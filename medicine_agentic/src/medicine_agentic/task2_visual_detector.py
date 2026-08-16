"""Adaptive RGB-D medicine-carton front-face boundary detector.

The detector first finds every independently placed copy of the approved
front-face reference with SIFT/RANSAC.  A strict quadrilateral quality gate
rejects glare-warped homographies.  When exactly one carton is hidden by
glare, the bundled RGB-D four-edge fitter recovers that carton from the same
colour/depth frame.  No fixed slots or fixed carton count are used.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from medicine_agentic.detector_provider import apply_grasp_policy
from medicine_agentic.task1_box import BoxCandidate


def _canonical_quad(polygon: Any) -> np.ndarray:
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    return np.roll(ordered, -int(np.argmin(ordered[:, 0] + ordered[:, 1])), axis=0)


def _long_axis_angle(rect: Any) -> float:
    (_, _), (width, height), angle = rect
    value = float(angle if width >= height else angle + 90.0)
    while value >= 90.0:
        value -= 180.0
    while value < -90.0:
        value += 180.0
    return value


def _rect_axes(polygon: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    (_, _), (a, b), angle = cv2.minAreaRect(np.asarray(polygon, np.float32))
    if a > b:
        a, b = b, a
        angle += 90.0
    theta = math.radians(angle)
    short_axis = np.asarray([math.cos(theta), math.sin(theta)], np.float32)
    long_axis = np.asarray([-math.sin(theta), math.cos(theta)], np.float32)
    if abs(float(short_axis[0])) < abs(float(long_axis[0])):
        short_axis, long_axis = long_axis, short_axis
    if short_axis[0] < 0:
        short_axis = -short_axis
    if long_axis[1] < 0:
        long_axis = -long_axis
    return float(a), float(b), short_axis, long_axis


def _strong_glare_fraction(rgb: np.ndarray, polygon: np.ndarray) -> float:
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 255)
    mask = cv2.erode(mask, np.ones((11, 11), np.uint8)) > 0
    if not np.any(mask):
        return 0.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return float(np.mean((hsv[..., 1][mask] < 35) & (hsv[..., 2][mask] > 235)))


def _normalize_glare_homographies(
    rgb: np.ndarray,
    instances: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[bool]]:
    """Replace only weak, overexposed homography shapes with row geometry.

    A reflection can leave enough coincidental SIFT matches for a rough centre
    while badly warping the projected corners.  The centre is still useful;
    width, height and direction are taken from the non-glared cartons in the
    same frame.  No slot position or carton index is assumed.
    """
    if len(instances) < 3:
        return instances, [False] * len(instances)
    polygons = [_canonical_quad(item["polygon_px"]) for item in instances]
    inliers = np.asarray([float(item.get("inliers", 0)) for item in instances])
    support_reference = float(np.percentile(inliers, 75))
    glare = np.asarray([_strong_glare_fraction(rgb, polygon) for polygon in polygons])
    suspicious = (glare >= 0.22) & (inliers <= max(12.0, 0.65 * support_reference))
    good_indices = np.flatnonzero(~suspicious)
    if not np.any(suspicious) or len(good_indices) < 2:
        return instances, suspicious.tolist()

    geometry = [_rect_axes(polygons[index]) for index in good_indices]
    expected_short = float(np.median([item[0] for item in geometry]))
    expected_long = float(np.median([item[1] for item in geometry]))
    short_axis = np.mean([item[2] for item in geometry], axis=0)
    short_axis /= max(float(np.linalg.norm(short_axis)), 1e-6)
    long_axis = np.asarray([-short_axis[1], short_axis[0]], np.float32)
    if long_axis[1] < 0:
        long_axis = -long_axis

    normalized = [dict(item) for item in instances]
    for index in np.flatnonzero(suspicious):
        center = np.asarray(instances[index]["center_px"], np.float32)
        polygon = np.asarray(
            [
                center - short_axis * expected_short / 2 - long_axis * expected_long / 2,
                center + short_axis * expected_short / 2 - long_axis * expected_long / 2,
                center + short_axis * expected_short / 2 + long_axis * expected_long / 2,
                center - short_axis * expected_short / 2 + long_axis * expected_long / 2,
            ],
            np.float32,
        )
        normalized[index]["polygon_px"] = polygon.tolist()
        normalized[index]["glare_geometry_normalized"] = True
        normalized[index]["glare_fraction"] = float(glare[index])
    return normalized, suspicious.tolist()


def _geometry_quality(
    instances: list[dict[str, Any]],
    *,
    minimum_fill: float = 0.84,
    maximum_opposite_ratio: float = 1.42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    measurements: list[dict[str, Any]] = []
    for instance in instances:
        polygon = np.asarray(instance["polygon_px"], np.float32)
        area = abs(float(cv2.contourArea(polygon)))
        (_, _), (a, b), _ = cv2.minAreaRect(polygon)
        short, long = sorted((float(a), float(b)))
        sides = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
        opposite_ratio = max(
            max(float(sides[0]), float(sides[2])) / max(min(float(sides[0]), float(sides[2])), 1.0),
            max(float(sides[1]), float(sides[3])) / max(min(float(sides[1]), float(sides[3])), 1.0),
        )
        measurements.append(
            {
                "short": short,
                "long": long,
                "area": area,
                "fill": area / max(short * long, 1.0),
                "opposite_ratio": opposite_ratio,
                "convex": bool(cv2.isContourConvex(polygon)),
            }
        )
    if not measurements:
        return [], []
    median_short = float(np.median([item["short"] for item in measurements]))
    median_long = float(np.median([item["long"] for item in measurements]))
    median_area = float(np.median([item["area"] for item in measurements]))
    reliable: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for index, (instance, item) in enumerate(zip(instances, measurements)):
        short_ratio = item["short"] / max(median_short, 1.0)
        long_ratio = item["long"] / max(median_long, 1.0)
        area_ratio = item["area"] / max(median_area, 1.0)
        passed = bool(
            item["convex"]
            and item["fill"] >= minimum_fill
            and item["opposite_ratio"] <= maximum_opposite_ratio
            and 0.72 <= short_ratio <= 1.30
            and 0.72 <= long_ratio <= 1.30
            and 0.68 <= area_ratio <= 1.35
        )
        diagnostics.append(
            {
                "index": index + 1,
                "passed": passed,
                "short_ratio": short_ratio,
                "long_ratio": long_ratio,
                "area_ratio": area_ratio,
                "fill": item["fill"],
                "opposite_ratio": item["opposite_ratio"],
            }
        )
        if passed:
            reliable.append(instance)
    return reliable, diagnostics


def _cached_missing_quad(
    rgb: np.ndarray,
    depth_z16: np.ndarray | None,
    current: list[np.ndarray],
    cached: list[np.ndarray],
    *,
    maximum_center_delta_px: float = 18.0,
) -> np.ndarray | None:
    """Recover one transiently missed carton from a recently verified row.

    Three current SIFT quads must map uniquely onto three cached quads.  The
    fourth cached quad is translated by their median motion and accepted only
    when the current frame still contains a pink face with compatible depth.
    This makes the fast path invalid as soon as a carton is removed or the row
    is rearranged.
    """

    if len(current) != 3 or len(cached) != 4:
        return None
    cached_centers = np.asarray([polygon.mean(axis=0) for polygon in cached])
    current_centers = np.asarray([polygon.mean(axis=0) for polygon in current])
    distances = np.linalg.norm(
        current_centers[:, None, :] - cached_centers[None, :, :], axis=2
    )
    pairs: list[tuple[int, int]] = []
    available = set(range(4))
    for current_index in np.argsort(np.min(distances, axis=1)):
        cached_index = min(available, key=lambda value: distances[current_index, value])
        if float(distances[current_index, cached_index]) > maximum_center_delta_px:
            return None
        pairs.append((int(current_index), int(cached_index)))
        available.remove(cached_index)
    if len(available) != 1:
        return None
    shifts = np.asarray(
        [current_centers[current_index] - cached_centers[cached_index] for current_index, cached_index in pairs]
    )
    if float(np.max(np.linalg.norm(shifts - np.median(shifts, axis=0), axis=1))) > 8.0:
        return None
    missing = _canonical_quad(cached[available.pop()] + np.median(shifts, axis=0))
    height, width = rgb.shape[:2]
    if (
        float(np.min(missing[:, 0])) < 0.0
        or float(np.max(missing[:, 0])) >= width
        or float(np.min(missing[:, 1])) < 0.0
        or float(np.max(missing[:, 1])) >= height
    ):
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(missing).astype(np.int32), 255)
    inside = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    pink = (hue >= 130) & (hue <= 179) & (saturation >= 8) & (saturation <= 170) & (value >= 105)
    if int(np.count_nonzero(inside)) == 0 or float(np.mean(pink[inside])) < 0.16:
        return None
    if depth_z16 is not None:
        missing_depth = depth_z16[inside]
        missing_depth = missing_depth[missing_depth > 0]
        anchor_depths = []
        for polygon in current:
            anchor_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(anchor_mask, np.rint(polygon).astype(np.int32), 255)
            values = depth_z16[anchor_mask > 0]
            values = values[values > 0]
            if values.size:
                anchor_depths.append(float(np.median(values)))
        if missing_depth.size < 32 or len(anchor_depths) < 2:
            return None
        if abs(float(np.median(missing_depth)) - float(np.median(anchor_depths))) > 35.0:
            return None
    return missing


class Task2AdaptiveVisualDetector:
    """Adaptive detector returning policy-checked individual front faces.

    Task2 historically required three or four formed cartons.  The same
    independent SIFT/RANSAC front-face recognition is also useful for Task3,
    where one or two flat dielines expose the identical printed front panel.
    Count, ROI and recovery policy therefore come from the isolated task
    detector config; Task2's defaults remain unchanged.
    """

    name = "task2_adaptive_rgbd"

    def __init__(self, config: dict[str, Any], face_bank: Any | None) -> None:
        self._config = dict(config)
        profile_name = str(
            self._config.get("adaptive_profile_name", "task2")
        ).strip()
        self.name = f"{profile_name or 'task2'}_adaptive_rgbd"
        allowed_value = self._config.get("adaptive_allowed_counts", [3, 4])
        allowed_counts: list[int] = []
        if isinstance(allowed_value, (list, tuple)):
            for value in allowed_value:
                if isinstance(value, bool):
                    continue
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    continue
                if 1 <= parsed <= 8:
                    allowed_counts.append(parsed)
        self._allowed_counts = tuple(sorted(set(allowed_counts or [3, 4])))
        self._maximum_instances = max(self._allowed_counts)
        self._sift_ratio = float(
            self._config.get("adaptive_sift_ratio", 0.80)
        )
        if not 0.75 <= self._sift_ratio <= 0.90:
            self._sift_ratio = 0.80
        self._homography_attempt_multiplier = max(
            2,
            min(
                6,
                int(
                    self._config.get(
                        "adaptive_homography_attempt_multiplier", 2
                    )
                ),
            ),
        )
        self._minimum_recovery_anchors = max(
            2,
            int(self._config.get("adaptive_minimum_recovery_anchors", 2)),
        )
        self._recovery_enabled = bool(
            self._config.get("adaptive_recovery_enabled", True)
        )
        self._reject_glare_matches = bool(
            self._config.get("adaptive_reject_glare_matches", True)
        )
        self._minimum_quad_fill = float(
            self._config.get("adaptive_minimum_quad_fill", 0.84)
        )
        if not 0.65 <= self._minimum_quad_fill <= 0.95:
            self._minimum_quad_fill = 0.84
        self._maximum_opposite_ratio = float(
            self._config.get("adaptive_maximum_opposite_side_ratio", 1.42)
        )
        if not 1.20 <= self._maximum_opposite_ratio <= 1.80:
            self._maximum_opposite_ratio = 1.42
        self._face_bank = face_bank
        self._lock = threading.Lock()
        self._last_error: str | None = None
        self._last_stage = "not_run"
        self._last_count = 0
        self._last_geometry: list[dict[str, Any]] = []
        self._last_glare_normalized: list[bool] = []
        self._cached_four_quads: list[np.ndarray] = []
        self._cached_four_at = 0.0
        self._remote_recovery = dict(self._config.get("rgbd_recovery_remote") or {})
        self._face: Any | None = None
        self._reference_gray: np.ndarray | None = None
        self._sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.018, edgeThreshold=16)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        self._reference_keypoints: tuple[Any, ...] = ()
        self._reference_descriptors: np.ndarray | None = None
        if face_bank is not None:
            for face in face_bank.faces:
                if str(face.face_type) != "front_large" or not bool(face.pick_allowed):
                    continue
                gray = cv2.imread(str(face.image_path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                keypoints, descriptors = self._sift.detectAndCompute(gray, None)
                if descriptors is None or len(keypoints) < 8:
                    continue
                self._face = face
                self._reference_gray = gray
                self._reference_keypoints = tuple(keypoints)
                self._reference_descriptors = descriptors
                break

    @property
    def ready(self) -> bool:
        return bool(self._face is not None and self._reference_descriptors is not None)

    def _sift_instances(self, rgb: np.ndarray) -> list[dict[str, Any]]:
        assert self._reference_gray is not None
        assert self._reference_descriptors is not None
        height, width = rgb.shape[:2]
        roi = self._config.get(
            "adaptive_roi_norm",
            self._config.get(
                "task2_adaptive_roi_norm",
                [0.39, 0.50, 0.75, 0.82],
            ),
        )
        if not isinstance(roi, (list, tuple)) or len(roi) != 4:
            roi = [0.39, 0.50, 0.75, 0.82]
        x0, y0, x1, y1 = (
            int(float(roi[0]) * width),
            int(float(roi[1]) * height),
            int(float(roi[2]) * width),
            int(float(roi[3]) * height),
        )
        include_polygon: np.ndarray | None = None
        polygon_value = self._config.get("adaptive_include_polygon_norm")
        if isinstance(polygon_value, (list, tuple)) and len(polygon_value) >= 3:
            try:
                parsed_polygon = np.asarray(
                    [
                        [float(point[0]) * width, float(point[1]) * height]
                        for point in polygon_value
                    ],
                    dtype=np.float32,
                )
                if (
                    parsed_polygon.ndim == 2
                    and parsed_polygon.shape[1] == 2
                    and np.all(np.isfinite(parsed_polygon))
                ):
                    include_polygon = parsed_polygon
            except (IndexError, TypeError, ValueError):
                include_polygon = None
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        scene = gray[y0:y1, x0:x1]
        scene_keypoints, scene_descriptors = self._sift.detectAndCompute(scene, None)
        if scene_descriptors is None or len(scene_keypoints) < 8:
            return []
        pairs = self._matcher.knnMatch(scene_descriptors, self._reference_descriptors, k=2)
        good = [
            pair[0]
            for pair in pairs
            if len(pair) == 2
            and pair[0].distance < self._sift_ratio * pair[1].distance
        ]
        if len(good) < 8:
            return []
        reference_points = np.float32(
            [self._reference_keypoints[item.trainIdx].pt for item in good]
        )
        scene_points = np.float32(
            [[scene_keypoints[item.queryIdx].pt[0] + x0, scene_keypoints[item.queryIdx].pt[1] + y0] for item in good]
        )
        active = np.ones(len(good), dtype=bool)
        ref_h, ref_w = self._reference_gray.shape
        reference_corners = np.float32(
            [[0, 0], [ref_w - 1, 0], [ref_w - 1, ref_h - 1], [0, ref_h - 1]]
        ).reshape(-1, 1, 2)
        short_range = self._config.get(
            "adaptive_short_side_px_range", [60.0, 115.0]
        )
        long_range = self._config.get(
            "adaptive_long_side_px_range", [105.0, 175.0]
        )
        try:
            short_min, short_max = map(float, short_range)
            long_min, long_max = map(float, long_range)
        except (TypeError, ValueError):
            short_min, short_max = 60.0, 115.0
            long_min, long_max = 105.0, 175.0
        instances: list[dict[str, Any]] = []
        slot_shape = self._config.get("adaptive_slot_grid_shape")
        slot_grid_polygon = include_polygon
        slot_polygon_value = self._config.get("adaptive_slot_polygon_norm")
        if (
            isinstance(slot_polygon_value, (list, tuple))
            and len(slot_polygon_value) >= 3
        ):
            try:
                parsed_slot_polygon = np.asarray(
                    [
                        [float(point[0]) * width, float(point[1]) * height]
                        for point in slot_polygon_value
                    ],
                    dtype=np.float32,
                )
                if (
                    parsed_slot_polygon.ndim == 2
                    and parsed_slot_polygon.shape[1] == 2
                    and np.all(np.isfinite(parsed_slot_polygon))
                ):
                    slot_grid_polygon = parsed_slot_polygon
            except (IndexError, TypeError, ValueError):
                pass
        slot_min_matches = max(
            6,
            int(self._config.get("adaptive_slot_min_matches", 8)),
        )
        slot_min_inliers = max(
            5,
            min(
                slot_min_matches,
                int(self._config.get("adaptive_slot_min_inliers", 7)),
            ),
        )
        try:
            slot_sift_ratio = float(
                self._config.get("adaptive_slot_sift_ratio", self._sift_ratio)
            )
        except (TypeError, ValueError):
            slot_sift_ratio = self._sift_ratio
        if not 0.75 <= slot_sift_ratio <= 0.96:
            slot_sift_ratio = self._sift_ratio
        slot_good = [
            pair[0]
            for pair in pairs
            if len(pair) == 2
            and pair[0].distance < slot_sift_ratio * pair[1].distance
        ]
        slot_reference_points = np.float32(
            [self._reference_keypoints[item.trainIdx].pt for item in slot_good]
        )
        slot_scene_points = np.float32(
            [
                [
                    scene_keypoints[item.queryIdx].pt[0] + x0,
                    scene_keypoints[item.queryIdx].pt[1] + y0,
                ]
                for item in slot_good
            ]
        )
        if (
            slot_grid_polygon is not None
            and isinstance(slot_shape, (list, tuple))
            and len(slot_shape) == 2
            and len(slot_good) >= slot_min_matches
        ):
            try:
                slot_rows = int(slot_shape[0])
                slot_columns = int(slot_shape[1])
            except (TypeError, ValueError):
                slot_rows = slot_columns = 0
            if 1 <= slot_rows <= 4 and 1 <= slot_columns <= 4:
                top_left, top_right, bottom_right, bottom_left = slot_grid_polygon

                def grid_point(u: float, v: float) -> np.ndarray:
                    return (
                        (1.0 - u) * (1.0 - v) * top_left
                        + u * (1.0 - v) * top_right
                        + u * v * bottom_right
                        + (1.0 - u) * v * bottom_left
                    )

                for row in range(slot_rows):
                    for column in range(slot_columns):
                        u0 = column / slot_columns
                        u1 = (column + 1) / slot_columns
                        v0 = row / slot_rows
                        v1 = (row + 1) / slot_rows
                        slot_polygon = np.asarray(
                            [
                                grid_point(u0, v0),
                                grid_point(u1, v0),
                                grid_point(u1, v1),
                                grid_point(u0, v1),
                            ],
                            dtype=np.float32,
                        )
                        indices = np.asarray(
                            [
                                index
                                for index, point in enumerate(slot_scene_points)
                                if cv2.pointPolygonTest(
                                    slot_polygon,
                                    (float(point[0]), float(point[1])),
                                    False,
                                )
                                >= 0.0
                            ],
                            dtype=np.int64,
                        )
                        if len(indices) < slot_min_matches:
                            continue
                        homography, mask = cv2.findHomography(
                            slot_reference_points[indices],
                            slot_scene_points[indices],
                            cv2.RANSAC,
                            4.0,
                            maxIters=8000,
                            confidence=0.999,
                        )
                        if homography is None or mask is None:
                            continue
                        local = mask.ravel().astype(bool)
                        inlier_count = int(np.count_nonzero(local))
                        if inlier_count < slot_min_inliers:
                            continue
                        polygon = cv2.perspectiveTransform(
                            reference_corners, homography
                        ).reshape(-1, 2)
                        projected = cv2.perspectiveTransform(
                            slot_reference_points[indices].reshape(-1, 1, 2),
                            homography,
                        ).reshape(-1, 2)
                        residual = np.linalg.norm(
                            projected - slot_scene_points[indices], axis=1
                        )
                        area = abs(
                            float(cv2.contourArea(polygon.astype(np.float32)))
                        )
                        (_, _), (a, b), _ = cv2.minAreaRect(
                            polygon.astype(np.float32)
                        )
                        short, long = sorted((float(a), float(b)))
                        center = polygon.mean(axis=0)
                        valid = bool(
                            np.isfinite(polygon).all()
                            and area > 4000.0
                            and short_min < short < short_max
                            and long_min < long < long_max
                            and cv2.pointPolygonTest(
                                include_polygon,
                                (float(center[0]), float(center[1])),
                                False,
                            )
                            >= 0.0
                        )
                        if valid:
                            instances.append(
                                {
                                    "polygon_px": polygon.tolist(),
                                    "center_px": center.tolist(),
                                    "inliers": inlier_count,
                                    "median_error_px": float(
                                        np.median(residual[local])
                                    ),
                                }
                            )
        for _ in range(
            max(
                8,
                self._maximum_instances * self._homography_attempt_multiplier,
            )
        ):
            if len(instances) >= self._maximum_instances:
                break
            indices = np.flatnonzero(active)
            if len(indices) < 8:
                break
            homography, mask = cv2.findHomography(
                reference_points[indices],
                scene_points[indices],
                cv2.RANSAC,
                4.0,
                maxIters=8000,
                confidence=0.999,
            )
            if homography is None or mask is None:
                break
            local = mask.ravel().astype(bool)
            inlier_count = int(np.count_nonzero(local))
            if inlier_count < 7:
                break
            polygon = cv2.perspectiveTransform(reference_corners, homography).reshape(-1, 2)
            projected = cv2.perspectiveTransform(
                reference_points[indices].reshape(-1, 1, 2), homography
            ).reshape(-1, 2)
            residual = np.linalg.norm(projected - scene_points[indices], axis=1)
            active[indices[residual < 7.0]] = False
            area = abs(float(cv2.contourArea(polygon.astype(np.float32))))
            (_, _), (a, b), _ = cv2.minAreaRect(polygon.astype(np.float32))
            short, long = sorted((float(a), float(b)))
            center = polygon.mean(axis=0)
            valid = bool(
                np.isfinite(polygon).all()
                and area > 4000.0
                and short_min < short < short_max
                and long_min < long < long_max
                and x0 - 15.0 < center[0] < x1 + 15.0
                and y0 - 20.0 < center[1] < y1 + 25.0
                and (
                    include_polygon is None
                    or cv2.pointPolygonTest(
                        include_polygon,
                        (float(center[0]), float(center[1])),
                        False,
                    )
                    >= 0.0
                )
            )
            if valid:
                instances.append(
                    {
                        "polygon_px": polygon.tolist(),
                        "center_px": center.tolist(),
                        "inliers": inlier_count,
                        "median_error_px": float(np.median(residual[local])),
                    }
                )
            if len(instances) >= self._maximum_instances:
                break
        kept: list[dict[str, Any]] = []
        for item in sorted(instances, key=lambda value: value["inliers"], reverse=True):
            center = np.asarray(item["center_px"], np.float32)
            if any(np.linalg.norm(center - np.asarray(other["center_px"], np.float32)) < 45.0 for other in kept):
                continue
            kept.append(item)
        return sorted(
            kept[: self._maximum_instances],
            key=lambda value: value["center_px"][0],
        )

    def _recover_one(
        self,
        rgb: np.ndarray,
        depth_z16: np.ndarray,
        depth_scale_m: float,
        anchors: list[dict[str, Any]],
        target_total: int,
        recovery_priors: list[list[float]] | None = None,
    ) -> np.ndarray | None:
        package_dir = Path(__file__).resolve().parent
        visual_script = package_dir / "task2_visual_quad_any.py"
        fit_script = package_dir / "task2_visual_quad_fit.py"
        if not visual_script.is_file() or not fit_script.is_file():
            raise RuntimeError("Task2 RGB-D boundary helper scripts are missing")
        with tempfile.TemporaryDirectory(prefix="task2-rgbd-") as temp_name:
            temp = Path(temp_name)
            capture = temp / "capture"
            output = temp / "output"
            capture.mkdir()
            output.mkdir()
            cv2.imwrite(str(capture / "color.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(capture / "depth.png"), depth_z16)
            (capture / "meta.json").write_text(
                json.dumps({"frame_id": 0, "depth_scale_m": float(depth_scale_m)}),
                encoding="utf-8",
            )
            sift_report = temp / "sift.json"
            sift_report.write_text(json.dumps({"instances": anchors}), encoding="utf-8")
            remote = self._recover_one_remote(
                temp,
                target_total=target_total,
                recovery_priors=recovery_priors or [],
            )
            if remote is not None:
                return remote
            environment = os.environ.copy()
            environment.update(
                {
                    "TASK2_CAPTURE": str(capture),
                    "TASK2_SIFT_REPORT": str(sift_report),
                    "TASK2_TARGET_TOTAL": str(target_total),
                    "TASK2_VISUAL_OUT": str(output),
                    "TASK2_VISUAL_FIT_SCRIPT": str(fit_script),
                    "TASK2_RECOVERY_PRIORS": json.dumps(
                        recovery_priors or []
                    ),
                }
            )
            completed = subprocess.run(
                [sys.executable, str(visual_script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=32,
                env=environment,
                text=True,
                check=False,
            )
            report_path = output / "report.json"
            if completed.returncode != 0 or not report_path.is_file():
                detail = completed.stderr.strip()[-400:]
                raise RuntimeError(f"RGB-D boundary helper failed: {detail or completed.returncode}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("passed") is not True or "polygon_px" not in report:
                return None
            return _canonical_quad(report["polygon_px"])

    def _recover_one_remote(
        self,
        temp: Path,
        *,
        target_total: int,
        recovery_priors: list[list[float]],
    ) -> np.ndarray | None:
        """Recover one glare-hidden carton on the LAN CPU, or fall back locally."""
        if self._remote_recovery.get("enabled") is not True:
            return None
        host = str(self._remote_recovery.get("host", "")).strip()
        python = str(self._remote_recovery.get("python", "")).strip()
        remote_script = str(self._remote_recovery.get("script", "")).strip()
        remote_root = str(self._remote_recovery.get("work_root", "/tmp/task2-rgbd"))
        if not host or not python or not remote_script:
            return None
        (temp / "settings.json").write_text(
            json.dumps({"target_total": target_total, "recovery_priors": recovery_priors}),
            encoding="utf-8",
        )
        remote_dir = f"{remote_root}-{os.getpid()}-{time.monotonic_ns()}"
        try:
            subprocess.run(
                ["scp", "-q", "-F", "/dev/null", "-r", str(temp), f"{host}:{remote_dir}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
                check=True,
            )
            command = f"{python} {remote_script} {remote_dir} {python}"
            subprocess.run(
                ["ssh", "-F", "/dev/null", "-o", "BatchMode=yes", host, command],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=4,
                check=True,
            )
            local_results = temp / "remote-report.json"
            subprocess.run(
                ["scp", "-q", "-F", "/dev/null", f"{host}:{remote_dir}/output/report.json", str(local_results)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=True,
            )
            report = json.loads(local_results.read_text(encoding="utf-8"))
            if report.get("passed") is not True or "polygon_px" not in report:
                return None
            return _canonical_quad(report["polygon_px"])
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, json.JSONDecodeError):
            return None
        finally:
            subprocess.run(
                ["ssh", "-F", "/dev/null", "-o", "BatchMode=yes", host, f"rm -rf -- {remote_dir}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
            )

    def _pink_fraction(self, rgb: np.ndarray, polygon: np.ndarray) -> float:
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 255)
        inside = mask > 0
        count = int(np.count_nonzero(inside))
        if count == 0:
            return 0.0
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue, saturation, value = cv2.split(hsv)
        pink = (hue >= 130) & (hue <= 179) & (saturation >= 8) & (saturation <= 170) & (value >= 105)
        return float(np.count_nonzero(pink & inside)) / float(count)

    def _candidate(self, rgb: np.ndarray, polygon: np.ndarray, *, recovered: bool, inliers: int = 7) -> BoxCandidate:
        assert self._face is not None
        polygon = _canonical_quad(polygon)
        rect = cv2.minAreaRect(polygon)
        long_side = max(float(rect[1][0]), float(rect[1][1]))
        short_side = min(float(rect[1][0]), float(rect[1][1]))
        area = abs(float(cv2.contourArea(polygon)))
        center = tuple(float(value) for value in polygon.mean(axis=0))
        score = 0.90 if recovered else min(0.99, 0.86 + 0.012 * max(0, inliers - 7))
        raw = BoxCandidate(
            center_px=center,
            suction_px=(int(round(center[0])), int(round(center[1]))),
            polygon_px=tuple((float(point[0]), float(point[1])) for point in polygon),
            long_side_px=long_side,
            short_side_px=short_side,
            angle_deg=_long_axis_angle(rect),
            rectangularity=min(1.0, area / max(long_side * short_side, 1.0)),
            bright_fill=self._pink_fraction(rgb, polygon),
            edge_clearance_px=short_side * 0.5,
            score=score,
            provider=f"{self.name}:{'rgbd_four_edge' if recovered else 'multi_sift'}",
            face_type=str(self._face.face_type),
            face_score=score,
            reference_face_id=str(self._face.id),
            graspable=False,
            grasp_blockers=(),
        )
        return apply_grasp_policy(
            raw,
            face_bank=self._face_bank,
            config=self._config,
        )

    def detect_rgbd(
        self,
        rgb: np.ndarray,
        depth_z16: np.ndarray | None,
        depth_scale_m: float | None,
    ) -> list[BoxCandidate]:
        with self._lock:
            if not self.ready:
                self._last_error = "approved front_large reference is unavailable"
                self._last_stage = "unavailable"
                self._last_count = 0
                return []
            try:
                instances = self._sift_instances(rgb)
                normalized_instances, glare_normalized = _normalize_glare_homographies(rgb, instances)
                self._last_glare_normalized = glare_normalized
                reject_glare_matches = getattr(
                    self, "_reject_glare_matches", True
                )
                if reject_glare_matches:
                    glare_priors = [
                        list(map(float, instance["center_px"]))
                        for instance, suspicious in zip(instances, glare_normalized)
                        if suspicious
                    ]
                    trusted_instances = [
                        instance
                        for instance, suspicious in zip(
                            normalized_instances, glare_normalized
                        )
                        if not suspicious
                    ]
                else:
                    # Dense Task1 stacks expose the same glossy print under
                    # slightly different angles.  A low-inlier glare signal
                    # is diagnostic, not proof that the measured homography
                    # is false.  Preserve every original SIFT quadrilateral;
                    # the independent geometry and RGB-D physical gates below
                    # still reject distorted or non-carton candidates.
                    glare_priors = []
                    trusted_instances = list(instances)
                reliable, geometry = _geometry_quality(
                    trusted_instances,
                    minimum_fill=self._minimum_quad_fill,
                    maximum_opposite_ratio=getattr(
                        self, "_maximum_opposite_ratio", 1.42
                    ),
                )
                rejected_priors = [
                    list(map(float, instance["center_px"]))
                    for instance, diagnostic in zip(trusted_instances, geometry)
                    if diagnostic.get("passed") is not True
                ]
                rejected_priors = glare_priors + rejected_priors
                self._last_geometry = geometry
                polygons = [_canonical_quad(item["polygon_px"]) for item in reliable]
                recovered_flags = [False] * len(polygons)
                stage = "sift_direct"
                next_allowed_count = next(
                    (
                        count
                        for count in self._allowed_counts
                        if count > len(reliable)
                    ),
                    None,
                )
                if (
                    self._recovery_enabled
                    and next_allowed_count is not None
                    and len(reliable) >= self._minimum_recovery_anchors
                    and depth_z16 is not None
                    and depth_scale_m
                ):
                    target_total = next_allowed_count
                    recovered = None
                    if (
                        target_total == 4
                        and time.monotonic() - self._cached_four_at <= 5.0
                    ):
                        recovered = _cached_missing_quad(
                            rgb,
                            depth_z16,
                            polygons,
                            self._cached_four_quads,
                        )
                        if recovered is not None:
                            stage = "verified_cached_four_recovery"
                    if recovered is None:
                        recovered = self._recover_one(
                            rgb,
                            depth_z16,
                            float(depth_scale_m),
                            reliable,
                            target_total,
                            rejected_priors,
                        )
                    if recovered is not None:
                        polygons.append(recovered)
                        recovered_flags.append(True)
                        if stage != "verified_cached_four_recovery":
                            stage = f"rgbd_four_edge_recovery_{target_total}"
                    elif len(reliable) in self._allowed_counts:
                        stage = f"sift_{len(reliable)}_no_{target_total}_body"
                    else:
                        polygons = []
                        recovered_flags = []
                        stage = (
                            f"{len(reliable)}_anchors_no_{target_total}_body"
                        )
                elif len(reliable) not in self._allowed_counts:
                    polygons = []
                    recovered_flags = []
                    stage = "unsupported_reliable_instance_count"
                direct = (
                    [
                        (_canonical_quad(item["polygon_px"]), False, int(item["inliers"]))
                        for item in reliable
                    ]
                    if polygons
                    else []
                )
                recovered_items = [
                    (polygon, True, 7)
                    for polygon in polygons[len(reliable):]
                ]
                ordered = sorted(
                    direct + recovered_items,
                    key=lambda item: float(item[0].mean(axis=0)[0]),
                )
                candidates = [
                    self._candidate(
                        rgb,
                        polygon,
                        recovered=recovered,
                        inliers=inliers,
                    )
                    for polygon, recovered, inliers in ordered
                ]
                self._last_error = None
                self._last_stage = stage
                self._last_count = len(candidates)
                if self._recovery_enabled and len(candidates) == 4:
                    self._cached_four_quads = [
                        _canonical_quad(candidate.polygon_px)
                        for candidate in candidates
                    ]
                    self._cached_four_at = time.monotonic()
                return candidates
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._last_stage = "error"
                self._last_count = 0
                return []

    def detect(self, rgb: np.ndarray) -> list[BoxCandidate]:
        return self.detect_rgbd(rgb, None, None)

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ready and self._last_error is None,
            "mode": "adaptive_count_multi_sift_plus_same_frame_rgbd_four_edge",
            "fixed_slots": False,
            "allowed_counts": list(self._allowed_counts),
            "sift_ratio": self._sift_ratio,
            "homography_attempt_multiplier": self._homography_attempt_multiplier,
            "recovery_enabled": self._recovery_enabled,
            "glare_rejection_enabled": self._reject_glare_matches,
            "minimum_quad_fill": self._minimum_quad_fill,
            "maximum_opposite_side_ratio": getattr(
                self, "_maximum_opposite_ratio", 1.42
            ),
            "last_stage": self._last_stage,
            "last_candidate_count": self._last_count,
            "last_geometry": self._last_geometry,
            "last_glare_normalized": self._last_glare_normalized,
            "last_error": self._last_error,
        }
