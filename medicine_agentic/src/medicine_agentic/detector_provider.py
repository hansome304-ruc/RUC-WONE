"""Pluggable 2-D carton detectors with one fail-closed grasp policy.

Detector backends are allowed to propose geometry and face identity.  They are
not allowed to authorize a grasp.  :class:`PolicyEnforcedDetectorProvider`
recomputes ``graspable`` from the immutable reference bank and deterministic
geometry checks after every inference call.

Heavy optional backends are imported only when selected in configuration.  The
default installation therefore remains independent from torch, Ultralytics,
SAM, and other model runtimes.
"""
from __future__ import annotations

import importlib
import math
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from medicine_agentic.reference_faces import ReferenceFaceBank
from medicine_agentic.task1_box import BoxCandidate, detect_cartons


POLICY_BLOCKERS = frozenset(
    {
        "candidate_geometry_invalid",
        "detection_score_low",
        "face_score_low",
        "face_type_mismatch",
        "face_unverified",
        "reference_bank_unavailable",
        "reference_face_missing",
        "reference_face_not_pick_allowed",
        "suction_clearance_low",
    }
)


class DetectorProvider(Protocol):
    """Minimal interface implemented by classical and optional model backends."""

    name: str

    def detect(self, rgb: np.ndarray) -> list[BoxCandidate]:
        """Return raw candidates. A wrapper will recompute grasp authorization."""

    def status(self) -> dict[str, Any]:
        """Return JSON-serializable diagnostic state."""


class ClassicalDetectorProvider:
    """Adapter around the existing HSV/contour proposal generator."""

    name = "classical"

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = dict(config)

    def detect(self, rgb: np.ndarray) -> list[BoxCandidate]:
        return detect_cartons(rgb, self._config)

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": True,
            "mode": "geometry_proposals_only",
        }


class ReferenceFeatureDetectorProvider:
    """Verify broad contour proposals against the supplied face images.

    SIFT descriptors are scale and rotation invariant. Geometry and colour
    only propose regions; carton identity comes from local reference-image
    correspondences and a RANSAC homography, not from a fixed pixel size.
    """

    name = "reference_feature"

    def __init__(
        self,
        config: dict[str, Any],
        face_bank: ReferenceFaceBank | None,
        *,
        config_dir: Path,
    ) -> None:
        if face_bank is None:
            raise ValueError("reference_feature requires a ready face bank")
        self._config = dict(config)
        self._sift = cv2.SIFT_create(
            nfeatures=int(config.get("reference_feature_max_features", 1200)),
            contrastThreshold=float(
                config.get("reference_feature_contrast_threshold", 0.04)
            ),
        )
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        self._ratio = float(config.get("reference_feature_ratio", 0.78))
        self._min_matches = int(
            config.get("reference_feature_min_matches", 5)
        )
        self._min_inliers = int(
            config.get("reference_feature_min_inliers", 4)
        )
        self._ransac_threshold_px = float(
            config.get("reference_feature_ransac_px", 8.0)
        )
        self._minimum_scene_hull_fraction = float(
            config.get(
                "reference_feature_min_scene_hull_fraction",
                0.008,
            )
        )
        self._minimum_reference_hull_fraction = float(
            config.get(
                "reference_feature_min_reference_hull_fraction",
                0.02,
            )
        )
        self._minimum_pink_fraction = float(
            config.get("reference_feature_min_pink_fraction", 0.008)
        )
        configured_face_types = config.get("reference_feature_face_types")
        allowed_face_types = (
            {
                str(value)
                for value in configured_face_types
                if isinstance(value, str) and value
            }
            if isinstance(configured_face_types, (list, tuple))
            else set()
        )
        self._references: list[
            tuple[Any, tuple[Any, ...], np.ndarray, np.ndarray]
        ] = []
        for face in face_bank.faces:
            if not face.pick_allowed:
                continue
            if allowed_face_types and str(face.face_type) not in allowed_face_types:
                continue
            gray = cv2.imread(str(face.image_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            keypoints, descriptors = self._sift.detectAndCompute(gray, None)
            if descriptors is None or len(keypoints) < self._min_matches:
                continue
            self._references.append((face, tuple(keypoints), descriptors, gray))
        if not self._references:
            raise ValueError("reference bank has no feature-ready pick faces")
        self._motif_helper: Any | None = None
        self._motif_raw_score_minimum = 1.0
        provider_options = config.get("provider_options", {})
        if not isinstance(provider_options, dict):
            raise ValueError("detector.provider_options must be an object")
        motif_settings = provider_options.get("motif_templates", [])
        if motif_settings:
            # Reuse the lightweight, deterministic motif implementation from
            # the optional YOLOE module without constructing Torch or a YOLO
            # model.  The module only imports those dependencies inside the
            # YOLOE detector constructor, which is deliberately not called.
            from medicine_agentic.yoloe_visual_prompt import (
                YOLOEVisualPromptDetector,
            )

            helper = object.__new__(YOLOEVisualPromptDetector)
            helper._options = dict(provider_options)
            helper._motifs = helper._load_motifs(
                motif_settings,
                config_dir=config_dir,
                face_bank=face_bank,
            )
            self._motif_helper = helper
            self._motif_raw_score_minimum = min(
                float(setting.get("min_score", 0.28))
                for setting in motif_settings
                if isinstance(setting, dict)
            )
        self._last_candidate_count = 0
        self._last_verified_count = 0
        self._last_motif_count = 0

    @staticmethod
    def _hull_fraction(points: np.ndarray, area: float) -> float:
        if len(points) < 3 or area <= 0.0:
            return 0.0
        hull = cv2.convexHull(
            np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        )
        return float(cv2.contourArea(hull)) / max(float(area), 1.0)

    def _pink_fraction(
        self,
        rgb: np.ndarray,
        candidate: BoxCandidate,
    ) -> float:
        points = np.round(np.asarray(candidate.polygon_px)).astype(np.int32)
        if len(points) < 3:
            return 0.0
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, points, 255)
        inside = mask > 0
        pixel_count = int(np.count_nonzero(inside))
        if pixel_count <= 0:
            return 0.0
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue, saturation, value = cv2.split(hsv)
        pink = (
            (hue >= int(self._config.get("pink_hue_min", 135)))
            & (hue <= int(self._config.get("pink_hue_max", 179)))
            & (
                saturation
                >= int(self._config.get("pink_saturation_min", 8))
            )
            & (
                saturation
                <= int(self._config.get("pink_saturation_max", 150))
            )
            & (value >= int(self._config.get("pink_value_min", 115)))
        )
        return float(np.count_nonzero(pink & inside)) / pixel_count

    def _detect_motifs(self, rgb: np.ndarray) -> list[BoxCandidate]:
        helper = self._motif_helper
        if helper is None:
            self._last_motif_count = 0
            return []
        options = helper._options
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue, saturation, value = cv2.split(hsv)
        pink_mask = np.where(
            (
                (hue >= int(options.get("pink_hue_min", 130)))
                & (hue <= int(options.get("pink_hue_max", 175)))
                & (
                    saturation
                    >= int(options.get("pink_saturation_min", 8))
                )
                & (
                    saturation
                    <= int(options.get("pink_saturation_max", 130))
                )
                & (value >= int(options.get("pink_value_min", 130)))
            ),
            255,
            0,
        ).astype(np.uint8)
        raw = helper._detect_motifs(rgb, pink_mask=pink_mask)
        self._last_motif_count = len(raw)
        policy_floor = max(
            float(self._config.get("min_detection_score", 0.6)),
            float(self._config.get("min_face_score", 0.6)),
        )
        result: list[BoxCandidate] = []
        for candidate in raw:
            raw_score = float(candidate.score)
            evidence = max(
                0.0,
                min(
                    1.0,
                    (raw_score - self._motif_raw_score_minimum)
                    / max(1.0 - self._motif_raw_score_minimum, 1e-6),
                ),
            )
            calibrated_score = min(
                1.0,
                policy_floor + (1.0 - policy_floor) * evidence,
            )
            center = (
                float(candidate.center_px[0]),
                float(candidate.center_px[1]),
            )
            result.append(
                replace(
                    candidate,
                    suction_px=(int(round(center[0])), int(round(center[1]))),
                    edge_clearance_px=max(
                        float(candidate.edge_clearance_px),
                        float(candidate.short_side_px) * 0.5,
                    ),
                    score=calibrated_score,
                    provider=f"{self.name}:motif",
                    face_score=calibrated_score,
                    graspable=False,
                    grasp_blockers=(),
                )
            )
        result.sort(key=lambda item: item.score, reverse=True)
        return result

    @staticmethod
    def _long_axis_angle(rect: Any) -> float:
        (_, _), (width, height), angle = rect
        value = float(angle if width >= height else angle + 90.0)
        while value >= 90.0:
            value -= 180.0
        while value < -90.0:
            value += 180.0
        return value

    def _slot_candidate(
        self,
        *,
        polygon: np.ndarray,
        face: Any,
        score: float,
        provider_suffix: str,
        rgb: np.ndarray,
    ) -> BoxCandidate | None:
        polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon):
            return None
        area = abs(float(cv2.contourArea(polygon)))
        rect = cv2.minAreaRect(polygon)
        long_side = max(float(rect[1][0]), float(rect[1][1]))
        short_side = min(float(rect[1][0]), float(rect[1][1]))
        if area < 2500.0 or not (65.0 <= long_side <= 190.0):
            return None
        if not (42.0 <= short_side <= 125.0):
            return None
        center = tuple(float(value) for value in rect[0])
        candidate = BoxCandidate(
            center_px=center,
            suction_px=(int(round(center[0])), int(round(center[1]))),
            polygon_px=tuple(
                (float(point[0]), float(point[1])) for point in polygon
            ),
            long_side_px=long_side,
            short_side_px=short_side,
            angle_deg=self._long_axis_angle(rect),
            rectangularity=min(1.0, area / max(long_side * short_side, 1.0)),
            bright_fill=0.0,
            edge_clearance_px=short_side * 0.5,
            score=float(score),
            provider=f"{self.name}:{provider_suffix}",
            face_type=str(face.face_type),
            face_score=float(score),
            reference_face_id=str(face.id),
            graspable=False,
            grasp_blockers=(),
        )
        pink_fraction = self._pink_fraction(rgb, candidate)
        minimum_pink = float(
            self._config.get("reference_feature_slot_min_pink_fraction", 0.08)
        )
        if pink_fraction < minimum_pink:
            return None
        return replace(candidate, bright_fill=pink_fraction)

    def _detect_sift_slots(self, rgb: np.ndarray) -> list[BoxCandidate]:
        columns = int(self._config.get("reference_feature_slot_columns", 0))
        if columns <= 0:
            return []
        roi = self._config.get("provider_options", {}).get(
            "roi_norm", [0.0, 0.0, 1.0, 1.0]
        )
        height, width = rgb.shape[:2]
        x0 = max(0, min(width - 1, int(round(float(roi[0]) * width))))
        y0 = max(0, min(height - 1, int(round(float(roi[1]) * height))))
        x1 = max(x0 + 1, min(width, int(round(float(roi[2]) * width))))
        y1 = max(y0 + 1, min(height, int(round(float(roi[3]) * height))))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        face, reference_keypoints, reference_descriptors, reference_gray = (
            self._references[0]
        )
        slot_width = float(x1 - x0) / float(columns)
        padding = float(
            self._config.get("reference_feature_slot_padding_fraction", 0.18)
        )
        ratio = float(self._config.get("reference_feature_slot_ratio", 0.80))
        minimum_inliers = int(
            self._config.get("reference_feature_slot_min_inliers", 6)
        )
        policy_floor = max(
            float(self._config.get("min_detection_score", 0.6)),
            float(self._config.get("min_face_score", 0.6)),
        )
        result: list[BoxCandidate] = []
        for column in range(columns):
            pad_px = padding * slot_width
            sx0 = max(x0, int(round(x0 + column * slot_width - pad_px)))
            sx1 = min(x1, int(round(x0 + (column + 1) * slot_width + pad_px)))
            crop = gray[y0:y1, sx0:sx1]
            keypoints, descriptors = self._sift.detectAndCompute(crop, None)
            if descriptors is None:
                continue
            pairs = self._matcher.knnMatch(descriptors, reference_descriptors, k=2)
            good = [
                first
                for pair in pairs
                if len(pair) == 2
                for first, second in [pair]
                if first.distance < ratio * second.distance
            ]
            if len(good) < minimum_inliers:
                continue
            source = np.float32(
                [keypoints[match.queryIdx].pt for match in good]
            ).reshape(-1, 1, 2)
            target = np.float32(
                [reference_keypoints[match.trainIdx].pt for match in good]
            ).reshape(-1, 1, 2)
            try:
                homography, mask = cv2.findHomography(
                    source, target, cv2.RANSAC, 5.0, maxIters=5000
                )
                if homography is None or mask is None:
                    continue
                inlier_count = int(np.count_nonzero(mask))
                if inlier_count < minimum_inliers:
                    continue
                inverse = np.linalg.inv(homography)
                ref_h, ref_w = reference_gray.shape
                corners = np.float32(
                    [[[0, 0], [ref_w - 1, 0], [ref_w - 1, ref_h - 1], [0, ref_h - 1]]]
                )
                polygon = cv2.perspectiveTransform(corners, inverse)[0]
            except (cv2.error, np.linalg.LinAlgError):
                continue
            polygon += np.asarray([sx0, y0], dtype=np.float32)
            center_x = float(np.mean(polygon[:, 0]))
            slot_left = x0 + column * slot_width
            slot_right = slot_left + slot_width
            if not (slot_left <= center_x <= slot_right):
                continue
            quality = min(1.0, max(0.0, inlier_count - minimum_inliers) / 10.0)
            score = policy_floor + (1.0 - policy_floor) * quality
            candidate = self._slot_candidate(
                polygon=polygon,
                face=face,
                score=score,
                provider_suffix="front_similarity_sift",
                rgb=rgb,
            )
            if candidate is not None:
                result.append(candidate)
        return result

    def _detect_template_slots(self, rgb: np.ndarray) -> list[BoxCandidate]:
        columns = int(self._config.get("reference_feature_slot_columns", 0))
        if columns <= 0:
            return []
        roi = self._config.get("provider_options", {}).get(
            "roi_norm", [0.0, 0.0, 1.0, 1.0]
        )
        height, width = rgb.shape[:2]
        x0, y0, x1, y1 = (
            int(round(float(roi[0]) * width)),
            int(round(float(roi[1]) * height)),
            int(round(float(roi[2]) * width)),
            int(round(float(roi[3]) * height)),
        )
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        face, _, _, reference = self._references[0]
        crop_norm = self._config.get(
            "reference_feature_template_crop_norm", [0.513, 0.226, 0.992, 0.978]
        )
        ref_h, ref_w = reference.shape
        cx0, cy0, cx1, cy1 = (
            int(round(float(crop_norm[0]) * ref_w)),
            int(round(float(crop_norm[1]) * ref_h)),
            int(round(float(crop_norm[2]) * ref_w)),
            int(round(float(crop_norm[3]) * ref_h)),
        )
        motif = cv2.rotate(reference[cy0:cy1, cx0:cx1], cv2.ROTATE_90_CLOCKWISE)
        full_center_in_motif = np.asarray(
            [(cy1 - cy0 - 1) - (ref_h * 0.5 - cy0), ref_w * 0.5 - cx0],
            dtype=np.float64,
        )
        minimum_score = float(
            self._config.get("reference_feature_template_min_score", 0.50)
        )
        widths = self._config.get("reference_feature_template_width_px", [35, 70, 5])
        angle_limit = int(
            self._config.get("reference_feature_template_angle_limit_deg", 20)
        )
        angle_step = int(
            self._config.get("reference_feature_template_angle_step_deg", 5)
        )
        slot_width = float(x1 - x0) / float(columns)
        result: list[BoxCandidate] = []
        for column in range(columns):
            pad = 0.15 * slot_width
            sx0 = max(x0, int(round(x0 + column * slot_width - pad)))
            sx1 = min(x1, int(round(x0 + (column + 1) * slot_width + pad)))
            scene = gray[y0:y1, sx0:sx1]
            best: tuple[float, tuple[int, int], float, float, np.ndarray] | None = None
            for target_width in range(int(widths[0]), int(widths[1]) + 1, int(widths[2])):
                scale = float(target_width) / float(motif.shape[1])
                base = cv2.resize(motif, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                base_center = ((base.shape[1] - 1) * 0.5, (base.shape[0] - 1) * 0.5)
                for angle in range(-angle_limit, angle_limit + 1, angle_step):
                    matrix = cv2.getRotationMatrix2D(base_center, float(angle), 1.0)
                    rotated = cv2.warpAffine(
                        base, matrix, (base.shape[1], base.shape[0]),
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                    if rotated.shape[0] >= scene.shape[0] or rotated.shape[1] >= scene.shape[1]:
                        continue
                    response = cv2.matchTemplate(scene, rotated, cv2.TM_CCOEFF_NORMED)
                    _, score, _, location = cv2.minMaxLoc(response)
                    if best is None or float(score) > best[0]:
                        best = (float(score), location, scale, float(angle), matrix)
            if best is None or best[0] < minimum_score:
                continue
            raw_score, location, scale, angle, matrix = best
            point = np.append(full_center_in_motif * scale, 1.0)
            transformed_center = matrix @ point
            center = np.asarray(
                [sx0 + location[0] + transformed_center[0], y0 + location[1] + transformed_center[1]],
                dtype=np.float32,
            )
            face_size = (float(ref_h) * scale, float(ref_w) * scale)
            polygon = cv2.boxPoints((tuple(center), face_size, angle))
            policy_floor = max(
                float(self._config.get("min_detection_score", 0.6)),
                float(self._config.get("min_face_score", 0.6)),
            )
            evidence = min(1.0, (raw_score - minimum_score) / max(1.0 - minimum_score, 1e-6))
            score = policy_floor + (1.0 - policy_floor) * evidence
            candidate = self._slot_candidate(
                polygon=polygon,
                face=face,
                score=score,
                provider_suffix="front_similarity_template",
                rgb=rgb,
            )
            if candidate is not None:
                result.append(candidate)
        return result

    @staticmethod
    def _candidate_gray(rgb: np.ndarray, candidate: BoxCandidate) -> np.ndarray | None:
        points = np.round(np.asarray(candidate.polygon_px)).astype(np.int32)
        x0 = max(0, int(points[:, 0].min()))
        y0 = max(0, int(points[:, 1].min()))
        x1 = min(rgb.shape[1], int(points[:, 0].max()) + 1)
        y1 = min(rgb.shape[0], int(points[:, 1].max()) + 1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        crop = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        local = points - np.asarray([x0, y0], dtype=np.int32)
        mask = np.zeros(crop.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, local, 255)
        return np.where(mask > 0, crop, 127).astype(np.uint8)

    def _verify(
        self,
        rgb: np.ndarray,
        candidate: BoxCandidate,
    ) -> BoxCandidate:
        gray = self._candidate_gray(rgb, candidate)
        if gray is None:
            return replace(candidate, provider=self.name)
        keypoints, descriptors = self._sift.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 2:
            return replace(candidate, provider=self.name)

        pink_fraction = self._pink_fraction(rgb, candidate)
        best: tuple[int, int, float, float, Any] | None = None
        for face, reference_keypoints, reference_descriptors, _ in self._references:
            pairs = self._matcher.knnMatch(
                descriptors,
                reference_descriptors,
                k=2,
            )
            good = [
                first
                for pair in pairs
                if len(pair) == 2
                for first, second in [pair]
                if first.distance < self._ratio * second.distance
            ]
            inliers = 0
            scene_hull_fraction = 0.0
            reference_hull_fraction = 0.0
            if len(good) >= 4:
                source = np.float32(
                    [keypoints[match.queryIdx].pt for match in good]
                ).reshape(-1, 1, 2)
                target = np.float32(
                    [reference_keypoints[match.trainIdx].pt for match in good]
                ).reshape(-1, 1, 2)
                try:
                    _, mask = cv2.findHomography(
                        source,
                        target,
                        cv2.RANSAC,
                        self._ransac_threshold_px,
                    )
                except cv2.error:
                    mask = None
                if mask is not None:
                    selected = mask.ravel().astype(bool)
                    inliers = int(np.count_nonzero(selected))
                    scene_points = source.reshape(-1, 2)[selected]
                    reference_points = target.reshape(-1, 2)[selected]
                    scene_hull_fraction = self._hull_fraction(
                        scene_points,
                        float(gray.shape[0] * gray.shape[1]),
                    )
                    reference_image_area = float(
                        max(
                            keypoint.pt[0] for keypoint in reference_keypoints
                        )
                        * max(
                            keypoint.pt[1] for keypoint in reference_keypoints
                        )
                    )
                    reference_hull_fraction = self._hull_fraction(
                        reference_points,
                        reference_image_area,
                    )
            score = (
                inliers,
                len(good),
                scene_hull_fraction,
                reference_hull_fraction,
            )
            if best is None or score > best[:4]:
                best = (
                    inliers,
                    len(good),
                    scene_hull_fraction,
                    reference_hull_fraction,
                    face,
                )

        if best is None:
            return replace(candidate, provider=self.name)
        (
            inliers,
            matches,
            scene_hull_fraction,
            reference_hull_fraction,
            face,
        ) = best
        verified = (
            matches >= self._min_matches and inliers >= self._min_inliers
            and scene_hull_fraction >= self._minimum_scene_hull_fraction
            and (
                reference_hull_fraction
                >= self._minimum_reference_hull_fraction
            )
            and pink_fraction >= self._minimum_pink_fraction
        )
        excess_quality = float(
            np.mean(
                [
                    min(
                        1.0,
                        max(0.0, matches - self._min_matches)
                        / max(self._min_matches, 1),
                    ),
                    min(
                        1.0,
                        max(0.0, inliers - self._min_inliers)
                        / max(self._min_inliers, 1),
                    ),
                    min(
                        1.0,
                        max(
                            0.0,
                            scene_hull_fraction
                            - self._minimum_scene_hull_fraction,
                        )
                        / max(self._minimum_scene_hull_fraction, 1e-6),
                    ),
                    min(
                        1.0,
                        max(
                            0.0,
                            reference_hull_fraction
                            - self._minimum_reference_hull_fraction,
                        )
                        / max(self._minimum_reference_hull_fraction, 1e-6),
                    ),
                ]
            )
        )
        score_floor = max(
            float(self._config.get("min_detection_score", 0.6)),
            float(self._config.get("min_face_score", 0.6)),
        )
        face_score = min(
            1.0,
            score_floor + (1.0 - score_floor) * excess_quality,
        )
        return replace(
            candidate,
            score=face_score,
            provider=self.name,
            face_type=str(face.face_type) if verified else "unknown",
            face_score=face_score if verified else 0.0,
            reference_face_id=str(face.id) if verified else None,
            graspable=False,
            grasp_blockers=() if verified else ("face_unverified",),
        )

    def detect(self, rgb: np.ndarray) -> list[BoxCandidate]:
        slot_mode = str(self._config.get("reference_feature_slot_mode", ""))
        if slot_mode == "sift":
            slots = self._detect_sift_slots(rgb)
        elif slot_mode == "template":
            slots = self._detect_template_slots(rgb)
        else:
            slots = []
        if slot_mode:
            self._last_candidate_count = len(slots)
            self._last_verified_count = len(slots)
            self._last_motif_count = 0
            return slots
        motifs = self._detect_motifs(rgb)
        if motifs:
            self._last_candidate_count = len(motifs)
            self._last_verified_count = len(motifs)
            return motifs
        proposals = detect_cartons(rgb, self._config)
        evaluated = [self._verify(rgb, candidate) for candidate in proposals]
        verified = [
            candidate
            for candidate in evaluated
            if candidate.face_type != "unknown"
        ]
        verified.sort(key=lambda item: item.score, reverse=True)
        self._last_candidate_count = len(proposals)
        self._last_verified_count = len(verified)
        return verified

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": True,
            "mode": "reference_image_sift_ransac",
            "reference_count": len(self._references),
            "last_candidate_count": self._last_candidate_count,
            "last_verified_count": self._last_verified_count,
            "last_motif_count": self._last_motif_count,
            "motif_first": self._motif_helper is not None,
            "spatial_consistency_gate": True,
            "pink_material_gate": True,
            "fixed_pixel_area_gate": False,
        }


class UnavailableDetectorProvider:
    """Keep the console alive while refusing all detections."""

    def __init__(self, name: str, error: str) -> None:
        self.name = name
        self.error = error

    def detect(self, rgb: np.ndarray) -> list[BoxCandidate]:
        del rgb
        return []

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": False,
            "mode": "unavailable",
            "error": self.error,
        }


def _finite_positive(value: float) -> bool:
    return math.isfinite(float(value)) and float(value) > 0.0


def _reference_by_id(
    bank: ReferenceFaceBank,
    reference_face_id: str,
) -> Any | None:
    try:
        return bank.face_by_id(reference_face_id)
    except (KeyError, LookupError):
        return None


def apply_grasp_policy(
    candidate: BoxCandidate,
    *,
    face_bank: ReferenceFaceBank | None,
    config: dict[str, Any],
) -> BoxCandidate:
    """Recompute the 2-D face-level grasp gate without trusting a backend.

    ``graspable`` here only means that the proposed image point lies on a
    verified, suction-allowed carton face with enough 2-D clearance.  It is
    never equivalent to a depth-checked, reachable robot target.
    """

    blockers = [
        str(blocker)
        for blocker in candidate.grasp_blockers
        if str(blocker) not in POLICY_BLOCKERS
    ]

    minimum_detection = float(config.get("min_detection_score", 0.68))
    if not math.isfinite(float(candidate.score)) or candidate.score < minimum_detection:
        blockers.append("detection_score_low")

    geometry_values = (
        candidate.long_side_px,
        candidate.short_side_px,
        candidate.edge_clearance_px,
    )
    if (
        len(candidate.polygon_px) < 4
        or not all(_finite_positive(value) for value in geometry_values)
        or not all(
            math.isfinite(float(value))
            for point in candidate.polygon_px
            for value in point
        )
    ):
        blockers.append("candidate_geometry_invalid")

    cup_radius = float(config.get("cup_radius_px", 8.0))
    safety_margin = float(config.get("cup_clearance_margin_px", 4.0))
    configured_clearance = float(config.get("min_edge_clearance_px", 0.0))
    required_clearance = max(configured_clearance, cup_radius + safety_margin)
    if (
        not math.isfinite(float(candidate.edge_clearance_px))
        or candidate.edge_clearance_px < required_clearance
    ):
        blockers.append("suction_clearance_low")

    reference = None
    if face_bank is None:
        blockers.append("reference_bank_unavailable")
    elif (
        candidate.face_type == "unknown"
        or not candidate.reference_face_id
    ):
        blockers.append("face_unverified")
    else:
        reference = _reference_by_id(face_bank, candidate.reference_face_id)
        if reference is None:
            blockers.append("reference_face_missing")
        elif str(reference.face_type) != candidate.face_type:
            blockers.append("face_type_mismatch")

    minimum_face_score = float(config.get("min_face_score", 0.75))
    if (
        not math.isfinite(float(candidate.face_score))
        or candidate.face_score < minimum_face_score
    ):
        blockers.append("face_score_low")

    if reference is not None and not bool(reference.pick_allowed):
        blockers.append("reference_face_not_pick_allowed")

    blockers = list(dict.fromkeys(blockers))
    return replace(
        candidate,
        graspable=not blockers,
        grasp_blockers=tuple(blockers),
    )


class PolicyEnforcedDetectorProvider:
    """Serialize inference and enforce the central 2-D grasp policy."""

    def __init__(
        self,
        backend: DetectorProvider,
        *,
        face_bank: ReferenceFaceBank | None,
        config: dict[str, Any],
        face_bank_error: str | None = None,
    ) -> None:
        self._backend = backend
        self._face_bank = face_bank
        self._config = dict(config)
        self._face_bank_error = face_bank_error
        self._last_error: str | None = None
        self._lock = threading.Lock()
        self.name = backend.name

    def detect(self, rgb: np.ndarray) -> list[BoxCandidate]:
        with self._lock:
            try:
                raw = self._backend.detect(rgb)
                if not isinstance(raw, list):
                    raw = list(raw)
                if not all(isinstance(item, BoxCandidate) for item in raw):
                    raise TypeError("detector returned a non-BoxCandidate value")
                candidates = [
                    apply_grasp_policy(
                        candidate,
                        face_bank=self._face_bank,
                        config=self._config,
                    )
                    for candidate in raw
                ]
                candidates.sort(key=lambda item: item.score, reverse=True)
                self._last_error = None
                return candidates
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                return []

    def status(self) -> dict[str, Any]:
        backend_status = self._backend.status()
        bank = self._face_bank
        return {
            "name": self.name,
            "ok": bool(backend_status.get("ok")) and self._last_error is None,
            "backend": backend_status,
            "last_error": self._last_error,
            "reference_bank": {
                "ready": bank is not None,
                "bank_id": None if bank is None else bank.bank_id,
                "content_sha256": None if bank is None else bank.content_sha256,
                "error": self._face_bank_error,
            },
            "policy": {
                "min_detection_score": float(
                    self._config.get("min_detection_score", 0.68)
                ),
                "min_face_score": float(self._config.get("min_face_score", 0.75)),
                "unknown_is_graspable": False,
            },
        }


def _load_plugin(
    factory_reference: str,
    *,
    provider_name: str,
    options: dict[str, Any],
    config_dir: Path,
    face_bank: ReferenceFaceBank | None,
) -> DetectorProvider:
    module_name, separator, attribute = factory_reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("plugin_factory must use the form 'module.path:factory'")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    backend = factory(
        options=dict(options),
        config_dir=config_dir,
        face_bank=face_bank,
    )
    if not hasattr(backend, "detect") or not hasattr(backend, "status"):
        raise TypeError(f"detector plugin {provider_name!r} has an invalid interface")
    if not getattr(backend, "name", None):
        raise TypeError(f"detector plugin {provider_name!r} has no name")
    return backend


def create_detector_provider(
    config: dict[str, Any],
    *,
    config_dir: Path,
    face_bank: ReferenceFaceBank | None,
    face_bank_error: str | None = None,
) -> PolicyEnforcedDetectorProvider:
    """Create a backend lazily and always wrap it in the central policy."""

    provider_name = str(config.get("provider", "classical")).strip()
    try:
        if provider_name == "classical":
            backend: DetectorProvider = ClassicalDetectorProvider(config)
        elif provider_name == "reference_feature":
            backend = ReferenceFeatureDetectorProvider(
                config,
                face_bank,
                config_dir=config_dir.resolve(),
            )
        else:
            factory_reference = config.get("plugin_factory")
            if not isinstance(factory_reference, str) or not factory_reference.strip():
                raise ValueError(
                    f"provider {provider_name!r} requires detector.plugin_factory"
                )
            options = config.get("provider_options", {})
            if not isinstance(options, dict):
                raise ValueError("detector.provider_options must be an object")
            backend = _load_plugin(
                factory_reference.strip(),
                provider_name=provider_name,
                options=options,
                config_dir=config_dir.resolve(),
                face_bank=face_bank,
            )
    except Exception as exc:
        backend = UnavailableDetectorProvider(
            provider_name or "invalid",
            f"{type(exc).__name__}: {exc}",
        )
    return PolicyEnforcedDetectorProvider(
        backend,
        face_bank=face_bank,
        config=config,
        face_bank_error=face_bank_error,
    )
