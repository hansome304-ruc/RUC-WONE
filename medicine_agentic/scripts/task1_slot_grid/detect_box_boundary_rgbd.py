#!/usr/bin/env python3
"""Depth-assisted, read-only detector for the open shipping-box inner boundary.

RGB limits the search to brown cardboard.  Aligned depth is transformed to
the left-arm base frame, where the box bottom is selected as one large,
approximately constant-height component.  The four inner edges are fitted to
the boundary of that bottom-plane component, not to the visually longest
cardboard line.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from medicine_agentic.packaging_camera import create_camera
from medicine_agentic.task1_box import load_cam_to_left


PROJECT = Path("/home/ubuntu/RUC-WONE/medicine_agentic")
CONFIG_PATH = PROJECT / "configs/packaging_console.json"
OUTPUT_DIR = Path("/tmp/center-box-rgbd-boundary")
ROI_NORM = (0.27, 0.27, 0.69, 0.82)
PLANE_TOLERANCE_M = 0.012


def resolve_from_config(value: str, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()


def robust_line(independent: np.ndarray, dependent: np.ndarray) -> tuple[float, float]:
    keep = np.isfinite(independent) & np.isfinite(dependent)
    x = independent[keep].astype(np.float64)
    y = dependent[keep].astype(np.float64)
    if x.size < 20:
        raise RuntimeError("not enough envelope points for a boundary line")
    for _ in range(5):
        slope, intercept = np.polyfit(x, y, 1)
        residual = np.abs(y - (slope * x + intercept))
        cutoff = max(2.5, float(np.quantile(residual, 0.72)))
        next_keep = residual <= cutoff
        if np.count_nonzero(next_keep) < 20 or np.all(next_keep):
            break
        x, y = x[next_keep], y[next_keep]
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def ransac_near_horizontal(
    points: np.ndarray,
    *,
    residual_threshold_px: float = 2.25,
    expected_slope: float = 0.0,
    maximum_slope_delta: float = 0.065,
) -> tuple[tuple[float, float], np.ndarray]:
    """Select the largest line consensus near the current depth-edge direction."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 20:
        raise RuntimeError("not enough transition points for depth-line RANSAC")
    best_inliers: np.ndarray | None = None
    best_score: tuple[int, float, float] | None = None
    # Test well-separated point pairs.  Nearby pairs amplify one-pixel depth
    # quantization into an implausible slope.
    for first in range(values.shape[0] - 1):
        for second in range(first + 1, values.shape[0]):
            dx = float(values[second, 0] - values[first, 0])
            if abs(dx) < 0.35 * float(values[-1, 0] - values[0, 0]):
                continue
            slope = float((values[second, 1] - values[first, 1]) / dx)
            if abs(slope - expected_slope) > maximum_slope_delta:
                continue
            intercept = float(values[first, 1] - slope * values[first, 0])
            residuals = np.abs(values[:, 1] - (slope * values[:, 0] + intercept))
            inliers = residuals <= residual_threshold_px
            count = int(np.count_nonzero(inliers))
            if count < 20:
                continue
            median_residual = float(np.median(residuals[inliers]))
            score = (count, -median_residual, -abs(slope - expected_slope))
            if best_score is None or score > best_score:
                best_score = score
                best_inliers = inliers
    if best_inliers is None:
        raise RuntimeError("no coherent near-horizontal depth transition was found")
    slope, intercept = np.polyfit(values[best_inliers, 0], values[best_inliers, 1], 1)
    if abs(float(slope) - expected_slope) > maximum_slope_delta:
        raise RuntimeError("depth transition consensus exceeded the local top-edge slope gate")
    residuals = np.abs(values[:, 1] - (slope * values[:, 0] + intercept))
    final_inliers = residuals <= residual_threshold_px
    if np.count_nonzero(final_inliers) >= 20:
        slope, intercept = np.polyfit(
            values[final_inliers, 0], values[final_inliers, 1], 1
        )
        best_inliers = final_inliers
    return (float(slope), float(intercept)), best_inliers


def intersect(
    horizontal: tuple[float, float], vertical: tuple[float, float]
) -> np.ndarray:
    a, b = horizontal  # y = a*x + b
    c, d = vertical    # x = c*y + d
    matrix = np.asarray([[-a, 1.0], [1.0, -c]], dtype=np.float64)
    return np.linalg.solve(matrix, np.asarray([b, d], dtype=np.float64))


def line_from_points_as_horizontal(start: np.ndarray, end: np.ndarray) -> tuple[float, float]:
    slope = float((end[1] - start[1]) / max(end[0] - start[0], 1e-9))
    return slope, float(start[1] - slope * start[0])


def line_from_points_as_vertical(start: np.ndarray, end: np.ndarray) -> tuple[float, float]:
    slope = float((end[0] - start[0]) / max(end[1] - start[1], 1e-9))
    return slope, float(start[0] - slope * start[1])


def angle_delta_deg(first: float, second: float) -> float:
    """Smallest signed difference between two undirected line angles."""

    return float((first - second + 90.0) % 180.0 - 90.0)


def robust_floor_plane(points_left: np.ndarray) -> tuple[np.ndarray, float, dict[str, object]]:
    """Fit the actual box-bottom plane in the left-base coordinate frame."""

    points = np.asarray(points_left, dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] < 1000:
        raise RuntimeError("too few metric bottom points for a floor plane")
    if points.shape[0] > 25000:
        indices = np.random.default_rng(7).choice(points.shape[0], 25000, replace=False)
        points = points[indices]
    keep = np.ones(points.shape[0], dtype=bool)
    normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    center = np.median(points, axis=0)
    for _ in range(5):
        selected = points[keep]
        center = selected.mean(axis=0)
        _, _, vectors = np.linalg.svd(selected - center, full_matrices=False)
        normal = vectors[-1]
        if normal[2] < 0.0:
            normal = -normal
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        residual = np.abs((points - center) @ normal)
        cutoff = max(0.0025, float(np.quantile(residual, 0.82)))
        next_keep = residual <= cutoff
        if np.count_nonzero(next_keep) < 1000 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
    selected = points[keep]
    center = selected.mean(axis=0)
    _, _, vectors = np.linalg.svd(selected - center, full_matrices=False)
    normal = vectors[-1]
    if normal[2] < 0.0:
        normal = -normal
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    offset = -float(normal @ center)
    residual = np.abs(points @ normal + offset)
    inlier_residual = residual[keep]
    return normal, offset, {
        "normal_left_base": np.round(normal, 9).tolist(),
        "offset_m": round(offset, 9),
        "point_left_base_m": np.round(center, 7).tolist(),
        "sample_count": int(points.shape[0]),
        "inlier_count": int(np.count_nonzero(keep)),
        "rms_residual_mm": round(float(np.sqrt(np.mean(inlier_residual**2))) * 1000.0, 4),
        "maximum_inlier_residual_mm": round(float(np.max(inlier_residual)) * 1000.0, 4),
    }


def refine_front_edge_with_rgb(
    bgr: np.ndarray,
    coarse_quad: np.ndarray,
    left_z: np.ndarray,
    valid: np.ndarray,
    floor_z: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Snap the coarse depth front edge to a nearby visible cardboard crease.

    The depth component identifies the correct floor but can have missing
    pixels near a corner.  Only RGB lines close to that depth edge are allowed;
    therefore the much clearer outer flap edge cannot replace the inner fold.
    """

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 32, 100)
    coarse_start = coarse_quad[3]
    coarse_end = coarse_quad[2]
    coarse_vector = coarse_end - coarse_start
    coarse_length = float(np.linalg.norm(coarse_vector))
    coarse_unit = coarse_vector / max(coarse_length, 1e-9)
    coarse_normal = np.asarray([-coarse_unit[1], coarse_unit[0]])
    coarse_angle = float(np.degrees(np.arctan2(coarse_vector[1], coarse_vector[0])))
    center = coarse_quad.mean(axis=0)
    image_height, image_width = left_z.shape
    # Crop Canny to a narrow quadrilateral around the depth transition.  This
    # prevents a collinear medicine-carton edge outside the shipping box from
    # winning merely because it is longer.
    band_start = coarse_start - 22.0 * coarse_unit
    band_end = coarse_end + 22.0 * coarse_unit
    band = np.asarray(
        [
            band_start - 23.0 * coarse_normal,
            band_end - 23.0 * coarse_normal,
            band_end + 23.0 * coarse_normal,
            band_start + 23.0 * coarse_normal,
        ],
        dtype=np.float64,
    )
    band_mask = np.zeros(edges.shape, dtype=np.uint8)
    cv2.fillConvexPoly(band_mask, np.rint(band).astype(np.int32), 255)
    edges = cv2.bitwise_and(edges, band_mask)
    raw = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=32,
        minLineLength=65,
        maxLineGap=28,
    )
    candidates: list[dict[str, object]] = []
    if raw is not None:
        for values in np.asarray(raw).reshape(-1, 4):
            start = values[:2].astype(np.float64)
            end = values[2:].astype(np.float64)
            if end[0] < start[0]:
                start, end = end, start
            vector = end - start
            length = float(np.linalg.norm(vector))
            angle = float(np.degrees(np.arctan2(vector[1], vector[0])))
            angle_delta = angle_delta_deg(angle, coarse_angle)
            if length < max(65.0, 0.28 * coarse_length) or abs(angle_delta) > 8.0:
                continue
            midpoint = (start + end) / 2.0
            signed_distance = float((midpoint - coarse_start) @ coarse_normal)
            perpendicular_distance = abs(signed_distance)
            projection_start = float((start - coarse_start) @ coarse_unit)
            projection_end = float((end - coarse_start) @ coarse_unit)
            overlap = max(
                0.0,
                min(max(projection_start, projection_end), coarse_length)
                - max(min(projection_start, projection_end), 0.0),
            )
            overlap_ratio = overlap / max(coarse_length, 1e-9)
            if perpendicular_distance > 12.0 or overlap_ratio < 0.28:
                continue
            inside_values: list[float] = []
            outside_values: list[float] = []
            for fraction in np.linspace(0.14, 0.86, 14):
                point = start + fraction * (end - start)
                inward = center - point
                inward /= max(float(np.linalg.norm(inward)), 1e-9)
                for offset, target in ((10.0, inside_values), (-14.0, outside_values)):
                    pixel = np.rint(point + offset * inward).astype(int)
                    x_i, y_i = int(pixel[0]), int(pixel[1])
                    if 0 <= x_i < image_width and 0 <= y_i < image_height and valid[y_i, x_i]:
                        target.append(float(left_z[y_i, x_i]))
            if len(inside_values) < 8 or len(outside_values) < 8:
                continue
            inside_median = float(np.median(inside_values))
            outside_median = float(np.median(outside_values))
            inside_floor_error = abs(inside_median - floor_z)
            depth_jump = outside_median - inside_median
            if inside_floor_error > 0.018 or depth_jump < 0.012:
                continue
            # Prefer a long crease close to the depth transition.  Angle is a
            # weak tie-breaker only; perspective is allowed to tilt the edge.
            score = (
                length
                + 35.0 * overlap_ratio
                - 6.0 * perpendicular_distance
                - abs(angle_delta)
                + 160.0 * depth_jump
                + 8.0 * signed_distance
            )
            candidates.append(
                {
                    "start": start,
                    "end": end,
                    "length_px": length,
                    "angle_deg": angle,
                    "angle_delta_from_depth_deg": angle_delta,
                    "distance_to_depth_edge_px": perpendicular_distance,
                    "signed_outward_distance_px": signed_distance,
                    "overlap_ratio": overlap_ratio,
                    "inside_floor_error_m": inside_floor_error,
                    "depth_jump_m": depth_jump,
                    "score": score,
                }
            )

    if not candidates:
        return coarse_quad.copy(), {
            "applied": False,
            "reason": "no RGB crease close enough to the depth front edge",
            "candidate_count": 0,
        }

    maximum_outward = max(
        float(item["signed_outward_distance_px"]) for item in candidates
    )
    outer_layer = [
        item
        for item in candidates
        if float(item["signed_outward_distance_px"]) >= maximum_outward - 2.0
    ]
    seed = max(outer_layer, key=lambda item: float(item["score"]))
    seed_fit = line_from_points_as_horizontal(
        np.asarray(seed["start"]), np.asarray(seed["end"])
    )
    supporting: list[dict[str, object]] = []
    support_points: list[np.ndarray] = []
    # Keep the fitted front edge on the same physical crease as the selected
    # outermost seed. Mixing the nearby inner fold back in here can pull the
    # result several pixels inward even though the seed itself is correct.
    for candidate in outer_layer:
        midpoint = (np.asarray(candidate["start"]) + np.asarray(candidate["end"])) / 2.0
        seed_y = seed_fit[0] * midpoint[0] + seed_fit[1]
        if abs(float(midpoint[1] - seed_y)) <= 3.0:
            supporting.append(candidate)
            start = np.asarray(candidate["start"], dtype=np.float64)
            end = np.asarray(candidate["end"], dtype=np.float64)
            support_points.extend(
                start + fraction * (end - start)
                for fraction in np.linspace(0.0, 1.0, 12)
            )
    points = np.asarray(support_points, dtype=np.float64)
    if points.shape[0] >= 20:
        front_fit = robust_line(points[:, 0], points[:, 1])
    else:
        slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
        front_fit = float(slope), float(intercept)
    visible_left_x = max(float(points[:, 0].min()), float(coarse_start[0] - 25.0))
    visible_right_x = min(float(points[:, 0].max()), float(coarse_end[0] + 25.0))
    if visible_right_x - visible_left_x < 0.68 * coarse_length:
        return coarse_quad.copy(), {
            "applied": False,
            "reason": "depth-validated front crease did not span both corners",
            "candidate_count": len(candidates),
        }
    refined = coarse_quad.copy()
    refined[3] = [
        visible_left_x,
        front_fit[0] * visible_left_x + front_fit[1],
    ]
    refined[2] = [
        visible_right_x,
        front_fit[0] * visible_right_x + front_fit[1],
    ]
    return refined, {
        "applied": True,
        "candidate_count": len(candidates),
        "outer_layer_candidate_count": len(outer_layer),
        "maximum_valid_outward_distance_px": round(maximum_outward, 3),
        "supporting_line_count": len(supporting),
        "selected_line_px": np.rint(
            np.concatenate([np.asarray(seed["start"]), np.asarray(seed["end"])])
        ).astype(int).tolist(),
        "selected_length_px": round(float(seed["length_px"]), 2),
        "selected_angle_deg": round(float(seed["angle_deg"]), 3),
        "selected_distance_to_depth_edge_px": round(
            float(seed["distance_to_depth_edge_px"]), 2
        ),
        "selected_signed_outward_distance_px": round(
            float(seed["signed_outward_distance_px"]), 2
        ),
        "selected_depth_jump_m": round(float(seed["depth_jump_m"]), 6),
        "selected_inside_floor_error_m": round(
            float(seed["inside_floor_error_m"]), 6
        ),
        "front_fit_y_equals_ax_plus_b": [round(front_fit[0], 8), round(front_fit[1], 4)],
        "visible_crease_x_span_px": [round(visible_left_x, 2), round(visible_right_x, 2)],
    }


def refine_back_edge_with_depth(
    coarse_quad: np.ndarray,
    left_z: np.ndarray,
    valid: np.ndarray,
    floor_z: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit the visible back inner edge from wall-to-floor depth transitions."""

    start = coarse_quad[0]
    end = coarse_quad[1]
    width = float(end[0] - start[0])
    expected_slope = float((end[1] - start[1]) / max(width, 1e-9))
    image_height, image_width = left_z.shape
    transition_points: list[list[float]] = []
    transition_jumps: list[float] = []
    # Avoid the two corners, where side-wall depth and alignment holes mix
    # with the back-wall transition.
    for x in np.arange(start[0] + 0.08 * width, end[0] - 0.08 * width, 2.0):
        fraction = float((x - start[0]) / max(width, 1e-9))
        coarse_y = float(start[1] + fraction * (end[1] - start[1]))
        best: tuple[float, float, float] | None = None
        x_i = int(round(x))
        for y_i in range(int(round(coarse_y)) - 24, int(round(coarse_y)) + 25):
            if x_i < 3 or x_i >= image_width - 3 or y_i < 13 or y_i >= image_height - 13:
                continue
            outside_window = left_z[y_i - 12 : y_i - 3, x_i - 2 : x_i + 3]
            outside_valid = valid[y_i - 12 : y_i - 3, x_i - 2 : x_i + 3]
            inside_window = left_z[y_i + 3 : y_i + 12, x_i - 2 : x_i + 3]
            inside_valid = valid[y_i + 3 : y_i + 12, x_i - 2 : x_i + 3]
            if np.count_nonzero(outside_valid) < 25 or np.count_nonzero(inside_valid) < 25:
                continue
            outside_median = float(np.median(outside_window[outside_valid]))
            inside_median = float(np.median(inside_window[inside_valid]))
            inside_error = abs(inside_median - floor_z)
            jump = outside_median - inside_median
            if inside_error > 0.018 or jump < 0.015:
                continue
            score = jump - 1.8 * inside_error - 0.00035 * abs(float(y_i) - coarse_y)
            if best is None or score > best[0]:
                best = (score, float(y_i), jump)
        if best is not None:
            transition_points.append([float(x), best[1]])
            transition_jumps.append(best[2])

    if len(transition_points) < 24:
        return coarse_quad.copy(), {
            "applied": False,
            "reason": "too few valid wall-to-floor depth transitions",
            "transition_count": len(transition_points),
        }
    points = np.asarray(transition_points, dtype=np.float64)
    try:
        back_fit, inliers = ransac_near_horizontal(
            points,
            expected_slope=expected_slope,
            maximum_slope_delta=0.075,
        )
    except RuntimeError as exc:
        return coarse_quad.copy(), {
            "applied": False,
            "reason": str(exc),
            "transition_count": len(transition_points),
        }

    left_fit = line_from_points_as_vertical(coarse_quad[0], coarse_quad[3])
    right_fit = line_from_points_as_vertical(coarse_quad[1], coarse_quad[2])
    refined = coarse_quad.copy()
    refined[0] = intersect(back_fit, left_fit)
    refined[1] = intersect(back_fit, right_fit)
    return refined, {
        "applied": True,
        "transition_count": len(transition_points),
        "inlier_count": int(np.count_nonzero(inliers)),
        "median_wall_minus_floor_m": round(float(np.median(transition_jumps)), 6),
        "back_fit_y_equals_ax_plus_b": [round(back_fit[0], 8), round(back_fit[1], 4)],
        "transition_y_range_px": [round(float(points[:, 1].min()), 2), round(float(points[:, 1].max()), 2)],
    }


def refine_back_edge_with_rgbd(
    bgr: np.ndarray,
    depth_quad: np.ndarray,
    left_z: np.ndarray,
    valid: np.ndarray,
    floor_z: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Use a visible crease for pixel position and depth for semantic validation."""

    start, end = depth_quad[0], depth_quad[1]
    vector = end - start
    length = float(np.linalg.norm(vector))
    unit = vector / max(length, 1e-9)
    normal = np.asarray([-unit[1], unit[0]])
    coarse_angle = float(np.degrees(np.arctan2(vector[1], vector[0])))
    center = depth_quad.mean(axis=0)
    band_start = start - 20.0 * unit
    band_end = end + 20.0 * unit
    band = np.asarray(
        [
            band_start - 22.0 * normal,
            band_end - 22.0 * normal,
            band_end + 22.0 * normal,
            band_start + 22.0 * normal,
        ]
    )
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 32, 100)
    band_mask = np.zeros(edges.shape, dtype=np.uint8)
    cv2.fillConvexPoly(band_mask, np.rint(band).astype(np.int32), 255)
    edges = cv2.bitwise_and(edges, band_mask)
    raw = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=30, minLineLength=65, maxLineGap=25
    )
    candidates: list[dict[str, object]] = []
    image_height, image_width = left_z.shape
    if raw is not None:
        for values in np.asarray(raw).reshape(-1, 4):
            p0 = values[:2].astype(np.float64)
            p1 = values[2:].astype(np.float64)
            if p1[0] < p0[0]:
                p0, p1 = p1, p0
            line_vector = p1 - p0
            line_length = float(np.linalg.norm(line_vector))
            angle = float(np.degrees(np.arctan2(line_vector[1], line_vector[0])))
            angle_delta = angle_delta_deg(angle, coarse_angle)
            if line_length < max(65.0, 0.32 * length) or abs(angle_delta) > 7.0:
                continue
            midpoint = (p0 + p1) / 2.0
            distance = abs(float((midpoint - start) @ normal))
            projection0 = float((p0 - start) @ unit)
            projection1 = float((p1 - start) @ unit)
            overlap = max(
                0.0,
                min(max(projection0, projection1), length)
                - max(min(projection0, projection1), 0.0),
            )
            overlap_ratio = overlap / max(length, 1e-9)
            if distance > 20.0 or overlap_ratio < 0.32:
                continue
            line_fit = line_from_points_as_horizontal(p0, p1)
            inside_values: list[float] = []
            outside_values: list[float] = []
            for fraction in np.linspace(0.15, 0.85, 14):
                point = p0 + fraction * (p1 - p0)
                inward = center - point
                inward /= max(float(np.linalg.norm(inward)), 1e-9)
                for offset, target in ((10.0, inside_values), (-10.0, outside_values)):
                    pixel = np.rint(point + offset * inward).astype(int)
                    x_i, y_i = int(pixel[0]), int(pixel[1])
                    if 0 <= x_i < image_width and 0 <= y_i < image_height and valid[y_i, x_i]:
                        target.append(float(left_z[y_i, x_i]))
            if len(inside_values) < 8 or len(outside_values) < 8:
                continue
            inside_median = float(np.median(inside_values))
            outside_median = float(np.median(outside_values))
            inside_error = abs(inside_median - floor_z)
            jump = outside_median - inside_median
            if inside_error > 0.018 or jump < 0.015:
                continue
            score = (
                line_length
                + 45.0 * overlap_ratio
                - 2.5 * distance
                + 180.0 * jump
                - abs(angle_delta)
            )
            candidates.append(
                {
                    "start": p0,
                    "end": p1,
                    "fit": line_fit,
                    "length_px": line_length,
                    "angle_deg": angle,
                    "angle_delta_from_depth_deg": angle_delta,
                    "distance_px": distance,
                    "overlap_ratio": overlap_ratio,
                    "depth_jump_m": jump,
                    "inside_floor_error_m": inside_error,
                    "score": score,
                }
            )
    if not candidates:
        return depth_quad.copy(), {
            "applied": False,
            "reason": "no visible top crease passed the depth transition gate",
            "candidate_count": 0,
        }
    seed = max(candidates, key=lambda item: float(item["score"]))
    seed_fit = seed["fit"]
    support = []
    support_points: list[np.ndarray] = []
    for candidate in candidates:
        midpoint = (np.asarray(candidate["start"]) + np.asarray(candidate["end"])) / 2.0
        if abs(float(midpoint[1] - (seed_fit[0] * midpoint[0] + seed_fit[1]))) <= 4.0:
            support.append(candidate)
            p0 = np.asarray(candidate["start"], dtype=np.float64)
            p1 = np.asarray(candidate["end"], dtype=np.float64)
            support_points.extend(
                p0 + fraction * (p1 - p0) for fraction in np.linspace(0.0, 1.0, 12)
            )
    points = np.asarray(support_points, dtype=np.float64)
    top_fit = robust_line(points[:, 0], points[:, 1]) if points.shape[0] >= 20 else seed_fit
    visible_left_x = max(float(points[:, 0].min()), float(start[0] - 24.0))
    visible_right_x = min(float(points[:, 0].max()), float(end[0] + 24.0))
    if visible_right_x - visible_left_x < 0.68 * length:
        return depth_quad.copy(), {
            "applied": False,
            "reason": "depth-validated visible top crease did not span both corners",
            "candidate_count": len(candidates),
        }
    # The visible crease fixes line position and orientation.  Extend that
    # validated line to the depth-derived side extents so an occluded/weak
    # corner does not shorten the inner opening.
    left_x = float(start[0])
    right_x = float(end[0])
    refined = depth_quad.copy()
    refined[0] = [left_x, top_fit[0] * left_x + top_fit[1]]
    refined[1] = [right_x, top_fit[0] * right_x + top_fit[1]]
    return refined, {
        "applied": True,
        "candidate_count": len(candidates),
        "supporting_line_count": len(support),
        "selected_line_px": np.rint(
            np.concatenate([np.asarray(seed["start"]), np.asarray(seed["end"])])
        ).astype(int).tolist(),
        "selected_depth_jump_m": round(float(seed["depth_jump_m"]), 6),
        "selected_inside_floor_error_m": round(float(seed["inside_floor_error_m"]), 6),
        "top_fit_y_equals_ax_plus_b": [round(top_fit[0], 8), round(top_fit[1], 4)],
        "visible_crease_x_span_px": [round(visible_left_x, 2), round(visible_right_x, 2)],
        "extended_depth_x_span_px": [round(left_x, 2), round(right_x, 2)],
    }


def refine_side_edge_with_rgbd(
    bgr: np.ndarray,
    coarse_quad: np.ndarray,
    left_z: np.ndarray,
    valid: np.ndarray,
    floor_z: float,
    *,
    side: str,
) -> tuple[tuple[float, float] | None, dict[str, object]]:
    """Fit one visible floor-to-side-wall crease with a depth transition gate."""

    if side == "left":
        start, end = coarse_quad[0], coarse_quad[3]
    elif side == "right":
        start, end = coarse_quad[1], coarse_quad[2]
    else:
        raise ValueError("side must be left or right")
    vector = end - start
    length = float(np.linalg.norm(vector))
    unit = vector / max(length, 1e-9)
    normal = np.asarray([-unit[1], unit[0]])
    coarse_angle = float(np.degrees(np.arctan2(vector[1], vector[0])))
    center = coarse_quad.mean(axis=0)
    midpoint_coarse = (start + end) / 2.0
    outward = midpoint_coarse - center
    outward /= max(float(np.linalg.norm(outward)), 1e-9)

    band_start = start - 18.0 * unit
    band_end = end + 18.0 * unit
    band = np.asarray(
        [
            band_start - 22.0 * normal,
            band_end - 22.0 * normal,
            band_end + 22.0 * normal,
            band_start + 22.0 * normal,
        ],
        dtype=np.float64,
    )
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 96)
    band_mask = np.zeros(edges.shape, dtype=np.uint8)
    cv2.fillConvexPoly(band_mask, np.rint(band).astype(np.int32), 255)
    edges = cv2.bitwise_and(edges, band_mask)
    raw = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=28, minLineLength=60, maxLineGap=24
    )
    candidates: list[dict[str, object]] = []
    image_height, image_width = left_z.shape
    if raw is not None:
        for values in np.asarray(raw).reshape(-1, 4):
            p0 = values[:2].astype(np.float64)
            p1 = values[2:].astype(np.float64)
            if float((p1 - p0) @ unit) < 0.0:
                p0, p1 = p1, p0
            line_vector = p1 - p0
            line_length = float(np.linalg.norm(line_vector))
            angle = float(np.degrees(np.arctan2(line_vector[1], line_vector[0])))
            delta = angle_delta_deg(angle, coarse_angle)
            if line_length < max(60.0, 0.32 * length) or abs(delta) > 9.0:
                continue
            midpoint = (p0 + p1) / 2.0
            signed_outward = float((midpoint - midpoint_coarse) @ outward)
            distance = abs(float((midpoint - start) @ normal))
            projection0 = float((p0 - start) @ unit)
            projection1 = float((p1 - start) @ unit)
            overlap = max(
                0.0,
                min(max(projection0, projection1), length)
                - max(min(projection0, projection1), 0.0),
            )
            overlap_ratio = overlap / max(length, 1e-9)
            if distance > 18.0 or overlap_ratio < 0.30:
                continue
            inside_values: list[float] = []
            outside_values: list[float] = []
            for fraction in np.linspace(0.16, 0.84, 14):
                point = p0 + fraction * (p1 - p0)
                inward = center - point
                inward /= max(float(np.linalg.norm(inward)), 1e-9)
                for offset, target in ((10.0, inside_values), (-12.0, outside_values)):
                    pixel = np.rint(point + offset * inward).astype(int)
                    x_i, y_i = int(pixel[0]), int(pixel[1])
                    if 0 <= x_i < image_width and 0 <= y_i < image_height and valid[y_i, x_i]:
                        target.append(float(left_z[y_i, x_i]))
            if len(inside_values) < 8 or len(outside_values) < 8:
                continue
            inside_median = float(np.median(inside_values))
            outside_median = float(np.median(outside_values))
            inside_error = abs(inside_median - floor_z)
            jump = outside_median - inside_median
            if inside_error > 0.020 or jump < 0.010:
                continue
            score = (
                line_length
                + 40.0 * overlap_ratio
                - 3.5 * distance
                - abs(delta)
                + 150.0 * jump
                + 5.0 * signed_outward
            )
            candidates.append(
                {
                    "start": p0,
                    "end": p1,
                    "length_px": line_length,
                    "angle_deg": angle,
                    "angle_delta_from_depth_deg": delta,
                    "distance_px": distance,
                    "signed_outward_distance_px": signed_outward,
                    "overlap_ratio": overlap_ratio,
                    "inside_floor_error_m": inside_error,
                    "depth_jump_m": jump,
                    "score": score,
                }
            )
    if not candidates:
        return None, {
            "applied": False,
            "reason": f"no {side} visible crease passed the depth transition gate",
            "candidate_count": 0,
        }
    seed = max(candidates, key=lambda item: float(item["score"]))
    seed_fit = line_from_points_as_vertical(
        np.asarray(seed["start"]), np.asarray(seed["end"])
    )
    support: list[dict[str, object]] = []
    support_points: list[np.ndarray] = []
    for candidate in candidates:
        point = (np.asarray(candidate["start"]) + np.asarray(candidate["end"])) / 2.0
        seed_x = seed_fit[0] * point[1] + seed_fit[1]
        if abs(float(point[0] - seed_x)) <= 4.0:
            support.append(candidate)
            p0 = np.asarray(candidate["start"], dtype=np.float64)
            p1 = np.asarray(candidate["end"], dtype=np.float64)
            support_points.extend(
                p0 + fraction * (p1 - p0) for fraction in np.linspace(0.0, 1.0, 12)
            )
    points = np.asarray(support_points, dtype=np.float64)
    if points.shape[0] >= 20:
        side_fit = robust_line(points[:, 1], points[:, 0])
    else:
        side_fit = seed_fit
    return side_fit, {
        "applied": True,
        "side": side,
        "candidate_count": len(candidates),
        "supporting_line_count": len(support),
        "selected_line_px": np.rint(
            np.concatenate([np.asarray(seed["start"]), np.asarray(seed["end"])])
        ).astype(int).tolist(),
        "selected_depth_jump_m": round(float(seed["depth_jump_m"]), 6),
        "selected_inside_floor_error_m": round(float(seed["inside_floor_error_m"]), 6),
        "selected_signed_outward_distance_px": round(
            float(seed["signed_outward_distance_px"]), 3
        ),
        "side_fit_x_equals_cy_plus_d": [round(side_fit[0], 8), round(side_fit[1], 4)],
    }


def fit_quad_from_component(component: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(component)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    width = x_max - x_min + 1
    height = y_max - y_min + 1

    column_x: list[float] = []
    column_top: list[float] = []
    column_bottom: list[float] = []
    for x in range(x_min, x_max + 1):
        column_y = np.flatnonzero(component[:, x])
        if column_y.size >= max(25, int(0.28 * height)):
            column_x.append(float(x))
            column_top.append(float(column_y.min()))
            column_bottom.append(float(column_y.max()))
    column_x_a = np.asarray(column_x)
    middle_columns = (
        (column_x_a >= x_min + 0.08 * width)
        & (column_x_a <= x_max - 0.08 * width)
    )
    top_fit = robust_line(
        column_x_a[middle_columns], np.asarray(column_top)[middle_columns]
    )
    bottom_fit = robust_line(
        column_x_a[middle_columns], np.asarray(column_bottom)[middle_columns]
    )

    row_y: list[float] = []
    row_left: list[float] = []
    row_right: list[float] = []
    for y in range(y_min, y_max + 1):
        row_x = np.flatnonzero(component[y])
        if row_x.size >= max(35, int(0.32 * width)):
            row_y.append(float(y))
            row_left.append(float(row_x.min()))
            row_right.append(float(row_x.max()))
    row_y_a = np.asarray(row_y)
    middle_rows = (
        (row_y_a >= y_min + 0.08 * height)
        & (row_y_a <= y_max - 0.08 * height)
    )
    left_fit = robust_line(row_y_a[middle_rows], np.asarray(row_left)[middle_rows])
    right_fit = robust_line(row_y_a[middle_rows], np.asarray(row_right)[middle_rows])

    return np.asarray(
        [
            intersect(top_fit, left_fit),
            intersect(top_fit, right_fit),
            intersect(bottom_fit, right_fit),
            intersect(bottom_fit, left_fit),
        ],
        dtype=np.float64,
    )


def edge_height_checks(
    quad: np.ndarray,
    left_z: np.ndarray,
    valid: np.ndarray,
    floor_z: float,
) -> list[dict[str, object]]:
    center = quad.mean(axis=0)
    names = ("back", "right", "front", "left")
    checks: list[dict[str, object]] = []
    image_height, image_width = left_z.shape
    for name, start, end in zip(names, quad, np.roll(quad, -1, axis=0), strict=True):
        edge = end - start
        samples_inside: list[float] = []
        samples_outside: list[float] = []
        for t in np.linspace(0.16, 0.84, 16):
            point = start + t * edge
            inward = center - point
            inward /= max(float(np.linalg.norm(inward)), 1e-9)
            for offset, target in ((10.0, samples_inside), (-14.0, samples_outside)):
                px = np.rint(point + offset * inward).astype(int)
                x, y = int(px[0]), int(px[1])
                if 0 <= x < image_width and 0 <= y < image_height and valid[y, x]:
                    target.append(float(left_z[y, x]))
        inside_median = float(np.median(samples_inside)) if samples_inside else None
        outside_median = float(np.median(samples_outside)) if samples_outside else None
        inside_error = None if inside_median is None else abs(inside_median - floor_z)
        height_jump = (
            None
            if inside_median is None or outside_median is None
            else outside_median - inside_median
        )
        checks.append(
            {
                "edge": name,
                "inside_samples": len(samples_inside),
                "outside_samples": len(samples_outside),
                "inside_median_z_m": inside_median,
                "outside_median_z_m": outside_median,
                "inside_floor_error_m": inside_error,
                "outside_minus_inside_m": height_jump,
                "valid": bool(
                    inside_error is not None
                    and inside_error <= 0.018
                    and height_jump is not None
                    and height_jump >= 0.012
                ),
            }
        )
    return checks


def draw_result(
    bgr: np.ndarray,
    quad: np.ndarray,
    component: np.ndarray,
    floor_z: float,
    accepted: bool,
) -> np.ndarray:
    output = bgr.copy()
    tint = np.zeros_like(output)
    tint[component] = (255, 120, 40)
    output = cv2.addWeighted(output, 1.0, tint, 0.28, 0.0)
    quad_i = np.rint(quad).astype(np.int32)
    color = (70, 255, 120) if accepted else (40, 80, 255)
    cv2.polylines(output, [quad_i], True, color, 4, cv2.LINE_AA)
    for index, point in enumerate(quad_i, start=1):
        cv2.circle(output, tuple(point), 7, (40, 190, 255), -1, cv2.LINE_AA)
        cv2.putText(
            output,
            f"P{index}",
            (int(point[0]) + 8, int(point[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (40, 190, 255),
            2,
            cv2.LINE_AA,
        )
    label = f"RGB-D INNER BOUNDARY | floor Z={floor_z:+.3f} m | {'PASS' if accepted else 'REJECT'}"
    cv2.putText(
        output,
        label,
        (max(12, int(quad[:, 0].min()) - 20), max(28, int(quad[:, 1].min()) - 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )
    return output


def main(
    *,
    camera=None,
    config: dict[str, object] | None = None,
    write_outputs: bool = True,
    emit_json: bool = True,
) -> tuple[dict[str, object], np.ndarray]:
    """Detect one frame, optionally reusing an already-open shared camera."""

    if write_outputs:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if config is None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    camera_cfg = config["camera"]
    if camera is None:
        camera = create_camera(camera_cfg, config_dir=CONFIG_PATH.parent)
    frame = camera.capture()
    if frame.depth_z16 is None:
        raise RuntimeError("the synchronized Center frame has no aligned depth")

    bgr = frame.bgr
    depth_m = frame.depth_z16.astype(np.float64) * float(frame.depth_scale_m)
    height, width = depth_m.shape
    intrinsics = np.asarray(camera_cfg["intrinsics"], dtype=np.float64)
    transform = load_cam_to_left(
        resolve_from_config(camera_cfg["cam_to_left_path"], CONFIG_PATH.parent)
    )
    v, u = np.indices((height, width), dtype=np.float64)
    z = depth_m
    x_camera = (u - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y_camera = (v - intrinsics[1, 2]) * z / intrinsics[1, 1]
    left_x = (
        transform[0, 0] * x_camera
        + transform[0, 1] * y_camera
        + transform[0, 2] * z
        + transform[0, 3]
    )
    left_y = (
        transform[1, 0] * x_camera
        + transform[1, 1] * y_camera
        + transform[1, 2] * z
        + transform[1, 3]
    )
    left_z = (
        transform[2, 0] * x_camera
        + transform[2, 1] * y_camera
        + transform[2, 2] * z
        + transform[2, 3]
    )
    valid = (z >= 0.25) & (z <= 1.6) & np.isfinite(left_z)

    x0 = int(round(ROI_NORM[0] * width))
    y0 = int(round(ROI_NORM[1] * height))
    x1 = int(round(ROI_NORM[2] * width))
    y1 = int(round(ROI_NORM[3] * height))
    roi = np.zeros_like(valid)
    roi[y0:y1, x0:x1] = True
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    brown = cv2.inRange(
        hsv,
        np.asarray([4, 22, 30], dtype=np.uint8),
        np.asarray([36, 245, 250], dtype=np.uint8),
    ) > 0
    candidate_pixels = valid & roi & brown
    values = left_z[candidate_pixels]
    if values.size < 1000:
        raise RuntimeError("too few brown RGB-D pixels in the search ROI")

    # Evaluate the most populated 3 mm height bins.  A wall only intersects a
    # height slice as a thin band; the box bottom produces a large rectangle.
    low, high = np.quantile(values, [0.005, 0.995])
    bin_edges = np.arange(float(low) - 0.003, float(high) + 0.006, 0.003)
    counts, _ = np.histogram(values, bins=bin_edges)
    ranked_bins = np.argsort(counts)[::-1][:40]
    best: dict[str, object] | None = None
    trials: list[dict[str, object]] = []
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for bin_index in ranked_bins:
        plane_z = float((bin_edges[bin_index] + bin_edges[bin_index + 1]) / 2.0)
        mask = (
            candidate_pixels & (np.abs(left_z - plane_z) <= PLANE_TOLERANCE_M)
        ).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for label in range(1, count):
            bx, by, bw, bh, area = (int(value) for value in stats[label])
            if area < 12000 or bw < 150 or bh < 110:
                continue
            component = labels == label
            contours, _ = cv2.findContours(
                component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contour = max(contours, key=cv2.contourArea)
            hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
            rectangularity = float(area / hull_area)
            fill = float(area / max(bw * bh, 1))
            aspect = float(bw / max(bh, 1))
            if not (0.80 <= aspect <= 2.2 and rectangularity >= 0.62 and fill >= 0.52):
                continue
            component_values = left_z[component & valid]
            median_z = float(np.median(component_values))
            mad_z = float(np.median(np.abs(component_values - median_z)))
            score = float(area * rectangularity * fill / (1.0 + 80.0 * mad_z))
            record = {
                "plane_seed_z_m": plane_z,
                "median_z_m": median_z,
                "mad_z_m": mad_z,
                "area_px": area,
                "bbox_px": [bx, by, bw, bh],
                "rectangularity": rectangularity,
                "fill": fill,
                "aspect": aspect,
                "score": score,
                "component": component,
            }
            trials.append({key: value for key, value in record.items() if key != "component"})
            if best is None or score > float(best["score"]):
                best = record
    if best is None:
        raise RuntimeError("no large constant-height cardboard bottom plane was found")

    component = np.asarray(best.pop("component"), dtype=bool)
    floor_z = float(best["median_z_m"])
    floor_points_left = np.column_stack(
        [left_x[component & valid], left_y[component & valid], left_z[component & valid]]
    )
    floor_normal_left, floor_plane_offset, floor_plane_report = robust_floor_plane(
        floor_points_left
    )
    depth_quad = fit_quad_from_component(component)
    back_refined_quad, depth_back_refinement = refine_back_edge_with_depth(
        depth_quad, left_z, valid, floor_z
    )
    top_refined_quad, rgbd_back_refinement = refine_back_edge_with_rgbd(
        bgr, back_refined_quad, left_z, valid, floor_z
    )
    edge_refined_quad, rgb_front_refinement = refine_front_edge_with_rgb(
        bgr, top_refined_quad, left_z, valid, floor_z
    )
    left_fit, rgbd_left_refinement = refine_side_edge_with_rgbd(
        bgr, edge_refined_quad, left_z, valid, floor_z, side="left"
    )
    right_fit, rgbd_right_refinement = refine_side_edge_with_rgbd(
        bgr, edge_refined_quad, left_z, valid, floor_z, side="right"
    )
    quad = edge_refined_quad.copy()
    if left_fit is not None and right_fit is not None:
        top_fit_values = rgbd_back_refinement.get("top_fit_y_equals_ax_plus_b")
        bottom_fit_values = rgb_front_refinement.get("front_fit_y_equals_ax_plus_b")
        if top_fit_values is not None and bottom_fit_values is not None:
            top_fit = (float(top_fit_values[0]), float(top_fit_values[1]))
            bottom_fit = (float(bottom_fit_values[0]), float(bottom_fit_values[1]))
            quad = np.asarray(
                [
                    intersect(top_fit, left_fit),
                    intersect(top_fit, right_fit),
                    intersect(bottom_fit, right_fit),
                    intersect(bottom_fit, left_fit),
                ],
                dtype=np.float64,
            )
    area = abs(float(cv2.contourArea(quad.astype(np.float32))))
    edge_checks = edge_height_checks(quad, left_z, valid, floor_z)
    inside_polygon = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(inside_polygon, np.rint(quad).astype(np.int32), 1)
    inside_valid = (inside_polygon > 0) & valid
    inside_floor_support = float(
        np.count_nonzero(inside_valid & (np.abs(left_z - floor_z) <= 0.018))
        / max(np.count_nonzero(inside_valid), 1)
    )
    accepted = bool(
        26000.0 <= area <= 65000.0
        and inside_floor_support >= 0.72
        and all(bool(check["valid"]) for check in edge_checks)
        and bool(rgbd_back_refinement.get("applied"))
        and bool(rgb_front_refinement.get("applied"))
        and bool(rgbd_left_refinement.get("applied"))
        and bool(rgbd_right_refinement.get("applied"))
    )

    diagnostics = {
        "algorithm": "brown RGB ROI + aligned-depth bottom plane + four height transitions",
        "motion_ready": False,
        "frame_number": frame.frame_number,
        "captured_at": frame.captured_at,
        "roi_px": [x0, y0, x1, y1],
        "coarse_depth_boundary_px": np.round(depth_quad, 2).tolist(),
        "boundary_px": np.round(quad, 2).tolist(),
        "boundary_norm": np.round(
            quad / np.asarray([width - 1, height - 1]), 6
        ).tolist(),
        "boundary_area_px": round(area, 1),
        "floor_z_left_base_m": round(floor_z, 6),
        "floor_plane_left_base": floor_plane_report,
        "inside_floor_support": round(inside_floor_support, 4),
        "edge_height_checks": edge_checks,
        "depth_back_refinement": depth_back_refinement,
        "rgbd_back_refinement": rgbd_back_refinement,
        "rgb_front_refinement": rgb_front_refinement,
        "rgbd_left_refinement": rgbd_left_refinement,
        "rgbd_right_refinement": rgbd_right_refinement,
        "accepted": accepted,
        "selected_component": best,
        "top_plane_trials": sorted(trials, key=lambda item: float(item["score"]), reverse=True)[:12],
    }
    if write_outputs:
        overlay = draw_result(bgr, quad, component, floor_z, accepted)
        component_view = np.zeros_like(bgr)
        component_view[component] = (255, 255, 255)
        cv2.imwrite(str(OUTPUT_DIR / "boundary-rgbd.jpg"), overlay)
        cv2.imwrite(str(OUTPUT_DIR / "bottom-plane-mask.png"), component_view)
        cv2.imwrite(str(OUTPUT_DIR / "rgb.jpg"), bgr)
        (OUTPUT_DIR / "diagnostics.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if emit_json:
        print(json.dumps(diagnostics, ensure_ascii=False))
    return diagnostics, bgr


if __name__ == "__main__":
    main()
