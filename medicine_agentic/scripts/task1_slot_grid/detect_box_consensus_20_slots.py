#!/usr/bin/env python3
"""Build an always-available 15-frame four-line estimate and 2x10 slot grid."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

import detect_box_boundary_rgbd as boundary_detector


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = Path("/home/ubuntu/RUC-WONE/medicine_agentic")
CONFIG_PATH = PROJECT / "configs/packaging_console.json"
TASK1_BOX_CONFIG_PATH = PROJECT / "configs/task1_box.json"
DETECTOR_OUTPUT = Path("/tmp/center-box-rgbd-boundary")
OUTPUT_DIR = Path("/tmp/center-box-consensus")
MINIMUM_SAMPLES = 12
MAXIMUM_SAMPLES = 15
MINIMUM_LINE_SAMPLES = 3
ROWS = 10
COLUMNS = 2
SHIPPING_BOX_INNER_SIZE_M = np.asarray([0.265, 0.255], dtype=np.float64)
CARTON_FOOTPRINT_M = np.asarray([0.130, 0.025], dtype=np.float64)
ARRAY_FOOTPRINT_M = np.asarray(
    [COLUMNS * CARTON_FOOTPRINT_M[0], ROWS * CARTON_FOOTPRINT_M[1]],
    dtype=np.float64,
)
CARTON_HEIGHT_M = 0.085
APPROACH_CLEARANCE_M = 0.060
CENTER_EXTRA_MARGIN_M = 0.001


def edge_record(payload: dict[str, object], name: str) -> dict[str, object]:
    for item in payload.get("edge_height_checks", []):
        if item.get("edge") == name:
            return item
    return {"valid": False}


def robust_y_line(
    lines: list[tuple[float, float]], x_reference: float
) -> tuple[tuple[float, float], np.ndarray, dict[str, float]]:
    values = np.asarray(lines, dtype=np.float64)
    slopes = values[:, 0]
    y_reference = slopes * x_reference + values[:, 1]
    median_slope = float(np.median(slopes))
    median_y = float(np.median(y_reference))
    slope_mad = float(np.median(np.abs(slopes - median_slope)))
    y_mad = float(np.median(np.abs(y_reference - median_y)))
    slope_limit = max(0.018, 3.5 * 1.4826 * slope_mad)
    y_limit = max(2.0, 3.5 * 1.4826 * y_mad)
    inliers = (
        (np.abs(slopes - median_slope) <= slope_limit)
        & (np.abs(y_reference - median_y) <= y_limit)
    )
    if np.count_nonzero(inliers) < min(MINIMUM_LINE_SAMPLES, values.shape[0]):
        # A result is mandatory in competition.  Keep the samples nearest the
        # robust median instead of aborting when only a few precise RGB-D
        # refinements survived the per-frame gates.
        order = np.argsort(np.abs(y_reference - median_y) + 20.0 * np.abs(slopes - median_slope))
        inliers = np.zeros(values.shape[0], dtype=bool)
        inliers[order[: min(MINIMUM_LINE_SAMPLES, values.shape[0])]] = True
    slope = float(np.median(slopes[inliers]))
    y_at_reference = float(np.median(y_reference[inliers]))
    intercept = y_at_reference - slope * x_reference
    predicted = slopes[inliers] * x_reference + values[inliers, 1]
    metrics = {
        "samples": int(values.shape[0]),
        "inliers": int(np.count_nonzero(inliers)),
        "anchor_peak_to_peak_px": float(np.ptp(predicted)),
        "anchor_std_px": float(np.std(predicted)),
        "slope_peak_to_peak": float(np.ptp(slopes[inliers])),
    }
    return (slope, intercept), inliers, metrics


def robust_x_line(
    lines: list[tuple[float, float]], y_reference: float
) -> tuple[tuple[float, float], np.ndarray, dict[str, float]]:
    values = np.asarray(lines, dtype=np.float64)
    slopes = values[:, 0]
    x_reference = slopes * y_reference + values[:, 1]
    median_slope = float(np.median(slopes))
    median_x = float(np.median(x_reference))
    slope_mad = float(np.median(np.abs(slopes - median_slope)))
    x_mad = float(np.median(np.abs(x_reference - median_x)))
    slope_limit = max(0.018, 3.5 * 1.4826 * slope_mad)
    x_limit = max(2.0, 3.5 * 1.4826 * x_mad)
    inliers = (
        (np.abs(slopes - median_slope) <= slope_limit)
        & (np.abs(x_reference - median_x) <= x_limit)
    )
    if np.count_nonzero(inliers) < min(MINIMUM_LINE_SAMPLES, values.shape[0]):
        order = np.argsort(np.abs(x_reference - median_x) + 20.0 * np.abs(slopes - median_slope))
        inliers = np.zeros(values.shape[0], dtype=bool)
        inliers[order[: min(MINIMUM_LINE_SAMPLES, values.shape[0])]] = True
    slope = float(np.median(slopes[inliers]))
    x_at_reference = float(np.median(x_reference[inliers]))
    intercept = x_at_reference - slope * y_reference
    predicted = slopes[inliers] * y_reference + values[inliers, 1]
    metrics = {
        "samples": int(values.shape[0]),
        "inliers": int(np.count_nonzero(inliers)),
        "anchor_peak_to_peak_px": float(np.ptp(predicted)),
        "anchor_std_px": float(np.std(predicted)),
        "slope_peak_to_peak": float(np.ptp(slopes[inliers])),
    }
    return (slope, intercept), inliers, metrics


def y_line_from_points(start: np.ndarray, end: np.ndarray) -> tuple[float, float]:
    slope = float((end[1] - start[1]) / max(end[0] - start[0], 1e-9))
    return slope, float(start[1] - slope * start[0])


def x_line_from_points(start: np.ndarray, end: np.ndarray) -> tuple[float, float]:
    slope = float((end[0] - start[0]) / max(end[1] - start[1], 1e-9))
    return slope, float(start[0] - slope * start[1])


def intersect(y_line: tuple[float, float], x_line: tuple[float, float]) -> np.ndarray:
    a, b = y_line
    c, d = x_line
    matrix = np.asarray([[-a, 1.0], [1.0, -c]], dtype=np.float64)
    return np.linalg.solve(matrix, np.asarray([b, d], dtype=np.float64))


def project(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float32).reshape(-1, 1, 2), transform
    ).reshape(-1, 2)


def resolve_from_config(value: str, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()


def pixels_to_left_plane(
    pixels: np.ndarray,
    plane_normal_left: np.ndarray,
    plane_offset_m: float,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    cam_to_left: np.ndarray,
) -> np.ndarray:
    """Undistort pixels and intersect their rays with the fitted bottom plane."""

    rotation = cam_to_left[:3, :3]
    translation = cam_to_left[:3, 3]
    undistorted = cv2.undistortPoints(
        np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2),
        intrinsics,
        distortion,
        P=intrinsics,
    ).reshape(-1, 2)
    points: list[np.ndarray] = []
    normal = np.asarray(plane_normal_left, dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    for u, v in undistorted:
        ray_camera = np.asarray(
            [
                (u - intrinsics[0, 2]) / intrinsics[0, 0],
                (v - intrinsics[1, 2]) / intrinsics[1, 1],
                1.0,
            ],
            dtype=np.float64,
        )
        ray_left = rotation @ ray_camera
        denominator = float(normal @ ray_left)
        if abs(denominator) < 1e-9:
            raise RuntimeError("camera ray is parallel to the box-bottom plane")
        ray_depth = float(-(normal @ translation + plane_offset_m) / denominator)
        if ray_depth <= 0.0:
            raise RuntimeError("box-bottom plane intersection is behind the camera")
        points.append(rotation @ (ray_depth * ray_camera) + translation)
    return np.asarray(points, dtype=np.float64)


def left_points_to_pixels(
    points_left: np.ndarray,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    cam_to_left: np.ndarray,
) -> np.ndarray:
    """Project left-base metric points back into the Center color image."""

    rotation = cam_to_left[:3, :3]
    translation = cam_to_left[:3, 3]
    points_camera = (rotation.T @ (np.asarray(points_left) - translation).T).T
    if np.any(points_camera[:, 2] <= 0.0):
        raise RuntimeError("metric slot projects behind the camera")
    projected, _ = cv2.projectPoints(
        points_camera,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        intrinsics,
        distortion,
    )
    return projected.reshape(-1, 2)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack([values, np.ones(values.shape[0], dtype=np.float64)])
    return (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3]


def metric_box_frame(
    corners_left: np.ndarray, plane_normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build horizontal footprint axes from the measured 3-D quadrilateral."""

    corners = np.asarray(corners_left, dtype=np.float64)
    x_hint = (corners[1] - corners[0]) + (corners[2] - corners[3])
    y_hint = (corners[3] - corners[0]) + (corners[2] - corners[1])
    x_hint[2] = 0.0
    y_hint[2] = 0.0
    x_axis = x_hint / max(float(np.linalg.norm(x_hint)), 1e-12)
    y_axis = y_hint - float(y_hint @ x_axis) * x_axis
    y_axis /= max(float(np.linalg.norm(y_axis)), 1e-12)
    if float(y_axis @ y_hint) < 0.0:
        y_axis = -y_axis
    center = corners.mean(axis=0)
    measured = np.asarray(
        [
            0.5 * (np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[2] - corners[3])),
            0.5 * (np.linalg.norm(corners[3] - corners[0]) + np.linalg.norm(corners[2] - corners[1])),
        ],
        dtype=np.float64,
    )
    return center, x_axis, y_axis, measured


def vertical_project_to_plane(
    points_left: np.ndarray, plane_normal: np.ndarray, plane_offset_m: float
) -> np.ndarray:
    """Keep robot-base X/Y and put each point exactly on the fitted bottom plane."""

    points = np.asarray(points_left, dtype=np.float64).copy()
    normal = np.asarray(plane_normal, dtype=np.float64)
    if abs(float(normal[2])) < 1e-9:
        raise RuntimeError("box-bottom plane is vertical")
    points[:, 2] = -(
        normal[0] * points[:, 0] + normal[1] * points[:, 1] + plane_offset_m
    ) / normal[2]
    return points


def erode_center_polygon(
    polygon_xy: np.ndarray,
    half_size_xy: np.ndarray,
    extra_margin_m: float,
) -> np.ndarray:
    """Minkowski-erode a convex opening by the oriented carton footprint."""

    polygon = np.asarray(polygon_xy, dtype=np.float64)
    signed_area = 0.5 * float(
        np.sum(
            polygon[:, 0] * np.roll(polygon[:, 1], -1)
            - polygon[:, 1] * np.roll(polygon[:, 0], -1)
        )
    )
    orientation = 1.0 if signed_area >= 0.0 else -1.0
    normals: list[np.ndarray] = []
    constants: list[float] = []
    for index in range(polygon.shape[0]):
        start = polygon[index]
        end = polygon[(index + 1) % polygon.shape[0]]
        edge = end - start
        inward = orientation * np.asarray([-edge[1], edge[0]], dtype=np.float64)
        inward /= max(float(np.linalg.norm(inward)), 1e-12)
        support = float(np.abs(inward) @ half_size_xy) + extra_margin_m
        normals.append(inward)
        constants.append(float(inward @ start) + support)
    vertices: list[np.ndarray] = []
    for index in range(polygon.shape[0]):
        previous = (index - 1) % polygon.shape[0]
        matrix = np.vstack([normals[previous], normals[index]])
        if abs(float(np.linalg.det(matrix))) < 1e-9:
            raise RuntimeError("adjacent box edges are parallel after erosion")
        vertices.append(
            np.linalg.solve(matrix, np.asarray([constants[previous], constants[index]]))
        )
    eroded = np.asarray(vertices, dtype=np.float64)
    for normal, constant in zip(normals, constants, strict=True):
        if np.any(eroded @ normal < constant - 1e-6):
            raise RuntimeError("carton footprint does not fit the detected opening")
    return eroded


def bilinear_quad(quad: np.ndarray, u: float, v: float) -> np.ndarray:
    p1, p2, p3, p4 = np.asarray(quad, dtype=np.float64)
    return (
        (1.0 - u) * (1.0 - v) * p1
        + u * (1.0 - v) * p2
        + u * v * p3
        + (1.0 - u) * v * p4
    )


def slot_id(row: int, column: int) -> int:
    return row + 1 if column == 1 else row + 11


def early_consensus_stable(frames: list[dict[str, object]]) -> bool:
    """Stop once the same robust four-line gates used by final output pass."""

    if len(frames) < MINIMUM_SAMPLES:
        return False
    boundaries = np.asarray(
        [frame["boundary_px"] for frame in frames], dtype=np.float64
    )
    if boundaries.ndim != 3 or boundaries.shape[1:] != (4, 2):
        return False
    reference_boundary = np.median(boundaries, axis=0)
    x_reference = float(reference_boundary[:, 0].mean())
    y_reference = float(reference_boundary[:, 1].mean())
    floor_values = np.asarray(
        [frame.get("floor_z_left_base_m") for frame in frames],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(floor_values)):
        return False
    floor_median = float(np.median(floor_values))
    if float(np.median(np.abs(floor_values - floor_median))) > 0.002:
        return False

    top_lines: list[tuple[float, float]] = []
    bottom_lines: list[tuple[float, float]] = []
    left_lines: list[tuple[float, float]] = []
    right_lines: list[tuple[float, float]] = []
    for frame in frames:
        top_refinement = frame.get("rgbd_back_refinement", {})
        bottom_refinement = frame.get("rgb_front_refinement", {})
        left_refinement = frame.get("rgbd_left_refinement", {})
        right_refinement = frame.get("rgbd_right_refinement", {})
        if bool(top_refinement.get("applied")) and bool(
            edge_record(frame, "back").get("valid")
        ):
            fit = top_refinement.get("top_fit_y_equals_ax_plus_b")
            if fit is not None:
                top_lines.append((float(fit[0]), float(fit[1])))
        if bool(bottom_refinement.get("applied")) and bool(
            edge_record(frame, "front").get("valid")
        ):
            fit = bottom_refinement.get("front_fit_y_equals_ax_plus_b")
            if fit is not None:
                bottom_lines.append((float(fit[0]), float(fit[1])))
        if bool(left_refinement.get("applied")) and bool(
            edge_record(frame, "left").get("valid")
        ):
            fit = left_refinement.get("side_fit_x_equals_cy_plus_d")
            if fit is not None:
                left_lines.append((float(fit[0]), float(fit[1])))
        if bool(right_refinement.get("applied")) and bool(
            edge_record(frame, "right").get("valid")
        ):
            fit = right_refinement.get("side_fit_x_equals_cy_plus_d")
            if fit is not None:
                right_lines.append((float(fit[0]), float(fit[1])))
    if not all(
        len(lines) >= MINIMUM_LINE_SAMPLES
        for lines in (top_lines, bottom_lines, left_lines, right_lines)
    ):
        return False
    top, _, top_metrics = robust_y_line(top_lines, x_reference)
    bottom, _, bottom_metrics = robust_y_line(bottom_lines, x_reference)
    left, _, left_metrics = robust_x_line(left_lines, y_reference)
    right, _, right_metrics = robust_x_line(right_lines, y_reference)
    boundary = np.asarray(
        [
            intersect(top, left),
            intersect(top, right),
            intersect(bottom, right),
            intersect(bottom, left),
        ],
        dtype=np.float64,
    )
    area = abs(float(cv2.contourArea(boundary.astype(np.float32))))
    return bool(
        25000.0 <= area <= 65000.0
        and top_metrics["anchor_peak_to_peak_px"] <= 4.0
        and bottom_metrics["anchor_peak_to_peak_px"] <= 4.0
        and left_metrics["anchor_peak_to_peak_px"] <= 7.0
        and right_metrics["anchor_peak_to_peak_px"] <= 7.0
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    slot_grid_cfg = config.get("task1_slot_grid", {})
    layout_clockwise_rotation_deg = float(
        slot_grid_cfg.get("layout_top_view_clockwise_rotation_deg", 0.0)
    )
    if (
        not np.isfinite(layout_clockwise_rotation_deg)
        or layout_clockwise_rotation_deg not in {0.0, 90.0, 180.0, 270.0}
    ):
        raise RuntimeError(
            "task1_slot_grid.layout_top_view_clockwise_rotation_deg "
            "must be 0, 90, 180 or 270"
        )
    camera_cfg = config["camera"]
    intrinsics = np.asarray(camera_cfg["intrinsics"], dtype=np.float64)
    distortion = np.asarray(
        camera_cfg.get("distortion_coefficients", [0.0, 0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    cam_to_left_payload = json.loads(
        resolve_from_config(
            camera_cfg["cam_to_left_path"], CONFIG_PATH.parent
        ).read_text(encoding="utf-8")
    )
    cam_to_left = np.asarray(cam_to_left_payload["cam_to_base"], dtype=np.float64)
    task1_box_cfg = json.loads(TASK1_BOX_CONFIG_PATH.read_text(encoding="utf-8"))
    cam_to_right_payload = json.loads(
        resolve_from_config(
            task1_box_cfg["camera"]["cam_to_right_path"],
            TASK1_BOX_CONFIG_PATH.parent,
        ).read_text(encoding="utf-8")
    )
    cam_to_right = np.asarray(cam_to_right_payload["cam_to_base"], dtype=np.float64)
    left_to_right = cam_to_right @ np.linalg.inv(cam_to_left)
    frames: list[dict[str, object]] = []
    latest_image: np.ndarray | None = None
    sampling_started = time.monotonic()
    early_stopped = False
    camera = boundary_detector.create_camera(
        camera_cfg,
        config_dir=CONFIG_PATH.parent,
    )
    try:
        for index in range(MAXIMUM_SAMPLES):
            try:
                payload, image = boundary_detector.main(
                    camera=camera,
                    config=config,
                    write_outputs=False,
                    emit_json=False,
                )
            except Exception:
                time.sleep(0.03)
                continue
            payload["sample"] = index + 1
            frames.append(payload)
            latest_image = image
            if early_consensus_stable(frames):
                early_stopped = True
                break
            time.sleep(0.03)
    finally:
        camera.close()
    if latest_image is None or not frames:
        raise RuntimeError("no RGB-D frame was available")
    sampling_duration_ms = (time.monotonic() - sampling_started) * 1000.0
    DETECTOR_OUTPUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DETECTOR_OUTPUT / "rgb.jpg"), latest_image)

    all_boundaries = [
        np.asarray(frame["boundary_px"], dtype=np.float64)
        for frame in frames
        if frame.get("boundary_px") is not None
    ]
    reference_boundary = np.median(np.stack(all_boundaries), axis=0)
    x_reference = float(reference_boundary[:, 0].mean())
    y_reference = float(reference_boundary[:, 1].mean())
    top_lines: list[tuple[float, float]] = []
    bottom_lines: list[tuple[float, float]] = []
    left_lines: list[tuple[float, float]] = []
    right_lines: list[tuple[float, float]] = []
    fallback_top_lines: list[tuple[float, float]] = []
    fallback_bottom_lines: list[tuple[float, float]] = []
    fallback_left_lines: list[tuple[float, float]] = []
    fallback_right_lines: list[tuple[float, float]] = []

    for frame in frames:
        boundary = np.asarray(frame["boundary_px"], dtype=np.float64)
        fallback_top_lines.append(y_line_from_points(boundary[0], boundary[1]))
        fallback_bottom_lines.append(y_line_from_points(boundary[3], boundary[2]))
        fallback_left_lines.append(x_line_from_points(boundary[0], boundary[3]))
        fallback_right_lines.append(x_line_from_points(boundary[1], boundary[2]))
        top_refinement = frame.get("rgbd_back_refinement", {})
        bottom_refinement = frame.get("rgb_front_refinement", {})
        left_refinement = frame.get("rgbd_left_refinement", {})
        right_refinement = frame.get("rgbd_right_refinement", {})
        if bool(top_refinement.get("applied")) and bool(edge_record(frame, "back").get("valid")):
            fit = top_refinement.get("top_fit_y_equals_ax_plus_b")
            if fit is not None:
                top_lines.append((float(fit[0]), float(fit[1])))
        if bool(bottom_refinement.get("applied")) and bool(edge_record(frame, "front").get("valid")):
            fit = bottom_refinement.get("front_fit_y_equals_ax_plus_b")
            if fit is not None:
                bottom_lines.append((float(fit[0]), float(fit[1])))
        if bool(left_refinement.get("applied")) and bool(edge_record(frame, "left").get("valid")):
            fit = left_refinement.get("side_fit_x_equals_cy_plus_d")
            if fit is not None:
                left_lines.append((float(fit[0]), float(fit[1])))
        if bool(right_refinement.get("applied")) and bool(edge_record(frame, "right").get("valid")):
            fit = right_refinement.get("side_fit_x_equals_cy_plus_d")
            if fit is not None:
                right_lines.append((float(fit[0]), float(fit[1])))

    # Prefer independently validated RGB-D creases only when enough frames
    # support that edge.  One intermittent crease must not replace fifteen
    # stable depth-quadrilateral observations and falsely degrade the plan.
    selected_top = (
        top_lines if len(top_lines) >= MINIMUM_LINE_SAMPLES else fallback_top_lines
    )
    selected_bottom = (
        bottom_lines
        if len(bottom_lines) >= MINIMUM_LINE_SAMPLES
        else fallback_bottom_lines
    )
    selected_left = (
        left_lines if len(left_lines) >= MINIMUM_LINE_SAMPLES else fallback_left_lines
    )
    selected_right = (
        right_lines
        if len(right_lines) >= MINIMUM_LINE_SAMPLES
        else fallback_right_lines
    )
    top, _, top_metrics = robust_y_line(selected_top, x_reference)
    bottom, _, bottom_metrics = robust_y_line(selected_bottom, x_reference)
    left, _, left_metrics = robust_x_line(selected_left, y_reference)
    right, _, right_metrics = robust_x_line(selected_right, y_reference)
    top_metrics["precise_samples"] = len(top_lines)
    bottom_metrics["precise_samples"] = len(bottom_lines)
    left_metrics["precise_samples"] = len(left_lines)
    right_metrics["precise_samples"] = len(right_lines)
    top_metrics["source"] = (
        "rgbd_crease" if selected_top is top_lines else "depth_quad"
    )
    bottom_metrics["source"] = (
        "rgbd_crease" if selected_bottom is bottom_lines else "depth_quad"
    )
    left_metrics["source"] = (
        "rgbd_crease" if selected_left is left_lines else "depth_quad"
    )
    right_metrics["source"] = (
        "rgbd_crease" if selected_right is right_lines else "depth_quad"
    )
    boundary = np.asarray(
        [
            intersect(top, left),
            intersect(top, right),
            intersect(bottom, right),
            intersect(bottom, left),
        ],
        dtype=np.float64,
    )
    area = abs(float(cv2.contourArea(boundary.astype(np.float32))))
    line_metrics = {
        "top": top_metrics,
        "bottom": bottom_metrics,
        "left": left_metrics,
        "right": right_metrics,
    }
    maximum_anchor_peak_to_peak = max(
        float(metrics["anchor_peak_to_peak_px"]) for metrics in line_metrics.values()
    )
    high_confidence = bool(
        25000.0 <= area <= 65000.0
        and all(len(lines) >= MINIMUM_LINE_SAMPLES for lines in (
            selected_top, selected_bottom, selected_left, selected_right
        ))
        and top_metrics["anchor_peak_to_peak_px"] <= 4.0
        and bottom_metrics["anchor_peak_to_peak_px"] <= 4.0
        and left_metrics["anchor_peak_to_peak_px"] <= 7.0
        and right_metrics["anchor_peak_to_peak_px"] <= 7.0
    )
    consensus_ready = True

    overlay = latest_image.copy()
    color = (70, 255, 120) if high_confidence else (40, 190, 255)
    boundary_i = np.rint(boundary).astype(np.int32)
    cv2.polylines(overlay, [boundary_i], True, color, 4, cv2.LINE_AA)
    for index, point in enumerate(boundary_i, start=1):
        cv2.circle(overlay, tuple(point), 7, (40, 190, 255), -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"P{index}",
            (int(point[0]) + 8, int(point[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 190, 255),
            2,
            cv2.LINE_AA,
        )

    plane_reports = [
        frame.get("floor_plane_left_base", {})
        for frame in frames
        if frame.get("floor_plane_left_base")
    ]
    if not plane_reports:
        raise RuntimeError("no fitted box-bottom plane was available")
    plane_normals = np.asarray(
        [report["normal_left_base"] for report in plane_reports], dtype=np.float64
    )
    fitted_plane_normal = np.median(plane_normals, axis=0)
    if fitted_plane_normal[2] < 0.0:
        fitted_plane_normal = -fitted_plane_normal
    fitted_plane_normal /= max(float(np.linalg.norm(fitted_plane_normal)), 1e-12)
    floor_z_values = [
        float(frame["floor_z_left_base_m"])
        for frame in frames
        if frame.get("floor_z_left_base_m") is not None
    ]
    if not floor_z_values:
        raise RuntimeError("no box-bottom height was available")
    # The carton is inserted vertically in the robot-base frame.  A fixed
    # horizontal projection plane prevents cardboard warp and depth noise from
    # turning an unstable fitted tilt into an X/Y placement shift.
    plane_normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    floor_z_left_m = float(np.median(floor_z_values))
    plane_offset = -floor_z_left_m
    boundary_left = pixels_to_left_plane(
        boundary, plane_normal, plane_offset, intrinsics, distortion, cam_to_left
    )
    box_center, detected_box_x_axis, detected_box_y_axis, measured_inner_size = metric_box_frame(
        boundary_left, plane_normal
    )
    clockwise_radians = np.radians(layout_clockwise_rotation_deg)
    clockwise_rotation = np.asarray(
        [
            [np.cos(clockwise_radians), np.sin(clockwise_radians), 0.0],
            [-np.sin(clockwise_radians), np.cos(clockwise_radians), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    box_x_axis = clockwise_rotation @ detected_box_x_axis
    box_y_axis = clockwise_rotation @ detected_box_y_axis
    box_center = vertical_project_to_plane(
        np.asarray([box_center]), plane_normal, plane_offset
    )[0]
    boundary_xy = np.column_stack(
        [
            (boundary_left - box_center) @ box_x_axis,
            (boundary_left - box_center) @ box_y_axis,
        ]
    )
    measured_layout_inner_size = np.ptp(boundary_xy, axis=0)
    full_array_fit_by_measured_boundary = True
    try:
        array_center_region_xy = erode_center_polygon(
            boundary_xy,
            0.5 * ARRAY_FOOTPRINT_M,
            CENTER_EXTRA_MARGIN_M,
        )
        array_center_xy = array_center_region_xy.mean(axis=0)
    except RuntimeError:
        # Competition output remains available even if noisy metric scale says
        # the known rigid 260x250 mm array is marginally larger.  Crucially,
        # the slot pitch is never compressed to hide that discrepancy.
        full_array_fit_by_measured_boundary = False
        array_center_region_xy = np.empty((0, 2), dtype=np.float64)
        array_center_xy = boundary_xy.mean(axis=0)
    box_yaw_deg = float(np.degrees(np.arctan2(box_x_axis[1], box_x_axis[0])))
    box_x_axis_right = left_to_right[:3, :3] @ box_x_axis
    box_yaw_right_deg = float(
        np.degrees(np.arctan2(box_x_axis_right[1], box_x_axis_right[0]))
    )
    adjacent_pitch_x = float(CARTON_FOOTPRINT_M[0])
    adjacent_pitch_y = float(CARTON_FOOTPRINT_M[1])
    metric_grid = {
        "nominal_shipping_box_inner_size_mm": np.round(SHIPPING_BOX_INNER_SIZE_M * 1000.0, 3).tolist(),
        "nominal_size_used_for_slot_scaling": False,
        "carton_footprint_mm": np.round(CARTON_FOOTPRINT_M * 1000.0, 3).tolist(),
        "carton_height_mm": CARTON_HEIGHT_M * 1000.0,
        "rigid_array_footprint_mm": np.round(ARRAY_FOOTPRINT_M * 1000.0, 3).tolist(),
        "layout_columns_rows": [COLUMNS, ROWS],
        "extra_center_margin_mm": CENTER_EXTRA_MARGIN_M * 1000.0,
        "measured_inner_size_mm": np.round(measured_layout_inner_size * 1000.0, 3).tolist(),
        "detected_inner_size_mm_before_layout_rotation": np.round(
            measured_inner_size * 1000.0, 3
        ).tolist(),
        "adjacent_center_pitch_mm": np.round(
            np.asarray([adjacent_pitch_x, adjacent_pitch_y]) * 1000.0, 3
        ).tolist(),
        "box_center_left_base_m": np.round(box_center, 6).tolist(),
        "bottom_plane_normal_left_base": np.round(plane_normal, 9).tolist(),
        "bottom_plane_offset_m": round(plane_offset, 9),
        "fitted_bottom_plane_normal_diagnostic": np.round(
            fitted_plane_normal, 9
        ).tolist(),
        "bottom_plane_rms_residual_mm": round(
            float(np.median([report["rms_residual_mm"] for report in plane_reports])), 4
        ),
        "x_axis_left_base": np.round(box_x_axis, 8).tolist(),
        "y_axis_left_base": np.round(box_y_axis, 8).tolist(),
        "detected_x_axis_left_base_before_layout_rotation": np.round(
            detected_box_x_axis, 8
        ).tolist(),
        "detected_y_axis_left_base_before_layout_rotation": np.round(
            detected_box_y_axis, 8
        ).tolist(),
        "layout_top_view_clockwise_rotation_deg": layout_clockwise_rotation_deg,
        "placement_vertical_axis_left_base": [0.0, 0.0, 1.0],
        "boundary_left_base_m": np.round(boundary_left, 6).tolist(),
        "boundary_local_xy_mm": np.round(boundary_xy * 1000.0, 3).tolist(),
        "full_array_fit_by_measured_boundary": full_array_fit_by_measured_boundary,
        "array_center_local_xy_mm": np.round(array_center_xy * 1000.0, 3).tolist(),
        "feasible_array_center_region_local_xy_mm": np.round(
            array_center_region_xy * 1000.0, 3
        ).tolist(),
        "carton_long_axis_yaw_left_base_deg": round(box_yaw_deg, 5),
        "carton_long_axis_yaw_right_base_deg": round(box_yaw_right_deg, 5),
    }

    slots: list[dict[str, object]] = []
    if consensus_ready:
        fill = overlay.copy()
        for row in range(ROWS):
            for column in range(COLUMNS):
                center_xy = array_center_xy + np.asarray(
                    [
                        (column - 0.5 * (COLUMNS - 1)) * CARTON_FOOTPRINT_M[0],
                        (row - 0.5 * (ROWS - 1)) * CARTON_FOOTPRINT_M[1],
                    ],
                    dtype=np.float64,
                )
                center_floor = (
                    box_center
                    + center_xy[0] * box_x_axis
                    + center_xy[1] * box_y_axis
                )
                half_x, half_y = 0.5 * CARTON_FOOTPRINT_M
                polygon_left = np.asarray(
                    [
                        center_floor - half_x * box_x_axis - half_y * box_y_axis,
                        center_floor + half_x * box_x_axis - half_y * box_y_axis,
                        center_floor + half_x * box_x_axis + half_y * box_y_axis,
                        center_floor - half_x * box_x_axis + half_y * box_y_axis,
                    ],
                    dtype=np.float64,
                )
                polygon_left = vertical_project_to_plane(
                    polygon_left, plane_normal, plane_offset
                )
                center_floor = vertical_project_to_plane(
                    np.asarray([center_floor]), plane_normal, plane_offset
                )[0]
                placement_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
                center_release = center_floor + CARTON_HEIGHT_M * placement_axis
                center_approach = center_release + APPROACH_CLEARANCE_M * placement_axis
                center_floor_right = transform_points(
                    np.asarray([center_floor]), left_to_right
                )[0]
                center_release_right = transform_points(
                    np.asarray([center_release]), left_to_right
                )[0]
                center_approach_right = transform_points(
                    np.asarray([center_approach]), left_to_right
                )[0]
                polygon = left_points_to_pixels(
                    polygon_left, intrinsics, distortion, cam_to_left
                )
                center = left_points_to_pixels(
                    np.asarray([center_floor]), intrinsics, distortion, cam_to_left
                )[0]
                identifier = slot_id(row, column)
                slot_color = (255, 180, 40) if column == 0 else (50, 210, 255)
                cv2.fillConvexPoly(fill, np.rint(polygon).astype(np.int32), slot_color)
                slots.append(
                    {
                        "slot_id": identifier,
                        "row": row,
                        "column": column,
                        "center_local_xy_mm": np.round(center_xy * 1000.0, 3).tolist(),
                        "center_px": np.round(center, 2).tolist(),
                        "polygon_px": np.round(polygon, 2).tolist(),
                        "floor_center_left_base_m": np.round(center_floor, 6).tolist(),
                        "release_surface_center_left_base_m": np.round(center_release, 6).tolist(),
                        "approach_center_left_base_m": np.round(center_approach, 6).tolist(),
                        "floor_center_right_base_m": np.round(center_floor_right, 6).tolist(),
                        "placement_completion_center_right_base_m": np.round(
                            center_release_right, 6
                        ).tolist(),
                        "carton_top_center_right_base_m": np.round(
                            center_release_right, 6
                        ).tolist(),
                        "approach_center_right_base_m": np.round(
                            center_approach_right, 6
                        ).tolist(),
                        "carton_long_axis_yaw_left_base_deg": round(box_yaw_deg, 5),
                        "carton_long_axis_yaw_right_base_deg": round(
                            box_yaw_right_deg, 5
                        ),
                        "motion_ready": False,
                    }
                )
        overlay = cv2.addWeighted(fill, 0.16, overlay, 0.84, 0.0)
        cv2.polylines(overlay, [boundary_i], True, color, 4, cv2.LINE_AA)
        for slot in slots:
            polygon_i = np.rint(np.asarray(slot["polygon_px"])).astype(np.int32)
            cv2.polylines(overlay, [polygon_i], True, (100, 230, 255), 1, cv2.LINE_AA)
            center_i = np.rint(np.asarray(slot["center_px"])).astype(int)
            label = f"{int(slot['slot_id']):02d}"
            cv2.putText(
                overlay,
                label,
                (int(center_i[0]) - 7, int(center_i[1]) + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    quality_label = "HIGH" if high_confidence else "DEGRADED"
    label = (
        f"PLACEMENT GRID READY | quality={quality_label} | "
        f"max line P-P={maximum_anchor_peak_to_peak:.2f}px"
    )
    cv2.putText(
        overlay,
        label,
        (max(12, int(boundary[:, 0].min()) - 35), max(28, int(boundary[:, 1].min()) - 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )
    payload = {
        "consensus_ready": consensus_ready,
        "placement_grid_ready": True,
        "high_confidence": high_confidence,
        "motion_ready": False,
        "sample_count": len(frames),
        "minimum_samples": MINIMUM_SAMPLES,
        "maximum_samples": MAXIMUM_SAMPLES,
        "early_stopped": early_stopped,
        "sampling_duration_ms": round(sampling_duration_ms, 3),
        "boundary_px": np.round(boundary, 2).tolist(),
        "boundary_area_px": round(area, 1),
        "line_metrics": line_metrics,
        "maximum_anchor_peak_to_peak_px": round(maximum_anchor_peak_to_peak, 4),
        "metric_grid": metric_grid,
        "slots": sorted(slots, key=lambda item: int(item["slot_id"])),
        "frames": frames,
    }
    cv2.imwrite(str(OUTPUT_DIR / "consensus-20-slots.jpg"), overlay)
    (OUTPUT_DIR / "consensus-20-slots.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "consensus_ready": consensus_ready,
        "placement_grid_ready": True,
        "high_confidence": high_confidence,
        "sample_count": len(frames),
        "early_stopped": early_stopped,
        "sampling_duration_ms": round(sampling_duration_ms, 3),
        "boundary_px": payload["boundary_px"],
        "line_metrics": line_metrics,
        "slot_count": len(slots),
        "motion_ready": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
