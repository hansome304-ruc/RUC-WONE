"""Pure geometry for projecting a fixed two-cup tool onto a carton face.

This module never reads hardware or issues robot commands.  It turns a locked
flange pose plus a one-time tool-axis calibration into the two real cup contact
patches seen by the fixed front camera.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np


def _unit_vector(value: Sequence[float], *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain 3 finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError(f"{name} must not be zero")
    return vector / norm


def quaternion_xyzw_to_matrix(value: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion_xyzw must contain 4 finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-9:
        raise ValueError("quaternion_xyzw must not be zero")
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def fixed_suction_axis_status(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = config if isinstance(config, dict) else {}
    blockers: list[str] = []
    enabled = cfg.get("enabled") is True
    calibrated = cfg.get("calibrated") is True

    if not calibrated:
        blockers.append("calibration_required")
    axis = cfg.get("axis_local_xyz")
    approach = cfg.get("approach_local_xyz")
    if calibrated:
        try:
            axis_vector = _unit_vector(axis, name="axis_local_xyz")
            approach_vector = _unit_vector(approach, name="approach_local_xyz")
            if abs(float(np.dot(axis_vector, approach_vector))) > 0.05:
                blockers.append("tool_axes_not_perpendicular")
        except (TypeError, ValueError) as exc:
            blockers.append(str(exc))

    dimensions: dict[str, float] = {}
    for field, default in (
        ("cup_center_spacing_mm", 50.0),
        ("cup_diameter_mm", 25.0),
        ("safety_margin_mm", 8.0),
    ):
        try:
            value = float(cfg.get(field, default))
        except (TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value) or value < 0.0 or (
            field != "safety_margin_mm" and value <= 0.0
        ):
            blockers.append(f"invalid_{field}")
        dimensions[field] = value

    blockers = list(dict.fromkeys(blockers))
    return {
        "enabled": enabled,
        "calibrated": calibrated,
        "ready": enabled and calibrated and not blockers,
        "calibration_version": cfg.get("calibration_version"),
        "axis_local_xyz": axis,
        "approach_local_xyz": approach,
        "dimensions": dimensions,
        "blockers": blockers,
        "activation_policy": "disabled_until_operator_confirms_calibration",
    }


@dataclass(frozen=True)
class FixedSuctionProjection:
    midpoint_left_base_m: tuple[float, float, float]
    cup_centers_left_base_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    midpoint_px: tuple[float, float]
    cup_centers_px: tuple[tuple[float, float], tuple[float, float]]
    footprints_px: tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]
    axis_angle_deg: float
    valid_2d: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "alignment": "fixed_tool_axis",
            "midpoint_left_base_m": list(self.midpoint_left_base_m),
            "cup_centers_left_base_m": [list(point) for point in self.cup_centers_left_base_m],
            "midpoint_px": list(self.midpoint_px),
            "cup_centers_px": [list(point) for point in self.cup_centers_px],
            "footprints_px": [
                [list(point) for point in footprint]
                for footprint in self.footprints_px
            ],
            "axis_angle_deg": self.axis_angle_deg,
            "angle_frame": "image_xy",
            "valid_2d": self.valid_2d,
            "blockers": list(self.blockers),
        }


def project_fixed_suction_axis(
    *,
    midpoint_left_base_m: Sequence[float],
    locked_flange_quaternion_xyzw: Sequence[float],
    axis_local_xyz: Sequence[float],
    approach_local_xyz: Sequence[float],
    cup_center_spacing_m: float,
    cup_diameter_m: float,
    safety_margin_m: float,
    cam_to_left: np.ndarray,
    intrinsics: np.ndarray,
    candidate_polygon_px: Sequence[Sequence[float]] | None = None,
    image_shape: tuple[int, int] | None = None,
    circle_samples: int = 24,
) -> FixedSuctionProjection:
    """Project the real fixed tool geometry into the camera image.

    ``cam_to_left`` maps camera coordinates into the left-base frame.  The
    local axes are stored only after the operator's one-time calibration.
    """

    midpoint = np.asarray(midpoint_left_base_m, dtype=np.float64)
    if midpoint.shape != (3,) or not np.all(np.isfinite(midpoint)):
        raise ValueError("midpoint_left_base_m must contain 3 finite values")
    transform = np.asarray(cam_to_left, dtype=np.float64)
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("cam_to_left must be a finite 4x4 matrix")
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("intrinsics must be a finite 3x3 matrix")
    if not math.isfinite(cup_center_spacing_m) or cup_center_spacing_m <= 0.0:
        raise ValueError("cup_center_spacing_m must be positive")
    if not math.isfinite(cup_diameter_m) or cup_diameter_m <= 0.0:
        raise ValueError("cup_diameter_m must be positive")
    if not math.isfinite(safety_margin_m) or safety_margin_m < 0.0:
        raise ValueError("safety_margin_m must be non-negative")
    if not 12 <= int(circle_samples) <= 128:
        raise ValueError("circle_samples must be 12..128")

    rotation = quaternion_xyzw_to_matrix(locked_flange_quaternion_xyzw)
    axis_local = _unit_vector(axis_local_xyz, name="axis_local_xyz")
    approach_local = _unit_vector(approach_local_xyz, name="approach_local_xyz")
    axis_local = axis_local - float(np.dot(axis_local, approach_local)) * approach_local
    axis_local = _unit_vector(axis_local, name="axis_local_xyz projected into cup plane")
    lateral_local = _unit_vector(
        np.cross(approach_local, axis_local),
        name="cup_plane_lateral_axis",
    )
    axis_base = rotation @ axis_local
    lateral_base = rotation @ lateral_local
    half_spacing = cup_center_spacing_m * 0.5
    cup_centers_base = (midpoint - half_spacing * axis_base, midpoint + half_spacing * axis_base)
    effective_radius = cup_diameter_m * 0.5 + safety_margin_m

    left_to_cam = np.linalg.inv(transform)

    def project(point_base: np.ndarray) -> tuple[float, float]:
        homogeneous = np.append(point_base, 1.0)
        point_camera = (left_to_cam @ homogeneous)[:3]
        if not np.all(np.isfinite(point_camera)) or point_camera[2] <= 1e-6:
            raise ValueError("fixed suction point is behind the camera")
        projected = matrix @ point_camera
        return float(projected[0] / projected[2]), float(projected[1] / projected[2])

    cup_centers_px = tuple(project(point) for point in cup_centers_base)
    midpoint_px = project(midpoint)
    footprints: list[tuple[tuple[float, float], ...]] = []
    for center in cup_centers_base:
        footprint = []
        for index in range(int(circle_samples)):
            angle = 2.0 * math.pi * index / int(circle_samples)
            point = center + effective_radius * (
                math.cos(angle) * axis_base + math.sin(angle) * lateral_base
            )
            footprint.append(project(point))
        footprints.append(tuple(footprint))

    blockers: list[str] = []
    polygon_array: np.ndarray | None = None
    if candidate_polygon_px is not None:
        polygon_array = np.asarray(candidate_polygon_px, dtype=np.float32)
        if polygon_array.ndim != 2 or polygon_array.shape[0] < 3 or polygon_array.shape[1] != 2:
            raise ValueError("candidate_polygon_px must be an Nx2 polygon")
        contour = polygon_array.reshape((-1, 1, 2))
        if any(
            cv2.pointPolygonTest(contour, point, False) < 0.0
            for footprint in footprints
            for point in footprint
        ):
            blockers.append("fixed_suction_footprint_outside_candidate")

    if image_shape is not None:
        image_height, image_width = int(image_shape[0]), int(image_shape[1])
        if image_height <= 0 or image_width <= 0:
            raise ValueError("image_shape must be positive")
        if any(
            point[0] < 0.0 or point[1] < 0.0 or point[0] >= image_width or point[1] >= image_height
            for footprint in footprints
            for point in footprint
        ):
            blockers.append("fixed_suction_footprint_outside_image")

    delta = np.asarray(cup_centers_px[1]) - np.asarray(cup_centers_px[0])
    angle_deg = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
    blockers = list(dict.fromkeys(blockers))
    return FixedSuctionProjection(
        midpoint_left_base_m=tuple(float(value) for value in midpoint),
        cup_centers_left_base_m=tuple(
            tuple(float(value) for value in point) for point in cup_centers_base
        ),
        midpoint_px=midpoint_px,
        cup_centers_px=cup_centers_px,
        footprints_px=(footprints[0], footprints[1]),
        axis_angle_deg=angle_deg,
        valid_2d=not blockers,
        blockers=tuple(blockers),
    )
