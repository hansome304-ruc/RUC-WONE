"""Task 1: deterministic medicine-carton detection and 3-D target estimation.

This module deliberately contains no robot-motion command.  Its output is a
checked surface point in the left-arm base frame plus visual evidence.  Motion
is added only after this perception gate is stable and the suction TCP has
been calibrated.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import requests


@dataclass(frozen=True)
class BoxCandidate:
    center_px: tuple[float, float]
    suction_px: tuple[int, int]
    polygon_px: tuple[tuple[float, float], ...]
    long_side_px: float
    short_side_px: float
    angle_deg: float
    rectangularity: float
    bright_fill: float
    edge_clearance_px: float
    score: float
    provider: str = "classical"
    face_type: str = "unknown"
    face_score: float = 0.0
    reference_face_id: str | None = None
    graspable: bool = False
    grasp_blockers: tuple[str, ...] = ("face_unverified",)
    grid_shape: tuple[int, int] = (1, 1)
    grid_index: tuple[int, int] = (0, 0)
    grid_parent_center_px: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["center_px"] = list(self.center_px)
        payload["suction_px"] = list(self.suction_px)
        payload["polygon_px"] = [list(point) for point in self.polygon_px]
        payload["grasp_blockers"] = list(self.grasp_blockers)
        payload["grid_shape"] = list(self.grid_shape)
        payload["grid_index"] = list(self.grid_index)
        payload["grid_parent_center_px"] = (
            None
            if self.grid_parent_center_px is None
            else list(self.grid_parent_center_px)
        )
        return payload


@dataclass(frozen=True)
class DualSuctionTarget:
    """Image-space target for the fixed two-cup suction assembly."""

    midpoint_px: tuple[float, float]
    cup_centers_px: tuple[tuple[float, float], tuple[float, float]]
    axis_angle_deg: float
    alignment: str
    carton_face_size_mm: tuple[float, float]
    cup_diameter_mm: float
    cup_edge_gap_mm: float
    cup_center_spacing_mm: float
    outer_span_mm: float
    projected_cup_radius_px: float
    effective_footprint_radius_px: float
    raw_long_end_margin_mm: float
    raw_short_side_margin_mm: float
    safety_margin_mm: float
    valid_2d: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid_2d": self.valid_2d,
            "midpoint_px": list(self.midpoint_px),
            "cup_centers_px": [
                list(center) for center in self.cup_centers_px
            ],
            "axis_angle_deg": self.axis_angle_deg,
            "angle_frame": "image_xy",
            "alignment": self.alignment,
            "carton_face_size_mm": list(self.carton_face_size_mm),
            "cup_diameter_mm": self.cup_diameter_mm,
            "cup_edge_gap_mm": self.cup_edge_gap_mm,
            "cup_center_spacing_mm": self.cup_center_spacing_mm,
            "outer_span_mm": self.outer_span_mm,
            "projected_cup_radius_px": self.projected_cup_radius_px,
            "effective_footprint_radius_px": (
                self.effective_footprint_radius_px
            ),
            "raw_edge_margins_mm": {
                "long_end": self.raw_long_end_margin_mm,
                "short_side": self.raw_short_side_margin_mm,
            },
            "safety_margin_mm": self.safety_margin_mm,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class DepthEstimate:
    median_mm: float
    spread_mm: float
    valid_samples: int
    samples_mm: tuple[float, ...]
    sample_pixels_px: tuple[tuple[int, int], ...]
    frame_age_s: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["samples_mm"] = list(self.samples_mm)
        payload["sample_pixels_px"] = [
            list(pixel) for pixel in self.sample_pixels_px
        ]
        return payload


@dataclass(frozen=True)
class LocatedBox:
    candidate: BoxCandidate
    depth: DepthEstimate | None
    point_camera_m: tuple[float, float, float] | None
    point_left_base_m: tuple[float, float, float] | None
    physical_size_m: tuple[float, float] | None
    surface_normal_left_base: tuple[float, float, float] | None
    surface_tilt_deg: float | None
    plane_residual_mm: float | None
    reachable: bool | None
    blockers: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "candidate": self.candidate.to_dict(),
            "depth": None if self.depth is None else self.depth.to_dict(),
            "point_camera_m": (
                None if self.point_camera_m is None else list(self.point_camera_m)
            ),
            "point_left_base_m": (
                None if self.point_left_base_m is None else list(self.point_left_base_m)
            ),
            "physical_size_m": (
                None if self.physical_size_m is None else list(self.physical_size_m)
            ),
            "surface_normal_left_base": (
                None
                if self.surface_normal_left_base is None
                else list(self.surface_normal_left_base)
            ),
            "surface_tilt_deg": self.surface_tilt_deg,
            "plane_residual_mm": self.plane_residual_mm,
            "reachable": self.reachable,
            "blockers": list(self.blockers),
        }


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"config root must be an object: {path}")
    return payload


def propose_task1_surface_grid(
    rgb: np.ndarray,
    seed_candidates: Iterable[BoxCandidate],
    *,
    roi_norm: Iterable[float],
    config: dict[str, Any],
    rows: int = 3,
    columns: int = 3,
) -> list[BoxCandidate]:
    """Fit the visible Task1 surface and return one candidate per carton.

    The 3 x 3 arrangement is only a layout prior.  Its image-space envelope
    and the lateral shift of every row are re-estimated from the current pink
    carton pixels, so a small displacement of the stack does not require a
    fixed pixel ROI or a new calibration.  At least one reference-verified
    seed is required before layout completion; all generated cells are still
    checked later by the RGB-D physical-size, dual-cup and layer gates.
    """

    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be uint8 HxWx3")
    rows = int(rows)
    columns = int(columns)
    if rows < 1 or columns < 1:
        return []
    try:
        roi_values = [float(value) for value in roi_norm]
    except (TypeError, ValueError):
        return []
    if len(roi_values) != 4:
        return []
    image_height, image_width = rgb.shape[:2]
    x0 = int(round(np.clip(roi_values[0], 0.0, 1.0) * image_width))
    y0 = int(round(np.clip(roi_values[1], 0.0, 1.0) * image_height))
    x1 = int(round(np.clip(roi_values[2], 0.0, 1.0) * image_width))
    y1 = int(round(np.clip(roi_values[3], 0.0, 1.0) * image_height))
    if x1 <= x0 or y1 <= y0:
        return []

    seeds = [
        candidate
        for candidate in seed_candidates
        if x0 <= float(candidate.center_px[0]) < x1
        and y0 <= float(candidate.center_px[1]) < y1
        and candidate.reference_face_id is not None
        and candidate.face_type in {"front_large", "back_large"}
    ]
    if not seeds:
        return []
    seed = max(seeds, key=lambda candidate: float(candidate.score))

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    pink = np.where(
        (
            (hue >= int(config.get("pink_hue_min", 130)))
            & (hue <= int(config.get("pink_hue_max", 175)))
            & (saturation >= int(config.get("pink_saturation_min", 8)))
            & (saturation <= int(config.get("pink_saturation_max", 130)))
            & (value >= int(config.get("pink_value_min", 130)))
        ),
        255,
        0,
    ).astype(np.uint8)
    roi_mask = pink[y0:y1, x0:x1]
    roi_mask = cv2.morphologyEx(
        roi_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(roi_mask, 8)
    retained = np.zeros_like(roi_mask)
    minimum_area = max(120, int(round(roi_mask.size * 0.001)))
    for label in range(1, count):
        component_x, component_y, width, height, area = stats[label]
        if (
            int(area) >= minimum_area
            and int(width) >= 18
            and int(height) >= 10
        ):
            retained[labels == label] = 255
    local_y, local_x = np.nonzero(retained)
    minimum_pixels = max(300, rows * columns * 20)
    if local_x.size < minimum_pixels:
        return []

    low_x, high_x = np.quantile(local_x, [0.01, 0.99])
    low_y, high_y = np.quantile(local_y, [0.01, 0.99])
    span_x = float(high_x - low_x)
    span_y = float(high_y - low_y)
    minimum_cell_side = float(config.get("task1_grid_minimum_cell_side_px", 36.0))
    if span_x / columns < minimum_cell_side or span_y / rows < minimum_cell_side:
        return []

    # Pink printing does not reach the cardboard edges.  Expand the fitted
    # envelope slightly, then use its cell dimensions for every row while
    # allowing the row centre to follow perspective and small placement skew.
    pad_x = 0.04 * span_x
    pad_y = 0.03 * span_y
    low_x = max(0.0, low_x - pad_x)
    high_x = min(float(x1 - x0 - 1), high_x + pad_x)
    low_y = max(0.0, low_y - pad_y)
    high_y = min(float(y1 - y0 - 1), high_y + pad_y)
    span_x = float(high_x - low_x)
    span_y = float(high_y - low_y)
    cell_width = span_x / columns
    cell_height = span_y / rows
    minimum_cell_pink = float(
        config.get("task1_grid_minimum_cell_pink_fraction", 0.018)
    )

    cells: list[BoxCandidate] = []
    global_center_x = (low_x + high_x) / 2.0
    for row in range(rows):
        band_low = low_y + row * cell_height
        band_high = low_y + (row + 1) * cell_height
        band = (
            (local_y >= max(0.0, band_low - 0.15 * cell_height))
            & (local_y <= min(float(y1 - y0 - 1), band_high + 0.15 * cell_height))
        )
        band_x = local_x[band]
        row_center_x = global_center_x
        if band_x.size >= 60:
            row_low, row_high = np.quantile(band_x, [0.02, 0.98])
            if float(row_high - row_low) >= 0.60 * span_x:
                row_center_x = float(row_low + row_high) / 2.0
        center_y = y0 + low_y + (row + 0.5) * cell_height
        for column in range(columns):
            center_x = (
                x0
                + row_center_x
                + (column - (columns - 1) / 2.0) * cell_width
            )
            left = int(max(x0, round(center_x - cell_width / 2.0)))
            right = int(min(x1, round(center_x + cell_width / 2.0)))
            top = int(max(y0, round(center_y - cell_height / 2.0)))
            bottom = int(min(y1, round(center_y + cell_height / 2.0)))
            if right <= left or bottom <= top:
                continue
            support = pink[top:bottom, left:right]
            pink_fraction = float(np.count_nonzero(support)) / float(support.size)
            if pink_fraction < minimum_cell_pink:
                continue
            half_width = cell_width / 2.0
            half_height = cell_height / 2.0
            cells.append(
                BoxCandidate(
                    center_px=(float(center_x), float(center_y)),
                    suction_px=(int(round(center_x)), int(round(center_y))),
                    polygon_px=(
                        (float(center_x - half_width), float(center_y - half_height)),
                        (float(center_x + half_width), float(center_y - half_height)),
                        (float(center_x + half_width), float(center_y + half_height)),
                        (float(center_x - half_width), float(center_y + half_height)),
                    ),
                    long_side_px=float(max(cell_width, cell_height)),
                    short_side_px=float(min(cell_width, cell_height)),
                    angle_deg=90.0 if cell_height >= cell_width else 0.0,
                    rectangularity=1.0,
                    bright_fill=pink_fraction,
                    edge_clearance_px=float(min(cell_width, cell_height) / 2.0),
                    score=float(seed.score),
                    provider="task1_dynamic_surface_grid",
                    face_type=str(seed.face_type),
                    face_score=float(seed.face_score),
                    reference_face_id=seed.reference_face_id,
                    graspable=bool(seed.graspable),
                    grasp_blockers=tuple(seed.grasp_blockers),
                    grid_shape=(rows, columns),
                    grid_index=(row, column),
                    grid_parent_center_px=(
                        float(x0 + global_center_x),
                        float(y0 + (low_y + high_y) / 2.0),
                    ),
                )
            )
    return cells


def plan_dual_suction_target(
    candidate: BoxCandidate,
    cfg: dict[str, Any],
    *,
    image_shape: tuple[int, int] | None = None,
) -> DualSuctionTarget | None:
    """Place both cup centers symmetrically along the carton's long axis.

    The target is image-space only. Its midpoint is always the detected
    large-face center; it is not the legacy single-cup clearance point.
    """

    dual_cfg = cfg.get("dual_suction", {})
    if not isinstance(dual_cfg, dict):
        dual_cfg = {}
    if not bool(dual_cfg.get("enabled", False)):
        return None

    blockers: list[str] = []

    def positive(name: str, default: float) -> float:
        try:
            value = float(dual_cfg.get(name, default))
        except (TypeError, ValueError):
            blockers.append("dual_suction_dimensions_inconsistent")
            return default
        if not math.isfinite(value) or value <= 0.0:
            blockers.append("dual_suction_dimensions_inconsistent")
            return default
        return value

    face_size = dual_cfg.get("carton_face_size_mm", [130.0, 85.0])
    try:
        face_long_mm, face_short_mm = (
            float(face_size[0]),
            float(face_size[1]),
        )
    except (IndexError, TypeError, ValueError):
        face_long_mm, face_short_mm = 130.0, 85.0
        blockers.append("dual_suction_dimensions_inconsistent")
    if (
        not math.isfinite(face_long_mm)
        or not math.isfinite(face_short_mm)
        or face_long_mm <= 0.0
        or face_short_mm <= 0.0
    ):
        face_long_mm, face_short_mm = 130.0, 85.0
        blockers.append("dual_suction_dimensions_inconsistent")

    cup_diameter_mm = positive("cup_diameter_mm", 25.0)
    cup_edge_gap_mm = positive("cup_edge_gap_mm", 25.0)
    cup_center_spacing_mm = positive("cup_center_spacing_mm", 50.0)
    outer_span_mm = positive("assembly_outer_span_mm", 75.0)
    alignment = str(
        dual_cfg.get("alignment", "carton_long_axis")
    ).strip().lower()
    alignment = {
        "long_axis": "carton_long_axis",
        "short_axis": "carton_short_axis",
    }.get(alignment, alignment)
    if alignment not in {"carton_long_axis", "carton_short_axis"}:
        alignment = "carton_long_axis"
        blockers.append("dual_suction_alignment_invalid")
    safety_margin_mm = max(
        0.0, float(dual_cfg.get("safety_margin_mm", 8.0))
    )

    tolerance_mm = float(dual_cfg.get("dimension_tolerance_mm", 0.5))
    if (
        abs(cup_center_spacing_mm - (cup_diameter_mm + cup_edge_gap_mm))
        > tolerance_mm
        or abs(outer_span_mm - (cup_center_spacing_mm + cup_diameter_mm))
        > tolerance_mm
        or outer_span_mm
        >= (face_short_mm if alignment == "carton_short_axis" else face_long_mm)
        or cup_diameter_mm
        >= (face_long_mm if alignment == "carton_short_axis" else face_short_mm)
    ):
        blockers.append("dual_suction_dimensions_inconsistent")

    required_faces = dual_cfg.get(
        "required_face_types", ["front_large", "back_large"]
    )
    enforce_required_face_types = (
        dual_cfg.get("enforce_required_face_types", True) is True
    )
    if enforce_required_face_types and (
        not isinstance(required_faces, list)
        or candidate.face_type not in {str(item) for item in required_faces}
    ):
        blockers.append("dual_suction_face_not_allowed")
    if not candidate.graspable:
        blockers.append("candidate_not_graspable")

    geometry = (
        candidate.center_px[0],
        candidate.center_px[1],
        candidate.long_side_px,
        candidate.short_side_px,
        candidate.angle_deg,
    )
    if (
        len(candidate.polygon_px) < 4
        or not all(math.isfinite(float(value)) for value in geometry)
        or candidate.long_side_px <= 0.0
        or candidate.short_side_px <= 0.0
    ):
        blockers.append("dual_suction_geometry_unavailable")

    carton_long_angle_deg = (
        float(candidate.angle_deg)
        if math.isfinite(float(candidate.angle_deg))
        else 0.0
    )
    angle_deg = (
        carton_long_angle_deg + 90.0
        if alignment == "carton_short_axis"
        else carton_long_angle_deg
    )
    while angle_deg >= 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    angle_rad = math.radians(angle_deg)
    axis = np.asarray(
        [math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float64
    )
    midpoint = np.asarray(candidate.center_px, dtype=np.float64)
    alignment_side_px = (
        float(candidate.short_side_px)
        if alignment == "carton_short_axis"
        else float(candidate.long_side_px)
    )
    alignment_side_mm = (
        face_short_mm if alignment == "carton_short_axis" else face_long_mm
    )
    half_spacing_px = (
        alignment_side_px * cup_center_spacing_mm / (2.0 * alignment_side_mm)
    )
    cup_a = midpoint - half_spacing_px * axis
    cup_b = midpoint + half_spacing_px * axis

    long_scale = float(candidate.long_side_px) / face_long_mm
    short_scale = float(candidate.short_side_px) / face_short_mm
    conservative_scale = max(long_scale, short_scale, 0.0)
    projected_radius_px = 0.5 * cup_diameter_mm * conservative_scale
    effective_radius_px = (
        0.5 * cup_diameter_mm + safety_margin_mm
    ) * conservative_scale
    minimum_clearance_px = max(
        effective_radius_px,
        float(dual_cfg.get("min_polygon_clearance_px", 4.0)),
    )

    if len(candidate.polygon_px) >= 3:
        polygon = np.asarray(
            candidate.polygon_px, dtype=np.float32
        ).reshape((-1, 1, 2))
        for cup in (cup_a, cup_b):
            clearance = cv2.pointPolygonTest(
                polygon, (float(cup[0]), float(cup[1])), True
            )
            if not math.isfinite(clearance) or clearance < minimum_clearance_px:
                blockers.append("dual_suction_margin_low")
                break

    if image_shape is not None:
        image_height, image_width = image_shape
        for cup in (cup_a, cup_b):
            if (
                cup[0] - effective_radius_px < 0.0
                or cup[1] - effective_radius_px < 0.0
                or cup[0] + effective_radius_px >= image_width
                or cup[1] + effective_radius_px >= image_height
            ):
                blockers.append("dual_suction_outside_image")
                break

    blockers = list(dict.fromkeys(blockers))
    return DualSuctionTarget(
        midpoint_px=(float(midpoint[0]), float(midpoint[1])),
        cup_centers_px=(
            (float(cup_a[0]), float(cup_a[1])),
            (float(cup_b[0]), float(cup_b[1])),
        ),
        axis_angle_deg=angle_deg,
        alignment=alignment,
        carton_face_size_mm=(face_long_mm, face_short_mm),
        cup_diameter_mm=cup_diameter_mm,
        cup_edge_gap_mm=cup_edge_gap_mm,
        cup_center_spacing_mm=cup_center_spacing_mm,
        outer_span_mm=outer_span_mm,
        projected_cup_radius_px=projected_radius_px,
        effective_footprint_radius_px=effective_radius_px,
        raw_long_end_margin_mm=(face_long_mm - outer_span_mm) / 2.0,
        raw_short_side_margin_mm=(face_short_mm - cup_diameter_mm) / 2.0,
        safety_margin_mm=safety_margin_mm,
        valid_2d=not blockers,
        blockers=tuple(blockers),
    )


def evaluate_dual_suction_depth(
    depth_z16: np.ndarray | None,
    depth_scale_m: float | None,
    target: DualSuctionTarget | None,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Check that both planned contact patches contain usable depth.

    Raw depth difference between the two cups is reported but is deliberately
    not gated: an oblique fixed camera naturally sees different Z values on a
    flat horizontal carton.
    """

    if target is None or depth_z16 is None or depth_scale_m is None:
        return {
            "available": False,
            "valid": False,
            "reason": "synchronized depth or dual-suction target unavailable",
            "cups": [],
        }
    if (
        depth_z16.dtype != np.uint16
        or depth_z16.ndim != 2
        or not math.isfinite(float(depth_scale_m))
        or depth_scale_m <= 0.0
    ):
        return {
            "available": False,
            "valid": False,
            "reason": "invalid aligned depth frame",
            "cups": [],
        }

    dual_cfg = cfg.get("dual_suction", {})
    if not isinstance(dual_cfg, dict):
        dual_cfg = {}
    minimum_ratio = float(dual_cfg.get("min_depth_valid_ratio", 0.8))
    sample_fraction = min(
        1.0, max(0.2, float(dual_cfg.get("depth_sample_radius_fraction", 0.7)))
    )
    radius_px = max(
        2, int(round(target.projected_cup_radius_px * sample_fraction))
    )
    height, width = depth_z16.shape
    cup_reports: list[dict[str, Any]] = []
    for center in target.cup_centers_px:
        cx, cy = (int(round(center[0])), int(round(center[1])))
        x0, x1 = max(0, cx - radius_px), min(width, cx + radius_px + 1)
        y0, y1 = max(0, cy - radius_px), min(height, cy + radius_px + 1)
        region = depth_z16[y0:y1, x0:x1]
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius_px**2
        samples = region[mask]
        valid_samples = samples[samples > 0]
        ratio = float(valid_samples.size) / max(int(samples.size), 1)
        median_m = (
            float(np.median(valid_samples.astype(np.float64)))
            * float(depth_scale_m)
            if valid_samples.size
            else None
        )
        cup_reports.append(
            {
                "center_px": list(center),
                "sample_radius_px": radius_px,
                "valid_ratio": ratio,
                "median_depth_m": median_m,
                "valid": ratio >= minimum_ratio and median_m is not None,
            }
        )

    medians = [
        float(report["median_depth_m"])
        for report in cup_reports
        if report["median_depth_m"] is not None
    ]
    return {
        "available": True,
        "valid": all(bool(report["valid"]) for report in cup_reports),
        "minimum_valid_ratio": minimum_ratio,
        "cups": cup_reports,
        "median_depth_delta_mm": (
            abs(medians[1] - medians[0]) * 1000.0
            if len(medians) == 2
            else None
        ),
        "surface_consistency_gated": False,
    }


def estimate_candidate_physical_size_rgbd(
    candidate: BoxCandidate,
    depth_z16: np.ndarray | None,
    depth_scale_m: float | None,
    intrinsics: np.ndarray | None,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate one candidate as one 130 x 85 mm carton face in 3-D.

    The check samples two interior points on each candidate axis, deprojects
    all four points with synchronized depth, and measures Euclidean distances
    in camera coordinates.  It therefore follows camera distance and surface
    perspective instead of relying on a fixed pixel-area threshold.
    """

    gate = cfg.get("physical_instance_gate", {})
    if not isinstance(gate, dict) or gate.get("enabled") is not True:
        return {
            "enabled": False,
            "available": False,
            "valid": True,
            "method": "disabled",
            "blockers": [],
        }
    expected = gate.get("expected_face_size_mm", [130.0, 85.0])
    tolerance = gate.get("tolerance_mm", [45.0, 35.0])
    try:
        expected_long = float(expected[0])
        expected_short = float(expected[1])
        tolerance_long = float(tolerance[0])
        tolerance_short = float(tolerance[1])
    except (IndexError, TypeError, ValueError):
        return {
            "enabled": True,
            "available": False,
            "valid": False,
            "method": "inner_axis_rgbd",
            "blockers": ["physical_size_config_invalid"],
        }
    if (
        expected_long <= 0.0
        or expected_short <= 0.0
        or tolerance_long <= 0.0
        or tolerance_short <= 0.0
    ):
        return {
            "enabled": True,
            "available": False,
            "valid": False,
            "method": "inner_axis_rgbd",
            "blockers": ["physical_size_config_invalid"],
        }
    if depth_z16 is None or depth_scale_m is None or intrinsics is None:
        return {
            "enabled": True,
            "available": False,
            "valid": False,
            "method": "inner_axis_rgbd",
            "expected_size_mm": [expected_long, expected_short],
            "tolerance_mm": [tolerance_long, tolerance_short],
            "blockers": ["physical_size_depth_unavailable"],
        }
    if depth_z16.dtype != np.uint16 or depth_z16.ndim != 2:
        return {
            "enabled": True,
            "available": False,
            "valid": False,
            "method": "inner_axis_rgbd",
            "blockers": ["physical_size_depth_invalid"],
        }
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return {
            "enabled": True,
            "available": False,
            "valid": False,
            "method": "inner_axis_rgbd",
            "blockers": ["physical_size_intrinsics_invalid"],
        }
    scale = float(depth_scale_m)
    if not math.isfinite(scale) or scale <= 0.0:
        return {
            "enabled": True,
            "available": False,
            "valid": False,
            "method": "inner_axis_rgbd",
            "blockers": ["physical_size_depth_scale_invalid"],
        }

    radius = max(1, int(gate.get("sample_radius_px", 5)))
    minimum_samples = max(1, int(gate.get("minimum_depth_samples", 8)))
    depth_mode = str(gate.get("depth_mode", "endpoint_rgbd")).strip().lower()
    if depth_mode not in {"endpoint_rgbd", "center_depth_projected"}:
        depth_mode = "endpoint_rgbd"
    span_fraction = float(gate.get("interior_span_fraction", 0.70))
    if not 0.30 <= span_fraction <= 0.90:
        span_fraction = 0.70
    half_span = span_fraction / 2.0
    height, width = depth_z16.shape

    def sample_depth_mm(pixel: np.ndarray) -> float | None:
        x = int(round(float(pixel[0])))
        y = int(round(float(pixel[1])))
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return None
        patch = depth_z16[y0:y1, x0:x1]
        values = patch[patch > 0]
        if values.size < minimum_samples:
            return None
        return float(np.median(values.astype(np.float64))) * scale * 1000.0

    angle = math.radians(float(candidate.angle_deg))
    long_axis = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
    short_axis = np.asarray([-long_axis[1], long_axis[0]], dtype=np.float64)
    center = np.asarray(candidate.center_px, dtype=np.float64)
    center_depth = (
        sample_depth_mm(center)
        if depth_mode == "center_depth_projected"
        else None
    )

    def measure_axis(axis: np.ndarray, side_px: float) -> tuple[float | None, list[Any]]:
        first_px = center - axis * float(side_px) * half_span
        second_px = center + axis * float(side_px) * half_span
        if depth_mode == "center_depth_projected":
            # Aligned RGB and depth can retain a few-pixel registration error on
            # a strongly oblique tabletop.  Sampling both ends then turns that
            # registration error into a false carton-length error.  Task scenes
            # with a known planar carton top may project both endpoints using
            # the robust centre depth; cup contact depth is still checked later
            # and independently at both real suction patches.
            first_depth = center_depth
            second_depth = center_depth
        else:
            first_depth = sample_depth_mm(first_px)
            second_depth = sample_depth_mm(second_px)
        samples = [
            [float(first_px[0]), float(first_px[1]), first_depth],
            [float(second_px[0]), float(second_px[1]), second_depth],
        ]
        if first_depth is None or second_depth is None:
            return None, samples
        first_point = deproject_pixel(
            (int(round(first_px[0])), int(round(first_px[1]))),
            first_depth,
            matrix,
        )
        second_point = deproject_pixel(
            (int(round(second_px[0])), int(round(second_px[1]))),
            second_depth,
            matrix,
        )
        measured_mm = (
            float(np.linalg.norm(second_point - first_point))
            * 1000.0
            / span_fraction
        )
        return measured_mm, samples

    measured_long, long_samples = measure_axis(long_axis, candidate.long_side_px)
    measured_short, short_samples = measure_axis(short_axis, candidate.short_side_px)
    blockers: list[str] = []
    if measured_long is None or measured_short is None:
        blockers.append("physical_size_depth_support_low")
    else:
        if abs(measured_long - expected_long) > tolerance_long:
            blockers.append("physical_size_long_mismatch")
        if abs(measured_short - expected_short) > tolerance_short:
            blockers.append("physical_size_short_mismatch")
    return {
        "enabled": True,
        "available": measured_long is not None and measured_short is not None,
        "valid": not blockers,
        "method": (
            "inner_axis_center_depth_projected"
            if depth_mode == "center_depth_projected"
            else "inner_axis_rgbd"
        ),
        "expected_size_mm": [expected_long, expected_short],
        "tolerance_mm": [tolerance_long, tolerance_short],
        "estimated_size_mm": (
            None
            if measured_long is None or measured_short is None
            else [measured_long, measured_short]
        ),
        "interior_span_fraction": span_fraction,
        "samples": {"long": long_samples, "short": short_samples},
        "blockers": blockers,
    }


def split_carton_grid_candidate(
    candidate: BoxCandidate,
    depth_z16: np.ndarray | None,
    depth_scale_m: float | None,
    intrinsics: np.ndarray | None,
    cfg: dict[str, Any],
    rgb: np.ndarray | None = None,
) -> list[BoxCandidate]:
    """Split one verified, merged carton array into individual top faces.

    Touching cartons often form one colour contour, and a repeated printed
    face can still make that large contour pass SIFT verification.  The split
    is not based on a fixed image ROI or fixed pixel area.  It estimates the
    contour's physical dimensions from synchronized depth and camera
    intrinsics, compares them with the configured 130 x 85 mm face, and only
    accepts an integer grid when both axes are close to an integer multiple.

    The inherited face identity remains valid for each repeated cell.  Actual
    contact depth is checked independently for every generated cell later.
    """

    grid_cfg = cfg.get("grid_split", {})
    if not isinstance(grid_cfg, dict) or grid_cfg.get("enabled", True) is not True:
        return [candidate]
    if depth_z16 is None or depth_scale_m is None or intrinsics is None:
        return [candidate]
    if depth_z16.dtype != np.uint16 or depth_z16.ndim != 2:
        return [candidate]
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return [candidate]
    if not math.isfinite(float(depth_scale_m)) or float(depth_scale_m) <= 0.0:
        return [candidate]

    dual_cfg = cfg.get("dual_suction", {})
    if not isinstance(dual_cfg, dict):
        dual_cfg = {}
    face_size = dual_cfg.get("carton_face_size_mm", [130.0, 85.0])
    try:
        face_long_m = float(face_size[0]) / 1000.0
        face_short_m = float(face_size[1]) / 1000.0
    except (IndexError, TypeError, ValueError):
        return [candidate]
    if face_long_m <= 0.0 or face_short_m <= 0.0:
        return [candidate]

    center_x = int(round(candidate.center_px[0]))
    center_y = int(round(candidate.center_px[1]))
    sample_radius = max(2, int(grid_cfg.get("center_depth_radius_px", 7)))
    height, width = depth_z16.shape
    x0 = max(0, center_x - sample_radius)
    x1 = min(width, center_x + sample_radius + 1)
    y0 = max(0, center_y - sample_radius)
    y1 = min(height, center_y + sample_radius + 1)
    region = depth_z16[y0:y1, x0:x1]
    if region.size == 0:
        return [candidate]
    valid_depth = region[region > 0]
    minimum_samples = max(1, int(grid_cfg.get("minimum_depth_samples", 12)))
    if valid_depth.size < minimum_samples:
        return [candidate]
    depth_m = (
        float(np.median(valid_depth.astype(np.float64)))
        * float(depth_scale_m)
    )
    minimum_depth = float(grid_cfg.get("minimum_depth_m", 0.3))
    maximum_depth = float(grid_cfg.get("maximum_depth_m", 1.5))
    if not minimum_depth <= depth_m <= maximum_depth:
        return [candidate]

    angle_rad = math.radians(float(candidate.angle_deg))
    primary_axis = np.asarray(
        [math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float64
    )
    secondary_axis = np.asarray(
        [-primary_axis[1], primary_axis[0]], dtype=np.float64
    )
    fx = float(matrix[0, 0])
    fy = float(matrix[1, 1])

    def directional_focal(axis: np.ndarray) -> float:
        return math.sqrt((fx * axis[0]) ** 2 + (fy * axis[1]) ** 2)

    estimated_primary_m = (
        float(candidate.long_side_px)
        * depth_m
        / max(directional_focal(primary_axis), 1e-9)
    )
    estimated_secondary_m = (
        float(candidate.short_side_px)
        * depth_m
        / max(directional_focal(secondary_axis), 1e-9)
    )
    max_count = max(1, int(grid_cfg.get("maximum_count_per_axis", 4)))
    shape_policy = str(grid_cfg.get("shape_policy", "any")).strip().lower()
    if shape_policy not in {
        "any",
        "bounded_2d_dynamic",
        "single_axis_dynamic",
    }:
        shape_policy = "any"
    count_tolerance = max(
        0.05, float(grid_cfg.get("integer_count_tolerance", 0.55))
    )

    def count_for(raw_count: float) -> int:
        return int(np.clip(round(raw_count), 1, max_count))

    # A merged row whose overall rectangle is 170 x 130 mm has its *primary*
    # rectangle axis along two carton short sides, not along the 130 mm carton
    # long side.  Evaluate both assignments and retain the lower-error one.
    # Without this swap, the generated cell centres can land on the seam
    # between two touching cartons.
    assignments: list[dict[str, Any]] = []
    for long_on_primary in (True, False):
        primary_unit = face_long_m if long_on_primary else face_short_m
        secondary_unit = face_short_m if long_on_primary else face_long_m
        raw_primary_count = estimated_primary_m / primary_unit
        raw_secondary_count = estimated_secondary_m / secondary_unit
        primary_count = count_for(raw_primary_count)
        secondary_count = count_for(raw_secondary_count)
        primary_error = abs(raw_primary_count - primary_count)
        secondary_error = abs(raw_secondary_count - secondary_count)
        assignments.append(
            {
                "long_on_primary": long_on_primary,
                "raw_primary_count": raw_primary_count,
                "raw_secondary_count": raw_secondary_count,
                "primary_count": primary_count,
                "secondary_count": secondary_count,
                "primary_error": primary_error,
                "secondary_error": secondary_error,
                "grid_shape": (
                    (secondary_count, primary_count)
                    if long_on_primary
                    else (primary_count, secondary_count)
                ),
                "valid": bool(
                    primary_error <= count_tolerance
                    and secondary_error <= count_tolerance
                    and (
                        shape_policy != "single_axis_dynamic"
                        or primary_count == 1
                        or secondary_count == 1
                    )
                ),
                "score": primary_error + secondary_error,
            }
        )
    valid_assignments = [item for item in assignments if item["valid"]]
    preferred_shape_value = grid_cfg.get("preferred_grid_shape")
    preferred_shape: tuple[int, int] | None = None
    if (
        isinstance(preferred_shape_value, (list, tuple))
        and len(preferred_shape_value) == 2
    ):
        try:
            parsed_shape = (
                int(preferred_shape_value[0]),
                int(preferred_shape_value[1]),
            )
            if min(parsed_shape) >= 1 and max(parsed_shape) <= max_count:
                preferred_shape = parsed_shape
        except (TypeError, ValueError):
            preferred_shape = None
    preferred_tolerance = max(
        count_tolerance,
        float(grid_cfg.get("preferred_grid_count_tolerance", count_tolerance)),
    )
    preferred_assignments = [
        item
        for item in assignments
        if shape_policy != "single_axis_dynamic"
        if item["grid_shape"] == preferred_shape
        and float(item["primary_error"]) <= preferred_tolerance
        and float(item["secondary_error"]) <= preferred_tolerance
    ]
    preferred_assignment = (
        min(preferred_assignments, key=lambda item: float(item["score"]))
        if preferred_assignments
        else None
    )
    assignment = (
        {**preferred_assignment, "valid": True, "preferred_override": True}
        if preferred_assignment is not None
        else (
            min(valid_assignments, key=lambda item: float(item["score"]))
            if valid_assignments
            else min(assignments, key=lambda item: float(item["score"]))
        )
    )

    area_ratio = (
        estimated_primary_m
        * estimated_secondary_m
        / max(face_long_m * face_short_m, 1e-9)
    )
    maximum_single_area_ratio = max(
        1.05,
        float(grid_cfg.get("maximum_single_carton_area_ratio", 1.45)),
    )

    def reject_unsafe_cluster(reason: str) -> list[BoxCandidate]:
        blockers = tuple(
            dict.fromkeys((*candidate.grasp_blockers, reason))
        )
        return [
            replace(
                candidate,
                graspable=False,
                grasp_blockers=blockers,
            )
        ]

    if not assignment["valid"]:
        if area_ratio > maximum_single_area_ratio:
            return reject_unsafe_cluster("touching_cartons_unsplit")
        return [candidate]

    primary_count = int(assignment["primary_count"])
    secondary_count = int(assignment["secondary_count"])
    if primary_count * secondary_count <= 1:
        if area_ratio > maximum_single_area_ratio:
            return reject_unsafe_cluster("touching_cartons_unsplit")
        return [candidate]

    if bool(assignment["long_on_primary"]):
        long_axis = primary_axis
        short_axis = secondary_axis
        long_count = primary_count
        short_count = secondary_count
        cell_long_px = float(candidate.long_side_px) / long_count
        cell_short_px = float(candidate.short_side_px) / short_count
        cell_angle_deg = float(candidate.angle_deg)
    else:
        long_axis = secondary_axis
        short_axis = -primary_axis
        long_count = secondary_count
        short_count = primary_count
        cell_long_px = float(candidate.short_side_px) / long_count
        cell_short_px = float(candidate.long_side_px) / short_count
        cell_angle_deg = float(candidate.angle_deg) + 90.0
        while cell_angle_deg >= 90.0:
            cell_angle_deg -= 180.0
        while cell_angle_deg < -90.0:
            cell_angle_deg += 180.0

    minimum_cell_side_px = float(grid_cfg.get("minimum_cell_side_px", 24.0))
    if min(cell_long_px, cell_short_px) < minimum_cell_side_px:
        return reject_unsafe_cluster("touching_cartons_unsplit")

    parent_center = np.asarray(candidate.center_px, dtype=np.float64)
    cells: list[BoxCandidate] = []
    for short_index in range(short_count):
        short_offset = (
            short_index - (short_count - 1) / 2.0
        ) * cell_short_px
        for long_index in range(long_count):
            long_offset = (
                long_index - (long_count - 1) / 2.0
            ) * cell_long_px
            center = (
                parent_center
                + long_offset * long_axis
                + short_offset * short_axis
            )
            half_long = cell_long_px / 2.0
            half_short = cell_short_px / 2.0
            polygon = tuple(
                (
                    float(center[0] + sx * half_long * long_axis[0]
                          + sy * half_short * short_axis[0]),
                    float(center[1] + sx * half_long * long_axis[1]
                          + sy * half_short * short_axis[1]),
                )
                for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
            )
            cells.append(
                BoxCandidate(
                    center_px=(float(center[0]), float(center[1])),
                    suction_px=(int(round(center[0])), int(round(center[1]))),
                    polygon_px=polygon,
                    long_side_px=cell_long_px,
                    short_side_px=cell_short_px,
                    angle_deg=cell_angle_deg,
                    rectangularity=float(candidate.rectangularity),
                    bright_fill=float(candidate.bright_fill),
                    edge_clearance_px=min(cell_long_px, cell_short_px) / 2.0,
                    score=float(candidate.score),
                    provider=str(candidate.provider) + ":grid_cell",
                    face_type=str(candidate.face_type),
                    face_score=float(candidate.face_score),
                    reference_face_id=candidate.reference_face_id,
                    graspable=bool(candidate.graspable),
                    grasp_blockers=tuple(candidate.grasp_blockers),
                    grid_shape=(short_count, long_count),
                    grid_index=(short_index, long_index),
                    grid_parent_center_px=(
                        float(candidate.center_px[0]),
                        float(candidate.center_px[1]),
                    ),
                )
            )

    if rgb is not None and grid_cfg.get("require_face_color_support", True) is True:
        if (
            rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[2] != 3
            or rgb.shape[:2] != depth_z16.shape[:2]
        ):
            return [candidate]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        pink = (
            (hue >= int(cfg.get("pink_hue_min", 135)))
            & (hue <= int(cfg.get("pink_hue_max", 179)))
            & (saturation >= int(cfg.get("pink_saturation_min", 8)))
            & (value >= int(cfg.get("pink_value_min", 115)))
        )
        cell_pink_fractions: list[float] = []
        for cell in cells:
            polygon = np.round(np.asarray(cell.polygon_px)).astype(np.int32)
            cell_mask = np.zeros(depth_z16.shape, dtype=np.uint8)
            cv2.fillConvexPoly(cell_mask, polygon, 255)
            inside = cell_mask > 0
            pixel_count = int(np.count_nonzero(inside))
            cell_pink_fractions.append(
                float(np.count_nonzero(pink & inside)) / max(pixel_count, 1)
            )
        minimum_fraction = float(
            grid_cfg.get("minimum_cell_pink_fraction", 0.01)
        )
        minimum_supported_ratio = float(
            grid_cfg.get("minimum_supported_cell_ratio", 0.50)
        )
        supported_ratio = float(
            np.mean(
                np.asarray(cell_pink_fractions, dtype=np.float64)
                >= minimum_fraction
            )
        )
        drop_unsupported = grid_cfg.get("drop_unsupported_cells", False) is True
        if drop_unsupported:
            cells = [
                cell
                for cell, fraction in zip(cells, cell_pink_fractions)
                if fraction >= minimum_fraction
            ]
            if not cells:
                return reject_unsafe_cluster(
                    "touching_cartons_split_unverified"
                )
        elif supported_ratio < minimum_supported_ratio:
            return reject_unsafe_cluster(
                "touching_cartons_split_unverified"
            )

    # Start at the geometric centre, then expand outwards.  The console may
    # later reorder these cells by measured layer height.
    cells.sort(
        key=lambda item: (
            (item.grid_index[0] - (short_count - 1) / 2.0) ** 2
            + (item.grid_index[1] - (long_count - 1) / 2.0) ** 2,
            item.grid_index,
        )
    )
    return cells


def _roi_pixels(
    shape: tuple[int, int],
    roi_norm: Iterable[float],
) -> tuple[int, int, int, int]:
    height, width = shape
    values = tuple(float(value) for value in roi_norm)
    if len(values) != 4:
        raise ValueError("roi_norm must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = values
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"invalid normalized ROI: {values}")
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )


def _long_axis_angle(rect: tuple[Any, Any, float]) -> float:
    (_, _), (width, height), angle = rect
    result = float(angle if width >= height else angle + 90.0)
    while result >= 90.0:
        result -= 180.0
    while result < -90.0:
        result += 180.0
    return result


def _bright_carton_mask(rgb: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)

    bright = (
        (value >= int(cfg.get("bright_value_min", 145)))
        & (saturation <= int(cfg.get("bright_saturation_max", 105)))
    )
    pink = (
        (hue >= int(cfg.get("pink_hue_min", 135)))
        & (hue <= int(cfg.get("pink_hue_max", 179)))
        & (saturation >= int(cfg.get("pink_saturation_min", 8)))
        & (value >= int(cfg.get("pink_value_min", 115)))
    )
    mask = np.where(bright | pink, 255, 0).astype(np.uint8)

    open_kernel = max(1, int(cfg.get("open_kernel_px", 3)))
    close_kernel = max(1, int(cfg.get("close_kernel_px", 7)))
    if open_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (close_kernel, close_kernel)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_cartons(rgb: np.ndarray, cfg: dict[str, Any]) -> list[BoxCandidate]:
    """Return filtered, score-sorted carton top-surface candidates."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must be uint8 HxWx3, got {rgb.dtype} {rgb.shape}")

    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = _roi_pixels((height, width), cfg["roi_norm"])
    roi_rgb = rgb[y1:y2, x1:x2]
    roi_mask = _bright_carton_mask(roi_rgb, cfg)
    contours, _ = cv2.findContours(
        roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = float(height * width)
    image_short = float(min(height, width))
    min_area_fraction = float(cfg.get("min_area_fraction", 0.0001))
    max_area_fraction = float(cfg.get("max_area_fraction", 0.30))
    min_short_fraction = float(cfg.get("min_short_side_fraction", 0.01))
    max_short_fraction = float(cfg.get("max_short_side_fraction", 0.50))
    expected_aspect = float(cfg.get("expected_aspect_ratio", 2.1))
    min_aspect = float(cfg.get("min_aspect_ratio", 1.45))
    max_aspect = float(cfg.get("max_aspect_ratio", 3.2))
    min_rectangularity = float(cfg.get("min_rectangularity", 0.50))
    min_bright_fill = float(cfg.get("min_bright_fill", 0.42))
    min_edge_clearance = float(cfg.get("min_edge_clearance_px", 8.0))

    candidates: list[BoxCandidate] = []
    for contour_local in contours:
        contour_area = float(cv2.contourArea(contour_local))
        if contour_area <= 0.0:
            continue
        rect_local = cv2.minAreaRect(contour_local)
        (_, _), (raw_width, raw_height), _ = rect_local
        long_side = float(max(raw_width, raw_height))
        short_side = float(min(raw_width, raw_height))
        rect_area = long_side * short_side
        area_fraction = rect_area / max(image_area, 1.0)
        if not (min_area_fraction <= area_fraction <= max_area_fraction):
            continue
        short_fraction = short_side / max(image_short, 1.0)
        if not (min_short_fraction <= short_fraction <= max_short_fraction):
            continue
        aspect = long_side / max(short_side, 1e-6)
        if not (min_aspect <= aspect <= max_aspect):
            continue
        rectangularity = min(1.0, contour_area / max(rect_area, 1e-6))
        if rectangularity < min_rectangularity:
            continue

        component = np.zeros_like(roi_mask)
        cv2.drawContours(component, [contour_local], -1, 255, thickness=-1)
        box_local = cv2.boxPoints(rect_local)
        rectangle_mask = np.zeros_like(roi_mask)
        cv2.fillConvexPoly(
            rectangle_mask, np.round(box_local).astype(np.int32), 255
        )
        rectangle_pixels = rectangle_mask > 0
        bright_fill = float(
            np.count_nonzero((roi_mask > 0) & rectangle_pixels)
            / max(np.count_nonzero(rectangle_pixels), 1)
        )
        if bright_fill < min_bright_fill:
            continue

        distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        _, edge_clearance, _, suction_local = cv2.minMaxLoc(distance)
        if float(edge_clearance) < min_edge_clearance:
            continue

        aspect_score = math.exp(
            -1.35 * abs(math.log(max(aspect, 1e-6) / expected_aspect))
        )
        clearance_score = min(
            1.0, float(edge_clearance) / max(min_edge_clearance * 1.8, 1.0)
        )
        score = float(
            0.38 * aspect_score
            + 0.27 * rectangularity
            + 0.22 * bright_fill
            + 0.13 * clearance_score
        )

        center_local = rect_local[0]
        polygon = tuple(
            (float(px + x1), float(py + y1)) for px, py in box_local
        )
        candidates.append(
            BoxCandidate(
                center_px=(
                    float(center_local[0] + x1),
                    float(center_local[1] + y1),
                ),
                suction_px=(int(suction_local[0] + x1), int(suction_local[1] + y1)),
                polygon_px=polygon,
                long_side_px=long_side,
                short_side_px=short_side,
                angle_deg=_long_axis_angle(rect_local),
                rectangularity=rectangularity,
                bright_fill=bright_fill,
                edge_clearance_px=float(edge_clearance),
                score=score,
            )
        )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates


def propose_task2_single_row(
    rgb: np.ndarray,
    depth_z16: np.ndarray | None,
    depth_scale_m: float | None,
    intrinsics: np.ndarray | None,
    seed_candidates: Iterable[BoxCandidate],
    *,
    roi_norm: Iterable[float],
    config: dict[str, Any],
    maximum_count: int = 4,
) -> list[BoxCandidate]:
    """Recover the Task2 source row as individual vertical cartons.

    The reference-face detector is still responsible for medicine-carton
    identity.  This helper only replaces its geometry: repeated adjacent
    faces can produce one oversized, tilted homography even though the real
    Task2 layout is a horizontal row of 1--4 vertical 130 x 85 mm cartons.
    Geometry is recovered from the live carton-colour support and RGB-D
    scale, then every generated cell is checked again by the normal physical
    size, task-region, depth and dual-suction gates.
    """

    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be uint8 HxWx3")
    if depth_z16 is None or depth_scale_m is None or intrinsics is None:
        return []
    if depth_z16.dtype != np.uint16 or depth_z16.shape != rgb.shape[:2]:
        return []
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return []
    scale = float(depth_scale_m)
    if not math.isfinite(scale) or scale <= 0.0:
        return []

    try:
        roi = [float(value) for value in roi_norm]
    except (TypeError, ValueError):
        return []
    if len(roi) != 4:
        return []
    image_height, image_width = rgb.shape[:2]
    x0 = int(round(np.clip(roi[0], 0.0, 1.0) * image_width))
    y0 = int(round(np.clip(roi[1], 0.0, 1.0) * image_height))
    x1 = int(round(np.clip(roi[2], 0.0, 1.0) * image_width))
    y1 = int(round(np.clip(roi[3], 0.0, 1.0) * image_height))
    if x1 <= x0 or y1 <= y0:
        return []

    verified_seeds = [
        candidate
        for candidate in seed_candidates
        if x0 <= float(candidate.center_px[0]) < x1
        and y0 <= float(candidate.center_px[1]) < y1
        and candidate.reference_face_id is not None
        and candidate.face_type in {"front_large", "back_large"}
    ]
    if not verified_seeds:
        return []
    identity_seed = max(verified_seeds, key=lambda item: float(item.score))

    dual_cfg = config.get("dual_suction", {})
    if not isinstance(dual_cfg, dict):
        dual_cfg = {}
    face_size = dual_cfg.get("carton_face_size_mm", [130.0, 85.0])
    try:
        face_long_m = float(face_size[0]) / 1000.0
        face_short_m = float(face_size[1]) / 1000.0
    except (IndexError, TypeError, ValueError):
        return []
    if face_long_m <= 0.0 or face_short_m <= 0.0:
        return []

    roi_rgb = rgb[y0:y1, x0:x1]
    mask_cfg = dict(config)
    mask_cfg["open_kernel_px"] = int(
        config.get("task2_row_open_kernel_px", 3)
    )
    mask_cfg["close_kernel_px"] = int(
        config.get("task2_row_close_kernel_px", 5)
    )
    mask = _bright_carton_mask(roi_rgb, mask_cfg)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    fx = float(matrix[0, 0])
    fy = float(matrix[1, 1])
    maximum_count = int(np.clip(maximum_count, 1, 4))
    minimum_fill = float(config.get("task2_row_minimum_fill", 0.24))
    maximum_axis_error = float(
        config.get("task2_row_maximum_axis_relative_error", 0.48)
    )
    minimum_component_area = int(
        config.get("task2_row_minimum_component_area_px", 1200)
    )

    proposals: list[tuple[float, list[BoxCandidate]]] = []
    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)
        if (
            bw < 24
            or bh < 40
            or int(cv2.contourArea(contour)) < minimum_component_area
        ):
            continue
        component_region = mask[by : by + bh, bx : bx + bw]
        fill = float(np.count_nonzero(component_region)) / max(bw * bh, 1)
        if fill < minimum_fill:
            continue

        # Depth is sampled from the central interior, avoiding the white flap
        # and the narrow seams between touching cartons.
        cx_local = bx + bw / 2.0
        cy_local = by + bh * 0.58
        cx = int(round(x0 + cx_local))
        cy = int(round(y0 + cy_local))
        radius = 8
        depth_region = depth_z16[
            max(0, cy - radius) : min(image_height, cy + radius + 1),
            max(0, cx - radius) : min(image_width, cx + radius + 1),
        ]
        valid_depth = depth_region[depth_region > 0]
        if valid_depth.size < 12:
            continue
        depth_m = float(np.median(valid_depth.astype(np.float64))) * scale
        if not 0.30 <= depth_m <= 1.50:
            continue

        expected_width_px = face_short_m * fx / depth_m
        expected_height_px = face_long_m * fy / depth_m
        if min(expected_width_px, expected_height_px) < 20.0:
            continue
        count = int(np.clip(round(bw / expected_width_px), 1, maximum_count))
        width_error = abs(bw / max(expected_width_px * count, 1e-6) - 1.0)
        height_error = abs(bh / max(expected_height_px, 1e-6) - 1.0)
        if width_error > maximum_axis_error or height_error > maximum_axis_error:
            continue

        # Keep the measured row centre, but use the RGB-D projected physical
        # face dimensions for each cell.  This prevents one bad homography
        # from producing a slanted box that crosses a carton boundary.
        row_center_y = float(y0 + by + bh / 2.0)
        row_left = float(x0 + bx)
        measured_pitch = float(bw) / count
        cells: list[BoxCandidate] = []
        for index in range(count):
            center_x = row_left + (index + 0.5) * measured_pitch
            center_y = row_center_y
            half_long = expected_height_px / 2.0
            half_short = expected_width_px / 2.0
            polygon = (
                (center_x - half_short, center_y - half_long),
                (center_x + half_short, center_y - half_long),
                (center_x + half_short, center_y + half_long),
                (center_x - half_short, center_y + half_long),
            )
            cells.append(
                BoxCandidate(
                    center_px=(center_x, center_y),
                    suction_px=(int(round(center_x)), int(round(center_y))),
                    polygon_px=polygon,
                    long_side_px=float(expected_height_px),
                    short_side_px=float(expected_width_px),
                    angle_deg=-90.0,
                    rectangularity=min(1.0, fill),
                    bright_fill=min(1.0, fill),
                    edge_clearance_px=min(
                        expected_height_px, expected_width_px
                    )
                    / 2.0,
                    score=float(identity_seed.score),
                    provider=str(identity_seed.provider) + ":task2_row_cell",
                    face_type=str(identity_seed.face_type),
                    face_score=float(identity_seed.face_score),
                    reference_face_id=identity_seed.reference_face_id,
                    graspable=bool(identity_seed.graspable),
                    grasp_blockers=tuple(identity_seed.grasp_blockers),
                    grid_shape=(count, 1),
                    grid_index=(index, 0),
                    grid_parent_center_px=(
                        row_left + bw / 2.0,
                        row_center_y,
                    ),
                )
            )
        score = (
            count * 10.0
            + fill
            - width_error
            - height_error
            + 0.15 * (row_center_y / max(image_height, 1))
        )
        proposals.append((score, cells))

    if not proposals:
        return []

    # Narrow green seams can leave one contour per carton instead of one
    # contour for the whole row.  Join compatible proposals so the same code
    # handles both touching and slightly separated 4 -> 3 -> 2 -> 1 layouts.
    cells = [cell for _, proposal in proposals for cell in proposal]
    cells.sort(key=lambda item: float(item.center_px[0]))
    rows: list[list[BoxCandidate]] = []
    for cell in cells:
        compatible_rows = [
            row
            for row in rows
            if abs(float(cell.center_px[1]) - np.median(
                [float(item.center_px[1]) for item in row]
            ))
            <= 0.45 * float(cell.long_side_px)
        ]
        if compatible_rows:
            compatible_rows[0].append(cell)
        else:
            rows.append([cell])

    normalized_rows: list[list[BoxCandidate]] = []
    for row in rows:
        row.sort(key=lambda item: float(item.center_px[0]))
        runs: list[list[BoxCandidate]] = []
        for cell in row:
            if not runs:
                runs.append([cell])
                continue
            previous = runs[-1][-1]
            maximum_gap = 1.85 * max(
                float(previous.short_side_px), float(cell.short_side_px)
            )
            if float(cell.center_px[0]) - float(previous.center_px[0]) <= maximum_gap:
                runs[-1].append(cell)
            else:
                runs.append([cell])
        normalized_rows.extend(runs)

    best = max(
        normalized_rows,
        key=lambda row: (
            min(len(row), maximum_count),
            float(np.mean([item.bright_fill for item in row])),
            float(np.mean([item.center_px[1] for item in row])),
        ),
    )[:maximum_count]
    count = len(best)
    parent_center = (
        float(np.mean([item.center_px[0] for item in best])),
        float(np.mean([item.center_px[1] for item in best])),
    )
    return [
        replace(
            cell,
            grid_shape=(count, 1),
            grid_index=(index, 0),
            grid_parent_center_px=parent_center,
        )
        for index, cell in enumerate(best)
    ]


class WebConsoleCamera:
    """Read the existing web-console stream without taking the RealSense lock."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8765",
        camera: str = "front",
        origin: str = "http://127.0.0.1:8765",
    ):
        self.base_url = base_url.rstrip("/")
        self.camera = camera
        self.origin = origin
        self.session = requests.Session()
        response = self.session.get(
            f"{self.base_url}/api/auth/session", timeout=(3.0, 5.0)
        )
        response.raise_for_status()
        auth = response.json()
        if not auth.get("authenticated"):
            raise RuntimeError("web console did not issue an authenticated session")
        self.csrf_token = str(auth.get("csrf_token", ""))

    def capture_rgb(self, timeout_s: float = 8.0) -> np.ndarray:
        url = f"{self.base_url}/api/cameras/{self.camera}/stream.mjpg"
        response = self.session.get(
            url, stream=True, timeout=(3.0, timeout_s)
        )
        response.raise_for_status()
        payload = bytearray()
        started = time.monotonic()
        try:
            for chunk in response.iter_content(chunk_size=16384):
                if chunk:
                    payload.extend(chunk)
                start = payload.find(b"\xff\xd8")
                end = payload.find(b"\xff\xd9", max(0, start + 2))
                if start >= 0 and end >= 0:
                    encoded = np.frombuffer(
                        bytes(payload[start : end + 2]), dtype=np.uint8
                    )
                    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise RuntimeError("failed to decode web-console JPEG frame")
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if time.monotonic() - started > timeout_s:
                    raise TimeoutError("timed out waiting for one MJPEG frame")
                if len(payload) > 8 * 1024 * 1024:
                    del payload[:4 * 1024 * 1024]
        finally:
            response.close()
        raise RuntimeError("web-console MJPEG stream ended without a JPEG frame")

    def depth_at(
        self,
        pixel: tuple[int, int],
        image_shape: tuple[int, int],
    ) -> dict[str, Any]:
        height, width = image_shape
        u, v = pixel
        params = {
            "x": min(max(float(u) / max(width - 1, 1), 0.0), 1.0),
            "y": min(max(float(v) / max(height - 1, 1), 0.0), 1.0),
        }
        response = self.session.get(
            f"{self.base_url}/api/cameras/{self.camera}/depth",
            params=params,
            timeout=(3.0, 5.0),
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(f"depth endpoint failed: {payload}")
        return dict(payload["depth"])

    def set_suction(self, engaged: bool) -> dict[str, Any]:
        """Available for the later motion stage; not used by detection."""
        response = self.session.post(
            f"{self.base_url}/api/suction",
            json={"engaged": bool(engaged)},
            headers={
                "Origin": self.origin,
                "X-CSRF-Token": self.csrf_token,
            },
            timeout=(3.0, 8.0),
        )
        response.raise_for_status()
        return dict(response.json())


def sample_candidate_depth(
    camera: WebConsoleCamera,
    candidate: BoxCandidate,
    image_shape: tuple[int, int],
    cfg: dict[str, Any],
) -> tuple[DepthEstimate | None, list[str]]:
    offsets = cfg.get(
        "depth_sample_offsets_px",
        [[0, 0], [-8, 0], [8, 0], [0, -8], [0, 8]],
    )
    minimum_ratio = float(cfg.get("min_depth_valid_ratio", 0.75))
    maximum_age = float(cfg.get("max_depth_age_s", 0.5))
    values: list[float] = []
    valid_pixels: list[tuple[int, int]] = []
    ages: list[float] = []
    query_errors: list[str] = []
    for dx, dy in offsets:
        pixel = (
            int(round(candidate.suction_px[0] + float(dx))),
            int(round(candidate.suction_px[1] + float(dy))),
        )
        try:
            depth = camera.depth_at(pixel, image_shape)
        except Exception as exc:
            query_errors.append(f"depth query failed at {pixel}: {exc}")
            continue
        if not bool(depth.get("available")):
            query_errors.append(
                f"depth unavailable at {pixel}: {depth.get('reason', '')}"
            )
            continue
        age = float(depth.get("age_s", math.inf))
        ratio = float(depth.get("target_valid_ratio", 0.0))
        value = float(depth.get("target_mm", 0.0))
        if bool(depth.get("stale")) or age > maximum_age:
            query_errors.append(f"stale depth at {pixel}: age={age:.3f}s")
            continue
        if ratio < minimum_ratio or value <= 0.0:
            query_errors.append(
                f"invalid depth at {pixel}: value={value:.1f} ratio={ratio:.2f}"
            )
            continue
        values.append(value)
        valid_pixels.append(pixel)
        ages.append(age)

    minimum_samples = int(cfg.get("min_depth_samples", 3))
    if len(values) < minimum_samples:
        errors = [
            f"only {len(values)} valid depth samples; require {minimum_samples}"
        ]
        errors.extend(query_errors[:2])
        return None, errors

    errors: list[str] = []
    median = float(np.median(np.asarray(values, dtype=np.float64)))
    spread = float(max(values) - min(values))
    return (
        DepthEstimate(
            median_mm=median,
            spread_mm=spread,
            valid_samples=len(values),
            samples_mm=tuple(values),
            sample_pixels_px=tuple(valid_pixels),
            frame_age_s=max(ages),
        ),
        errors,
    )


def deproject_pixel(
    pixel: tuple[int, int],
    depth_mm: float,
    intrinsics: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"intrinsics must be 3x3, got {matrix.shape}")
    u, v = (float(value) for value in pixel)
    z = float(depth_mm) / 1000.0
    x = (u - matrix[0, 2]) * z / matrix[0, 0]
    y = (v - matrix[1, 2]) * z / matrix[1, 1]
    return np.array([x, y, z], dtype=np.float64)


def transform_point(point: np.ndarray, transform: np.ndarray) -> np.ndarray:
    point_value = np.asarray(point, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if point_value.shape != (3,) or matrix.shape != (4, 4):
        raise ValueError("point must be (3,) and transform must be (4,4)")
    return (matrix @ np.array([*point_value, 1.0], dtype=np.float64))[:3]


def load_cam_to_left(path: str | Path) -> np.ndarray:
    payload = load_json(path)
    transform = np.asarray(payload["cam_to_base"], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"cam_to_base must be 4x4: {path}")
    return transform


def _inside_workspace(point: np.ndarray, workspace: dict[str, Any]) -> bool:
    for index, axis in enumerate(("x", "y", "z")):
        lower, upper = (float(value) for value in workspace[axis])
        if not lower <= float(point[index]) <= upper:
            return False
    return True


def estimate_surface_plane(
    depth: DepthEstimate,
    intrinsics: np.ndarray,
    cam_to_left: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Fit the sampled top surface and return left-base normal, tilt and residual.

    Raw depth naturally changes across a horizontal surface when the camera is
    oblique.  Plane residual is therefore the correct flatness check; comparing
    the raw minimum and maximum depth would reject valid cartons.
    """
    points = np.vstack(
        [
            deproject_pixel(pixel, value, intrinsics)
            for pixel, value in zip(
                depth.sample_pixels_px, depth.samples_mm, strict=True
            )
        ]
    )
    center = points.mean(axis=0)
    _, _, vectors = np.linalg.svd(points - center)
    normal_camera = vectors[-1]
    normal_left = np.asarray(cam_to_left, dtype=np.float64)[:3, :3] @ normal_camera
    if normal_left[2] < 0.0:
        normal_camera = -normal_camera
        normal_left = -normal_left
    normal_left = normal_left / max(float(np.linalg.norm(normal_left)), 1e-12)
    residual_mm = float(
        np.max(np.abs((points - center) @ normal_camera)) * 1000.0
    )
    tilt_deg = float(
        math.degrees(math.acos(float(np.clip(normal_left[2], -1.0, 1.0))))
    )
    return normal_left, tilt_deg, residual_mm


def locate_candidate(
    candidate: BoxCandidate,
    depth: DepthEstimate | None,
    intrinsics: np.ndarray | None,
    cam_to_left: np.ndarray | None,
    cfg: dict[str, Any],
    inherited_blockers: Iterable[str] = (),
) -> LocatedBox:
    blockers = list(inherited_blockers)
    point_camera: np.ndarray | None = None
    point_base: np.ndarray | None = None
    size: tuple[float, float] | None = None
    surface_normal: np.ndarray | None = None
    surface_tilt: float | None = None
    plane_residual: float | None = None
    reachable: bool | None = None

    if not candidate.graspable:
        if candidate.grasp_blockers:
            blockers.extend(
                f"2D grasp policy: {reason}"
                for reason in candidate.grasp_blockers
            )
        else:
            blockers.append("2D grasp policy did not authorize this candidate")
    if candidate.score < float(cfg.get("min_detection_score", 0.68)):
        blockers.append(
            f"detection score {candidate.score:.3f} below "
            f"{float(cfg.get('min_detection_score', 0.68)):.3f}"
        )
    if depth is None:
        blockers.append("no valid target depth")
    elif intrinsics is None:
        blockers.append("camera intrinsics unavailable")
    else:
        point_camera = deproject_pixel(
            candidate.suction_px, depth.median_mm, intrinsics
        )
        focal_mean = float((intrinsics[0, 0] + intrinsics[1, 1]) * 0.5)
        size = (
            candidate.long_side_px * point_camera[2] / focal_mean,
            candidate.short_side_px * point_camera[2] / focal_mean,
        )
        physical = cfg.get(
            "physical_size_limits_m",
            {"long": [0.08, 0.25], "short": [0.03, 0.13]},
        )
        if not float(physical["long"][0]) <= size[0] <= float(physical["long"][1]):
            blockers.append(f"estimated long side {size[0]:.3f}m outside limits")
        if not float(physical["short"][0]) <= size[1] <= float(
            physical["short"][1]
        ):
            blockers.append(f"estimated short side {size[1]:.3f}m outside limits")

        if cam_to_left is None:
            blockers.append("camera-to-left-base calibration unavailable")
        else:
            point_base = transform_point(point_camera, cam_to_left)
            surface_normal, surface_tilt, plane_residual = estimate_surface_plane(
                depth, intrinsics, cam_to_left
            )
            maximum_residual = float(cfg.get("max_plane_residual_mm", 3.0))
            maximum_tilt = float(cfg.get("max_surface_tilt_deg", 18.0))
            if plane_residual > maximum_residual:
                blockers.append(
                    f"top surface plane residual {plane_residual:.1f}mm exceeds "
                    f"{maximum_residual:.1f}mm"
                )
            if surface_tilt > maximum_tilt:
                blockers.append(
                    f"top surface tilt {surface_tilt:.1f}deg exceeds "
                    f"{maximum_tilt:.1f}deg"
                )
            reachable = _inside_workspace(point_base, cfg["left_workspace_m"])
            if not reachable:
                blockers.append(
                    "target surface point is outside the configured left-arm workspace"
                )

    return LocatedBox(
        candidate=candidate,
        depth=depth,
        point_camera_m=(
            None if point_camera is None else tuple(float(x) for x in point_camera)
        ),
        point_left_base_m=(
            None if point_base is None else tuple(float(x) for x in point_base)
        ),
        physical_size_m=size,
        surface_normal_left_base=(
            None
            if surface_normal is None
            else tuple(float(x) for x in surface_normal)
        ),
        surface_tilt_deg=surface_tilt,
        plane_residual_mm=plane_residual,
        reachable=reachable,
        blockers=tuple(dict.fromkeys(blockers)),
    )


def draw_overlay(
    rgb: np.ndarray,
    candidates: list[BoxCandidate],
    selected: LocatedBox | None,
    cfg: dict[str, Any],
    dual_targets: dict[int, Any] | None = None,
) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = _roi_pixels(rgb.shape[:2], cfg["roi_norm"])
    roi_is_full_frame = (
        x1 <= 0
        and y1 <= 0
        and x2 >= rgb.shape[1]
        and y2 >= rgb.shape[0]
    )
    if not roi_is_full_frame:
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 215, 255), 2)
        cv2.putText(
            bgr,
            "TASK1 SEARCH AREA",
            (x1 + 4, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 215, 255),
            2,
            cv2.LINE_AA,
        )

    selected_candidate = None if selected is None else selected.candidate
    for index, candidate in enumerate(candidates, start=1):
        is_selected = candidate is selected_candidate
        if is_selected and candidate.graspable:
            color = (60, 220, 60)
        elif is_selected:
            color = (0, 150, 255)
        else:
            color = (180, 140, 40)
        polygon = np.round(np.asarray(candidate.polygon_px)).astype(np.int32)
        cv2.polylines(bgr, [polygon], True, color, 2)
        dual_target = (
            dual_targets.get(id(candidate))
            if dual_targets is not None
            else None
        )
        if dual_target is None:
            dual_target = plan_dual_suction_target(
                candidate, cfg, image_shape=rgb.shape[:2]
            )
        if dual_target is not None:
            cup_points = [
                tuple(int(round(value)) for value in center)
                for center in dual_target.cup_centers_px
            ]
            midpoint = tuple(
                int(round(value)) for value in dual_target.midpoint_px
            )
            cv2.line(bgr, cup_points[0], cup_points[1], color, 2)
            for cup_point in cup_points:
                cv2.circle(
                    bgr,
                    cup_point,
                    max(3, int(round(dual_target.projected_cup_radius_px))),
                    color,
                    2,
                )
                cv2.circle(bgr, cup_point, 2, color, -1)
            cv2.drawMarker(
                bgr,
                midpoint,
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2,
            )
        else:
            suction = candidate.suction_px
            cv2.circle(
                bgr,
                suction,
                max(4, int(round(float(cfg.get("cup_radius_px", 8))))),
                color,
                2,
            )
            cv2.circle(bgr, suction, 3, color, -1)
        cv2.putText(
            bgr,
            (
                f"#{index} det={candidate.score:.2f} "
                f"face={candidate.face_type}:{candidate.face_score:.2f} "
                + (
                    f"dual={'PASS' if dual_target.valid_2d else 'NO-PICK'}"
                    if dual_target is not None
                    else (
                        "PICK-2D" if candidate.graspable else "NO-PICK"
                    )
                )
            ),
            (int(candidate.center_px[0]) - 50, int(candidate.center_px[1]) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )

    if selected is not None:
        lines = []
        if selected.depth is not None:
            lines.append(
                f"depth={selected.depth.median_mm:.0f}mm "
                f"spread={selected.depth.spread_mm:.1f}mm"
            )
        if selected.point_left_base_m is not None:
            x, y, z = selected.point_left_base_m
            lines.append(f"L-base xyz=({x:.3f},{y:.3f},{z:.3f})m")
        if selected.surface_tilt_deg is not None:
            lines.append(
                f"surface tilt={selected.surface_tilt_deg:.1f}deg "
                f"plane residual={selected.plane_residual_mm:.1f}mm"
            )
        lines.append("READY" if selected.ok else "BLOCKED: " + selected.blockers[0])
        background = (35, 100, 35) if selected.ok else (35, 35, 180)
        top = 28
        cv2.rectangle(bgr, (12, 8), (min(bgr.shape[1] - 12, 620), 20 + 24 * len(lines)), background, -1)
        for index, line in enumerate(lines):
            cv2.putText(
                bgr,
                line[:85],
                (20, top + index * 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return bgr
