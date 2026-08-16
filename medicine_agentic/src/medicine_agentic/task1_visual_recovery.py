"""Task1-only 3x3 medicine-carton recovery.

The primary detector still owns carton identity: this module never creates a
candidate unless at least two individually verified Task1 faces are present.
It uses the current frame's colour, edge and aligned-depth evidence only to
recover a bounded number of faces missed by SIFT.  Task2 deliberately does
not import or call this module.
"""
from __future__ import annotations

import math
import threading
from dataclasses import replace
from typing import Any, Iterable

import cv2
import numpy as np

from medicine_agentic.task1_box import BoxCandidate, propose_task1_surface_grid
from medicine_agentic.task2_visual_detector import Task2AdaptiveVisualDetector


class Task1StackOccupancyPrior:
    """Track consumed cells in the fixed 3 x 3 x 3 Task1 stack.

    Array index zero is physical layer 1 (bottom) and index two is layer 3
    (top). A lower layer is unavailable until all nine cells above it have
    been committed after successful test lifts.
    """

    def __init__(
        self,
        *,
        layers: int = 3,
        rows: int = 3,
        columns: int = 3,
    ) -> None:
        self.layers = int(layers)
        self.rows = int(rows)
        self.columns = int(columns)
        if min(self.layers, self.rows, self.columns) < 1:
            raise ValueError("Task1 stack dimensions must be positive")
        self._picked = np.zeros(
            (self.layers, self.rows, self.columns),
            dtype=np.bool_,
        )
        self._lock = threading.Lock()

    def _active_layer_unlocked(self) -> int | None:
        for layer in range(self.layers, 0, -1):
            if not bool(np.all(self._picked[layer - 1])):
                return layer
        return None

    def _snapshot_unlocked(self) -> dict[str, Any]:
        active_layer = self._active_layer_unlocked()
        picked_count = int(np.count_nonzero(self._picked))
        active_picked = (
            0
            if active_layer is None
            else int(np.count_nonzero(self._picked[active_layer - 1]))
        )
        return {
            "schema": "task1_stack_occupancy_v1",
            "shape": [self.layers, self.rows, self.columns],
            "layer_number_by_array_index": list(range(1, self.layers + 1)),
            "active_layer": active_layer,
            "picked_count": picked_count,
            "remaining_count": int(self._picked.size - picked_count),
            "active_layer_picked_count": active_picked,
            "active_layer_remaining_count": (
                0
                if active_layer is None
                else self.rows * self.columns - active_picked
            ),
            "picked": self._picked.tolist(),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_unlocked()

    def filter_candidates(
        self,
        candidates: Iterable[BoxCandidate],
        *,
        image_shape: tuple[int, int],
        layout_polygon_norm: Iterable[Iterable[float]] | None,
    ) -> tuple[list[BoxCandidate], dict[str, Any]]:
        items = list(candidates)
        if layout_polygon_norm is None:
            with self._lock:
                report = self._snapshot_unlocked()
            report.update(
                {
                    "mapping_configured": False,
                    "input_candidate_count": len(items),
                    "eligible_candidate_count": len(items),
                    "unassigned_candidate_count": len(items),
                    "filtered_picked_candidate_count": 0,
                    "filtered_picked_slots": [],
                }
            )
            return items, report
        annotated, unassigned = assign_task1_stack_slots(
            items,
            image_shape=image_shape,
            layout_polygon_norm=layout_polygon_norm,
            rows=self.rows,
            columns=self.columns,
        )
        with self._lock:
            active_layer = self._active_layer_unlocked()
            kept: list[BoxCandidate] = []
            filtered: list[BoxCandidate] = []
            for candidate in annotated:
                row, column = candidate.grid_index
                blocked = bool(
                    active_layer is None
                    or self._picked[active_layer - 1, row, column]
                )
                if blocked:
                    filtered.append(candidate)
                else:
                    kept.append(candidate)
            report = self._snapshot_unlocked()
        report.update(
            {
                "mapping_configured": True,
                "input_candidate_count": len(annotated) + len(unassigned),
                "eligible_candidate_count": len(kept),
                "unassigned_candidate_count": len(unassigned),
                "filtered_picked_candidate_count": len(filtered),
                "filtered_picked_slots": [
                    {
                        "layer": active_layer,
                        "row_index": int(item.grid_index[0]),
                        "column_index": int(item.grid_index[1]),
                    }
                    for item in filtered
                ],
            }
        )
        return kept, report

    def constrain_layer_estimate(
        self,
        estimate: dict[str, Any],
    ) -> dict[str, Any]:
        constrained = dict(estimate)
        if constrained.get("valid") is not True:
            return constrained
        with self._lock:
            active_layer = self._active_layer_unlocked()
        if active_layer is None:
            constrained["valid"] = False
            constrained["error"] = "all Task1 stack cells have been picked"
            return constrained
        vision_layer = constrained.get("layer")
        constrained.update(
            {
                "layer": active_layer,
                "vision_layer": vision_layer,
                "prior_layer": active_layer,
                "prior_constrained": vision_layer != active_layer,
                "layer_source": "task1_stack_occupancy_prior",
            }
        )
        return constrained

    def validate_detection_ticket(
        self,
        detection: dict[str, Any],
    ) -> tuple[int, int, int]:
        stack_payload = detection.get("task1_stack_prior")
        if not isinstance(stack_payload, dict):
            raise ValueError("Task1 detection has no stack-prior payload")
        selected_slot = stack_payload.get("selected_slot")
        if not isinstance(selected_slot, dict):
            raise ValueError("Task1 detection has no selected 3 x 3 stack slot")
        try:
            layer = int(selected_slot["layer"])
            row = int(selected_slot["row_index"])
            column = int(selected_slot["column_index"])
            estimated_layer = int(detection["layer_estimate"]["layer"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Task1 detection has an invalid stack slot") from exc
        self.validate_pick(layer=layer, row=row, column=column)
        if layer != estimated_layer:
            raise ValueError(
                "Task1 stack layer disagrees with the constrained layer"
            )
        return layer, row, column

    def validate_pick(self, *, layer: int, row: int, column: int) -> None:
        with self._lock:
            self._validate_pick_unlocked(layer=layer, row=row, column=column)

    def _validate_pick_unlocked(
        self,
        *,
        layer: int,
        row: int,
        column: int,
    ) -> None:
        active_layer = self._active_layer_unlocked()
        if active_layer is None:
            raise ValueError("all Task1 stack cells have been picked")
        if layer != active_layer:
            raise ValueError(
                f"stale Task1 layer {layer}; active layer is {active_layer}"
            )
        if not (0 <= row < self.rows and 0 <= column < self.columns):
            raise ValueError("Task1 stack slot is outside the 3 x 3 grid")
        if bool(self._picked[layer - 1, row, column]):
            raise ValueError("Task1 stack slot was already picked")

    def mark_picked(
        self,
        *,
        layer: int,
        row: int,
        column: int,
    ) -> dict[str, Any]:
        with self._lock:
            self._validate_pick_unlocked(
                layer=layer,
                row=row,
                column=column,
            )
            self._picked[layer - 1, row, column] = True
            return self._snapshot_unlocked()


def draw_task1_stack_debug_overlay(
    image_bgr: np.ndarray,
    *,
    stack_report: dict[str, Any] | None,
    layout_polygon_norm: Iterable[Iterable[float]] | None,
) -> np.ndarray:
    """Draw the fixed stack grid and occupancy state on a captured frame."""

    output = image_bgr.copy()
    if not isinstance(stack_report, dict) or layout_polygon_norm is None:
        return output
    height, width = output.shape[:2]
    try:
        shape = tuple(int(value) for value in stack_report["shape"])
        normalized = np.asarray(list(layout_polygon_norm), dtype=np.float32)
        picked = np.asarray(stack_report["picked"], dtype=np.bool_)
    except (KeyError, TypeError, ValueError):
        return output
    if (
        len(shape) != 3
        or normalized.shape != (4, 2)
        or picked.shape != shape
        or height < 2
        or width < 2
        or not np.all(np.isfinite(normalized))
    ):
        return output
    layers, rows, columns = shape
    source = normalized * np.asarray(
        [width - 1, height - 1],
        dtype=np.float32,
    )
    if (
        not cv2.isContourConvex(source)
        or abs(float(cv2.contourArea(source))) < 16.0
    ):
        return output
    destination = np.asarray(
        [[0.0, 0.0], [columns, 0.0], [columns, rows], [0.0, rows]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(destination, source)
    active_value = stack_report.get("active_layer")
    try:
        active_layer = None if active_value is None else int(active_value)
    except (TypeError, ValueError):
        return output
    if active_layer is not None and not 1 <= active_layer <= layers:
        return output
    selected = stack_report.get("selected_slot")
    selected_key: tuple[int, int, int] | None = None
    if isinstance(selected, dict):
        try:
            selected_key = (
                int(selected["layer"]),
                int(selected["row_index"]),
                int(selected["column_index"]),
            )
        except (KeyError, TypeError, ValueError):
            selected_key = None

    cell_polygons: list[
        tuple[np.ndarray, tuple[int, int, int], tuple[int, int, int]]
    ] = []
    tint = output.copy()
    display_layer = active_layer if active_layer is not None else 1
    for row in range(rows):
        for column in range(columns):
            grid_corners = np.asarray(
                [
                    [column, row],
                    [column + 1, row],
                    [column + 1, row + 1],
                    [column, row + 1],
                ],
                dtype=np.float32,
            ).reshape(1, 4, 2)
            polygon = np.round(
                cv2.perspectiveTransform(grid_corners, transform).reshape(4, 2)
            ).astype(np.int32)
            is_picked = bool(picked[display_layer - 1, row, column])
            is_selected = selected_key == (active_layer, row, column)
            color = (
                (60, 220, 60)
                if is_selected
                else ((55, 55, 220) if is_picked else (0, 210, 255))
            )
            cv2.fillConvexPoly(tint, polygon, color)
            cell_polygons.append(
                (polygon, color, (row, column, int(is_picked)))
            )
    cv2.addWeighted(tint, 0.16, output, 0.84, 0.0, dst=output)

    for polygon, color, cell in cell_polygons:
        row, column, is_picked = cell
        is_selected = selected_key == (active_layer, row, column)
        cv2.polylines(output, [polygon], True, color, 3 if is_selected else 1)
        center = tuple(int(value) for value in np.mean(polygon, axis=0))
        label = (
            "TARGET"
            if is_selected
            else ("PICKED" if is_picked else f"R{row + 1}C{column + 1}")
        )
        for text_color, thickness in (((0, 0, 0), 3), (color, 1)):
            cv2.putText(
                output,
                label,
                (center[0] - 24, center[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

    picked_count = int(
        stack_report.get("picked_count", np.count_nonzero(picked))
    )
    remaining_count = int(
        stack_report.get("remaining_count", picked.size - picked_count)
    )
    active_label = "DONE" if active_layer is None else f"L{active_layer}"
    summary = (
        f"TASK1 STACK {active_label} | PICKED {picked_count}/{picked.size} | "
        f"REMAIN {remaining_count}"
    )
    summary_top = max(8, height - 44)
    cv2.rectangle(
        output,
        (8, summary_top),
        (min(width - 8, 540), summary_top + 28),
        (18, 18, 18),
        -1,
    )
    cv2.putText(
        output,
        summary,
        (16, summary_top + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def assign_task1_stack_slots(
    candidates: Iterable[BoxCandidate],
    *,
    image_shape: tuple[int, int],
    layout_polygon_norm: Iterable[Iterable[float]],
    rows: int = 3,
    columns: int = 3,
) -> tuple[list[BoxCandidate], list[BoxCandidate]]:
    """Assign stable row/column identities through a calibrated quadrilateral."""

    items = list(candidates)
    height, width = (int(image_shape[0]), int(image_shape[1]))
    try:
        normalized = np.asarray(list(layout_polygon_norm), dtype=np.float32)
    except (TypeError, ValueError):
        return [], items
    if (
        normalized.shape != (4, 2)
        or height < 2
        or width < 2
        or not np.all(np.isfinite(normalized))
    ):
        return [], items
    source = normalized * np.asarray(
        [width - 1, height - 1],
        dtype=np.float32,
    )
    if (
        not cv2.isContourConvex(source)
        or abs(float(cv2.contourArea(source))) < 16.0
    ):
        return [], items
    destination = np.asarray(
        [[0.0, 0.0], [columns, 0.0], [columns, rows], [0.0, rows]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    parent_center = tuple(float(value) for value in np.mean(source, axis=0))
    annotated: list[BoxCandidate] = []
    unassigned: list[BoxCandidate] = []
    for candidate in items:
        if candidate.grid_shape == (rows, columns):
            row, column = candidate.grid_index
            if 0 <= row < rows and 0 <= column < columns:
                annotated.append(candidate)
                continue
        point = np.asarray(candidate.center_px, dtype=np.float32).reshape(1, 1, 2)
        try:
            grid_point = cv2.perspectiveTransform(point, transform).reshape(2)
        except cv2.error:
            unassigned.append(candidate)
            continue
        grid_x, grid_y = (float(grid_point[0]), float(grid_point[1]))
        tolerance = 0.08
        if not (
            -tolerance <= grid_x <= columns + tolerance
            and -tolerance <= grid_y <= rows + tolerance
        ):
            unassigned.append(candidate)
            continue
        column = min(columns - 1, max(0, int(math.floor(grid_x))))
        row = min(rows - 1, max(0, int(math.floor(grid_y))))
        annotated.append(
            replace(
                candidate,
                grid_shape=(rows, columns),
                grid_index=(row, column),
                grid_parent_center_px=parent_center,
            )
        )
    return annotated, unassigned


class Task1AdaptiveVisualDetector:
    """Task1 identity detector with its own public type and recovery path.

    The existing SIFT/RANSAC matcher is used only as an internal primitive,
    with its RGB-D row recovery forcibly disabled.  Task1 never enters the
    Task2 recovery branch; its missing faces are handled below.
    """

    name = "task1_3x3_adaptive_rgbd"

    def __init__(self, config: dict[str, Any], face_bank: Any | None) -> None:
        isolated = dict(config)
        isolated["adaptive_recovery_enabled"] = False
        self._identity = Task2AdaptiveVisualDetector(isolated, face_bank)
        allowed: list[int] = []
        for value in isolated.get("adaptive_allowed_counts", range(1, 10)):
            if isinstance(value, bool):
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if 1 <= parsed <= 9:
                allowed.append(parsed)
        task1_counts = tuple(sorted(set(allowed or range(1, 10))))
        self._identity._allowed_counts = task1_counts
        self._identity._maximum_instances = max(task1_counts)

    @property
    def ready(self) -> bool:
        return self._identity.ready

    def detect(self, rgb: np.ndarray) -> list[BoxCandidate]:
        # No aligned depth is passed to the internal matcher, so even a bad
        # configuration cannot enter Task2's four-edge recovery path.
        return self._identity.detect(rgb)

    def status(self) -> dict[str, Any]:
        payload = dict(self._identity.status())
        payload.update(
            {
                "name": self.name,
                "mode": "task1_multi_sift_plus_task1_rgbd_3x3_recovery",
                "recovery_enabled": False,
            }
        )
        return payload


def _edge_support(
    edge_image: np.ndarray,
    polygon: np.ndarray,
    *,
    search_radius_px: int,
) -> tuple[float, int, list[float]]:
    """Return sampled support along each transferred carton edge."""

    height, width = edge_image.shape
    per_edge: list[float] = []
    radius = max(1, int(search_radius_px))
    for index in range(4):
        start = polygon[index]
        end = polygon[(index + 1) % 4]
        length = max(8, int(round(float(np.linalg.norm(end - start)))))
        samples = np.linspace(start, end, length, dtype=np.float32)
        supported = 0
        considered = 0
        for x_value, y_value in samples:
            x = int(round(float(x_value)))
            y = int(round(float(y_value)))
            if not (0 <= x < width and 0 <= y < height):
                continue
            considered += 1
            patch = edge_image[
                max(0, y - radius) : min(height, y + radius + 1),
                max(0, x - radius) : min(width, x + radius + 1),
            ]
            if np.any(patch):
                supported += 1
        per_edge.append(float(supported) / float(max(considered, 1)))
    return float(np.mean(per_edge)), sum(value >= 0.35 for value in per_edge), per_edge


def _axial_angle_error(first_deg: float, second_deg: float) -> float:
    return abs((float(first_deg) - float(second_deg) + 90.0) % 180.0 - 90.0)


def _line_intersection(
    first_point: np.ndarray,
    first_direction: np.ndarray,
    second_point: np.ndarray,
    second_direction: np.ndarray,
) -> np.ndarray | None:
    cross = float(
        first_direction[0] * second_direction[1]
        - first_direction[1] * second_direction[0]
    )
    if abs(cross) < 1e-4:
        return None
    delta = second_point - first_point
    distance = float(
        (delta[0] * second_direction[1] - delta[1] * second_direction[0])
        / cross
    )
    return first_point + distance * first_direction


def _refine_polygon_from_edges(
    edge_image: np.ndarray,
    polygon: np.ndarray,
    *,
    search_radius_px: int,
    maximum_angle_error_deg: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Fit each predicted side to a real Hough segment in its local corridor."""

    height, width = edge_image.shape
    fitted_lines: list[tuple[np.ndarray, np.ndarray]] = []
    edge_reports: list[dict[str, Any]] = []
    for index in range(4):
        start = polygon[index].astype(np.float64)
        end = polygon[(index + 1) % 4].astype(np.float64)
        vector = end - start
        expected_length = float(np.linalg.norm(vector))
        if expected_length < 8.0:
            return None, {"valid": False, "reason": "predicted_edge_too_short"}
        direction = vector / expected_length
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        predicted_angle = math.degrees(math.atan2(direction[1], direction[0]))

        corridor = np.zeros(edge_image.shape, dtype=np.uint8)
        cv2.line(
            corridor,
            tuple(np.round(start).astype(int)),
            tuple(np.round(end).astype(int)),
            255,
            max(3, 2 * int(search_radius_px) + 1),
        )
        local_edges = cv2.bitwise_and(edge_image, corridor)
        segments = cv2.HoughLinesP(
            local_edges,
            1.0,
            np.pi / 360.0,
            threshold=max(12, int(round(0.20 * expected_length))),
            minLineLength=max(12, int(round(0.28 * expected_length))),
            maxLineGap=max(6, int(round(0.12 * expected_length))),
        )
        best: tuple[float, np.ndarray, np.ndarray, dict[str, Any]] | None = None
        if segments is not None:
            for raw in np.asarray(segments).reshape(-1, 4):
                point_a = np.asarray(raw[:2], dtype=np.float64)
                point_b = np.asarray(raw[2:], dtype=np.float64)
                segment = point_b - point_a
                length = float(np.linalg.norm(segment))
                if length < 1.0:
                    continue
                segment_direction = segment / length
                angle = math.degrees(
                    math.atan2(segment_direction[1], segment_direction[0])
                )
                angle_error = _axial_angle_error(angle, predicted_angle)
                if angle_error > maximum_angle_error_deg:
                    continue
                midpoint = 0.5 * (point_a + point_b)
                relative = midpoint - start
                perpendicular_distance = abs(float(np.dot(relative, normal)))
                along = float(np.dot(relative, direction))
                if (
                    perpendicular_distance > float(search_radius_px) + 2.0
                    or along < -0.20 * expected_length
                    or along > 1.20 * expected_length
                ):
                    continue
                if float(np.dot(segment_direction, direction)) < 0.0:
                    segment_direction = -segment_direction
                score = (
                    min(1.5, length / expected_length)
                    - 0.018 * perpendicular_distance
                    - 0.018 * angle_error
                )
                details = {
                    "hough": True,
                    "length_px": length,
                    "angle_error_deg": angle_error,
                    "offset_px": float(np.dot(relative, normal)),
                    "score": score,
                }
                if best is None or score > best[0]:
                    best = (score, midpoint, segment_direction, details)
        if best is None:
            fitted_lines.append((start, direction))
            edge_reports.append(
                {
                    "hough": False,
                    "length_px": expected_length,
                    "angle_error_deg": 0.0,
                    "offset_px": 0.0,
                    "score": 0.0,
                }
            )
        else:
            fitted_lines.append((best[1], best[2]))
            edge_reports.append(best[3])

    refined_points: list[np.ndarray] = []
    for index in range(4):
        previous = fitted_lines[(index - 1) % 4]
        current = fitted_lines[index]
        intersection = _line_intersection(
            previous[0], previous[1], current[0], current[1]
        )
        if intersection is None:
            return None, {
                "valid": False,
                "reason": "refined_edges_parallel",
                "edges": edge_reports,
            }
        refined_points.append(intersection)
    refined = np.asarray(refined_points, dtype=np.float32)
    if not np.all(np.isfinite(refined)):
        return None, {
            "valid": False,
            "reason": "refined_polygon_non_finite",
            "edges": edge_reports,
        }
    original_area = abs(float(cv2.contourArea(polygon.astype(np.float32))))
    refined_area = abs(float(cv2.contourArea(refined)))
    center_shift = float(
        np.linalg.norm(np.mean(refined, axis=0) - np.mean(polygon, axis=0))
    )
    area_ratio = refined_area / max(original_area, 1.0)
    valid = bool(
        cv2.isContourConvex(np.round(refined).astype(np.int32))
        and 0.55 <= area_ratio <= 1.55
        and center_shift <= float(search_radius_px) + 8.0
        and np.all(refined[:, 0] >= 0.0)
        and np.all(refined[:, 0] < width)
        and np.all(refined[:, 1] >= 0.0)
        and np.all(refined[:, 1] < height)
        and sum(bool(item["hough"]) for item in edge_reports) >= 2
    )
    return (
        refined if valid else None,
        {
            "valid": valid,
            "center_shift_px": center_shift,
            "area_ratio": area_ratio,
            "fitted_edge_count": sum(
                bool(item["hough"]) for item in edge_reports
            ),
            "edges": edge_reports,
        },
    )


def _depth_support(
    depth_z16: np.ndarray | None,
    depth_scale_m: float | None,
    polygon: np.ndarray,
    *,
    minimum_samples: int,
    maximum_mad_mm: float,
) -> dict[str, Any]:
    if depth_z16 is None or depth_scale_m is None:
        return {"available": False, "valid": False, "samples": 0}
    if depth_z16.dtype != np.uint16 or depth_z16.ndim != 2:
        return {"available": True, "valid": False, "samples": 0}
    scale = float(depth_scale_m)
    if not math.isfinite(scale) or scale <= 0.0:
        return {"available": True, "valid": False, "samples": 0}

    center = np.mean(polygon, axis=0)
    inner = center + 0.62 * (polygon - center)
    mask = np.zeros(depth_z16.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(inner).astype(np.int32), 255)
    values = depth_z16[(mask > 0) & (depth_z16 > 0)].astype(np.float64)
    if values.size < minimum_samples:
        return {
            "available": True,
            "valid": False,
            "samples": int(values.size),
        }
    values_mm = values * scale * 1000.0
    median_mm = float(np.median(values_mm))
    mad_mm = float(np.median(np.abs(values_mm - median_mm)))
    return {
        "available": True,
        "valid": mad_mm <= maximum_mad_mm,
        "samples": int(values.size),
        "median_mm": median_mm,
        "mad_mm": mad_mm,
    }


def _translated_polygon(
    template: BoxCandidate,
    target_center: tuple[float, float],
) -> np.ndarray:
    polygon = np.asarray(template.polygon_px, dtype=np.float32)
    source_center = np.mean(polygon, axis=0)
    target = np.asarray(target_center, dtype=np.float32)
    return polygon + (target - source_center)


def _pink_fraction(
    rgb: np.ndarray,
    polygon: np.ndarray,
    config: dict[str, Any],
) -> float:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    pink = (
        (hue >= int(config.get("pink_hue_min", 130)))
        & (hue <= int(config.get("pink_hue_max", 175)))
        & (saturation >= int(config.get("pink_saturation_min", 8)))
        & (saturation <= int(config.get("pink_saturation_max", 130)))
        & (value >= int(config.get("pink_value_min", 130)))
    )
    center = np.mean(polygon, axis=0)
    central_polygon = center + 0.55 * (polygon - center)
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(central_polygon).astype(np.int32), 255)
    inside = mask > 0
    count = int(np.count_nonzero(inside))
    if count == 0:
        return 0.0
    return float(np.count_nonzero(pink & inside)) / float(count)


def _maximum_polygon_overlap(
    polygon: np.ndarray,
    candidates: Iterable[BoxCandidate],
) -> float:
    """Return intersection divided by the smaller face area."""

    polygon = np.asarray(polygon, dtype=np.float32)
    area = abs(float(cv2.contourArea(polygon)))
    if area <= 1.0:
        return 1.0
    maximum = 0.0
    for candidate in candidates:
        other = np.asarray(candidate.polygon_px, dtype=np.float32)
        other_area = abs(float(cv2.contourArea(other)))
        if other_area <= 1.0:
            continue
        try:
            intersection, _ = cv2.intersectConvexConvex(polygon, other)
        except cv2.error:
            continue
        maximum = max(
            maximum,
            float(intersection) / min(area, other_area),
        )
    return maximum


def _same_row_vertical_alignment(
    proposal: BoxCandidate,
    proposals: Iterable[BoxCandidate],
    verified: Iterable[BoxCandidate],
) -> float | None:
    """Estimate the current row centre from directly measured neighbours."""

    proposal_list = list(proposals)
    if not proposal_list:
        return None
    row_values: list[float] = []
    for candidate in verified:
        nearest = min(
            proposal_list,
            key=lambda item: float(
                np.linalg.norm(
                    np.asarray(candidate.center_px, dtype=np.float64)
                    - np.asarray(item.center_px, dtype=np.float64)
                )
            ),
        )
        distance = float(
            np.linalg.norm(
                np.asarray(candidate.center_px, dtype=np.float64)
                - np.asarray(nearest.center_px, dtype=np.float64)
            )
        )
        maximum_distance = 0.70 * math.hypot(
            float(nearest.long_side_px), float(nearest.short_side_px)
        )
        if (
            nearest.grid_index[0] == proposal.grid_index[0]
            and distance <= maximum_distance
        ):
            row_values.append(float(candidate.center_px[1]))
    if not row_values:
        return None
    return float(np.median(np.asarray(row_values, dtype=np.float64)))


def recover_task1_grid_candidates(
    rgb: np.ndarray,
    depth_z16: np.ndarray | None,
    depth_scale_m: float | None,
    seed_candidates: Iterable[BoxCandidate],
    *,
    roi_norm: Iterable[float],
    config: dict[str, Any],
) -> tuple[list[BoxCandidate], dict[str, Any]]:
    """Merge measured Task1 faces with evidence-backed 3x3 recoveries.

    Direct SIFT quadrilaterals always win.  A missing cell is proposed from
    Task1's colour-supported surface grid, inherits perspective only from the
    nearest verified face, and must independently show boundary and planar
    depth evidence in the current synchronized RGB-D frame.
    """

    seeds = list(seed_candidates)
    report: dict[str, Any] = {
        "enabled": bool(config.get("task1_surface_recovery_enabled", False)),
        "seed_count": len(seeds),
        "proposal_count": 0,
        "recovered_count": 0,
        "rejections": {},
    }
    if not report["enabled"]:
        report["stage"] = "disabled"
        return seeds, report
    minimum_seeds = max(
        2,
        int(config.get("task1_surface_recovery_minimum_verified_seeds", 2)),
    )
    verified = [
        candidate
        for candidate in seeds
        if candidate.face_type == "front_large"
        and bool(candidate.reference_face_id)
        and len(candidate.polygon_px) >= 4
    ]
    if len(verified) < minimum_seeds:
        report["stage"] = "insufficient_verified_seeds"
        return seeds, report
    if len(verified) >= 9:
        report["stage"] = "complete_3x3_direct"
        return seeds, report

    proposals = propose_task1_surface_grid(
        rgb,
        verified,
        roi_norm=roi_norm,
        config=config,
        rows=3,
        columns=3,
    )
    report["proposal_count"] = len(proposals)
    if not proposals:
        report["stage"] = "no_colour_grid"
        return seeds, report

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 45, 125)
    maximum_new = max(
        0,
        min(8, int(config.get("task1_surface_recovery_max_new_candidates", 3))),
    )
    minimum_edge_support = float(
        config.get("task1_surface_recovery_minimum_edge_support", 0.18)
    )
    minimum_supported_edges = max(
        1,
        min(
            4,
            int(config.get("task1_surface_recovery_minimum_supported_edges", 2)),
        ),
    )
    search_radius = max(
        1,
        int(config.get("task1_surface_recovery_edge_search_radius_px", 5)),
    )
    refinement_radius = max(
        6,
        int(config.get("task1_surface_recovery_refinement_radius_px", 24)),
    )
    refinement_angle_error = min(
        30.0,
        max(
            5.0,
            float(
                config.get(
                    "task1_surface_recovery_refinement_angle_error_deg",
                    20.0,
                )
            ),
        ),
    )
    minimum_depth_samples = max(
        8,
        int(config.get("task1_surface_recovery_minimum_depth_samples", 24)),
    )
    maximum_depth_mad = max(
        1.0,
        float(config.get("task1_surface_recovery_maximum_depth_mad_mm", 18.0)),
    )
    maximum_anchor_depth_delta = max(
        25.0,
        float(
            config.get(
                "task1_surface_recovery_maximum_anchor_depth_delta_mm",
                65.0,
            )
        ),
    )
    minimum_center_pink = max(
        0.0,
        float(
            config.get(
                "task1_surface_recovery_minimum_center_pink_fraction",
                0.015,
            )
        ),
    )
    duplicate_distance_ratio = float(
        config.get("task1_surface_recovery_duplicate_distance_ratio", 0.55)
    )
    maximum_direct_overlap = min(
        0.80,
        max(
            0.05,
            float(
                config.get(
                    "task1_surface_recovery_maximum_direct_overlap", 0.20
                )
            ),
        ),
    )
    maximum_row_shift = max(
        0.0,
        float(config.get("task1_surface_recovery_maximum_row_shift_px", 12.0)),
    )

    anchor_depths_mm: list[float] = []
    for seed in verified:
        seed_depth = _depth_support(
            depth_z16,
            depth_scale_m,
            np.asarray(seed.polygon_px, dtype=np.float32),
            minimum_samples=max(8, minimum_depth_samples // 2),
            maximum_mad_mm=max(30.0, maximum_depth_mad),
        )
        if seed_depth.get("valid") is True:
            anchor_depths_mm.append(float(seed_depth["median_mm"]))

    rejection_counts: dict[str, int] = {}
    accepted: list[tuple[float, BoxCandidate, dict[str, Any]]] = []
    for proposal in proposals:
        nearest_seed = min(
            verified,
            key=lambda candidate: float(
                np.linalg.norm(
                    np.asarray(candidate.center_px)
                    - np.asarray(proposal.center_px)
                )
            ),
        )
        duplicate_threshold = duplicate_distance_ratio * max(
            1.0,
            min(float(proposal.long_side_px), float(proposal.short_side_px)),
        )
        if any(
            np.linalg.norm(
                np.asarray(candidate.center_px)
                - np.asarray(proposal.center_px)
            )
            <= duplicate_threshold
            for candidate in seeds
        ):
            rejection_counts["direct_measurement_present"] = (
                rejection_counts.get("direct_measurement_present", 0) + 1
            )
            continue

        initial_polygon = _translated_polygon(nearest_seed, proposal.center_px)
        polygon, refinement = _refine_polygon_from_edges(
            edges,
            initial_polygon,
            search_radius_px=refinement_radius,
            maximum_angle_error_deg=refinement_angle_error,
        )
        if polygon is None:
            rejection_counts["edge_refinement_failed"] = (
                rejection_counts.get("edge_refinement_failed", 0) + 1
            )
            continue
        row_center_y = _same_row_vertical_alignment(
            proposal,
            proposals,
            verified,
        )
        row_shift_y = 0.0
        if row_center_y is not None:
            row_shift_y = float(
                np.clip(
                    row_center_y - float(np.mean(polygon[:, 1])),
                    -maximum_row_shift,
                    maximum_row_shift,
                )
            )
            polygon = polygon + np.asarray([0.0, row_shift_y], dtype=np.float32)
        refinement["same_row_center_y_px"] = row_center_y
        refinement["row_alignment_shift_y_px"] = row_shift_y
        direct_overlap = _maximum_polygon_overlap(polygon, seeds)
        if direct_overlap > maximum_direct_overlap:
            rejection_counts["overlaps_direct_measurement"] = (
                rejection_counts.get("overlaps_direct_measurement", 0) + 1
            )
            continue
        center_pink_fraction = _pink_fraction(rgb, polygon, config)
        if center_pink_fraction < minimum_center_pink:
            rejection_counts["missing_center_colour"] = (
                rejection_counts.get("missing_center_colour", 0) + 1
            )
            continue
        edge_mean, supported_edges, per_edge = _edge_support(
            edges,
            polygon,
            search_radius_px=search_radius,
        )
        if edge_mean < minimum_edge_support or supported_edges < minimum_supported_edges:
            rejection_counts["weak_edges"] = rejection_counts.get("weak_edges", 0) + 1
            continue
        depth = _depth_support(
            depth_z16,
            depth_scale_m,
            polygon,
            minimum_samples=minimum_depth_samples,
            maximum_mad_mm=maximum_depth_mad,
        )
        if depth.get("valid") is not True:
            rejection_counts["depth_surface_invalid"] = (
                rejection_counts.get("depth_surface_invalid", 0) + 1
            )
            continue
        if not anchor_depths_mm or min(
            abs(float(depth["median_mm"]) - anchor_depth)
            for anchor_depth in anchor_depths_mm
        ) > maximum_anchor_depth_delta:
            rejection_counts["depth_incompatible_with_verified_faces"] = (
                rejection_counts.get(
                    "depth_incompatible_with_verified_faces", 0
                )
                + 1
            )
            continue

        polygon_center = np.mean(polygon, axis=0)
        (_, _), (side_a, side_b), rect_angle = cv2.minAreaRect(
            polygon.astype(np.float32)
        )
        long_side, short_side = sorted(
            (float(side_a), float(side_b)), reverse=True
        )
        long_angle = float(
            rect_angle if side_a >= side_b else rect_angle + 90.0
        )
        while long_angle >= 90.0:
            long_angle -= 180.0
        while long_angle < -90.0:
            long_angle += 180.0
        polygon_area = abs(float(cv2.contourArea(polygon.astype(np.float32))))
        rectangularity = polygon_area / max(long_side * short_side, 1.0)
        recovered = replace(
            nearest_seed,
            center_px=(float(polygon_center[0]), float(polygon_center[1])),
            suction_px=(
                int(round(float(polygon_center[0]))),
                int(round(float(polygon_center[1]))),
            ),
            polygon_px=tuple(
                (float(point[0]), float(point[1])) for point in polygon
            ),
            long_side_px=long_side,
            short_side_px=short_side,
            angle_deg=long_angle,
            rectangularity=rectangularity,
            bright_fill=float(proposal.bright_fill),
            edge_clearance_px=short_side / 2.0,
            score=min(0.84, float(nearest_seed.score)),
            face_score=min(0.84, float(nearest_seed.face_score)),
            provider="task1_rgbd_3x3_recovery",
            grid_shape=(3, 3),
            grid_index=proposal.grid_index,
            grid_parent_center_px=proposal.grid_parent_center_px,
        )
        evidence_score = (
            0.55 * min(1.0, edge_mean)
            + 0.30 * min(1.0, float(proposal.bright_fill) / 0.08)
            + 0.15 * min(1.0, float(depth["samples"]) / 200.0)
        )
        accepted.append(
            (
                evidence_score,
                recovered,
                {
                    "grid_index": list(proposal.grid_index),
                    "edge_refinement": refinement,
                    "maximum_direct_overlap": direct_overlap,
                    "edge_support": edge_mean,
                    "supported_edges": supported_edges,
                    "per_edge_support": per_edge,
                    "pink_fraction": float(proposal.bright_fill),
                    "center_pink_fraction": center_pink_fraction,
                    "depth": depth,
                },
            )
        )

    accepted.sort(key=lambda item: item[0], reverse=True)
    selected = accepted[:maximum_new]
    report["stage"] = "merged_current_frame_evidence"
    report["recovered_count"] = len(selected)
    report["rejections"] = rejection_counts
    report["recovered"] = [item[2] for item in selected]
    return [*seeds, *(item[1] for item in selected)], report
