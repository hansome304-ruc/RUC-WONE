from __future__ import annotations

from dataclasses import replace
import unittest

import cv2
import numpy as np

from medicine_agentic.task1_box import BoxCandidate
from medicine_agentic.task1_visual_recovery import (
    Task1AdaptiveVisualDetector,
    Task1StackOccupancyPrior,
    assign_task1_stack_slots,
    draw_task1_stack_debug_overlay,
    recover_task1_grid_candidates,
)
from medicine_agentic.task2_visual_detector import Task2AdaptiveVisualDetector


def candidate(row: int, column: int) -> BoxCandidate:
    center_x = 64.0 + 70.0 * column
    center_y = 106.0 + 78.0 * row
    half_x, half_y = 29.0, 31.0
    return BoxCandidate(
        center_px=(center_x, center_y),
        suction_px=(int(center_x), int(center_y)),
        polygon_px=(
            (center_x - half_x, center_y - half_y),
            (center_x + half_x, center_y - half_y),
            (center_x + half_x, center_y + half_y),
            (center_x - half_x, center_y + half_y),
        ),
        long_side_px=62.0,
        short_side_px=58.0,
        angle_deg=90.0,
        rectangularity=0.95,
        bright_fill=0.30,
        edge_clearance_px=20.0,
        score=0.95,
        provider="task1_3x3_adaptive_rgbd:multi_sift",
        face_type="front_large",
        face_score=0.95,
        reference_face_id="front_large_01",
        graspable=True,
        grasp_blockers=(),
    )


class Task1VisualRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rgb = np.zeros((360, 480, 3), dtype=np.uint8)
        self.depth = np.full((360, 480), 1200, dtype=np.uint16)
        for row in range(3):
            for column in range(3):
                item = candidate(row, column)
                polygon = np.asarray(item.polygon_px, dtype=np.int32)
                cv2.fillConvexPoly(self.rgb, polygon, (255, 205, 225))
                cv2.polylines(self.rgb, [polygon], True, (35, 35, 35), 2)
                cv2.fillConvexPoly(self.depth, polygon, 900)
        self.config = {
            "task1_surface_recovery_enabled": True,
            "task1_surface_recovery_minimum_verified_seeds": 2,
            "task1_surface_recovery_max_new_candidates": 3,
            "task1_surface_recovery_minimum_edge_support": 0.18,
            "task1_surface_recovery_minimum_supported_edges": 2,
            "task1_surface_recovery_edge_search_radius_px": 5,
            "task1_surface_recovery_refinement_radius_px": 24,
            "task1_surface_recovery_refinement_angle_error_deg": 20.0,
            "task1_surface_recovery_maximum_row_shift_px": 12.0,
            "task1_surface_recovery_minimum_center_pink_fraction": 0.015,
            "task1_surface_recovery_minimum_depth_samples": 24,
            "task1_surface_recovery_maximum_depth_mad_mm": 18.0,
            "task1_surface_recovery_maximum_anchor_depth_delta_mm": 65.0,
            "task1_surface_recovery_duplicate_distance_ratio": 0.55,
            "task1_surface_recovery_maximum_direct_overlap": 0.2,
            "pink_hue_min": 130,
            "pink_hue_max": 175,
            "pink_saturation_min": 8,
            "pink_saturation_max": 130,
            "pink_value_min": 130,
            "task1_grid_minimum_cell_side_px": 30,
            "task1_grid_minimum_cell_pink_fraction": 0.018,
        }

    def test_recovers_one_missing_face_without_replacing_direct_quads(self) -> None:
        seeds = [
            candidate(row, column)
            for row in range(3)
            for column in range(3)
            if (row, column) != (0, 0)
        ]
        merged, report = recover_task1_grid_candidates(
            self.rgb,
            self.depth,
            0.001,
            seeds,
            roi_norm=[0.0, 0.1, 0.7, 1.0],
            config=self.config,
        )

        self.assertEqual(merged[: len(seeds)], seeds)
        recovered = [
            item
            for item in merged
            if item.provider == "task1_rgbd_3x3_recovery"
        ]
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].grid_index, (0, 0))
        np.testing.assert_allclose(
            recovered[0].center_px,
            candidate(0, 0).center_px,
            atol=3.0,
        )
        self.assertEqual(report["recovered_count"], 1)
        self.assertTrue(
            report["recovered"][0]["edge_refinement"]["valid"]
        )

    def test_does_not_invent_a_removed_carton(self) -> None:
        missing = np.asarray(candidate(0, 0).polygon_px, dtype=np.int32)
        cv2.fillConvexPoly(self.rgb, missing, (20, 90, 30))
        # The empty support plane is deliberately close to a first-layer
        # carton; colour/edge evidence, not a large depth jump, must reject it.
        cv2.fillConvexPoly(self.depth, missing, 925)
        seeds = [
            candidate(row, column)
            for row in range(3)
            for column in range(3)
            if (row, column) != (0, 0)
        ]
        merged, report = recover_task1_grid_candidates(
            self.rgb,
            self.depth,
            0.001,
            seeds,
            roi_norm=[0.0, 0.1, 0.7, 1.0],
            config=self.config,
        )

        self.assertEqual(len(merged), len(seeds))
        self.assertEqual(report["recovered_count"], 0)

    def test_task1_and_task2_recovery_switches_are_isolated(self) -> None:
        task1 = Task1AdaptiveVisualDetector(
            {
                "adaptive_allowed_counts": list(range(1, 10)),
                "adaptive_recovery_enabled": True,
            },
            None,
        )
        task2 = Task2AdaptiveVisualDetector({}, None)

        self.assertEqual(
            task1.status()["allowed_counts"], list(range(1, 10))
        )
        self.assertFalse(task1.status()["recovery_enabled"])
        self.assertEqual(task2._allowed_counts, (3, 4))
        self.assertTrue(task2._recovery_enabled)

    def test_assigns_direct_candidates_to_stable_perspective_slots(self) -> None:
        items = [
            replace(
                candidate(row, column),
                center_px=(50.0 + 100.0 * column, 50.0 + 100.0 * row),
            )
            for row in range(3)
            for column in range(3)
        ]

        assigned, unassigned = assign_task1_stack_slots(
            items,
            image_shape=(300, 300),
            layout_polygon_norm=[
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
        )

        self.assertEqual(unassigned, [])
        self.assertEqual(
            [item.grid_index for item in assigned],
            [(row, column) for row in range(3) for column in range(3)],
        )
        self.assertTrue(all(item.grid_shape == (3, 3) for item in assigned))

    def test_stack_prior_filters_picked_cells_and_advances_one_layer(self) -> None:
        prior = Task1StackOccupancyPrior()
        cells = [
            replace(
                candidate(row, column),
                grid_shape=(3, 3),
                grid_index=(row, column),
            )
            for row in range(3)
            for column in range(3)
        ]
        layout = [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]

        prior.mark_picked(layer=3, row=0, column=0)
        eligible, report = prior.filter_candidates(
            cells[:2],
            image_shape=(360, 480),
            layout_polygon_norm=layout,
        )

        self.assertEqual([item.grid_index for item in eligible], [(0, 1)])
        self.assertEqual(report["filtered_picked_candidate_count"], 1)
        self.assertTrue(report["picked"][2][0][0])

        for row in range(3):
            for column in range(3):
                if (row, column) != (0, 0):
                    prior.mark_picked(layer=3, row=row, column=column)

        transitioned = prior.snapshot()
        self.assertEqual(transitioned["active_layer"], 2)
        self.assertEqual(transitioned["active_layer_remaining_count"], 9)

        eligible, _ = prior.filter_candidates(
            cells[:1],
            image_shape=(360, 480),
            layout_polygon_norm=layout,
        )
        self.assertEqual([item.grid_index for item in eligible], [(0, 0)])

    def test_stack_prior_overrides_reflection_layer_after_transition(self) -> None:
        prior = Task1StackOccupancyPrior()
        for row in range(3):
            for column in range(3):
                prior.mark_picked(layer=3, row=row, column=column)

        constrained = prior.constrain_layer_estimate(
            {
                "valid": True,
                "layer": 3,
                "height_above_table_m": 0.081,
            }
        )

        self.assertEqual(constrained["vision_layer"], 3)
        self.assertEqual(constrained["layer"], 2)
        self.assertEqual(constrained["prior_layer"], 2)
        self.assertTrue(constrained["prior_constrained"])
        ticket = prior.validate_detection_ticket(
            {
                "layer_estimate": constrained,
                "task1_stack_prior": {
                    "selected_slot": {
                        "layer": 2,
                        "row_index": 0,
                        "column_index": 0,
                    }
                },
            }
        )
        self.assertEqual(ticket, (2, 0, 0))
        with self.assertRaisesRegex(ValueError, "active layer is 2"):
            prior.validate_pick(layer=3, row=0, column=0)

    def test_stack_debug_overlay_marks_picked_and_selected_cells(self) -> None:
        prior = Task1StackOccupancyPrior()
        prior.mark_picked(layer=3, row=0, column=0)
        report = prior.snapshot()
        report["selected_slot"] = {
            "layer": 3,
            "row_index": 1,
            "column_index": 1,
        }
        original = np.zeros((300, 300, 3), dtype=np.uint8)

        overlay = draw_task1_stack_debug_overlay(
            original,
            stack_report=report,
            layout_polygon_norm=[
                [0.1, 0.1],
                [0.9, 0.1],
                [0.9, 0.9],
                [0.1, 0.9],
            ],
        )

        self.assertFalse(np.any(original))
        self.assertTrue(np.any(overlay))
        self.assertGreater(int(overlay[50, 50, 2]), int(overlay[50, 50, 1]))
        self.assertGreater(
            int(overlay[130, 130, 1]),
            int(overlay[130, 130, 2]),
        )



if __name__ == "__main__":
    unittest.main()
