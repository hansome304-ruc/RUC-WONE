import json
import subprocess
import unittest
from pathlib import Path

import cv2
import numpy as np

from medicine_agentic.packaging_console import carton_interior_depth_fallback
from medicine_agentic.task1_box import BoxCandidate
from unittest import mock


class Task2RuntimeRegressionTests(unittest.TestCase):
    @staticmethod
    def candidate() -> BoxCandidate:
        return BoxCandidate(
            center_px=(60.0, 70.0),
            suction_px=(60, 70),
            polygon_px=((20.0, 10.0), (100.0, 10.0), (100.0, 130.0), (20.0, 130.0)),
            long_side_px=120.0,
            short_side_px=80.0,
            angle_deg=90.0,
            rectangularity=1.0,
            bright_fill=0.8,
            edge_clearance_px=40.0,
            score=0.9,
        )

    def test_interior_depth_recovers_glare_hole(self) -> None:
        depth = np.full((160, 140), 1100, dtype=np.uint16)
        cv2.fillConvexPoly(
            depth,
            np.asarray(self.candidate().polygon_px, dtype=np.int32),
            950,
        )
        cv2.circle(depth, (60, 70), 22, 0, -1)
        result = carton_interior_depth_fallback(
            self.candidate(), depth, 0.001
        )
        self.assertTrue(result["valid"])
        self.assertGreater(result["valid_ratio"], 0.40)
        self.assertAlmostEqual(result["median_depth_m"], 0.95, places=3)

    def test_interior_depth_rejects_sparse_support(self) -> None:
        depth = np.zeros((160, 140), dtype=np.uint16)
        depth[60:65, 55:60] = 950
        result = carton_interior_depth_fallback(
            self.candidate(), depth, 0.001
        )
        self.assertFalse(result["valid"])
    def test_task2_surface_gate_keeps_observed_middle_carton(self) -> None:
        config = json.loads(
            Path("configs/packaging_console.json").read_text(encoding="utf-8")
        )
        profile = config["task_profiles"]["task2"]
        lower, upper = profile["surface_z_range_left_base_m"]
        tolerance = profile["surface_z_gate_tolerance_m"]
        observed_middle_carton_z_m = 0.04913072843986099
        self.assertLessEqual(observed_middle_carton_z_m, upper + tolerance)
        self.assertGreaterEqual(observed_middle_carton_z_m, lower - tolerance)

    def test_visual_fits_use_bounded_parallel_workers(self) -> None:
        source = Path("src/medicine_agentic/task2_visual_quad_any.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("ThreadPoolExecutor", source)
        self.assertIn("max_workers=min(3, max(1, len(remaining)))", source)

    def test_task2_reports_recognized_boxes_before_depth_pick_filter(self) -> None:
        source = Path("src/medicine_agentic/packaging_console.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('if task_id in {"task1", "task2"}', source)
        self.assertIn('"instance_count": len(reported_candidates)', source)

    def test_task2_complete_four_count_stops_after_first_frame(self) -> None:
        source = Path("src/medicine_agentic/packaging_console.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('if int(result.get("instance_count", 0)) >= 4:', source)
        self.assertIn('"policy": "complete_maximum_count_early_stop"', source)


if __name__ == "__main__":
    unittest.main()
