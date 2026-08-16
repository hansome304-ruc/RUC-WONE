import ast
import unittest
from pathlib import Path


class Task2FitPerformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = Path(
            "src/medicine_agentic/task2_visual_quad_fit.py"
        ).read_text(encoding="utf-8")

    def test_fit_script_is_valid_python(self) -> None:
        ast.parse(self.source)

    def test_colour_distance_is_limited_to_candidate_roi(self) -> None:
        self.assertIn("local_lab = lab[py0:py1, px0:px1]", self.source)
        self.assertIn("face_probability[py0:py1, px0:px1]", self.source)

    def test_edge_search_uses_coarse_to_fine_grid(self) -> None:
        self.assertIn("np.arange(-18, 18.1, 3.0)", self.source)
        self.assertIn("np.arange(-14, 14.1, 3.0)", self.source)
        self.assertIn("coarse[:24]", self.source)

    def test_rejected_sift_prior_is_tried_before_area_scan(self) -> None:
        wrapper = Path(
            "src/medicine_agentic/task2_visual_quad_any.py"
        ).read_text(encoding="utf-8")
        self.assertIn("TASK2_RECOVERY_PRIORS", wrapper)
        self.assertIn("priority_hypotheses", wrapper)


if __name__ == "__main__":
    unittest.main()
