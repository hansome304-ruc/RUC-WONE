import threading
import unittest
from unittest import mock

import cv2
import numpy as np

from medicine_agentic.task2_visual_detector import (
    Task2AdaptiveVisualDetector,
    _cached_missing_quad,
)


def quad(cx: float) -> np.ndarray:
    return np.asarray(
        [[cx - 38, 410], [cx + 38, 410], [cx + 38, 530], [cx - 38, 530]],
        dtype=np.float32,
    )


class CachedMissingQuadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cached = [quad(value) for value in (560, 650, 740, 830)]
        self.rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.depth = np.full((720, 1280), 1100, dtype=np.uint16)
        for polygon in self.cached:
            cv2.fillConvexPoly(self.rgb, polygon.astype(np.int32), (235, 190, 235))
            cv2.fillConvexPoly(self.depth, polygon.astype(np.int32), 950)

    def test_recovers_missing_middle_carton_from_current_evidence(self) -> None:
        current = [self.cached[index] + (2, -1) for index in (0, 2, 3)]
        recovered = _cached_missing_quad(self.rgb, self.depth, current, self.cached)
        self.assertIsNotNone(recovered)
        np.testing.assert_allclose(recovered.mean(axis=0), [652, 469], atol=1.0)

    def test_rejects_removed_carton_when_current_pixels_are_table(self) -> None:
        current = [self.cached[index] for index in (0, 2, 3)]
        missing = np.zeros(self.rgb.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(missing, self.cached[1].astype(np.int32), 255)
        self.rgb[missing > 0] = (30, 100, 30)
        self.depth[missing > 0] = 1100
        self.assertIsNone(
            _cached_missing_quad(self.rgb, self.depth, current, self.cached)
        )

    def test_rejects_rearranged_row(self) -> None:
        current = [self.cached[0], self.cached[2] + (30, 0), self.cached[3]]
        self.assertIsNone(
            _cached_missing_quad(self.rgb, self.depth, current, self.cached)
        )

    def test_detector_passes_rejected_sift_center_to_recovery(self) -> None:
        source = __import__(
            "inspect"
        ).getsource(__import__(
            "medicine_agentic.task2_visual_detector",
            fromlist=["Task2AdaptiveVisualDetector"],
        ).Task2AdaptiveVisualDetector.detect_rgbd)
        self.assertIn("rejected_priors", source)

    def test_flat_profile_accepts_one_or_two_direct_sift_faces(self) -> None:
        detector = Task2AdaptiveVisualDetector.__new__(
            Task2AdaptiveVisualDetector
        )
        detector._lock = threading.Lock()
        detector._face = object()
        detector._reference_descriptors = np.ones((8, 128), np.float32)
        detector._allowed_counts = (1, 2)
        detector._minimum_recovery_anchors = 2
        detector._recovery_enabled = False
        detector._minimum_quad_fill = 0.70
        detector._cached_four_quads = []
        detector._cached_four_at = 0.0
        detector._last_error = None
        detector._last_stage = "not_run"
        detector._last_count = 0
        detector._last_geometry = []
        detector._last_glare_normalized = []
        detector._candidate = lambda rgb, polygon, **kwargs: polygon
        rgb = np.zeros((720, 1280, 3), dtype=np.uint8)

        for count in (1, 2):
            detector._sift_instances = lambda image, count=count: [
                {
                    "polygon_px": quad(120.0 + 100.0 * index).tolist(),
                    "center_px": [120.0 + 100.0 * index, 470.0],
                    "inliers": 12,
                }
                for index in range(count)
            ]
            candidates = detector.detect_rgbd(rgb, None, None)
            self.assertEqual(len(candidates), count)
            self.assertEqual(detector._last_stage, "sift_direct")

    def test_default_task2_counts_remain_three_or_four(self) -> None:
        detector = Task2AdaptiveVisualDetector(
            {"adaptive_profile_name": "task2"},
            None,
        )
        self.assertEqual(detector._allowed_counts, (3, 4))
        self.assertEqual(detector._sift_ratio, 0.80)
        self.assertEqual(detector._homography_attempt_multiplier, 2)
        self.assertTrue(detector._recovery_enabled)
        self.assertTrue(detector._reject_glare_matches)
        self.assertEqual(detector._minimum_quad_fill, 0.84)

    def test_task1_can_keep_six_measured_quads_despite_glare_signal(self) -> None:
        detector = Task2AdaptiveVisualDetector.__new__(
            Task2AdaptiveVisualDetector
        )
        detector._lock = threading.Lock()
        detector._face = object()
        detector._reference_descriptors = np.ones((8, 128), np.float32)
        detector._allowed_counts = (1, 2, 3, 4, 5, 6)
        detector._minimum_recovery_anchors = 2
        detector._recovery_enabled = False
        detector._reject_glare_matches = False
        detector._minimum_quad_fill = 0.70
        detector._cached_four_quads = []
        detector._cached_four_at = 0.0
        detector._last_error = None
        detector._last_stage = "not_run"
        detector._last_count = 0
        detector._last_geometry = []
        detector._last_glare_normalized = []
        detector._candidate = lambda rgb, polygon, **kwargs: polygon
        rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
        instances = [
            {
                "polygon_px": quad(120.0 + 90.0 * index).tolist(),
                "center_px": [120.0 + 90.0 * index, 470.0],
                "inliers": 12,
            }
            for index in range(6)
        ]
        detector._sift_instances = lambda image: instances
        normalized = [dict(instance) for instance in instances]
        glare_flags = [False, True, False, False, True, True]

        with mock.patch(
            "medicine_agentic.task2_visual_detector._normalize_glare_homographies",
            return_value=(normalized, glare_flags),
        ):
            candidates = detector.detect_rgbd(rgb, None, None)

        self.assertEqual(len(candidates), 6)
        self.assertEqual(detector._last_stage, "sift_direct")
        self.assertEqual(detector._last_glare_normalized, glare_flags)


if __name__ == "__main__":
    unittest.main()
