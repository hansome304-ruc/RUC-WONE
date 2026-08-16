from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from medicine_agentic.yoloe_visual_prompt import (
    FacePrompt,
    MotifTemplate,
    YOLOEVisualPromptDetector,
    _candidate_from_polygon,
    _deduplicate,
    _surface_evidence,
)


PROMPT = FacePrompt(
    class_id=0,
    face_type="front_large",
    reference_face_id="front_large_01",
    bbox_xyxy=(10.0, 10.0, 90.0, 60.0),
)


def polygon(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.asarray(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )


class YOLOEVisualPromptGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.options = {
            "min_box_area_fraction": 0.001,
            "max_box_area_fraction": 0.10,
            "min_box_aspect_ratio": 1.15,
            "max_box_aspect_ratio": 2.0,
            "min_mask_rectangularity": 0.78,
        }

    def candidate(self, points: np.ndarray, confidence: float = 0.4):
        return _candidate_from_polygon(
            points,
            confidence=confidence,
            prompt=PROMPT,
            frame_shape=(480, 848),
            roi_px=(290, 190, 575, 335),
            options=self.options,
        )

    def test_accepts_carton_geometry_and_finds_interior_suction_point(self) -> None:
        candidate = self.candidate(polygon(320, 260, 395, 310))
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.face_type, "front_large")
        self.assertEqual(candidate.reference_face_id, "front_large_01")
        self.assertGreater(candidate.edge_clearance_px, 20.0)
        self.assertTrue(320 < candidate.suction_px[0] < 395)
        self.assertTrue(260 < candidate.suction_px[1] < 310)

    def test_rejects_merged_row_and_outside_roi(self) -> None:
        self.assertIsNone(self.candidate(polygon(320, 260, 505, 312)))
        self.assertIsNone(self.candidate(polygon(50, 260, 125, 310)))

    def test_rejects_geometry_without_enough_pink_surface(self) -> None:
        points = polygon(320, 260, 395, 310)
        no_pink = np.zeros((480, 848), dtype=np.uint8)
        candidate = _candidate_from_polygon(
            points,
            confidence=0.4,
            prompt=PROMPT,
            frame_shape=(480, 848),
            roi_px=(0, 0, 848, 480),
            options={
                **self.options,
                "verify_pink_color": True,
                "min_pink_fraction": 0.5,
            },
            pink_mask=no_pink,
        )
        self.assertIsNone(candidate)

    def test_area_gate_is_resolution_normalized(self) -> None:
        candidate = _candidate_from_polygon(
            polygon(488, 400, 614, 482),
            confidence=0.4,
            prompt=PROMPT,
            frame_shape=(720, 1280),
            roi_px=(0, 0, 1280, 720),
            options=self.options,
        )
        self.assertIsNotNone(candidate)

    def test_deduplicate_keeps_the_highest_confidence_instance(self) -> None:
        strong = self.candidate(polygon(320, 260, 395, 310), 0.6)
        weak = self.candidate(polygon(322, 261, 396, 311), 0.2)
        other = self.candidate(polygon(410, 260, 485, 310), 0.4)
        assert strong is not None and weak is not None and other is not None
        kept = _deduplicate([weak, other, strong], iou_threshold=0.45)
        self.assertEqual([item.score for item in kept], [0.6, 0.4])

    def test_surface_evidence_never_selects_a_cropped_border(self) -> None:
        pink = np.full((80, 100), 255, dtype=np.uint8)
        evidence = _surface_evidence(
            polygon(0, 10, 50, 70),
            pink_mask=pink,
        )
        self.assertIsNotNone(evidence)
        assert evidence is not None
        _, _, suction, clearance = evidence
        self.assertGreater(suction[0], 0)
        self.assertGreater(suction[1], 10)
        self.assertLess(suction[0], 50)
        self.assertLess(suction[1], 70)
        self.assertGreater(clearance, 1.0)

    def test_motif_fallback_runs_when_yolo_returns_no_results(self) -> None:
        candidate = self.candidate(polygon(320, 260, 395, 310), 0.6)
        assert candidate is not None

        class EmptyModel:
            @staticmethod
            def predict(*args, **kwargs):
                return []

        detector = YOLOEVisualPromptDetector.__new__(
            YOLOEVisualPromptDetector
        )
        detector._options = {}
        detector._model = EmptyModel()
        detector._embedding_ready = True
        detector._inference_count = 0
        detector._last_latency_ms = None
        with patch.object(
            YOLOEVisualPromptDetector,
            "_detect_motifs",
            return_value=[candidate],
        ) as motif:
            result = detector.detect(
                np.full((480, 848, 3), 200, dtype=np.uint8)
            )

        motif.assert_called_once()
        self.assertEqual(result, [candidate])
        self.assertEqual(detector._inference_count, 1)
        self.assertIsNotNone(detector._last_latency_ms)

    def test_motif_matching_is_limited_to_configured_roi(self) -> None:
        motif = np.zeros((16, 16), dtype=np.uint8)
        cv2.line(motif, (2, 2), (13, 13), 255, 2)
        cv2.circle(motif, (11, 4), 2, 255, -1)
        rgb = np.full((60, 100, 3), [255, 200, 220], dtype=np.uint8)
        for x in (8, 68):
            patch_rgb = cv2.cvtColor(motif, cv2.COLOR_GRAY2RGB)
            rgb[20:36, x : x + 16] = patch_rgb

        detector = YOLOEVisualPromptDetector.__new__(
            YOLOEVisualPromptDetector
        )
        detector._options = {
            "roi_norm": [0.50, 0.0, 1.0, 1.0],
            "motif_max_peaks_per_variant": 2,
            "motif_max_raw_candidates": 4,
            "motif_min_visible_fraction": 0.5,
            "motif_min_pink_fraction": 0.0,
        }
        detector._motifs = (
            MotifTemplate(
                image_path=Path("front.png"),
                gray=motif,
                face_type="front_large",
                reference_face_id="front_large_01",
                box_center_offset_px=(0.0, 0.0),
                box_size_px=(16.0, 16.0),
                box_angle_deg=0.0,
                min_score=0.45,
                angle_min_deg=0.0,
                angle_max_deg=0.0,
                angle_step_deg=5.0,
                scale_min=1.0,
                scale_max=1.0,
                scale_steps=1,
            ),
        )
        pink_mask = np.full(rgb.shape[:2], 255, dtype=np.uint8)

        matches = detector._detect_motifs(rgb, pink_mask=pink_mask)

        self.assertTrue(matches)
        self.assertTrue(all(match.center_px[0] >= 50.0 for match in matches))


if __name__ == "__main__":
    unittest.main()
