from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from medicine_agentic.detector_provider import (
    ReferenceFeatureDetectorProvider,
    apply_grasp_policy,
    create_detector_provider,
)
from medicine_agentic.task1_box import BoxCandidate


class FakeBank:
    bank_id = "test_bank"
    content_sha256 = "a" * 64

    def __init__(self, *, pick_allowed: bool = True) -> None:
        self.reference = SimpleNamespace(
            id="front_large_01",
            face_type="front_large",
            pick_allowed=pick_allowed,
        )

    def face_by_id(self, reference_id: str):
        if reference_id != self.reference.id:
            raise KeyError(reference_id)
        return self.reference


def candidate(**changes) -> BoxCandidate:
    values = {
        "center_px": (100.0, 80.0),
        "suction_px": (100, 80),
        "polygon_px": (
            (60.0, 60.0),
            (140.0, 60.0),
            (140.0, 100.0),
            (60.0, 100.0),
        ),
        "long_side_px": 80.0,
        "short_side_px": 40.0,
        "angle_deg": 0.0,
        "rectangularity": 0.95,
        "bright_fill": 0.9,
        "edge_clearance_px": 20.0,
        "score": 0.92,
        "provider": "fake_visual_prompt",
        "face_type": "front_large",
        "face_score": 0.94,
        "reference_face_id": "front_large_01",
        # Deliberately lie. The central policy must recompute this field.
        "graspable": True,
        "grasp_blockers": (),
    }
    values.update(changes)
    return BoxCandidate(**values)


class DetectorProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "min_detection_score": 0.68,
            "min_face_score": 0.75,
            "min_edge_clearance_px": 12,
            "cup_radius_px": 8,
            "cup_clearance_margin_px": 4,
        }

    def test_policy_allows_verified_pick_face_with_clearance(self) -> None:
        checked = apply_grasp_policy(
            candidate(),
            face_bank=FakeBank(),
            config=self.config,
        )
        self.assertTrue(checked.graspable)
        self.assertEqual(checked.grasp_blockers, ())

    def test_missing_bank_and_unknown_face_fail_closed(self) -> None:
        checked = apply_grasp_policy(
            candidate(
                face_type="unknown",
                reference_face_id=None,
                graspable=True,
            ),
            face_bank=None,
            config=self.config,
        )
        self.assertFalse(checked.graspable)
        self.assertIn("reference_bank_unavailable", checked.grasp_blockers)
        self.assertIn("face_score_low", apply_grasp_policy(
            candidate(face_score=0.1),
            face_bank=FakeBank(),
            config=self.config,
        ).grasp_blockers)

    def test_reference_permission_and_clearance_are_central(self) -> None:
        forbidden = apply_grasp_policy(
            candidate(),
            face_bank=FakeBank(pick_allowed=False),
            config=self.config,
        )
        self.assertFalse(forbidden.graspable)
        self.assertIn(
            "reference_face_not_pick_allowed",
            forbidden.grasp_blockers,
        )

        too_close = apply_grasp_policy(
            candidate(edge_clearance_px=11.9),
            face_bank=FakeBank(),
            config=self.config,
        )
        self.assertFalse(too_close.graspable)
        self.assertIn("suction_clearance_low", too_close.grasp_blockers)

    def test_missing_optional_plugin_is_reported_without_importing_at_startup(self) -> None:
        provider = create_detector_provider(
            {
                **self.config,
                "provider": "yoloe_visual_prompt",
                "plugin_factory": "module_that_does_not_exist:create",
            },
            config_dir=Path("."),
            face_bank=None,
        )
        self.assertEqual(provider.detect(np.zeros((8, 8, 3), dtype=np.uint8)), [])
        status = provider.status()
        self.assertFalse(status["ok"])
        self.assertIn("ModuleNotFoundError", status["backend"]["error"])

    def test_reference_feature_pink_gate_rejects_black_robot_material(self) -> None:
        provider = object.__new__(ReferenceFeatureDetectorProvider)
        provider._config = {
            "pink_hue_min": 135,
            "pink_hue_max": 179,
            "pink_saturation_min": 8,
            "pink_saturation_max": 150,
            "pink_value_min": 115,
        }
        carton_rgb = np.full((120, 180, 3), [255, 200, 220], dtype=np.uint8)
        robot_rgb = np.zeros_like(carton_rgb)
        self.assertGreater(provider._pink_fraction(carton_rgb, candidate()), 0.9)
        self.assertEqual(provider._pink_fraction(robot_rgb, candidate()), 0.0)

    def test_motif_fallback_calibrates_full_face_center_for_policy(self) -> None:
        class FakeMotifHelper:
            _options = {
                "pink_hue_min": 130,
                "pink_hue_max": 175,
                "pink_saturation_min": 8,
                "pink_saturation_max": 130,
                "pink_value_min": 130,
            }

            @staticmethod
            def _detect_motifs(rgb, *, pink_mask):
                del rgb, pink_mask
                return [
                    candidate(
                        center_px=(110.4, 90.6),
                        suction_px=(80, 70),
                        short_side_px=60.0,
                        edge_clearance_px=4.0,
                        score=0.34,
                        face_score=0.34,
                    )
                ]

        provider = object.__new__(ReferenceFeatureDetectorProvider)
        provider._config = {
            "min_detection_score": 0.6,
            "min_face_score": 0.6,
        }
        provider._motif_helper = FakeMotifHelper()
        provider._motif_raw_score_minimum = 0.28
        provider._last_motif_count = 0
        detected = provider._detect_motifs(
            np.zeros((120, 180, 3), dtype=np.uint8)
        )
        self.assertEqual(len(detected), 1)
        self.assertEqual(detected[0].suction_px, (110, 91))
        self.assertEqual(detected[0].edge_clearance_px, 30.0)
        self.assertGreaterEqual(detected[0].score, 0.6)
        self.assertEqual(detected[0].provider, "reference_feature:motif")


if __name__ == "__main__":
    unittest.main()
