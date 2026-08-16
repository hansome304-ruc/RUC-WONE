from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from medicine_agentic.task1_box import (
    BoxCandidate,
    evaluate_dual_suction_depth,
    plan_dual_suction_target,
)


def config() -> dict:
    return {
        "dual_suction": {
            "enabled": True,
            "alignment": "long_axis",
            "carton_face_size_mm": [130.0, 85.0],
            "cup_diameter_mm": 25.0,
            "cup_edge_gap_mm": 25.0,
            "cup_center_spacing_mm": 50.0,
            "assembly_outer_span_mm": 75.0,
            "safety_margin_mm": 8.0,
            "min_polygon_clearance_px": 4.0,
            "required_face_types": ["front_large", "back_large"],
            "min_depth_valid_ratio": 0.8,
        }
    }


def candidate(
    *,
    center: tuple[float, float] = (220.0, 150.0),
    long_side: float = 130.0,
    short_side: float = 85.0,
    angle_deg: float = 0.0,
    face_type: str = "front_large",
    graspable: bool = True,
) -> BoxCandidate:
    rect = (center, (long_side, short_side), angle_deg)
    polygon = tuple(
        (float(point[0]), float(point[1])) for point in cv2.boxPoints(rect)
    )
    return BoxCandidate(
        center_px=center,
        # Deliberately offset: the dual target must ignore this legacy point.
        suction_px=(int(center[0] + 11), int(center[1] - 7)),
        polygon_px=polygon,
        long_side_px=long_side,
        short_side_px=short_side,
        angle_deg=angle_deg,
        rectangularity=0.98,
        bright_fill=0.95,
        edge_clearance_px=35.0,
        score=0.95,
        provider="test",
        face_type=face_type,
        face_score=0.95,
        reference_face_id="front_large_01",
        graspable=graspable,
        grasp_blockers=() if graspable else ("face_unverified",),
    )


class DualSuctionTargetTests(unittest.TestCase):
    def test_midpoint_spacing_and_long_axis_are_exact(self) -> None:
        item = candidate(angle_deg=-17.0)
        target = plan_dual_suction_target(item, config(), image_shape=(480, 848))
        self.assertIsNotNone(target)
        assert target is not None
        self.assertTrue(target.valid_2d, target.blockers)
        np.testing.assert_allclose(target.midpoint_px, item.center_px)
        np.testing.assert_allclose(
            np.mean(np.asarray(target.cup_centers_px), axis=0),
            item.center_px,
        )
        self.assertAlmostEqual(
            float(
                np.linalg.norm(
                    np.asarray(target.cup_centers_px[1])
                    - np.asarray(target.cup_centers_px[0])
                )
            ),
            50.0,
            places=5,
        )
        vector = (
            np.asarray(target.cup_centers_px[1])
            - np.asarray(target.cup_centers_px[0])
        )
        angle = math.degrees(math.atan2(vector[1], vector[0]))
        self.assertAlmostEqual(angle, -17.0, places=5)
        self.assertEqual(target.raw_long_end_margin_mm, 27.5)
        self.assertEqual(target.raw_short_side_margin_mm, 30.0)

    def test_vertical_candidate_places_cups_above_and_below_center(self) -> None:
        target = plan_dual_suction_target(
            candidate(angle_deg=-90.0), config(), image_shape=(480, 848)
        )
        assert target is not None
        self.assertTrue(target.valid_2d, target.blockers)
        self.assertAlmostEqual(target.cup_centers_px[0][0], 220.0, places=5)
        self.assertAlmostEqual(target.cup_centers_px[1][0], 220.0, places=5)
        self.assertAlmostEqual(
            abs(
                target.cup_centers_px[1][1]
                - target.cup_centers_px[0][1]
            ),
            50.0,
            places=5,
        )

    def test_side_face_and_cropped_contact_fail_closed(self) -> None:
        side = plan_dual_suction_target(
            candidate(face_type="long_side_a"),
            config(),
            image_shape=(480, 848),
        )
        assert side is not None
        self.assertFalse(side.valid_2d)
        self.assertIn("dual_suction_face_not_allowed", side.blockers)

        cropped = plan_dual_suction_target(
            candidate(center=(30.0, 30.0)),
            config(),
            image_shape=(480, 848),
        )
        assert cropped is not None
        self.assertFalse(cropped.valid_2d)
        self.assertIn("dual_suction_outside_image", cropped.blockers)

    def test_inconsistent_hardware_dimensions_fail_closed(self) -> None:
        bad = config()
        bad["dual_suction"]["cup_center_spacing_mm"] = 42.0
        target = plan_dual_suction_target(candidate(), bad)
        assert target is not None
        self.assertFalse(target.valid_2d)
        self.assertIn("dual_suction_dimensions_inconsistent", target.blockers)

    def test_depth_support_checks_both_contact_patches(self) -> None:
        target = plan_dual_suction_target(candidate(), config())
        assert target is not None
        depth = np.full((300, 440), 1000, dtype=np.uint16)
        report = evaluate_dual_suction_depth(depth, 0.001, target, config())
        self.assertTrue(report["available"])
        self.assertTrue(report["valid"])
        self.assertEqual(len(report["cups"]), 2)
        self.assertTrue(all(cup["valid_ratio"] == 1.0 for cup in report["cups"]))
        self.assertEqual(report["median_depth_delta_mm"], 0.0)


if __name__ == "__main__":
    unittest.main()
