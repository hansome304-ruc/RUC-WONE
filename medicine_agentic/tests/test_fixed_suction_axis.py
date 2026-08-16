from __future__ import annotations

import unittest

import numpy as np

from medicine_agentic.fixed_suction_axis import (
    fixed_suction_axis_status,
    project_fixed_suction_axis,
    quaternion_xyzw_to_matrix,
)


class FixedSuctionAxisTests(unittest.TestCase):
    def test_status_remains_disabled_until_calibrated(self) -> None:
        status = fixed_suction_axis_status(
            {
                "enabled": False,
                "calibrated": False,
                "cup_center_spacing_mm": 50.0,
                "cup_diameter_mm": 25.0,
                "safety_margin_mm": 8.0,
            }
        )
        self.assertFalse(status["ready"])
        self.assertIn("calibration_required", status["blockers"])

    def test_identity_quaternion(self) -> None:
        np.testing.assert_allclose(
            quaternion_xyzw_to_matrix([0.0, 0.0, 0.0, 1.0]),
            np.eye(3),
            atol=1e-12,
        )

    def test_projects_real_fifty_millimetre_spacing(self) -> None:
        projection = project_fixed_suction_axis(
            midpoint_left_base_m=[0.0, 0.0, 1.0],
            locked_flange_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
            axis_local_xyz=[1.0, 0.0, 0.0],
            approach_local_xyz=[0.0, 0.0, 1.0],
            cup_center_spacing_m=0.05,
            cup_diameter_m=0.025,
            safety_margin_m=0.0,
            cam_to_left=np.eye(4),
            intrinsics=np.asarray(
                [[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]]
            ),
            candidate_polygon_px=[
                [240.0, 190.0],
                [400.0, 190.0],
                [400.0, 290.0],
                [240.0, 290.0],
            ],
            image_shape=(480, 640),
        )
        np.testing.assert_allclose(projection.cup_centers_px, [[295.0, 240.0], [345.0, 240.0]])
        self.assertTrue(projection.valid_2d)
        self.assertEqual(projection.blockers, ())

    def test_rejects_when_either_contact_patch_crosses_box_edge(self) -> None:
        projection = project_fixed_suction_axis(
            midpoint_left_base_m=[0.0, 0.0, 1.0],
            locked_flange_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
            axis_local_xyz=[1.0, 0.0, 0.0],
            approach_local_xyz=[0.0, 0.0, 1.0],
            cup_center_spacing_m=0.05,
            cup_diameter_m=0.025,
            safety_margin_m=0.008,
            cam_to_left=np.eye(4),
            intrinsics=np.asarray(
                [[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]]
            ),
            candidate_polygon_px=[
                [300.0, 215.0],
                [340.0, 215.0],
                [340.0, 265.0],
                [300.0, 265.0],
            ],
        )
        self.assertFalse(projection.valid_2d)
        self.assertIn("fixed_suction_footprint_outside_candidate", projection.blockers)


if __name__ == "__main__":
    unittest.main()
