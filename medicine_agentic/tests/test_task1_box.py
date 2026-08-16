from __future__ import annotations

import unittest

import cv2
import numpy as np

from medicine_agentic.task1_box import (
    DepthEstimate,
    deproject_pixel,
    detect_cartons,
    estimate_surface_plane,
    transform_point,
)


class Task1BoxTest(unittest.TestCase):
    def test_detects_one_rotated_carton_in_roi(self) -> None:
        rgb = np.zeros((360, 640, 3), dtype=np.uint8)
        rgb[:] = [45, 112, 91]
        rect = ((390.0, 190.0), (130.0, 58.0), -12.0)
        polygon = np.round(cv2.boxPoints(rect)).astype(np.int32)
        cv2.fillConvexPoly(rgb, polygon, (220, 190, 215))

        config = {
            "roi_norm": [0.42, 0.25, 0.82, 0.78],
            "bright_value_min": 140,
            "bright_saturation_max": 110,
            "pink_hue_min": 135,
            "pink_hue_max": 179,
            "pink_saturation_min": 8,
            "pink_value_min": 110,
            "min_area_fraction": 0.002,
            "max_area_fraction": 0.04,
            "min_short_side_fraction": 0.06,
            "max_short_side_fraction": 0.20,
            "expected_aspect_ratio": 2.2,
            "min_aspect_ratio": 1.7,
            "max_aspect_ratio": 2.8,
            "min_rectangularity": 0.75,
            "min_bright_fill": 0.7,
            "min_edge_clearance_px": 10,
        }
        candidates = detect_cartons(rgb, config)
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].center_px[0], 390.0, delta=3.0)
        self.assertAlmostEqual(candidates[0].center_px[1], 190.0, delta=3.0)
        self.assertGreater(candidates[0].score, 0.8)
        self.assertGreater(candidates[0].edge_clearance_px, 20.0)

    def test_deproject_and_transform(self) -> None:
        intrinsics = np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        )
        camera_point = deproject_pixel((370, 215), 1000.0, intrinsics)
        np.testing.assert_allclose(camera_point, [0.1, -0.05, 1.0])

        transform = np.eye(4)
        transform[:3, 3] = [0.2, 0.3, -0.4]
        base_point = transform_point(camera_point, transform)
        np.testing.assert_allclose(base_point, [0.3, 0.25, 0.6])

    def test_plane_check_accepts_oblique_but_flat_surface(self) -> None:
        intrinsics = np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        )
        # Depth changes across the image, yet all five points lie on one plane.
        pixels = ((320, 240), (300, 240), (340, 240), (320, 220), (320, 260))
        depths = []
        for u, v in pixels:
            # Plane z = 1 + 0.1*x.  Solve using x=(u-cx)z/fx.
            z = 1.0 / (1.0 - 0.1 * (u - 320.0) / 500.0)
            depths.append(z * 1000.0)
        estimate = DepthEstimate(
            median_mm=float(np.median(depths)),
            spread_mm=float(max(depths) - min(depths)),
            valid_samples=5,
            samples_mm=tuple(depths),
            sample_pixels_px=pixels,
            frame_age_s=0.01,
        )
        normal, tilt, residual = estimate_surface_plane(
            estimate, intrinsics, np.eye(4)
        )
        self.assertLess(residual, 1e-6)
        self.assertLess(tilt, 6.0)
        self.assertGreater(normal[2], 0.99)


if __name__ == "__main__":
    unittest.main()
