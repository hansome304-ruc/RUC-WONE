import unittest

import cv2
import numpy as np

from medicine_agentic.packaging_console import locate_task1_shipping_box_slots


class Task1SlotGridTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "rows": 10,
            "columns": 2,
            "fill_order": "right_column_then_left",
            "opening_corners_norm": [
                [0.10, 0.10],
                [0.90, 0.10],
                [0.90, 0.90],
                [0.10, 0.90],
            ],
            "occupancy_inset_scale": 0.64,
            "occupied_maximum_saturation": 200.0,
            "occupied_minimum_value": 150.0,
            "occupied_minimum_light_ratio": 0.55,
        }

    def _empty_box(self) -> np.ndarray:
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[:] = (45, 80, 125)
        return image

    def test_empty_box_still_produces_all_twenty_slots(self) -> None:
        detection, overlay = locate_task1_shipping_box_slots(
            self._empty_box(),
            self.config,
        )

        self.assertEqual(detection["capacity"], 20)
        self.assertEqual(len(detection["slots"]), 20)
        self.assertEqual(detection["occupied_count"], 0)
        self.assertEqual(detection["next_slot"]["index"], 1)
        self.assertEqual(detection["next_slot"]["column"], 2)
        self.assertEqual(overlay.shape, (240, 320, 3))

    def test_eleven_visible_cartons_select_slot_twelve(self) -> None:
        image = self._empty_box()
        height, width = image.shape[:2]
        left = int(round(0.10 * (width - 1)))
        right = int(round(0.90 * (width - 1)))
        top = int(round(0.10 * (height - 1)))
        bottom = int(round(0.90 * (height - 1)))
        middle = (left + right) // 2
        row_height = (bottom - top) / 10.0

        # The configured sequence fills the right column first (slots 01-10),
        # then the left column from top to bottom (slot 11 onward).
        cv2.rectangle(image, (middle, top), (right, bottom), (235, 235, 235), -1)
        cv2.rectangle(
            image,
            (left, top),
            (middle, int(round(top + row_height))),
            (235, 235, 235),
            -1,
        )

        detection, _ = locate_task1_shipping_box_slots(image, self.config)

        self.assertEqual(detection["occupied_count"], 11)
        self.assertEqual(detection["empty_count"], 9)
        self.assertEqual(detection["next_slot"]["index"], 12)
        self.assertEqual(detection["next_slot"]["row"], 2)
        self.assertEqual(detection["next_slot"]["column"], 1)


if __name__ == "__main__":
    unittest.main()
