from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from medicine_agentic.runtime_parameters import RuntimeParameterStore


def defaults() -> dict:
    return {
        "task1": {
            "transit_z_m": 0.1,
            "pre_contact_clearance_m": 0.025,
            "test_lift_m": 0.02,
            "contact_flange_z_m_by_layer": {
                "1": 0.04,
                "2": 0.065,
                "3": 0.09,
            },
        },
        "task2": {
            "transit_z_m": 0.1,
            "pre_contact_clearance_m": 0.025,
            "test_lift_m": 0.02,
            "contact_flange_z_m": 0.0273,
        },
        "task3": {
            "transit_z_m": 0.1,
            "pre_contact_clearance_m": 0.025,
            "test_lift_m": 0.02,
            "contact_flange_z_m": 0.015,
        },
    }


class RuntimeParameterStoreTests(unittest.TestCase):
    def test_update_is_immediate_atomic_and_persistent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "operator_parameters.json"
            store = RuntimeParameterStore(path, defaults())
            snapshot = store.update_task(
                "task2",
                {"contact_flange_z_m": 0.0245, "test_lift_m": 0.015},
            )
            self.assertEqual(snapshot["revision"], 1)
            self.assertEqual(store.task("task2")["contact_flange_z_m"], 0.0245)
            self.assertTrue(path.is_file())
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

            reloaded = RuntimeParameterStore(path, defaults())
            self.assertEqual(reloaded.task("task2")["contact_flange_z_m"], 0.0245)
            self.assertEqual(reloaded.task("task2")["test_lift_m"], 0.015)

    def test_invalid_values_do_not_replace_active_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = RuntimeParameterStore(
                Path(temporary) / "operator_parameters.json",
                defaults(),
            )
            before = store.snapshot()
            with self.assertRaisesRegex(ValueError, "between"):
                store.update_task("task2", {"test_lift_m": 0.5})
            self.assertEqual(store.snapshot(), before)

    def test_named_pose_round_trips_with_task_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "operator_parameters.json"
            store = RuntimeParameterStore(path, defaults())
            pose = {
                "frame": "left_base",
                "position_m": [0.2, -0.1, 0.1],
                "quaternion_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
                "joint_positions_rad": [0.0, 0.0, 0.0, 1.5, 0.0, -1.5],
                "captured_at": 1234.5,
            }
            snapshot = store.save_pose("left", "safe_transport", pose)
            self.assertEqual(
                snapshot["poses"]["left"]["safe_transport"]["position_m"],
                pose["position_m"],
            )
            reloaded = RuntimeParameterStore(path, defaults())
            saved = reloaded.snapshot()["poses"]["left"]["safe_transport"]
            self.assertEqual(saved["joint_positions_rad"], pose["joint_positions_rad"])
            self.assertEqual(
                reloaded.pose("left", "safe_transport")["position_m"],
                pose["position_m"],
            )
            with self.assertRaisesRegex(ValueError, "does not exist"):
                reloaded.pose("right", "safe_transport")

    def test_malformed_file_falls_back_to_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "operator_parameters.json"
            path.write_text(json.dumps({"tasks": {"task2": {"test_lift_m": 9}}}))
            store = RuntimeParameterStore(path, defaults())
            self.assertEqual(store.task("task2")["test_lift_m"], 0.02)
            self.assertEqual(store.snapshot()["poses"], {"left": {}, "right": {}})


if __name__ == "__main__":
    unittest.main()
