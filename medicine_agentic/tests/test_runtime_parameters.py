from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
    def test_task2_allows_zero_pre_contact_clearance_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = RuntimeParameterStore(
                Path(temporary) / "operator_parameters.json",
                defaults(),
            )
            snapshot = store.update_task(
                "task2",
                {"pre_contact_clearance_m": 0.0},
            )
            self.assertEqual(
                snapshot["tasks"]["task2"]["pre_contact_clearance_m"],
                0.0,
            )
            with self.assertRaisesRegex(ValueError, "between"):
                store.update_task(
                    "task1",
                    {"pre_contact_clearance_m": 0.0},
                )

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

    def test_task1_allows_100mm_lift_and_minus85mm_contact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = RuntimeParameterStore(
                Path(temporary) / "operator_parameters.json",
                defaults(),
            )
            layer_map = {"1": -0.085, "2": -0.060, "3": -0.035}
            snapshot = store.update_task(
                "task1",
                {
                    "test_lift_m": 0.10,
                    "contact_flange_z_m_by_layer": layer_map,
                },
            )
            self.assertEqual(snapshot["tasks"]["task1"]["test_lift_m"], 0.10)
            self.assertEqual(
                snapshot["tasks"]["task1"]["contact_flange_z_m_by_layer"],
                layer_map,
            )
            with self.assertRaisesRegex(ValueError, "between"):
                store.update_task("task1", {"test_lift_m": 0.101})
            with self.assertRaisesRegex(ValueError, "between"):
                store.update_task(
                    "task1",
                    {
                        "contact_flange_z_m_by_layer": {
                            "1": -0.101,
                            "2": -0.060,
                            "3": -0.035,
                        }
                    },
                )
            with self.assertRaisesRegex(ValueError, "between"):
                store.update_task("task2", {"test_lift_m": 0.10})

    def test_named_pose_round_trips_with_task_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "operator_parameters.json"
            store = RuntimeParameterStore(path, defaults())
            pose = {
                "frame": "left_base",
                "position_m": [0.2, -0.1, 0.1],
                "quaternion_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
                "joint_positions_rad": [0.0, 0.0, 0.0, 1.5, 0.0, -1.5],
                "gripper_position_m": 0.0,
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
            self.assertEqual(saved["gripper_position_m"], 0.0)
            self.assertEqual(
                reloaded.pose("left", "safe_transport")["position_m"],
                pose["position_m"],
            )
            with self.assertRaisesRegex(ValueError, "does not exist"):
                reloaded.pose("right", "safe_transport")

    def test_named_pose_delete_is_persistent_and_rejects_missing_pose(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "operator_parameters.json"
            store = RuntimeParameterStore(path, defaults())
            pose = {
                "position_m": [0.2, -0.1, 0.1],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "joint_positions_rad": [0.0] * 6,
            }
            store.save_pose("left", "temporary_pose", pose)

            snapshot = store.delete_pose("left", "temporary_pose")

            self.assertEqual(snapshot["revision"], 2)
            self.assertNotIn("temporary_pose", snapshot["poses"]["left"])
            reloaded = RuntimeParameterStore(path, defaults())
            with self.assertRaisesRegex(ValueError, "does not exist"):
                reloaded.pose("left", "temporary_pose")
            with self.assertRaisesRegex(ValueError, "does not exist"):
                reloaded.delete_pose("left", "temporary_pose")

    def test_named_pose_delete_rolls_back_when_persistence_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = RuntimeParameterStore(
                Path(temporary) / "operator_parameters.json",
                defaults(),
            )
            pose = {
                "position_m": [0.2, -0.1, 0.1],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "joint_positions_rad": [0.0] * 6,
            }
            store.save_pose("right", "temporary_pose", pose)
            before = store.snapshot()

            with mock.patch.object(store, "_persist", side_effect=OSError("disk")):
                with self.assertRaisesRegex(OSError, "disk"):
                    store.delete_pose("right", "temporary_pose")

            self.assertEqual(store.snapshot(), before)
            self.assertEqual(
                store.pose("right", "temporary_pose")["position_m"],
                pose["position_m"],
            )

    def test_malformed_file_falls_back_to_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "operator_parameters.json"
            path.write_text(json.dumps({"tasks": {"task2": {"test_lift_m": 9}}}))
            store = RuntimeParameterStore(path, defaults())
            self.assertEqual(store.task("task2")["test_lift_m"], 0.02)
            self.assertEqual(store.snapshot()["poses"], {"left": {}, "right": {}})


if __name__ == "__main__":
    unittest.main()
