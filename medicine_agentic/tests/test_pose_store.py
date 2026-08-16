from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from medicine_agentic.airbot_readonly import normalize_end_pose, summarize_arm_samples
from medicine_agentic.pose_store import (
    PoseStore,
    PoseStoreError,
    document_sha256,
    new_pose_document,
    validate_pose_document,
)
from medicine_agentic.pose_cli import run as run_pose_cli


def arm_record(joint_offset: float = 0.0) -> dict:
    joints = [0.0, -0.2, 0.5, 0.0, 0.0, 0.0]
    joints[0] += joint_offset
    return {
        "joint_position_rad": joints,
        "flange_pose_in_base": {
            "position_m": [0.3, 0.0, 0.2],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "eef_feedback_m": [0.07],
        "driver_state": "IDLE",
        "control_mode": "SERVO_JOINT_POS",
        "capture_metrics": {"stable": True},
    }


def pose_record() -> dict:
    return {
        "kind": "paired_joint_pose",
        "status": "draft",
        "captured_at": "2026-07-30T00:00:00.000Z",
        "source": "test",
        "instruction": "",
        "scene_id": "test",
        "context": {"base_docked": True, "lift_height_mm": 0.0},
        "tooling": {},
        "arms": {"left": arm_record(), "right": arm_record(0.1)},
        "validation": {"stable": True, "collision_free": "unproven"},
    }


class PoseStoreTests(unittest.TestCase):
    def test_new_document_hash_is_valid(self) -> None:
        payload = new_pose_document()
        self.assertEqual(payload["content_sha256"], document_sha256(payload))
        self.assertEqual(validate_pose_document(payload), [])
        self.assertEqual(payload, new_pose_document())

    def test_upsert_requires_explicit_hash_to_replace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PoseStore(Path(directory) / "poses.json")
            first = store.upsert_pose("home", pose_record())
            self.assertEqual(first["revision"], 1)
            with self.assertRaises(PoseStoreError):
                store.upsert_pose("home", pose_record(), replace=True)

    def test_atomic_save_and_read_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PoseStore(Path(directory) / "poses.json")
            saved = store.upsert_pose("home", pose_record())
            reloaded = json.loads(store.path.read_text())
            self.assertEqual(saved, reloaded)
            self.assertEqual(validate_pose_document(reloaded), [])

    def test_sample_summary_rejects_motion(self) -> None:
        samples = []
        for index in range(5):
            samples.append(
                {
                    "joint_position_rad": [0.01 * index, -0.2, 0.5, 0.0, 0.0, 0.0],
                    "flange_position_m": [0.3 + 0.002 * index, 0.0, 0.2],
                    "flange_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "eef_feedback_m": [0.07],
                    "driver_state": "IDLE",
                    "control_mode": "SERVO_JOINT_POS",
                }
            )
        summary = summarize_arm_samples(samples)
        self.assertFalse(summary["capture_metrics"]["stable"])

    def test_normalize_end_pose_accepts_both_sdk_shapes(self) -> None:
        position, quaternion = normalize_end_pose(
            ([0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 2.0])
        )
        np.testing.assert_allclose(position, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(quaternion, [0.0, 0.0, 0.0, 1.0])

    def test_operator_can_explicitly_approve_a_stable_pose(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.json"
            store = PoseStore(path)
            store.upsert_pose("home", pose_record())
            with patch("builtins.print"):
                code = run_pose_cli(
                    [
                        "--store",
                        str(path),
                        "approve",
                        "home",
                        "--confirm",
                        "VALIDATE home",
                    ]
                )
            self.assertEqual(code, 0)
            approved = store.load()["poses"]["home"]
            self.assertEqual(approved["status"], "validated")
            self.assertTrue(approved["validation"]["collision_free"])
            self.assertFalse(
                approved["validation"]["automatic_full_link_collision_check"]
            )
        position, quaternion = normalize_end_pose(
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
        )
        np.testing.assert_allclose(position, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(quaternion, [0.0, 0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
