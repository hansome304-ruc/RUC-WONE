from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from medicine_agentic.tcp_pivot import (
    PivotCalibrationError,
    PivotSampleStore,
    content_sha256,
    solve_pivot_translation,
    summarize_flange_samples,
    write_accepted_result,
)


def axis_angle_quaternion(axis: list[float], angle_rad: float) -> list[float]:
    vector = np.asarray(axis, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    sine = math.sin(angle_rad / 2.0)
    return [
        float(vector[0] * sine),
        float(vector[1] * sine),
        float(vector[2] * sine),
        float(math.cos(angle_rad / 2.0)),
    ]


def quaternion_matrix(quaternion: list[float]) -> np.ndarray:
    x, y, z, w = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def synthetic_samples(*, noise_m: float = 0.0) -> tuple[list[dict], np.ndarray]:
    tcp = np.asarray([0.031, -0.014, 0.112], dtype=np.float64)
    pivot = np.asarray([0.42, -0.08, 0.24], dtype=np.float64)
    orientations = [
        ([1, 0, 0], -0.65),
        ([1, 0, 0], 0.55),
        ([0, 1, 0], -0.60),
        ([0, 1, 0], 0.70),
        ([0, 0, 1], -0.75),
        ([0, 0, 1], 0.60),
        ([1, 1, 0], 0.72),
        ([1, -1, 0], -0.68),
        ([1, 0, 1], 0.66),
        ([0, 1, 1], -0.64),
        ([1, 1, 1], 0.78),
        ([1, -1, 1], -0.70),
    ]
    rng = np.random.default_rng(20260730)
    samples = []
    for index, (axis, angle) in enumerate(orientations, start=1):
        quaternion = axis_angle_quaternion(axis, angle)
        rotation = quaternion_matrix(quaternion)
        translation = pivot - rotation @ tcp
        if noise_m:
            translation += rng.normal(scale=noise_m, size=3)
        samples.append(
            {
                "sample_id": f"sample_{index:04d}",
                "captured_at": "2026-07-30T00:00:00.000Z",
                "label": "",
                "joint_position_rad": [0.0, -0.2, 0.5, 0.0, 0.0, 0.0],
                "flange_pose_in_base": {
                    "position_m": translation.tolist(),
                    "quaternion_xyzw": quaternion,
                },
                "driver_state": "IDLE",
                "control_mode": "SERVO_JOINT_POS",
                "capture_metrics": {"stable": True},
            }
        )
    return samples, tcp


class PivotCalibrationTests(unittest.TestCase):
    def test_exact_synthetic_pivot_recovers_tcp(self) -> None:
        samples, expected_tcp = synthetic_samples()
        result = solve_pivot_translation(samples)
        self.assertTrue(result["acceptance"]["accepted"], result["acceptance"])
        np.testing.assert_allclose(
            result["flange_to_tcp"]["translation_m"],
            expected_tcp,
            atol=1e-10,
        )
        self.assertLess(result["metrics"]["rms_residual_m"], 1e-10)

    def test_noisy_synthetic_pivot_passes_and_is_accurate(self) -> None:
        samples, expected_tcp = synthetic_samples(noise_m=0.00025)
        result = solve_pivot_translation(samples)
        self.assertTrue(result["acceptance"]["accepted"], result["acceptance"])
        np.testing.assert_allclose(
            result["flange_to_tcp"]["translation_m"],
            expected_tcp,
            atol=0.001,
        )
        self.assertLess(result["metrics"]["rms_residual_m"], 0.001)

    def test_same_solver_supports_right_gripper_tcp(self) -> None:
        samples, expected_tcp = synthetic_samples()
        result = solve_pivot_translation(
            samples,
            arm="right",
            tcp_frame="right_gripper_tcp",
        )
        self.assertTrue(result["acceptance"]["accepted"])
        self.assertEqual(result["arm"], "right")
        self.assertEqual(result["frames"]["tcp"], "right_gripper_tcp")
        np.testing.assert_allclose(
            result["translation_m"],
            expected_tcp,
            atol=1e-10,
        )

    def test_right_sample_store_records_right_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document = PivotSampleStore(
                Path(directory) / "right_samples.json",
                arm="right",
            ).load()
            self.assertEqual(document["arm"], "right")
            self.assertEqual(document["base_frame"], "right_base/base_link")
            self.assertEqual(document["tcp_frame"], "right_gripper_tcp")

    def test_degenerate_orientations_fail_conditioning(self) -> None:
        samples, _ = synthetic_samples()
        first_pose = samples[0]["flange_pose_in_base"]
        for sample in samples:
            sample["flange_pose_in_base"] = json.loads(json.dumps(first_pose))
        result = solve_pivot_translation(samples)
        self.assertFalse(result["acceptance"]["accepted"])
        self.assertLess(result["metrics"]["matrix_rank"], 3)
        self.assertTrue(result["acceptance"]["failures"])

    def test_stationary_burst_summary_rejects_motion(self) -> None:
        raw = []
        for index in range(5):
            raw.append(
                {
                    "joint_position_rad": [0.01 * index, -0.2, 0.5, 0, 0, 0],
                    "flange_position_m": [0.3 + 0.002 * index, 0.0, 0.2],
                    "flange_quaternion_xyzw": [0, 0, 0, 1],
                    "driver_state": "IDLE",
                    "control_mode": "SERVO_JOINT_POS",
                }
            )
        self.assertFalse(
            summarize_flange_samples(raw)["capture_metrics"]["stable"]
        )

    def test_sample_append_and_result_write_are_readable(self) -> None:
        samples, _ = synthetic_samples()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = PivotSampleStore(root / "samples.json")
            for sample in samples:
                record = dict(sample)
                record.pop("sample_id")
                record.pop("captured_at")
                record.pop("label")
                store.append(record)
            document = store.load()
            self.assertEqual(len(document["samples"]), len(samples))
            self.assertEqual(document["content_sha256"], content_sha256(document))

            result = solve_pivot_translation(
                document["samples"],
                source_samples_sha256=document["content_sha256"],
            )
            output = root / "left_suction_tcp.json"
            saved = write_accepted_result(output, result)
            self.assertEqual(saved, json.loads(output.read_text(encoding="utf-8")))
            with self.assertRaises(PivotCalibrationError):
                write_accepted_result(output, result)


if __name__ == "__main__":
    unittest.main()
