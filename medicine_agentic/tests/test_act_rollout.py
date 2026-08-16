from __future__ import annotations

import json
import threading
import time
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from medicine_agentic.act_rollout import (
    ActRolloutController,
    ActRolloutSafetyViolation,
)


class FakeArm:
    def __init__(self, name: str, *, follow_commands: bool = True) -> None:
        self.name = name
        self.follow_commands = follow_commands
        self.joints = np.zeros(6, dtype=np.float64)
        self.gripper = 0.001
        self.mode = "IDLE"
        self.servo_calls: list[list[float]] = []
        self.gripper_calls: list[float] = []

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        return None

    def get_state(self) -> str:
        return "IDLE"

    def switch_mode(self, mode: str) -> bool:
        self.mode = mode
        return True

    def get_control_mode(self) -> str:
        return self.mode

    def set_speed_profile(self, _profile: str) -> bool:
        return True

    def get_joint_pos(self) -> list[float]:
        return self.joints.tolist()

    def get_eef_pos(self) -> list[float]:
        return [self.gripper]

    def servo_joint_pos(self, joints: list[float]) -> bool:
        target = np.asarray(joints, dtype=np.float64)
        self.servo_calls.append(target.tolist())
        if self.follow_commands:
            self.joints = target
        return True

    def servo_eef_pos(self, position: float) -> bool:
        self.gripper = float(position)
        self.gripper_calls.append(self.gripper)
        return True


class FakeInference:
    def __init__(
        self,
        *,
        block_after: int | None = None,
        joint_delta: float = 0.10,
        per_step_delta: float = 0.0,
    ) -> None:
        self.calls = 0
        self.requested_horizons: list[int] = []
        self.block_after = block_after
        self.joint_delta = float(joint_delta)
        self.per_step_delta = float(per_step_delta)
        self.blocked = threading.Event()
        self.release = threading.Event()

    def status(self, *, force: bool = False) -> dict:
        return {"ready": True}

    def predict(self, *, state, horizon, **_kwargs) -> dict:
        self.calls += 1
        self.requested_horizons.append(int(horizon))
        if self.block_after is not None and self.calls >= self.block_after:
            self.blocked.set()
            self.release.wait(2.0)
        state_array = np.asarray(state, dtype=np.float64)
        actions = np.repeat(state_array[None, :], horizon, axis=0)
        actions[:, [0, 7]] += self.joint_delta
        actions[:, [0, 7]] += (
            np.arange(horizon, dtype=np.float64)[:, None] * self.per_step_delta
        )
        return {
            "actions": actions.tolist(),
            "action_representation": "absolute_joint_target",
        }


def wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached before timeout")


class ActRolloutControllerTests(unittest.TestCase):
    def make_controller(
        self,
        *,
        inference: FakeInference | None = None,
        checker=None,
        execute_steps: int = 1,
        debug_log_dir: str | None = None,
        follow_commands: bool = True,
        tracking_error_blocking: bool = True,
        camera_arm_timing_blocking: bool = True,
        max_command_step_rad: float = 0.35,
        configured_speed_profile: str = "DEFAULT",
        speed_profile: str | None = "FAST",
    ) -> tuple[ActRolloutController, dict[str, FakeArm]]:
        arms: dict[str, FakeArm] = {}

        def arm_factory(_host: str, port: int) -> FakeArm:
            name = "left" if port == 50051 else "right"
            arm = FakeArm(name, follow_commands=follow_commands)
            arms[name] = arm
            return arm

        def frame_provider(_first: bool, *, captured_after=None):
            timestamp = max(
                time.time(),
                0.0 if captured_after is None else captured_after + 0.001,
            )
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            return {
                name: SimpleNamespace(bgr=frame, captured_at=timestamp)
                for name in ("front", "left_wrist", "right_wrist")
            }

        config = {
                "enabled": True,
                "speed_profile": configured_speed_profile,
                "command_hz": 60,
                "horizon": 25,
                "execute_steps_per_inference": execute_steps,
                "feedback_every_n": 1,
                "max_camera_arm_delta_ms": 200,
                "camera_arm_timing_blocking": camera_arm_timing_blocking,
                "max_command_step_rad": max_command_step_rad,
                "tracking_error_blocking": tracking_error_blocking,
            }
        if debug_log_dir is not None:
            config.update(
                {
                    "debug_log_enabled": True,
                    "debug_log_dir": debug_log_dir,
                }
            )
        controller = ActRolloutController(
            config,
            inference=inference or FakeInference(),
            frame_provider=frame_provider,
            interlock=lambda: None,
            start_pose_checker=checker
            or (lambda _state: {"out_of_range": [], "blocking": False}),
            arm_factory=arm_factory,
            servo_mode="SERVO_JOINT_POS",
            speed_profile=speed_profile,
        )
        return controller, arms

    def test_delta_actions_stay_anchored_to_chunk_observation(self) -> None:
        controller, _arms = self.make_controller()
        observation_joints = {
            "left": np.asarray([1.0, 2.0, 3.0, 0.0, -1.0, -2.0]),
            "right": np.asarray([-1.0, -2.0, -3.0, 0.0, 1.0, 2.0]),
        }
        observation_grippers = {"left": 0.02, "right": 0.03}
        previous_joints = {
            "left": observation_joints["left"] + 0.05,
            "right": observation_joints["right"] - 0.05,
        }
        previous_grippers = {"left": 0.021, "right": 0.029}
        action = np.asarray([0.1] * 6 + [0.002] + [-0.1] * 6 + [-0.003])

        joints, grippers, max_step = controller._action_targets(
            action,
            previous_joints,
            previous_grippers,
            "delta_target_minus_observation_state",
            observation_joints,
            observation_grippers,
        )

        np.testing.assert_allclose(
            joints["left"], observation_joints["left"] + 0.1
        )
        np.testing.assert_allclose(
            joints["right"], observation_joints["right"] - 0.1
        )
        self.assertAlmostEqual(grippers["left"], 0.02 + 0.002 * controller.gripper_scale)
        self.assertAlmostEqual(grippers["right"], 0.03 - 0.003 * controller.gripper_scale)
        self.assertAlmostEqual(max_step, 0.05)
        controller.close()

    def test_large_model_step_is_rejected_before_servo_send(self) -> None:
        controller, arms = self.make_controller(
            inference=FakeInference(joint_delta=2.0),
        )
        controller.max_command_step_rad = 0.15
        controller.start()
        wait_for(lambda: controller.status()["state"] == "error")
        self.assertIn("ACT command step 2.000 rad exceeds limit 0.150 rad", controller.status()["error"])
        self.assertEqual(controller.status()["command_count"], 0)
        for arm in arms.values():
            self.assertEqual(arm.servo_calls, [])
        controller.close()

    def test_max_command_step_accepts_40_degrees(self) -> None:
        controller, _arms = self.make_controller(
            max_command_step_rad=np.deg2rad(40.0),
        )
        self.assertAlmostEqual(
            controller.status()["max_command_step_rad"],
            np.deg2rad(40.0),
        )
        controller.close()

    def test_max_command_step_rejects_more_than_40_degrees(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid act_rollout safety settings"):
            self.make_controller(max_command_step_rad=np.deg2rad(40.1))

    def test_delta_chunk_step_limit_is_anchored_to_observation(self) -> None:
        controller, _arms = self.make_controller()
        controller.max_command_step_rad = 0.15
        observation_joints = {
            "left": np.zeros(6, dtype=np.float64),
            "right": np.zeros(6, dtype=np.float64),
        }
        previous_joints = {
            "left": np.zeros(6, dtype=np.float64),
            "right": np.zeros(6, dtype=np.float64),
        }
        with self.assertRaisesRegex(
            ActRolloutSafetyViolation,
            r"right ACT command step 2\.140 rad exceeds limit 0\.150 rad",
        ):
            controller._action_targets(
                np.asarray([0.0] * 7 + [0.0, 0.0, 0.0, -2.14, 0.0, 0.0, 0.0]),
                previous_joints,
                {"left": 0.02, "right": 0.02},
                "delta_target_minus_observation_state",
                observation_joints,
                {"left": 0.02, "right": 0.02},
            )
        controller.close()

    def test_chunk_boundary_requires_post_command_frames_and_feedback(self) -> None:
        frame_cutoffs: list[float | None] = []
        frame_timestamps: list[float] = []
        inference = FakeInference(block_after=2)
        controller, _arms = self.make_controller(inference=inference)

        def frame_provider(_first: bool, *, captured_after=None):
            frame_cutoffs.append(captured_after)
            timestamp = time.time()
            if captured_after is not None:
                timestamp = captured_after + 0.002
            frame_timestamps.append(timestamp)
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            return {
                name: SimpleNamespace(bgr=frame, captured_at=timestamp)
                for name in ("front", "left_wrist", "right_wrist")
            }

        controller.frame_provider = frame_provider
        controller.start()
        wait_for(inference.blocked.is_set)
        self.assertIsNone(frame_cutoffs[0])
        self.assertIsNotNone(frame_cutoffs[1])
        self.assertGreater(frame_timestamps[1], frame_cutoffs[1])
        controller.stop()
        inference.release.set()
        wait_for(lambda: controller.status()["state"] == "stopped")
        controller.close()

    def test_stale_post_command_frame_fails_closed(self) -> None:
        inference = FakeInference()
        controller, _arms = self.make_controller(inference=inference)

        def stale_frame_provider(_first: bool, *, captured_after=None):
            timestamp = time.time()
            if captured_after is not None:
                timestamp = captured_after
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            return {
                name: SimpleNamespace(bgr=frame, captured_at=timestamp)
                for name in ("front", "left_wrist", "right_wrist")
            }

        controller.frame_provider = stale_frame_provider
        controller.start()
        wait_for(lambda: controller.status()["state"] == "error")
        self.assertIn(
            "observation predates the completed action chunk",
            controller.status()["error"],
        )
        controller.close()

    def test_camera_arm_timing_can_be_diagnostic_only(self) -> None:
        inference = FakeInference(block_after=1)
        controller, _arms = self.make_controller(
            inference=inference,
            camera_arm_timing_blocking=False,
        )

        def stale_frame_provider(_first: bool, *, captured_after=None):
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            return {
                name: SimpleNamespace(bgr=frame, captured_at=time.time() - 1.0)
                for name in ("front", "left_wrist", "right_wrist")
            }

        controller.frame_provider = stale_frame_provider
        controller.start()
        wait_for(inference.blocked.is_set)
        status = controller.status()
        self.assertNotEqual(status["state"], "error")
        self.assertFalse(status["camera_arm_timing_blocking"])
        controller.stop()
        inference.release.set()
        wait_for(lambda: controller.status()["state"] == "stopped")
        controller.close()

    def test_default_speed_profile_is_reported(self) -> None:
        controller, _arms = self.make_controller(speed_profile=None)
        self.assertEqual(controller.status()["speed_profile"], "DEFAULT")
        controller.close()

    def test_configured_fast_speed_profile_is_reported_for_act_only(self) -> None:
        controller, _arms = self.make_controller(
            configured_speed_profile="FAST",
            speed_profile=None,
        )
        self.assertEqual(controller.status()["speed_profile"], "FAST")
        controller.close()

    def test_invalid_configured_speed_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "act_rollout.speed_profile"):
            self.make_controller(
                configured_speed_profile="TURBO",
                speed_profile=None,
            )

    def test_tracking_error_can_be_diagnostic_only(self) -> None:
        inference = FakeInference(block_after=2, joint_delta=0.30)
        controller, _arms = self.make_controller(
            inference=inference,
            follow_commands=False,
            tracking_error_blocking=False,
        )
        controller.start()
        wait_for(inference.blocked.is_set)
        status = controller.status()
        self.assertEqual(status["state"], "running")
        self.assertEqual(status["command_count"], 1)
        self.assertGreaterEqual(status["tracking_warning_count"], 1)
        self.assertGreater(status["max_tracking_errors_rad"]["left"], 0.29)

        controller.stop()
        inference.release.set()
        wait_for(lambda: controller.status()["state"] == "stopped")
        controller.close()

    def test_debug_trace_persists_actions_timing_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inference = FakeInference(block_after=2)
            controller, _arms = self.make_controller(
                inference=inference,
                debug_log_dir=temporary,
            )
            started = controller.start()
            trace_path = Path(started["debug_log_path"])
            wait_for(inference.blocked.is_set)
            controller.stop()
            inference.release.set()
            wait_for(lambda: controller.status()["state"] == "stopped")
            controller.close()

            chunk = json.loads((trace_path / "chunk_000000.json").read_text())
            execution = json.loads(
                (trace_path / "execution_000000.json").read_text()
            )
            summary = json.loads(
                (trace_path / "session_summary.json").read_text()
            )
            self.assertEqual(len(chunk["actions"]), 1)
            self.assertEqual(len(chunk["actions"][0]), 14)
            self.assertEqual(inference.requested_horizons, [1, 1])
            self.assertEqual(len(execution["records"]), 1)
            self.assertIn("rpc_elapsed_ms", execution["records"][0])
            self.assertEqual(summary["state"], "stopped")

    def test_executes_the_complete_model_action_horizon(self) -> None:
        inference = FakeInference(block_after=2)
        controller, _arms = self.make_controller(
            inference=inference,
            execute_steps=25,
        )
        controller.start()
        wait_for(inference.blocked.is_set)
        self.assertEqual(controller.status()["command_count"], 25)

        result = controller.stop()
        self.assertTrue(result["hold_confirmed"])
        inference.release.set()
        wait_for(lambda: controller.status()["state"] == "stopped")
        controller.close()

    def test_start_can_select_the_number_of_front_actions(self) -> None:
        inference = FakeInference(block_after=2)
        controller, _arms = self.make_controller(
            inference=inference,
            execute_steps=20,
        )
        started = controller.start(execute_steps_per_inference=5)
        self.assertEqual(started["execute_steps_per_inference"], 5)
        wait_for(inference.blocked.is_set)
        self.assertEqual(controller.status()["command_count"], 5)
        self.assertEqual(inference.requested_horizons, [5, 5])
        self.assertEqual(
            controller.status()["execute_steps_per_inference"],
            5,
        )

        controller.stop()
        inference.release.set()
        wait_for(lambda: controller.status()["state"] == "stopped")
        controller.close()

    def test_start_rejects_execute_steps_beyond_model_horizon(self) -> None:
        controller, _arms = self.make_controller(execute_steps=20)
        with self.assertRaisesRegex(ValueError, r"\[1, 25\]"):
            controller.start(execute_steps_per_inference=26)
        controller.close()

    def test_first_model_frame_is_not_rate_limited_and_stop_holds(self) -> None:
        inference = FakeInference(block_after=2)
        controller, arms = self.make_controller(inference=inference)
        controller.start()
        wait_for(inference.blocked.is_set)
        result = controller.stop()
        self.assertTrue(result["hold_confirmed"])
        self.assertIn(result["state"], {"stopping", "stopped"})
        counts = {name: len(arm.servo_calls) for name, arm in arms.items()}
        time.sleep(0.08)
        self.assertEqual(
            counts,
            {name: len(arm.servo_calls) for name, arm in arms.items()},
        )
        for arm in arms.values():
            steps = np.diff(np.asarray([[0.0] * 6, *arm.servo_calls]), axis=0)
            self.assertGreater(float(np.max(np.abs(steps))), 0.099)
        inference.release.set()
        wait_for(lambda: controller.status()["state"] == "stopped")
        controller.close()

    def test_later_model_frames_are_not_rate_limited(self) -> None:
        controller, arms = self.make_controller(
            inference=FakeInference(
                block_after=2,
                joint_delta=0.10,
                per_step_delta=0.04,
            ),
            execute_steps=2,
        )
        controller.start()
        inference = controller.inference
        wait_for(inference.blocked.is_set)
        self.assertNotEqual(controller.status()["state"], "error")
        for arm in arms.values():
            steps = np.diff(np.asarray([[0.0] * 6, *arm.servo_calls]), axis=0)
            self.assertGreater(float(np.max(np.abs(steps))), 0.039)
        controller.stop()
        inference.release.set()
        wait_for(lambda: controller.status()["state"] == "stopped")
        controller.close()

    def test_stop_during_inference_prevents_later_servo_send(self) -> None:
        inference = FakeInference(block_after=2)
        controller, arms = self.make_controller(inference=inference)
        controller.start()
        wait_for(inference.blocked.is_set)
        result = controller.stop()
        self.assertTrue(result["hold_confirmed"])
        counts = {name: len(arm.servo_calls) for name, arm in arms.items()}
        inference.release.set()
        wait_for(lambda: controller.status()["state"] == "stopped")
        self.assertEqual(
            counts,
            {name: len(arm.servo_calls) for name, arm in arms.items()},
        )
        controller.close()

    def test_large_start_pose_deviation_is_diagnostic_only(self) -> None:
        def checker(_state):
            return {
                "out_of_range": [
                    {
                        "index": 7,
                        "name": "right_joint_1",
                        "distance_to_range": 0.20,
                    }
                ]
            }

        inference = FakeInference(block_after=2)
        controller, arms = self.make_controller(
            checker=checker,
            inference=inference,
        )
        controller.start()
        wait_for(inference.blocked.is_set)
        status = controller.status()
        self.assertNotEqual(status["state"], "error")
        self.assertTrue(status["start_pose_diagnostic"]["diagnostic_only"])
        self.assertFalse(status["start_pose_diagnostic"]["blocking"])
        self.assertEqual(
            status["start_pose_diagnostic"]["outside_margin"][0]["name"],
            "right_joint_1",
        )
        for arm in arms.values():
            self.assertEqual(arm.mode, "SERVO_JOINT_POS")
            self.assertTrue(arm.gripper_calls)
        controller.stop()
        inference.release.set()
        wait_for(lambda: controller.status()["state"] == "stopped")
        controller.close()


if __name__ == "__main__":
    unittest.main()
