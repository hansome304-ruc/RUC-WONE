from __future__ import annotations

import threading
import unittest

from medicine_agentic.cartesian_jog import (
    CartesianJogConflict,
    CartesianJogController,
    CartesianJogSafetyViolation,
    CartesianJogUnavailable,
)


WORKSPACE = {
    "x_min": 0.10,
    "x_max": 0.60,
    "y_min": -0.40,
    "y_max": 0.40,
    "z_min": 0.08,
    "z_max": 0.60,
}


class FakeArm:
    def __init__(self) -> None:
        self.position = [0.30, 0.0, 0.25]
        self.quaternion = [0.0, 0.70710678, 0.0, 0.70710678]
        self.connected = False
        self.disconnect_count = 0
        self.mode = "SERVO_JOINT_POS"
        self.speed = None
        self.state = "IDLE"
        self.moves: list[tuple[list[list[float]], bool]] = []
        self.waypoint_moves: list[tuple[list[list[list[float]]], bool]] = []
        self.joint_positions = [0.2, -0.1, 0.3, 1.2, 0.1, -1.1]
        self.joint_moves: list[tuple[list[float], bool]] = []
        self.move_result = True
        self.follow_target = True
        self.speed_params = {}

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.connected = False
        self.disconnect_count += 1

    def get_end_pose(self):
        return [list(self.position), list(self.quaternion)]

    def get_state(self):
        return self.state

    def switch_mode(self, mode):
        self.mode = mode
        return True

    def set_speed_profile(self, speed):
        self.speed = speed
        self.speed_params = {
            "servo_node.moveit_servo.scale.linear": 0.05,
            "servo_node.moveit_servo.scale.rotational": 0.05,
            "servo_node.moveit_servo.scale.joint": 0.05,
            "sdk_server.max_velocity_scaling_factor": 0.1,
            "sdk_server.max_acceleration_scaling_factor": 0.02,
        }
        return True

    def get_params(self, names):
        return {name: self.speed_params[name] for name in names}

    def get_control_mode(self):
        return self.mode

    def move_to_cart_pose(self, target, *, blocking):
        self.moves.append((target, blocking))
        if self.move_result and self.follow_target:
            self.position = list(target[0])
            self.quaternion = list(target[1])
        return self.move_result

    def move_with_cart_waypoints(self, waypoints, *, blocking):
        self.waypoint_moves.append((waypoints, blocking))
        if self.move_result and self.follow_target:
            self.position = list(waypoints[-1][0])
            self.quaternion = list(waypoints[-1][1])
        return self.move_result

    def get_joint_pos(self):
        return list(self.joint_positions)

    def move_to_joint_pos(self, target, *, blocking):
        self.joint_moves.append((list(target), blocking))
        if self.move_result and self.follow_target:
            self.joint_positions = list(target)
        return self.move_result


class FakeFactory:
    def __init__(self, arm: FakeArm) -> None:
        self.arm = arm
        self.calls: list[tuple[str, int]] = []

    def __call__(self, host: str, port: int) -> FakeArm:
        self.calls.append((host, port))
        return self.arm


def controller(
    arm: FakeArm,
    *,
    dry_run: bool = True,
    teleop_running=lambda: False,
    workspace=WORKSPACE,
    capture_workspace=None,
    endpoint=None,
) -> tuple[CartesianJogController, FakeFactory]:
    factory = FakeFactory(arm)
    config = {
        "enabled": True,
        "dry_run": dry_run,
        "enable_token": "operator-secret",
        "workspace": workspace,
        "workspace_profile": "task1_folded_carton_pick",
        "sample_interval_s": 0.0,
        "max_position_error_m": 0.001,
        "max_downward_step_mm": 5,
        "feedback_timeout_s": 0.1,
        "restore_feedback_timeout_s": 0.1,
        "feedback_stable_samples": 3,
        "safe_vertical_pose": {
            "position_m": [0.30, 0.0, 0.25],
            "quaternion_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
            "transit_z_m": 0.30,
            "restore_token": "restore-secret",
            "rotation_steps": 4,
        },
        "home_joint_pose": {
            "enabled": True,
            "joint_positions_rad": [0.0, 0.0, 0.0, 1.5, 0.0, -1.5],
            "position_tolerance_rad": 0.12,
            "feedback_timeout_s": 2.0,
        },
    }
    if capture_workspace is not None:
        config["capture_workspace"] = capture_workspace
    if endpoint is not None:
        config["endpoint"] = endpoint
    instance = CartesianJogController(
        config,
        arm_factory=factory,
        teleop_running=teleop_running,
        planning_mode="PLANNING_POS",
        waypoint_mode="PLANNING_WAYPOINTS_PATH",
        slow_speed="SLOW",
    )
    return instance, factory


class CartesianJogTests(unittest.TestCase):
    def test_read_current_pose_only_reads_feedback(self) -> None:
        arm = FakeArm()
        jog, factory = controller(arm, dry_run=False)

        pose = jog.read_current_pose()

        self.assertEqual(factory.calls, [("localhost", 50051)])
        self.assertEqual(pose["position_m"], arm.position)
        for actual, expected in zip(pose["quaternion_xyzw"], arm.quaternion):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(pose["joint_positions_rad"], arm.joint_positions)
        self.assertTrue(pose["read_only"])
        self.assertEqual(arm.moves, [])
        self.assertEqual(arm.joint_moves, [])

    def test_explicit_feedback_read_is_allowed_during_teleop(self) -> None:
        arm = FakeArm()
        jog, factory = controller(
            arm,
            dry_run=False,
            teleop_running=lambda: True,
        )

        pose = jog.read_current_pose(allow_during_teleop=True)

        self.assertEqual(factory.calls, [("localhost", 50051)])
        self.assertTrue(pose["read_only"])
        self.assertTrue(pose["captured_during_teleop"])
        self.assertEqual(arm.moves, [])
        self.assertEqual(arm.joint_moves, [])

    def test_xy_workspace_and_capture_offsets_can_be_disabled(self) -> None:
        workspace = {
            **WORKSPACE,
            "enforce_xy": False,
            "x_min": 0.20,
            "x_max": 0.40,
            "y_min": -0.10,
            "y_max": 0.10,
        }
        arm = FakeArm()
        arm.position = [1.20, -1.10, 0.25]
        jog, _factory = controller(arm, workspace=workspace)

        jog.capture_orientation()
        arm.position = [1.80, -1.60, 0.25]
        jog.enable("dry", area_clear=True, estop_ready=True)
        result = jog.move_fixed_orientation_path(
            [[2.50, 2.00, 0.30]],
            operation="xy_unbounded_test",
        )

        self.assertTrue(result["dry_run"])
        self.assertFalse(jog.status()["xy_workspace_enforced"])
        self.assertFalse(jog.status()["xy_capture_offset_enforced"])
        with self.assertRaises(CartesianJogSafetyViolation):
            jog.move_fixed_orientation_path(
                [[2.50, 2.00, 0.01]],
                operation="z_floor_guard_test",
            )

    def test_constructor_and_status_never_connect(self) -> None:
        arm = FakeArm()
        jog, factory = controller(arm)
        status = jog.status()
        self.assertEqual(factory.calls, [])
        self.assertFalse(status["enabled"])
        self.assertEqual(status["endpoint"], {
            "arm": "left",
            "host": "localhost",
            "port": 50051,
        })
        self.assertTrue(status["dry_run"])
        self.assertEqual(
            status["workspace_profile"],
            "task1_folded_carton_pick",
        )
        self.assertEqual(status["workspace_m"], WORKSPACE)
        self.assertEqual(status["capture_workspace_m"], WORKSPACE)

    def test_capture_workspace_is_separate_from_motion_workspace(self) -> None:
        capture_workspace = {
            **WORKSPACE,
            "x_max": 0.75,
        }
        arm = FakeArm()
        arm.position[0] = 0.70
        jog, _factory = controller(
            arm,
            capture_workspace=capture_workspace,
        )

        captured = jog.capture_orientation()
        self.assertEqual(captured["position_m"], arm.position)
        status = jog.status()
        self.assertTrue(status["orientation_captured"])
        self.assertEqual(status["workspace_m"], WORKSPACE)
        self.assertEqual(
            status["capture_workspace_m"],
            capture_workspace,
        )

        with self.assertRaises(CartesianJogSafetyViolation):
            jog.enable("dry", area_clear=True, estop_ready=True)
        self.assertFalse(jog.status()["enabled"])

    def test_capture_at_zero_is_read_only_and_cannot_enable_motion(self) -> None:
        motion_workspace = {**WORKSPACE, "z_min": 0.02}
        capture_workspace = {**motion_workspace, "z_min": 0.0}
        arm = FakeArm()
        arm.position[2] = 0.0
        jog, _factory = controller(
            arm,
            dry_run=False,
            workspace=motion_workspace,
            capture_workspace=capture_workspace,
        )

        jog.capture_orientation()
        with self.assertRaises(CartesianJogSafetyViolation):
            jog.enable("operator-secret", area_clear=True, estop_ready=True)
        self.assertFalse(jog.status()["enabled"])
        self.assertEqual(arm.mode, "SERVO_JOINT_POS")
        self.assertIsNone(arm.speed)
        self.assertEqual(arm.moves, [])

    def test_motion_z_boundary_rejects_target_below_minimum_before_mode_switch(
        self,
    ) -> None:
        motion_workspace = {**WORKSPACE, "z_min": 0.02}
        capture_workspace = {**motion_workspace, "z_min": 0.0}
        arm = FakeArm()
        arm.position[2] = 0.02
        jog, _factory = controller(
            arm,
            dry_run=False,
            workspace=motion_workspace,
            capture_workspace=capture_workspace,
        )

        jog.capture_orientation()
        jog.enable("operator-secret", area_clear=True, estop_ready=True)
        self.assertEqual(arm.mode, "SERVO_JOINT_POS")
        self.assertIsNone(arm.speed)
        self.assertEqual(arm.moves, [])

        with self.assertRaises(CartesianJogSafetyViolation):
            jog.jog("z", -1)
        self.assertFalse(jog.status()["enabled"])
        self.assertEqual(arm.mode, "SERVO_JOINT_POS")
        self.assertIsNone(arm.speed)
        self.assertEqual(arm.moves, [])

    def test_enable_rejects_live_pose_moved_outside_motion_workspace(self) -> None:
        arm = FakeArm()
        jog, _factory = controller(arm)
        jog.capture_orientation()
        arm.position[0] = WORKSPACE["x_max"] + 0.02

        with self.assertRaises(CartesianJogSafetyViolation):
            jog.enable("dry", area_clear=True, estop_ready=True)

        status = jog.status()
        self.assertFalse(status["enabled"])
        self.assertFalse(status["orientation_captured"])
        self.assertEqual(arm.moves, [])
        self.assertIsNone(arm.speed)
        self.assertFalse(arm.connected)

    def test_enable_rejects_live_orientation_drift(self) -> None:
        arm = FakeArm()
        jog, _factory = controller(arm)
        jog.capture_orientation()
        arm.quaternion = [0.0, 0.0, 0.0, 1.0]

        with self.assertRaises(CartesianJogSafetyViolation):
            jog.enable("dry", area_clear=True, estop_ready=True)

        status = jog.status()
        self.assertFalse(status["enabled"])
        self.assertFalse(status["orientation_captured"])
        self.assertEqual(arm.moves, [])
        self.assertEqual(arm.mode, "SERVO_JOINT_POS")
        self.assertFalse(arm.connected)

    def test_capture_workspace_cannot_bypass_motion_workspace_during_jog(self) -> None:
        capture_workspace = {
            **WORKSPACE,
            "x_max": 0.75,
        }
        arm = FakeArm()
        jog, _factory = controller(
            arm,
            capture_workspace=capture_workspace,
        )
        jog.capture_orientation()
        jog.enable("dry", area_clear=True, estop_ready=True)

        # Simulate an out-of-band displacement that remains legal for capture
        # but is outside the narrower motion workspace.
        arm.position[0] = 0.70
        with self.assertRaises(CartesianJogSafetyViolation):
            jog.jog("x", -1)
        self.assertFalse(jog.status()["enabled"])

    def test_capture_rejects_position_outside_capture_workspace(self) -> None:
        arm = FakeArm()
        arm.position[0] = 0.70
        jog, _factory = controller(arm)
        with self.assertRaises(CartesianJogSafetyViolation):
            jog.capture_orientation()
        self.assertFalse(jog.status()["orientation_captured"])

    def test_absent_config_is_fail_closed(self) -> None:
        arm = FakeArm()
        factory = FakeFactory(arm)
        jog = CartesianJogController(arm_factory=factory)
        self.assertFalse(jog.status()["available"])
        with self.assertRaises(CartesianJogUnavailable):
            jog.capture_orientation()
        self.assertEqual(factory.calls, [])

    def test_capture_records_stable_pose_origin_and_releases_arm(self) -> None:
        arm = FakeArm()
        jog, factory = controller(arm)
        result = jog.capture_orientation()
        self.assertEqual(result["operation"], "capture_orientation")
        self.assertEqual(result["position_m"], arm.position)
        self.assertEqual(result["position_spread_m"], 0.0)
        self.assertEqual(factory.calls, [("localhost", 50051)])
        self.assertFalse(arm.connected)
        self.assertEqual(arm.disconnect_count, 1)
        self.assertTrue(jog.status()["orientation_captured"])
        self.assertEqual(jog.status()["current_position_m"], arm.position)
        self.assertFalse(jog.status()["enabled"])

    def test_unstable_orientation_capture_fails_and_disables(self) -> None:
        arm = FakeArm()
        jog, _factory = controller(arm)
        jog.capture_orientation()
        self.assertTrue(jog.status()["orientation_captured"])
        samples = iter(
            [
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0.02, 0, 0.9998],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ]
        )

        def get_pose():
            return [list(arm.position), next(samples)]

        arm.get_end_pose = get_pose
        with self.assertRaises(CartesianJogSafetyViolation):
            jog.capture_orientation()
        self.assertFalse(jog.status()["enabled"])
        self.assertFalse(jog.status()["orientation_captured"])
        self.assertFalse(arm.connected)

    def test_capture_requires_idle_arm_state(self) -> None:
        arm = FakeArm()
        arm.state = "MOVING"
        jog, _factory = controller(arm)
        with self.assertRaises(CartesianJogSafetyViolation):
            jog.capture_orientation()
        status = jog.status()
        self.assertFalse(status["orientation_captured"])
        self.assertIn("IDLE", status["last_error"])

    def test_capture_rejects_position_motion(self) -> None:
        arm = FakeArm()
        jog, _factory = controller(arm)
        positions = iter(
            [
                [0.30, 0.0, 0.25],
                [0.30, 0.0, 0.25],
                [0.302, 0.0, 0.25],
                [0.30, 0.0, 0.25],
                [0.30, 0.0, 0.25],
            ]
        )

        def get_pose():
            return [next(positions), list(arm.quaternion)]

        arm.get_end_pose = get_pose
        with self.assertRaises(CartesianJogSafetyViolation):
            jog.capture_orientation()
        self.assertFalse(jog.status()["orientation_captured"])

    def test_enable_requires_capture_token_and_confirmations(self) -> None:
        arm = FakeArm()
        jog, _factory = controller(arm, dry_run=False)
        with self.assertRaises(CartesianJogSafetyViolation):
            jog.enable(
                "operator-secret",
                area_clear=True,
                estop_ready=True,
            )
        jog.capture_orientation()
        with self.assertRaises(CartesianJogSafetyViolation):
            jog.enable("wrong", area_clear=True, estop_ready=True)
        jog.capture_orientation()
        with self.assertRaises(CartesianJogSafetyViolation):
            jog.enable(
                "operator-secret",
                area_clear=False,
                estop_ready=True,
            )
        jog.capture_orientation()
        result = jog.enable(
            "operator-secret",
            area_clear=True,
            estop_ready=True,
        )
        self.assertTrue(result["enabled"])
        self.assertFalse(jog.disable()["enabled"])

    def test_dry_run_uses_live_position_and_never_commands_motion(self) -> None:
        arm = FakeArm()
        jog, _factory = controller(arm)
        jog.capture_orientation()
        jog.enable("dry-confirmation", area_clear=True, estop_ready=True)
        arm.position = [0.31, -0.02, 0.24]
        result = jog.jog("x", 5)
        self.assertFalse(result["executed"])
        self.assertEqual(result["current_position_m"], [0.31, -0.02, 0.24])
        self.assertAlmostEqual(result["target_position_m"][0], 0.315)
        self.assertEqual(arm.moves, [])
        self.assertIsNone(arm.speed)
        self.assertFalse(arm.connected)

    def test_real_jog_locks_orientation_mode_speed_and_blocking(self) -> None:
        arm = FakeArm()
        jog, _factory = controller(arm, dry_run=False)
        locked = jog.capture_orientation()["quaternion_xyzw"]
        jog.enable("operator-secret", area_clear=True, estop_ready=True)
        result = jog.jog("y", -2)
        self.assertTrue(result["executed"])
        self.assertEqual(arm.mode, "PLANNING_POS")
        self.assertEqual(arm.speed, "SLOW")
        self.assertEqual(len(arm.moves), 1)
        target, blocking = arm.moves[0]
        self.assertTrue(blocking)
        self.assertEqual(target[1], locked)
        self.assertAlmostEqual(target[0][1], -0.002)
        self.assertFalse(arm.connected)

    def test_fixed_orientation_path_executes_absolute_targets_in_order(self) -> None:
        arm = FakeArm()
        jog, _factory = controller(arm, dry_run=False)
        locked = jog.capture_orientation()["quaternion_xyzw"]
        jog.enable("operator-secret", area_clear=True, estop_ready=True)
        targets = [
            [0.30, 0.0, 0.30],
            [0.36, -0.04, 0.30],
            [0.36, -0.04, 0.12],
        ]

        result = jog.move_fixed_orientation_path(
            targets,
            operation="task1_approach",
        )

        self.assertTrue(result["executed"])
        self.assertEqual(result["operation"], "task1_approach")
        self.assertEqual(result["completed_targets_m"], targets)
        self.assertEqual(len(arm.moves), 3)
        self.assertEqual([move[0][0] for move in arm.moves], targets)
        self.assertTrue(all(move[0][1] == locked for move in arm.moves))
        self.assertTrue(all(move[1] for move in arm.moves))
        self.assertEqual(arm.position, targets[-1])
        self.assertEqual(jog.status()["current_position_m"], targets[-1])
        self.assertFalse(arm.connected)

    def test_restore_safe_vertical_uses_one_bounded_waypoint_path(self) -> None:
        arm = FakeArm()
        arm.position = [0.32, -0.02, 0.20]
        arm.quaternion = [0.0, 0.6755902, 0.0, 0.7372773]
        jog, _factory = controller(arm, dry_run=False)

        result = jog.restore_safe_vertical(
            "restore-secret",
            area_clear=True,
            estop_ready=True,
            suction_released=True,
        )

        self.assertTrue(result["executed"])
        self.assertEqual(len(arm.waypoint_moves), 1)
        waypoints, blocking = arm.waypoint_moves[0]
        self.assertTrue(blocking)
        self.assertEqual(waypoints[0][0], [0.32, -0.02, 0.20])
        self.assertEqual(waypoints[-1][0], [0.30, 0.0, 0.25])
        self.assertEqual(
            waypoints[-1][1],
            [0.0, 0.7071067811865476, 0.0, 0.7071067811865476],
        )
        self.assertGreaterEqual(max(point[0][2] for point in waypoints), 0.30)
        self.assertEqual(arm.mode, "PLANNING_POS")
        self.assertEqual(arm.speed, "SLOW")
        status = jog.status()
        self.assertFalse(status["enabled"])
        self.assertTrue(status["orientation_captured"])
        self.assertEqual(status["current_position_m"], [0.30, 0.0, 0.25])
        self.assertFalse(arm.connected)

    def test_restore_safe_vertical_recovers_from_previous_jog_error(self) -> None:
        arm = FakeArm()
        jog, _factory = controller(arm, dry_run=False)
        jog.capture_orientation()
        jog.enable("operator-secret", area_clear=True, estop_ready=True)
        arm.follow_target = False
        with self.assertRaises(CartesianJogSafetyViolation):
            jog.jog("x", 2)
        self.assertEqual(jog.status()["state"], "idle")
        self.assertTrue(jog.status()["last_error"])

        arm.follow_target = True
        result = jog.restore_safe_vertical(
            "restore-secret",
            area_clear=True,
            estop_ready=True,
            suction_released=True,
        )
        self.assertTrue(result["executed"])
        self.assertEqual(jog.status()["state"], "captured")

    def test_restore_safe_vertical_requires_all_confirmations(self) -> None:
        for field in ("area_clear", "estop_ready", "suction_released"):
            arm = FakeArm()
            jog, _factory = controller(arm, dry_run=False)
            confirmations = {
                "area_clear": True,
                "estop_ready": True,
                "suction_released": True,
            }
            confirmations[field] = False
            with self.subTest(field=field):
                with self.assertRaises(CartesianJogSafetyViolation):
                    jog.restore_safe_vertical(
                        "restore-secret",
                        **confirmations,
                    )
                self.assertEqual(arm.waypoint_moves, [])

    def test_restore_safe_vertical_rejects_pose_outside_motion_workspace(self) -> None:
        arm = FakeArm()
        arm.position = [0.65, 0.0, 0.25]
        jog, _factory = controller(arm, dry_run=False)
        with self.assertRaises(CartesianJogSafetyViolation):
            jog.restore_safe_vertical(
                "restore-secret",
                area_clear=True,
                estop_ready=True,
                suction_released=True,
            )
        self.assertEqual(arm.waypoint_moves, [])

    def test_restore_safe_vertical_allows_large_orientation_change(self) -> None:
        arm = FakeArm()
        arm.quaternion = [0.0, 0.0, 0.0, 1.0]
        jog, _factory = controller(arm, dry_run=False)
        result = jog.restore_safe_vertical(
            "restore-secret",
            area_clear=True,
            estop_ready=True,
            suction_released=True,
        )
        self.assertTrue(result["executed"])
        self.assertEqual(len(arm.waypoint_moves), 1)

    def test_restore_feedback_failure_never_sends_second_motion(self) -> None:
        arm = FakeArm()
        arm.position = [0.31, 0.0, 0.25]
        arm.follow_target = False
        jog, _factory = controller(arm, dry_run=False)
        with self.assertRaises(CartesianJogSafetyViolation):
            jog.restore_safe_vertical(
                "restore-secret",
                area_clear=True,
                estop_ready=True,
                suction_released=True,
            )
        self.assertEqual(len(arm.waypoint_moves), 1)
        self.assertEqual(arm.mode, "PLANNING_WAYPOINTS_PATH")
        self.assertFalse(jog.status()["orientation_captured"])

    def test_restore_dry_run_never_moves_or_captures(self) -> None:
        arm = FakeArm()
        arm.position = [0.31, 0.0, 0.25]
        jog, _factory = controller(arm, dry_run=True)
        result = jog.restore_safe_vertical(
            "restore-secret",
            area_clear=True,
            estop_ready=True,
            suction_released=True,
        )
        self.assertFalse(result["executed"])
        self.assertEqual(arm.waypoint_moves, [])
        self.assertFalse(jog.status()["orientation_captured"])

    def test_reset_home_moves_only_left_endpoint_to_shared_home_joints(self) -> None:
        arm = FakeArm()
        jog, factory = controller(arm, dry_run=False)
        jog.capture_orientation()
        jog.enable("operator-secret", area_clear=True, estop_ready=True)

        result = jog.reset_home()

        target = [0.0, 0.0, 0.0, 1.5, 0.0, -1.5]
        self.assertTrue(result["executed"])
        self.assertEqual(result["arm"], "left")
        self.assertEqual(result["target_joint_positions_rad"], target)
        self.assertEqual(result["actual_joint_positions_rad"], target)
        self.assertEqual(arm.joint_moves, [(target, True)])
        self.assertEqual(arm.mode, "PLANNING_POS")
        self.assertEqual(arm.speed, "SLOW")
        self.assertTrue(all(call == ("localhost", 50051) for call in factory.calls))
        status = jog.status()
        self.assertFalse(status["enabled"])
        self.assertFalse(status["orientation_captured"])
        self.assertEqual(
            status["home_joint_pose"]["joint_positions_rad"],
            target,
        )
        self.assertFalse(arm.connected)

    def test_reset_home_can_target_only_right_endpoint(self) -> None:
        arm = FakeArm()
        jog, factory = controller(
            arm,
            dry_run=False,
            endpoint={"arm": "right", "host": "localhost", "port": 50053},
        )

        result = jog.reset_home()

        target = [0.0, 0.0, 0.0, 1.5, 0.0, -1.5]
        self.assertTrue(result["executed"])
        self.assertEqual(result["arm"], "right")
        self.assertEqual(arm.joint_moves, [(target, True)])
        self.assertTrue(all(call == ("localhost", 50053) for call in factory.calls))
        self.assertEqual(jog.status()["endpoint"]["arm"], "right")
        self.assertEqual(jog.status()["home_joint_pose"]["affects"], ["right_arm"])

    def test_reset_home_dry_run_never_sends_joint_motion(self) -> None:
        arm = FakeArm()
        jog, _factory = controller(arm, dry_run=True)

        result = jog.reset_home()

        self.assertFalse(result["executed"])
        self.assertEqual(arm.joint_moves, [])
        self.assertIsNone(arm.speed)
        self.assertFalse(arm.connected)

    def test_reset_home_is_blocked_while_teleop_runs(self) -> None:
        arm = FakeArm()
        jog, factory = controller(
            arm,
            dry_run=False,
            teleop_running=lambda: True,
        )

        with self.assertRaises(CartesianJogConflict):
            jog.reset_home()

        self.assertEqual(factory.calls, [])
        self.assertEqual(arm.joint_moves, [])

    def test_saved_pose_moves_to_recorded_joint_target(self) -> None:
        arm = FakeArm()
        jog, factory = controller(arm, dry_run=False)
        target = [0.1, -0.2, 0.3, 1.4, -0.1, -1.3]

        result = jog.move_to_saved_joint_pose(target, pose_name="left_watcher")

        self.assertTrue(result["executed"])
        self.assertEqual(result["pose_name"], "left_watcher")
        self.assertEqual(result["target_joint_positions_rad"], target)
        self.assertEqual(arm.joint_moves, [(target, True)])
        self.assertTrue(all(call == ("localhost", 50051) for call in factory.calls))

    def test_saved_pose_is_blocked_while_teleop_runs(self) -> None:
        arm = FakeArm()
        jog, factory = controller(
            arm,
            dry_run=False,
            teleop_running=lambda: True,
        )
        with self.assertRaises(CartesianJogConflict):
            jog.move_to_saved_joint_pose([0.0] * 6, pose_name="blocked")
        self.assertEqual(factory.calls, [])
        self.assertEqual(arm.joint_moves, [])

    def test_invalid_steps_downward_limit_and_workspace_fail_closed(self) -> None:
        for axis, step in (("q", 1), ("x", 3), ("x", 0)):
            arm = FakeArm()
            jog, _factory = controller(arm)
            jog.capture_orientation()
            jog.enable("dry", area_clear=True, estop_ready=True)
            with self.subTest(axis=axis, step=step):
                with self.assertRaises(CartesianJogSafetyViolation):
                    jog.jog(axis, step)
                self.assertFalse(jog.status()["enabled"])

        arm = FakeArm()
        arm.position[0] = WORKSPACE["x_max"] - 0.001
        jog, _factory = controller(arm)
        jog.capture_orientation()
        jog.enable("dry", area_clear=True, estop_ready=True)
        with self.assertRaises(CartesianJogSafetyViolation):
            jog.jog("x", 2)
        self.assertFalse(jog.status()["enabled"])

    def test_downward_five_mm_is_allowed_above_workspace_floor(self) -> None:
        arm = FakeArm()
        jog, _factory = controller(arm)
        jog.capture_orientation()
        jog.enable("dry", area_clear=True, estop_ready=True)

        result = jog.jog("z", -5)

        self.assertFalse(result["executed"])
        self.assertAlmostEqual(result["target_position_m"][2], 0.245)
        self.assertTrue(jog.status()["enabled"])

    def test_ten_mm_fast_step_is_allowed_except_downward_z(self) -> None:
        arm = FakeArm()
        jog, _factory = controller(arm)
        jog.capture_orientation()
        jog.enable("dry", area_clear=True, estop_ready=True)

        result = jog.jog("x", 10)

        self.assertFalse(result["executed"])
        self.assertAlmostEqual(result["target_position_m"][0], 0.31)
        self.assertIn(10, jog.status()["allowed_step_mm"])

        jog.capture_orientation()
        jog.enable("dry", area_clear=True, estop_ready=True)
        with self.assertRaises(CartesianJogSafetyViolation):
            jog.jog("z", -10)
        self.assertFalse(jog.status()["enabled"])

    def test_cumulative_offset_from_capture_is_bounded(self) -> None:
        arm = FakeArm()
        jog, _factory = controller(
            arm,
            workspace=WORKSPACE,
        )
        jog.capture_orientation()
        jog.enable("dry", area_clear=True, estop_ready=True)
        arm.position[0] += 0.151
        with self.assertRaises(CartesianJogSafetyViolation):
            jog.jog("x", 1)
        self.assertFalse(jog.status()["enabled"])

    def test_teleop_interlock_blocks_capture_enable_and_jog(self) -> None:
        running = {"value": True}
        arm = FakeArm()
        jog, factory = controller(
            arm,
            teleop_running=lambda: running["value"],
        )
        with self.assertRaises(CartesianJogConflict):
            jog.capture_orientation()
        self.assertEqual(factory.calls, [])
        running["value"] = False
        jog.capture_orientation()
        running["value"] = True
        with self.assertRaises(CartesianJogConflict):
            jog.enable("dry", area_clear=True, estop_ready=True)
        running["value"] = False
        jog.capture_orientation()
        jog.enable("dry", area_clear=True, estop_ready=True)
        running["value"] = True
        with self.assertRaises(CartesianJogConflict):
            jog.jog("x", 1)
        self.assertFalse(jog.status()["enabled"])

    def test_post_motion_error_fails_and_disables(self) -> None:
        arm = FakeArm()
        arm.follow_target = False
        jog, _factory = controller(arm, dry_run=False)
        jog.capture_orientation()
        jog.enable("operator-secret", area_clear=True, estop_ready=True)
        with self.assertRaises(CartesianJogSafetyViolation):
            jog.jog("x", 2)
        status = jog.status()
        self.assertFalse(status["enabled"])
        self.assertFalse(status["orientation_captured"])
        self.assertFalse(arm.connected)

    def test_orientation_drift_is_rejected_before_mode_switch(self) -> None:
        arm = FakeArm()
        jog, _factory = controller(arm, dry_run=False)
        jog.capture_orientation()
        jog.enable("operator-secret", area_clear=True, estop_ready=True)
        arm.quaternion = [0, 0, 0, 1]
        with self.assertRaises(CartesianJogSafetyViolation):
            jog.jog("x", 1)
        self.assertEqual(arm.mode, "SERVO_JOINT_POS")
        self.assertEqual(arm.moves, [])
        status = jog.status()
        self.assertFalse(status["enabled"])
        self.assertFalse(status["orientation_captured"])

    def test_busy_lock_serializes_actions(self) -> None:
        arm = FakeArm()
        jog, _factory = controller(arm)
        entered = threading.Event()
        release = threading.Event()
        original = arm.get_end_pose

        def blocked_pose():
            entered.set()
            release.wait(timeout=1.0)
            return original()

        arm.get_end_pose = blocked_pose
        errors: list[Exception] = []

        def capture():
            try:
                jog.capture_orientation()
            except Exception as exc:  # pragma: no cover - diagnostic only
                errors.append(exc)

        thread = threading.Thread(target=capture)
        thread.start()
        self.assertTrue(entered.wait(timeout=1.0))
        with self.assertRaises(CartesianJogConflict):
            jog.capture_orientation()
        release.set()
        thread.join(timeout=1.0)
        self.assertEqual(errors, [])
        self.assertFalse(jog.status()["busy"])

    def test_close_clears_orientation_and_enable(self) -> None:
        arm = FakeArm()
        jog, _factory = controller(arm)
        jog.capture_orientation()
        jog.enable("dry", area_clear=True, estop_ready=True)
        jog.close()
        status = jog.status()
        self.assertFalse(status["enabled"])
        self.assertFalse(status["orientation_captured"])


if __name__ == "__main__":
    unittest.main()
