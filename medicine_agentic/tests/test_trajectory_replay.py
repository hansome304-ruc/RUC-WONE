from __future__ import annotations

import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path

from medicine_agentic.trajectory_replay import TrajectoryReplay


def jsonl(rows):
    return "".join(json.dumps(row) + "\n" for row in rows)


class FakeArm:
    def __init__(self) -> None:
        self.joints = [-0.4] * 6
        self.mode = "IDLE"
        self.commands = 0
        self.planning_commands = 0
        self.gripper = 0.0
        self.gripper_planning_commands = 0
        self.gripper_commands = 0
        self.speed_profiles = []
        self.events = []

    def connect(self):
        return True

    def disconnect(self):
        return None

    def get_state(self):
        return "IDLE"

    def get_joint_pos(self):
        return list(self.joints)

    def switch_mode(self, mode):
        self.mode = mode
        self.events.append(("mode", mode))
        return True

    def get_control_mode(self):
        return self.mode

    def move_to_joint_pos(self, joints, *, blocking=False):
        self.joints = list(joints)
        self.planning_commands += 1
        self.events.append(("position", list(joints), blocking))
        return True

    def servo_joint_pos(self, joints):
        self.joints = list(joints)
        self.commands += 1
        self.events.append(("servo", list(joints)))
        return True

    def set_speed_profile(self, profile):
        self.speed_profiles.append(profile)
        self.events.append(("speed_profile", profile))

    def move_eef_pos(self, target, *, blocking=False):
        self.gripper = float(target)
        self.gripper_planning_commands += 1
        self.events.append(("position_gripper", float(target), blocking))
        return True

    def servo_eef_pos(self, target):
        self.gripper = float(target)
        self.gripper_commands += 1
        self.events.append(("servo_gripper", float(target)))
        return True

    def get_eef_pos(self):
        return [self.gripper]


class LaggingGripperArm(FakeArm):
    """Accept gripper commands but keep feedback fixed for limit tests."""

    def move_eef_pos(self, target, *, blocking=False):
        self.gripper_planning_commands += 1
        self.events.append(("position_gripper", float(target), blocking))
        return True

    def servo_eef_pos(self, target):
        self.gripper_commands += 1
        self.events.append(("servo_gripper", float(target)))
        return True


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.episode_root = self.root / "finalized"
        self.recording_id = "act_20260806_test_12345678"
        self.episode = self.episode_root / self.recording_id
        self.episode.mkdir(parents=True)
        (self.episode / "observations").mkdir()
        (self.episode / "actions").mkdir()
        (self.episode / "sensors").mkdir()
        (self.episode / "aligned").mkdir()
        arms = ("left", "right")
        arm_rows = [
            {"timestamp": 100.0, "joint_positions": [0.0] * 6, "gripper": 0.0},
            {"timestamp": 100.1, "joint_positions": [0.01] * 6, "gripper": 0.0},
        ]
        for group in ("observations", "actions"):
            for arm in arms:
                (self.episode / group / f"{arm}_arm.jsonl").write_text(
                    jsonl(arm_rows), encoding="utf-8"
                )
        camera_names = ("front", "left_wrist", "right_wrist")
        image_files = {}
        for name in camera_names:
            video = f"sensors/cam_{name}_rgb.mp4"
            timestamps = f"sensors/cam_{name}_rgb.mp4.tsf"
            frames = f"sensors/cam_{name}_frames.jsonl"
            (self.episode / video).write_bytes(b"")
            (self.episode / timestamps).write_bytes(b"")
            camera_offset = {
                "front": -0.001,
                "left_wrist": 0.0,
                "right_wrist": 0.001,
            }[name]
            (self.episode / frames).write_text(
                jsonl(
                    [
                        {
                            "index": index,
                            "captured_at": timestamp + camera_offset,
                            "device_timestamp_ms": (timestamp + camera_offset) * 1000.0,
                            "sync_timestamp_ms": (timestamp + camera_offset) * 1000.0,
                            "timestamp_domain": "timestamp_domain.global_time",
                            "timestamp_source": "device_global_time",
                            "sync_bundle_id": index + 1,
                        }
                        for index, timestamp in enumerate((100.0, 100.1))
                    ]
                ),
                encoding="utf-8",
            )
            image_files[name] = {
                "video": video,
                "timestamps": timestamps,
                "frame_metadata": frames,
            }
        aligned = []
        for index, timestamp in enumerate((100.0, 100.1)):
            def aligned_arm_sample(*, joint_offset=0.0):
                return {
                    "joint_positions": [joint_offset + 0.01 * index] * 6,
                    "gripper": 0.01 + 0.005 * index,
                    "source_indices": [0, 1],
                    "source_timestamps": [timestamp - 0.01, timestamp + 0.01],
                    "interpolation_alpha": 0.5,
                    "nearest_delta_ms": 10.0,
                    "source_gap_ms": 20.0,
                }

            aligned.append(
                {
                    "index": index,
                    "camera_frame_index": index,
                    "sync_bundle_id": index + 1,
                    "timestamp": timestamp,
                    "camera_timestamps": {
                        "front": timestamp - 0.001,
                        "left_wrist": timestamp,
                        "right_wrist": timestamp + 0.001,
                    },
                    "action": {
                        arm: aligned_arm_sample(joint_offset=0.2)
                        for arm in arms
                    },
                    "observation": {
                        arm: aligned_arm_sample()
                        for arm in arms
                    },
                }
            )
        (self.episode / "aligned/samples.jsonl").write_text(
            jsonl(aligned), encoding="utf-8"
        )
        meta = {
            "version": "medicine_act_episode_v1",
            "recording_id": self.recording_id,
            "label": "test",
            "purpose": "act_bimanual",
            "selected_arms": list(arms),
            "status": "completed",
            "frame_count": 2,
            "camera_names": list(camera_names),
            "camera_frame_counts": {name: 2 for name in camera_names},
            "observation_sample_counts": {arm: 2 for arm in arms},
            "action_sample_counts": {arm: 2 for arm in arms},
            "files": {
                "observations": {arm: f"observations/{arm}_arm.jsonl" for arm in arms},
                "actions": {arm: f"actions/{arm}_arm.jsonl" for arm in arms},
                "images": image_files,
                "aligned_samples": "aligned/samples.jsonl",
            },
            "act": {
                "synchronization": {"max_allowed_skew_ms": 25.0},
                "training_alignment": {
                    "timestamp_basis": "device_global_time",
                    "aligned_sample_count": 2,
                    "max_allowed_arm_sample_gap_ms": 60.0,
                    "max_allowed_camera_arm_delta_ms": 30.0,
                },
            },
        }
        (self.episode / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        manifest = []
        for path in sorted(self.episode.rglob("*")):
            if path.is_file():
                relative = path.relative_to(self.episode).as_posix()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                manifest.append(f"{digest}  {relative}\n")
        (self.episode / "checksums.sha256").write_text("".join(manifest), encoding="utf-8")
        (self.episode / "READY").write_text("{}", encoding="utf-8")
        self.fake_arms = []

        class LaggingRightArm(FakeArm):
            def servo_joint_pos(self, joints):
                self.commands += 1
                self.events.append(("servo", list(joints)))
                return True

        def factory(_host, port):
            arm = LaggingRightArm() if port == 50053 else FakeArm()
            self.fake_arms.append(arm)
            return arm

        self.replay = TrajectoryReplay(
            {
                "enabled": True,
                "episode_root": str(self.episode_root),
                "official_episode_root": str(self.root / "eps"),
                "calibration_episode_root": str(self.root / "trajectories"),
                "command_hz": 50,
                "initial_feedback_poll_s": 0.01,
                "replay_gripper": True,
                "gripper_replay_arms": ["right"],
                "gripper_feedback_every_n": 1,
                "max_gripper_tracking_error_m": 0.012,
                "gripper_scale": 2.0,
                "gripper_min_m": 0.0,
                "gripper_max_m": 0.072,
            },
            config_dir=self.root,
            interlock=lambda: None,
            arm_factory=factory,
            servo_mode="SERVO_JOINT_POS",
            planning_mode="PLANNING_POS",
            initial_speed_profile="SLOW",
            speed_profile="FAST",
        )

    def tearDown(self):
        self.replay.close()
        self.temporary.cleanup()

    def test_configuration_cannot_enable_left_gripper_replay(self):
        with self.assertRaisesRegex(ValueError, "must be exactly.*right"):
            TrajectoryReplay(
                {
                    "enabled": True,
                    "gripper_replay_arms": ["left", "right"],
                },
                config_dir=self.root,
                interlock=lambda: None,
                arm_factory=lambda _host, _port: FakeArm(),
            )

    def test_fast_odd_frame_profile_releases_suction_once(self):
        recording_id = "act_20260815_044616_zhuangxiang_ba97f7be"
        episode = self.root / "eps" / "20260815" / "episode_0001"
        (episode / "actions").mkdir(parents=True)
        for arm in ("left", "right"):
            rows = [
                {
                    "timestamp": 200.0 + 0.1 * index,
                    "joint_positions": [0.2 * index] * 6,
                    "gripper": 0.01 + (0.001 * index if arm == "right" else 0.0),
                }
                for index in range(7)
            ]
            (episode / f"actions/{arm}_arm.jsonl").write_text(
                jsonl(rows), encoding="utf-8"
            )
        (episode / "meta.json").write_text(
            json.dumps(
                {
                    "version": "v0.1",
                    "created_at": 200.0,
                    "finished_at": 200.5,
                    "task_meta": {"task_name": "act_bimanual"},
                    "actions": {"left_arm": {}, "right_arm": {}},
                    "extra": {
                        "recording_id": recording_id,
                        "purpose": "act_bimanual",
                        "recording_strategy": "official_open_pdd_direct_rgbd_v1",
                    },
                }
            ),
            encoding="utf-8",
        )
        fake_arms = []

        def factory(_host, _port):
            arm = FakeArm() if not fake_arms else LaggingGripperArm()
            fake_arms.append(arm)
            return arm

        suction_state = {"engaged": True}
        suction_calls = []

        def suction_status():
            return {"available": True, "engaged": suction_state["engaged"]}

        def suction_setter(engaged):
            suction_calls.append(engaged)
            suction_state["engaged"] = engaged
            return {"engaged": engaged}

        interlock_contexts = []

        def interlock(*, allow_suction_engaged=False):
            interlock_contexts.append(allow_suction_engaged)
            return None

        replay = TrajectoryReplay(
            {
                "enabled": True,
                "episode_root": str(self.episode_root),
                "official_episode_root": str(self.root / "eps"),
                "calibration_episode_root": str(self.root / "trajectories"),
                "command_hz": 50,
                "initial_feedback_poll_s": 0.01,
                "replay_gripper": True,
                "gripper_replay_arms": ["right"],
                "gripper_feedback_every_n": 1,
                "max_gripper_tracking_error_m": 0.012,
                "max_recorded_joint_step_rad": 1.0,
                "gripper_scale": 1.0,
                "gripper_min_m": 0.0,
                "gripper_max_m": 0.072,
                "recording_profiles": {
                    recording_id: {
                        "retain_frame_numbers_1_based": "odd",
                        "playback_speed_scale": 2.0,
                        "first_retained_frame_1_based": 3,
                        "suction_release_frame_1_based": 5,
                        "max_tracking_error_rad": None,
                        "max_gripper_tracking_error_m": None,
                    }
                },
            },
            config_dir=self.root,
            interlock=interlock,
            arm_factory=factory,
            servo_mode="SERVO_JOINT_POS",
            planning_mode="PLANNING_POS",
            initial_speed_profile="SLOW",
            speed_profile="FAST",
            suction_status=suction_status,
            suction_setter=suction_setter,
        )
        try:
            preflight = replay.preflight(recording_id)["preflight"]
            self.assertEqual(preflight["source_sample_count"], 7)
            self.assertEqual(preflight["sample_count"], 3)
            self.assertEqual(preflight["retained_frame_stride"], 2)
            self.assertEqual(preflight["retained_frame_parity"], "odd")
            self.assertEqual(preflight["speed_scale"], 2.0)
            self.assertEqual(preflight["first_source_frame_1_based"], 3)
            self.assertEqual(
                preflight["discarded_leading_source_frame_count"], 2
            )
            self.assertAlmostEqual(preflight["recorded_duration_s"], 0.6)
            self.assertAlmostEqual(preflight["default_replay_duration_s"], 0.2)
            self.assertTrue(preflight["suction_replayed"])
            self.assertEqual(preflight["suction_release_frame_1_based"], 5)
            self.assertIsNone(preflight["max_tracking_error_rad"])
            self.assertIsNone(preflight["max_gripper_tracking_error_m"])

            replay.start(recording_id=recording_id, confirmation=recording_id)
            completed = replay.wait(recording_id, timeout_s=2.0)
            self.assertEqual(completed["state"], "completed")
            self.assertIsNone(completed["max_tracking_error_rad"])
            self.assertIsNone(completed["max_gripper_tracking_error_m"])
            self.assertEqual(completed["source_sample_count"], 7)
            self.assertEqual(completed["sample_count"], 3)
            self.assertEqual(completed["current_source_frame_1_based"], 7)
            self.assertEqual(completed["suction_release_state"], "released")
            self.assertIsNotNone(completed["suction_released_at"])
            self.assertEqual(suction_calls, [False])
            self.assertFalse(suction_state["engaged"])
            self.assertIn(True, interlock_contexts)
            self.assertEqual(fake_arms[0].gripper_commands, 0)
            self.assertGreater(fake_arms[1].gripper_commands, 0)
            self.assertGreater(completed["gripper_max_tracking_error_m"], 0.012)
        finally:
            replay.close()

    def test_preflight_and_replay(self):
        preflight = self.replay.preflight(self.recording_id)["preflight"]
        self.assertEqual(preflight["sample_count"], 2)
        self.assertEqual(preflight["speed_scale"], 1.0)
        self.assertFalse(preflight["requires_near_start_pose"])
        self.assertTrue(preflight["automatically_positions_at_first_frame"])
        self.assertTrue(preflight["gripper_replayed"])
        self.assertEqual(preflight["gripper_replay_arms"], ["right"])
        self.assertTrue(preflight["left_gripper_commands_forbidden"])
        self.assertEqual(preflight["gripper_scale"], 2.0)
        started = self.replay.start(
            recording_id=self.recording_id,
            confirmation=self.recording_id,
            max_tracking_error_rad=0.5,
        )
        self.assertIn(started["state"], {"starting", "replaying"})
        deadline = time.monotonic() + 2.0
        while self.replay.status()["active"] and time.monotonic() < deadline:
            time.sleep(0.01)
        status = self.replay.status()
        self.assertEqual(status["state"], "completed")
        self.assertEqual(status["speed_scale"], 1.0)
        self.assertEqual(self.replay.max_tracking_error_rad, 0.35)
        self.assertEqual(status["max_tracking_error_rad"], 0.5)
        self.assertEqual(len(self.fake_arms), 2)
        self.assertTrue(all(arm.planning_commands == 1 for arm in self.fake_arms))
        self.assertTrue(all(arm.commands > 0 for arm in self.fake_arms))
        left_arm, right_arm = self.fake_arms
        self.assertEqual(left_arm.gripper_planning_commands, 0)
        self.assertEqual(left_arm.gripper_commands, 0)
        self.assertEqual(right_arm.gripper_planning_commands, 1)
        self.assertGreater(right_arm.gripper_commands, 0)
        self.assertEqual(status["gripper_replay_arms"], ["right"])
        self.assertTrue(status["left_gripper_commands_forbidden"])
        self.assertLessEqual(status["gripper_max_tracking_error_m"], 0.012)
        self.assertTrue(
            all(arm.speed_profiles == ["SLOW", "FAST"] for arm in self.fake_arms)
        )
        for index, arm in enumerate(self.fake_arms):
            slow_index = arm.events.index(("speed_profile", "SLOW"))
            position_index = next(
                index for index, event in enumerate(arm.events) if event[0] == "position"
            )
            fast_index = arm.events.index(("speed_profile", "FAST"))
            servo_index = next(
                index for index, event in enumerate(arm.events) if event[0] == "servo"
            )
            self.assertLess(slow_index, position_index)
            self.assertEqual(arm.events[position_index][1], [0.0] * 6)
            self.assertLess(position_index, fast_index)
            self.assertLess(fast_index, servo_index)
            self.assertTrue(arm.events[position_index][2])
            if index == 1:
                gripper_position_index = next(
                    event_index
                    for event_index, event in enumerate(arm.events)
                    if event[0] == "position_gripper"
                )
                gripper_servo_index = next(
                    event_index
                    for event_index, event in enumerate(arm.events)
                    if event[0] == "servo_gripper"
                )
                self.assertLess(gripper_position_index, gripper_servo_index)
                self.assertTrue(arm.events[gripper_position_index][2])
                self.assertAlmostEqual(arm.events[gripper_position_index][1], 0.02)

    def test_left_suction_arm_eef_api_is_never_called(self):
        class LeftSuctionArm(FakeArm):
            def move_eef_pos(self, target, *, blocking=False):
                raise AssertionError("left suction EEF planning must never be called")

            def servo_eef_pos(self, target):
                raise AssertionError("left suction EEF servo must never be called")

            def get_eef_pos(self):
                raise AssertionError("left suction EEF feedback must never be called")

        created = []

        def factory(_host, port):
            arm = LeftSuctionArm() if port == 50051 else FakeArm()
            created.append(arm)
            return arm

        self.replay.close()
        self.replay._arm_factory = factory
        self.replay.start(
            recording_id=self.recording_id,
            confirmation=self.recording_id,
        )
        completed = self.replay.wait(self.recording_id, timeout_s=2.0)
        self.assertEqual(completed["state"], "completed")
        self.assertEqual(created[0].gripper_planning_commands, 0)
        self.assertEqual(created[0].gripper_commands, 0)
        self.assertGreater(created[1].gripper_commands, 0)

    def test_right_gripper_feedback_error_aborts_replay(self):
        class StuckRightGripper(FakeArm):
            def get_eef_pos(self):
                return [0.0]

        def factory(_host, port):
            return StuckRightGripper() if port == 50053 else FakeArm()

        self.replay._arm_factory = factory
        self.replay.start(
            recording_id=self.recording_id,
            confirmation=self.recording_id,
        )
        with self.assertRaisesRegex(Exception, "right gripper tracking error"):
            self.replay.wait(self.recording_id, timeout_s=2.0)
        status = self.replay.status()
        self.assertEqual(status["state"], "error")
        self.assertGreater(status["gripper_max_tracking_error_m"], 0.012)

    def test_calibration_trajectory_replays_one_arm_without_gripper(self):
        recording_id = "trajectory_20260812_expand_box_12345678"
        episode = self.root / "trajectories" / recording_id
        (episode / "actions").mkdir(parents=True)
        (episode / "actions/left_arm.jsonl").write_text(
            jsonl(
                [
                    {
                        "timestamp": 200.0,
                        "joint_positions": [0.0] * 6,
                        "gripper": 0.01,
                    },
                    {
                        "timestamp": 200.1,
                        "joint_positions": [0.01] * 6,
                        "gripper": 0.02,
                    },
                ]
            ),
            encoding="utf-8",
        )
        (episode / "meta.json").write_text(
            json.dumps(
                {
                    "version": "medicine_calibration_episode_v1",
                    "recording_id": recording_id,
                    "status": "completed",
                    "selected_arms": ["left"],
                    "files": {"actions": {"left": "actions/left_arm.jsonl"}},
                }
            ),
            encoding="utf-8",
        )

        preflight = self.replay.preflight(
            recording_id,
            replay_gripper=False,
        )["preflight"]
        self.assertEqual(preflight["arms"], ["left"])
        self.assertEqual(
            preflight["trajectory_source"],
            "calibration_follower_action",
        )
        self.assertFalse(preflight["gripper_replayed"])

        started = self.replay.start(
            recording_id=recording_id,
            confirmation=recording_id,
            replay_gripper=False,
            allow_suction_engaged=True,
        )
        self.assertIn(started["state"], {"starting", "positioning", "replaying"})
        completed = self.replay.wait(recording_id, timeout_s=2.0)
        self.assertEqual(completed["state"], "completed")
        self.assertEqual(len(self.fake_arms), 1)
        self.assertGreater(self.fake_arms[0].commands, 0)
        self.assertEqual(self.fake_arms[0].gripper_planning_commands, 0)
        self.assertEqual(self.fake_arms[0].gripper_commands, 0)

    def test_official_open_pdd_episode_preflight_and_replay(self):
        recording_id = "act_20260814_fold_60_1ffdc902"
        episode = self.root / "eps" / "task3" / "episode_0060"
        (episode / "actions").mkdir(parents=True)
        left_rows = [
            {
                "timestamp": 200.000,
                "joint_positions": [0.0] * 6,
                "gripper": 0.01,
            },
            {
                "timestamp": 200.100,
                "joint_positions": [0.01] * 6,
                "gripper": 0.02,
            },
        ]
        right_rows = [
            {
                "timestamp": 200.001,
                "joint_positions": [0.1] * 6,
                "gripper": 0.015,
            },
            {
                "timestamp": 200.101,
                "joint_positions": [0.11] * 6,
                "gripper": 0.025,
            },
        ]
        (episode / "actions/left_arm.jsonl").write_text(
            jsonl(left_rows), encoding="utf-8"
        )
        (episode / "actions/right_arm.jsonl").write_text(
            jsonl(right_rows), encoding="utf-8"
        )
        (episode / "meta.json").write_text(
            json.dumps(
                {
                    "version": "v0.1",
                    "created_at": 200.0,
                    "finished_at": 200.2,
                    "task_meta": {"task_name": "act_bimanual"},
                    "actions": {"left_arm": {}, "right_arm": {}},
                    "extra": {
                        "recording_id": recording_id,
                        "purpose": "act_bimanual",
                        "recording_strategy": "official_open_pdd_direct_rgbd_v1",
                    },
                }
            ),
            encoding="utf-8",
        )

        preflight = self.replay.preflight(recording_id)["preflight"]

        self.assertEqual(preflight["arms"], ["left", "right"])
        self.assertEqual(preflight["sample_count"], 2)
        self.assertEqual(
            preflight["trajectory_source"],
            "official_follower_observation",
        )
        started = self.replay.start(
            recording_id=recording_id,
            confirmation=recording_id,
        )
        self.assertIn(started["state"], {"starting", "positioning", "replaying"})
        completed = self.replay.wait(recording_id, timeout_s=2.0)
        self.assertEqual(completed["state"], "completed")
        self.assertEqual(len(self.fake_arms), 2)


if __name__ == "__main__":
    unittest.main()
