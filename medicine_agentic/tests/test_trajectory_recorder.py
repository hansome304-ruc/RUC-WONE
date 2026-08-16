from __future__ import annotations

import json
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path

import cv2
import h5py
import numpy as np

from medicine_agentic.act_dataset import validate_episode
from medicine_agentic.act_export import prepare_training_dataset
from medicine_agentic.packaging_camera import CameraFrame
from medicine_agentic.trajectory_recorder import (
    RecordingConflict,
    RecordingUnavailable,
    SharedCameraBundle,
    TrajectoryRecorder,
)


class FakeCamera:
    def __init__(self, name="front") -> None:
        self.name = name
        self._lock = threading.Lock()
        self._frame_number = 0

    def profile(self):
        return {
            "name": self.name,
            "serial": f"fake-{self.name}",
            "mode": "offline",
            "color": {
                "width": 160,
                "height": 96,
                "fps": 10,
                "format": "bgr8",
            },
            "depth": {
                "width": 160,
                "height": 96,
                "fps": 10,
                "format": "z16",
            },
            "source_color_profile": {
                "width": 160,
                "height": 96,
                "fps": 10,
                "format": "bgr8",
            },
            "intrinsics": [
                [100.0, 0.0, 80.0],
                [0.0, 100.0, 48.0],
                [0.0, 0.0, 1.0],
            ],
            "profile_approved": False,
        }

    def capture(self) -> CameraFrame:
        with self._lock:
            self._frame_number += 1
            frame_number = self._frame_number
        image = np.full(
            (96, 160, 3),
            (frame_number % 255, 80, 140),
            dtype=np.uint8,
        )
        return CameraFrame(
            bgr=image,
            depth_z16=np.full((96, 160), 800, dtype=np.uint16),
            depth_scale_m=0.001,
            captured_at=time.time(),
            frame_number=frame_number,
            device_timestamp_ms=float(frame_number * 100),
        )


class FakeSynchronizedBundle:
    def __init__(
        self,
        directory: Path,
        *,
        sync_skew_ms: float = 1.25,
        exposure_age_s: float = 0.0,
    ) -> None:
        self.directory = directory
        self.fps = 10.0
        self.bundle_id = 0
        self.last_metadata = None
        self.sync_skew_ms = sync_skew_ms
        self.exposure_age_s = exposure_age_s
        self._next_capture_at = time.monotonic()
        self.begin_session_count = 0
        self.latest_only_calls = []

    def begin_session(self):
        self.begin_session_count += 1
        self.last_metadata = None

    def profiles(self):
        return {
            name: {
                "name": name,
                "mode": "fake_synchronized_bundle",
                "color": {"fps": self.fps, "format": "bgr8"},
            }
            for name in ("front", "left_wrist", "right_wrist")
        }

    def capture(self, *, latest_only=False, captured_after=None):
        self.latest_only_calls.append(bool(latest_only))
        delay_s = self._next_capture_at - time.monotonic()
        if delay_s > 0:
            time.sleep(delay_s)
        self._next_capture_at = max(
            self._next_capture_at + 1.0 / self.fps,
            time.monotonic(),
        )
        self.bundle_id += 1
        captured_at = time.time() - self.exposure_age_s
        if captured_after is not None:
            captured_at = max(captured_at, captured_after + 0.001)
        camera_timestamps = {
            name: captured_at + offset * 0.0005
            for offset, name in enumerate(
                ("front", "left_wrist", "right_wrist")
            )
        }
        self.last_metadata = {
            "bundle_id": self.bundle_id,
            "sync_skew_ms": self.sync_skew_ms,
            "cameras": {
                name: {
                    "captured_at": timestamp + 0.04,
                    "device_timestamp_ms": timestamp * 1000.0,
                    "sync_timestamp_ms": timestamp * 1000.0,
                    "timestamp_domain": "timestamp_domain.global_time",
                }
                for name, timestamp in camera_timestamps.items()
            },
        }
        return {
            name: CameraFrame(
                bgr=np.full(
                    (96, 160, 3),
                    (self.bundle_id % 255, 80 + offset, 140),
                    dtype=np.uint8,
                ),
                depth_z16=None,
                depth_scale_m=None,
                captured_at=camera_timestamps[name],
                frame_number=self.bundle_id,
                device_timestamp_ms=camera_timestamps[name] * 1000.0,
            )
            for offset, name in enumerate(
                ("front", "left_wrist", "right_wrist")
            )
        }


class FakeArmReader:
    def __init__(self, *, arm_names, host="localhost", **_kwargs) -> None:
        self.arm_names = tuple(arm_names)
        self.host = host
        self.connected = False
        self.sample_index = 0

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def capture_selected(self):
        if not self.connected:
            raise RuntimeError("fake arms are disconnected")
        self.sample_index += 1
        timestamp_ns = time.time_ns()
        arms = {}
        for offset, name in enumerate(self.arm_names):
            source_offset = 1.0 if self.host == "leader.test" else 0.0
            arms[name] = {
                "timestamp_ns": timestamp_ns + offset * 100_000,
                "joint_position_rad": [source_offset + 0.01 * self.sample_index] * 6,
                "flange_position_m": [0.2, -0.1 + offset * 0.2, 0.3],
                "flange_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "eef_feedback_m": [0.02],
                "driver_state": "IDLE",
                "control_mode": "ONLINE_TRAJ",
            }
        return {
            "timestamp_ns": timestamp_ns,
            "paired_sample_skew_ms": 0.1 if len(arms) == 2 else 0.0,
            "arms": arms,
        }


class StaticFakeArmReader(FakeArmReader):
    def capture_selected(self):
        sample = super().capture_selected()
        for arm in sample["arms"].values():
            arm["joint_position_rad"] = [0.1] * 6
            arm["eef_feedback_m"] = [0.02]
        return sample


def wait_for_state(recorder: TrajectoryRecorder, states: set[str], timeout=4.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = recorder.status()
        if status["state"] in states:
            return status
        time.sleep(0.02)
    raise AssertionError(f"recorder did not reach {states}: {recorder.status()}")


class TrajectoryRecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        runtime_dir = self.root / "runtime"
        runtime_dir.mkdir()
        (runtime_dir / "arm_services.json").write_text(
            json.dumps({"remote_host": "leader.test"}), encoding="utf-8"
        )
        self.recorder = TrajectoryRecorder(
            FakeCamera(),
            {
                "enabled": True,
                "output_dir": "episodes",
                "camera_fps_limit": 10,
                "arm_sample_hz": 20,
                "max_duration_s": 5,
                "min_free_gb": 0,
                "act_output_dir": "act/finalized",
                "act_failed_dir": "act/failed",
                "act": {
                    "leader_host": "stale.test",
                    "leader_runtime_config": "runtime/arm_services.json",
                    "leader_ports": {"left": 50050, "right": 50052},
                    "camera_names": ["front", "left_wrist", "right_wrist"],
                    "synchronized_bundle": {
                        "max_skew_ms": 30.0,
                        "encoder_queue_size": 4,
                        "require_contiguous_bundles": True,
                    },
                },
            },
            config_dir=self.root,
            arm_factory=FakeArmReader,
            act_bundle=FakeSynchronizedBundle(self.root / "sync"),
        )

    def tearDown(self) -> None:
        self.recorder.close()
        self.temporary.cleanup()

    def test_capture_act_observation_is_synchronized_and_read_only(self) -> None:
        observation = self.recorder.capture_act_observation()
        self.assertEqual(len(observation["state"]), 14)
        self.assertTrue(np.all(np.isfinite(observation["state"])))
        self.assertEqual(
            set(observation["frames_bgr"]),
            {"front", "left_wrist", "right_wrist"},
        )
        self.assertEqual(observation["hardware_access"], "feedback_only")
        self.assertLessEqual(
            observation["camera_arm_delta_ms"],
            self.recorder.act_max_camera_arm_delta_ms,
        )
        self.assertEqual(observation["timing_validation"], "strict")

    def test_rollout_frame_capture_requests_latest_bundle(self) -> None:
        bundle = self.recorder._act_bundle
        self.assertIsInstance(bundle, FakeSynchronizedBundle)

        first = self.recorder.capture_act_frames(begin_stream=True)
        second = self.recorder.capture_act_frames()

        self.assertEqual(set(first), {"front", "left_wrist", "right_wrist"})
        self.assertEqual(set(second), {"front", "left_wrist", "right_wrist"})
        self.assertEqual(bundle.begin_session_count, 1)
        self.assertEqual(bundle.latest_only_calls, [True, True])

    def test_delayed_preview_is_allowed_only_while_both_arms_are_static(self) -> None:
        self.recorder._act_bundle = FakeSynchronizedBundle(
            self.root / "sync-static",
            exposure_age_s=0.08,
        )
        self.recorder._arm_factory = StaticFakeArmReader
        observation = self.recorder.capture_act_observation()
        self.assertEqual(
            observation["timing_validation"],
            "static_delay_compensation",
        )
        self.assertGreater(
            observation["camera_arm_delta_ms"],
            self.recorder.act_max_camera_arm_delta_ms,
        )
        self.assertLessEqual(
            observation["camera_arm_delta_ms"],
            self.recorder.act_preview_max_camera_arm_delta_ms,
        )
        self.assertEqual(observation["static_joint_delta_rad"], 0.0)
        self.assertEqual(observation["static_eef_delta_m"], 0.0)

    def test_delayed_preview_is_rejected_when_arm_feedback_changes(self) -> None:
        self.recorder._act_bundle = FakeSynchronizedBundle(
            self.root / "sync-moving",
            exposure_age_s=0.08,
        )
        with self.assertRaisesRegex(RecordingUnavailable, "arms are moving"):
            self.recorder.capture_act_observation()

    def test_shared_bundle_uses_exposure_time_instead_of_arrival_time(self) -> None:
        root = self.root / "shared"
        sync = root / "sync"
        slot = sync / "slots" / "1"
        slot.mkdir(parents=True)
        names = ["front", "left_wrist", "right_wrist"]
        camera_meta = {}
        for offset, name in enumerate(names):
            relative = f"slots/1/{name}.bgr"
            np.full((2, 3, 3), offset, dtype=np.uint8).tofile(sync / relative)
            device_ms = 1_785_960_000_000.0 + offset * 0.5
            camera_meta[name] = {
                "file": relative,
                "color_shape": [2, 3, 3],
                "captured_at": device_ms / 1000.0 + 0.04,
                "device_timestamp_ms": device_ms,
                "sync_timestamp_ms": device_ms,
                "timestamp_domain": "timestamp_domain.global_time",
            }
        metadata = {
            "version": "ruc_video_sync_bundle_v1",
            "state": "ready",
            "bundle_id": 1,
            "ring_size": 2,
            "camera_names": names,
            "sync_skew_ms": 1.0,
            "cameras": camera_meta,
        }
        encoded = json.dumps(metadata)
        (sync / "meta.json").write_text(encoded, encoding="utf-8")
        (slot / "meta.json").write_text(encoded, encoding="utf-8")

        bundle = SharedCameraBundle(
            root=root,
            camera_names=tuple(names),
            fps=30.0,
            timeout_s=0.2,
            max_skew_ms=30.0,
        )
        frames = bundle.capture()
        for name in names:
            expected = camera_meta[name]["device_timestamp_ms"] / 1000.0
            self.assertAlmostEqual(frames[name].captured_at, expected, places=6)
            self.assertNotAlmostEqual(
                frames[name].captured_at,
                camera_meta[name]["captured_at"],
                places=3,
            )

    def test_shared_bundle_latest_only_skips_stale_backlog(self) -> None:
        root = self.root / "shared-latest"
        sync = root / "sync"
        ring_size = 4
        bundle_id = 10
        slot = sync / "slots" / str(bundle_id % ring_size)
        slot.mkdir(parents=True)
        names = ["front", "left_wrist", "right_wrist"]
        camera_meta = {}
        for offset, name in enumerate(names):
            relative = f"slots/{bundle_id % ring_size}/{name}.bgr"
            np.full((2, 3, 3), offset, dtype=np.uint8).tofile(sync / relative)
            device_ms = 1_785_960_000_000.0 + offset * 0.5
            camera_meta[name] = {
                "file": relative,
                "color_shape": [2, 3, 3],
                "captured_at": device_ms / 1000.0 + 0.04,
                "device_timestamp_ms": device_ms,
                "sync_timestamp_ms": device_ms,
                "timestamp_domain": "timestamp_domain.global_time",
            }
        metadata = {
            "version": "ruc_video_sync_bundle_v1",
            "state": "ready",
            "bundle_id": bundle_id,
            "ring_size": ring_size,
            "camera_names": names,
            "sync_skew_ms": 1.0,
            "cameras": camera_meta,
        }
        encoded = json.dumps(metadata)
        (sync / "meta.json").write_text(encoded, encoding="utf-8")
        (slot / "meta.json").write_text(encoded, encoding="utf-8")

        latest = SharedCameraBundle(
            root=root,
            camera_names=tuple(names),
            fps=30.0,
            timeout_s=0.2,
            max_skew_ms=30.0,
        )
        latest.last_bundle_id = 1
        frames = latest.capture(latest_only=True)
        self.assertEqual(latest.last_bundle_id, bundle_id)
        self.assertTrue(
            all(frame.frame_number == bundle_id for frame in frames.values())
        )

        contiguous = SharedCameraBundle(
            root=root,
            camera_names=tuple(names),
            fps=30.0,
            timeout_s=0.1,
            max_skew_ms=30.0,
        )
        contiguous.last_bundle_id = 1
        with self.assertRaisesRegex(RecordingUnavailable, "ring buffer overrun"):
            contiguous.capture()

    def test_shared_bundle_waits_for_exposure_after_chunk_cutoff(self) -> None:
        root = self.root / "shared-post-command"
        sync = root / "sync"
        names = ["front", "left_wrist", "right_wrist"]
        ring_size = 4

        def write_bundle(bundle_id: int, captured_at: float) -> None:
            slot_index = bundle_id % ring_size
            slot = sync / "slots" / str(slot_index)
            slot.mkdir(parents=True, exist_ok=True)
            camera_meta = {}
            for offset, name in enumerate(names):
                relative = f"slots/{slot_index}/{name}.bgr"
                np.full((2, 3, 3), bundle_id + offset, dtype=np.uint8).tofile(
                    sync / relative
                )
                device_ms = (captured_at + offset * 0.0005) * 1000.0
                camera_meta[name] = {
                    "file": relative,
                    "color_shape": [2, 3, 3],
                    "captured_at": captured_at + 0.04,
                    "device_timestamp_ms": device_ms,
                    "sync_timestamp_ms": device_ms,
                    "timestamp_domain": "timestamp_domain.global_time",
                }
            metadata = {
                "version": "ruc_video_sync_bundle_v1",
                "state": "ready",
                "bundle_id": bundle_id,
                "ring_size": ring_size,
                "camera_names": names,
                "sync_skew_ms": 1.0,
                "cameras": camera_meta,
            }
            encoded = json.dumps(metadata)
            (slot / "meta.json").write_text(encoded, encoding="utf-8")
            (sync / "meta.json").write_text(encoded, encoding="utf-8")

        cutoff = time.time()
        write_bundle(1, cutoff - 0.05)

        def publish_fresh_bundle() -> None:
            time.sleep(0.03)
            write_bundle(2, cutoff + 0.01)

        publisher = threading.Thread(target=publish_fresh_bundle)
        publisher.start()
        bundle = SharedCameraBundle(
            root=root,
            camera_names=tuple(names),
            fps=30.0,
            timeout_s=0.3,
            max_skew_ms=30.0,
        )
        frames = bundle.capture(latest_only=True, captured_after=cutoff)
        publisher.join(timeout=1.0)

        self.assertEqual(bundle.last_bundle_id, 2)
        self.assertTrue(all(frame.frame_number == 2 for frame in frames.values()))
        self.assertGreater(
            min(frame.captured_at for frame in frames.values()), cutoff
        )

    def test_records_calibration_compatible_rgb_and_both_arms(self) -> None:
        started = self.recorder.start(
            label="left handeye 01",
            purpose="calibration_left",
        )
        self.assertEqual(started["state"], "starting")
        wait_for_state(self.recorder, {"recording"})
        time.sleep(0.35)
        stopping = self.recorder.stop()
        self.assertEqual(stopping["state"], "stopping")
        saved = wait_for_state(self.recorder, {"saved"})

        output = Path(saved["saved_path"])
        self.assertTrue(output.is_dir())
        self.assertIn("left_handeye_01", output.name)
        metadata = json.loads((output / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["label"], "left handeye 01")
        self.assertEqual(metadata["selected_arms"], ["left", "right"])
        self.assertGreaterEqual(metadata["frame_count"], 2)
        self.assertGreater(metadata["arm_sample_counts"]["left"], 1)
        self.assertGreater(metadata["arm_sample_counts"]["right"], 1)
        self.assertFalse(metadata["safety"]["motion_commands"])
        self.assertFalse(metadata["recorder"]["depth_frames_saved"])

        rgb_path = output / "sensors" / "cam_front_rgb.mp4"
        tsf_path = output / "sensors" / "cam_front_rgb.mp4.tsf"
        self.assertTrue(rgb_path.is_file())
        self.assertTrue(tsf_path.is_file())
        capture = cv2.VideoCapture(str(rgb_path))
        try:
            video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        self.assertEqual(video_frames, metadata["frame_count"])

        raw_tsf = tsf_path.read_bytes()
        self.assertEqual(raw_tsf[:4], b"TSF1")
        self.assertEqual(struct.unpack("<Q", raw_tsf[8:16])[0], rgb_path.stat().st_size)
        self.assertEqual((len(raw_tsf) - 16) // 8, metadata["frame_count"])

        for arm in ("left", "right"):
            lines = (
                output / "actions" / f"{arm}_arm.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), metadata["arm_sample_counts"][arm])
            sample = json.loads(lines[0])
            self.assertEqual(len(sample["joint_positions"]), 6)
            self.assertEqual(len(sample["ee_positions"]), 7)
            self.assertIsInstance(sample["timestamp"], float)

    def test_rejects_parallel_start_and_supports_single_arm_trajectory(self) -> None:
        self.recorder.start(label="../../unsafe name", purpose="trajectory_left")
        wait_for_state(self.recorder, {"recording"})
        with self.assertRaises(RecordingConflict):
            self.recorder.start(label="second", purpose="trajectory_right")
        self.recorder.stop()
        saved = wait_for_state(self.recorder, {"saved"})
        output = Path(saved["saved_path"])
        self.assertTrue((output / "actions" / "left_arm.jsonl").is_file())
        self.assertFalse((output / "actions" / "right_arm.jsonl").exists())
        self.assertNotIn("..", output.name)
        self.assertIn("unsafe_name", output.name)

    def test_exposes_recorder_owned_preview_without_second_capture(self) -> None:
        started = self.recorder.start(
            label="preview test",
            purpose="trajectory_left",
        )
        self.assertIn("preview_test", started["target_path"])
        wait_for_state(self.recorder, {"recording"})

        deadline = time.monotonic() + 2.0
        active, preview = self.recorder.active_preview_frame()
        while preview is None and time.monotonic() < deadline:
            time.sleep(0.02)
            active, preview = self.recorder.active_preview_frame()
        self.assertTrue(active)
        self.assertIsNotNone(preview)
        self.assertEqual(preview.bgr.shape, (96, 160, 3))
        self.assertIsNotNone(preview.frame_number)

        self.recorder.stop()
        wait_for_state(self.recorder, {"saved"})
        self.assertEqual(self.recorder.active_preview_frame(), (False, None))

    def test_act_episode_records_follower_observations_and_leader_actions(self) -> None:
        self.recorder.start(label="act1_test", purpose="act_bimanual")
        wait_for_state(self.recorder, {"recording"})
        time.sleep(0.35)
        self.recorder.stop()
        saved = wait_for_state(self.recorder, {"saved"})

        output = Path(saved["saved_path"])
        self.assertEqual(output.parent, self.root / "act" / "finalized")
        self.assertTrue((output / "READY").is_file())
        self.assertTrue((output / "checksums.sha256").is_file())
        metadata = json.loads((output / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["version"], "medicine_act_episode_v1")
        self.assertEqual(metadata["act"]["observation_source"], "follower_joint_feedback")
        self.assertEqual(metadata["act"]["action_source"], "leader_joint_feedback")
        self.assertEqual(metadata["act"]["leader_endpoint"]["host"], "leader.test")
        self.assertGreater(metadata["observation_sample_counts"]["left"], 1)
        self.assertGreater(metadata["action_sample_counts"]["right"], 1)

        follower = json.loads(
            (output / "observations" / "left_arm.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        leader = json.loads(
            (output / "actions" / "left_arm.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        self.assertLess(follower["joint_positions"][0], 0.5)
        self.assertGreater(leader["joint_positions"][0], 0.5)
        manifest = (output / "checksums.sha256").read_text(encoding="utf-8")
        self.assertIn("sensors/cam_front_rgb.mp4", manifest)
        self.assertIn("sensors/cam_left_wrist_rgb.mp4", manifest)
        self.assertIn("sensors/cam_right_wrist_rgb.mp4", manifest)
        self.assertEqual(
            metadata["act"]["camera_names"],
            ["front", "left_wrist", "right_wrist"],
        )

        self.assertEqual(len(set(metadata["camera_frame_counts"].values())), 1)
        self.assertEqual(
            metadata["act"]["alignment"], "source_synchronized_bundle"
        )
        self.assertEqual(
            metadata["act"]["training_alignment"]["timestamp_basis"],
            "device_global_time",
        )
        self.assertEqual(
            metadata["act"]["synchronization"]["max_skew_ms"], 1.25
        )
        self.assertEqual(
            metadata["act"]["synchronization"]["max_allowed_skew_ms"], 30.0
        )
        self.assertEqual(
            metadata["act"]["synchronization"]["missed_bundle_count"], 0
        )
        self.assertEqual(
            metadata["act"]["synchronization"][
                "encoder_backpressure_drop_count"
            ],
            0,
        )
        self.assertEqual(metadata["recorder"]["video_encoder_mode"], "per_camera_async")
        for camera_name in ("front", "left_wrist", "right_wrist"):
            image_files = metadata["files"]["images"][camera_name]
            self.assertTrue((output / image_files["video"]).is_file())
            self.assertTrue((output / image_files["timestamps"]).is_file())
            self.assertGreater(metadata["camera_frame_counts"][camera_name], 1)
            first_frame_meta = json.loads(
                (output / image_files["frame_metadata"])
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(
                first_frame_meta["timestamp_source"], "device_global_time"
            )
            self.assertAlmostEqual(
                first_frame_meta["captured_at"] * 1000.0,
                first_frame_meta["sync_timestamp_ms"],
                places=3,
            )
        self.assertIn("observations/left_arm.jsonl", manifest)
        self.assertNotIn("READY", manifest)
        self.assertTrue(validate_episode(output).valid)

        processed_root = self.root / "processed"
        prepared = prepare_training_dataset(
            self.root / "act" / "finalized",
            processed_root,
            image_width=160,
            image_height=96,
        )
        self.assertEqual(prepared["tasks"], {"act1": 1})
        training_path = processed_root / "act1" / "episode_0.hdf5"
        self.assertTrue((processed_root / "act1" / "dataset_manifest.json").is_file())
        with h5py.File(training_path, "r") as training:
            aligned_count = metadata["act"]["training_alignment"][
                "aligned_sample_count"
            ]
            self.assertEqual(training.attrs["timestamp_basis"], "device_global_time")
            self.assertEqual(training["observations/qpos"].shape, (aligned_count, 14))
            self.assertEqual(training["action"].shape, (aligned_count, 14))
            self.assertEqual(
                training["observations/images/cam_high"].shape,
                (aligned_count, 96, 160, 3),
            )
            self.assertEqual(
                sorted(training["observations/images"]),
                ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            )
            self.assertTrue(
                np.all(np.diff(training["timestamps/aligned"][:]) > 0.0)
            )
            self.assertLess(float(training["observations/qpos"][0, 0]), 0.5)
            self.assertGreater(float(training["action"][0, 0]), 0.5)

        with (output / "actions" / "left_arm.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("{}\n")
        corrupted = validate_episode(output, verify_video=False)
        self.assertFalse(corrupted.valid)
        self.assertTrue(any("checksum mismatch" in error for error in corrupted.errors))

    def test_act_single_arm_mode_records_held_arm_action_from_follower(self) -> None:
        self.recorder.start(
            label="act_single_left",
            purpose="act_bimanual",
            action_from_observation_arms=("right",),
        )
        wait_for_state(self.recorder, {"recording"})
        time.sleep(0.35)
        self.recorder.stop()
        saved = wait_for_state(self.recorder, {"saved"})

        output = Path(saved["saved_path"])
        metadata = json.loads((output / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["act"]["action_source"], "mixed_by_arm")
        self.assertEqual(
            metadata["act"]["action_source_by_arm"],
            {
                "left": "leader_joint_feedback",
                "right": "held_follower_joint_feedback",
            },
        )
        left_action = json.loads(
            (output / "actions" / "left_arm.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        right_action = json.loads(
            (output / "actions" / "right_arm.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        self.assertGreater(left_action["joint_positions"][0], 0.5)
        self.assertLess(right_action["joint_positions"][0], 0.5)

    def test_act_resets_synchronized_bundle_at_each_episode_boundary(self) -> None:
        bundle = self.recorder._act_bundle
        self.assertIsInstance(bundle, FakeSynchronizedBundle)

        for label in ("act_first", "act_second"):
            self.recorder.start(label=label, purpose="act_bimanual")
            wait_for_state(self.recorder, {"recording"})
            time.sleep(0.35)
            self.recorder.stop()
            saved = wait_for_state(self.recorder, {"saved"})
            self.assertTrue(Path(saved["saved_path"]).is_dir())

        self.assertEqual(bundle.begin_session_count, 2)

    def test_strict_act_gate_rejects_over_skew_bundle(self) -> None:
        self.recorder.close()
        self.recorder = TrajectoryRecorder(
            FakeCamera(),
            {
                "enabled": True,
                "output_dir": "episodes",
                "camera_fps_limit": 10,
                "arm_sample_hz": 20,
                "max_duration_s": 5,
                "min_free_gb": 0,
                "act_output_dir": "act/finalized",
                "act_failed_dir": "act/failed",
                "act": {
                    "leader_host": "leader.test",
                    "leader_ports": {"left": 50050, "right": 50052},
                    "camera_names": ["front", "left_wrist", "right_wrist"],
                    "synchronized_bundle": {
                        "max_skew_ms": 30.0,
                        "encoder_queue_size": 4,
                        "require_contiguous_bundles": True,
                    },
                },
            },
            config_dir=self.root,
            arm_factory=FakeArmReader,
            act_bundle=FakeSynchronizedBundle(
                self.root / "bad-sync", sync_skew_ms=50.0
            ),
        )
        self.recorder.start(label="bad sync", purpose="act_bimanual")
        wait_for_state(self.recorder, {"recording"})
        time.sleep(0.25)
        self.recorder.stop()
        failed = wait_for_state(self.recorder, {"error"})

        self.assertIn("strict ACT quality gate failed", failed["error"])
        output = Path(failed["saved_path"])
        self.assertEqual(output.parent, self.root / "act" / "failed")
        self.assertFalse((output / "READY").exists())

    def test_delete_recording_archives_exact_metadata_match(self) -> None:
        episode = self.recorder.output_dir / "trajectory_delete_me"
        episode.mkdir(parents=True)
        (episode / "meta.json").write_text(
            json.dumps(
                {
                    "recording_id": "trajectory_delete_me",
                    "label": "delete me",
                    "purpose": "trajectory_both",
                    "status": "completed",
                    "created_at": time.time(),
                }
            ),
            encoding="utf-8",
        )

        result = self.recorder.delete_recording("trajectory_delete_me")

        self.assertTrue(result["deleted"])
        self.assertTrue(result["recoverable"])
        self.assertFalse(episode.exists())
        archived = Path(result["archived_path"])
        self.assertTrue((archived / "meta.json").is_file())
        self.assertIn("deleted-by-ui", archived.parts)
        self.assertEqual(self.recorder.list_recordings(), [])

        with self.assertRaises(FileNotFoundError):
            self.recorder.delete_recording("../trajectory_delete_me")

    def test_official_completed_episode_is_replay_ready(self) -> None:
        episode = self.recorder.act_output_dir / "task3" / "episode_0060"
        (episode / "actions").mkdir(parents=True)
        for arm in ("left", "right"):
            (episode / "actions" / f"{arm}_arm.jsonl").write_text(
                "{}\n{}\n",
                encoding="utf-8",
            )
        (episode / "meta.json").write_text(
            json.dumps(
                {
                    "version": "v0.1",
                    "created_at": 100.0,
                    "finished_at": 101.0,
                    "task_meta": {
                        "task_name": "act_bimanual",
                        "prompt": "fold_60",
                    },
                    "extra": {
                        "recording_id": "act_20260814_fold_60_1ffdc902",
                        "label": "fold_60",
                        "purpose": "act_bimanual",
                        "recording_strategy": "official_open_pdd_direct_rgbd_v1",
                    },
                }
            ),
            encoding="utf-8",
        )

        items = self.recorder.list_recordings()

        self.assertEqual(len(items), 1)
        self.assertTrue(items[0]["replay_ready"])


if __name__ == "__main__":
    unittest.main()
