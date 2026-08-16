"""Validated replay of finalized ACT and calibration joint trajectories."""
from __future__ import annotations

import bisect
import inspect
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from medicine_agentic.act_dataset import validate_episode


SAFE_ID = re.compile(r"^(?:act|trajectory)_[0-9A-Za-z._-]+$")
ACTIVE_STATES = frozenset({"starting", "positioning", "replaying", "stopping"})


class ReplayError(RuntimeError):
    pass


class ReplayConflict(ReplayError):
    pass


class ReplayUnavailable(ReplayError):
    pass


@dataclass
class ReplaySession:
    recording_id: str
    episode_path: Path
    speed_scale: float
    max_tracking_error_rad: float | None
    requested_at: float
    gripper_replayed: bool = False
    gripper_replay_arms: tuple[str, ...] = ()
    max_gripper_tracking_error_m: float | None = None
    gripper_last_tracking_error_m: float | None = None
    gripper_max_tracking_error_m: float = 0.0
    source_sample_count: int = 0
    source_frame_numbers_1_based: tuple[int, ...] = ()
    current_source_frame_1_based: int = 0
    retained_frame_stride: int = 1
    retained_frame_parity: str = "all"
    suction_release_frame_1_based: int | None = None
    suction_release_state: str = "not_configured"
    suction_release_started_at: float | None = None
    suction_released_at: float | None = None
    allow_suction_engaged: bool = False
    trajectory_source: str = "follower_observation"
    state: str = "starting"
    started_at: float | None = None
    finished_at: float | None = None
    current_index: int = 0
    sample_count: int = 0
    error: str | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    worker: threading.Thread | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "active": self.state in ACTIVE_STATES,
            "state": self.state,
            "recording_id": self.recording_id,
            "episode_path": str(self.episode_path),
            "speed_scale": self.speed_scale,
            "max_tracking_error_rad": self.max_tracking_error_rad,
            "requested_at": self.requested_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "current_index": self.current_index,
            "sample_count": self.sample_count,
            "progress": (
                self.current_index / self.sample_count
                if self.sample_count
                else 0.0
            ),
            "error": self.error,
            "suction_replayed": self.suction_release_frame_1_based is not None,
            "suction_release_frame_1_based": self.suction_release_frame_1_based,
            "suction_release_state": self.suction_release_state,
            "suction_release_started_at": self.suction_release_started_at,
            "suction_released_at": self.suction_released_at,
            "source_sample_count": self.source_sample_count,
            "current_source_frame_1_based": self.current_source_frame_1_based,
            "retained_frame_stride": self.retained_frame_stride,
            "retained_frame_parity": self.retained_frame_parity,
            "gripper_replayed": self.gripper_replayed,
            "gripper_replay_arms": list(self.gripper_replay_arms),
            "left_gripper_commands_forbidden": True,
            "max_gripper_tracking_error_m": self.max_gripper_tracking_error_m,
            "gripper_last_tracking_error_m": self.gripper_last_tracking_error_m,
            "gripper_max_tracking_error_m": self.gripper_max_tracking_error_m,
            "trajectory_source": self.trajectory_source,
        }


class TrajectoryReplay:
    """Replay the executed follower trajectory with strict safety gates."""

    def __init__(
        self,
        config: dict[str, Any] | None,
        *,
        config_dir: Path,
        interlock: Callable[..., str | None],
        arm_factory: Callable[[str, int], Any] | None = None,
        servo_mode: Any | None = None,
        planning_mode: Any | None = None,
        initial_speed_profile: Any | None = None,
        speed_profile: Any | None = None,
        suction_status: Callable[[], dict[str, Any]] | None = None,
        suction_setter: Callable[[bool], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        cfg = dict(config or {})
        self.enabled = cfg.get("enabled") is True
        root = Path(str(cfg.get("episode_root", "../recordings/act/finalized"))).expanduser()
        self.episode_root = (root if root.is_absolute() else config_dir / root).resolve()
        official_root = Path(
            str(cfg.get("official_episode_root", "../recordings/eps"))
        ).expanduser()
        self.official_episode_root = (
            official_root
            if official_root.is_absolute()
            else config_dir / official_root
        ).resolve()
        calibration_root = Path(
            str(cfg.get("calibration_episode_root", "../recordings/trajectories"))
        ).expanduser()
        self.calibration_episode_root = (
            calibration_root
            if calibration_root.is_absolute()
            else config_dir / calibration_root
        ).resolve()
        self.arm_host = str(cfg.get("arm_host", "localhost"))
        raw_ports = cfg.get("arm_ports", {"left": 50051, "right": 50053})
        if not isinstance(raw_ports, dict):
            raise ValueError("trajectory_replay.arm_ports must be an object")
        self.arm_ports = {
            "left": int(raw_ports.get("left", 50051)),
            "right": int(raw_ports.get("right", 50053)),
        }
        # Replay follows the recorded timeline.  Deliberately do not expose a
        # speed multiplier: the operator asked for faithful, normal-speed
        # playback after automatic start-pose positioning.
        self.default_speed_scale = 1.0
        self.min_speed_scale = 1.0
        self.max_speed_scale = 1.0
        self.command_hz = float(cfg.get("command_hz", 50.0))
        self.initial_move_timeout_s = float(
            cfg.get("initial_move_timeout_s", 45.0)
        )
        self.initial_joint_tolerance_rad = float(
            cfg.get("initial_joint_tolerance_rad", 0.06)
        )
        self.initial_feedback_poll_s = float(
            cfg.get("initial_feedback_poll_s", 0.05)
        )
        self.initial_feedback_stable_samples = int(
            cfg.get("initial_feedback_stable_samples", 3)
        )
        self.max_recorded_joint_step_rad = float(
            cfg.get("max_recorded_joint_step_rad", 0.25)
        )
        self.max_recorded_joint_velocity_rad_s = float(
            cfg.get("max_recorded_joint_velocity_rad_s", 5.0)
        )
        self.max_tracking_error_rad = float(cfg.get("max_tracking_error_rad", 0.35))
        self.feedback_every_n = int(cfg.get("feedback_every_n", 10))
        self.replay_gripper = cfg.get("replay_gripper", True) is True
        raw_gripper_replay_arms = cfg.get("gripper_replay_arms", ["right"])
        if raw_gripper_replay_arms != ["right"]:
            raise ValueError(
                "trajectory_replay.gripper_replay_arms must be exactly ['right']; "
                "left gripper commands are forbidden because a suction cup is installed"
            )
        # This is deliberately a hard-coded allow-list, not merely a default.
        # The left end effector is a suction assembly that can be damaged by a
        # gripper-open command, so replay must never touch any left EEF API.
        self.gripper_replay_arms = frozenset({"right"})
        self.gripper_feedback_every_n = int(
            cfg.get("gripper_feedback_every_n", 5)
        )
        self.max_gripper_tracking_error_m = float(
            cfg.get("max_gripper_tracking_error_m", 0.012)
        )
        self.gripper_scale = float(cfg.get("gripper_scale", 0.072 / 0.0471))
        self.gripper_min_m = float(cfg.get("gripper_min_m", 0.0))
        self.gripper_max_m = float(cfg.get("gripper_max_m", 0.072))
        self.max_recorded_gripper_step_m = float(
            cfg.get("max_recorded_gripper_step_m", 0.02)
        )
        self.max_recorded_gripper_velocity_m_s = float(
            cfg.get("max_recorded_gripper_velocity_m_s", 0.5)
        )
        raw_profiles = cfg.get("recording_profiles", {})
        if not isinstance(raw_profiles, dict):
            raise ValueError("trajectory_replay.recording_profiles must be an object")
        self.recording_profiles: dict[str, dict[str, Any]] = {}
        required_profile_keys = frozenset(
            {
                "retain_frame_numbers_1_based",
                "playback_speed_scale",
                "suction_release_frame_1_based",
                "max_tracking_error_rad",
                "max_gripper_tracking_error_m",
            }
        )
        optional_profile_keys = frozenset(
            {"first_retained_frame_1_based"}
        )
        for recording_id, raw_profile in raw_profiles.items():
            if not isinstance(recording_id, str) or not SAFE_ID.fullmatch(recording_id):
                raise ValueError("trajectory_replay recording profile has invalid ID")
            if (
                not isinstance(raw_profile, dict)
                or not required_profile_keys.issubset(raw_profile)
                or not frozenset(raw_profile).issubset(
                    required_profile_keys | optional_profile_keys
                )
            ):
                raise ValueError(
                    "trajectory_replay recording profile fields must include "
                    "retain_frame_numbers_1_based, playback_speed_scale, and "
                    "suction_release_frame_1_based, max_tracking_error_rad, and "
                    "max_gripper_tracking_error_m; first_retained_frame_1_based "
                    "is optional"
                )
            parity = raw_profile["retain_frame_numbers_1_based"]
            speed_scale = float(raw_profile["playback_speed_scale"])
            release_frame = int(raw_profile["suction_release_frame_1_based"])
            first_retained_frame = int(
                raw_profile.get("first_retained_frame_1_based", 1)
            )
            raw_tracking_error = raw_profile["max_tracking_error_rad"]
            tracking_error = (
                None
                if raw_tracking_error is None
                else float(raw_tracking_error)
            )
            raw_gripper_tracking_error = raw_profile[
                "max_gripper_tracking_error_m"
            ]
            gripper_tracking_error = (
                None
                if raw_gripper_tracking_error is None
                else float(raw_gripper_tracking_error)
            )
            if parity != "odd" or speed_scale != 2.0:
                raise ValueError(
                    "trajectory_replay fast recording profile must retain odd frames at 2x"
                )
            if release_frame < 1 or release_frame % 2 != 1:
                raise ValueError(
                    "suction release frame must be a retained positive odd frame"
                )
            if first_retained_frame < 1 or first_retained_frame % 2 != 1:
                raise ValueError(
                    "first retained frame must be a positive odd frame"
                )
            if release_frame < first_retained_frame:
                raise ValueError(
                    "suction release frame cannot precede the first retained frame"
                )
            if tracking_error is not None and not 0.05 <= tracking_error <= 1.0:
                raise ValueError(
                    "recording profile max_tracking_error_rad must be between 0.05 and 1.0"
                )
            if gripper_tracking_error is not None and not (
                0.001 <= gripper_tracking_error <= 0.05
            ):
                raise ValueError(
                    "recording profile max_gripper_tracking_error_m must be "
                    "between 0.001 and 0.05"
                )
            self.recording_profiles[recording_id] = {
                "retain_frame_numbers_1_based": "odd",
                "playback_speed_scale": 2.0,
                "first_retained_frame_1_based": first_retained_frame,
                "suction_release_frame_1_based": release_frame,
                "max_tracking_error_rad": tracking_error,
                "max_gripper_tracking_error_m": gripper_tracking_error,
            }
        if not (
            10.0 <= self.command_hz <= 100.0
            and 1.0 <= self.initial_move_timeout_s <= 120.0
            and 0.01 <= self.initial_joint_tolerance_rad <= 0.25
            and 0.01 <= self.initial_feedback_poll_s <= 0.5
            and 1 <= self.initial_feedback_stable_samples <= 20
            and 0.01 <= self.max_recorded_joint_step_rad <= 1.0
            and 0.1 <= self.max_recorded_joint_velocity_rad_s <= 20.0
            and 0.05 <= self.max_tracking_error_rad <= 1.0
            and 1 <= self.feedback_every_n <= 100
            and 1 <= self.gripper_feedback_every_n <= 100
            and 0.001 <= self.max_gripper_tracking_error_m <= 0.05
            and 0.1 <= self.gripper_scale <= 5.0
            and 0.0 <= self.gripper_min_m < self.gripper_max_m <= 0.2
            and 0.001 <= self.max_recorded_gripper_step_m <= 0.05
            and 0.01 <= self.max_recorded_gripper_velocity_m_s <= 2.0
        ):
            raise ValueError("invalid trajectory_replay safety settings")
        self._interlock = interlock
        try:
            parameters = inspect.signature(interlock).parameters
        except (TypeError, ValueError):
            parameters = {}
        self._interlock_accepts_suction_context = bool(
            "allow_suction_engaged" in parameters
            or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        )
        self._arm_factory = arm_factory or self._default_arm_factory
        self._servo_mode = servo_mode
        self._planning_mode = planning_mode
        self._initial_speed_profile = initial_speed_profile
        self._speed_profile = speed_profile
        self._suction_status = suction_status
        self._suction_setter = suction_setter
        self._sleep = sleep
        self._lock = threading.RLock()
        self._session: ReplaySession | None = None
        self._last_session: ReplaySession | None = None
        self._active_arms: dict[str, Any] = {}

    @staticmethod
    def _default_arm_factory(host: str, port: int) -> Any:
        try:
            from airbot_py.arm import AIRBOTPlay
        except ImportError as exc:  # pragma: no cover
            raise ReplayUnavailable("airbot_py is unavailable") from exc
        return AIRBOTPlay(url=host, port=port)

    def status(self) -> dict[str, Any]:
        with self._lock:
            session = self._session or self._last_session
            payload = (
                session.snapshot()
                if session is not None
                else {
                    "active": False,
                    "state": "idle",
                    "recording_id": None,
                    "current_index": 0,
                    "sample_count": 0,
                    "progress": 0.0,
                    "error": None,
                    "suction_replayed": False,
                    "gripper_replayed": self.replay_gripper,
                    "gripper_replay_arms": ["right"] if self.replay_gripper else [],
                    "left_gripper_commands_forbidden": True,
                }
            )
        payload.update(
            {
                "enabled": self.enabled,
                "default_speed_scale": self.default_speed_scale,
                "minimum_speed_scale": self.min_speed_scale,
                "maximum_speed_scale": self.max_speed_scale,
                "requires_exact_recording_id_confirmation": True,
                "requires_near_start_pose": False,
                "automatically_positions_at_first_frame": True,
            }
        )
        payload.setdefault("gripper_replayed", self.replay_gripper)
        payload.setdefault(
            "gripper_replay_arms", ["right"] if self.replay_gripper else []
        )
        payload.setdefault("left_gripper_commands_forbidden", True)
        payload["gripper_feedback_every_n"] = self.gripper_feedback_every_n
        payload.setdefault(
            "max_gripper_tracking_error_m",
            self.max_gripper_tracking_error_m,
        )
        payload.setdefault("trajectory_source", "follower_observation")
        return payload

    def _interlock_blocker(self, session: ReplaySession | None = None) -> str | None:
        allow_suction_engaged = bool(
            session is not None and session.allow_suction_engaged
        )
        if self._interlock_accepts_suction_context:
            return self._interlock(
                allow_suction_engaged=allow_suction_engaged,
            )
        return self._interlock()

    def _episode_path(self, recording_id: str) -> Path:
        if not SAFE_ID.fullmatch(recording_id):
            raise ValueError("invalid recording_id")
        if recording_id.startswith("trajectory_"):
            candidate = (self.calibration_episode_root / recording_id).resolve()
            if candidate.parent != self.calibration_episode_root:
                raise ValueError("unsafe recording_id")
            return candidate

        candidate = (self.episode_root / recording_id).resolve()
        if candidate.parent != self.episode_root:
            raise ValueError("unsafe recording_id")
        if candidate.is_dir():
            return candidate

        # The official open_pdd recorder stores episodes by task and sequence
        # (for example eps/task3/episode_0060), while the operator-facing ID is
        # kept in meta.json.extra.recording_id. Resolve that ID without ever
        # accepting a client-supplied filesystem path.
        metadata_paths = list(self.official_episode_root.glob("episode_*/meta.json"))
        metadata_paths.extend(
            self.official_episode_root.glob("*/episode_*/meta.json")
        )
        matches: list[Path] = []
        for metadata_path in metadata_paths:
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            extra = payload.get("extra") if isinstance(payload, dict) else None
            if isinstance(extra, dict) and extra.get("recording_id") == recording_id:
                resolved = metadata_path.parent.resolve()
                try:
                    resolved.relative_to(self.official_episode_root)
                except ValueError:
                    continue
                matches.append(resolved)
        if len(matches) > 1:
            raise ReplayUnavailable("recording_id matches multiple official episodes")
        return matches[0] if matches else candidate

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ReplayUnavailable(
                            f"{path.name}:{line_number} must contain an object"
                        )
                    rows.append(row)
        except (OSError, json.JSONDecodeError) as exc:
            raise ReplayUnavailable(f"cannot read replay samples {path}: {exc}") from exc
        return rows

    def _load_calibration_episode(
        self,
        path: Path,
        recording_id: str,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        try:
            meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReplayUnavailable(
                f"cannot read calibration trajectory metadata: {exc}"
            ) from exc
        if not isinstance(meta, dict):
            raise ReplayUnavailable("calibration trajectory metadata must be an object")
        if meta.get("version") != "medicine_calibration_episode_v1":
            raise ReplayUnavailable("unsupported calibration trajectory version")
        if meta.get("recording_id") != recording_id:
            raise ReplayUnavailable("calibration trajectory recording_id mismatch")
        if meta.get("status") != "completed":
            raise ReplayUnavailable("calibration trajectory is not completed")
        arms = [str(name) for name in meta.get("selected_arms", [])]
        if len(arms) != 1 or arms[0] not in self.arm_ports:
            raise ReplayUnavailable(
                "calibration trajectory must contain exactly one valid arm"
            )
        arm = arms[0]
        relative = (
            meta.get("files", {}).get("actions", {}).get(arm)
            if isinstance(meta.get("files"), dict)
            else None
        )
        if not isinstance(relative, str) or not relative:
            relative = f"actions/{arm}_arm.jsonl"
        sample_path = (path / relative).resolve()
        try:
            sample_path.relative_to(path)
        except ValueError as exc:
            raise ReplayUnavailable("unsafe calibration trajectory sample path") from exc
        source_rows = self._read_jsonl(sample_path)
        rows = [
            {
                "timestamp": row.get("timestamp"),
                "observation": {
                    arm: {
                        "joint_positions": row.get("joint_positions"),
                        "gripper": row.get("gripper"),
                    }
                },
            }
            for row in source_rows
        ]
        return arms, rows

    def _load(
        self,
        recording_id: str,
        *,
        replay_gripper: bool | None = None,
    ) -> tuple[Path, list[str], list[dict[str, Any]], str]:
        path = self._episode_path(recording_id)
        replay_gripper = (
            self.replay_gripper if replay_gripper is None else replay_gripper
        )
        if recording_id.startswith("trajectory_"):
            arms, rows = self._load_calibration_episode(path, recording_id)
            trajectory_source = "calibration_follower_action"
        else:
            arms, rows, trajectory_source = self._load_act_episode(
                path,
                recording_id,
            )
        if len(rows) < 2:
            raise ReplayUnavailable("episode has fewer than two aligned samples")
        try:
            timestamps = np.asarray(
                [row["timestamp"] for row in rows], dtype=np.float64
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplayUnavailable("episode timestamps are invalid") from exc
        if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
            raise ReplayUnavailable("aligned timestamps are not finite and increasing")
        for arm in arms:
            try:
                joints = np.asarray(
                    [row["observation"][arm]["joint_positions"] for row in rows],
                    dtype=np.float64,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ReplayUnavailable(f"{arm} observation joints are invalid") from exc
            if joints.shape != (len(rows), 6) or not np.all(np.isfinite(joints)):
                raise ReplayUnavailable(f"{arm} observation joints are invalid")
            steps = np.max(np.abs(np.diff(joints, axis=0)), axis=1)
            velocities = steps / np.diff(timestamps)
            if float(np.max(steps)) > self.max_recorded_joint_step_rad:
                raise ReplayUnavailable(f"{arm} observation has an unsafe joint step")
            if float(np.max(velocities)) > self.max_recorded_joint_velocity_rad_s:
                raise ReplayUnavailable(f"{arm} observation has an unsafe joint velocity")
            if replay_gripper and arm in self.gripper_replay_arms:
                try:
                    raw_gripper = np.asarray(
                        [row["observation"][arm]["gripper"] for row in rows],
                        dtype=np.float64,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ReplayUnavailable(
                        f"{arm} observation gripper is invalid"
                    ) from exc
                if raw_gripper.shape != (len(rows),) or not np.all(
                    np.isfinite(raw_gripper)
                ):
                    raise ReplayUnavailable(f"{arm} observation gripper is invalid")
                source_max = self.gripper_max_m / self.gripper_scale
                if float(np.min(raw_gripper)) < -0.005 or float(
                    np.max(raw_gripper)
                ) > source_max + 0.005:
                    raise ReplayUnavailable(f"{arm} observation gripper is out of range")
                gripper = np.clip(
                    raw_gripper * self.gripper_scale,
                    self.gripper_min_m,
                    self.gripper_max_m,
                )
                gripper_steps = np.abs(np.diff(gripper))
                gripper_velocities = gripper_steps / np.diff(timestamps)
                if float(np.max(gripper_steps)) > self.max_recorded_gripper_step_m:
                    raise ReplayUnavailable(
                        f"{arm} observation has an unsafe gripper step"
                    )
                if (
                    float(np.max(gripper_velocities))
                    > self.max_recorded_gripper_velocity_m_s
                ):
                    raise ReplayUnavailable(
                        f"{arm} observation has an unsafe gripper velocity"
                    )
        return path, arms, rows, trajectory_source

    def _apply_recording_profile(
        self,
        recording_id: str,
        arms: list[str],
        rows: list[dict[str, Any]],
        *,
        replay_gripper: bool,
    ) -> tuple[list[dict[str, Any]], tuple[int, ...], dict[str, Any]]:
        """Apply a configured virtual replay profile without mutating the episode."""

        profile = self.recording_profiles.get(recording_id)
        if profile is None:
            return (
                rows,
                tuple(range(1, len(rows) + 1)),
                {
                    "speed_scale": 1.0,
                    "frame_stride": 1,
                    "frame_parity": "all",
                    "suction_release_frame_1_based": None,
                    "max_tracking_error_rad": None,
                    "max_gripper_tracking_error_m": self.max_gripper_tracking_error_m,
                },
            )
        source_timestamps = np.asarray(
            [row["timestamp"] for row in rows], dtype=np.float64
        )
        first_retained_frame = int(
            profile.get("first_retained_frame_1_based", 1)
        )
        retained_indices = list(
            range(first_retained_frame - 1, len(rows), 2)
        )
        if len(retained_indices) < 2:
            raise ReplayUnavailable(
                "fast odd-frame replay profile retains fewer than two samples"
            )
        release_frame = int(profile["suction_release_frame_1_based"])
        if release_frame > len(rows):
            raise ReplayUnavailable(
                "configured suction release frame exceeds source sample count"
            )
        speed_scale = float(profile["playback_speed_scale"])
        source_origin = float(source_timestamps[retained_indices[0]])
        profiled_rows: list[dict[str, Any]] = []
        for source_index in retained_indices:
            row = dict(rows[source_index])
            row["timestamp"] = source_origin + (
                float(source_timestamps[source_index]) - source_origin
            ) / speed_scale
            profiled_rows.append(row)
        self._validate_execution_rows(
            arms,
            profiled_rows,
            replay_gripper=replay_gripper,
            context="fast odd-frame replay",
        )
        return (
            profiled_rows,
            tuple(index + 1 for index in retained_indices),
            {
                "speed_scale": speed_scale,
                "frame_stride": 2,
                "frame_parity": "odd",
                "first_retained_frame_1_based": first_retained_frame,
                "suction_release_frame_1_based": release_frame,
                "max_tracking_error_rad": profile["max_tracking_error_rad"],
                "max_gripper_tracking_error_m": profile[
                    "max_gripper_tracking_error_m"
                ],
            },
        )

    def _validate_execution_rows(
        self,
        arms: list[str],
        rows: list[dict[str, Any]],
        *,
        replay_gripper: bool,
        context: str,
    ) -> None:
        timestamps = np.asarray([row["timestamp"] for row in rows], dtype=np.float64)
        deltas = np.diff(timestamps)
        if not np.all(np.isfinite(timestamps)) or np.any(deltas <= 0.0):
            raise ReplayUnavailable(f"{context} timestamps are invalid")
        for arm in arms:
            joints = np.asarray(
                [row["observation"][arm]["joint_positions"] for row in rows],
                dtype=np.float64,
            )
            joint_steps = np.max(np.abs(np.diff(joints, axis=0)), axis=1)
            joint_velocities = joint_steps / deltas
            if float(np.max(joint_steps)) > self.max_recorded_joint_step_rad:
                raise ReplayUnavailable(f"{context} has an unsafe {arm} joint step")
            if (
                float(np.max(joint_velocities))
                > self.max_recorded_joint_velocity_rad_s
            ):
                raise ReplayUnavailable(
                    f"{context} has an unsafe {arm} joint velocity"
                )
            if replay_gripper and arm in self.gripper_replay_arms:
                gripper = np.clip(
                    np.asarray(
                        [row["observation"][arm]["gripper"] for row in rows],
                        dtype=np.float64,
                    )
                    * self.gripper_scale,
                    self.gripper_min_m,
                    self.gripper_max_m,
                )
                gripper_steps = np.abs(np.diff(gripper))
                gripper_velocities = gripper_steps / deltas
                if float(np.max(gripper_steps)) > self.max_recorded_gripper_step_m:
                    raise ReplayUnavailable(
                        f"{context} has an unsafe {arm} gripper step"
                    )
                if (
                    float(np.max(gripper_velocities))
                    > self.max_recorded_gripper_velocity_m_s
                ):
                    raise ReplayUnavailable(
                        f"{context} has an unsafe {arm} gripper velocity"
                    )

    def _require_suction_ready_for_profile(self) -> None:
        if self._suction_status is None or self._suction_setter is None:
            raise ReplayUnavailable("profiled replay suction control is unavailable")
        status = self._suction_status()
        if not isinstance(status, dict) or status.get("available") is not True:
            raise ReplayUnavailable("suction control is unavailable for profiled replay")
        if status.get("engaged") is not True:
            raise ReplayConflict(
                "suction must be engaged before this profiled replay starts"
            )

    def _release_suction(self) -> Any:
        if self._suction_setter is None or self._suction_status is None:
            raise ReplayUnavailable("profiled replay suction control is unavailable")
        result = self._suction_setter(False)
        if isinstance(result, dict) and result.get("engaged") is not False:
            raise ReplayUnavailable("suction release command was not confirmed")
        status = self._suction_status()
        if not isinstance(status, dict) or status.get("engaged") is not False:
            raise ReplayUnavailable("suction remained engaged after release command")
        return result

    def _load_act_episode(
        self,
        path: Path,
        recording_id: str,
    ) -> tuple[list[str], list[dict[str, Any]], str]:
        # Replay consumes the verified follower-observation timeline: this is
        # the path the physical robot actually executed. Leader actions remain
        # untouched as ACT training targets, but can include teleop tracking
        # lag that is unsafe and misleading for physical trajectory validation.
        # Replay deliberately
        # does not require strict camera-exposure provenance, so legacy
        # episodes can be mechanically inspected without ever becoming valid
        # training input.
        try:
            raw_meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReplayUnavailable(f"cannot read trajectory metadata: {exc}") from exc
        extra = raw_meta.get("extra") if isinstance(raw_meta, dict) else None
        if (
            isinstance(extra, dict)
            and extra.get("recording_strategy")
            in {
                "official_open_pdd_shared_rgbd_v1",
                "official_open_pdd_direct_rgbd_v1",
            }
        ):
            return self._load_official_act_episode(
                path,
                recording_id,
                raw_meta,
            )

        validation = validate_episode(
            path,
            verify_video=False,
            require_training_timing=False,
        )
        if not validation.valid:
            raise ReplayUnavailable(
                "episode validation failed: " + "; ".join(validation.errors)
            )
        meta = validation.metadata or {}
        arms = [str(name) for name in meta.get("selected_arms", [])]
        if not arms or any(name not in self.arm_ports for name in arms):
            raise ReplayUnavailable("episode selected_arms are invalid")
        aligned_relative = meta.get("files", {}).get("aligned_samples")
        return (
            arms,
            self._read_jsonl(path / str(aligned_relative)),
            "follower_observation",
        )

    def _load_official_act_episode(
        self,
        path: Path,
        recording_id: str,
        meta: dict[str, Any],
    ) -> tuple[list[str], list[dict[str, Any]], str]:
        """Normalize official open_pdd follower feedback into replay rows."""
        extra = meta.get("extra")
        if not isinstance(extra, dict) or extra.get("recording_id") != recording_id:
            raise ReplayUnavailable("official episode recording_id mismatch")
        if not meta.get("finished_at"):
            raise ReplayUnavailable("official episode is not completed")
        purpose = str(extra.get("purpose", ""))
        if not purpose.startswith("act_"):
            raise ReplayUnavailable("official episode is not an ACT trajectory")

        action_meta = meta.get("actions")
        if not isinstance(action_meta, dict):
            raise ReplayUnavailable("official episode actions metadata is invalid")
        arms = [
            arm
            for arm in ("left", "right")
            if f"{arm}_arm" in action_meta
            and (path / "actions" / f"{arm}_arm.jsonl").is_file()
        ]
        if not arms:
            raise ReplayUnavailable("official episode contains no arm trajectory")

        samples: dict[str, list[dict[str, Any]]] = {}
        timestamps: dict[str, np.ndarray] = {}
        for arm in arms:
            rows = self._read_jsonl(path / "actions" / f"{arm}_arm.jsonl")
            if len(rows) < 2:
                raise ReplayUnavailable(f"official {arm} trajectory has fewer than two samples")
            try:
                arm_timestamps = np.asarray(
                    [row["timestamp"] for row in rows],
                    dtype=np.float64,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ReplayUnavailable(
                    f"official {arm} trajectory timestamps are invalid"
                ) from exc
            if not np.all(np.isfinite(arm_timestamps)) or np.any(
                np.diff(arm_timestamps) <= 0.0
            ):
                raise ReplayUnavailable(
                    f"official {arm} trajectory timestamps are not increasing"
                )
            samples[arm] = rows
            timestamps[arm] = arm_timestamps

        common_start = max(values[0] for values in timestamps.values())
        common_end = min(values[-1] for values in timestamps.values())
        sample_count = min(len(samples[arm]) for arm in arms)
        if sample_count < 2 or common_end <= common_start:
            raise ReplayUnavailable("official arm trajectories do not overlap")
        # Controller streams are sampled independently and can be offset by a
        # fraction of one frame. A shared overlap timeline preserves the
        # recorded duration while keeping both arms synchronized for replay.
        timeline = np.linspace(common_start, common_end, sample_count)

        interpolated: dict[str, dict[str, np.ndarray]] = {}
        for arm in arms:
            source_rows = samples[arm]
            try:
                joints = np.asarray(
                    [row["joint_positions"] for row in source_rows],
                    dtype=np.float64,
                )
                gripper = np.asarray(
                    [row["gripper"] for row in source_rows],
                    dtype=np.float64,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ReplayUnavailable(
                    f"official {arm} trajectory samples are invalid"
                ) from exc
            if joints.shape != (len(source_rows), 6) or gripper.shape != (
                len(source_rows),
            ):
                raise ReplayUnavailable(
                    f"official {arm} trajectory samples have invalid shapes"
                )
            official_gripper = np.interp(timeline, timestamps[arm], gripper)
            if not np.all(np.isfinite(official_gripper)):
                raise ReplayUnavailable(
                    f"official {arm} trajectory gripper is invalid"
                )
            if (
                float(np.min(official_gripper)) < -0.005
                or float(np.max(official_gripper)) > self.gripper_max_m + 0.005
            ):
                raise ReplayUnavailable(
                    f"official {arm} trajectory gripper is out of range"
                )
            if float(np.max(np.abs(np.diff(official_gripper)))) > (
                self.max_recorded_gripper_step_m
            ):
                raise ReplayUnavailable(
                    f"official {arm} trajectory has an unsafe gripper step"
                )
            # The official feedback occasionally contains a physically valid
            # step a few percent above the configured servo velocity limit.
            # Slew-limit only that gripper channel; never relax the joint or
            # discontinuity gates. This keeps replay commands within the same
            # safety envelope while preserving the recorded target sequence.
            limited_gripper = official_gripper.copy()
            for index, delta_s in enumerate(np.diff(timeline), start=1):
                maximum_delta = (
                    self.max_recorded_gripper_velocity_m_s * delta_s * (1.0 - 1e-9)
                )
                limited_gripper[index] = np.clip(
                    limited_gripper[index],
                    limited_gripper[index - 1] - maximum_delta,
                    limited_gripper[index - 1] + maximum_delta,
                )
            interpolated[arm] = {
                "joints": np.column_stack(
                    [
                        np.interp(timeline, timestamps[arm], joints[:, index])
                        for index in range(6)
                    ]
                ),
                # Legacy aligned episodes store a raw sensor range and the
                # common command path multiplies by gripper_scale. Normalize
                # official metre values here so that scale is not applied
                # twice during replay.
                "gripper": limited_gripper / self.gripper_scale,
            }

        rows = [
            {
                "timestamp": float(timestamp),
                "observation": {
                    arm: {
                        "joint_positions": interpolated[arm]["joints"][index].tolist(),
                        "gripper": float(interpolated[arm]["gripper"][index]),
                    }
                    for arm in arms
                },
            }
            for index, timestamp in enumerate(timeline)
        ]
        return arms, rows, "official_follower_observation"

    def preflight(
        self,
        recording_id: str,
        *,
        replay_gripper: bool | None = None,
    ) -> dict[str, Any]:
        effective_replay_gripper = (
            self.replay_gripper if replay_gripper is None else replay_gripper
        )
        path, arms, rows, trajectory_source = self._load(
            recording_id,
            replay_gripper=effective_replay_gripper,
        )
        source_rows = rows
        rows, source_frame_numbers, profile = self._apply_recording_profile(
            recording_id,
            arms,
            rows,
            replay_gripper=effective_replay_gripper,
        )
        timestamps = [float(row["timestamp"]) for row in rows]
        source_timestamps = [float(row["timestamp"]) for row in source_rows]
        effective_gripper_arms = tuple(
            arm
            for arm in arms
            if effective_replay_gripper and arm in self.gripper_replay_arms
        )
        return {
            "ok": True,
            "preflight": {
                "recording_id": recording_id,
                "path": str(path),
                "arms": arms,
                "sample_count": len(rows),
                "source_sample_count": len(source_rows),
                "recorded_duration_s": source_timestamps[-1] - source_timestamps[0],
                "default_replay_duration_s": timestamps[-1] - timestamps[0],
                "speed_scale": profile["speed_scale"],
                "retained_frame_stride": profile["frame_stride"],
                "retained_frame_parity": profile["frame_parity"],
                "first_source_frame_1_based": source_frame_numbers[0],
                "last_source_frame_1_based": source_frame_numbers[-1],
                "discarded_leading_source_frame_count": (
                    source_frame_numbers[0] - 1
                ),
                "suction_replayed": (
                    profile["suction_release_frame_1_based"] is not None
                ),
                "suction_release_frame_1_based": profile[
                    "suction_release_frame_1_based"
                ],
                "suction_release_requires_engaged_at_start": (
                    profile["suction_release_frame_1_based"] is not None
                ),
                "max_tracking_error_rad": (
                    profile["max_tracking_error_rad"]
                    if recording_id in self.recording_profiles
                    else self.max_tracking_error_rad
                ),
                "gripper_replayed": bool(effective_gripper_arms),
                "gripper_replay_arms": list(effective_gripper_arms),
                "left_gripper_commands_forbidden": True,
                "gripper_feedback_every_n": self.gripper_feedback_every_n,
                "max_gripper_tracking_error_m": profile[
                    "max_gripper_tracking_error_m"
                ],
                "gripper_scale": self.gripper_scale,
                "gripper_range_m": [self.gripper_min_m, self.gripper_max_m],
                "requires_near_start_pose": False,
                "automatically_positions_at_first_frame": True,
                "trajectory_source": trajectory_source,
                "confirmation_text": recording_id,
            },
        }

    def start(
        self,
        *,
        recording_id: str,
        confirmation: str,
        replay_gripper: bool | None = None,
        allow_suction_engaged: bool = False,
        max_tracking_error_rad: float | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise ReplayUnavailable("trajectory replay is disabled")
        if confirmation != recording_id:
            raise ValueError("confirmation must exactly match recording_id")
        effective_replay_gripper = (
            self.replay_gripper if replay_gripper is None else replay_gripper
        )
        path, arms, rows, trajectory_source = self._load(
            recording_id,
            replay_gripper=effective_replay_gripper,
        )
        source_sample_count = len(rows)
        rows, source_frame_numbers, profile = self._apply_recording_profile(
            recording_id,
            arms,
            rows,
            replay_gripper=effective_replay_gripper,
        )
        release_frame = profile["suction_release_frame_1_based"]
        effective_tracking_error = (
            profile["max_tracking_error_rad"]
            if max_tracking_error_rad is None
            and recording_id in self.recording_profiles
            else (
                self.max_tracking_error_rad
                if max_tracking_error_rad is None
                else float(max_tracking_error_rad)
            )
        )
        if effective_tracking_error is not None and not (
            0.05 <= effective_tracking_error <= 1.0
        ):
            raise ValueError("max_tracking_error_rad must be between 0.05 and 1.0")
        if release_frame is not None:
            self._require_suction_ready_for_profile()
        pending_session = ReplaySession(
            recording_id=recording_id,
            episode_path=path,
            speed_scale=profile["speed_scale"],
            max_tracking_error_rad=effective_tracking_error,
            requested_at=time.time(),
            gripper_replayed=effective_replay_gripper,
            max_gripper_tracking_error_m=profile[
                "max_gripper_tracking_error_m"
            ],
            allow_suction_engaged=(allow_suction_engaged or release_frame is not None),
            source_sample_count=source_sample_count,
            source_frame_numbers_1_based=source_frame_numbers,
            current_source_frame_1_based=source_frame_numbers[0],
            retained_frame_stride=profile["frame_stride"],
            retained_frame_parity=profile["frame_parity"],
            suction_release_frame_1_based=release_frame,
            suction_release_state=(
                "pending" if release_frame is not None else "not_configured"
            ),
        )
        blocker = self._interlock_blocker(pending_session)
        if blocker:
            raise ReplayConflict(blocker)
        effective_gripper_arms = tuple(
            arm
            for arm in arms
            if effective_replay_gripper and arm in self.gripper_replay_arms
        )
        pending_session.gripper_replayed = bool(effective_gripper_arms)
        pending_session.gripper_replay_arms = effective_gripper_arms
        with self._lock:
            if self._session is not None and self._session.state in ACTIVE_STATES:
                raise ReplayConflict("another replay is already active")
            session = pending_session
            session.episode_path = path
            session.sample_count = len(rows)
            session.trajectory_source = trajectory_source
            worker = threading.Thread(
                target=self._run,
                args=(session,),
                name=f"trajectory-replay-{recording_id}",
                daemon=True,
            )
            session.worker = worker
            self._session = session
            worker.start()
            return session.snapshot()

    def wait(self, recording_id: str, *, timeout_s: float) -> dict[str, Any]:
        if not math.isfinite(timeout_s) or not 0.1 <= timeout_s <= 300.0:
            raise ValueError("timeout_s must be between 0.1 and 300 seconds")
        with self._lock:
            session = self._session or self._last_session
            if session is None or session.recording_id != recording_id:
                raise ReplayConflict("requested replay session is unavailable")
            worker = session.worker
        if worker is not None:
            worker.join(timeout=timeout_s)
        snapshot = session.snapshot()
        if snapshot["active"]:
            session.state = "stopping"
            session.stop_event.set()
            if worker is not None:
                worker.join(timeout=5.0)
            raise ReplayConflict("trajectory replay did not finish before timeout")
        if snapshot["state"] != "completed":
            raise ReplayUnavailable(
                str(snapshot.get("error") or f"trajectory replay {snapshot['state']}")
            )
        return snapshot

    def stop(self) -> dict[str, Any]:
        with self._lock:
            session = self._session
            if session is None or session.state not in ACTIVE_STATES:
                raise ReplayConflict("no replay is active")
            session.state = "stopping"
            session.stop_event.set()
            return session.snapshot()

    @staticmethod
    def _enum_name(value: Any) -> str:
        return getattr(value, "name", str(value).split(".")[-1])

    @staticmethod
    def _disconnect(arm: Any) -> None:
        for method_name in ("disconnect", "close", "shutdown"):
            method = getattr(arm, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass
                return

    def _hold(self, arms: dict[str, Any]) -> None:
        try:
            _planning_mode, servo_mode = self._motion_modes()
        except Exception:
            servo_mode = None
        for arm in arms.values():
            try:
                joints = [float(value) for value in arm.get_joint_pos()]
                if servo_mode is not None:
                    arm.switch_mode(servo_mode)
                arm.servo_joint_pos(joints)
            except Exception:
                pass

    def _motion_modes(self) -> tuple[Any, Any]:
        if self._planning_mode is not None and self._servo_mode is not None:
            return self._planning_mode, self._servo_mode
        try:
            from airbot_py.arm import RobotMode
        except ImportError as exc:  # pragma: no cover
            raise ReplayUnavailable("AIRBOT replay modes are unavailable") from exc
        return (
            self._planning_mode or RobotMode.PLANNING_POS,
            self._servo_mode or RobotMode.SERVO_JOINT_POS,
        )

    def _replay_speed_profile(self) -> Any:
        if self._speed_profile is not None:
            return self._speed_profile
        try:
            from airbot_py.arm import SpeedProfile
        except ImportError as exc:  # pragma: no cover
            raise ReplayUnavailable("AIRBOT replay speed profile is unavailable") from exc
        return SpeedProfile.FAST

    def _first_frame_speed_profile(self) -> Any:
        if self._initial_speed_profile is not None:
            return self._initial_speed_profile
        try:
            from airbot_py.arm import SpeedProfile
        except ImportError as exc:  # pragma: no cover
            raise ReplayUnavailable("AIRBOT positioning speed profile is unavailable") from exc
        return SpeedProfile.SLOW

    @staticmethod
    def _start_planning_move(arm: Any, target: np.ndarray) -> Any:
        move_method = getattr(arm, "move_to_joint_pos", None)
        if not callable(move_method):
            raise ReplayUnavailable("follower arm lacks joint-position planning")
        # AIRBOT exposes move_to_joint_pos as a streaming RPC.  With
        # blocking=False the SDK returns after ACCEPTED and leaves the result
        # stream open, so a following SetMode(SERVO_JOINT_POS) can be rejected
        # even after feedback already reports IDLE.  Drain the stream through
        # FINISHED before changing controller mode.  Both arms still move
        # concurrently because callers run this function in a thread pool.
        return move_method(target.tolist(), blocking=True)

    @staticmethod
    def _start_gripper_move(arm: Any, target_m: float) -> Any:
        move_method = getattr(arm, "move_eef_pos", None)
        if not callable(move_method):
            raise ReplayUnavailable("follower arm lacks gripper-position planning")
        return move_method(float(target_m), blocking=True)

    @staticmethod
    def _send_servo_target(
        arm: Any,
        joints: np.ndarray,
        gripper_m: float | None,
    ) -> bool:
        if arm.servo_joint_pos(joints.tolist()) is False:
            return False
        if gripper_m is not None:
            gripper_method = getattr(arm, "servo_eef_pos", None)
            if not callable(gripper_method):
                raise ReplayUnavailable("follower arm lacks gripper servo control")
            if gripper_method(float(gripper_m)) is False:
                return False
        return True

    def _position_grippers_at_first_frame(
        self,
        session: ReplaySession,
        arms: dict[str, Any],
        targets: dict[str, float],
    ) -> None:
        """Position and verify only explicitly allow-listed grippers."""

        with ThreadPoolExecutor(max_workers=len(arms)) as command_pool:
            futures = {
                name: command_pool.submit(
                    self._start_gripper_move,
                    arms[name],
                    targets[name],
                )
                for name in arms
            }
            for name, future in futures.items():
                if future.result() is False:
                    raise ReplayUnavailable(
                        f"{name} gripper rejected automatic first-frame positioning"
                    )
        for name, arm in arms.items():
            self._check_gripper_tracking(session, name, arm, targets[name])

    @staticmethod
    def _read_gripper_position(arm: Any, name: str) -> float:
        feedback_method = getattr(arm, "get_eef_pos", None)
        if not callable(feedback_method):
            raise ReplayUnavailable(f"{name} arm lacks gripper-position feedback")
        feedback = np.asarray(feedback_method(), dtype=np.float64).reshape(-1)
        if feedback.size < 1 or not math.isfinite(float(feedback[0])):
            raise ReplayUnavailable(f"{name} arm returned invalid gripper feedback")
        return float(feedback[0])

    def _check_gripper_tracking(
        self,
        session: ReplaySession,
        name: str,
        arm: Any,
        target_m: float,
    ) -> None:
        current_m = self._read_gripper_position(arm, name)
        error_m = abs(current_m - float(target_m))
        session.gripper_last_tracking_error_m = error_m
        session.gripper_max_tracking_error_m = max(
            session.gripper_max_tracking_error_m,
            error_m,
        )
        if (
            session.max_gripper_tracking_error_m is not None
            and error_m > session.max_gripper_tracking_error_m
        ):
            raise ReplayUnavailable(
                f"{name} gripper tracking error {error_m:.4f} m exceeds limit "
                f"{session.max_gripper_tracking_error_m:.4f} m"
            )

    def _position_at_first_frame(
        self,
        session: ReplaySession,
        arms: dict[str, Any],
        targets: dict[str, np.ndarray],
        planning_mode: Any,
    ) -> bool:
        """Move both arms concurrently to the episode's first action sample."""

        session.state = "positioning"
        positioning_speed = self._first_frame_speed_profile()
        for name, arm in arms.items():
            if arm.switch_mode(planning_mode) is not True:
                raise ReplayUnavailable(f"{name} arm rejected PLANNING_POS")
            mode_method = getattr(arm, "get_control_mode", None)
            if callable(mode_method) and self._enum_name(mode_method()) != "PLANNING_POS":
                raise ReplayUnavailable(f"{name} arm did not enter PLANNING_POS")
            set_speed_profile = getattr(arm, "set_speed_profile", None)
            if not callable(set_speed_profile):
                raise ReplayUnavailable(f"{name} arm lacks a speed profile API")
            if set_speed_profile(positioning_speed) is False:
                raise ReplayUnavailable(
                    f"{name} arm rejected slow first-frame positioning profile"
                )

        with ThreadPoolExecutor(max_workers=len(arms)) as command_pool:
            futures = {
                name: command_pool.submit(
                    self._start_planning_move,
                    arms[name],
                    targets[name],
                )
                for name in arms
            }
            for name, future in futures.items():
                if future.result() is False:
                    raise ReplayUnavailable(
                        f"{name} arm rejected automatic first-frame positioning"
                    )

        stable = {name: 0 for name in arms}
        deadline = time.monotonic() + self.initial_move_timeout_s
        while time.monotonic() < deadline:
            if session.stop_event.is_set():
                return False
            blocker = self._interlock_blocker(session)
            if blocker:
                raise ReplayConflict(blocker)
            all_reached = True
            diagnostics = []
            for name, arm in arms.items():
                current = np.asarray(arm.get_joint_pos(), dtype=np.float64)
                if current.shape != (6,) or not np.all(np.isfinite(current)):
                    raise ReplayUnavailable(f"{name} arm returned invalid joint feedback")
                error = float(np.max(np.abs(current - targets[name])))
                state_method = getattr(arm, "get_state", None)
                state = self._enum_name(state_method()) if callable(state_method) else "IDLE"
                reached = state == "IDLE" and error <= self.initial_joint_tolerance_rad
                stable[name] = stable[name] + 1 if reached else 0
                if stable[name] < self.initial_feedback_stable_samples:
                    all_reached = False
                diagnostics.append(f"{name}: state={state}, error={error:.3f} rad")
            if all_reached:
                return True
            session.stop_event.wait(self.initial_feedback_poll_s)
        raise ReplayUnavailable(
            "automatic first-frame positioning did not settle ("
            + "; ".join(diagnostics)
            + ")"
        )

    def _run(self, session: ReplaySession) -> None:
        arms: dict[str, Any] = {}
        try:
            _path, arm_names, rows, _trajectory_source = self._load(
                session.recording_id,
                replay_gripper=session.gripper_replayed,
            )
            rows, source_frame_numbers, profile = self._apply_recording_profile(
                session.recording_id,
                arm_names,
                rows,
                replay_gripper=session.gripper_replayed,
            )
            if source_frame_numbers != session.source_frame_numbers_1_based:
                raise ReplayUnavailable("replay profile changed after preflight")
            release_frame = profile["suction_release_frame_1_based"]
            release_index = (
                source_frame_numbers.index(release_frame)
                if release_frame is not None
                else None
            )
            blocker = self._interlock_blocker(session)
            if blocker:
                raise ReplayConflict(blocker)
            timestamps = [float(row["timestamp"]) for row in rows]
            origin = timestamps[0]
            relative = [value - origin for value in timestamps]
            replay_joints = {
                name: np.asarray(
                    [row["observation"][name]["joint_positions"] for row in rows],
                    dtype=np.float64,
                )
                for name in arm_names
            }
            replay_grippers = (
                {
                    name: np.clip(
                        np.asarray(
                            [row["observation"][name]["gripper"] for row in rows],
                            dtype=np.float64,
                        )
                        * self.gripper_scale,
                        self.gripper_min_m,
                        self.gripper_max_m,
                    )
                    for name in session.gripper_replay_arms
                }
                if session.gripper_replayed
                else {}
            )
            for name in arm_names:
                arm = self._arm_factory(self.arm_host, self.arm_ports[name])
                if arm.connect() is not True:
                    raise ReplayUnavailable(f"cannot connect to follower {name} arm")
                arms[name] = arm
            with self._lock:
                self._active_arms = dict(arms)
            for name, arm in arms.items():
                state_method = getattr(arm, "get_state", None)
                if callable(state_method) and self._enum_name(state_method()) != "IDLE":
                    raise ReplayConflict(f"{name} follower arm must be IDLE")
            planning_mode, servo_mode = self._motion_modes()
            positioned = self._position_at_first_frame(
                session,
                arms,
                {name: replay_joints[name][0] for name in arm_names},
                planning_mode,
            )
            if not positioned:
                self._hold(arms)
                session.finished_at = time.time()
                session.state = "stopped"
                return
            if session.gripper_replayed:
                self._position_grippers_at_first_frame(
                    session,
                    {name: arms[name] for name in session.gripper_replay_arms},
                    {
                        name: float(replay_grippers[name][0])
                        for name in session.gripper_replay_arms
                    },
                )
            for name, arm in arms.items():
                if arm.switch_mode(servo_mode) is not True:
                    raise ReplayUnavailable(f"{name} arm rejected SERVO_JOINT_POS")
                mode_method = getattr(arm, "get_control_mode", None)
                if callable(mode_method) and self._enum_name(mode_method()) != "SERVO_JOINT_POS":
                    raise ReplayUnavailable(f"{name} arm did not enter SERVO_JOINT_POS")
            # The demonstrations are collected by the factory Follow worker
            # with SpeedProfile.FAST.  Replaying under the SDK default joint
            # scale (0.1 instead of 1.0) creates artificial lag and trips the
            # tracking-error gate during otherwise valid fast segments.
            speed_profile = self._replay_speed_profile()
            for name, arm in arms.items():
                set_speed_profile = getattr(arm, "set_speed_profile", None)
                if not callable(set_speed_profile):
                    raise ReplayUnavailable(f"{name} arm lacks a speed profile API")
                if set_speed_profile(speed_profile) is False:
                    raise ReplayUnavailable(f"{name} arm rejected replay speed profile")
            session.started_at = time.time()
            session.state = "replaying"
            replay_started = time.monotonic()
            command_period = 1.0 / self.command_hz
            command_index = 0
            suction_release_future = None
            with (
                ThreadPoolExecutor(max_workers=len(arms)) as command_pool,
                ThreadPoolExecutor(max_workers=1) as suction_pool,
            ):
                while not session.stop_event.is_set():
                    elapsed = time.monotonic() - replay_started
                    source_time = elapsed
                    if source_time >= relative[-1]:
                        source_time = relative[-1]
                        finished = True
                    else:
                        finished = False
                    right = bisect.bisect_left(relative, source_time)
                    if right <= 0:
                        left = right = 0
                        alpha = 0.0
                    elif right >= len(relative):
                        left = right = len(relative) - 1
                        alpha = 0.0
                    else:
                        left = right - 1
                        alpha = (source_time - relative[left]) / (
                            relative[right] - relative[left]
                        )
                    targets = {
                        name: (
                            replay_joints[name][left]
                            + alpha * (replay_joints[name][right] - replay_joints[name][left])
                        )
                        for name in arm_names
                    }
                    gripper_targets = (
                        {
                            name: float(
                                replay_grippers[name][left]
                                + alpha
                                * (
                                    replay_grippers[name][right]
                                    - replay_grippers[name][left]
                                )
                            )
                            for name in session.gripper_replay_arms
                        }
                        if session.gripper_replayed
                        else {}
                    )
                    blocker = self._interlock_blocker(session)
                    if blocker:
                        raise ReplayConflict(blocker)
                    futures = {
                        name: command_pool.submit(
                            self._send_servo_target,
                            arms[name],
                            targets[name],
                            gripper_targets.get(name),
                        )
                        for name in arm_names
                    }
                    for name, future in futures.items():
                        if future.result() is False:
                            raise ReplayUnavailable(
                                f"{name} arm or gripper rejected replay target"
                            )
                    session.current_index = min(right, len(rows) - 1)
                    session.current_source_frame_1_based = source_frame_numbers[
                        session.current_index
                    ]
                    if (
                        release_index is not None
                        and suction_release_future is None
                        and source_time >= relative[release_index]
                    ):
                        session.suction_release_state = "releasing"
                        session.suction_release_started_at = time.time()
                        suction_release_future = suction_pool.submit(
                            self._release_suction
                        )
                    if (
                        suction_release_future is not None
                        and suction_release_future.done()
                        and session.suction_release_state != "released"
                    ):
                        suction_release_future.result()
                        session.suction_release_state = "released"
                        session.suction_released_at = time.time()
                        session.allow_suction_engaged = False
                    if command_index % self.feedback_every_n == 0:
                        for name, arm in arms.items():
                            current = np.asarray(arm.get_joint_pos(), dtype=np.float64)
                            tracking_error = float(
                                np.max(np.abs(current - targets[name]))
                            )
                            if (
                                session.max_tracking_error_rad is not None
                                and tracking_error > session.max_tracking_error_rad
                            ):
                                raise ReplayUnavailable(
                                    f"{name} tracking error {tracking_error:.3f} rad exceeds limit"
                                )
                    if (
                        session.gripper_replayed
                        and command_index % self.gripper_feedback_every_n == 0
                    ):
                        for name in session.gripper_replay_arms:
                            self._check_gripper_tracking(
                                session,
                                name,
                                arms[name],
                                gripper_targets[name],
                            )
                    command_index += 1
                    if finished:
                        break
                    session.stop_event.wait(command_period)
                if release_index is not None:
                    if suction_release_future is None:
                        raise ReplayUnavailable(
                            "replay ended before configured suction release frame"
                        )
                    suction_release_future.result()
                    session.suction_release_state = "released"
                    session.suction_released_at = (
                        session.suction_released_at or time.time()
                    )
                    session.allow_suction_engaged = False
            self._hold(arms)
            session.finished_at = time.time()
            session.state = "stopped" if session.stop_event.is_set() else "completed"
        except Exception as exc:
            self._hold(arms)
            session.error = f"{type(exc).__name__}: {exc}"
            session.finished_at = time.time()
            session.state = "error"
        finally:
            for arm in arms.values():
                self._disconnect(arm)
            with self._lock:
                self._active_arms = {}
                self._last_session = session
                self._session = None

    def close(self, *, timeout_s: float = 5.0) -> None:
        with self._lock:
            session = self._session
            if session is None:
                return
            session.stop_event.set()
            worker = session.worker
        if worker is not None:
            worker.join(timeout=max(0.1, timeout_s))
