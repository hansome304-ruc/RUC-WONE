"""Closed-loop ACT execution and trace capture for the packaging console.

The controller owns both follower-arm command connections while a rollout is
active. Interlocks are checked before every send, stop synchronously serializes
a hold-current command after any in-flight servo RPC, and optional per-session
traces preserve model outputs and command timing without writing on the control
thread.
"""
from __future__ import annotations

import json
import math
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ACTIVE_STATES = frozenset({"starting", "running", "stopping"})
ARM_NAMES = ("left", "right")
CAMERA_MAP = {
    "cam_high": "front",
    "cam_left_wrist": "left_wrist",
    "cam_right_wrist": "right_wrist",
}


class ActRolloutError(RuntimeError):
    """Base class for rollout errors exposed by the local API."""


class ActRolloutConflict(ActRolloutError):
    """Another controller currently owns the robot."""


class ActRolloutUnavailable(ActRolloutError):
    """The model, cameras, or robot endpoints are unavailable."""


class ActRolloutSafetyViolation(ActRolloutError):
    """A bounded rollout safety check rejected the request or model output."""


@dataclass
class ActRolloutSession:
    session_id: str
    requested_at: float
    execute_steps_per_inference: int
    state: str = "starting"
    started_at: float | None = None
    finished_at: float | None = None
    inference_count: int = 0
    command_count: int = 0
    last_inference_ms: float | None = None
    last_observation_age_ms: float | None = None
    last_camera_arm_delta_ms: float | None = None
    last_max_command_step_rad: float | None = None
    last_tracking_errors_rad: dict[str, float] | None = None
    max_tracking_errors_rad: dict[str, float] = field(
        default_factory=lambda: {"left": 0.0, "right": 0.0}
    )
    tracking_warning_count: int = 0
    start_pose_diagnostic: dict[str, Any] | None = None
    error: str | None = None
    stop_reason: str | None = None
    hold_requested_at: float | None = None
    hold_completed_at: float | None = None
    hold_confirmed: bool = False
    debug_log_path: str | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    worker: threading.Thread | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "active": self.state in ACTIVE_STATES,
            "state": self.state,
            "session_id": self.session_id,
            "requested_at": self.requested_at,
            "execute_steps_per_inference": self.execute_steps_per_inference,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "inference_count": self.inference_count,
            "command_count": self.command_count,
            "last_inference_ms": self.last_inference_ms,
            "last_observation_age_ms": self.last_observation_age_ms,
            "last_camera_arm_delta_ms": self.last_camera_arm_delta_ms,
            "last_max_command_step_rad": self.last_max_command_step_rad,
            "last_tracking_errors_rad": self.last_tracking_errors_rad,
            "max_tracking_errors_rad": dict(self.max_tracking_errors_rad),
            "tracking_warning_count": self.tracking_warning_count,
            "start_pose_diagnostic": self.start_pose_diagnostic,
            "error": self.error,
            "stop_reason": self.stop_reason,
            "hold_requested_at": self.hold_requested_at,
            "hold_completed_at": self.hold_completed_at,
            "hold_confirmed": self.hold_confirmed,
            "debug_log_path": self.debug_log_path,
        }


class ActRolloutController:
    """Continuously replan short ACT chunks with bounded joint servo output."""

    def __init__(
        self,
        config: dict[str, Any] | None,
        *,
        inference: Any,
        frame_provider: Callable[..., Mapping[str, Any]],
        interlock: Callable[[], str | None],
        start_pose_checker: Callable[[Sequence[float]], dict[str, Any]],
        arm_factory: Callable[[str, int], Any] | None = None,
        servo_mode: Any | None = None,
        speed_profile: Any | None = None,
    ) -> None:
        cfg = dict(config or {})
        self.enabled = cfg.get("enabled") is True
        self.speed_profile_name = str(
            cfg.get("speed_profile", "DEFAULT")
        ).strip().upper()
        if self.speed_profile_name not in {"SLOW", "DEFAULT", "FAST"}:
            raise ValueError(
                "act_rollout.speed_profile must be SLOW, DEFAULT or FAST"
            )
        self.inference = inference
        self.frame_provider = frame_provider
        self.interlock = interlock
        self.start_pose_checker = start_pose_checker
        self.arm_factory = arm_factory or self._default_arm_factory
        self.arm_host = str(cfg.get("arm_host", "localhost"))
        raw_ports = cfg.get("arm_ports", {"left": 50051, "right": 50053})
        if not isinstance(raw_ports, dict):
            raise ValueError("act_rollout.arm_ports must be an object")
        self.arm_ports = {
            "left": int(raw_ports.get("left", 50051)),
            "right": int(raw_ports.get("right", 50053)),
        }
        self.command_hz = float(cfg.get("command_hz", 30.0))
        self.horizon = int(cfg.get("horizon", 25))
        self.execute_steps = int(cfg.get("execute_steps_per_inference", 3))
        self.max_tracking_error_rad = float(
            cfg.get("max_tracking_error_rad", 0.20)
        )
        self.max_command_step_rad = float(
            cfg.get("max_command_step_rad", 0.15)
        )
        self.tracking_error_blocking = (
            cfg.get("tracking_error_blocking", True) is True
        )
        self.feedback_every_n = int(cfg.get("feedback_every_n", 3))
        self.max_inference_s = float(cfg.get("max_inference_s", 0.50))
        self.max_observation_age_s = float(
            cfg.get("max_observation_age_s", 0.25)
        )
        self.max_camera_arm_delta_ms = float(
            cfg.get("max_camera_arm_delta_ms", 60.0)
        )
        self.camera_arm_timing_blocking = (
            cfg.get("camera_arm_timing_blocking", True) is True
        )
        self.feedback_sample_hz = float(cfg.get("feedback_sample_hz", 50.0))
        self.feedback_history_warmup_s = float(
            cfg.get("feedback_history_warmup_s", 0.15)
        )
        self.post_chunk_delay_s = float(cfg.get("post_chunk_delay_s", 0.0))
        self.start_joint_outside_margin_rad = float(
            cfg.get("start_joint_outside_margin_rad", 0.12)
        )
        self.start_gripper_outside_margin_m = float(
            cfg.get("start_gripper_outside_margin_m", 0.015)
        )
        self.gripper_scale = float(cfg.get("gripper_scale", 0.072 / 0.0471))
        self.gripper_min_m = float(cfg.get("gripper_min_m", 0.0))
        self.gripper_max_m = float(cfg.get("gripper_max_m", 0.072))
        self.joint_absolute_limit_rad = float(
            cfg.get("joint_absolute_limit_rad", math.pi + 0.2)
        )
        self.debug_log_enabled = cfg.get("debug_log_enabled") is True
        raw_debug_log_dir = str(cfg.get("debug_log_dir", "")).strip()
        self.debug_log_dir = (
            Path(raw_debug_log_dir).expanduser().resolve()
            if self.debug_log_enabled and raw_debug_log_dir
            else None
        )
        if not (
            10.0 <= self.command_hz <= 60.0
            and 1 <= self.horizon <= 100
            and 1 <= self.execute_steps <= self.horizon
            and 0.03 <= self.max_tracking_error_rad <= 0.5
            and 0.01 <= self.max_command_step_rad <= math.radians(40.0)
            and 1 <= self.feedback_every_n <= 30
            and 0.05 <= self.max_inference_s <= 3.0
            and 0.05 <= self.max_observation_age_s <= 1.0
            and 10.0 <= self.max_camera_arm_delta_ms <= 200.0
            and 20.0 <= self.feedback_sample_hz <= 200.0
            and 0.05 <= self.feedback_history_warmup_s <= 0.5
            and 0.0 <= self.post_chunk_delay_s <= 1.0
            and 0.01 <= self.start_joint_outside_margin_rad <= 0.5
            and 0.001 <= self.start_gripper_outside_margin_m <= 0.05
            and 0.1 <= self.gripper_scale <= 5.0
            and 0.0 <= self.gripper_min_m < self.gripper_max_m <= 0.2
            and math.pi <= self.joint_absolute_limit_rad <= 2.0 * math.pi
        ):
            raise ValueError("invalid act_rollout safety settings")

        self._servo_mode = servo_mode
        self._speed_profile = speed_profile
        self._lock = threading.RLock()
        self._command_lock = threading.Lock()
        self._session: ActRolloutSession | None = None
        self._last_session: ActRolloutSession | None = None
        self._active_arms: dict[str, Any] = {}
        self._motion_armed = False
        self._trace_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="act-rollout-trace",
        )
        history_size = max(
            16,
            int(math.ceil(self.feedback_sample_hz * self.max_observation_age_s))
            + 8,
        )
        self._feedback_history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._feedback_history_lock = threading.Lock()

    @staticmethod
    def _default_arm_factory(host: str, port: int) -> Any:
        try:
            from airbot_py.arm import AIRBOTPlay
        except ImportError as exc:  # pragma: no cover - hardware only
            raise ActRolloutUnavailable("airbot_py is unavailable") from exc
        return AIRBOTPlay(url=host, port=port)

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

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _write_trace_file(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _submit_trace(
        self,
        session: ActRolloutSession,
        filename: str,
        payload: Any,
    ) -> None:
        if not session.debug_log_path:
            return
        path = Path(session.debug_log_path) / filename
        safe_payload = self._json_safe(payload)
        try:
            self._trace_executor.submit(
                self._write_trace_file,
                path,
                safe_payload,
            )
        except RuntimeError:
            # Trace persistence is diagnostic-only and must never affect motion.
            pass

    def _prepare_trace(self, session: ActRolloutSession) -> None:
        if self.debug_log_dir is None:
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(session.requested_at))
        path = self.debug_log_dir / f"{timestamp}_{session.session_id}"
        try:
            path.mkdir(parents=True, exist_ok=False)
        except OSError:
            return
        session.debug_log_path = str(path)
        self._submit_trace(
            session,
            "session_start.json",
            {
                "schema": "medicine_act_rollout_trace_v1",
                "session_id": session.session_id,
                "requested_at": session.requested_at,
                "execute_steps_per_inference": session.execute_steps_per_inference,
                "command_hz": self.command_hz,
                "horizon": self.horizon,
                "speed_profile": self._motion_speed_profile_name(),
                "initial_action_alignment": False,
                "per_step_action_clipping": False,
                "tracking_error_blocking": self.tracking_error_blocking,
                "tracking_error_diagnostic_threshold_rad": (
                    self.max_tracking_error_rad
                ),
                "max_command_step_rad": self.max_command_step_rad,
                "post_chunk_delay_s": self.post_chunk_delay_s,
            },
        )

    def _trace_chunk(
        self,
        session: ActRolloutSession,
        chunk_index: int,
        observation: Mapping[str, Any],
        actions: np.ndarray,
    ) -> None:
        self._submit_trace(
            session,
            f"chunk_{chunk_index:06d}.json",
            {
                "schema": "medicine_act_rollout_chunk_v1",
                "session_id": session.session_id,
                "chunk_index": chunk_index,
                "captured_at": observation["timestamp"],
                "observation_state": observation["state"],
                "arm_pair_skew_ms": observation["arm_pair_skew_ms"],
                "camera_arm_delta_ms": observation["camera_arm_delta_ms"],
                "frame_timestamps": observation.get("frame_timestamps", {}),
                "frame_shapes": observation.get("frame_shapes", {}),
                "captured_after": observation.get("captured_after"),
                "post_command_frame_margin_ms": observation.get(
                    "post_command_frame_margin_ms"
                ),
                "inference_ms": session.last_inference_ms,
                "observation_age_ms": session.last_observation_age_ms,
                "actions": actions,
            },
        )

    def _trace_execution(
        self,
        session: ActRolloutSession,
        chunk_index: int,
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        self._submit_trace(
            session,
            f"execution_{chunk_index:06d}.json",
            {
                "schema": "medicine_act_rollout_execution_v1",
                "session_id": session.session_id,
                "chunk_index": chunk_index,
                "command_hz_requested": self.command_hz,
                "records": records,
            },
        )

    def _trace_summary(self, session: ActRolloutSession) -> None:
        self._submit_trace(
            session,
            "session_summary.json",
            {
                "schema": "medicine_act_rollout_summary_v1",
                **session.snapshot(),
            },
        )

    def _motion_mode(self) -> Any:
        if self._servo_mode is not None:
            return self._servo_mode
        try:
            from airbot_py.arm import RobotMode
        except ImportError as exc:  # pragma: no cover - hardware only
            raise ActRolloutUnavailable("AIRBOT servo mode is unavailable") from exc
        return RobotMode.SERVO_JOINT_POS

    def _motion_speed_profile_name(self) -> str:
        if self._speed_profile is not None:
            return self._enum_name(self._speed_profile)
        return self.speed_profile_name

    def _motion_speed_profile(self) -> Any:
        if self._speed_profile is not None:
            return self._speed_profile
        try:
            from airbot_py.arm import SpeedProfile
        except ImportError as exc:  # pragma: no cover - hardware only
            raise ActRolloutUnavailable("AIRBOT speed profile is unavailable") from exc
        try:
            return getattr(SpeedProfile, self.speed_profile_name)
        except AttributeError as exc:  # pragma: no cover - SDK contract only
            raise ActRolloutUnavailable(
                f"AIRBOT does not support {self.speed_profile_name} speed profile"
            ) from exc

    def status(self) -> dict[str, Any]:
        with self._lock:
            session = self._session or self._last_session
            payload = (
                session.snapshot()
                if session is not None
                else {
                    "enabled": self.enabled,
                    "active": False,
                    "state": "idle",
                    "session_id": None,
                    "error": None,
                    "stop_reason": None,
                    "hold_confirmed": False,
                }
            )
        payload.update(
            {
                "enabled": self.enabled,
                "execution_enabled": self.enabled,
                "command_hz": self.command_hz,
                "horizon": self.horizon,
                "execute_steps_per_inference": int(
                    payload.get("execute_steps_per_inference", self.execute_steps)
                ),
                "speed_profile": self._motion_speed_profile_name(),
                "initial_action_alignment": False,
                "per_step_action_clipping": False,
                "max_command_step_rad": self.max_command_step_rad,
                "tracking_error_blocking": self.tracking_error_blocking,
                "max_camera_arm_delta_ms": self.max_camera_arm_delta_ms,
                "camera_arm_timing_blocking": self.camera_arm_timing_blocking,
                "post_chunk_delay_s": self.post_chunk_delay_s,
                "stop_semantics": "synchronous_hold_current",
            }
        )
        return payload

    def start(
        self,
        *,
        execute_steps_per_inference: int | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise ActRolloutUnavailable("ACT rollout is disabled")
        execute_steps = (
            self.execute_steps
            if execute_steps_per_inference is None
            else execute_steps_per_inference
        )
        if (
            isinstance(execute_steps, bool)
            or not isinstance(execute_steps, int)
            or not 1 <= execute_steps <= self.horizon
        ):
            raise ValueError(
                f"execute_steps_per_inference must be an integer in [1, {self.horizon}]"
            )
        blocker = self.interlock()
        if blocker:
            raise ActRolloutConflict(blocker)
        inference_status = self.inference.status(force=True)
        if inference_status.get("ready") is not True:
            raise ActRolloutUnavailable(
                str(inference_status.get("error") or "ACT inference is not ready")
            )
        with self._lock:
            if self._session is not None and self._session.state in ACTIVE_STATES:
                raise ActRolloutConflict("ACT rollout is already active")
            session = ActRolloutSession(
                session_id=f"rollout-{uuid.uuid4().hex}",
                requested_at=time.time(),
                execute_steps_per_inference=execute_steps,
            )
            self._prepare_trace(session)
            worker = threading.Thread(
                target=self._run,
                args=(session,),
                name=f"act-rollout-{session.session_id[-8:]}",
                daemon=True,
            )
            session.worker = worker
            self._session = session
            worker.start()
            return session.snapshot()

    def stop(self, *, reason: str = "operator_stop") -> dict[str, Any]:
        with self._lock:
            session = self._session
            if session is None or session.state not in ACTIVE_STATES:
                raise ActRolloutConflict("ACT rollout is not active")
            session.state = "stopping"
            session.stop_reason = str(reason)[:128]
            session.hold_requested_at = time.time()
            session.stop_event.set()
            arms = dict(self._active_arms)
            motion_armed = self._motion_armed

        # Serialize after any in-flight servo send. Once stop_event is set the
        # worker cannot enter another send, so this hold is the final command.
        if arms and motion_armed:
            with self._command_lock:
                if not session.hold_confirmed:
                    self._hold_current(arms)
                    session.hold_completed_at = time.time()
                    session.hold_confirmed = True
        else:
            # During preflight no motion mode has been entered and no command
            # has been sent; there is therefore nothing physical to arrest.
            with self._lock:
                session.hold_completed_at = time.time()
                session.hold_confirmed = session.command_count == 0
        return session.snapshot()

    def _capture_feedback(self, arms: Mapping[str, Any]) -> dict[str, Any]:
        started_ns = time.time_ns()

        def read_one(name: str) -> tuple[np.ndarray, float, int]:
            arm = arms[name]
            read_started = time.time_ns()
            joints = np.asarray(arm.get_joint_pos(), dtype=np.float64)
            eef = np.asarray(arm.get_eef_pos(), dtype=np.float64).reshape(-1)
            read_finished = time.time_ns()
            if joints.shape != (6,) or not np.all(np.isfinite(joints)):
                raise ActRolloutUnavailable(f"{name} arm returned invalid joints")
            if eef.size < 1 or not np.all(np.isfinite(eef)):
                raise ActRolloutUnavailable(f"{name} arm returned invalid gripper feedback")
            return joints, float(eef[0]), (read_started + read_finished) // 2

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {name: pool.submit(read_one, name) for name in ARM_NAMES}
            values = {name: futures[name].result() for name in ARM_NAMES}
        finished_ns = time.time_ns()
        timestamps = [values[name][2] for name in ARM_NAMES]
        state: list[float] = []
        for name in ARM_NAMES:
            state.extend(float(value) for value in values[name][0])
            state.append(float(values[name][1]))
        return {
            "state": state,
            "timestamp": (started_ns + finished_ns) / 2e9,
            "arm_pair_skew_ms": (max(timestamps) - min(timestamps)) / 1e6,
            "joints": {name: values[name][0] for name in ARM_NAMES},
            "grippers": {name: values[name][1] for name in ARM_NAMES},
        }

    def _reset_feedback_history(self) -> None:
        with self._feedback_history_lock:
            self._feedback_history.clear()

    def _remember_feedback(self, feedback: Mapping[str, Any]) -> None:
        sample = {
            "state": list(feedback["state"]),
            "timestamp": float(feedback["timestamp"]),
            "arm_pair_skew_ms": float(feedback["arm_pair_skew_ms"]),
            "joints": {
                name: np.asarray(feedback["joints"][name], dtype=np.float64).copy()
                for name in ARM_NAMES
            },
            "grippers": {
                name: float(feedback["grippers"][name]) for name in ARM_NAMES
            },
        }
        with self._feedback_history_lock:
            self._feedback_history.append(sample)

    def _sample_feedback_history(
        self,
        arms: Mapping[str, Any],
        stop_event: threading.Event,
    ) -> None:
        period = 1.0 / self.feedback_sample_hz
        while not stop_event.is_set():
            started = time.monotonic()
            try:
                with self._command_lock:
                    feedback = self._capture_feedback(arms)
                self._remember_feedback(feedback)
            except Exception:
                # The foreground observation/command path remains authoritative
                # and will expose any persistent arm communication failure.
                pass
            remaining = period - (time.monotonic() - started)
            if remaining > 0.0:
                stop_event.wait(remaining)

    def _nearest_feedback(
        self,
        timestamp: float,
        *,
        captured_after: float | None = None,
    ) -> dict[str, Any] | None:
        with self._feedback_history_lock:
            samples = tuple(self._feedback_history)
        if captured_after is not None:
            samples = tuple(
                sample
                for sample in samples
                if float(sample["timestamp"]) > float(captured_after)
            )
        if not samples:
            return None
        return min(samples, key=lambda item: abs(float(item["timestamp"]) - timestamp))

    def _capture_observation(
        self,
        arms: Mapping[str, Any],
        *,
        first: bool,
        captured_after: float | None = None,
    ) -> dict[str, Any]:
        with ThreadPoolExecutor(max_workers=2) as pool:
            frames_future = pool.submit(
                self.frame_provider,
                first,
                captured_after=captured_after,
            )
            with self._command_lock:
                current_feedback = self._capture_feedback(arms)
            self._remember_feedback(current_feedback)
            frames = dict(frames_future.result())
        if set(frames) != set(CAMERA_MAP.values()):
            raise ActRolloutUnavailable("ACT camera bundle is incomplete")
        timestamps = np.asarray(
            [float(getattr(frames[name], "captured_at")) for name in CAMERA_MAP.values()],
            dtype=np.float64,
        )
        if captured_after is not None and float(np.min(timestamps)) <= float(
            captured_after
        ):
            raise ActRolloutSafetyViolation(
                "ACT observation predates the completed action chunk"
            )
        target_timestamp = float(np.median(timestamps))
        feedback = self._nearest_feedback(
            target_timestamp,
            captured_after=captured_after,
        ) or current_feedback
        if captured_after is not None and float(feedback["timestamp"]) <= float(
            captured_after
        ):
            raise ActRolloutSafetyViolation(
                "ACT arm feedback predates the completed action chunk"
            )
        camera_arm_delta_ms = float(
            np.max(np.abs(timestamps - float(feedback["timestamp"]))) * 1000.0
        )
        if not np.all(np.isfinite(timestamps)) or not math.isfinite(
            camera_arm_delta_ms
        ):
            raise ActRolloutUnavailable("ACT observation timestamps are invalid")
        if (
            self.camera_arm_timing_blocking
            and camera_arm_delta_ms > self.max_camera_arm_delta_ms
        ):
            raise ActRolloutSafetyViolation(
                "ACT camera/arm timing is stale: "
                f"{camera_arm_delta_ms:.1f} ms > {self.max_camera_arm_delta_ms:.1f} ms"
            )
        feedback["frames_bgr"] = {
            model_name: np.asarray(frames[source_name].bgr).copy()
            for model_name, source_name in CAMERA_MAP.items()
        }
        feedback["frame_timestamps"] = {
            model_name: float(getattr(frames[source_name], "captured_at"))
            for model_name, source_name in CAMERA_MAP.items()
        }
        feedback["frame_shapes"] = {
            model_name: list(np.asarray(frames[source_name].bgr).shape)
            for model_name, source_name in CAMERA_MAP.items()
        }
        feedback["camera_arm_delta_ms"] = camera_arm_delta_ms
        feedback["captured_after"] = captured_after
        feedback["post_command_frame_margin_ms"] = (
            None
            if captured_after is None
            else (float(np.min(timestamps)) - float(captured_after)) * 1000.0
        )
        return feedback

    def _checked_start_pose(self, state: Sequence[float]) -> dict[str, Any]:
        diagnostic = dict(self.start_pose_checker(state))
        outside_margin = []
        for item in diagnostic.get("out_of_range", []):
            try:
                index = int(item["index"])
                distance = float(item["distance_to_range"])
            except (KeyError, TypeError, ValueError):
                raise ActRolloutSafetyViolation("start-pose diagnostic is malformed")
            limit = (
                self.start_gripper_outside_margin_m
                if index in (6, 13)
                else self.start_joint_outside_margin_rad
            )
            if distance > limit:
                outside_margin.append({**item, "diagnostic_margin": limit})
        # The rollout now performs a one-time, rate-limited interpolation to
        # the first predicted action. Training-start coverage is therefore
        # useful operator context, but must not reject an otherwise valid
        # observation before that alignment can run.
        diagnostic["blocking"] = False
        diagnostic["blockers"] = []
        diagnostic["outside_margin"] = outside_margin
        diagnostic["diagnostic_only"] = True
        return diagnostic

    def _predict(
        self,
        session: ActRolloutSession,
        observation: Mapping[str, Any],
    ) -> tuple[np.ndarray, str]:
        age_s = time.time() - float(observation["timestamp"])
        if age_s < -0.05 or age_s > self.max_observation_age_s:
            raise ActRolloutSafetyViolation(
                f"ACT observation age {age_s * 1000.0:.1f} ms exceeds limit"
            )
        requested_horizon = session.execute_steps_per_inference
        started = time.monotonic()
        response = self.inference.predict(
            state=observation["state"],
            frames_bgr=observation["frames_bgr"],
            horizon=requested_horizon,
            request_id=uuid.uuid4().hex,
            session_id=session.session_id,
        )
        elapsed_s = time.monotonic() - started
        if elapsed_s > self.max_inference_s:
            raise ActRolloutSafetyViolation(
                f"ACT inference took {elapsed_s * 1000.0:.1f} ms and is stale"
            )
        actions = np.asarray(response.get("actions"), dtype=np.float64)
        if actions.shape != (requested_horizon, 14) or not np.all(
            np.isfinite(actions)
        ):
            raise ActRolloutSafetyViolation("ACT returned an invalid action chunk")
        action_representation = str(response.get("action_representation", ""))
        if action_representation not in {
            "absolute_joint_target",
            "delta_target_minus_observation_state",
        }:
            raise ActRolloutSafetyViolation(
                "ACT returned an unsupported action representation"
            )
        joint_indices = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
        if (
            action_representation == "absolute_joint_target"
            and float(np.max(np.abs(actions[:, joint_indices])))
            > self.joint_absolute_limit_rad
        ):
            raise ActRolloutSafetyViolation("ACT action exceeds absolute joint limit")
        chunk_index = session.inference_count
        session.inference_count += 1
        session.last_inference_ms = elapsed_s * 1000.0
        session.last_observation_age_ms = age_s * 1000.0
        session.last_camera_arm_delta_ms = float(
            observation["camera_arm_delta_ms"]
        )
        self._trace_chunk(session, chunk_index, observation, actions)
        return actions, action_representation

    def _enter_servo_mode(self, arms: Mapping[str, Any]) -> None:
        servo_mode = self._motion_mode()
        speed_profile = self._motion_speed_profile()
        with self._command_lock:
            for name, arm in arms.items():
                if arm.switch_mode(servo_mode) is not True:
                    raise ActRolloutUnavailable(
                        f"{name} arm rejected SERVO_JOINT_POS"
                    )
                mode_method = getattr(arm, "get_control_mode", None)
                if callable(mode_method) and self._enum_name(
                    mode_method()
                ) != "SERVO_JOINT_POS":
                    raise ActRolloutUnavailable(
                        f"{name} arm did not enter SERVO_JOINT_POS"
                    )
                set_speed = getattr(arm, "set_speed_profile", None)
                if not callable(set_speed) or set_speed(speed_profile) is False:
                    raise ActRolloutUnavailable(
                        f"{name} arm rejected rollout speed profile"
                    )

    @staticmethod
    def _send_target(
        arm: Any,
        joints: np.ndarray,
        gripper_m: float,
    ) -> bool:
        if arm.servo_joint_pos(joints.tolist()) is False:
            return False
        gripper_method = getattr(arm, "servo_eef_pos", None)
        if not callable(gripper_method):
            raise ActRolloutUnavailable("follower arm lacks gripper servo control")
        return gripper_method(float(gripper_m)) is not False

    def _hold_current(self, arms: Mapping[str, Any]) -> None:
        def hold_one(name: str) -> bool:
            arm = arms[name]
            joints = np.asarray(arm.get_joint_pos(), dtype=np.float64)
            if joints.shape != (6,) or not np.all(np.isfinite(joints)):
                raise ActRolloutUnavailable(f"{name} hold feedback is invalid")
            return arm.servo_joint_pos(joints.tolist()) is not False

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {name: pool.submit(hold_one, name) for name in ARM_NAMES}
            for name in ARM_NAMES:
                if not futures[name].result():
                    raise ActRolloutUnavailable(f"{name} arm rejected hold-current")

    def _action_targets(
        self,
        action: np.ndarray,
        previous_joints: Mapping[str, np.ndarray],
        previous_grippers: Mapping[str, float],
        action_representation: str,
        observation_joints: Mapping[str, np.ndarray],
        observation_grippers: Mapping[str, float],
    ) -> tuple[dict[str, np.ndarray], dict[str, float], float]:
        joints: dict[str, np.ndarray] = {}
        grippers: dict[str, float] = {}
        max_step = 0.0
        for arm_index, name in enumerate(ARM_NAMES):
            offset = arm_index * 7
            model_joints = np.asarray(action[offset : offset + 6], dtype=np.float64)
            if action_representation == "absolute_joint_target":
                desired = model_joints
                desired_gripper_raw = float(action[offset + 6])
            elif action_representation == "delta_target_minus_observation_state":
                # Every row in a predicted chunk is relative to the observation
                # supplied for that inference request. It is not an incremental
                # command relative to the preceding row in the chunk.
                desired = observation_joints[name] + model_joints
                desired_gripper_raw = (
                    observation_grippers[name] / self.gripper_scale
                    + float(action[offset + 6])
                )
            else:
                raise ActRolloutSafetyViolation(
                    "ACT returned an unsupported action representation"
                )
            if float(np.max(np.abs(desired))) > self.joint_absolute_limit_rad:
                raise ActRolloutSafetyViolation(
                    f"{name} ACT target exceeds absolute joint limit"
                )
            delta = desired - previous_joints[name]
            command_step = float(np.max(np.abs(delta)))
            if command_step > self.max_command_step_rad:
                raise ActRolloutSafetyViolation(
                    f"{name} ACT command step {command_step:.3f} rad exceeds "
                    f"limit {self.max_command_step_rad:.3f} rad"
                )
            joints[name] = desired.copy()
            max_step = max(max_step, command_step)
            desired_gripper = float(
                np.clip(
                    desired_gripper_raw * self.gripper_scale,
                    self.gripper_min_m,
                    self.gripper_max_m,
                )
            )
            grippers[name] = desired_gripper
        return joints, grippers, max_step

    def _validate_action_chunk(
        self,
        actions: np.ndarray,
        previous_joints: Mapping[str, np.ndarray],
        previous_grippers: Mapping[str, float],
        action_representation: str,
        observation_joints: Mapping[str, np.ndarray],
        observation_grippers: Mapping[str, float],
    ) -> None:
        simulated_joints = {
            name: np.asarray(previous_joints[name], dtype=np.float64).copy()
            for name in ARM_NAMES
        }
        simulated_grippers = {
            name: float(previous_grippers[name]) for name in ARM_NAMES
        }
        for action in actions:
            simulated_joints, simulated_grippers, _ = self._action_targets(
                action,
                simulated_joints,
                simulated_grippers,
                action_representation,
                observation_joints,
                observation_grippers,
            )

    def _run(self, session: ActRolloutSession) -> None:
        arms: dict[str, Any] = {}
        motion_armed = False
        feedback_stop = threading.Event()
        feedback_thread: threading.Thread | None = None
        current_chunk_index = -1
        current_execution_records: list[dict[str, Any]] = []
        try:
            blocker = self.interlock()
            if blocker:
                raise ActRolloutConflict(blocker)
            for name in ARM_NAMES:
                arm = self.arm_factory(self.arm_host, self.arm_ports[name])
                if arm.connect() is not True:
                    raise ActRolloutUnavailable(
                        f"cannot connect to follower {name} arm"
                    )
                arms[name] = arm
            with self._lock:
                self._active_arms = dict(arms)
            for name, arm in arms.items():
                state_method = getattr(arm, "get_state", None)
                if callable(state_method) and self._enum_name(
                    state_method()
                ) != "IDLE":
                    raise ActRolloutConflict(f"{name} follower arm must be IDLE")

            self._reset_feedback_history()
            feedback_thread = threading.Thread(
                target=self._sample_feedback_history,
                args=(arms, feedback_stop),
                name=f"act-feedback-{session.session_id[-8:]}",
                daemon=True,
            )
            feedback_thread.start()
            if session.stop_event.wait(self.feedback_history_warmup_s):
                session.finished_at = time.time()
                session.state = "stopped"
                return

            observation = self._capture_observation(arms, first=True)
            diagnostic = self._checked_start_pose(observation["state"])
            session.start_pose_diagnostic = diagnostic
            actions, action_representation = self._predict(session, observation)
            if session.stop_event.is_set():
                session.finished_at = time.time()
                session.state = "stopped"
                return
            previous_joints = {
                name: np.asarray(observation["joints"][name], dtype=np.float64)
                for name in ARM_NAMES
            }
            previous_grippers = {
                name: float(observation["grippers"][name]) for name in ARM_NAMES
            }
            observation_joints = {
                name: values.copy() for name, values in previous_joints.items()
            }
            observation_grippers = dict(previous_grippers)
            left_gripper_override = getattr(
                self.inference,
                "left_gripper_observation_override",
                None,
            )
            if left_gripper_override is not None:
                observation_grippers["left"] = (
                    float(left_gripper_override) * self.gripper_scale
                )
            self._validate_action_chunk(
                actions,
                previous_joints,
                previous_grippers,
                action_representation,
                observation_joints,
                observation_grippers,
            )
            self._enter_servo_mode(arms)
            motion_armed = True
            with self._lock:
                self._motion_armed = True
            session.started_at = time.time()
            session.state = "running"
            period = 1.0 / self.command_hz
            first_capture = False
            while not session.stop_event.is_set():
                current_chunk_index = session.inference_count - 1
                current_execution_records = []
                for step_index in range(
                    0,
                    session.execute_steps_per_inference,
                ):
                    if session.stop_event.is_set():
                        break
                    blocker = self.interlock()
                    if blocker:
                        raise ActRolloutConflict(blocker)
                    targets, grippers, max_step = self._action_targets(
                        actions[step_index],
                        previous_joints,
                        previous_grippers,
                        action_representation,
                        observation_joints,
                        observation_grippers,
                    )
                    command_started_at = time.time()
                    command_started = time.monotonic()
                    with self._command_lock:
                        if session.stop_event.is_set():
                            break
                        with ThreadPoolExecutor(max_workers=2) as pool:
                            futures = {
                                name: pool.submit(
                                    self._send_target,
                                    arms[name],
                                    targets[name],
                                    grippers[name],
                                )
                                for name in ARM_NAMES
                            }
                            for name in ARM_NAMES:
                                if not futures[name].result():
                                    raise ActRolloutUnavailable(
                                        f"{name} arm rejected ACT target"
                                    )
                    previous_joints = targets
                    previous_grippers = grippers
                    session.command_count += 1
                    session.last_max_command_step_rad = max_step
                    command_finished_at = time.time()
                    rpc_elapsed_ms = (time.monotonic() - command_started) * 1000.0
                    feedback_record = None
                    tracking_errors: dict[str, float] = {}
                    tracking_violation: str | None = None

                    if session.command_count % self.feedback_every_n == 0:
                        with self._command_lock:
                            feedback = self._capture_feedback(arms)
                        for name in ARM_NAMES:
                            tracking_error = float(
                                np.max(
                                    np.abs(
                                        feedback["joints"][name]
                                        - previous_joints[name]
                                    )
                                )
                            )
                            tracking_errors[name] = tracking_error
                            session.max_tracking_errors_rad[name] = max(
                                session.max_tracking_errors_rad[name],
                                tracking_error,
                            )
                            if tracking_error > self.max_tracking_error_rad:
                                session.tracking_warning_count += 1
                                tracking_violation = (
                                    f"{name} tracking error {tracking_error:.3f} "
                                    f"rad exceeds diagnostic threshold"
                                )
                        session.last_tracking_errors_rad = dict(tracking_errors)
                        feedback_record = {
                            "timestamp": feedback["timestamp"],
                            "state": feedback["state"],
                            "arm_pair_skew_ms": feedback["arm_pair_skew_ms"],
                        }
                    remaining = period - (time.monotonic() - command_started)
                    if remaining > 0.0:
                        session.stop_event.wait(remaining)
                    current_execution_records.append(
                        {
                            "step_index": step_index,
                            "command_index": session.command_count - 1,
                            "started_at": command_started_at,
                            "finished_at": command_finished_at,
                            "rpc_elapsed_ms": rpc_elapsed_ms,
                            "cycle_elapsed_ms": (
                                time.monotonic() - command_started
                            )
                            * 1000.0,
                            "max_joint_step_rad": max_step,
                            "target_state": [
                                *targets["left"].tolist(),
                                grippers["left"],
                                *targets["right"].tolist(),
                                grippers["right"],
                            ],
                            "tracking_errors_rad": tracking_errors,
                            "tracking_error_warning": tracking_violation,
                            "feedback": feedback_record,
                        }
                    )
                    if tracking_violation and self.tracking_error_blocking:
                        raise ActRolloutSafetyViolation(tracking_violation)

                self._trace_execution(
                    session,
                    current_chunk_index,
                    current_execution_records,
                )
                current_execution_records = []

                if session.stop_event.is_set():
                    break
                # Give the robot a short fixed settling window after the chunk.
                # No joint-error gate or timeout can reject the rollout here.
                if session.stop_event.wait(self.post_chunk_delay_s):
                    break
                # A new inference must be conditioned on sensor exposures and
                # arm feedback sampled after the entire preceding chunk cycle.
                # Capturing this cutoff here also includes the final 30 Hz wait.
                captured_after = time.time()
                observation = self._capture_observation(
                    arms,
                    first=first_capture,
                    captured_after=captured_after,
                )
                first_capture = False
                if session.stop_event.is_set():
                    break
                previous_joints = {
                    name: np.asarray(observation["joints"][name], dtype=np.float64)
                    for name in ARM_NAMES
                }
                previous_grippers = {
                    name: float(observation["grippers"][name]) for name in ARM_NAMES
                }
                observation_joints = {
                    name: values.copy() for name, values in previous_joints.items()
                }
                observation_grippers = dict(previous_grippers)
                if left_gripper_override is not None:
                    observation_grippers["left"] = (
                        float(left_gripper_override) * self.gripper_scale
                    )
                actions, action_representation = self._predict(session, observation)
                self._validate_action_chunk(
                    actions,
                    previous_joints,
                    previous_grippers,
                    action_representation,
                    observation_joints,
                    observation_grippers,
                )

            with self._command_lock:
                if not session.hold_confirmed:
                    self._hold_current(arms)
                    session.hold_completed_at = time.time()
                    session.hold_confirmed = True
            session.finished_at = time.time()
            session.state = "stopped"
        except Exception as exc:
            if current_execution_records and current_chunk_index >= 0:
                self._trace_execution(
                    session,
                    current_chunk_index,
                    current_execution_records,
                )
            try:
                if arms and motion_armed:
                    with self._command_lock:
                        if not session.hold_confirmed:
                            self._hold_current(arms)
                            session.hold_completed_at = time.time()
                            session.hold_confirmed = True
            except Exception as hold_exc:
                session.error = (
                    f"{type(exc).__name__}: {exc}; hold failed: "
                    f"{type(hold_exc).__name__}: {hold_exc}"
                )
            else:
                session.error = f"{type(exc).__name__}: {exc}"
            session.finished_at = time.time()
            session.state = "error"
        finally:
            feedback_stop.set()
            if feedback_thread is not None:
                feedback_thread.join(timeout=1.0)
            for arm in arms.values():
                self._disconnect(arm)
            # Queue the terminal summary before publishing the terminal state.
            # Callers often close immediately after observing stopped/error;
            # publishing first could shut the trace writer down too early.
            self._trace_summary(session)
            with self._lock:
                self._active_arms = {}
                self._motion_armed = False
                self._last_session = session
                self._session = None

    def close(self, *, timeout_s: float = 5.0) -> None:
        with self._lock:
            session = self._session
            worker = None if session is None else session.worker
        if session is not None and session.state in ACTIVE_STATES:
            try:
                self.stop(reason="service_shutdown")
            except ActRolloutError:
                session.stop_event.set()
        if worker is not None:
            worker.join(timeout=max(0.1, timeout_s))
        self._trace_executor.shutdown(wait=True)
