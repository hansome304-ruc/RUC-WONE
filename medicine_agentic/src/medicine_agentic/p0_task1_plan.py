"""Fail-closed, planning-only state machine for P0 Task 1.

This module intentionally has no robot, suction, camera, or network adapter.
It consumes two already persisted inputs:

* a pose document managed by :mod:`medicine_agentic.pose_store`;
* a detection report produced by ``scripts/task1_detect_box.py``.

The result is an auditable list of actions that *would* be requested later.
Every action is marked ``executed=False`` and the report records zero hardware
commands.  A future hardware executor must live in a separate module and must
not be reachable through this API.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from medicine_agentic.pose_store import PoseStore, PoseStoreError
from medicine_agentic.tcp_pivot import content_sha256 as tcp_content_sha256


SCHEMA_VERSION = 1
DEFAULT_REQUIRED_POSES = (
    "home",
    "task1_observe",
    "pre_pick_carton",
    "safe_transport_carton",
    "pre_place_carton",
    "place_slot_0_contact",
    "post_place",
    "recovery_high",
)


@dataclass(frozen=True)
class BoxCandidate:
    center_px: tuple[float, float]
    suction_px: tuple[int, int]
    polygon_px: tuple[tuple[float, float], ...]
    long_side_px: float
    short_side_px: float
    angle_deg: float
    rectangularity: float
    bright_fill: float
    edge_clearance_px: float
    score: float
    provider: str = "legacy_unknown"
    face_type: str = "unknown"
    face_score: float = 0.0
    reference_face_id: str | None = None
    graspable: bool = False
    grasp_blockers: tuple[str, ...] = (
        "legacy report has no 2-D grasp authorization",
    )


@dataclass(frozen=True)
class DepthEstimate:
    median_mm: float
    spread_mm: float
    valid_samples: int
    samples_mm: tuple[float, ...]
    sample_pixels_px: tuple[tuple[int, int], ...]
    frame_age_s: float


@dataclass(frozen=True)
class LocatedBox:
    candidate: BoxCandidate
    depth: DepthEstimate | None
    point_camera_m: tuple[float, float, float] | None
    point_left_base_m: tuple[float, float, float] | None
    physical_size_m: tuple[float, float] | None
    surface_normal_left_base: tuple[float, float, float] | None
    surface_tilt_deg: float | None
    plane_residual_mm: float | None
    reachable: bool | None
    blockers: tuple[str, ...]


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def load_cam_to_left(path: str | Path) -> np.ndarray:
    payload = load_json(path)
    transform = np.asarray(payload["cam_to_base"], dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"cam_to_base must be 4x4: {path}")
    return transform


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


class P0State(str, Enum):
    START = "START"
    CONFIG_GATE = "CONFIG_GATE"
    POSE_GATE = "POSE_GATE"
    PLAN_HOME = "PLAN_HOME"
    PLAN_OBSERVE = "PLAN_OBSERVE"
    PERCEPTION_GATE = "PERCEPTION_GATE"
    PLAN_PRE_PICK = "PLAN_PRE_PICK"
    PLAN_VERTICAL_DESCENT = "PLAN_VERTICAL_DESCENT"
    PLAN_SUCTION_ON = "PLAN_SUCTION_ON"
    PLAN_TEST_LIFT = "PLAN_TEST_LIFT"
    PLAN_VERIFY_ATTACHED = "PLAN_VERIFY_ATTACHED"
    PLAN_FULL_LIFT = "PLAN_FULL_LIFT"
    PLAN_SAFE_TRANSPORT = "PLAN_SAFE_TRANSPORT"
    PLAN_PRE_PLACE = "PLAN_PRE_PLACE"
    PLAN_PLACE_DESCENT = "PLAN_PLACE_DESCENT"
    PLAN_SUCTION_OFF = "PLAN_SUCTION_OFF"
    PLAN_RETRACT = "PLAN_RETRACT"
    PLAN_VERIFY_PLACED = "PLAN_VERIFY_PLACED"
    DRY_RUN_COMPLETE = "DRY_RUN_COMPLETE"
    FAILED = "FAILED"


class FailureCode(str, Enum):
    NONE = "NONE"
    CONFIG_INVALID = "CONFIG_INVALID"
    DRY_RUN_REQUIRED = "DRY_RUN_REQUIRED"
    MOTION_MUST_BE_DISABLED = "MOTION_MUST_BE_DISABLED"
    SUCTION_MUST_BE_DISABLED = "SUCTION_MUST_BE_DISABLED"
    ROBOT_ADAPTER_FORBIDDEN = "ROBOT_ADAPTER_FORBIDDEN"
    INPUT_FILE_MISSING = "INPUT_FILE_MISSING"
    CAMERA_EXTRINSIC_NOT_READY = "CAMERA_EXTRINSIC_NOT_READY"
    TCP_NOT_CALIBRATED = "TCP_NOT_CALIBRATED"
    POSE_STORE_INVALID = "POSE_STORE_INVALID"
    ROBOT_ID_MISMATCH = "ROBOT_ID_MISMATCH"
    POSE_MISSING = "POSE_MISSING"
    POSE_NOT_VALIDATED = "POSE_NOT_VALIDATED"
    POSE_UNSTABLE = "POSE_UNSTABLE"
    POSE_COLLISION_UNPROVEN = "POSE_COLLISION_UNPROVEN"
    DETECTION_INVALID = "DETECTION_INVALID"
    DETECTION_FAILED = "DETECTION_FAILED"
    DETECTION_STALE = "DETECTION_STALE"
    TARGET_NOT_LOCALIZED = "TARGET_NOT_LOCALIZED"
    TARGET_BLOCKED = "TARGET_BLOCKED"
    TARGET_UNREACHABLE = "TARGET_UNREACHABLE"
    INVALID_TRANSITION = "INVALID_TRANSITION"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class PlanConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SafetyConfig:
    dry_run: bool
    motion_commands_enabled: bool
    suction_commands_enabled: bool
    robot_adapter: str


@dataclass(frozen=True)
class GateConfig:
    robot_id: str
    required_poses: tuple[str, ...]
    required_pose_status: str
    require_pose_stable: bool
    require_collision_free: bool
    require_camera_extrinsic: bool
    require_suction_tcp: bool
    max_detection_age_s: float | None


@dataclass(frozen=True)
class MotionPlanConfig:
    pre_contact_height_m: float
    passive_compression_m: float
    probe_lift_m: float
    full_lift_m: float
    release_clearance_m: float
    pick_descent_speed_m_s: float
    place_descent_speed_m_s: float
    speed_scale: float
    slot_id: str


@dataclass(frozen=True)
class P0Task1Config:
    config_path: Path
    pose_store_path: Path
    detection_report_path: Path
    task1_box_config_path: Path
    suction_tcp_path: Path
    log_root: Path
    safety: SafetyConfig
    gates: GateConfig
    plan: MotionPlanConfig


@dataclass(frozen=True)
class PlanAction:
    state: P0State
    kind: str
    description: str
    parameters: dict[str, Any]
    executed: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload


@dataclass(frozen=True)
class PlanReport:
    run_id: str
    status: str
    state: P0State
    ready: bool
    failure_code: FailureCode
    message: str
    actions: tuple[PlanAction, ...]
    log_dir: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "mode": "dry_run",
            "status": self.status,
            "state": self.state.value,
            "ready": self.ready,
            "task_physically_completed": False,
            "failure_code": self.failure_code.value,
            "message": self.message,
            "planned_action_count": len(self.actions),
            "actions": [action.to_dict() for action in self.actions],
            "safety_accounting": {
                "motion_commands_issued": 0,
                "suction_commands_issued": 0,
                "camera_connections_opened": 0,
                "robot_connections_opened": 0,
            },
            "log_dir": str(self.log_dir),
        }


def _resolve_path(value: Any, base_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PlanConfigError(f"{field} must be a non-empty path string")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _positive_float(raw: dict[str, Any], key: str, default: float) -> float:
    value = float(raw.get(key, default))
    if not math.isfinite(value) or value <= 0.0:
        raise PlanConfigError(f"plan.{key} must be a finite positive number")
    return value


def load_plan_config(path: str | Path) -> P0Task1Config:
    config_path = Path(path).expanduser().resolve()
    raw = load_json(config_path)
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise PlanConfigError(
            f"schema_version must be {SCHEMA_VERSION}, got "
            f"{raw.get('schema_version')!r}"
        )

    inputs = raw.get("inputs")
    safety_raw = raw.get("safety")
    gates_raw = raw.get("gates")
    plan_raw = raw.get("plan")
    for name, value in (
        ("inputs", inputs),
        ("safety", safety_raw),
        ("gates", gates_raw),
        ("plan", plan_raw),
    ):
        if not isinstance(value, dict):
            raise PlanConfigError(f"{name} must be an object")

    base_dir = config_path.parent
    required = gates_raw.get("required_poses", DEFAULT_REQUIRED_POSES)
    if not isinstance(required, list) or not required:
        raise PlanConfigError("gates.required_poses must be a non-empty list")
    required_poses = tuple(str(name) for name in required)
    if any(not name or any(character.isspace() for character in name) for name in required_poses):
        raise PlanConfigError("required pose names must be non-empty without whitespace")
    if len(set(required_poses)) != len(required_poses):
        raise PlanConfigError("gates.required_poses contains duplicates")

    maximum_age = gates_raw.get("max_detection_age_s", 30.0)
    if maximum_age is not None:
        maximum_age = float(maximum_age)
        if not math.isfinite(maximum_age) or maximum_age <= 0.0:
            raise PlanConfigError(
                "gates.max_detection_age_s must be null or a finite positive number"
            )

    speed_scale = float(plan_raw.get("speed_scale", 0.10))
    if not math.isfinite(speed_scale) or not 0.0 < speed_scale <= 0.20:
        raise PlanConfigError("plan.speed_scale must be in (0, 0.20] for P0 dry-run")

    return P0Task1Config(
        config_path=config_path,
        pose_store_path=_resolve_path(
            inputs.get("pose_store"), base_dir, "inputs.pose_store"
        ),
        detection_report_path=_resolve_path(
            inputs.get("detection_report"), base_dir, "inputs.detection_report"
        ),
        task1_box_config_path=_resolve_path(
            inputs.get("task1_box_config"), base_dir, "inputs.task1_box_config"
        ),
        suction_tcp_path=_resolve_path(
            inputs.get("suction_tcp", "calibration/left_suction_tcp.json"),
            base_dir,
            "inputs.suction_tcp",
        ),
        log_root=_resolve_path(
            inputs.get("log_root", "../artifacts/p0_task1_runs"),
            base_dir,
            "inputs.log_root",
        ),
        safety=SafetyConfig(
            dry_run=bool(safety_raw.get("dry_run", False)),
            motion_commands_enabled=bool(
                safety_raw.get("motion_commands_enabled", False)
            ),
            suction_commands_enabled=bool(
                safety_raw.get("suction_commands_enabled", False)
            ),
            robot_adapter=str(safety_raw.get("robot_adapter", "")),
        ),
        gates=GateConfig(
            robot_id=str(gates_raw.get("robot_id", "")),
            required_poses=required_poses,
            required_pose_status=str(
                gates_raw.get("required_pose_status", "validated")
            ),
            require_pose_stable=bool(
                gates_raw.get("require_pose_stable", True)
            ),
            require_collision_free=bool(
                gates_raw.get("require_collision_free", True)
            ),
            require_camera_extrinsic=bool(
                gates_raw.get("require_camera_extrinsic", True)
            ),
            require_suction_tcp=bool(
                gates_raw.get("require_suction_tcp", True)
            ),
            max_detection_age_s=maximum_age,
        ),
        plan=MotionPlanConfig(
            pre_contact_height_m=_positive_float(
                plan_raw, "pre_contact_height_m", 0.08
            ),
            passive_compression_m=_positive_float(
                plan_raw, "passive_compression_m", 0.004
            ),
            probe_lift_m=_positive_float(plan_raw, "probe_lift_m", 0.02),
            full_lift_m=_positive_float(plan_raw, "full_lift_m", 0.08),
            release_clearance_m=_positive_float(
                plan_raw, "release_clearance_m", 0.004
            ),
            pick_descent_speed_m_s=_positive_float(
                plan_raw, "pick_descent_speed_m_s", 0.01
            ),
            place_descent_speed_m_s=_positive_float(
                plan_raw, "place_descent_speed_m_s", 0.01
            ),
            speed_scale=speed_scale,
            slot_id=str(plan_raw.get("slot_id", "slot_0")),
        ),
    )


def _parse_tuple(
    raw: dict[str, Any],
    key: str,
    length: int,
    *,
    optional: bool = False,
) -> tuple[Any, ...] | None:
    value = raw.get(key)
    if value is None and optional:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{key} must contain {length} values")
    return tuple(value)


def located_box_from_mapping(raw: dict[str, Any]) -> LocatedBox:
    """Rehydrate the stable ``task1_box`` output without touching a camera."""
    candidate_raw = raw.get("candidate")
    if not isinstance(candidate_raw, dict):
        raise ValueError("selected.candidate must be an object")
    polygon_raw = candidate_raw.get("polygon_px")
    if not isinstance(polygon_raw, list) or len(polygon_raw) < 4:
        raise ValueError("selected.candidate.polygon_px must contain at least 4 points")
    grasp_blockers_raw = candidate_raw.get(
        "grasp_blockers",
        ["legacy report has no 2-D grasp authorization"],
    )
    if not isinstance(grasp_blockers_raw, list):
        raise ValueError("selected.candidate.grasp_blockers must be a list")
    candidate = BoxCandidate(
        center_px=tuple(
            float(value)
            for value in _parse_tuple(candidate_raw, "center_px", 2) or ()
        ),
        suction_px=tuple(
            int(value)
            for value in _parse_tuple(candidate_raw, "suction_px", 2) or ()
        ),
        polygon_px=tuple(
            tuple(float(value) for value in point)
            for point in polygon_raw
        ),
        long_side_px=float(candidate_raw["long_side_px"]),
        short_side_px=float(candidate_raw["short_side_px"]),
        angle_deg=float(candidate_raw["angle_deg"]),
        rectangularity=float(candidate_raw["rectangularity"]),
        bright_fill=float(candidate_raw["bright_fill"]),
        edge_clearance_px=float(candidate_raw["edge_clearance_px"]),
        score=float(candidate_raw["score"]),
        provider=str(candidate_raw.get("provider", "legacy_unknown")),
        face_type=str(candidate_raw.get("face_type", "unknown")),
        face_score=float(candidate_raw.get("face_score", 0.0)),
        reference_face_id=(
            None
            if candidate_raw.get("reference_face_id") is None
            else str(candidate_raw["reference_face_id"])
        ),
        graspable=candidate_raw.get("graspable") is True,
        grasp_blockers=tuple(
            str(value) for value in grasp_blockers_raw
        ),
    )

    depth_raw = raw.get("depth")
    depth = None
    if depth_raw is not None:
        if not isinstance(depth_raw, dict):
            raise ValueError("selected.depth must be an object or null")
        samples = tuple(float(value) for value in depth_raw["samples_mm"])
        pixels = tuple(
            tuple(int(value) for value in pixel)
            for pixel in depth_raw["sample_pixels_px"]
        )
        if len(samples) != len(pixels):
            raise ValueError("selected.depth samples and pixels have different lengths")
        depth = DepthEstimate(
            median_mm=float(depth_raw["median_mm"]),
            spread_mm=float(depth_raw["spread_mm"]),
            valid_samples=int(depth_raw["valid_samples"]),
            samples_mm=samples,
            sample_pixels_px=pixels,
            frame_age_s=float(depth_raw["frame_age_s"]),
        )

    def optional_floats(key: str, length: int) -> tuple[float, ...] | None:
        values = _parse_tuple(raw, key, length, optional=True)
        return None if values is None else tuple(float(value) for value in values)

    blockers_raw = raw.get("blockers", [])
    if not isinstance(blockers_raw, list):
        raise ValueError("selected.blockers must be a list")
    return LocatedBox(
        candidate=candidate,
        depth=depth,
        point_camera_m=optional_floats("point_camera_m", 3),
        point_left_base_m=optional_floats("point_left_base_m", 3),
        physical_size_m=optional_floats("physical_size_m", 2),
        surface_normal_left_base=optional_floats(
            "surface_normal_left_base", 3
        ),
        surface_tilt_deg=(
            None
            if raw.get("surface_tilt_deg") is None
            else float(raw["surface_tilt_deg"])
        ),
        plane_residual_mm=(
            None
            if raw.get("plane_residual_mm") is None
            else float(raw["plane_residual_mm"])
        ),
        reachable=(
            None if raw.get("reachable") is None else bool(raw["reachable"])
        ),
        blockers=tuple(str(blocker) for blocker in blockers_raw),
    )


class JsonlRunLog:
    def __init__(self, root: Path, run_id: str) -> None:
        self.run_dir = root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.sequence = 0

    def event(
        self,
        previous: P0State,
        current: P0State,
        outcome: str,
        *,
        failure_code: FailureCode = FailureCode.NONE,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.sequence += 1
        payload = {
            "timestamp": utc_now(),
            "sequence": self.sequence,
            "from_state": previous.value,
            "to_state": current.value,
            "outcome": outcome,
            "failure_code": failure_code.value,
            "details": details or {},
        }
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
            )

    def summary(self, payload: dict[str, Any]) -> None:
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".summary.",
            suffix=".tmp",
            dir=self.run_dir,
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.summary_path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)


_HAPPY_PATH = (
    P0State.START,
    P0State.CONFIG_GATE,
    P0State.POSE_GATE,
    P0State.PLAN_HOME,
    P0State.PLAN_OBSERVE,
    P0State.PERCEPTION_GATE,
    P0State.PLAN_PRE_PICK,
    P0State.PLAN_VERTICAL_DESCENT,
    P0State.PLAN_SUCTION_ON,
    P0State.PLAN_TEST_LIFT,
    P0State.PLAN_VERIFY_ATTACHED,
    P0State.PLAN_FULL_LIFT,
    P0State.PLAN_SAFE_TRANSPORT,
    P0State.PLAN_PRE_PLACE,
    P0State.PLAN_PLACE_DESCENT,
    P0State.PLAN_SUCTION_OFF,
    P0State.PLAN_RETRACT,
    P0State.PLAN_VERIFY_PLACED,
    P0State.DRY_RUN_COMPLETE,
)
_ALLOWED_TRANSITIONS = {
    state: {next_state, P0State.FAILED}
    for state, next_state in zip(_HAPPY_PATH, _HAPPY_PATH[1:])
}
_ALLOWED_TRANSITIONS[P0State.DRY_RUN_COMPLETE] = set()
_ALLOWED_TRANSITIONS[P0State.FAILED] = set()


class P0Task1DryRunPlanner:
    """Traverse P0 Task 1 while producing plans and never executing them."""

    def __init__(self, config: P0Task1Config, *, run_id: str | None = None) -> None:
        self.config = config
        self.run_id = run_id or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "_"
            + uuid.uuid4().hex[:8]
        )
        self.log = JsonlRunLog(config.log_root, self.run_id)
        self.state = P0State.START
        self.actions: list[PlanAction] = []
        self.pose_document: dict[str, Any] | None = None
        self.target: LocatedBox | None = None

    def run(self) -> PlanReport:
        try:
            failure = self._gate_config()
            if failure is not None:
                return self._blocked(*failure)
            failure = self._gate_poses()
            if failure is not None:
                return self._blocked(*failure)

            self._plan_named_pose(P0State.PLAN_HOME, "home")
            self._plan_named_pose(P0State.PLAN_OBSERVE, "task1_observe")

            failure = self._gate_perception()
            if failure is not None:
                return self._blocked(*failure)
            assert self.target is not None
            assert self.target.point_left_base_m is not None

            self._plan_pick_and_place(self.target)
            self._transition(
                P0State.DRY_RUN_COMPLETE,
                "READY",
                details={
                    "planned_action_count": len(self.actions),
                    "task_physically_completed": False,
                },
            )
            return self._finish(
                ready=True,
                code=FailureCode.NONE,
                message=(
                    "dry-run plan is ready; no motion, suction, camera, or robot "
                    "command was issued"
                ),
            )
        except Exception as exc:  # fail closed and preserve an audit trail
            return self._blocked(
                FailureCode.INTERNAL_ERROR,
                f"{type(exc).__name__}: {exc}",
                {"exception_type": type(exc).__name__},
            )

    def _gate_config(
        self,
    ) -> tuple[FailureCode, str, dict[str, Any]] | None:
        self._transition(P0State.CONFIG_GATE, "CHECKING")
        safety = self.config.safety
        if not safety.dry_run:
            return (
                FailureCode.DRY_RUN_REQUIRED,
                "safety.dry_run must be true",
                {},
            )
        if safety.motion_commands_enabled:
            return (
                FailureCode.MOTION_MUST_BE_DISABLED,
                "motion commands must remain disabled in this planner",
                {},
            )
        if safety.suction_commands_enabled:
            return (
                FailureCode.SUCTION_MUST_BE_DISABLED,
                "suction commands must remain disabled in this planner",
                {},
            )
        if safety.robot_adapter.strip().lower() not in {"none", "disabled"}:
            return (
                FailureCode.ROBOT_ADAPTER_FORBIDDEN,
                "robot_adapter must be 'none' or 'disabled'",
                {"robot_adapter": safety.robot_adapter},
            )
        if not self.config.gates.robot_id:
            return (
                FailureCode.CONFIG_INVALID,
                "gates.robot_id must be non-empty",
                {},
            )
        if not self.config.plan.slot_id:
            return (
                FailureCode.CONFIG_INVALID,
                "plan.slot_id must be non-empty",
                {},
            )
        if not self.config.task1_box_config_path.is_file():
            return (
                FailureCode.INPUT_FILE_MISSING,
                "task1_box config does not exist",
                {"path": str(self.config.task1_box_config_path)},
            )
        try:
            box_config = load_json(self.config.task1_box_config_path)
        except Exception as exc:
            return (
                FailureCode.CONFIG_INVALID,
                f"cannot read task1_box config: {exc}",
                {"path": str(self.config.task1_box_config_path)},
            )

        if self.config.gates.require_suction_tcp:
            try:
                tcp = load_json(self.config.suction_tcp_path)
                translation = np.asarray(tcp.get("translation_m"), dtype=np.float64)
                if translation.shape != (3,) or not np.all(np.isfinite(translation)):
                    raise ValueError("translation_m must contain 3 finite values")
                if tcp.get("calibrated") is not True:
                    raise ValueError("calibrated is not true")
                if tcp.get("usable_for_motion") is not True:
                    raise ValueError("usable_for_motion is not true")
                stored_hash = tcp.get("content_sha256")
                if stored_hash is not None and stored_hash != tcp_content_sha256(tcp):
                    raise ValueError("content_sha256 does not match")
            except Exception as exc:
                return (
                    FailureCode.TCP_NOT_CALIBRATED,
                    f"left suction TCP is not ready: {exc}",
                    {"path": str(self.config.suction_tcp_path)},
                )

        camera_config = box_config.get("camera")
        if not isinstance(camera_config, dict):
            return (
                FailureCode.CONFIG_INVALID,
                "task1_box config has no camera object",
                {},
            )
        if self.config.gates.require_camera_extrinsic:
            calibration_value = camera_config.get("cam_to_left_path")
            try:
                calibration_path = _resolve_path(
                    calibration_value,
                    self.config.task1_box_config_path.parent,
                    "camera.cam_to_left_path",
                )
                transform = load_cam_to_left(calibration_path)
                rotation = transform[:3, :3]
                valid = (
                    np.all(np.isfinite(transform))
                    and np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
                    and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
                    and abs(float(np.linalg.det(rotation)) - 1.0) <= 1e-3
                )
                if not valid:
                    raise ValueError("transform is not a finite rigid transform")
            except Exception as exc:
                return (
                    FailureCode.CAMERA_EXTRINSIC_NOT_READY,
                    f"camera-to-left-base calibration is not ready: {exc}",
                    {"path": str(calibration_value)},
                )
        return None

    def _gate_poses(
        self,
    ) -> tuple[FailureCode, str, dict[str, Any]] | None:
        self._transition(P0State.POSE_GATE, "CHECKING")
        if not self.config.pose_store_path.is_file():
            return (
                FailureCode.INPUT_FILE_MISSING,
                "pose store does not exist",
                {"path": str(self.config.pose_store_path)},
            )
        try:
            document = PoseStore(
                self.config.pose_store_path,
                robot_id=self.config.gates.robot_id,
            ).load()
        except (OSError, ValueError, PoseStoreError) as exc:
            return (
                FailureCode.POSE_STORE_INVALID,
                str(exc),
                {"path": str(self.config.pose_store_path)},
            )
        self.pose_document = document
        if document.get("robot_id") != self.config.gates.robot_id:
            return (
                FailureCode.ROBOT_ID_MISMATCH,
                "pose store belongs to another robot",
                {
                    "expected": self.config.gates.robot_id,
                    "actual": document.get("robot_id"),
                },
            )

        poses = document["poses"]
        missing = [
            name for name in self.config.gates.required_poses if name not in poses
        ]
        if missing:
            return (
                FailureCode.POSE_MISSING,
                "required taught poses are missing",
                {"missing": missing},
            )
        expected_status = self.config.gates.required_pose_status
        wrong_status = {
            name: poses[name].get("status")
            for name in self.config.gates.required_poses
            if poses[name].get("status") != expected_status
        }
        if wrong_status:
            return (
                FailureCode.POSE_NOT_VALIDATED,
                f"required poses must have status {expected_status!r}",
                {"poses": wrong_status},
            )
        if self.config.gates.require_pose_stable:
            unstable = [
                name
                for name in self.config.gates.required_poses
                if poses[name].get("validation", {}).get("stable") is not True
            ]
            if unstable:
                return (
                    FailureCode.POSE_UNSTABLE,
                    "required poses have not passed stability validation",
                    {"poses": unstable},
                )
        if self.config.gates.require_collision_free:
            unproven = [
                name
                for name in self.config.gates.required_poses
                if poses[name].get("validation", {}).get("collision_free")
                is not True
            ]
            if unproven:
                return (
                    FailureCode.POSE_COLLISION_UNPROVEN,
                    "required poses are not marked collision_free=true",
                    {"poses": unproven},
                )
        return None

    def _gate_perception(
        self,
    ) -> tuple[FailureCode, str, dict[str, Any]] | None:
        self._transition(P0State.PERCEPTION_GATE, "CHECKING")
        if not self.config.detection_report_path.is_file():
            return (
                FailureCode.INPUT_FILE_MISSING,
                "offline detection report does not exist",
                {"path": str(self.config.detection_report_path)},
            )
        try:
            report = load_json(self.config.detection_report_path)
        except Exception as exc:
            return (
                FailureCode.DETECTION_INVALID,
                f"cannot read detection report: {exc}",
                {"path": str(self.config.detection_report_path)},
            )
        if report.get("ok") is not True:
            return (
                FailureCode.DETECTION_FAILED,
                "detection report is not successful",
                {"error": report.get("error")},
            )
        timestamp = report.get("timestamp")
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            return (
                FailureCode.DETECTION_INVALID,
                "detection report timestamp is invalid",
                {},
            )
        maximum_age = self.config.gates.max_detection_age_s
        age = datetime.now(timezone.utc).timestamp() - float(timestamp)
        if age < -2.0:
            return (
                FailureCode.DETECTION_INVALID,
                "detection report timestamp is in the future",
                {"age_s": age},
            )
        if maximum_age is not None and age > maximum_age:
            return (
                FailureCode.DETECTION_STALE,
                "detection report is too old",
                {"age_s": age, "max_age_s": maximum_age},
            )
        selected = report.get("selected")
        if not isinstance(selected, dict):
            return (
                FailureCode.DETECTION_INVALID,
                "detection report has no selected object",
                {},
            )
        try:
            target = located_box_from_mapping(selected)
        except Exception as exc:
            return (
                FailureCode.DETECTION_INVALID,
                f"selected object is malformed: {exc}",
                {},
            )
        self.target = target
        if target.candidate.graspable is not True:
            return (
                FailureCode.TARGET_BLOCKED,
                "selected target lacks 2-D face-level grasp authorization",
                {
                    "provider": target.candidate.provider,
                    "face_type": target.candidate.face_type,
                    "grasp_blockers": list(target.candidate.grasp_blockers),
                },
            )
        if target.blockers:
            return (
                FailureCode.TARGET_BLOCKED,
                "selected target contains perception blockers",
                {"blockers": list(target.blockers)},
            )
        if target.reachable is not True:
            return (
                FailureCode.TARGET_UNREACHABLE,
                "selected target is not confirmed reachable",
                {"reachable": target.reachable},
            )
        required_vectors: Iterable[tuple[str, tuple[float, ...] | None]] = (
            ("point_camera_m", target.point_camera_m),
            ("point_left_base_m", target.point_left_base_m),
            ("surface_normal_left_base", target.surface_normal_left_base),
        )
        missing = [
            name
            for name, vector in required_vectors
            if vector is None
            or not all(math.isfinite(float(value)) for value in vector)
        ]
        if target.depth is None or missing:
            return (
                FailureCode.TARGET_NOT_LOCALIZED,
                "selected target lacks checked 3-D evidence",
                {"missing": missing},
            )
        return None

    def _plan_pick_and_place(self, target: LocatedBox) -> None:
        x, y, z = (float(value) for value in target.point_left_base_m or ())
        p = self.config.plan
        self._add_action(
            P0State.PLAN_PRE_PICK,
            "CARTESIAN_POSE_PLAN",
            "Resolve the dynamic pre-pick pose above the detected suction point.",
            {
                "arm": "left",
                "frame": "left_base",
                "reference_pose": "pre_pick_carton",
                "target_position_m": [x, y, z + p.pre_contact_height_m],
                "tool_axis": "base_vertical_down",
                "speed_scale": p.speed_scale,
            },
        )
        self._add_action(
            P0State.PLAN_VERTICAL_DESCENT,
            "CARTESIAN_LINEAR_PLAN",
            "Plan a capped vertical descent with passive cup compression.",
            {
                "arm": "left",
                "frame": "left_base",
                "surface_position_m": [x, y, z],
                "maximum_overtravel_m": p.passive_compression_m,
                "speed_m_s": p.pick_descent_speed_m_s,
            },
        )
        self._add_action(
            P0State.PLAN_SUCTION_ON,
            "DIGITAL_OUTPUT_PLAN",
            "Would request suction engagement; never emitted by this planner.",
            {"channel": "left_suction", "requested_state": True},
        )
        self._add_action(
            P0State.PLAN_TEST_LIFT,
            "CARTESIAN_LINEAR_PLAN",
            "Plan the mandatory 20 mm-class probe lift before transport.",
            {
                "arm": "left",
                "delta_in_left_base_m": [0.0, 0.0, p.probe_lift_m],
                "speed_m_s": p.pick_descent_speed_m_s,
            },
        )
        self._add_action(
            P0State.PLAN_VERIFY_ATTACHED,
            "VERIFICATION_PLAN",
            "Require visual motion consistency; calibrated acoustics may add a second gate.",
            {
                "expected": "carton_attached_after_probe_lift",
                "visual_probe_required": True,
                "acoustic_default": "supporting_only_until_holdout_passes",
                "failure_code": "EMPTY_SUCTION_OR_CARTON_NOT_LIFTED",
            },
        )
        self._add_action(
            P0State.PLAN_FULL_LIFT,
            "CARTESIAN_LINEAR_PLAN",
            "Plan full vertical clearance only after attachment verification.",
            {
                "arm": "left",
                "delta_in_left_base_m": [
                    0.0,
                    0.0,
                    p.full_lift_m - p.probe_lift_m,
                ],
                "speed_scale": p.speed_scale,
            },
        )
        self._plan_named_pose(
            P0State.PLAN_SAFE_TRANSPORT, "safe_transport_carton"
        )
        self._plan_named_pose(P0State.PLAN_PRE_PLACE, "pre_place_carton")
        self._add_action(
            P0State.PLAN_PLACE_DESCENT,
            "CARTESIAN_LINEAR_PLAN",
            "Plan a capped vertical descent to the taught slot contact reference.",
            {
                "arm": "left",
                "slot_id": p.slot_id,
                "reference_pose": "place_slot_0_contact",
                "release_clearance_m": p.release_clearance_m,
                "speed_m_s": p.place_descent_speed_m_s,
            },
        )
        self._add_action(
            P0State.PLAN_SUCTION_OFF,
            "DIGITAL_OUTPUT_PLAN",
            "Would request suction release; never emitted by this planner.",
            {"channel": "left_suction", "requested_state": False},
        )
        self._plan_named_pose(P0State.PLAN_RETRACT, "post_place")
        self._add_action(
            P0State.PLAN_VERIFY_PLACED,
            "VERIFICATION_PLAN",
            "Require a fresh image showing the carton inside the requested slot.",
            {
                "expected": "slot_occupied_and_carton_not_attached",
                "slot_id": p.slot_id,
                "failure_code": "PLACE_NOT_CONFIRMED",
            },
        )

    def _plan_named_pose(self, state: P0State, pose_name: str) -> None:
        assert self.pose_document is not None
        pose = self.pose_document["poses"][pose_name]
        self._add_action(
            state,
            "NAMED_JOINT_POSE_PLAN",
            f"Resolve validated paired joint pose {pose_name!r}.",
            {
                "pose_name": pose_name,
                "pose_store_revision": self.pose_document["revision"],
                "pose_store_sha256": self.pose_document["content_sha256"],
                "captured_at": pose.get("captured_at"),
                "speed_scale": self.config.plan.speed_scale,
            },
        )

    def _add_action(
        self,
        state: P0State,
        kind: str,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        self._transition(
            state,
            "PLANNED_NOT_EXECUTED",
            details={"kind": kind},
        )
        self.actions.append(
            PlanAction(
                state=state,
                kind=kind,
                description=description,
                parameters=parameters,
                executed=False,
            )
        )

    def _transition(
        self,
        next_state: P0State,
        outcome: str,
        *,
        failure_code: FailureCode = FailureCode.NONE,
        details: dict[str, Any] | None = None,
    ) -> None:
        if next_state not in _ALLOWED_TRANSITIONS[self.state]:
            raise RuntimeError(
                f"{FailureCode.INVALID_TRANSITION.value}: "
                f"{self.state.value} -> {next_state.value}"
            )
        previous = self.state
        self.state = next_state
        self.log.event(
            previous,
            next_state,
            outcome,
            failure_code=failure_code,
            details=details,
        )

    def _blocked(
        self,
        code: FailureCode,
        message: str,
        details: dict[str, Any],
    ) -> PlanReport:
        if self.state not in {P0State.FAILED, P0State.DRY_RUN_COMPLETE}:
            self._transition(
                P0State.FAILED,
                "BLOCKED",
                failure_code=code,
                details={"message": message, **details},
            )
        return self._finish(ready=False, code=code, message=message)

    def _finish(
        self,
        *,
        ready: bool,
        code: FailureCode,
        message: str,
    ) -> PlanReport:
        report = PlanReport(
            run_id=self.run_id,
            status="DRY_RUN_READY" if ready else "BLOCKED",
            state=self.state,
            ready=ready,
            failure_code=code,
            message=message,
            actions=tuple(self.actions),
            log_dir=self.log.run_dir,
        )
        self.log.summary(report.to_dict())
        return report
