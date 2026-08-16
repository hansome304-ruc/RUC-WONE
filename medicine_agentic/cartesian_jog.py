"""Fail-closed Cartesian jog controller for the left AIRBOT arm.

The controller intentionally exposes only small, base-frame XYZ increments.
Callers cannot choose an arm, endpoint, quaternion, or arbitrary Cartesian
target.  Construction never connects to hardware; every operation borrows the
left-arm connection on ``localhost:50051`` and releases it before returning.
"""
from __future__ import annotations

import copy
import math
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence


LEFT_ARM_HOST = "localhost"
LEFT_ARM_PORT = 50051
ALLOWED_STEP_MM = frozenset({1, 2, 5, 10})
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
SLOW_SPEED_PARAMETERS = {
    "servo_node.moveit_servo.scale.linear": 0.05,
    "servo_node.moveit_servo.scale.rotational": 0.05,
    "servo_node.moveit_servo.scale.joint": 0.05,
    "sdk_server.max_velocity_scaling_factor": 0.1,
    "sdk_server.max_acceleration_scaling_factor": 0.02,
}


class CartesianJogError(RuntimeError):
    """Base class for jog-controller failures."""


class CartesianJogUnavailable(CartesianJogError):
    """The feature or left-arm connection is unavailable."""


class CartesianJogConflict(CartesianJogError):
    """Another operation or teleoperation conflicts with this operation."""


class CartesianJogSafetyViolation(CartesianJogError):
    """A safety precondition was not met."""


class CartesianJogTimeout(CartesianJogError):
    """A bounded operation did not complete in time."""


@dataclass(frozen=True)
class Workspace:
    """Axis-aligned workspace in the physical left-arm base frame, in metres."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    enforce_xy: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Workspace":
        try:
            enforce_xy = value.get("enforce_xy", True)
            if not isinstance(enforce_xy, bool):
                raise TypeError("enforce_xy must be a boolean")
            workspace = cls(
                x_min=float(value["x_min"]),
                x_max=float(value["x_max"]),
                y_min=float(value["y_min"]),
                y_max=float(value["y_max"]),
                z_min=float(value["z_min"]),
                z_max=float(value["z_max"]),
                enforce_xy=enforce_xy,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "workspace requires finite x_min/x_max, y_min/y_max and "
                "z_min/z_max values"
            ) from exc
        values = (
            workspace.x_min,
            workspace.x_max,
            workspace.y_min,
            workspace.y_max,
            workspace.z_min,
            workspace.z_max,
        )
        if not all(math.isfinite(item) for item in values):
            raise ValueError("workspace bounds must be finite")
        if not (
            workspace.x_min < workspace.x_max
            and workspace.y_min < workspace.y_max
            and workspace.z_min < workspace.z_max
        ):
            raise ValueError("workspace minimums must be less than maximums")
        return workspace

    def contains(self, position_m: Sequence[float]) -> bool:
        x, y, z = _finite_vector(position_m, length=3, field="position")
        return bool(
            (
                not self.enforce_xy
                or (
                    self.x_min <= x <= self.x_max
                    and self.y_min <= y <= self.y_max
                )
            )
            and self.z_min <= z <= self.z_max
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "z_min": self.z_min,
            "z_max": self.z_max,
        }


def _finite_vector(
    value: Sequence[float] | Any,
    *,
    length: int,
    field: str,
) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise CartesianJogUnavailable(f"{field} must be numeric") from exc
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise CartesianJogUnavailable(
            f"{field} must contain {length} finite values"
        )
    return result


def _normalise_quaternion(value: Sequence[float] | Any) -> tuple[float, ...]:
    quaternion = _finite_vector(value, length=4, field="quaternion_xyzw")
    norm = math.sqrt(sum(item * item for item in quaternion))
    if norm < 1e-12:
        raise CartesianJogUnavailable("quaternion_xyzw has zero norm")
    result = tuple(item / norm for item in quaternion)
    if result[3] < 0.0:
        result = tuple(-item for item in result)
    return result


def _quaternion_error_deg(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    first = _normalise_quaternion(left)
    second = _normalise_quaternion(right)
    cosine = min(1.0, max(-1.0, abs(sum(a * b for a, b in zip(first, second)))))
    return math.degrees(2.0 * math.acos(cosine))


def _slerp_quaternion(
    start: Sequence[float],
    end: Sequence[float],
    fraction: float,
) -> tuple[float, ...]:
    first = _normalise_quaternion(start)
    second = _normalise_quaternion(end)
    dot = sum(a * b for a, b in zip(first, second))
    if dot < 0.0:
        second = tuple(-item for item in second)
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _normalise_quaternion(
            [a + fraction * (b - a) for a, b in zip(first, second)]
        )
    angle = math.acos(dot)
    sine = math.sin(angle)
    left_weight = math.sin((1.0 - fraction) * angle) / sine
    right_weight = math.sin(fraction * angle) / sine
    return _normalise_quaternion(
        [
            left_weight * a + right_weight * b
            for a, b in zip(first, second)
        ]
    )


def _position_error_m(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    first = _finite_vector(left, length=3, field="left_position_m")
    second = _finite_vector(right, length=3, field="right_position_m")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _median_quaternion(
    values: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    normalised = [_normalise_quaternion(value) for value in values]
    reference = normalised[0]
    aligned = [
        tuple(-item for item in value)
        if sum(a * b for a, b in zip(value, reference)) < 0.0
        else value
        for value in normalised
    ]
    return _normalise_quaternion(
        [_median([value[index] for value in aligned]) for index in range(4)]
    )


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", str(value).split(".")[-1])).upper()


def _normalise_end_pose(value: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise CartesianJogUnavailable(
            "get_end_pose() must return ([x,y,z], [qx,qy,qz,qw])"
        )
    return (
        _finite_vector(value[0], length=3, field="position_m"),
        _normalise_quaternion(value[1]),
    )


class CartesianJogController:
    """Small-step, fixed-orientation Cartesian control for left:50051 only."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        arm_factory: Callable[[str, int], Any] | None = None,
        teleop_running: Callable[[], bool] | None = None,
        planning_mode: Any | None = None,
        waypoint_mode: Any | None = None,
        slow_speed: Any | None = None,
        speed_profiles: Mapping[str, Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        settings = dict(config or {})
        endpoint = settings.get("endpoint", {})
        if endpoint is None:
            endpoint = {}
        if not isinstance(endpoint, Mapping):
            raise ValueError("endpoint must be an object")
        arm_name = str(endpoint.get("arm", "left")).strip().lower()
        if arm_name not in {"left", "right"}:
            raise ValueError("endpoint.arm must be left or right")
        host = str(endpoint.get("host", LEFT_ARM_HOST)).strip()
        try:
            port = int(endpoint.get("port", LEFT_ARM_PORT))
        except (TypeError, ValueError) as exc:
            raise ValueError("endpoint.port must be an integer") from exc
        if not host or not 1 <= port <= 65535:
            raise ValueError("endpoint requires a host and port 1..65535")
        self._arm_name = arm_name
        self._arm_host = host
        self._arm_port = port
        self._arm_label = f"{arm_name} arm"
        self._arm_compound = f"{arm_name}-arm"
        self._feedback_enabled = bool(
            config is not None and settings.get("enabled", False)
        )
        self._feature_enabled = self._feedback_enabled
        self._dry_run = bool(settings.get("dry_run", True))
        workspace_value = settings.get("workspace")
        self._workspace = (
            Workspace.from_mapping(workspace_value)
            if isinstance(workspace_value, Mapping)
            else None
        )
        capture_workspace_value = settings.get("capture_workspace")
        if capture_workspace_value is not None:
            if not isinstance(capture_workspace_value, Mapping):
                raise ValueError("capture_workspace must be an object or null")
            self._capture_workspace = Workspace.from_mapping(
                capture_workspace_value
            )
        else:
            # Backwards compatibility: before these ranges were separated,
            # orientation capture used the motion workspace.
            self._capture_workspace = self._workspace
        workspace_profile_value = settings.get("workspace_profile", "")
        if not isinstance(workspace_profile_value, str):
            raise ValueError("workspace_profile must be a string")
        self._workspace_profile = workspace_profile_value.strip()
        calibrated_floor_profiles = settings.get(
            "calibrated_workspace_floor_z_m_by_profile",
            {},
        )
        if not isinstance(calibrated_floor_profiles, Mapping):
            raise ValueError(
                "calibrated_workspace_floor_z_m_by_profile must be an object"
            )
        self._calibrated_workspace_floor_z_m_by_profile: dict[str, float] = {}
        for profile_name, profile_floor_value in calibrated_floor_profiles.items():
            if not isinstance(profile_name, str) or not profile_name.strip():
                raise ValueError("calibrated workspace profile names must be strings")
            try:
                profile_floor = float(profile_floor_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "calibrated workspace profile floors must be finite"
                ) from exc
            if not math.isfinite(profile_floor):
                raise ValueError(
                    "calibrated workspace profile floors must be finite"
                )
            if self._workspace is None:
                raise ValueError(
                    "calibrated workspace profiles require a motion workspace"
                )
            if not -0.10 <= profile_floor <= self._workspace.z_min:
                raise ValueError(
                    "calibrated workspace profile floors must be between "
                    "-100 mm and the generic motion floor"
                )
            self._calibrated_workspace_floor_z_m_by_profile[
                profile_name.strip()
            ] = profile_floor
        token = settings.get("enable_token")
        self._enable_token = str(token) if token is not None else ""
        if not self._dry_run and not self._enable_token:
            self._feature_enabled = False

        self._capture_sample_count = int(settings.get("capture_sample_count", 5))
        self._verify_sample_count = int(settings.get("verify_sample_count", 3))
        self._sample_interval_s = float(settings.get("sample_interval_s", 0.03))
        self._max_capture_spread_deg = float(
            settings.get("max_capture_spread_deg", 0.2)
        )
        self._max_capture_position_spread_m = float(
            settings.get("max_capture_position_spread_m", 0.0005)
        )
        self._max_pre_orientation_error_deg = float(
            settings.get("max_pre_orientation_error_deg", 1.0)
        )
        self._max_position_error_m = float(
            settings.get("max_position_error_m", 0.0008)
        )
        self._max_orientation_error_deg = float(
            settings.get("max_orientation_error_deg", 0.5)
        )
        self._max_downward_step_mm = int(
            settings.get("max_downward_step_mm", 2)
        )
        manual_jog_workspace_enforced = settings.get(
            "manual_jog_workspace_enforced",
            True,
        )
        if not isinstance(manual_jog_workspace_enforced, bool):
            raise ValueError("manual_jog_workspace_enforced must be a boolean")
        self._manual_jog_workspace_enforced = manual_jog_workspace_enforced
        self._feedback_timeout_s = float(
            settings.get("feedback_timeout_s", 5.0)
        )
        self._restore_feedback_timeout_s = float(
            settings.get(
                "restore_feedback_timeout_s",
                max(self._feedback_timeout_s, 30.0),
            )
        )
        self._feedback_stable_samples = int(
            settings.get("feedback_stable_samples", 3)
        )
        max_offset_value = settings.get(
            "max_offset_from_capture_m",
            {"x": 0.15, "y": 0.15, "z": 0.10},
        )
        if not isinstance(max_offset_value, Mapping):
            raise ValueError("max_offset_from_capture_m must be an object")
        try:
            self._max_offset_from_capture_m = tuple(
                float(max_offset_value[axis]) for axis in ("x", "y", "z")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "max_offset_from_capture_m requires finite x/y/z values"
            ) from exc
        if not all(
            math.isfinite(value) and value > 0.0
            for value in self._max_offset_from_capture_m
        ):
            raise ValueError(
                "max_offset_from_capture_m values must be positive and finite"
            )
        if self._capture_sample_count < 3 or self._verify_sample_count < 1:
            raise ValueError("capture_sample_count >= 3 and verify_sample_count >= 1")
        numeric_settings = (
            self._sample_interval_s,
            self._max_capture_spread_deg,
            self._max_capture_position_spread_m,
            self._max_pre_orientation_error_deg,
            self._max_position_error_m,
            self._max_orientation_error_deg,
            self._feedback_timeout_s,
            self._restore_feedback_timeout_s,
        )
        if (
            not all(math.isfinite(value) and value >= 0.0 for value in numeric_settings)
            or self._max_position_error_m <= 0.0
            or self._feedback_timeout_s <= 0.0
            or self._restore_feedback_timeout_s <= 0.0
            or self._feedback_stable_samples < 2
            or self._max_downward_step_mm not in ALLOWED_STEP_MM
        ):
            raise ValueError("invalid Cartesian jog tolerance or interval setting")

        safe_pose_value = settings.get("safe_vertical_pose")
        self._safe_position: tuple[float, ...] | None = None
        self._safe_quaternion: tuple[float, ...] | None = None
        self._safe_transit_z_m: float | None = None
        self._safe_restore_token = ""
        self._safe_rotation_steps = 0
        if safe_pose_value is not None:
            if not isinstance(safe_pose_value, Mapping):
                raise ValueError("safe_vertical_pose must be an object or null")
            try:
                safe_position = _finite_vector(
                    safe_pose_value["position_m"],
                    length=3,
                    field="safe_vertical_pose.position_m",
                )
                safe_quaternion = _normalise_quaternion(
                    safe_pose_value["quaternion_xyzw"]
                )
                transit_z_m = float(safe_pose_value["transit_z_m"])
                restore_token = str(safe_pose_value["restore_token"])
                rotation_steps = int(safe_pose_value.get("rotation_steps", 4))
            except (CartesianJogError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "safe_vertical_pose requires a valid position, quaternion, "
                    "transit_z_m and restore_token"
                ) from exc
            if not math.isfinite(transit_z_m) or transit_z_m <= 0.0:
                raise ValueError("safe_vertical_pose.transit_z_m must be positive")
            if not 1 <= rotation_steps <= 20:
                raise ValueError("safe_vertical_pose.rotation_steps must be 1..20")
            if not restore_token:
                raise ValueError("safe_vertical_pose.restore_token is required")
            if not self._workspace or not self._workspace.contains(safe_position):
                raise ValueError("safe vertical position is outside motion workspace")
            transit_position = (
                safe_position[0],
                safe_position[1],
                transit_z_m,
            )
            if (
                transit_z_m < safe_position[2]
                or not self._workspace.contains(transit_position)
            ):
                raise ValueError(
                    "safe_vertical_pose.transit_z_m must be at or above the "
                    "target and inside motion workspace"
                )
            self._safe_position = safe_position
            self._safe_quaternion = safe_quaternion
            self._safe_transit_z_m = transit_z_m
            self._safe_restore_token = restore_token
            self._safe_rotation_steps = rotation_steps

        home_pose_value = settings.get("home_joint_pose")
        self._home_joint_positions: tuple[float, ...] | None = None
        self._home_joint_tolerance_rad = 0.12
        self._home_feedback_timeout_s = 30.0
        if home_pose_value is not None:
            if not isinstance(home_pose_value, Mapping):
                raise ValueError("home_joint_pose must be an object or null")
            try:
                home_enabled = home_pose_value.get("enabled", True)
                if not isinstance(home_enabled, bool):
                    raise TypeError("enabled must be a boolean")
                home_positions = _finite_vector(
                    home_pose_value["joint_positions_rad"],
                    length=6,
                    field="home_joint_pose.joint_positions_rad",
                )
                home_tolerance = float(
                    home_pose_value.get("position_tolerance_rad", 0.12)
                )
                home_timeout = float(
                    home_pose_value.get("feedback_timeout_s", 30.0)
                )
            except (CartesianJogError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "home_joint_pose requires six finite joint positions and "
                    "valid tolerance/timeout values"
                ) from exc
            if not 0.001 <= home_tolerance <= 0.5:
                raise ValueError(
                    "home_joint_pose.position_tolerance_rad must be 0.001..0.5"
                )
            if not 1.0 <= home_timeout <= 120.0:
                raise ValueError(
                    "home_joint_pose.feedback_timeout_s must be 1..120"
                )
            if home_enabled:
                self._home_joint_positions = home_positions
            self._home_joint_tolerance_rad = home_tolerance
            self._home_feedback_timeout_s = home_timeout

        self._arm_factory = arm_factory or self._default_arm_factory
        self._teleop_running = teleop_running or (lambda: False)
        self._planning_mode = planning_mode
        self._waypoint_mode = waypoint_mode
        self._slow_speed = slow_speed
        self._speed_profiles = {
            str(name).strip().upper(): profile
            for name, profile in dict(speed_profiles or {}).items()
        }
        if slow_speed is not None:
            self._speed_profiles.setdefault("SLOW", slow_speed)
        self._sleep = sleep

        self._state_lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._active_arm_lock = threading.RLock()
        self._active_arm: Any | None = None
        self._emergency_stop_requested = threading.Event()
        self._session_id = uuid.uuid4().hex
        self._busy = False
        self._enabled = False
        self._locked_quaternion: tuple[float, ...] | None = None
        self._capture_position: tuple[float, ...] | None = None
        self._current_position: tuple[float, ...] | None = None
        self._last_result: dict[str, Any] | None = None
        self._last_error = ""

    @staticmethod
    def _default_arm_factory(host: str, port: int) -> Any:
        try:
            from airbot_py.arm import AIRBOTPlay
        except ImportError as exc:  # pragma: no cover - only on dosw1
            raise CartesianJogUnavailable("airbot_py is unavailable") from exc
        return AIRBOTPlay(url=host, port=port)

    def _teleop_is_running(self) -> bool:
        try:
            return bool(self._teleop_running())
        except Exception as exc:
            raise CartesianJogSafetyViolation(
                "teleop status is unavailable; refusing Cartesian control"
            ) from exc

    def _require_available(self) -> None:
        if not self._feature_enabled:
            raise CartesianJogUnavailable(
                "Cartesian jog is disabled or lacks a real-motion enable token"
            )
        if self._workspace is None:
            raise CartesianJogUnavailable("Cartesian jog workspace is not configured")
        if self._capture_workspace is None:
            raise CartesianJogUnavailable(
                "Cartesian jog capture workspace is not configured"
            )

    def _require_feedback_available(self) -> None:
        if not self._feedback_enabled:
            raise CartesianJogUnavailable("arm feedback endpoint is disabled")

    @contextmanager
    def _operation(self, name: str) -> Iterator[None]:
        if self._emergency_stop_requested.is_set():
            raise CartesianJogConflict("operator stop is active")
        if not self._operation_lock.acquire(blocking=False):
            raise CartesianJogConflict("another Cartesian jog operation is busy")
        if self._emergency_stop_requested.is_set():
            self._operation_lock.release()
            raise CartesianJogConflict("operator stop is active")
        with self._state_lock:
            self._busy = True
            self._last_error = ""
        try:
            yield
        except Exception as exc:
            with self._state_lock:
                self._enabled = False
                # A failed capture, precondition, mode transition, motion, or
                # verification invalidates the whole jog session.  The
                # operator must return to a known downward pose and recapture.
                self._locked_quaternion = None
                self._capture_position = None
                self._current_position = None
                self._last_error = f"{name}: {type(exc).__name__}: {exc}"
            raise
        finally:
            with self._state_lock:
                self._busy = False
            self._operation_lock.release()

    @contextmanager
    def _arm_session(self) -> Iterator[Any]:
        arm = self._arm_factory(self._arm_host, self._arm_port)
        connected = False
        try:
            connected = bool(arm.connect())
            if not connected:
                raise CartesianJogUnavailable(
                    f"cannot connect to {self._arm_label} on "
                    f"{self._arm_host}:{self._arm_port}"
                )
            with self._active_arm_lock:
                self._active_arm = arm
            yield arm
        finally:
            with self._active_arm_lock:
                if self._active_arm is arm:
                    self._active_arm = None
            if connected or arm is not None:
                for method_name in ("disconnect", "close", "shutdown"):
                    method = getattr(arm, method_name, None)
                    if callable(method):
                        try:
                            method()
                        except Exception:
                            pass
                        break

    def _read_pose(self, arm: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
        try:
            return _normalise_end_pose(arm.get_end_pose())
        except CartesianJogError:
            raise
        except Exception as exc:
            raise CartesianJogUnavailable(
                f"failed to read {self._arm_compound} pose: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _read_gripper_position(self, arm: Any) -> float:
        method = getattr(arm, "get_eef_pos", None)
        if not callable(method):
            raise CartesianJogUnavailable(
                f"{self._arm_label} lacks gripper-position feedback"
            )
        try:
            return _finite_vector(
                method(),
                length=1,
                field=f"{self._arm_name}_gripper_position_m",
            )[0]
        except CartesianJogError:
            raise
        except Exception as exc:
            raise CartesianJogUnavailable(
                f"failed to read {self._arm_compound} gripper position: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _read_pose_burst(
        self,
        arm: Any,
        count: int,
    ) -> tuple[tuple[float, ...], tuple[float, ...], float, float]:
        positions: list[tuple[float, ...]] = []
        quaternions: list[tuple[float, ...]] = []
        for index in range(count):
            position, quaternion = self._read_pose(arm)
            positions.append(position)
            quaternions.append(quaternion)
            if index + 1 < count and self._sample_interval_s:
                self._sleep(self._sample_interval_s)
        median_position = tuple(
            _median([position[index] for position in positions])
            for index in range(3)
        )
        median_quaternion = _median_quaternion(quaternions)
        spread_deg = max(
            _quaternion_error_deg(value, median_quaternion)
            for value in quaternions
        )
        position_spread_m = max(
            math.sqrt(
                sum(
                    (value[index] - median_position[index]) ** 2
                    for index in range(3)
                )
            )
            for value in positions
        )
        return (
            median_position,
            median_quaternion,
            spread_deg,
            position_spread_m,
        )

    def _require_idle_if_supported(self, arm: Any) -> None:
        state_method = getattr(arm, "get_state", None)
        if not callable(state_method):
            return
        try:
            state = _enum_name(state_method())
        except Exception as exc:
            raise CartesianJogUnavailable(
                f"failed to read {self._arm_compound} state: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if state != "IDLE":
            raise CartesianJogSafetyViolation(
                f"{self._arm_label} must be IDLE, current state is {state}"
            )

    def status(self) -> dict[str, Any]:
        """Return state only; this method never connects to or commands AIRBOT."""

        with self._state_lock:
            if not self._feature_enabled:
                state = "disabled"
            elif self._busy:
                state = "moving" if self._enabled else "busy"
            elif self._enabled:
                state = "enabled"
            elif self._locked_quaternion is not None:
                state = "captured"
            else:
                state = "idle"
            return {
                "configured": self._feature_enabled,
                "available": bool(
                    self._feature_enabled
                    and self._workspace is not None
                    and self._capture_workspace is not None
                ),
                "state": state,
                "session_id": self._session_id,
                "endpoint": {
                    "arm": self._arm_name,
                    "host": self._arm_host,
                    "port": self._arm_port,
                },
                "frame": f"{self._arm_name}_base",
                # Display/audit metadata only; both workspaces below remain the
                # sole authority for capture and motion admission.
                "workspace_profile": self._workspace_profile,
                "dry_run": self._dry_run,
                "enabled": self._enabled,
                "armed": self._enabled,
                "busy": self._busy,
                # The application layer supplies its shared teleop snapshot.
                # Safety-critical operations still re-check the callback.
                "teleop_running": None,
                "teleop_error": "",
                "orientation_captured": self._locked_quaternion is not None,
                "locked_quaternion_xyzw": (
                    list(self._locked_quaternion)
                    if self._locked_quaternion is not None
                    else None
                ),
                "locked_quaternion": (
                    list(self._locked_quaternion)
                    if self._locked_quaternion is not None
                    else None
                ),
                "current_position_m": (
                    list(self._current_position)
                    if self._current_position is not None
                    else None
                ),
                "capture_position_m": (
                    list(self._capture_position)
                    if self._capture_position is not None
                    else None
                ),
                "allowed_step_mm": sorted(ALLOWED_STEP_MM),
                "max_downward_z_step_mm": self._max_downward_step_mm,
                "manual_jog_workspace_enforced": (
                    self._manual_jog_workspace_enforced
                ),
                "feedback_timeout_s": self._feedback_timeout_s,
                "restore_feedback_timeout_s": self._restore_feedback_timeout_s,
                "feedback_stable_samples": self._feedback_stable_samples,
                "safe_vertical_pose": {
                    "available": bool(
                        self._safe_position is not None
                        and self._safe_quaternion is not None
                        and self._safe_transit_z_m is not None
                    ),
                    "position_m": (
                        list(self._safe_position)
                        if self._safe_position is not None
                        else None
                    ),
                    "quaternion_xyzw": (
                        list(self._safe_quaternion)
                        if self._safe_quaternion is not None
                        else None
                    ),
                    "transit_z_m": self._safe_transit_z_m,
                    "planner": "AIRBOT Cartesian waypoint planner",
                },
                "home_joint_pose": {
                    "available": self._home_joint_positions is not None,
                    "joint_positions_rad": (
                        list(self._home_joint_positions)
                        if self._home_joint_positions is not None
                        else None
                    ),
                    "position_tolerance_rad": self._home_joint_tolerance_rad,
                    "feedback_timeout_s": self._home_feedback_timeout_s,
                    "planner": "AIRBOT joint planner",
                    "affects": [f"{self._arm_name}_arm"],
                },
                "max_offset_from_capture_m": {
                    axis: self._max_offset_from_capture_m[index]
                    for index, axis in enumerate(("x", "y", "z"))
                },
                "workspace_m": (
                    self._workspace.as_dict()
                    if self._workspace is not None
                    else None
                ),
                "capture_workspace_m": (
                    self._capture_workspace.as_dict()
                    if self._capture_workspace is not None
                    else None
                ),
                "xy_workspace_enforced": (
                    self._workspace.enforce_xy
                    if self._workspace is not None
                    else None
                ),
                "xy_capture_offset_enforced": (
                    self._workspace.enforce_xy
                    if self._workspace is not None
                    else None
                ),
                "last_result": copy.deepcopy(self._last_result),
                "last_error": self._last_error,
            }

    def read_current_pose(
        self,
        *,
        allow_during_teleop: bool = False,
    ) -> dict[str, Any]:
        """Read Cartesian and joint feedback without changing control mode."""

        self._require_feedback_available()
        if self._teleop_is_running() and not allow_during_teleop:
            raise CartesianJogSafetyViolation(
                "stop teleoperation before reading a standalone arm pose"
            )
        if not self._operation_lock.acquire(blocking=False):
            raise CartesianJogConflict("another Cartesian jog operation is busy")
        try:
            with self._arm_session() as arm:
                position, quaternion = self._read_pose(arm)
                joints = self._read_joint_positions(arm)
                gripper_position = self._read_gripper_position(arm)
            return {
                "arm": self._arm_name,
                "frame": f"{self._arm_name}_base",
                "position_m": list(position),
                "quaternion_xyzw": list(quaternion),
                "joint_positions_rad": list(joints),
                "gripper_position_m": gripper_position,
                "captured_at": time.time(),
                "read_only": True,
                "captured_during_teleop": bool(self._teleop_is_running()),
            }
        finally:
            self._operation_lock.release()

    def capture_orientation(self) -> dict[str, Any]:
        """Capture a stable quaternion and the bounded session origin position."""

        with self._operation("capture_orientation"):
            self._require_available()
            # A recapture attempt invalidates the previous reference immediately.
            # An unstable/failed recapture must never leave a stale quaternion armed.
            with self._state_lock:
                self._enabled = False
                self._locked_quaternion = None
                self._capture_position = None
                self._current_position = None
            if self._teleop_is_running():
                raise CartesianJogConflict(
                    "stop teleoperation before capturing the locked orientation"
                )
            with self._arm_session() as arm:
                self._require_idle_if_supported(arm)
                (
                    position,
                    quaternion,
                    spread_deg,
                    position_spread_m,
                ) = self._read_pose_burst(
                    arm, self._capture_sample_count
                )
                self._require_idle_if_supported(arm)
            if (
                not self._capture_workspace
                or not self._capture_workspace.contains(position)
            ):
                raise CartesianJogSafetyViolation(
                    "current left-arm position is outside the configured "
                    "capture workspace"
                )
            if spread_deg > self._max_capture_spread_deg:
                raise CartesianJogSafetyViolation(
                    "left-arm orientation is not stable "
                    f"({spread_deg:.3f} deg > {self._max_capture_spread_deg:.3f} deg)"
                )
            if position_spread_m > self._max_capture_position_spread_m:
                raise CartesianJogSafetyViolation(
                    "left-arm position is not stable "
                    f"({position_spread_m * 1000.0:.3f} mm > "
                    f"{self._max_capture_position_spread_m * 1000.0:.3f} mm)"
                )
            with self._state_lock:
                self._locked_quaternion = quaternion
                self._capture_position = position
                self._current_position = position
                self._last_result = {
                    "operation": "capture_orientation",
                    "position_m": list(position),
                    "quaternion_xyzw": list(quaternion),
                    "sample_count": self._capture_sample_count,
                    "orientation_spread_deg": spread_deg,
                    "position_spread_m": position_spread_m,
                }
                return copy.deepcopy(self._last_result)

    def enable(
        self,
        enable_token: str,
        *,
        area_clear: bool,
        estop_ready: bool,
    ) -> dict[str, Any]:
        """Arm jog only after a fresh, stable, read-only live-pose check."""

        with self._operation("enable"):
            self._require_available()
            if self._teleop_is_running():
                raise CartesianJogConflict("stop teleoperation before enabling jog")
            with self._state_lock:
                self._enabled = False
                locked_quaternion = self._locked_quaternion
                capture_position = self._capture_position
            if locked_quaternion is None:
                raise CartesianJogSafetyViolation(
                    "capture the downward orientation before enabling jog"
                )
            if capture_position is None:
                raise CartesianJogSafetyViolation("capture position is missing")
            if not self._workspace or not self._workspace.contains(
                capture_position
            ):
                raise CartesianJogSafetyViolation(
                    "captured left-arm position is outside the configured "
                    "motion workspace"
                )
            if not isinstance(enable_token, str) or not enable_token:
                raise CartesianJogSafetyViolation("an explicit enable token is required")
            if not self._dry_run and enable_token != self._enable_token:
                raise CartesianJogSafetyViolation("invalid enable token")
            if not area_clear or not estop_ready:
                raise CartesianJogSafetyViolation(
                    "area_clear and estop_ready must both be confirmed"
                )

            # Enabling is deliberately read-only.  Do not switch mode, set a
            # speed profile, or send any pose while establishing this snapshot.
            with self._arm_session() as arm:
                self._require_idle_if_supported(arm)
                (
                    live_position,
                    live_quaternion,
                    orientation_spread_deg,
                    position_spread_m,
                ) = self._read_pose_burst(arm, self._verify_sample_count)
                self._require_idle_if_supported(arm)

            if not self._workspace or not self._workspace.contains(live_position):
                raise CartesianJogSafetyViolation(
                    "live left-arm position is outside the configured motion workspace"
                )
            orientation_error_deg = _quaternion_error_deg(
                live_quaternion,
                locked_quaternion,
            )
            if orientation_error_deg > self._max_pre_orientation_error_deg:
                raise CartesianJogSafetyViolation(
                    "live orientation differs from the locked orientation "
                    f"by {orientation_error_deg:.3f} deg"
                )
            if orientation_spread_deg > self._max_capture_spread_deg:
                raise CartesianJogSafetyViolation(
                    "live left-arm orientation is not stable "
                    f"({orientation_spread_deg:.3f} deg > "
                    f"{self._max_capture_spread_deg:.3f} deg)"
                )
            if position_spread_m > self._max_capture_position_spread_m:
                raise CartesianJogSafetyViolation(
                    "live left-arm position is not stable "
                    f"({position_spread_m * 1000.0:.3f} mm > "
                    f"{self._max_capture_position_spread_m * 1000.0:.3f} mm)"
                )
            for index, coordinate in enumerate(live_position):
                if (
                    index < 2
                    and self._workspace is not None
                    and not self._workspace.enforce_xy
                ):
                    continue
                if (
                    abs(coordinate - capture_position[index])
                    > self._max_offset_from_capture_m[index]
                ):
                    axis_name = ("x", "y", "z")[index]
                    raise CartesianJogSafetyViolation(
                        f"live pose exceeds the allowed {axis_name.upper()} "
                        "offset from the captured pose"
                    )
            if self._teleop_is_running():
                raise CartesianJogConflict(
                    "teleoperation started during the enable check"
                )
            with self._state_lock:
                self._current_position = live_position
                self._enabled = True
                self._last_result = {
                    "operation": "enable",
                    "position_m": list(live_position),
                    "orientation_error_deg": orientation_error_deg,
                    "orientation_spread_deg": orientation_spread_deg,
                    "position_spread_m": position_spread_m,
                }
        return self.status()

    def disable(self) -> dict[str, Any]:
        """Disable future jogs.  It never issues an arm command."""

        with self._state_lock:
            self._enabled = False
        return self.status()

    def clear_emergency_stop(self) -> None:
        """Allow a newly operator-confirmed transaction after a stop."""

        self._emergency_stop_requested.clear()

    def emergency_stop(self) -> dict[str, Any]:
        """Interrupt an active planning RPC and hold the measured joint pose."""

        self._emergency_stop_requested.set()
        with self._state_lock:
            self._enabled = False
        with self._active_arm_lock:
            arm = self._active_arm
        if arm is None:
            return {
                "ok": True,
                "active_session": False,
                "hold_commanded": False,
                "session_released": True,
            }

        hold_commanded = False
        errors: list[str] = []
        try:
            joints = [float(value) for value in arm.get_joint_pos()]
            if len(joints) != 6 or not all(math.isfinite(value) for value in joints):
                raise RuntimeError("arm did not return six finite joints")
            from airbot_py.arm import RobotMode

            if arm.switch_mode(RobotMode.SERVO_JOINT_POS) is False:
                raise RuntimeError("AIRBOT rejected emergency joint-servo mode")
            arm.servo_joint_pos(joints)
            self._sleep(0.08)
            hold_commanded = True
        except Exception as exc:
            errors.append(f"hold: {type(exc).__name__}: {exc}")
        finally:
            try:
                arm.disconnect()
            except Exception as exc:
                errors.append(f"disconnect: {type(exc).__name__}: {exc}")
            with self._active_arm_lock:
                if self._active_arm is arm:
                    self._active_arm = None
        return {
            "ok": not errors,
            "active_session": True,
            "hold_commanded": hold_commanded,
            "session_released": True,
            "errors": errors,
        }

    def close(self) -> None:
        """Fail closed; no persistent AIRBOT connection exists to close."""

        with self._state_lock:
            self._enabled = False
            self._locked_quaternion = None
            self._capture_position = None
            self._current_position = None

    def _motion_symbols(self) -> tuple[Any, Any, Any]:
        if (
            self._planning_mode is not None
            and self._waypoint_mode is not None
            and self._slow_speed is not None
        ):
            return self._planning_mode, self._waypoint_mode, self._slow_speed
        try:
            from airbot_py.arm import RobotMode, SpeedProfile
        except ImportError as exc:  # pragma: no cover - only on dosw1
            raise CartesianJogUnavailable(
                "AIRBOT motion enums are unavailable"
            ) from exc
        return (
            RobotMode.PLANNING_POS,
            RobotMode.PLANNING_WAYPOINTS_PATH,
            SpeedProfile.SLOW,
        )

    def _switch_and_verify_mode(
        self,
        arm: Any,
        mode: Any,
        expected_name: str,
    ) -> None:
        if arm.switch_mode(mode) is not True:
            raise CartesianJogUnavailable(
                f"{self._arm_label} rejected {expected_name}"
            )
        mode_method = getattr(arm, "get_control_mode", None)
        if callable(mode_method):
            actual_mode = _enum_name(mode_method())
            if actual_mode != expected_name:
                raise CartesianJogSafetyViolation(
                    f"{self._arm_label} did not enter {expected_name} "
                    f"(current={actual_mode})"
                )

    def _set_and_verify_slow_speed(
        self,
        arm: Any,
        slow_speed: Any,
    ) -> dict[str, float]:
        arm.set_speed_profile(slow_speed)
        get_params = getattr(arm, "get_params", None)
        if not callable(get_params):
            raise CartesianJogUnavailable(
                f"{self._arm_label} cannot read back SLOW speed parameters"
            )
        actual = get_params(list(SLOW_SPEED_PARAMETERS))
        if not isinstance(actual, Mapping):
            raise CartesianJogUnavailable(
                f"{self._arm_label} returned invalid SLOW speed parameters"
            )
        verified: dict[str, float] = {}
        for key, expected in SLOW_SPEED_PARAMETERS.items():
            if key not in actual:
                raise CartesianJogUnavailable(
                    f"SLOW speed readback is missing {key}"
                )
            try:
                value = float(actual[key])
            except (TypeError, ValueError) as exc:
                raise CartesianJogUnavailable(
                    f"invalid SLOW speed readback for {key}"
                ) from exc
            if not math.isfinite(value) or not math.isclose(
                value,
                expected,
                rel_tol=1e-6,
                abs_tol=1e-9,
            ):
                raise CartesianJogUnavailable(
                    f"SLOW speed readback mismatch for {key}: "
                    f"expected {expected}, got {actual[key]!r}"
                )
            verified[key] = value
        return verified

    def _set_and_verify_named_speed(
        self,
        arm: Any,
        profile_name: str,
        slow_speed: Any,
    ) -> dict[str, float]:
        """Apply one bounded SDK speed profile and verify finite scaling."""

        normalized = str(profile_name).strip().upper()
        if normalized not in {"SLOW", "DEFAULT", "FAST"}:
            raise CartesianJogUnavailable(
                "speed profile must be SLOW, DEFAULT or FAST"
            )
        if normalized == "SLOW":
            return self._set_and_verify_slow_speed(arm, slow_speed)
        profile = self._speed_profiles.get(normalized)
        if profile is None:
            try:
                from airbot_py.arm import SpeedProfile
            except ImportError as exc:  # pragma: no cover - only on dosw1
                raise CartesianJogUnavailable(
                    "AIRBOT speed profiles are unavailable"
                ) from exc
            profile = getattr(SpeedProfile, normalized)
        accepted = arm.set_speed_profile(profile)
        if accepted is False:
            raise CartesianJogUnavailable(
                f"{self._arm_label} rejected {normalized} speed profile"
            )
        get_params = getattr(arm, "get_params", None)
        if not callable(get_params):
            raise CartesianJogUnavailable(
                f"{self._arm_label} cannot read back {normalized} speed parameters"
            )
        actual = get_params(list(SLOW_SPEED_PARAMETERS))
        if not isinstance(actual, Mapping):
            raise CartesianJogUnavailable(
                f"{self._arm_label} returned invalid {normalized} speed parameters"
            )
        verified: dict[str, float] = {}
        for key in SLOW_SPEED_PARAMETERS:
            try:
                value = float(actual[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise CartesianJogUnavailable(
                    f"invalid {normalized} speed readback for {key}"
                ) from exc
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise CartesianJogUnavailable(
                    f"unsafe {normalized} speed readback for {key}: {value!r}"
                )
            verified[key] = value
        return verified

    def _prepare_for_motion(
        self,
        arm: Any,
        *,
        mode: Any,
        expected_mode_name: str,
        slow_speed: Any,
        speed_profile: str = "DEFAULT",
    ) -> dict[str, float]:
        try:
            state_method = getattr(arm, "get_state", None)
            if callable(state_method):
                state = _enum_name(state_method())
                if state != "IDLE":
                    raise CartesianJogSafetyViolation(
                        f"{self._arm_label} must be IDLE, current state is {state}"
                    )
            self._switch_and_verify_mode(arm, mode, expected_mode_name)
            return self._set_and_verify_named_speed(
                arm,
                speed_profile,
                slow_speed,
            )
        except CartesianJogError:
            raise
        except Exception as exc:
            raise CartesianJogUnavailable(
                f"failed to prepare {self._arm_label}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _read_joint_positions(self, arm: Any) -> tuple[float, ...]:
        try:
            return _finite_vector(
                arm.get_joint_pos(),
                length=6,
                field=f"{self._arm_name}_arm_joint_positions_rad",
            )
        except CartesianJogError:
            raise
        except Exception as exc:
            raise CartesianJogUnavailable(
                f"failed to read {self._arm_compound} joint positions: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _wait_for_joint_target(
        self,
        arm: Any,
        target: Sequence[float],
    ) -> tuple[tuple[float, ...], float]:
        target_positions = _finite_vector(
            target,
            length=6,
            field="home_joint_target_rad",
        )
        poll_interval = max(self._sample_interval_s, 0.05)
        attempts = max(
            self._feedback_stable_samples,
            int(math.ceil(self._home_feedback_timeout_s / poll_interval))
            + self._feedback_stable_samples,
        )
        stable = 0
        last_positions: tuple[float, ...] | None = None
        last_error = math.inf
        last_state = "UNKNOWN"
        for attempt in range(attempts):
            state_method = getattr(arm, "get_state", None)
            last_state = (
                _enum_name(state_method())
                if callable(state_method)
                else "IDLE"
            )
            last_positions = self._read_joint_positions(arm)
            last_error = max(
                abs(actual - expected)
                for actual, expected in zip(last_positions, target_positions)
            )
            if (
                last_state == "IDLE"
                and last_error <= self._home_joint_tolerance_rad
            ):
                stable += 1
                if stable >= self._feedback_stable_samples:
                    return last_positions, last_error
            else:
                stable = 0
            if attempt + 1 < attempts:
                self._sleep(poll_interval)
        raise CartesianJogSafetyViolation(
            f"{self._arm_compound} home feedback did not settle "
            f"(state={last_state}, max_joint_error={last_error:.4f} rad)"
        )

    def _wait_for_verified_target(
        self,
        arm: Any,
        target_position: Sequence[float],
        target_quaternion: Sequence[float],
        *,
        timeout_s: float | None = None,
    ) -> tuple[tuple[float, ...], tuple[float, ...], float, float]:
        poll_interval = max(self._sample_interval_s, 0.02)
        feedback_timeout_s = (
            self._feedback_timeout_s
            if timeout_s is None
            else timeout_s
        )
        attempts = max(
            self._feedback_stable_samples,
            int(math.ceil(feedback_timeout_s / poll_interval))
            + self._feedback_stable_samples,
        )
        stable = 0
        last_position: tuple[float, ...] | None = None
        last_quaternion: tuple[float, ...] | None = None
        last_position_error = math.inf
        last_orientation_error = math.inf
        last_state = "UNKNOWN"
        for attempt in range(attempts):
            state_method = getattr(arm, "get_state", None)
            last_state = (
                _enum_name(state_method())
                if callable(state_method)
                else "IDLE"
            )
            last_position, last_quaternion = self._read_pose(arm)
            last_position_error = _position_error_m(
                last_position,
                target_position,
            )
            last_orientation_error = _quaternion_error_deg(
                last_quaternion,
                target_quaternion,
            )
            if (
                last_state == "IDLE"
                and last_position_error <= self._max_position_error_m
                and last_orientation_error <= self._max_orientation_error_deg
            ):
                stable += 1
                if stable >= self._feedback_stable_samples:
                    return (
                        last_position,
                        last_quaternion,
                        last_position_error,
                        last_orientation_error,
                    )
            else:
                stable = 0
            if attempt + 1 < attempts and self._sample_interval_s:
                self._sleep(self._sample_interval_s)
        raise CartesianJogSafetyViolation(
            "post-motion feedback did not settle "
            f"(state={last_state}, "
            f"position_error={last_position_error * 1000.0:.3f} mm, "
            f"orientation_error={last_orientation_error:.3f} deg)"
        )

    @staticmethod
    def _append_linear_position_segment(
        waypoints: list[list[list[float]]],
        target_position: Sequence[float],
        quaternion: Sequence[float],
        *,
        max_step_m: float = 0.005,
    ) -> None:
        start_position = tuple(waypoints[-1][0])
        target = _finite_vector(
            target_position,
            length=3,
            field="target_position_m",
        )
        distance = _position_error_m(start_position, target)
        if distance < 1e-9:
            return
        steps = max(1, int(math.ceil(distance / max_step_m)))
        for index in range(1, steps + 1):
            fraction = index / steps
            position = [
                start + fraction * (end - start)
                for start, end in zip(start_position, target)
            ]
            waypoints.append([position, list(_normalise_quaternion(quaternion))])

    def _safe_vertical_waypoints(
        self,
        current_position: Sequence[float],
        current_quaternion: Sequence[float],
    ) -> list[list[list[float]]]:
        if (
            self._safe_position is None
            or self._safe_quaternion is None
            or self._safe_transit_z_m is None
            or self._workspace is None
        ):
            raise CartesianJogUnavailable("safe vertical pose is not configured")
        start_position = _finite_vector(
            current_position,
            length=3,
            field="current_position_m",
        )
        start_quaternion = _normalise_quaternion(current_quaternion)
        transit_z = max(start_position[2], self._safe_transit_z_m)
        transit_position = [start_position[0], start_position[1], transit_z]
        waypoints: list[list[list[float]]] = [
            [list(start_position), list(start_quaternion)]
        ]
        self._append_linear_position_segment(
            waypoints,
            transit_position,
            start_quaternion,
        )
        rotation_position = list(waypoints[-1][0])
        for index in range(1, self._safe_rotation_steps + 1):
            waypoints.append(
                [
                    list(rotation_position),
                    list(
                        _slerp_quaternion(
                            start_quaternion,
                            self._safe_quaternion,
                            index / self._safe_rotation_steps,
                        )
                    ),
                ]
            )
        self._append_linear_position_segment(
            waypoints,
            [self._safe_position[0], self._safe_position[1], transit_z],
            self._safe_quaternion,
        )
        self._append_linear_position_segment(
            waypoints,
            self._safe_position,
            self._safe_quaternion,
        )
        if not all(self._workspace.contains(item[0]) for item in waypoints):
            raise CartesianJogSafetyViolation(
                "safe vertical path leaves the configured motion workspace"
            )
        return waypoints

    def jog(self, axis: str, step_mm: int) -> dict[str, Any]:
        """Move 1/2/5/10 mm along one physical left-base axis."""

        if axis not in AXIS_INDEX:
            with self._state_lock:
                self._enabled = False
            raise CartesianJogSafetyViolation("axis must be one of x/y/z")
        if isinstance(step_mm, bool) or not isinstance(step_mm, int):
            with self._state_lock:
                self._enabled = False
            raise CartesianJogSafetyViolation("step_mm must be a signed integer")
        if step_mm == 0 or abs(step_mm) not in ALLOWED_STEP_MM:
            with self._state_lock:
                self._enabled = False
            raise CartesianJogSafetyViolation(
                "absolute step_mm must be one of 1, 2, 5 or 10"
            )
        if axis == "z" and step_mm < 0 and abs(step_mm) > self._max_downward_step_mm:
            with self._state_lock:
                self._enabled = False
            raise CartesianJogSafetyViolation(
                f"downward Z steps are limited to {self._max_downward_step_mm} mm"
            )

        with self._operation("jog"):
            self._require_available()
            if self._teleop_is_running():
                raise CartesianJogConflict("teleoperation is running")
            with self._state_lock:
                if not self._enabled:
                    raise CartesianJogSafetyViolation("Cartesian jog is not enabled")
                locked_quaternion = self._locked_quaternion
                capture_position = self._capture_position
            if locked_quaternion is None:
                raise CartesianJogSafetyViolation("locked orientation is missing")
            if capture_position is None:
                raise CartesianJogSafetyViolation("capture position is missing")

            with self._arm_session() as arm:
                current_position, current_quaternion = self._read_pose(arm)
                pre_orientation_error = _quaternion_error_deg(
                    current_quaternion,
                    locked_quaternion,
                )
                if pre_orientation_error > self._max_pre_orientation_error_deg:
                    raise CartesianJogSafetyViolation(
                        "current orientation differs from the locked orientation "
                        f"by {pre_orientation_error:.3f} deg"
                    )
                target_position = list(current_position)
                target_position[AXIS_INDEX[axis]] += step_mm / 1000.0
                if self._manual_jog_workspace_enforced:
                    if not self._workspace or not self._workspace.contains(
                        current_position
                    ):
                        raise CartesianJogSafetyViolation(
                            "current left-arm position is outside the configured workspace"
                        )
                    if not self._workspace.contains(target_position):
                        raise CartesianJogSafetyViolation(
                            "target left-arm position is outside the configured workspace"
                        )
                    for index, coordinate in enumerate(target_position):
                        if index < 2 and not self._workspace.enforce_xy:
                            continue
                        if (
                            abs(coordinate - capture_position[index])
                            > self._max_offset_from_capture_m[index]
                        ):
                            axis_name = ("x", "y", "z")[index]
                            raise CartesianJogSafetyViolation(
                                f"target exceeds the allowed {axis_name.upper()} "
                                "offset from the captured pose"
                            )

                target_pose = [target_position, list(locked_quaternion)]
                if self._dry_run:
                    actual_for_cache = tuple(current_position)
                    result = {
                        "operation": "jog",
                        "dry_run": True,
                        "executed": False,
                        "axis": axis,
                        "step_mm": step_mm,
                        "current_position_m": list(current_position),
                        "target_position_m": target_position,
                        "locked_quaternion_xyzw": list(locked_quaternion),
                        "pre_orientation_error_deg": pre_orientation_error,
                    }
                else:
                    # Re-check after obtaining the connection, immediately before motion.
                    if self._teleop_is_running():
                        raise CartesianJogConflict("teleoperation started before motion")
                    planning_mode, _waypoint_mode, slow_speed = (
                        self._motion_symbols()
                    )
                    verified_speed_parameters = self._prepare_for_motion(
                        arm,
                        mode=planning_mode,
                        expected_mode_name="PLANNING_POS",
                        slow_speed=slow_speed,
                    )
                    try:
                        moved = arm.move_to_cart_pose(target_pose, blocking=True)
                    except TimeoutError as exc:
                        raise CartesianJogTimeout(
                            "left-arm Cartesian motion timed out"
                        ) from exc
                    except Exception as exc:
                        raise CartesianJogUnavailable(
                            f"left-arm Cartesian motion failed: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    if moved is not True:
                        raise CartesianJogUnavailable(
                            "left arm rejected Cartesian target"
                        )
                    (
                        actual_position,
                        actual_quaternion,
                        position_error,
                        orientation_error,
                    ) = self._wait_for_verified_target(
                        arm,
                        target_position,
                        locked_quaternion,
                    )
                    result = {
                        "operation": "jog",
                        "dry_run": False,
                        "executed": True,
                        "axis": axis,
                        "step_mm": step_mm,
                        "current_position_m": list(current_position),
                        "target_position_m": target_position,
                        "actual_position_m": list(actual_position),
                        "locked_quaternion_xyzw": list(locked_quaternion),
                        "position_error_m": position_error,
                        "orientation_error_deg": orientation_error,
                        "feedback_stable_samples": (
                            self._feedback_stable_samples
                        ),
                        "verified_speed_parameters": verified_speed_parameters,
                    }
                    actual_for_cache = tuple(actual_position)
            with self._state_lock:
                self._current_position = actual_for_cache
                self._last_result = result
                return copy.deepcopy(result)

    def move_fixed_orientation_path(
        self,
        target_positions_m: Sequence[Sequence[float]],
        *,
        operation: str = "fixed_orientation_path",
        calibrated_minimum_z_m: float | None = None,
        calibrated_workspace_profile: str | None = None,
        speed_profile: str = "DEFAULT",
        verify_each_target: bool = True,
    ) -> dict[str, Any]:
        """Execute a short absolute XYZ path while preserving the captured pose."""

        targets = [
            _finite_vector(value, length=3, field="target_position_m")
            for value in target_positions_m
        ]
        if not targets or len(targets) > 8:
            raise CartesianJogSafetyViolation(
                "fixed-orientation path requires between 1 and 8 targets"
            )
        if not isinstance(verify_each_target, bool):
            raise CartesianJogSafetyViolation(
                "verify_each_target must be a boolean"
            )

        with self._operation(operation):
            self._require_available()
            if self._teleop_is_running():
                raise CartesianJogConflict("teleoperation is running")
            with self._state_lock:
                if not self._enabled:
                    raise CartesianJogSafetyViolation(
                        "Cartesian jog is not enabled"
                    )
                locked_quaternion = self._locked_quaternion
                capture_position = self._capture_position
            if locked_quaternion is None or capture_position is None:
                raise CartesianJogSafetyViolation(
                    "captured fixed orientation is missing"
                )
            if self._workspace is None:
                raise CartesianJogUnavailable("motion workspace is not configured")

            effective_workspace = self._workspace
            if calibrated_workspace_profile is not None:
                if calibrated_minimum_z_m is not None:
                    raise CartesianJogSafetyViolation(
                        "use either a calibrated minimum Z or a calibrated "
                        "workspace profile, not both"
                    )
                if (
                    not isinstance(calibrated_workspace_profile, str)
                    or not calibrated_workspace_profile.strip()
                ):
                    raise CartesianJogSafetyViolation(
                        "calibrated workspace profile must be a non-empty string"
                    )
                profile_name = calibrated_workspace_profile.strip()
                calibrated_floor = (
                    self._calibrated_workspace_floor_z_m_by_profile.get(
                        profile_name
                    )
                )
                if calibrated_floor is None:
                    raise CartesianJogSafetyViolation(
                        f"unknown calibrated workspace profile: {profile_name}"
                    )
                effective_workspace = Workspace(
                    x_min=self._workspace.x_min,
                    x_max=self._workspace.x_max,
                    y_min=self._workspace.y_min,
                    y_max=self._workspace.y_max,
                    z_min=min(self._workspace.z_min, calibrated_floor),
                    z_max=self._workspace.z_max,
                    enforce_xy=self._workspace.enforce_xy,
                )
            elif calibrated_minimum_z_m is not None:
                try:
                    calibrated_floor = float(calibrated_minimum_z_m)
                except (TypeError, ValueError) as exc:
                    raise CartesianJogSafetyViolation(
                        "calibrated minimum Z must be finite"
                    ) from exc
                if not math.isfinite(calibrated_floor):
                    raise CartesianJogSafetyViolation(
                        "calibrated minimum Z must be finite"
                    )
                capture_floor = (
                    self._capture_workspace.z_min
                    if self._capture_workspace is not None
                    else self._workspace.z_min
                )
                if (
                    calibrated_floor < capture_floor - 1e-9
                    or self._workspace.z_min - calibrated_floor > 0.005 + 1e-9
                ):
                    raise CartesianJogSafetyViolation(
                        "calibrated minimum Z exceeds the bounded 5 mm relaxation"
                    )
                effective_workspace = Workspace(
                    x_min=self._workspace.x_min,
                    x_max=self._workspace.x_max,
                    y_min=self._workspace.y_min,
                    y_max=self._workspace.y_max,
                    z_min=min(self._workspace.z_min, calibrated_floor),
                    z_max=self._workspace.z_max,
                    enforce_xy=self._workspace.enforce_xy,
                )

            for target in targets:
                if not effective_workspace.contains(target):
                    raise CartesianJogSafetyViolation(
                        "fixed-orientation target is outside the configured workspace"
                    )

            with self._arm_session() as arm:
                current_position, current_quaternion = self._read_pose(arm)
                pre_orientation_error = _quaternion_error_deg(
                    current_quaternion,
                    locked_quaternion,
                )
                if pre_orientation_error > self._max_pre_orientation_error_deg:
                    raise CartesianJogSafetyViolation(
                        "current orientation differs from the locked orientation "
                        f"by {pre_orientation_error:.3f} deg"
                    )
                # A calibrated pick may end up to 5 mm below the generic jog
                # floor.  The following lift must be allowed to start from
                # that same bounded contact workspace; otherwise descent is
                # accepted but the first upward target is rejected.
                if not effective_workspace.contains(current_position):
                    raise CartesianJogSafetyViolation(
                        "current left-arm position is outside the configured workspace"
                    )

                result: dict[str, Any] = {
                    "operation": operation,
                    "dry_run": self._dry_run,
                    "executed": False,
                    "current_position_m": list(current_position),
                    "target_positions_m": [list(value) for value in targets],
                    "effective_minimum_z_m": effective_workspace.z_min,
                    "calibrated_workspace_profile": calibrated_workspace_profile,
                    "locked_quaternion_xyzw": list(locked_quaternion),
                    "pre_orientation_error_deg": pre_orientation_error,
                    "verify_each_target": verify_each_target,
                }
                actual_position = current_position
                actual_quaternion = current_quaternion
                completed_targets: list[list[float]] = []
                sdk_reported_success_by_target: list[bool] = []
                verified_speed_parameters: dict[str, Any] | None = None
                final_position_error = 0.0
                final_orientation_error = pre_orientation_error
                if not self._dry_run:
                    if self._teleop_is_running():
                        raise CartesianJogConflict(
                            "teleoperation started before motion"
                        )
                    planning_mode, _waypoint_mode, slow_speed = (
                        self._motion_symbols()
                    )
                    verified_speed_parameters = self._prepare_for_motion(
                        arm,
                        mode=planning_mode,
                        expected_mode_name="PLANNING_POS",
                        slow_speed=slow_speed,
                        speed_profile=speed_profile,
                    )
                    for target_index, target in enumerate(targets):
                        if self._teleop_is_running():
                            raise CartesianJogConflict(
                                "teleoperation started during motion"
                            )
                        target_pose = [list(target), list(locked_quaternion)]
                        try:
                            moved = arm.move_to_cart_pose(
                                target_pose,
                                blocking=True,
                            )
                        except TimeoutError as exc:
                            raise CartesianJogTimeout(
                                "fixed-orientation Cartesian path timed out"
                            ) from exc
                        except Exception as exc:
                            raise CartesianJogUnavailable(
                                "fixed-orientation Cartesian path failed: "
                                f"{type(exc).__name__}: {exc}"
                            ) from exc
                        if moved is not True:
                            raise CartesianJogUnavailable(
                                f"{operation} target was rejected by the arm"
                            )
                        if verify_each_target or target_index == len(targets) - 1:
                            (
                                actual_position,
                                actual_quaternion,
                                final_position_error,
                                final_orientation_error,
                            ) = self._wait_for_verified_target(
                                arm,
                                target,
                                locked_quaternion,
                            )
                        sdk_reported_success_by_target.append(moved is True)
                        completed_targets.append(list(target))
                    result.update(
                        {
                            "executed": True,
                            "completed_targets_m": completed_targets,
                            "sdk_reported_success_by_target": (
                                sdk_reported_success_by_target
                            ),
                            "actual_position_m": list(actual_position),
                            "actual_quaternion_xyzw": list(actual_quaternion),
                            "position_error_m": final_position_error,
                            "orientation_error_deg": final_orientation_error,
                            "feedback_stable_samples": (
                                self._feedback_stable_samples
                            ),
                            "verified_speed_parameters": (
                                verified_speed_parameters
                            ),
                            "speed_profile": str(speed_profile).strip().upper(),
                        }
                    )
            with self._state_lock:
                self._current_position = tuple(actual_position)
                self._last_result = result
                return copy.deepcopy(result)

    def move_to_fixed_orientation_entry(
        self,
        target_position_m: Sequence[float],
        target_quaternion_xyzw: Sequence[float],
        *,
        transit_z_m: float,
        enable_token: str,
        area_clear: bool,
        estop_ready: bool,
        operation: str = "fixed_orientation_entry",
        calibrated_workspace_profile: str | None = None,
        use_configured_safe_transit: bool = True,
    ) -> dict[str, Any]:
        """Reach one approach pose and arm fixed-orientation motion.

        The bounded target sequence uses the configured, known-reachable
        vertical transit height, changes XY and orientation together, and
        then descends to the requested approach pose.  It lifts first only
        when the arm starts below that transit height.  This avoids both an
        unnecessary taught-pose detour and an unreachable downward pose at a
        high watcher position while retaining feedback checks.
        """

        target_position = _finite_vector(
            target_position_m,
            length=3,
            field="target_position_m",
        )
        target_quaternion = _normalise_quaternion(target_quaternion_xyzw)
        try:
            requested_transit_z = float(transit_z_m)
        except (TypeError, ValueError) as exc:
            raise CartesianJogSafetyViolation(
                "transit_z_m must be finite"
            ) from exc
        if not math.isfinite(requested_transit_z):
            raise CartesianJogSafetyViolation("transit_z_m must be finite")
        if not isinstance(use_configured_safe_transit, bool):
            raise CartesianJogSafetyViolation(
                "use_configured_safe_transit must be a boolean"
            )

        with self._operation(operation):
            self._require_available()
            if not isinstance(enable_token, str) or not enable_token:
                raise CartesianJogSafetyViolation(
                    "an explicit enable token is required"
                )
            if not self._dry_run and enable_token != self._enable_token:
                raise CartesianJogSafetyViolation("invalid enable token")
            if not area_clear or not estop_ready:
                raise CartesianJogSafetyViolation(
                    "area_clear and estop_ready must both be confirmed"
                )
            if self._teleop_is_running():
                raise CartesianJogConflict(
                    "stop teleoperation before entering the fixed orientation"
                )
            if self._workspace is None:
                raise CartesianJogUnavailable("motion workspace is not configured")
            effective_workspace = self._workspace
            if calibrated_workspace_profile is not None:
                if (
                    not isinstance(calibrated_workspace_profile, str)
                    or not calibrated_workspace_profile.strip()
                ):
                    raise CartesianJogSafetyViolation(
                        "calibrated workspace profile must be a non-empty string"
                    )
                profile_name = calibrated_workspace_profile.strip()
                calibrated_floor = (
                    self._calibrated_workspace_floor_z_m_by_profile.get(
                        profile_name
                    )
                )
                if calibrated_floor is None:
                    raise CartesianJogSafetyViolation(
                        f"unknown calibrated workspace profile: {profile_name}"
                    )
                effective_workspace = Workspace(
                    x_min=self._workspace.x_min,
                    x_max=self._workspace.x_max,
                    y_min=self._workspace.y_min,
                    y_max=self._workspace.y_max,
                    z_min=min(self._workspace.z_min, calibrated_floor),
                    z_max=self._workspace.z_max,
                    enforce_xy=self._workspace.enforce_xy,
                )
            if not effective_workspace.contains(target_position):
                raise CartesianJogSafetyViolation(
                    "fixed-orientation entry target is outside the configured workspace"
                )
            with self._state_lock:
                self._enabled = False
                self._locked_quaternion = None
                self._capture_position = None
                self._current_position = None

            with self._arm_session() as arm:
                self._require_idle_if_supported(arm)
                (
                    current_position,
                    current_quaternion,
                    orientation_spread_deg,
                    position_spread_m,
                ) = self._read_pose_burst(arm, self._verify_sample_count)
                self._require_idle_if_supported(arm)
                if not effective_workspace.contains(current_position):
                    raise CartesianJogSafetyViolation(
                        "current arm position is outside the configured motion workspace"
                    )
                if orientation_spread_deg > self._max_capture_spread_deg:
                    raise CartesianJogSafetyViolation(
                        "arm orientation is not stable before fixed-orientation entry"
                    )
                if position_spread_m > self._max_capture_position_spread_m:
                    raise CartesianJogSafetyViolation(
                        "arm position is not stable before fixed-orientation entry "
                        f"({position_spread_m * 1000.0:.3f} mm > "
                        f"{self._max_capture_position_spread_m * 1000.0:.3f} mm)"
                    )

                configured_safe_transit_z = (
                    self._safe_transit_z_m
                    if use_configured_safe_transit
                    and self._safe_transit_z_m is not None
                    else requested_transit_z
                )
                transit_z = max(
                    requested_transit_z,
                    target_position[2],
                    configured_safe_transit_z,
                )
                transit_start = [
                    current_position[0],
                    current_position[1],
                    transit_z,
                ]
                transit_target = [
                    target_position[0],
                    target_position[1],
                    transit_z,
                ]
                waypoints: list[list[list[float]]] = [
                    [list(current_position), list(current_quaternion)]
                ]
                if current_position[2] < transit_z - 1e-6:
                    self._append_linear_position_segment(
                        waypoints,
                        transit_start,
                        current_quaternion,
                    )
                transition_start_position = tuple(waypoints[-1][0])
                transition_distance = _position_error_m(
                    transition_start_position,
                    transit_target,
                )
                transition_steps = max(
                    self._safe_rotation_steps,
                    int(math.ceil(transition_distance / 0.005)),
                    1,
                )
                for index in range(1, transition_steps + 1):
                    fraction = index / transition_steps
                    waypoints.append(
                        [
                            [
                                start + fraction * (end - start)
                                for start, end in zip(
                                    transition_start_position,
                                    transit_target,
                                )
                            ],
                            list(
                                _slerp_quaternion(
                                    current_quaternion,
                                    target_quaternion,
                                    fraction,
                                )
                            ),
                        ]
                    )
                self._append_linear_position_segment(
                    waypoints,
                    target_position,
                    target_quaternion,
                )
                if not all(
                    effective_workspace.contains(waypoint[0])
                    for waypoint in waypoints
                ):
                    raise CartesianJogSafetyViolation(
                        "fixed-orientation entry path leaves the configured workspace"
                    )
                entry_targets: list[
                    tuple[tuple[float, ...], tuple[float, ...]]
                ] = []
                if current_position[2] < transit_z - 1e-6:
                    entry_targets.append(
                        (tuple(transit_start), tuple(current_quaternion))
                    )
                entry_targets.append(
                    (tuple(transit_target), tuple(target_quaternion))
                )
                if _position_error_m(transit_target, target_position) > 1e-6:
                    entry_targets.append(
                        (tuple(target_position), tuple(target_quaternion))
                    )

                result: dict[str, Any] = {
                    "operation": operation,
                    "dry_run": self._dry_run,
                    "executed": False,
                    "start_position_m": list(current_position),
                    "start_quaternion_xyzw": list(current_quaternion),
                    "target_position_m": list(target_position),
                    "target_quaternion_xyzw": list(target_quaternion),
                    "transit_z_m": transit_z,
                    "effective_minimum_z_m": effective_workspace.z_min,
                    "calibrated_workspace_profile": calibrated_workspace_profile,
                    "use_configured_safe_transit": use_configured_safe_transit,
                    "waypoint_count": len(waypoints),
                    "planner": "verified_planning_pos_segments",
                    "planned_target_count": len(entry_targets),
                }
                actual_position = current_position
                actual_quaternion = current_quaternion
                if not self._dry_run:
                    if self._teleop_is_running():
                        raise CartesianJogConflict(
                            "teleoperation started before fixed-orientation entry"
                        )
                    planning_mode, _waypoint_mode, slow_speed = (
                        self._motion_symbols()
                    )
                    verified_speed_parameters = self._prepare_for_motion(
                        arm,
                        mode=planning_mode,
                        expected_mode_name="PLANNING_POS",
                        slow_speed=slow_speed,
                    )
                    move_method = getattr(arm, "move_to_cart_pose", None)
                    if not callable(move_method):
                        raise CartesianJogUnavailable(
                            "arm lacks Cartesian pose planning"
                        )
                    completed_targets: list[list[float]] = []
                    sdk_reported_success_by_target: list[bool] = []
                    position_error = 0.0
                    orientation_error = _quaternion_error_deg(
                        current_quaternion,
                        target_quaternion,
                    )
                    for segment_position, segment_quaternion in entry_targets:
                        if self._teleop_is_running():
                            raise CartesianJogConflict(
                                "teleoperation started during fixed-orientation entry"
                            )
                        try:
                            moved = move_method(
                                [
                                    list(segment_position),
                                    list(segment_quaternion),
                                ],
                                blocking=True,
                            )
                        except TimeoutError as exc:
                            raise CartesianJogTimeout(
                                "fixed-orientation entry timed out"
                            ) from exc
                        except Exception as exc:
                            raise CartesianJogUnavailable(
                                "fixed-orientation entry failed: "
                                f"{type(exc).__name__}: {exc}"
                            ) from exc
                        (
                            actual_position,
                            actual_quaternion,
                            position_error,
                            orientation_error,
                        ) = self._wait_for_verified_target(
                            arm,
                            segment_position,
                            segment_quaternion,
                            timeout_s=self._restore_feedback_timeout_s,
                        )
                        self._require_idle_if_supported(arm)
                        sdk_reported_success_by_target.append(moved is True)
                        completed_targets.append(list(segment_position))
                    result.update(
                        {
                            "executed": True,
                            "completed_targets_m": completed_targets,
                            "sdk_reported_success_by_target": (
                                sdk_reported_success_by_target
                            ),
                            "actual_position_m": list(actual_position),
                            "actual_quaternion_xyzw": list(actual_quaternion),
                            "position_error_m": position_error,
                            "orientation_error_deg": orientation_error,
                            "feedback_stable_samples": (
                                self._feedback_stable_samples
                            ),
                            "verified_speed_parameters": (
                                verified_speed_parameters
                            ),
                        }
                    )

            with self._state_lock:
                if result.get("executed") is True:
                    self._locked_quaternion = target_quaternion
                    self._capture_position = tuple(actual_position)
                    self._current_position = tuple(actual_position)
                    self._enabled = True
                self._last_result = result
                return copy.deepcopy(result)

    def restore_safe_vertical(
        self,
        restore_token: str,
        *,
        area_clear: bool,
        estop_ready: bool,
        suction_released: bool,
    ) -> dict[str, Any]:
        """Return the left flange to one configured, bounded downward pose."""

        with self._operation("restore_safe_vertical"):
            self._require_available()
            if (
                self._safe_position is None
                or self._safe_quaternion is None
                or self._safe_transit_z_m is None
            ):
                raise CartesianJogUnavailable(
                    "safe vertical pose is not configured"
                )
            if restore_token != self._safe_restore_token:
                raise CartesianJogSafetyViolation(
                    "invalid safe vertical restore token"
                )
            if not area_clear or not estop_ready or not suction_released:
                raise CartesianJogSafetyViolation(
                    "area_clear, estop_ready and suction_released must all be confirmed"
                )
            if self._teleop_is_running():
                raise CartesianJogConflict(
                    "stop teleoperation before restoring the safe vertical pose"
                )
            with self._state_lock:
                # Recovery is independent from the previous jog session.  A
                # stale captured pose must never remain armed across this move.
                self._enabled = False
                self._locked_quaternion = None
                self._capture_position = None
                self._current_position = None

            def validate_start_and_measure(
                position: Sequence[float],
                quaternion: Sequence[float],
            ) -> tuple[float, float]:
                if not self._workspace or not self._workspace.contains(position):
                    raise CartesianJogSafetyViolation(
                        "current left-arm position is outside the configured "
                        "motion workspace"
                    )
                distance_m = _position_error_m(position, self._safe_position)
                orientation_delta_deg = _quaternion_error_deg(
                    quaternion,
                    self._safe_quaternion,
                )
                return distance_m, orientation_delta_deg

            with self._arm_session() as arm:
                self._require_idle_if_supported(arm)
                (
                    current_position,
                    current_quaternion,
                    orientation_spread_deg,
                    position_spread_m,
                ) = self._read_pose_burst(arm, self._verify_sample_count)
                self._require_idle_if_supported(arm)
                if orientation_spread_deg > self._max_capture_spread_deg:
                    raise CartesianJogSafetyViolation(
                        "left-arm orientation is not stable before restore"
                    )
                if position_spread_m > self._max_capture_position_spread_m:
                    raise CartesianJogSafetyViolation(
                        "left-arm position is not stable before restore"
                    )
                distance_m, orientation_delta_deg = validate_start_and_measure(
                    current_position,
                    current_quaternion,
                )
                waypoints = self._safe_vertical_waypoints(
                    current_position,
                    current_quaternion,
                )
                result: dict[str, Any] = {
                    "operation": "restore_safe_vertical",
                    "dry_run": self._dry_run,
                    "executed": False,
                    "start_position_m": list(current_position),
                    "start_quaternion_xyzw": list(current_quaternion),
                    "target_position_m": list(self._safe_position),
                    "target_quaternion_xyzw": list(self._safe_quaternion),
                    "transit_z_m": self._safe_transit_z_m,
                    "start_distance_m": distance_m,
                    "orientation_delta_deg": orientation_delta_deg,
                    "waypoint_count": len(waypoints),
                }
                actual_position = current_position
                actual_quaternion = current_quaternion
                if not self._dry_run:
                    if self._teleop_is_running():
                        raise CartesianJogConflict(
                            "teleoperation started before safe-pose restore"
                        )
                    planning_mode, waypoint_mode, slow_speed = (
                        self._motion_symbols()
                    )
                    waypoint_mode_entered = False
                    command_sent = False
                    target_verified = False
                    try:
                        self._require_idle_if_supported(arm)
                        self._switch_and_verify_mode(
                            arm,
                            waypoint_mode,
                            "PLANNING_WAYPOINTS_PATH",
                        )
                        waypoint_mode_entered = True
                        verified_speed_parameters = self._set_and_verify_named_speed(
                            arm,
                            "DEFAULT",
                            slow_speed,
                        )
                        if self._teleop_is_running():
                            raise CartesianJogConflict(
                                "teleoperation started during safe-pose setup"
                            )
                        self._require_idle_if_supported(arm)
                        command_position, command_quaternion = self._read_pose(arm)
                        distance_m, orientation_delta_deg = validate_start_and_measure(
                            command_position,
                            command_quaternion,
                        )
                        waypoints = self._safe_vertical_waypoints(
                            command_position,
                            command_quaternion,
                        )
                        move_method = getattr(
                            arm,
                            "move_with_cart_waypoints",
                            None,
                        )
                        if not callable(move_method):
                            raise CartesianJogUnavailable(
                                "left arm lacks Cartesian waypoint motion"
                            )
                        command_sent = True
                        try:
                            moved = move_method(waypoints, blocking=True)
                        except TimeoutError as exc:
                            raise CartesianJogTimeout(
                                "safe vertical restore timed out"
                            ) from exc
                        except Exception as exc:
                            raise CartesianJogUnavailable(
                                "safe vertical restore failed: "
                                f"{type(exc).__name__}: {exc}"
                            ) from exc
                        if moved is not True:
                            raise CartesianJogUnavailable(
                                "left arm rejected the safe vertical path"
                            )
                        (
                            actual_position,
                            actual_quaternion,
                            position_error,
                            orientation_error,
                        ) = self._wait_for_verified_target(
                            arm,
                            self._safe_position,
                            self._safe_quaternion,
                            timeout_s=self._restore_feedback_timeout_s,
                        )
                        target_verified = True
                        self._require_idle_if_supported(arm)
                        self._switch_and_verify_mode(
                            arm,
                            planning_mode,
                            "PLANNING_POS",
                        )
                        waypoint_mode_entered = False
                        result.update(
                            {
                                "executed": True,
                                "start_position_m": list(command_position),
                                "start_quaternion_xyzw": list(command_quaternion),
                                "start_distance_m": distance_m,
                                "orientation_delta_deg": orientation_delta_deg,
                                "waypoint_count": len(waypoints),
                                "actual_position_m": list(actual_position),
                                "actual_quaternion_xyzw": list(actual_quaternion),
                                "position_error_m": position_error,
                                "orientation_error_deg": orientation_error,
                                "feedback_stable_samples": (
                                    self._feedback_stable_samples
                                ),
                                "verified_speed_parameters": (
                                    verified_speed_parameters
                                ),
                                "restored_planning_pos": True,
                            }
                        )
                    finally:
                        # Before a command, restoring the previous planning
                        # mode is safe. After a command, do so only after the
                        # final target has been positively verified.
                        if waypoint_mode_entered and (
                            not command_sent or target_verified
                        ):
                            try:
                                state_method = getattr(arm, "get_state", None)
                                if (
                                    not callable(state_method)
                                    or _enum_name(state_method()) == "IDLE"
                                ):
                                    arm.switch_mode(planning_mode)
                            except Exception:
                                pass

            with self._state_lock:
                self._enabled = False
                if result.get("executed") is True:
                    self._locked_quaternion = self._safe_quaternion
                    self._capture_position = tuple(actual_position)
                    self._current_position = tuple(actual_position)
                self._last_result = result
                return copy.deepcopy(result)

    def reset_home(self, *, speed_profile: str = "DEFAULT") -> dict[str, Any]:
        """Move only the configured arm to its validated AIRBOT home pose."""

        with self._operation("reset_home"):
            if not self._feature_enabled:
                raise CartesianJogUnavailable(
                    f"{self._arm_compound} motion is disabled"
                )
            if self._home_joint_positions is None:
                raise CartesianJogUnavailable(
                    f"{self._arm_compound} home joint pose is not configured"
                )
            if self._teleop_is_running():
                raise CartesianJogConflict(
                    f"stop teleoperation before resetting the {self._arm_label}"
                )
            with self._state_lock:
                self._enabled = False
                self._locked_quaternion = None
                self._capture_position = None
                self._current_position = None

            target = self._home_joint_positions
            with self._arm_session() as arm:
                self._require_idle_if_supported(arm)
                start_joints = self._read_joint_positions(arm)
                result: dict[str, Any] = {
                    "operation": "reset_home",
                    "dry_run": self._dry_run,
                    "executed": False,
                    "arm": self._arm_name,
                    "start_joint_positions_rad": list(start_joints),
                    "target_joint_positions_rad": list(target),
                    "position_tolerance_rad": self._home_joint_tolerance_rad,
                }
                actual_joints = start_joints
                final_joint_error = max(
                    abs(actual - expected)
                    for actual, expected in zip(start_joints, target)
                )
                final_position: tuple[float, ...] | None = None
                final_quaternion: tuple[float, ...] | None = None
                if not self._dry_run:
                    if self._teleop_is_running():
                        raise CartesianJogConflict(
                            "teleoperation started before "
                            f"{self._arm_compound} reset"
                        )
                    planning_mode, _waypoint_mode, slow_speed = (
                        self._motion_symbols()
                    )
                    verified_speed_parameters = self._prepare_for_motion(
                        arm,
                        mode=planning_mode,
                        expected_mode_name="PLANNING_POS",
                        slow_speed=slow_speed,
                        speed_profile=speed_profile,
                    )
                    if self._teleop_is_running():
                        raise CartesianJogConflict(
                            "teleoperation started during "
                            f"{self._arm_compound} reset setup"
                        )
                    move_method = getattr(arm, "move_to_joint_pos", None)
                    if not callable(move_method):
                        raise CartesianJogUnavailable(
                            f"{self._arm_label} lacks joint-position planning"
                        )
                    try:
                        moved = move_method(list(target), blocking=True)
                    except TimeoutError as exc:
                        raise CartesianJogTimeout(
                            f"{self._arm_compound} home motion timed out"
                        ) from exc
                    except Exception as exc:
                        raise CartesianJogUnavailable(
                            f"{self._arm_compound} home motion failed: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    if moved is False:
                        raise CartesianJogUnavailable(
                            f"{self._arm_label} rejected the configured "
                            "home joint pose"
                        )
                    actual_joints, final_joint_error = (
                        self._wait_for_joint_target(arm, target)
                    )
                    final_position, final_quaternion = self._read_pose(arm)
                    result.update(
                        {
                            "executed": True,
                            "actual_joint_positions_rad": list(actual_joints),
                            "max_joint_error_rad": final_joint_error,
                            "actual_position_m": list(final_position),
                            "actual_quaternion_xyzw": list(final_quaternion),
                            "feedback_stable_samples": (
                                self._feedback_stable_samples
                            ),
                            "verified_speed_parameters": (
                                verified_speed_parameters
                            ),
                            "speed_profile": str(speed_profile).strip().upper(),
                        }
                    )
            with self._state_lock:
                if final_position is not None:
                    self._current_position = final_position
                self._last_result = result
                return copy.deepcopy(result)

    def move_to_saved_joint_pose(
        self,
        joint_positions_rad: Any,
        *,
        pose_name: str,
        speed_profile: str = "DEFAULT",
        gripper_position_m: float | None = None,
    ) -> dict[str, Any]:
        """Restore one operator-taught pose, including its optional gripper."""

        target = _finite_vector(
            joint_positions_rad,
            length=6,
            field="saved_pose.joint_positions_rad",
        )
        gripper_target = None
        if gripper_position_m is not None:
            gripper_target = _finite_vector(
                [gripper_position_m],
                length=1,
                field="saved_pose.gripper_position_m",
            )[0]
            if not 0.0 <= gripper_target <= 0.10:
                raise CartesianJogUnavailable(
                    "saved_pose.gripper_position_m must be between 0.0 and 0.1 m"
                )
        with self._operation("move_saved_pose"):
            if not self._feature_enabled:
                raise CartesianJogUnavailable(
                    f"{self._arm_compound} motion is disabled"
                )
            if self._teleop_is_running():
                raise CartesianJogConflict(
                    f"stop teleoperation before moving the {self._arm_label}"
                )
            with self._state_lock:
                self._enabled = False
                self._locked_quaternion = None
                self._capture_position = None
                self._current_position = None

            with self._arm_session() as arm:
                self._require_idle_if_supported(arm)
                start_joints = self._read_joint_positions(arm)
                result: dict[str, Any] = {
                    "operation": "move_saved_pose",
                    "pose_name": str(pose_name),
                    "dry_run": self._dry_run,
                    "executed": False,
                    "arm": self._arm_name,
                    "start_joint_positions_rad": list(start_joints),
                    "target_joint_positions_rad": list(target),
                    "target_gripper_position_m": gripper_target,
                    "position_tolerance_rad": self._home_joint_tolerance_rad,
                }
                final_position: tuple[float, ...] | None = None
                if not self._dry_run:
                    if self._teleop_is_running():
                        raise CartesianJogConflict(
                            "teleoperation started before saved-pose motion"
                        )
                    planning_mode, _waypoint_mode, slow_speed = self._motion_symbols()
                    verified_speed_parameters = self._prepare_for_motion(
                        arm,
                        mode=planning_mode,
                        expected_mode_name="PLANNING_POS",
                        slow_speed=slow_speed,
                        speed_profile=speed_profile,
                    )
                    if self._teleop_is_running():
                        raise CartesianJogConflict(
                            "teleoperation started during saved-pose setup"
                        )
                    move_method = getattr(arm, "move_to_joint_pos", None)
                    if not callable(move_method):
                        raise CartesianJogUnavailable(
                            f"{self._arm_label} lacks joint-position planning"
                        )
                    try:
                        moved = move_method(list(target), blocking=True)
                    except TimeoutError as exc:
                        raise CartesianJogTimeout(
                            f"{self._arm_compound} saved-pose motion timed out"
                        ) from exc
                    except Exception as exc:
                        raise CartesianJogUnavailable(
                            f"{self._arm_compound} saved-pose motion failed: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    if moved is False:
                        raise CartesianJogUnavailable(
                            f"{self._arm_label} rejected saved pose {pose_name}"
                        )
                    actual_joints, final_joint_error = self._wait_for_joint_target(
                        arm,
                        target,
                    )
                    actual_gripper_position = None
                    if gripper_target is not None:
                        gripper_method = getattr(arm, "move_eef_pos", None)
                        if not callable(gripper_method):
                            raise CartesianJogUnavailable(
                                f"{self._arm_label} lacks gripper-position planning"
                            )
                        try:
                            gripper_moved = gripper_method(
                                gripper_target,
                                blocking=True,
                            )
                        except TimeoutError as exc:
                            raise CartesianJogTimeout(
                                f"{self._arm_compound} gripper restore timed out"
                            ) from exc
                        except Exception as exc:
                            raise CartesianJogUnavailable(
                                f"{self._arm_compound} gripper restore failed: "
                                f"{type(exc).__name__}: {exc}"
                            ) from exc
                        if gripper_moved is False:
                            raise CartesianJogUnavailable(
                                f"{self._arm_label} rejected gripper target "
                                f"for saved pose {pose_name}"
                            )
                        actual_gripper_position = self._read_gripper_position(arm)
                    final_position, final_quaternion = self._read_pose(arm)
                    result.update(
                        {
                            "executed": True,
                            "actual_joint_positions_rad": list(actual_joints),
                            "max_joint_error_rad": final_joint_error,
                            "actual_position_m": list(final_position),
                            "actual_quaternion_xyzw": list(final_quaternion),
                            "actual_gripper_position_m": actual_gripper_position,
                            "verified_speed_parameters": verified_speed_parameters,
                            "speed_profile": str(speed_profile).strip().upper(),
                        }
                    )
            with self._state_lock:
                if final_position is not None:
                    self._current_position = final_position
                self._last_result = result
                return copy.deepcopy(result)

    def move_gripper_to_position(
        self,
        target_position_m: float,
        *,
        operation: str = "move_gripper_to_position",
        speed_profile: str = "DEFAULT",
    ) -> dict[str, Any]:
        """Move only this controller's gripper and return direct feedback."""

        target = _finite_vector(
            [target_position_m],
            length=1,
            field="gripper_target_position_m",
        )[0]
        if not 0.0 <= target <= 0.10:
            raise CartesianJogSafetyViolation(
                "gripper target position must be between 0.0 and 0.1 m"
            )
        with self._operation(operation):
            if not self._feature_enabled:
                raise CartesianJogUnavailable(
                    f"{self._arm_compound} motion is disabled"
                )
            if self._teleop_is_running():
                raise CartesianJogConflict(
                    f"stop teleoperation before moving the {self._arm_label} gripper"
                )
            with self._state_lock:
                self._enabled = False
                self._locked_quaternion = None
                self._capture_position = None
                self._current_position = None
            with self._arm_session() as arm:
                self._require_idle_if_supported(arm)
                initial = self._read_gripper_position(arm)
                result: dict[str, Any] = {
                    "operation": operation,
                    "dry_run": self._dry_run,
                    "executed": False,
                    "arm": self._arm_name,
                    "initial_gripper_position_m": initial,
                    "target_gripper_position_m": target,
                }
                if not self._dry_run:
                    planning_mode, _waypoint_mode, slow_speed = self._motion_symbols()
                    verified_speed_parameters = self._prepare_for_motion(
                        arm,
                        mode=planning_mode,
                        expected_mode_name="PLANNING_POS",
                        slow_speed=slow_speed,
                        speed_profile=speed_profile,
                    )
                    move_method = getattr(arm, "move_eef_pos", None)
                    if not callable(move_method):
                        raise CartesianJogUnavailable(
                            f"{self._arm_label} lacks gripper-position planning"
                        )
                    try:
                        moved = move_method(target, blocking=True)
                    except TimeoutError as exc:
                        raise CartesianJogTimeout(
                            f"{self._arm_compound} gripper motion timed out"
                        ) from exc
                    except Exception as exc:
                        raise CartesianJogUnavailable(
                            f"{self._arm_compound} gripper motion failed: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    if moved is False:
                        raise CartesianJogUnavailable(
                            f"{self._arm_label} rejected the gripper target"
                        )
                    actual = self._read_gripper_position(arm)
                    result.update(
                        {
                            "executed": True,
                            "actual_gripper_position_m": actual,
                            "gripper_error_m": abs(actual - target),
                            "verified_speed_parameters": verified_speed_parameters,
                            "speed_profile": str(speed_profile).strip().upper(),
                        }
                    )
            with self._state_lock:
                self._last_result = result
                return copy.deepcopy(result)
