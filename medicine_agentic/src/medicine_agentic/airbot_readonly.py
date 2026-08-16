from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np


ARM_PORTS = {"left": 50051, "right": 50053}
ARM_NAMES = ("left", "right")


class ArmReadError(RuntimeError):
    pass


def enum_name(value: Any) -> str:
    return getattr(value, "name", str(value).split(".")[-1])


def normalize_end_pose(value: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        position = np.asarray(value[0], dtype=np.float64)
        quaternion = np.asarray(value[1], dtype=np.float64)
    else:
        flat = np.asarray(value, dtype=np.float64).reshape(-1)
        if flat.shape != (7,):
            raise ArmReadError(
                "get_end_pose() must return ([x,y,z], [qx,qy,qz,qw]) or 7 values"
            )
        position, quaternion = flat[:3], flat[3:]
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ArmReadError(
            f"invalid end pose shapes: position={position.shape}, quaternion={quaternion.shape}"
        )
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
        raise ArmReadError("end pose contains non-finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-9:
        raise ArmReadError("end-pose quaternion has zero norm")
    return position, quaternion / norm


def _median_quaternion(quaternions: np.ndarray) -> np.ndarray:
    aligned = quaternions.copy()
    reference = aligned[0]
    for index in range(len(aligned)):
        if float(np.dot(aligned[index], reference)) < 0.0:
            aligned[index] = -aligned[index]
    median = np.median(aligned, axis=0)
    return median / np.linalg.norm(median)


def _quaternion_distance_deg(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.clip(abs(np.dot(left, right)), -1.0, 1.0))
    return math.degrees(2.0 * math.acos(cosine))


def summarize_arm_samples(
    samples: list[dict[str, Any]],
    *,
    max_joint_ptp_rad: float = 0.003,
    max_position_ptp_m: float = 0.001,
    max_orientation_spread_deg: float = 0.2,
) -> dict[str, Any]:
    if len(samples) < 3:
        raise ArmReadError("at least 3 arm samples are required")
    joints = np.asarray([sample["joint_position_rad"] for sample in samples], dtype=np.float64)
    positions = np.asarray(
        [sample["flange_position_m"] for sample in samples], dtype=np.float64
    )
    quaternions = np.asarray(
        [sample["flange_quaternion_xyzw"] for sample in samples], dtype=np.float64
    )
    eef_values = [sample.get("eef_feedback_m", []) for sample in samples]
    if joints.shape[1:] != (6,) or positions.shape[1:] != (3,) or quaternions.shape[1:] != (4,):
        raise ArmReadError("arm sample shapes are invalid")
    median_quaternion = _median_quaternion(quaternions)
    orientation_spread = max(
        _quaternion_distance_deg(value, median_quaternion) for value in quaternions
    )
    joint_ptp = np.ptp(joints, axis=0)
    position_ptp = np.ptp(positions, axis=0)
    eef_array = np.asarray(eef_values, dtype=np.float64)
    if eef_array.ndim == 2 and eef_array.shape[0] == len(samples):
        eef_median = np.median(eef_array, axis=0).tolist()
    else:
        eef_median = []
    stable = bool(
        float(np.max(joint_ptp)) <= max_joint_ptp_rad
        and float(np.max(position_ptp)) <= max_position_ptp_m
        and orientation_spread <= max_orientation_spread_deg
    )
    return {
        "joint_position_rad": np.median(joints, axis=0).tolist(),
        "flange_pose_in_base": {
            "position_m": np.median(positions, axis=0).tolist(),
            "quaternion_xyzw": median_quaternion.tolist(),
        },
        "eef_feedback_m": eef_median,
        "driver_state": str(samples[-1]["driver_state"]),
        "control_mode": str(samples[-1]["control_mode"]),
        "capture_metrics": {
            "sample_count": len(samples),
            "joint_peak_to_peak_rad": joint_ptp.tolist(),
            "max_joint_peak_to_peak_rad": float(np.max(joint_ptp)),
            "position_peak_to_peak_m": position_ptp.tolist(),
            "max_position_peak_to_peak_m": float(np.max(position_ptp)),
            "orientation_spread_deg": float(orientation_spread),
            "stable": stable,
            "thresholds": {
                "max_joint_peak_to_peak_rad": max_joint_ptp_rad,
                "max_position_peak_to_peak_m": max_position_ptp_m,
                "max_orientation_spread_deg": max_orientation_spread_deg,
            },
        },
    }


@dataclass
class ArmConnection:
    name: str
    port: int
    arm: Any


class AirbotReadOnly:
    """Read AIRBOT feedback without switching mode or issuing motion."""

    def __init__(
        self,
        host: str = "localhost",
        ports: dict[str, int] | None = None,
        arm_names: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        self.host = host
        self.ports = dict(ports or ARM_PORTS)
        requested = tuple(arm_names or ARM_NAMES)
        if (
            not requested
            or len(set(requested)) != len(requested)
            or any(name not in ARM_NAMES for name in requested)
        ):
            raise ValueError("arm_names must contain unique values from left/right")
        missing_ports = [name for name in requested if name not in self.ports]
        if missing_ports:
            raise ValueError(
                "missing AIRBOT port(s): " + ", ".join(missing_ports)
            )
        self.arm_names = requested
        self.connections: dict[str, ArmConnection] = {}
        self._capture_pool: ThreadPoolExecutor | None = None

    def connect(self) -> None:
        try:
            from airbot_py.arm import AIRBOTPlay
        except ImportError as exc:  # pragma: no cover - only available on dosw1
            raise ArmReadError(
                "airbot_py is unavailable; run this command inside the dos-w1 Conda environment"
            ) from exc
        failures: list[str] = []
        for name in self.arm_names:
            port = int(self.ports[name])
            arm = AIRBOTPlay(url=self.host, port=port)
            try:
                connected = bool(arm.connect())
            except Exception as exc:
                connected = False
                failures.append(f"{name}:{port}: {type(exc).__name__}: {exc}")
            if not connected:
                failures.append(f"{name}:{port}: connection failed")
                for existing in self.connections.values():
                    self._disconnect(existing.arm)
                self.connections.clear()
                raise ArmReadError("; ".join(dict.fromkeys(failures)))
            self.connections[name] = ArmConnection(name=name, port=port, arm=arm)
        self._capture_pool = ThreadPoolExecutor(
            max_workers=len(self.arm_names),
            thread_name_prefix=f"airbot-feedback-{self.host}",
        )

    def close(self) -> None:
        if self._capture_pool is not None:
            self._capture_pool.shutdown(wait=True, cancel_futures=True)
            self._capture_pool = None
        for connection in self.connections.values():
            self._disconnect(connection.arm)
        self.connections.clear()

    def __enter__(self) -> "AirbotReadOnly":
        self.connect()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

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

    def capture_pair(self) -> dict[str, Any]:
        if set(self.connections) != {"left", "right"}:
            raise ArmReadError("both arms must be connected before capture")
        return self.capture_selected()

    def capture_selected(self) -> dict[str, Any]:
        """Read selected endpoints concurrently without changing control state."""

        if set(self.connections) != set(self.arm_names):
            raise ArmReadError("selected arms must be connected before capture")
        if self._capture_pool is None:
            raise ArmReadError("feedback capture pool is unavailable")
        pair_started_ns = time.time_ns()
        arms: dict[str, dict[str, Any]] = {}
        timestamps: dict[str, int] = {}
        futures = {
            name: self._capture_pool.submit(
                self._capture_one,
                name,
                self.connections[name].arm,
            )
            for name in self.arm_names
        }
        for name in self.arm_names:
            arm_sample = futures[name].result()
            timestamp_ns = int(arm_sample["timestamp_ns"])
            timestamps[name] = timestamp_ns
            arms[name] = arm_sample
        pair_completed_ns = time.time_ns()
        skew_ms = (
            0.0
            if len(timestamps) < 2
            else (max(timestamps.values()) - min(timestamps.values())) / 1e6
        )
        return {
            "timestamp_ns": (pair_started_ns + pair_completed_ns) // 2,
            "capture_duration_ms": (pair_completed_ns - pair_started_ns) / 1e6,
            "paired_sample_skew_ms": skew_ms,
            "arms": arms,
        }

    def capture_selected_fast(self) -> dict[str, Any]:
        """ACT fast path: concurrently read only joints and EEF feedback."""

        if set(self.connections) != set(self.arm_names):
            raise ArmReadError("selected arms must be connected before capture")
        if self._capture_pool is None:
            raise ArmReadError("feedback capture pool is unavailable")
        pair_started_ns = time.time_ns()
        futures = {
            name: self._capture_pool.submit(
                self._capture_one_fast,
                name,
                self.connections[name].arm,
            )
            for name in self.arm_names
        }
        arms = {name: futures[name].result() for name in self.arm_names}
        pair_completed_ns = time.time_ns()
        timestamps = [int(sample["timestamp_ns"]) for sample in arms.values()]
        return {
            "timestamp_ns": (pair_started_ns + pair_completed_ns) // 2,
            "capture_duration_ms": (pair_completed_ns - pair_started_ns) / 1e6,
            "paired_sample_skew_ms": (
                0.0 if len(timestamps) < 2 else (max(timestamps) - min(timestamps)) / 1e6
            ),
            "arms": arms,
        }

    @staticmethod
    def _capture_one_fast(name: str, arm: Any) -> dict[str, Any]:
        started_ns = time.time_ns()
        try:
            joints = np.asarray(arm.get_joint_pos(), dtype=np.float64)
            eef = np.asarray(arm.get_eef_pos(), dtype=np.float64).reshape(-1)
        except Exception as exc:
            raise ArmReadError(
                f"failed to read {name} ACT feedback: {type(exc).__name__}: {exc}"
            ) from exc
        completed_ns = time.time_ns()
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise ArmReadError(f"{name} arm returned invalid joints: {joints}")
        if not np.all(np.isfinite(eef)):
            raise ArmReadError(f"{name} arm returned invalid EEF feedback: {eef}")
        return {
            "timestamp_ns": (started_ns + completed_ns) // 2,
            "read_duration_ms": (completed_ns - started_ns) / 1e6,
            "joint_position_rad": joints.tolist(),
            "flange_position_m": [],
            "flange_quaternion_xyzw": [],
            "eef_feedback_m": eef.tolist(),
            "driver_state": None,
            "control_mode": None,
        }

    @staticmethod
    def _capture_one(name: str, arm: Any) -> dict[str, Any]:
        started_ns = time.time_ns()
        try:
            joints = np.asarray(arm.get_joint_pos(), dtype=np.float64)
            position, quaternion = normalize_end_pose(arm.get_end_pose())
            eef = np.asarray(arm.get_eef_pos(), dtype=np.float64).reshape(-1)
            state = enum_name(arm.get_state())
            mode = enum_name(arm.get_control_mode())
        except Exception as exc:
            raise ArmReadError(
                f"failed to read {name} arm: {type(exc).__name__}: {exc}"
            ) from exc
        completed_ns = time.time_ns()
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise ArmReadError(f"{name} arm returned invalid joints: {joints}")
        if not np.all(np.isfinite(eef)):
            raise ArmReadError(f"{name} arm returned invalid EEF feedback: {eef}")
        timestamp_ns = (started_ns + completed_ns) // 2
        return {
            "timestamp_ns": timestamp_ns,
            "read_duration_ms": (completed_ns - started_ns) / 1e6,
            "joint_position_rad": joints.tolist(),
            "flange_position_m": position.tolist(),
            "flange_quaternion_xyzw": quaternion.tolist(),
            "eef_feedback_m": eef.tolist(),
            "driver_state": state,
            "control_mode": mode,
        }

    def collect_pairs(
        self,
        *,
        count: int = 20,
        interval_s: float = 0.05,
    ) -> list[dict[str, Any]]:
        if count < 3:
            raise ValueError("sample count must be at least 3")
        if interval_s < 0.0:
            raise ValueError("sample interval must be non-negative")
        pairs = []
        for index in range(count):
            pairs.append(self.capture_pair())
            if index + 1 < count and interval_s > 0.0:
                time.sleep(interval_s)
        return pairs
