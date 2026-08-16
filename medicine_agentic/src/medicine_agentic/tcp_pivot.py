from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from medicine_agentic.pose_store import utc_now

try:
    import fcntl
except ImportError:  # pragma: no cover - dosw1 and development macOS provide it
    fcntl = None


SAMPLE_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
CALIBRATION_TYPE = "pivot_tcp_translation"


class PivotCalibrationError(RuntimeError):
    pass


def _payload_without_hash(payload: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    result.pop("content_sha256", None)
    return result


def content_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        _payload_without_hash(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def quaternion_xyzw_to_matrix(value: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise PivotCalibrationError("quaternion_xyzw must contain 4 finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise PivotCalibrationError("quaternion_xyzw has zero norm")
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quaternion_distance_deg(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.clip(abs(np.dot(left, right)), -1.0, 1.0))
    return math.degrees(2.0 * math.acos(cosine))


def _median_quaternion(quaternions: np.ndarray) -> np.ndarray:
    aligned = quaternions.copy()
    reference = aligned[0]
    for index in range(len(aligned)):
        if float(np.dot(aligned[index], reference)) < 0.0:
            aligned[index] = -aligned[index]
    median = np.median(aligned, axis=0)
    norm = float(np.linalg.norm(median))
    if norm < 1e-12:
        raise PivotCalibrationError("captured quaternions have no stable median")
    result = median / norm
    if result[3] < 0.0:
        result = -result
    return result


def summarize_flange_samples(
    samples: Sequence[dict[str, Any]],
    *,
    max_joint_ptp_rad: float = 0.003,
    max_position_ptp_m: float = 0.001,
    max_orientation_spread_deg: float = 0.2,
) -> dict[str, Any]:
    """Collapse a short, stationary read-only burst into one flange pose."""

    if len(samples) < 3:
        raise PivotCalibrationError("at least 3 feedback samples are required")
    joints = np.asarray(
        [sample["joint_position_rad"] for sample in samples],
        dtype=np.float64,
    )
    positions = np.asarray(
        [sample["flange_position_m"] for sample in samples],
        dtype=np.float64,
    )
    quaternions = np.asarray(
        [sample["flange_quaternion_xyzw"] for sample in samples],
        dtype=np.float64,
    )
    if joints.shape != (len(samples), 6) or not np.all(np.isfinite(joints)):
        raise PivotCalibrationError(f"invalid captured joint array: {joints.shape}")
    if positions.shape != (len(samples), 3) or not np.all(np.isfinite(positions)):
        raise PivotCalibrationError(f"invalid captured position array: {positions.shape}")
    if quaternions.shape != (len(samples), 4) or not np.all(np.isfinite(quaternions)):
        raise PivotCalibrationError(f"invalid captured quaternion array: {quaternions.shape}")
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms < 1e-12):
        raise PivotCalibrationError("captured quaternion has zero norm")
    quaternions = quaternions / norms[:, np.newaxis]
    median_quaternion = _median_quaternion(quaternions)
    joint_ptp = np.ptp(joints, axis=0)
    position_ptp = np.ptp(positions, axis=0)
    orientation_spread_deg = max(
        _quaternion_distance_deg(value, median_quaternion)
        for value in quaternions
    )
    stable = bool(
        float(np.max(joint_ptp)) <= max_joint_ptp_rad
        and float(np.max(position_ptp)) <= max_position_ptp_m
        and orientation_spread_deg <= max_orientation_spread_deg
    )
    return {
        "joint_position_rad": np.median(joints, axis=0).tolist(),
        "flange_pose_in_base": {
            "position_m": np.median(positions, axis=0).tolist(),
            "quaternion_xyzw": median_quaternion.tolist(),
        },
        "driver_state": str(samples[-1].get("driver_state", "UNKNOWN")),
        "control_mode": str(samples[-1].get("control_mode", "UNKNOWN")),
        "capture_metrics": {
            "feedback_sample_count": len(samples),
            "max_joint_peak_to_peak_rad": float(np.max(joint_ptp)),
            "max_position_peak_to_peak_m": float(np.max(position_ptp)),
            "orientation_spread_deg": float(orientation_spread_deg),
            "stable": stable,
            "thresholds": {
                "max_joint_peak_to_peak_rad": float(max_joint_ptp_rad),
                "max_position_peak_to_peak_m": float(max_position_ptp_m),
                "max_orientation_spread_deg": float(max_orientation_spread_deg),
            },
        },
    }


def _tcp_frame_for_arm(arm: str) -> str:
    if arm == "left":
        return "left_suction_tcp"
    if arm == "right":
        return "right_gripper_tcp"
    raise ValueError("arm must be 'left' or 'right'")


def new_sample_document(
    *,
    robot_id: str = "dosw1",
    arm: str = "left",
    tcp_frame: str | None = None,
) -> dict[str, Any]:
    if arm not in {"left", "right"}:
        raise ValueError("arm must be 'left' or 'right'")
    payload: dict[str, Any] = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "calibration_type": CALIBRATION_TYPE,
        "robot_id": robot_id,
        "arm": arm,
        "base_frame": f"{arm}_base/base_link",
        "flange_frame": f"{arm}_base/end_link",
        "tcp_frame": tcp_frame or _tcp_frame_for_arm(arm),
        "revision": 0,
        "updated_at": utc_now(),
        "units": {
            "joint_position": "rad",
            "position": "m",
            "quaternion": "xyzw",
        },
        "samples": [],
    }
    payload["content_sha256"] = content_sha256(payload)
    return payload


def _finite_vector(value: Any, length: int, field: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PivotCalibrationError(f"{field} must be numeric") from exc
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise PivotCalibrationError(f"{field} must contain {length} finite values")
    return vector


def validate_pivot_sample(sample: Any) -> list[str]:
    if not isinstance(sample, dict):
        return ["sample must be an object"]
    errors: list[str] = []
    if not isinstance(sample.get("sample_id"), str) or not sample.get("sample_id"):
        errors.append("sample_id must be a non-empty string")
    try:
        _finite_vector(sample.get("joint_position_rad"), 6, "joint_position_rad")
    except PivotCalibrationError as exc:
        errors.append(str(exc))
    flange = sample.get("flange_pose_in_base")
    if not isinstance(flange, dict):
        errors.append("flange_pose_in_base must be an object")
        return errors
    try:
        _finite_vector(flange.get("position_m"), 3, "flange position_m")
    except PivotCalibrationError as exc:
        errors.append(str(exc))
    try:
        quaternion = _finite_vector(
            flange.get("quaternion_xyzw"),
            4,
            "flange quaternion_xyzw",
        )
        if abs(float(np.linalg.norm(quaternion)) - 1.0) > 1e-4:
            errors.append("flange quaternion_xyzw must be normalized")
    except PivotCalibrationError as exc:
        errors.append(str(exc))
    metrics = sample.get("capture_metrics")
    if not isinstance(metrics, dict) or metrics.get("stable") is not True:
        errors.append("capture_metrics.stable must be true")
    return errors


def validate_sample_document(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return ["sample document root must be an object"]
    errors: list[str] = []
    if payload.get("schema_version") != SAMPLE_SCHEMA_VERSION:
        errors.append(f"schema_version must be {SAMPLE_SCHEMA_VERSION}")
    if payload.get("calibration_type") != CALIBRATION_TYPE:
        errors.append(f"calibration_type must be {CALIBRATION_TYPE!r}")
    arm = payload.get("arm")
    if arm not in {"left", "right"}:
        errors.append("arm must be 'left' or 'right'")
    elif payload.get("base_frame") != f"{arm}_base/base_link":
        errors.append("base_frame does not match arm")
    elif payload.get("flange_frame") != f"{arm}_base/end_link":
        errors.append("flange_frame does not match arm")
    if not isinstance(payload.get("tcp_frame"), str) or not payload.get("tcp_frame"):
        errors.append("tcp_frame must be a non-empty string")
    if not isinstance(payload.get("revision"), int) or payload.get("revision", -1) < 0:
        errors.append("revision must be a non-negative integer")
    samples = payload.get("samples")
    if not isinstance(samples, list):
        errors.append("samples must be a list")
        return errors
    identifiers: set[str] = set()
    for index, sample in enumerate(samples):
        sample_errors = validate_pivot_sample(sample)
        errors.extend(f"samples[{index}]: {error}" for error in sample_errors)
        if isinstance(sample, dict) and isinstance(sample.get("sample_id"), str):
            identifier = sample["sample_id"]
            if identifier in identifiers:
                errors.append(f"duplicate sample_id: {identifier}")
            identifiers.add(identifier)
    expected_hash = payload.get("content_sha256")
    if expected_hash is not None and expected_hash != content_sha256(payload):
        errors.append("content_sha256 does not match the sample document")
    return errors


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


class PivotSampleStore:
    """Append-only, lock-protected store for manually posed flange samples."""

    def __init__(
        self,
        path: str | Path,
        *,
        robot_id: str = "dosw1",
        arm: str = "left",
        tcp_frame: str | None = None,
    ) -> None:
        self.path = Path(path).expanduser()
        self.robot_id = robot_id
        self.arm = arm
        self.tcp_frame = tcp_frame or _tcp_frame_for_arm(arm)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return new_sample_document(
                robot_id=self.robot_id,
                arm=self.arm,
                tcp_frame=self.tcp_frame,
            )
        with self.path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        errors = validate_sample_document(payload)
        if errors:
            raise PivotCalibrationError(
                f"invalid pivot sample store {self.path}: " + "; ".join(errors)
            )
        if payload.get("robot_id") != self.robot_id:
            raise PivotCalibrationError(
                f"sample store robot_id={payload.get('robot_id')!r}, "
                f"expected {self.robot_id!r}"
            )
        if payload.get("arm") != self.arm:
            raise PivotCalibrationError(
                f"sample store arm={payload.get('arm')!r}, expected {self.arm!r}"
            )
        return payload

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def append(self, sample: dict[str, Any], *, label: str = "") -> dict[str, Any]:
        with self._lock():
            current = self.load()
            candidate = copy.deepcopy(current)
            sequence = len(candidate["samples"]) + 1
            record = copy.deepcopy(sample)
            record["sample_id"] = f"sample_{sequence:04d}"
            record["label"] = str(label)
            record["captured_at"] = utc_now()
            errors = validate_pivot_sample(record)
            if errors:
                raise PivotCalibrationError(
                    "refusing to append invalid pivot sample: " + "; ".join(errors)
                )
            candidate["samples"].append(record)
            candidate["revision"] = int(current["revision"]) + 1
            candidate["updated_at"] = utc_now()
            candidate["content_sha256"] = content_sha256(candidate)
            errors = validate_sample_document(candidate)
            if errors:
                raise PivotCalibrationError(
                    "refusing to save invalid pivot sample store: " + "; ".join(errors)
                )
            _atomic_write_json(self.path, candidate)
            reloaded = self.load()
            if reloaded["content_sha256"] != candidate["content_sha256"]:
                raise PivotCalibrationError("pivot sample store read-back hash mismatch")
            return record


def _orientation_span_deg(rotations: np.ndarray) -> float:
    maximum = 0.0
    for left_index in range(len(rotations)):
        for right_index in range(left_index + 1, len(rotations)):
            relative = rotations[left_index].T @ rotations[right_index]
            cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
            maximum = max(maximum, math.degrees(math.acos(cosine)))
    return maximum


def solve_pivot_translation(
    samples: Sequence[dict[str, Any]],
    *,
    robot_id: str = "dosw1",
    arm: str = "left",
    tcp_frame: str | None = None,
    source_samples_sha256: str | None = None,
    min_samples: int = 8,
    max_rms_residual_m: float = 0.002,
    max_residual_m: float = 0.005,
    max_condition_number: float = 100.0,
    min_normalized_singular_value: float = 0.10,
    min_orientation_span_deg: float = 25.0,
    max_tcp_offset_m: float = 0.30,
) -> dict[str, Any]:
    """Solve ``R_i p_flange + t_i = p_pivot`` using centered least squares."""

    if not samples:
        raise PivotCalibrationError("pivot solve requires at least one sample")
    if arm not in {"left", "right"}:
        raise ValueError("arm must be 'left' or 'right'")
    resolved_tcp_frame = tcp_frame or _tcp_frame_for_arm(arm)
    if min_samples < 4:
        raise ValueError("min_samples must be at least 4")
    for name, value in (
        ("max_rms_residual_m", max_rms_residual_m),
        ("max_residual_m", max_residual_m),
        ("max_condition_number", max_condition_number),
        ("min_normalized_singular_value", min_normalized_singular_value),
        ("min_orientation_span_deg", min_orientation_span_deg),
        ("max_tcp_offset_m", max_tcp_offset_m),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")

    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    for index, sample in enumerate(samples):
        errors = validate_pivot_sample(sample)
        if errors:
            raise PivotCalibrationError(
                f"invalid pivot sample {index}: " + "; ".join(errors)
            )
        flange = sample["flange_pose_in_base"]
        rotations.append(
            quaternion_xyzw_to_matrix(flange["quaternion_xyzw"])
        )
        translations.append(
            _finite_vector(flange["position_m"], 3, "flange position_m")
        )
    rotation_array = np.stack(rotations)
    translation_array = np.stack(translations)

    centered_rotations = rotation_array - np.mean(rotation_array, axis=0)
    centered_translations = translation_array - np.mean(translation_array, axis=0)
    matrix = centered_rotations.reshape(-1, 3)
    target = -centered_translations.reshape(-1)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank_threshold = max(
        float(singular_values[0]) * np.finfo(np.float64).eps * max(matrix.shape),
        1e-12,
    )
    rank = int(np.sum(singular_values > rank_threshold))
    translation_flange_to_tcp, *_ = np.linalg.lstsq(matrix, target, rcond=None)

    tcp_points = np.einsum(
        "nij,j->ni",
        rotation_array,
        translation_flange_to_tcp,
    ) + translation_array
    pivot_point = np.mean(tcp_points, axis=0)
    residual_vectors = tcp_points - pivot_point
    residual_norms = np.linalg.norm(residual_vectors, axis=1)
    rms_residual_m = float(np.sqrt(np.mean(np.square(residual_norms))))
    maximum_residual_m = float(np.max(residual_norms))
    median_residual_m = float(np.median(residual_norms))
    smallest_singular_value = float(singular_values[-1])
    largest_singular_value = float(singular_values[0])
    condition_number = (
        largest_singular_value / smallest_singular_value
        if smallest_singular_value > rank_threshold
        else math.inf
    )
    normalized_smallest_singular_value = smallest_singular_value / math.sqrt(
        len(samples)
    )
    orientation_span_deg = _orientation_span_deg(rotation_array)
    tcp_offset_m = float(np.linalg.norm(translation_flange_to_tcp))

    failures: list[str] = []
    if len(samples) < min_samples:
        failures.append(f"sample_count {len(samples)} is below {min_samples}")
    if rank < 3:
        failures.append(f"rotation excitation rank is {rank}, expected 3")
    if not math.isfinite(condition_number) or condition_number > max_condition_number:
        failures.append(
            f"condition_number {condition_number:.6g} exceeds {max_condition_number:.6g}"
        )
    if normalized_smallest_singular_value < min_normalized_singular_value:
        failures.append(
            "normalized_smallest_singular_value "
            f"{normalized_smallest_singular_value:.6g} is below "
            f"{min_normalized_singular_value:.6g}"
        )
    if orientation_span_deg < min_orientation_span_deg:
        failures.append(
            f"orientation_span_deg {orientation_span_deg:.3f} is below "
            f"{min_orientation_span_deg:.3f}"
        )
    if rms_residual_m > max_rms_residual_m:
        failures.append(
            f"rms_residual_m {rms_residual_m:.6f} exceeds "
            f"{max_rms_residual_m:.6f}"
        )
    if maximum_residual_m > max_residual_m:
        failures.append(
            f"max_residual_m {maximum_residual_m:.6f} exceeds "
            f"{max_residual_m:.6f}"
        )
    if tcp_offset_m > max_tcp_offset_m:
        failures.append(
            f"tcp_offset_m {tcp_offset_m:.6f} exceeds {max_tcp_offset_m:.6f}"
        )

    thresholds = {
        "min_samples": int(min_samples),
        "max_rms_residual_m": float(max_rms_residual_m),
        "max_residual_m": float(max_residual_m),
        "max_condition_number": float(max_condition_number),
        "min_normalized_singular_value": float(min_normalized_singular_value),
        "min_orientation_span_deg": float(min_orientation_span_deg),
        "max_tcp_offset_m": float(max_tcp_offset_m),
    }
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "calibration_type": CALIBRATION_TYPE,
        "robot_id": robot_id,
        "arm": arm,
        "created_at": utc_now(),
        "source_samples_sha256": source_samples_sha256,
        "frames": {
            "base": f"{arm}_base/base_link",
            "flange": f"{arm}_base/end_link",
            "tcp": resolved_tcp_frame,
        },
        "units": {
            "translation": "m",
            "residual": "m",
            "orientation": "deg",
        },
        # Convenience fields are intentionally duplicated at the top level so
        # generic P0 gates do not need to understand the solver's diagnostics
        # layout.  The content hash protects both representations.
        "calibrated": not failures,
        "usable_for_motion": not failures,
        "translation_m": translation_flange_to_tcp.tolist(),
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "rotation_source": "mechanical_mount_assumed_aligned",
        "flange_to_tcp": {
            "translation_m": translation_flange_to_tcp.tolist(),
            "rotation_calibrated": False,
            "note": "Pivot calibration estimates translation only.",
        },
        "pivot_point_in_base_m": pivot_point.tolist(),
        "sample_count": len(samples),
        "metrics": {
            "rms_residual_m": rms_residual_m,
            "median_residual_m": median_residual_m,
            "max_residual_m": maximum_residual_m,
            # JSON has no portable representation for infinity.  Degenerate
            # capture sets this to null and is already rejected by acceptance.
            "condition_number": (
                float(condition_number)
                if math.isfinite(condition_number)
                else None
            ),
            "matrix_rank": rank,
            "singular_values": singular_values.tolist(),
            "normalized_smallest_singular_value": (
                float(normalized_smallest_singular_value)
            ),
            "orientation_span_deg": float(orientation_span_deg),
            "tcp_offset_m": tcp_offset_m,
            "residual_norms_m": residual_norms.tolist(),
        },
        "fit": {
            "rms_residual_mm": rms_residual_m * 1000.0,
            "max_residual_mm": maximum_residual_m * 1000.0,
        },
        "acceptance": {
            "accepted": not failures,
            "failures": failures,
            "thresholds": thresholds,
        },
    }
    result["content_sha256"] = content_sha256(result)
    return result


def write_accepted_result(
    path: str | Path,
    result: dict[str, Any],
    *,
    replace: bool = False,
    expected_current_sha256: str | None = None,
) -> dict[str, Any]:
    target = Path(path).expanduser()
    if result.get("acceptance", {}).get("accepted") is not True:
        raise PivotCalibrationError("refusing to write a pivot result that failed acceptance")
    if result.get("content_sha256") != content_sha256(result):
        raise PivotCalibrationError("pivot result content_sha256 is invalid")
    lock_path = target.with_suffix(target.suffix + ".lock")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        if target.exists():
            with target.open("r", encoding="utf-8") as stream:
                current = json.load(stream)
            current_hash = current.get("content_sha256")
            if not replace:
                raise PivotCalibrationError(
                    f"result already exists: {target}; use --replace with "
                    f"--expected-output-sha256 {current_hash}"
                )
            if expected_current_sha256 is None or expected_current_sha256 != current_hash:
                raise PivotCalibrationError(
                    "existing result hash mismatch: "
                    f"expected {expected_current_sha256!r}, current {current_hash!r}"
                )
        _atomic_write_json(target, result)
        with target.open("r", encoding="utf-8") as stream:
            reloaded = json.load(stream)
        if reloaded.get("content_sha256") != result["content_sha256"]:
            raise PivotCalibrationError("pivot result read-back hash mismatch")
        return reloaded
    finally:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
