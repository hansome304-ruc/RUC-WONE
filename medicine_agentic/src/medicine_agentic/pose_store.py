from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

try:
    import fcntl
except ImportError:  # pragma: no cover - dosw1 and development macOS both provide it
    fcntl = None


SCHEMA_VERSION = 1
HARD_JOINT_LIMITS_RAD = (
    (-3.151, 2.089),
    (-2.963, 0.181),
    (-0.094, 3.161),
    (-3.012, 3.012),
    (-1.859, 1.859),
    (-3.017, 3.017),
)
DEFAULT_SOFT_MARGIN_RAD = math.radians(5.0)


class PoseStoreError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    value = Path(path).expanduser()
    if not value.is_file():
        return None
    digest = hashlib.sha256()
    with value.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_without_hash(payload: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    result.pop("content_sha256", None)
    return result


def document_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        _payload_without_hash(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def new_pose_document(robot_id: str = "dosw1") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "robot_id": robot_id,
        "revision": 0,
        # Keep an absent store deterministic.  ``save`` replaces this sentinel
        # with the real wall-clock time.  Without a stable value, an optimistic
        # concurrency check could fail simply because two calls happened a few
        # milliseconds apart before the file existed.
        "updated_at": "1970-01-01T00:00:00.000Z",
        "units": {
            "joint_position": "rad",
            "position": "m",
            "quaternion": "xyzw",
            "eef_feedback": "m",
            "lift": "mm",
        },
        "poses": {},
    }
    payload["content_sha256"] = document_sha256(payload)
    return payload


def soft_joint_limits(
    margin_rad: float = DEFAULT_SOFT_MARGIN_RAD,
) -> tuple[tuple[float, float], ...]:
    if margin_rad < 0.0:
        raise ValueError("joint soft-limit margin must be non-negative")
    return tuple((low + margin_rad, high - margin_rad) for low, high in HARD_JOINT_LIMITS_RAD)


def validate_joint_position(
    joints: Any,
    *,
    margin_rad: float = DEFAULT_SOFT_MARGIN_RAD,
) -> list[str]:
    errors: list[str] = []
    values = np.asarray(joints, dtype=np.float64)
    if values.shape != (6,):
        return [f"joint_position_rad must have shape (6,), got {values.shape}"]
    if not np.all(np.isfinite(values)):
        return ["joint_position_rad contains non-finite values"]
    for index, (value, limits) in enumerate(zip(values, soft_joint_limits(margin_rad))):
        low, high = limits
        if not low <= float(value) <= high:
            errors.append(
                f"joint {index + 1}={float(value):.6f}rad outside "
                f"soft limits [{low:.6f}, {high:.6f}]"
            )
    return errors


def _validate_arm_record(name: str, arm: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(arm, dict):
        return [f"{name} arm record must be an object"]
    errors.extend(
        f"{name}: {error}"
        for error in validate_joint_position(arm.get("joint_position_rad"))
    )
    flange = arm.get("flange_pose_in_base")
    if not isinstance(flange, dict):
        errors.append(f"{name}: flange_pose_in_base must be an object")
        return errors
    position = np.asarray(flange.get("position_m"), dtype=np.float64)
    quaternion = np.asarray(flange.get("quaternion_xyzw"), dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        errors.append(f"{name}: flange position must contain 3 finite values")
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        errors.append(f"{name}: flange quaternion must contain 4 finite values")
    else:
        norm = float(np.linalg.norm(quaternion))
        if abs(norm - 1.0) > 1e-4:
            errors.append(f"{name}: flange quaternion norm is {norm:.8f}, expected 1")
    return errors


def validate_pose_document(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return ["pose store root must be an object"]
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {SCHEMA_VERSION}, got {payload.get('schema_version')!r}"
        )
    if not isinstance(payload.get("robot_id"), str) or not payload.get("robot_id"):
        errors.append("robot_id must be a non-empty string")
    if not isinstance(payload.get("revision"), int) or int(payload.get("revision", -1)) < 0:
        errors.append("revision must be a non-negative integer")
    poses = payload.get("poses")
    if not isinstance(poses, dict):
        errors.append("poses must be an object")
        return errors
    for pose_name, pose in poses.items():
        if not isinstance(pose_name, str) or not pose_name:
            errors.append("pose names must be non-empty strings")
            continue
        if not isinstance(pose, dict):
            errors.append(f"{pose_name}: pose must be an object")
            continue
        arms = pose.get("arms")
        if not isinstance(arms, dict) or set(arms) != {"left", "right"}:
            errors.append(f"{pose_name}: arms must contain exactly left and right")
            continue
        errors.extend(f"{pose_name}: {error}" for error in _validate_arm_record("left", arms["left"]))
        errors.extend(f"{pose_name}: {error}" for error in _validate_arm_record("right", arms["right"]))
    expected_hash = payload.get("content_sha256")
    if expected_hash is not None and expected_hash != document_sha256(payload):
        errors.append("content_sha256 does not match the document")
    return errors


class PoseStore:
    def __init__(self, path: str | Path, *, robot_id: str = "dosw1") -> None:
        self.path = Path(path).expanduser()
        self.robot_id = robot_id
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.history_dir = self.path.parent / f"{self.path.stem}_history"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return new_pose_document(self.robot_id)
        with self.path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        errors = validate_pose_document(payload)
        if errors:
            raise PoseStoreError(
                f"invalid pose store {self.path}: " + "; ".join(errors)
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

    def save(
        self,
        payload: dict[str, Any],
        *,
        expected_current_sha256: str | None = None,
    ) -> dict[str, Any]:
        with self._lock():
            current = self.load()
            current_hash = current["content_sha256"]
            if (
                expected_current_sha256 is not None
                and expected_current_sha256 != current_hash
            ):
                raise PoseStoreError(
                    "pose store changed since it was read: "
                    f"expected {expected_current_sha256}, current {current_hash}"
                )
            candidate = copy.deepcopy(payload)
            candidate["updated_at"] = utc_now()
            candidate["content_sha256"] = document_sha256(candidate)
            errors = validate_pose_document(candidate)
            if errors:
                raise PoseStoreError("refusing to save invalid pose store: " + "; ".join(errors))
            if self.path.exists():
                self.history_dir.mkdir(parents=True, exist_ok=True)
                backup = self.history_dir / (
                    f"revision_{int(current['revision']):04d}_{current_hash[:12]}.json"
                )
                if not backup.exists():
                    backup.write_text(
                        json.dumps(current, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
            self._atomic_write(candidate)
            reloaded = self.load()
            if reloaded["content_sha256"] != candidate["content_sha256"]:
                raise PoseStoreError("pose store read-back hash mismatch")
            return reloaded

    def upsert_pose(
        self,
        name: str,
        pose: dict[str, Any],
        *,
        replace: bool = False,
        expected_pose_sha256: str | None = None,
    ) -> dict[str, Any]:
        if not name or any(character.isspace() for character in name):
            raise PoseStoreError("pose name must be non-empty and contain no whitespace")
        current = self.load()
        existing = current["poses"].get(name)
        if existing is not None and not replace:
            raise PoseStoreError(
                f"pose {name!r} already exists; use --replace with its expected hash"
            )
        if existing is not None:
            existing_hash = hashlib.sha256(
                json.dumps(
                    existing,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            if expected_pose_sha256 is None:
                raise PoseStoreError(
                    f"replacing {name!r} requires --expected-pose-sha256 {existing_hash}"
                )
            if expected_pose_sha256 != existing_hash:
                raise PoseStoreError(
                    f"pose {name!r} hash mismatch: expected {expected_pose_sha256}, "
                    f"current {existing_hash}"
                )
        candidate = copy.deepcopy(current)
        candidate["revision"] = int(current["revision"]) + 1
        candidate["poses"][name] = copy.deepcopy(pose)
        return self.save(
            candidate,
            expected_current_sha256=current["content_sha256"],
        )

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.path)
            directory_descriptor = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
