"""Small, hot-reloadable operator parameters for the packaging console.

The main JSON configuration describes hardware and stable task contracts.  This
module deliberately owns only the few scalar values that are tuned at the
workstation.  Updates are validated, applied in memory, and atomically persisted
without restarting the HTTP service.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any


TASK_IDS = frozenset({"task1", "task2", "task3"})
ARM_IDS = frozenset({"left", "right"})
POSE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
GRIPPER_POSITION_LIMITS_M = (0.0, 0.10)
SCALAR_LIMITS = {
    "transit_z_m": (0.05, 0.35),
    "pre_contact_clearance_m": (0.005, 0.08),
    "test_lift_m": (0.005, 0.05),
    "contact_flange_z_m": (-0.03, 0.20),
}
TASK1_TEST_LIFT_LIMITS_M = (0.005, 0.10)
TASK1_CONTACT_FLANGE_Z_LIMITS_M = (-0.10, 0.20)
TASK1_TRANSIT_Z_LIMITS_M = (0.0, 0.35)


class RuntimeParameterStore:
    """Thread-safe, validated store for operator-tunable task parameters."""

    def __init__(self, path: Path, defaults: dict[str, Any]) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._defaults = self._validate_tasks(defaults)
        self._tasks = copy.deepcopy(self._defaults)
        self._poses: dict[str, dict[str, Any]] = {"left": {}, "right": {}}
        self._revision = 0
        self._updated_at: float | None = None
        self._load_existing()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": 1,
                "revision": self._revision,
                "updated_at": self._updated_at,
                "path": str(self.path),
                "tasks": copy.deepcopy(self._tasks),
                "poses": copy.deepcopy(self._poses),
            }

    def task(self, task_id: str) -> dict[str, Any]:
        if task_id not in TASK_IDS:
            raise ValueError("task_id must be task1, task2 or task3")
        with self._lock:
            return copy.deepcopy(self._tasks.get(task_id, {}))

    def update_task(
        self,
        task_id: str,
        values: dict[str, Any],
        *,
        source: str = "operator",
    ) -> dict[str, Any]:
        if task_id not in TASK_IDS:
            raise ValueError("task_id must be task1, task2 or task3")
        if not isinstance(values, dict) or not values:
            raise ValueError("values must be a non-empty object")
        allowed = set(SCALAR_LIMITS)
        if task_id == "task1":
            allowed.add("contact_flange_z_m_by_layer")
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(
                "unsupported runtime parameter(s): " + ", ".join(sorted(unknown))
            )
        with self._lock:
            candidate = copy.deepcopy(self._tasks)
            candidate.setdefault(task_id, {}).update(copy.deepcopy(values))
            candidate = self._validate_tasks(candidate)
            previous_tasks = self._tasks
            previous_revision = self._revision
            previous_updated_at = self._updated_at
            self._tasks = candidate
            self._revision += 1
            self._updated_at = time.time()
            try:
                self._persist(source=source)
            except OSError:
                self._tasks = previous_tasks
                self._revision = previous_revision
                self._updated_at = previous_updated_at
                raise
            return self.snapshot()

    def save_pose(
        self,
        arm: str,
        name: str,
        pose: dict[str, Any],
        *,
        source: str = "operator",
    ) -> dict[str, Any]:
        if arm not in ARM_IDS:
            raise ValueError("arm must be left or right")
        if not isinstance(name, str) or not POSE_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                "pose name must use lowercase letters, digits and underscores"
            )
        validated_pose = self._validate_pose(arm, pose)
        with self._lock:
            previous_poses = copy.deepcopy(self._poses)
            previous_revision = self._revision
            previous_updated_at = self._updated_at
            self._poses.setdefault(arm, {})[name] = validated_pose
            self._revision += 1
            self._updated_at = time.time()
            try:
                self._persist(source=source)
            except OSError:
                self._poses = previous_poses
                self._revision = previous_revision
                self._updated_at = previous_updated_at
                raise
            return self.snapshot()

    def pose(self, arm: str, name: str) -> dict[str, Any]:
        if arm not in ARM_IDS:
            raise ValueError("arm must be left or right")
        if not isinstance(name, str) or not POSE_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                "pose name must use lowercase letters, digits and underscores"
            )
        with self._lock:
            pose = self._poses.get(arm, {}).get(name)
            if pose is None:
                raise ValueError(f"saved pose does not exist: {arm}.{name}")
            return copy.deepcopy(pose)

    def delete_pose(
        self,
        arm: str,
        name: str,
        *,
        source: str = "operator",
    ) -> dict[str, Any]:
        if arm not in ARM_IDS:
            raise ValueError("arm must be left or right")
        if not isinstance(name, str) or not POSE_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                "pose name must use lowercase letters, digits and underscores"
            )
        with self._lock:
            if name not in self._poses.get(arm, {}):
                raise ValueError(f"saved pose does not exist: {arm}.{name}")
            previous_poses = copy.deepcopy(self._poses)
            previous_revision = self._revision
            previous_updated_at = self._updated_at
            del self._poses[arm][name]
            self._revision += 1
            self._updated_at = time.time()
            try:
                self._persist(source=source)
            except OSError:
                self._poses = previous_poses
                self._revision = previous_revision
                self._updated_at = previous_updated_at
                raise
            return self.snapshot()

    def _load_existing(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            tasks = payload.get("tasks", {})
            if not isinstance(tasks, dict):
                raise ValueError("tasks must be an object")
            merged = copy.deepcopy(self._defaults)
            for task_id, values in tasks.items():
                if task_id in TASK_IDS and isinstance(values, dict):
                    merged.setdefault(task_id, {}).update(values)
            self._tasks = self._validate_tasks(merged)
            poses = payload.get("poses", {})
            if poses is not None:
                if not isinstance(poses, dict):
                    raise ValueError("poses must be an object")
                validated_poses: dict[str, dict[str, Any]] = {
                    "left": {},
                    "right": {},
                }
                for arm, arm_poses in poses.items():
                    if arm not in ARM_IDS:
                        continue
                    if not isinstance(arm_poses, dict):
                        raise ValueError(f"poses.{arm} must be an object")
                    for name, pose in arm_poses.items():
                        if not POSE_NAME_PATTERN.fullmatch(str(name)):
                            raise ValueError(f"invalid pose name: {name}")
                        validated_poses[arm][str(name)] = self._validate_pose(
                            arm,
                            pose,
                        )
                self._poses = validated_poses
            self._revision = max(0, int(payload.get("revision", 0)))
            updated_at = payload.get("updated_at")
            if updated_at is not None and math.isfinite(float(updated_at)):
                self._updated_at = float(updated_at)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # A malformed operator file must never prevent the robot console from
            # starting.  Stable defaults remain active until the next valid save.
            self._tasks = copy.deepcopy(self._defaults)
            self._poses = {"left": {}, "right": {}}

    @staticmethod
    def _validate_pose(arm: str, pose: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(pose, dict):
            raise ValueError("pose must be an object")
        try:
            position = [float(value) for value in pose["position_m"]]
            quaternion = [float(value) for value in pose["quaternion_xyzw"]]
            joints = [float(value) for value in pose["joint_positions_rad"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "pose requires position_m, quaternion_xyzw and joint_positions_rad"
            ) from exc
        if len(position) != 3 or len(quaternion) != 4 or len(joints) != 6:
            raise ValueError("pose vectors must have lengths 3, 4 and 6")
        if not all(math.isfinite(value) for value in position + quaternion + joints):
            raise ValueError("pose vectors must contain finite numbers")
        quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
        if not 0.99 <= quaternion_norm <= 1.01:
            raise ValueError("pose quaternion must be normalized")
        validated = {
            "arm": arm,
            "frame": str(pose.get("frame", f"{arm}_base")),
            "position_m": position,
            "quaternion_xyzw": quaternion,
            "joint_positions_rad": joints,
            "captured_at": float(pose.get("captured_at", time.time())),
        }
        if pose.get("gripper_position_m") is not None:
            gripper_position = float(pose["gripper_position_m"])
            minimum, maximum = GRIPPER_POSITION_LIMITS_M
            if (
                not math.isfinite(gripper_position)
                or not minimum <= gripper_position <= maximum
            ):
                raise ValueError(
                    "pose gripper_position_m must be between "
                    f"{minimum} and {maximum} m"
                )
            validated["gripper_position_m"] = gripper_position
        return validated

    @classmethod
    def _validate_tasks(cls, tasks: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(tasks, dict):
            raise ValueError("runtime task parameters must be an object")
        validated: dict[str, Any] = {}
        for task_id, raw_values in tasks.items():
            if task_id not in TASK_IDS:
                continue
            if not isinstance(raw_values, dict):
                raise ValueError(f"{task_id} parameters must be an object")
            values: dict[str, Any] = {}
            for key, raw_value in raw_values.items():
                if key in SCALAR_LIMITS:
                    value = float(raw_value)
                    minimum, maximum = SCALAR_LIMITS[key]
                    if task_id == "task1" and key == "transit_z_m":
                        minimum, maximum = TASK1_TRANSIT_Z_LIMITS_M
                    if task_id == "task2" and key == "pre_contact_clearance_m":
                        minimum = 0.0
                    if task_id == "task1" and key == "test_lift_m":
                        minimum, maximum = TASK1_TEST_LIFT_LIMITS_M
                    if not math.isfinite(value) or not minimum <= value <= maximum:
                        raise ValueError(
                            f"{task_id}.{key} must be between {minimum} and {maximum} m"
                        )
                    values[key] = value
                elif key == "contact_flange_z_m_by_layer" and task_id == "task1":
                    if not isinstance(raw_value, dict):
                        raise ValueError(
                            "task1.contact_flange_z_m_by_layer must be an object"
                        )
                    layers: dict[str, float] = {}
                    for layer in ("1", "2", "3"):
                        if layer not in raw_value:
                            raise ValueError(f"task1 contact Z is missing layer {layer}")
                        value = float(raw_value[layer])
                        minimum, maximum = TASK1_CONTACT_FLANGE_Z_LIMITS_M
                        if not math.isfinite(value) or not minimum <= value <= maximum:
                            raise ValueError(
                                f"task1 layer {layer} contact Z must be between "
                                f"{minimum} and {maximum} m"
                            )
                        layers[layer] = value
                    values[key] = layers
                else:
                    raise ValueError(f"unsupported runtime parameter: {task_id}.{key}")
            validated[task_id] = values
        return validated

    def _persist(self, *, source: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "revision": self._revision,
            "updated_at": self._updated_at,
            "source": source,
            "tasks": self._tasks,
            "poses": self._poses,
        }
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
