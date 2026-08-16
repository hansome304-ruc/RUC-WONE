"""Private operator-controlled HTTP console for medicine packaging."""
from __future__ import annotations

import argparse
import copy
import fcntl
import ipaddress
import json
import math
import mimetypes
import os
import signal
import shutil
import stat
import subprocess
import sys
import termios
import threading
import time
import uuid
import urllib.error
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence
from urllib.parse import parse_qs, quote, unquote, urlsplit

import cv2
import numpy as np

from medicine_agentic.act_inference import (
    ActInferenceClient,
    ActInferenceProtocolError,
    ActInferenceUnavailable,
)
from medicine_agentic.act_rollout import (
    ActRolloutConflict,
    ActRolloutController,
    ActRolloutSafetyViolation,
    ActRolloutUnavailable,
)
from medicine_agentic.cartesian_jog import (
    CartesianJogConflict,
    CartesianJogController,
    CartesianJogSafetyViolation,
    CartesianJogTimeout,
    CartesianJogUnavailable,
)
from medicine_agentic.packaging_camera import (
    CameraUnavailable,
    PackagingCamera,
    create_camera,
)
from medicine_agentic.detector_provider import (
    DetectorProvider,
    create_detector_provider,
)
from medicine_agentic.fixed_suction_axis import (
    fixed_suction_axis_status,
    project_fixed_suction_axis,
    quaternion_xyzw_to_matrix,
)
from medicine_agentic.reference_faces import (
    ReferenceFaceBank,
    load_reference_face_bank,
)
from medicine_agentic.runtime_parameters import RuntimeParameterStore
from medicine_agentic.task1_box import (
    BoxCandidate,
    LocatedBox,
    deproject_pixel,
    draw_overlay,
    estimate_candidate_physical_size_rgbd,
    evaluate_dual_suction_depth,
    load_cam_to_left,
    plan_dual_suction_target,
    propose_task1_surface_grid,
    propose_task2_single_row,
    split_carton_grid_candidate,
    transform_point,
)
from medicine_agentic.task2_visual_detector import Task2AdaptiveVisualDetector
from medicine_agentic.task1_visual_recovery import (
    Task1AdaptiveVisualDetector,
    Task1StackOccupancyPrior,
    draw_task1_stack_debug_overlay,
    recover_task1_grid_candidates,
)
from medicine_agentic.trajectory_recorder import (
    RecordingConflict,
    RecordingUnavailable,
    TrajectoryRecorder,
)
from medicine_agentic.trajectory_replay import (
    ReplayConflict,
    ReplayUnavailable,
    TrajectoryReplay,
)
from medicine_agentic.base_trajectory import (
    BaseTrajectoryConflict,
    BaseTrajectoryController,
    BaseTrajectorySafetyViolation,
    BaseTrajectoryUnavailable,
)
from medicine_agentic.teleop_launcher import (
    TeleopLauncher,
    TeleopLaunchConflict,
    TeleopLaunchUnavailable,
)


SERVICE_NAME = "medicine-packaging-console"
SERVICE_VERSION = "0.7.0"
DEFAULT_BIND = "127.0.0.1"
DEFAULT_PORT = 8899
RESERVED_PORTS = frozenset({8765, 8766, 8888, 9999})
TASK_IDS = frozenset({"task1", "task2", "task3"})
# Each task uses its saved observation pose before one-call recognition flows.
# The explicit workflow detection steps remain perception-only; their watcher
# transition is a separate, operator-visible step.
WATCHER_RECOGNITION_POSES = {
    "task1": "left_watcher",
    "task2": "left_watcher",
    "task3": "left_box_watcher",
}
FIXED_SUCTION_DEPTH_FALLBACK_PRIMARY_RATIO = 0.70
FIXED_SUCTION_DEPTH_FALLBACK_MIN_RATIO = 0.15
FIXED_SUCTION_DEPTH_FALLBACK_MAX_DELTA_MM = 15.0


def _quaternion_error_deg(
    first_xyzw: Sequence[float],
    second_xyzw: Sequence[float],
) -> float:
    """Return the shortest angular distance between two quaternions."""

    first = np.asarray(first_xyzw, dtype=np.float64)
    second = np.asarray(second_xyzw, dtype=np.float64)
    if (
        first.shape != (4,)
        or second.shape != (4,)
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
    ):
        raise ValueError("quaternions must contain four finite values")
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-9 or second_norm <= 1e-9:
        raise ValueError("quaternions must have non-zero norms")
    cosine = abs(float(np.dot(first / first_norm, second / second_norm)))
    cosine = min(1.0, max(-1.0, cosine))
    return math.degrees(2.0 * math.acos(cosine))


def apply_fixed_suction_depth_plane_fallback(
    support: dict[str, Any],
) -> dict[str, Any]:
    """Accept bounded RealSense holes when both cups remain on one plane."""

    result = copy.deepcopy(support)
    if result.get("valid") is True or result.get("available") is not True:
        return result
    cups = result.get("cups")
    if not isinstance(cups, list) or len(cups) != 2:
        return result
    try:
        strict_ratio = float(result.get("minimum_valid_ratio", 0.8))
        ratios = [float(cup.get("valid_ratio")) for cup in cups]
        medians = [float(cup.get("median_depth_m")) for cup in cups]
    except (TypeError, ValueError):
        return result
    if not all(math.isfinite(value) for value in (*ratios, *medians)):
        return result
    delta_mm = abs(medians[1] - medians[0]) * 1000.0
    fallback_valid = bool(
        max(ratios) >= min(
            strict_ratio,
            FIXED_SUCTION_DEPTH_FALLBACK_PRIMARY_RATIO,
        )
        and min(ratios) >= FIXED_SUCTION_DEPTH_FALLBACK_MIN_RATIO
        and delta_mm <= FIXED_SUCTION_DEPTH_FALLBACK_MAX_DELTA_MM
    )
    result["plane_fallback"] = {
        "used": fallback_valid,
        "minimum_primary_valid_ratio": (
            FIXED_SUCTION_DEPTH_FALLBACK_PRIMARY_RATIO
        ),
        "minimum_secondary_valid_ratio": (
            FIXED_SUCTION_DEPTH_FALLBACK_MIN_RATIO
        ),
        "maximum_median_depth_delta_mm": (
            FIXED_SUCTION_DEPTH_FALLBACK_MAX_DELTA_MM
        ),
        "measured_valid_ratios": ratios,
        "measured_median_depth_delta_mm": delta_mm,
    }
    if fallback_valid:
        result["valid"] = True
    return result


def apply_task1_verified_center_depth_fallback(
    support: dict[str, Any],
    verified_center_depth_m: float | None,
    *,
    maximum_delta_m: float = 0.012,
) -> dict[str, Any]:
    """Fill one Task1 RealSense cup hole from a verified carton centre plane.

    The caller must already have verified the carton face geometry and both
    fixed cup footprints.  This fallback only repairs depth availability; it
    does not bypass the fixed-axis polygon projection.
    """

    result = copy.deepcopy(support)
    if result.get("valid") is True or result.get("available") is not True:
        return result
    try:
        center_depth = float(verified_center_depth_m)
        maximum_delta = float(maximum_delta_m)
    except (TypeError, ValueError):
        return result
    if (
        not math.isfinite(center_depth)
        or center_depth <= 0.0
        or not math.isfinite(maximum_delta)
        or not 0.001 <= maximum_delta <= 0.030
    ):
        return result
    cups = result.get("cups")
    if not isinstance(cups, list) or len(cups) != 2:
        return result

    valid_indices: list[int] = []
    for index, cup in enumerate(cups):
        try:
            depth = float(cup.get("median_depth_m"))
        except (TypeError, ValueError):
            continue
        if cup.get("valid") is True and math.isfinite(depth) and depth > 0.0:
            valid_indices.append(index)
    if len(valid_indices) != 1:
        return result

    valid_index = valid_indices[0]
    missing_index = 1 - valid_index
    valid_depth = float(cups[valid_index]["median_depth_m"])
    delta_m = abs(valid_depth - center_depth)
    fallback_valid = delta_m <= maximum_delta
    result["verified_center_plane_fallback"] = {
        "used": fallback_valid,
        "source": "verified_physical_instance_center_depth",
        "verified_center_depth_m": center_depth,
        "measured_cup_index": valid_index,
        "missing_cup_index": missing_index,
        "measured_depth_delta_mm": delta_m * 1000.0,
        "maximum_depth_delta_mm": maximum_delta * 1000.0,
    }
    if not fallback_valid:
        return result

    missing_cup = cups[missing_index]
    missing_cup["sensor_valid_ratio"] = missing_cup.get("valid_ratio")
    missing_cup["sensor_median_depth_m"] = missing_cup.get("median_depth_m")
    missing_cup["median_depth_m"] = center_depth
    missing_cup["valid"] = True
    missing_cup["depth_source"] = "verified_physical_instance_center_depth"
    result["valid"] = True
    result["median_depth_delta_mm"] = delta_m * 1000.0
    result["surface_consistency_gated"] = True
    return result


ACT_START_POSE_JOINT_NAMES = (
    "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4",
    "left_joint_5", "left_joint_6", "left_gripper_raw", "right_joint_1",
    "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5",
    "right_joint_6", "right_gripper_raw",
)
ACT_START_POSE_MIN = np.asarray(
    [-1.0297932625, -0.1329442263, -0.0116350045, -1.9151979685,
     -0.2096208185, -0.4430838525, 0.0002624177, 0.8161669374,
     -0.1993209720, -0.0310902577, 1.4188983440, 0.0589379705,
     -1.9266422987, 0.0001079996],
    dtype=np.float64,
)
ACT_START_POSE_MAX = np.asarray(
    [-0.6918058991, -0.0249866489, 0.0619897768, 0.4156176150,
     0.2383560389, 1.7858778238, 0.0003913740, 0.8932250142,
     0.0108720530, 0.0742367059, 1.9148164988, 0.1276035756,
     -1.4463645219, 0.0049463590],
    dtype=np.float64,
)


def describe_act_start_pose(state: Sequence[float]) -> dict[str, Any]:
    """Describe start-pose coverage without authorizing or blocking motion."""

    qpos = np.asarray(state, dtype=np.float64)
    if qpos.shape != (14,) or not np.all(np.isfinite(qpos)):
        raise ValueError("ACT start-pose diagnostic requires 14 finite values")
    outside = []
    for index, name in enumerate(ACT_START_POSE_JOINT_NAMES):
        value = float(qpos[index])
        minimum = float(ACT_START_POSE_MIN[index])
        maximum = float(ACT_START_POSE_MAX[index])
        if value < minimum:
            direction, distance = "below", minimum - value
        elif value > maximum:
            direction, distance = "above", value - maximum
        else:
            continue
        outside.append({
            "index": index,
            "name": name,
            "value": value,
            "range": [minimum, maximum],
            "direction": direction,
            "distance_to_range": distance,
        })
    return {
        "version": "medicine_act_start_pose_diagnostic_v1",
        "reference_episode_count": 25,
        "within_training_start_range": not outside,
        "warning": (
            None if not outside
            else "current pose is outside one or more demonstrated start ranges"
        ),
        "out_of_range": outside,
        "diagnostic_only": True,
        "blocking": False,
    }


LIVE_RGBD_MAX_AGE_S = 3.0
OVERLAY_CACHE_SIZE = 8
SYSTEM_FOLLOW_DESIRED_PATH = Path("/run/ruc-teleop/follow-desired")
SYSTEM_FOLLOW_READY_PATHS = (
    Path(
        "/home/ubuntu/RUC-WONE/teleop_standalone/runtime/"
        "follow-left-udp-ready.json"
    ),
    Path(
        "/home/ubuntu/RUC-WONE/teleop_standalone/runtime/"
        "follow-right-udp-ready.json"
    ),
)
SYSTEM_FOLLOW_READY_MAX_AGE_S = 5.0
TELEOP_FOLLOW_HOLD_PATH = Path(
    "/home/ubuntu/RUC-WONE/teleop_standalone/runtime/arm_follow_hold.json"
)
TELEOP_MODE_STATE_PATH = Path(
    "/home/ubuntu/RUC-WONE/medicine_agentic/runtime/teleop_mode.json"
)
TELEOP_MODES = frozenset({"dual", "left_only", "right_only"})
PREVIEW_JPEG_CACHE_TTL_S = 0.075
SUCTION_STATE_PATH = Path(
    "/home/ubuntu/RUC-WONE/medicine_agentic/runtime/suction_state.json"
)
TASK1_SLOT_PLAN_STATE_PATH = Path(
    "/home/ubuntu/RUC-WONE/medicine_agentic/runtime/task1_slot_plan.json"
)
SUCTION_SERIAL_LOCK_PATH = Path(
    "/home/ubuntu/RUC-WONE/teleop_standalone/runtime/suction_serial.lock"
)


class SuctionUnavailable(RuntimeError):
    """Raised when the dedicated suction serial channel cannot be used."""


class WristCameraUnavailable(RuntimeError):
    """Raised when a configured shared wrist-camera stream is unavailable."""


class SuctionController:
    """Control suction locally and notify the peer console of state changes."""

    def __init__(self, config: dict[str, Any] | None) -> None:
        cfg = config or {}
        self.enabled = cfg.get("enabled") is True
        self.peer_sync_url = str(
            cfg.get(
                "peer_sync_url",
                "http://127.0.0.1:9999/api/suction/sync",
            )
        ).strip()
        self.device = Path(
            str(
                cfg.get(
                    "device",
                    "/dev/serial/by-path/"
                    "pci-0000:c4:00.3-usb-0:3:1.0-port0",
                )
            )
        )
        self.baud_rate = int(cfg.get("baud_rate", 115200))
        self.on_commands = tuple(
            str(value)
            for value in cfg.get(
                "on_commands",
                ("#005P2500T0000!", "#006P2500T0000!"),
            )
        )
        self.off_commands = tuple(
            str(value)
            for value in cfg.get(
                "off_commands",
                (
                    "#255P1500T0000!",
                    "#005P1500T0000!",
                    "#006P1500T0000!",
                ),
            )
        )
        self.settle_s = float(cfg.get("settle_s", 0.5))
        if not self.on_commands or not self.off_commands:
            raise ValueError("suction on_commands and off_commands are required")
        if self.baud_rate <= 0:
            raise ValueError("suction baud_rate must be positive")
        if not 0.0 <= self.settle_s <= 5.0:
            raise ValueError("suction settle_s must be between 0 and 5 seconds")
        self._lock = threading.RLock()
        saved_state = self._read_saved_state()
        saved_engaged = saved_state.get("engaged")
        self._engaged: bool | None = (
            saved_engaged if isinstance(saved_engaged, bool) else None
        )

    @staticmethod
    def _read_saved_state() -> dict[str, Any]:
        try:
            return _read_json(SUCTION_STATE_PATH)
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    def _remember_state(
        self,
        engaged: bool,
        *,
        source: str,
        updated_at: float | None = None,
    ) -> dict[str, Any]:
        timestamp = float(updated_at if updated_at is not None else time.time())
        with self._lock:
            saved = self._read_saved_state()
            saved_at = saved.get("updated_at")
            if isinstance(saved_at, (int, float)) and timestamp < float(saved_at):
                return saved
            state = {
                "engaged": engaged,
                "source": source,
                "updated_at": timestamp,
            }
            self._engaged = engaged
            _write_json_atomic(SUCTION_STATE_PATH, state)
            return state

    def sync_state(self, engaged: bool, *, source: str, updated_at: float) -> dict[str, Any]:
        if not isinstance(engaged, bool):
            raise ValueError("engaged must be a boolean")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        if not isinstance(updated_at, (int, float)):
            raise ValueError("updated_at must be a number")
        return self._remember_state(
            engaged,
            source=source.strip(),
            updated_at=float(updated_at),
        )

    def _notify_peer(self, state: dict[str, Any]) -> dict[str, Any]:
        if not self.peer_sync_url:
            return {"ok": False, "error": "peer sync URL is not configured"}
        payload = json.dumps(state).encode("utf-8")
        last_error = "peer sync failed"
        for attempt in range(1, 4):
            request = urllib.request.Request(
                self.peer_sync_url,
                method="POST",
                data=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=0.35) as response:
                    result = json.loads(response.read().decode("utf-8"))
                if not isinstance(result, dict) or result.get("ok") is not True:
                    raise ValueError(f"invalid peer response: {result}")
                return {"ok": True, "peer": self.peer_sync_url, "attempts": attempt}
            except Exception as exc:
                last_error = str(exc)
                if attempt < 3:
                    time.sleep(0.05 * attempt)
        return {
            "ok": False,
            "peer": self.peer_sync_url,
            "attempts": 3,
            "error": last_error,
        }

    def status(self) -> dict[str, Any]:
        available = False
        resolved = ""
        error = ""
        if not self.enabled:
            error = "suction control is disabled by configuration"
        else:
            try:
                resolved_path = self.device.resolve(strict=True)
                resolved = str(resolved_path)
                if not stat.S_ISCHR(resolved_path.stat().st_mode):
                    raise OSError(
                        f"suction device is not a character device: {resolved}"
                    )
                if not os.access(resolved_path, os.W_OK):
                    raise PermissionError(
                        f"suction device is not writable: {resolved}"
                    )
                available = True
            except OSError as exc:
                error = str(exc)
        with self._lock:
            saved = self._read_saved_state()
            saved_engaged = saved.get("engaged")
            if isinstance(saved_engaged, bool):
                self._engaged = saved_engaged
            engaged = self._engaged
        return {
            "enabled": self.enabled,
            "available": available,
            "configured_device": str(self.device),
            "resolved_device": resolved,
            "baud_rate": self.baud_rate,
            "engaged": engaged,
            "settle_s": self.settle_s,
            "write_confirmation": "kernel_tx_drain",
            "device_ack_supported": False,
            "error": error,
        }

    def set_engaged(self, engaged: bool) -> dict[str, Any]:
        if not isinstance(engaged, bool):
            raise ValueError("engaged must be a boolean")
        status = self.status()
        if status["available"] is not True:
            raise SuctionUnavailable(
                status["error"] or "suction serial device is unavailable"
            )
        commands = self.on_commands if engaged else self.off_commands
        device = str(self.device)
        try:
            with self._lock:
                SUCTION_SERIAL_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
                lock_fd = os.open(
                    SUCTION_SERIAL_LOCK_PATH,
                    os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                )
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                    subprocess.run(
                        [
                            "stty", "-F", device, str(self.baud_rate), "cs8",
                            "-cstopb", "-parenb", "-ixon", "-ixoff",
                            "-crtscts", "raw",
                        ],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=2.0,
                    )
                    fd = os.open(
                        device,
                        os.O_WRONLY | os.O_NOCTTY | getattr(os, "O_CLOEXEC", 0),
                    )
                    try:
                        for command in commands:
                            pending = memoryview(command.encode("ascii"))
                            while pending:
                                written = os.write(fd, pending)
                                if written <= 0:
                                    raise OSError(
                                        "suction serial write returned no progress"
                                    )
                                pending = pending[written:]
                            time.sleep(0.08)
                        termios.tcdrain(fd)
                    finally:
                        os.close(fd)
                finally:
                    os.close(lock_fd)
                state = self._remember_state(
                    engaged,
                    source="8899_serial_command",
                )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise SuctionUnavailable(
                f"suction serial setup failed: {detail}"
            ) from exc
        except SuctionUnavailable:
            raise
        except Exception as exc:
            raise SuctionUnavailable(
                f"suction serial write failed: {exc}"
            ) from exc
        return {
            "engaged": engaged,
            "device": device,
            "resolved_device": status["resolved_device"],
            "commands": list(commands),
            "write_confirmed": True,
            "device_acknowledged": False,
            "state": state,
            "peer_sync": self._notify_peer(state),
        }


class OverlayUnavailable(LookupError):
    """Raised when an immutable detection overlay is absent or has expired."""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"config root must be an object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _is_loopback(bind: str) -> bool:
    if bind.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(bind).is_loopback
    except ValueError:
        return False


def _encode_jpeg(bgr: Any, quality: int = 88) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), max(50, min(int(quality), 100))],
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode JPEG")
    return encoded.tobytes()


def classify_carton_height(
    measured_height_m: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Classify a measured carton height using configured layer thresholds."""

    measured_height = float(measured_height_m)
    layer_height = float(config.get("layer_height_m", 0.025))
    allowed_layers = sorted(
        {
            int(value)
            for value in config.get("allowed_layers", [1, 2, 3])
            if int(value) > 0
        }
    )
    if (
        not math.isfinite(measured_height)
        or measured_height < 0.0
        or not allowed_layers
        or not 0.005 <= layer_height <= 0.10
    ):
        raise ValueError("layer height configuration is invalid")

    configured_minima = config.get("layer_minimum_height_m")
    classification_method = "nearest_nominal_height"
    minimum_height_by_layer: dict[str, float] = {}
    if configured_minima is not None:
        if not isinstance(configured_minima, dict):
            raise ValueError("layer minimum-height thresholds must be an object")
        previous_threshold = -math.inf
        for candidate_layer in allowed_layers[1:]:
            raw_threshold = configured_minima.get(
                str(candidate_layer), configured_minima.get(candidate_layer)
            )
            if raw_threshold is None:
                raise ValueError(
                    "layer minimum-height threshold is missing for "
                    f"layer {candidate_layer}"
                )
            threshold = float(raw_threshold)
            if not math.isfinite(threshold) or threshold <= previous_threshold:
                raise ValueError(
                    "layer minimum-height thresholds must increase by layer"
                )
            minimum_height_by_layer[str(candidate_layer)] = threshold
            previous_threshold = threshold

        layer = allowed_layers[0]
        for candidate_layer in allowed_layers[1:]:
            if measured_height >= minimum_height_by_layer[str(candidate_layer)]:
                layer = candidate_layer
        classification_method = "configured_minimum_height_thresholds"
    else:
        layer = min(
            allowed_layers,
            key=lambda value: abs(measured_height - value * layer_height),
        )

    expected_height = layer * layer_height
    classification_error = abs(measured_height - expected_height)
    maximum_error = float(config.get("maximum_layer_error_m", 0.010))
    within_nominal_error = classification_error <= maximum_error
    valid = (
        True
        if classification_method == "configured_minimum_height_thresholds"
        else within_nominal_error
    )
    return {
        "enabled": True,
        "valid": valid,
        "layer": layer,
        "measured_height_m": measured_height,
        "expected_height_m": expected_height,
        "classification_error_m": classification_error,
        "maximum_layer_error_m": maximum_error,
        "within_nominal_error": within_nominal_error,
        "layer_height_m": layer_height,
        "classification_method": classification_method,
        "layer_minimum_height_m": minimum_height_by_layer,
        "error": "" if valid else "measured height is not near a configured layer",
    }


def estimate_carton_layer(
    bgr: np.ndarray,
    depth_z16: np.ndarray | None,
    depth_scale_m: float | None,
    intrinsics: Any,
    point_camera_m: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Estimate 1/2/3 carton layers from height above the green table plane."""

    if config.get("enabled") is not True:
        return {"enabled": False, "valid": False, "error": "disabled"}
    if depth_z16 is None or depth_scale_m is None:
        raise ValueError("layer estimation requires synchronized depth")
    if bgr.shape[:2] != depth_z16.shape[:2]:
        raise ValueError("layer estimation requires aligned color and depth")
    matrix = np.asarray(intrinsics, dtype=np.float64)
    point = np.asarray(point_camera_m, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("layer estimation requires a finite 3x3 intrinsic matrix")
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("layer estimation requires a finite camera point")

    lower = np.asarray(
        config.get("table_hsv_lower", [70, 45, 25]),
        dtype=np.int16,
    )
    upper = np.asarray(
        config.get("table_hsv_upper", [100, 255, 255]),
        dtype=np.int16,
    )
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("table HSV bounds must contain three values")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    depth_m = depth_z16.astype(np.float64) * float(depth_scale_m)
    minimum_depth = float(config.get("minimum_depth_m", 0.30))
    maximum_depth = float(config.get("maximum_depth_m", 1.50))
    table_mask = (
        (hsv[:, :, 0] >= lower[0])
        & (hsv[:, :, 0] <= upper[0])
        & (hsv[:, :, 1] >= lower[1])
        & (hsv[:, :, 1] <= upper[1])
        & (hsv[:, :, 2] >= lower[2])
        & (hsv[:, :, 2] <= upper[2])
        & np.isfinite(depth_m)
        & (depth_m >= minimum_depth)
        & (depth_m <= maximum_depth)
    )
    rows, columns = np.nonzero(table_mask)
    sample_limit = int(config.get("sample_limit", 16000))
    minimum_inliers = int(config.get("minimum_inliers", 800))
    if len(columns) < max(3, minimum_inliers):
        raise ValueError(
            f"not enough green-table depth samples: {len(columns)}"
        )
    rng = np.random.default_rng(int(config.get("random_seed", 7)))
    if len(columns) > sample_limit:
        selected = rng.choice(len(columns), size=sample_limit, replace=False)
        columns = columns[selected]
        rows = rows[selected]
    z_values = depth_m[rows, columns]
    fx = float(matrix[0, 0])
    fy = float(matrix[1, 1])
    cx = float(matrix[0, 2])
    cy = float(matrix[1, 2])
    points = np.column_stack(
        (
            (columns.astype(np.float64) - cx) * z_values / fx,
            (rows.astype(np.float64) - cy) * z_values / fy,
            z_values,
        )
    )

    threshold = float(config.get("ransac_inlier_threshold_m", 0.004))
    iterations = int(config.get("ransac_iterations", 400))
    best_mask: np.ndarray | None = None
    best_count = 0
    for _ in range(iterations):
        indices = rng.choice(len(points), size=3, replace=False)
        first, second, third = points[indices]
        normal = np.cross(second - first, third - first)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            continue
        normal /= norm
        offset = -float(np.dot(normal, first))
        inliers = np.abs(points @ normal + offset) <= threshold
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_mask = inliers
    minimum_ratio = float(config.get("minimum_inlier_ratio", 0.10))
    if (
        best_mask is None
        or best_count < minimum_inliers
        or best_count / len(points) < minimum_ratio
    ):
        raise ValueError(
            "green-table plane RANSAC has insufficient support: "
            f"{best_count}/{len(points)}"
        )

    inlier_points = points[best_mask]
    centroid = np.mean(inlier_points, axis=0)
    _u, _s, vh = np.linalg.svd(
        inlier_points - centroid,
        full_matrices=False,
    )
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(np.dot(normal, centroid))
    residuals = np.abs(points @ normal + offset)
    refined_mask = residuals <= threshold
    refined_count = int(np.count_nonzero(refined_mask))
    if refined_count < minimum_inliers:
        raise ValueError("refined green-table plane has insufficient support")

    measured_height = abs(float(np.dot(point, normal) + offset))
    result = classify_carton_height(measured_height, config)
    result.update({
        "table_plane_camera": {
            "normal_xyz": [float(value) for value in normal],
            "offset_m": offset,
            "sample_count": int(len(points)),
            "inlier_count": refined_count,
            "inlier_ratio": float(refined_count / len(points)),
            "median_residual_m": float(np.median(residuals[refined_mask])),
        },
    })
    return result


def classify_carton_layer_from_plane(
    point_camera_m: Any,
    table_plane_camera: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Classify one carton point using an already fitted table plane."""

    point = np.asarray(point_camera_m, dtype=np.float64)
    normal = np.asarray(
        table_plane_camera.get("normal_xyz", []), dtype=np.float64
    )
    offset = float(table_plane_camera.get("offset_m", math.nan))
    if (
        point.shape != (3,)
        or normal.shape != (3,)
        or not np.all(np.isfinite(point))
        or not np.all(np.isfinite(normal))
        or not math.isfinite(offset)
    ):
        raise ValueError("table-plane layer classification inputs are invalid")
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-9:
        raise ValueError("table-plane normal is invalid")
    normal = normal / normal_norm

    measured_height = abs(float(np.dot(point, normal) + offset))
    result = classify_carton_height(measured_height, config)
    result["table_plane_camera"] = dict(table_plane_camera)
    return result


def select_highest_layer_nearest_base(
    candidates: list[Any],
    layer_estimates: dict[int, dict[str, Any]],
    base_points: dict[int, tuple[float, float, float]],
) -> Any | None:
    """Select the highest valid layer, then the shortest left-base XY reach."""

    eligible = [
        item
        for item in candidates
        if layer_estimates.get(id(item), {}).get("valid") is True
        and id(item) in base_points
    ]
    if not eligible:
        return None

    def priority(item: Any) -> tuple[float, float]:
        layer = float(layer_estimates[id(item)].get("layer", 0))
        point = base_points[id(item)]
        planar_distance = math.hypot(float(point[0]), float(point[1]))
        return (-layer, planar_distance)

    return min(eligible, key=priority)


def axial_orientation_error_deg(angle_deg: float, target_deg: float) -> float:
    """Smallest unsigned error between two unoriented image axes.

    A carton long axis has no arrow, so +90 and -90 degrees describe the
    same vertical axis.  The result is therefore wrapped modulo 180 degrees.
    """

    return abs((float(angle_deg) - float(target_deg) + 90.0) % 180.0 - 90.0)


def allow_fixed_axis_validation_after_generic_margin(
    task_id: str,
    generic_target: Any,
    *,
    fixed_axis_ready: bool,
) -> bool:
    """Let a calibrated task tool axis supersede only a margin-only precheck.

    The generic target follows the carton short axis, while Task1 and Task2
    ultimately use their calibrated fixed tool axes.  A small polygon error
    may reject the former even though both real cups fit.  Every other generic
    blocker stays fail-closed, and the real fixed projection/depth checks still
    run afterwards.
    """
    if generic_target is None:
        return False
    if bool(getattr(generic_target, "valid_2d", False)):
        return True
    blockers = set(getattr(generic_target, "blockers", ()) or ())
    return bool(
        task_id in {"task1", "task2"}
        and fixed_axis_ready
        and blockers == {"dual_suction_margin_low"}
    )


def physical_instance_center_depth_m(report: dict[str, Any]) -> float | None:
    """Recover the verified centre depth from a physical-size report.

    The center-depth-projected physical gate repeats one robust centre sample
    at its four axis endpoints. Reusing that sample gives recognition a stable
    3-D panel centre when a real suction-cup patch contains a RealSense hole.
    The real cup patches remain independently gated before picking.
    """

    if report.get("method") != "inner_axis_center_depth_projected":
        return None
    samples = report.get("samples")
    if not isinstance(samples, dict):
        return None
    depths_mm: list[float] = []
    for axis_samples in samples.values():
        if not isinstance(axis_samples, (list, tuple)):
            continue
        for sample in axis_samples:
            if not isinstance(sample, (list, tuple)) or len(sample) < 3:
                continue
            try:
                depth_mm = float(sample[2])
            except (TypeError, ValueError):
                continue
            if math.isfinite(depth_mm) and depth_mm > 0.0:
                depths_mm.append(depth_mm)
    if not depths_mm:
        return None
    return float(np.median(np.asarray(depths_mm, dtype=np.float64))) / 1000.0


def carton_interior_depth_fallback(
    candidate: BoxCandidate,
    depth_z16: np.ndarray | None,
    depth_scale_m: float | None,
    *,
    minimum_valid_ratio: float = 0.40,
    erosion_px: int = 8,
) -> dict[str, Any]:
    """Estimate carton surface depth when a glare hole covers its centre.

    Only the eroded interior of an already verified carton quadrilateral is
    sampled.  This deliberately cannot create a visual detection; it only
    supplies a provisional 3-D centre for later height and fixed-suction
    safety gates.
    """

    unavailable = {
        "available": False,
        "valid": False,
        "method": "eroded_carton_interior_median",
        "minimum_valid_ratio": float(minimum_valid_ratio),
    }
    if depth_z16 is None or depth_scale_m is None:
        return unavailable
    polygon = np.asarray(candidate.polygon_px, dtype=np.float32).reshape(-1, 2)
    if len(polygon) < 4 or not np.all(np.isfinite(polygon)):
        return unavailable
    mask = np.zeros(depth_z16.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 255)
    radius = max(1, int(erosion_px))
    kernel_size = radius * 2 + 1
    interior = cv2.erode(
        mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        ),
    ) > 0
    values = depth_z16[interior]
    valid_values = values[values > 0].astype(np.float64)
    sample_count = int(values.size)
    valid_count = int(valid_values.size)
    valid_ratio = valid_count / max(sample_count, 1)
    result = {
        **unavailable,
        "available": sample_count > 0,
        "sample_count": sample_count,
        "valid_count": valid_count,
        "valid_ratio": valid_ratio,
    }
    if valid_count < 32 or valid_ratio < float(minimum_valid_ratio):
        return result
    median_raw = float(np.median(valid_values))
    absolute_deviation = np.abs(valid_values - median_raw)
    mad_raw = float(np.median(absolute_deviation))
    depth_m = median_raw * float(depth_scale_m)
    mad_m = mad_raw * float(depth_scale_m)
    result.update(
        {
            "valid": bool(
                math.isfinite(depth_m)
                and 0.20 <= depth_m <= 2.0
                and mad_m <= 0.030
            ),
            "median_depth_m": depth_m,
            "median_absolute_deviation_m": mad_m,
        }
    )
    return result


def normalize_task3_front_face_geometry(
    candidate: BoxCandidate,
    physical_report: dict[str, Any],
    intrinsics: Any,
    expected_size_mm: Any,
) -> BoxCandidate | None:
    """Stabilize a verified Task3 bear face using Task2-style RGB-D scale."""

    depth_m = physical_instance_center_depth_m(physical_report)
    if depth_m is None:
        return None
    return normalize_front_face_geometry_at_depth(
        candidate,
        depth_m,
        intrinsics,
        expected_size_mm,
    )


def normalize_front_face_geometry_at_depth(
    candidate: BoxCandidate,
    depth_m: float,
    intrinsics: Any,
    expected_size_mm: Any,
) -> BoxCandidate | None:
    """Project one independently matched bear face at its measured depth."""

    matrix = np.asarray(intrinsics, dtype=np.float64)
    try:
        expected_long_m = float(expected_size_mm[0]) / 1000.0
        expected_short_m = float(expected_size_mm[1]) / 1000.0
    except (IndexError, TypeError, ValueError):
        return None
    if (
        not math.isfinite(float(depth_m))
        or float(depth_m) <= 0.0
        or matrix.shape != (3, 3)
        or not np.all(np.isfinite(matrix))
        or expected_long_m <= 0.0
        or expected_short_m <= 0.0
    ):
        return None
    angle = math.radians(float(candidate.angle_deg))
    long_axis = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
    short_axis = np.asarray([-long_axis[1], long_axis[0]], dtype=np.float64)
    fx = float(matrix[0, 0])
    fy = float(matrix[1, 1])

    def directional_focal(axis: np.ndarray) -> float:
        return math.sqrt((fx * axis[0]) ** 2 + (fy * axis[1]) ** 2)

    long_side_px = expected_long_m * directional_focal(long_axis) / depth_m
    short_side_px = expected_short_m * directional_focal(short_axis) / depth_m
    if (
        not math.isfinite(long_side_px)
        or not math.isfinite(short_side_px)
        or min(long_side_px, short_side_px) < 18.0
    ):
        return None
    center = np.asarray(candidate.center_px, dtype=np.float64)
    half_long = 0.5 * long_side_px
    half_short = 0.5 * short_side_px
    polygon = tuple(
        (
            float(center[0] + sx * half_long * long_axis[0]
                  + sy * half_short * short_axis[0]),
            float(center[1] + sx * half_long * long_axis[1]
                  + sy * half_short * short_axis[1]),
        )
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    )
    return replace(
        candidate,
        polygon_px=polygon,
        long_side_px=float(long_side_px),
        short_side_px=float(short_side_px),
        edge_clearance_px=float(short_side_px) * 0.5,
        provider=f"{candidate.provider}:rgbd_face_rect",
    )


def recover_task3_verified_flat_face_geometry(
    candidate: BoxCandidate,
    flat_footprint_report: dict[str, Any],
    intrinsics: Any,
    expected_face_size_mm: Any,
    *,
    require_verified_front: bool = True,
) -> BoxCandidate | None:
    """Recover the 130 x 85 face inside one located Task3 flat stack.

    Strict mode preserves the original SIFT and complete-footprint checks.
    Task3's fixed lower-left layout can opt into a tolerant mode after the two
    physical stacks have already been separated. In that mode the outer
    205 x 130 mm dieline may be partially folded; only a usable centre depth is
    required to project the actual 130 x 85 mm suction face.
    """

    base_invalid = (
        candidate.graspable is not True
        or not str(candidate.provider).startswith("reference_feature")
    )
    strict_valid = (
        candidate.face_type == "front_large"
        and bool(candidate.reference_face_id)
        and flat_footprint_report.get("valid") is True
    )
    relaxed_valid = physical_instance_center_depth_m(
        flat_footprint_report
    ) is not None
    if base_invalid or not (
        strict_valid if require_verified_front else relaxed_valid
    ):
        return None
    normalized = normalize_task3_front_face_geometry(
        candidate,
        flat_footprint_report,
        intrinsics,
        expected_face_size_mm,
    )
    if normalized is None:
        return None
    return replace(
        normalized,
        face_type="front_large",
        reference_face_id=(
            candidate.reference_face_id if require_verified_front else None
        ),
        provider=(
            f"{candidate.provider}:"
            + (
                "task3_verified_flat_face"
                if require_verified_front
                else "task3_roi_front_face"
            )
            + ":rgbd_face_rect"
        ),
    )


def normalized_roi_contains_point(
    center_px: tuple[float, float],
    image_shape: tuple[int, int],
    roi_norm: Any,
) -> bool:
    """Return whether a pixel centre lies inside a normalized task region."""

    if not isinstance(roi_norm, (list, tuple)) or len(roi_norm) != 4:
        return False
    try:
        x0, y0, x1, y1 = (float(value) for value in roi_norm)
    except (TypeError, ValueError):
        return False
    height, width = image_shape
    if not (
        0.0 <= x0 < x1 <= 1.0
        and 0.0 <= y0 < y1 <= 1.0
        and height > 0
        and width > 0
    ):
        return False
    x, y = float(center_px[0]), float(center_px[1])
    return x0 * width <= x <= x1 * width and y0 * height <= y <= y1 * height


def normalized_polygon_contains_point(
    center_px: tuple[float, float],
    image_shape: tuple[int, int],
    polygon_norm: Any,
) -> bool:
    """Return whether a pixel centre lies inside a normalized polygon."""

    if not isinstance(polygon_norm, (list, tuple)) or len(polygon_norm) < 3:
        return False
    height, width = image_shape
    if height <= 0 or width <= 0:
        return False
    try:
        normalized = np.asarray(polygon_norm, dtype=np.float32).reshape(-1, 2)
    except (TypeError, ValueError):
        return False
    if (
        len(normalized) < 3
        or not np.all(np.isfinite(normalized))
        or np.any(normalized < 0.0)
        or np.any(normalized > 1.0)
    ):
        return False
    polygon = normalized * np.asarray([width, height], dtype=np.float32)
    point = (float(center_px[0]), float(center_px[1]))
    return cv2.pointPolygonTest(polygon, point, False) >= 0.0


def normalized_roi_intersects_polygon(
    polygon_px: Any,
    image_shape: tuple[int, int],
    roi_norm: Any,
) -> bool:
    """Return whether a candidate footprint overlaps a normalized region.

    Finished cartons inside the shipping box must not become pick targets.  A
    centre-only test is insufficient near the box boundary, because a carton
    footprint can overlap the finished-goods region while its centre remains
    just outside.  Detector candidates are convex rotated rectangles, so an
    exact convex-polygon intersection is both cheap and deterministic.
    """

    if not isinstance(roi_norm, (list, tuple)) or len(roi_norm) != 4:
        return False
    try:
        x0, y0, x1, y1 = (float(value) for value in roi_norm)
    except (TypeError, ValueError):
        return False
    height, width = image_shape
    if not (
        0.0 <= x0 < x1 <= 1.0
        and 0.0 <= y0 < y1 <= 1.0
        and height > 0
        and width > 0
    ):
        return False
    try:
        polygon = np.asarray(polygon_px, dtype=np.float32).reshape(-1, 2)
    except (TypeError, ValueError):
        return False
    if len(polygon) < 3 or not np.all(np.isfinite(polygon)):
        return False
    roi_polygon = np.asarray(
        [
            [x0 * width, y0 * height],
            [x1 * width, y0 * height],
            [x1 * width, y1 * height],
            [x0 * width, y1 * height],
        ],
        dtype=np.float32,
    )
    polygon = cv2.convexHull(polygon).reshape(-1, 2)
    overlap_area, _ = cv2.intersectConvexConvex(polygon, roi_polygon)
    return bool(float(overlap_area) > 0.5)


def _shipping_box_depth_sample(
    depth_m: np.ndarray,
    pixel: Sequence[float],
    *,
    radius_px: int,
    quantile: float,
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> tuple[float | None, float]:
    """Return a robust local depth and valid-pixel ratio."""

    height, width = depth_m.shape
    u, v = (int(round(float(value))) for value in pixel)
    x0, x1 = max(0, u - radius_px), min(width, u + radius_px + 1)
    y0, y1 = max(0, v - radius_px), min(height, v + radius_px + 1)
    if x0 >= x1 or y0 >= y1:
        return None, 0.0
    region = depth_m[y0:y1, x0:x1]
    valid = (
        np.isfinite(region)
        & (region >= minimum_depth_m)
        & (region <= maximum_depth_m)
    )
    valid_ratio = float(np.count_nonzero(valid) / max(1, region.size))
    if not np.any(valid):
        return None, valid_ratio
    return (
        float(np.quantile(region[valid], min(1.0, max(0.0, quantile)))),
        valid_ratio,
    )


def shipping_box_rim_plane_depth_m(depths_m: Sequence[float]) -> float | None:
    """Estimate the rim plane while rejecting one contaminated probe."""

    depths = np.asarray(depths_m, dtype=np.float64)
    depths = depths[np.isfinite(depths) & (depths > 0.0)]
    if depths.size < 3:
        return None
    return float(np.median(depths))


def project_pixel_to_base_z_plane(
    pixel: Sequence[float],
    plane_z_m: float,
    intrinsics: np.ndarray,
    cam_to_base: np.ndarray,
) -> np.ndarray:
    """Intersect a camera pixel ray with a horizontal base-frame plane."""

    matrix = np.asarray(intrinsics, dtype=np.float64)
    transform = np.asarray(cam_to_base, dtype=np.float64)
    if matrix.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("pixel-plane projection calibration is invalid")
    u, v = (float(value) for value in pixel)
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    if (
        not all(math.isfinite(value) for value in (u, v, plane_z_m, fx, fy, cx, cy))
        or fx <= 0.0
        or fy <= 0.0
        or not np.all(np.isfinite(transform))
    ):
        raise ValueError("pixel-plane projection inputs are invalid")
    camera_ray = np.asarray([(u - cx) / fx, (v - cy) / fy, 1.0])
    base_ray = transform[:3, :3] @ camera_ray
    origin = transform[:3, 3]
    denominator = float(base_ray[2])
    if abs(denominator) < 1e-9:
        raise ValueError("camera ray is parallel to the base-frame plane")
    scale = (float(plane_z_m) - float(origin[2])) / denominator
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("base-frame plane is behind the camera")
    point = origin + base_ray * scale
    point[2] = float(plane_z_m)
    return point


def shipping_box_region_depth_statistics(
    depth_m: np.ndarray,
    opening_polygon_px: np.ndarray,
    intrinsics: np.ndarray,
    cam_to_left: np.ndarray,
    *,
    minimum_depth_m: float,
    maximum_depth_m: float,
    bottom_region_scale: float,
    rim_height_quantile: float,
    bottom_height_quantile: float,
) -> dict[str, Any]:
    """Measure rim and bottom from a contour-following base-frame point cloud."""

    height, width = depth_m.shape
    inner_mask = np.zeros((height, width), dtype=np.uint8)
    bottom_mask = np.zeros((height, width), dtype=np.uint8)
    inner = np.round(opening_polygon_px).astype(np.int32)
    center = np.mean(opening_polygon_px, axis=0)
    bottom = np.round(
        center
        + (opening_polygon_px - center)
        * min(0.90, max(0.25, float(bottom_region_scale)))
    ).astype(np.int32)
    cv2.fillConvexPoly(inner_mask, inner, 1)
    cv2.fillConvexPoly(bottom_mask, bottom, 1)
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= minimum_depth_m)
        & (depth_m <= maximum_depth_m)
    )
    region_valid = (inner_mask == 1) & valid
    bottom_valid = (bottom_mask == 1) & valid
    ys, xs = np.nonzero(region_valid)
    bottom_ys, bottom_xs = np.nonzero(bottom_valid)
    region_total = int(np.count_nonzero(inner_mask))
    bottom_total = int(np.count_nonzero(bottom_mask))
    if xs.size < 100 or bottom_xs.size < 50:
        return {
            "rim_depth_m": None,
            "bottom_depth_m": None,
            "rim_z_m": None,
            "bottom_z_m": None,
            "bottom_point_left_base_m": None,
            "rim_valid_ratio": float(xs.size / max(1, region_total)),
            "bottom_valid_ratio": float(bottom_xs.size / max(1, bottom_total)),
        }

    matrix = np.asarray(intrinsics, dtype=np.float64)
    transform = np.asarray(cam_to_left, dtype=np.float64)

    def base_points(px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        depths = depth_m[py, px]
        camera_points = np.column_stack(
            (
                (px.astype(np.float64) - matrix[0, 2])
                * depths
                / matrix[0, 0],
                (py.astype(np.float64) - matrix[1, 2])
                * depths
                / matrix[1, 1],
                depths,
            )
        )
        points = camera_points @ transform[:3, :3].T + transform[:3, 3]
        return points, depths

    region_points, region_depths = base_points(xs, ys)
    bottom_points, bottom_depths = base_points(bottom_xs, bottom_ys)
    rim_z_m = float(np.quantile(region_points[:, 2], rim_height_quantile))
    bottom_z_m = float(
        np.quantile(bottom_points[:, 2], bottom_height_quantile)
    )
    rim_band = region_points[:, 2] >= rim_z_m
    bottom_band = bottom_points[:, 2] <= bottom_z_m
    rim_depth_m = float(np.median(region_depths[rim_band]))
    bottom_depth_m = float(np.median(bottom_depths[bottom_band]))
    bottom_xy = np.median(bottom_points[bottom_band, :2], axis=0)
    return {
        "rim_depth_m": rim_depth_m,
        "bottom_depth_m": bottom_depth_m,
        "rim_z_m": rim_z_m,
        "bottom_z_m": bottom_z_m,
        "bottom_point_left_base_m": [
            float(bottom_xy[0]),
            float(bottom_xy[1]),
            bottom_z_m,
        ],
        "rim_valid_ratio": float(xs.size / max(1, region_total)),
        "bottom_valid_ratio": float(bottom_xs.size / max(1, bottom_total)),
    }


def task1_slot_placement_target(
    slot: dict[str, Any] | None,
    metric_grid: dict[str, Any],
) -> dict[str, Any] | None:
    """Build the Task1 left-TCP/right-base target for one persisted slot."""

    if not isinstance(slot, dict):
        return None
    return {
        "slot_id": int(slot["slot_id"]),
        "coordinate_frame": "right_base",
        "semantic": "carton_top_face_center_after_vertical_insertion",
        "placement_completion_center_m": copy.deepcopy(
            slot["placement_completion_center_right_base_m"]
        ),
        "approach_center_m": copy.deepcopy(
            slot["approach_center_right_base_m"]
        ),
        "carton_long_axis_yaw_deg": float(
            slot["carton_long_axis_yaw_right_base_deg"]
        ),
        "left_suction_tcp_target": {
            "coordinate_frame": "left_base",
            "release_center_m": copy.deepcopy(
                slot["release_surface_center_left_base_m"]
            ),
            "approach_center_m": copy.deepcopy(
                slot["approach_center_left_base_m"]
            ),
            "pixel_center": copy.deepcopy(slot.get("center_px")),
            "depth_source": "multiframe_rgbd_box_bottom_plane",
            "bottom_plane_rms_residual_mm": metric_grid.get(
                "bottom_plane_rms_residual_mm"
            ),
        },
    }


def apply_task1_slot_progress(
    detection: dict[str, Any],
    placed_slot_ids: Sequence[int],
) -> dict[str, Any]:
    """Apply persistent placement progress without re-running RGB-D."""

    result = copy.deepcopy(detection)
    source_slots = result.get("slots")
    if not isinstance(source_slots, list) or len(source_slots) != 20:
        raise ValueError("Task1 persistent slot plan must contain 20 slots")
    valid_identifiers = {int(slot["slot_id"]) for slot in source_slots}
    placed = {int(value) for value in placed_slot_ids}
    if not placed.issubset(valid_identifiers):
        raise ValueError("Task1 persistent slot progress contains unknown slots")
    slots: list[dict[str, Any]] = []
    for source in source_slots:
        slot = copy.deepcopy(source)
        identifier = int(slot["slot_id"])
        initially_occupied = bool(
            slot.get("occupied_at_first_detection", slot.get("occupied") is True)
        )
        slot["occupied_at_first_detection"] = initially_occupied
        slot["placed_by_workflow"] = identifier in placed
        slot["occupied"] = initially_occupied or identifier in placed
        slots.append(slot)
    slots.sort(key=lambda item: int(item["slot_id"]))
    next_slot = next(
        (copy.deepcopy(slot) for slot in slots if slot["occupied"] is not True),
        None,
    )
    occupied_count = sum(slot["occupied"] is True for slot in slots)
    metric_grid = result.get("metric_grid", {})
    result.update(
        {
            "slots": slots,
            "occupied_count": occupied_count,
            "empty_count": len(slots) - occupied_count,
            "next_slot": next_slot,
            "placement_target": task1_slot_placement_target(
                next_slot,
                metric_grid if isinstance(metric_grid, dict) else {},
            ),
            "target_ready": next_slot is not None,
            "blockers": [] if next_slot is not None else ["shipping_box_full"],
            "slot_plan_progress": {
                "placed_count": len(placed),
                "capacity": len(slots),
                "next_placement_number": (
                    None if next_slot is None else len(placed) + 1
                ),
                "next_slot_id": (
                    None if next_slot is None else int(next_slot["slot_id"])
                ),
                "placed_slot_ids": sorted(placed),
                "remaining_count": len(slots) - occupied_count,
            },
        }
    )
    candidate = result.get("candidate")
    if isinstance(candidate, dict):
        candidate["slots"] = copy.deepcopy(slots)
    candidates = result.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        candidates[0]["slots"] = copy.deepcopy(slots)
    return result


def annotate_task1_rgbd_slots(
    bgr: np.ndarray,
    rgbd_result: dict[str, Any],
    config: dict[str, Any] | None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Add occupancy and a next placement target to the metric RGB-D grid."""

    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Task1 slot annotation requires one uint8 BGR frame")
    cfg = dict(config or {})
    source_slots = rgbd_result.get("slots")
    if not isinstance(source_slots, list) or len(source_slots) != 20:
        raise ValueError("RGB-D slot result must contain exactly 20 slots")

    maximum_saturation = float(
        cfg.get("occupied_maximum_saturation", 200.0)
    )
    minimum_value = float(cfg.get("occupied_minimum_value", 150.0))
    minimum_light_ratio = float(
        cfg.get("occupied_minimum_light_ratio", 0.55)
    )
    cardboard_hsv_lower = np.asarray(
        cfg.get("empty_cardboard_hsv_lower", [8, 45, 75]),
        dtype=np.float64,
    )
    cardboard_hsv_upper = np.asarray(
        cfg.get("empty_cardboard_hsv_upper", [30, 255, 220]),
        dtype=np.float64,
    )
    minimum_cardboard_ratio = float(
        cfg.get("empty_minimum_cardboard_ratio", 0.55)
    )
    inset_scale = float(cfg.get("occupancy_inset_scale", 0.64))
    if not 0.20 <= inset_scale <= 0.95:
        raise ValueError("task1_slot_grid.occupancy_inset_scale is invalid")
    if not 0.0 <= minimum_light_ratio <= 1.0:
        raise ValueError(
            "task1_slot_grid.occupied_minimum_light_ratio is invalid"
        )
    if cardboard_hsv_lower.shape != (3,) or cardboard_hsv_upper.shape != (3,):
        raise ValueError("Task1 empty-cardboard HSV bounds must have three values")
    if not 0.0 <= minimum_cardboard_ratio <= 1.0:
        raise ValueError(
            "task1_slot_grid.empty_minimum_cardboard_ratio is invalid"
        )

    height, width = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    slots: list[dict[str, Any]] = []
    for source in source_slots:
        if not isinstance(source, dict):
            raise ValueError("RGB-D slot entries must be objects")
        slot = copy.deepcopy(source)
        polygon = np.asarray(slot.get("polygon_px"), dtype=np.float64)
        if polygon.shape != (4, 2) or not np.all(np.isfinite(polygon)):
            raise ValueError("RGB-D slot polygon must contain four finite points")
        center = np.mean(polygon, axis=0)
        occupancy_polygon = center + (polygon - center) * inset_scale
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(
            mask,
            np.rint(occupancy_polygon).astype(np.int32),
            255,
        )
        pixels = hsv[mask > 0]
        light_ratio = 0.0
        cardboard_ratio = 0.0
        if pixels.size:
            light_ratio = float(
                np.mean(
                    (pixels[:, 1] <= maximum_saturation)
                    & (pixels[:, 2] >= minimum_value)
                )
            )
            cardboard_ratio = float(
                np.mean(
                    np.all(pixels >= cardboard_hsv_lower, axis=1)
                    & np.all(pixels <= cardboard_hsv_upper, axis=1)
                )
            )
        identifier = int(slot["slot_id"])
        slot.update(
            {
                "id": f"slot-{identifier:02d}",
                "index": identifier,
                "row": int(slot["row"]) + 1,
                "column": int(slot["column"]) + 1,
                "occupied": bool(
                    cardboard_ratio < minimum_cardboard_ratio
                    and light_ratio >= minimum_light_ratio
                ),
                "occupied_at_first_detection": bool(
                    cardboard_ratio < minimum_cardboard_ratio
                    and light_ratio >= minimum_light_ratio
                ),
                "occupancy_light_ratio": light_ratio,
                "empty_cardboard_ratio": cardboard_ratio,
            }
        )
        slots.append(slot)

    slots.sort(key=lambda item: int(item["slot_id"]))
    occupied_count = sum(slot["occupied"] is True for slot in slots)
    next_slot = next(
        (copy.deepcopy(slot) for slot in slots if slot["occupied"] is not True),
        None,
    )
    metric_grid = rgbd_result.get("metric_grid", {})
    placement_target = task1_slot_placement_target(next_slot, metric_grid)

    overlay = bgr.copy()
    fill = overlay.copy()
    for slot in slots:
        polygon_i = np.rint(slot["polygon_px"]).astype(np.int32)
        is_next = (
            next_slot is not None
            and int(slot["slot_id"]) == int(next_slot["slot_id"])
        )
        colour = (
            (50, 190, 255)
            if is_next
            else (179, 242, 93)
            if slot["occupied"] is True
            else (255, 160, 40)
        )
        cv2.fillConvexPoly(fill, polygon_i, colour)
    overlay = cv2.addWeighted(fill, 0.16, overlay, 0.84, 0.0)
    for slot in slots:
        polygon_i = np.rint(slot["polygon_px"]).astype(np.int32)
        is_next = (
            next_slot is not None
            and int(slot["slot_id"]) == int(next_slot["slot_id"])
        )
        colour = (
            (50, 190, 255)
            if is_next
            else (179, 242, 93)
            if slot["occupied"] is True
            else (255, 160, 40)
        )
        cv2.polylines(
            overlay,
            [polygon_i],
            True,
            colour,
            3 if is_next else 1,
            cv2.LINE_AA,
        )
        center = np.mean(polygon_i, axis=0).astype(int)
        cv2.putText(
            overlay,
            f"{int(slot['slot_id']):02d}",
            (int(center[0]) - 8, int(center[1]) + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            colour,
            1,
            cv2.LINE_AA,
        )
    if next_slot is not None:
        precise_center = np.rint(
            np.asarray(next_slot.get("center_px"), dtype=np.float64)
        ).astype(int)
        if precise_center.shape == (2,):
            center_xy = (int(precise_center[0]), int(precise_center[1]))
            cv2.drawMarker(
                overlay,
                center_xy,
                (0, 0, 255),
                cv2.MARKER_CROSS,
                22,
                2,
                cv2.LINE_AA,
            )
            cv2.circle(overlay, center_xy, 7, (255, 255, 255), 2, cv2.LINE_AA)
            release = np.asarray(
                next_slot.get("release_surface_center_left_base_m"),
                dtype=np.float64,
            )
            if release.shape == (3,) and np.all(np.isfinite(release)):
                cv2.putText(
                    overlay,
                    "TCP L:[{:.3f},{:.3f},{:.3f}]m".format(*release),
                    (max(10, center_xy[0] - 125), max(25, center_xy[1] - 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
    boundary = np.asarray(rgbd_result.get("boundary_px"), dtype=np.float64)
    if boundary.shape == (4, 2) and np.all(np.isfinite(boundary)):
        cv2.polylines(
            overlay,
            [np.rint(boundary).astype(np.int32)],
            True,
            (80, 255, 80),
            4,
            cv2.LINE_AA,
        )
    summary = f"TASK1 RGB-D SLOTS {occupied_count}/20 OCCUPIED"
    if next_slot is not None:
        summary += f"  NEXT {int(next_slot['slot_id']):02d}"
    cv2.putText(
        overlay,
        summary,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (50, 190, 255),
        2,
        cv2.LINE_AA,
    )

    metric_grid = copy.deepcopy(rgbd_result.get("metric_grid", {}))
    candidate = {
        "center_px": np.mean(boundary, axis=0).tolist()
        if boundary.shape == (4, 2)
        else None,
        "polygon_px": boundary.tolist()
        if boundary.shape == (4, 2)
        else None,
        "slots": copy.deepcopy(slots),
    }
    detection = {
        "type": "task1_shipping_box_rgbd_slots",
        "detected_2d": True,
        "target_ready": next_slot is not None,
        "motion_ready": False,
        "count": 20,
        "capacity": 20,
        "rows": 10,
        "columns": 2,
        "occupied_count": occupied_count,
        "empty_count": 20 - occupied_count,
        "fill_order": "right_column_then_left",
        "geometry_source": "rgbd_four_inner_edges_and_metric_ray_plane_grid",
        "occupancy_source": "per_slot_cardboard_removal_with_light_carton_gate",
        "quality": {
            "high_confidence": rgbd_result.get("high_confidence") is True,
            "sample_count": rgbd_result.get("sample_count"),
            "maximum_anchor_peak_to_peak_px": rgbd_result.get(
                "maximum_anchor_peak_to_peak_px"
            ),
            "full_array_fit_by_measured_boundary": metric_grid.get(
                "full_array_fit_by_measured_boundary"
            ),
        },
        "metric_grid": metric_grid,
        "candidate": candidate,
        "candidates": [candidate],
        "slots": slots,
        "next_slot": next_slot,
        "placement_target": placement_target,
        "blockers": [] if next_slot is not None else ["shipping_box_full"],
    }
    initially_occupied_ids = [
        int(slot["slot_id"]) for slot in slots if slot["occupied"] is True
    ]
    return apply_task1_slot_progress(detection, initially_occupied_ids), overlay


def measure_task1_top_barcode_stripes(
    bgr: np.ndarray,
    polygon_px: np.ndarray,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Measure barcode-like parallel stripes on the staged top's right side."""

    cfg = dict(config or {})
    polygon = np.asarray(polygon_px, dtype=np.float32)
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Task1 barcode evidence requires uint8 BGR")
    if polygon.shape != (4, 2) or not np.all(np.isfinite(polygon)):
        raise ValueError("Task1 barcode evidence requires a four-point polygon")

    # The fixed station leaves the long carton edge approximately horizontal.
    # Sum/difference ordering also keeps the rectification deterministic when
    # the box is displaced by a few degrees.
    sums = polygon.sum(axis=1)
    differences = polygon[:, 0] - polygon[:, 1]
    ordered = np.asarray(
        [
            polygon[int(np.argmin(sums))],
            polygon[int(np.argmax(differences))],
            polygon[int(np.argmax(sums))],
            polygon[int(np.argmin(differences))],
        ],
        dtype=np.float32,
    )
    warp_width = max(120, int(cfg.get("barcode_warp_width_px", 260)))
    warp_height = max(32, int(cfg.get("barcode_warp_height_px", 64)))
    destination = np.asarray(
        [
            [0.0, 0.0],
            [float(warp_width - 1), 0.0],
            [float(warp_width - 1), float(warp_height - 1)],
            [0.0, float(warp_height - 1)],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(ordered, destination)
    rectified = cv2.warpPerspective(bgr, homography, (warp_width, warp_height))

    roi_value = cfg.get("barcode_roi_norm", [0.48, 0.08, 0.98, 0.92])
    if not isinstance(roi_value, (list, tuple)) or len(roi_value) != 4:
        raise ValueError("task1_staged_top.barcode_roi_norm is invalid")
    x0, y0, x1, y1 = [float(value) for value in roi_value]
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError("task1_staged_top.barcode_roi_norm is invalid")
    left = int(round(x0 * warp_width))
    top = int(round(y0 * warp_height))
    right = int(round(x1 * warp_width))
    bottom = int(round(y1 * warp_height))
    gray = cv2.cvtColor(rectified[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < 8 or gray.shape[1] < 16:
        raise ValueError("task1_staged_top barcode ROI is too small")

    gradient_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gradient_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    edge_percentile = float(cfg.get("barcode_edge_percentile", 80.0))
    if not 50.0 <= edge_percentile <= 95.0:
        raise ValueError("task1_staged_top.barcode_edge_percentile is invalid")
    edge_threshold = max(
        float(cfg.get("barcode_minimum_gradient", 35.0)),
        float(np.percentile(gradient_x, edge_percentile)),
    )
    vertical_support = np.mean(gradient_x >= edge_threshold, axis=0)
    strong_columns = vertical_support >= float(
        cfg.get("barcode_minimum_line_support", 0.18)
    )
    run_widths: list[int] = []
    run_start: int | None = None
    for index, strong in enumerate(np.append(strong_columns, False)):
        if strong and run_start is None:
            run_start = index
        elif not strong and run_start is not None:
            run_widths.append(index - run_start)
            run_start = None
    maximum_run_width = max(1, int(cfg.get("barcode_maximum_stripe_width_px", 8)))
    stripe_widths = [width for width in run_widths if width <= maximum_run_width]
    stripe_count = len(stripe_widths)
    anisotropy = float(
        np.mean(gradient_x) / max(float(np.mean(gradient_y)), 1e-6)
    )
    intensity_contrast = float(
        np.percentile(gray, 90.0) - np.percentile(gray, 10.0)
    )
    minimum_stripes = max(4, int(cfg.get("barcode_minimum_stripe_count", 8)))
    minimum_anisotropy = float(cfg.get("barcode_minimum_anisotropy", 1.10))
    minimum_contrast = float(cfg.get("barcode_minimum_contrast", 35.0))
    valid = bool(
        stripe_count >= minimum_stripes
        and anisotropy >= minimum_anisotropy
        and intensity_contrast >= minimum_contrast
    )
    return {
        "valid": valid,
        "policy": "parallel_dark_stripes_on_rectified_top_right_side",
        "roi_norm": [x0, y0, x1, y1],
        "stripe_count": stripe_count,
        "minimum_stripe_count": minimum_stripes,
        "stripe_widths_px": stripe_widths,
        "gradient_anisotropy_x_over_y": anisotropy,
        "minimum_anisotropy": minimum_anisotropy,
        "intensity_contrast": intensity_contrast,
        "minimum_contrast": minimum_contrast,
        "edge_threshold": edge_threshold,
        "strong_column_fraction": float(np.mean(strong_columns)),
    }


def detect_task1_staged_carton_top_rgbd(
    bgr: np.ndarray,
    depth_z16: np.ndarray,
    depth_scale_m: float,
    intrinsics: np.ndarray,
    cam_to_left: np.ndarray,
    config: dict[str, Any] | None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Locate the narrow 130 x 25 mm top of Task1's staged upright carton.

    Step 4 places one carton in a fixed lower-image station.  Its printed
    vertical side is not a grasp surface.  The horizontal top is isolated by
    converting every synchronized depth pixel to left-base Z, then applying
    the configured station ROI, height, metric-size, colour and position
    gates.  The returned target is the geometric centre of that top plane.
    """

    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Task1 staged-top detection requires uint8 BGR")
    if (
        depth_z16.dtype != np.uint16
        or depth_z16.ndim != 2
        or depth_z16.shape != bgr.shape[:2]
    ):
        raise ValueError("Task1 staged-top depth must match the BGR frame")
    scale = float(depth_scale_m)
    matrix = np.asarray(intrinsics, dtype=np.float64)
    transform = np.asarray(cam_to_left, dtype=np.float64)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Task1 staged-top depth scale is invalid")
    if matrix.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("Task1 staged-top calibration matrices are invalid")

    cfg = dict(config or {})
    height, width = bgr.shape[:2]
    roi_value = cfg.get("roi_norm", [0.30, 0.84, 0.55, 0.99])
    if not isinstance(roi_value, (list, tuple)) or len(roi_value) != 4:
        raise ValueError("task1_staged_top.roi_norm must contain four values")
    x0, y0, x1, y1 = [float(value) for value in roi_value]
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError("task1_staged_top.roi_norm is invalid")
    left = int(round(x0 * width))
    top = int(round(y0 * height))
    right = int(round(x1 * width))
    bottom = int(round(y1 * height))
    if right - left < 16 or bottom - top < 16:
        raise ValueError("task1_staged_top.roi_norm is too small")

    surface_range = cfg.get("surface_z_range_left_base_m", [0.07, 0.115])
    if not isinstance(surface_range, (list, tuple)) or len(surface_range) != 2:
        raise ValueError(
            "task1_staged_top.surface_z_range_left_base_m must contain two values"
        )
    surface_low, surface_high = [float(value) for value in surface_range]
    if not (
        math.isfinite(surface_low)
        and math.isfinite(surface_high)
        and surface_low < surface_high
    ):
        raise ValueError("task1_staged_top surface Z range is invalid")

    roi_depth = depth_z16[top:bottom, left:right]
    rows, columns = np.mgrid[top:bottom, left:right]
    camera_z = roi_depth.astype(np.float64) * scale
    camera_x = (columns - matrix[0, 2]) * camera_z / matrix[0, 0]
    camera_y = (rows - matrix[1, 2]) * camera_z / matrix[1, 1]
    base_x = (
        transform[0, 0] * camera_x
        + transform[0, 1] * camera_y
        + transform[0, 2] * camera_z
        + transform[0, 3]
    )
    base_y = (
        transform[1, 0] * camera_x
        + transform[1, 1] * camera_y
        + transform[1, 2] * camera_z
        + transform[1, 3]
    )
    base_z = (
        transform[2, 0] * camera_x
        + transform[2, 1] * camera_y
        + transform[2, 2] * camera_z
        + transform[2, 3]
    )
    valid_depth = roi_depth > 0
    height_mask = np.where(
        valid_depth & (base_z >= surface_low) & (base_z <= surface_high),
        255,
        0,
    ).astype(np.uint8)
    open_kernel = max(1, int(cfg.get("morphology_open_px", 3)))
    close_kernel = max(1, int(cfg.get("morphology_close_px", 7)))
    if open_kernel > 1:
        height_mask = cv2.morphologyEx(
            height_mask,
            cv2.MORPH_OPEN,
            np.ones((open_kernel, open_kernel), dtype=np.uint8),
        )
    if close_kernel > 1:
        height_mask = cv2.morphologyEx(
            height_mask,
            cv2.MORPH_CLOSE,
            np.ones((close_kernel, close_kernel), dtype=np.uint8),
        )

    expected_size = cfg.get("expected_top_size_mm", [130.0, 25.0])
    tolerance = cfg.get("top_size_tolerance_mm", [20.0, 8.0])
    if (
        not isinstance(expected_size, (list, tuple))
        or len(expected_size) != 2
        or not isinstance(tolerance, (list, tuple))
        or len(tolerance) != 2
    ):
        raise ValueError("Task1 staged-top size settings are invalid")
    expected_long, expected_short = [float(value) / 1000.0 for value in expected_size]
    tolerance_long, tolerance_short = [float(value) / 1000.0 for value in tolerance]
    minimum_area = float(cfg.get("minimum_area_px", 900.0))
    maximum_area = float(cfg.get("maximum_area_px", 7000.0))
    minimum_fill = float(cfg.get("minimum_rectangularity", 0.50))
    minimum_pink = float(cfg.get("minimum_pink_fraction", 0.04))
    maximum_height_spread = float(cfg.get("maximum_surface_z_spread_m", 0.025))
    expected_center = cfg.get("expected_center_norm", [0.405, 0.92])
    maximum_center_distance = float(
        cfg.get("maximum_center_distance_norm", 0.12)
    )
    if not isinstance(expected_center, (list, tuple)) or len(expected_center) != 2:
        raise ValueError("task1_staged_top.expected_center_norm is invalid")
    expected_center_px = np.asarray(
        [float(expected_center[0]) * width, float(expected_center[1]) * height],
        dtype=np.float64,
    )
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    pink_lower = np.asarray(cfg.get("pink_hsv_lower", [130, 8, 100]), np.uint8)
    pink_upper = np.asarray(cfg.get("pink_hsv_upper", [179, 180, 255]), np.uint8)
    if pink_lower.shape != (3,) or pink_upper.shape != (3,):
        raise ValueError("Task1 staged-top pink HSV bounds are invalid")

    contours, _ = cv2.findContours(
        height_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    candidates: list[dict[str, Any]] = []
    valid_candidates: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0.0:
            continue
        image_rect = cv2.minAreaRect(contour)
        rect_width, rect_height = [float(value) for value in image_rect[1]]
        if rect_width >= rect_height:
            image_long_px = rect_width
            image_short_px = rect_height
            image_long_axis_angle_deg = float(image_rect[2])
        else:
            image_long_px = rect_height
            image_short_px = rect_width
            image_long_axis_angle_deg = float(image_rect[2]) + 90.0
        image_rect_area = rect_width * rect_height
        rectangularity = area / max(image_rect_area, 1.0)
        component = np.zeros_like(height_mask)
        cv2.drawContours(component, [contour], -1, 255, thickness=-1)
        selected = (component > 0) & (height_mask > 0) & valid_depth
        selected_count = int(np.count_nonzero(selected))
        blockers: list[str] = []
        if not minimum_area <= area <= maximum_area:
            blockers.append("staged_top_area_mismatch")
        if rectangularity < minimum_fill:
            blockers.append("staged_top_rectangularity_low")
        if selected_count < int(cfg.get("minimum_depth_samples", 500)):
            blockers.append("staged_top_depth_samples_low")

        physical_long = physical_short = surface_z = depth_m = None
        base_center_xy: tuple[float, float] | None = None
        surface_spread = None
        if selected_count:
            base_points = np.column_stack(
                (base_x[selected], base_y[selected])
            ).astype(np.float32)
            base_rect = cv2.minAreaRect(base_points)
            physical_long, physical_short = sorted(
                (float(base_rect[1][0]), float(base_rect[1][1])),
                reverse=True,
            )
            base_center_xy = (
                float(base_rect[0][0]),
                float(base_rect[0][1]),
            )
            surface_values = base_z[selected]
            surface_z = float(np.median(surface_values))
            depth_m = float(np.median(camera_z[selected]))
            low_z, high_z = np.quantile(surface_values, [0.05, 0.95])
            surface_spread = float(high_z - low_z)
            if abs(physical_long - expected_long) > tolerance_long:
                blockers.append("staged_top_long_size_mismatch")
            if abs(physical_short - expected_short) > tolerance_short:
                blockers.append("staged_top_short_size_mismatch")
            if surface_spread > maximum_height_spread:
                blockers.append("staged_top_surface_not_planar")

        polygon = cv2.boxPoints(image_rect)
        polygon[:, 0] += float(left)
        polygon[:, 1] += float(top)
        center_px = np.asarray(
            [float(image_rect[0][0] + left), float(image_rect[0][1] + top)],
            dtype=np.float64,
        )
        center_distance_norm = float(
            np.linalg.norm(
                (center_px - expected_center_px)
                / np.asarray([width, height], dtype=np.float64)
            )
        )
        if center_distance_norm > maximum_center_distance:
            blockers.append("staged_top_station_position_mismatch")
        barcode_evidence = measure_task1_top_barcode_stripes(
            bgr,
            polygon,
            cfg,
        )
        if barcode_evidence["valid"] is not True:
            blockers.append("staged_top_barcode_not_found")
        full_component = np.zeros((height, width), dtype=np.uint8)
        full_component[top:bottom, left:right] = component
        pixels = hsv[full_component > 0]
        pink_fraction = (
            0.0
            if not pixels.size
            else float(
                np.mean(
                    np.all(pixels >= pink_lower, axis=1)
                    & np.all(pixels <= pink_upper, axis=1)
                )
            )
        )
        if pink_fraction < minimum_pink:
            blockers.append("staged_top_pink_material_low")
        blockers = list(dict.fromkeys(blockers))
        size_error = (
            math.inf
            if physical_long is None or physical_short is None
            else (
                abs(physical_long - expected_long) / max(tolerance_long, 1e-6)
                + abs(physical_short - expected_short) / max(tolerance_short, 1e-6)
            )
        )
        candidate = {
            "center_px": center_px.tolist(),
            "polygon_px": polygon.tolist(),
            "point_left_base_m": (
                None
                if base_center_xy is None or surface_z is None
                else [base_center_xy[0], base_center_xy[1], surface_z]
            ),
            "median_depth_m": depth_m,
            "physical_size_m": (
                None
                if physical_long is None or physical_short is None
                else [physical_long, physical_short]
            ),
            "image_size_px": [image_long_px, image_short_px],
            "image_long_axis_angle_deg": image_long_axis_angle_deg,
            "surface_z_spread_m": surface_spread,
            "area_px": area,
            "rectangularity": rectangularity,
            "pink_fraction": pink_fraction,
            "barcode_evidence": barcode_evidence,
            "center_distance_norm": center_distance_norm,
            "depth_sample_count": selected_count,
            "score": float(
                1.0
                / (
                    1.0
                    + max(0.0, size_error)
                    + 2.0 * center_distance_norm
                )
            ),
            "valid": not blockers,
            "blockers": blockers,
        }
        candidates.append(candidate)
        if not blockers:
            valid_candidates.append(candidate)

    candidates.sort(key=lambda item: (item["valid"], item["score"]), reverse=True)
    valid_candidates.sort(key=lambda item: item["score"], reverse=True)
    selected_candidate = valid_candidates[0] if valid_candidates else None
    fixed_station_fallback_used = False
    if cfg.get("fixed_station_partial_surface_fallback_enabled") is True:
        # At the calibrated Task1 hand-off station the right tool hides part of
        # the 130 x 25 mm face.  The height mask can therefore split the face
        # into two pieces or merge one piece with the tool.  Validate material,
        # depth, height and station position on the visible anchor, then return
        # the configured full-face centre instead of the visible-piece centre.
        anchor_min_area = float(cfg.get("fixed_station_anchor_minimum_area_px", 700.0))
        anchor_max_area = float(cfg.get("fixed_station_anchor_maximum_area_px", 7000.0))
        anchor_min_depth = int(cfg.get("fixed_station_anchor_minimum_depth_samples", 700))
        anchor_min_pink = float(cfg.get("fixed_station_anchor_minimum_pink_fraction", 0.05))
        anchor_max_spread = float(cfg.get("fixed_station_anchor_maximum_z_spread_m", 0.04))
        anchor_max_distance = float(cfg.get("fixed_station_anchor_maximum_center_distance_norm", 0.065))
        anchor_long_range = cfg.get("fixed_station_anchor_long_size_range_mm", [40.0, 180.0])
        anchor_short_range = cfg.get("fixed_station_anchor_short_size_range_mm", [15.0, 80.0])
        if (
            not isinstance(anchor_long_range, (list, tuple))
            or len(anchor_long_range) != 2
            or not isinstance(anchor_short_range, (list, tuple))
            or len(anchor_short_range) != 2
        ):
            raise ValueError("Task1 staged-top fixed-station anchor size ranges are invalid")
        anchor_long_low, anchor_long_high = [float(value) / 1000.0 for value in anchor_long_range]
        anchor_short_low, anchor_short_high = [float(value) / 1000.0 for value in anchor_short_range]
        anchors: list[dict[str, Any]] = []
        for observed in candidates:
            physical_size = observed.get("physical_size_m")
            point = observed.get("point_left_base_m")
            spread = observed.get("surface_z_spread_m")
            if (
                not isinstance(physical_size, list)
                or len(physical_size) != 2
                or not isinstance(point, list)
                or len(point) != 3
                or spread is None
            ):
                continue
            observed_long, observed_short = [float(value) for value in physical_size]
            if not (
                anchor_min_area <= float(observed["area_px"]) <= anchor_max_area
                and int(observed["depth_sample_count"]) >= anchor_min_depth
                and float(observed["pink_fraction"]) >= anchor_min_pink
                and float(spread) <= anchor_max_spread
                and float(observed["center_distance_norm"]) <= anchor_max_distance
                and anchor_long_low <= observed_long <= anchor_long_high
                and anchor_short_low <= observed_short <= anchor_short_high
            ):
                continue
            anchor_score = (
                4.0 * float(observed["center_distance_norm"])
                + abs(observed_short - expected_short)
                + 0.25 * float(spread)
            )
            anchors.append({"score": anchor_score, "candidate": observed})

        if anchors:
            anchors.sort(key=lambda item: item["score"])
            observed = anchors[0]["candidate"]
            observed_surface_z = float(observed["point_left_base_m"][2])
            configured_surface_z = cfg.get("fixed_surface_z_left_base_m")
            surface_z = (
                observed_surface_z
                if configured_surface_z is None
                else float(configured_surface_z)
            )
            fixed_surface_z_tolerance = float(
                cfg.get("fixed_surface_z_tolerance_m", 0.015)
            )
            if not (
                math.isfinite(surface_z)
                and surface_low <= surface_z <= surface_high
                and math.isfinite(fixed_surface_z_tolerance)
                and 0.003 <= fixed_surface_z_tolerance <= 0.03
                and abs(observed_surface_z - surface_z)
                <= fixed_surface_z_tolerance
            ):
                anchors = []
        if anchors:
            observed = anchors[0]["candidate"]
            observed_surface_z = float(observed["point_left_base_m"][2])
            configured_surface_z = cfg.get("fixed_surface_z_left_base_m")
            surface_z = (
                observed_surface_z
                if configured_surface_z is None
                else float(configured_surface_z)
            )
            ray_camera = np.asarray(
                [
                    (expected_center_px[0] - matrix[0, 2]) / matrix[0, 0],
                    (expected_center_px[1] - matrix[1, 2]) / matrix[1, 1],
                    1.0,
                ],
                dtype=np.float64,
            )
            ray_left = transform[:3, :3] @ ray_camera
            camera_origin_left = transform[:3, 3]
            if abs(float(ray_left[2])) > 1e-8:
                ray_distance = (surface_z - float(camera_origin_left[2])) / float(ray_left[2])
            else:
                ray_distance = math.nan
            target_point = camera_origin_left + ray_distance * ray_left
            observed_image_size = observed.get("image_size_px", [0.0, 0.0])
            observed_physical_size = observed["physical_size_m"]
            full_long_px = (
                float(observed_image_size[0])
                * expected_long
                / max(float(observed_physical_size[0]), 1e-6)
            )
            full_short_px = (
                float(observed_image_size[1])
                * expected_short
                / max(float(observed_physical_size[1]), 1e-6)
            )
            long_axis_angle = float(
                cfg.get(
                    "fixed_surface_long_axis_angle_deg",
                    observed.get("image_long_axis_angle_deg", 0.0),
                )
            )
            if (
                math.isfinite(ray_distance)
                and ray_distance > 0.0
                and np.all(np.isfinite(target_point))
                and 20.0 <= full_long_px <= float(width)
                and 8.0 <= full_short_px <= float(height)
            ):
                full_polygon = cv2.boxPoints(
                    (
                        (float(expected_center_px[0]), float(expected_center_px[1])),
                        (float(full_long_px), float(full_short_px)),
                        long_axis_angle,
                    )
                )
                selected_candidate = copy.deepcopy(observed)
                selected_candidate.update(
                    {
                        "center_px": expected_center_px.tolist(),
                        "point_left_base_m": target_point.tolist(),
                        "observed_partial_polygon_px": copy.deepcopy(observed["polygon_px"]),
                        "polygon_px": full_polygon.tolist(),
                        "physical_size_m": [expected_long, expected_short],
                        "image_size_px": [full_long_px, full_short_px],
                        "image_long_axis_angle_deg": long_axis_angle,
                        "valid": True,
                        "blockers": [],
                        "fixed_station_partial_surface_fallback": {
                            "used": True,
                            "policy": "visible_rgbd_anchor_plus_calibrated_full_face_center",
                            "observed_center_px": copy.deepcopy(observed["center_px"]),
                            "calibrated_center_px": expected_center_px.tolist(),
                            "observed_physical_size_m": copy.deepcopy(observed_physical_size),
                            "observed_surface_z_left_base_m": observed_surface_z,
                            "target_surface_z_left_base_m": surface_z,
                            "fixed_surface_z_tolerance_m": fixed_surface_z_tolerance,
                            "required_pink_fraction": anchor_min_pink,
                            "required_maximum_z_spread_m": anchor_max_spread,
                        },
                    }
                )
                fixed_station_fallback_used = True
    blockers = [] if selected_candidate is not None else [
        "no_staged_carton_top_passed_rgbd_geometry"
    ]
    if not contours:
        blockers.insert(0, "no_surface_in_staged_top_height_band")
    detection = {
        "type": "task1_staged_upright_carton_top",
        "task_id": "task1",
        "target_ready": selected_candidate is not None,
        "geometry_source": (
            "fixed_station_visible_rgbd_anchor_and_calibrated_full_face_center"
            if fixed_station_fallback_used
            else "fixed_station_roi_left_base_height_metric_130x25_and_top_barcode"
        ),
        "selection_policy": (
            "validated_visible_anchor_then_fixed_full_face_center"
            if fixed_station_fallback_used
            else "best_barcode_top_near_fixed_station"
        ),
        "roi_norm": [x0, y0, x1, y1],
        "surface_z_range_left_base_m": [surface_low, surface_high],
        "count": len(valid_candidates),
        "candidate": copy.deepcopy(selected_candidate),
        "candidates": copy.deepcopy(candidates),
        "point_left_base_m": (
            None
            if selected_candidate is None
            else copy.deepcopy(selected_candidate["point_left_base_m"])
        ),
        "blockers": blockers,
    }

    overlay = bgr.copy()
    cv2.rectangle(overlay, (left, top), (right - 1, bottom - 1), (0, 220, 255), 2)
    for candidate in candidates:
        colour = (40, 210, 40) if candidate["valid"] else (0, 90, 255)
        polygon_i = np.rint(candidate["polygon_px"]).astype(np.int32)
        cv2.polylines(overlay, [polygon_i], True, colour, 2, cv2.LINE_AA)
        center_i = tuple(np.rint(candidate["center_px"]).astype(int))
        cv2.circle(overlay, center_i, 5, colour, -1, cv2.LINE_AA)
    if fixed_station_fallback_used and selected_candidate is not None:
        reconstructed_polygon = np.rint(
            selected_candidate["polygon_px"]
        ).astype(np.int32)
        reconstructed_center = tuple(
            np.rint(selected_candidate["center_px"]).astype(int)
        )
        cv2.polylines(
            overlay,
            [reconstructed_polygon],
            True,
            (40, 210, 40),
            3,
            cv2.LINE_AA,
        )
        cv2.drawMarker(
            overlay,
            reconstructed_center,
            (40, 210, 40),
            cv2.MARKER_CROSS,
            18,
            3,
            cv2.LINE_AA,
        )
    label = "TASK1 STAGED TOP READY" if selected_candidate else "TASK1 STAGED TOP NOT READY"
    cv2.putText(
        overlay,
        label,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (40, 210, 40) if selected_candidate else (0, 90, 255),
        2,
        cv2.LINE_AA,
    )
    return detection, overlay


def locate_open_shipping_box_rgbd(
    bgr: np.ndarray,
    depth_z16: np.ndarray | None,
    depth_scale_m: float | None,
    intrinsics: Any,
    cam_to_left: np.ndarray | None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Locate the open Task2 shipping box without commanding either arm.

    The colour mask proposes the outer cardboard footprint.  Four depth probes
    on the box rim are then transformed independently into the left-arm base
    frame.  Averaging those 3-D rim points avoids the several-centimetre error
    produced by deprojecting the image centre at the deeper box-bottom depth.
    Workflows that only release above the opening can set
    ``require_cavity_depth`` false so contents do not hide the target.
    """

    cfg = dict(config or {})
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("shipping-box detection requires one uint8 BGR frame")
    height, width = bgr.shape[:2]
    roi_norm = cfg.get("roi_norm", [0.30, 0.0, 0.68, 0.45])
    if not isinstance(roi_norm, (list, tuple)) or len(roi_norm) != 4:
        raise ValueError("task2_workflow.shipping_box_detection.roi_norm is invalid")
    x0, y0, x1, y1 = (float(value) for value in roi_norm)
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError("shipping-box ROI must be normalized inside the image")
    left = max(0, min(width - 1, int(math.floor(x0 * width))))
    top = max(0, min(height - 1, int(math.floor(y0 * height))))
    right = max(left + 1, min(width, int(math.ceil(x1 * width))))
    bottom = max(top + 1, min(height, int(math.ceil(y1 * height))))
    overlay = bgr.copy()
    cv2.rectangle(overlay, (left, top), (right - 1, bottom - 1), (255, 160, 40), 2)

    crop = bgr[top:bottom, left:right]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hsv_lower = np.asarray(cfg.get("hsv_lower", [5, 45, 45]), dtype=np.uint8)
    hsv_upper = np.asarray(cfg.get("hsv_upper", [18, 230, 175]), dtype=np.uint8)
    if hsv_lower.shape != (3,) or hsv_upper.shape != (3,):
        raise ValueError("shipping-box HSV bounds must contain three values")
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    kernel_px = max(1, min(31, int(cfg.get("morphology_kernel_px", 5))))
    if kernel_px % 2 == 0:
        kernel_px += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_px, kernel_px),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    roi_area = float((right - left) * (bottom - top))
    minimum_area_fraction = float(cfg.get("minimum_area_fraction", 0.12))
    maximum_area_fraction = float(cfg.get("maximum_area_fraction", 0.85))
    minimum_rectangularity = float(cfg.get("minimum_rectangularity", 0.58))
    minimum_aspect_ratio = float(cfg.get("minimum_aspect_ratio", 1.0))
    maximum_aspect_ratio = float(cfg.get("maximum_aspect_ratio", 1.8))
    geometry_candidates: list[
        tuple[float, np.ndarray, tuple[Any, Any, Any], float, float]
    ] = []
    for local_contour in contours:
        contour = local_contour + np.asarray([[[left, top]]], dtype=local_contour.dtype)
        area_px = float(cv2.contourArea(contour))
        area_fraction = area_px / max(1.0, roi_area)
        if not minimum_area_fraction <= area_fraction <= maximum_area_fraction:
            continue
        rect = cv2.minAreaRect(contour)
        rect_width, rect_height = (float(value) for value in rect[1])
        if min(rect_width, rect_height) < 40.0:
            continue
        aspect_ratio = max(rect_width, rect_height) / max(
            1.0,
            min(rect_width, rect_height),
        )
        rectangularity = area_px / max(1.0, rect_width * rect_height)
        if (
            not minimum_aspect_ratio <= aspect_ratio <= maximum_aspect_ratio
            or rectangularity < minimum_rectangularity
        ):
            continue
        geometry_score = min(
            1.0,
            0.55 * rectangularity
            + 0.30 * min(1.0, area_fraction / 0.30)
            + 0.15
            * max(
                0.0,
                1.0
                - abs(aspect_ratio - float(cfg.get("expected_aspect_ratio", 1.2)))
                / 0.8,
            ),
        )
        geometry_candidates.append(
            (geometry_score, contour, rect, area_px, rectangularity)
        )

    base_result: dict[str, Any] = {
        "type": "task2_shipping_box_opening",
        "detected_2d": False,
        "target_ready": False,
        "count": 0,
        "candidate": None,
        "candidates": [],
        "point_left_base_m": None,
        "blockers": [],
        "roi_norm": [x0, y0, x1, y1],
    }
    if not geometry_candidates:
        base_result["blockers"] = ["shipping_box_not_found"]
        cv2.putText(
            overlay,
            "SHIPPING BOX NOT FOUND",
            (left + 8, max(24, top + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (40, 60, 255),
            2,
            cv2.LINE_AA,
        )
        return base_result, overlay

    score, contour, rect, area_px, rectangularity = max(
        geometry_candidates,
        key=lambda item: (item[0], item[3]),
    )
    center = np.asarray(rect[0], dtype=np.float64)
    rect_width, rect_height = (float(value) for value in rect[1])
    theta = math.radians(float(rect[2]))
    width_axis = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float64)
    height_axis = np.asarray([-math.sin(theta), math.cos(theta)], dtype=np.float64)
    if rect_width >= rect_height:
        long_axis, long_size_px = width_axis, rect_width
        short_axis, short_size_px = height_axis, rect_height
    else:
        long_axis, long_size_px = height_axis, rect_height
        short_axis, short_size_px = width_axis, rect_width
    if long_axis[0] < 0.0:
        long_axis = -long_axis
    if short_axis[1] < 0.0:
        short_axis = -short_axis

    center_short_offset = float(cfg.get("opening_center_short_offset_fraction", 0.05))
    opening_center_px = center + short_axis * short_size_px * center_short_offset
    opening_long_scale = float(cfg.get("opening_long_scale", 0.74))
    opening_short_scale = float(cfg.get("opening_short_scale", 0.72))
    opening_long_px = long_size_px * opening_long_scale
    opening_short_px = short_size_px * opening_short_scale
    opening_polygon = np.asarray(
        [
            opening_center_px - long_axis * opening_long_px / 2.0 - short_axis * opening_short_px / 2.0,
            opening_center_px + long_axis * opening_long_px / 2.0 - short_axis * opening_short_px / 2.0,
            opening_center_px + long_axis * opening_long_px / 2.0 + short_axis * opening_short_px / 2.0,
            opening_center_px - long_axis * opening_long_px / 2.0 + short_axis * opening_short_px / 2.0,
        ],
        dtype=np.float64,
    )
    rim_long_scale = float(cfg.get("rim_probe_long_scale", 0.90))
    rim_short_scale = float(cfg.get("rim_probe_short_scale", 0.72))
    rim_pixels = np.asarray(
        [
            opening_center_px - long_axis * long_size_px * rim_long_scale / 2.0,
            opening_center_px + long_axis * long_size_px * rim_long_scale / 2.0,
            opening_center_px - short_axis * short_size_px * rim_short_scale / 2.0,
            opening_center_px + short_axis * short_size_px * rim_short_scale / 2.0,
        ],
        dtype=np.float64,
    )
    outer_polygon = cv2.boxPoints(rect).astype(np.float64)
    candidate: dict[str, Any] = {
        "center_px": [float(value) for value in opening_center_px],
        "suction_px": [int(round(value)) for value in opening_center_px],
        "polygon_px": opening_polygon.tolist(),
        "outer_polygon_px": outer_polygon.tolist(),
        "rim_probe_pixels_px": rim_pixels.tolist(),
        "long_side_px": float(opening_long_px),
        "short_side_px": float(opening_short_px),
        "score": float(score),
        "rectangularity": float(rectangularity),
        "area_px": float(area_px),
        "point_left_base_m": None,
    }
    base_result.update(
        {
            "detected_2d": True,
            "count": 1,
            "candidate": candidate,
            "candidates": [candidate],
            "score": float(score),
        }
    )

    blockers: list[str] = []
    require_cavity_depth = bool(cfg.get("require_cavity_depth", True))
    fixed_rim_z_raw = cfg.get("fixed_rim_z_left_base_m")
    fixed_rim_z_m: float | None = None
    if fixed_rim_z_raw is not None:
        fixed_rim_z_m = float(fixed_rim_z_raw)
        if not math.isfinite(fixed_rim_z_m) or not -0.10 <= fixed_rim_z_m <= 0.40:
            raise ValueError("shipping-box fixed rim Z must be -0.10..0.40 m")
    if depth_z16 is None or depth_scale_m is None:
        blockers.append("shipping_box_depth_unavailable")
    elif (
        depth_z16.ndim != 2
        or depth_z16.shape != (height, width)
        or not math.isfinite(float(depth_scale_m))
        or float(depth_scale_m) <= 0.0
    ):
        blockers.append("shipping_box_depth_invalid")
    matrix = None
    try:
        matrix = np.asarray(intrinsics, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError
    except (TypeError, ValueError):
        blockers.append("shipping_box_intrinsics_unavailable")
        matrix = None
    transform = None
    try:
        transform = np.asarray(cam_to_left, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError
    except (TypeError, ValueError):
        blockers.append("shipping_box_handeye_unavailable")
        transform = None

    rim_points_base: list[np.ndarray] = []
    rim_depths_m: list[float] = []
    rim_valid_ratios: list[float] = []
    bottom_point_base: np.ndarray | None = None
    bottom_depth_m: float | None = None
    bottom_valid_ratio = 0.0
    minimum_valid_ratio = float(cfg.get("minimum_depth_valid_ratio", 0.65))
    if not blockers and matrix is not None and transform is not None:
        depth_m = depth_z16.astype(np.float64) * float(depth_scale_m)
        minimum_depth_m = float(cfg.get("minimum_depth_m", 0.30))
        maximum_depth_m = float(cfg.get("maximum_depth_m", 1.50))
        rim_quantile = float(cfg.get("rim_depth_quantile", 0.20))
        region_depth = shipping_box_region_depth_statistics(
            depth_m,
            opening_polygon,
            matrix,
            transform,
            minimum_depth_m=minimum_depth_m,
            maximum_depth_m=maximum_depth_m,
            bottom_region_scale=float(cfg.get("bottom_region_scale", 0.55)),
            rim_height_quantile=float(
                cfg.get("rim_height_quantile", 0.80)
            ),
            bottom_height_quantile=float(
                cfg.get("bottom_height_quantile", 0.20)
            ),
        )
        rim_depth_value = region_depth["rim_depth_m"]
        rim_valid_ratio = float(region_depth["rim_valid_ratio"] or 0.0)
        rim_valid_ratios = [rim_valid_ratio] * 4
        if rim_depth_value is not None and rim_valid_ratio >= minimum_valid_ratio:
            for pixel in rim_pixels:
                depth_value = float(rim_depth_value)
                point_camera = deproject_pixel(
                    (int(round(pixel[0])), int(round(pixel[1]))),
                    depth_value * 1000.0,
                    matrix,
                )
                rim_depths_m.append(depth_value)
                rim_points_base.append(transform_point(point_camera, transform))
        bottom_depth_m = region_depth["bottom_depth_m"]
        bottom_valid_ratio = float(region_depth["bottom_valid_ratio"] or 0.0)
        if fixed_rim_z_m is None and len(rim_points_base) != 4:
            blockers.append("shipping_box_rim_depth_incomplete")
        if (
            require_cavity_depth
            and (
                bottom_depth_m is None
                or bottom_valid_ratio < minimum_valid_ratio
            )
        ):
            blockers.append("shipping_box_bottom_depth_invalid")
        elif region_depth.get("bottom_point_left_base_m") is not None:
            bottom_point_base = np.asarray(
                region_depth["bottom_point_left_base_m"], dtype=np.float64
            )

    opening_center_base: np.ndarray | None = None
    rim_z_m: float | None = None
    rim_plane_depth_m: float | None = None
    cavity_depth_m: float | None = None
    opening_size_m: list[float] | None = None
    yaw_deg: float | None = None
    if (
        (fixed_rim_z_m is not None or len(rim_points_base) == 4)
        and matrix is not None
        and transform is not None
    ):
        if fixed_rim_z_m is not None:
            try:
                opening_center_base = project_pixel_to_base_z_plane(
                    opening_center_px,
                    fixed_rim_z_m,
                    matrix,
                    transform,
                )
                rim_z_m = fixed_rim_z_m
                center_camera = transform_point(
                    opening_center_base,
                    np.linalg.inv(transform),
                )
                rim_plane_depth_m = float(center_camera[2])
            except (TypeError, ValueError, np.linalg.LinAlgError):
                blockers.append("shipping_box_rim_plane_invalid")
        else:
            rim_plane_depth_m = shipping_box_rim_plane_depth_m(rim_depths_m)
            if rim_plane_depth_m is None:
                blockers.append("shipping_box_rim_depth_incomplete")
            else:
                center_camera = deproject_pixel(
                    (
                        int(round(opening_center_px[0])),
                        int(round(opening_center_px[1])),
                    ),
                    rim_plane_depth_m * 1000.0,
                    matrix,
                )
                opening_center_base = transform_point(center_camera, transform)
                rim_z_m = float(region_depth.get("rim_z_m", opening_center_base[2]))
                opening_center_base[2] = rim_z_m
        if opening_center_base is None or rim_z_m is None:
            blockers.append("shipping_box_rim_plane_invalid")
        else:
            long_pixels = (
                opening_center_px - long_axis * opening_long_px / 2.0,
                opening_center_px + long_axis * opening_long_px / 2.0,
            )
            short_pixels = (
                opening_center_px - short_axis * opening_short_px / 2.0,
                opening_center_px + short_axis * opening_short_px / 2.0,
            )
            if fixed_rim_z_m is not None:
                long_points = [
                    project_pixel_to_base_z_plane(
                        pixel,
                        fixed_rim_z_m,
                        matrix,
                        transform,
                    )
                    for pixel in long_pixels
                ]
                short_points = [
                    project_pixel_to_base_z_plane(
                        pixel,
                        fixed_rim_z_m,
                        matrix,
                        transform,
                    )
                    for pixel in short_pixels
                ]
            else:
                long_points = [
                    transform_point(
                        deproject_pixel(
                            (int(round(pixel[0])), int(round(pixel[1]))),
                            rim_plane_depth_m * 1000.0,
                            matrix,
                        ),
                        transform,
                    )
                    for pixel in long_pixels
                ]
                short_points = [
                    transform_point(
                        deproject_pixel(
                            (int(round(pixel[0])), int(round(pixel[1]))),
                            rim_plane_depth_m * 1000.0,
                            matrix,
                        ),
                        transform,
                    )
                    for pixel in short_pixels
                ]
            if bottom_point_base is not None:
                cavity_depth_m = rim_z_m - float(bottom_point_base[2])
            long_span = float(
                np.linalg.norm(long_points[1][:2] - long_points[0][:2])
            )
            short_span = float(
                np.linalg.norm(short_points[1][:2] - short_points[0][:2])
            )
            opening_size_m = [long_span, short_span]
            long_vector = long_points[1][:2] - long_points[0][:2]
            yaw_deg = math.degrees(math.atan2(long_vector[1], long_vector[0]))
        if yaw_deg is not None:
            while yaw_deg >= 90.0:
                yaw_deg -= 180.0
            while yaw_deg < -90.0:
                yaw_deg += 180.0
        if opening_size_m is None:
            blockers.append("shipping_box_rim_plane_invalid")
        else:
            if require_cavity_depth:
                minimum_cavity_m = float(
                    cfg.get("minimum_cavity_depth_m", 0.025)
                )
                maximum_cavity_m = float(
                    cfg.get("maximum_cavity_depth_m", 0.20)
                )
                if (
                    cavity_depth_m is None
                    or not minimum_cavity_m
                    <= cavity_depth_m
                    <= maximum_cavity_m
                ):
                    blockers.append("shipping_box_cavity_depth_invalid")
            minimum_opening_m = float(cfg.get("minimum_opening_size_m", 0.16))
            maximum_opening_m = float(cfg.get("maximum_opening_size_m", 0.40))
            if any(
                not minimum_opening_m <= value <= maximum_opening_m
                for value in opening_size_m
            ):
                blockers.append("shipping_box_opening_size_invalid")
        workspace = cfg.get("left_base_workspace", {})
        if isinstance(workspace, dict):
            axes = {"x": 0, "y": 1, "z": 2}
            for axis, index in axes.items():
                minimum = float(workspace.get(f"{axis}_min", -math.inf))
                maximum = float(workspace.get(f"{axis}_max", math.inf))
                if not minimum <= float(opening_center_base[index]) <= maximum:
                    blockers.append("shipping_box_outside_configured_workspace")
                    break

    minimum_score = float(cfg.get("minimum_score", 0.65))
    if score < minimum_score:
        blockers.append("shipping_box_score_low")
    blockers = list(dict.fromkeys(blockers))
    candidate.update(
        {
            "point_left_base_m": (
                None
                if opening_center_base is None
                else [float(value) for value in opening_center_base]
            ),
            "opening_center_left_base_m": (
                None
                if opening_center_base is None
                else [float(value) for value in opening_center_base]
            ),
            "bottom_point_left_base_m": (
                None
                if bottom_point_base is None
                else [float(value) for value in bottom_point_base]
            ),
            "rim_points_left_base_m": [
                [float(value) for value in point] for point in rim_points_base
            ],
            "opening_size_m": opening_size_m,
            "yaw_left_base_deg": yaw_deg,
            "rim_z_m": rim_z_m,
            "rim_plane_depth_m": rim_plane_depth_m,
            "bottom_z_m": (
                None if bottom_point_base is None else float(bottom_point_base[2])
            ),
            "cavity_depth_m": cavity_depth_m,
            "rim_depths_m": rim_depths_m,
            "rim_depth_valid_ratios": rim_valid_ratios,
            "bottom_depth_m": bottom_depth_m,
            "bottom_depth_valid_ratio": bottom_valid_ratio,
            "cavity_depth_required": require_cavity_depth,
            "rim_height_source": (
                "configured_base_plane"
                if fixed_rim_z_m is not None
                else "rgbd_region"
            ),
        }
    )
    base_result.update(
        {
            "target_ready": not blockers and opening_center_base is not None,
            "point_left_base_m": candidate["point_left_base_m"],
            "opening_center_left_base_m": candidate["opening_center_left_base_m"],
            "opening_size_m": opening_size_m,
            "yaw_left_base_deg": yaw_deg,
            "rim_z_m": rim_z_m,
            "bottom_z_m": candidate["bottom_z_m"],
            "cavity_depth_m": cavity_depth_m,
            "cavity_depth_required": require_cavity_depth,
            "rim_height_source": candidate["rim_height_source"],
            "blockers": blockers,
        }
    )

    ready = base_result["target_ready"] is True
    colour = (93, 242, 179) if ready else (40, 80, 255)
    cv2.polylines(
        overlay,
        [np.rint(outer_polygon).astype(np.int32)],
        True,
        (0, 215, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.polylines(
        overlay,
        [np.rint(opening_polygon).astype(np.int32)],
        True,
        colour,
        3,
        cv2.LINE_AA,
    )
    for pixel in rim_pixels:
        cv2.circle(
            overlay,
            tuple(np.rint(pixel).astype(int)),
            7,
            colour,
            2,
            cv2.LINE_AA,
        )
    cv2.drawMarker(
        overlay,
        tuple(np.rint(opening_center_px).astype(int)),
        colour,
        markerType=cv2.MARKER_CROSS,
        markerSize=28,
        thickness=3,
    )
    label = "SHIPPING BOX READY" if ready else "SHIPPING BOX BLOCKED"
    if opening_center_base is not None:
        label += (
            f"  X {opening_center_base[0]:.3f}"
            f"  Y {opening_center_base[1]:.3f}"
            f"  Z {opening_center_base[2]:.3f}"
        )
    cv2.putText(
        overlay,
        label,
        (left + 8, max(24, top + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        colour,
        2,
        cv2.LINE_AA,
    )
    return base_result, overlay


def cluster_carton_instances(candidates: list[Any]) -> list[list[Any]]:
    """Group neighbouring single-carton faces into disconnected layouts."""

    remaining = list(candidates)
    groups: list[list[Any]] = []
    while remaining:
        group = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            for candidate in list(remaining):
                center = np.asarray(candidate.center_px, dtype=np.float64)
                connected = any(
                    float(
                        np.linalg.norm(
                            center
                            - np.asarray(member.center_px, dtype=np.float64)
                        )
                    )
                    <= 1.75
                    * max(
                        float(candidate.long_side_px),
                        float(member.long_side_px),
                    )
                    for member in group
                )
                if connected:
                    group.append(candidate)
                    remaining.remove(candidate)
                    changed = True
        groups.append(group)
    groups.sort(
        key=lambda group: (
            -len(group),
            min(float(item.center_px[0]) for item in group),
        )
    )
    return groups


def build_task2_detector_config(
    detector_config: dict[str, Any],
    task_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a front-face-only detector profile for the Task2 station.

    Task2 may only pick the printed bear front.  Reuse the deterministic motif
    matcher, but constrain it to the task ROI and discard templates for every
    other face.  The normal SIFT fallback is restricted to the same face type.
    """

    task2_config = copy.deepcopy(detector_config)
    if str(task2_config.get("provider", "")) != "reference_feature":
        return task2_config
    provider_options = dict(task2_config.get("provider_options", {}))
    front_motifs = [
        copy.deepcopy(setting)
        for setting in provider_options.get("motif_templates", [])
        if isinstance(setting, dict)
        and str(setting.get("face_type", "")) == "front_large"
    ]
    profile = task_profile if isinstance(task_profile, dict) else {}
    remote_recovery = profile.get("rgbd_recovery_remote")
    if isinstance(remote_recovery, dict):
        task2_config["rgbd_recovery_remote"] = copy.deepcopy(remote_recovery)
    vertical_motifs = [
        setting
        for setting in front_motifs
        if "vertical" in str(setting.get("image", "")).lower()
    ]
    selected_motifs = vertical_motifs or front_motifs
    for setting in selected_motifs:
        setting["min_score"] = float(
            profile.get("front_similarity_min_score", 0.285)
        )
        setting["angle_min_deg"] = float(
            profile.get("front_similarity_angle_min_deg", -20.0)
        )
        setting["angle_max_deg"] = float(
            profile.get("front_similarity_angle_max_deg", 20.0)
        )
        setting["angle_step_deg"] = float(
            profile.get("front_similarity_angle_step_deg", 5.0)
        )
        setting["scale_min"] = float(
            profile.get("front_similarity_scale_min", 0.75)
        )
        setting["scale_max"] = float(
            profile.get("front_similarity_scale_max", 1.15)
        )
        setting["scale_steps"] = int(
            profile.get("front_similarity_scale_steps", 5)
        )
    provider_options["motif_templates"] = selected_motifs
    roi_norm = profile.get("include_roi_norm")
    if isinstance(roi_norm, (list, tuple)) and len(roi_norm) == 4:
        provider_options["roi_norm"] = [float(value) for value in roi_norm]
        task2_config["adaptive_roi_norm"] = [
            float(value) for value in roi_norm
        ]
    task2_config["adaptive_profile_name"] = str(
        profile.get("adaptive_profile_name", "task2")
    )
    task2_config["adaptive_allowed_counts"] = list(
        profile.get("adaptive_allowed_counts", [3, 4])
    )
    task2_config["adaptive_recovery_enabled"] = bool(
        profile.get("adaptive_recovery_enabled", True)
    )
    task2_config["provider_options"] = provider_options
    task2_config["reference_feature_face_types"] = ["front_large"]
    slot_mode = profile.get("front_similarity_slot_mode")
    if isinstance(slot_mode, str) and slot_mode in {"sift", "template"}:
        provider_options["motif_templates"] = []
        task2_config["reference_feature_slot_mode"] = slot_mode
        task2_config["reference_feature_slot_columns"] = int(
            profile.get("front_similarity_slot_columns", 1)
        )
        task2_config["reference_feature_slot_ratio"] = float(
            profile.get("front_similarity_sift_ratio", 0.80)
        )
        task2_config["reference_feature_slot_min_inliers"] = int(
            profile.get("front_similarity_sift_min_inliers", 6)
        )
        task2_config["reference_feature_max_features"] = int(
            profile.get("front_similarity_sift_max_features", 2500)
        )
        task2_config["reference_feature_contrast_threshold"] = float(
            profile.get("front_similarity_sift_contrast_threshold", 0.02)
        )
        task2_config["reference_feature_template_min_score"] = float(
            profile.get("front_similarity_template_min_score", 0.50)
        )
    return task2_config


def build_task1_detector_config(
    detector_config: dict[str, Any],
    task_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build Task1's isolated per-face SIFT/RGB-D 3 x 3 profile.

    All three visible columns participate in recognition.  The workflow still
    processes the operator-requested 18 cartons one at a time; the 3 x 3
    setting controls only the maximum number of visible candidates per frame.
    Missing-face recovery remains Task1-only and never changes Task2.
    """

    profile = task_profile if isinstance(task_profile, dict) else {}
    task1_config = build_task3_detector_config(detector_config, profile)
    task1_config["adaptive_profile_name"] = str(
        profile.get("adaptive_profile_name", "task1_3x3")
    )
    task1_config["adaptive_allowed_counts"] = list(
        profile.get("adaptive_allowed_counts", [1, 2, 3, 4, 5, 6, 7, 8, 9])
    )
    task1_config["adaptive_recovery_enabled"] = bool(
        profile.get("adaptive_recovery_enabled", False)
    )
    task1_config["adaptive_reject_glare_matches"] = bool(
        profile.get("adaptive_reject_glare_matches", True)
    )
    task1_config["adaptive_include_polygon_norm"] = copy.deepcopy(
        profile.get("include_polygon_norm")
    )
    task1_config["adaptive_homography_attempt_multiplier"] = int(
        profile.get("adaptive_homography_attempt_multiplier", 4)
    )
    task1_config["adaptive_sift_ratio"] = float(
        profile.get("adaptive_sift_ratio", 0.86)
    )
    task1_config["adaptive_maximum_opposite_side_ratio"] = float(
        profile.get("adaptive_maximum_opposite_side_ratio", 1.42)
    )
    task1_config["adaptive_slot_grid_shape"] = list(
        profile.get("adaptive_slot_grid_shape", [3, 3])
    )
    task1_config["adaptive_slot_polygon_norm"] = copy.deepcopy(
        profile.get("adaptive_slot_polygon_norm")
    )
    task1_config["adaptive_slot_sift_ratio"] = float(
        profile.get("adaptive_slot_sift_ratio", 0.95)
    )
    task1_config["adaptive_slot_min_matches"] = int(
        profile.get("adaptive_slot_min_matches", 6)
    )
    task1_config["adaptive_slot_min_inliers"] = int(
        profile.get("adaptive_slot_min_inliers", 5)
    )
    return task1_config


def build_task3_detector_config(
    detector_config: dict[str, Any],
    task_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a bear-front-only detector profile for tabletop flat cartons.

    Task3 shares the printed large face with a formed carton, but it must not
    inherit Task1's repeated-stack motif short circuit. Height, orientation,
    2 x 2 layout and tabletop gates are applied later by the Task3 profile.
    """

    profile = task_profile if isinstance(task_profile, dict) else {}
    task3_config = build_task2_detector_config(detector_config, profile)
    task3_config["adaptive_profile_name"] = "task3"
    task3_config["adaptive_allowed_counts"] = list(
        profile.get("adaptive_allowed_counts", [1, 2])
    )
    task3_config["adaptive_recovery_enabled"] = bool(
        profile.get("adaptive_recovery_enabled", False)
    )
    task3_config["adaptive_minimum_quad_fill"] = float(
        profile.get("adaptive_minimum_quad_fill", 0.70)
    )
    task3_config["adaptive_long_side_px_range"] = list(
        profile.get("front_similarity_long_side_px_range", [80.0, 180.0])
    )
    task3_config["adaptive_short_side_px_range"] = list(
        profile.get("front_similarity_short_side_px_range", [50.0, 120.0])
    )
    return task3_config


def should_recover_task_row(
    task_id: str,
    *,
    individual_front_similarity: bool,
    task_profile: dict[str, Any],
) -> bool:
    """Allow verified Task2 identity to seed safe RGB-D row recovery."""

    if task_id == "task2":
        return (
            task_profile.get("recover_row_from_verified_identity", True)
            is True
        )
    return task_id == "task3" and not individual_front_similarity


def flange_offset_for_orientation(
    calibration: dict[str, Any],
    quaternion_xyzw: Sequence[float],
) -> np.ndarray:
    """Return base-frame TCP-to-flange translation for one flange pose."""

    flange_to_tcp = calibration.get("flange_to_tcp", {})
    local_translation = np.asarray(
        (
            flange_to_tcp.get("translation_m")
            if isinstance(flange_to_tcp, dict)
            else calibration.get("translation_m")
        ),
        dtype=np.float64,
    )
    if local_translation.shape != (3,) or not np.all(
        np.isfinite(local_translation)
    ):
        local_translation = np.asarray(
            calibration.get("translation_m"), dtype=np.float64
        )
    if local_translation.shape != (3,) or not np.all(
        np.isfinite(local_translation)
    ):
        raise ValueError("suction flange-to-TCP translation is invalid")
    rotation = quaternion_xyzw_to_matrix(quaternion_xyzw)
    return -(rotation @ local_translation)


def segmented_linear_positions(
    start_position_m: Sequence[float],
    end_position_m: Sequence[float],
    maximum_step_m: float,
) -> list[list[float]]:
    """Split one Cartesian translation into bounded endpoint waypoints."""

    start = np.asarray(start_position_m, dtype=np.float64)
    end = np.asarray(end_position_m, dtype=np.float64)
    step = float(maximum_step_m)
    if (
        start.shape != (3,)
        or end.shape != (3,)
        or not np.all(np.isfinite(start))
        or not np.all(np.isfinite(end))
        or not math.isfinite(step)
        or not 0.005 <= step <= 0.10
    ):
        raise ValueError("segmented Cartesian path inputs are invalid")
    distance = float(np.linalg.norm(end - start))
    count = max(1, int(math.ceil(distance / step)))
    return [
        (start + (index / count) * (end - start)).tolist()
        for index in range(1, count + 1)
    ]


def apply_top_view_clockwise_yaw_xyzw(
    quaternion_xyzw: Sequence[float],
    clockwise_degrees: float,
) -> np.ndarray:
    """Rotate a flange clockwise about base +Z when viewed from above."""

    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    degrees = float(clockwise_degrees)
    if (
        quaternion.shape != (4,)
        or not np.all(np.isfinite(quaternion))
        or not math.isfinite(degrees)
    ):
        raise ValueError("flange quaternion/yaw adjustment is invalid")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-9:
        raise ValueError("flange quaternion/yaw adjustment is invalid")
    quaternion = quaternion / norm
    # Looking from +Z toward the work surface, clockwise is negative yaw.
    half_angle = math.radians(-degrees) * 0.5
    yaw = np.asarray(
        [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)],
        dtype=np.float64,
    )
    x1, y1, z1, w1 = yaw
    x2, y2, z2, w2 = quaternion
    result = np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )
    return result / np.linalg.norm(result)


def shipping_box_detections_consistent(
    current: dict[str, Any],
    previous: dict[str, Any],
    *,
    xy_tolerance_m: float,
    pixel_tolerance_px: float,
    size_tolerance_m: float,
) -> dict[str, Any]:
    """Compare stable box geometry without rejecting harmless rim-Z noise."""

    current_point = np.asarray(current.get("point_left_base_m"), dtype=np.float64)
    previous_point = np.asarray(previous.get("point_left_base_m"), dtype=np.float64)
    current_size = np.asarray(current.get("opening_size_m"), dtype=np.float64)
    previous_size = np.asarray(previous.get("opening_size_m"), dtype=np.float64)
    current_px = np.asarray(
        (current.get("candidate") or {}).get("center_px"), dtype=np.float64
    )
    previous_px = np.asarray(
        (previous.get("candidate") or {}).get("center_px"), dtype=np.float64
    )
    if any(
        value.shape != shape or not np.all(np.isfinite(value))
        for value, shape in (
            (current_point, (3,)),
            (previous_point, (3,)),
            (current_size, (2,)),
            (previous_size, (2,)),
            (current_px, (2,)),
            (previous_px, (2,)),
        )
    ):
        return {"valid": False, "reason": "incomplete_geometry"}
    xy_distance_m = float(np.linalg.norm(current_point[:2] - previous_point[:2]))
    pixel_distance_px = float(np.linalg.norm(current_px - previous_px))
    size_delta_m = float(
        np.max(np.abs(np.sort(current_size) - np.sort(previous_size)))
    )
    z_delta_m = abs(float(current_point[2] - previous_point[2]))
    return {
        "valid": bool(
            xy_distance_m <= xy_tolerance_m
            and pixel_distance_px <= pixel_tolerance_px
            and size_delta_m <= size_tolerance_m
        ),
        "xy_distance_m": xy_distance_m,
        "pixel_distance_px": pixel_distance_px,
        "size_delta_m": size_delta_m,
        "z_delta_m": z_delta_m,
        "xy_tolerance_m": xy_tolerance_m,
        "pixel_tolerance_px": pixel_tolerance_px,
        "size_tolerance_m": size_tolerance_m,
    }


def shipping_box_image_detections_consistent(
    current: dict[str, Any],
    previous: dict[str, Any],
    *,
    pixel_tolerance_px: float,
    side_tolerance_fraction: float,
) -> dict[str, Any]:
    """Confirm that two frames show the same 2-D shipping-box footprint."""

    current_candidate = current.get("candidate") or {}
    previous_candidate = previous.get("candidate") or {}
    current_px = np.asarray(current_candidate.get("center_px"), dtype=np.float64)
    previous_px = np.asarray(previous_candidate.get("center_px"), dtype=np.float64)
    current_sides = np.asarray(
        [
            current_candidate.get("long_side_px"),
            current_candidate.get("short_side_px"),
        ],
        dtype=np.float64,
    )
    previous_sides = np.asarray(
        [
            previous_candidate.get("long_side_px"),
            previous_candidate.get("short_side_px"),
        ],
        dtype=np.float64,
    )
    if any(
        value.shape != (2,) or not np.all(np.isfinite(value))
        for value in (current_px, previous_px, current_sides, previous_sides)
    ):
        return {"valid": False, "reason": "incomplete_image_geometry"}
    pixel_distance_px = float(np.linalg.norm(current_px - previous_px))
    side_delta_fraction = float(
        np.max(
            np.abs(np.sort(current_sides) - np.sort(previous_sides))
            / np.maximum(1.0, np.sort(previous_sides))
        )
    )
    return {
        "valid": bool(
            pixel_distance_px <= pixel_tolerance_px
            and side_delta_fraction <= side_tolerance_fraction
        ),
        "pixel_distance_px": pixel_distance_px,
        "side_delta_fraction": side_delta_fraction,
        "pixel_tolerance_px": pixel_tolerance_px,
        "side_tolerance_fraction": side_tolerance_fraction,
    }


class Task3FrontPanelProjector:
    """Project the verified 130 x 85 mm bear front inside a flat dieline.

    Task3 proposals describe the complete flat carton footprint.  The suction
    cups, however, may only touch the printed front panel.  A local SIFT
    homography recovers that panel inside each already-separated carton cell;
    failure is intentionally fail-closed.
    """

    def __init__(
        self,
        face_bank: ReferenceFaceBank | None,
        *,
        minimum_matches: int = 8,
        minimum_inliers: int = 6,
        ratio: float = 0.78,
        ransac_px: float = 6.0,
    ) -> None:
        self._minimum_matches = max(4, int(minimum_matches))
        self._minimum_inliers = max(4, int(minimum_inliers))
        self._ratio = float(ratio)
        self._ransac_px = float(ransac_px)
        self._sift = cv2.SIFT_create(nfeatures=1600)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        self._reference_id = ""
        self._reference_shape: tuple[int, int] | None = None
        self._reference_keypoints: tuple[Any, ...] = ()
        self._reference_descriptors: np.ndarray | None = None
        if face_bank is None:
            return
        for face in face_bank.faces:
            if str(face.face_type) != "front_large" or not face.pick_allowed:
                continue
            gray = cv2.imread(str(face.image_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            keypoints, descriptors = self._sift.detectAndCompute(gray, None)
            if descriptors is None or len(keypoints) < self._minimum_matches:
                continue
            self._reference_id = str(face.id)
            self._reference_shape = gray.shape[:2]
            self._reference_keypoints = tuple(keypoints)
            self._reference_descriptors = descriptors
            break

    @property
    def ready(self) -> bool:
        return bool(
            self._reference_id
            and self._reference_shape is not None
            and self._reference_descriptors is not None
        )

    def project(
        self,
        rgb: np.ndarray,
        candidate: BoxCandidate,
    ) -> BoxCandidate | None:
        if not self.ready or rgb.ndim != 3 or rgb.shape[2] != 3:
            return None
        polygon = np.asarray(candidate.polygon_px, dtype=np.float32).reshape(-1, 2)
        if len(polygon) < 4 or not np.all(np.isfinite(polygon)):
            return None
        x0 = max(0, int(math.floor(float(np.min(polygon[:, 0])))))
        y0 = max(0, int(math.floor(float(np.min(polygon[:, 1])))))
        x1 = min(rgb.shape[1], int(math.ceil(float(np.max(polygon[:, 0])))) + 1)
        y1 = min(rgb.shape[0], int(math.ceil(float(np.max(polygon[:, 1])))) + 1)
        if x1 - x0 < 32 or y1 - y0 < 32:
            return None
        crop = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        local_polygon = np.round(
            polygon - np.asarray([x0, y0], dtype=np.float32)
        ).astype(np.int32)
        mask = np.zeros(crop.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, local_polygon, 255)
        masked = np.where(mask > 0, crop, 127).astype(np.uint8)
        keypoints, descriptors = self._sift.detectAndCompute(masked, None)
        if descriptors is None or len(keypoints) < self._minimum_matches:
            return None
        pairs = self._matcher.knnMatch(
            self._reference_descriptors,
            descriptors,
            k=2,
        )
        good = [
            first
            for pair in pairs
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < self._ratio * second.distance
        ]
        if len(good) < self._minimum_matches:
            return None
        source = np.float32(
            [self._reference_keypoints[item.queryIdx].pt for item in good]
        ).reshape(-1, 1, 2)
        target = np.float32(
            [keypoints[item.trainIdx].pt for item in good]
        ).reshape(-1, 1, 2)
        try:
            homography, inlier_mask = cv2.findHomography(
                source,
                target,
                cv2.RANSAC,
                self._ransac_px,
            )
        except cv2.error:
            return None
        if homography is None or inlier_mask is None:
            return None
        if int(np.count_nonzero(inlier_mask)) < self._minimum_inliers:
            return None
        reference_height, reference_width = self._reference_shape
        corners = np.asarray(
            [[0.0, 0.0], [reference_width - 1.0, 0.0],
             [reference_width - 1.0, reference_height - 1.0],
             [0.0, reference_height - 1.0]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        try:
            projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        except cv2.error:
            return None
        projected += np.asarray([x0, y0], dtype=np.float32)
        if not np.all(np.isfinite(projected)) or not cv2.isContourConvex(
            projected.astype(np.float32)
        ):
            return None
        candidate_contour = cv2.convexHull(polygon).reshape(-1, 2)
        if any(
            cv2.pointPolygonTest(
                candidate_contour.astype(np.float32),
                (float(point[0]), float(point[1])),
                True,
            ) < -5.0
            for point in projected
        ):
            return None
        candidate_area = abs(float(cv2.contourArea(candidate_contour)))
        panel_area = abs(float(cv2.contourArea(projected.astype(np.float32))))
        area_ratio = panel_area / max(candidate_area, 1.0)
        if not 0.18 <= area_ratio <= 1.05:
            return None
        rect = cv2.minAreaRect(projected.astype(np.float32))
        (center_x, center_y), (width, height), angle = rect
        if min(width, height) < 18.0:
            return None
        if width >= height:
            long_side, short_side, long_angle = width, height, angle
        else:
            long_side, short_side, long_angle = height, width, angle + 90.0
        while long_angle >= 90.0:
            long_angle -= 180.0
        while long_angle < -90.0:
            long_angle += 180.0
        panel_polygon = tuple(
            (float(point[0]), float(point[1])) for point in projected
        )
        blockers = tuple(
            blocker
            for blocker in candidate.grasp_blockers
            if blocker not in {"physical_size", "touching_cartons_unsplit"}
        )
        return replace(
            candidate,
            center_px=(float(center_x), float(center_y)),
            suction_px=(int(round(center_x)), int(round(center_y))),
            polygon_px=panel_polygon,
            long_side_px=float(long_side),
            short_side_px=float(short_side),
            angle_deg=float(long_angle),
            edge_clearance_px=float(short_side) * 0.5,
            provider=f"{candidate.provider}:task3_front_homography",
            face_type="front_large",
            reference_face_id=self._reference_id,
            graspable=not blockers,
            grasp_blockers=blockers,
        )


def compute_fixed_suction_axis_preview(
    samples: dict[str, Any],
    *,
    expected_spacing_m: float = 0.05,
    spacing_tolerance_m: float = 0.006,
    max_orientation_delta_deg: float = 0.30,
) -> dict[str, Any] | None:
    """Estimate the fixed cup axis from same-marker XY alignment samples.

    The operator is only required to align each cup centre over the same table
    marker in XY.  Sample height is deliberately ignored: a different Z for
    A and B must not tilt the calibrated cup axis.
    """

    if "A" not in samples or "B" not in samples:
        return None
    position_a = np.asarray(
        samples["A"]["flange_position_left_base_m"], dtype=np.float64
    )
    position_b = np.asarray(
        samples["B"]["flange_position_left_base_m"], dtype=np.float64
    )
    quaternion_a = np.asarray(
        samples["A"]["flange_quaternion_xyzw"], dtype=np.float64
    )
    quaternion_b = np.asarray(
        samples["B"]["flange_quaternion_xyzw"], dtype=np.float64
    )
    if (
        position_a.shape != (3,)
        or position_b.shape != (3,)
        or quaternion_a.shape != (4,)
        or quaternion_b.shape != (4,)
        or not np.all(np.isfinite(position_a))
        or not np.all(np.isfinite(position_b))
        or not np.all(np.isfinite(quaternion_a))
        or not np.all(np.isfinite(quaternion_b))
    ):
        return None
    quaternion_a /= max(float(np.linalg.norm(quaternion_a)), 1e-12)
    quaternion_b /= max(float(np.linalg.norm(quaternion_b)), 1e-12)
    delta_base = position_a - position_b
    planar_delta_base = np.asarray(
        [delta_base[0], delta_base[1], 0.0], dtype=np.float64
    )
    planar_spacing_m = float(np.linalg.norm(planar_delta_base))
    raw_spacing_m = float(np.linalg.norm(delta_base))
    orientation_dot = float(
        np.clip(abs(np.dot(quaternion_a, quaternion_b)), 0.0, 1.0)
    )
    orientation_delta_deg = float(
        np.degrees(2.0 * np.arccos(orientation_dot))
    )
    checks = {
        "planar_spacing": (
            planar_spacing_m > 1e-6
            and abs(planar_spacing_m - expected_spacing_m)
            <= spacing_tolerance_m
        ),
        "orientation": orientation_delta_deg <= max_orientation_delta_deg,
    }
    axis_local: list[float] | None = None
    approach_local: list[float] | None = None
    if planar_spacing_m > 1e-6:
        rotation = quaternion_xyzw_to_matrix(quaternion_a)
        axis = rotation.T @ (planar_delta_base / planar_spacing_m)
        approach = rotation.T @ np.asarray(
            [0.0, 0.0, -1.0], dtype=np.float64
        )
        approach /= max(float(np.linalg.norm(approach)), 1e-12)
        axis = axis - float(np.dot(axis, approach)) * approach
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm > 1e-6:
            axis /= axis_norm
            axis_local = axis.tolist()
            approach_local = approach.tolist()
        else:
            checks["planar_spacing"] = False
    return {
        "valid": all(checks.values()) and axis_local is not None,
        "checks": checks,
        "measured_spacing_mm": planar_spacing_m * 1000.0,
        "raw_3d_spacing_mm": raw_spacing_m * 1000.0,
        "orientation_delta_deg": orientation_delta_deg,
        "ignored_z_delta_mm": abs(float(delta_base[2])) * 1000.0,
        "z_ignored": True,
        "sampling_policy": "same_marker_xy_only",
        "axis_local_xyz": axis_local,
        "approach_local_xyz": approach_local,
    }


class PackagingConsoleApp:
    """Thread-safe state for API handlers."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        config_path: Path,
        bind: str,
        port: int,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.bind = bind
        self.port = port
        wrist_camera_cfg = config.get("wrist_cameras", {})
        if not isinstance(wrist_camera_cfg, dict):
            raise ValueError("wrist_cameras config must be an object")
        self.wrist_camera_enabled = wrist_camera_cfg.get("enabled") is True
        self.wrist_camera_base_url = str(
            wrist_camera_cfg.get("base_url", "http://127.0.0.1:8877")
        ).rstrip("/")
        self.wrist_camera_names = {
            "left": str(wrist_camera_cfg.get("left_name", "left")),
            "right": str(wrist_camera_cfg.get("right_name", "right")),
        }
        self.wrist_camera_timeout_s = max(
            0.25,
            min(float(wrist_camera_cfg.get("timeout_s", 3.0)), 10.0),
        )
        if self.wrist_camera_enabled:
            parsed_wrist_base = urlsplit(self.wrist_camera_base_url)
            if (
                parsed_wrist_base.scheme != "http"
                or parsed_wrist_base.hostname not in {"127.0.0.1", "localhost"}
                or parsed_wrist_base.username is not None
                or parsed_wrist_base.password is not None
                or parsed_wrist_base.query
                or parsed_wrist_base.fragment
            ):
                raise ValueError(
                    "wrist_cameras.base_url must be a loopback HTTP URL"
                )
        camera_cfg = config.get("camera", {})
        detector_cfg = config.get("detector", {})
        if not isinstance(camera_cfg, dict):
            raise ValueError("camera config must be an object")
        if not isinstance(detector_cfg, dict):
            raise ValueError("detector config must be an object")
        self.detector_cfg = detector_cfg
        task_profiles_cfg = config.get("task_profiles", {})
        if not isinstance(task_profiles_cfg, dict):
            raise ValueError("task_profiles config must be an object")
        self.task_profiles_cfg = {
            str(task_id): dict(profile)
            for task_id, profile in task_profiles_cfg.items()
            if isinstance(profile, dict)
        }
        fixed_axis_cfg = config.get("fixed_suction_axis", {})
        if not isinstance(fixed_axis_cfg, dict):
            raise ValueError("fixed_suction_axis config must be an object")
        self.fixed_suction_axis_cfg = dict(fixed_axis_cfg)
        self._fixed_axis_pending_path = (
            config_path.parent
            / "calibration"
            / "fixed_suction_axis_pending.json"
        )
        self._fixed_axis_calibration_session: dict[str, Any] = {
            "marker": None,
            "samples": {},
            "preview": None,
            "saved": False,
        }
        if self._fixed_axis_pending_path.exists():
            try:
                pending = _read_json(self._fixed_axis_pending_path)
                if isinstance(pending.get("calibration_session"), dict):
                    pending = pending["calibration_session"]
                if isinstance(pending, dict) and pending.get("saved") is not True:
                    pending_samples = pending.get("samples", {})
                    if not isinstance(pending_samples, dict):
                        pending_samples = {}
                    pending["samples"] = pending_samples
                    pending["preview"] = compute_fixed_suction_axis_preview(
                        pending_samples,
                        expected_spacing_m=float(
                            self.fixed_suction_axis_cfg.get(
                                "cup_center_spacing_mm", 50.0
                            )
                        )
                        / 1000.0,
                    )
                    pending["saved"] = False
                    self._fixed_axis_calibration_session = pending
            except Exception:
                pass
        face_bank_cfg = config.get("reference_face_bank", {})
        if not isinstance(face_bank_cfg, dict):
            raise ValueError("reference_face_bank config must be an object")
        face_bank: ReferenceFaceBank | None = None
        face_bank_error: str | None = None
        manifest_setting = face_bank_cfg.get("manifest")
        if isinstance(manifest_setting, str) and manifest_setting.strip():
            manifest_path = Path(manifest_setting).expanduser()
            if not manifest_path.is_absolute():
                manifest_path = config_path.parent / manifest_path
            try:
                face_bank = load_reference_face_bank(manifest_path.resolve())
            except Exception as exc:
                face_bank_error = f"{type(exc).__name__}: {exc}"
        else:
            face_bank_error = "reference face bank manifest is not configured"
        self.task3_front_panel_projector = Task3FrontPanelProjector(face_bank)
        task1_detector_cfg = build_task1_detector_config(
            detector_cfg,
            self.task_profiles_cfg.get("task1"),
        )
        self.detector: DetectorProvider = Task1AdaptiveVisualDetector(
            task1_detector_cfg,
            face_bank,
        )
        task2_detector_cfg = build_task2_detector_config(
            detector_cfg,
            self.task_profiles_cfg.get("task2"),
        )
        self.task2_detector: DetectorProvider = Task2AdaptiveVisualDetector(
            task2_detector_cfg,
            face_bank,
        )

        task3_detector_cfg = build_task3_detector_config(
            detector_cfg,
            self.task_profiles_cfg.get("task3"),
        )
        self.task3_detector: DetectorProvider = Task2AdaptiveVisualDetector(
            task3_detector_cfg,
            face_bank,
        )
        self.camera: PackagingCamera = create_camera(
            camera_cfg,
            config_dir=config_path.parent,
        )
        self.cam_to_left: np.ndarray | None = None
        self.cam_to_left_error: str | None = None
        cam_to_left_setting = camera_cfg.get("cam_to_left_path")
        if isinstance(cam_to_left_setting, str) and cam_to_left_setting.strip():
            cam_to_left_path = Path(cam_to_left_setting).expanduser()
            if not cam_to_left_path.is_absolute():
                cam_to_left_path = config_path.parent / cam_to_left_path
            try:
                self.cam_to_left = load_cam_to_left(cam_to_left_path.resolve())
            except Exception as exc:
                self.cam_to_left_error = f"{type(exc).__name__}: {exc}"
        else:
            self.cam_to_left_error = "camera.cam_to_left_path is not configured"
        recorder_cfg = config.get("trajectory_recorder", {})
        if not isinstance(recorder_cfg, dict):
            raise ValueError("trajectory_recorder config must be an object")
        self.trajectory_recorder = TrajectoryRecorder(
            self.camera,
            recorder_cfg,
            config_dir=config_path.parent,
        )
        act_inference_cfg = config.get("act_inference", {})
        if not isinstance(act_inference_cfg, dict):
            raise ValueError("act_inference config must be an object")
        self.act_inference = ActInferenceClient(
            act_inference_cfg,
            config_dir=config_path.parent,
        )
        teleop_cfg = config.get("teleop_launcher", {})
        if not isinstance(teleop_cfg, dict):
            raise ValueError("teleop_launcher config must be an object")
        self.teleop_launcher = TeleopLauncher(
            teleop_cfg,
            config_dir=config_path.parent,
        )
        cartesian_jog_cfg = config.get("cartesian_jog")
        if cartesian_jog_cfg is not None and not isinstance(
            cartesian_jog_cfg,
            dict,
        ):
            raise ValueError("cartesian_jog config must be an object")
        self._motion_transition_lock = threading.RLock()
        self.cartesian_jog = CartesianJogController(
            cartesian_jog_cfg,
            teleop_running=self._teleop_blocks_cartesian_jog,
        )
        right_arm_home_cfg = config.get("right_arm_home")
        if right_arm_home_cfg is not None and not isinstance(
            right_arm_home_cfg,
            dict,
        ):
            raise ValueError("right_arm_home config must be an object")
        self.right_arm_home = CartesianJogController(
            right_arm_home_cfg,
            teleop_running=self._teleop_blocks_cartesian_jog,
        )
        replay_cfg = config.get("trajectory_replay", {})
        if not isinstance(replay_cfg, dict):
            raise ValueError("trajectory_replay config must be an object")
        self.trajectory_replay = TrajectoryReplay(
            replay_cfg,
            config_dir=config_path.parent,
            interlock=self._trajectory_replay_blocker,
            suction_status=lambda: self.suction.status(),
            suction_setter=lambda engaged: self.suction.set_engaged(engaged),
        )
        rollout_cfg = config.get("act_rollout", {})
        if not isinstance(rollout_cfg, dict):
            raise ValueError("act_rollout config must be an object")
        self.act_rollout = ActRolloutController(
            rollout_cfg,
            inference=self.act_inference,
            frame_provider=self.trajectory_recorder.capture_act_frames,
            interlock=self._act_rollout_blocker,
            start_pose_checker=describe_act_start_pose,
        )
        suction_cfg = config.get("suction")
        if suction_cfg is not None and not isinstance(suction_cfg, dict):
            raise ValueError("suction config must be an object")
        self.suction = SuctionController(suction_cfg)
        base_trajectory_cfg = config.get("base_trajectory", {})
        if not isinstance(base_trajectory_cfg, dict):
            raise ValueError("base_trajectory config must be an object")
        self.base_trajectory = BaseTrajectoryController(
            base_trajectory_cfg,
            config_dir=config_path.parent,
            interlock=self._base_trajectory_blocker,
        )
        task1_pick_cfg = config.get("task1_pick")
        if task1_pick_cfg is not None and not isinstance(task1_pick_cfg, dict):
            raise ValueError("task1_pick config must be an object")
        self.task1_pick_cfg = dict(task1_pick_cfg or {})
        self.task1_pick_enabled = (
            self.task1_pick_cfg.get("enabled") is True
        )
        task1_fixed_place_cfg = config.get("task1_fixed_trajectory_place", {})
        if not isinstance(task1_fixed_place_cfg, dict):
            raise ValueError("task1_fixed_trajectory_place config must be an object")
        self.task1_fixed_place_cfg = dict(task1_fixed_place_cfg)
        self.task1_fixed_place_enabled = (
            self.task1_fixed_place_cfg.get("enabled") is True
        )
        self.task1_fixed_place_left_pose = str(
            self.task1_fixed_place_cfg.get("left_pose", "zhuangxiang")
        )
        self.task1_fixed_place_right_pose = str(
            self.task1_fixed_place_cfg.get("right_pose", "zhuangxiang")
        )
        self.task1_fixed_place_recording_id = str(
            self.task1_fixed_place_cfg.get("recording_id", "")
        )
        self.task1_fixed_place_speed_profile = str(
            self.task1_fixed_place_cfg.get("joint_speed_profile", "DEFAULT")
        ).upper()
        self.task1_fixed_place_replay_timeout_s = float(
            self.task1_fixed_place_cfg.get("replay_timeout_s", 90.0)
        )
        if self.task1_fixed_place_enabled and (
            not self.task1_fixed_place_left_pose
            or not self.task1_fixed_place_right_pose
            or not self.task1_fixed_place_recording_id
            or self.task1_fixed_place_speed_profile
            not in {"SLOW", "DEFAULT", "FAST"}
            or not 1.0 <= self.task1_fixed_place_replay_timeout_s <= 300.0
        ):
            raise ValueError("invalid task1_fixed_trajectory_place settings")
        self.task1_pick_calibration: dict[str, Any] | None = None
        self.task1_pick_error = ""
        calibration_setting = self.task1_pick_cfg.get("tcp_calibration")
        if self.task1_pick_enabled:
            try:
                if not isinstance(calibration_setting, str) or not calibration_setting:
                    raise ValueError("task1_pick.tcp_calibration is required")
                calibration_path = Path(calibration_setting).expanduser()
                if not calibration_path.is_absolute():
                    calibration_path = config_path.parent / calibration_path
                calibration = _read_json(calibration_path.resolve())
                contact = calibration.get("contact_sample")
                if not isinstance(contact, dict):
                    raise ValueError("TCP calibration has no contact_sample")
                offset = np.asarray(
                    contact.get("surface_to_target_flange_offset_in_base_m"),
                    dtype=np.float64,
                )
                locked = np.asarray(
                    calibration.get("locked_flange_quaternion_xyzw"),
                    dtype=np.float64,
                )
                if offset.shape != (3,) or not np.all(np.isfinite(offset)):
                    raise ValueError("contact offset must contain 3 finite values")
                if locked.shape != (4,) or not np.all(np.isfinite(locked)):
                    raise ValueError("locked quaternion must contain 4 finite values")
                if calibration.get("usable_for_motion") is not True:
                    raise ValueError("TCP calibration is not usable_for_motion")
                self.task1_pick_calibration = calibration
            except Exception as exc:
                self.task1_pick_error = f"{type(exc).__name__}: {exc}"
        task1_slot_grid_cfg = config.get("task1_slot_grid", {})
        if not isinstance(task1_slot_grid_cfg, dict):
            raise ValueError("task1_slot_grid config must be an object")
        self.task1_slot_grid_cfg = dict(task1_slot_grid_cfg)
        self.task1_slot_grid_enabled = (
            self.task1_slot_grid_cfg.get("enabled") is True
        )
        task1_slot_plan_setting = Path(
            str(
                self.task1_slot_grid_cfg.get(
                    "persistent_plan_path",
                    TASK1_SLOT_PLAN_STATE_PATH,
                )
            )
        ).expanduser()
        self.task1_slot_plan_state_path = (
            task1_slot_plan_setting
            if task1_slot_plan_setting.is_absolute()
            else (config_path.parent / task1_slot_plan_setting).resolve()
        )
        self.task1_slot_plan_reuse_enabled = (
            self.task1_slot_grid_cfg.get("reuse_persistent_plan", True) is True
        )
        task1_box_placement_cfg = config.get("task1_box_placement", {})
        if not isinstance(task1_box_placement_cfg, dict):
            raise ValueError("task1_box_placement config must be an object")
        self.task1_box_placement_cfg = dict(task1_box_placement_cfg)
        self.task1_box_placement_enabled = (
            self.task1_box_placement_cfg.get("enabled") is True
        )
        self._task1_right_clearance_state: dict[str, Any] | None = None
        task1_staged_top_cfg = config.get("task1_staged_top", {})
        if not isinstance(task1_staged_top_cfg, dict):
            raise ValueError("task1_staged_top config must be an object")
        self.task1_staged_top_cfg = dict(task1_staged_top_cfg)
        self.task1_staged_top_enabled = (
            self.task1_staged_top_cfg.get("enabled") is True
        )
        self.task1_staged_top_calibration: dict[str, Any] | None = None
        self.task1_staged_top_error = ""
        staged_calibration_setting = self.task1_staged_top_cfg.get(
            "tcp_calibration",
            "calibration/left_suction_tcp.json",
        )
        if self.task1_staged_top_enabled:
            try:
                if (
                    not isinstance(staged_calibration_setting, str)
                    or not staged_calibration_setting.strip()
                ):
                    raise ValueError(
                        "task1_staged_top.tcp_calibration is required"
                    )
                staged_calibration_path = Path(
                    staged_calibration_setting
                ).expanduser()
                if not staged_calibration_path.is_absolute():
                    staged_calibration_path = (
                        config_path.parent / staged_calibration_path
                    )
                staged_calibration = _read_json(
                    staged_calibration_path.resolve()
                )
                staged_contact = staged_calibration.get("contact_sample")
                staged_locked = np.asarray(
                    staged_calibration.get("locked_flange_quaternion_xyzw"),
                    dtype=np.float64,
                )
                if not isinstance(staged_contact, dict):
                    raise ValueError(
                        "staged-top TCP calibration has no contact_sample"
                    )
                staged_z_offset = float(
                    staged_contact.get("surface_to_flange_z_offset_m")
                )
                if (
                    staged_locked.shape != (4,)
                    or not np.all(np.isfinite(staged_locked))
                    or not math.isfinite(staged_z_offset)
                ):
                    raise ValueError(
                        "staged-top TCP calibration is incomplete"
                    )
                if staged_calibration.get("usable_for_motion") is not True:
                    raise ValueError(
                        "staged-top TCP calibration is not usable_for_motion"
                    )
                self.task1_staged_top_calibration = staged_calibration
            except Exception as exc:
                self.task1_staged_top_error = f"{type(exc).__name__}: {exc}"
        task2_pick_cfg = config.get("task2_pick")
        if task2_pick_cfg is not None and not isinstance(task2_pick_cfg, dict):
            raise ValueError("task2_pick config must be an object")
        self.task2_pick_cfg = dict(task2_pick_cfg or {})
        self.task2_pick_enabled = self.task2_pick_cfg.get("enabled") is True
        task2_workflow_cfg = config.get("task2_workflow", {})
        if not isinstance(task2_workflow_cfg, dict):
            raise ValueError("task2_workflow config must be an object")
        self.task2_workflow_cfg = dict(task2_workflow_cfg)
        shipping_box_cfg = self.task2_workflow_cfg.get(
            "shipping_box_detection",
            {},
        )
        if not isinstance(shipping_box_cfg, dict):
            raise ValueError(
                "task2_workflow.shipping_box_detection must be an object"
            )
        self.task2_shipping_box_cfg = dict(shipping_box_cfg)
        self.task2_shipping_box_enabled = (
            self.task2_shipping_box_cfg.get("enabled") is True
        )
        shipping_box_placement_cfg = self.task2_workflow_cfg.get(
            "shipping_box_placement",
            {},
        )
        if not isinstance(shipping_box_placement_cfg, dict):
            raise ValueError(
                "task2_workflow.shipping_box_placement must be an object"
            )
        self.task2_shipping_box_placement_cfg = dict(
            shipping_box_placement_cfg
        )
        self.task2_joint_speed_profile = str(
            self.task2_workflow_cfg.get("joint_speed_profile", "DEFAULT")
        ).strip().upper()
        if self.task2_joint_speed_profile not in {"SLOW", "DEFAULT", "FAST"}:
            raise ValueError(
                "task2_workflow.joint_speed_profile must be SLOW, DEFAULT or FAST"
            )
        self.task2_left_ready_pose = str(
            self.task2_workflow_cfg.get("left_ready_pose", "paper_init")
        ).strip()
        self.task2_right_ready_pose = str(
            self.task2_workflow_cfg.get("right_ready_pose", "init_pose")
        ).strip()
        if not self.task2_left_ready_pose or not self.task2_right_ready_pose:
            raise ValueError("task2_workflow ready pose names cannot be empty")
        self.task2_pick_calibration: dict[str, Any] | None = None
        self.task2_pick_error = ""
        if self.task2_pick_enabled:
            try:
                task2_calibration_setting = self.task2_pick_cfg.get(
                    "tcp_calibration"
                )
                if (
                    not isinstance(task2_calibration_setting, str)
                    or not task2_calibration_setting
                ):
                    raise ValueError("task2_pick.tcp_calibration is required")
                task2_calibration_path = Path(
                    task2_calibration_setting
                ).expanduser()
                if not task2_calibration_path.is_absolute():
                    task2_calibration_path = (
                        config_path.parent / task2_calibration_path
                    )
                calibration = _read_json(task2_calibration_path.resolve())
                contact = calibration.get("contact_sample")
                if not isinstance(contact, dict):
                    raise ValueError("TCP calibration has no contact_sample")
                offset = np.asarray(
                    contact.get("surface_to_target_flange_offset_in_base_m"),
                    dtype=np.float64,
                )
                locked = np.asarray(
                    calibration.get("locked_flange_quaternion_xyzw"),
                    dtype=np.float64,
                )
                contact_z = float(contact.get("absolute_contact_flange_z_m"))
                surface = np.asarray(
                    contact.get("carton_surface_center_in_base_m"),
                    dtype=np.float64,
                )
                if offset.shape != (3,) or not np.all(np.isfinite(offset)):
                    raise ValueError("contact offset must contain 3 finite values")
                if locked.shape != (4,) or not np.all(np.isfinite(locked)):
                    raise ValueError("locked quaternion must contain 4 finite values")
                if surface.shape != (3,) or not np.all(np.isfinite(surface)):
                    raise ValueError("contact surface must contain 3 finite values")
                if not math.isfinite(contact_z):
                    raise ValueError("absolute contact flange Z must be finite")
                if calibration.get("usable_for_motion") is not True:
                    raise ValueError("TCP calibration is not usable_for_motion")
                self.task2_pick_calibration = calibration
            except Exception as exc:
                self.task2_pick_error = f"{type(exc).__name__}: {exc}"
        task3_pick_cfg = config.get("task3_pick")
        if task3_pick_cfg is not None and not isinstance(task3_pick_cfg, dict):
            raise ValueError("task3_pick config must be an object")
        self.task3_pick_cfg = dict(task3_pick_cfg or {})
        self.task3_pick_enabled = self.task3_pick_cfg.get("enabled") is True
        self.task3_pick_calibration: dict[str, Any] | None = None
        self.task3_pick_error = ""
        if self.task3_pick_enabled:
            try:
                task3_calibration_setting = self.task3_pick_cfg.get(
                    "tcp_calibration"
                )
                if (
                    not isinstance(task3_calibration_setting, str)
                    or not task3_calibration_setting
                ):
                    raise ValueError("task3_pick.tcp_calibration is required")
                task3_calibration_path = Path(
                    task3_calibration_setting
                ).expanduser()
                if not task3_calibration_path.is_absolute():
                    task3_calibration_path = (
                        config_path.parent / task3_calibration_path
                    )
                calibration = _read_json(task3_calibration_path.resolve())
                contact = calibration.get("contact_sample")
                if not isinstance(contact, dict):
                    raise ValueError("TCP calibration has no contact_sample")
                offset = np.asarray(
                    contact.get("surface_to_target_flange_offset_in_base_m"),
                    dtype=np.float64,
                )
                locked = np.asarray(
                    calibration.get("locked_flange_quaternion_xyzw"),
                    dtype=np.float64,
                )
                table_surface_z = float(
                    self.task3_pick_cfg.get("table_surface_z_m")
                )
                if offset.shape != (3,) or not np.all(np.isfinite(offset)):
                    raise ValueError("contact offset must contain 3 finite values")
                if locked.shape != (4,) or not np.all(np.isfinite(locked)):
                    raise ValueError("locked quaternion must contain 4 finite values")
                if not math.isfinite(table_surface_z):
                    raise ValueError("task3 table surface Z must be finite")
                if calibration.get("usable_for_motion") is not True:
                    raise ValueError("TCP calibration is not usable_for_motion")
                self.task3_pick_calibration = calibration
            except Exception as exc:
                self.task3_pick_error = f"{type(exc).__name__}: {exc}"
        task3_expand_cfg = config.get("task3_expand", {})
        if not isinstance(task3_expand_cfg, dict):
            raise ValueError("task3_expand config must be an object")
        self.task3_expand_cfg = dict(task3_expand_cfg)
        runtime_cfg = config.get("runtime_parameters", {})
        if not isinstance(runtime_cfg, dict):
            raise ValueError("runtime_parameters config must be an object")
        runtime_path_setting = Path(
            str(runtime_cfg.get("path", "runtime/operator_parameters.json"))
        ).expanduser()
        runtime_path = (
            runtime_path_setting
            if runtime_path_setting.is_absolute()
            else config_path.parent / runtime_path_setting
        ).resolve()
        task3_contact_z: float | None = None
        try:
            task3_contact = (self.task3_pick_calibration or {})[
                "contact_sample"
            ]
            task3_contact_z = float(self.task3_pick_cfg["table_surface_z_m"]) + float(
                task3_contact["surface_to_target_flange_offset_in_base_m"][2]
            )
        except (KeyError, IndexError, TypeError, ValueError):
            pass
        runtime_defaults: dict[str, Any] = {
            "task1": {
                "transit_z_m": self.task1_pick_cfg.get("transit_z_m", 0.10),
                "pre_contact_clearance_m": self.task1_pick_cfg.get(
                    "pre_contact_clearance_m", 0.025
                ),
                "test_lift_m": self.task1_pick_cfg.get("test_lift_m", 0.02),
                "contact_flange_z_m_by_layer": self.task1_pick_cfg.get(
                    "contact_flange_z_m_by_layer",
                    {"1": 0.04, "2": 0.065, "3": 0.09},
                ),
            },
            "task2": {
                "transit_z_m": self.task2_pick_cfg.get("transit_z_m", 0.10),
                "pre_contact_clearance_m": self.task2_pick_cfg.get(
                    "pre_contact_clearance_m", 0.025
                ),
                "test_lift_m": self.task2_pick_cfg.get("test_lift_m", 0.02),
                "contact_flange_z_m": self.task2_pick_cfg.get(
                    "contact_flange_z_m", 0.04
                ),
            },
            "task3": {
                "transit_z_m": self.task3_pick_cfg.get("transit_z_m", 0.10),
                "pre_contact_clearance_m": self.task3_pick_cfg.get(
                    "pre_contact_clearance_m", 0.025
                ),
                "test_lift_m": self.task3_pick_cfg.get("test_lift_m", 0.02),
                "contact_flange_z_m": (
                    task3_contact_z if task3_contact_z is not None else 0.025
                ),
            },
        }
        self.runtime_parameters = RuntimeParameterStore(
            runtime_path,
            runtime_defaults,
        )
        static_setting = Path(
            str(config.get("static_dir", "../src/medicine_agentic/web_static"))
        ).expanduser()
        self.static_dir = (
            static_setting
            if static_setting.is_absolute()
            else config_path.parent / static_setting
        ).resolve()
        self._lock = threading.Lock()
        self._preview_lock = threading.Lock()
        self._preview_jpeg: bytes | None = None
        self._preview_jpeg_cached_at = 0.0
        self._last_detection: dict[str, Any] | None = None
        self._last_shipping_box_detection: dict[str, Any] | None = None
        self._workflow_detection_cache: dict[str, dict[str, Any]] = {}
        self._task1_stack_prior = Task1StackOccupancyPrior()
        self._task2_workflow_status: dict[str, Any] = {
            "state": "idle",
            "step_index": None,
            "stage": None,
            "message": "Task2 workflow has not started",
            "error": None,
            "updated_at": time.time(),
        }
        self._overlay_jpegs: OrderedDict[str, bytes] = OrderedDict()
        self._restore_task1_slot_plan()

    def wrist_camera_frame_url(self, side: str) -> str:
        """Return the fixed loopback video-service URL for one wrist camera."""

        normalized = str(side).strip().lower()
        if not self.wrist_camera_enabled:
            raise WristCameraUnavailable("wrist-camera relay is disabled")
        if normalized not in self.wrist_camera_names:
            raise ValueError("wrist camera must be left or right")
        camera_name = self.wrist_camera_names[normalized].strip()
        if not camera_name:
            raise WristCameraUnavailable(
                f"{normalized} wrist camera is not configured"
            )
        return (
            f"{self.wrist_camera_base_url}/api/cameras/"
            f"{quote(camera_name, safe='')}/frame.jpg"
        )

    def health(self) -> dict[str, Any]:
        teleop_enabled = self.teleop_launcher.enabled
        teleop_status = self.teleop_launcher.status()
        jog_status = self._cartesian_jog_snapshot(teleop_status)
        jog_available = jog_status.get("available") is True
        jog_dry_run = jog_status.get("dry_run") is True
        suction_status = self.suction.status()
        rollout_status = self.act_rollout.status()
        return {
            "ok": True,
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "safety": {
                "mode": (
                    "operator_teleop"
                    if teleop_enabled
                    else "read_only"
                ),
                "dry_run": not teleop_enabled,
                "motion_api": teleop_enabled,
                "teleop_enable_api": teleop_enabled,
                "cartesian_jog_api": jog_available,
                "cartesian_jog_dry_run": jog_dry_run,
                "cartesian_jog_arm": "left",
                "direct_cartesian_step_api": bool(
                    jog_available and not jog_dry_run
                ),
                "autonomous_motion_api": False,
                "act_rollout_api": rollout_status.get("enabled") is True,
                "act_rollout_stop_semantics": (
                    "synchronous_hold_current"
                ),
                "bounded_task_skill_api": bool(
                    rollout_status.get("enabled") is True
                    or self.task1_pick_status().get("ready") is True
                    or self.task2_pick_status().get("ready") is True
                    or self.task3_pick_status().get("ready") is True
                    or jog_status.get("home_joint_pose", {}).get("available")
                    is True
                    or self._right_arm_home_snapshot().get(
                        "home_joint_pose",
                        {},
                    ).get("available") is True
                ),
                "bounded_task_skills": [
                    "task1.pick_carton",
                    "task2.pick_carton",
                    "task3.pick_flat_carton",
                    "left_arm.reset_home",
                    "right_arm.reset_home",
                    "act.rollout",
                ],
                "direct_joint_command_api": False,
                "suction_api": suction_status.get("available") is True,
                "chassis_api": False,
                "base_trajectory_api": self.base_trajectory.enabled,
                "navigation_api": False,
            },
            "server": {"bind": self.bind, "port": self.port},
            "detector": self.detector.status(),
            "recording": {
                "enabled": self.trajectory_recorder.enabled,
                "hardware_access": "feedback_only",
                "writes_local_files": True,
                "motion_api": False,
                "playback_api": False,
            },
            "act_inference": self.act_inference.cached_status(),
            "act_rollout": rollout_status,
            "teleop": {
                "enabled": self.teleop_launcher.enabled,
                "control": "existing_supervised_follow",
                "direct_joint_commands": False,
                "automatic_arm_service_recovery": False,
                "camera_access": False,
            },
            "cartesian_jog": jog_status,
            "suction": suction_status,
            "fixed_suction_axis": self.fixed_suction_axis_status()[
                "fixed_suction_axis"
            ],
            "task_profiles": self.task_profiles_status()["profiles"],
            "task1_pick": self.task1_pick_status(),
            "task2_pick": self.task2_pick_status(),
            "task3_pick": self.task3_pick_status(),
        }

    def fixed_suction_axis_status(self) -> dict[str, Any]:
        """Return calibration readiness without opening camera or arm links."""

        return {
            "ok": True,
            "fixed_suction_axis": fixed_suction_axis_status(
                self.fixed_suction_axis_cfg
            ),
            "calibration_session": json.loads(
                json.dumps(self._fixed_axis_calibration_session)
            ),
        }

    def _persist_fixed_axis_calibration_session(self) -> None:
        """Keep an unfinished A/B calibration across a service restart."""

        self._fixed_axis_pending_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._fixed_axis_pending_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(
                self._fixed_axis_calibration_session,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, self._fixed_axis_pending_path)

    def lock_fixed_axis_marker(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Lock one operator-clicked RGB-D marker without commanding motion."""

        self._require_exact_payload(payload, frozenset({"pixel_x", "pixel_y"}))
        pixel_x = float(payload.get("pixel_x"))
        pixel_y = float(payload.get("pixel_y"))
        if not math.isfinite(pixel_x) or not math.isfinite(pixel_y):
            raise ValueError("marker pixel must be finite")
        frame = self._capture_for_read()
        if frame.depth_z16 is None or frame.depth_scale_m is None:
            raise CameraUnavailable("synchronized depth is unavailable")
        if self.camera.intrinsics is None or self.cam_to_left is None:
            raise CameraUnavailable("RGB-D calibration is unavailable")
        height, width = frame.depth_z16.shape
        u, v = int(round(pixel_x)), int(round(pixel_y))
        if not (0 <= u < width and 0 <= v < height):
            raise ValueError("marker pixel is outside the RGB-D frame")
        radius = 6
        x0, x1 = max(0, u - radius), min(width, u + radius + 1)
        y0, y1 = max(0, v - radius), min(height, v + radius + 1)
        region = frame.depth_z16[y0:y1, x0:x1]
        samples = region[region > 0]
        if samples.size < 20:
            raise CameraUnavailable(
                "orange marker has insufficient valid depth; click its top center"
            )
        depth_m = float(np.median(samples.astype(np.float64))) * float(
            frame.depth_scale_m
        )
        point_camera = deproject_pixel(
            (u, v),
            depth_m * 1000.0,
            np.asarray(self.camera.intrinsics, dtype=np.float64),
        )
        point_base = transform_point(point_camera, self.cam_to_left)
        marker = {
            "pixel": [u, v],
            "depth_m": depth_m,
            "point_left_base_m": [float(value) for value in point_base],
            "captured_at": frame.captured_at,
        }
        with self._lock:
            self._fixed_axis_calibration_session = {
                "marker": marker,
                "samples": {},
                "preview": None,
                "saved": False,
            }
            self._persist_fixed_axis_calibration_session()
        return self.fixed_suction_axis_status()

    def sample_fixed_axis_cup(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Record current flange feedback after an operator aligns one cup."""

        self._require_exact_payload(payload, frozenset({"cup"}))
        cup = str(payload.get("cup", "")).upper()
        if cup not in {"A", "B"}:
            raise ValueError("cup must be A or B")
        session = self._fixed_axis_calibration_session
        if not isinstance(session.get("marker"), dict):
            raise CartesianJogSafetyViolation("lock the orange marker first")
        jog = self._cartesian_jog_snapshot()
        if jog.get("enabled") is not True or jog.get("busy") is True:
            raise CartesianJogSafetyViolation(
                "enable fixed-orientation XYZ jog and wait until it is idle"
            )
        position = np.asarray(jog.get("current_position_m"), dtype=np.float64)
        quaternion = np.asarray(
            jog.get("locked_quaternion_xyzw"), dtype=np.float64
        )
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise CartesianJogSafetyViolation("current flange position is unavailable")
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise CartesianJogSafetyViolation("locked flange orientation is unavailable")
        quaternion /= np.linalg.norm(quaternion)
        sample = {
            "cup": cup,
            "flange_position_left_base_m": position.tolist(),
            "flange_quaternion_xyzw": quaternion.tolist(),
            "captured_at": time.time(),
        }
        with self._lock:
            samples = dict(session.get("samples", {}))
            samples[cup] = sample
            session["samples"] = samples
            session["preview"] = compute_fixed_suction_axis_preview(
                samples,
                expected_spacing_m=float(
                    self.fixed_suction_axis_cfg.get(
                        "cup_center_spacing_mm", 50.0
                    )
                )
                / 1000.0,
            )
            session["saved"] = False
            self._persist_fixed_axis_calibration_session()
        return self.fixed_suction_axis_status()

    def commit_fixed_axis_calibration(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Persist a validated two-cup calibration; no robot command is sent."""

        self._require_exact_payload(payload, frozenset())
        session = self._fixed_axis_calibration_session
        preview = session.get("preview")
        if not isinstance(preview, dict) or preview.get("valid") is not True:
            raise CartesianJogSafetyViolation(
                "record both cup positions and pass all calibration checks first"
            )
        stamp = time.strftime("%Y%m%d_%H%M%S")
        record = {
            "schema_version": 1,
            "calibration_type": "same_marker_two_cup_axis",
            "created_at_local": stamp,
            "marker": session.get("marker"),
            "samples": session.get("samples"),
            "result": preview,
        }
        calibration_dir = self.config_path.parent / "calibration"
        calibration_dir.mkdir(parents=True, exist_ok=True)
        record_path = calibration_dir / f"fixed_suction_axis_{stamp}.json"
        record_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        updated = dict(self.fixed_suction_axis_cfg)
        updated.update(
            {
                "enabled": True,
                "calibrated": True,
                "calibration_version": stamp,
                "axis_local_xyz": preview["axis_local_xyz"],
                "approach_local_xyz": preview["approach_local_xyz"],
                "calibration_record": str(record_path),
            }
        )
        config_payload = _read_json(self.config_path)
        config_payload["fixed_suction_axis"] = updated
        backup_path = self.config_path.with_name(
            f"{self.config_path.stem}.before_fixed_axis_{stamp}.json"
        )
        shutil.copy2(self.config_path, backup_path)
        temporary_path = self.config_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, self.config_path)
        self.fixed_suction_axis_cfg = updated
        self.config["fixed_suction_axis"] = updated
        session["saved"] = True
        session["calibration_record"] = str(record_path)
        self._fixed_axis_pending_path.unlink(missing_ok=True)
        return self.fixed_suction_axis_status()

    def task_profiles_status(self) -> dict[str, Any]:
        defaults: dict[str, dict[str, Any]] = {
            "task1": {
                "label": "药盒装箱",
                "recognition_profile": "task1_stack",
                "height_policy": "highest_of_1_2_3_layers",
                "recognition_ready": True,
                "suction_skill_ready": True,
            },
            "task2": {
                "label": "装药板与闭盒",
                "recognition_profile": "task2_single_carton",
                "height_policy": "single_carton_surface",
                "recognition_ready": False,
                "suction_skill_ready": False,
            },
            "task3": {
                "label": "扁盒展开与闭盒",
                "recognition_profile": "task3_flat_carton",
                "height_policy": "tabletop_flat_carton",
                "recognition_ready": False,
                "suction_skill_ready": False,
            },
        }
        profiles: dict[str, dict[str, Any]] = {}
        for task_id in ("task1", "task2", "task3"):
            configured = {
                **defaults[task_id],
                **self.task_profiles_cfg.get(task_id, {}),
            }
            profiles[task_id] = {
                "label": str(configured.get("label", task_id)),
                "recognition_profile": str(
                    configured.get("recognition_profile", "")
                ),
                "height_policy": str(configured.get("height_policy", "")),
                "recognition_ready": (
                    configured.get("recognition_ready") is True
                ),
                "suction_skill_ready": (
                    configured.get("suction_skill_ready") is True
                ),
                "initial_total_count": int(
                    configured.get("initial_total_count", 0)
                ),
                "maximum_visible_instances": int(
                    configured.get("maximum_visible_instances", 0)
                ),
                "process_target_count": int(
                    configured.get("process_target_count", 0)
                ),
                "physical_instance_gate": dict(
                    configured.get("physical_instance_gate", {})
                ),
                "surface_z_range_left_base_m": configured.get(
                    "surface_z_range_left_base_m"
                ),
                "workflow_locked": configured.get("workflow_locked") is True,
                "workflow_version": configured.get("workflow_version"),
                "workflow_cycle_target_count": int(
                    configured.get("workflow_cycle_target_count", 0)
                ),
                "workflow_suction_policy": configured.get(
                    "workflow_suction_policy"
                ),
                "workflow_steps": copy.deepcopy(
                    configured.get("workflow_steps", [])
                ),
            }
        return {
            "ok": True,
            "default_task": "task1",
            "profiles": profiles,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            last_detection = self._last_detection
        image_ready = self.camera.state == "ready"
        live_rgbd = self.camera.live_rgbd_is_fresh(
            max_age_s=LIVE_RGBD_MAX_AGE_S
        )
        camera_profile = self.camera.profile()
        camera_profile_approved = (
            camera_profile.get("profile_approved") is True
            and self.cam_to_left is not None
        )
        detector_status = self.detector.status()
        reference_status = detector_status.get("reference_bank", {})
        detector_ready = detector_status.get("ok") is True
        reference_ready = (
            isinstance(reference_status, dict)
            and reference_status.get("ready") is True
        )
        teleop_status = self.teleop_launcher.status()
        jog_status = self._cartesian_jog_snapshot(teleop_status)
        suction_status = self.suction.status()
        return {
            "ok": True,
            "timestamp": time.time(),
            "task": {"active": "task1_pick_carton", "state": "idle"},
            "task2_workflow": self.task2_workflow_status(),
            "detector": detector_status,
            "fixed_suction_axis": self.fixed_suction_axis_status()[
                "fixed_suction_axis"
            ],
            "camera": {
                "mode": self.camera.mode,
                "state": self.camera.state,
                "error": self.camera.error,
                "live_rgbd": live_rgbd,
                "profile_approved": camera_profile_approved,
                "cam_to_left_ready": self.cam_to_left is not None,
                "cam_to_left_error": self.cam_to_left_error,
            },
            "gates": [
                {
                    "id": "operator_teleop",
                    "label": "操作员控制模式",
                    "passed": True,
                    "detail": (
                        "自主运动和直接关节指令保持禁用；仅允许操作员确认的 "
                        "Follow 遥操，以及左臂定姿 XYZ 离散微调。"
                    ),
                },
                {
                    "id": "chassis_excluded",
                    "label": "固定工位 / 无底盘",
                    "passed": True,
                    "detail": "底盘与导航不在打包项目的系统边界内。",
                },
                {
                    "id": "image_source_ready",
                    "label": "图像源可用",
                    "passed": image_ready,
                    "detail": (
                        (
                            "已通过 Web Console 共享实时画面，未独占相机设备。"
                            if self.camera.mode in {"shared", "shared_memory"}
                            else "已能提供画面；离线图仅用于界面和二维检测测试。"
                        )
                        if image_ready
                        else (self.camera.error or "相机尚未就绪。")
                    ),
                },
                {
                    "id": "detector_backend_ready",
                    "label": "二维候选检测器",
                    "passed": detector_ready,
                    "detail": (
                        f"{detector_status.get('name', 'unknown')} 已就绪。"
                        if detector_ready
                        else (
                            detector_status.get("last_error")
                            or detector_status.get("backend", {}).get("error")
                            or "检测器尚未就绪。"
                        )
                    ),
                },
                {
                    "id": "reference_face_bank",
                    "label": "药盒六面参考库",
                    "passed": reference_ready,
                    "detail": (
                        f"已锁定 {reference_status.get('bank_id')}。"
                        if reference_ready
                        else (
                            reference_status.get("error")
                            if isinstance(reference_status, dict)
                            else "参考面库尚未配置。"
                        )
                    ),
                },
                {
                    "id": "live_rgbd",
                    "label": "实时 RGB-D",
                    "passed": live_rgbd,
                    "detail": (
                        (
                            "已通过 Web Console 取得同步 RGB-D，未独占相机设备。"
                            if self.camera.mode in {"shared", "shared_memory"}
                            else "独立控制台已直接取得同步 RGB-D。"
                        )
                        if live_rgbd
                        else (
                            (
                                "共享通道当前传输彩色帧，不传输完整深度帧；"
                                "二维检测可用，三维抓取目标仍保持禁用。"
                            )
                            if self.camera.mode == "shared"
                            else (
                                "离线模式或 RealSense 未就绪，"
                                "不能输出三维抓取目标。"
                            )
                        )
                    ),
                },
                {
                    "id": "camera_profile_approved",
                    "label": "RGB-D配置批准",
                    "passed": camera_profile_approved,
                    "detail": (
                        "1280×720内参与相机到左臂基座外参均已加载。"
                        if camera_profile_approved
                        else (
                            self.cam_to_left_error
                            or "当前RGB-D档案尚未批准用于三维定位。"
                        )
                    ),
                },
                {
                    "id": "left_cartesian_jog",
                    "label": "左吸盘定姿 XYZ 微调",
                    "passed": jog_status.get("available") is True,
                    "detail": (
                        "当前为干运行：只校验离散目标，不发送机械臂运动。"
                        if jog_status.get("dry_run") is True
                        else "仅控制左臂，锁定捕获方向并执行低速离散 XYZ 步进。"
                    ),
                },
                {
                    "id": "left_dual_suction",
                    "label": "左臂双吸盘",
                    "passed": suction_status.get("available") is True,
                    "detail": (
                        "双通道吸紧与释放命令已接入独立串口。"
                        if suction_status.get("available") is True
                        else (
                            suction_status.get("error")
                            or "吸盘串口尚未就绪。"
                        )
                    ),
                },
                {
                    "id": "task1_pick_calibration",
                    "label": "任务一自动抓取标定",
                    "passed": self.task1_pick_status().get("ready") is True,
                    "detail": (
                        "固定竖直姿态、接触高度和100 mm试抬参数已锁定。"
                        if self.task1_pick_status().get("ready") is True
                        else (
                            self.task1_pick_error
                            or "任务一自动抓取尚未配置。"
                        )
                    ),
                },
                {
                    "id": "autonomous_motion_disabled",
                    "label": "通用自主运动禁用",
                    "passed": True,
                    "detail": (
                        "不开放通用单关节、底盘、导航或任意轨迹控制；"
                        "ACT 仅通过有互锁和限速的专用闭环入口运行。"
                    ),
                },
                {
                    "id": "bounded_act_rollout",
                    "label": "ACT 闭环控制",
                    "passed": self.act_rollout.enabled,
                    "detail": (
                        "仅开放带互锁、限速和同步保持停止的 ACT rollout；"
                        "不开放通用单关节、底盘或导航控制。"
                    ),
                },
            ],
            "last_detection": last_detection,
            "recording": self.trajectory_recorder.status(),
            "act_inference": self.act_inference.cached_status(),
            "act_rollout": self.act_rollout.status(),
            "replay": self.trajectory_replay.status(),
            "teleop": teleop_status,
            "cartesian_jog": jog_status,
            "suction": suction_status,
            "base_trajectory": self.base_trajectory.status(),
            "task1_pick": self.task1_pick_status(),
            "task2_pick": self.task2_pick_status(),
            "task3_pick": self.task3_pick_status(),
        }

    def camera_profile(self) -> dict[str, Any]:
        return {"ok": True, "camera": self.camera.profile()}

    def recording_status(self) -> dict[str, Any]:
        return {"ok": True, "recording": self.trajectory_recorder.status()}

    def act_inference_status(self) -> dict[str, Any]:
        return {"ok": True, "act_inference": self.act_inference.status(force=True)}

    def select_act_inference_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        if set(payload) != {"profile"} or not isinstance(payload.get("profile"), str):
            raise ValueError("request must contain exactly one string profile field")
        if self.act_rollout.status().get("active") is True:
            raise RecordingConflict("stop ACT rollout before switching models")
        status = self.act_inference.select_profile(payload["profile"])
        return {
            "ok": True,
            "act_inference": status,
            "robot_command_sent": False,
        }

    def predict_act_preview(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a model action chunk without issuing any robot command."""

        allowed = frozenset({"horizon", "session_id"})
        unknown = frozenset(payload) - allowed
        if unknown:
            raise ValueError(
                "request fields may contain only horizon and session_id"
            )
        if self.trajectory_recorder.status().get("active") is True:
            raise RecordingConflict("stop trajectory recording before ACT preview")
        if self.trajectory_replay.status().get("active") is True:
            raise RecordingConflict("stop trajectory replay before ACT preview")
        if self._teleop_blocks_cartesian_jog():
            raise RecordingConflict("stop teleoperation before ACT preview")
        for label, controller in (
            ("left-arm Cartesian control", self.cartesian_jog),
            ("right-arm control", self.right_arm_home),
        ):
            controller_status = controller.status()
            if controller_status.get("busy") or controller_status.get("enabled"):
                raise RecordingConflict(f"disable {label} before ACT preview")
        status = self.act_inference.status(force=True)
        if status.get("ready") is not True:
            raise ActInferenceUnavailable(
                str(status.get("error") or "ACT inference service is not ready")
            )
        horizon_value = payload.get("horizon")
        horizon = None if horizon_value is None else int(horizon_value)
        session_id = str(payload.get("session_id") or f"preview-{uuid.uuid4().hex}")
        if len(session_id) > 128:
            raise ValueError("session_id is too long")
        observation = self.trajectory_recorder.capture_act_observation()
        source_frames = observation["frames_bgr"]
        result = self.act_inference.predict(
            state=observation["state"],
            frames_bgr={
                "cam_high": source_frames["front"],
                "cam_left_wrist": source_frames["left_wrist"],
                "cam_right_wrist": source_frames["right_wrist"],
            },
            horizon=horizon,
            request_id=uuid.uuid4().hex,
            session_id=session_id,
        )
        result["preview_only"] = True
        result["robot_command_sent"] = False
        result["observation_timing"] = {
            "captured_at": observation["captured_at"],
            "camera_arm_delta_ms": observation["camera_arm_delta_ms"],
            "arm_pair_skew_ms": observation["arm_pair_skew_ms"],
            "validation": observation["timing_validation"],
            "static_joint_delta_rad": observation["static_joint_delta_rad"],
            "static_eef_delta_m": observation["static_eef_delta_m"],
        }
        result["observation_state"] = observation["state"]
        result["start_pose_diagnostic"] = describe_act_start_pose(observation["state"])
        return result

    def act_rollout_status(self) -> dict[str, Any]:
        return {"ok": True, "act_rollout": self.act_rollout.status()}

    def start_act_rollout(self, payload: dict[str, Any]) -> dict[str, Any]:
        allowed_fields = frozenset({"execute_steps_per_inference"})
        if not frozenset(payload).issubset(allowed_fields):
            raise ValueError(
                "request fields may only contain: execute_steps_per_inference"
            )
        execute_steps = payload.get("execute_steps_per_inference")
        if execute_steps is not None and (
            isinstance(execute_steps, bool) or not isinstance(execute_steps, int)
        ):
            raise ValueError("execute_steps_per_inference must be an integer")
        with self._motion_transition_lock:
            rollout = self.act_rollout.start(
                execute_steps_per_inference=execute_steps
            )
        return {"ok": True, "act_rollout": rollout}

    def stop_act_rollout(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_exact_payload(payload, frozenset())
        rollout = self.act_rollout.stop(reason="operator_stop")
        return {"ok": True, "act_rollout": rollout}

    def _act_rollout_blocker(self) -> str | None:
        if self.trajectory_recorder.status().get("active") is True:
            return "stop trajectory recording before ACT rollout"
        if self.trajectory_replay.status().get("active") is True:
            return "stop trajectory replay before ACT rollout"
        teleop_status = self.teleop_launcher.status()
        if (
            self._teleop_status_blocks_cartesian_jog(teleop_status)
            or self._system_follow_ownership_active()
        ):
            return "stop teleoperation before ACT rollout"
        for label, controller in (
            ("left-arm Cartesian control", self.cartesian_jog),
            ("right-arm control", self.right_arm_home),
        ):
            status = controller.status()
            if status.get("busy") or status.get("enabled"):
                return f"disable {label} before ACT rollout"
        return None

    def _base_trajectory_blocker(self) -> str | None:
        if self.act_rollout.status().get("active") is True:
            return "stop ACT rollout before base trajectory operation"
        if self.trajectory_recorder.status().get("active") is True:
            return "stop arm trajectory recording before base trajectory operation"
        if self.trajectory_replay.status().get("active") is True:
            return "stop arm trajectory replay before base trajectory operation"
        if (
            self._teleop_status_blocks_cartesian_jog(
                self.teleop_launcher.status()
            )
            or self._system_follow_ownership_active()
        ):
            return "stop teleoperation before base trajectory operation"
        for label, controller in (
            ("left-arm Cartesian control", self.cartesian_jog),
            ("right-arm control", self.right_arm_home),
        ):
            status = controller.status()
            if status.get("busy") or status.get("enabled"):
                return f"disable {label} before base trajectory operation"
        return None

    def recordings(self) -> dict[str, Any]:
        return {
            "ok": True,
            "recordings": self.trajectory_recorder.list_recordings(),
        }

    def start_recording(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.act_rollout.status().get("active") is True:
            raise RecordingConflict("stop ACT rollout before recording")
        if self.trajectory_replay.status().get("active") is True:
            raise RecordingConflict("stop trajectory replay before recording")
        label = payload.get("label", "recording")
        purpose = payload.get("purpose", "projection_validation")
        if not isinstance(label, str):
            raise ValueError("label must be a string")
        if not isinstance(purpose, str):
            raise ValueError("purpose must be a string")
        return {
            "ok": True,
            "recording": self.trajectory_recorder.start(
                label=label,
                purpose=purpose,
            ),
        }

    def stop_recording(self) -> dict[str, Any]:
        return {
            "ok": True,
            "recording": self.trajectory_recorder.stop(),
        }

    def delete_recording(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_exact_payload(payload, frozenset({"recording_id"}))
        recording_id = payload.get("recording_id")
        if not isinstance(recording_id, str):
            raise ValueError("recording_id must be a string")
        if self.trajectory_replay.status().get("active") is True:
            raise RecordingConflict("stop trajectory replay before deleting an episode")
        return {
            "ok": True,
            "deletion": self.trajectory_recorder.delete_recording(recording_id),
            "recordings": self.trajectory_recorder.list_recordings(),
        }

    def replay_status(self) -> dict[str, Any]:
        return {"ok": True, "replay": self.trajectory_replay.status()}

    def replay_preflight(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_exact_payload(payload, frozenset({"recording_id"}))
        recording_id = payload.get("recording_id")
        if not isinstance(recording_id, str):
            raise ValueError("recording_id must be a string")
        return self.trajectory_replay.preflight(recording_id)

    def start_replay(self, payload: dict[str, Any]) -> dict[str, Any]:
        allowed = frozenset({"recording_id", "confirmation"})
        if frozenset(payload) != allowed:
            raise ValueError("request fields must be recording_id and confirmation")
        recording_id = payload.get("recording_id")
        confirmation = payload.get("confirmation")
        if not isinstance(recording_id, str) or not isinstance(confirmation, str):
            raise ValueError("recording_id and confirmation must be strings")
        with self._motion_transition_lock:
            replay = self.trajectory_replay.start(
                recording_id=recording_id,
                confirmation=confirmation,
            )
        return {"ok": True, "replay": replay}

    def stop_replay(self) -> dict[str, Any]:
        return {"ok": True, "replay": self.trajectory_replay.stop()}

    def base_trajectory_status(self) -> dict[str, Any]:
        return {"ok": True, "base_trajectory": self.base_trajectory.status()}

    def base_trajectories(self) -> dict[str, Any]:
        return {
            "ok": True,
            "base_trajectories": self.base_trajectory.list_recordings(),
        }

    def start_base_trajectory_recording(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        allowed = frozenset({"label"})
        if not frozenset(payload).issubset(allowed):
            raise ValueError("request fields may contain only: label")
        label = payload.get("label", "base-trajectory")
        if not isinstance(label, str):
            raise ValueError("label must be a string")
        return {
            "ok": True,
            "base_trajectory": self.base_trajectory.start_recording(label),
        }

    def stop_base_trajectory_recording(self) -> dict[str, Any]:
        return {
            "ok": True,
            "base_trajectory": self.base_trajectory.stop_recording(),
        }

    def base_trajectory_replay_preflight(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_exact_payload(payload, frozenset({"recording_id"}))
        recording_id = payload.get("recording_id")
        if not isinstance(recording_id, str):
            raise ValueError("recording_id must be a string")
        return {
            "ok": True,
            "preflight": self.base_trajectory.preflight(recording_id),
        }

    def start_base_trajectory_replay(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        allowed = frozenset({"recording_id", "confirmation"})
        if frozenset(payload) != allowed:
            raise ValueError("request fields must be recording_id and confirmation")
        recording_id = payload.get("recording_id")
        confirmation = payload.get("confirmation")
        if not isinstance(recording_id, str) or not isinstance(confirmation, str):
            raise ValueError("recording_id and confirmation must be strings")
        with self._motion_transition_lock:
            replay = self.base_trajectory.start_replay(
                recording_id,
                confirmation,
            )
        return {"ok": True, "base_trajectory": replay}

    def stop_base_trajectory_replay(self) -> dict[str, Any]:
        return {
            "ok": True,
            "base_trajectory": self.base_trajectory.stop_replay(),
        }

    def _trajectory_replay_blocker(
        self,
        *,
        allow_suction_engaged: bool = False,
    ) -> str | None:
        if self.act_rollout.status().get("active") is True:
            return "stop ACT rollout before replay"
        if self.trajectory_recorder.status().get("active") is True:
            return "stop trajectory recording before replay"
        if (
            self._teleop_status_blocks_cartesian_jog(
                self.teleop_launcher.status()
            )
            or self._system_follow_ownership_active()
        ):
            return "stop teleoperation before replay"
        for label, controller in (
            ("left-arm Cartesian control", self.cartesian_jog),
            ("right-arm control", self.right_arm_home),
        ):
            status = controller.status()
            if status.get("busy") or status.get("enabled"):
                return f"disable {label} before replay"
        suction = self.suction.status()
        if suction.get("engaged") is True and not allow_suction_engaged:
            return "turn suction off before replay"
        return None

    def _teleop_mode_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "mode": "dual",
            "hold_pose_by_arm": {"left": "init_pose", "right": "init_pose"},
        }
        try:
            saved = _read_json(TELEOP_MODE_STATE_PATH)
            if saved.get("mode") in TELEOP_MODES:
                state["mode"] = saved["mode"]
            saved_poses = saved.get("hold_pose_by_arm")
            if isinstance(saved_poses, dict):
                for arm in ("left", "right"):
                    value = saved_poses.get(arm)
                    if isinstance(value, str) and value:
                        state["hold_pose_by_arm"][arm] = value
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        return state

    def _follow_hold_state(self) -> dict[str, bool]:
        sides = {"left": False, "right": False}
        try:
            payload = _read_json(TELEOP_FOLLOW_HOLD_PATH)
            raw_sides = payload.get("sides")
            if isinstance(raw_sides, dict):
                sides = {
                    arm: raw_sides.get(arm) is True
                    for arm in ("left", "right")
                }
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        return sides

    def _write_follow_hold_state(self, *, inactive_arm: str | None) -> None:
        now = time.time()
        payload: dict[str, Any] = {}
        try:
            payload = _read_json(TELEOP_FOLLOW_HOLD_PATH)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        payload.update(
            {
                "version": 1,
                "sides": {
                    "left": inactive_arm == "left",
                    "right": inactive_arm == "right",
                },
                "updated_at": now,
                "updated_side": inactive_arm or "both",
                "source": "medicine_packaging_console_single_arm_mode",
            }
        )
        _write_json_atomic(TELEOP_FOLLOW_HOLD_PATH, payload)

    @staticmethod
    def _inactive_arm_for_mode(mode: str) -> str | None:
        if mode == "left_only":
            return "right"
        if mode == "right_only":
            return "left"
        return None

    def _teleop_mode_snapshot(self) -> dict[str, Any]:
        state = self._teleop_mode_state()
        mode = str(state["mode"])
        inactive_arm = self._inactive_arm_for_mode(mode)
        runtime = self.runtime_parameters.snapshot()
        raw_poses = runtime.get("poses", {})
        available_hold_poses = {
            arm: sorted(
                str(name)
                for name in (
                    raw_poses.get(arm, {})
                    if isinstance(raw_poses, dict)
                    else {}
                )
            )
            for arm in ("left", "right")
        }
        selected_pose = (
            state["hold_pose_by_arm"].get(inactive_arm)
            if inactive_arm is not None
            else None
        )
        return {
            "operator_mode": mode,
            "active_arm": (
                "left" if mode == "left_only"
                else "right" if mode == "right_only"
                else "both"
            ),
            "inactive_arm": inactive_arm,
            "hold_pose": selected_pose,
            "hold_pose_by_arm": dict(state["hold_pose_by_arm"]),
            "available_hold_poses": available_hold_poses,
            "follow_hold": self._follow_hold_state(),
        }

    def _augment_teleop_status(self, status: dict[str, Any]) -> dict[str, Any]:
        augmented = dict(status)
        augmented.update(self._teleop_mode_snapshot())
        return augmented

    def teleop_status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "teleop": self._augment_teleop_status(
                self.teleop_launcher.status()
            ),
        }

    @staticmethod
    def _teleop_status_blocks_cartesian_jog(
        status: dict[str, Any],
    ) -> bool:
        follow = status.get("follow", {})
        # A latched stop such as ``web_console_shutdown_stop`` records why the
        # old Follow session was shut down.  It must not masquerade as a live
        # teleop owner after running/desired/busy/tmux have all become false.
        # The actual ownership signals below still block every real Follow
        # session and transition.
        return bool(
            status.get("running")
            or status.get("busy")
            or status.get("desired")
            or (
                isinstance(follow, dict)
                and follow.get("tmux")
            )
        )

    def _teleop_blocks_cartesian_jog(self) -> bool:
        return bool(
            self._teleop_status_blocks_cartesian_jog(
                self.teleop_launcher.status()
            )
            or self._system_follow_ownership_active()
            or (
                getattr(self, "trajectory_replay", None) is not None
                and self.trajectory_replay.status().get("active") is True
            )
            or
            (
                getattr(self, "act_rollout", None) is not None
                and self.act_rollout.status().get("active") is True
            )
        )

    @staticmethod
    def _system_follow_ownership_active() -> bool:
        """Detect the systemd UDP Follow stack used by port 9999.

        The legacy launcher status only observes its old tmux session.  The
        production standalone teleop stack instead owns a desired-state marker
        and refreshes two receiver readiness files.  Either signal must block
        the 8899 Cartesian controller.
        """

        if SYSTEM_FOLLOW_DESIRED_PATH.exists():
            return True
        now = time.time()
        for path in SYSTEM_FOLLOW_READY_PATHS:
            try:
                if path.is_file() and now - path.stat().st_mtime <= (
                    SYSTEM_FOLLOW_READY_MAX_AGE_S
                ):
                    return True
            except OSError:
                continue
        return False

    def _cartesian_jog_snapshot(
        self,
        teleop_status: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        snapshot = self.cartesian_jog.status()
        try:
            status = teleop_status or self.teleop_launcher.status()
            snapshot["teleop_running"] = (
                self._teleop_status_blocks_cartesian_jog(status)
                or self._system_follow_ownership_active()
            )
            snapshot["teleop_error"] = ""
        except Exception as exc:
            snapshot["teleop_running"] = None
            snapshot["teleop_error"] = (
                "teleop status unavailable: "
                f"{type(exc).__name__}: {exc}"
            )
        return snapshot

    def _right_arm_home_snapshot(
        self,
        teleop_status: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        snapshot = self.right_arm_home.status()
        try:
            status = teleop_status or self.teleop_launcher.status()
            snapshot["teleop_running"] = (
                self._teleop_status_blocks_cartesian_jog(status)
                or self._system_follow_ownership_active()
            )
            snapshot["teleop_error"] = ""
        except Exception as exc:
            snapshot["teleop_running"] = None
            snapshot["teleop_error"] = (
                "teleop status unavailable: "
                f"{type(exc).__name__}: {exc}"
            )
        return snapshot

    @staticmethod
    def _require_exact_payload(
        payload: dict[str, Any],
        expected_keys: frozenset[str],
    ) -> None:
        if frozenset(payload) != expected_keys:
            raise ValueError(
                "request fields must be exactly: "
                + (", ".join(sorted(expected_keys)) if expected_keys else "(none)")
            )

    def start_teleop(self, payload: dict[str, Any]) -> dict[str, Any]:
        base_fields = frozenset(
            {"confirm", "area_clear", "estop_ready", "initial_pose_aligned"}
        )
        extended_fields = base_fields | {"mode", "hold_pose"}
        if frozenset(payload) not in {base_fields, extended_fields}:
            raise ValueError(
                "request fields must be the four safety fields, optionally with "
                "mode and hold_pose"
            )
        if self.act_rollout.status().get("active") is True:
            raise TeleopLaunchConflict("stop ACT rollout before teleoperation")
        if self.trajectory_replay.status().get("active") is True:
            raise TeleopLaunchConflict("stop trajectory replay before teleoperation")
        if self.trajectory_recorder.status().get("active") is True:
            raise TeleopLaunchConflict("stop trajectory recording before teleoperation")
        current_mode = self._teleop_mode_state()
        mode = payload.get("mode", current_mode["mode"])
        if mode not in TELEOP_MODES:
            raise ValueError("mode must be dual, left_only or right_only")
        inactive_arm = self._inactive_arm_for_mode(str(mode))
        hold_pose = payload.get("hold_pose")
        if inactive_arm is not None:
            if hold_pose is None:
                hold_pose = current_mode["hold_pose_by_arm"].get(inactive_arm)
            if not isinstance(hold_pose, str) or not hold_pose:
                raise ValueError("single-arm mode requires hold_pose")
        elif hold_pose not in {None, ""}:
            raise ValueError("dual-arm mode does not use hold_pose")
        safety_payload = {name: payload[name] for name in base_fields}
        with self._motion_transition_lock:
            jog = self.cartesian_jog.status()
            if jog.get("enabled") or jog.get("busy"):
                raise TeleopLaunchConflict(
                    "disable left-arm Cartesian jog before starting teleoperation"
                )
            preposition_result = None
            if inactive_arm is not None:
                pose = self.runtime_parameters.pose(inactive_arm, str(hold_pose))
                controller = (
                    self.cartesian_jog
                    if inactive_arm == "left"
                    else self.right_arm_home
                )
                move_kwargs: dict[str, Any] = {
                    "pose_name": f"single_arm_hold:{hold_pose}",
                }
                if pose.get("gripper_position_m") is not None:
                    move_kwargs["gripper_position_m"] = pose["gripper_position_m"]
                preposition_result = controller.move_to_saved_joint_pose(
                    pose["joint_positions_rad"],
                    **move_kwargs,
                )
            next_mode = dict(current_mode)
            next_mode["mode"] = mode
            if inactive_arm is not None:
                next_mode["hold_pose_by_arm"][inactive_arm] = hold_pose
            next_mode.update(
                {
                    "version": 1,
                    "updated_at": time.time(),
                    "source": "medicine_packaging_console",
                }
            )
            _write_json_atomic(TELEOP_MODE_STATE_PATH, next_mode)
            self._write_follow_hold_state(inactive_arm=inactive_arm)
            result = self.teleop_launcher.start(safety_payload)
            # Any accepted Follow start invalidates a previously captured tool
            # orientation, because teleoperation is expected to change the pose.
            self.cartesian_jog.close()
            if isinstance(result.get("teleop"), dict):
                result["teleop"] = self._augment_teleop_status(result["teleop"])
            result["single_arm_preposition"] = preposition_result
            return result

    def stop_teleop(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.teleop_launcher.stop(payload)
        if isinstance(result.get("teleop"), dict):
            result["teleop"] = self._augment_teleop_status(result["teleop"])
        return result

    def hard_restart_teleop(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.act_rollout.status().get("active") is True:
            raise TeleopLaunchConflict("stop ACT rollout before hard restart")
        if self.trajectory_replay.status().get("active") is True:
            raise TeleopLaunchConflict("stop trajectory replay before hard restart")
        if self.trajectory_recorder.status().get("active") is True:
            raise TeleopLaunchConflict("stop trajectory recording before hard restart")
        with self._motion_transition_lock:
            jog = self.cartesian_jog.status()
            if jog.get("enabled") or jog.get("busy"):
                raise TeleopLaunchConflict(
                    "disable left-arm Cartesian jog before hard restart"
                )
            result = self.teleop_launcher.hard_restart(payload)
            # Rebuilding the arm SDKs invalidates feedback captured before the
            # restart, including the Cartesian jog orientation snapshot.
            self.cartesian_jog.close()
        if isinstance(result.get("teleop"), dict):
            result["teleop"] = self._augment_teleop_status(result["teleop"])
        return result

    def cartesian_jog_status(self) -> dict[str, Any]:
        return {"ok": True, "cartesian_jog": self._cartesian_jog_snapshot()}

    def suction_status(self) -> dict[str, Any]:
        return {"ok": True, "suction": self.suction.status()}

    def runtime_parameters_status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "runtime_parameters": self.runtime_parameters.snapshot(),
        }

    def read_current_pose(self, arm: str) -> dict[str, Any]:
        if arm == "left":
            controller = self.cartesian_jog
        elif arm == "right":
            controller = self.right_arm_home
        else:
            raise ValueError("arm must be left or right")
        # Named-pose teaching is a feedback-only operation and is expected to
        # happen while the operator is holding the robot through Follow.  This
        # exemption applies only to pose reads; all jog/home/motion interlocks
        # remain unchanged.
        return {
            "ok": True,
            "pose": controller.read_current_pose(allow_during_teleop=True),
        }

    def capture_runtime_pose(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_exact_payload(payload, frozenset({"arm", "name"}))
        arm = payload.get("arm")
        name = payload.get("name")
        if not isinstance(arm, str):
            raise ValueError("arm must be a string")
        if not isinstance(name, str):
            raise ValueError("name must be a string")
        pose = self.read_current_pose(arm)["pose"]
        if pose.get("gripper_position_m") is not None:
            try:
                gripper_position = float(pose["gripper_position_m"])
            except (TypeError, ValueError):
                gripper_position = float("nan")
            if not math.isfinite(gripper_position) or not 0.0 <= gripper_position <= 0.1:
                pose = dict(pose)
                pose.pop("gripper_position_m", None)
        snapshot = self.runtime_parameters.save_pose(
            arm,
            name,
            pose,
            source="captured_current_arm_pose",
        )
        return {
            "ok": True,
            "pose": pose,
            "saved_as": {"arm": arm, "name": name},
            "runtime_parameters": snapshot,
        }

    def move_to_runtime_pose(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_exact_payload(payload, frozenset({"arm", "name"}))
        arm = payload.get("arm")
        name = payload.get("name")
        if not isinstance(arm, str):
            raise ValueError("arm must be a string")
        if not isinstance(name, str):
            raise ValueError("name must be a string")
        pose = self.runtime_parameters.pose(arm, name)
        controller = self.cartesian_jog if arm == "left" else self.right_arm_home
        with self._motion_transition_lock:
            if self.trajectory_recorder.status().get("active") is True:
                raise CartesianJogConflict(
                    "stop trajectory recording before moving to a saved pose"
                )
            move_kwargs: dict[str, Any] = {"pose_name": name}
            if pose.get("gripper_position_m") is not None:
                move_kwargs["gripper_position_m"] = pose["gripper_position_m"]
            result = controller.move_to_saved_joint_pose(
                pose["joint_positions_rad"],
                **move_kwargs,
            )
        return {
            "ok": True,
            "moved_to": {"arm": arm, "name": name},
            "result": result,
        }

    def delete_runtime_pose(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_exact_payload(
            payload,
            frozenset({"arm", "name", "confirm_name"}),
        )
        arm = payload.get("arm")
        name = payload.get("name")
        confirm_name = payload.get("confirm_name")
        if not isinstance(arm, str):
            raise ValueError("arm must be a string")
        if not isinstance(name, str):
            raise ValueError("name must be a string")
        if not isinstance(confirm_name, str):
            raise ValueError("confirm_name must be a string")
        if confirm_name != name:
            raise ValueError("confirm_name must exactly match name")
        snapshot = self.runtime_parameters.delete_pose(
            arm,
            name,
            source="web_operator_delete_pose",
        )
        return {
            "ok": True,
            "deleted": {"arm": arm, "name": name},
            "runtime_parameters": snapshot,
        }

    def update_runtime_parameters(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_exact_payload(payload, frozenset({"task_id", "values"}))
        task_id = payload.get("task_id")
        values = payload.get("values")
        if not isinstance(task_id, str):
            raise ValueError("task_id must be a string")
        if not isinstance(values, dict):
            raise ValueError("values must be an object")
        snapshot = self.runtime_parameters.update_task(
            task_id,
            values,
            source="web_operator",
        )
        return {"ok": True, "runtime_parameters": snapshot}

    def capture_runtime_contact_z(self, payload: dict[str, Any]) -> dict[str, Any]:
        allowed = frozenset({"task_id"})
        allowed_with_layer = frozenset({"task_id", "layer"})
        if frozenset(payload) not in {allowed, allowed_with_layer}:
            raise ValueError("request fields must be task_id and optional layer")
        task_id = payload.get("task_id")
        if task_id not in {"task1", "task2", "task3"}:
            raise ValueError("task_id must be task1, task2 or task3")
        current_pose = self.read_current_pose("left")["pose"]
        contact_z = float(current_pose["position_m"][2])
        if task_id == "task1":
            layer = payload.get("layer")
            if layer not in (1, 2, 3):
                raise ValueError("task1 capture requires layer 1, 2 or 3")
            task_values = self.runtime_parameters.task("task1")
            layer_map = dict(task_values["contact_flange_z_m_by_layer"])
            layer_map[str(layer)] = contact_z
            values: dict[str, Any] = {
                "contact_flange_z_m_by_layer": layer_map,
            }
        else:
            if "layer" in payload:
                raise ValueError("layer is only valid for task1")
            values = {"contact_flange_z_m": contact_z}
        snapshot = self.runtime_parameters.update_task(
            task_id,
            values,
            source="captured_current_left_flange_z",
        )
        return {
            "ok": True,
            "captured": {
                "task_id": task_id,
                "layer": payload.get("layer"),
                "contact_flange_z_m": contact_z,
            },
            "runtime_parameters": snapshot,
        }

    def _effective_pick_cfg(self, task_id: str) -> dict[str, Any]:
        if task_id not in {"task1", "task2", "task3"}:
            raise ValueError("task_id must be task1, task2 or task3")
        base = getattr(self, f"{task_id}_pick_cfg")
        effective = dict(base)
        runtime_parameters = getattr(self, "runtime_parameters", None)
        if runtime_parameters is not None:
            effective.update(runtime_parameters.task(task_id))
        return effective

    def set_suction(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_exact_payload(payload, frozenset({"engaged"}))
        engaged = payload.get("engaged")
        if not isinstance(engaged, bool):
            raise ValueError("engaged must be a boolean")
        result = self.suction.set_engaged(engaged)
        return {
            "ok": True,
            "result": result,
            "suction": self.suction.status(),
        }

    def sync_suction_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_exact_payload(
            payload,
            frozenset({"engaged", "source", "updated_at"}),
        )
        state = self.suction.sync_state(
            payload.get("engaged"),
            source=payload.get("source"),
            updated_at=payload.get("updated_at"),
        )
        return {"ok": True, "state": state, "suction": self.suction.status()}

    def task1_pick_status(self) -> dict[str, Any]:
        pick_cfg = self._effective_pick_cfg("task1")
        jog = self._cartesian_jog_snapshot()
        suction = self.suction.status()
        ready = bool(
            self.task1_pick_enabled
            and self.task1_pick_calibration is not None
            and not self.task1_pick_error
            and jog.get("available") is True
            and suction.get("available") is True
        )
        return {
            "enabled": self.task1_pick_enabled,
            "ready": ready,
            "requires_jog_enabled": True,
            "jog_enabled": jog.get("enabled") is True,
            "suction_available": suction.get("available") is True,
            "tcp_ready": self.task1_pick_calibration is not None,
            "transit_z_m": float(
                pick_cfg.get("transit_z_m", 0.10)
            ),
            "pre_contact_clearance_m": float(
                pick_cfg.get("pre_contact_clearance_m", 0.025)
            ),
            "test_lift_m": float(
                pick_cfg.get("test_lift_m", 0.02)
            ),
            "height_mode": "vision_layer_to_fixed_z",
            "contact_flange_z_m_by_layer": dict(
                pick_cfg.get(
                    "contact_flange_z_m_by_layer",
                    {},
                )
            ),
            "error": self.task1_pick_error,
        }

    def task2_pick_status(self) -> dict[str, Any]:
        pick_cfg = self._effective_pick_cfg("task2")
        jog = self._cartesian_jog_snapshot()
        suction = self.suction.status()
        calibration = self.task2_pick_calibration or {}
        contact = calibration.get("contact_sample", {})
        ready = bool(
            self.task2_pick_enabled
            and self.task2_pick_calibration is not None
            and not self.task2_pick_error
            and jog.get("available") is True
            and suction.get("available") is True
        )
        return {
            "enabled": self.task2_pick_enabled,
            "ready": ready,
            "requires_jog_enabled": True,
            "jog_enabled": jog.get("enabled") is True,
            "suction_available": suction.get("available") is True,
            "tcp_ready": self.task2_pick_calibration is not None,
            "transit_z_m": float(pick_cfg.get("transit_z_m", 0.10)),
            "pre_contact_clearance_m": float(
                pick_cfg.get("pre_contact_clearance_m", 0.025)
            ),
            "test_lift_m": float(pick_cfg.get("test_lift_m", 0.02)),
            "height_mode": "sampled_single_carton_fixed_z",
            "contact_flange_z_m": pick_cfg.get(
                "contact_flange_z_m",
                contact.get("absolute_contact_flange_z_m"),
            ),
            "surface_z_m": (
                contact.get("carton_surface_center_in_base_m", [None, None, None])[2]
                if isinstance(contact.get("carton_surface_center_in_base_m"), list)
                and len(contact.get("carton_surface_center_in_base_m")) >= 3
                else None
            ),
            "selection_policy": "leftmost_image_x_valid_task2_front_carton",
            "error": self.task2_pick_error,
        }

    def task3_pick_status(self) -> dict[str, Any]:
        pick_cfg = self._effective_pick_cfg("task3")
        jog = self._cartesian_jog_snapshot()
        suction = self.suction.status()
        calibration = self.task3_pick_calibration or {}
        contact = calibration.get("contact_sample", {})
        offset = contact.get("surface_to_target_flange_offset_in_base_m")
        table_surface_z = self.task3_pick_cfg.get("table_surface_z_m")
        contact_flange_z = pick_cfg.get("contact_flange_z_m")
        try:
            if contact_flange_z is None:
                candidate_contact_z = float(table_surface_z) + float(offset[2])
            else:
                candidate_contact_z = float(contact_flange_z)
            if math.isfinite(candidate_contact_z):
                contact_flange_z = candidate_contact_z
        except (IndexError, TypeError, ValueError):
            contact_flange_z = None
        ready = bool(
            self.task3_pick_enabled
            and self.task3_pick_calibration is not None
            and not self.task3_pick_error
            and contact_flange_z is not None
            and jog.get("available") is True
            and suction.get("available") is True
        )
        return {
            "enabled": self.task3_pick_enabled,
            "ready": ready,
            "requires_jog_enabled": True,
            "jog_enabled": jog.get("enabled") is True,
            "suction_available": suction.get("available") is True,
            "tcp_ready": self.task3_pick_calibration is not None,
            "transit_z_m": float(pick_cfg.get("transit_z_m", 0.10)),
            "pre_contact_clearance_m": float(
                pick_cfg.get("pre_contact_clearance_m", 0.025)
            ),
            "test_lift_m": float(pick_cfg.get("test_lift_m", 0.02)),
            "height_mode": "table_surface_plus_tcp_offset",
            "table_surface_z_m": table_surface_z,
            "surface_to_flange_z_offset_m": (
                None
                if not isinstance(offset, list) or len(offset) < 3
                else offset[2]
            ),
            "contact_flange_z_m": contact_flange_z,
            "selection_policy": "nearest_left_base_flat_carton_panel",
            "error": self.task3_pick_error,
        }

    def task1_pick_skill_descriptor(self) -> dict[str, Any]:
        """Machine-readable contract for the bounded Task-1 pick skill."""

        return {
            "ok": True,
            "skill": {
                "id": "task1.pick_carton",
                "version": "1.0.0",
                "method": "POST",
                "endpoint": "/api/skills/task1/pick-carton",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
                "selection_policy": (
                    "highest_layer_then_nearest_left_base_xy"
                ),
                "effect": (
                    "lock current calibrated downward pose, detect one carton, "
                    "approach, engage both suction cups, and test-lift 100 mm"
                ),
                "postcondition": "one carton held 100 mm above contact height",
                "ready": self.task1_pick_status().get("ready") is True,
            },
        }

    def task2_pick_skill_descriptor(self) -> dict[str, Any]:
        """Machine-readable contract for the bounded Task-2 carton pick."""

        return {
            "ok": True,
            "skill": {
                "id": "task2.pick_carton",
                "version": "1.0.0",
                "method": "POST",
                "endpoint": "/api/skills/task2/pick-carton",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
                "selection_policy": (
                    "leftmost_image_x_valid_task2_front_carton"
                ),
                "effect": (
                    "detect the leftmost safe Task-2 front carton, approach at the "
                    "sampled fixed contact height, engage both suction cups, "
                    "and test-lift 20 mm"
                ),
                "postcondition": "one Task-2 carton held 20 mm above contact height",
                "ready": self.task2_pick_status().get("ready") is True,
            },
        }

    def task3_pick_skill_descriptor(self) -> dict[str, Any]:
        """Machine-readable contract for the bounded Task-3 flat-carton pick."""

        return {
            "ok": True,
            "skill": {
                "id": "task3.pick_flat_carton",
                "version": "1.0.0",
                "method": "POST",
                "endpoint": "/api/skills/task3/pick-flat-carton",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
                "selection_policy": "nearest_left_base_flat_carton_panel",
                "effect": (
                    "detect one Task-3 flat carton, approach at the configured "
                    "table height plus calibrated suction TCP offset, engage "
                    "both suction cups, and test-lift 20 mm"
                ),
                "postcondition": (
                    "one Task-3 flat carton held 20 mm above table contact height"
                ),
                "ready": self.task3_pick_status().get("ready") is True,
            },
        }

    def task3_expand_skill_descriptor(self) -> dict[str, Any]:
        """Describe Task3 step 3's lift, expand-pose, and replay sequence."""

        cfg = getattr(self, "task3_expand_cfg", {})
        jog = self._cartesian_jog_snapshot()
        suction = self.suction.status()
        replay = self.trajectory_replay.status()
        enabled = cfg.get("enabled") is True
        expand_pose_name = str(cfg.get("expand_pose_name", "expand_box"))
        trajectory_path = Path(str(cfg.get("trajectory_path", ""))).expanduser()
        pose_ready = True
        try:
            self.runtime_parameters.pose("left", expand_pose_name)
        except ValueError:
            pose_ready = False
        return {
            "ok": True,
            "skill": {
                "id": "task3.expand_carton",
                "version": "1.0.0",
                "method": "POST",
                "endpoint": "/api/skills/task3/expand-carton",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
                "effect": (
                    "keep Task3 suction engaged, raise the left arm vertically "
                    f"to the safe height, move to left.{expand_pose_name}, then "
                    f"replay {trajectory_path} at the recorded speed"
                ),
                "postcondition": (
                    "the configured expand-box trajectory has completed while "
                    "Task3 suction remains engaged"
                ),
                "expand_pose": f"left.{expand_pose_name}",
                "trajectory_path": str(trajectory_path),
                "ready": bool(
                    enabled
                    and self.task3_pick_calibration is not None
                    and pose_ready
                    and trajectory_path.is_dir()
                    and replay.get("enabled") is True
                    and replay.get("active") is not True
                    and jog.get("busy") is not True
                    and suction.get("available") is True
                    and suction.get("engaged") is True
                ),
            },
        }

    def task2_reset_both_skill_descriptor(self) -> dict[str, Any]:
        """Describe Task2's parallel, configurable-speed home reset."""

        left = self._cartesian_jog_snapshot()
        right = self._right_arm_home_snapshot()
        return {
            "ok": True,
            "skill": {
                "id": "task2.reset_both_home",
                "version": "1.0.0",
                "method": "POST",
                "endpoint": "/api/skills/task2/reset-both-home",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
                "execution": "parallel",
                "speed_profile": self.task2_joint_speed_profile,
                "ready": bool(
                    left.get("home_joint_pose", {}).get("available") is True
                    and right.get("home_joint_pose", {}).get("available") is True
                    and left.get("busy") is not True
                    and right.get("busy") is not True
                ),
            },
        }

    def run_task2_reset_both_skill(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Reset both arms concurrently and wait for both feedback checks."""

        self._require_exact_payload(payload, frozenset())
        started_at = time.time()
        started_monotonic = time.monotonic()
        with self._motion_transition_lock:
            if self.trajectory_recorder.status().get("active") is True:
                raise CartesianJogConflict(
                    "stop trajectory recording before resetting both arms"
                )
            with ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="task2-reset",
            ) as executor:
                futures = {
                    "left": executor.submit(
                        self.cartesian_jog.reset_home,
                        speed_profile=self.task2_joint_speed_profile,
                    ),
                    "right": executor.submit(
                        self.right_arm_home.reset_home,
                        speed_profile=self.task2_joint_speed_profile,
                    ),
                }
                results: dict[str, Any] = {}
                errors: list[Exception] = []
                for arm, future in futures.items():
                    try:
                        results[arm] = future.result()
                    except Exception as exc:  # propagate after both settle
                        errors.append(exc)
                if errors:
                    raise errors[0]
        return {
            "ok": True,
            "result": {
                "operation": "task2_reset_both_home",
                "execution": "parallel",
                "speed_profile": self.task2_joint_speed_profile,
                "arms": results,
            },
            "skill": {
                "id": "task2.reset_both_home",
                "version": "1.0.0",
                "status": "succeeded",
                "started_at": started_at,
                "duration_ms": (time.monotonic() - started_monotonic) * 1000.0,
                "affected_arms": ["left", "right"],
            },
            "cartesian_jog": self._cartesian_jog_snapshot(),
            "right_arm_home": self._right_arm_home_snapshot(),
        }

    def task2_ready_poses_skill_descriptor(self) -> dict[str, Any]:
        """Describe Task2's post-pick left/right ready-pose transition."""

        pose_ready = True
        try:
            self.runtime_parameters.pose("left", self.task2_left_ready_pose)
            self.runtime_parameters.pose("right", self.task2_right_ready_pose)
        except ValueError:
            pose_ready = False
        return {
            "ok": True,
            "skill": {
                "id": "task2.move_ready_poses",
                "version": "1.0.0",
                "method": "POST",
                "endpoint": "/api/skills/task2/move-ready-poses",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
                "speed_profile": self.task2_joint_speed_profile,
                "execution": "parallel",
                "targets": {
                    "left": self.task2_left_ready_pose,
                    "right": self.task2_right_ready_pose,
                },
                "ready": pose_ready,
            },
        }

    @staticmethod
    def _task2_subtask_init_pose_names(subtask_id: int) -> tuple[str, str]:
        if subtask_id not in {2, 3}:
            raise ValueError("Task2 subtask init pose id must be 2 or 3")
        return (
            f"subtask{subtask_id}_left_init",
            f"subtask{subtask_id}_right_init",
        )

    def task2_subtask_init_poses_skill_descriptor(
        self,
        subtask_id: int,
    ) -> dict[str, Any]:
        """Describe a fixed Task2 inter-ACT dual-arm pose transition."""

        left_name, right_name = self._task2_subtask_init_pose_names(subtask_id)
        pose_ready = True
        try:
            self.runtime_parameters.pose("left", left_name)
            self.runtime_parameters.pose("right", right_name)
        except ValueError:
            pose_ready = False
        return {
            "ok": True,
            "skill": {
                "id": f"task2.move_subtask{subtask_id}_init_poses",
                "version": "1.0.0",
                "method": "POST",
                "endpoint": (
                    f"/api/skills/task2/move-subtask{subtask_id}-init-poses"
                ),
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
                "speed_profile": self.task2_joint_speed_profile,
                "execution": "parallel",
                "targets": {"left": left_name, "right": right_name},
                "ready": pose_ready,
            },
        }

    def task1_watcher_pose_skill_descriptor(self) -> dict[str, Any]:
        """Describe Task1's left-first pre-recognition arm transition."""

        pose_ready = True
        try:
            self.runtime_parameters.pose("left", "left_watcher")
        except ValueError:
            pose_ready = False
        right = self._right_arm_home_snapshot()
        return {
            "ok": True,
            "skill": {
                "id": "task1.move_watcher_pose",
                "version": "1.0.0",
                "method": "POST",
                "endpoint": "/api/skills/task1/move-watcher-pose",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
                "execution": "parallel",
                "targets": {
                    "left": "left_watcher",
                    "right": "system_initial_pose",
                },
                "ready": bool(
                    pose_ready
                    and right.get("home_joint_pose", {}).get("available") is True
                    and right.get("busy") is not True
                ),
            },
        }

    def task2_watcher_pose_skill_descriptor(self) -> dict[str, Any]:
        """Describe Task2's standalone pre-recognition arm transition."""

        pose_ready = True
        try:
            self.runtime_parameters.pose("left", "left_watcher")
        except ValueError:
            pose_ready = False
        right = self._right_arm_home_snapshot()
        return {
            "ok": True,
            "skill": {
                "id": "task2.move_watcher_pose",
                "version": "1.0.0",
                "method": "POST",
                "endpoint": "/api/skills/task2/move-watcher-pose",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
                "targets": {
                    "left": "left_watcher",
                    "right": "system_initial_pose",
                },
                "ready": bool(
                    pose_ready
                    and right.get("home_joint_pose", {}).get("available") is True
                    and right.get("busy") is not True
                ),
            },
        }

    def task3_watcher_pose_skill_descriptor(self) -> dict[str, Any]:
        """Describe Task3's standalone pre-recognition arm transition."""

        pose_ready = True
        try:
            self.runtime_parameters.pose("left", "left_box_watcher")
        except ValueError:
            pose_ready = False
        right = self._right_arm_home_snapshot()
        return {
            "ok": True,
            "skill": {
                "id": "task3.move_watcher_pose",
                "version": "1.0.0",
                "method": "POST",
                "endpoint": "/api/skills/task3/move-watcher-pose",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
                "targets": {
                    "left": "left_box_watcher",
                    "right": "system_initial_pose",
                },
                "ready": bool(
                    pose_ready
                    and right.get("home_joint_pose", {}).get("available") is True
                    and right.get("busy") is not True
                ),
            },
        }

    def _set_task2_workflow_status(
        self,
        *,
        state: str,
        step_index: int,
        stage: str,
        message: str,
        error: str | None = None,
    ) -> None:
        self._task2_workflow_status = {
            "state": state,
            "step_index": step_index,
            "step_number": step_index + 1,
            "stage": stage,
            "message": message,
            "error": error,
            "updated_at": time.time(),
        }

    def task2_workflow_status(self) -> dict[str, Any]:
        return copy.deepcopy(
            getattr(
                self,
                "_task2_workflow_status",
                {
                    "state": "idle",
                    "step_index": None,
                    "stage": None,
                    "message": "Task2 workflow has not started",
                    "error": None,
                    "updated_at": None,
                },
            )
        )

    @staticmethod
    def _require_executed_motion(result: Any, label: str) -> None:
        if not isinstance(result, dict) or result.get("executed") is not True:
            raise CartesianJogSafetyViolation(
                f"{label} did not return executed=true feedback"
            )

    def run_task2_watcher_pose_skill(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Reset right home, then move only the left arm to left_watcher."""

        self._require_exact_payload(payload, frozenset())
        started_at = time.time()
        started_monotonic = time.monotonic()
        self._set_task2_workflow_status(
            state="running",
            step_index=1,
            stage="right_home",
            message="Task2 step 2: resetting right arm to system initial pose",
        )
        try:
            with self._motion_transition_lock:
                if self.trajectory_recorder.status().get("active") is True:
                    raise CartesianJogConflict(
                        "stop trajectory recording before Task2 watcher motion"
                    )
                right_result = self.right_arm_home.reset_home(
                    speed_profile=self.task2_joint_speed_profile,
                )
                self._require_executed_motion(right_result, "Task2 step 2 right arm")
                self._set_task2_workflow_status(
                    state="running",
                    step_index=1,
                    stage="left_watcher",
                    message=(
                        "Task2 step 2: right arm verified; moving left arm to "
                        "left_watcher"
                    ),
                )
                watcher_pose = self.runtime_parameters.pose("left", "left_watcher")
                left_result = self.cartesian_jog.move_to_saved_joint_pose(
                    watcher_pose["joint_positions_rad"],
                    pose_name="left_watcher",
                    speed_profile=self.task2_joint_speed_profile,
                )
                self._require_executed_motion(left_result, "Task2 step 2 left arm")
        except Exception as exc:
            self._set_task2_workflow_status(
                state="failed",
                step_index=1,
                stage="failed",
                message="Task2 step 2 stopped before both arm checks completed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._set_task2_workflow_status(
            state="succeeded",
            step_index=1,
            stage="feedback_verified",
            message="Task2 step 2 complete: right home and left_watcher verified",
        )
        return {
            "ok": True,
            "result": {
                "operation": "task2_move_watcher_pose",
                "execution": "right_then_left",
                "speed_profile": self.task2_joint_speed_profile,
                "targets": {
                    "left": "left_watcher",
                    "right": "system_initial_pose",
                },
                "motions": {"right": right_result, "left": left_result},
            },
            "skill": {
                "id": "task2.move_watcher_pose",
                "version": "1.0.0",
                "status": "succeeded",
                "started_at": started_at,
                "duration_ms": (time.monotonic() - started_monotonic) * 1000.0,
                "affected_arms": ["right", "left"],
            },
            "cartesian_jog": self._cartesian_jog_snapshot(),
            "right_arm_home": self._right_arm_home_snapshot(),
            "task2_workflow": self.task2_workflow_status(),
        }

    def _move_task1_watcher_and_right_home_parallel(self) -> dict[str, Any]:
        """Move both Task1 arms concurrently and verify both motion replies."""

        watcher_pose = self.runtime_parameters.pose("left", "left_watcher")
        with ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="task1-watcher",
        ) as executor:
            futures = {
                "left": executor.submit(
                    self.cartesian_jog.move_to_saved_joint_pose,
                    watcher_pose["joint_positions_rad"],
                    pose_name="left_watcher",
                    speed_profile="DEFAULT",
                ),
                "right": executor.submit(
                    self.right_arm_home.reset_home,
                    speed_profile=self.task2_joint_speed_profile,
                ),
            }
            results: dict[str, Any] = {}
            errors: list[Exception] = []
            for arm, future in futures.items():
                try:
                    results[arm] = future.result()
                except Exception as exc:  # wait for both arms to settle
                    errors.append(exc)
            if errors:
                raise errors[0]
        for arm, result in results.items():
            self._require_executed_motion(result, f"Task1 step 1 {arm} arm")
        return {
            "operation": "task1_move_watcher_pose",
            "execution": "parallel",
            "speed_profile": "DEFAULT",
            "targets": {
                "left": "left_watcher",
                "right": "system_initial_pose",
            },
            "motions": results,
        }

    def run_task1_watcher_pose_skill(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Move left to left_watcher while resetting right to system home."""

        self._require_exact_payload(payload, frozenset())
        started_at = time.time()
        started_monotonic = time.monotonic()
        with self._motion_transition_lock:
            if self.trajectory_recorder.status().get("active") is True:
                raise CartesianJogConflict(
                    "stop trajectory recording before Task1 watcher motion"
                )
            result = self._move_task1_watcher_and_right_home_parallel()
        return {
            "ok": True,
            "result": result,
            "skill": {
                "id": "task1.move_watcher_pose",
                "version": "1.0.0",
                "status": "succeeded",
                "started_at": started_at,
                "duration_ms": (time.monotonic() - started_monotonic) * 1000.0,
                "affected_arms": ["left", "right"],
            },
            "cartesian_jog": self._cartesian_jog_snapshot(),
            "right_arm_home": self._right_arm_home_snapshot(),
        }

    def task1_fixed_trajectory_place_skill_descriptor(self) -> dict[str, Any]:
        """Describe Task1 step 4's dual-pose transition and fixed replay."""

        ready = self.task1_fixed_place_enabled
        preflight: dict[str, Any] | None = None
        error = ""
        try:
            self.runtime_parameters.pose(
                "left", self.task1_fixed_place_left_pose
            )
            self.runtime_parameters.pose(
                "right", self.task1_fixed_place_right_pose
            )
            preflight = self.trajectory_replay.preflight(
                self.task1_fixed_place_recording_id
            )["preflight"]
        except Exception as exc:
            ready = False
            error = f"{type(exc).__name__}: {exc}"
        return {
            "ok": True,
            "skill": {
                "id": "task1.place_carton_fixed_trajectory",
                "version": "1.0.0",
                "method": "POST",
                "endpoint": "/api/skills/task1/place-carton-fixed-trajectory",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
                "execution": "parallel_poses_then_replay",
                "targets": {
                    "left": self.task1_fixed_place_left_pose,
                    "right": self.task1_fixed_place_right_pose,
                },
                "recording_id": self.task1_fixed_place_recording_id,
                "ready": ready,
                "preflight": preflight,
                "error": error,
            },
        }

    def run_task1_fixed_trajectory_place_skill(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Move both zhuangxiang poses concurrently, then replay the fixed path."""

        self._require_exact_payload(payload, frozenset())
        if not self.task1_fixed_place_enabled:
            raise ReplayUnavailable("Task1 fixed-trajectory placement is disabled")
        started_at = time.time()
        started_monotonic = time.monotonic()
        recording_id = self.task1_fixed_place_recording_id
        stages: list[dict[str, Any]] = []
        with self._motion_transition_lock:
            preflight = self.trajectory_replay.preflight(recording_id)["preflight"]
            if preflight.get("arms") != ["left", "right"]:
                raise ReplayUnavailable(
                    "Task1 fixed placement trajectory must contain both arms"
                )
            if preflight.get("max_tracking_error_rad") is not None:
                raise ReplayUnavailable(
                    "Task1 fixed placement trajectory tracking limit was not removed"
                )
            if preflight.get("max_gripper_tracking_error_m") is not None:
                raise ReplayUnavailable(
                    "Task1 fixed placement gripper tracking limit was not removed"
                )
            suction_before = self.suction.status()
            if suction_before.get("available") is not True:
                raise SuctionUnavailable(
                    suction_before.get("error") or "suction is unavailable"
                )
            if suction_before.get("engaged") is not True:
                raise CartesianJogSafetyViolation(
                    "Task1 step 4 requires suction engaged before placement"
                )
            blocker = self._trajectory_replay_blocker(
                allow_suction_engaged=True
            )
            if blocker:
                raise ReplayConflict(blocker)
            left_pose = self.runtime_parameters.pose(
                "left", self.task1_fixed_place_left_pose
            )
            right_pose = self.runtime_parameters.pose(
                "right", self.task1_fixed_place_right_pose
            )
            right_move_kwargs: dict[str, Any] = {
                "pose_name": self.task1_fixed_place_right_pose,
                "speed_profile": self.task1_fixed_place_speed_profile,
            }
            if right_pose.get("gripper_position_m") is not None:
                right_move_kwargs["gripper_position_m"] = right_pose[
                    "gripper_position_m"
                ]
            with ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="task1-fixed-place",
            ) as executor:
                futures = {
                    "left": executor.submit(
                        self.cartesian_jog.move_to_saved_joint_pose,
                        left_pose["joint_positions_rad"],
                        pose_name=self.task1_fixed_place_left_pose,
                        speed_profile=self.task1_fixed_place_speed_profile,
                    ),
                    "right": executor.submit(
                        self.right_arm_home.move_to_saved_joint_pose,
                        right_pose["joint_positions_rad"],
                        **right_move_kwargs,
                    ),
                }
                motions: dict[str, Any] = {}
                errors: list[Exception] = []
                for arm, future in futures.items():
                    try:
                        motions[arm] = future.result()
                    except Exception as exc:  # wait for both arms to settle
                        errors.append(exc)
                if errors:
                    raise errors[0]
            for arm, result in motions.items():
                self._require_executed_motion(
                    result,
                    f"Task1 step 4 {arm} zhuangxiang pose",
                )
            stages.append(
                {
                    "name": "move_both_to_zhuangxiang",
                    "status": "completed",
                    "execution": "parallel",
                }
            )
            suction_at_replay = self.suction.status()
            if suction_at_replay.get("engaged") is not True:
                raise CartesianJogSafetyViolation(
                    "Task1 suction released before fixed trajectory replay"
                )
            replay_started = self.trajectory_replay.start(
                recording_id=recording_id,
                confirmation=recording_id,
                allow_suction_engaged=True,
            )
            replay_completed = self.trajectory_replay.wait(
                recording_id,
                timeout_s=self.task1_fixed_place_replay_timeout_s,
            )
            if replay_completed.get("state") != "completed":
                raise ReplayUnavailable(
                    "Task1 fixed placement trajectory did not complete"
                )
            if replay_completed.get("suction_release_state") != "released":
                raise ReplayUnavailable(
                    "Task1 fixed placement did not confirm suction release"
                )
            stages.append(
                {
                    "name": "replay_zhuangxiang_trajectory",
                    "status": "completed",
                    "speed_scale": replay_completed.get("speed_scale"),
                }
            )
        return {
            "ok": True,
            "result": {
                "operation": "task1_place_carton_fixed_trajectory",
                "execution": "parallel_poses_then_replay",
                "targets": {
                    "left": self.task1_fixed_place_left_pose,
                    "right": self.task1_fixed_place_right_pose,
                },
                "motions": motions,
                "recording_id": recording_id,
                "replay_started": replay_started,
                "replay_completed": replay_completed,
            },
            "skill": {
                "id": "task1.place_carton_fixed_trajectory",
                "version": "1.0.0",
                "status": "succeeded",
                "started_at": started_at,
                "duration_ms": (time.monotonic() - started_monotonic) * 1000.0,
                "affected_arms": ["left", "right"],
                "stages": stages,
            },
            "cartesian_jog": self._cartesian_jog_snapshot(),
            "right_arm_home": self._right_arm_home_snapshot(),
            "suction": self.suction.status(),
            "replay": replay_completed,
        }

    def run_task3_watcher_pose_skill(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Reset right home, then move only the left arm to left_box_watcher."""

        self._require_exact_payload(payload, frozenset())
        started_at = time.time()
        started_monotonic = time.monotonic()
        with self._motion_transition_lock:
            if self.trajectory_recorder.status().get("active") is True:
                raise CartesianJogConflict(
                    "stop trajectory recording before Task3 watcher motion"
                )
            right_result = self.right_arm_home.reset_home(
                speed_profile=self.task2_joint_speed_profile,
            )
            self._require_executed_motion(right_result, "Task3 step 1 right arm")
            watcher_pose = self.runtime_parameters.pose(
                "left", "left_box_watcher"
            )
            left_result = self.cartesian_jog.move_to_saved_joint_pose(
                watcher_pose["joint_positions_rad"],
                pose_name="left_box_watcher",
                speed_profile=self.task2_joint_speed_profile,
            )
            self._require_executed_motion(left_result, "Task3 step 1 left arm")
        return {
            "ok": True,
            "result": {
                "operation": "task3_move_watcher_pose",
                "execution": "right_then_left",
                "speed_profile": self.task2_joint_speed_profile,
                "targets": {
                    "left": "left_box_watcher",
                    "right": "system_initial_pose",
                },
                "motions": {"right": right_result, "left": left_result},
            },
            "skill": {
                "id": "task3.move_watcher_pose",
                "version": "1.0.0",
                "status": "succeeded",
                "started_at": started_at,
                "duration_ms": (time.monotonic() - started_monotonic) * 1000.0,
                "affected_arms": ["right", "left"],
            },
            "cartesian_jog": self._cartesian_jog_snapshot(),
            "right_arm_home": self._right_arm_home_snapshot(),
        }

    def run_task2_ready_poses_skill(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Move both arms concurrently to the Task2 ready poses."""

        self._require_exact_payload(payload, frozenset())
        suction_before = self.suction.status()
        if suction_before.get("available") is not True:
            raise SuctionUnavailable(
                suction_before.get("error") or "suction is unavailable"
            )
        started_at = time.time()
        started_monotonic = time.monotonic()
        self._set_task2_workflow_status(
            state="running",
            step_index=4,
            stage="parallel_ready_poses",
            message=(
                "Task2 step 5: moving left to paper_init and right to "
                "init_pose in parallel"
            ),
        )
        try:
            with self._motion_transition_lock:
                if self.trajectory_recorder.status().get("active") is True:
                    raise CartesianJogConflict(
                        "stop trajectory recording before Task2 ready-pose motion"
                    )
                right_pose = self.runtime_parameters.pose(
                    "right",
                    self.task2_right_ready_pose,
                )
                left_pose = self.runtime_parameters.pose(
                    "left",
                    self.task2_left_ready_pose,
                )
                right_move_kwargs: dict[str, Any] = {
                    "pose_name": self.task2_right_ready_pose,
                    "speed_profile": self.task2_joint_speed_profile,
                }
                if right_pose.get("gripper_position_m") is not None:
                    right_move_kwargs["gripper_position_m"] = right_pose[
                        "gripper_position_m"
                    ]
                left_move_kwargs: dict[str, Any] = {
                    "pose_name": self.task2_left_ready_pose,
                    "speed_profile": self.task2_joint_speed_profile,
                }
                if left_pose.get("gripper_position_m") is not None:
                    left_move_kwargs["gripper_position_m"] = left_pose[
                        "gripper_position_m"
                    ]
                with ThreadPoolExecutor(
                    max_workers=2,
                    thread_name_prefix="task2-ready",
                ) as executor:
                    futures = {
                        "right": executor.submit(
                            self.right_arm_home.move_to_saved_joint_pose,
                            right_pose["joint_positions_rad"],
                            **right_move_kwargs,
                        ),
                        "left": executor.submit(
                            self.cartesian_jog.move_to_saved_joint_pose,
                            left_pose["joint_positions_rad"],
                            **left_move_kwargs,
                        ),
                    }
                    results: dict[str, Any] = {}
                    errors: list[Exception] = []
                    for arm, future in futures.items():
                        try:
                            results[arm] = future.result()
                        except Exception as exc:  # wait for both arms to settle
                            errors.append(exc)
                    if errors:
                        raise errors[0]
                for arm, result in results.items():
                    self._require_executed_motion(
                        result,
                        f"Task2 step 5 {arm} arm",
                    )
        except Exception as exc:
            self._set_task2_workflow_status(
                state="failed",
                step_index=4,
                stage="failed",
                message="Task2 step 5 did not complete both parallel motions",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._set_task2_workflow_status(
            state="succeeded",
            step_index=4,
            stage="feedback_verified",
            message="Task2 step 5 complete: both ready poses verified",
        )
        return {
            "ok": True,
            "result": {
                "operation": "task2_move_ready_poses",
                "execution": "parallel",
                "speed_profile": self.task2_joint_speed_profile,
                "targets": {
                    "right": self.task2_right_ready_pose,
                    "left": self.task2_left_ready_pose,
                },
                "motions": {
                    "right": results["right"],
                    "left": results["left"],
                },
            },
            "skill": {
                "id": "task2.move_ready_poses",
                "version": "1.0.0",
                "status": "succeeded",
                "started_at": started_at,
                "duration_ms": (time.monotonic() - started_monotonic) * 1000.0,
                "affected_arms": ["right", "left"],
            },
            "cartesian_jog": self._cartesian_jog_snapshot(),
            "right_arm_home": self._right_arm_home_snapshot(),
            "suction": self.suction.status(),
            "task2_workflow": self.task2_workflow_status(),
        }

    def run_task2_subtask_init_poses_skill(
        self,
        payload: dict[str, Any],
        *,
        subtask_id: int,
    ) -> dict[str, Any]:
        """Move both arms to a fixed inter-ACT dataset initial pose pair."""

        self._require_exact_payload(payload, frozenset())
        left_name, right_name = self._task2_subtask_init_pose_names(subtask_id)
        step_index = 6 if subtask_id == 2 else 8
        step_number = step_index + 1
        suction_before = self.suction.status()
        if suction_before.get("available") is not True:
            raise SuctionUnavailable(
                suction_before.get("error") or "suction is unavailable"
            )
        started_at = time.time()
        started_monotonic = time.monotonic()
        stage = f"parallel_subtask{subtask_id}_init_poses"
        self._set_task2_workflow_status(
            state="running",
            step_index=step_index,
            stage=stage,
            message=(
                f"Task2 step {step_number}: moving left to {left_name} and "
                f"right to {right_name} in parallel"
            ),
        )
        try:
            with self._motion_transition_lock:
                if self.trajectory_recorder.status().get("active") is True:
                    raise CartesianJogConflict(
                        "stop trajectory recording before Task2 inter-ACT "
                        "pose motion"
                    )
                left_pose = self.runtime_parameters.pose("left", left_name)
                right_pose = self.runtime_parameters.pose("right", right_name)
                left_move_kwargs: dict[str, Any] = {
                    "pose_name": left_name,
                    "speed_profile": self.task2_joint_speed_profile,
                }
                right_move_kwargs: dict[str, Any] = {
                    "pose_name": right_name,
                    "speed_profile": self.task2_joint_speed_profile,
                }
                if left_pose.get("gripper_position_m") is not None:
                    left_move_kwargs["gripper_position_m"] = left_pose[
                        "gripper_position_m"
                    ]
                if right_pose.get("gripper_position_m") is not None:
                    right_move_kwargs["gripper_position_m"] = right_pose[
                        "gripper_position_m"
                    ]
                with ThreadPoolExecutor(
                    max_workers=2,
                    thread_name_prefix=f"task2-subtask{subtask_id}-init",
                ) as executor:
                    futures = {
                        "left": executor.submit(
                            self.cartesian_jog.move_to_saved_joint_pose,
                            left_pose["joint_positions_rad"],
                            **left_move_kwargs,
                        ),
                        "right": executor.submit(
                            self.right_arm_home.move_to_saved_joint_pose,
                            right_pose["joint_positions_rad"],
                            **right_move_kwargs,
                        ),
                    }
                    results: dict[str, Any] = {}
                    errors: list[Exception] = []
                    for arm, future in futures.items():
                        try:
                            results[arm] = future.result()
                        except Exception as exc:  # wait for both arms to settle
                            errors.append(exc)
                    if errors:
                        raise errors[0]
                for arm, result in results.items():
                    self._require_executed_motion(
                        result,
                        f"Task2 step {step_number} {arm} arm",
                    )
        except Exception as exc:
            self._set_task2_workflow_status(
                state="failed",
                step_index=step_index,
                stage="failed",
                message=(
                    f"Task2 step {step_number} did not complete both parallel "
                    "motions"
                ),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._set_task2_workflow_status(
            state="succeeded",
            step_index=step_index,
            stage=f"subtask{subtask_id}_init_verified",
            message=(
                f"Task2 step {step_number} complete: {left_name} and "
                f"{right_name} verified"
            ),
        )
        return {
            "ok": True,
            "result": {
                "operation": f"task2_move_subtask{subtask_id}_init_poses",
                "execution": "parallel",
                "speed_profile": self.task2_joint_speed_profile,
                "targets": {"left": left_name, "right": right_name},
                "motions": {
                    "left": results["left"],
                    "right": results["right"],
                },
            },
            "skill": {
                "id": f"task2.move_subtask{subtask_id}_init_poses",
                "version": "1.0.0",
                "status": "succeeded",
                "started_at": started_at,
                "duration_ms": (time.monotonic() - started_monotonic) * 1000.0,
                "affected_arms": ["left", "right"],
            },
            "cartesian_jog": self._cartesian_jog_snapshot(),
            "right_arm_home": self._right_arm_home_snapshot(),
            "suction": self.suction.status(),
            "task2_workflow": self.task2_workflow_status(),
        }

    def left_arm_reset_home_skill_descriptor(self) -> dict[str, Any]:
        """Machine-readable contract for the fixed left-arm home action."""

        jog = self._cartesian_jog_snapshot()
        home = jog.get("home_joint_pose", {})
        return {
            "ok": True,
            "skill": {
                "id": "left_arm.reset_home",
                "version": "1.0.0",
                "method": "POST",
                "endpoint": "/api/skills/left-arm/reset-home",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
                "effect": (
                    "move only the left arm at SLOW speed to the shared "
                    "AIRBOT initial joint pose; right arm is unchanged"
                ),
                "target_joint_positions_rad": home.get(
                    "joint_positions_rad"
                ),
                "postcondition": (
                    "left-arm joint feedback is within configured tolerance"
                ),
                "ready": bool(
                    home.get("available") is True
                    and jog.get("busy") is not True
                ),
            },
        }

    def run_left_arm_reset_home_skill(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute the fixed left-arm home action through one agent call."""

        self._require_exact_payload(payload, frozenset())
        started_at = time.time()
        started_monotonic = time.monotonic()
        with self._motion_transition_lock:
            if self.trajectory_recorder.status().get("active") is True:
                raise CartesianJogConflict(
                    "stop trajectory recording before resetting the left arm"
                )
            result = self.cartesian_jog.reset_home()
        return {
            "ok": True,
            "result": result,
            "skill": {
                "id": "left_arm.reset_home",
                "version": "1.0.0",
                "status": "succeeded",
                "started_at": started_at,
                "duration_ms": (
                    time.monotonic() - started_monotonic
                ) * 1000.0,
                "affected_arms": ["left"],
                "right_arm_commanded": False,
            },
            "cartesian_jog": self._cartesian_jog_snapshot(),
        }

    def right_arm_reset_home_skill_descriptor(self) -> dict[str, Any]:
        """Machine-readable contract for the fixed right-arm home action."""

        status = self._right_arm_home_snapshot()
        home = status.get("home_joint_pose", {})
        return {
            "ok": True,
            "skill": {
                "id": "right_arm.reset_home",
                "version": "1.0.0",
                "method": "POST",
                "endpoint": "/api/skills/right-arm/reset-home",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
                "effect": (
                    "move only the right arm at SLOW speed to the shared "
                    "AIRBOT initial joint pose; left arm is unchanged"
                ),
                "target_joint_positions_rad": home.get(
                    "joint_positions_rad"
                ),
                "postcondition": (
                    "right-arm joint feedback is within configured tolerance"
                ),
                "ready": bool(
                    home.get("available") is True
                    and status.get("busy") is not True
                    and status.get("teleop_running") is False
                ),
            },
        }

    def run_right_arm_reset_home_skill(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute the fixed right-arm home action through one agent call."""

        self._require_exact_payload(payload, frozenset())
        started_at = time.time()
        started_monotonic = time.monotonic()
        with self._motion_transition_lock:
            if self.trajectory_recorder.status().get("active") is True:
                raise CartesianJogConflict(
                    "stop trajectory recording before resetting the right arm"
                )
            result = self.right_arm_home.reset_home()
        return {
            "ok": True,
            "result": result,
            "skill": {
                "id": "right_arm.reset_home",
                "version": "1.0.0",
                "status": "succeeded",
                "started_at": started_at,
                "duration_ms": (
                    time.monotonic() - started_monotonic
                ) * 1000.0,
                "affected_arms": ["right"],
                "left_arm_commanded": False,
            },
            "right_arm_home": self._right_arm_home_snapshot(),
        }

    def run_task1_pick_skill(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the full bounded pick skill through one agent-facing call."""

        self._require_exact_payload(payload, frozenset())
        if not self.task1_pick_enabled:
            raise CartesianJogUnavailable("task1 automatic pick is disabled")
        if self.task1_pick_calibration is None:
            raise CartesianJogUnavailable(
                self.task1_pick_error or "left suction TCP is unavailable"
            )
        suction_before = self.suction.status()
        if suction_before.get("available") is not True:
            raise SuctionUnavailable(
                suction_before.get("error") or "suction is unavailable"
            )
        started_at = time.time()
        started_monotonic = time.monotonic()
        with self._motion_transition_lock:
            try:
                jog_before = self.cartesian_jog.status()
                initialization: dict[str, Any]
                if jog_before.get("enabled") is True:
                    initialization = {
                        "status": "reused",
                        "detail": "existing fixed-orientation session reused",
                    }
                else:
                    capture = self.cartesian_jog.capture_orientation()
                    enabled = self.cartesian_jog.enable(
                        "ENABLE_LEFT_CARTESIAN_JOG",
                        area_clear=True,
                        estop_ready=True,
                    )
                    initialization = {
                        "status": "completed",
                        "capture": capture,
                        "enabled": enabled.get("enabled") is True,
                    }
                response = self.pick_detected_carton(
                    {"confirm": "PICK_DETECTED_CARTON"}
                )
            finally:
                # The bounded pick may arm Cartesian control internally, but
                # the manual hold-to-repeat XYZ interface must stay disabled.
                self.cartesian_jog.disable()

        duration_ms = (time.monotonic() - started_monotonic) * 1000.0
        response["skill"] = {
            "id": "task1.pick_carton",
            "version": "1.0.0",
            "status": "succeeded",
            "started_at": started_at,
            "duration_ms": duration_ms,
            "selection_policy": "highest_layer_then_nearest_left_base_xy",
            "stages": [
                {
                    "name": "orientation_lock_and_enable",
                    **initialization,
                },
                {"name": "rgbd_detect_and_select", "status": "completed"},
                {"name": "fixed_orientation_approach", "status": "completed"},
                {"name": "dual_suction_engage", "status": "completed"},
                {"name": "test_lift_100mm", "status": "completed"},
            ],
        }
        return response

    def run_task2_pick_skill(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the bounded Task-2 single-layer carton pick skill."""

        self._require_exact_payload(payload, frozenset())
        if not self.task2_pick_enabled:
            raise CartesianJogUnavailable("task2 automatic pick is disabled")
        if self.task2_pick_calibration is None:
            raise CartesianJogUnavailable(
                self.task2_pick_error or "left suction TCP is unavailable"
            )
        suction_before = self.suction.status()
        if suction_before.get("available") is not True:
            raise SuctionUnavailable(
                suction_before.get("error") or "suction is unavailable"
            )
        started_at = time.time()
        started_monotonic = time.monotonic()
        with self._motion_transition_lock:
            jog_before = self.cartesian_jog.status()
            initialization: dict[str, Any]
            if jog_before.get("enabled") is True:
                initialization = {
                    "status": "reused",
                    "detail": "existing fixed-orientation session reused",
                }
            else:
                capture = self.cartesian_jog.capture_orientation()
                enabled = self.cartesian_jog.enable(
                    "ENABLE_LEFT_CARTESIAN_JOG",
                    area_clear=True,
                    estop_ready=True,
                )
                initialization = {
                    "status": "completed",
                    "capture": capture,
                    "enabled": enabled.get("enabled") is True,
                }
            response = self.pick_detected_carton(
                {"confirm": "PICK_TASK2_SINGLE_CARTON"},
                task_id="task2",
            )

        response["skill"] = {
            "id": "task2.pick_carton",
            "version": "1.0.0",
            "status": "succeeded",
            "started_at": started_at,
            "duration_ms": (time.monotonic() - started_monotonic) * 1000.0,
            "selection_policy": "leftmost_image_x_valid_task2_front_carton",
            "stages": [
                {"name": "orientation_lock_and_enable", **initialization},
                {"name": "rgbd_detect_task2_single", "status": "completed"},
                {"name": "sampled_height_approach", "status": "completed"},
                {"name": "dual_suction_engage", "status": "completed"},
                {"name": "test_lift_20mm", "status": "completed"},
            ],
        }
        return response

    def run_task3_pick_skill(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the bounded Task-3 tabletop flat-carton pick skill."""

        self._require_exact_payload(payload, frozenset())
        if not self.task3_pick_enabled:
            raise CartesianJogUnavailable("task3 automatic pick is disabled")
        if self.task3_pick_calibration is None:
            raise CartesianJogUnavailable(
                self.task3_pick_error or "left suction TCP is unavailable"
            )
        suction_before = self.suction.status()
        if suction_before.get("available") is not True:
            raise SuctionUnavailable(
                suction_before.get("error") or "suction is unavailable"
            )
        started_at = time.time()
        started_monotonic = time.monotonic()
        with self._motion_transition_lock:
            try:
                jog_before = self.cartesian_jog.status()
                initialization: dict[str, Any]
                if jog_before.get("enabled") is True:
                    initialization = {
                        "status": "reused",
                        "detail": "existing fixed-orientation session reused",
                    }
                else:
                    capture = self.cartesian_jog.capture_orientation()
                    enabled = self.cartesian_jog.enable(
                        "ENABLE_LEFT_CARTESIAN_JOG",
                        area_clear=True,
                        estop_ready=True,
                    )
                    initialization = {
                        "status": "completed",
                        "capture": capture,
                        "enabled": enabled.get("enabled") is True,
                    }
                response = self.pick_detected_carton(
                    {"confirm": "PICK_TASK3_FLAT_CARTON"},
                    task_id="task3",
                )
            finally:
                # Step 3 may use this session internally, but must not leave
                # manual hold-to-repeat XYZ jog enabled for later steps.
                self.cartesian_jog.disable()

        response["skill"] = {
            "id": "task3.pick_flat_carton",
            "version": "1.0.0",
            "status": "succeeded",
            "started_at": started_at,
            "duration_ms": (time.monotonic() - started_monotonic) * 1000.0,
            "selection_policy": "nearest_left_base_flat_carton_panel",
            "stages": [
                {"name": "orientation_lock_and_enable", **initialization},
                {"name": "rgbd_detect_task3_flat", "status": "completed"},
                {"name": "table_height_approach", "status": "completed"},
                {"name": "dual_suction_engage", "status": "completed"},
                {"name": "test_lift_20mm", "status": "completed"},
            ],
        }
        return response

    def run_task3_expand_skill(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Raise safely, move to left.expand_box, then replay its trajectory."""

        self._require_exact_payload(payload, frozenset())
        cfg = getattr(self, "task3_expand_cfg", {})
        if cfg.get("enabled") is not True:
            raise CartesianJogUnavailable("task3 carton expansion is disabled")
        if self.task3_pick_calibration is None:
            raise CartesianJogUnavailable(
                self.task3_pick_error or "task3 suction orientation is unavailable"
            )

        safe_height_m = float(
            cfg.get("safe_height_m", cfg.get("rotation_flange_z_m", 0.10))
        )
        expand_pose_name = str(cfg.get("expand_pose_name", "expand_box")).strip()
        trajectory_path = Path(str(cfg.get("trajectory_path", ""))).expanduser()
        replay_timeout_s = float(cfg.get("replay_timeout_s", 60.0))
        max_tracking_error_rad = float(
            cfg.get(
                "max_tracking_error_rad",
                self.trajectory_replay.max_tracking_error_rad,
            )
        )
        if not (
            math.isfinite(safe_height_m)
            and 0.08 <= safe_height_m <= 0.25
            and expand_pose_name
            and trajectory_path.is_absolute()
            and math.isfinite(replay_timeout_s)
            and 5.0 <= replay_timeout_s <= 180.0
            and math.isfinite(max_tracking_error_rad)
            and 0.05 <= max_tracking_error_rad <= 1.0
        ):
            raise CartesianJogUnavailable("task3 expansion sequence is invalid")
        trajectory_path = trajectory_path.resolve()
        recording_id = trajectory_path.name
        preflight = self.trajectory_replay.preflight(
            recording_id,
            replay_gripper=False,
        )["preflight"]
        if Path(str(preflight.get("path"))).resolve() != trajectory_path:
            raise ReplayUnavailable(
                "task3 replay path does not match the configured trajectory"
            )
        if preflight.get("arms") != ["left"]:
            raise ReplayUnavailable("task3 expansion trajectory must be left-arm only")
        if preflight.get("trajectory_source") != "calibration_follower_action":
            raise ReplayUnavailable(
                "task3 expansion requires a calibration follower trajectory"
            )

        started_at = time.time()
        started_monotonic = time.monotonic()
        with self._motion_transition_lock:
            if self.trajectory_recorder.status().get("active") is True:
                raise CartesianJogConflict(
                    "stop trajectory recording before expanding the carton"
                )
            if self.trajectory_replay.status().get("active") is True:
                raise ReplayConflict("another trajectory replay is already active")
            suction_before = self.suction.status()
            if suction_before.get("available") is not True:
                raise SuctionUnavailable(
                    suction_before.get("error") or "suction is unavailable"
                )
            if suction_before.get("engaged") is not True:
                raise CartesianJogSafetyViolation(
                    "Task3 carton must remain attached before expansion"
                )
            jog_before = self.cartesian_jog.status()
            if jog_before.get("busy") is True:
                raise CartesianJogConflict("left arm is busy")
            current_pose = self.cartesian_jog.read_current_pose()
            current_position = np.asarray(
                current_pose.get("position_m"), dtype=np.float64
            )
            current_quaternion = np.asarray(
                current_pose.get("quaternion_xyzw"), dtype=np.float64
            )
            if (
                current_position.shape != (3,)
                or current_quaternion.shape != (4,)
                or not np.all(np.isfinite(current_position))
                or not np.all(np.isfinite(current_quaternion))
            ):
                raise CartesianJogSafetyViolation(
                    "Task3 current left-arm pose is unavailable"
                )
            safe_target = current_position.copy()
            safe_target[2] = max(float(safe_target[2]), safe_height_m)
            safe_lift = self.cartesian_jog.move_to_fixed_orientation_entry(
                safe_target.tolist(),
                current_quaternion.tolist(),
                transit_z_m=safe_height_m,
                enable_token="ENABLE_LEFT_CARTESIAN_JOG",
                area_clear=True,
                estop_ready=True,
                operation="task3_expand_raise_to_safe_height",
            )
            suction_after_lift = self.suction.status()
            if suction_after_lift.get("engaged") is not True:
                raise CartesianJogSafetyViolation(
                    "Task3 suction released during the safe-height lift"
                )
            expand_pose = self.runtime_parameters.pose("left", expand_pose_name)
            expand_pose_move = self.cartesian_jog.move_to_saved_joint_pose(
                expand_pose["joint_positions_rad"],
                pose_name=expand_pose_name,
                speed_profile="DEFAULT",
            )
            suction_at_expand_pose = self.suction.status()
            if suction_at_expand_pose.get("engaged") is not True:
                raise CartesianJogSafetyViolation(
                    "Task3 suction released while moving to left.expand_box"
                )
            replay_started = self.trajectory_replay.start(
                recording_id=recording_id,
                confirmation=recording_id,
                replay_gripper=False,
                allow_suction_engaged=True,
                max_tracking_error_rad=max_tracking_error_rad,
            )
            replay_completed = self.trajectory_replay.wait(
                recording_id,
                timeout_s=replay_timeout_s,
            )
            suction_after = self.suction.status()
            if suction_after.get("engaged") is not True:
                raise CartesianJogSafetyViolation(
                    "Task3 trajectory completed but suction is no longer engaged"
                )

        return {
            "ok": True,
            "result": {
                "operation": "task3_expand_carton",
                "executed": replay_completed.get("state") == "completed",
                "start_position_m": current_position.tolist(),
                "safe_height_m": safe_height_m,
                "safe_target_position_m": safe_target.tolist(),
                "expand_pose": f"left.{expand_pose_name}",
                "trajectory_path": str(trajectory_path),
                "recording_id": recording_id,
                "max_tracking_error_rad": max_tracking_error_rad,
                "motions": {
                    "safe_lift": safe_lift,
                    "expand_pose": expand_pose_move,
                },
                "replay": {
                    "started": replay_started,
                    "completed": replay_completed,
                    "preflight": preflight,
                },
            },
            "skill": {
                "id": "task3.expand_carton",
                "version": "1.0.0",
                "status": "succeeded",
                "started_at": started_at,
                "duration_ms": (time.monotonic() - started_monotonic) * 1000.0,
                "affected_arms": ["left"],
                "suction_preserved": True,
                "stages": [
                    {"name": "raise_to_safe_height", "status": "completed"},
                    {"name": "move_to_left_expand_box", "status": "completed"},
                    {"name": "replay_expand_box_trajectory", "status": "completed"},
                ],
            },
            "cartesian_jog": self._cartesian_jog_snapshot(),
            "suction": suction_after,
        }

    def run_watch_detect_pick_skill(
        self,
        payload: dict[str, Any],
        *,
        task_id: str,
    ) -> dict[str, Any]:
        """Run each task's own recognition prelude, then pick that exact target."""

        self._require_exact_payload(payload, frozenset())
        if task_id not in TASK_IDS:
            raise ValueError("task_id must be task1, task2 or task3")
        pick_enabled = bool(getattr(self, f"{task_id}_pick_enabled"))
        pick_calibration = getattr(self, f"{task_id}_pick_calibration")
        pick_error = getattr(self, f"{task_id}_pick_error")
        if not pick_enabled:
            raise CartesianJogUnavailable(f"{task_id} automatic pick is disabled")
        if pick_calibration is None:
            raise CartesianJogUnavailable(
                pick_error or "left suction TCP is unavailable"
            )
        suction_before = self.suction.status()
        if suction_before.get("available") is not True:
            raise SuctionUnavailable(
                suction_before.get("error") or "suction is unavailable"
            )
        if suction_before.get("engaged") is True:
            raise CartesianJogSafetyViolation(
                "release suction before starting observe-and-pick"
            )

        started_at = time.time()
        started_monotonic = time.monotonic()
        with self._motion_transition_lock:
            if self.trajectory_recorder.status().get("active") is True:
                raise CartesianJogConflict(
                    "stop trajectory recording before observe-and-pick"
                )
            recognition_stages: list[dict[str, Any]] = []
            watcher_stage = self._move_to_recognition_pose(task_id)
            if watcher_stage is not None:
                recognition_stages.append(watcher_stage)

            detection_response = self.detect(task_id=task_id)
            cached_detection = detection_response["detection"]
            if cached_detection.get("target_ready") is not True:
                blockers = cached_detection.get("blockers") or [
                    "detection target is not ready"
                ]
                raise CartesianJogSafetyViolation(
                    "observe stage found no pickable carton: "
                    + "; ".join(map(str, blockers))
                )

            transition_stages: list[dict[str, Any]] = []
            prepared_pre_contact: list[float] | None = None
            if task_id in TASK_IDS:
                entry = getattr(
                    self, f"_prepare_{task_id}_direct_pick_entry"
                )(cached_detection)
                prepared_pre_contact = list(entry["pre_contact_position_m"])
                capture = {
                    "source": entry.get(
                        "orientation_source", f"{task_id}_pick_calibration"
                    ),
                    "quaternion_xyzw": entry["saved_orientation_xyzw"],
                }
                enabled = {"enabled": True}
                transition_stages.append(
                    {
                        "name": "move_directly_to_pick_contact",
                        "status": "completed",
                        "result": entry["motion"],
                    }
                )
            confirm = {
                "task1": "PICK_DETECTED_CARTON",
                "task2": "PICK_TASK2_SINGLE_CARTON",
                "task3": "PICK_TASK3_FLAT_CARTON",
            }[task_id]
            pick_arguments: dict[str, Any] = {
                "task_id": task_id,
                "detection_override": cached_detection,
            }
            if prepared_pre_contact is not None:
                pick_arguments["prepared_pre_contact_position_m"] = (
                    prepared_pre_contact
                )
            response = self.pick_detected_carton(
                {"confirm": confirm},
                **pick_arguments,
            )

        response["skill"] = {
            "id": f"{task_id}.watch_detect_pick",
            "display_name": "药盒识别 + 药盒吸取",
            "version": "1.0.0",
            "status": "succeeded",
            "started_at": started_at,
            "duration_ms": (time.monotonic() - started_monotonic) * 1000.0,
            "cached_detection_id": cached_detection.get("id"),
            "steps": [
                {
                    "name": "药盒识别",
                    "status": "completed",
                    "stages": [
                        *(stage["name"] for stage in recognition_stages),
                        "rgbd_detect_and_cache",
                    ],
                },
                {
                    "name": "药盒吸取",
                    "status": "completed",
                    "stages": [
                        *(stage["name"] for stage in transition_stages),
                        "lock_downward_orientation",
                        "pick_cached_target",
                    ],
                },
            ],
            "stages": [
                *recognition_stages,
                {"name": "rgbd_detect_and_cache", "status": "completed"},
                *transition_stages,
                {
                    "name": "lock_downward_orientation",
                    "status": "completed",
                    "capture": capture,
                    "enabled": enabled.get("enabled") is True,
                },
                {"name": "pick_cached_target", "status": "completed"},
            ],
        }
        return response

    def _move_to_recognition_pose(
        self,
        task_id: str,
    ) -> dict[str, Any] | None:
        """Move to a task's saved camera observation pose when required."""

        if task_id == "task1":
            result = self._move_task1_watcher_and_right_home_parallel()
            return {
                "name": "move_left_watcher_and_right_home_parallel",
                "status": "completed",
                "result": result,
            }
        pose_name = WATCHER_RECOGNITION_POSES.get(task_id)
        if pose_name is None:
            return None
        watcher_pose = self.runtime_parameters.pose("left", pose_name)
        watcher_move = self.cartesian_jog.move_to_saved_joint_pose(
            watcher_pose["joint_positions_rad"],
            pose_name=pose_name,
            speed_profile=(
                self.task2_joint_speed_profile
                if task_id == "task2"
                else "DEFAULT"
            ),
        )
        return {
            "name": f"move_{pose_name}",
            "status": "completed",
            "result": watcher_move,
        }

    def _detect_task2_shipping_box_once(
        self,
        *,
        task_id: str = "task2",
    ) -> dict[str, Any]:
        """Capture and locate the open shipping box; never command motion."""

        if task_id not in {"task2", "task3"}:
            raise ValueError("shipping-box task_id must be task2 or task3")
        detection_cfg = dict(self.task2_shipping_box_cfg)
        if task_id == "task3":
            detection_cfg["require_cavity_depth"] = bool(
                detection_cfg.get("task3_require_cavity_depth", False)
            )

        frame = self._capture_for_read()
        detection, overlay = locate_open_shipping_box_rgbd(
            frame.bgr,
            frame.depth_z16,
            frame.depth_scale_m,
            self.camera.intrinsics,
            self.cam_to_left,
            detection_cfg,
        )
        detection_id = uuid.uuid4().hex
        detection.update(
            {
                "id": detection_id,
                "task_id": task_id,
                "profile": f"{task_id}_shipping_box_opening",
                "captured_at": frame.captured_at,
                "overlay_url": f"/api/camera/frame.jpg?overlay={detection_id}",
                "frame": {
                    "number": frame.frame_number,
                    "device_timestamp_ms": frame.device_timestamp_ms,
                    "has_depth": frame.depth_z16 is not None,
                    "depth_scale_m": frame.depth_scale_m,
                },
            }
        )
        overlay_jpeg = _encode_jpeg(overlay)
        with self._lock:
            self._overlay_jpegs[detection_id] = overlay_jpeg
            while len(self._overlay_jpegs) > OVERLAY_CACHE_SIZE:
                self._overlay_jpegs.popitem(last=False)
        return {"ok": True, "detection": detection}

    def run_task2_detect_shipping_box_step(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Run Task2 step 12 and cache a stable opening-centre pose."""

        return self.run_shipping_box_detection_step(payload, task_id="task2")

    def run_task3_detect_shipping_box_step(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Run Task3 step 7 using the proven Task2 detection workflow."""

        return self.run_shipping_box_detection_step(payload, task_id="task3")

    def run_shipping_box_detection_step(
        self,
        payload: dict[str, Any],
        *,
        task_id: str,
    ) -> dict[str, Any]:
        """Detect the shared shipping box and cache it per workflow."""

        if task_id not in {"task2", "task3"}:
            raise ValueError("shipping-box task_id must be task2 or task3")
        self._require_exact_payload(payload, frozenset())
        if not self.task2_shipping_box_enabled:
            raise CartesianJogSafetyViolation(
                f"{task_id.capitalize()} shipping-box detection is disabled"
            )
        started_at = time.time()
        started_monotonic = time.monotonic()
        attempts = max(
            1,
            min(5, int(self.task2_shipping_box_cfg.get("detection_attempts", 3))),
        )
        required = max(
            1,
            min(
                attempts,
                int(
                    self.task2_shipping_box_cfg.get(
                        "required_consistent_detections",
                        2,
                    )
                ),
            ),
        )
        tolerance_m = float(
            self.task2_shipping_box_cfg.get("consensus_tolerance_m", 0.025)
        )
        if not math.isfinite(tolerance_m) or not 0.003 <= tolerance_m <= 0.10:
            tolerance_m = 0.025
        pixel_tolerance_px = float(
            self.task2_shipping_box_cfg.get("consensus_pixel_tolerance_px", 35.0)
        )
        size_tolerance_m = float(
            self.task2_shipping_box_cfg.get("consensus_size_tolerance_m", 0.04)
        )
        image_side_tolerance_fraction = float(
            self.task2_shipping_box_cfg.get(
                "consensus_image_side_tolerance_fraction", 0.08
            )
        )
        retry_delay_s = max(
            0.0,
            min(
                0.20,
                float(
                    self.task2_shipping_box_cfg.get(
                        "retry_delay_s",
                        0.05,
                    )
                ),
            ),
        )

        with self._motion_transition_lock:
            if self.trajectory_recorder.status().get("active") is True:
                raise CartesianJogConflict(
                    "stop trajectory recording before detecting the shipping box"
                )
            responses: list[dict[str, Any]] = []
            ready_detections: list[dict[str, Any]] = []
            image_detections: list[tuple[dict[str, Any], dict[str, Any]]] = []
            consensus_attempts: list[dict[str, Any]] = []
            selected_response: dict[str, Any] | None = None
            for attempt_index in range(attempts):
                response = self._detect_task2_shipping_box_once(task_id=task_id)
                responses.append(response)
                detection = response["detection"]
                image_matches: list[
                    tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
                ] = []
                if detection.get("detected_2d") is True:
                    for previous_response, previous_detection in image_detections:
                        comparison = shipping_box_image_detections_consistent(
                            detection,
                            previous_detection,
                            pixel_tolerance_px=pixel_tolerance_px,
                            side_tolerance_fraction=image_side_tolerance_fraction,
                        )
                        if comparison.get("valid") is True:
                            image_matches.append(
                                (previous_response, previous_detection, comparison)
                            )
                if detection.get("target_ready") is True:
                    point = np.asarray(
                        detection.get("point_left_base_m"),
                        dtype=np.float64,
                    )
                    if point.shape == (3,) and np.all(np.isfinite(point)):
                        if required <= 1:
                            detection["temporal_consensus"] = {
                                "valid": True,
                                "attempts_used": attempt_index + 1,
                                "required": required,
                                "tolerance_m": tolerance_m,
                            }
                            selected_response = response
                            break
                        if image_matches:
                            detection["temporal_consensus"] = {
                                "valid": True,
                                "mode": "one_rgbd_safe_plus_stable_image_geometry",
                                "attempts_used": attempt_index + 1,
                                "required": required,
                                **image_matches[0][2],
                            }
                            selected_response = response
                            break
                        for previous in ready_detections:
                            comparison = shipping_box_detections_consistent(
                                detection,
                                previous,
                                xy_tolerance_m=tolerance_m,
                                pixel_tolerance_px=pixel_tolerance_px,
                                size_tolerance_m=size_tolerance_m,
                            )
                            consensus_attempts.append(comparison)
                            if comparison.get("valid") is True:
                                detection["temporal_consensus"] = {
                                    "valid": True,
                                    "attempts_used": attempt_index + 1,
                                    "required": required,
                                    "tolerance_m": tolerance_m,
                                    **comparison,
                                }
                                selected_response = response
                                break
                        ready_detections.append(detection)
                        if selected_response is not None:
                            break
                elif image_matches:
                    for previous_response, previous_detection, comparison in image_matches:
                        if previous_detection.get("target_ready") is True:
                            previous_detection["temporal_consensus"] = {
                                "valid": True,
                                "mode": "one_rgbd_safe_plus_stable_image_geometry",
                                "attempts_used": attempt_index + 1,
                                "required": required,
                                **comparison,
                            }
                            selected_response = previous_response
                            break
                    if selected_response is not None:
                        break
                if detection.get("detected_2d") is True:
                    image_detections.append((response, detection))
                if attempt_index + 1 < attempts and retry_delay_s > 0.0:
                    time.sleep(retry_delay_s)

            if selected_response is None:
                selected_response = max(
                    responses,
                    key=lambda item: (
                        item["detection"].get("target_ready") is True,
                        float(item["detection"].get("score", 0.0)),
                    ),
                )
                detection = copy.deepcopy(selected_response["detection"])
                detection["target_ready"] = False
                detection["blockers"] = list(
                    dict.fromkeys(
                        (
                            *detection.get("blockers", []),
                            "shipping_box_target_not_temporally_consistent",
                        )
                    )
                )
                detection["temporal_consensus"] = {
                    "valid": False,
                    "attempts_used": attempts,
                    "required": required,
                    "tolerance_m": tolerance_m,
                    "comparisons": consensus_attempts,
                    "ready_attempts": [
                        {
                            "point_left_base_m": item.get("point_left_base_m"),
                            "center_px": (item.get("candidate") or {}).get(
                                "center_px"
                            ),
                            "opening_size_m": item.get("opening_size_m"),
                        }
                        for item in ready_detections
                    ],
                }
                selected_response = {**selected_response, "detection": detection}

            detection = selected_response["detection"]
            if detection.get("target_ready") is not True:
                blockers = detection.get("blockers") or [
                    "shipping-box target is not ready"
                ]
                with self._lock:
                    self._last_shipping_box_detection = copy.deepcopy(detection)
                diagnostic_fields = (
                    "opening_size_m",
                    "opening_center_left_base_m",
                    "rim_z_m",
                    "bottom_z_m",
                    "cavity_depth_m",
                    "score",
                    "overlay_url",
                    "temporal_consensus",
                )
                diagnostics = {
                    field: detection.get(field)
                    for field in diagnostic_fields
                    if detection.get(field) is not None
                }
                raise CartesianJogSafetyViolation(
                    f"{task_id.capitalize()} shipping-box detection failed: "
                    + "; ".join(map(str, blockers))
                    + "; diagnostics="
                    + json.dumps(diagnostics, ensure_ascii=False, separators=(",", ":"))
                )
            cached_at = time.time()
            self._workflow_detection_cache[f"{task_id}_shipping_box"] = {
                "cached_at": cached_at,
                "detection": copy.deepcopy(detection),
            }
            with self._lock:
                self._last_shipping_box_detection = copy.deepcopy(detection)

        return {
            "ok": True,
            "detection": detection,
            "shipping_box": copy.deepcopy(detection),
            "cache": {
                "task_id": task_id,
                "target": "shipping_box_opening",
                "detection_id": detection.get("id"),
                "cached_at": cached_at,
            },
            "skill": {
                "id": f"{task_id}.detect_shipping_box",
                "status": "succeeded",
                "started_at": started_at,
                "duration_ms": (time.monotonic() - started_monotonic) * 1000.0,
                "stages": [
                    {
                        "name": "capture_synchronized_front_rgbd",
                        "status": "completed",
                    },
                    {
                        "name": "locate_open_shipping_box_rim",
                        "status": "completed",
                    },
                    {
                        "name": "transform_opening_center_to_left_base",
                        "status": "completed",
                    },
                    {
                        "name": "cache_shipping_box_target",
                        "status": "completed",
                    },
                ],
            },
        }

    def task2_place_shipping_box_preflight(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a read-only Task2 placement plan from the cached box target."""

        return self.shipping_box_placement_preflight(payload, task_id="task2")

    def task3_place_shipping_box_preflight(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Build Task3 step 8 preflight using the Task2 placement workflow."""

        return self.shipping_box_placement_preflight(payload, task_id="task3")

    def shipping_box_placement_preflight(
        self,
        payload: dict[str, Any],
        *,
        task_id: str,
    ) -> dict[str, Any]:
        """Build a read-only placement plan using task-specific cache and TCP."""

        if task_id not in {"task2", "task3"}:
            raise ValueError("shipping-box task_id must be task2 or task3")
        self._require_exact_payload(payload, frozenset())
        cfg = self.task2_shipping_box_placement_cfg
        blocker_prefix = task_id
        blockers: list[str] = []
        cached = self._workflow_detection_cache.get(f"{task_id}_shipping_box")
        detection: dict[str, Any] | None = None
        cache_age_s: float | None = None
        maximum_age_s = float(cfg.get("target_cache_max_age_s", 30.0))
        if not math.isfinite(maximum_age_s) or not 5.0 <= maximum_age_s <= 120.0:
            raise ValueError(
                "shipping-box target_cache_max_age_s must be 5..120"
            )
        if not isinstance(cached, dict):
            blockers.append(
                "run_task2_step12_before_step13"
                if task_id == "task2"
                else "run_task3_step7_before_step8"
            )
        else:
            cached_at = float(cached.get("cached_at", 0.0))
            cache_age_s = max(0.0, time.time() - cached_at)
            if cache_age_s > maximum_age_s:
                blockers.append(f"{blocker_prefix}_shipping_box_target_expired")
            raw_detection = cached.get("detection")
            if isinstance(raw_detection, dict):
                detection = copy.deepcopy(raw_detection)
            if detection is None or detection.get("target_ready") is not True:
                blockers.append(f"{blocker_prefix}_shipping_box_target_invalid")

        calibration = getattr(self, f"{task_id}_pick_calibration")
        if calibration is None:
            blockers.append(f"{blocker_prefix}_left_suction_tcp_unavailable")
        pose_name = str(cfg.get("orientation_pose", "system_home")).strip()
        orientation_quaternion = np.asarray([], dtype=np.float64)
        if pose_name != "system_home":
            blockers.append(f"{blocker_prefix}_shipping_box_requires_system_home")
        try:
            live_pose = self.cartesian_jog.read_current_pose()
            live_joints = np.asarray(
                live_pose.get("joint_positions_rad"), dtype=np.float64
            )
            orientation_quaternion = np.asarray(
                live_pose.get("quaternion_xyzw"), dtype=np.float64
            )
            home = self.cartesian_jog.status().get("home_joint_pose", {})
            home_joints = np.asarray(
                home.get("joint_positions_rad"), dtype=np.float64
            )
            home_tolerance = float(home.get("position_tolerance_rad", 0.12))
            if (
                live_joints.shape != (6,)
                or home_joints.shape != (6,)
                or not np.all(np.isfinite(live_joints))
                or not np.all(np.isfinite(home_joints))
                or not math.isfinite(home_tolerance)
                or float(np.max(np.abs(live_joints - home_joints)))
                > home_tolerance
            ):
                blockers.append(f"{blocker_prefix}_left_arm_not_at_system_home")
        except (
            CartesianJogConflict,
            CartesianJogSafetyViolation,
            CartesianJogUnavailable,
            TypeError,
            ValueError,
        ):
            blockers.append(f"{blocker_prefix}_system_home_feedback_unavailable")
        if orientation_quaternion.shape != (4,) or not np.all(
            np.isfinite(orientation_quaternion)
        ):
            blockers.append(f"{blocker_prefix}_shipping_box_orientation_invalid")
        flange_offset = np.asarray([], dtype=np.float64)
        if isinstance(calibration, dict) and orientation_quaternion.shape == (4,):
            try:
                flange_offset = flange_offset_for_orientation(
                    calibration, orientation_quaternion
                )
            except (TypeError, ValueError):
                blockers.append(f"{blocker_prefix}_left_suction_tcp_offset_invalid")

        plan: dict[str, Any] | None = None
        if detection is not None and flange_offset.shape == (3,):
            center = np.asarray(
                detection.get("opening_center_left_base_m"),
                dtype=np.float64,
            )
            opening_size = np.asarray(
                detection.get("opening_size_m"),
                dtype=np.float64,
            )
            rim_z_m = float(detection.get("rim_z_m", math.nan))
            approach_z_m = float(cfg.get("approach_flange_z_m", 0.22))
            opening_margin_m = float(cfg.get("minimum_opening_margin_m", 0.015))
            carton_footprint = np.asarray(
                cfg.get("carton_footprint_m", [0.13, 0.085]),
                dtype=np.float64,
            )
            if (
                center.shape != (3,)
                or not np.all(np.isfinite(center))
                or not math.isfinite(rim_z_m)
            ):
                blockers.append(f"{blocker_prefix}_shipping_box_center_invalid")
            elif (
                not math.isfinite(approach_z_m)
                or not 0.12 <= approach_z_m <= 0.35
            ):
                raise ValueError("task2 shipping-box placement heights are invalid")
            else:
                # The operator-confirmed approach height is already the safe
                # release height.  Keep only one target above the opening;
                # an additional descent risks striking the shipping-box rim.
                release_flange = center + flange_offset
                release_flange[2] = approach_z_m
                release_tcp = release_flange - flange_offset
                if (
                    opening_size.shape != (2,)
                    or not np.all(np.isfinite(opening_size))
                    or carton_footprint.shape != (2,)
                    or not np.all(np.isfinite(carton_footprint))
                    or np.any(carton_footprint <= 0.0)
                    or np.any(
                        np.sort(opening_size)
                        < np.sort(carton_footprint) + 2.0 * opening_margin_m
                    )
                ):
                    blockers.append(
                        f"{blocker_prefix}_shipping_box_opening_margin_invalid"
                    )
                workspace = cfg.get("flange_workspace", {})
                if not isinstance(workspace, dict):
                    raise ValueError(
                        "task2 shipping-box flange_workspace must be an object"
                    )
                for target in (release_flange,):
                    if any(
                        not float(workspace.get(f"{axis}_min", -math.inf))
                        <= float(target[index])
                        <= float(workspace.get(f"{axis}_max", math.inf))
                        for axis, index in {"x": 0, "y": 1, "z": 2}.items()
                    ):
                        blockers.append(
                            f"{blocker_prefix}_shipping_box_flange_target_outside_workspace"
                        )
                        break
                plan = {
                    "detection_id": detection.get("id"),
                    "cache_age_s": cache_age_s,
                    "opening_center_left_base_m": center.tolist(),
                    "opening_size_m": opening_size.tolist(),
                    "carton_footprint_m": carton_footprint.tolist(),
                    "minimum_opening_margin_m": opening_margin_m,
                    "rim_z_m": rim_z_m,
                    "release_tcp_clearance_above_rim_m": float(
                        release_tcp[2] - rim_z_m
                    ),
                    "release_tcp_left_base_m": release_tcp.tolist(),
                    "release_flange_left_base_m": release_flange.tolist(),
                    "approach_flange_left_base_m": release_flange.tolist(),
                    "surface_to_flange_offset_left_base_m": flange_offset.tolist(),
                    "orientation_pose": pose_name,
                    "orientation_quaternion_xyzw": orientation_quaternion.tolist(),
                    "speed_profile": str(cfg.get("speed_profile", "SLOW")).upper(),
                    "motion_policy": "system_home_direct_release_above_opening",
                }

        suction = self.suction.status()
        if suction.get("available") is not True:
            blockers.append(f"{blocker_prefix}_suction_unavailable")
        elif suction.get("engaged") is not True:
            blockers.append(f"{blocker_prefix}_suction_must_be_engaged")
        if self.trajectory_recorder.status().get("active") is True:
            blockers.append("stop_trajectory_recording")
        if self.trajectory_replay.status().get("active") is True:
            blockers.append("stop_trajectory_replay")
        if self.act_rollout.status().get("active") is True:
            blockers.append("stop_act_rollout")
        if (
            self._teleop_status_blocks_cartesian_jog(
                self.teleop_launcher.status()
            )
            or self._system_follow_ownership_active()
        ):
            blockers.append("stop_teleoperation")
        jog = self.cartesian_jog.status()
        if jog.get("busy") is True:
            blockers.append("left_arm_busy")
        blockers = list(dict.fromkeys(blockers))
        return {
            "ok": True,
            "preflight": {
                "ready": not blockers and plan is not None,
                "read_only": True,
                "blockers": blockers,
                "plan": plan,
                "suction": suction,
                "cartesian_jog": jog,
            },
        }

    def run_task2_place_shipping_box_step(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Move above the cached box opening and release without descending."""

        return self.run_shipping_box_placement_step(payload, task_id="task2")

    def run_task3_place_shipping_box_step(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Run Task3 step 8 using the proven Task2 placement workflow."""

        return self.run_shipping_box_placement_step(payload, task_id="task3")

    def run_shipping_box_placement_step(
        self,
        payload: dict[str, Any],
        *,
        task_id: str,
    ) -> dict[str, Any]:
        """Place the held carton using task-specific cache and TCP data."""

        if task_id not in {"task2", "task3"}:
            raise ValueError("shipping-box task_id must be task2 or task3")
        self._require_exact_payload(payload, frozenset({"confirmation"}))
        expected = (
            str(
                self.task2_shipping_box_placement_cfg.get(
                    "confirmation_token",
                    "PLACE_TASK2_IN_SHIPPING_BOX",
                )
            )
            if task_id == "task2"
            else "PLACE_TASK3_IN_SHIPPING_BOX"
        )
        if payload.get("confirmation") != expected:
            raise CartesianJogSafetyViolation(
                f"{task_id.capitalize()} shipping-box placement confirmation is invalid"
            )
        started_at = time.time()
        started_monotonic = time.monotonic()
        with self._motion_transition_lock:
            preflight = (
                self.task3_place_shipping_box_preflight({})
                if task_id == "task3"
                else self.task2_place_shipping_box_preflight({})
            )["preflight"]
            if preflight.get("ready") is not True:
                raise CartesianJogSafetyViolation(
                    f"{task_id.capitalize()} shipping-box placement preflight failed: "
                    + "; ".join(map(str, preflight.get("blockers") or []))
                )
            plan = copy.deepcopy(preflight["plan"])
            cfg = self.task2_shipping_box_placement_cfg
            pose_name = str(cfg.get("orientation_pose", "system_home")).strip()
            if pose_name != "system_home":
                raise CartesianJogSafetyViolation(
                    f"{task_id.capitalize()} shipping-box placement requires system_home"
                )
            pose_motion = self.cartesian_jog.reset_home(
                speed_profile=str(cfg.get("speed_profile", "SLOW")).upper(),
            )
            if self.suction.status().get("engaged") is not True:
                raise CartesianJogSafetyViolation(
                    "Task2 suction released before the shipping-box approach"
                )
            preflight = (
                self.task3_place_shipping_box_preflight({})
                if task_id == "task3"
                else self.task2_place_shipping_box_preflight({})
            )["preflight"]
            if preflight.get("ready") is not True:
                raise CartesianJogSafetyViolation(
                    f"{task_id.capitalize()} shipping-box post-home preflight failed: "
                    + "; ".join(map(str, preflight.get("blockers") or []))
                )
            plan = copy.deepcopy(preflight["plan"])
            capture = self.cartesian_jog.capture_orientation()
            expected_quaternion = plan["orientation_quaternion_xyzw"]
            orientation_error_deg = _quaternion_error_deg(
                capture["quaternion_xyzw"],
                expected_quaternion,
            )
            maximum_error_deg = float(
                cfg.get("maximum_system_home_orientation_error_deg", 3.0)
            )
            if orientation_error_deg > maximum_error_deg:
                raise CartesianJogSafetyViolation(
                    f"{task_id.capitalize()} placement orientation differs from system_home by "
                    f"{orientation_error_deg:.3f} deg"
                )
            enabled = self.cartesian_jog.enable(
                "ENABLE_LEFT_CARTESIAN_JOG",
                area_clear=True,
                estop_ready=True,
            )
            current_position = np.asarray(
                enabled.get("current_position_m"),
                dtype=np.float64,
            )
            if current_position.shape != (3,) or not np.all(
                np.isfinite(current_position)
            ):
                raise CartesianJogSafetyViolation(
                    "Task2 placement start position is unavailable"
                )
            vertical_safe = current_position.copy()
            vertical_safe[2] = float(
                plan["approach_flange_left_base_m"][2]
            )
            approach = self.cartesian_jog.move_fixed_orientation_path(
                [
                    vertical_safe.tolist(),
                    plan["approach_flange_left_base_m"],
                ],
                operation="task2_place_shipping_box_direct_release_approach",
            )
            if self.suction.status().get("engaged") is not True:
                raise CartesianJogSafetyViolation(
                    "Task2 suction released before reaching the box opening"
                )
            suction_release = self.suction.set_engaged(False)
            if self.suction.settle_s:
                time.sleep(self.suction.settle_s)
            if self.suction.status().get("engaged") is not False:
                raise CartesianJogSafetyViolation(
                    "Task2 suction did not report a released state"
                )
            self._workflow_detection_cache.pop(f"{task_id}_shipping_box", None)

        return {
            "ok": True,
            "placement": {
                "executed": True,
                "released": True,
                "plan": plan,
                "initial_vertical_safe_flange_left_base_m": (
                    vertical_safe.tolist()
                ),
                "orientation_error_deg": orientation_error_deg,
                "motions": {
                    "orientation_pose": pose_motion,
                    "approach_to_release_height": approach,
                },
                "suction": suction_release,
            },
            "skill": {
                "id": f"{task_id}.place_shipping_box",
                "status": "succeeded",
                "started_at": started_at,
                "duration_ms": (time.monotonic() - started_monotonic) * 1000.0,
                "stages": [
                    {"name": "validate_cached_shipping_box_target", "status": "completed"},
                    {"name": "verify_system_home_orientation", "status": "completed"},
                    {"name": "approach_above_shipping_box", "status": "completed"},
                    {"name": "release_suction", "status": "completed"},
                ],
            },
        }

    def run_observe_carton_step(
        self,
        payload: dict[str, Any],
        *,
        task_id: str,
    ) -> dict[str, Any]:
        """Run one task-specific recognition step and cache only that target."""

        self._require_exact_payload(payload, frozenset())
        if task_id not in TASK_IDS:
            raise ValueError("task_id must be task1, task2 or task3")
        started_at = time.time()
        started_monotonic = time.monotonic()
        with self._motion_transition_lock:
            if self.trajectory_recorder.status().get("active") is True:
                raise CartesianJogConflict(
                    "stop trajectory recording before observing a carton"
                )
            recognition_stages: list[dict[str, Any]] = []
            watcher_stage = self._move_to_recognition_pose(task_id)
            if watcher_stage is not None:
                recognition_stages.append(watcher_stage)
            detection_response = self.detect(task_id=task_id)
            detection = detection_response["detection"]
            if detection.get("target_ready") is not True:
                blockers = detection.get("blockers") or [
                    "detection target is not ready"
                ]
                raise CartesianJogSafetyViolation(
                    "observe stage found no pickable carton: "
                    + "; ".join(map(str, blockers))
                )
            cached_at = time.time()
            self._workflow_detection_cache[task_id] = {
                "cached_at": cached_at,
                "detection": copy.deepcopy(detection),
            }
        return {
            "ok": True,
            "detection": detection,
            "cache": {
                "task_id": task_id,
                "detection_id": detection.get("id"),
                "cached_at": cached_at,
            },
            "skill": {
                "id": f"{task_id}.observe_carton",
                "status": "succeeded",
                "started_at": started_at,
                "duration_ms": (time.monotonic() - started_monotonic) * 1000.0,
                "stages": [
                    *recognition_stages,
                    {"name": "rgbd_detect_and_cache", "status": "completed"},
                ],
            },
        }

    def run_task1_detect_carton_step(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Detect Task1 source cartons and destination slots concurrently."""

        self._require_exact_payload(payload, frozenset())
        started_at = time.time()
        started_monotonic = time.monotonic()
        with self._motion_transition_lock:
            if self.trajectory_recorder.status().get("active") is True:
                raise CartesianJogConflict(
                    "stop trajectory recording before detecting a Task1 carton"
                )
            with ThreadPoolExecutor(max_workers=2) as executor:
                carton_future = executor.submit(self.detect, task_id="task1")
                box_slots_future = executor.submit(
                    self.run_task1_detect_box_slots_step,
                    {},
                    lock_motion=False,
                )
                detection = carton_future.result()["detection"]
                box_slots_response = box_slots_future.result()
            if detection.get("target_ready") is not True:
                blockers = detection.get("blockers") or [
                    "detection target is not ready"
                ]
                raise CartesianJogSafetyViolation(
                    "Task1 carton detection found no pickable target: "
                    + "; ".join(map(str, blockers))
                )
            cached_at = time.time()
            self._workflow_detection_cache["task1"] = {
                "cached_at": cached_at,
                "detection": copy.deepcopy(detection),
            }
            box_slots_detection = box_slots_response["detection"]
        return {
            "ok": True,
            "detection": detection,
            "box_slots_detection": box_slots_detection,
            "placement_target": copy.deepcopy(
                box_slots_response.get("placement_target")
            ),
            "cache": {
                "task_id": "task1",
                "detection_id": detection.get("id"),
                "cached_at": cached_at,
                "box_slots_detection_id": box_slots_detection.get("id"),
                "next_slot_id": (
                    None
                    if box_slots_detection.get("next_slot") is None
                    else int(box_slots_detection["next_slot"]["slot_id"])
                ),
            },
            "skill": {
                "id": "task1.detect_carton",
                "version": "3.0.0",
                "status": "succeeded",
                "started_at": started_at,
                "duration_ms": (time.monotonic() - started_monotonic) * 1000.0,
                "affected_arms": [],
                "stages": [
                    {
                        "name": "source_carton_rgbd_detect_and_cache",
                        "status": "completed",
                    },
                    {
                        "name": "destination_box_slots_rgbd_detect_and_cache",
                        "status": "completed",
                    },
                ],
            },
        }

    def run_task2_detect_carton_step(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Detect and cache a Task2 carton without commanding either arm."""

        self._require_exact_payload(payload, frozenset())
        started_at = time.time()
        started_monotonic = time.monotonic()
        with self._motion_transition_lock:
            if self.trajectory_recorder.status().get("active") is True:
                raise CartesianJogConflict(
                    "stop trajectory recording before detecting a Task2 carton"
                )
            detection_response = self.detect(task_id="task2")
            detection = detection_response["detection"]
            if detection.get("target_ready") is not True:
                blockers = detection.get("blockers") or [
                    "detection target is not ready"
                ]
                raise CartesianJogSafetyViolation(
                    "Task2 carton detection found no pickable target: "
                    + "; ".join(map(str, blockers))
                )
            cached_at = time.time()
            self._workflow_detection_cache["task2"] = {
                "cached_at": cached_at,
                "detection": copy.deepcopy(detection),
            }
        return {
            "ok": True,
            "detection": detection,
            "cache": {
                "task_id": "task2",
                "detection_id": detection.get("id"),
                "cached_at": cached_at,
            },
            "skill": {
                "id": "task2.detect_carton",
                "version": "1.0.0",
                "status": "succeeded",
                "started_at": started_at,
                "duration_ms": (time.monotonic() - started_monotonic) * 1000.0,
                "affected_arms": [],
                "stages": [
                    {"name": "rgbd_detect_and_cache", "status": "completed"},
                ],
            },
        }

    def _task1_slot_detection_quality_ready(
        self,
        detection: dict[str, Any],
    ) -> bool:
        """Return whether one RGB-D grid is safe to lock for repeated use."""

        quality = detection.get("quality")
        metric_grid = detection.get("metric_grid")
        if not isinstance(quality, dict) or not isinstance(metric_grid, dict):
            return False
        cfg = self.task1_box_placement_cfg
        try:
            return bool(
                quality.get("high_confidence") is True
                and int(quality.get("sample_count", 0))
                >= int(cfg.get("minimum_depth_consensus_samples", 12))
                and float(metric_grid.get("bottom_plane_rms_residual_mm"))
                <= float(cfg.get("maximum_depth_plane_rms_mm", 8.0))
                and float(quality.get("maximum_anchor_peak_to_peak_px"))
                <= float(cfg.get("maximum_anchor_peak_to_peak_px", 7.0))
            )
        except (TypeError, ValueError):
            return False

    def _restore_task1_slot_plan(self) -> None:
        """Restore the locked Task1 grid and progress after a service restart."""

        self.task1_slot_plan_error = ""
        if not getattr(self, "task1_slot_plan_reuse_enabled", False):
            return
        path = getattr(
            self,
            "task1_slot_plan_state_path",
            TASK1_SLOT_PLAN_STATE_PATH,
        )
        try:
            state = _read_json(path)
            if int(state.get("schema_version")) != 1:
                raise ValueError("unsupported Task1 slot-plan schema")
            configured_rotation = float(
                self.task1_slot_grid_cfg.get(
                    "layout_top_view_clockwise_rotation_deg", 0.0
                )
            )
            if abs(
                float(state.get("layout_top_view_clockwise_rotation_deg"))
                - configured_rotation
            ) > 1e-6:
                raise ValueError("saved Task1 slot-plan rotation is obsolete")
            detection = state.get("detection")
            if not isinstance(detection, dict):
                raise ValueError("saved Task1 slot plan has no detection")
            if not self._task1_slot_detection_quality_ready(detection):
                raise ValueError("saved Task1 slot plan has insufficient quality")
            detected_at = float(state.get("detected_at"))
            if not math.isfinite(detected_at) or detected_at <= 0.0:
                raise ValueError("saved Task1 slot plan has no timestamp")
            restored = apply_task1_slot_progress(
                detection,
                state.get("placed_slot_ids", []),
            )
            self._workflow_detection_cache["task1_box_slots"] = {
                "cached_at": detected_at,
                "detection": restored,
                "persistent_plan": True,
            }
        except FileNotFoundError:
            return
        except Exception as exc:
            self.task1_slot_plan_error = f"{type(exc).__name__}: {exc}"

    def _persist_task1_slot_plan(
        self,
        detection: dict[str, Any],
        placed_slot_ids: Sequence[int],
        *,
        detected_at: float,
    ) -> dict[str, Any]:
        """Atomically save the full 20-slot geometry and current sequence."""

        progressed = apply_task1_slot_progress(detection, placed_slot_ids)
        progress = progressed["slot_plan_progress"]
        payload = {
            "schema_version": 1,
            "layout_top_view_clockwise_rotation_deg": float(
                self.task1_slot_grid_cfg.get(
                    "layout_top_view_clockwise_rotation_deg", 0.0
                )
            ),
            "placement_flange_top_view_clockwise_yaw_deg": float(
                self.task1_box_placement_cfg.get(
                    "placement_flange_top_view_clockwise_yaw_deg", 0.0
                )
            ),
            "detected_at": float(detected_at),
            "updated_at": time.time(),
            "placed_slot_ids": copy.deepcopy(progress["placed_slot_ids"]),
            "placed_count": int(progress["placed_count"]),
            "next_slot_id": progress["next_slot_id"],
            "detection": progressed,
        }
        path = getattr(
            self,
            "task1_slot_plan_state_path",
            TASK1_SLOT_PLAN_STATE_PATH,
        )
        _write_json_atomic(path, payload)
        self._workflow_detection_cache["task1_box_slots"] = {
            "cached_at": float(detected_at),
            "detection": copy.deepcopy(progressed),
            "persistent_plan": True,
        }
        return progressed

    def _advance_task1_slot_plan(self, completed_slot_id: int) -> dict[str, Any]:
        """Commit one successful insertion and select the next saved slot."""

        cached = self._workflow_detection_cache.get("task1_box_slots")
        detection = cached.get("detection") if isinstance(cached, dict) else None
        if not isinstance(detection, dict) or not isinstance(cached, dict):
            raise CartesianJogSafetyViolation("Task1 slot plan disappeared")
        progress = detection.get("slot_plan_progress", {})
        placed = [int(value) for value in progress.get("placed_slot_ids", [])]
        identifier = int(completed_slot_id)
        if identifier in placed:
            raise CartesianJogSafetyViolation(
                f"Task1 slot {identifier} was already committed"
            )
        placed.append(identifier)
        return self._persist_task1_slot_plan(
            detection,
            placed,
            detected_at=float(cached.get("cached_at", time.time())),
        )

    def run_task1_detect_box_slots_step(
        self,
        payload: dict[str, Any],
        *,
        lock_motion: bool = True,
    ) -> dict[str, Any]:
        """Detect the Task1 box and cache a rigid metric 2x10 slot grid."""

        self._require_exact_payload(payload, frozenset())
        if not self.task1_slot_grid_enabled:
            raise RuntimeError("Task1 RGB-D slot-grid perception is disabled")
        started_at = time.time()
        started_monotonic = time.monotonic()
        cfg = self.task1_slot_grid_cfg
        cached_plan = self._workflow_detection_cache.get("task1_box_slots")
        if (
            self.task1_slot_plan_reuse_enabled
            and isinstance(cached_plan, dict)
            and cached_plan.get("persistent_plan") is True
            and isinstance(cached_plan.get("detection"), dict)
        ):
            detection = copy.deepcopy(cached_plan["detection"])
            return {
                "ok": True,
                "detection": detection,
                "placement_target": copy.deepcopy(
                    detection.get("placement_target")
                ),
                "cache": {
                    "task_id": "task1",
                    "key": "task1_box_slots",
                    "detection_id": detection.get("id"),
                    "cached_at": cached_plan.get("cached_at"),
                    "persistent_plan": True,
                    "reused_without_camera": True,
                    "next_slot_id": detection.get(
                        "slot_plan_progress", {}
                    ).get("next_slot_id"),
                },
                "skill": {
                    "id": "task1.detect_box_slots",
                    "version": "3.0.0",
                    "status": "succeeded",
                    "started_at": started_at,
                    "duration_ms": (
                        time.monotonic() - started_monotonic
                    ) * 1000.0,
                    "affected_arms": [],
                    "stages": [
                        {
                            "name": "reuse_persistent_rotated_slot_plan",
                            "status": "completed",
                        }
                    ],
                },
            }
        script_setting = Path(
            str(cfg.get("script", "../scripts/task1_slot_grid/detect_box_consensus_20_slots.py"))
        ).expanduser()
        script_path = (
            script_setting
            if script_setting.is_absolute()
            else self.config_path.parent / script_setting
        ).resolve()
        result_setting = Path(
            str(
                cfg.get(
                    "result_json_path",
                    "/tmp/center-box-consensus/consensus-20-slots.json",
                )
            )
        ).expanduser()
        result_path = (
            result_setting
            if result_setting.is_absolute()
            else self.config_path.parent / result_setting
        ).resolve()
        rgb_setting = Path(
            str(
                cfg.get(
                    "raw_rgb_path",
                    "/tmp/center-box-rgbd-boundary/rgb.jpg",
                )
            )
        ).expanduser()
        rgb_path = (
            rgb_setting
            if rgb_setting.is_absolute()
            else self.config_path.parent / rgb_setting
        ).resolve()
        if not script_path.is_file():
            raise RuntimeError(f"Task1 RGB-D slot script is missing: {script_path}")

        project_root = self.config_path.parent.parent.resolve()
        environment = os.environ.copy()
        source_root = str(project_root / "src")
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            source_root
            if not existing_pythonpath
            else source_root + os.pathsep + existing_pythonpath
        )
        timeout_s = max(10.0, min(float(cfg.get("timeout_s", 60.0)), 120.0))
        motion_context = (
            self._motion_transition_lock if lock_motion else nullcontext()
        )
        with motion_context:
            if self.trajectory_recorder.status().get("active") is True:
                raise CartesianJogConflict(
                    "stop trajectory recording before detecting Task1 box slots"
                )
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(project_root),
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(
                    "Task1 RGB-D slot detection failed: " + detail[-1200:]
                )
            if not result_path.is_file():
                raise RuntimeError("Task1 RGB-D slot detector returned no JSON result")
            rgbd_result = _read_json(result_path)
            if rgbd_result.get("placement_grid_ready") is not True:
                raise RuntimeError("Task1 RGB-D slot detector returned no placement grid")
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("Task1 RGB-D slot detector returned no RGB frame")
            detection, overlay = annotate_task1_rgbd_slots(
                bgr,
                rgbd_result,
                cfg,
            )
            detection_id = uuid.uuid4().hex
            detection["id"] = detection_id
            detection["overlay_url"] = (
                f"/api/camera/frame.jpg?overlay={detection_id}"
            )
            cached_at = time.time()
            if self._task1_slot_detection_quality_ready(detection):
                initially_occupied = [
                    int(slot["slot_id"])
                    for slot in detection.get("slots", [])
                    if slot.get("occupied_at_first_detection") is True
                ]
                detection = self._persist_task1_slot_plan(
                    detection,
                    initially_occupied,
                    detected_at=cached_at,
                )
            else:
                self._workflow_detection_cache["task1_box_slots"] = {
                    "cached_at": cached_at,
                    "detection": copy.deepcopy(detection),
                    "persistent_plan": False,
                }
            overlay_jpeg = _encode_jpeg(overlay)
            with self._lock:
                self._overlay_jpegs[detection_id] = overlay_jpeg
                while len(self._overlay_jpegs) > OVERLAY_CACHE_SIZE:
                    self._overlay_jpegs.popitem(last=False)

        return {
            "ok": True,
            "detection": detection,
            "placement_target": copy.deepcopy(detection["placement_target"]),
            "cache": {
                "task_id": "task1",
                "key": "task1_box_slots",
                "detection_id": detection_id,
                "cached_at": cached_at,
                "persistent_plan": bool(
                    self._workflow_detection_cache.get(
                        "task1_box_slots", {}
                    ).get("persistent_plan") is True
                ),
                "reused_without_camera": False,
                "next_slot_id": (
                    None
                    if detection["next_slot"] is None
                    else int(detection["next_slot"]["slot_id"])
                ),
            },
            "skill": {
                "id": "task1.detect_box_slots",
                "version": "3.0.0",
                "status": "succeeded",
                "started_at": started_at,
                "duration_ms": (time.monotonic() - started_monotonic) * 1000.0,
                "affected_arms": [],
                "stages": [
                    {"name": "rgbd_four_inner_edge_consensus", "status": "completed"},
                    {"name": "rigid_metric_2x10_grid", "status": "completed"},
                    {
                        "name": "clockwise_rotated_layout_and_persistent_plan",
                        "status": (
                            "completed"
                            if self._task1_slot_detection_quality_ready(detection)
                            else "quality_not_locked"
                        ),
                    },
                    {"name": "next_slot_and_right_base_center", "status": "completed"},
                ],
            },
        }

    def task1_detect_box_slots_skill_descriptor(self) -> dict[str, Any]:
        """Describe Task1 step 7 RGB-D slot-grid perception."""

        return {
            "ok": True,
            "skill": {
                "id": "task1.detect_box_slots",
                "version": "3.0.0",
                "method": "POST",
                "endpoint": "/api/skills/task1/detect-box-slots",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
                "effect": (
                    "首次检测纸箱并生成俯视顺时针90度的刚性2x10三维槽位；"
                    "持久化全部坐标、已放数量和下一槽位，后续循环不再调用相机"
                ),
                "affected_arms": [],
                "ready": bool(self.task1_slot_grid_enabled),
                "persistent_plan_loaded": bool(
                    isinstance(
                        self._workflow_detection_cache.get("task1_box_slots"),
                        dict,
                    )
                    and self._workflow_detection_cache["task1_box_slots"].get(
                        "persistent_plan"
                    )
                    is True
                ),
                "persistent_plan_error": getattr(
                    self, "task1_slot_plan_error", ""
                ),
            },
        }

    def run_task1_confirm_box_slots_step(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Return the slot result produced concurrently by Task1 step 2."""

        self._require_exact_payload(payload, frozenset())
        started_at = time.time()
        started_monotonic = time.monotonic()
        cached = self._workflow_detection_cache.get("task1_box_slots")
        if not isinstance(cached, dict) or not isinstance(
            cached.get("detection"), dict
        ):
            refreshed = self.run_task1_detect_box_slots_step({})
            refreshed["skill"] = {
                "id": "task1.confirm_box_slots",
                "version": "1.0.0",
                "status": "succeeded",
                "started_at": started_at,
                "duration_ms": (
                    time.monotonic() - started_monotonic
                ) * 1000.0,
                "affected_arms": [],
                "stages": [
                    {
                        "name": "missing_step2_cache_refreshed",
                        "status": "completed",
                    }
                ],
            }
            return refreshed

        detection = copy.deepcopy(cached["detection"])
        return {
            "ok": True,
            "detection": detection,
            "placement_target": copy.deepcopy(
                detection.get("placement_target")
            ),
            "cache": {
                "task_id": "task1",
                "key": "task1_box_slots",
                "detection_id": detection.get("id"),
                "cached_at": cached.get("cached_at"),
                "age_s": max(
                    0.0,
                    time.time() - float(cached.get("cached_at", time.time())),
                ),
                "next_slot_id": (
                    None
                    if detection.get("next_slot") is None
                    else int(detection["next_slot"]["slot_id"])
                ),
            },
            "skill": {
                "id": "task1.confirm_box_slots",
                "version": "1.0.0",
                "status": "succeeded",
                "started_at": started_at,
                "duration_ms": (
                    time.monotonic() - started_monotonic
                ) * 1000.0,
                "affected_arms": [],
                "stages": [
                    {"name": "reuse_step2_slot_cache", "status": "completed"}
                ],
            },
        }

    def task1_confirm_box_slots_skill_descriptor(self) -> dict[str, Any]:
        """Describe the Task1 step 7 cached-slot confirmation."""

        return {
            "ok": True,
            "skill": {
                "id": "task1.confirm_box_slots",
                "version": "1.0.0",
                "method": "POST",
                "endpoint": "/api/skills/task1/confirm-box-slots",
                "effect": (
                    "reuse the slot grid generated concurrently in Task1 step 2; "
                    "refresh only when that cache is absent"
                ),
                "affected_arms": [],
                "ready": bool(self.task1_slot_grid_enabled),
            },
        }

    def _detect_task1_staged_top_once(self) -> dict[str, Any]:
        """Capture one frame and locate only the step-5 upright-carton top."""

        if not self.task1_staged_top_enabled:
            raise CartesianJogUnavailable(
                "Task1 staged-carton top detection is disabled"
            )
        if self.camera.intrinsics is None or self.cam_to_left is None:
            raise CartesianJogUnavailable(
                self.cam_to_left_error
                or "Task1 staged-top RGB-D calibration is unavailable"
            )
        frame = self._capture_for_read()
        if frame.depth_z16 is None or frame.depth_scale_m is None:
            raise CameraUnavailable(
                "Task1 staged-top detection requires synchronized depth"
            )
        detection, overlay = detect_task1_staged_carton_top_rgbd(
            frame.bgr,
            frame.depth_z16,
            frame.depth_scale_m,
            self.camera.intrinsics,
            self.cam_to_left,
            self.task1_staged_top_cfg,
        )
        detection_id = uuid.uuid4().hex
        detection.update(
            {
                "id": detection_id,
                "captured_at": frame.captured_at,
                "frame_number": frame.frame_number,
                "overlay_url": (
                    f"/api/camera/frame.jpg?overlay={detection_id}"
                ),
            }
        )
        overlay_jpeg = _encode_jpeg(overlay)
        with self._lock:
            self._overlay_jpegs[detection_id] = overlay_jpeg
            while len(self._overlay_jpegs) > OVERLAY_CACHE_SIZE:
                self._overlay_jpegs.popitem(last=False)
        return {"ok": True, "detection": detection}

    def detect_task1_staged_top(self) -> dict[str, Any]:
        """Require repeated metric agreement before moving toward the top."""

        cfg = self.task1_staged_top_cfg
        attempts = max(1, min(8, int(cfg.get("detection_attempts", 5))))
        required = max(
            1,
            min(attempts, int(cfg.get("required_consistent_detections", 2))),
        )
        tolerance = float(cfg.get("consensus_tolerance_m", 0.015))
        if not math.isfinite(tolerance) or not 0.003 <= tolerance <= 0.05:
            raise ValueError(
                "task1_staged_top.consensus_tolerance_m is invalid"
            )
        responses: list[dict[str, Any]] = []
        ready: list[dict[str, Any]] = []
        seen_frames: set[tuple[Any, Any]] = set()
        interval_s = max(
            0.0,
            min(0.25, float(cfg.get("detection_interval_s", 0.04))),
        )
        for attempt_index in range(attempts):
            if attempt_index and interval_s:
                time.sleep(interval_s)
            response = self._detect_task1_staged_top_once()
            responses.append(response)
            detection = response["detection"]
            frame_key = (
                detection.get("frame_number"),
                detection.get("captured_at"),
            )
            if frame_key in seen_frames:
                continue
            seen_frames.add(frame_key)
            point = np.asarray(
                detection.get("point_left_base_m"), dtype=np.float64
            )
            if (
                detection.get("target_ready") is not True
                or point.shape != (3,)
                or not np.all(np.isfinite(point))
            ):
                continue
            if required <= 1:
                detection["temporal_consensus"] = {
                    "valid": True,
                    "attempts_used": attempt_index + 1,
                    "required": required,
                    "tolerance_m": tolerance,
                }
                return response
            for previous in ready:
                previous_point = np.asarray(
                    previous["point_left_base_m"], dtype=np.float64
                )
                distance = float(np.linalg.norm(point - previous_point))
                if distance <= tolerance:
                    detection["temporal_consensus"] = {
                        "valid": True,
                        "attempts_used": attempt_index + 1,
                        "required": required,
                        "matched_distance_m": distance,
                        "tolerance_m": tolerance,
                        "policy": "repeated_130x25_top_center_in_fixed_station",
                    }
                    return response
            ready.append(detection)

        best_response = max(
            responses,
            key=lambda response: (
                response["detection"].get("target_ready") is True,
                float(
                    (response["detection"].get("candidate") or {}).get(
                        "score", 0.0
                    )
                ),
            ),
        )
        best = copy.deepcopy(best_response["detection"])
        best["target_ready"] = False
        best["blockers"] = list(
            dict.fromkeys(
                (*best.get("blockers", []), "staged_top_not_temporally_consistent")
            )
        )
        best["temporal_consensus"] = {
            "valid": False,
            "attempts_used": attempts,
            "required": required,
            "tolerance_m": tolerance,
        }
        return {**best_response, "detection": best}

    def _verify_task1_staged_top_suction_axis(
        self,
        detection: dict[str, Any],
    ) -> dict[str, Any]:
        """Require both cup centres, but not their rims, on the 25 mm top."""

        calibration = self.task1_staged_top_calibration
        candidate = detection.get("candidate")
        point = np.asarray(detection.get("point_left_base_m"), np.float64)
        if calibration is None or not isinstance(candidate, dict):
            raise CartesianJogUnavailable(
                self.task1_staged_top_error
                or "Task1 staged-top suction calibration is unavailable"
            )
        polygon = np.asarray(candidate.get("polygon_px"), np.float32)
        if point.shape != (3,) or polygon.shape != (4, 2):
            raise CartesianJogSafetyViolation(
                "Task1 staged-top target geometry is incomplete"
            )
        projection = project_fixed_suction_axis(
            midpoint_left_base_m=point,
            locked_flange_quaternion_xyzw=(
                self._task1_staged_top_locked_quaternion()
            ),
            axis_local_xyz=self.fixed_suction_axis_cfg["axis_local_xyz"],
            approach_local_xyz=self.fixed_suction_axis_cfg[
                "approach_local_xyz"
            ],
            cup_center_spacing_m=float(
                self.fixed_suction_axis_cfg.get("cup_center_spacing_mm", 50.0)
            )
            / 1000.0,
            cup_diameter_m=float(
                self.fixed_suction_axis_cfg.get("cup_diameter_mm", 25.0)
            )
            / 1000.0,
            safety_margin_m=0.0,
            cam_to_left=self.cam_to_left,
            intrinsics=np.asarray(self.camera.intrinsics, np.float64),
            candidate_polygon_px=polygon,
            image_shape=(
                int(self.camera.profile()["color"]["height"]),
                int(self.camera.profile()["color"]["width"]),
            ),
        )
        minimum_center_margin = float(
            self.task1_staged_top_cfg.get(
                "minimum_cup_center_margin_px", 0.0
            )
        )
        center_clearances = [
            float(
                cv2.pointPolygonTest(
                    polygon.reshape((-1, 1, 2)),
                    (float(center[0]), float(center[1])),
                    True,
                )
            )
            for center in projection.cup_centers_px
        ]
        centers_valid = all(
            value >= minimum_center_margin for value in center_clearances
        )
        if not centers_valid:
            raise CartesianJogSafetyViolation(
                "Task1 staged-top cup centres do not both lie on the narrow top"
            )
        result = projection.to_dict()
        result.update(
            {
                "narrow_top_policy": "cup_centres_inside_top; rim overhang allowed",
                "cup_center_clearances_px": center_clearances,
                "minimum_cup_center_margin_px": minimum_center_margin,
                "centers_valid": True,
            }
        )
        return result

    def _task1_staged_top_locked_quaternion(self) -> np.ndarray:
        """Return the step-5-only flange orientation adjustment."""

        calibration = self.task1_staged_top_calibration
        if calibration is None:
            raise CartesianJogUnavailable(
                self.task1_staged_top_error
                or "Task1 staged-top suction calibration is unavailable"
            )
        clockwise_degrees = float(
            self.task1_staged_top_cfg.get(
                "flange_top_view_clockwise_yaw_deg", 0.0
            )
        )
        if not math.isfinite(clockwise_degrees) or abs(clockwise_degrees) > 15.0:
            raise CartesianJogUnavailable(
                "Task1 staged-top flange yaw adjustment is invalid"
            )
        return apply_top_view_clockwise_yaw_xyzw(
            calibration["locked_flange_quaternion_xyzw"],
            clockwise_degrees,
        )

    def _prepare_task1_right_arm_clearance(self) -> dict[str, Any] | None:
        """Open and move the right arm away before the left-arm lift."""

        if not getattr(self, "task1_box_placement_enabled", False):
            return None
        cfg = self.task1_box_placement_cfg
        post_suction_wait = float(cfg.get("post_suction_wait_s", 1.0))
        post_open_wait = float(
            cfg.get("post_right_gripper_open_wait_s", 0.5)
        )
        open_position = float(cfg.get("right_gripper_fully_open_m", 0.06))
        open_tolerance = float(
            cfg.get("right_gripper_open_tolerance_m", 0.012)
        )
        y_clearance = float(cfg.get("right_y_negative_clearance_m", 0.15))
        if not (
            0.0 <= post_suction_wait <= 5.0
            and 0.0 <= post_open_wait <= 5.0
            and 0.0 <= open_position <= 0.10
            and 0.001 <= open_tolerance <= 0.02
            and 0.01 <= y_clearance <= 0.20
        ):
            raise CartesianJogUnavailable(
                "Task1 right-arm clearance settings are invalid"
            )
        if post_suction_wait:
            time.sleep(post_suction_wait)
        initial_pose = self.right_arm_home.read_current_pose()
        initial_position = np.asarray(
            initial_pose.get("position_m"), dtype=np.float64
        )
        initial_quaternion = np.asarray(
            initial_pose.get("quaternion_xyzw"), dtype=np.float64
        )
        if (
            initial_position.shape != (3,)
            or not np.all(np.isfinite(initial_position))
            or initial_quaternion.shape != (4,)
            or not np.all(np.isfinite(initial_quaternion))
        ):
            raise CartesianJogUnavailable(
                "Task1 right-arm initial feedback pose is invalid"
            )
        gripper = self.right_arm_home.move_gripper_to_position(
            open_position,
            operation="task1_right_gripper_fully_open",
            speed_profile="DEFAULT",
        )
        self._require_executed_motion(
            gripper,
            "Task1 right gripper fully open",
        )
        if post_open_wait:
            time.sleep(post_open_wait)
        # The AIRBOT gripper reports before its fingers have mechanically
        # settled even though the blocking command has returned.  Verify the
        # feedback after the operator-requested wait instead of rejecting the
        # transient first sample.
        settled_pose = self.right_arm_home.read_current_pose()
        settled_position = settled_pose.get("gripper_position_m")
        if settled_position is None:
            raise CartesianJogUnavailable(
                "Task1 right gripper settled feedback is unavailable"
            )
        settled_position = float(settled_position)
        settled_error = abs(settled_position - open_position)
        gripper["settled_actual_gripper_position_m"] = settled_position
        gripper["settled_gripper_error_m"] = settled_error
        gripper["open_tolerance_m"] = open_tolerance
        if not math.isfinite(settled_position) or settled_error > open_tolerance:
            raise CartesianJogSafetyViolation(
                "Task1 right gripper did not reach the fully-open target "
                f"after settling (actual={settled_position:.4f} m, "
                f"target={open_position:.4f} m)"
            )
        clearance_target = initial_position.copy()
        clearance_target[1] -= y_clearance
        clearance_motion = self.right_arm_home.move_to_fixed_orientation_entry(
            clearance_target.tolist(),
            initial_quaternion.tolist(),
            transit_z_m=float(initial_position[2]),
            enable_token=str(
                cfg.get("right_enable_token", "ENABLE_RIGHT_ARM_HOME")
            ),
            area_clear=True,
            estop_ready=True,
            operation="task1_right_y_negative_clearance",
            use_configured_safe_transit=False,
        )
        self._require_executed_motion(
            clearance_motion,
            "Task1 right-arm Y-negative clearance",
        )
        actual_clearance = np.asarray(
            clearance_motion.get("actual_position_m", clearance_target),
            dtype=np.float64,
        )
        state = {
            "initial_position_m": initial_position.tolist(),
            "initial_quaternion_xyzw": initial_quaternion.tolist(),
            "initial_joint_positions_rad": copy.deepcopy(
                initial_pose.get("joint_positions_rad")
            ),
            "clearance_position_m": actual_clearance.tolist(),
            "right_gripper": gripper,
            "y_negative_clearance": clearance_motion,
            "captured_at": float(initial_pose.get("captured_at", time.time())),
            "ready_for_parallel_retreat": True,
        }
        self._task1_right_clearance_state = copy.deepcopy(state)
        return state

    def run_task1_pick_staged_top_step(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Task1 step 5: move to the taught or detected staged-top pick pose."""

        self._require_exact_payload(payload, frozenset())
        if not self.task1_staged_top_enabled:
            raise CartesianJogUnavailable("Task1 staged-top pick is disabled")
        if self.task1_staged_top_calibration is None:
            raise CartesianJogUnavailable(
                self.task1_staged_top_error
                or "Task1 staged-top suction calibration is unavailable"
            )
        suction_before = self.suction.status()
        if suction_before.get("available") is not True:
            raise SuctionUnavailable(
                suction_before.get("error") or "suction is unavailable"
            )
        if suction_before.get("engaged") is True:
            raise CartesianJogSafetyViolation(
                "release suction after the fixed placement before Task1 step 5"
            )
        started_at = time.time()
        started_monotonic = time.monotonic()
        cfg = self.task1_staged_top_cfg
        with self._motion_transition_lock:
            if self.trajectory_recorder.status().get("active") is True:
                raise CartesianJogConflict(
                    "stop trajectory recording before Task1 staged-top pick"
                )
            if self.trajectory_replay.status().get("active") is True:
                raise CartesianJogConflict(
                    "stop trajectory replay before Task1 staged-top pick"
                )
            if self.act_rollout.status().get("active") is True:
                raise CartesianJogConflict(
                    "stop ACT rollout before Task1 staged-top pick"
                )
            if (
                self._teleop_status_blocks_cartesian_jog(
                    self.teleop_launcher.status()
                )
                or self._system_follow_ownership_active()
            ):
                raise CartesianJogConflict(
                    "stop teleoperation before Task1 staged-top pick"
                )
            if self.cartesian_jog.status().get("busy") is True:
                raise CartesianJogConflict(
                    "left arm is busy before Task1 staged-top pick"
                )
            if (
                getattr(self, "task1_box_placement_enabled", False)
                and self.right_arm_home.status().get("busy") is True
            ):
                raise CartesianJogConflict(
                    "right arm is busy before Task1 staged-top pick"
                )
            self._task1_right_clearance_state = None
            fixed_pose_enabled = cfg.get("fixed_contact_pose_enabled") is True
            detection: dict[str, Any]
            target: np.ndarray | None = None
            flange_xy_offset = np.zeros(2, dtype=np.float64)
            if fixed_pose_enabled:
                if cfg.get("fixed_contact_pose_frame", "left_base") != "left_base":
                    raise CartesianJogUnavailable(
                        "Task1 staged-top fixed contact pose must use left_base"
                    )
                contact = np.asarray(
                    cfg.get("fixed_contact_flange_position_m"),
                    dtype=np.float64,
                )
                locked_quaternion = np.asarray(
                    cfg.get("fixed_contact_flange_quaternion_xyzw"),
                    dtype=np.float64,
                )
                quaternion_norm = float(np.linalg.norm(locked_quaternion))
                if (
                    contact.shape != (3,)
                    or not np.all(np.isfinite(contact))
                    or locked_quaternion.shape != (4,)
                    or not np.all(np.isfinite(locked_quaternion))
                    or not 0.999 <= quaternion_norm <= 1.001
                ):
                    raise CartesianJogUnavailable(
                        "Task1 staged-top fixed contact pose is invalid"
                    )
                locked_quaternion = locked_quaternion / quaternion_norm
                detection = {
                    "target_ready": True,
                    "mode": "operator_taught_fixed_contact_pose",
                    "frame": "left_base",
                    "point_left_base_m": None,
                    "blockers": [],
                }
                suction_axis = {
                    "centers_valid": True,
                    "verification": "operator_taught_fixed_contact_pose",
                }
            else:
                detection_response = self.detect_task1_staged_top()
                detection = detection_response["detection"]
                if detection.get("target_ready") is not True:
                    raise CartesianJogSafetyViolation(
                        "Task1 staged-carton top is not ready: "
                        + "; ".join(map(str, detection.get("blockers") or []))
                    )
                suction_axis = self._verify_task1_staged_top_suction_axis(
                    detection
                )
                target = np.asarray(
                    detection.get("point_left_base_m"), dtype=np.float64
                )
                calibration = self.task1_staged_top_calibration
                locked_quaternion = self._task1_staged_top_locked_quaternion()
                z_offset = float(
                    calibration["contact_sample"]["surface_to_flange_z_offset_m"]
                )
                flange_xy_offset = np.asarray(
                    cfg.get("flange_center_offset_left_base_m", [0.0, 0.0]),
                    dtype=np.float64,
                )
                if (
                    flange_xy_offset.shape != (2,)
                    or not np.all(np.isfinite(flange_xy_offset))
                    or np.any(np.abs(flange_xy_offset) > 0.02)
                ):
                    raise CartesianJogUnavailable(
                        "Task1 staged-top flange-centre offset is invalid"
                    )
                contact = target.copy()
                contact[:2] += flange_xy_offset
                contact[2] = float(target[2]) + z_offset
            workspace = cfg.get(
                "target_workspace_left_base_m",
                {
                    "x": [-0.05, 0.10],
                    "y": [-0.28, -0.08],
                    "z": [0.06, 0.13],
                },
            )
            if not isinstance(workspace, dict):
                raise CartesianJogUnavailable(
                    "task1_staged_top target workspace is invalid"
                )
            try:
                workspace_valid = all(
                    float(workspace[axis][0])
                    <= float((contact if fixed_pose_enabled else target)[index])
                    <= float(workspace[axis][1])
                    for index, axis in enumerate(("x", "y", "z"))
                )
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                raise CartesianJogUnavailable(
                    "task1_staged_top target workspace is invalid"
                ) from exc
            if not workspace_valid:
                raise CartesianJogSafetyViolation(
                    "Task1 staged-top pose is outside its calibrated station workspace"
                )
            clearance = float(cfg.get("pre_contact_clearance_m", 0.025))
            lift_distance = float(cfg.get("test_lift_m", 0.05))
            lift_speed_profile = str(
                cfg.get("lift_speed_profile", "DEFAULT")
            ).strip().upper()
            pre_contact = contact.copy()
            pre_contact[2] += clearance
            lift_target_z = float(contact[2]) + lift_distance
            requested_transit = max(
                float(cfg.get("transit_z_m", 0.12)),
                float(pre_contact[2]),
            )
            if not (
                0.005 <= clearance <= 0.08
                and 0.01 <= lift_distance <= 0.20
                and 0.08 <= requested_transit <= 0.25
                and lift_target_z <= 0.50
                and lift_speed_profile in {"DEFAULT", "FAST"}
            ):
                raise CartesianJogUnavailable(
                    "Task1 staged-top motion heights are invalid"
                )
            entry = self.cartesian_jog.move_to_fixed_orientation_entry(
                pre_contact.tolist(),
                locked_quaternion.tolist(),
                transit_z_m=requested_transit,
                enable_token="ENABLE_LEFT_CARTESIAN_JOG",
                area_clear=True,
                estop_ready=True,
                operation="task1_staged_top_direct_entry",
                calibrated_workspace_profile=str(
                    cfg.get("calibrated_workspace_profile", "task1_pick")
                ),
                use_configured_safe_transit=False,
            )
            descent = self.cartesian_jog.move_fixed_orientation_path(
                [contact.tolist()],
                operation="task1_staged_top_vertical_contact",
                calibrated_workspace_profile=str(
                    cfg.get("calibrated_workspace_profile", "task1_pick")
                ),
            )
            suction_result = self.suction.set_engaged(True)
            if suction_result.get("engaged") is not True:
                raise SuctionUnavailable(
                    suction_result.get("error")
                    or "Task1 staged-top suction did not engage"
                )
            right_clearance = self._prepare_task1_right_arm_clearance()
            if right_clearance is None and self.suction.settle_s:
                time.sleep(self.suction.settle_s)
            lift_target = contact.copy()
            lift_target[2] = lift_target_z
            try:
                lift = self.cartesian_jog.move_fixed_orientation_path(
                    [lift_target.tolist()],
                    operation="task1_staged_top_test_lift",
                    calibrated_workspace_profile=str(
                        cfg.get("calibrated_workspace_profile", "task1_pick")
                    ),
                    speed_profile=lift_speed_profile,
                )
            except (
                CartesianJogSafetyViolation,
                CartesianJogUnavailable,
                CartesianJogTimeout,
            ) as exc:
                raise type(exc)(
                    "staged-top suction is engaged, but the lift did not complete: "
                    f"{exc}"
                ) from exc

        return {
            "ok": True,
            "detection": detection,
            "result": {
                "operation": "task1_pick_staged_upright_carton_top",
                "executed": True,
                "flange_center_alignment": (
                    "operator_taught_fixed_contact_pose"
                    if fixed_pose_enabled
                    else "detected_top_geometric_center"
                ),
                "surface_center_left_base_m": (
                    None if target is None else target.tolist()
                ),
                "flange_center_offset_left_base_m": flange_xy_offset.tolist(),
                "flange_top_view_clockwise_yaw_deg": float(
                    cfg.get("flange_top_view_clockwise_yaw_deg", 0.0)
                ),
                "locked_flange_quaternion_xyzw": locked_quaternion.tolist(),
                "contact_flange_position_m": contact.tolist(),
                "pre_contact_flange_position_m": pre_contact.tolist(),
                "test_lift_flange_position_m": lift_target.tolist(),
                "test_lift_speed_profile": lift_speed_profile,
                "suction_axis": suction_axis,
                "motions": {
                    "direct_entry": entry,
                    "vertical_contact": descent,
                    "test_lift": lift,
                    "right_clearance": (
                        None
                        if right_clearance is None
                        else {
                            "executed": True,
                            "gripper": right_clearance["right_gripper"],
                            "y_negative": right_clearance[
                                "y_negative_clearance"
                            ],
                        }
                    ),
                },
                "suction": suction_result,
            },
            "cartesian_jog": self._cartesian_jog_snapshot(),
            "suction": self.suction.status(),
            "skill": {
                "id": "task1.pick_staged_carton_top",
                "version": "1.0.0",
                "status": "succeeded",
                "started_at": started_at,
                "duration_ms": (time.monotonic() - started_monotonic) * 1000.0,
                "affected_arms": ["left"],
                "stages": [
                    {
                        "name": (
                            "load_operator_taught_fixed_contact_pose"
                            if fixed_pose_enabled
                            else "detect_130x25_top_plane_and_barcode"
                        ),
                        "status": "completed",
                    },
                    {
                        "name": (
                            "use_operator_verified_dual_cup_alignment"
                            if fixed_pose_enabled
                            else "verify_two_cup_centres"
                        ),
                        "status": "completed",
                    },
                    {
                        "name": (
                            "move_to_taught_flange_pose"
                            if fixed_pose_enabled
                            else "align_flange_center"
                        ),
                        "status": "completed",
                    },
                    {
                        "name": "right_gripper_open_and_y_negative_clearance",
                        "status": (
                            "completed"
                            if right_clearance is not None
                            else "not_configured"
                        ),
                    },
                    {"name": "vertical_suction_and_lift", "status": "completed"},
                ],
            },
        }

    def task1_place_in_box_skill_descriptor(self) -> dict[str, Any]:
        """Describe Task1 step 6's parallel left placement/right retreat."""

        cached = self._workflow_detection_cache.get("task1_box_slots")
        detection = cached.get("detection") if isinstance(cached, dict) else None
        next_slot = (
            detection.get("next_slot") if isinstance(detection, dict) else None
        )
        progress = (
            detection.get("slot_plan_progress", {})
            if isinstance(detection, dict)
            else {}
        )
        return {
            "ok": True,
            "skill": {
                "id": "task1.place_in_box",
                "version": "3.0.0",
                "method": "POST",
                "endpoint": "/api/skills/task1/place-in-box",
                "effect": (
                    "左臂复用第2步持久化的顺时针90度RGB-D槽位计划，"
                    "让法兰绕当前吸盘中心顺时针旋转90度，再用标定TCP"
                    "将吸盘中心对准当前编号槽位三维中心；"
                    "左臂不再上升，使用短段路径经箱体中心上方到目标；"
                    "右臂同时沿Z正方向"
                    "抬升150毫米，然后返回右臂系统初始关节位姿"
                ),
                "affected_arms": ["left", "right"],
                "execution": "parallel_left_place_right_retreat_and_system_home",
                "slot_plan_progress": copy.deepcopy(progress),
                "ready": bool(
                    self.task1_box_placement_enabled
                    and isinstance(next_slot, dict)
                    and (
                        isinstance(self._task1_right_clearance_state, dict)
                        or self.right_arm_home.status().get("available") is True
                    )
                    and self.task1_staged_top_calibration is not None
                    and self.suction.status().get("engaged") is True
                ),
            },
        }

    def run_task1_place_in_box_step(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Place directly from the current height while the right arm homes."""

        self._require_exact_payload(payload, frozenset())
        if not self.task1_box_placement_enabled:
            raise CartesianJogUnavailable("Task1 box placement is disabled")
        started_at = time.time()
        started_monotonic = time.monotonic()
        cfg = self.task1_box_placement_cfg
        with self._motion_transition_lock:
            if self.trajectory_recorder.status().get("active") is True:
                raise CartesianJogConflict(
                    "stop trajectory recording before Task1 box placement"
                )
            if self.trajectory_replay.status().get("active") is True:
                raise CartesianJogConflict(
                    "stop trajectory replay before Task1 box placement"
                )
            if self.act_rollout.status().get("active") is True:
                raise CartesianJogConflict(
                    "stop ACT rollout before Task1 box placement"
                )
            if (
                self._teleop_status_blocks_cartesian_jog(
                    self.teleop_launcher.status()
                )
                or self._system_follow_ownership_active()
            ):
                raise CartesianJogConflict(
                    "stop teleoperation before Task1 box placement"
                )
            if self.suction.status().get("engaged") is not True:
                raise CartesianJogSafetyViolation(
                    "Task1 step 6 requires left suction engaged"
                )
            right_state = copy.deepcopy(self._task1_right_clearance_state)
            right_already_system_home = False
            if not isinstance(right_state, dict) or right_state.get(
                "ready_for_parallel_retreat"
            ) is not True:
                # A service reload after a failed left-arm entry clears the
                # in-memory step-5 state even though the parallel right-arm
                # branch may already have reached system home.  Recover only
                # after live joint feedback proves that exact safe condition.
                right_pose = self.right_arm_home.read_current_pose()
                right_live_joints = np.asarray(
                    right_pose.get("joint_positions_rad"), dtype=np.float64
                )
                right_home_status = self.right_arm_home.status().get(
                    "home_joint_pose", {}
                )
                right_home_joints = np.asarray(
                    right_home_status.get("joint_positions_rad"),
                    dtype=np.float64,
                )
                right_home_tolerance = float(
                    right_home_status.get("position_tolerance_rad", 0.12)
                )
                if (
                    right_live_joints.shape != (6,)
                    or right_home_joints.shape != (6,)
                    or not np.all(np.isfinite(right_live_joints))
                    or not np.all(np.isfinite(right_home_joints))
                    or not math.isfinite(right_home_tolerance)
                    or float(
                        np.max(np.abs(right_live_joints - right_home_joints))
                    )
                    > right_home_tolerance
                ):
                    raise CartesianJogSafetyViolation(
                        "Task1 step 6 requires step-5 right clearance or "
                        "verified right system home"
                    )
                right_already_system_home = True
                right_state = {"ready_for_parallel_retreat": False}
            cached = self._workflow_detection_cache.get("task1_box_slots")
            detection = (
                cached.get("detection") if isinstance(cached, dict) else None
            )
            next_slot = (
                detection.get("next_slot")
                if isinstance(detection, dict)
                else None
            )
            if not isinstance(next_slot, dict):
                raise CartesianJogSafetyViolation(
                    "Task1 step 6 has no cached empty box slot"
                )
            progress_before = detection.get("slot_plan_progress", {})
            placement_sequence_number = int(
                progress_before.get("placed_count", 0)
            ) + 1
            persistent_plan = bool(
                isinstance(cached, dict)
                and cached.get("persistent_plan") is True
            )
            maximum_cache_age_s = float(cfg.get("target_cache_max_age_s", 300.0))
            if (
                not math.isfinite(maximum_cache_age_s)
                or not 10.0 <= maximum_cache_age_s <= 900.0
            ):
                raise CartesianJogUnavailable(
                    "Task1 step 6 target_cache_max_age_s must be 10..900"
                )
            cached_at = float(cached.get("cached_at", 0.0))
            cache_age_s = max(0.0, time.time() - cached_at)
            if not math.isfinite(cached_at) or cached_at <= 0.0:
                raise CartesianJogSafetyViolation(
                    "Task1 step 6 cached slot has no capture timestamp"
                )
            if not persistent_plan and cache_age_s > maximum_cache_age_s:
                raise CartesianJogSafetyViolation(
                    "Task1 step 6 depth slot is stale; rerun step 2"
                )
            quality = detection.get("quality")
            metric_grid = detection.get("metric_grid")
            if not isinstance(quality, dict) or not isinstance(metric_grid, dict):
                raise CartesianJogSafetyViolation(
                    "Task1 step 6 cached slot has no RGB-D quality evidence"
                )
            minimum_depth_samples = int(
                cfg.get("minimum_depth_consensus_samples", 12)
            )
            maximum_depth_rms_mm = float(
                cfg.get("maximum_depth_plane_rms_mm", 8.0)
            )
            maximum_anchor_spread_px = float(
                cfg.get("maximum_anchor_peak_to_peak_px", 7.0)
            )
            sample_count = int(quality.get("sample_count", 0))
            depth_rms_mm = float(
                metric_grid.get("bottom_plane_rms_residual_mm", math.nan)
            )
            anchor_spread_px = float(
                quality.get("maximum_anchor_peak_to_peak_px", math.nan)
            )
            if not (
                3 <= minimum_depth_samples <= 30
                and 1.0 <= maximum_depth_rms_mm <= 20.0
                and 1.0 <= maximum_anchor_spread_px <= 20.0
            ):
                raise CartesianJogUnavailable(
                    "Task1 step 6 RGB-D quality thresholds are invalid"
                )
            if (
                quality.get("high_confidence") is not True
                or sample_count < minimum_depth_samples
                or not math.isfinite(depth_rms_mm)
                or depth_rms_mm > maximum_depth_rms_mm
                or not math.isfinite(anchor_spread_px)
                or anchor_spread_px > maximum_anchor_spread_px
            ):
                raise CartesianJogSafetyViolation(
                    "Task1 step 6 RGB-D slot quality is insufficient; rerun step 2"
                )
            release_surface = np.asarray(
                next_slot.get("release_surface_center_left_base_m"),
                dtype=np.float64,
            )
            approach_surface = np.asarray(
                next_slot.get("approach_center_left_base_m"),
                dtype=np.float64,
            )
            floor_center = np.asarray(
                next_slot.get("floor_center_left_base_m"),
                dtype=np.float64,
            )
            live_pose = self.cartesian_jog.read_current_pose()
            current_flange_position = np.asarray(
                live_pose.get("position_m"), dtype=np.float64
            )
            current_quaternion = np.asarray(
                live_pose.get("quaternion_xyzw"), dtype=np.float64
            )
            if (
                release_surface.shape != (3,)
                or approach_surface.shape != (3,)
                or floor_center.shape != (3,)
                or current_flange_position.shape != (3,)
                or current_quaternion.shape != (4,)
                or not np.all(np.isfinite(release_surface))
                or not np.all(np.isfinite(approach_surface))
                or not np.all(np.isfinite(floor_center))
                or not np.all(np.isfinite(current_flange_position))
                or not np.all(np.isfinite(current_quaternion))
            ):
                raise CartesianJogUnavailable(
                    "Task1 step 6 cached slot or live suction pose is invalid"
                )
            quaternion_norm = float(np.linalg.norm(current_quaternion))
            if not 0.995 <= quaternion_norm <= 1.005:
                raise CartesianJogSafetyViolation(
                    "Task1 step 6 live flange quaternion is invalid"
                )
            current_quaternion = current_quaternion / quaternion_norm
            orientation = quaternion_xyzw_to_matrix(current_quaternion)
            flange_down_axis = orientation[:, 0]
            vertical_tilt_deg = math.degrees(
                math.acos(float(np.clip(-flange_down_axis[2], -1.0, 1.0)))
            )
            maximum_vertical_tilt_deg = float(
                cfg.get("maximum_vertical_tilt_deg", 5.0)
            )
            if (
                not math.isfinite(maximum_vertical_tilt_deg)
                or not 0.5 <= maximum_vertical_tilt_deg <= 10.0
            ):
                raise CartesianJogUnavailable(
                    "Task1 step 6 maximum_vertical_tilt_deg is invalid"
                )
            if vertical_tilt_deg > maximum_vertical_tilt_deg:
                raise CartesianJogSafetyViolation(
                    "Task1 step 6 requires the current flange to be vertical"
                )
            placement_clockwise_yaw_deg = float(
                cfg.get("placement_flange_top_view_clockwise_yaw_deg", 90.0)
            )
            if (
                not math.isfinite(placement_clockwise_yaw_deg)
                or placement_clockwise_yaw_deg not in {0.0, 90.0, 180.0, 270.0}
            ):
                raise CartesianJogUnavailable(
                    "Task1 step 6 placement flange yaw must be 0, 90, 180 or 270"
                )
            placement_quaternion = apply_top_view_clockwise_yaw_xyzw(
                current_quaternion,
                placement_clockwise_yaw_deg,
            )
            calibration = self.task1_staged_top_calibration
            if not isinstance(calibration, dict):
                raise CartesianJogUnavailable(
                    self.task1_staged_top_error
                    or "Task1 step 6 suction TCP calibration is unavailable"
                )
            try:
                current_tcp_to_flange_offset = flange_offset_for_orientation(
                    calibration, current_quaternion
                )
                placement_tcp_to_flange_offset = flange_offset_for_orientation(
                    calibration, placement_quaternion
                )
            except (TypeError, ValueError) as exc:
                raise CartesianJogUnavailable(
                    "Task1 step 6 suction TCP offset is invalid"
                ) from exc
            carton_height_m = float(
                metric_grid.get("carton_height_mm", math.nan)
            ) / 1000.0
            approach_delta = approach_surface - release_surface
            if (
                not math.isfinite(carton_height_m)
                or not 0.05 <= carton_height_m <= 0.15
                or abs(float(release_surface[2] - floor_center[2]) - carton_height_m)
                > 0.005
                or float(np.linalg.norm(approach_delta[:2])) > 0.002
                or not 0.03 <= float(approach_delta[2]) <= 0.12
            ):
                raise CartesianJogSafetyViolation(
                    "Task1 step 6 RGB-D slot center geometry is inconsistent"
                )
            release_flange = release_surface + placement_tcp_to_flange_offset
            approach_flange = approach_surface + placement_tcp_to_flange_offset
            current_suction_center = (
                current_flange_position - current_tcp_to_flange_offset
            )
            orientation_rotation_flange = (
                current_suction_center + placement_tcp_to_flange_offset
            )
            box_center = np.asarray(
                metric_grid.get("box_center_left_base_m"), dtype=np.float64
            )
            configured_transfer_z = float(
                cfg.get("left_transfer_flange_z_m", 0.16)
            )
            transfer_step_m = float(
                cfg.get("left_transfer_max_step_m", 0.05)
            )
            approach_step_m = float(
                cfg.get("left_approach_max_step_m", 0.04)
            )
            if (
                box_center.shape != (3,)
                or not np.all(np.isfinite(box_center))
                or not math.isfinite(configured_transfer_z)
                or not 0.10 <= configured_transfer_z <= 0.22
            ):
                raise CartesianJogUnavailable(
                    "Task1 step 6 box-centre transfer settings are invalid"
                )
            transfer_flange = np.asarray(
                [
                    float(box_center[0]),
                    float(box_center[1]),
                    min(float(orientation_rotation_flange[2]), configured_transfer_z),
                ],
                dtype=np.float64,
            )
            if transfer_flange[2] < approach_flange[2] + 0.04:
                raise CartesianJogSafetyViolation(
                    "Task1 step 6 current height is too low for a safe box-centre transfer"
                )
            try:
                transfer_waypoints = segmented_linear_positions(
                    orientation_rotation_flange,
                    transfer_flange,
                    transfer_step_m,
                )
                approach_waypoints = segmented_linear_positions(
                    transfer_flange,
                    approach_flange,
                    approach_step_m,
                )
            except ValueError as exc:
                raise CartesianJogUnavailable(
                    "Task1 step 6 segmented transfer path is invalid"
                ) from exc
            segmented_approach_waypoints = (
                transfer_waypoints + approach_waypoints
            )
            workspace = cfg.get("left_slot_workspace_m", {})
            try:
                targets_inside = all(
                    float(workspace[axis][0])
                    <= float(target[index])
                    <= float(workspace[axis][1])
                    for target in (release_flange, approach_flange)
                    for index, axis in enumerate(("x", "y", "z"))
                )
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                raise CartesianJogUnavailable(
                    "Task1 step 6 left slot workspace is invalid"
                ) from exc
            if not targets_inside:
                raise CartesianJogSafetyViolation(
                    "Task1 step 6 slot is outside the left-arm workspace"
                )
            right_z_retreat = float(
                cfg.get("right_z_positive_retreat_m", 0.15)
            )
            right_clearance = np.asarray(
                right_state.get("clearance_position_m"), dtype=np.float64
            )
            if not 0.01 <= right_z_retreat <= 0.20 or (
                not right_already_system_home
                and (
                    right_clearance.shape != (3,)
                    or not np.all(np.isfinite(right_clearance))
                )
            ):
                raise CartesianJogUnavailable(
                    "Task1 step 6 right-arm return state is invalid"
                )

            def move_left_into_slot() -> dict[str, Any]:
                orientation_lock = self.cartesian_jog.move_to_fixed_orientation_entry(
                    orientation_rotation_flange.tolist(),
                    placement_quaternion.tolist(),
                    transit_z_m=float(current_flange_position[2]),
                    enable_token="ENABLE_LEFT_CARTESIAN_JOG",
                    area_clear=True,
                    estop_ready=True,
                    operation="task1_left_box_slot_clockwise90_about_suction_center",
                    calibrated_workspace_profile=str(
                        cfg.get("calibrated_workspace_profile", "task1_pick")
                    ),
                    use_configured_safe_transit=False,
                )
                transfer = self.cartesian_jog.move_fixed_orientation_path(
                    transfer_waypoints,
                    operation="task1_left_box_slot_segmented_transfer",
                    calibrated_workspace_profile=str(
                        cfg.get("calibrated_workspace_profile", "task1_pick")
                    ),
                )
                approach = self.cartesian_jog.move_fixed_orientation_path(
                    approach_waypoints,
                    operation="task1_left_box_slot_segmented_approach",
                    calibrated_workspace_profile=str(
                        cfg.get("calibrated_workspace_profile", "task1_pick")
                    ),
                )
                descent = self.cartesian_jog.move_fixed_orientation_path(
                    [release_flange.tolist()],
                    operation="task1_left_box_slot_descent",
                    calibrated_workspace_profile=str(
                        cfg.get("calibrated_workspace_profile", "task1_pick")
                    ),
                )
                self._require_executed_motion(
                    orientation_lock,
                    "Task1 step 6 left orientation lock",
                )
                self._require_executed_motion(
                    transfer,
                    "Task1 step 6 left segmented box-centre transfer",
                )
                self._require_executed_motion(
                    approach,
                    "Task1 step 6 left segmented slot approach",
                )
                self._require_executed_motion(
                    descent,
                    "Task1 step 6 left slot descent",
                )
                return {
                    "executed": True,
                    "orientation_lock": orientation_lock,
                    "segmented_transfer": transfer,
                    "segmented_approach": approach,
                    "descent": descent,
                }

            def retreat_and_return_right() -> dict[str, Any]:
                if right_already_system_home:
                    return {
                        "executed": True,
                        "already_system_home": True,
                    }
                raised = right_clearance.copy()
                raised[2] += right_z_retreat
                retreat = self.right_arm_home.move_fixed_orientation_path(
                    [raised.tolist()],
                    operation="task1_right_z_positive_retreat",
                )
                returned = self.right_arm_home.reset_home(
                    speed_profile="DEFAULT",
                )
                self._require_executed_motion(
                    retreat,
                    "Task1 step 6 right Z-positive retreat",
                )
                self._require_executed_motion(
                    returned,
                    "Task1 step 6 right return to system initial pose",
                )
                return {
                    "executed": True,
                    "z_positive_retreat": retreat,
                    "return_system_initial_pose": returned,
                }

            with ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="task1-direct-place-and-right-home",
            ) as executor:
                futures = {
                    "left": executor.submit(move_left_into_slot),
                    "right": executor.submit(retreat_and_return_right),
                }
                motions: dict[str, Any] = {}
                errors: list[Exception] = []
                for arm, future in futures.items():
                    try:
                        motions[arm] = future.result()
                    except Exception as exc:
                        errors.append(exc)
                if errors:
                    raise errors[0]
            self._task1_right_clearance_state = None
            progressed_detection = self._advance_task1_slot_plan(
                int(next_slot["slot_id"])
            )
            progress_after = progressed_detection["slot_plan_progress"]
        return {
            "ok": True,
            "result": {
                "operation": "task1_place_in_box",
                "execution": "parallel_left_place_right_retreat_and_system_home",
                "slot_id": int(next_slot["slot_id"]),
                "placement_sequence_number": placement_sequence_number,
                "slot_plan_capacity": int(progress_after["capacity"]),
                "placed_count_after": int(progress_after["placed_count"]),
                "remaining_slot_count": int(progress_after["remaining_count"]),
                "next_slot_id": progress_after["next_slot_id"],
                "slot_plan_persistent": True,
                "depth_slot_cache_age_s": cache_age_s,
                "depth_plane_rms_residual_mm": depth_rms_mm,
                "depth_consensus_sample_count": sample_count,
                "flange_vertical_tilt_deg": vertical_tilt_deg,
                "placement_flange_top_view_clockwise_yaw_deg": (
                    placement_clockwise_yaw_deg
                ),
                "current_flange_quaternion_xyzw": current_quaternion.tolist(),
                "placement_flange_quaternion_xyzw": (
                    placement_quaternion.tolist()
                ),
                "current_suction_center_left_base_m": current_suction_center.tolist(),
                "target_suction_center_left_base_m": release_surface.tolist(),
                "current_tcp_to_flange_offset_left_base_m": (
                    current_tcp_to_flange_offset.tolist()
                ),
                "tcp_to_flange_offset_left_base_m": (
                    placement_tcp_to_flange_offset.tolist()
                ),
                "left_orientation_rotation_flange_position_m": (
                    orientation_rotation_flange.tolist()
                ),
                "left_approach_flange_position_m": approach_flange.tolist(),
                "left_release_flange_position_m": release_flange.tolist(),
                "left_transfer_flange_position_m": transfer_flange.tolist(),
                "left_segmented_approach_waypoint_count": len(
                    segmented_approach_waypoints
                ),
                "left_entry_mode": "segmented_via_box_center_without_lift",
                "right_z_positive_retreat_m": right_z_retreat,
                "right_return_target": "system_initial_joint_pose",
                "right_already_system_home": right_already_system_home,
                "motions": motions,
                "suction_remains_engaged": True,
            },
            "skill": {
                "id": "task1.place_in_box",
                "version": "3.0.0",
                "status": "succeeded",
                "started_at": started_at,
                "duration_ms": (
                    time.monotonic() - started_monotonic
                ) * 1000.0,
                "affected_arms": ["left", "right"],
                "stages": [
                    {
                        "name": "left_place_and_right_retreat_return",
                        "status": "completed",
                        "execution": "parallel",
                    }
                ],
            },
            "cartesian_jog": self._cartesian_jog_snapshot(),
            "right_arm_home": self._right_arm_home_snapshot(),
            "suction": self.suction.status(),
        }

    def task1_pick_staged_top_skill_descriptor(self) -> dict[str, Any]:
        return {
            "ok": True,
            "skill": {
                "id": "task1.pick_staged_carton_top",
                "version": "1.0.0",
                "method": "POST",
                "endpoint": "/api/skills/task1/pick-staged-carton-top",
                "effect": (
                    "use the operator-taught fixed contact pose when enabled; "
                    "otherwise detect the fixed-station 130x25 mm upright-carton "
                    "top, then descend, engage suction and lift"
                ),
                "affected_arms": ["left"],
                "ready": bool(
                    self.task1_staged_top_enabled
                    and self.task1_staged_top_calibration is not None
                    and (
                        self.task1_staged_top_cfg.get(
                            "fixed_contact_pose_enabled"
                        )
                        is True
                        or (
                            self.camera.state == "ready"
                            and self.camera.live_rgbd_is_fresh(
                                max_age_s=LIVE_RGBD_MAX_AGE_S
                            )
                        )
                    )
                ),
                "error": self.task1_staged_top_error,
            },
        }

    def run_task3_detect_carton_step(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Detect and cache a Task3 carton without commanding either arm."""

        self._require_exact_payload(payload, frozenset())
        started_at = time.time()
        started_monotonic = time.monotonic()
        with self._motion_transition_lock:
            if self.trajectory_recorder.status().get("active") is True:
                raise CartesianJogConflict(
                    "stop trajectory recording before detecting a Task3 carton"
                )
            detection = self.detect(task_id="task3")["detection"]
            if detection.get("target_ready") is not True:
                blockers = detection.get("blockers") or [
                    "detection target is not ready"
                ]
                raise CartesianJogSafetyViolation(
                    "Task3 carton detection found no pickable target: "
                    + "; ".join(map(str, blockers))
                )
            cached_at = time.time()
            self._workflow_detection_cache["task3"] = {
                "cached_at": cached_at,
                "detection": copy.deepcopy(detection),
            }
        return {
            "ok": True,
            "detection": detection,
            "cache": {
                "task_id": "task3",
                "detection_id": detection.get("id"),
                "cached_at": cached_at,
            },
            "skill": {
                "id": "task3.detect_carton",
                "version": "1.0.0",
                "status": "succeeded",
                "started_at": started_at,
                "duration_ms": (time.monotonic() - started_monotonic) * 1000.0,
                "affected_arms": [],
                "stages": [
                    {"name": "rgbd_detect_and_cache", "status": "completed"},
                ],
            },
        }

    def task1_detect_carton_skill_descriptor(self) -> dict[str, Any]:
        """Describe Task1's perception-only left-18 detection step."""

        camera_ready = self.camera.state == "ready"
        live_rgbd = self.camera.live_rgbd_is_fresh(
            max_age_s=LIVE_RGBD_MAX_AGE_S
        )
        return {
            "ok": True,
            "skill": {
                "id": "task1.detect_carton",
                "version": "2.0.0",
                "method": "POST",
                "endpoint": "/api/skills/task1/detect-carton",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
                "effect": (
                    "concurrently detect and cache the Task1 source cartons "
                    "and the destination shipping-box 2x10 slot grid without arm motion"
                ),
                "affected_arms": [],
                "ready": bool(camera_ready and live_rgbd),
            },
        }

    def task2_detect_carton_skill_descriptor(self) -> dict[str, Any]:
        """Describe Task2's perception-only detection and cache step."""

        camera_ready = self.camera.state == "ready"
        live_rgbd = self.camera.live_rgbd_is_fresh(
            max_age_s=LIVE_RGBD_MAX_AGE_S
        )
        return {
            "ok": True,
            "skill": {
                "id": "task2.detect_carton",
                "version": "1.0.0",
                "method": "POST",
                "endpoint": "/api/skills/task2/detect-carton",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
                "effect": "detect and cache one Task2 carton without arm motion",
                "affected_arms": [],
                "ready": bool(camera_ready and live_rgbd),
            },
        }

    def task3_detect_carton_skill_descriptor(self) -> dict[str, Any]:
        """Describe Task3's perception-only detection and cache step."""

        camera_ready = self.camera.state == "ready"
        live_rgbd = self.camera.live_rgbd_is_fresh(
            max_age_s=LIVE_RGBD_MAX_AGE_S
        )
        return {
            "ok": True,
            "skill": {
                "id": "task3.detect_carton",
                "version": "1.0.0",
                "method": "POST",
                "endpoint": "/api/skills/task3/detect-carton",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
                "effect": "detect and cache one Task3 carton without arm motion",
                "affected_arms": [],
                "ready": bool(camera_ready and live_rgbd),
            },
        }

    def run_pick_cached_carton_step(
        self,
        payload: dict[str, Any],
        *,
        task_id: str,
    ) -> dict[str, Any]:
        """Pick the exact target produced by the preceding observe step."""

        self._require_exact_payload(payload, frozenset())
        if task_id not in {"task1", "task2", "task3"}:
            raise ValueError("task_id must be task1, task2 or task3")
        cached = self._workflow_detection_cache.get(task_id)
        if not isinstance(cached, dict):
            raise CartesianJogSafetyViolation(
                "run the carton recognition step before the pick step"
            )
        cached_at = float(cached.get("cached_at", 0.0))
        if time.time() - cached_at > 60.0:
            self._workflow_detection_cache.pop(task_id, None)
            raise CartesianJogSafetyViolation(
                "cached carton target is older than 60 seconds; recognize again"
            )
        detection = copy.deepcopy(cached.get("detection"))
        if not isinstance(detection, dict) or detection.get("target_ready") is not True:
            raise CartesianJogSafetyViolation("cached carton target is invalid")

        started_at = time.time()
        started_monotonic = time.monotonic()
        with self._motion_transition_lock:
            try:
                if self.trajectory_recorder.status().get("active") is True:
                    raise CartesianJogConflict(
                        "stop trajectory recording before picking a carton"
                    )
                prepared_pre_contact: list[float] | None = None
                if task_id in TASK_IDS:
                    entry = getattr(
                        self, f"_prepare_{task_id}_direct_pick_entry"
                    )(detection)
                    prepared_pre_contact = list(entry["pre_contact_position_m"])
                    transition_stage = {
                        "name": (
                            "move_via_home_to_pick_approach"
                            if task_id == "task3"
                            else (
                                "move_directly_from_watcher_to_pick_approach"
                                if task_id == "task1"
                                else "move_directly_to_pick_approach"
                            )
                        ),
                        "status": "completed",
                        "result": entry["motion"],
                        "entry_pose": entry.get("entry_pose"),
                        "entry_pose_motion": entry.get("entry_pose_motion"),
                        "orientation_source": entry.get(
                            "orientation_source", f"{task_id}_pick_calibration"
                        ),
                    }
                confirm = {
                    "task1": "PICK_DETECTED_CARTON",
                    "task2": "PICK_TASK2_SINGLE_CARTON",
                    "task3": "PICK_TASK3_FLAT_CARTON",
                }[task_id]
                pick_arguments: dict[str, Any] = {
                    "task_id": task_id,
                    "detection_override": detection,
                }
                if prepared_pre_contact is not None:
                    pick_arguments["prepared_pre_contact_position_m"] = (
                        prepared_pre_contact
                    )
                response = self.pick_detected_carton(
                    {"confirm": confirm},
                    **pick_arguments,
                )
                self._workflow_detection_cache.pop(task_id, None)
            finally:
                if task_id in {"task1", "task3"}:
                    # Fixed workflow step 3 may arm XYZ jog internally, but
                    # manual hold-to-repeat jogging must not remain enabled.
                    self.cartesian_jog.disable()
        response["skill"] = {
            "id": f"{task_id}.pick_cached_carton",
            "status": "succeeded",
            "started_at": started_at,
            "duration_ms": (time.monotonic() - started_monotonic) * 1000.0,
            "cached_detection_id": detection.get("id"),
            "stages": [
                transition_stage,
                {"name": "pick_cached_target", "status": "completed"},
            ],
        }
        return response

    def _prepare_task1_direct_pick_entry(
        self,
        detection: dict[str, Any],
    ) -> dict[str, Any]:
        return self._prepare_direct_pick_entry(detection, task_id="task1")

    def _prepare_task2_direct_pick_entry(
        self,
        detection: dict[str, Any],
    ) -> dict[str, Any]:
        return self._prepare_direct_pick_entry(detection, task_id="task2")

    def _prepare_task3_direct_pick_entry(
        self,
        detection: dict[str, Any],
    ) -> dict[str, Any]:
        return self._prepare_direct_pick_entry(detection, task_id="task3")

    def _prepare_direct_pick_entry(
        self,
        detection: dict[str, Any],
        *,
        task_id: str,
    ) -> dict[str, Any]:
        """Move from a watcher pose to the task's approach pose."""

        if task_id not in TASK_IDS:
            raise ValueError("direct pick entry requires task1, task2 or task3")
        if detection.get("task_id") != task_id or detection.get(
            "target_ready"
        ) is not True:
            raise CartesianJogSafetyViolation(
                f"cached {task_id} carton target is not ready"
            )
        pick_cfg = self._effective_pick_cfg(task_id)
        calibration = getattr(self, f"{task_id}_pick_calibration")
        if not isinstance(calibration, dict):
            raise CartesianJogUnavailable(
                getattr(self, f"{task_id}_pick_error")
                or f"{task_id} pick calibration is unavailable"
            )
        suction_before = self.suction.status()
        if suction_before.get("available") is not True:
            raise SuctionUnavailable(
                suction_before.get("error") or "suction is unavailable"
            )
        if suction_before.get("engaged") is True:
            raise CartesianJogSafetyViolation(
                f"release suction before moving to the {task_id} pick approach"
            )
        expected_quaternion = np.asarray(
            calibration.get("locked_flange_quaternion_xyzw"),
            dtype=np.float64,
        )
        if task_id in {"task1", "task3"}:
            entry_pose_name = "system_home"
            if task_id == "task1":
                entry_pose_name = None
            entry_pose = None
            saved_quaternion = expected_quaternion.copy()
            orientation_source = f"{task_id}_pick_calibration"
        else:
            entry_pose_name = None
            entry_pose = self.runtime_parameters.pose("left", "left_pick_ready")
            saved_quaternion = np.asarray(
                entry_pose.get("quaternion_xyzw"),
                dtype=np.float64,
            )
            orientation_source = "left.left_pick_ready"
        if (
            saved_quaternion.shape != (4,)
            or expected_quaternion.shape != (4,)
            or not np.all(np.isfinite(saved_quaternion))
            or not np.all(np.isfinite(expected_quaternion))
        ):
            raise CartesianJogUnavailable(
                f"{task_id} downward flange orientation is unavailable"
            )
        saved_norm = float(np.linalg.norm(saved_quaternion))
        expected_norm = float(np.linalg.norm(expected_quaternion))
        if saved_norm < 1e-12 or expected_norm < 1e-12:
            raise CartesianJogUnavailable(
                f"{task_id} downward flange orientation has zero norm"
            )
        saved_quaternion /= saved_norm
        expected_quaternion /= expected_norm
        orientation_dot = float(
            np.clip(abs(np.dot(saved_quaternion, expected_quaternion)), 0.0, 1.0)
        )
        orientation_error_deg = float(
            np.degrees(2.0 * np.arccos(orientation_dot))
        )
        max_orientation_error = float(
            pick_cfg.get("max_locked_orientation_error_deg", 1.0)
        )
        if orientation_error_deg > max_orientation_error:
            raise CartesianJogSafetyViolation(
                "saved flange orientation differs from the calibrated "
                f"{task_id} suction orientation by {orientation_error_deg:.3f} deg"
            )

        surface_center = np.asarray(
            detection.get("point_left_base_m"),
            dtype=np.float64,
        )
        contact_offset = np.asarray(
            calibration.get("contact_sample", {}).get(
                "surface_to_target_flange_offset_in_base_m"
            ),
            dtype=np.float64,
        )
        shared_target_offset = np.asarray(
            getattr(self, "config", {}).get("shared_pick", {}).get(
                "target_offset_left_base_m",
                [0.0, 0.0, 0.0],
            ),
            dtype=np.float64,
        )
        task_target_offset = np.asarray(
            pick_cfg.get("target_offset_left_base_m", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        if (
            surface_center.shape != (3,)
            or contact_offset.shape != (3,)
            or shared_target_offset.shape != (3,)
            or task_target_offset.shape != (3,)
            or not np.all(np.isfinite(surface_center))
            or not np.all(np.isfinite(contact_offset))
            or not np.all(np.isfinite(shared_target_offset))
            or not np.all(np.isfinite(task_target_offset))
        ):
            raise CartesianJogSafetyViolation(
                f"{task_id} pick geometry does not contain finite XYZ values"
            )
        combined_target_offset = shared_target_offset + task_target_offset
        if np.any(np.abs(combined_target_offset) > 0.03):
            raise CartesianJogUnavailable(
                f"{task_id} combined pick target offset exceeds 30 mm"
            )
        try:
            configured_contact_z = pick_cfg.get("contact_flange_z_m")
            if task_id == "task1":
                layer_estimate = detection.get("layer_estimate")
                if (
                    not isinstance(layer_estimate, dict)
                    or layer_estimate.get("valid") is not True
                ):
                    raise CartesianJogSafetyViolation(
                        "detected carton has no valid 1/2/3-layer estimate"
                    )
                layer = int(layer_estimate.get("layer", 0))
                layer_map = pick_cfg.get("contact_flange_z_m_by_layer", {})
                if not isinstance(layer_map, dict):
                    raise CartesianJogUnavailable(
                        "task1 contact Z map is not configured"
                    )
                contact_z = float(layer_map[str(layer)])
            elif configured_contact_z is not None:
                contact_z = float(configured_contact_z)
            elif task_id == "task2":
                contact_z = float(
                    calibration["contact_sample"][
                        "absolute_contact_flange_z_m"
                    ]
                )
            else:
                contact_z = float(pick_cfg["table_surface_z_m"]) + float(
                    calibration["contact_sample"][
                        "surface_to_target_flange_offset_in_base_m"
                    ][2]
                )
            clearance = float(pick_cfg.get("pre_contact_clearance_m", 0.025))
            transit_z = float(pick_cfg.get("transit_z_m", 0.10))
        except (KeyError, TypeError, ValueError) as exc:
            raise CartesianJogUnavailable(
                f"{task_id} approach height is unavailable"
            ) from exc
        minimum_transit_z = 0.0 if task_id == "task1" else 0.05
        if not (
            math.isfinite(contact_z)
            and 0.0 <= clearance <= 0.08
            and minimum_transit_z <= transit_z <= 0.35
        ):
            raise CartesianJogUnavailable(
                f"{task_id} approach parameters are invalid"
            )

        contact = surface_center + contact_offset + combined_target_offset
        contact[2] = contact_z
        pre_contact = contact.copy()
        pre_contact[2] += clearance
        entry_pose_motion: dict[str, Any] | None = None
        if task_id == "task3":
            # Leave the low, sideways Task3 watcher pose through the operator-
            # configured system home joint pose.  The following Cartesian
            # entry uses the calibrated vertical suction quaternion; neither
            # init_pose nor left_pick_ready is a Task3 motion waypoint.
            entry_pose_motion = self.cartesian_jog.reset_home()
            if entry_pose_motion.get("executed") is not True:
                raise CartesianJogSafetyViolation(
                    "task3 left-arm reset to system home did not execute"
                )
        motion = self.cartesian_jog.move_to_fixed_orientation_entry(
            pre_contact.tolist(),
            saved_quaternion.tolist(),
            transit_z_m=transit_z,
            enable_token="ENABLE_LEFT_CARTESIAN_JOG",
            area_clear=True,
            estop_ready=True,
            operation=f"{task_id}_direct_pick_entry",
            **(
                {
                    "calibrated_workspace_profile": str(
                        pick_cfg["calibrated_workspace_profile"]
                    )
                }
                if task_id == "task1"
                and pick_cfg.get("calibrated_workspace_profile") is not None
                else {}
            ),
            **(
                {"use_configured_safe_transit": False}
                if task_id == "task1"
                else {}
            ),
        )
        if motion.get("executed") is not True:
            raise CartesianJogSafetyViolation(
                f"{task_id} direct pick entry did not execute"
            )
        return {
            "pre_contact_position_m": pre_contact.tolist(),
            "contact_position_m": contact.tolist(),
            "saved_orientation_xyzw": saved_quaternion.tolist(),
            "orientation_error_deg": orientation_error_deg,
            "shared_target_offset_left_base_m": shared_target_offset.tolist(),
            "task_target_offset_left_base_m": task_target_offset.tolist(),
            "combined_target_offset_left_base_m": (
                combined_target_offset.tolist()
            ),
            "entry_pose": (
                (
                    "left.system_home"
                    if entry_pose_name == "system_home"
                    else f"left.{entry_pose_name}"
                )
                if entry_pose_name is not None
                else None
            ),
            "entry_pose_motion": entry_pose_motion,
            "orientation_source": orientation_source,
            "detected_layer": (
                int(detection.get("layer_estimate", {}).get("layer", 0))
                if task_id == "task1"
                else None
            ),
            "motion": motion,
        }

    def pick_detected_carton(
        self,
        payload: dict[str, Any],
        *,
        task_id: str = "task1",
        detection_override: dict[str, Any] | None = None,
        prepared_pre_contact_position_m: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        self._require_exact_payload(payload, frozenset({"confirm"}))
        if task_id not in {"task1", "task2", "task3"}:
            raise ValueError("task_id must be task1, task2 or task3")
        if task_id == "task1":
            pick_cfg = self._effective_pick_cfg("task1")
            pick_enabled = self.task1_pick_enabled
            pick_calibration = self.task1_pick_calibration
            pick_error = self.task1_pick_error
        elif task_id == "task2":
            pick_cfg = self._effective_pick_cfg("task2")
            pick_enabled = self.task2_pick_enabled
            pick_calibration = self.task2_pick_calibration
            pick_error = self.task2_pick_error
        else:
            pick_cfg = self._effective_pick_cfg("task3")
            pick_enabled = self.task3_pick_enabled
            pick_calibration = self.task3_pick_calibration
            pick_error = self.task3_pick_error
        expected_token = str(
            pick_cfg.get(
                "confirm_token",
                {
                    "task1": "PICK_DETECTED_CARTON",
                    "task2": "PICK_TASK2_SINGLE_CARTON",
                    "task3": "PICK_TASK3_FLAT_CARTON",
                }[task_id],
            )
        )
        if payload.get("confirm") != expected_token:
            raise ValueError(f"confirm must be {expected_token}")
        if not pick_enabled:
            raise CartesianJogUnavailable(f"{task_id} automatic pick is disabled")
        if pick_calibration is None:
            raise CartesianJogUnavailable(
                pick_error or "left suction TCP is unavailable"
            )

        transit_z = float(pick_cfg.get("transit_z_m", 0.10))
        pre_contact_clearance = float(
            pick_cfg.get("pre_contact_clearance_m", 0.025)
        )
        test_lift = float(pick_cfg.get("test_lift_m", 0.02))
        max_orientation_error = float(
            pick_cfg.get(
                "max_locked_orientation_error_deg",
                1.0,
            )
        )
        minimum_clearance = 0.0 if task_id == "task2" else 0.005
        minimum_transit_z = 0.0 if task_id == "task1" else 0.05
        maximum_test_lift = 0.10 if task_id == "task1" else 0.05
        if not (
            minimum_transit_z <= transit_z <= 0.35
            and minimum_clearance <= pre_contact_clearance <= 0.08
            and 0.005 <= test_lift <= maximum_test_lift
            and 0.1 <= max_orientation_error <= 5.0
        ):
            raise CartesianJogUnavailable(f"{task_id} pick parameters are invalid")

        with self._motion_transition_lock:
            if self.trajectory_recorder.status().get("active") is True:
                raise CartesianJogConflict(
                    "stop trajectory recording before automatic pick"
                )
            jog_before = self.cartesian_jog.status()
            if jog_before.get("enabled") is not True:
                raise CartesianJogSafetyViolation(
                    "enable left-arm XYZ control before automatic pick"
                )
            current_position = np.asarray(
                jog_before.get("current_position_m"),
                dtype=np.float64,
            )
            current_quaternion = np.asarray(
                jog_before.get("locked_quaternion_xyzw"),
                dtype=np.float64,
            )
            if current_position.shape != (3,) or not np.all(
                np.isfinite(current_position)
            ):
                raise CartesianJogSafetyViolation(
                    "current left-arm position is unavailable"
                )
            if current_quaternion.shape != (4,) or not np.all(
                np.isfinite(current_quaternion)
            ):
                raise CartesianJogSafetyViolation(
                    "locked left-arm orientation is unavailable"
                )
            expected_quaternion = np.asarray(
                pick_calibration[
                    "locked_flange_quaternion_xyzw"
                ],
                dtype=np.float64,
            )
            current_quaternion /= np.linalg.norm(current_quaternion)
            expected_quaternion /= np.linalg.norm(expected_quaternion)
            orientation_dot = float(
                np.clip(
                    abs(np.dot(current_quaternion, expected_quaternion)),
                    0.0,
                    1.0,
                )
            )
            orientation_error_deg = float(
                np.degrees(2.0 * np.arccos(orientation_dot))
            )
            if orientation_error_deg > max_orientation_error:
                raise CartesianJogSafetyViolation(
                    "captured suction orientation differs from the calibrated "
                    f"downward orientation by {orientation_error_deg:.3f} deg"
                )
            suction_before = self.suction.status()
            if suction_before.get("available") is not True:
                raise SuctionUnavailable(
                    suction_before.get("error")
                    or "suction serial device is unavailable"
                )
            if suction_before.get("engaged") is True:
                raise CartesianJogSafetyViolation(
                    "release suction before starting a new automatic pick"
                )

            if detection_override is None:
                detection_result = (
                    self.detect()
                    if task_id == "task1"
                    else self.detect(task_id=task_id)
                )
                detection = detection_result["detection"]
            else:
                detection = copy.deepcopy(detection_override)
                if detection.get("task_id") != task_id:
                    raise CartesianJogSafetyViolation(
                        "cached detection does not match the active task"
                    )
            if detection.get("target_ready") is not True:
                blockers = detection.get("blockers") or [
                    "detection target is not ready"
                ]
                raise CartesianJogSafetyViolation(
                    "cannot pick detected carton: " + "; ".join(map(str, blockers))
                )
            surface_center = np.asarray(
                detection.get("point_left_base_m"),
                dtype=np.float64,
            )
            layer_estimate = detection.get("layer_estimate")
            layer: int | None = None
            task1_stack_ticket: tuple[int, int, int] | None = None
            if task_id == "task1":
                if (
                    not isinstance(layer_estimate, dict)
                    or layer_estimate.get("valid") is not True
                ):
                    raise CartesianJogSafetyViolation(
                        "detected carton has no valid 1/2/3-layer estimate"
                    )
                layer = int(layer_estimate.get("layer", 0))
                stack_prior = getattr(self, "_task1_stack_prior", None)
                if stack_prior is not None and isinstance(
                    detection.get("task1_stack_prior"), dict
                ):
                    try:
                        task1_stack_ticket = (
                            stack_prior.validate_detection_ticket(detection)
                        )
                    except ValueError as exc:
                        raise CartesianJogSafetyViolation(
                            "Task1 stack slot is stale or already picked: "
                            f"{exc}"
                        ) from exc
                contact_z_by_layer = pick_cfg.get(
                    "contact_flange_z_m_by_layer",
                    {},
                )
                if not isinstance(contact_z_by_layer, dict):
                    raise CartesianJogUnavailable(
                        "task1 contact Z map is not configured"
                    )
                try:
                    fixed_contact_z = float(contact_z_by_layer[str(layer)])
                except (KeyError, TypeError, ValueError) as exc:
                    raise CartesianJogUnavailable(
                        f"task1 has no fixed contact Z for layer {layer}"
                    ) from exc
            elif task_id == "task2":
                try:
                    configured_contact_z = pick_cfg.get("contact_flange_z_m")
                    fixed_contact_z = float(
                        configured_contact_z
                        if configured_contact_z is not None
                        else (
                            pick_calibration["contact_sample"][
                                "absolute_contact_flange_z_m"
                            ]
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise CartesianJogUnavailable(
                        "task2 sampled single-carton contact Z is unavailable"
                    ) from exc
            else:
                try:
                    configured_contact_z = pick_cfg.get("contact_flange_z_m")
                    if configured_contact_z is None:
                        fixed_contact_z = float(pick_cfg["table_surface_z_m"]) + float(
                            pick_calibration["contact_sample"][
                                "surface_to_target_flange_offset_in_base_m"
                            ][2]
                        )
                    else:
                        fixed_contact_z = float(configured_contact_z)
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    raise CartesianJogUnavailable(
                        "task3 contact flange Z is unavailable"
                    ) from exc
                if not math.isfinite(fixed_contact_z):
                    raise CartesianJogUnavailable(
                        "task3 table contact Z is not finite"
                    )
            contact_offset = np.asarray(
                pick_calibration["contact_sample"][
                    "surface_to_target_flange_offset_in_base_m"
                ],
                dtype=np.float64,
            )
            shared_target_offset = np.asarray(
                getattr(self, "config", {}).get("shared_pick", {}).get(
                    "target_offset_left_base_m",
                    [0.0, 0.0, 0.0],
                ),
                dtype=np.float64,
            )
            task_target_offset = np.asarray(
                pick_cfg.get("target_offset_left_base_m", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            )
            if (
                shared_target_offset.shape != (3,)
                or task_target_offset.shape != (3,)
                or not np.all(np.isfinite(shared_target_offset))
                or not np.all(np.isfinite(task_target_offset))
            ):
                raise CartesianJogUnavailable(
                    "pick target offsets must contain three finite values"
                )
            combined_target_offset = shared_target_offset + task_target_offset
            if np.any(np.abs(combined_target_offset) > 0.03):
                raise CartesianJogUnavailable(
                    f"{task_id} combined pick target offset exceeds 30 mm"
                )
            if surface_center.shape != (3,) or not np.all(
                np.isfinite(surface_center)
            ):
                raise CartesianJogSafetyViolation(
                    "detected carton has no finite left-base surface center"
                )
            contact = surface_center + contact_offset + combined_target_offset
            contact[2] = fixed_contact_z
            pre_contact = contact.copy()
            pre_contact[2] += pre_contact_clearance
            transit_height = max(
                transit_z,
                float(current_position[2]),
                float(pre_contact[2]),
            )
            if prepared_pre_contact_position_m is not None:
                prepared_pre_contact = np.asarray(
                    prepared_pre_contact_position_m,
                    dtype=np.float64,
                )
                if (
                    prepared_pre_contact.shape != (3,)
                    or not np.all(np.isfinite(prepared_pre_contact))
                    or np.linalg.norm(prepared_pre_contact - pre_contact) > 0.001
                    or np.linalg.norm(current_position - pre_contact) > 0.003
                ):
                    raise CartesianJogSafetyViolation(
                        f"prepared {task_id} approach pose does not match "
                        "the current target"
                    )
                approach_targets = (
                    []
                    if np.linalg.norm(current_position - contact) <= 0.001
                    else [contact.tolist()]
                )
            else:
                approach_targets = [
                    [
                        float(current_position[0]),
                        float(current_position[1]),
                        transit_height,
                    ],
                    [float(contact[0]), float(contact[1]), transit_height],
                    pre_contact.tolist(),
                    contact.tolist(),
                ]
            if approach_targets:
                approach = self.cartesian_jog.move_fixed_orientation_path(
                    approach_targets,
                    operation=f"{task_id}_pick_approach",
                    calibrated_minimum_z_m=(
                        fixed_contact_z if task_id == "task3" else None
                    ),
                    **(
                        {
                            "calibrated_workspace_profile": str(
                                pick_cfg["calibrated_workspace_profile"]
                            )
                        }
                        if task_id == "task1"
                        and pick_cfg.get("calibrated_workspace_profile") is not None
                        else {}
                    ),
                )
            else:
                approach = {
                    "operation": f"{task_id}_pick_approach",
                    "executed": False,
                    "skipped": True,
                    "reason": "already_at_contact_pose",
                }
            suction_result = self.suction.set_engaged(True)
            if (
                task_id == "task1"
                and suction_result.get("engaged") is not True
            ):
                raise SuctionUnavailable(
                    suction_result.get("error")
                    or f"{task_id} suction did not engage"
                )
            if self.suction.settle_s:
                time.sleep(self.suction.settle_s)
            lift_target = contact.copy()
            lift_target[2] += test_lift
            try:
                lift = self.cartesian_jog.move_fixed_orientation_path(
                    [lift_target.tolist()],
                    operation=f"{task_id}_pick_test_lift",
                    calibrated_minimum_z_m=(
                        fixed_contact_z if task_id == "task3" else None
                    ),
                    **(
                        {
                            "calibrated_workspace_profile": str(
                                pick_cfg["calibrated_workspace_profile"]
                            )
                        }
                        if task_id == "task1"
                        and pick_cfg.get("calibrated_workspace_profile") is not None
                        else {}
                    ),
                )
            except (
                CartesianJogSafetyViolation,
                CartesianJogUnavailable,
                CartesianJogTimeout,
            ) as exc:
                raise type(exc)(
                    "suction is engaged, but the test lift did not complete: "
                    f"{exc}"
                ) from exc
            task1_stack_commit: dict[str, Any] | None = None
            if task1_stack_ticket is not None:
                stack_layer, stack_row, stack_column = task1_stack_ticket
                task1_stack_commit = stack_prior.mark_picked(
                    layer=stack_layer,
                    row=stack_row,
                    column=stack_column,
                )
                detection["task1_stack_prior_after_pick"] = copy.deepcopy(
                    task1_stack_commit
                )
            return {
                "ok": True,
                "result": {
                    "operation": f"{task_id}_pick_detected_carton",
                    "task_id": task_id,
                    "executed": True,
                    "detection_id": detection.get("id"),
                    "surface_center_left_base_m": surface_center.tolist(),
                    "shared_target_offset_left_base_m": (
                        shared_target_offset.tolist()
                    ),
                    "task_target_offset_left_base_m": task_target_offset.tolist(),
                    "combined_target_offset_left_base_m": (
                        combined_target_offset.tolist()
                    ),
                    "detected_layer": layer,
                    "task1_stack_prior": task1_stack_commit,
                    "layer_estimate": layer_estimate,
                    "contact_flange_position_m": contact.tolist(),
                    "pre_contact_flange_position_m": pre_contact.tolist(),
                    "test_lift_flange_position_m": lift_target.tolist(),
                    "locked_orientation_error_deg": orientation_error_deg,
                    "approach": approach,
                    "suction": suction_result,
                    "lift": lift,
                },
                "detection": detection,
                "cartesian_jog": self._cartesian_jog_snapshot(),
                "suction": self.suction.status(),
                f"{task_id}_pick": (
                    {
                        "task1": self.task1_pick_status,
                        "task2": self.task2_pick_status,
                        "task3": self.task3_pick_status,
                    }[task_id]()
                ),
            }

    def capture_cartesian_jog_orientation(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_exact_payload(
            payload,
            frozenset({"confirm", "vertical_down_confirmed"}),
        )
        if payload.get("confirm") != "CAPTURE_LEFT_SUCTION_DOWN":
            raise ValueError("confirm must be CAPTURE_LEFT_SUCTION_DOWN")
        if payload.get("vertical_down_confirmed") is not True:
            raise ValueError("vertical_down_confirmed=true is required")
        with self._motion_transition_lock:
            result = self.cartesian_jog.capture_orientation()
        return {
            "ok": True,
            "result": result,
            "cartesian_jog": self._cartesian_jog_snapshot(),
        }

    def enable_cartesian_jog(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_exact_payload(
            payload,
            frozenset({"confirm", "area_clear", "estop_ready"}),
        )
        if payload.get("confirm") != "ENABLE_LEFT_CARTESIAN_JOG":
            raise ValueError("confirm must be ENABLE_LEFT_CARTESIAN_JOG")
        with self._motion_transition_lock:
            self.cartesian_jog.enable(
                payload.get("confirm"),
                area_clear=payload.get("area_clear") is True,
                estop_ready=payload.get("estop_ready") is True,
            )
        return {
            "ok": True,
            "cartesian_jog": self._cartesian_jog_snapshot(),
        }

    def quick_enable_cartesian_jog(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Capture the current tool pose and enable XYZ jog in one request."""

        self._require_exact_payload(payload, frozenset({"confirm"}))
        confirm = payload.get("confirm")
        if confirm != "ENABLE_LEFT_CARTESIAN_JOG":
            raise ValueError("confirm must be ENABLE_LEFT_CARTESIAN_JOG")
        with self._motion_transition_lock:
            capture = self.cartesian_jog.capture_orientation()
            self.cartesian_jog.enable(
                confirm,
                area_clear=True,
                estop_ready=True,
            )
        return {
            "ok": True,
            "result": {"capture": capture, "enabled": True},
            "cartesian_jog": self._cartesian_jog_snapshot(),
        }

    def disable_cartesian_jog(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_exact_payload(payload, frozenset())
        self.cartesian_jog.disable()
        return {
            "ok": True,
            "cartesian_jog": self._cartesian_jog_snapshot(),
        }

    def restore_cartesian_jog_safe_pose(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_exact_payload(
            payload,
            frozenset(
                {
                    "confirm",
                    "area_clear",
                    "estop_ready",
                    "suction_released",
                }
            ),
        )
        if payload.get("confirm") != "RESTORE_LEFT_SAFE_VERTICAL":
            raise ValueError(
                "confirm must be RESTORE_LEFT_SAFE_VERTICAL"
            )
        with self._motion_transition_lock:
            if self.trajectory_recorder.status().get("active") is True:
                raise CartesianJogConflict(
                    "stop trajectory recording before restoring the safe pose"
                )
            result = self.cartesian_jog.restore_safe_vertical(
                payload.get("confirm"),
                area_clear=payload.get("area_clear") is True,
                estop_ready=payload.get("estop_ready") is True,
                suction_released=payload.get("suction_released") is True,
            )
        return {
            "ok": True,
            "result": result,
            "cartesian_jog": self._cartesian_jog_snapshot(),
        }

    def move_cartesian_jog(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_exact_payload(
            payload,
            frozenset({"axis", "direction", "step_mm"}),
        )
        axis = payload.get("axis")
        direction = payload.get("direction")
        step_mm = payload.get("step_mm")
        if not isinstance(axis, str):
            raise ValueError("axis must be x, y or z")
        if isinstance(direction, bool) or direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        if isinstance(step_mm, bool) or not isinstance(step_mm, int):
            raise ValueError("step_mm must be an integer")
        with self._motion_transition_lock:
            result = self.cartesian_jog.jog(
                axis,
                direction * step_mm,
            )
        return {
            "ok": True,
            "result": result,
            "cartesian_jog": self._cartesian_jog_snapshot(),
        }

    def _capture_for_read(self) -> Any:
        recording_active, preview = (
            self.trajectory_recorder.active_preview_frame()
        )
        if recording_active:
            if preview is None:
                raise CameraUnavailable("recording preview is not ready yet")
            return preview
        return self.camera.capture()

    def frame_jpeg(self, *, overlay_id: str | None = None) -> bytes:
        if overlay_id is not None:
            with self._lock:
                cached = self._overlay_jpegs.get(overlay_id)
            if cached is None:
                raise OverlayUnavailable(
                    "detection overlay is unknown or has expired"
                )
            return cached
        recording_active, preview = (
            self.trajectory_recorder.active_preview_frame()
        )
        if recording_active:
            if preview is None:
                raise CameraUnavailable("recording preview is not ready yet")
            return _encode_jpeg(preview.bgr)
        # One 8899 page polls at 10 FPS.  Reuse a very recent encoded frame so
        # multiple tabs do not repeat the full RGB-D copy and JPEG encode.
        with self._preview_lock:
            now = time.monotonic()
            if (
                self._preview_jpeg is not None
                and now - self._preview_jpeg_cached_at
                <= PREVIEW_JPEG_CACHE_TTL_S
            ):
                return self._preview_jpeg
            frame = self.camera.capture()
            encoded = _encode_jpeg(frame.bgr)
            self._preview_jpeg = encoded
            self._preview_jpeg_cached_at = time.monotonic()
            return encoded

    def detect(self, *, task_id: str = "task1") -> dict[str, Any]:
        """Run bounded multi-frame detection and require a stable 3-D target."""

        if task_id not in {"task1", "task2", "task3"}:
            raise ValueError("task_id must be task1, task2 or task3")
        profile = dict(self.task_profiles_cfg.get(task_id, {}))
        attempt_limit = 5 if task_id in {"task1", "task3"} else 3
        attempts = max(
            1,
            min(
                attempt_limit,
                int(profile.get("detection_attempts", 1)),
            ),
        )
        required = max(
            1,
            min(
                attempts,
                int(profile.get("required_consistent_detections", 1)),
            ),
        )
        tolerance = float(profile.get("consensus_tolerance_m", 0.030))
        if not math.isfinite(tolerance) or not 0.005 <= tolerance <= 0.100:
            tolerance = 0.030
        results: list[dict[str, Any]] = []
        ready_results: list[dict[str, Any]] = []
        adaptive_task2 = task_id == "task2" and isinstance(
            self.task2_detector,
            Task2AdaptiveVisualDetector,
        )
        complete_frame_task1 = task_id == "task1"
        for attempt_index in range(attempts):
            response = self._detect_once(task_id=task_id)
            result = response.get("detection")
            if not isinstance(result, dict):
                raise RuntimeError("single-frame detection returned no payload")
            results.append(response)
            if result.get("target_ready") is not True:
                continue
            point = np.asarray(result.get("point_left_base_m"), dtype=np.float64)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                continue
            # Task2 has an adaptive three/four-carton model.  Do not return
            # early merely because the selected suction point is stable: a
            # glare-obscured fourth carton may be recovered in a later frame.
            # Collect every bounded attempt, then prefer the highest count
            # supported by at least the configured number of frames.
            if complete_frame_task1:
                # Task1 has six simultaneously visible source cartons.  A
                # stable suction point from an incomplete early frame must
                # not hide a later frame containing more independently
                # measured cartons.  Collect every bounded attempt, then
                # choose the most complete frame whose own target also has
                # the configured temporal support.
                ready_results.append(result)
                continue
            if adaptive_task2:
                ready_results.append(result)
                if int(result.get("instance_count", 0)) >= 4:
                    result["temporal_consensus"] = {
                        "valid": True,
                        "attempts_used": attempt_index + 1,
                        "required": required,
                        "carton_count": 4,
                        "count_support": {"3": 0, "4": 1},
                        "policy": "complete_maximum_count_early_stop",
                        "tolerance_m": tolerance,
                    }
                    with self._lock:
                        self._last_detection = result
                    return response
                continue
            if required <= 1:
                result["temporal_consensus"] = {
                    "valid": True,
                    "attempts_used": attempt_index + 1,
                    "required": required,
                    "tolerance_m": tolerance,
                }
                with self._lock:
                    self._last_detection = result
                return response
            for previous in ready_results:
                previous_point = np.asarray(
                    previous.get("point_left_base_m"), dtype=np.float64
                )
                delta = float(np.linalg.norm(point - previous_point))
                if delta <= tolerance:
                    result["temporal_consensus"] = {
                        "valid": True,
                        "attempts_used": attempt_index + 1,
                        "required": required,
                        "matched_distance_m": delta,
                        "tolerance_m": tolerance,
                    }
                    with self._lock:
                        self._last_detection = result
                    return response
            ready_results.append(result)

        if complete_frame_task1 and ready_results:
            stable_results: list[tuple[dict[str, Any], int, float]] = []
            for current_index, current in enumerate(ready_results):
                current_point = np.asarray(
                    current.get("point_left_base_m"), dtype=np.float64
                )
                distances = [
                    float(
                        np.linalg.norm(
                            current_point
                            - np.asarray(
                                other.get("point_left_base_m"),
                                dtype=np.float64,
                            )
                        )
                    )
                    for other_index, other in enumerate(ready_results)
                    if other_index != current_index
                ]
                matched = [distance for distance in distances if distance <= tolerance]
                support = 1 + len(matched)
                if support >= required:
                    stable_results.append(
                        (
                            current,
                            support,
                            min(matched) if matched else 0.0,
                        )
                    )
            if stable_results:
                winner, support, matched_distance = max(
                    stable_results,
                    key=lambda item: (
                        int(
                            (item[0].get("layer_estimate") or {}).get(
                                "layer", 0
                            )
                        ),
                        int(
                            item[0].get(
                                "recognized_count",
                                item[0].get("instance_count", 0),
                            )
                        ),
                        int(item[0].get("safe_grasp_candidate_count", 0)),
                    ),
                )
                for response in reversed(results):
                    if response.get("detection") is winner:
                        winner["temporal_consensus"] = {
                            "valid": True,
                            "attempts_used": attempts,
                            "required": required,
                            "matched_support": support,
                            "matched_distance_m": matched_distance,
                            "carton_count": int(
                                winner.get(
                                    "recognized_count",
                                    winner.get("instance_count", 0),
                                )
                            ),
                            "policy": (
                                "prefer_highest_stable_layer_then_most_complete_"
                                "task1_frame"
                            ),
                            "tolerance_m": tolerance,
                        }
                        with self._lock:
                            self._last_detection = winner
                        return response

        if adaptive_task2 and ready_results:
            support = {
                count: [
                    result
                    for result in ready_results
                    if int(result.get("instance_count", 0)) == count
                ]
                for count in (3, 4)
            }
            # A recovered fourth carton has already passed four independent
            # RGB/depth edge gates plus non-overlap and physical-size checks;
            # one such complete observation is stronger evidence than a frame
            # where glare simply hid it.  Three-carton absence still needs the
            # configured repeated support.
            supported_counts = [
                count
                for count in (3, 4)
                if len(support[count]) >= (1 if count == 4 else required)
            ]
            if supported_counts:
                winning_count = max(supported_counts)
                winner = support[winning_count][-1]
                for response in reversed(results):
                    if response.get("detection") is winner:
                        winner["temporal_consensus"] = {
                            "valid": True,
                            "attempts_used": attempts,
                            "required": required,
                            "carton_count": winning_count,
                            "count_support": {
                                "3": len(support[3]),
                                "4": len(support[4]),
                            },
                            "policy": "prefer_highest_supported_adaptive_count",
                            "tolerance_m": tolerance,
                        }
                        with self._lock:
                            self._last_detection = winner
                        return response

        best_response = max(
            results,
            key=lambda response: (
                response["detection"].get("target_ready") is True,
                int(response["detection"].get("instance_count", 0)),
            ),
        )
        best = best_response["detection"]
        if required > 1 and ready_results:
            best = copy.deepcopy(best)
            best["target_ready"] = False
            best["blockers"] = list(
                dict.fromkeys(
                    (*best.get("blockers", []), "target_not_temporally_consistent")
                )
            )
            best["temporal_consensus"] = {
                "valid": False,
                "attempts_used": attempts,
                "required": required,
                "tolerance_m": tolerance,
            }
            best_response = {**best_response, "detection": best}
        with self._lock:
            self._last_detection = best
        return best_response

    def _detect_once(self, *, task_id: str = "task1") -> dict[str, Any]:
        if task_id not in {"task1", "task2", "task3"}:
            raise ValueError("task_id must be task1, task2 or task3")
        frame = self._capture_for_read()
        rgb = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB)
        task_profile = dict(self.task_profiles_cfg.get(task_id, {}))
        # OpenCV's homography RANSAC otherwise changes its hypothesis between
        # identical frames, which makes repeated carton patterns flicker.
        cv2.setRNGSeed(int(task_profile.get("opencv_rng_seed", 0)))
        active_detector = {
            "task1": self.detector,
            "task2": self.task2_detector,
            "task3": self.task3_detector,
        }[task_id]
        if isinstance(active_detector, Task2AdaptiveVisualDetector):
            raw_candidates = active_detector.detect_rgbd(
                rgb,
                frame.depth_z16,
                frame.depth_scale_m,
            )
        else:
            raw_candidates = active_detector.detect(rgb)
        task1_recovery_report: dict[str, Any] | None = None
        task1_stack_report: dict[str, Any] | None = None
        if task_id == "task1":
            raw_candidates, task1_recovery_report = (
                recover_task1_grid_candidates(
                    rgb,
                    frame.depth_z16,
                    frame.depth_scale_m,
                    raw_candidates,
                    roi_norm=task_profile.get(
                        "include_roi_norm", [0.0, 0.0, 1.0, 1.0]
                    ),
                    config={**self.detector_cfg, **task_profile},
                )
            )
        individual_front_similarity = (
            task_profile.get("require_individual_front_similarity") is True
        )
        if individual_front_similarity:
            task_roi = task_profile.get(
                "include_roi_norm", [0.0, 0.0, 1.0, 1.0]
            )
            physical_profile = task_profile.get("physical_instance_gate", {})
            if not isinstance(physical_profile, dict):
                physical_profile = {}
            expected_face_size = physical_profile.get(
                "expected_face_size_mm", [130.0, 85.0]
            )
            long_side_range = task_profile.get(
                "front_similarity_long_side_px_range", [80.0, 180.0]
            )
            short_side_range = task_profile.get(
                "front_similarity_short_side_px_range", [50.0, 120.0]
            )
            verified_front_candidates: list[BoxCandidate] = []
            for candidate in raw_candidates:
                if (
                    candidate.face_type != "front_large"
                    or not candidate.reference_face_id
                    or not normalized_roi_contains_point(
                        candidate.center_px,
                        rgb.shape[:2],
                        task_roi,
                    )
                ):
                    continue
                normalized = None
                is_motif = ":motif" in str(candidate.provider)
                if (
                    is_motif
                    and frame.depth_z16 is not None
                    and frame.depth_scale_m is not None
                    and self.camera.intrinsics is not None
                ):
                    center_x = int(round(float(candidate.center_px[0])))
                    center_y = int(round(float(candidate.center_px[1])))
                    radius = 6
                    depth_patch = frame.depth_z16[
                        max(0, center_y - radius) : min(
                            frame.depth_z16.shape[0], center_y + radius + 1
                        ),
                        max(0, center_x - radius) : min(
                            frame.depth_z16.shape[1], center_x + radius + 1
                        ),
                    ]
                    valid_depth = depth_patch[depth_patch > 0]
                    if valid_depth.size >= 8:
                        depth_m = float(
                            np.median(valid_depth.astype(np.float64))
                        ) * float(frame.depth_scale_m)
                        normalized = normalize_front_face_geometry_at_depth(
                            candidate,
                            depth_m,
                            self.camera.intrinsics,
                            expected_face_size,
                        )
                if is_motif and normalized is None:
                    continue
                accepted = normalized or candidate
                if is_motif:
                    try:
                        size_valid = (
                            float(long_side_range[0])
                            <= float(accepted.long_side_px)
                            <= float(long_side_range[1])
                            and float(short_side_range[0])
                            <= float(accepted.short_side_px)
                            <= float(short_side_range[1])
                        )
                    except (IndexError, TypeError, ValueError):
                        size_valid = False
                    if not size_valid:
                        continue
                verified_front_candidates.append(accepted)
            raw_candidates = verified_front_candidates
        # Task1's adaptive SIFT/RANSAC detector already returns one measured
        # perspective quadrilateral per visible carton.  The legacy surface
        # grid is useful only as an explicitly enabled recovery mode: it
        # replaces those measurements with synthetic axis-aligned cells, so
        # it must stay off when pixel geometry is used to plan suction.
        if task_id == "task1" and bool(
            task_profile.get("surface_grid_completion_enabled", False)
        ):
            fitted_grid = propose_task1_surface_grid(
                rgb,
                raw_candidates,
                roi_norm=task_profile.get(
                    "include_roi_norm", [0.0, 0.0, 1.0, 1.0]
                ),
                config={**self.detector_cfg, **task_profile},
                rows=3,
                columns=3,
            )
            if fitted_grid:
                raw_candidates = fitted_grid
        if task_id == "task1":
            raw_candidates, task1_stack_report = (
                self._task1_stack_prior.filter_candidates(
                    raw_candidates,
                    image_shape=rgb.shape[:2],
                    layout_polygon_norm=task_profile.get(
                        "adaptive_slot_polygon_norm",
                        task_profile.get("include_polygon_norm"),
                    ),
                )
            )
        if should_recover_task_row(
            task_id,
            individual_front_similarity=individual_front_similarity,
            task_profile=task_profile,
        ) and not isinstance(active_detector, Task2AdaptiveVisualDetector):
            # One verified Task2 front establishes medicine-carton identity;
            # the existing RGB-D row fitter can then recover glare-obscured
            # neighbours before every normal per-instance safety gate runs.
            # Task3 also uses this fitter when it is not in per-face mode.
            row_config = {**self.detector_cfg, **task_profile}
            if task_id == "task3":
                flat_size_mm = task_profile.get("grid_cell_size_mm")
                if (
                    isinstance(flat_size_mm, (list, tuple))
                    and len(flat_size_mm) == 2
                ):
                    row_dual_config = dict(
                        self.detector_cfg.get("dual_suction", {})
                    )
                    row_dual_config["carton_face_size_mm"] = [
                        float(flat_size_mm[0]),
                        float(flat_size_mm[1]),
                    ]
                    row_config["dual_suction"] = row_dual_config
            fitted_row = propose_task2_single_row(
                rgb,
                frame.depth_z16,
                frame.depth_scale_m,
                self.camera.intrinsics,
                raw_candidates,
                roi_norm=task_profile.get(
                    "include_roi_norm", [0.0, 0.0, 1.0, 1.0]
                ),
                config=row_config,
                maximum_count=int(
                    task_profile.get("grid_maximum_count_per_axis", 4)
                ),
            )
            if fitted_row:
                if task_id == "task3":
                    fitted_row = [
                        replace(
                            candidate,
                            provider=str(candidate.provider).replace(
                                ":task2_row_cell", ":task3_row_cell"
                            ),
                        )
                        for candidate in fitted_row
                    ]
                raw_candidates = fitted_row
        split_config = copy.deepcopy(self.detector_cfg)
        grid_config = dict(split_config.get("grid_split", {}))
        grid_config["enabled"] = True
        grid_config["maximum_count_per_axis"] = int(
            task_profile.get(
                "grid_maximum_count_per_axis",
                3 if task_id == "task1" else (4 if task_id == "task2" else 2),
            )
        )
        grid_shape_policy = str(
            task_profile.get("grid_shape_policy", "any")
        ).strip().lower()
        grid_config["shape_policy"] = grid_shape_policy
        grid_config["drop_unsupported_cells"] = bool(
            task_profile.get("drop_unsupported_grid_cells", False)
        )
        task_grid_cell_size = task_profile.get("grid_cell_size_mm")
        if (
            isinstance(task_grid_cell_size, (list, tuple))
            and len(task_grid_cell_size) == 2
        ):
            split_dual_config = dict(split_config.get("dual_suction", {}))
            split_dual_config["carton_face_size_mm"] = [
                float(task_grid_cell_size[0]),
                float(task_grid_cell_size[1]),
            ]
            split_config["dual_suction"] = split_dual_config
        task_cell_fraction = task_profile.get(
            "grid_minimum_cell_pink_fraction"
        )
        if task_cell_fraction is not None:
            grid_config["minimum_cell_pink_fraction"] = float(
                task_cell_fraction
            )
        preferred_grid_shape = task_profile.get("preferred_grid_shape")
        if (
            grid_shape_policy != "single_axis_dynamic"
            and
            isinstance(preferred_grid_shape, (list, tuple))
            and len(preferred_grid_shape) == 2
        ):
            grid_config["preferred_grid_shape"] = [
                int(preferred_grid_shape[0]),
                int(preferred_grid_shape[1]),
            ]
            grid_config["preferred_grid_count_tolerance"] = float(
                task_profile.get(
                    "preferred_grid_count_tolerance",
                    grid_config.get("integer_count_tolerance", 0.55),
                )
            )
        else:
            grid_config.pop("preferred_grid_shape", None)
            grid_config.pop("preferred_grid_count_tolerance", None)
        split_config["grid_split"] = grid_config
        split_candidates = (
            list(raw_candidates)
            if individual_front_similarity
            else [
                cell
                for candidate in raw_candidates
                for cell in split_carton_grid_candidate(
                    candidate,
                    frame.depth_z16,
                    frame.depth_scale_m,
                    self.camera.intrinsics,
                    split_config,
                    rgb=rgb,
                )
            ]
        )
        task3_projector = getattr(self, "task3_front_panel_projector", None)
        if task_id == "task3" and task3_projector is not None:
            require_front_panel_verification = (
                task_profile.get("require_front_panel_verification", True)
                is True
            )
            task3_flat_instance_config: dict[str, Any] | None = None
            task3_physical_gate = task_profile.get(
                "physical_instance_gate", {}
            )
            if not isinstance(task3_physical_gate, dict):
                task3_physical_gate = {}
            task3_face_size_mm = task3_physical_gate.get(
                "expected_face_size_mm", [130.0, 85.0]
            )
            task3_flat_size_mm = task_profile.get(
                "grid_cell_size_mm", [205.0, 130.0]
            )
            if (
                isinstance(task3_flat_size_mm, (list, tuple))
                and len(task3_flat_size_mm) == 2
            ):
                flat_gate = copy.deepcopy(task3_physical_gate)
                flat_gate["enabled"] = True
                flat_gate["expected_face_size_mm"] = list(task3_flat_size_mm)
                flat_gate["tolerance_mm"] = list(
                    task_profile.get(
                        "flat_footprint_tolerance_mm", [35.0, 30.0]
                    )
                )
                task3_flat_instance_config = copy.deepcopy(self.detector_cfg)
                task3_flat_instance_config["physical_instance_gate"] = flat_gate
            front_panel_candidates: list[BoxCandidate] = []
            for candidate in split_candidates:
                provider_name = str(candidate.provider)
                independently_verified_front = any(
                    marker in provider_name
                    for marker in (
                        ":motif",
                        ":front_similarity_",
                        ":multi_sift",
                        ":rgbd_four_edge",
                    )
                )
                projected = (
                    replace(
                        candidate,
                        provider=f"{candidate.provider}:task3_front_similarity",
                    )
                    if individual_front_similarity
                    and independently_verified_front
                    and candidate.face_type == "front_large"
                    and bool(candidate.reference_face_id)
                    else task3_projector.project(rgb, candidate)
                )
                if projected is None and task3_flat_instance_config is not None:
                    flat_report = estimate_candidate_physical_size_rgbd(
                        candidate,
                        frame.depth_z16,
                        frame.depth_scale_m,
                        self.camera.intrinsics,
                        task3_flat_instance_config,
                    )
                    projected = recover_task3_verified_flat_face_geometry(
                        candidate,
                        flat_report,
                        self.camera.intrinsics,
                        task3_face_size_mm,
                        require_verified_front=(
                            require_front_panel_verification
                        ),
                    )
                if projected is None:
                    if require_front_panel_verification:
                        candidate = replace(
                            candidate,
                            graspable=False,
                            grasp_blockers=tuple(
                                dict.fromkeys(
                                    (
                                        *candidate.grasp_blockers,
                                        "task3_front_panel_unverified",
                                    )
                                )
                            ),
                        )
                    front_panel_candidates.append(candidate)
                else:
                    front_panel_candidates.append(projected)
            split_candidates = front_panel_candidates
        physical_gate = dict(task_profile.get("physical_instance_gate", {}))
        instance_config = copy.deepcopy(self.detector_cfg)
        instance_config["physical_instance_gate"] = physical_gate
        instance_reports: dict[int, dict[str, Any]] = {}
        candidates: list[Any] = []
        for candidate in split_candidates:
            report = estimate_candidate_physical_size_rgbd(
                candidate,
                frame.depth_z16,
                frame.depth_scale_m,
                self.camera.intrinsics,
                instance_config,
            )
            if (
                task_id == "task3"
                and report.get("valid") is True
                and candidate.face_type == "front_large"
                and bool(candidate.reference_face_id)
                and (
                    ":task3_front_similarity" in str(candidate.provider)
                    or str(candidate.provider).endswith(
                        "task3_front_homography"
                    )
                )
            ):
                normalized = normalize_task3_front_face_geometry(
                    candidate,
                    report,
                    self.camera.intrinsics,
                    physical_gate.get("expected_face_size_mm", [130.0, 85.0]),
                )
                if normalized is not None:
                    candidate = normalized
                    report = estimate_candidate_physical_size_rgbd(
                        candidate,
                        frame.depth_z16,
                        frame.depth_scale_m,
                        self.camera.intrinsics,
                        instance_config,
                    )
            gated_candidate = candidate
            if report.get("valid") is not True:
                gated_candidate = replace(
                    candidate,
                    graspable=False,
                    grasp_blockers=tuple(
                        dict.fromkeys(
                            (*candidate.grasp_blockers, "physical_size")
                        )
                    ),
                )
            candidates.append(gated_candidate)
            instance_reports[id(gated_candidate)] = report
        detector_status = dict(active_detector.status())
        if task1_recovery_report is not None:
            detector_status["task1_surface_recovery"] = task1_recovery_report
        if task1_stack_report is not None:
            detector_status["task1_stack_prior"] = task1_stack_report
        detector_status["task_profile"] = (
            "task1_3x3x3_dynamic_instances"
            if task_id == "task1"
            else (
                "task2_four_carton_instances"
                if task_id == "task2"
                else "task3_flat_carton_2x2_instances"
            )
        )
        depth_eval_cfg = self.detector_cfg
        if task_id in {"task2", "task3"}:
            depth_eval_cfg = dict(self.detector_cfg)
            task_pick_cfg = (
                self.task2_pick_cfg
                if task_id == "task2"
                else self.task3_pick_cfg
            )
            task_dual_cfg = dict(depth_eval_cfg.get("dual_suction", {}))
            task_dual_cfg["min_depth_valid_ratio"] = float(
                task_pick_cfg.get("min_depth_valid_ratio", 0.4)
            )
            depth_eval_cfg["dual_suction"] = task_dual_cfg
        minimum_score = float(self.detector_cfg.get("min_detection_score", 0.68))
        detected_candidates = [
            item for item in candidates if item.score >= minimum_score
        ]
        physical_instance_candidates = [
            item
            for item in detected_candidates
            if instance_reports.get(id(item), {}).get("valid") is True
        ]
        dual_plan_config = copy.deepcopy(self.detector_cfg)
        task_dual_config = dict(dual_plan_config.get("dual_suction", {}))
        task_alignment = task_profile.get("dual_suction_alignment")
        if task_alignment is not None:
            task_dual_config["alignment"] = str(task_alignment)
        task_safety_margin = task_profile.get("dual_suction_safety_margin_mm")
        if task_safety_margin is not None:
            task_dual_config["safety_margin_mm"] = float(task_safety_margin)
        required_face_types = task_profile.get("required_face_types")
        if isinstance(required_face_types, list) and required_face_types:
            task_dual_config["required_face_types"] = [
                str(value) for value in required_face_types
            ]
        enforce_required_face_types = task_profile.get(
            "dual_suction_enforce_required_face_types"
        )
        if enforce_required_face_types is not None:
            task_dual_config["enforce_required_face_types"] = bool(
                enforce_required_face_types
            )
        dual_plan_config["dual_suction"] = task_dual_config
        dual_targets = {
            id(item): plan_dual_suction_target(
                item,
                dual_plan_config,
                image_shape=rgb.shape[:2],
            )
            for item in candidates
        }
        graspable_candidates = [
            item for item in physical_instance_candidates if item.graspable
        ]
        fixed_status = fixed_suction_axis_status(self.fixed_suction_axis_cfg)
        fixed_selection_required = bool(
            fixed_status.get("ready") is True
            and self.cam_to_left is not None
            and self.camera.intrinsics is not None
        )
        dual_graspable_candidates = [
            item
            for item in graspable_candidates
            if allow_fixed_axis_validation_after_generic_margin(
                task_id,
                dual_targets[id(item)],
                fixed_axis_ready=fixed_selection_required,
            )
        ]
        dual_depth_supports = {
            id(item): evaluate_dual_suction_depth(
                frame.depth_z16,
                frame.depth_scale_m,
                dual_targets[id(item)],
                depth_eval_cfg,
            )
            for item in dual_graspable_candidates
        }
        depth_ready_candidates = [
            item
            for item in dual_graspable_candidates
            if dual_depth_supports[id(item)].get("valid") is True
        ]

        candidate_midpoint_depths: dict[int, float] = {}
        candidate_camera_points: dict[int, tuple[float, float, float]] = {}
        candidate_base_points: dict[int, tuple[float, float, float]] = {}
        if self.camera.intrinsics is not None:
            # Build a provisional 3-D midpoint for every geometrically valid
            # carton that has any usable cup-depth samples.  The generic
            # carton-axis depth gate must not discard a candidate before the
            # real calibrated fixed cups (which can project on a different
            # image axis) have been evaluated.
            for item in dual_graspable_candidates:
                support = dual_depth_supports[id(item)]
                cup_depths = [
                    float(cup["median_depth_m"])
                    for cup in support.get("cups", [])
                    if cup.get("median_depth_m") is not None
                ]
                if cup_depths:
                    midpoint_depth = float(
                        np.median(np.asarray(cup_depths, dtype=np.float64))
                    )
                else:
                    midpoint_depth = physical_instance_center_depth_m(
                        instance_reports.get(id(item), {})
                    )
                    if midpoint_depth is None and isinstance(
                        active_detector,
                        Task2AdaptiveVisualDetector,
                    ):
                        interior_fallback = carton_interior_depth_fallback(
                            item,
                            frame.depth_z16,
                            frame.depth_scale_m,
                        )
                        report = dict(instance_reports.get(id(item), {}))
                        report["interior_depth_fallback"] = interior_fallback
                        instance_reports[id(item)] = report
                        if interior_fallback.get("valid") is True:
                            midpoint_depth = float(
                                interior_fallback["median_depth_m"]
                            )
                    if midpoint_depth is None:
                        continue
                midpoint = dual_targets[id(item)].midpoint_px
                point = deproject_pixel(
                    (int(round(midpoint[0])), int(round(midpoint[1]))),
                    midpoint_depth * 1000.0,
                    np.asarray(self.camera.intrinsics, dtype=np.float64),
                )
                candidate_midpoint_depths[id(item)] = midpoint_depth
                candidate_camera_points[id(item)] = tuple(
                    float(value) for value in point
                )
                if self.cam_to_left is not None:
                    point_base = transform_point(point, self.cam_to_left)
                    candidate_base_points[id(item)] = tuple(
                        float(value) for value in point_base
                    )

        surface_range_value = task_profile.get("surface_z_range_left_base_m")
        surface_range: tuple[float, float] | None = None
        if (
            isinstance(surface_range_value, (list, tuple))
            and len(surface_range_value) == 2
        ):
            lower = float(surface_range_value[0])
            upper = float(surface_range_value[1])
            if math.isfinite(lower) and math.isfinite(upper) and lower < upper:
                surface_range = (lower, upper)
        required_orientation_value = task_profile.get(
            "required_long_axis_orientation_image_deg"
        )
        required_orientation: float | None = None
        orientation_tolerance = max(
            0.0,
            float(task_profile.get("orientation_tolerance_deg", 20.0)),
        )
        if required_orientation_value is not None:
            parsed_orientation = float(required_orientation_value)
            if (
                math.isfinite(parsed_orientation)
                and 0.0 < orientation_tolerance <= 90.0
            ):
                required_orientation = parsed_orientation
        include_roi = task_profile.get("include_roi_norm")
        include_polygon = task_profile.get("include_polygon_norm")
        exclude_rois_value = task_profile.get("exclude_roi_norms", [])
        exclude_rois = (
            list(exclude_rois_value)
            if isinstance(exclude_rois_value, (list, tuple))
            else []
        )
        # Recognition and grasp safety are separate contracts.  A 3x3 grid
        # cell remains a recognized medicine carton even when sparse/noisy
        # depth makes it unsuitable for this cycle's fixed-suction grasp.
        recognized_task_candidates = [
            item
            for item in detected_candidates
            if (
                include_roi is None
                or normalized_roi_contains_point(
                    item.center_px,
                    rgb.shape[:2],
                    include_roi,
                )
            )
            if (
                include_polygon is None
                or normalized_polygon_contains_point(
                    item.center_px,
                    rgb.shape[:2],
                    include_polygon,
                )
            )
            if not any(
                normalized_roi_intersects_polygon(
                    item.polygon_px,
                    rgb.shape[:2],
                    roi,
                )
                for roi in exclude_rois
            )
        ]
        recognized_task_ids = {
            id(item) for item in recognized_task_candidates
        }
        task_instance_candidates: list[Any] = []
        surface_gate_tolerance = max(
            0.0,
            float(task_profile.get("surface_z_gate_tolerance_m", 0.0)),
        )
        for item in physical_instance_candidates:
            report = dict(instance_reports.get(id(item), {}))
            point = candidate_base_points.get(id(item))
            include_valid = (
                True
                if include_roi is None
                else normalized_roi_contains_point(
                    item.center_px,
                    rgb.shape[:2],
                    include_roi,
                )
            )
            polygon_valid = (
                True
                if include_polygon is None
                else normalized_polygon_contains_point(
                    item.center_px,
                    rgb.shape[:2],
                    include_polygon,
                )
            )
            excluded_indices = [
                index
                for index, roi in enumerate(exclude_rois)
                if normalized_roi_intersects_polygon(
                    item.polygon_px,
                    rgb.shape[:2],
                    roi,
                )
            ]
            region_valid = include_valid and polygon_valid and not excluded_indices
            report["task_region_gate"] = {
                "enabled": (
                    include_roi is not None
                    or include_polygon is not None
                    or bool(exclude_rois)
                ),
                "include_roi_norm": include_roi,
                "include_polygon_norm": include_polygon,
                "exclude_roi_norms": exclude_rois,
                "inside_include_roi": include_valid,
                "inside_include_polygon": polygon_valid,
                "matched_exclusion_indices": excluded_indices,
                "valid": region_valid,
            }
            if not region_valid:
                report["valid"] = False
                report["blockers"] = list(
                    dict.fromkeys(
                        (*report.get("blockers", []), "task_region_mismatch")
                    )
                )
            surface_valid = True
            if surface_range is not None:
                surface_valid = bool(
                    point is not None
                    and surface_range[0] - surface_gate_tolerance
                    <= float(point[2])
                    <= surface_range[1] + surface_gate_tolerance
                )
                report["surface_z_gate"] = {
                    "enabled": True,
                    "range_left_base_m": list(surface_range),
                    "tolerance_m": surface_gate_tolerance,
                    "measured_left_base_m": (
                        None if point is None else float(point[2])
                    ),
                    "valid": surface_valid,
                }
                if not surface_valid:
                    report["valid"] = False
                    report["blockers"] = list(
                        dict.fromkeys(
                            (*report.get("blockers", []), "task_surface_height_mismatch")
                        )
                    )
            orientation_valid = True
            if required_orientation is not None:
                orientation_error = axial_orientation_error_deg(
                    float(item.angle_deg),
                    required_orientation,
                )
                orientation_valid = orientation_error <= orientation_tolerance
                report["task_orientation_gate"] = {
                    "enabled": True,
                    "axis": "carton_long_axis",
                    "frame": "image_xy",
                    "required_deg": required_orientation,
                    "tolerance_deg": orientation_tolerance,
                    "measured_deg": float(item.angle_deg),
                    "error_deg": orientation_error,
                    "valid": orientation_valid,
                }
                if not orientation_valid:
                    report["valid"] = False
                    report["blockers"] = list(
                        dict.fromkeys(
                            (*report.get("blockers", []), "task_orientation_mismatch")
                        )
                    )
            instance_reports[id(item)] = report
            if region_valid and surface_valid and orientation_valid:
                task_instance_candidates.append(item)

        instance_groups = cluster_carton_instances(task_instance_candidates)
        layout_candidates = list(task_instance_candidates)
        if task_id in {"task2", "task3"} and instance_groups:
            reference_xy = np.asarray(
                self.task2_pick_cfg.get(
                    "station_reference_left_base_xy_m",
                    [0.25606636, -0.18517659],
                ),
                dtype=np.float64,
            )

            def group_distance(group: list[Any]) -> float:
                distances: list[float] = []
                for item in group:
                    point = candidate_base_points.get(id(item))
                    if point is None:
                        continue
                    if task_id == "task2" and reference_xy.shape == (2,):
                        distances.append(
                            math.hypot(
                                float(point[0]) - float(reference_xy[0]),
                                float(point[1]) - float(reference_xy[1]),
                            )
                        )
                    else:
                        distances.append(
                            math.hypot(float(point[0]), float(point[1]))
                        )
                return min(distances) if distances else float("inf")

            layout_candidates = list(min(instance_groups, key=group_distance))
        layout_ids = {id(item) for item in layout_candidates}
        layout_graspable_candidates = [
            item for item in graspable_candidates if id(item) in layout_ids
        ]
        layout_dual_graspable_candidates = [
            item for item in dual_graspable_candidates if id(item) in layout_ids
        ]
        layout_depth_ready_candidates = [
            item for item in depth_ready_candidates if id(item) in layout_ids
        ]

        layer_cfg = (
            self.task1_pick_cfg.get("layer_estimation", {})
            if task_id == "task1"
            else {}
        )
        if not isinstance(layer_cfg, dict):
            layer_cfg = {}
        candidate_layer_estimates: dict[int, dict[str, Any]] = {}
        layer_fit_error: str | None = None
        if (
            layer_cfg.get("enabled") is True
            and candidate_camera_points
            and self.camera.intrinsics is not None
        ):
            try:
                provisional = next(
                    item
                    for item in physical_instance_candidates
                    if id(item) in candidate_camera_points
                )
                fitted = estimate_carton_layer(
                    frame.bgr,
                    frame.depth_z16,
                    frame.depth_scale_m,
                    self.camera.intrinsics,
                    candidate_camera_points[id(provisional)],
                    layer_cfg,
                )
                plane = fitted["table_plane_camera"]
                for item in physical_instance_candidates:
                    point = candidate_camera_points.get(id(item))
                    if point is not None:
                        candidate_layer_estimates[id(item)] = (
                            classify_carton_layer_from_plane(
                                point,
                                plane,
                                layer_cfg,
                            )
                        )
            except Exception as exc:
                layer_fit_error = f"{type(exc).__name__}: {exc}"
        if task_id == "task1":
            candidate_layer_estimates = {
                item_id: self._task1_stack_prior.constrain_layer_estimate(
                    estimate
                )
                for item_id, estimate in candidate_layer_estimates.items()
            }

        task_instance_ids = {id(item) for item in task_instance_candidates}
        layer_ready_candidates = [
            item
            for item in task_instance_candidates
            if id(item) in task_instance_ids
            if candidate_layer_estimates.get(id(item), {}).get("valid") is True
        ]
        if (
            task_id == "task1"
            and layer_cfg.get("enabled") is True
            and layer_ready_candidates
        ):
            task_instance_candidates = list(layer_ready_candidates)
            instance_groups = cluster_carton_instances(task_instance_candidates)
            layout_candidates = list(task_instance_candidates)
            layout_ids = {id(item) for item in layout_candidates}
            layout_graspable_candidates = [
                item for item in graspable_candidates if id(item) in layout_ids
            ]
            layout_dual_graspable_candidates = [
                item
                for item in dual_graspable_candidates
                if id(item) in layout_ids
            ]
            layout_depth_ready_candidates = [
                item for item in depth_ready_candidates if id(item) in layout_ids
            ]
        # Validate the real, calibrated fixed tool against every otherwise
        # viable RGB-D candidate before ranking.  Selecting first and only
        # then projecting the fixed cups can reject a cropped edge carton even
        # when another carton on the same highest layer is fully graspable.
        tcp_calibration = {
            "task1": self.task1_pick_calibration,
            "task2": self.task2_pick_calibration,
            "task3": self.task3_pick_calibration,
        }[task_id]
        fixed_projection_payloads: dict[int, dict[str, Any]] = {}
        fixed_depth_supports: dict[int, dict[str, Any]] = {}
        fixed_projection_failures: dict[int, tuple[str, ...]] = {}
        fixed_valid_candidate_ids: set[int] = set()
        if fixed_selection_required:
            for item in layout_dual_graspable_candidates:
                item_id = id(item)
                point_left_base = candidate_base_points.get(item_id)
                generic_target = dual_targets.get(item_id)
                if point_left_base is None or generic_target is None:
                    fixed_projection_failures[item_id] = (
                        "fixed_suction_candidate_point_unavailable",
                    )
                    continue
                try:
                    if tcp_calibration is None:
                        raise ValueError(
                            "locked flange calibration is unavailable"
                        )
                    projection = project_fixed_suction_axis(
                        midpoint_left_base_m=point_left_base,
                        locked_flange_quaternion_xyzw=tcp_calibration[
                            "locked_flange_quaternion_xyzw"
                        ],
                        axis_local_xyz=self.fixed_suction_axis_cfg[
                            "axis_local_xyz"
                        ],
                        approach_local_xyz=self.fixed_suction_axis_cfg[
                            "approach_local_xyz"
                        ],
                        cup_center_spacing_m=float(
                            self.fixed_suction_axis_cfg.get(
                                "cup_center_spacing_mm", 50.0
                            )
                        ) / 1000.0,
                        cup_diameter_m=float(
                            self.fixed_suction_axis_cfg.get(
                                "cup_diameter_mm", 25.0
                            )
                        ) / 1000.0,
                        safety_margin_m=float(
                            task_dual_config.get(
                                "safety_margin_mm",
                                self.fixed_suction_axis_cfg.get(
                                    "safety_margin_mm", 8.0
                                ),
                            )
                        ) / 1000.0,
                        cam_to_left=self.cam_to_left,
                        intrinsics=np.asarray(
                            self.camera.intrinsics, dtype=np.float64
                        ),
                        candidate_polygon_px=item.polygon_px,
                        image_shape=rgb.shape[:2],
                    )
                    projection_payload = projection.to_dict()
                    projection_payload["projected_cup_radius_px"] = float(
                        generic_target.projected_cup_radius_px
                    )
                    fixed_depth_target = SimpleNamespace(
                        cup_centers_px=projection.cup_centers_px,
                        projected_cup_radius_px=(
                            generic_target.projected_cup_radius_px
                        ),
                    )
                    fixed_depth_support = (
                        apply_fixed_suction_depth_plane_fallback(
                            evaluate_dual_suction_depth(
                                frame.depth_z16,
                                frame.depth_scale_m,
                                fixed_depth_target,
                                depth_eval_cfg,
                            )
                        )
                    )
                    if (
                        task_id == "task1"
                        and fixed_depth_support.get("valid") is not True
                    ):
                        fixed_depth_support = (
                            apply_task1_verified_center_depth_fallback(
                                fixed_depth_support,
                                physical_instance_center_depth_m(
                                    instance_reports.get(item_id, {})
                                ),
                                maximum_delta_m=float(
                                    self.task1_pick_cfg.get(
                                        "fixed_depth_center_plane_fallback_max_delta_m",
                                        0.012,
                                    )
                                ),
                            )
                        )
                    projection_payload["depth_support"] = fixed_depth_support
                    fixed_projection_payloads[item_id] = projection_payload
                    fixed_depth_supports[item_id] = fixed_depth_support
                    fixed_cup_depths = [
                        float(cup["median_depth_m"])
                        for cup in fixed_depth_support.get("cups", [])
                        if cup.get("median_depth_m") is not None
                    ]
                    if fixed_cup_depths:
                        fixed_midpoint_depth = float(
                            np.median(
                                np.asarray(fixed_cup_depths, dtype=np.float64)
                            )
                        )
                        fixed_midpoint = projection.midpoint_px
                        fixed_camera_point = deproject_pixel(
                            (
                                int(round(fixed_midpoint[0])),
                                int(round(fixed_midpoint[1])),
                            ),
                            fixed_midpoint_depth * 1000.0,
                            np.asarray(
                                self.camera.intrinsics, dtype=np.float64
                            ),
                        )
                        candidate_midpoint_depths[item_id] = (
                            fixed_midpoint_depth
                        )
                        candidate_camera_points[item_id] = tuple(
                            float(value) for value in fixed_camera_point
                        )
                        fixed_base_point = transform_point(
                            fixed_camera_point,
                            self.cam_to_left,
                        )
                        candidate_base_points[item_id] = tuple(
                            float(value) for value in fixed_base_point
                        )
                    failure_reasons = list(projection.blockers)
                    if fixed_depth_support.get("valid") is not True:
                        failure_reasons.append("fixed_suction_depth_invalid")
                    failure_reasons = list(dict.fromkeys(failure_reasons))
                    if failure_reasons:
                        fixed_projection_failures[item_id] = tuple(
                            failure_reasons
                        )
                    else:
                        fixed_valid_candidate_ids.add(item_id)
                except Exception as exc:
                    fixed_projection_failures[item_id] = (
                        "fixed_suction_projection_failed: "
                        f"{type(exc).__name__}: {exc}",
                    )

        selection_depth_ready_candidates = (
            [
                item
                for item in layout_dual_graspable_candidates
                if id(item) in fixed_valid_candidate_ids
            ]
            if fixed_selection_required
            else list(layout_depth_ready_candidates)
        )
        selection_layer_ready_candidates = (
            [
                item
                for item in layer_ready_candidates
                if id(item) in fixed_valid_candidate_ids
            ]
            if fixed_selection_required
            else list(layer_ready_candidates)
        )
        highest_nearest_candidate = select_highest_layer_nearest_base(
            selection_layer_ready_candidates,
            candidate_layer_estimates,
            candidate_base_points,
        )
        fallback_candidate = (
            selection_depth_ready_candidates[0]
            if selection_depth_ready_candidates
            else (
                None
                if fixed_selection_required
                else (
                    layout_dual_graspable_candidates[0]
                    if layout_dual_graspable_candidates
                    else (
                        layout_graspable_candidates[0]
                        if layout_graspable_candidates
                        else None
                    )
                )
            )
        )
        nearest_base_candidate = min(
            (
                item
                for item in selection_depth_ready_candidates
                if id(item) in candidate_base_points
            ),
            key=lambda item: math.hypot(
                float(candidate_base_points[id(item)][0]),
                float(candidate_base_points[id(item)][1]),
            ),
            default=None,
        )
        task2_station_candidate = None
        task2_station_matches: list[Any] = []
        task2_station_radius_required = bool(
            task_profile.get("require_station_radius_gate", False)
        )
        if task_id == "task2":
            reference_xy = np.asarray(
                self.task2_pick_cfg.get(
                    "station_reference_left_base_xy_m",
                    [0.25606636, -0.18517659],
                ),
                dtype=np.float64,
            )
            max_station_distance = float(
                self.task2_pick_cfg.get("max_station_distance_m", 0.18)
            )
            if (
                reference_xy.shape == (2,)
                and np.all(np.isfinite(reference_xy))
                and 0.01 <= max_station_distance <= 0.50
            ):
                candidates_with_points = [
                    item
                    for item in selection_depth_ready_candidates
                    if id(item) in candidate_base_points
                ]
                task2_station_matches = [
                    item
                    for item in candidates_with_points
                    if math.hypot(
                        float(candidate_base_points[id(item)][0])
                        - float(reference_xy[0]),
                        float(candidate_base_points[id(item)][1])
                        - float(reference_xy[1]),
                    ) <= max_station_distance
                ]
                selectable_station_candidates = (
                    task2_station_matches
                    if task2_station_radius_required
                    else candidates_with_points
                )
                if selectable_station_candidates:
                    # Task 2 has a fixed horizontal row of identical cartons.
                    # Selecting by distance to the calibration reference can
                    # jump between neighbours as RGB-D depth varies by a few
                    # millimetres.  Image X is the stable task-space identity:
                    # always consume the leftmost currently safe carton.
                    task2_station_candidate = min(
                        selectable_station_candidates,
                        key=lambda item: (
                            float(item.center_px[0]),
                            float(item.center_px[1]),
                        ),
                    )
        selected_candidate = (
            task2_station_candidate
            if task_id == "task2"
            else (
                (
                    nearest_base_candidate
                    if nearest_base_candidate is not None
                    else fallback_candidate
                )
                if task_id == "task3"
                else (
                    highest_nearest_candidate
                    if highest_nearest_candidate is not None
                    else (
                        max(
                            selection_layer_ready_candidates,
                            key=lambda item: int(
                                candidate_layer_estimates[id(item)].get("layer", 0)
                            ),
                        )
                        if selection_layer_ready_candidates
                        else fallback_candidate
                    )
                )
            )
        )
        selected_dual_target = (
            None
            if selected_candidate is None
            else dual_targets[id(selected_candidate)]
        )
        detected_2d = bool(detected_candidates)
        graspable_2d = bool(layout_graspable_candidates)
        dual_suction_ready_2d = bool(
            selection_depth_ready_candidates
            if fixed_selection_required
            else layout_dual_graspable_candidates
        )
        camera_profile_approved = (
            self.camera.profile().get("profile_approved") is True
        )
        depth_support = fixed_depth_supports.get(id(selected_candidate))
        if depth_support is None:
            depth_support = dual_depth_supports.get(id(selected_candidate))
        if depth_support is None:
            depth_support = evaluate_dual_suction_depth(
                frame.depth_z16,
                frame.depth_scale_m,
                selected_dual_target,
                depth_eval_cfg,
            )
        midpoint_depth_m: float | None = None
        point_camera_m: tuple[float, float, float] | None = None
        point_left_base_m: tuple[float, float, float] | None = None
        coordinate_error: str | None = None
        layer_estimate: dict[str, Any] | None = None
        layer_error: str | None = None

        if (
            selected_dual_target is not None
            and depth_support.get("valid") is True
            and self.camera.intrinsics is not None
            and self.cam_to_left is not None
            and camera_profile_approved
        ):
            try:
                midpoint_depth_m = candidate_midpoint_depths[id(selected_candidate)]
                point_camera = np.asarray(
                    candidate_camera_points[id(selected_candidate)],
                    dtype=np.float64,
                )
                point_left_base = transform_point(
                    point_camera,
                    self.cam_to_left,
                )
                point_camera_m = tuple(float(value) for value in point_camera)
                point_left_base_m = tuple(
                    float(value) for value in point_left_base
                )
            except Exception as exc:
                coordinate_error = f"{type(exc).__name__}: {exc}"

        if layer_cfg.get("enabled") is True and point_camera_m is not None:
            layer_estimate = candidate_layer_estimates.get(id(selected_candidate))
            if layer_estimate is None:
                layer_error = layer_fit_error or "carton layer estimate is unavailable"
            elif layer_estimate.get("valid") is not True:
                layer_error = str(
                    layer_estimate.get("error")
                    or "carton layer estimate is invalid"
                )

        fixed_projection_payload = copy.deepcopy(
            fixed_projection_payloads.get(id(selected_candidate))
        )
        fixed_projection_blockers: list[str] = []
        if fixed_selection_required and selected_candidate is None:
            failure_reasons = list(
                dict.fromkeys(
                    reason
                    for reasons in fixed_projection_failures.values()
                    for reason in reasons
                )
            )
            if layout_depth_ready_candidates and failure_reasons:
                fixed_projection_blockers.append(
                    "no candidate passed the calibrated fixed-suction gate: "
                    + ", ".join(failure_reasons)
                )

        blockers: list[str] = []
        if frame.depth_z16 is None:
            blockers.append(
                "no depth is available from the synchronized frame"
            )
        elif self.camera.intrinsics is None:
            blockers.append(
                "camera intrinsics are unavailable for RGB-D deprojection"
            )
        elif self.cam_to_left is None:
            blockers.append(
                self.cam_to_left_error
                or "camera-to-left-base calibration is unavailable"
            )
        elif not camera_profile_approved:
            blockers.append("the active RGB-D calibration profile is not approved")
        elif selected_dual_target is not None and depth_support.get("valid") is not True:
            blockers.append("dual-suction contact patches have invalid depth")
        elif coordinate_error is not None:
            blockers.append("3-D target conversion failed: " + coordinate_error)
        elif layer_cfg.get("enabled") is True and layer_error is not None:
            blockers.append("carton layer estimation failed: " + layer_error)
        blockers.extend(fixed_projection_blockers)
        if (
            task_id == "task1"
            and task1_stack_report is not None
            and int(task1_stack_report.get("input_candidate_count", 0)) > 0
            and int(task1_stack_report.get("eligible_candidate_count", 0)) == 0
        ):
            blockers.append(
                "Task1 stack prior found no unpicked cell on the active layer"
            )
        if (
            task_id == "task2"
            and task2_station_radius_required
            and depth_ready_candidates
            and not task2_station_matches
        ):
            blockers.append(
                "未在任务二标定工位附近找到可吸取的单层药盒"
            )
        if not candidates:
            if detector_status.get("ok") is not True:
                blockers.insert(
                    0,
                    "detector unavailable: "
                    + str(
                        detector_status.get("last_error")
                        or detector_status.get("backend", {}).get("error")
                        or "unknown error"
                    ),
                )
            else:
                blockers.insert(
                    0,
                    "no medicine-carton candidate passed geometric filters",
                )
        elif not detected_candidates:
            best_candidate = candidates[0]
            blockers.insert(
                0,
                f"detection score {best_candidate.score:.3f} below "
                f"{minimum_score:.3f}",
            )
        elif not physical_instance_candidates:
            reasons = sorted(
                {
                    reason
                    for item in detected_candidates
                    for reason in instance_reports.get(id(item), {}).get(
                        "blockers", []
                    )
                }
            )
            blockers.insert(
                0,
                "no candidate passed the RGB-D single-carton physical-size gate"
                + (": " + ", ".join(reasons) if reasons else ""),
            )
        elif not task_instance_candidates:
            blockers.insert(
                0,
                "no single-carton instance matched the active task region, height and orientation profile",
            )
        elif not graspable_2d:
            candidate_reasons = ", ".join(
                selected_candidate.grasp_blockers
                if selected_candidate is not None
                else ("face_unverified",)
            )
            blockers.insert(
                0,
                "no candidate passed the 2-D grasp policy: " + candidate_reasons,
            )
        elif not dual_suction_ready_2d:
            dual_reasons = (
                selected_dual_target.blockers
                if selected_dual_target is not None
                else ("dual_suction_geometry_unavailable",)
            )
            blockers.insert(
                0,
                "no candidate passed the dual-suction geometry gate: "
                + ", ".join(dual_reasons),
            )

        selected = None
        if selected_candidate is not None:
            selected = LocatedBox(
                candidate=selected_candidate,
                depth=None,
                point_camera_m=point_camera_m,
                point_left_base_m=point_left_base_m,
                physical_size_m=None,
                surface_normal_left_base=None,
                surface_tilt_deg=None,
                plane_residual_mm=None,
                reachable=None,
                blockers=tuple(blockers),
            )
        # The task overlay/API must describe only active source-zone cartons.
        # Falling back to every global candidate when the source is empty would
        # redraw completed cartons inside the shipping box as diagnostic
        # NO-PICK boxes, which is misleading and can leak finished goods into
        # downstream agent observations.
        reported_candidates = list(
            recognized_task_candidates
            if task_id in {"task1", "task2"}
            else layout_candidates
        )
        overlay_dual_targets: dict[int, Any] = dict(dual_targets)
        for item_id, projection_payload in fixed_projection_payloads.items():
            overlay_dual_targets[item_id] = SimpleNamespace(
                midpoint_px=tuple(projection_payload["midpoint_px"]),
                cup_centers_px=tuple(
                    tuple(point)
                    for point in projection_payload["cup_centers_px"]
                ),
                projected_cup_radius_px=float(
                    projection_payload["projected_cup_radius_px"]
                ),
                valid_2d=bool(projection_payload.get("valid_2d")),
            )
        overlay = draw_overlay(
            rgb,
            reported_candidates,
            selected,
            dual_plan_config,
            dual_targets=overlay_dual_targets,
        )
        if task_id == "task1" and task1_stack_report is not None:
            debug_stack_report = copy.deepcopy(task1_stack_report)
            if (
                selected_candidate is not None
                and selected_candidate.grid_shape == (3, 3)
                and debug_stack_report.get("active_layer") is not None
            ):
                debug_stack_report["selected_slot"] = {
                    "layer": int(debug_stack_report["active_layer"]),
                    "row_index": int(selected_candidate.grid_index[0]),
                    "column_index": int(selected_candidate.grid_index[1]),
                }
            overlay = draw_task1_stack_debug_overlay(
                overlay,
                stack_report=debug_stack_report,
                layout_polygon_norm=task_profile.get(
                    "adaptive_slot_polygon_norm",
                    task_profile.get("include_polygon_norm"),
                ),
            )
        overlay_jpeg = _encode_jpeg(overlay)
        detection_id = uuid.uuid4().hex

        def candidate_payload(item: Any) -> dict[str, Any]:
            payload = item.to_dict()
            item_id = id(item)
            target = dual_targets[item_id]
            item_fixed_projection = fixed_projection_payloads.get(item_id)
            target_payload = (
                copy.deepcopy(item_fixed_projection)
                if item_fixed_projection is not None
                else (None if target is None else target.to_dict())
            )
            if (
                target_payload is not None
                and item_fixed_projection is None
                and item_id in dual_depth_supports
            ):
                target_payload["depth_support"] = dual_depth_supports[item_id]
            payload["dual_suction"] = target_payload
            if not fixed_selection_required:
                payload["fixed_suction_eligible"] = None
            elif item_id in fixed_valid_candidate_ids:
                payload["fixed_suction_eligible"] = True
            elif item_id in fixed_projection_failures:
                payload["fixed_suction_eligible"] = False
            else:
                payload["fixed_suction_eligible"] = None
            payload["fixed_suction_blockers"] = list(
                fixed_projection_failures.get(item_id, ())
            )
            payload["physical_instance"] = instance_reports.get(item_id)
            payload["recognized_2d"] = item_id in recognized_task_ids
            payload["pickable_3d"] = bool(
                instance_reports.get(item_id, {}).get("valid") is True
                and item.graspable
            )
            payload["selected_for_pick"] = item is selected_candidate
            payload["layer_estimate"] = candidate_layer_estimates.get(item_id)
            base_point = candidate_base_points.get(item_id)
            payload["point_left_base_m"] = (
                None if base_point is None else list(base_point)
            )
            payload["left_base_planar_distance_m"] = (
                None
                if base_point is None
                else math.hypot(float(base_point[0]), float(base_point[1]))
            )
            if item is selected_candidate:
                payload["midpoint_depth_m"] = midpoint_depth_m
                payload["point_camera_m"] = (
                    None if point_camera_m is None else list(point_camera_m)
                )
                payload["point_left_base_m"] = (
                    None
                    if point_left_base_m is None
                    else list(point_left_base_m)
                )
            return payload

        dual_target_payload: dict[str, Any] | None = None
        if fixed_projection_payload is not None:
            dual_target_payload = dict(fixed_projection_payload)
            dual_target_payload["midpoint_depth_m"] = midpoint_depth_m
            dual_target_payload["point_camera_m"] = (
                None if point_camera_m is None else list(point_camera_m)
            )
            dual_target_payload["point_left_base_m"] = (
                None if point_left_base_m is None else list(point_left_base_m)
            )
        elif selected_dual_target is not None:
            dual_target_payload = selected_dual_target.to_dict()
            dual_target_payload["depth_support"] = depth_support
            dual_target_payload["midpoint_depth_m"] = midpoint_depth_m
            dual_target_payload["point_camera_m"] = (
                None if point_camera_m is None else list(point_camera_m)
            )
            dual_target_payload["point_left_base_m"] = (
                None if point_left_base_m is None else list(point_left_base_m)
            )
        safe_grasp_candidate_count = (
            len(fixed_valid_candidate_ids)
            if fixed_selection_required
            else len(layout_depth_ready_candidates)
        )
        minimum_pickable_instances = max(
            1,
            int(task_profile.get("minimum_pickable_instances", 1)),
        )
        pickable_count_ready = (
            safe_grasp_candidate_count >= minimum_pickable_instances
        )
        target_ready = bool(
            selected_candidate is not None
            and dual_suction_ready_2d
            and pickable_count_ready
            and depth_support.get("valid") is True
            and point_left_base_m is not None
            and not blockers
        )
        task1_stack_payload: dict[str, Any] | None = None
        if task1_stack_report is not None:
            task1_stack_payload = copy.deepcopy(task1_stack_report)
            if (
                selected_candidate is not None
                and selected_candidate.grid_shape == (3, 3)
                and task1_stack_payload.get("active_layer") is not None
            ):
                task1_stack_payload["selected_slot"] = {
                    "layer": int(task1_stack_payload["active_layer"]),
                    "row_index": int(selected_candidate.grid_index[0]),
                    "column_index": int(selected_candidate.grid_index[1]),
                }
        detection = {
            "id": detection_id,
            "task_id": task_id,
            "task1_stack_prior": task1_stack_payload,
            "detected_2d": detected_2d,
            "graspable_2d": graspable_2d,
            "dual_suction_ready_2d": dual_suction_ready_2d,
            "target_ready": target_ready,
            "fixed_suction_selection": {
                "required": fixed_selection_required,
                "valid_candidate_count": len(fixed_valid_candidate_ids),
                "rejected_candidate_count": len(fixed_projection_failures),
            },
            "count": len(reported_candidates),
            "recognized_count": len(reported_candidates),
            "pickable_instance_count": len(layout_candidates),
            "safe_grasp_candidate_count": safe_grasp_candidate_count,
            "minimum_pickable_instances": minimum_pickable_instances,
            "pickable_count_ready": pickable_count_ready,
            "instance_count": len(reported_candidates),
            "global_instance_count": len(physical_instance_candidates),
            "task_instance_count": len(task_instance_candidates),
            "layout_group_count": len(instance_groups),
            "raw_candidate_count": len(raw_candidates),
            "physical_gate_diagnostics": [
                {
                    "candidate": item.to_dict(),
                    "physical_instance": instance_reports.get(id(item)),
                }
                for item in detected_candidates
            ],
            "grid_cell_count": sum(
                1 for item in candidates if item.grid_shape != (1, 1)
            ),
            "selection_policy": (
                "leftmost_image_x_valid_task2_front_carton"
                if task_id == "task2"
                else (
                    "nearest_left_base_flat_carton_panel"
                    if task_id == "task3"
                    else "highest_layer_then_nearest_left_base_xy"
                )
            ),
            "height_policy": (
                "sampled_single_carton_fixed_z"
                if task_id == "task2"
                else (
                    "tabletop_flat_carton"
                    if task_id == "task3"
                    else "vision_layer_to_fixed_z"
                )
            ),
            "layout": {
                "profile": str(task_profile.get("recognition_profile", task_id)),
                "initial_total_count": int(
                    task_profile.get("initial_total_count", 0)
                ),
                "maximum_visible_instances": int(
                    task_profile.get("maximum_visible_instances", 0)
                ),
                "process_target_count": int(
                    task_profile.get("process_target_count", 0)
                ),
                "observed_instance_count": len(reported_candidates),
                "pickable_instance_count": len(layout_candidates),
                "safe_grasp_candidate_count": safe_grasp_candidate_count,
                "minimum_pickable_instances": minimum_pickable_instances,
                "pickable_count_ready": pickable_count_ready,
                "global_observed_instance_count": len(
                    recognized_task_candidates
                    if task_id == "task1"
                    else physical_instance_candidates
                ),
                "decreasing_count_expected": True,
            },
            "candidate": (
                None
                if selected_candidate is None
                else candidate_payload(selected_candidate)
            ),
            "candidates": [
                candidate_payload(candidate)
                for candidate in reported_candidates
            ],
            "dual_suction_target": dual_target_payload,
            "midpoint_depth_m": midpoint_depth_m,
            "point_camera_m": (
                None if point_camera_m is None else list(point_camera_m)
            ),
            "point_left_base_m": (
                None if point_left_base_m is None else list(point_left_base_m)
            ),
            "pick_target_record": (
                None
                if selected_candidate is None
                else {
                    "selection_policy": (
                        "leftmost_image_x_valid_task2_front_carton"
                        if task_id == "task2"
                        else (
                            "nearest_left_base_flat_carton_panel"
                            if task_id == "task3"
                            else "highest_layer_then_nearest_left_base_xy"
                        )
                    ),
                    "candidate_center_px": [
                        float(selected_candidate.center_px[0]),
                        float(selected_candidate.center_px[1]),
                    ],
                    "suction_midpoint_px": (
                        None
                        if dual_target_payload is None
                        else list(dual_target_payload.get("midpoint_px", []))
                    ),
                    "point_camera_m": (
                        None if point_camera_m is None else list(point_camera_m)
                    ),
                    "point_left_base_m": (
                        None
                        if point_left_base_m is None
                        else list(point_left_base_m)
                    ),
                    "captured_at": frame.captured_at,
                    "frame_number": frame.frame_number,
                }
            ),
            "layer_estimate": layer_estimate,
            "detector": detector_status,
            "blockers": blockers,
            "overlay_url": f"/api/camera/frame.jpg?overlay={detection_id}",
            "captured_at": frame.captured_at,
            "frame": {
                "number": frame.frame_number,
                "device_timestamp_ms": frame.device_timestamp_ms,
                "has_depth": frame.depth_z16 is not None,
                "depth_scale_m": frame.depth_scale_m,
            },
        }
        with self._lock:
            self._last_detection = detection
            self._overlay_jpegs[detection_id] = overlay_jpeg
            while len(self._overlay_jpegs) > OVERLAY_CACHE_SIZE:
                self._overlay_jpegs.popitem(last=False)
        return {"ok": True, "detection": detection}

    def close(self) -> None:
        self.base_trajectory.close()
        self.act_rollout.close()
        self.trajectory_replay.close()
        self.cartesian_jog.close()
        # ACT finalization closes three encoders, aligns samples, writes
        # checksums and atomically renames the episode.  Eight seconds was too
        # short under load and could strand an otherwise recoverable
        # ``.inprogress`` directory during a service restart.
        self.trajectory_recorder.close(timeout_s=25.0)
        self.camera.close()


class PackagingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True
    request_queue_size = 32

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        app: PackagingConsoleApp | None,
    ) -> None:
        self.app = app
        super().__init__(server_address, handler_class)


class PackagingRequestHandler(BaseHTTPRequestHandler):
    _gripper_proxy_session_lock = threading.RLock()
    _gripper_proxy_cookie = ""
    _gripper_proxy_csrf = ""
    server_version = "MedicinePackagingConsole/0.1"
    protocol_version = "HTTP/1.1"

    @property
    def app(self) -> PackagingConsoleApp:
        app = self.server.app  # type: ignore[attr-defined]
        if app is None:
            raise RuntimeError("packaging console app is not initialized")
        return app

    def _base_headers(self) -> None:
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; base-uri 'none'; connect-src 'self'; "
            "frame-ancestors 'none'; form-action 'none'; img-src 'self' data:; "
            "object-src 'none'; script-src 'self'; style-src 'self'",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str,
        *,
        head_only: bool = False,
    ) -> None:
        # The browser polls several camera and status resources.  Keeping each
        # HTTP/1.1 socket alive consumes one ThreadingHTTPServer thread while
        # it waits for the next request.  That collided with the gRPC threads
        # created by ACT recording and exhausted the service TasksMax.  One
        # request per connection is deliberate for this loopback-only console.
        self.close_connection = True
        self.send_response(status)
        self._base_headers()
        self.send_header("Connection", "close")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head_only:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # A timed-out image element may close its SSH-forwarded socket
                # while a frame is being encoded.  The request is already
                # complete from the server's perspective.
                return

    def _send_json(
        self,
        status: int,
        payload: dict[str, Any],
        *,
        head_only: bool = False,
    ) -> None:
        body = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self._send_bytes(
            status,
            body,
            "application/json; charset=utf-8",
            head_only=head_only,
        )

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"ok": False, "error": message})

    def _host_is_allowed(self) -> bool:
        raw_host = self.headers.get("Host", "").strip()
        if not raw_host:
            return False
        try:
            hostname = urlsplit(f"//{raw_host}").hostname
        except ValueError:
            return False
        return bool(hostname) and _is_loopback(str(hostname))

    def _origin_is_allowed(self) -> bool:
        raw_origin = self.headers.get("Origin", "").strip()
        if not raw_origin:
            return True
        try:
            parsed = urlsplit(raw_origin)
            hostname = parsed.hostname
            port = parsed.port
        except ValueError:
            return False
        return (
            parsed.scheme == "http"
            and bool(hostname)
            and _is_loopback(str(hostname))
            and port == self.app.port
            and parsed.path in ("", "/")
            and not parsed.query
            and not parsed.fragment
        )

    def _static_path(self, request_path: str) -> Path | None:
        decoded = unquote(request_path)
        if "\x00" in decoded:
            return None
        relative = "index.html" if decoded in ("", "/") else decoded.lstrip("/")
        candidate = (self.app.static_dir / relative).resolve()
        try:
            if os.path.commonpath(
                (str(candidate), str(self.app.static_dir))
            ) != str(self.app.static_dir):
                return None
        except ValueError:
            return None
        return candidate

    def _serve_static(self, request_path: str, *, head_only: bool = False) -> None:
        path = self._static_path(request_path)
        if path is None:
            self._send_error_json(HTTPStatus.FORBIDDEN, "path traversal is forbidden")
            return
        if path.is_dir():
            path = path / "index.html"
        if not path.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, "resource not found")
            return
        try:
            body = path.read_bytes()
        except OSError as exc:
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"cannot read static resource: {exc}",
            )
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in (
            "application/javascript",
            "application/json",
        ):
            content_type += "; charset=utf-8"
        self._send_bytes(
            HTTPStatus.OK,
            body,
            content_type,
            head_only=head_only,
        )

    def _serve_wrist_camera_frame(
        self,
        side: str,
        *,
        head_only: bool = False,
    ) -> None:
        """Relay one fresh wrist frame from the local video service."""

        try:
            upstream_url = self.app.wrist_camera_frame_url(side)
        except ValueError as exc:
            self._send_error_json(HTTPStatus.NOT_FOUND, str(exc))
            return
        except WristCameraUnavailable as exc:
            self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            return
        if head_only:
            self._send_bytes(
                HTTPStatus.OK,
                b"",
                "image/jpeg",
                head_only=True,
            )
            return

        request = urllib.request.Request(
            upstream_url,
            headers={"User-Agent": "medicine-packaging-console/0.6"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.app.wrist_camera_timeout_s,
            ) as upstream:
                content_type = str(
                    upstream.headers.get(
                        "Content-Type",
                        "image/jpeg",
                    )
                )
                if not content_type.lower().startswith("image/jpeg"):
                    raise WristCameraUnavailable(
                        "wrist-camera service returned a non-JPEG response"
                    )
                body = upstream.read(4 * 1024 * 1024 + 1)
                if len(body) > 4 * 1024 * 1024:
                    raise WristCameraUnavailable(
                        "wrist-camera frame exceeds the 4 MiB limit"
                    )
                if not body:
                    raise WristCameraUnavailable(
                        "wrist-camera service returned an empty frame"
                    )
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            WristCameraUnavailable,
        ) as exc:
            self._send_error_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                f"{side} wrist camera unavailable: {exc}",
            )
            return
        self._send_bytes(HTTPStatus.OK, body, "image/jpeg")

    def _proxy_gripper_signal_lock(
        self,
        *,
        method: str,
        body: bytes | None = None,
        head_only: bool = False,
    ) -> None:
        """Use the single 9999 lock implementation and state as source of truth."""
        try:
            for attempt in range(2):
                with self._gripper_proxy_session_lock:
                    if not self._gripper_proxy_cookie or not self._gripper_proxy_csrf:
                        session_request = urllib.request.Request(
                            "http://127.0.0.1:9999/api/auth/session",
                            headers={"User-Agent": "medicine-packaging-console/0.6"},
                            method="GET",
                        )
                        with urllib.request.urlopen(session_request, timeout=4.0) as upstream:
                            session_payload = json.loads(upstream.read(64 * 1024).decode("utf-8"))
                            set_cookie = str(upstream.headers.get("Set-Cookie", ""))
                        cookie = set_cookie.split(";", 1)[0].strip()
                        csrf = str(session_payload.get("csrf_token", ""))
                        if not cookie or not csrf:
                            raise ValueError("9999 did not issue an anonymous control session")
                        type(self)._gripper_proxy_cookie = cookie
                        type(self)._gripper_proxy_csrf = csrf
                    cookie = self._gripper_proxy_cookie
                    csrf = self._gripper_proxy_csrf
                request_headers = {
                    "User-Agent": "medicine-packaging-console/0.6",
                    "Content-Type": "application/json",
                    "Cookie": cookie,
                }
                if method == "POST":
                    request_headers["X-CSRF-Token"] = csrf
                    request_headers["Origin"] = "http://127.0.0.1:9999"
                request = urllib.request.Request(
                    "http://127.0.0.1:9999/api/gripper-signal-lock",
                    data=body if method == "POST" else None,
                    headers=request_headers,
                    method=method,
                )
                try:
                    with urllib.request.urlopen(request, timeout=4.0) as upstream:
                        response_body = upstream.read(64 * 1024 + 1)
                        if len(response_body) > 64 * 1024:
                            raise ValueError("9999 gripper-lock response is too large")
                        payload = json.loads(response_body.decode("utf-8"))
                        if not isinstance(payload, dict):
                            raise ValueError("9999 gripper-lock response is not an object")
                        self._send_json(HTTPStatus.OK, payload, head_only=head_only)
                        return
                except urllib.error.HTTPError as exc:
                    if exc.code == HTTPStatus.UNAUTHORIZED and attempt == 0:
                        exc.read()
                        with self._gripper_proxy_session_lock:
                            type(self)._gripper_proxy_cookie = ""
                            type(self)._gripper_proxy_csrf = ""
                        continue
                    raise
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read(64 * 1024).decode("utf-8"))
                message = payload.get("error") or payload.get("message")
            except Exception:
                message = None
            self._send_error_json(exc.code, message or f"9999 gripper-lock rejected request: {exc.code}")
        except (OSError, TimeoutError, urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
            self._send_error_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                f"9999 gripper-lock unavailable: {exc}",
            )

    def _route_get(self, *, head_only: bool = False) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/api/health":
            self._send_json(HTTPStatus.OK, self.app.health(), head_only=head_only)
            return
        if parsed.path == "/api/status":
            self._send_json(HTTPStatus.OK, self.app.status(), head_only=head_only)
            return
        if parsed.path == "/api/runtime-parameters":
            self._send_json(
                HTTPStatus.OK,
                self.app.runtime_parameters_status(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/current-pose":
            arm_values = parse_qs(parsed.query).get("arm", [])
            arm = arm_values[0] if arm_values else "left"
            try:
                payload = self.app.read_current_pose(arm)
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except CartesianJogConflict as exc:
                self._send_error_json(HTTPStatus.CONFLICT, str(exc))
                return
            except CartesianJogSafetyViolation as exc:
                self._send_error_json(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
                return
            except CartesianJogUnavailable as exc:
                self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            self._send_json(HTTPStatus.OK, payload, head_only=head_only)
            return
        if parsed.path == "/api/camera/profile":
            self._send_json(
                HTTPStatus.OK,
                self.app.camera_profile(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/calibrations/fixed-suction-axis/status":
            self._send_json(
                HTTPStatus.OK,
                self.app.fixed_suction_axis_status(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/tasks/profiles":
            self._send_json(
                HTTPStatus.OK,
                self.app.task_profiles_status(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/recordings/status":
            self._send_json(
                HTTPStatus.OK,
                self.app.recording_status(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/act/status":
            self._send_json(
                HTTPStatus.OK,
                self.app.act_inference_status(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/act/rollout/status":
            self._send_json(
                HTTPStatus.OK,
                self.app.act_rollout_status(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/recordings":
            self._send_json(
                HTTPStatus.OK,
                self.app.recordings(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/base-trajectory/status":
            self._send_json(
                HTTPStatus.OK,
                self.app.base_trajectory_status(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/base-trajectories":
            self._send_json(
                HTTPStatus.OK,
                self.app.base_trajectories(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/replay/status":
            self._send_json(
                HTTPStatus.OK,
                self.app.replay_status(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/teleop/status":
            self._send_json(
                HTTPStatus.OK,
                self.app.teleop_status(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/cartesian-jog/status":
            self._send_json(
                HTTPStatus.OK,
                self.app.cartesian_jog_status(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/suction/status":
            self._send_json(
                HTTPStatus.OK,
                self.app.suction_status(),
                head_only=head_only,
            )
            return
        if parsed.path in {
            "/api/skills/task2/place-shipping-box/preflight",
            "/api/skills/task3/place-shipping-box/preflight",
        }:
            task_id = parsed.path.split("/")[3]
            self._send_json(
                HTTPStatus.OK,
                self.app.shipping_box_placement_preflight(
                    {}, task_id=task_id
                ),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/gripper-signal-lock":
            self._proxy_gripper_signal_lock(method="GET", head_only=head_only)
            return
        if parsed.path == "/api/task1/pick/status":
            self._send_json(
                HTTPStatus.OK,
                {"ok": True, "task1_pick": self.app.task1_pick_status()},
                head_only=head_only,
            )
            return
        if parsed.path == "/api/task2/pick/status":
            self._send_json(
                HTTPStatus.OK,
                {"ok": True, "task2_pick": self.app.task2_pick_status()},
                head_only=head_only,
            )
            return
        if parsed.path == "/api/task3/pick/status":
            self._send_json(
                HTTPStatus.OK,
                {"ok": True, "task3_pick": self.app.task3_pick_status()},
                head_only=head_only,
            )
            return
        if parsed.path == "/api/skills/task1/pick-carton":
            self._send_json(
                HTTPStatus.OK,
                self.app.task1_pick_skill_descriptor(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/skills/task2/pick-carton":
            self._send_json(
                HTTPStatus.OK,
                self.app.task2_pick_skill_descriptor(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/skills/task2/reset-both-home":
            self._send_json(
                HTTPStatus.OK,
                self.app.task2_reset_both_skill_descriptor(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/skills/task2/move-ready-poses":
            self._send_json(
                HTTPStatus.OK,
                self.app.task2_ready_poses_skill_descriptor(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/skills/task2/move-subtask2-init-poses":
            self._send_json(
                HTTPStatus.OK,
                self.app.task2_subtask_init_poses_skill_descriptor(2),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/skills/task2/move-subtask3-init-poses":
            self._send_json(
                HTTPStatus.OK,
                self.app.task2_subtask_init_poses_skill_descriptor(3),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/skills/task1/move-watcher-pose":
            self._send_json(
                HTTPStatus.OK,
                self.app.task1_watcher_pose_skill_descriptor(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/skills/task1/place-carton-fixed-trajectory":
            self._send_json(
                HTTPStatus.OK,
                self.app.task1_fixed_trajectory_place_skill_descriptor(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/skills/task2/move-watcher-pose":
            self._send_json(
                HTTPStatus.OK,
                self.app.task2_watcher_pose_skill_descriptor(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/skills/task3/move-watcher-pose":
            self._send_json(
                HTTPStatus.OK,
                self.app.task3_watcher_pose_skill_descriptor(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/skills/task1/detect-carton":
            self._send_json(
                HTTPStatus.OK,
                self.app.task1_detect_carton_skill_descriptor(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/skills/task1/detect-box-slots":
            self._send_json(
                HTTPStatus.OK,
                self.app.task1_detect_box_slots_skill_descriptor(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/skills/task1/confirm-box-slots":
            self._send_json(
                HTTPStatus.OK,
                self.app.task1_confirm_box_slots_skill_descriptor(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/skills/task1/pick-staged-carton-top":
            self._send_json(
                HTTPStatus.OK,
                self.app.task1_pick_staged_top_skill_descriptor(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/skills/task1/place-in-box":
            self._send_json(
                HTTPStatus.OK,
                self.app.task1_place_in_box_skill_descriptor(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/skills/task2/detect-carton":
            self._send_json(
                HTTPStatus.OK,
                self.app.task2_detect_carton_skill_descriptor(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/skills/task3/detect-carton":
            self._send_json(
                HTTPStatus.OK,
                self.app.task3_detect_carton_skill_descriptor(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/skills/task3/pick-flat-carton":
            self._send_json(
                HTTPStatus.OK,
                self.app.task3_pick_skill_descriptor(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/skills/task3/expand-carton":
            self._send_json(
                HTTPStatus.OK,
                self.app.task3_expand_skill_descriptor(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/skills/left-arm/reset-home":
            self._send_json(
                HTTPStatus.OK,
                self.app.left_arm_reset_home_skill_descriptor(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/skills/right-arm/reset-home":
            self._send_json(
                HTTPStatus.OK,
                self.app.right_arm_reset_home_skill_descriptor(),
                head_only=head_only,
            )
            return
        if parsed.path == "/api/camera/frame.jpg":
            try:
                overlay_values = parse_qs(parsed.query).get("overlay", [])
                overlay_id = overlay_values[0] if overlay_values else None
                body = self.app.frame_jpeg(overlay_id=overlay_id)
            except OverlayUnavailable as exc:
                self._send_error_json(HTTPStatus.NOT_FOUND, str(exc))
                return
            except CameraUnavailable as exc:
                self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            except Exception as exc:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"frame capture failed: {exc}",
                )
                return
            self._send_bytes(
                HTTPStatus.OK,
                body,
                "image/jpeg",
                head_only=head_only,
            )
            return
        wrist_prefix = "/api/wrist-cameras/"
        wrist_suffix = "/frame.jpg"
        if (
            parsed.path.startswith(wrist_prefix)
            and parsed.path.endswith(wrist_suffix)
        ):
            side = parsed.path[
                len(wrist_prefix) : -len(wrist_suffix)
            ]
            self._serve_wrist_camera_frame(side, head_only=head_only)
            return
        if parsed.path.startswith("/api/"):
            self._send_error_json(HTTPStatus.NOT_FOUND, "API endpoint not found")
            return
        self._serve_static(parsed.path, head_only=head_only)

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_is_allowed():
            self._send_error_json(HTTPStatus.MISDIRECTED_REQUEST, "Host is not allowed")
            return
        self._route_get()

    def do_HEAD(self) -> None:  # noqa: N802
        if not self._host_is_allowed():
            self._send_json(
                HTTPStatus.MISDIRECTED_REQUEST,
                {"ok": False, "error": "Host is not allowed"},
                head_only=True,
            )
            return
        self._route_get(head_only=True)

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_is_allowed():
            self.close_connection = True
            self._send_error_json(HTTPStatus.MISDIRECTED_REQUEST, "Host is not allowed")
            return
        parsed = urlsplit(self.path)
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.close_connection = True
            self._send_error_json(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
            return
        if content_length < 0 or content_length > 4096:
            self.close_connection = True
            self._send_error_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "body too large")
            return
        raw = self.rfile.read(content_length) if content_length else b""
        if parsed.path not in (
            "/api/detect",
            "/api/calibrations/fixed-suction-axis/lock-marker",
            "/api/calibrations/fixed-suction-axis/sample-cup",
            "/api/calibrations/fixed-suction-axis/commit",
            "/api/recordings/start",
            "/api/recordings/stop",
            "/api/recordings/delete",
            "/api/base-trajectory/record/start",
            "/api/base-trajectory/record/stop",
            "/api/base-trajectory/replay/preflight",
            "/api/base-trajectory/replay/start",
            "/api/base-trajectory/replay/stop",
            "/api/replay/preflight",
            "/api/replay/start",
            "/api/replay/stop",
            "/api/teleop/start",
            "/api/teleop/stop",
            "/api/teleop/hard-restart",
            "/api/cartesian-jog/capture-orientation",
            "/api/cartesian-jog/enable",
            "/api/cartesian-jog/quick-enable",
            "/api/cartesian-jog/disable",
            "/api/cartesian-jog/move",
            "/api/cartesian-jog/restore-safe-vertical",
            "/api/suction",
            "/api/suction/sync",
            "/api/gripper-signal-lock",
            "/api/task1/pick",
            "/api/skills/task1/pick-carton",
            "/api/skills/task1/move-watcher-pose",
            "/api/skills/task1/place-carton-fixed-trajectory",
            "/api/skills/task1/detect-carton",
            "/api/skills/task1/detect-box-slots",
            "/api/skills/task1/confirm-box-slots",
            "/api/skills/task1/pick-staged-carton-top",
            "/api/skills/task1/place-in-box",
            "/api/skills/task2/pick-carton",
            "/api/skills/task2/reset-both-home",
            "/api/skills/task2/move-watcher-pose",
            "/api/skills/task2/detect-carton",
            "/api/skills/task2/move-ready-poses",
            "/api/skills/task2/move-subtask2-init-poses",
            "/api/skills/task2/move-subtask3-init-poses",
            "/api/skills/task2/detect-shipping-box",
            "/api/skills/task2/place-shipping-box",
            "/api/skills/task3/move-watcher-pose",
            "/api/skills/task3/detect-carton",
            "/api/skills/task3/detect-shipping-box",
            "/api/skills/task3/place-shipping-box",
            "/api/skills/task3/pick-flat-carton",
            "/api/skills/task3/expand-carton",
            "/api/skills/task1/watch-detect-pick",
            "/api/skills/task2/watch-detect-pick",
            "/api/skills/task3/watch-detect-pick",
            "/api/skills/task1/observe-carton",
            "/api/skills/task2/observe-carton",
            "/api/skills/task3/observe-carton",
            "/api/skills/task1/pick-cached-carton",
            "/api/skills/task2/pick-cached-carton",
            "/api/skills/task3/pick-cached-carton",
            "/api/act/rollout/start",
            "/api/act/rollout/stop",
            "/api/skills/left-arm/reset-home",
            "/api/skills/right-arm/reset-home",
            "/api/runtime-parameters",
            "/api/runtime-parameters/capture-contact-z",
            "/api/runtime-poses/capture",
            "/api/runtime-poses/move",
            "/api/runtime-poses/delete",
            "/api/act/model",
            "/api/act/predict-preview",
        ):
            self._send_error_json(HTTPStatus.NOT_FOUND, "API endpoint not found")
            return
        peer_sync_request = (
            parsed.path == "/api/suction/sync"
            and _is_loopback(self.client_address[0])
        )
        if not peer_sync_request and (parsed.path.startswith((
            "/api/teleop/",
            "/api/cartesian-jog/",
            "/api/replay/",
            "/api/act/rollout/",
            "/api/calibrations/fixed-suction-axis/",
            "/api/base-trajectory/",
        )) or parsed.path in {
            "/api/recordings/delete",
            "/api/base-trajectory/record/start",
            "/api/base-trajectory/record/stop",
            "/api/base-trajectory/replay/preflight",
            "/api/base-trajectory/replay/start",
            "/api/base-trajectory/replay/stop",
            "/api/suction",
            "/api/suction/sync",
            "/api/gripper-signal-lock",
            "/api/task1/pick",
            "/api/skills/task1/pick-carton",
            "/api/skills/task1/move-watcher-pose",
            "/api/skills/task1/place-carton-fixed-trajectory",
            "/api/skills/task1/detect-carton",
            "/api/skills/task1/detect-box-slots",
            "/api/skills/task1/confirm-box-slots",
            "/api/skills/task1/pick-staged-carton-top",
            "/api/skills/task1/place-in-box",
            "/api/skills/task2/pick-carton",
            "/api/skills/task2/reset-both-home",
            "/api/skills/task2/move-watcher-pose",
            "/api/skills/task2/detect-carton",
            "/api/skills/task2/move-ready-poses",
            "/api/skills/task2/move-subtask2-init-poses",
            "/api/skills/task2/move-subtask3-init-poses",
            "/api/skills/task2/detect-shipping-box",
            "/api/skills/task2/place-shipping-box",
            "/api/skills/task3/move-watcher-pose",
            "/api/skills/task3/detect-carton",
            "/api/skills/task3/detect-shipping-box",
            "/api/skills/task3/place-shipping-box",
            "/api/skills/task3/pick-flat-carton",
            "/api/skills/task3/expand-carton",
            "/api/skills/task1/watch-detect-pick",
            "/api/skills/task2/watch-detect-pick",
            "/api/skills/task3/watch-detect-pick",
            "/api/skills/task1/observe-carton",
            "/api/skills/task2/observe-carton",
            "/api/skills/task3/observe-carton",
            "/api/skills/task1/pick-cached-carton",
            "/api/skills/task2/pick-cached-carton",
            "/api/skills/task3/pick-cached-carton",
            "/api/skills/left-arm/reset-home",
            "/api/skills/right-arm/reset-home",
            "/api/runtime-parameters",
            "/api/runtime-parameters/capture-contact-z",
            "/api/runtime-poses/capture",
            "/api/runtime-poses/move",
            "/api/runtime-poses/delete",
            "/api/act/model",
            "/api/act/predict-preview",
        }):
            content_type = self.headers.get("Content-Type", "")
            if content_type.split(";", 1)[0].strip().lower() != "application/json":
                self._send_error_json(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    "motion-control requests require application/json",
                )
                return
            if not self._origin_is_allowed():
                self._send_error_json(
                    HTTPStatus.FORBIDDEN,
                    "Origin is not allowed",
                )
                return
        payload: dict[str, Any] = {}
        if content_length:
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_error_json(HTTPStatus.BAD_REQUEST, "body must be valid JSON")
                return
            if not isinstance(payload, dict):
                self._send_error_json(HTTPStatus.BAD_REQUEST, "JSON body must be an object")
                return
        if parsed.path == "/api/gripper-signal-lock":
            self._proxy_gripper_signal_lock(
                method="POST",
                body=json.dumps(payload).encode("utf-8"),
            )
            return
        if parsed.path.startswith("/api/calibrations/fixed-suction-axis/"):
            actions = {
                "/api/calibrations/fixed-suction-axis/lock-marker":
                    self.app.lock_fixed_axis_marker,
                "/api/calibrations/fixed-suction-axis/sample-cup":
                    self.app.sample_fixed_axis_cup,
                "/api/calibrations/fixed-suction-axis/commit":
                    self.app.commit_fixed_axis_calibration,
            }
            try:
                result = actions[parsed.path](payload)
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except CameraUnavailable as exc:
                self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            except CartesianJogSafetyViolation as exc:
                self._send_error_json(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
                return
            except Exception as exc:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "fixed suction calibration failed: "
                    f"{type(exc).__name__}: {exc}",
                )
                return
            self._send_json(HTTPStatus.OK, result)
            return
        if parsed.path == "/api/act/model":
            try:
                result = self.app.select_act_inference_profile(payload)
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except RecordingConflict as exc:
                self._send_error_json(HTTPStatus.CONFLICT, str(exc))
                return
            except ActInferenceUnavailable as exc:
                self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            except Exception as exc:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"ACT model switch failed: {type(exc).__name__}: {exc}",
                )
                return
            self._send_json(HTTPStatus.OK, result)
            return

        if parsed.path in {
            "/api/act/rollout/start",
            "/api/act/rollout/stop",
        }:
            try:
                action = (
                    self.app.start_act_rollout
                    if parsed.path.endswith("/start")
                    else self.app.stop_act_rollout
                )
                result = action(payload)
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except ActRolloutConflict as exc:
                self._send_error_json(HTTPStatus.CONFLICT, str(exc))
                return
            except ActRolloutSafetyViolation as exc:
                self._send_error_json(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
                return
            except ActRolloutUnavailable as exc:
                self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            except Exception as exc:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"ACT rollout failed: {type(exc).__name__}: {exc}",
                )
                return
            self._send_json(HTTPStatus.OK, result)
            return

        if parsed.path == "/api/act/predict-preview":
            try:
                result = self.app.predict_act_preview(payload)
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except RecordingConflict as exc:
                self._send_error_json(HTTPStatus.CONFLICT, str(exc))
                return
            except (RecordingUnavailable, ActInferenceUnavailable) as exc:
                self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            except ActInferenceProtocolError as exc:
                self._send_error_json(HTTPStatus.BAD_GATEWAY, str(exc))
                return
            except Exception as exc:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"ACT preview failed: {type(exc).__name__}: {exc}",
                )
                return
            self._send_json(HTTPStatus.OK, result)
            return
        if parsed.path in {
            "/api/runtime-parameters",
            "/api/runtime-parameters/capture-contact-z",
            "/api/runtime-poses/capture",
            "/api/runtime-poses/move",
            "/api/runtime-poses/delete",
        }:
            try:
                if parsed.path == "/api/runtime-parameters":
                    result = self.app.update_runtime_parameters(payload)
                elif parsed.path == "/api/runtime-poses/capture":
                    result = self.app.capture_runtime_pose(payload)
                elif parsed.path == "/api/runtime-poses/move":
                    result = self.app.move_to_runtime_pose(payload)
                elif parsed.path == "/api/runtime-poses/delete":
                    result = self.app.delete_runtime_pose(payload)
                else:
                    result = self.app.capture_runtime_contact_z(payload)
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except CartesianJogUnavailable as exc:
                self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            except CartesianJogConflict as exc:
                self._send_error_json(HTTPStatus.CONFLICT, str(exc))
                return
            except CartesianJogSafetyViolation as exc:
                self._send_error_json(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
                return
            except CartesianJogTimeout as exc:
                self._send_error_json(HTTPStatus.GATEWAY_TIMEOUT, str(exc))
                return
            except OSError as exc:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"cannot persist runtime parameters: {exc}",
                )
                return
            self._send_json(HTTPStatus.OK, result)
            return
        if parsed.path.startswith("/api/cartesian-jog/"):
            actions = {
                "/api/cartesian-jog/capture-orientation":
                    self.app.capture_cartesian_jog_orientation,
                "/api/cartesian-jog/enable":
                    self.app.enable_cartesian_jog,
                "/api/cartesian-jog/quick-enable":
                    self.app.quick_enable_cartesian_jog,
                "/api/cartesian-jog/disable":
                    self.app.disable_cartesian_jog,
                "/api/cartesian-jog/move":
                    self.app.move_cartesian_jog,
                "/api/cartesian-jog/restore-safe-vertical":
                    self.app.restore_cartesian_jog_safe_pose,
            }
            try:
                result = actions[parsed.path](payload)
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except CartesianJogConflict as exc:
                self._send_error_json(HTTPStatus.CONFLICT, str(exc))
                return
            except CartesianJogSafetyViolation as exc:
                self._send_error_json(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    str(exc),
                )
                return
            except CartesianJogUnavailable as exc:
                self._send_error_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    str(exc),
                )
                return
            except CartesianJogTimeout as exc:
                self._send_error_json(HTTPStatus.GATEWAY_TIMEOUT, str(exc))
                return
            except Exception as exc:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "left-arm Cartesian jog failed: "
                    f"{type(exc).__name__}: {exc}",
                )
                return
            self._send_json(HTTPStatus.OK, result)
            return
        if parsed.path == "/api/suction":
            try:
                result = self.app.set_suction(payload)
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except SuctionUnavailable as exc:
                self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            except Exception as exc:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"suction control failed: {type(exc).__name__}: {exc}",
                )
                return
            self._send_json(HTTPStatus.OK, result)
            return
        if parsed.path == "/api/suction/sync":
            try:
                result = self.app.sync_suction_state(payload)
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send_json(HTTPStatus.OK, result)
            return
        if parsed.path in {
            "/api/task1/pick",
            "/api/skills/task1/pick-carton",
            "/api/skills/task1/move-watcher-pose",
            "/api/skills/task1/place-carton-fixed-trajectory",
            "/api/skills/task1/detect-carton",
            "/api/skills/task1/detect-box-slots",
            "/api/skills/task1/confirm-box-slots",
            "/api/skills/task1/pick-staged-carton-top",
            "/api/skills/task1/place-in-box",
            "/api/skills/task2/pick-carton",
            "/api/skills/task2/reset-both-home",
            "/api/skills/task2/move-watcher-pose",
            "/api/skills/task2/detect-carton",
            "/api/skills/task2/move-ready-poses",
            "/api/skills/task2/move-subtask2-init-poses",
            "/api/skills/task2/move-subtask3-init-poses",
            "/api/skills/task2/detect-shipping-box",
            "/api/skills/task2/place-shipping-box",
            "/api/skills/task3/move-watcher-pose",
            "/api/skills/task3/detect-carton",
            "/api/skills/task3/detect-shipping-box",
            "/api/skills/task3/place-shipping-box",
            "/api/skills/task3/pick-flat-carton",
            "/api/skills/task3/expand-carton",
            "/api/skills/task1/watch-detect-pick",
            "/api/skills/task2/watch-detect-pick",
            "/api/skills/task3/watch-detect-pick",
            "/api/skills/task1/observe-carton",
            "/api/skills/task2/observe-carton",
            "/api/skills/task3/observe-carton",
            "/api/skills/task1/pick-cached-carton",
            "/api/skills/task2/pick-cached-carton",
            "/api/skills/task3/pick-cached-carton",
            "/api/skills/left-arm/reset-home",
            "/api/skills/right-arm/reset-home",
        }:
            try:
                if parsed.path == "/api/skills/left-arm/reset-home":
                    result = self.app.run_left_arm_reset_home_skill(payload)
                elif parsed.path == "/api/skills/right-arm/reset-home":
                    result = self.app.run_right_arm_reset_home_skill(payload)
                elif parsed.path == "/api/skills/task1/pick-carton":
                    result = self.app.run_task1_pick_skill(payload)
                elif parsed.path == "/api/skills/task1/move-watcher-pose":
                    result = self.app.run_task1_watcher_pose_skill(payload)
                elif parsed.path == "/api/skills/task1/place-carton-fixed-trajectory":
                    result = self.app.run_task1_fixed_trajectory_place_skill(payload)
                elif parsed.path == "/api/skills/task1/detect-carton":
                    result = self.app.run_task1_detect_carton_step(payload)
                elif parsed.path == "/api/skills/task1/detect-box-slots":
                    result = self.app.run_task1_detect_box_slots_step(payload)
                elif parsed.path == "/api/skills/task1/confirm-box-slots":
                    result = self.app.run_task1_confirm_box_slots_step(payload)
                elif parsed.path == "/api/skills/task1/pick-staged-carton-top":
                    result = self.app.run_task1_pick_staged_top_step(payload)
                elif parsed.path == "/api/skills/task1/place-in-box":
                    result = self.app.run_task1_place_in_box_step(payload)
                elif parsed.path == "/api/skills/task2/pick-carton":
                    result = self.app.run_task2_pick_skill(payload)
                elif parsed.path == "/api/skills/task2/reset-both-home":
                    result = self.app.run_task2_reset_both_skill(payload)
                elif parsed.path == "/api/skills/task2/move-watcher-pose":
                    result = self.app.run_task2_watcher_pose_skill(payload)
                elif parsed.path == "/api/skills/task2/detect-carton":
                    result = self.app.run_task2_detect_carton_step(payload)
                elif parsed.path == "/api/skills/task2/move-ready-poses":
                    result = self.app.run_task2_ready_poses_skill(payload)
                elif parsed.path == "/api/skills/task2/move-subtask2-init-poses":
                    result = self.app.run_task2_subtask_init_poses_skill(
                        payload,
                        subtask_id=2,
                    )
                elif parsed.path == "/api/skills/task2/move-subtask3-init-poses":
                    result = self.app.run_task2_subtask_init_poses_skill(
                        payload,
                        subtask_id=3,
                    )
                elif parsed.path == "/api/skills/task2/detect-shipping-box":
                    result = self.app.run_task2_detect_shipping_box_step(payload)
                elif parsed.path == "/api/skills/task2/place-shipping-box":
                    result = self.app.run_task2_place_shipping_box_step(payload)
                elif parsed.path == "/api/skills/task3/move-watcher-pose":
                    result = self.app.run_task3_watcher_pose_skill(payload)
                elif parsed.path == "/api/skills/task3/detect-carton":
                    result = self.app.run_task3_detect_carton_step(payload)
                elif parsed.path == "/api/skills/task3/detect-shipping-box":
                    result = self.app.run_task3_detect_shipping_box_step(payload)
                elif parsed.path == "/api/skills/task3/place-shipping-box":
                    result = self.app.run_task3_place_shipping_box_step(payload)
                elif parsed.path == "/api/skills/task3/pick-flat-carton":
                    result = self.app.run_task3_pick_skill(payload)
                elif parsed.path == "/api/skills/task3/expand-carton":
                    result = self.app.run_task3_expand_skill(payload)
                elif parsed.path.endswith("/watch-detect-pick"):
                    task_id = parsed.path.split("/")[3]
                    result = self.app.run_watch_detect_pick_skill(
                        payload,
                        task_id=task_id,
                    )
                elif parsed.path.endswith("/observe-carton"):
                    task_id = parsed.path.split("/")[3]
                    result = self.app.run_observe_carton_step(
                        payload,
                        task_id=task_id,
                    )
                elif parsed.path.endswith("/pick-cached-carton"):
                    task_id = parsed.path.split("/")[3]
                    result = self.app.run_pick_cached_carton_step(
                        payload,
                        task_id=task_id,
                    )
                else:
                    result = self.app.pick_detected_carton(payload)
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except CartesianJogConflict as exc:
                self._send_error_json(HTTPStatus.CONFLICT, str(exc))
                return
            except CartesianJogSafetyViolation as exc:
                self._send_error_json(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    str(exc),
                )
                return
            except (CartesianJogUnavailable, SuctionUnavailable) as exc:
                self._send_error_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    str(exc),
                )
                return
            except CartesianJogTimeout as exc:
                self._send_error_json(HTTPStatus.GATEWAY_TIMEOUT, str(exc))
                return
            except ReplayConflict as exc:
                self._send_error_json(HTTPStatus.CONFLICT, str(exc))
                return
            except ReplayUnavailable as exc:
                self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            except Exception as exc:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "bounded arm skill failed: "
                    f"{type(exc).__name__}: {exc}",
                )
                return
            self._send_json(HTTPStatus.OK, result)
            return
        if parsed.path == "/api/recordings/start":
            try:
                result = self.app.start_recording(payload)
            except RecordingConflict as exc:
                self._send_error_json(HTTPStatus.CONFLICT, str(exc))
                return
            except RecordingUnavailable as exc:
                self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except Exception as exc:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"recording start failed: {exc}",
                )
                return
            self._send_json(HTTPStatus.ACCEPTED, result)
            return
        if parsed.path == "/api/recordings/stop":
            try:
                result = self.app.stop_recording()
            except RecordingConflict as exc:
                self._send_error_json(HTTPStatus.CONFLICT, str(exc))
                return
            except Exception as exc:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"recording stop failed: {exc}",
                )
                return
            self._send_json(HTTPStatus.ACCEPTED, result)
            return
        if parsed.path == "/api/recordings/delete":
            try:
                result = self.app.delete_recording(payload)
            except RecordingConflict as exc:
                self._send_error_json(HTTPStatus.CONFLICT, str(exc))
                return
            except FileNotFoundError as exc:
                self._send_error_json(HTTPStatus.NOT_FOUND, str(exc))
                return
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except Exception as exc:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"recording delete failed: {exc}",
                )
                return
            self._send_json(HTTPStatus.OK, result)
            return
        if parsed.path in {
            "/api/base-trajectory/record/start",
            "/api/base-trajectory/record/stop",
            "/api/base-trajectory/replay/preflight",
            "/api/base-trajectory/replay/start",
            "/api/base-trajectory/replay/stop",
        }:
            try:
                if parsed.path == "/api/base-trajectory/record/start":
                    result = self.app.start_base_trajectory_recording(payload)
                    status = HTTPStatus.ACCEPTED
                elif parsed.path == "/api/base-trajectory/record/stop":
                    result = self.app.stop_base_trajectory_recording()
                    status = HTTPStatus.ACCEPTED
                elif parsed.path == "/api/base-trajectory/replay/preflight":
                    result = self.app.base_trajectory_replay_preflight(payload)
                    status = HTTPStatus.OK
                elif parsed.path == "/api/base-trajectory/replay/start":
                    result = self.app.start_base_trajectory_replay(payload)
                    status = HTTPStatus.ACCEPTED
                else:
                    result = self.app.stop_base_trajectory_replay()
                    status = HTTPStatus.ACCEPTED
            except BaseTrajectoryConflict as exc:
                self._send_error_json(HTTPStatus.CONFLICT, str(exc))
                return
            except BaseTrajectorySafetyViolation as exc:
                self._send_error_json(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
                return
            except BaseTrajectoryUnavailable as exc:
                self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except Exception as exc:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"base trajectory operation failed: {type(exc).__name__}: {exc}",
                )
                return
            self._send_json(status, result)
            return
        if parsed.path in {
            "/api/replay/preflight",
            "/api/replay/start",
            "/api/replay/stop",
        }:
            try:
                if parsed.path == "/api/replay/preflight":
                    result = self.app.replay_preflight(payload)
                    status = HTTPStatus.OK
                elif parsed.path == "/api/replay/start":
                    result = self.app.start_replay(payload)
                    status = HTTPStatus.ACCEPTED
                else:
                    result = self.app.stop_replay()
                    status = HTTPStatus.ACCEPTED
            except ReplayConflict as exc:
                self._send_error_json(HTTPStatus.CONFLICT, str(exc))
                return
            except ReplayUnavailable as exc:
                self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except Exception as exc:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"trajectory replay failed: {type(exc).__name__}: {exc}",
                )
                return
            self._send_json(status, result)
            return
        if parsed.path == "/api/teleop/start":
            try:
                result = self.app.start_teleop(payload)
            except TeleopLaunchConflict as exc:
                self._send_error_json(HTTPStatus.CONFLICT, str(exc))
                return
            except TeleopLaunchUnavailable as exc:
                self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except Exception as exc:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"teleop start failed: {exc}",
                )
                return
            self._send_json(HTTPStatus.ACCEPTED, result)
            return
        if parsed.path == "/api/teleop/stop":
            try:
                result = self.app.stop_teleop(payload)
            except TeleopLaunchConflict as exc:
                self._send_error_json(HTTPStatus.CONFLICT, str(exc))
                return
            except TeleopLaunchUnavailable as exc:
                self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except Exception as exc:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"teleop stop failed: {exc}",
                )
                return
            self._send_json(HTTPStatus.ACCEPTED, result)
            return
        if parsed.path == "/api/teleop/hard-restart":
            try:
                result = self.app.hard_restart_teleop(payload)
            except TeleopLaunchConflict as exc:
                self._send_error_json(HTTPStatus.CONFLICT, str(exc))
                return
            except TeleopLaunchUnavailable as exc:
                self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except Exception as exc:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"teleop hard restart failed: {exc}",
                )
                return
            self._send_json(HTTPStatus.ACCEPTED, result)
            return
        try:
            if frozenset(payload) not in {frozenset(), frozenset({"task_id"})}:
                raise ValueError("request fields may contain only: task_id")
            task_id = payload.get("task_id", "task1")
            if not isinstance(task_id, str):
                raise ValueError("task_id must be a string")
            result = self.app.detect(task_id=task_id)
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except CameraUnavailable as exc:
            self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            return
        except Exception as exc:
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"2D detection failed: {exc}",
            )
            return
        self._send_json(HTTPStatus.OK, result)

    def do_PUT(self) -> None:  # noqa: N802
        self.close_connection = True
        self._send_error_json(HTTPStatus.METHOD_NOT_ALLOWED, "method not allowed")

    def do_DELETE(self) -> None:  # noqa: N802
        self.close_connection = True
        self._send_error_json(HTTPStatus.METHOD_NOT_ALLOWED, "method not allowed")

    def log_message(self, fmt: str, *args: Any) -> None:
        request_line = str(args[0]) if args else ""
        status_code = str(args[1]) if len(args) > 1 else ""
        if (
            request_line.startswith("GET /api/camera/frame.jpg")
            and status_code == "200"
        ):
            return
        print(
            f"[packaging-console] {self.address_string()} "
            f"{self.log_date_time_string()} {fmt % args}"
        )


def build_parser(default_config: Path | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Start the private medicine-packaging console. "
            "It records feedback and may enable the existing operator Follow "
            "lifecycle, but has no autonomous motion, direct joint, suction, "
            "chassis, navigation, or playback API."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        required=default_config is None,
    )
    parser.add_argument("--bind", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--camera-mode",
        choices=("offline", "realsense", "shared"),
        default=None,
        help="Override the configured camera source.",
    )
    return parser


def run_server(
    config_path: Path,
    *,
    bind_override: str | None = None,
    port_override: int | None = None,
    camera_mode_override: str | None = None,
) -> int:
    config_path = config_path.expanduser().resolve()
    config = _read_json(config_path)
    server_cfg = config.get("server", {})
    if not isinstance(server_cfg, dict):
        raise ValueError("server config must be an object")
    bind = str(
        bind_override if bind_override is not None else server_cfg.get("bind", DEFAULT_BIND)
    )
    port = int(
        port_override if port_override is not None else server_cfg.get("port", DEFAULT_PORT)
    )
    if not _is_loopback(bind):
        raise ValueError(
            f"refusing non-loopback bind {bind!r}; this private console is local-only"
        )
    if not 1 <= port <= 65535:
        raise ValueError(f"invalid TCP port: {port}")
    if port in RESERVED_PORTS:
        raise ValueError(
            f"refusing reserved TCP port {port}; use the dedicated packaging port "
            f"{DEFAULT_PORT}"
        )
    if camera_mode_override is not None:
        camera_cfg = config.get("camera")
        if not isinstance(camera_cfg, dict):
            raise ValueError("camera config must be an object")
        camera_cfg["mode"] = camera_mode_override

    # Bind first. A port conflict must not open or disturb the RealSense.
    server = PackagingHTTPServer((bind, port), PackagingRequestHandler, None)
    try:
        app = PackagingConsoleApp(
            config,
            config_path=config_path,
            bind=bind,
            port=port,
        )
        server.app = app
    except Exception:
        server.server_close()
        raise
    stop_once = threading.Event()

    def request_stop(signum: int, _frame: Any) -> None:
        if not stop_once.is_set():
            stop_once.set()
            print(f"\n[packaging-console] received signal {signum}, stopping")
            threading.Thread(target=server.shutdown, daemon=True).start()

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    print(
        f"[packaging-console] operator console at http://{bind}:{port} "
        f"(camera={app.camera.mode}, state={app.camera.state})"
    )
    if app.camera.error:
        print(f"[packaging-console] camera error: {app.camera.error}")
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        app.close()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
    return 0


def main(argv: list[str] | None = None, *, default_config: Path | None = None) -> int:
    args = build_parser(default_config).parse_args(argv)
    try:
        return run_server(
            args.config,
            bind_override=args.bind,
            port_override=args.port,
            camera_mode_override=args.camera_mode,
        )
    except (OSError, ValueError) as exc:
        print(f"[packaging-console] startup failed: {exc}")
        return 2
