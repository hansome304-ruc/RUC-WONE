"""Record and replay the mobile base trajectory through the 9999 API.

This module deliberately does not publish ROS messages.  The 9999 service
remains the single chassis owner and keeps its existing speed, deadman and
charging interlocks authoritative.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable


class BaseTrajectoryError(RuntimeError):
    """Base class for trajectory errors."""


class BaseTrajectoryConflict(BaseTrajectoryError):
    """Another base operation is active."""


class BaseTrajectoryUnavailable(BaseTrajectoryError):
    """9999 or the odometry source is unavailable."""


class BaseTrajectorySafetyViolation(BaseTrajectoryError):
    """A replay safety preflight failed."""


def _angle_error(target: float, current: float) -> float:
    return (target - current + math.pi) % (2.0 * math.pi) - math.pi


def _safe_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BaseTrajectoryUnavailable(f"{name} is not numeric") from exc
    if not math.isfinite(result):
        raise BaseTrajectoryUnavailable(f"{name} is not finite")
    return result


class _NineNineNineClient:
    """Small authenticated HTTP client for the local 9999 service."""

    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._lock = threading.RLock()
        self._cookie = ""
        self._csrf = ""

    def _clear_session(self) -> None:
        with self._lock:
            self._cookie = ""
            self._csrf = ""

    def _ensure_session(self) -> tuple[str, str]:
        with self._lock:
            if self._cookie and self._csrf:
                return self._cookie, self._csrf
        request = urllib.request.Request(
            self.base_url + "/api/auth/session",
            headers={"User-Agent": "medicine-packaging-console/base-trajectory"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read(64 * 1024).decode("utf-8"))
                cookie_header = str(response.headers.get("Set-Cookie", ""))
        except (OSError, TimeoutError, urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
            raise BaseTrajectoryUnavailable(f"9999 session unavailable: {exc}") from exc
        cookie = cookie_header.split(";", 1)[0].strip()
        csrf = str(payload.get("csrf_token", "")) if isinstance(payload, dict) else ""
        if not cookie or not csrf:
            raise BaseTrajectoryUnavailable("9999 did not issue a control session")
        with self._lock:
            self._cookie = cookie
            self._csrf = csrf
        return cookie, csrf

    def request(self, path: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
        for attempt in range(2):
            cookie, csrf = self._ensure_session()
            headers = {
                "User-Agent": "medicine-packaging-console/base-trajectory",
                "Accept": "application/json",
                "Cookie": cookie,
            }
            data = None
            if method != "GET":
                data = json.dumps(payload or {}).encode("utf-8")
                headers.update(
                    {
                        "Content-Type": "application/json",
                        "Content-Length": str(len(data)),
                        "Origin": self.base_url,
                        "X-CSRF-Token": csrf,
                    }
                )
            request = urllib.request.Request(
                self.base_url + path,
                data=data,
                headers=headers,
                method=method,
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    body = response.read(256 * 1024 + 1)
                    if len(body) > 256 * 1024:
                        raise BaseTrajectoryUnavailable("9999 response is too large")
                    result = json.loads(body.decode("utf-8"))
                    if not isinstance(result, dict):
                        raise BaseTrajectoryUnavailable("9999 response is not an object")
                    if result.get("ok") is False:
                        raise BaseTrajectoryUnavailable(str(result.get("error") or "9999 rejected request"))
                    return result
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403) and attempt == 0:
                    self._clear_session()
                    continue
                try:
                    detail = json.loads(exc.read(64 * 1024).decode("utf-8"))
                    message = detail.get("error") if isinstance(detail, dict) else None
                except Exception:
                    message = None
                raise BaseTrajectoryUnavailable(
                    str(message or f"9999 rejected request: HTTP {exc.code}")
                ) from exc
            except (OSError, TimeoutError, urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
                raise BaseTrajectoryUnavailable(f"9999 request failed: {exc}") from exc
        raise BaseTrajectoryUnavailable("9999 session refresh failed")

    def state(self) -> dict[str, Any]:
        payload = self.request("/api/base/state")
        data = payload.get("data", payload)
        if not isinstance(data, dict):
            raise BaseTrajectoryUnavailable("9999 base state has no data object")
        pose = data.get("pose")
        velocity = data.get("velocity") or {}
        if not isinstance(pose, dict):
            raise BaseTrajectoryUnavailable("9999 base state has no pose")
        return {
            "stamp": _safe_float(data.get("stamp", time.time()), "stamp"),
            "x": _safe_float(pose.get("x"), "pose.x"),
            "y": _safe_float(pose.get("y"), "pose.y"),
            "yaw": _safe_float(pose.get("yaw"), "pose.yaw"),
            "vx": _safe_float(velocity.get("x", 0.0), "velocity.x"),
            "vy": _safe_float(velocity.get("y", 0.0), "velocity.y"),
            "wz": _safe_float(velocity.get("yaw", 0.0), "velocity.yaw"),
            "charging_interlock": data.get("charging_interlock"),
        }

    def move(
        self,
        command: str,
        *,
        speed: float,
        yaw_speed: float,
        duration: float,
        hold_id: str,
        sequence: int,
    ) -> dict[str, Any]:
        return self.request(
            "/api/base/move",
            method="POST",
            payload={
                "command": command,
                "speed": speed,
                "yaw_speed": yaw_speed,
                "duration": duration,
                "hold_id": hold_id,
                "sequence": sequence,
            },
        )


class BaseTrajectoryController:
    """Thread-safe recorder/replayer for the mobile base."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        config: dict[str, Any] | None,
        *,
        config_dir: Path,
        interlock: Callable[[], str | None] | None = None,
    ) -> None:
        cfg = dict(config or {})
        self.enabled = cfg.get("enabled", True) is True
        self.api_url = str(cfg.get("api_url", "http://127.0.0.1:9999")).rstrip("/")
        if not re.fullmatch(r"http://(?:127\.0\.0\.1|localhost):\d+", self.api_url):
            raise ValueError("base_trajectory.api_url must be a loopback HTTP URL")
        self.sample_hz = min(20.0, max(2.0, float(cfg.get("sample_hz", 10.0))))
        self.control_hz = min(20.0, max(2.0, float(cfg.get("control_hz", 10.0))))
        self.max_duration_s = min(1800.0, max(1.0, float(cfg.get("max_duration_s", 300.0))))
        self.max_points = min(20000, max(20, int(cfg.get("max_points", 6000))))
        self.max_linear_mps = min(0.20, max(0.01, float(cfg.get("max_linear_mps", 0.10))))
        self.max_yaw_radps = min(0.50, max(0.02, float(cfg.get("max_yaw_radps", 0.25))))
        self.start_tolerance_m = min(1.0, max(0.02, float(cfg.get("start_tolerance_m", 0.12))))
        self.start_yaw_tolerance_rad = min(math.pi, max(0.05, float(cfg.get("start_yaw_tolerance_rad", 0.35))))
        self.waypoint_tolerance_m = min(0.5, max(0.02, float(cfg.get("waypoint_tolerance_m", 0.06))))
        self.waypoint_yaw_tolerance_rad = min(math.pi, max(0.08, float(cfg.get("waypoint_yaw_tolerance_rad", 0.20))))
        self.replay_timeout_s = min(3600.0, max(10.0, float(cfg.get("replay_timeout_s", 900.0))))
        storage_setting = Path(str(cfg.get("storage_dir", "runtime/base_trajectories"))).expanduser()
        self.storage_dir = storage_setting if storage_setting.is_absolute() else (config_dir / storage_setting).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._client = _NineNineNineClient(self.api_url, min(10.0, max(0.5, float(cfg.get("request_timeout_s", 3.0)))))
        self._interlock = interlock
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._mode = "disabled" if not self.enabled else "idle"
        self._error = ""
        self._message = ""
        self._recording: dict[str, Any] | None = None
        self._replay: dict[str, Any] | None = None
        self._hold_id = ""
        self._sequence = 0
        self._last_saved: dict[str, Any] | None = None

    def _check_enabled(self) -> None:
        if not self.enabled:
            raise BaseTrajectoryUnavailable("base trajectory feature is disabled")

    def _check_interlock(self) -> None:
        if self._interlock is not None:
            blocker = self._interlock()
            if blocker:
                raise BaseTrajectoryConflict(blocker)

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._mode = "error"
            self._error = message
            self._message = message

    def _point(self, state: dict[str, Any], t_s: float) -> dict[str, Any]:
        return {
            "t_s": round(max(0.0, float(t_s)), 4),
            "stamp": state["stamp"],
            "x": state["x"],
            "y": state["y"],
            "yaw": state["yaw"],
            "velocity": {
                "x": state["vx"],
                "y": state["vy"],
                "yaw": state["wz"],
            },
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            recording = dict(self._recording or {})
            replay = dict(self._replay or {})
            return {
                "enabled": self.enabled,
                "mode": self._mode,
                "active": self._mode in {"recording", "replaying", "stopping"},
                "error": self._error,
                "message": self._message,
                "recording": recording or None,
                "replay": replay or None,
                "last_saved": dict(self._last_saved or {}) or None,
            }

    def _load(self, recording_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", recording_id):
            raise ValueError("recording_id is invalid")
        path = (self.storage_dir / f"{recording_id}.json").resolve()
        try:
            path.relative_to(self.storage_dir.resolve())
        except ValueError as exc:
            raise ValueError("recording_id escapes storage directory") from exc
        if not path.is_file():
            raise ValueError(f"unknown base recording: {recording_id}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read base recording: {exc}") from exc
        points = payload.get("points") if isinstance(payload, dict) else None
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError("base recording has fewer than two points")
        return payload

    def list_recordings(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for path in sorted(self.storage_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                points = payload.get("points", [])
                if not isinstance(points, list) or not points:
                    continue
                result.append(
                    {
                        "id": str(payload.get("id") or path.stem),
                        "label": str(payload.get("label") or path.stem),
                        "created_at": payload.get("created_at"),
                        "duration_s": float(payload.get("duration_s", points[-1].get("t_s", 0.0))),
                        "point_count": len(points),
                        "path": str(path),
                        "start_pose": {key: points[0].get(key) for key in ("x", "y", "yaw")},
                        "end_pose": {key: points[-1].get(key) for key in ("x", "y", "yaw")},
                    }
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return result

    def start_recording(self, label: str = "base-trajectory") -> dict[str, Any]:
        self._check_enabled()
        self._check_interlock()
        if not isinstance(label, str) or not label.strip():
            raise ValueError("label must be a non-empty string")
        label = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip())[:48] or "base-trajectory"
        with self._lock:
            if self._mode in {"recording", "replaying", "stopping"}:
                raise BaseTrajectoryConflict("another base trajectory operation is active")
        state = self._client.state()
        recording_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{label}-{uuid.uuid4().hex[:8]}"
        now = time.time()
        record = {
            "id": recording_id,
            "label": label,
            "created_at": now,
            "started_at": now,
            "duration_s": 0.0,
            "point_count": 1,
            "start_pose": {key: state[key] for key in ("x", "y", "yaw")},
            "points": [self._point(state, 0.0)],
        }
        with self._lock:
            self._recording = record
            self._replay = None
            self._error = ""
            self._message = "Recording base trajectory"
            self._mode = "recording"
            self._stop_event.clear()
            self._worker = threading.Thread(target=self._record_loop, name="base-trajectory-record", daemon=True)
            self._worker.start()
        return self.status()["recording"] | {"state": "recording"}

    def _record_loop(self) -> None:
        started = time.monotonic()
        period = 1.0 / self.sample_hz
        next_tick = started + period
        try:
            while not self._stop_event.is_set():
                wait_s = max(0.0, next_tick - time.monotonic())
                if self._stop_event.wait(wait_s):
                    break
                elapsed = time.monotonic() - started
                state = self._client.state()
                with self._lock:
                    if self._recording is None:
                        break
                    points = self._recording["points"]
                    if len(points) >= self.max_points or elapsed >= self.max_duration_s:
                        break
                    points.append(self._point(state, elapsed))
                    self._recording["duration_s"] = elapsed
                    self._recording["point_count"] = len(points)
                next_tick += period
            self._finalize_recording()
        except Exception as exc:
            self._set_error(f"Base trajectory operation failed: {type(exc).__name__}: {exc}")

    def _finalize_recording(self) -> None:
        with self._lock:
            record = dict(self._recording or {})
            if not record:
                return
            points = list(record.get("points", []))
        if len(points) < 2:
            self._set_error("Base trajectory recording needs at least two points")
            return
        record["duration_s"] = float(points[-1].get("t_s", 0.0))
        record["point_count"] = len(points)
        record["ended_at"] = time.time()
        record["format_version"] = self.FORMAT_VERSION
        target = (self.storage_dir / f"{record['id']}.json").resolve()
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        summary = {
            "id": record["id"],
            "label": record["label"],
            "created_at": record["created_at"],
            "duration_s": record["duration_s"],
            "point_count": record["point_count"],
            "path": str(target),
            "start_pose": record["start_pose"],
            "end_pose": {key: points[-1].get(key) for key in ("x", "y", "yaw")},
        }
        with self._lock:
            self._last_saved = summary
            self._recording = summary
            self._mode = "idle"
            self._message = "Base trajectory recording saved"
            self._stop_event.clear()

    def stop_recording(self) -> dict[str, Any]:
        with self._lock:
            if self._mode != "recording":
                return self.status()["recording"] or {"state": self._mode}
            self._mode = "stopping"
            worker = self._worker
            self._stop_event.set()
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=6.0)
        with self._lock:
            return dict(self._recording or {"state": self._mode})

    def preflight(self, recording_id: str) -> dict[str, Any]:
        self._check_enabled()
        self._check_interlock()
        payload = self._load(recording_id)
        points = payload["points"]
        current = self._client.state()
        first = points[0]
        distance = math.hypot(current["x"] - float(first["x"]), current["y"] - float(first["y"]))
        yaw_error = abs(_angle_error(float(first["yaw"]), current["yaw"]))
        charging = current.get("charging_interlock")
        blockers: list[str] = []
        if charging is True or (isinstance(charging, dict) and charging.get("motion_allowed") is not True):
            blockers.append("Base is charging; replay is blocked")
        if distance > self.start_tolerance_m:
            blockers.append(f"Current position is {distance:.3f} m from recording start; limit is {self.start_tolerance_m:.3f} m")
        if yaw_error > self.start_yaw_tolerance_rad:
            blockers.append(f"Current yaw differs by {yaw_error:.3f} rad from recording start; limit is {self.start_yaw_tolerance_rad:.3f} rad")
        return {
            "ok": not blockers,
            "ready": not blockers,
            "recording": {
                "id": payload.get("id", recording_id),
                "label": payload.get("label", recording_id),
                "duration_s": payload.get("duration_s", points[-1].get("t_s", 0.0)),
                "point_count": len(points),
            },
            "current_pose": {key: current[key] for key in ("x", "y", "yaw")},
            "start_pose": {key: first[key] for key in ("x", "y", "yaw")},
            "start_distance_m": distance,
            "start_yaw_error_rad": yaw_error,
            "blockers": blockers,
        }

    def start_replay(self, recording_id: str, confirmation: str) -> dict[str, Any]:
        self._check_enabled()
        if confirmation != "REPLAY_BASE_TRAJECTORY":
            raise ValueError("confirmation must be REPLAY_BASE_TRAJECTORY")
        with self._lock:
            if self._mode in {"recording", "replaying", "stopping"}:
                raise BaseTrajectoryConflict("another base trajectory operation is active")
        preflight = self.preflight(recording_id)
        if preflight["ready"] is not True:
            raise BaseTrajectorySafetyViolation("; ".join(preflight["blockers"]))
        payload = self._load(recording_id)
        with self._lock:
            self._replay = {
                "recording_id": recording_id,
                "label": payload.get("label", recording_id),
                "point_count": len(payload["points"]),
                "duration_s": float(payload.get("duration_s", payload["points"][-1].get("t_s", 0.0))),
                "point_index": 0,
                "started_at": time.time(),
                "state": "replaying",
            }
            self._recording = None
            self._error = ""
            self._message = "Base trajectory replay started"
            self._mode = "replaying"
            self._stop_event.clear()
            self._hold_id = f"base-replay-{uuid.uuid4().hex[:24]}"
            self._sequence = 0
            self._worker = threading.Thread(
                target=self._replay_loop,
                args=(payload,),
                name="base-trajectory-replay",
                daemon=True,
            )
            self._worker.start()
        return self.status()["replay"] | {"state": "replaying", "preflight": preflight}

    def _send_stop(self) -> None:
        with self._lock:
            hold_id = self._hold_id
            self._sequence += 1
            sequence = self._sequence
        if hold_id:
            try:
                self._client.move(
                    "stop",
                    speed=0.0,
                    yaw_speed=0.0,
                    duration=0.0,
                    hold_id=hold_id,
                    sequence=sequence,
                )
            except Exception:
                pass

    def _replay_loop(self, payload: dict[str, Any]) -> None:
        points = payload["points"]
        period = 1.0 / self.control_hz
        index = 1
        started = time.monotonic()
        try:
            while not self._stop_event.is_set() and index < len(points):
                if time.monotonic() - started > self.replay_timeout_s:
                    raise BaseTrajectorySafetyViolation("Base trajectory replay timed out")
                current = self._client.state()
                target = points[index]
                dx = float(target["x"]) - current["x"]
                dy = float(target["y"]) - current["y"]
                distance = math.hypot(dx, dy)
                yaw_error = _angle_error(float(target["yaw"]), current["yaw"])
                if distance <= self.waypoint_tolerance_m and abs(yaw_error) <= self.waypoint_yaw_tolerance_rad:
                    index += 1
                    with self._lock:
                        if self._replay:
                            self._replay["point_index"] = index
                            self._replay["progress"] = index / max(1, len(points) - 1)
                    continue
                base_x = math.cos(current["yaw"]) * dx + math.sin(current["yaw"]) * dy
                base_y = -math.sin(current["yaw"]) * dx + math.cos(current["yaw"]) * dy
                if distance <= self.waypoint_tolerance_m * 1.8 and abs(yaw_error) > self.waypoint_yaw_tolerance_rad:
                    command = "turn-left" if yaw_error > 0 else "turn-right"
                    speed = 0.0
                    yaw_speed = min(self.max_yaw_radps, max(0.08, abs(yaw_error) * 0.8))
                elif abs(base_x) >= abs(base_y):
                    command = "forward" if base_x >= 0 else "backward"
                    speed = min(self.max_linear_mps, max(0.025, abs(base_x) * 0.8))
                    yaw_speed = 0.0
                else:
                    command = "left" if base_y >= 0 else "right"
                    speed = min(self.max_linear_mps, max(0.025, abs(base_y) * 0.8))
                    yaw_speed = 0.0
                with self._lock:
                    self._sequence += 1
                    sequence = self._sequence
                    hold_id = self._hold_id
                self._client.move(
                    command,
                    speed=speed,
                    yaw_speed=yaw_speed,
                    duration=min(0.45, max(0.25, period * 2.5)),
                    hold_id=hold_id,
                    sequence=sequence,
                )
                with self._lock:
                    if self._replay:
                        self._replay["point_index"] = index
                        self._replay["progress"] = index / max(1, len(points) - 1)
                        self._replay["current_pose"] = {key: current[key] for key in ("x", "y", "yaw")}
                self._stop_event.wait(period)
            if not self._stop_event.is_set():
                self._send_stop()
                with self._lock:
                    if self._replay:
                        self._replay["state"] = "completed"
                    self._mode = "idle"
                    self._message = "Base trajectory replay completed"
        except Exception as exc:
            self._send_stop()
            self._set_error(f"Base trajectory operation failed: {type(exc).__name__}: {exc}")

    def stop_replay(self) -> dict[str, Any]:
        with self._lock:
            if self._mode != "replaying":
                return self.status()["replay"] or {"state": self._mode}
            self._mode = "stopping"
            self._stop_event.set()
        self._send_stop()
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=4.0)
        with self._lock:
            replay = dict(self._replay or {})
            replay["state"] = "stopped"
            self._replay = replay
            self._mode = "idle"
            self._message = "Recording base trajectory"
            return replay

    def close(self) -> None:
        with self._lock:
            mode = self._mode
        if mode == "recording":
            self.stop_recording()
        elif mode == "replaying":
            self.stop_replay()

