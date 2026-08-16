"""Operator-confirmed lifecycle control for the existing AIRBOT Follow."""
from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class TeleopLaunchUnavailable(RuntimeError):
    """Raised when the configured Follow lifecycle is unavailable."""


class TeleopLaunchConflict(RuntimeError):
    """Raised when another lifecycle operation or stale session conflicts."""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_path(
    value: Any,
    *,
    config_dir: Path,
    default: str,
) -> Path:
    setting = Path(str(value or default)).expanduser()
    return (
        setting if setting.is_absolute() else (config_dir / setting)
    ).resolve()


def _clean_output(value: str, limit: int = 800) -> str:
    cleaned = re.sub(
        r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])",
        "",
        value,
    ).replace("\r", "").strip()
    return cleaned if len(cleaned) <= limit else cleaned[-limit:]


class TeleopLauncher:
    """Control only the canonical Follow start/stop scripts.

    Starting requires all four AIRBOT endpoints to be stable beforehand.
    This class never resets CAN, starts arm containers, or accepts a command,
    host, port, path, or environment variable from HTTP input.
    """

    START_KEYS = frozenset(
        {
            "confirm",
            "area_clear",
            "estop_ready",
            "initial_pose_aligned",
        }
    )
    STOP_KEYS = frozenset({"confirm"})
    HARD_RESTART_KEYS = frozenset(
        {
            "confirm",
            "area_clear",
            "estop_ready",
            "master_arms_stable",
        }
    )

    def __init__(
        self,
        config: dict[str, Any],
        *,
        config_dir: Path,
    ) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.repo_root = _resolve_path(
            config.get("repo_root"),
            config_dir=config_dir,
            default="../..",
        )
        self.runtime_config_path = _resolve_path(
            config.get("arm_runtime_config"),
            config_dir=config_dir,
            default="../../web_console/runtime/arm_services.json",
        )
        self.safety_path = _resolve_path(
            config.get("teleop_safety"),
            config_dir=config_dir,
            default="../../web_console/runtime/teleop_safety.json",
        )
        self.health_path = _resolve_path(
            config.get("follow_health"),
            config_dir=config_dir,
            default="../../web_console/runtime/teleop_follow_health.json",
        )
        self.supervisor_path = _resolve_path(
            config.get("follow_supervisor"),
            config_dir=config_dir,
            default="../../web_console/runtime/teleop_follow_supervisor.json",
        )
        self.desired_path = _resolve_path(
            config.get("follow_desired"),
            config_dir=config_dir,
            default="../../web_console/runtime/teleop_follow_desired.json",
        )
        # The production 8888 stack uses the root-owned systemd/UDP Follow
        # controller.  Its authoritative desired-state marker and selected
        # lead live outside the legacy JSON supervisor files.
        self.desired_marker_path = _resolve_path(
            config.get("follow_desired_marker"),
            config_dir=config_dir,
            default="/run/ruc-teleop/follow-desired",
        )
        self.lead_env_path = _resolve_path(
            config.get("follow_lead_env"),
            config_dir=config_dir,
            default="/etc/ruc-teleop/teleop.env",
        )
        self.follow_start_script = self._script_path(
            config.get("follow_start_script"),
            "server/start_teleop_follow.sh",
        )
        self.follow_check_script = self._script_path(
            config.get("follow_check_script"),
            "server/check_teleop_follow.sh",
        )
        self.follow_stop_script = self._script_path(
            config.get("follow_stop_script"),
            "server/stop_teleop_follow.sh",
        )
        self.hard_restart_script = self._script_path(
            config.get("hard_restart_script"),
            "server/hard_restart_teleop_stack.sh",
        )
        self.start_timeout_s = max(
            8.0,
            min(float(config.get("start_timeout_s", 18.0)), 45.0),
        )
        self.stop_timeout_s = max(
            3.0,
            min(float(config.get("stop_timeout_s", 8.0)), 20.0),
        )
        self.status_timeout_s = max(
            0.5,
            min(float(config.get("status_timeout_s", 2.0)), 5.0),
        )
        self.hard_restart_timeout_s = max(
            60.0,
            min(float(config.get("hard_restart_timeout_s", 180.0)), 240.0),
        )
        self.endpoint_timeout_s = max(
            0.05,
            min(float(config.get("endpoint_timeout_s", 0.25)), 2.0),
        )
        self.endpoint_stability_delay_s = max(
            0.05,
            min(
                float(config.get("endpoint_stability_delay_s", 0.25)),
                1.0,
            ),
        )
        self._lock = threading.RLock()
        self._busy = False
        self._state = "stopped" if self.enabled else "disabled"
        self._message = (
            "四个端点就绪后可启动 Follow"
            if self.enabled
            else "遥操生命周期入口未配置"
        )
        self._error = ""
        self._operation_id: str | None = None
        self._operation: str | None = None
        self._requested_at: float | None = None
        self._updated_at = time.time()
        self._thread: threading.Thread | None = None
        self._active_process: subprocess.Popen[str] | None = None
        self._cancel_start = False

    def _script_path(self, value: Any, default: str) -> Path:
        setting = Path(str(value or default)).expanduser()
        resolved = (
            setting if setting.is_absolute() else self.repo_root / setting
        ).resolve()
        allowed = (self.repo_root / "server").resolve()
        try:
            inside_server = (
                os.path.commonpath((str(resolved), str(allowed)))
                == str(allowed)
            )
        except ValueError:
            inside_server = False
        if not inside_server:
            raise ValueError(
                f"teleop script must stay inside {allowed}: {resolved}"
            )
        return resolved

    @staticmethod
    def _normalize_host(value: Any) -> str:
        host = str(value or "").strip()
        if (
            not host
            or len(host) > 253
            or not re.fullmatch(r"[A-Za-z0-9.-]+", host)
            or host.startswith((".", "-"))
            or host.endswith((".", "-"))
            or ".." in host
        ):
            return ""
        return host

    def _runtime_config(self) -> dict[str, str]:
        payload = _load_object(self.runtime_config_path)
        return {
            "remote_host": self._normalize_host(
                payload.get("remote_host")
            ),
        }

    @staticmethod
    def _is_local_lead(host: str) -> bool:
        return host in {"127.0.0.1", "localhost"}

    def _endpoint_ready(self, host: str, port: int) -> bool:
        try:
            with socket.create_connection(
                (host, port),
                timeout=self.endpoint_timeout_s,
            ):
                return True
        except OSError:
            return False

    def _endpoints(self, lead_host: str) -> dict[str, Any]:
        lead_ports = (50050, 50052)
        follower_ports = (50051, 50053)
        lead = [
            self._endpoint_ready(lead_host, port)
            if lead_host
            else False
            for port in lead_ports
        ]
        follower = [
            self._endpoint_ready("127.0.0.1", port)
            for port in follower_ports
        ]
        return {
            "lead_host": lead_host or None,
            "lead_ports": dict(zip(map(str, lead_ports), lead)),
            "follower_ports": dict(zip(map(str, follower_ports), follower)),
            "lead_ready": bool(lead_host) and all(lead),
            "follower_ready": all(follower),
        }

    def _stable_endpoints(self, lead_host: str) -> dict[str, Any]:
        first = self._endpoints(lead_host)
        if not (first["lead_ready"] and first["follower_ready"]):
            return first
        time.sleep(self.endpoint_stability_delay_s)
        return self._endpoints(lead_host)

    def _check_follow(self) -> dict[str, Any]:
        default = {
            "ok": False,
            "tmux": False,
            "state": "missing",
            "heartbeat_age_s": None,
            "error": "",
        }
        if not self.enabled or not self.follow_check_script.is_file():
            return default
        try:
            result = subprocess.run(
                ["bash", str(self.follow_check_script)],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=self.status_timeout_s,
                check=False,
            )
            payload = json.loads(result.stdout.strip() or "{}")
        except (
            OSError,
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
        ) as exc:
            return {
                **default,
                "state": "unknown",
                "error": f"{type(exc).__name__}: {exc}",
            }
        if not isinstance(payload, dict):
            return default
        return {**default, **payload, "ok": result.returncode == 0}

    def _desired_matches(self, lead_host: str) -> bool:
        if self._is_local_lead(lead_host):
            # Local Follow is owned by the fixed user services/tmux session
            # selected by the canonical wrapper scripts. It has no root-owned
            # desired marker or /etc lead environment.
            return True
        desired = _load_object(self.desired_path)
        expected = {
            "lead_urls": [lead_host, lead_host],
            "follow_urls": ["localhost", "localhost"],
            "lead_ports": [50050, 50052],
            "follow_ports": [50051, 50053],
        }
        legacy_matches = desired.get("enabled") is True and all(
            desired.get(key) == value
            for key, value in expected.items()
        )
        if legacy_matches:
            return True
        if not self.desired_marker_path.exists():
            return False
        configured_lead = ""
        try:
            for line in self.lead_env_path.read_text(
                encoding="utf-8"
            ).splitlines():
                key, separator, value = line.partition("=")
                if separator and key.strip() == "LEAD_URL":
                    configured_lead = self._normalize_host(value.strip())
        except OSError:
            return False
        return bool(configured_lead) and configured_lead == lead_host

    def _desired_enabled(self) -> bool:
        return (
            _load_object(self.desired_path).get("enabled") is True
            or self.desired_marker_path.exists()
        )

    def _safety(self) -> dict[str, Any]:
        payload = _load_object(self.safety_path)
        return {
            "latched": payload.get("latched") is True,
            "reason": str(payload.get("reason", "")),
        }

    def status(self) -> dict[str, Any]:
        lead_host = self._runtime_config()["remote_host"]
        endpoints = self._endpoints(lead_host)
        check = self._check_follow()
        supervisor = _load_object(self.supervisor_path)
        safety = self._safety()
        running = check.get("ok") is True
        desired_enabled = self._desired_enabled()
        desired_matches = self._desired_matches(lead_host)
        supervisor_state = str(supervisor.get("state") or "")

        with self._lock:
            state = self._state
            message = self._message
            error = self._error
            if self._busy:
                state = self._state
            elif running:
                state = "running"
                message = "Follow 心跳正常"
                error = ""
            elif desired_enabled:
                state = (
                    supervisor_state
                    if supervisor_state
                    and supervisor_state not in {"stopped", "missing"}
                    else "unknown"
                )
                message = (
                    "Follow 仍处于启用状态但心跳不健康；请正常停止或使用实体急停"
                )
            elif state == "running":
                state = "stopped"
                message = "Follow 已停止"
            elif state != "error":
                state = "stopped" if self.enabled else "disabled"
                message = (
                    "四个端点就绪，可启动 Follow"
                    if endpoints["lead_ready"]
                    and endpoints["follower_ready"]
                    else "等待主臂和执行臂端点全部就绪"
                )
            self._state = state
            self._message = message
            self._error = error
            return {
                "enabled": self.enabled,
                "state": state,
                "busy": self._busy,
                "operation": self._operation,
                "operation_id": self._operation_id,
                "running": running,
                "desired": desired_enabled,
                "desired_matches": desired_matches,
                "message": message,
                "error": error or check.get("error") or None,
                **endpoints,
                "follow": {
                    "tmux": bool(check.get("tmux")),
                    "heartbeat_age_s": check.get("heartbeat_age_s"),
                    "supervisor_state": supervisor_state or None,
                    "restart_count": supervisor.get("restart_count"),
                },
                "safety": safety,
                "requested_at": self._requested_at,
                "updated_at": self._updated_at,
            }

    @staticmethod
    def _require_exact_payload(
        payload: dict[str, Any],
        expected_keys: frozenset[str],
    ) -> None:
        keys = frozenset(payload)
        if keys != expected_keys:
            raise ValueError(
                "request fields must be exactly: "
                + ", ".join(sorted(expected_keys))
            )

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            raise TeleopLaunchUnavailable("teleop launcher is disabled")
        self._require_exact_payload(payload, self.START_KEYS)
        if payload.get("confirm") != "START_FOLLOW":
            raise ValueError("confirm must be START_FOLLOW")
        for field in (
            "area_clear",
            "estop_ready",
            "initial_pose_aligned",
        ):
            if payload.get(field) is not True:
                raise ValueError(f"{field}=true is required")

        snapshot = self.status()
        if snapshot["safety"]["latched"]:
            raise TeleopLaunchConflict(
                "teleop safety is latched: "
                + (snapshot["safety"]["reason"] or "unknown reason")
            )
        if snapshot["running"]:
            if not snapshot["desired_matches"]:
                raise TeleopLaunchConflict(
                    "a healthy Follow session uses a different configuration; "
                    "stop it explicitly before starting"
                )
            return {
                "ok": True,
                "already_running": True,
                "teleop": snapshot,
            }
        if snapshot["desired"] or snapshot["follow"]["tmux"]:
            raise TeleopLaunchConflict(
                "a stale or waiting Follow session exists; stop it explicitly "
                "before starting"
            )
        lead_host = str(snapshot["lead_host"] or "")
        stable = self._stable_endpoints(lead_host)
        if not (stable["lead_ready"] and stable["follower_ready"]):
            raise TeleopLaunchConflict(
                "all lead and follower endpoints must be stable before Follow "
                "can be armed"
            )

        with self._lock:
            if self._busy:
                raise TeleopLaunchConflict(
                    "another teleop lifecycle operation is in progress"
                )
            operation_id = uuid.uuid4().hex
            self._busy = True
            self._state = "starting"
            self._operation = "start"
            self._operation_id = operation_id
            self._message = "端点稳定，正在启动 Follow"
            self._error = ""
            self._cancel_start = False
            self._requested_at = time.time()
            self._updated_at = self._requested_at
            self._thread = threading.Thread(
                target=self._run_start,
                args=(operation_id, lead_host),
                name="medicine-teleop-start",
                daemon=True,
            )
            self._thread.start()
        return {
            "ok": True,
            "already_running": False,
            "operation_id": operation_id,
            "teleop": self.status(),
        }

    def stop(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            raise TeleopLaunchUnavailable("teleop launcher is disabled")
        self._require_exact_payload(payload, self.STOP_KEYS)
        if payload.get("confirm") != "STOP_FOLLOW":
            raise ValueError("confirm must be STOP_FOLLOW")
        with self._lock:
            operation_id = uuid.uuid4().hex
            self._cancel_start = True
            process = self._active_process
            self._busy = True
            self._state = "stopping"
            self._operation = "stop"
            self._operation_id = operation_id
            self._message = "正在正常停止并解除 Follow supervisor"
            self._error = ""
            self._requested_at = time.time()
            self._updated_at = self._requested_at
        if process is not None and process.poll() is None:
            self._terminate_process_group(process)
        with self._lock:
            self._thread = threading.Thread(
                target=self._run_stop,
                args=(operation_id,),
                name="medicine-teleop-stop",
                daemon=True,
            )
            self._thread.start()
        return {
            "ok": True,
            "operation_id": operation_id,
            "teleop": self.status(),
        }

    def hard_restart(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Rebuild both master/follower SDK pairs and then arm Follow.

        The HTTP caller cannot choose a command, path, port, CAN interface, or
        host.  The saved runtime lead host and the fixed root controller are
        the only inputs to the privileged restart transaction.
        """

        if not self.enabled:
            raise TeleopLaunchUnavailable("teleop launcher is disabled")
        self._require_exact_payload(payload, self.HARD_RESTART_KEYS)
        if payload.get("confirm") != "HARD_RESTART_TELEOP":
            raise ValueError("confirm must be HARD_RESTART_TELEOP")
        for field in ("area_clear", "estop_ready", "master_arms_stable"):
            if payload.get(field) is not True:
                raise ValueError(f"{field}=true is required")

        snapshot = self.status()
        if snapshot["safety"]["latched"]:
            raise TeleopLaunchConflict(
                "teleop safety is latched: "
                + (snapshot["safety"]["reason"] or "unknown reason")
            )
        lead_host = str(snapshot["lead_host"] or "")
        if not lead_host:
            raise TeleopLaunchUnavailable(
                "saved master-arm host is missing; configure it before restart"
            )

        with self._lock:
            if self._busy:
                raise TeleopLaunchConflict(
                    "another teleop lifecycle operation is in progress"
                )
            operation_id = uuid.uuid4().hex
            self._busy = True
            self._state = "hard-restarting"
            self._operation = "hard-restart"
            self._operation_id = operation_id
            self._message = (
                "正在停止并重建主臂、执行臂和 Follow；请勿移动主臂"
            )
            self._error = ""
            self._cancel_start = False
            self._requested_at = time.time()
            self._updated_at = self._requested_at
            self._thread = threading.Thread(
                target=self._run_hard_restart,
                args=(operation_id, lead_host),
                name="medicine-teleop-hard-restart",
                daemon=True,
            )
            self._thread.start()
        return {
            "ok": True,
            "operation_id": operation_id,
            "teleop": self.status(),
        }

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[str]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass

    def _run_script(
        self,
        script: Path,
        env: dict[str, str],
        timeout_s: float,
    ) -> None:
        if not script.is_file():
            raise TeleopLaunchUnavailable(
                f"teleop script is missing: {script}"
            )
        try:
            process = subprocess.Popen(
                ["bash", str(script)],
                cwd=self.repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            raise TeleopLaunchUnavailable(
                f"cannot start {script.name}: {exc}"
            ) from exc
        with self._lock:
            self._active_process = process
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            self._terminate_process_group(process)
            raise TeleopLaunchUnavailable(
                f"{script.name} timed out after {timeout_s:.0f}s"
            ) from exc
        finally:
            with self._lock:
                if self._active_process is process:
                    self._active_process = None
        if process.returncode != 0:
            detail = _clean_output(stderr or stdout)
            raise TeleopLaunchUnavailable(
                f"{script.name} failed"
                + (f": {detail}" if detail else "")
            )

    def _run_start(self, operation_id: str, lead_host: str) -> None:
        start_attempted = False
        try:
            stable = self._stable_endpoints(lead_host)
            if not (stable["lead_ready"] and stable["follower_ready"]):
                raise TeleopLaunchUnavailable(
                    "arm endpoint stability was lost before Follow start"
                )
            check_before = self._check_follow()
            if (
                check_before.get("ok") is True
                or check_before.get("tmux") is True
                or self._desired_enabled()
            ):
                raise TeleopLaunchConflict(
                    "Follow state changed during startup; no replacement was attempted"
                )
            with self._lock:
                if self._cancel_start:
                    raise TeleopLaunchUnavailable(
                        "Follow start was cancelled by operator stop"
                    )
            env = os.environ.copy()
            env.update(
                {
                    "LEAD_URL": lead_host,
                    "FOLLOW_URL": "localhost",
                    "LEFT_LEAD_PORT": "50050",
                    "RIGHT_LEAD_PORT": "50052",
                    "LEFT_PORT": "50051",
                    "RIGHT_PORT": "50053",
                    "AIRBOT_TELEOP_START_WAIT_S": "12",
                }
            )
            start_attempted = True
            self._run_script(
                self.follow_start_script,
                env,
                self.start_timeout_s,
            )
            with self._lock:
                cancelled = self._cancel_start
            if cancelled or self._check_follow().get("ok") is not True:
                raise TeleopLaunchUnavailable(
                    "Follow did not publish a healthy heartbeat; "
                    "the request was disarmed"
                )
            if not self._desired_matches(lead_host):
                raise TeleopLaunchUnavailable(
                    "Follow heartbeat is healthy but its endpoint configuration "
                    "does not match; the request was disarmed"
                )
        except Exception as exc:
            with self._lock:
                superseded_by_stop = (
                    self._operation_id != operation_id
                    or self._cancel_start
                )
            if start_attempted and not superseded_by_stop:
                try:
                    self._run_script(
                        self.follow_stop_script,
                        os.environ.copy(),
                        self.stop_timeout_s,
                    )
                except Exception as stop_exc:
                    exc = TeleopLaunchUnavailable(
                        f"{exc}; automatic disarm also failed: {stop_exc}"
                    )
            with self._lock:
                if self._operation_id == operation_id:
                    self._busy = False
                    self._state = "error"
                    self._error = str(exc)
                    self._message = "遥操启动失败"
                    self._updated_at = time.time()
            return
        with self._lock:
            if self._operation_id == operation_id:
                self._busy = False
                self._state = "running"
                self._error = ""
                self._message = "Follow 遥操已启动"
                self._updated_at = time.time()

    def _run_stop(self, operation_id: str) -> None:
        try:
            self._run_script(
                self.follow_stop_script,
                os.environ.copy(),
                self.stop_timeout_s,
            )
            check = self._check_follow()
            if check.get("ok") is True or self._desired_enabled():
                raise TeleopLaunchUnavailable(
                    "Follow stop could not be verified"
                )
        except Exception as exc:
            with self._lock:
                if self._operation_id == operation_id:
                    self._busy = False
                    self._state = "error"
                    self._error = str(exc)
                    self._message = (
                        "无法确认 Follow 已停止；必要时使用实体急停"
                    )
                    self._updated_at = time.time()
            return
        with self._lock:
            if self._operation_id == operation_id:
                self._busy = False
                self._state = "stopped"
                self._error = ""
                self._message = "Follow 已正常停止"
                self._updated_at = time.time()

    def _run_hard_restart(self, operation_id: str, lead_host: str) -> None:
        try:
            env = os.environ.copy()
            env["LEAD_URL"] = lead_host
            first_failure: TeleopLaunchUnavailable | None = None
            for attempt in (1, 2):
                try:
                    self._run_script(
                        self.hard_restart_script,
                        env,
                        self.hard_restart_timeout_s,
                    )
                    break
                except TeleopLaunchUnavailable as exc:
                    with self._lock:
                        cancelled = (
                            self._operation_id != operation_id
                            or self._cancel_start
                        )
                        if not cancelled and attempt == 1:
                            self._message = (
                                "首次彻底重启未完成；正在再次完整清理并重试"
                            )
                            self._updated_at = time.time()
                    if cancelled:
                        raise
                    if attempt == 2:
                        raise TeleopLaunchUnavailable(
                            "hard restart failed after one automatic full retry; "
                            f"first attempt: {first_failure}; retry: {exc}"
                        ) from exc
                    first_failure = exc
            with self._lock:
                cancelled = self._cancel_start
            if cancelled:
                raise TeleopLaunchUnavailable(
                    "hard restart was cancelled by operator stop"
                )
            stable = self._stable_endpoints(lead_host)
            if not (stable["lead_ready"] and stable["follower_ready"]):
                raise TeleopLaunchUnavailable(
                    "hard restart finished but one or more SDK endpoints are unavailable"
                )
            if self._check_follow().get("ok") is not True:
                raise TeleopLaunchUnavailable(
                    "hard restart finished but both Follow workers are not healthy"
                )
            if not self._desired_matches(lead_host):
                raise TeleopLaunchUnavailable(
                    "hard restart is healthy but its endpoint configuration does not match"
                )
        except Exception as exc:
            with self._lock:
                superseded_by_stop = (
                    self._operation_id != operation_id
                    or self._cancel_start
                )
            if not superseded_by_stop:
                try:
                    self._run_script(
                        self.follow_stop_script,
                        os.environ.copy(),
                        self.stop_timeout_s,
                    )
                except Exception as stop_exc:
                    exc = TeleopLaunchUnavailable(
                        f"{exc}; automatic Follow disarm also failed: {stop_exc}"
                    )
            with self._lock:
                if self._operation_id == operation_id:
                    self._busy = False
                    self._state = "error"
                    self._error = str(exc)
                    self._message = "彻底重启失败；Follow 已保持停止"
                    self._updated_at = time.time()
            return
        with self._lock:
            if self._operation_id == operation_id:
                self._busy = False
                self._state = "running"
                self._error = ""
                self._message = "主臂、执行臂和双臂 Follow 已彻底重启"
                self._updated_at = time.time()
