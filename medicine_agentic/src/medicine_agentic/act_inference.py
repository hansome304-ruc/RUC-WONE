"""Read-only bridge from the packaging console to the remote ACT policy service."""
from __future__ import annotations

import base64
import json
import stat
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import cv2
import numpy as np


SERVICE_VERSIONS = {
    "medicine_act_inference_v1",
    "medicine_act_inference_v2_terminal15",
    "medicine_zr0_inference_v1",
}
CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
ABSOLUTE_ACTION_REPRESENTATION = "absolute_joint_target"
DELTA_ACTION_REPRESENTATION = "delta_target_minus_observation_state"
ACTION_REPRESENTATIONS = {
    ABSOLUTE_ACTION_REPRESENTATION,
    DELTA_ACTION_REPRESENTATION,
}
LEGACY_ACTION_REPRESENTATIONS = {
    "medicine_act_inference_v1": ABSOLUTE_ACTION_REPRESENTATION,
    "medicine_zr0_inference_v1": ABSOLUTE_ACTION_REPRESENTATION,
}


class ActInferenceUnavailable(RuntimeError):
    """The configured ACT inference service cannot be reached."""


class ActInferenceProtocolError(RuntimeError):
    """The ACT service returned a response that violates the pinned contract."""


def validate_prediction_response(
    response: dict[str, Any], *, selected_horizon: int
) -> dict[str, Any]:
    try:
        actions = np.asarray(response["actions"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ActInferenceProtocolError("ACT response has no numeric actions") from exc
    if actions.shape != (selected_horizon, 14) or not np.all(np.isfinite(actions)):
        raise ActInferenceProtocolError("ACT response has invalid action dimensions or values")
    termination_supported = response.get("termination_supported", False)
    if not isinstance(termination_supported, bool):
        raise ActInferenceProtocolError("ACT termination capability must be boolean")
    result = dict(response)
    result["actions"] = actions.tolist()
    action_representation = response.get("action_representation")
    if action_representation is None:
        action_representation = LEGACY_ACTION_REPRESENTATIONS.get(
            str(response.get("service", ""))
        )
    if action_representation not in ACTION_REPRESENTATIONS:
        raise ActInferenceProtocolError(
            "ACT response has no supported action_representation"
        )
    result["action_representation"] = action_representation
    if termination_supported:
        try:
            done_probs = np.asarray(response["done_probs"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise ActInferenceProtocolError("terminal ACT response has no numeric done_probs") from exc
        if (
            done_probs.shape != (selected_horizon,)
            or not np.all(np.isfinite(done_probs))
            or np.any(done_probs < 0.0)
            or np.any(done_probs > 1.0)
        ):
            raise ActInferenceProtocolError("ACT response has invalid done probabilities")
        result["done_probs"] = done_probs.tolist()
    elif "done_probs" in response:
        raise ActInferenceProtocolError("legacy ACT response unexpectedly contains done_probs")
    result["termination_supported"] = termination_supported
    result["execution_enabled"] = False
    return result


def _resolve_path(value: str, *, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else config_dir / path).resolve()


def _encode_jpeg(frame_bgr: np.ndarray) -> str:
    frame = np.asarray(frame_bgr)
    if (
        frame.ndim != 3
        or frame.shape[2] != 3
        or frame.dtype != np.uint8
        or frame.shape[0] < 1
        or frame.shape[1] < 1
    ):
        raise ValueError("ACT camera frames must be non-empty uint8 BGR images")
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("ACT camera frame cannot be JPEG encoded")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def build_prediction_payload(
    *,
    state: Sequence[float],
    frames_bgr: Mapping[str, np.ndarray],
    horizon: int,
    request_id: str,
    session_id: str,
) -> dict[str, Any]:
    qpos = np.asarray(state, dtype=np.float32)
    if qpos.shape != (14,) or not np.all(np.isfinite(qpos)):
        raise ValueError("ACT state must contain 14 finite values")
    if set(frames_bgr) != set(CAMERA_NAMES):
        raise ValueError("ACT observation requires front, left-wrist and right-wrist frames")
    if not 1 <= int(horizon) <= 100:
        raise ValueError("ACT horizon must be between 1 and 100")
    return {
        "request_id": request_id,
        "session_id": session_id,
        "horizon": int(horizon),
        "observation": {
            "state": qpos.tolist(),
            "images": {name: _encode_jpeg(frames_bgr[name]) for name in CAMERA_NAMES},
        },
    }


class ActInferenceClient:
    """Authenticated, model-pinned HTTP client with no robot command capability."""

    def __init__(self, config: dict[str, Any] | None, *, config_dir: Path) -> None:
        config = dict(config or {})
        self.enabled = config.get("enabled") is True
        self.execution_enabled = config.get("execution_enabled") is True
        self._profile_lock = threading.RLock()
        self._switch_lock = threading.Lock()
        self._profiles: dict[str, dict[str, str]] = {}
        raw_profiles = config.get("profiles")
        if raw_profiles is not None:
            if not isinstance(raw_profiles, dict) or not raw_profiles:
                raise ValueError("act_inference.profiles must be a non-empty object")
            for profile_id, raw_profile in raw_profiles.items():
                if (
                    not isinstance(profile_id, str)
                    or not profile_id
                    or len(profile_id) > 32
                    or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in profile_id)
                    or not isinstance(raw_profile, dict)
                ):
                    raise ValueError("act_inference profile IDs must be lowercase safe names")
                base_url = str(raw_profile.get("base_url", "")).rstrip("/")
                expected_hash = str(
                    raw_profile.get("expected_model_sha256", "")
                ).lower()
                expected_representation = str(
                    raw_profile.get(
                        "expected_action_representation",
                        ABSOLUTE_ACTION_REPRESENTATION,
                    )
                )
                self._validate_profile_contract(
                    base_url=base_url,
                    expected_model_sha256=expected_hash,
                    expected_action_representation=expected_representation,
                )
                self._profiles[profile_id] = {
                    "id": profile_id,
                    "label": str(raw_profile.get("label") or profile_id),
                    "task": str(raw_profile.get("task") or profile_id),
                    "base_url": base_url,
                    "expected_model_sha256": expected_hash,
                    "expected_action_representation": expected_representation,
                }
            self.active_profile = str(config.get("active_profile", ""))
            if self.active_profile not in self._profiles:
                raise ValueError("act_inference.active_profile is not configured")
            active = self._profiles[self.active_profile]
            self.base_url = active["base_url"]
            self.expected_model_sha256 = active["expected_model_sha256"]
            self.expected_action_representation = active[
                "expected_action_representation"
            ]
        else:
            self.active_profile = "default"
            self.base_url = str(config.get("base_url", "")).rstrip("/")
            self.expected_model_sha256 = str(
                config.get("expected_model_sha256", "")
            ).lower()
            self.expected_action_representation = str(
                config.get(
                    "expected_action_representation",
                    ABSOLUTE_ACTION_REPRESENTATION,
                )
            )
            self._profiles[self.active_profile] = {
                "id": self.active_profile,
                "label": "默认模型",
                "task": "default",
                "base_url": self.base_url,
                "expected_model_sha256": self.expected_model_sha256,
                "expected_action_representation": self.expected_action_representation,
            }
        self.timeout_s = max(0.25, min(float(config.get("timeout_s", 5.0)), 30.0))
        self.default_horizon = max(1, min(int(config.get("default_horizon", 25)), 100))
        raw_left_gripper_override = config.get("left_gripper_observation_override")
        self.left_gripper_observation_override = (
            None
            if raw_left_gripper_override is None
            else float(raw_left_gripper_override)
        )
        if (
            self.left_gripper_observation_override is not None
            and (
                not np.isfinite(self.left_gripper_observation_override)
                or not 0.0 <= self.left_gripper_observation_override <= 0.1
            )
        ):
            raise ValueError(
                "act_inference.left_gripper_observation_override must be "
                "a finite value in [0.0, 0.1]"
            )
        self._token = ""
        self._status_lock = threading.Lock()
        self._cached_status: tuple[float, dict[str, Any]] | None = None
        self._cache_ttl_s = max(0.1, min(float(config.get("status_cache_s", 2.0)), 30.0))
        if not self.enabled:
            return
        self._validate_profile_contract(
            base_url=self.base_url,
            expected_model_sha256=self.expected_model_sha256,
            expected_action_representation=self.expected_action_representation,
        )
        token_setting = str(config.get("token_file", "")).strip()
        if not token_setting:
            raise ValueError("act_inference.token_file is required")
        token_path = _resolve_path(token_setting, config_dir=config_dir)
        try:
            mode = stat.S_IMODE(token_path.stat().st_mode)
            self._token = token_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"cannot read act_inference.token_file: {exc}") from exc
        if mode & 0o077:
            raise ValueError("act_inference.token_file must not be accessible by group or other")
        if len(self._token) < 32:
            raise ValueError("act_inference token must contain at least 32 characters")

    @staticmethod
    def _validate_profile_contract(
        *,
        base_url: str,
        expected_model_sha256: str,
        expected_action_representation: str,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("ACT model base_url must be a plain HTTP origin")
        if len(expected_model_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in expected_model_sha256
        ):
            raise ValueError("ACT model hash must be a lowercase SHA256")
        if expected_action_representation not in ACTION_REPRESENTATIONS:
            raise ValueError("ACT model action representation is unsupported")

    def profiles(self) -> dict[str, Any]:
        with self._profile_lock:
            return {
                "active": self.active_profile,
                "items": [
                    {
                        "id": profile["id"],
                        "label": profile["label"],
                        "task": profile["task"],
                        "base_url": profile["base_url"],
                        "model_sha256": profile["expected_model_sha256"],
                        "action_representation": profile[
                            "expected_action_representation"
                        ],
                    }
                    for profile in self._profiles.values()
                ],
            }

    def select_profile(self, profile_id: str) -> dict[str, Any]:
        if not self.enabled:
            raise ActInferenceUnavailable("ACT inference is disabled")
        with self._switch_lock:
            with self._profile_lock:
                if profile_id not in self._profiles:
                    raise ValueError("unknown ACT model profile")
                previous = self.active_profile
                profile = self._profiles[profile_id]
                self.active_profile = profile_id
                self.base_url = profile["base_url"]
                self.expected_model_sha256 = profile["expected_model_sha256"]
                self.expected_action_representation = profile[
                    "expected_action_representation"
                ]
            with self._status_lock:
                self._cached_status = None
            status = self.status(force=True)
            if status.get("ready") is True:
                return status
            with self._profile_lock:
                rollback = self._profiles[previous]
                self.active_profile = previous
                self.base_url = rollback["base_url"]
                self.expected_model_sha256 = rollback["expected_model_sha256"]
                self.expected_action_representation = rollback[
                    "expected_action_representation"
                ]
            with self._status_lock:
                self._cached_status = None
            raise ActInferenceUnavailable(
                str(status.get("error") or "selected ACT model is not ready")
            )

    @staticmethod
    def _json_response(response: Any) -> dict[str, Any]:
        body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise ActInferenceProtocolError("ACT response exceeds 2 MiB")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ActInferenceProtocolError("ACT response is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ActInferenceProtocolError("ACT response must be a JSON object")
        return payload

    def _request(
        self,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        authenticated: bool,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise ActInferenceUnavailable("ACT inference is disabled")
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if authenticated:
            headers["Authorization"] = f"Bearer {self._token}"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method="POST" if body is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return self._json_response(response)
        except urllib.error.HTTPError as exc:
            try:
                details = self._json_response(exc).get("error", f"HTTP {exc.code}")
            except ActInferenceProtocolError:
                details = f"HTTP {exc.code}"
            raise ActInferenceUnavailable(f"ACT service rejected the request: {details}") from exc
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise ActInferenceUnavailable(
                f"cannot reach ACT service: {type(exc).__name__}: {exc}"
            ) from exc

    def _validate_identity(self, payload: dict[str, Any]) -> None:
        if payload.get("ok") is not True or payload.get("service") not in SERVICE_VERSIONS:
            raise ActInferenceProtocolError("unexpected ACT service identity")
        if payload.get("model_sha256") != self.expected_model_sha256:
            raise ActInferenceProtocolError("ACT service model SHA256 does not match configuration")
        observed_representation = payload.get("action_representation") or (
            LEGACY_ACTION_REPRESENTATIONS.get(str(payload.get("service", "")))
        )
        if observed_representation != self.expected_action_representation:
            raise ActInferenceProtocolError(
                "ACT service action representation does not match configuration"
            )

    def _profile_fields(self) -> dict[str, Any]:
        profile = self._profiles[self.active_profile]
        return {
            "profile_id": self.active_profile,
            "profile_label": profile["label"],
            "profile_task": profile["task"],
            "profiles": self.profiles(),
        }

    def status(self, *, force: bool = False) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "ready": False,
                "execution_enabled": False,
                "left_gripper_observation_override": (
                    self.left_gripper_observation_override
                ),
                "error": "ACT inference is disabled",
            }
        now = time.monotonic()
        with self._status_lock:
            if (
                not force
                and self._cached_status is not None
                and now - self._cached_status[0] <= self._cache_ttl_s
            ):
                return dict(self._cached_status[1])
        try:
            with self._profile_lock:
                payload = self._request("/healthz", authenticated=False)
                self._validate_identity(payload)
                status = {
                    "enabled": True,
                    "ready": True,
                    "execution_enabled": self.execution_enabled,
                    "base_url": self.base_url,
                    "service": payload["service"],
                    "model_sha256": payload["model_sha256"],
                    "device": payload.get("device"),
                    "action_representation": payload.get("action_representation")
                    or LEGACY_ACTION_REPRESENTATIONS.get(payload["service"]),
                    "left_gripper_observation_override": (
                        self.left_gripper_observation_override
                    ),
                    "error": None,
                    **self._profile_fields(),
                }
        except (ActInferenceUnavailable, ActInferenceProtocolError) as exc:
            with self._profile_lock:
                status = {
                    "enabled": True,
                    "ready": False,
                    "execution_enabled": self.execution_enabled,
                    "base_url": self.base_url,
                    "model_sha256": self.expected_model_sha256,
                    "action_representation": self.expected_action_representation,
                    "left_gripper_observation_override": (
                        self.left_gripper_observation_override
                    ),
                    "error": str(exc),
                    **self._profile_fields(),
                }
        with self._status_lock:
            self._cached_status = (now, dict(status))
        return status

    def cached_status(self) -> dict[str, Any]:
        """Return the last model status without performing network I/O."""

        if not self.enabled:
            return {
                "enabled": False,
                "ready": False,
                "execution_enabled": False,
                "left_gripper_observation_override": (
                    self.left_gripper_observation_override
                ),
                "error": "ACT inference is disabled",
                "cached": True,
            }
        now = time.monotonic()
        with self._status_lock:
            cached = self._cached_status
            if cached is not None:
                checked_at, payload = cached
                snapshot = dict(payload)
                snapshot["cached"] = True
                snapshot["cache_age_s"] = max(0.0, now - checked_at)
                return snapshot
        with self._profile_lock:
            return {
                "enabled": True,
                "ready": None,
                "execution_enabled": self.execution_enabled,
                "base_url": self.base_url,
                "model_sha256": self.expected_model_sha256,
                "action_representation": self.expected_action_representation,
                "left_gripper_observation_override": (
                    self.left_gripper_observation_override
                ),
                "error": None,
                "cached": False,
                **self._profile_fields(),
            }

    def predict(
        self,
        *,
        state: Sequence[float],
        frames_bgr: Mapping[str, np.ndarray],
        horizon: int | None = None,
        request_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        selected_horizon = self.default_horizon if horizon is None else int(horizon)
        model_state = np.asarray(state, dtype=np.float32).copy()
        if self.left_gripper_observation_override is not None:
            if model_state.shape != (14,) or not np.all(np.isfinite(model_state)):
                raise ValueError("ACT state must contain 14 finite values")
            model_state[6] = self.left_gripper_observation_override
        request_payload = build_prediction_payload(
            state=model_state,
            frames_bgr=frames_bgr,
            horizon=selected_horizon,
            request_id=request_id,
            session_id=session_id,
        )
        with self._profile_lock:
            response = self._request(
                "/v1/act/predict", payload=request_payload, authenticated=True
            )
            self._validate_identity(response)
        result = validate_prediction_response(
            response, selected_horizon=selected_horizon
        )
        result["execution_enabled"] = self.execution_enabled
        if self.left_gripper_observation_override is not None:
            result["observation_overrides"] = {
                "left_gripper_raw": self.left_gripper_observation_override,
            }
        return result
