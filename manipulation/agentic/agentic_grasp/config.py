"""Centralised settings, all overridable via `.env` or environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass


def _get(name: str, default: str | None = None) -> str:
    val = os.getenv(name, default)
    if val is None:
        raise RuntimeError(f"Required env var {name!r} is unset and has no default")
    return val


def _get_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # --- perception server ---
    perception_url: str = _get("PERCEPTION_URL", "http://127.0.0.1:9100")
    perception_token: str = os.getenv("PERCEPTION_TOKEN", "")

    # --- hardware ---
    cam_front_serial: str = _get("CAM_FRONT_SERIAL", "420222072569")
    cam_left_serial: str = _get("CAM_LEFT_SERIAL", "347622072392")
    cam_right_serial: str = _get("CAM_RIGHT_SERIAL", "347622071407")
    left_arm_port: int = int(_get("LEFT_ARM_PORT", "50051"))
    right_arm_port: int = int(_get("RIGHT_ARM_PORT", "50053"))

    # --- calibration ---
    calib_dir: Path = Path(_get("CALIB_DIR", "/home/ubuntu/calib/result"))
    gripper_tcp_offset: float = _get_float("GRIPPER_TCP_OFFSET", 0.13)
    side_approach_min_y: float = _get_float("SIDE_APPROACH_MIN_Y", 0.20)
    grasp_z_min: float = _get_float("GRASP_Z_MIN", 0.12)
    grasp_z_max: float = _get_float("GRASP_Z_MAX", 0.24)
    grasp_pre_z_min: float = _get_float("GRASP_PRE_Z_MIN", 0.12)
    grasp_max_abs_approach_z: float = _get_float("GRASP_MAX_ABS_APPROACH_Z", 0.35)
    grasp_max_abs_closing_z: float = _get_float("GRASP_MAX_ABS_CLOSING_Z", 0.30)
    execute_single_attempt: bool = _get("EXECUTE_SINGLE_ATTEMPT", "1").lower() not in {"0", "false", "no"}

    # --- workspace safety box, in arm base frame (m) ---
    ws_x_min: float = _get_float("WS_X_MIN", 0.20)
    ws_x_max: float = _get_float("WS_X_MAX", 0.55)
    ws_y_min: float = _get_float("WS_Y_MIN", -0.40)
    ws_y_max: float = _get_float("WS_Y_MAX", 0.40)
    ws_z_min: float = _get_float("WS_Z_MIN", 0.05)
    ws_z_max: float = _get_float("WS_Z_MAX", 0.40)

    @property
    def left_calib_path(self) -> Path:
        return self.calib_dir / "cam_to_leftbase.json"

    @property
    def right_calib_path(self) -> Path:
        return self.calib_dir / "cam_to_rightbase.json"

    @property
    def intrinsics_path(self) -> Path:
        return self.calib_dir / "camera_intrinsics.json"


settings = Settings()
