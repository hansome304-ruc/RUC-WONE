from __future__ import annotations

import json
import shutil
import subprocess
import urllib.request
from http.cookiejar import CookieJar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from medicine_agentic.airbot_readonly import AirbotReadOnly
from medicine_agentic.pose_store import PoseStore


REQUIRED_P0_POSES = (
    "home",
    "task1_observe",
    "safe_transport_empty",
    "safe_transport_carton",
    "pre_pick_carton",
    "pre_place_carton",
    "place_slot_0_contact",
    "post_place",
    "recovery_high",
    "pre_pick_blister",
    "pre_insert",
)


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    blocking_for_motion: bool
    detail: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _matrix_from_calibration(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("calibration root is not an object")
    value = payload.get("cam_to_base", payload.get("matrix"))
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"expected a finite 4x4 transform, got {matrix.shape}")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError("homogeneous transform last row is invalid")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError("rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > 2e-3:
        raise ValueError(f"rotation determinant is {determinant:.6f}, expected +1")
    return matrix


def _tcp_check(path: Path, name: str) -> Check:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        calibrated = bool(payload.get("calibrated", False))
        usable = bool(payload.get("usable_for_motion", False))
        translation = payload.get("translation_m")
        if not isinstance(translation, list) or len(translation) != 3:
            raise ValueError("translation_m must contain three values")
        if not np.all(np.isfinite(np.asarray(translation, dtype=np.float64))):
            raise ValueError("translation_m contains non-finite values")
        residual = payload.get("fit", {}).get("rms_residual_mm")
        detail = (
            f"calibrated={calibrated}, usable_for_motion={usable}, "
            f"translation_m={translation}, rms_residual_mm={residual}"
        )
        return Check(
            name=name,
            ok=calibrated and usable,
            blocking_for_motion=True,
            detail=detail,
            evidence={"path": str(path), "config": payload},
        )
    except Exception as exc:
        return Check(
            name=name,
            ok=False,
            blocking_for_motion=True,
            detail=f"{type(exc).__name__}: {exc}",
            evidence={"path": str(path)},
        )


def _audio_check() -> Check:
    executable = shutil.which("arecord")
    if executable is None:
        return Check(
            name="optional_audio_input",
            ok=False,
            blocking_for_motion=False,
            detail="arecord is unavailable; acoustic verification will be skipped",
            evidence={},
        )
    try:
        result = subprocess.run(
            [executable, "-l"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        output = (result.stdout + result.stderr).strip()
        available = result.returncode == 0 and "card " in output.lower()
        return Check(
            name="optional_audio_input",
            ok=available,
            blocking_for_motion=False,
            detail=(
                "capture device found"
                if available
                else "no ALSA capture device found; acoustic verification will be skipped"
            ),
            evidence={"arecord": executable, "listing": output},
        )
    except Exception as exc:
        return Check(
            name="optional_audio_input",
            ok=False,
            blocking_for_motion=False,
            detail=f"{type(exc).__name__}: {exc}",
            evidence={"arecord": executable},
        )


def _web_console_check(url: str) -> Check:
    try:
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(CookieJar())
        )
        with opener.open(
            f"{url.rstrip('/')}/api/auth/session", timeout=2.0
        ) as response:
            authenticated = bool(json.load(response).get("authenticated"))
        with opener.open(f"{url.rstrip('/')}/api/health", timeout=2.0) as response:
            payload = json.load(response)
        ok = authenticated and bool(payload.get("ok"))
        return Check(
            name="web_console_live",
            ok=ok,
            blocking_for_motion=True,
            detail=f"authenticated={authenticated}, health_ok={payload.get('ok')}",
            evidence={"url": url, "health": payload},
        )
    except Exception as exc:
        return Check(
            name="web_console_live",
            ok=False,
            blocking_for_motion=True,
            detail=f"{type(exc).__name__}: {exc}",
            evidence={"url": url},
        )


def run_doctor(
    *,
    project_root: Path,
    live: bool = False,
    host: str = "localhost",
    left_port: int = 50051,
    right_port: int = 50053,
) -> dict[str, Any]:
    checks: list[Check] = []
    task_config_path = project_root / "configs" / "task1_box.json"
    try:
        task_config = json.loads(task_config_path.read_text(encoding="utf-8"))
        camera = task_config["camera"]
        calibration_path = Path(camera["cam_to_left_path"]).expanduser()
        matrix = _matrix_from_calibration(calibration_path)
        checks.append(
            Check(
                name="camera_to_left_base_file",
                ok=True,
                blocking_for_motion=True,
                detail="4x4 transform is finite and rigid; physical validation is still required",
                evidence={
                    "path": str(calibration_path),
                    "translation_m": matrix[:3, 3].tolist(),
                },
            )
        )
    except Exception as exc:
        task_config = {}
        camera = {}
        checks.append(
            Check(
                name="camera_to_left_base_file",
                ok=False,
                blocking_for_motion=True,
                detail=f"{type(exc).__name__}: {exc}",
                evidence={"task_config": str(task_config_path)},
            )
        )

    pose_path = project_root / "configs" / "p0_poses.json"
    try:
        pose_payload = PoseStore(pose_path).load()
        names = set(pose_payload["poses"])
        missing = [name for name in REQUIRED_P0_POSES if name not in names]
        checks.append(
            Check(
                name="paired_named_poses",
                ok=not missing,
                blocking_for_motion=True,
                detail=(
                    "all required paired poses are recorded"
                    if not missing
                    else f"{len(missing)} pose(s) still need teleoperation capture"
                ),
                evidence={
                    "path": str(pose_path),
                    "recorded": sorted(names),
                    "missing": missing,
                    "revision": pose_payload["revision"],
                    "content_sha256": pose_payload["content_sha256"],
                },
            )
        )
    except Exception as exc:
        checks.append(
            Check(
                name="paired_named_poses",
                ok=False,
                blocking_for_motion=True,
                detail=f"{type(exc).__name__}: {exc}",
                evidence={"path": str(pose_path)},
            )
        )

    checks.append(
        _tcp_check(
            project_root / "configs" / "calibration" / "left_suction_tcp.json",
            "left_suction_tcp",
        )
    )
    checks.append(
        _tcp_check(
            project_root / "configs" / "calibration" / "right_gripper_tcp.json",
            "right_gripper_tcp",
        )
    )
    checks.append(_audio_check())

    if live:
        web_url = str(camera.get("web_console_url", "http://127.0.0.1:8765"))
        checks.append(_web_console_check(web_url))
        try:
            with AirbotReadOnly(
                host=host, ports={"left": left_port, "right": right_port}
            ) as reader:
                pair = reader.capture_pair()
            checks.append(
                Check(
                    name="dual_arm_feedback_live",
                    ok=True,
                    blocking_for_motion=True,
                    detail="read-only feedback received from both arms",
                    evidence=pair,
                )
            )
        except Exception as exc:
            checks.append(
                Check(
                    name="dual_arm_feedback_live",
                    ok=False,
                    blocking_for_motion=True,
                    detail=f"{type(exc).__name__}: {exc}",
                    evidence={"host": host, "ports": [left_port, right_port]},
                )
            )

    blockers = [
        check.name
        for check in checks
        if check.blocking_for_motion and not check.ok
    ]
    return {
        "ok": not blockers,
        "mode": "live_read_only" if live else "files_only",
        "safe_to_record_poses": (
            live
            and all(
                check.ok
                for check in checks
                if check.name == "dual_arm_feedback_live"
            )
        ),
        "safe_to_execute_motion": False,
        "motion_unlock_note": (
            "The doctor never authorizes motion. Physical pose/TCP capture, "
            "transition validation, an operator at the emergency stop, and an "
            "explicit one-run execution token are separate gates."
        ),
        "blocking_checks": blockers,
        "checks": [check.to_dict() for check in checks],
    }
