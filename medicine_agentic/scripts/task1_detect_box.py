#!/usr/bin/env python3
"""Recognize one Task-1 medicine carton and produce auditable 3-D evidence.

Default mode reads the already-running web console.  It never connects to or
moves either arm, and it never changes suction state.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from medicine_agentic.detector_provider import create_detector_provider
from medicine_agentic.reference_faces import load_reference_face_bank
from medicine_agentic.task1_box import (
    WebConsoleCamera,
    draw_overlay,
    load_cam_to_left,
    load_json,
    locate_candidate,
    sample_candidate_depth,
)


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "task1_box.json",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Offline RGB image. If omitted, capture one frame from web console.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "task1" / "latest",
    )
    parser.add_argument(
        "--pixel-only",
        action="store_true",
        help="Skip live depth and base-frame localization.",
    )
    return parser


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> int:
    args = build_parser().parse_args()
    config = load_json(args.config)
    detector_cfg = dict(config["detector"])
    camera_cfg = dict(config["camera"])
    bank_cfg = config.get("reference_face_bank", {})
    if not isinstance(bank_cfg, dict):
        raise ValueError("reference_face_bank config must be an object")
    face_bank = None
    face_bank_error = None
    manifest_setting = bank_cfg.get("manifest")
    if isinstance(manifest_setting, str) and manifest_setting.strip():
        manifest_path = Path(manifest_setting).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = args.config.resolve().parent / manifest_path
        try:
            face_bank = load_reference_face_bank(manifest_path.resolve())
        except Exception as exc:
            face_bank_error = f"{type(exc).__name__}: {exc}"
    else:
        face_bank_error = "reference face bank manifest is not configured"
    detector = create_detector_provider(
        detector_cfg,
        config_dir=args.config.resolve().parent,
        face_bank=face_bank,
        face_bank_error=face_bank_error,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    camera = None
    if args.image is not None:
        rgb = _load_rgb(args.image)
        source = str(args.image.resolve())
    else:
        camera = WebConsoleCamera(
            base_url=str(camera_cfg.get("web_console_url", "http://127.0.0.1:8765")),
            camera=str(camera_cfg.get("name", "front")),
        )
        rgb = camera.capture_rgb()
        source = f"web-console:{camera.camera}"

    candidates = detector.detect(rgb)
    located = []
    intrinsics = np.asarray(camera_cfg["intrinsics"], dtype=np.float64)
    cam_to_left = None
    calibration_error = None
    if not args.pixel_only:
        try:
            cam_to_left = load_cam_to_left(camera_cfg["cam_to_left_path"])
        except Exception as exc:
            calibration_error = f"failed to load hand-eye calibration: {exc}"

    for candidate in candidates:
        depth = None
        blockers: list[str] = []
        if args.pixel_only:
            blockers.append("pixel-only mode: no 3-D localization")
        elif camera is None:
            blockers.append("offline image has no synchronized depth")
        else:
            depth, depth_errors = sample_candidate_depth(
                camera, candidate, rgb.shape[:2], detector_cfg
            )
            blockers.extend(depth_errors)
        if calibration_error is not None:
            blockers.append(calibration_error)
        located.append(
            locate_candidate(
                candidate,
                depth,
                intrinsics,
                cam_to_left,
                detector_cfg,
                inherited_blockers=blockers,
            )
        )

    # The closest valid top surface wins; score resolves nearly equal depths.
    def rank(item):
        depth = float("inf") if item.depth is None else item.depth.median_mm
        return (bool(item.blockers), depth, -item.candidate.score)

    located.sort(key=rank)
    selected = located[0] if located else None

    bgr = draw_overlay(rgb, candidates, selected, detector_cfg)
    capture_path = args.out_dir / "capture.jpg"
    overlay_path = args.out_dir / "detection_overlay.jpg"
    report_path = args.out_dir / "detection.json"
    cv2.imwrite(str(capture_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(overlay_path), bgr)

    report = {
        "ok": bool(selected and selected.ok),
        "timestamp": time.time(),
        "source": source,
        "config": str(args.config.resolve()),
        "detector": detector.status(),
        "candidate_count": len(candidates),
        "selected": None if selected is None else selected.to_dict(),
        "candidates": [item.to_dict() for item in located],
        "artifacts": {
            "capture": str(capture_path.resolve()),
            "overlay": str(overlay_path.resolve()),
        },
    }
    if selected is None:
        report["error"] = "no medicine-carton candidate passed geometric filters"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
