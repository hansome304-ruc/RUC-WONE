#!/usr/bin/env python3
"""End-to-end pipeline against the perception server.

Defaults to the dummy /grasp_dummy endpoint so you can validate the wiring
without ZeroGrasp installed. Pass --endpoint /grasp once the real model is up.

`--dry-run` prints the planned poses and never sends a motion command.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

from agentic_grasp.skills import GraspSkills
from agentic_grasp.config import settings
from agentic_grasp.grasp_profiles import PROFILES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="a bottle")
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--endpoint", default="/grasp_dummy",
                        help="/grasp_dummy for the geometric fallback, /grasp for ZeroGrasp")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cam", default=settings.cam_front_serial)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--strategy", choices=sorted(PROFILES), default="default",
                        help="Grasp selection profile. side-bottle forces a horizontal side-entry bottle grasp.")
    parser.add_argument("--pre-grasp-back", type=float, default=0.08,
                        help="Pre-grasp retreat distance along approach axis, in metres")
    parser.add_argument("--side-overdrive", type=float, default=None,
                        help="Override extra forward push for --strategy side-bottle, in metres")
    parser.add_argument("--lift-height", type=float, default=0.10,
                        help="Vertical lift distance after closing, in metres")
    parser.add_argument("--multi-attempt-real", action="store_true",
                        help="Allow real robot runs to try multiple candidates after a failed near-object move")
    parser.add_argument("--speed", default="SLOW", choices=["SLOW", "DEFAULT", "FAST"])
    parser.add_argument("--record-video", action="store_true",
                        help="Record the front camera view during the run to debug_out/grasp_run_*.mp4")
    parser.add_argument("--record-out", default=None,
                        help="Optional mp4 path for --record-video")
    parser.add_argument("--record-fps", type=float, default=10.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    skills = GraspSkills(
        cam_serial=args.cam,
        connect_arms=not args.dry_run,
        speed=args.speed,
        require_calib=not args.dry_run,
    )

    def _on_sigint(*_):
        logging.warning("Ctrl-C, going limp")
        skills.estop()
        # NB: don't call skills.close() here — we want the arms held at their
        # current pose by GRAVITY_COMP, not forcefully disconnected mid-grasp.
        sys.exit(130)

    signal.signal(signal.SIGINT, _on_sigint)

    try:
        if args.record_video:
            record_out = Path(args.record_out) if args.record_out else Path(
                "debug_out") / f"grasp_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            skills.start_video_recording(record_out, fps=args.record_fps)
        skills.home(dry_run=args.dry_run)
        ok, attempt = skills.grasp_object(
            prompt=args.prompt,
            arm=args.arm,
            endpoint=args.endpoint,
            max_attempts=args.max_attempts,
            pre_grasp_back=args.pre_grasp_back,
            lift_height=args.lift_height,
            execute_single_attempt=not args.multi_attempt_real,
            strategy=args.strategy,
            side_overdrive=args.side_overdrive,
            dry_run=args.dry_run,
        )
        print(f"grasp success={ok}")
        if attempt:
            print(f"executed grasp xyz={attempt.T_eef_in_base[:3,3].round(3).tolist()} "
                  f"width={attempt.grasp_in_cam.width:.3f} score={attempt.grasp_in_cam.score:.2f}")
        return 0 if ok else 1
    finally:
        # Ensure RealSense + AIRBOT gRPC streams are released even on success,
        # so the script exits promptly and the next run doesn't hit a stale
        # `Another request is in progress` on the server side. We do NOT change
        # the arm mode here — the arms keep whatever pose / brake state the
        # last command left them in.
        try:
            skills.stop_video_recording()
            skills.close()
        except Exception as exc:
            logging.warning("skills.close() raised at shutdown: %s", exc)


if __name__ == "__main__":
    sys.exit(main())
