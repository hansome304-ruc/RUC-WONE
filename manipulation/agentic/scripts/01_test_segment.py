#!/usr/bin/env python3
"""Capture a frame from cam_front and run /segment, save an overlay PNG.

Run after 00_test_health.py passes.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from agentic_grasp.perception_client import PerceptionClient
from agentic_grasp.sensors import RealSenseRGBD
from agentic_grasp.skills import GraspSkills
from agentic_grasp.config import settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="a bottle")
    parser.add_argument("--cam", default=settings.cam_front_serial)
    parser.add_argument("--out", default="debug_out/segment_overlay.png")
    parser.add_argument(
        "--save-rgb",
        action="store_true",
        help="also save the raw RGB frame next to the overlay",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cam = RealSenseRGBD(args.cam)
    try:
        frame = cam.capture()
    finally:
        cam.close()
    print(f"captured frame: rgb={frame.rgb.shape} depth={frame.depth_mm.shape} K[0,0]={frame.K[0,0]:.1f}")

    if args.save_rgb:
        import cv2
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out.replace(".png", "_rgb.png"), frame.rgb[..., ::-1])

    client = PerceptionClient()
    seg = client.segment(frame.rgb, prompt=args.prompt)
    print(f"segment: bbox={seg.bbox} score={seg.score:.3f} mask_px={int(seg.mask.sum())}")

    GraspSkills._save_segmentation_overlay(frame.rgb, seg, args.out)
    print(f"overlay saved → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
