#!/usr/bin/env python3
"""Sanity-check calibration: project arm base origin and a few axis tips
back into the RGB image and see if they land where the arm bases physically are.

Requires ~/calib/result/ to exist.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from agentic_grasp.config import settings
from agentic_grasp.sensors import RealSenseRGBD, load_calib


def project(K: np.ndarray, T_cam_to_base: np.ndarray, p_in_base: np.ndarray):
    p_in_cam = (np.linalg.inv(T_cam_to_base) @ np.array([*p_in_base, 1.0]))[:3]
    if p_in_cam[2] <= 0:
        return None
    u = K[0, 0] * p_in_cam[0] / p_in_cam[2] + K[0, 2]
    v = K[1, 1] * p_in_cam[1] / p_in_cam[2] + K[1, 2]
    return int(round(u)), int(round(v))


def draw_frame(img, K, T_cam_to_base, label, colour, length=0.10):
    """Draw the base_link xyz frame as 3 coloured arrows."""
    origin = project(K, T_cam_to_base, np.zeros(3))
    if origin is None:
        return img
    cv2.circle(img, origin, 6, colour, -1)
    cv2.putText(img, label, (origin[0] + 8, origin[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    for axis_idx, axis_colour, axis_label in [
        (0, (0, 0, 255), "x"),
        (1, (0, 255, 0), "y"),
        (2, (255, 0, 0), "z"),
    ]:
        tip3d = np.zeros(3); tip3d[axis_idx] = length
        tip = project(K, T_cam_to_base, tip3d)
        if tip is None:
            continue
        cv2.arrowedLine(img, origin, tip, axis_colour, 2, tipLength=0.2)
        cv2.putText(img, axis_label, tip, cv2.FONT_HERSHEY_SIMPLEX, 0.5, axis_colour, 2)
    return img


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", default=settings.cam_front_serial)
    parser.add_argument("--out", default="debug_out/calib_overlay.png")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    calib = load_calib(settings.left_calib_path, settings.right_calib_path, settings.intrinsics_path)
    print("calib loaded ok")
    print("T_cam_to_left[:3,3] =", calib.T_cam_to_left[:3, 3])
    print("T_cam_to_right[:3,3] =", calib.T_cam_to_right[:3, 3])

    cam = RealSenseRGBD(args.cam)
    try:
        frame = cam.capture()
    finally:
        cam.close()

    bgr = frame.rgb[..., ::-1].copy()
    bgr = draw_frame(bgr, calib.K_calib, calib.T_cam_to_left, "L_base", (0, 255, 255))
    bgr = draw_frame(bgr, calib.K_calib, calib.T_cam_to_right, "R_base", (255, 255, 0))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, bgr)
    print(f"overlay saved → {args.out}")
    print("check: yellow dot on left arm base, cyan dot on right arm base, x→fwd y→left z→up")
    return 0


if __name__ == "__main__":
    sys.exit(main())
