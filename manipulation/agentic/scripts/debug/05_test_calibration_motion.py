#!/usr/bin/env python3
"""Diagnostic script to verify hand-eye calibration using physical motion.

This script commands the specified robot arm to move exactly 10cm (0.1m)
along the CAMERA's optical axis (Z-axis, straight away from the camera lens)
using the hand-eye calibration matrix.

If the calibration is correct:
  - The arm should move in a straight line directly away from the camera lens.
If the calibration is wrong (e.g., rotated by 45 degrees):
  - The arm will move diagonally (skewed by ~45 degrees) relative to the camera lens.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import numpy as np

from agentic_grasp.config import settings
from agentic_grasp.sensors import load_calib
from agentic_grasp.transforms import pose_for_airbot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--distance", type=float, default=0.10, help="Distance to move in metres (default: 0.10m)")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="z", 
                        help="Camera axis to move along: x (horizontal in image), y (vertical in image), z (depth)")
    parser.add_argument("--speed", default="SLOW", choices=["SLOW", "DEFAULT", "FAST"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    # 1. Load calibration
    try:
        calib = load_calib(settings.left_calib_path, settings.right_calib_path, settings.intrinsics_path)
        log.info("Calibration loaded successfully.")
    except FileNotFoundError as exc:
        log.error("Failed to load calibration: %s", exc)
        return 1

    # 2. Connect to the robot arm
    from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile
    port = settings.left_arm_port if args.arm == "left" else settings.right_arm_port
    log.info("Connecting to %s arm on port %d...", args.arm, port)
    
    arm = AIRBOTPlay(url="localhost", port=port)
    if not arm.connect():
        log.error("Failed to connect to %s arm gRPC!", args.arm)
        return 1

    arm.switch_mode(RobotMode.PLANNING_POS)
    arm.set_speed_profile(getattr(SpeedProfile, args.speed))

    # 3. Get current end-effector pose in base frame
    # AIRBOTPlay.get_current_pose() or get_end_pose() returns [xyz, quat_xyzw]
    # Based on AttributeError, the correct method name is get_end_pose
    curr_pose = arm.get_end_pose()
    if not curr_pose:
        log.error("Failed to get current arm pose!")
        return 1
    
    curr_xyz, curr_quat = curr_pose
    log.info("Current pose in base frame: xyz=%s, quat=%s", curr_xyz, curr_quat)

    # Convert current pose to 4x4 matrix T_eef_in_base
    from agentic_grasp.transforms import R_from_quat_xyzw, make_T
    R_curr = R_from_quat_xyzw(curr_quat)
    T_curr = make_T(R_curr, curr_xyz)

    # 4. Compute the movement direction along the CAMERA's specified axis
    # Camera X-axis = T_cam_to_base[:, 0] (horizontal in image)
    # Camera Y-axis = T_cam_to_base[:, 1] (vertical in image)
    # Camera Z-axis = T_cam_to_base[:, 2] (depth / optical axis)
    T_cam_to_base = calib.T_cam_to_left if args.arm == "left" else calib.T_cam_to_right
    
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_idx = axis_map[args.axis]
    direction_in_base = T_cam_to_base[:3, axis_idx]
    
    # Normalize the direction vector
    direction_in_base = direction_in_base / np.linalg.norm(direction_in_base)
    log.info("Camera %s-axis direction in base frame: %s", args.axis.upper(), direction_in_base.tolist())

    # 5. Compute the target pose
    # Translate the current end-effector position along the camera's specified axis direction
    target_xyz = np.array(curr_xyz) + direction_in_base * args.distance
    T_target = make_T(R_curr, target_xyz) # Keep the same orientation
    target_pose = pose_for_airbot(T_target)

    # Safety workspace check
    x, y, z = target_xyz
    c = settings
    if not (c.ws_x_min <= x <= c.ws_x_max and c.ws_y_min <= y <= c.ws_y_max and c.ws_z_min <= z <= c.ws_z_max):
        log.error("SAFETY BLOCK: Target position xyz=(%.3f, %.3f, %.3f) is outside safety workspace!", x, y, z)
        return 1

    print("\n" + "="*60)
    print(f"🚨 WARNING: The {args.arm} arm is about to move {args.distance*100:.1f}cm along the CAMERA {args.axis.upper()}-axis.")
    print("Please keep your hand near the emergency stop (E-stop) or spacebar!")
    print("="*60)
    input("Press Enter to start the motion...")

    log.info("Moving to target: xyz=%s...", target_xyz.round(3).tolist())
    ok = arm.move_to_cart_pose(target_pose, blocking=True)
    
    if ok:
        log.info("Motion completed successfully!")
        print("\n" + "="*60)
        print("🤔 OBSERVATION TIME (观察与判定方法):")
        if args.axis == "x":
            print("Look at the CAMERA'S LIVE IMAGE FEED (看相机的实时图像画面):")
            print("Did the robot gripper move PERFECTLY HORIZONTALLY (left <-> right) across the screen?")
            print("  - [YES] -> Your hand-eye calibration rotation is CORRECT.")
            print("  - [NO]  -> Your calibration is WRONG (it moved diagonally up/down on screen).")
        elif args.axis == "y":
            print("Look at the CAMERA'S LIVE IMAGE FEED (看相机的实时图像画面):")
            print("Did the robot gripper move PERFECTLY VERTICALLY (up <-> down) across the screen?")
            print("  - [YES] -> Your hand-eye calibration rotation is CORRECT.")
            print("  - [NO]  -> Your calibration is WRONG (it moved diagonally left/right on screen).")
        else:
            print("Did the robot arm move in a straight line DIRECTLY AWAY from the camera lens?")
            print("  - [YES] -> Your hand-eye calibration rotation is CORRECT.")
            print("  - [NO]  -> Your hand-eye calibration rotation is WRONG.")
        print("="*60)
    else:
        log.error("Motion planning failed! The target might be kinematically unreachable.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
