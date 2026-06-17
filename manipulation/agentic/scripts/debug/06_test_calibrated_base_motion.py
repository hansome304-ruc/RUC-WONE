#!/usr/bin/env python3
"""Diagnostic script to verify hand-eye calibration by moving along the CALIBRATED base axes.

This script commands the robot arm to move along the X, Y, or Z axis of the 
CALIBRATED base coordinate frame (the one drawn in calib_overlay.png).

If the calibration is correct:
  - Moving along calibrated X should make the arm move physically straight forward.
  - Moving along calibrated Y should make the arm move physically straight left.
If the calibration is wrong (e.g., rotated by 45 degrees):
  - Moving along calibrated X will make the arm move physically at a 45-degree angle (diagonally).
"""
from __future__ import annotations

import argparse
import logging
import sys
import numpy as np

from agentic_grasp.config import settings
from agentic_grasp.sensors import load_calib
from agentic_grasp.transforms import pose_for_airbot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--distance", type=float, default=0.10, help="Distance to move in metres (default: 0.10m)")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="x", 
                        help="Calibrated base axis to move along: x (red arrow), y (green arrow), z (blue arrow)")
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
    curr_pose = arm.get_end_pose()
    if not curr_pose:
        log.error("Failed to get current arm pose!")
        return 1
    
    curr_xyz, curr_quat = curr_pose
    log.info("Current physical pose: xyz=%s, quat=%s", curr_xyz, curr_quat)

    # Convert current pose to 4x4 matrix T_eef_in_base
    from agentic_grasp.transforms import R_from_quat_xyzw, make_T
    R_curr = R_from_quat_xyzw(curr_quat)
    T_curr = make_T(R_curr, curr_xyz)

    # 4. Compute the direction vector of the CALIBRATED base axis
    # The hand-eye calibration matrix T_cam_to_base defines the calibrated base frame relative to the camera.
    # To move along the calibrated base's axis, we define the direction in the calibrated base frame:
    # x_axis = [1, 0, 0], y_axis = [0, 1, 0], z_axis = [0, 0, 1]
    #
    # Since the robot's physical controller only understands the ACTUAL physical base frame,
    # we must project the calibrated axis onto the actual physical base frame.
    #
    # If the calibration is wrong (rotated), the calibrated axis will be rotated relative to the actual base.
    # The rotation from calibrated base to actual base is defined by the rotation part of T_cam_to_base.
    T_cam_to_base = calib.T_cam_to_left if args.arm == "left" else calib.T_cam_to_right
    R_calib = T_cam_to_base[:3, :3]
    
    # Define unit vector in calibrated base frame
    unit_vector = np.zeros(3)
    axis_map = {"x": 0, "y": 1, "z": 2}
    unit_vector[axis_map[args.axis]] = 1.0
    
    # Transform this vector to the ACTUAL base frame using the calibration rotation
    # Note: If calibration is correct, R_calib @ unit_vector should align with the physical axes.
    # If calibration is wrong (e.g. rotated by 45 deg), this will rotate the motion by 45 deg physically!
    direction_in_actual_base = R_calib @ unit_vector
    direction_in_actual_base = direction_in_actual_base / np.linalg.norm(direction_in_actual_base)
    
    log.info("Direction in ACTUAL physical base frame: %s", direction_in_actual_base.tolist())

    # 5. Compute the target pose
    target_xyz = np.array(curr_xyz) + direction_in_actual_base * args.distance
    T_target = make_T(R_curr, target_xyz)
    target_pose = pose_for_airbot(T_target)

    # Safety workspace check
    x, y, z = target_xyz
    c = settings
    if not (c.ws_x_min <= x <= c.ws_x_max and c.ws_y_min <= y <= c.ws_y_max and c.ws_z_min <= z <= c.ws_z_max):
        log.error("SAFETY BLOCK: Target position xyz=(%.3f, %.3f, %.3f) is outside safety workspace!", x, y, z)
        return 1

    print("\n" + "="*60)
    print(f"🚨 WARNING: The {args.arm} arm is about to move {args.distance*100:.1f}cm along the CALIBRATED {args.axis.upper()}-axis.")
    print("Please keep your hand near the emergency stop (E-stop) or spacebar!")
    print("="*60)
    input("Press Enter to start the motion...")

    log.info("Moving to target: xyz=%s...", target_xyz.round(3).tolist())
    ok = arm.move_to_cart_pose(target_pose, blocking=True)
    
    if ok:
        log.info("Motion completed successfully!")
        print("\n" + "="*60)
        print("🤔 OBSERVATION TIME (观察与判定方法):")
        print(f"Did the robot arm move physically straight along the robot's physical {args.axis.upper()}-axis?")
        if args.axis == "x":
            print("  - [YES] -> It moved physically STRAIGHT FORWARD. Calibration is CORRECT.")
            print("  - [NO]  -> It moved DIAGONALLY (e.g., skewed by 45 degrees). Calibration is WRONG!")
        elif args.axis == "y":
            print("  - [YES] -> It moved physically STRAIGHT LEFT/RIGHT. Calibration is CORRECT.")
            print("  - [NO]  -> It moved DIAGONALLY. Calibration is WRONG!")
        print("="*60)
    else:
        log.error("Motion planning failed! The target might be kinematically unreachable.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
