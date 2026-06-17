#!/usr/bin/env python3
"""Diagnostic script to verify hand-eye calibration by moving along the PHYSICAL base axes.

This script commands the robot arm to move along the X, Y, or Z axis of its 
ACTUAL physical base coordinate frame (completely independent of calibration).

We can then compare this physical motion with the calibrated axes drawn in calib_overlay.png.
"""
from __future__ import annotations

import argparse
import logging
import sys
import numpy as np

from agentic_grasp.config import settings
from agentic_grasp.transforms import pose_for_airbot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--distance", type=float, default=0.10, help="Distance to move in metres (default: 0.10m)")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="x", 
                        help="Physical base axis to move along: x (straight forward), y (straight left), z (straight up)")
    parser.add_argument("--speed", default="SLOW", choices=["SLOW", "DEFAULT", "FAST"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    # 1. Connect to the robot arm
    from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile
    port = settings.left_arm_port if args.arm == "left" else settings.right_arm_port
    log.info("Connecting to %s arm on port %d...", args.arm, port)
    
    arm = AIRBOTPlay(url="localhost", port=port)
    if not arm.connect():
        log.error("Failed to connect to %s arm gRPC!", args.arm)
        return 1

    arm.switch_mode(RobotMode.PLANNING_POS)
    arm.set_speed_profile(getattr(SpeedProfile, args.speed))

    # 2. Get current end-effector pose in base frame
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

    # 3. Compute the direction vector of the PHYSICAL base axis
    # This is pure physical motion, completely independent of calibration!
    # x_axis = [1, 0, 0] (straight forward)
    # y_axis = [0, 1, 0] (straight left)
    # z_axis = [0, 0, 1] (straight up)
    direction_in_actual_base = np.zeros(3)
    axis_map = {"x": 0, "y": 1, "z": 2}
    direction_in_actual_base[axis_map[args.axis]] = 1.0
    
    log.info("Direction in ACTUAL physical base frame: %s", direction_in_actual_base.tolist())

    # 4. Compute the target pose
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
    print(f"🚨 WARNING: The {args.arm} arm is about to move {args.distance*100:.1f}cm along the PHYSICAL {args.axis.upper()}-axis.")
    print("Please keep your hand near the emergency stop (E-stop) or spacebar!")
    print("="*60)
    input("Press Enter to start the motion...")

    log.info("Moving to target: xyz=%s...", target_xyz.round(3).tolist())
    ok = arm.move_to_cart_pose(target_pose, blocking=True)
    
    if ok:
        log.info("Motion completed successfully!")
        print("\n" + "="*60)
        print("🤔 OBSERVATION TIME (观察与判定方法):")
        print(f"The robot arm just moved physically straight along its physical {args.axis.upper()}-axis.")
        print("Now, open 'debug_out/calib_overlay.png' and check the corresponding arrow:")
        if args.axis == "x":
            print("  - Compare the physical forward motion with the RED X-axis arrow in the image.")
            print("  - If they point in the SAME direction in the camera frame, calibration is CORRECT.")
            print("  - If they point in DIFFERENT directions (e.g., red arrow is tilted 45 deg), calibration is WRONG!")
        elif args.axis == "y":
            print("  - Compare the physical left motion with the GREEN Y-axis arrow in the image.")
            print("  - If they point in the SAME direction, calibration is CORRECT.")
            print("  - If they point in DIFFERENT directions, calibration is WRONG!")
        print("="*60)
    else:
        log.error("Motion planning failed! The target might be kinematically unreachable.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
