#!/usr/bin/env python3
import numpy as np
from agentic_grasp.config import settings
from agentic_grasp.sensors import load_calib

def project(K: np.ndarray, T_cam_to_base: np.ndarray, p_in_base: np.ndarray):
    p_in_cam = (np.linalg.inv(T_cam_to_base) @ np.array([*p_in_base, 1.0]))[:3]
    u = K[0, 0] * p_in_cam[0] / p_in_cam[2] + K[0, 2]
    v = K[1, 1] * p_in_cam[1] / p_in_cam[2] + K[1, 2]
    return int(round(u)), int(round(v))

def main():
    calib = load_calib(settings.left_calib_path, settings.right_calib_path, settings.intrinsics_path)
    K = calib.K_calib
    
    for arm, T in [("Left", calib.T_cam_to_left), ("Right", calib.T_cam_to_right)]:
        print(f"\n=== {arm} Arm Base Frame Projections ===")
        origin = project(K, T, np.zeros(3))
        print(f"Origin (0,0,0) pixel: {origin}")
        
        # X axis (red)
        x_tip = project(K, T, np.array([0.10, 0, 0]))
        print(f"X-axis (forward 10cm) pixel: {x_tip} (vector: {x_tip[0]-origin[0]}, {x_tip[1]-origin[1]})")
        
        # Y axis (green)
        y_tip = project(K, T, np.array([0, 0.10, 0]))
        print(f"Y-axis (left 10cm) pixel: {y_tip} (vector: {y_tip[0]-origin[0]}, {y_tip[1]-origin[1]})")
        
        # Z axis (blue)
        z_tip = project(K, T, np.array([0, 0, 0.10]))
        print(f"Z-axis (up 10cm) pixel: {z_tip} (vector: {z_tip[0]-origin[0]}, {z_tip[1]-origin[1]})")

if __name__ == "__main__":
    main()
