#!/usr/bin/env python3
import numpy as np
from agentic_grasp.config import settings
from agentic_grasp.sensors import load_calib

def main():
    calib = load_calib(settings.left_calib_path, settings.right_calib_path, settings.intrinsics_path)
    K = calib.K_calib
    
    # Project left base origin [0, 0, 0]
    p_in_cam_l = (np.linalg.inv(calib.T_cam_to_left) @ np.array([0, 0, 0, 1.0]))[:3]
    u_l = K[0, 0] * p_in_cam_l[0] / p_in_cam_l[2] + K[0, 2]
    v_l = K[1, 1] * p_in_cam_l[1] / p_in_cam_l[2] + K[1, 2]
    
    # Project right base origin [0, 0, 0]
    p_in_cam_r = (np.linalg.inv(calib.T_cam_to_right) @ np.array([0, 0, 0, 1.0]))[:3]
    u_r = K[0, 0] * p_in_cam_r[0] / p_in_cam_r[2] + K[0, 2]
    v_r = K[1, 1] * p_in_cam_r[1] / p_in_cam_r[2] + K[1, 2]
    
    print(f"Left base in camera frame: {p_in_cam_l.round(3).tolist()}")
    print(f"Left base pixel: ({int(round(u_l))}, {int(round(v_l))})")
    
    print(f"Right base in camera frame: {p_in_cam_r.round(3).tolist()}")
    print(f"Right base pixel: ({int(round(u_r))}, {int(round(v_r))})")

if __name__ == "__main__":
    main()
