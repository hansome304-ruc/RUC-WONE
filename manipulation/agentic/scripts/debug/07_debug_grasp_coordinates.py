#!/usr/bin/env python3
import numpy as np
from agentic_grasp.config import settings
from agentic_grasp.sensors import RealSenseRGBD, load_calib
from agentic_grasp.perception_client import PerceptionClient
from agentic_grasp.transforms import apply_tcp_offset, pose_for_airbot

def main():
    calib = load_calib(settings.left_calib_path, settings.right_calib_path, settings.intrinsics_path)
    cam = RealSenseRGBD(settings.cam_front_serial)
    try:
        frame = cam.capture()
    finally:
        cam.close()
        
    client = PerceptionClient()
    seg = client.segment(frame.rgb, prompt="a bottle")
    print(f"Segmented bbox: {seg.bbox}, score: {seg.score:.2f}, mask_px: {int(seg.mask.sum())}")
    
    grasps = client.grasps(frame.rgb, frame.depth_mm, frame.K, seg.mask, endpoint="/grasp", top_k=5)
    print(f"Received {len(grasps)} grasps.")
    
    T_cam_to_left = calib.T_cam_to_left
    print("\nT_cam_to_left:")
    print(T_cam_to_left)
    
    for i, g in enumerate(grasps):
        print(f"\n--- Grasp {i+1} (Score: {g.score:.2f}) ---")
        t_cam = g.T_cam[:3, 3]
        print(f"t_cam (translation in camera frame): {t_cam.round(3).tolist()}")
        
        T_in_base = T_cam_to_left @ g.T_cam
        t_base = T_in_base[:3, 3]
        print(f"t_base (translation in left arm base frame, before TCP offset): {t_base.round(3).tolist()}")
        
        T_eef = apply_tcp_offset(T_in_base, settings.gripper_tcp_offset)
        t_eef = T_eef[:3, 3]
        print(f"t_eef (translation in left arm base frame, after TCP offset): {t_eef.round(3).tolist()}")
        
        # Check approach vector (z-axis of T_eef)
        z_axis_base = T_eef[:3, 2]
        print(f"Approach vector in base frame (z-axis): {z_axis_base.round(3).tolist()}")

if __name__ == "__main__":
    main()
