#!/usr/bin/env python3
"""Script to move left arm to HOME and capture camera view and grasp predictions."""
from __future__ import annotations

import logging
import sys
import time
import importlib.util
from pathlib import Path

import cv2
import numpy as np

from agentic_grasp.skills import GraspSkills
from agentic_grasp.config import settings

log = logging.getLogger(__name__)


def load_visualize_helpers():
    helper_path = Path(__file__).with_name("04_visualize_grasps.py")
    spec = importlib.util.spec_from_file_location("visualize_grasps_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper module: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.draw_grasp_pose, module.project_point


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    log.info("Initializing GraspSkills (connecting to left arm)...")
    skills = GraspSkills(
        cam_serial=settings.cam_front_serial,
        connect_arms=True,
        speed="SLOW",
        require_calib=True,
    )
    
    try:
        log.info("Moving left arm to HOME...")
        skills.home(dry_run=False)
        log.info("Arm is at HOME. Waiting 2 seconds for camera exposure to stabilize...")
        time.sleep(2.0)
        
        log.info("Capturing frame...")
        frame = skills.capture()
        
        # Save raw RGB image
        rgb_path = "debug_out/home_view_rgb.png"
        cv2.imwrite(rgb_path, frame.rgb[..., ::-1])
        log.info("Saved raw RGB image to %s", rgb_path)
        
        log.info("Calling /segment with prompt 'a bottle'...")
        seg = skills.client.segment(frame.rgb, prompt="a bottle")
        log.info("Segmented object: bbox=%s, score=%.2f, mask_px=%d", 
                 seg.bbox, seg.score, int(seg.mask.sum()))
        
        if int(seg.mask.sum()) < 100:
            log.error("Mask is too small!")
            return 1
            
        log.info("Calling /grasp to predict grasp poses...")
        grasps = skills.client.grasps(frame.rgb, frame.depth_mm, frame.K, seg.mask, endpoint="/grasp", top_k=10)
        log.info("Received %d grasp candidates from server", len(grasps))
        
        # Draw visualization
        bgr = frame.rgb[..., ::-1].copy()
        
        # Draw segmentation mask overlay
        overlay = bgr.copy()
        overlay[seg.mask.astype(bool)] = [0, 0, 100]
        bgr = cv2.addWeighted(bgr, 0.7, overlay, 0.3, 0)
        
        # Draw bounding box
        cv2.rectangle(bgr, (seg.bbox[0], seg.bbox[1]), (seg.bbox[2], seg.bbox[3]), (0, 255, 0), 2)
        
        # Draw all predicted grasps.
        draw_grasp_pose, project_point = load_visualize_helpers()
        
        if self_calib := skills.calib:
            T_cam_to_base = self_calib.T_cam_to_left
        else:
            T_cam_to_base = np.eye(4)
            
        for i, g in enumerate(grasps):
            T_in_base = T_cam_to_base @ g.T_cam
            T_eef = T_in_base # without tcp offset for visualization of raw grasp
            
            if i == 0:
                draw_grasp_pose(bgr, frame.K, g.T_cam, g.score, g.width)
                center_2d = project_point(frame.K, g.T_cam[:3, 3])
                if center_2d is not None:
                    cv2.circle(bgr, center_2d, 8, (0, 0, 255), 2)
                    cv2.putText(bgr, f"TOP-1 (xyz in base: {T_in_base[:3,3].round(3).tolist()})", 
                                (center_2d[0] - 50, center_2d[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            else:
                temp_img = bgr.copy()
                draw_grasp_pose(temp_img, frame.K, g.T_cam, g.score, g.width)
                bgr = cv2.addWeighted(bgr, 0.85, temp_img, 0.15, 0)
                
        out_path = "debug_out/home_grasp_visualize.png"
        cv2.imwrite(out_path, bgr)
        log.info("Saved visualization to %s", out_path)
        
    finally:
        skills.close()
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
