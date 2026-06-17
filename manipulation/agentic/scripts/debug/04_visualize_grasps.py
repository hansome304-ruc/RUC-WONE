#!/usr/bin/env python3
"""Debug script to visualize the predicted grasps in 3D and 2D.

Generates:
  1. debug_out/grasp_visualize_2d.png: Projects 3D grasp poses back to 2D image.
  2. debug_out/grasp_pointcloud_debug.png (Optional): Visualizes depth point cloud.
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
from agentic_grasp.perception_client import PerceptionClient


log = logging.getLogger(__name__)


def project_point(K: np.ndarray, p: np.ndarray) -> tuple[int, int] | None:
    """Project a 3D point in camera frame to pixel coordinates."""
    if p[2] <= 0:
        return None
    u = K[0, 0] * p[0] / p[2] + K[0, 2]
    v = K[1, 1] * p[1] / p[2] + K[1, 2]
    return int(round(u)), int(round(v))


def draw_grasp_pose(img: np.ndarray, K: np.ndarray, T_cam: np.ndarray, score: float, width: float):
    """Draw a 3D grasp pose on the 2D image as a coordinate frame and gripper fingers."""
    # 1. Draw coordinate axis at grasp center
    center_3d = T_cam[:3, 3]
    center_2d = project_point(K, center_3d)
    if center_2d is None:
        return

    # Draw small circle at grasp center
    cv2.circle(img, center_2d, 4, (0, 255, 255), -1)

    # Draw coordinate frame (x=red, y=green, z=blue)
    axis_len = 0.04  # 4 cm
    for axis_idx, color in [(0, (0, 0, 255)), (1, (0, 255, 0)), (2, (255, 0, 0))]:
        tip_in_grasp = np.zeros(4)
        tip_in_grasp[axis_idx] = axis_len
        tip_in_grasp[3] = 1.0
        tip_3d = (T_cam @ tip_in_grasp)[:3]
        tip_2d = project_point(K, tip_3d)
        if tip_2d is not None:
            cv2.line(img, center_2d, tip_2d, color, 2)

    # 2. Draw gripper fingers (along the local y-axis, which is the closing direction)
    # Gripper fingers are placed at +/- (width/2) along the local y-axis
    half_w = width / 2.0
    finger_l = np.array([0.0, -half_w, 0.0, 1.0])
    finger_r = np.array([0.0, half_w, 0.0, 1.0])
    
    fl_3d = (T_cam @ finger_l)[:3]
    fr_3d = (T_cam @ finger_r)[:3]
    
    fl_2d = project_point(K, fl_3d)
    fr_2d = project_point(K, fr_3d)
    
    if fl_2d is not None and fr_2d is not None:
        # Draw the gripper bar connecting the fingers
        cv2.line(img, fl_2d, fr_2d, (255, 0, 255), 2)
        # Draw fingers extending forward along local z-axis (approach direction)
        finger_len = 0.02  # 2 cm finger length
        for f_3d, f_2d in [(fl_3d, fl_2d), (fr_3d, fr_2d)]:
            tip_3d = f_3d + T_cam[:3, 2] * finger_len
            tip_2d = project_point(K, tip_3d)
            if tip_2d is not None:
                cv2.line(img, f_2d, tip_2d, (255, 0, 255), 2)
                
    # Draw score text
    cv2.putText(img, f"{score:.2f}", (center_2d[0] + 5, center_2d[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="a bottle")
    parser.add_argument("--cam", default=settings.cam_front_serial)
    parser.add_argument("--out", default="debug_out/grasp_visualize_2d.png")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    # 1. Capture Frame from RealSense
    log.info("Connecting to RealSense camera %s...", args.cam)
    cam = RealSenseRGBD(args.cam)
    try:
        frame = cam.capture()
    finally:
        cam.close()
    log.info("Captured image %dx%d", frame.rgb.shape[1], frame.rgb.shape[0])

    # 2. Call Perception Server
    client = PerceptionClient()
    log.info("Calling /segment with prompt: %r...", args.prompt)
    seg = client.segment(frame.rgb, prompt=args.prompt)
    log.info("Segmented object: bbox=%s, score=%.2f, mask_px=%d", 
             seg.bbox, seg.score, int(seg.mask.sum()))
    
    if int(seg.mask.sum()) < 100:
        log.error("Mask is too small or empty! Camera might be blocked or prompt is wrong.")
        return 1

    # Check depth values inside the mask
    masked_depth = frame.depth_mm[seg.mask.astype(bool)]
    valid_depth_mask = masked_depth > 0
    if not valid_depth_mask.any():
        log.error("CRITICAL: All depth values inside the segmented mask are ZERO!")
        log.error("This means the depth camera is blind to this object (too close, reflective, or hardware issue).")
        return 1
    
    min_d, max_d, mean_d = masked_depth[valid_depth_mask].min(), masked_depth.max(), masked_depth[valid_depth_mask].mean()
    log.info("Depth stats inside mask (mm): min=%d, max=%d, mean=%.1f, valid_ratio=%.1f%%",
             min_d, max_d, mean_d, (valid_depth_mask.sum() / len(masked_depth)) * 100)

    log.info("Calling /grasp to predict grasp poses...")
    grasps = client.grasps(frame.rgb, frame.depth_mm, frame.K, seg.mask, endpoint="/grasp", top_k=args.top_k)
    log.info("Received %d grasp candidates from server", len(grasps))

    if not grasps:
        log.warning("No grasp candidates returned by the server!")
        return 1

    # 3. Draw visualization
    bgr = frame.rgb[..., ::-1].copy()
    
    # Draw segmentation mask overlay
    overlay = bgr.copy()
    overlay[seg.mask.astype(bool)] = [0, 0, 100]  # Translucent red for mask
    bgr = cv2.addWeighted(bgr, 0.7, overlay, 0.3, 0)
    
    # Draw bounding box
    cv2.rectangle(bgr, (seg.bbox[0], seg.bbox[1]), (seg.bbox[2], seg.bbox[3]), (0, 255, 0), 2)
    
    # Draw all predicted grasps
    for i, g in enumerate(grasps):
        if i == 0:
            # Highlight the top-1 grasp with thicker lines and a distinct color
            draw_grasp_pose(bgr, frame.K, g.T_cam, g.score, g.width)
            center_2d = project_point(frame.K, g.T_cam[:3, 3])
            if center_2d is not None:
                cv2.circle(bgr, center_2d, 8, (0, 0, 255), 2) # Red circle for top-1
                cv2.putText(bgr, "TOP-1", (center_2d[0] - 20, center_2d[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            # Draw other grasps very faintly so they don't clutter the image
            # We can temporarily blend a faint version
            temp_img = bgr.copy()
            draw_grasp_pose(temp_img, frame.K, g.T_cam, g.score, g.width)
            bgr = cv2.addWeighted(bgr, 0.85, temp_img, 0.15, 0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, bgr)
    log.info("SUCCESS! Debug visualization saved to: %s", args.out)
    log.info("Please open this image to check if the 3D grasp frames align with the physical object.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
