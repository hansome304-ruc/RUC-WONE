"""ZeroGrasp inference wrapper.

Adapts demo.py's pipeline (file-based) to memory-based inputs and converts
GraspNet-convention grasp poses to the convention expected by
`agentic_grasp/perception_client.py`.

Conversion rationale
--------------------
demo.py builds the rotation matrix as

    R = rotation_6d_to_matrix(cat([-gnormal, tangent], dim=-1))

i.e. **GraspNet convention**: column 0 = approach (into the object), column 1 =
closing (between fingers), column 2 = binormal.

The client (agentic_grasp.perception_client + transforms.offset_along_pose_z)
assumes column 2 is the approach direction. We multiply on the right by a
fixed orthogonal matrix that swaps cols 0 and 2 (with a sign flip to keep
det = +1):

    R_client = R_graspnet @ SWAP

After this transformation:
    R_client[:, 2] = R_graspnet[:, 0]   # approach
    R_client[:, 1] = R_graspnet[:, 1]   # closing
    R_client[:, 0] = -R_graspnet[:, 2]  # binormal (sign irrelevant: gripper
                                        #           is symmetric across this axis)

Translation is GraspNet's grasp-center, in metres, in camera frame — passed
through unchanged because the client treats T_cam[:3, 3] as the fingertip /
grasp center anyway.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch as th
import torch.nn.functional as F
import yaml

log = logging.getLogger(__name__)


# Right-multiply on rotation: swap col 0 <-> col 2, flipping col 0 sign so det = +1.
_SWAP_GN_TO_CLIENT = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


class ZeroGraspPipeline:
    """Loads ZeroGrasp once, then serves grasp predictions on (rgb, depth, mask, K) inputs.

    Heavy upstream imports are deferred to __init__ so this module is cheap to
    import (e.g. when running with SERVER_STUB=1).

    Args:
        checkpoint: path to the .ckpt (PyTorch Lightning) file.
        config_path: path to configs/demo.yaml.
        meta_template: path to demo/000.meta.yml — its schema is mirrored when
            we synthesise per-request meta files for fetch_data().
        device: torch device.
    """

    def __init__(
        self,
        checkpoint: str,
        config_path: str,
        device: str = "cuda",
    ):
        self.device = device
        self.checkpoint = checkpoint
        self.config_path = config_path

        # parse_config in zerograsp/utils/config.py uses argparse internally;
        # we have to spoof sys.argv to call it without forking a process.
        from zerograsp.utils.config import parse_config

        old_argv = sys.argv
        sys.argv = [
            "perception_server",
            "--config", config_path,
            "--checkpoint", checkpoint,
            # parse_config requires these even though they are dummies for our flow:
            "--img_path", "demo/000.rgb.png",
            "--depth_path", "demo/000.depth.png",
            "--mask_path", "demo/000.mask.png",
            "--camera_info_path", "demo/000.meta.yml",
        ]
        try:
            self.config = parse_config()
        finally:
            sys.argv = old_argv

        # demo.py sets this; required so the model regenerates octrees from
        # depth on each call (rather than expecting cached training tensors).
        self.config.update_octree = True

        log.info(
            "ZeroGrasp model resolution: %dx%d (from config.img_height/width)",
            self.config.img_height, self.config.img_width,
        )

        log.info("loading ZeroGrasp from checkpoint=%s", checkpoint)
        from main import BaseTrainer  # ZeroGrasp's main.py
        self.model = BaseTrainer.load_from_checkpoint(
            checkpoint, config=self.config, strict=False
        )
        self.model.to(device)
        self.model.eval()
        log.info("ZeroGrasp model ready on %s", device)

    # ----------------------------- public API -------------------------------

    @th.no_grad()
    def predict(
        self,
        rgb: np.ndarray,
        depth_mm: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Run ZeroGrasp on a single image, return up to `top_k` grasp candidates.

        Args:
            rgb: (H, W, 3) uint8 in RGB order.
            depth_mm: (H, W) uint16 millimetres (0 = invalid).
            mask: (H, W) uint8 — single object, any nonzero value = foreground.
            K: (3, 3) intrinsics (pinhole, no distortion).
            top_k: max grasps to return after collision filtering + NMS.

        Returns:
            list of grasp dicts with keys T_cam (4x4 list), width, score, depth.
        """
        from zerograsp.utils.dataset import fetch_data
        from zerograsp.utils.math import unnormalize_pts, rotation_6d_to_matrix
        from zerograsp.nets.utils import get_xyz_from_octree
        from zerograsp.utils.collision_detector import ModelFreeCollisionDetector
        from graspnetAPI import GraspGroup

        # Defensive shape handling. Upstream fetch_data() does
        # `np.logical_and(mask == obj_id, depth > 10)` which requires both
        # operands to broadcast — and that breaks if mask sneaks in as
        # (H, W, C). Collapse to (H, W) before we hit fetch_data.
        if mask.ndim == 3:
            mask = mask.max(axis=2)

        # Empty mask early-exit. fetch_data() crashes on `np.unique(mask)[1:]`
        # when there are no foreground pixels.
        if int((mask > 0).sum()) == 0:
            log.warning("predict: empty mask, returning no grasps")
            return []
        if rgb.shape[:2] != depth_mm.shape[:2] or rgb.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"shape mismatch: rgb={rgb.shape[:2]} depth={depth_mm.shape[:2]} "
                f"mask={mask.shape[:2]} (must all be H, W)"
            )

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            paths = self._dump_inputs(td, rgb, depth_mm, mask, K)

            batch = fetch_data(
                str(paths["rgb"]),
                str(paths["depth"]),
                str(paths["mask"]),
                str(paths["meta"]),
                self.config,
                1.0,
            )

            grid_res = 1 << self.config.min_lod
            output = self.model.model(batch)

            z_min = batch[-2][0]
            pts_3d_in = batch[3][0]
            rays_3d = batch[4][0]

            octrees_out = output["octrees_out"]
            pcd, batch_id_t = get_xyz_from_octree(
                octrees_out, self.config.max_lod, nempty=True, return_batch=True,
            )
            pcd = unnormalize_pts(pcd, z_min, self.config.grid_size, grid_res)
            normals = octrees_out.normals[self.config.max_lod]
            signal = octrees_out.features[self.config.max_lod]
            sdf = signal[:, :1]

            batch_id = batch_id_t.cpu().numpy()
            pcd_np = pcd.cpu().numpy()
            normals_np = F.normalize(normals, dim=-1).cpu().numpy()
            sdf_np = sdf.cpu().numpy()
            # Push points off the surface by their predicted SDF (mirrors demo.py).
            pcd_np = pcd_np - normals_np * sdf_np

            obj_ids = th.unique(pts_3d_in.labels, sorted=True)
            if len(obj_ids) == 0:
                return []

            # Server contract: client always sends a single-object mask, so
            # we only emit grasps for the first object (index 0 in batch_id).
            i = 0
            obj_pts_mask = batch_id == i
            if not obj_pts_mask.any():
                return []

            masked_pcd = pcd_np[obj_pts_mask]
            masked_signal = signal[obj_pts_mask, 1:]

            quality = masked_signal[:, :1].cpu().numpy()
            tangent = masked_signal[:, 2:5]
            gnormal = masked_signal[:, 5:8]
            R_gn = rotation_6d_to_matrix(
                th.cat([-gnormal, tangent], dim=-1)
            ).cpu().numpy()  # (N, 3, 3) — GraspNet convention
            depth_norm = masked_signal[:, 8:9].cpu().numpy()
            width_norm = masked_signal[:, 9:10].cpu().numpy()
            translation = masked_pcd / 1000.0  # mm -> m
            height = 0.02 * np.ones_like(quality)

            # Same scale constants as demo.py.
            GRASP_MAX_WIDTH = 0.1
            GRASP_MAX_DEPTH = 0.04
            grasp_array = np.concatenate(
                [
                    quality,
                    np.clip(width_norm * GRASP_MAX_WIDTH, 0.0, GRASP_MAX_WIDTH),
                    height,
                    np.clip(depth_norm * GRASP_MAX_DEPTH, 0.0, GRASP_MAX_DEPTH),
                    R_gn.reshape(-1, 9),
                    translation.reshape(-1, 3),
                    -1 * np.ones_like(quality),
                ],
                axis=-1,
            )
            gg = GraspGroup(grasp_array).sort_by_score()

            # Collision detection against the visible depth-pcd (every 5th point).
            depth_pcd = rays_3d.reshape(-1, 3)[::5].cpu().numpy()
            cloud = th.from_numpy(pcd_np / 1000.0).float().to(self.device)
            cloud_nrm = th.from_numpy(normals_np).float().to(self.device)
            depth_cloud = th.from_numpy(depth_pcd / 1000.0).float().to(self.device)
            mfc = ModelFreeCollisionDetector(cloud, cloud_nrm, depth_cloud)
            collision_mask, delta_width, refined_depth = mfc.detect(gg)
            gg.grasp_group_array[:, 1] = gg.grasp_group_array[:, 1] + delta_width
            gg.grasp_group_array[:, 3] = refined_depth
            if (~collision_mask).sum() > 0:
                gg = gg[~collision_mask]

            gg = gg.nms(0.03, 30.0 / 180 * np.pi).sort_by_score()
            gg = gg[: max(1, int(top_k))]

            return [self._gn_to_client(g) for g in gg]

    # ----------------------------- internals --------------------------------

    def _dump_inputs(
        self,
        td: Path,
        rgb: np.ndarray,
        depth_mm: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
    ) -> dict[str, Path]:
        """Persist arrays to disk so we can reuse upstream fetch_data() unchanged.

        Two non-obvious things happen here:

        1. **Resize to model resolution.** ZeroGrasp's fetch_data assumes RGB,
           depth and rays are all at (config.img_height, config.img_width) =
           (1024, 1280). It explicitly resizes RGB but NOT depth, then does
           `camera_rays * depth[:, :, None]` which would broadcast-crash if
           shapes disagree. So we resize everything (and rescale K) on the
           server side. Already-aligned inputs are a no-op.

        2. **`left_p` meta schema.** fetch_data calls `extract_camera_matrix`
           with `proj=True`, which reads the 12-element flat `left_p` 3x4
           projection matrix from the YAML. We synthesise it from the request
           K with last column = 0 (no stereo baseline). Skipping
           `rectified_width` / `width` keys disables the upstream's
           rectification rescale, so K passes through verbatim.
        """
        target_h = int(self.config.img_height)
        target_w = int(self.config.img_width)
        src_h, src_w = rgb.shape[:2]

        if (src_h, src_w) != (target_h, target_w):
            sx = target_w / src_w
            sy = target_h / src_h
            rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            depth_mm = cv2.resize(
                depth_mm.astype(np.uint16),
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            K = np.asarray(K, dtype=np.float64).copy()
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy

        rgb_path = td / "rgb.png"
        depth_path = td / "depth.png"
        mask_path = td / "mask.png"
        meta_path = td / "meta.yml"

        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        if depth_mm.dtype != np.uint16:
            depth_mm = depth_mm.astype(np.uint16)
        cv2.imwrite(str(depth_path), depth_mm)

        # Single-object mask: any nonzero treated as object id 1.
        # `np.unique(mask)[1:]` in fetch_data picks {1} for our binary mask.
        mask_u8 = (mask > 0).astype(np.uint8)
        cv2.imwrite(str(mask_path), mask_u8)

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        left_p = [
            fx,  0.0, cx,  0.0,
            0.0, fy,  cy,  0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        meta = {
            "left_p": left_p,
            "height": target_h,
            "width": target_w,
            # Intentionally NO `rectified_width` -> extract_camera_matrix skips
            # the rectification rescale block.
        }
        with meta_path.open("w") as fp:
            yaml.safe_dump(meta, fp)

        return {"rgb": rgb_path, "depth": depth_path, "mask": mask_path, "meta": meta_path}

    @staticmethod
    def _gn_to_client(gn) -> dict[str, Any]:
        """graspnetAPI.Grasp -> wire schema. Performs the GraspNet -> client R basis swap."""
        T_cam = np.eye(4, dtype=np.float64)
        T_cam[:3, :3] = gn.rotation_matrix @ _SWAP_GN_TO_CLIENT
        T_cam[:3, 3] = gn.translation
        return {
            "T_cam": T_cam.tolist(),
            "width": float(gn.width),
            "score": float(gn.score),
            "depth": float(gn.depth),
        }


