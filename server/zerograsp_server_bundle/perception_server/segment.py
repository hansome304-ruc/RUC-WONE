"""Text-prompt -> binary mask, using HuggingFace transformers.

Pipeline: prompt --[GroundingDINO]--> top-1 bbox --[SAM]--> mask.

We deliberately avoid third-party "grounded-sam" wrappers and the official
SAM-2 repo: both have aggressive dependency footprints / build steps that
fight with this image's pinned versions. transformers gives us
GroundingDINO + SAM (1) with zero extra compilation.

Models (all overridable via env vars in server.py):
    GD_MODEL  default "IDEA-Research/grounding-dino-tiny"   (~170 MB)
    SAM_MODEL default "facebook/sam-vit-base"               (~375 MB)

Both are cached under HF_HOME (mounted to host's ~/.cache/huggingface).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)


class SegmentPipeline:
    """One-shot text-to-mask pipeline. Stateless across calls."""

    def __init__(
        self,
        gd_model_id: str = "IDEA-Research/grounding-dino-tiny",
        sam_model_id: str = "facebook/sam-vit-base",
        device: str = "cuda",
        gd_box_threshold: float = 0.30,
        gd_text_threshold: float = 0.25,
    ):
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            SamModel,
            SamProcessor,
        )

        log.info("loading GroundingDINO from %s", gd_model_id)
        self.gd_proc = AutoProcessor.from_pretrained(gd_model_id)
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            gd_model_id
        ).to(device)
        self.gd_model.eval()

        log.info("loading SAM from %s", sam_model_id)
        self.sam_proc = SamProcessor.from_pretrained(sam_model_id)
        self.sam_model = SamModel.from_pretrained(sam_model_id).to(device)
        self.sam_model.eval()

        self.device = device
        self.box_threshold = gd_box_threshold
        self.text_threshold = gd_text_threshold

    @torch.no_grad()
    def predict(
        self,
        rgb: np.ndarray,
        prompt: str | None = None,
        bbox: list[int] | None = None,
    ) -> dict[str, Any]:
        """Returns {bbox: [x1,y1,x2,y2], mask: (H,W) uint8 in {0,1}, score: float}."""
        if rgb.dtype != np.uint8:
            raise TypeError(f"rgb must be uint8, got {rgb.dtype}")
        H, W = rgb.shape[:2]

        if bbox is not None:
            bx, sc = list(map(int, bbox)), 1.0
        else:
            if not prompt:
                raise ValueError("Either `prompt` or `bbox` must be provided")
            bx, sc = self._dino(rgb, prompt)
            if bx is None:
                # No detection: return a zero mask but still a valid bbox so
                # the client can decide what to do (e.g. retry with a different
                # prompt or fall back to /grasp_dummy).
                return {
                    "bbox": [0, 0, W, H],
                    "mask": np.zeros((H, W), dtype=np.uint8),
                    "score": 0.0,
                }

        mask = self._sam(rgb, bx)
        return {
            "bbox": bx,
            "mask": mask.astype(np.uint8),
            "score": float(sc),
        }

    # ----------------------------- internals --------------------------------

    def _dino(self, rgb: np.ndarray, prompt: str) -> tuple[list[int] | None, float]:
        # GroundingDINO expects each label to end with a "."; otherwise it
        # treats the whole string as one phrase, which is usually what we want
        # for a single object prompt like "a green bottle".
        text = prompt.strip()
        if not text.endswith("."):
            text = text + "."

        H, W = rgb.shape[:2]
        inputs = self.gd_proc(images=rgb, text=text, return_tensors="pt").to(self.device)
        out = self.gd_model(**inputs)

        # Older transformers versions used `box_threshold=` / `text_threshold=`,
        # newer ones renamed to `threshold=` / `text_threshold=`. We try both.
        post = self.gd_proc.post_process_grounded_object_detection
        try:
            results = post(
                out,
                inputs.input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(H, W)],
            )[0]
        except TypeError:
            results = post(
                out,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(H, W)],
            )[0]

        if len(results["boxes"]) == 0:
            return None, 0.0

        scores = results["scores"]
        idx = int(scores.argmax().item())
        b = results["boxes"][idx].tolist()
        bbox_int = [int(round(b[0])), int(round(b[1])), int(round(b[2])), int(round(b[3]))]
        return bbox_int, float(scores[idx].item())

    def _sam(self, rgb: np.ndarray, bbox: list[int]) -> np.ndarray:
        # SAM input_boxes wants shape [batch, num_boxes_per_image, 4]
        # in absolute pixel coords [x1, y1, x2, y2].
        inputs = self.sam_proc(
            rgb,
            input_boxes=[[bbox]],
            return_tensors="pt",
        ).to(self.device)
        out = self.sam_model(**inputs)

        # post_process_masks is on the image_processor (renamed from
        # `mask_processor` in some versions; we try both).
        post = getattr(self.sam_proc, "post_process_masks", None) or \
               self.sam_proc.image_processor.post_process_masks
        masks = post(
            out.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]
        # Shape: (num_boxes_for_image=1, num_masks_per_box=3, H, W) typically.
        # Pick the highest-IoU mask of the (only) box.
        if masks.dim() == 4:
            masks = masks[0]
        ious = out.iou_scores[0, 0].cpu().numpy()
        best = int(ious.argmax())
        return masks[best].numpy().astype(np.uint8)
