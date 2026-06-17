"""HTTP client for the remote perception server (SAM2 / GroundingDINO / ZeroGrasp).

The server exposes:
  GET  /healthz                                -> {"ok": true, "ts": ..., "stub": ...}
  POST /segment   {rgb_b64, H, W, prompt|bbox} -> {"bbox", "mask_b64", "score", ...}
  POST /grasp     {rgb_b64, depth_b64, K, mask_b64, H, W} -> {"grasps": [...]}
  POST /grasp_dummy (same payload as /grasp)   -> geometric fallback
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests

from agentic_grasp.config import settings


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _from_b64(b64: str, dtype, shape) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=dtype).reshape(shape)


@dataclass
class Segmentation:
    bbox: list[int]            # [x1, y1, x2, y2]
    mask: np.ndarray           # (H, W) uint8 in {0, 1}
    score: float
    raw: dict


@dataclass
class Grasp:
    T_cam: np.ndarray          # (4, 4), grasp pose in camera frame
    width: float               # gripper opening width, metres
    score: float
    raw: dict


class PerceptionClient:
    def __init__(self, url: str | None = None, token: str | None = None, timeout: float = 60.0):
        self.url = (url or settings.perception_url).rstrip("/")
        self.token = token if token is not None else settings.perception_token
        self.timeout = timeout
        self._sess = requests.Session()
        if self.token:
            self._sess.headers["Authorization"] = f"Bearer {self.token}"

    # ---------- low-level ----------

    def _post(self, path: str, payload: dict, timeout: float | None = None) -> dict:
        r = self._sess.post(f"{self.url}{path}", json=payload, timeout=timeout or self.timeout)
        r.raise_for_status()
        return r.json()

    def healthz(self) -> dict:
        r = self._sess.get(f"{self.url}/healthz", timeout=5)
        r.raise_for_status()
        return r.json()

    # ---------- typed wrappers ----------

    def segment(
        self,
        rgb: np.ndarray,
        prompt: str | None = None,
        bbox: list[int] | None = None,
    ) -> Segmentation:
        if rgb.dtype != np.uint8:
            raise TypeError(f"rgb must be uint8, got {rgb.dtype}")
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must be (H,W,3), got {rgb.shape}")
        H, W = rgb.shape[:2]
        payload: dict[str, Any] = {"rgb_b64": _b64(rgb), "H": H, "W": W}
        if prompt is not None:
            payload["prompt"] = prompt
        if bbox is not None:
            payload["bbox"] = list(map(int, bbox))
        if prompt is None and bbox is None:
            raise ValueError("Need at least `prompt` or `bbox` to /segment")
        r = self._post("/segment", payload)
        mask = _from_b64(r["mask_b64"], np.uint8, (H, W))
        return Segmentation(bbox=list(map(int, r["bbox"])), mask=mask, score=float(r["score"]), raw=r)

    def grasps(
        self,
        rgb: np.ndarray,
        depth_mm: np.ndarray,
        K: np.ndarray,
        mask: np.ndarray,
        endpoint: str = "/grasp",
        top_k: int = 10,
    ) -> list[Grasp]:
        H, W = rgb.shape[:2]
        if depth_mm.dtype != np.uint16:
            depth_mm = depth_mm.astype(np.uint16)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        payload = {
            "rgb_b64": _b64(rgb),
            "depth_b64": _b64(depth_mm),
            "mask_b64": _b64(mask),
            "K": np.asarray(K, dtype=np.float64).tolist(),
            "H": H, "W": W,
            "top_k": top_k,
        }
        r = self._post(endpoint, payload, timeout=120)
        out: list[Grasp] = []
        for g in r.get("grasps", [])[:top_k]:
            out.append(Grasp(
                T_cam=np.asarray(g["T_cam"], dtype=np.float64).reshape(4, 4),
                width=float(g.get("width", 0.06)),
                score=float(g.get("score", 0.0)),
                raw=g,
            ))
        out.sort(key=lambda g: g.score, reverse=True)
        return out
