"""Pydantic models matching the contract in `agentic_grasp/perception_client.py`.

Conventions:
- All image arrays are sent as raw-bytes base64 (see `encoding.py`).
- `T_cam` is a 4x4 row-major SE(3) matrix in **camera frame**, with the
  convention used by the client:
      T_cam[:3, 3] = grasp center / fingertip position (m)
      T_cam[:3, 2] = approach direction (gripper points TOWARDS the object)
      T_cam[:3, 0:2] = perpendicular to approach (gripper plane)
  This is NOT the GraspNet R-column convention — see grasp.py for the
  conversion matrix (GraspNet has approach as col 0, we want it as col 2).
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HealthzResp(BaseModel):
    ok: bool = True
    ts: float
    stub: bool = False
    model_loaded: dict[str, bool] = Field(default_factory=dict)
    cuda_mem_mb: dict[str, float] | None = None


# --------------------------------- /segment ---------------------------------


class SegReq(BaseModel):
    rgb_b64: str
    H: int
    W: int
    prompt: str | None = None
    bbox: list[int] | None = None  # [x1, y1, x2, y2] in absolute pixels


class SegResp(BaseModel):
    bbox: list[int]
    mask_b64: str
    score: float
    stub: bool = False
    debug: dict | None = None


# ---------------------------------- /grasp ----------------------------------


class GraspItem(BaseModel):
    # disable Pydantic v2's "model_" namespace warning so depth/score fields
    # don't accidentally collide.
    model_config = ConfigDict(protected_namespaces=())

    T_cam: list[list[float]]
    width: float
    score: float
    depth: float | None = None
    obj_id: int | None = None


class GraspReq(BaseModel):
    rgb_b64: str
    depth_b64: str
    mask_b64: str
    K: list[list[float]]
    H: int
    W: int
    top_k: int = 10


class GraspResp(BaseModel):
    grasps: list[GraspItem]
    stub: bool = False
    debug: dict | None = None
