"""FastAPI perception server.

Endpoints (matching agentic_grasp/perception_client.py):
    GET  /healthz
    POST /segment        (SAM + GroundingDINO)
    POST /grasp          (ZeroGrasp; collision filtered, NMS'd)
    POST /grasp_dummy    (geometric centroid fallback; runs without GPU)

Operational knobs (env vars):
    SERVER_HOST                  default 0.0.0.0
    SERVER_PORT                  default 9100
    SERVER_STUB                  "1" -> skip model loading, return canned data
    SERVER_WARMUP                "1" -> run one inference at startup (default 1)
    SERVER_REQUIRE_AUTH_TOKEN    if set, /segment and /grasp require Bearer auth
    ZEROGRASP_CHECKPOINT
    ZEROGRASP_CONFIG
    ZEROGRASP_META_TEMPLATE
    GD_MODEL                     HuggingFace id for GroundingDINO
    SAM_MODEL                    HuggingFace id for SAM
    HF_HOME                      HuggingFace cache (Docker mount-point)

GPU access is serialised with a single asyncio.Lock — single-worker uvicorn
is the only supported deployment.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from .encoding import b64_to_np, np_to_b64
from .schemas import (
    GraspItem,
    GraspReq,
    GraspResp,
    HealthzResp,
    SegReq,
    SegResp,
)

log = logging.getLogger("perception_server")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


# ----------------------------- lifespan / state -----------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.stub = os.getenv("SERVER_STUB", "0") == "1"
    app.state.gpu_lock = asyncio.Lock()
    app.state.segment = None
    app.state.grasp = None
    app.state.auth_token = os.getenv("SERVER_REQUIRE_AUTH_TOKEN", "") or None

    if app.state.stub:
        log.warning(
            "SERVER_STUB=1 — models will NOT be loaded; canned responses only.",
        )
    else:
        from .grasp import ZeroGraspPipeline
        from .segment import SegmentPipeline

        ckpt = os.getenv(
            "ZEROGRASP_CHECKPOINT",
            "checkpoints/mirage_cvpr2025/mirage/epoch=1-step=80000.ckpt",
        )
        cfg = os.getenv("ZEROGRASP_CONFIG", "configs/demo.yaml")
        gd_id = os.getenv("GD_MODEL", "IDEA-Research/grounding-dino-tiny")
        sam_id = os.getenv("SAM_MODEL", "facebook/sam-vit-base")

        log.info("loading SegmentPipeline (gd=%s sam=%s)", gd_id, sam_id)
        app.state.segment = SegmentPipeline(gd_model_id=gd_id, sam_model_id=sam_id)

        log.info("loading ZeroGraspPipeline (ckpt=%s cfg=%s)", ckpt, cfg)
        app.state.grasp = ZeroGraspPipeline(checkpoint=ckpt, config_path=cfg)

        if os.getenv("SERVER_WARMUP", "1") == "1":
            try:
                _warmup(app)
            except Exception as exc:  # pragma: no cover - never fatal
                log.warning("warmup failed (continuing): %s", exc)

    log.info(
        "perception_server ready: stub=%s grasp_loaded=%s segment_loaded=%s",
        app.state.stub,
        app.state.grasp is not None,
        app.state.segment is not None,
    )
    yield


def _warmup(app: FastAPI) -> None:
    """Run one inference using demo/000.* so all CUDA kernels JIT-compile.

    This avoids a 10–30 s stall on the first real request (PTX -> SASS for
    the octree extension on RTX 3090).
    """
    import cv2

    rgb_bgr = cv2.imread("demo/000.rgb.png", cv2.IMREAD_COLOR)
    depth = cv2.imread("demo/000.depth.png", cv2.IMREAD_UNCHANGED)
    mask = cv2.imread("demo/000.mask.png", cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None or depth is None or mask is None:
        log.info("warmup: demo/000.* missing, skipping")
        return

    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16)
    # The repo's demo masks are RGB (per-instance colour). Collapse to a
    # 2D binary mask. Real client requests already send 2D masks.
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    mask_bin = (mask > 0).astype(np.uint8)

    H, W = rgb.shape[:2]
    # The demo's actual K isn't strictly required for warmup — any plausible
    # pinhole matrix works. Reading the real one would couple us to demo/000.meta.yml.
    K = np.array(
        [[600.0, 0.0, W / 2], [0.0, 600.0, H / 2], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    log.info("warmup: running one ZeroGrasp inference …")
    t0 = time.time()
    n = len(app.state.grasp.predict(rgb, depth, mask_bin, K, top_k=1))
    log.info("warmup ok: %d grasp(s) in %.2fs", n, time.time() - t0)


app = FastAPI(
    title="perception_server",
    version="0.1.0",
    lifespan=lifespan,
)


# ------------------------------- error handler -------------------------------


@app.exception_handler(Exception)
async def _on_exception(request, exc):
    log.exception("unhandled error in %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal", "msg": f"{type(exc).__name__}: {exc}"}},
    )


# --------------------------------- helpers ----------------------------------


def _check_auth(authorization: str | None) -> None:
    expected = app.state.auth_token
    if not expected:
        return
    got = (authorization or "").removeprefix("Bearer ").strip()
    if got != expected:
        raise HTTPException(401, "invalid or missing bearer token")


# ---------------------------------- /healthz --------------------------------


@app.get("/healthz", response_model=HealthzResp)
async def healthz():
    cuda_mem = None
    if torch.cuda.is_available():
        cuda_mem = {
            "alloc_mb": float(torch.cuda.memory_allocated()) / 1e6,
            "reserved_mb": float(torch.cuda.memory_reserved()) / 1e6,
        }
    return HealthzResp(
        ok=True,
        ts=time.time(),
        stub=app.state.stub,
        model_loaded={
            "segment": app.state.segment is not None,
            "grasp": app.state.grasp is not None,
        },
        cuda_mem_mb=cuda_mem,
    )


# ---------------------------------- /segment --------------------------------


@app.post("/segment", response_model=SegResp)
async def segment(req: SegReq, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    if not req.prompt and not req.bbox:
        raise HTTPException(400, "Either `prompt` or `bbox` must be provided")

    rgb = b64_to_np(req.rgb_b64, np.uint8, (req.H, req.W, 3))

    if app.state.stub or app.state.segment is None:
        h0, h1 = req.H // 4, 3 * req.H // 4
        w0, w1 = req.W // 4, 3 * req.W // 4
        m = np.zeros((req.H, req.W), dtype=np.uint8)
        m[h0:h1, w0:w1] = 1
        return SegResp(
            bbox=req.bbox or [w0, h0, w1, h1],
            mask_b64=np_to_b64(m),
            score=0.99,
            stub=True,
        )

    async with app.state.gpu_lock:
        out = app.state.segment.predict(rgb, prompt=req.prompt, bbox=req.bbox)

    return SegResp(
        bbox=list(map(int, out["bbox"])),
        mask_b64=np_to_b64(out["mask"].astype(np.uint8)),
        score=float(out["score"]),
    )


# ---------------------------------- /grasp ----------------------------------


@app.post("/grasp", response_model=GraspResp)
async def grasp(req: GraspReq, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    rgb = b64_to_np(req.rgb_b64, np.uint8, (req.H, req.W, 3))
    depth_mm = b64_to_np(req.depth_b64, np.uint16, (req.H, req.W))
    mask = b64_to_np(req.mask_b64, np.uint8, (req.H, req.W))
    K = np.asarray(req.K, dtype=np.float64)

    if app.state.stub or app.state.grasp is None:
        return _stub_grasp(rgb, depth_mm, mask, K, req.top_k)

    async with app.state.gpu_lock:
        out = app.state.grasp.predict(rgb, depth_mm, mask, K, top_k=req.top_k)

    return GraspResp(grasps=[GraspItem(**g) for g in out])


@app.post("/grasp_dummy", response_model=GraspResp)
async def grasp_dummy(req: GraspReq, authorization: str | None = Header(default=None)):
    """Geometric fallback: top-down grasp at the centroid of the masked depth.

    Always available regardless of model loading state. Useful when ZeroGrasp
    is being recompiled / restarted, or the object is in a corner case the
    model fails on.
    """
    _check_auth(authorization)
    rgb = b64_to_np(req.rgb_b64, np.uint8, (req.H, req.W, 3))
    depth_mm = b64_to_np(req.depth_b64, np.uint16, (req.H, req.W))
    mask = b64_to_np(req.mask_b64, np.uint8, (req.H, req.W))
    K = np.asarray(req.K, dtype=np.float64)
    return _stub_grasp(rgb, depth_mm, mask, K, req.top_k)


def _stub_grasp(rgb, depth_mm, mask, K, top_k):
    """Centroid-of-masked-depth, top-down grasp. Client-frame R (z = approach)."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return GraspResp(grasps=[], stub=True)
    cx, cy = float(xs.mean()), float(ys.mean())
    md = depth_mm[mask > 0]
    md = md[md > 0]
    z_m = float(np.median(md)) / 1000.0 if len(md) else 0.5
    x_m = (cx - K[0, 2]) * z_m / K[0, 0]
    y_m = (cy - K[1, 2]) * z_m / K[1, 1]

    # Top-down: approach = camera +z, gripper closes along +x.
    # (Identity rotation: client convention has approach = R[:, 2] = e_z.)
    T = np.eye(4)
    T[:3, 3] = [x_m, y_m, z_m]
    return GraspResp(
        grasps=[GraspItem(T_cam=T.tolist(), width=0.06, score=0.5, depth=0.02)],
        stub=True,
    )
