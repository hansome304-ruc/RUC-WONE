# Deployment Guide — perception server bundle

This RUC-WONE server bundle is meant to be merged into your existing
`~/zyh/ZeroGrasp/` checkout on the GPU box (`user@10.42.115.70`).
It carries the perception-server overlay and environment pins; it does not
replace the upstream ZeroGrasp repo or model checkpoints.

## What's in this bundle

```
RUC-WONE/server/zerograsp_server_bundle/
├── DEPLOY.md                     # this file
├── constraints.txt               # numpy<2 / ocnn==2.2.4 / triton==2.2.0
├── requirements-server.txt       # FastAPI + transformers (SAM/GD)
├── docker/
│   ├── Dockerfile                # replaces ZeroGrasp/docker/Dockerfile
│   └── run_server.sh             # host-side container launcher
├── serve.sh                      # in-container uvicorn entry point
└── perception_server/            # the FastAPI app
    ├── __init__.py
    ├── encoding.py               # base64 ↔ ndarray (mirrors client)
    ├── schemas.py                # Pydantic wire schemas
    ├── segment.py                # GroundingDINO + SAM
    ├── grasp.py                  # ZeroGrasp wrapper
    └── server.py                 # FastAPI app
```

## One-time deploy

From the machine that has this bundle (your laptop / dev box):

```bash
# 1) Sync to the GPU server. The trailing slashes here matter.
rsync -avzh --progress \
    /home/ubuntu/RUC-WONE/server/zerograsp_server_bundle/ \
    user@10.42.115.70:~/zyh/ZeroGrasp/
```

This adds:
- `constraints.txt`, `requirements-server.txt`, `serve.sh` (top-level)
- `docker/Dockerfile` **replaces** the existing one
- `docker/run_server.sh` (new)
- `perception_server/` (new directory)

> If you don't want to overwrite the existing `docker/Dockerfile`, sync to a
> staging path first and diff before applying:
> ```bash
> rsync -avzh ./ user@10.42.115.70:~/zerograsp_server_bundle.staged/
> ssh user@10.42.115.70 "diff ~/zerograsp_server_bundle.staged/docker/Dockerfile ~/zyh/ZeroGrasp/docker/Dockerfile"
> ```

## On the GPU server

```bash
ssh user@10.42.115.70
cd ~/zyh/ZeroGrasp

# A. Submodule MUST be at origin/main (5e84aea or later).
#    Without this the ofe forward signature won't match model.py:313.
cd submodules/octree_feature_extractor && git checkout main && cd ../..

# B. Rebuild image. Most layers are cached; this typically takes 1–3 min.
./docker/build.sh

# C. Smoke test in stub mode (no models, validates network / FastAPI plumbing).
SERVER_STUB=1 ./docker/run_server.sh
# In another shell:
curl http://localhost:9100/healthz

# D. Real run (loads ZeroGrasp + SAM + GroundingDINO; takes ~30 s + warmup).
./docker/run_server.sh           # foreground; Ctrl-C to stop

# Or detached:
DETACH=1 ./docker/run_server.sh
docker logs -f perception-server
```

## Smoke test from the robot side

On the robot machine, with `agentic_grasp` set up:

```bash
cd /home/ubuntu/RUC-WONE/manipulation/agentic
conda activate dos-w1
# .env should have:  PERCEPTION_URL=http://10.42.115.70:9100
python scripts/00_test_health.py
python scripts/01_test_segment.py --prompt "a green bottle"
python scripts/03_dummy_grasp.py --dry-run --prompt "a bottle"
python scripts/run_scripted.py    --dry-run --prompt "a bottle"
```

`00_test_health.py` should print `stub=False` and `model_loaded={"segment": True, "grasp": True}` once the real server is up.

## Operational notes

- **Single GPU worker.** The server holds an `asyncio.Lock` around model
  forward passes. Don't bump `--workers` in `serve.sh`.
- **Cold starts.** First inference JIT-compiles the octree CUDA extension
  (PTX → sm_86). The `SERVER_WARMUP=1` lifespan hook absorbs that cost so
  the first real request is fast. Disable it with `SERVER_WARMUP=0` if it
  ever becomes flaky.
- **Auth.** Set `SERVER_REQUIRE_AUTH_TOKEN=somesecret`; the client side
  already supports `PERCEPTION_TOKEN` and sends `Authorization: Bearer …`.
- **HF cache.** `~/.cache:/root/.cache` is bind-mounted; HuggingFace assets
  download once and persist on the host.
- **Logs.** `docker logs -f perception-server` (detached) or just stay in
  the foreground `run_server.sh` shell.

## Updating just the server code (no rebuild)

Because `~/zyh/ZeroGrasp` is bind-mounted into `/opt/app` inside the
container, edits to `perception_server/*.py` apply on the next request *if*
you restart the uvicorn worker:

```bash
# from the host:
docker exec perception-server pkill -HUP -f uvicorn || \
    docker restart perception-server
```

For changes to anything in `requirements-server.txt` you DO need to
`./docker/build.sh` again (the layer is cached up to the COPY of
`requirements-server.txt`, so it's quick).

## What about the upstream `requirements.txt`?

Untouched on purpose. Hard pins live in `constraints.txt` so future
`./docker/build.sh` runs apply them via `pip install -c constraints.txt
-r requirements.txt`. If upstream ever bumps a dependency we want to allow,
just remove the corresponding line from `constraints.txt`.

## Known landmines (already handled, kept here for posterity)

| Symptom | Root cause | Mitigation |
| --- | --- | --- |
| `_ARRAY_API not found`, `numpy.dtype size changed` | numpy 2.x ABI vs torch 2.2 wheels | `numpy<2` in constraints, plus a final `--force-reinstall` in Dockerfile |
| `Autotuner.__init__() got an unexpected keyword argument 'pre_hook'` | ocnn ≥2.3 needs triton 3, torch 2.2 ships triton 2.2 | `ocnn==2.2.4` in constraints |
| `OctreeFeatureExtractor.forward() missing 2 required positional arguments` | `.gitmodules` pins `d602008` (old 8-arg API), but `model.py` calls 6-arg API from `5e84aea` | DEPLOY step A: `git checkout main` in submodule |
| `wandb: command not found` | upstream `docker/run.sh` wraps `wandb docker-run` | use `docker/run_server.sh` instead |
| GraspNet R has approach in col 0, but client expects col 2 | upstream convention difference | `_SWAP_GN_TO_CLIENT` in `grasp.py` |
