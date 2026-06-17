#!/usr/bin/env bash
# Start the perception server container on the host.
#
# Usage (from ~/zyh/ZeroGrasp on the GPU box):
#     ./docker/run_server.sh                # GPU 0, port 9100, real models
#     ./docker/run_server.sh 1              # GPU 1
#     SERVER_STUB=1 ./docker/run_server.sh  # smoke test without models
#
# Interactive vs detached:
#     DETACH=1 ./docker/run_server.sh       # docker run -d (logs via `docker logs`)
set -euo pipefail

GPU_IDS=${1:-0}
PORT=${PORT:-9100}
NAME=${NAME:-perception-server}

cd "$(dirname "$0")/.."

DETACH_ARGS=( -it --rm )
if [[ "${DETACH:-0}" == "1" ]]; then
    DETACH_ARGS=( -d --restart unless-stopped )
fi

# Stop any previous instance with the same name (only matters in detached mode).
docker rm -f "$NAME" >/dev/null 2>&1 || true

exec docker run "${DETACH_ARGS[@]}" \
    --name "$NAME" \
    --mount type=bind,source="$(pwd)",target=/opt/app \
    --ipc=host \
    -v "$HOME/.cache:/root/.cache" \
    --gpus "\"device=${GPU_IDS}\"" \
    -p "${PORT}:9100" \
    -e SERVER_PORT=9100 \
    -e SERVER_STUB="${SERVER_STUB:-0}" \
    -e SERVER_WARMUP="${SERVER_WARMUP:-1}" \
    -e SERVER_REQUIRE_AUTH_TOKEN="${SERVER_REQUIRE_AUTH_TOKEN:-}" \
    -e ZEROGRASP_CHECKPOINT="${ZEROGRASP_CHECKPOINT:-checkpoints/mirage_cvpr2025/mirage/epoch=1-step=80000.ckpt}" \
    -e ZEROGRASP_CONFIG="${ZEROGRASP_CONFIG:-configs/demo.yaml}" \
    -e GD_MODEL="${GD_MODEL:-IDEA-Research/grounding-dino-tiny}" \
    -e SAM_MODEL="${SAM_MODEL:-facebook/sam-vit-base}" \
    -e HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
    -e TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" \
    -e HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-}}" \
    -e HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-}}" \
    -e NO_PROXY="${NO_PROXY:-${no_proxy:-}}" \
    -e http_proxy="${http_proxy:-${HTTP_PROXY:-}}" \
    -e https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" \
    -e no_proxy="${no_proxy:-${NO_PROXY:-}}" \
    -e LOG_LEVEL="${LOG_LEVEL:-info}" \
    zerograsp:latest /bin/bash /opt/app/serve.sh
