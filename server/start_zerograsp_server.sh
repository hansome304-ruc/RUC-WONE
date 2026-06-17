#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="${ZEROGRASP_BUNDLE_DIR:-$SCRIPT_DIR/zerograsp_server_bundle}"
GPU_ID="${ZEROGRASP_GPU_ID:-${1:-0}}"
PORT="${ZEROGRASP_PORT:-9100}"
NAME="${ZEROGRASP_CONTAINER_NAME:-perception-server}"
IMAGE="${ZEROGRASP_IMAGE:-zerograsp:latest}"
SERVER_STUB="${SERVER_STUB:-0}"
ZEROGRASP_CHECKPOINT="${ZEROGRASP_CHECKPOINT:-checkpoints/mirage_cvpr2025/mirage/epoch=1-step=80000.ckpt}"
ZEROGRASP_CONFIG="${ZEROGRASP_CONFIG:-configs/demo.yaml}"

if [[ ! -d "$BUNDLE_DIR" ]]; then
  echo "[zerograsp] bundle not found: $BUNDLE_DIR" >&2
  exit 1
fi
if [[ ! -x "$BUNDLE_DIR/docker/run_server.sh" ]]; then
  echo "[zerograsp] launcher not executable: $BUNDLE_DIR/docker/run_server.sh" >&2
  exit 1
fi
if [[ "$SERVER_STUB" != "1" ]]; then
  missing=0
  if [[ ! -f "$BUNDLE_DIR/$ZEROGRASP_CHECKPOINT" ]]; then
    echo "[zerograsp] checkpoint not found: $BUNDLE_DIR/$ZEROGRASP_CHECKPOINT" >&2
    missing=1
  fi
  if [[ ! -f "$BUNDLE_DIR/$ZEROGRASP_CONFIG" ]]; then
    echo "[zerograsp] config not found: $BUNDLE_DIR/$ZEROGRASP_CONFIG" >&2
    missing=1
  fi
  if [[ "$missing" == "1" ]]; then
    cat >&2 <<EOF

[zerograsp] Real-model mode needs a full ZeroGrasp checkout, not only the
server bundle. Put/copy this bundle into the full repo, or point the launcher
at it:

  rsync -avzh --progress $SCRIPT_DIR/zerograsp_server_bundle/ user@10.42.115.70:~/zyh/ZeroGrasp/
  ZEROGRASP_BUNDLE_DIR=~/zyh/ZeroGrasp bash server/start_zerograsp_server.sh

For a pure client smoke test without model files:

  SERVER_STUB=1 bash server/start_zerograsp_server.sh
EOF
    exit 1
  fi
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "[zerograsp] docker command not found" >&2
  exit 1
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  cat >&2 <<EOF
[zerograsp] Docker image not found: $IMAGE

This launcher uses the existing ZeroGrasp Docker image. Build it on the GPU
server/full ZeroGrasp checkout first, for example:

  cd ~/zyh/ZeroGrasp
  ./docker/build.sh

Then start:

  cd /home/ubuntu/RUC-WONE
  bash server/start_zerograsp_server.sh

For client-only plumbing tests, SERVER_STUB=1 still needs the same image:

  SERVER_STUB=1 bash server/start_zerograsp_server.sh
EOF
  exit 1
fi

echo "[zerograsp] bundle: $BUNDLE_DIR"
echo "[zerograsp] image:  $IMAGE"
echo "[zerograsp] gpu:    $GPU_ID"
echo "[zerograsp] port:   $PORT"
echo "[zerograsp] name:   $NAME"
echo "[zerograsp] stub:   $SERVER_STUB"

cd "$BUNDLE_DIR"
DETACH="${DETACH:-1}" PORT="$PORT" NAME="$NAME" SERVER_STUB="$SERVER_STUB" \
  ZEROGRASP_CHECKPOINT="$ZEROGRASP_CHECKPOINT" ZEROGRASP_CONFIG="$ZEROGRASP_CONFIG" \
  ./docker/run_server.sh "$GPU_ID"

echo "[zerograsp] started container: $NAME"
echo "[zerograsp] logs: docker logs -f $NAME"
echo "[zerograsp] check: bash server/check_zerograsp_server.sh"
