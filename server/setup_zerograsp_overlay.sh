#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash server/setup_zerograsp_overlay.sh [options]

Sync RUC-WONE's ZeroGrasp server overlay into a full ZeroGrasp checkout.

Default target:
  1. ./third_party/ZeroGrasp if it exists
  2. ~/zyh/ZeroGrasp otherwise

Options:
  --zerograsp-dir DIR   Full ZeroGrasp checkout to overlay
  --overlay-dir DIR     Overlay source; defaults to server/zerograsp_server_bundle
  --build               Run ./docker/build.sh after syncing
  --start               Start the perception server after syncing
  --offline-cache       Use already-downloaded HuggingFace cache, no network
  --gpu ID              GPU id(s) for docker/run_server.sh; default 0
  --port PORT           Host port; default 9100
  --stub                Start in SERVER_STUB=1 mode
  -h, --help            Show this help

Examples:
  bash server/setup_zerograsp_overlay.sh \
    --zerograsp-dir /home/user/zyh/ZeroGrasp --build --start

  SERVER_STUB=1 bash server/setup_zerograsp_overlay.sh \
    --zerograsp-dir /home/user/zyh/ZeroGrasp --start

  bash server/setup_zerograsp_overlay.sh --offline-cache --start
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OVERLAY_DIR="${ZEROGRASP_OVERLAY_DIR:-$SCRIPT_DIR/zerograsp_server_bundle}"
if [[ -d "$REPO_ROOT/third_party/ZeroGrasp" ]]; then
  DEFAULT_ZEROGRASP_DIR="$REPO_ROOT/third_party/ZeroGrasp"
else
  DEFAULT_ZEROGRASP_DIR="$HOME/zyh/ZeroGrasp"
fi

ZEROGRASP_DIR="${ZEROGRASP_DIR:-$DEFAULT_ZEROGRASP_DIR}"
BUILD=0
START=0
OFFLINE_CACHE=0
GPU_ID="${ZEROGRASP_GPU_ID:-0}"
PORT="${ZEROGRASP_PORT:-9100}"
SERVER_STUB="${SERVER_STUB:-0}"
ZEROGRASP_CHECKPOINT="${ZEROGRASP_CHECKPOINT:-checkpoints/mirage_cvpr2025/mirage/epoch=1-step=80000.ckpt}"
ZEROGRASP_CONFIG="${ZEROGRASP_CONFIG:-configs/demo.yaml}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-$HOME/.cache/huggingface/hub}"
CONTAINER_HF_CACHE_ROOT="${CONTAINER_HF_CACHE_ROOT:-/root/.cache/huggingface/hub}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zerograsp-dir)
      ZEROGRASP_DIR="$2"
      shift 2
      ;;
    --overlay-dir)
      OVERLAY_DIR="$2"
      shift 2
      ;;
    --build)
      BUILD=1
      shift
      ;;
    --start)
      START=1
      shift
      ;;
    --offline-cache)
      OFFLINE_CACHE=1
      shift
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --stub)
      SERVER_STUB=1
      shift
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      echo "[zerograsp] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$OVERLAY_DIR" ]]; then
  echo "[zerograsp] overlay not found: $OVERLAY_DIR" >&2
  exit 1
fi
if [[ ! -d "$ZEROGRASP_DIR" ]]; then
  cat >&2 <<EOF
[zerograsp] ZeroGrasp checkout not found: $ZEROGRASP_DIR

Create or clone the full ZeroGrasp repo first. For zjlab, the recommended
location is:

  /home/user/zyh/ZeroGrasp

Then rerun this script with:

  bash server/setup_zerograsp_overlay.sh --zerograsp-dir /home/user/zyh/ZeroGrasp
EOF
  exit 1
fi
if ! command -v rsync >/dev/null 2>&1; then
  echo "[zerograsp] rsync command not found" >&2
  exit 1
fi

echo "[zerograsp] overlay: $OVERLAY_DIR"
echo "[zerograsp] target:  $ZEROGRASP_DIR"

rsync -av \
  --exclude="__pycache__" \
  --exclude=".git" \
  "$OVERLAY_DIR/" "$ZEROGRASP_DIR/"

chmod +x \
  "$ZEROGRASP_DIR/docker/build.sh" \
  "$ZEROGRASP_DIR/docker/run_server.sh" \
  "$ZEROGRASP_DIR/serve.sh"

missing=()
for path in \
  requirements.txt \
  submodules/octree_feature_extractor \
  docker/build.sh \
  docker/run_server.sh \
  perception_server \
  serve.sh \
  requirements-server.txt \
  constraints.txt
do
  if [[ ! -e "$ZEROGRASP_DIR/$path" ]]; then
    missing+=("$path")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "[zerograsp] target does not look like a complete ZeroGrasp checkout:" >&2
  for path in "${missing[@]}"; do
    echo "  - missing $path" >&2
  done
  exit 1
fi

model_missing=()
if [[ "$SERVER_STUB" != "1" ]]; then
  for path in "$ZEROGRASP_CONFIG" "$ZEROGRASP_CHECKPOINT"; do
    if [[ ! -f "$ZEROGRASP_DIR/$path" ]]; then
      model_missing+=("$path")
    fi
  done
fi

if (( ${#model_missing[@]} > 0 )); then
  echo "[zerograsp] real-model files are missing:" >&2
  for path in "${model_missing[@]}"; do
    echo "  - $ZEROGRASP_DIR/$path" >&2
  done
  if (( START )); then
    echo "[zerograsp] refusing to start real-model server; use --stub for a plumbing test." >&2
    exit 1
  fi
fi

if (( BUILD )); then
  echo "[zerograsp] building docker image"
  (cd "$ZEROGRASP_DIR" && ./docker/build.sh)
fi

latest_hf_snapshot() {
  local model_cache_dir="$1"
  local snapshot_dir="$HF_CACHE_ROOT/$model_cache_dir/snapshots"
  if [[ ! -d "$snapshot_dir" ]]; then
    return 1
  fi
  find "$snapshot_dir" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %f\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR == 1 {print $2}'
}

if (( OFFLINE_CACHE )); then
  gd_snapshot="$(latest_hf_snapshot "models--IDEA-Research--grounding-dino-tiny" || true)"
  sam_snapshot="$(latest_hf_snapshot "models--facebook--sam-vit-base" || true)"
  if [[ -z "$gd_snapshot" || -z "$sam_snapshot" ]]; then
    cat >&2 <<EOF
[zerograsp] HuggingFace cache snapshot not found.

Expected cache directories:
  $HF_CACHE_ROOT/models--IDEA-Research--grounding-dino-tiny/snapshots/
  $HF_CACHE_ROOT/models--facebook--sam-vit-base/snapshots/

If the models were downloaded under another user, copy or move that
~/.cache/huggingface directory to this account first.
EOF
    exit 1
  fi
  export GD_MODEL="$CONTAINER_HF_CACHE_ROOT/models--IDEA-Research--grounding-dino-tiny/snapshots/$gd_snapshot"
  export SAM_MODEL="$CONTAINER_HF_CACHE_ROOT/models--facebook--sam-vit-base/snapshots/$sam_snapshot"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  echo "[zerograsp] offline cache enabled"
  echo "[zerograsp] GD_MODEL=$GD_MODEL"
  echo "[zerograsp] SAM_MODEL=$SAM_MODEL"
fi

if (( START )); then
  echo "[zerograsp] starting perception server"
  (cd "$ZEROGRASP_DIR" && \
    DETACH="${DETACH:-1}" PORT="$PORT" SERVER_STUB="$SERVER_STUB" \
    GD_MODEL="${GD_MODEL:-}" SAM_MODEL="${SAM_MODEL:-}" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" \
    ZEROGRASP_CHECKPOINT="$ZEROGRASP_CHECKPOINT" ZEROGRASP_CONFIG="$ZEROGRASP_CONFIG" \
    ./docker/run_server.sh "$GPU_ID")
  echo "[zerograsp] logs: docker logs -f perception-server"
fi

echo "[zerograsp] done"
