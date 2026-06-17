#!/usr/bin/env bash
# Build the ZeroGrasp perception-server image from a full ZeroGrasp checkout.
#
# This server bundle is meant to be overlaid onto the upstream ZeroGrasp repo.
# The build needs upstream files such as requirements.txt and submodules/.
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE="${ZEROGRASP_IMAGE:-zerograsp:latest}"
DOCKERFILE="${ZEROGRASP_DOCKERFILE:-docker/Dockerfile}"

missing=()
for path in \
  requirements.txt \
  constraints.txt \
  requirements-server.txt \
  serve.sh \
  perception_server \
  submodules/octree_feature_extractor
do
  if [[ ! -e "$path" ]]; then
    missing+=("$path")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "[zerograsp] cannot build yet. Missing required ZeroGrasp files:" >&2
  for path in "${missing[@]}"; do
    echo "  - $path" >&2
  done
  cat >&2 <<'EOF'

This directory contains the perception server overlay. Sync it into a full
ZeroGrasp checkout on the GPU server first, for example:

  rsync -avzh --progress --exclude="__pycache__" --exclude=".git" \
    /home/ubuntu/RUC-WONE/server/zerograsp_server_bundle/ \
    user@10.42.115.70:~/zyh/ZeroGrasp/

Then build on the GPU server:

  ssh user@10.42.115.70
  cd ~/zyh/ZeroGrasp
  ./docker/build.sh
EOF
  exit 1
fi

echo "[zerograsp] building image: $IMAGE"
exec docker build -t "$IMAGE" -f "$DOCKERFILE" .
