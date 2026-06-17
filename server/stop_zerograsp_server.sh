#!/usr/bin/env bash
set -euo pipefail

NAME="${ZEROGRASP_CONTAINER_NAME:-perception-server}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "[zerograsp] stopped container: $NAME"
