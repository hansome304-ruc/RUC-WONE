#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LIFT_PYTHON:-/home/ubuntu/hq_v2/SJJ_control/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

PYTHONPATH="$ROOT:${PYTHONPATH:-}" exec "$PYTHON" -m locomotion.lift.cli "$@"
