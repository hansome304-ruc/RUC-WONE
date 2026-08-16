#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec conda run --no-capture-output -n dos-w1 \
  env PYTHONPATH="${PROJECT_ROOT}/src" \
  python "${PROJECT_ROOT}/scripts/act_dataset.py" "$@"
