#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PACKAGING_PORT:-8899}"
BIND="${PACKAGING_BIND:-127.0.0.1}"

if ss -H -ltn "sport = :${PORT}" | grep -q .; then
  echo "Refusing to start: TCP port ${PORT} is already occupied." >&2
  exit 2
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is unavailable; initialize Conda before running this script." >&2
  exit 2
fi

echo "Starting the private collection console on http://${BIND}:${PORT}"
exec conda run --no-capture-output -n dos-w1 \
  env PYTHONPATH="${PROJECT_ROOT}/src" \
  python "${PROJECT_ROOT}/scripts/packaging_console.py" \
  --bind "${BIND}" --port "${PORT}"
