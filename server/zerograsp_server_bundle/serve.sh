#!/usr/bin/env bash
# Run inside the zerograsp container.
#
# Single-worker uvicorn — there's only one GPU and the server already
# serialises GPU access with an asyncio lock. Don't bump --workers.
set -euo pipefail

cd /opt/app
export PYTHONPATH="/opt/app:${PYTHONPATH:-}"

exec uvicorn perception_server.server:app \
    --host "${SERVER_HOST:-0.0.0.0}" \
    --port "${SERVER_PORT:-9100}" \
    --workers 1 \
    --log-level "${LOG_LEVEL:-info}" \
    --no-access-log
