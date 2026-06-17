#!/usr/bin/env bash
set -euo pipefail

SESSION="${AIRBOT_SERVER_SESSION:-airbot_servers}"
TELEOP_SESSION="${AIRBOT_TELEOP_SESSION:-airbot_teleop}"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux kill-session -t "$TELEOP_SESSION" 2>/dev/null || true
echo "[server] stopped tmux session: $SESSION"
echo "[server] stopped tmux session: $TELEOP_SESSION"
