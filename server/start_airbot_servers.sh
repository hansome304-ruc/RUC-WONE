#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash server/start_airbot_servers.sh [execution|all|lead]

Modes:
  execution  Start only competition/execution arms: can_left, can_right. Default.
  all        Start lead arms and execution arms. Use this before teleoperation.
  lead       Start only lead/master arms.

Environment overrides:
  LEFT_IFACE, RIGHT_IFACE, LEFT_LEAD_IFACE, RIGHT_LEAD_IFACE
  LEFT_PORT, RIGHT_PORT, LEFT_LEAD_PORT, RIGHT_LEAD_PORT
  AIRBOT_SERVER_SESSION, AIRBOT_SERVER_WAIT_S, STOP_AIRBOT_DOCKERS=1
EOF
}

SESSION="${AIRBOT_SERVER_SESSION:-airbot_servers}"
MODE="${1:-${AIRBOT_SERVER_MODE:-execution}}"

LEFT_LEAD_IFACE="${LEFT_LEAD_IFACE:-can_left_lead}"
LEFT_IFACE="${LEFT_IFACE:-can_left}"
RIGHT_LEAD_IFACE="${RIGHT_LEAD_IFACE:-can_right_lead}"
RIGHT_IFACE="${RIGHT_IFACE:-can_right}"

LEFT_LEAD_PORT="${LEFT_LEAD_PORT:-50050}"
LEFT_PORT="${LEFT_PORT:-50051}"
RIGHT_LEAD_PORT="${RIGHT_LEAD_PORT:-50052}"
RIGHT_PORT="${RIGHT_PORT:-50053}"

case "$MODE" in
  -h|--help|help)
    usage
    exit 0
    ;;
  exec|follow|execution)
    MODE="execution"
    SPECS=(
      "left_execution:${LEFT_IFACE}:${LEFT_PORT}"
      "right_execution:${RIGHT_IFACE}:${RIGHT_PORT}"
    )
    ;;
  lead|master)
    MODE="lead"
    SPECS=(
      "left_lead:${LEFT_LEAD_IFACE}:${LEFT_LEAD_PORT}"
      "right_lead:${RIGHT_LEAD_IFACE}:${RIGHT_LEAD_PORT}"
    )
    ;;
  all|four|4)
    MODE="all"
    SPECS=(
      "left_lead:${LEFT_LEAD_IFACE}:${LEFT_LEAD_PORT}"
      "left_execution:${LEFT_IFACE}:${LEFT_PORT}"
      "right_lead:${RIGHT_LEAD_IFACE}:${RIGHT_LEAD_PORT}"
      "right_execution:${RIGHT_IFACE}:${RIGHT_PORT}"
    )
    ;;
  *)
    echo "[server] unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ "${STOP_AIRBOT_DOCKERS:-0}" == "1" ]] && command -v docker >/dev/null 2>&1; then
  docker ps -a | grep -vE "robohmi|CONTAINER" | awk '{print $1}' | xargs -r docker stop >/dev/null 2>&1 || true
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n airbots
tmux set-option -t "$SESSION" mouse on
tmux set-option -t "$SESSION" remain-on-exit on

run_server() {
  local pane="$1"
  local label="$2"
  local iface="$3"
  local port="$4"
  tmux send-keys -t "$pane" \
    "echo '[server] ${label}: ${iface}:${port}'; source ~/miniconda3/etc/profile.d/conda.sh && conda activate airbot && airbot_server -i ${iface} -p ${port}" C-m
}

for ((i = 1; i < ${#SPECS[@]}; i++)); do
  tmux split-window -t "$SESSION:airbots" >/dev/null
done
tmux select-layout -t "$SESSION:airbots" tiled >/dev/null

mapfile -t PANE_IDS < <(tmux list-panes -t "$SESSION:airbots" -F "#{pane_id}")
if [[ "${#PANE_IDS[@]}" -ne "${#SPECS[@]}" ]]; then
  echo "[server] expected ${#SPECS[@]} panes, got ${#PANE_IDS[@]}" >&2
  exit 1
fi

PORTS_TO_CHECK=()
for idx in "${!SPECS[@]}"; do
  IFS=":" read -r label iface port <<<"${SPECS[$idx]}"
  run_server "${PANE_IDS[$idx]}" "$label" "$iface" "$port"
  PORTS_TO_CHECK+=("$port")
  sleep "${AIRBOT_SERVER_START_GAP_S:-1}"
done

echo "[server] started tmux session: $SESSION"
echo "[server] mode: $MODE"
echo "[server] ports: ${PORTS_TO_CHECK[*]}"
echo "[server] attach logs: tmux attach -t $SESSION"

AIRBOT_SERVER_PORTS="${PORTS_TO_CHECK[*]}" "$(dirname "$0")/check_airbot_servers.sh"
