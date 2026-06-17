#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash server/start_teleop_follow.sh

Starts the lead/follow teleoperation loop in tmux.

Local four-arm mode:
  bash server/start_airbot_servers.sh all
  bash server/start_teleop_follow.sh

Split host mode:
  # On the lead/master-arm host:
  bash server/start_airbot_servers.sh lead

  # On the robot/follower-arm host:
  bash server/start_airbot_servers.sh execution
  LEAD_URL=<lead-host-ip> bash server/start_teleop_follow.sh

Environment overrides:
  LEAD_URL, FOLLOW_URL
  LEFT_LEAD_URL, RIGHT_LEAD_URL, LEFT_FOLLOW_URL, RIGHT_FOLLOW_URL
  LEFT_LEAD_PORT, RIGHT_LEAD_PORT, LEFT_PORT, RIGHT_PORT
  AIRBOT_TELEOP_SESSION, AIRBOT_DATA_CLIENT_DIR
EOF
}

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

SESSION="${AIRBOT_TELEOP_SESSION:-airbot_teleop}"
DATA_CLIENT_DIR="${AIRBOT_DATA_CLIENT_DIR:-/home/ubuntu/dos_w1/data_collection_client}"

LEFT_LEAD_URL="${LEFT_LEAD_URL:-${LEAD_URL:-localhost}}"
RIGHT_LEAD_URL="${RIGHT_LEAD_URL:-${LEAD_URL:-localhost}}"
LEFT_FOLLOW_URL="${LEFT_FOLLOW_URL:-${FOLLOW_URL:-localhost}}"
RIGHT_FOLLOW_URL="${RIGHT_FOLLOW_URL:-${FOLLOW_URL:-localhost}}"

LEFT_LEAD_PORT="${LEFT_LEAD_PORT:-50050}"
LEFT_PORT="${LEFT_PORT:-50051}"
RIGHT_LEAD_PORT="${RIGHT_LEAD_PORT:-50052}"
RIGHT_PORT="${RIGHT_PORT:-50053}"

if [[ ! -f "$DATA_CLIENT_DIR/test_task_follow.py" ]]; then
  echo "[teleop] missing: $DATA_CLIENT_DIR/test_task_follow.py" >&2
  exit 1
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n follow
tmux set-option -t "$SESSION" mouse on
tmux set-option -t "$SESSION" remain-on-exit on

printf -v run_dir "%q" "$DATA_CLIENT_DIR"
printf -v left_lead_url "%q" "$LEFT_LEAD_URL"
printf -v right_lead_url "%q" "$RIGHT_LEAD_URL"
printf -v left_follow_url "%q" "$LEFT_FOLLOW_URL"
printf -v right_follow_url "%q" "$RIGHT_FOLLOW_URL"
tmux send-keys -t "$SESSION:follow.0" \
  "cd ${run_dir} && source ~/miniconda3/etc/profile.d/conda.sh && conda activate airbot_data && python3 test_task_follow.py --lead-url ${left_lead_url} ${right_lead_url} --follow-url ${left_follow_url} ${right_follow_url} -lp ${LEFT_LEAD_PORT} ${RIGHT_LEAD_PORT} -fp ${LEFT_PORT} ${RIGHT_PORT}" C-m

echo "[teleop] started tmux session: $SESSION"
echo "[teleop] lead:"
echo "[teleop]   left  ${LEFT_LEAD_URL}:${LEFT_LEAD_PORT}"
echo "[teleop]   right ${RIGHT_LEAD_URL}:${RIGHT_LEAD_PORT}"
echo "[teleop] follow:"
echo "[teleop]   left  ${LEFT_FOLLOW_URL}:${LEFT_PORT}"
echo "[teleop]   right ${RIGHT_FOLLOW_URL}:${RIGHT_PORT}"
echo "[teleop] attach logs: tmux attach -t $SESSION"
