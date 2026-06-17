#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash server/check_airbot_servers.sh [execution|all|lead]

Defaults to execution mode unless AIRBOT_SERVER_PORTS is set.

Environment:
  AIRBOT_SERVER_HOST=192.168.x.x
  AIRBOT_SERVER_HOSTS="host_for_port1 host_for_port2 ..."
  AIRBOT_SERVER_PORTS="50050 50052"
EOF
}

MODE="${1:-${AIRBOT_SERVER_MODE:-execution}}"

if [[ -n "${AIRBOT_SERVER_PORTS:-}" ]]; then
  read -r -a PORTS <<<"${AIRBOT_SERVER_PORTS}"
else
  case "$MODE" in
    -h|--help|help)
      usage
      exit 0
      ;;
    exec|follow|execution)
      PORTS=(
        "${LEFT_PORT:-50051}"
        "${RIGHT_PORT:-50053}"
      )
      ;;
    lead|master)
      PORTS=(
        "${LEFT_LEAD_PORT:-50050}"
        "${RIGHT_LEAD_PORT:-50052}"
      )
      ;;
    all|four|4)
      PORTS=(
        "${LEFT_LEAD_PORT:-50050}"
        "${LEFT_PORT:-50051}"
        "${RIGHT_LEAD_PORT:-50052}"
        "${RIGHT_PORT:-50053}"
      )
      ;;
    *)
      echo "[server] unknown mode: $MODE" >&2
      usage >&2
      exit 2
      ;;
  esac
fi

if [[ -n "${AIRBOT_SERVER_HOSTS:-}" ]]; then
  read -r -a HOSTS <<<"${AIRBOT_SERVER_HOSTS}"
  if [[ "${#HOSTS[@]}" -ne "${#PORTS[@]}" ]]; then
    echo "[server] AIRBOT_SERVER_HOSTS count must match ports count" >&2
    exit 2
  fi
else
  HOSTS=()
  for _ in "${PORTS[@]}"; do
    HOSTS+=("${AIRBOT_SERVER_HOST:-127.0.0.1}")
  done
fi

port_open() {
  local host="$1"
  local port="$2"
  python3 - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.25)
try:
    sock.connect((sys.argv[1], int(sys.argv[2])))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}

timeout_s="${AIRBOT_SERVER_WAIT_S:-60}"
for i in $(seq 1 "$timeout_s"); do
  ready=0
  for idx in "${!PORTS[@]}"; do
    if port_open "${HOSTS[$idx]}" "${PORTS[$idx]}"; then
      ready=$((ready + 1))
    fi
  done
  echo "[server] ${i}s: ${ready}/${#PORTS[@]} ports listening"
  if [[ "$ready" -eq "${#PORTS[@]}" ]]; then
    echo "[server] ready"
    exit 0
  fi
  sleep 1
done

echo "[server] timeout waiting for ports:" >&2
for idx in "${!PORTS[@]}"; do
  echo "[server]   ${HOSTS[$idx]}:${PORTS[$idx]}" >&2
done
exit 1
