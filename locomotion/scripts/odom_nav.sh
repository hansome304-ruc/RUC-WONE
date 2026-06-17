#!/usr/bin/env bash
set -euo pipefail

LOCOMOTION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${LOCOMOTION_ROOT}:${PYTHONPATH:-}"
exec /usr/bin/python3 -m robots.airbots.movebase.ros1_odom_nav "$@"
