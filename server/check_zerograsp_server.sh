#!/usr/bin/env bash
set -euo pipefail

URL="${PERCEPTION_URL:-${ZEROGRASP_URL:-http://127.0.0.1:9100}}"
TIMEOUT="${ZEROGRASP_CHECK_TIMEOUT_S:-5}"

python3 - "$URL" "$TIMEOUT" <<'PY'
import json
import sys
import urllib.error
import urllib.request

url = sys.argv[1].rstrip("/")
timeout = float(sys.argv[2])
try:
    with urllib.request.urlopen(f"{url}/healthz", timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
except Exception as exc:
    print(f"[zerograsp] healthz failed: {url}/healthz: {exc}", file=sys.stderr)
    sys.exit(1)

print(json.dumps(payload, ensure_ascii=False, indent=2))
if not payload.get("ok", False):
    sys.exit(1)
PY
