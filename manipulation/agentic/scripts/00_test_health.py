#!/usr/bin/env python3
"""Probe the perception server's /healthz."""
from __future__ import annotations

import logging
import sys

from agentic_grasp.perception_client import PerceptionClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    client = PerceptionClient()
    print(f"perception url: {client.url}")
    try:
        info = client.healthz()
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1
    print("ok response:", info)
    if info.get("stub"):
        print("note: server is in STUB mode — every /segment will return a fake bbox")
    return 0


if __name__ == "__main__":
    sys.exit(main())
