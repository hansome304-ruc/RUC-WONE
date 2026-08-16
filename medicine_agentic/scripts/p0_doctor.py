#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from medicine_agentic.p0_doctor import run_doctor


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="P0 readiness checks. This command never moves the robot."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also read web-console health and one feedback sample from each arm.",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--left-port", type=int, default=50051)
    parser.add_argument("--right-port", type=int, default=50053)
    args = parser.parse_args()
    report = run_doctor(
        project_root=ROOT,
        live=args.live,
        host=args.host,
        left_port=args.left_port,
        right_port=args.right_port,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
