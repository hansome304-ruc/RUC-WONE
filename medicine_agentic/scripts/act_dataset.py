#!/usr/bin/env python3
"""Inspect and validate the ACT episodes produced by the 8899 console."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from medicine_agentic.act_dataset import discover_ready, validate_episode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "recordings" / "act" / "finalized"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--no-video", action="store_true", help="skip MP4 frame-count check")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "episode",
        nargs="?",
        type=Path,
        help="one episode directory; omit to validate every READY episode under --root",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    episodes = [args.episode] if args.episode else discover_ready(args.root)
    results = [
        validate_episode(episode, verify_video=not args.no_video) for episode in episodes
    ]
    payload = {
        "root": str(args.root.expanduser().resolve()),
        "episode_count": len(results),
        "valid_count": sum(result.valid for result in results),
        "results": [result.as_dict() for result in results],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif not results:
        print(f"No READY episodes found under {payload['root']}")
    else:
        for result in results:
            state = "OK" if result.valid else "INVALID"
            print(f"[{state}] {result.path}")
            for error in result.errors:
                print(f"  - {error}")
        print(f"{payload['valid_count']}/{payload['episode_count']} valid")
    return 0 if all(result.valid for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
