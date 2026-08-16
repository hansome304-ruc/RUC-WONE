#!/usr/bin/env python3
"""Build canonical three-camera ACT HDF5 training datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from medicine_agentic.act_export import StationaryDedupConfig, prepare_training_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PROJECT_ROOT / "recordings" / "act" / "finalized",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "recordings" / "act" / "processed",
    )
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="atomically replace an existing generated output directory",
    )
    parser.add_argument(
        "--no-stationary-dedup",
        action="store_true",
        help="disable conservative compression of sustained stationary intervals",
    )
    parser.add_argument("--stationary-joint-tolerance-rad", type=float, default=2.5e-3)
    parser.add_argument("--stationary-gripper-tolerance", type=float, default=5e-4)
    parser.add_argument("--stationary-min-duration-seconds", type=float, default=0.3)
    parser.add_argument("--stationary-keep-every-n-frames", type=int, default=5)
    parser.add_argument("--stationary-anchor-multiplier", type=float, default=2.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    stationary_dedup = StationaryDedupConfig(
        enabled=not args.no_stationary_dedup,
        joint_tolerance_rad=args.stationary_joint_tolerance_rad,
        gripper_tolerance=args.stationary_gripper_tolerance,
        min_duration_seconds=args.stationary_min_duration_seconds,
        keep_every_n_frames=args.stationary_keep_every_n_frames,
        anchor_multiplier=args.stationary_anchor_multiplier,
    )
    result = prepare_training_dataset(
        args.source_root,
        args.output_root,
        image_width=args.image_width,
        image_height=args.image_height,
        replace=args.replace,
        stationary_dedup=stationary_dedup,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
