"""CLI for the P0 Task-1 planning-only state machine."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from medicine_agentic.p0_task1_plan import (
    P0Task1DryRunPlanner,
    PlanConfigError,
    load_plan_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "p0_task1_plan.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an auditable P0 Task-1 plan. This command cannot connect "
            "to or control a robot, camera, or suction output."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--pose-store",
        type=Path,
        help="Override the read-only pose document input.",
    )
    parser.add_argument(
        "--detection-report",
        type=Path,
        help="Override the offline task1_box detection.json input.",
    )
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--report-out", type=Path)
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_plan_config(args.config)
        if args.pose_store is not None:
            config = replace(
                config, pose_store_path=args.pose_store.expanduser().resolve()
            )
        if args.detection_report is not None:
            config = replace(
                config,
                detection_report_path=args.detection_report.expanduser().resolve(),
            )
        if args.log_root is not None:
            config = replace(config, log_root=args.log_root.expanduser().resolve())
        report = P0Task1DryRunPlanner(config, run_id=args.run_id).run()
        payload = report.to_dict()
        exit_code = 0 if report.ready else 2
    except (OSError, ValueError, PlanConfigError) as exc:
        payload = {
            "schema_version": 1,
            "mode": "dry_run",
            "status": "BLOCKED",
            "ready": False,
            "task_physically_completed": False,
            "failure_code": "CONFIG_INVALID",
            "message": f"{type(exc).__name__}: {exc}",
            "safety_accounting": {
                "motion_commands_issued": 0,
                "suction_commands_issued": 0,
                "camera_connections_opened": 0,
                "robot_connections_opened": 0,
            },
        }
        exit_code = 2

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(text + "\n", encoding="utf-8")
    return exit_code


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()

