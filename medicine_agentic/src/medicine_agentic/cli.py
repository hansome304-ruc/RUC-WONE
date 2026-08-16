from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from medicine_agentic.config import load_config
from medicine_agentic.dry_run import DryRunSkillExecutor
from medicine_agentic.workflow import MedicineWorkflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "demo.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run medicine workflows in dry-run mode.")
    parser.add_argument(
        "--workflow",
        choices=("all", "pack", "load", "erect"),
        default="all",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--fail-once",
        action="append",
        default=[],
        metavar="KEY",
        help="Inject one retryable failure, e.g. insert_item:BLISTER:2",
    )
    parser.add_argument("--report-out", type=Path)
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    failures = {key: 1 for key in args.fail_once}
    executor = DryRunSkillExecutor(failures=failures)
    workflow = MedicineWorkflow(config=config, executor=executor)

    runners = {
        "pack": workflow.run_pack,
        "load": workflow.run_load,
        "erect": workflow.run_erect,
    }
    selected = tuple(runners) if args.workflow == "all" else (args.workflow,)
    reports = [runners[name]().to_dict() for name in selected]
    payload = {
        "mode": "dry_run",
        "ok": all(report["ok"] for report in reports),
        "reports": reports,
        "safe_stop_calls": list(executor.safe_stop_calls),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["ok"] else 1


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()

