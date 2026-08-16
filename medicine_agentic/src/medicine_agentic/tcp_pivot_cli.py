from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from medicine_agentic.airbot_readonly import ArmReadError, enum_name, normalize_end_pose
from medicine_agentic.pose_store import validate_joint_position
from medicine_agentic.tcp_pivot import (
    PivotCalibrationError,
    PivotSampleStore,
    solve_pivot_translation,
    summarize_flange_samples,
    validate_sample_document,
    write_accepted_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEFT_SAMPLES = (
    PROJECT_ROOT / "artifacts" / "calibration" / "left_suction_tcp" / "pivot_samples.json"
)
DEFAULT_RIGHT_SAMPLES = (
    PROJECT_ROOT / "artifacts" / "calibration" / "right_gripper_tcp" / "pivot_samples.json"
)
DEFAULT_LEFT_OUTPUT = PROJECT_ROOT / "configs" / "calibration" / "left_suction_tcp.json"
DEFAULT_RIGHT_OUTPUT = PROJECT_ROOT / "configs" / "calibration" / "right_gripper_tcp.json"


class FlangeReadOnly:
    """Connect to one follower and expose feedback methods only."""

    def __init__(
        self,
        *,
        arm_name: str,
        host: str = "localhost",
        port: int,
    ) -> None:
        self.arm_name = arm_name
        self.host = host
        self.port = int(port)
        self.arm: Any | None = None

    def __enter__(self) -> "FlangeReadOnly":
        try:
            from airbot_py.arm import AIRBOTPlay
        except ImportError as exc:  # pragma: no cover - available on dosw1
            raise ArmReadError(
                "airbot_py is unavailable; run inside the dos-w1 Conda environment"
            ) from exc
        arm = AIRBOTPlay(url=self.host, port=self.port)
        try:
            connected = bool(arm.connect())
        except Exception as exc:
            raise ArmReadError(
                f"{self.arm_name} arm connection failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not connected:
            raise ArmReadError(
                f"{self.arm_name} arm connection failed: {self.host}:{self.port}"
            )
        self.arm = arm
        return self

    def __exit__(self, *_args: Any) -> None:
        if self.arm is None:
            return
        for method_name in ("disconnect", "close", "shutdown"):
            method = getattr(self.arm, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass
                break
        self.arm = None

    def read(self) -> dict[str, Any]:
        if self.arm is None:
            raise ArmReadError(f"{self.arm_name} arm is not connected")
        try:
            joints = np.asarray(self.arm.get_joint_pos(), dtype=np.float64)
            position, quaternion = normalize_end_pose(self.arm.get_end_pose())
            state = enum_name(self.arm.get_state())
            mode = enum_name(self.arm.get_control_mode())
        except Exception as exc:
            raise ArmReadError(
                f"failed to read {self.arm_name} flange feedback: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise ArmReadError(
                f"{self.arm_name} arm returned invalid joints: {joints}"
            )
        return {
            "timestamp_ns": time.time_ns(),
            "joint_position_rad": joints.tolist(),
            "flange_position_m": position.tolist(),
            "flange_quaternion_xyzw": quaternion.tolist(),
            "driver_state": state,
            "control_mode": mode,
        }

    def collect(
        self,
        *,
        count: int,
        interval_s: float,
    ) -> list[dict[str, Any]]:
        if count < 3:
            raise ValueError("sample-count must be at least 3")
        if interval_s < 0.0:
            raise ValueError("sample-interval-s must be non-negative")
        result = []
        for index in range(count):
            result.append(self.read())
            if index + 1 < count and interval_s > 0.0:
                time.sleep(interval_s)
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only tool TCP pivot calibration. "
            "This command contains no robot-motion operation."
        )
    )
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument(
        "--samples",
        type=Path,
        default=None,
        help="Defaults to the selected arm's artifact directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to the selected arm's TCP config.",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Defaults to 50051 for left and 50053 for right.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="Read one selected-arm flange feedback sample.")

    capture = subparsers.add_parser(
        "capture",
        help="Append one stationary flange pose while the tool tip stays on the pivot.",
    )
    capture.add_argument("--label", default="")
    capture.add_argument("--sample-count", type=int, default=20)
    capture.add_argument("--sample-interval-s", type=float, default=0.05)
    capture.add_argument("--no-prompt", action="store_true")

    subparsers.add_parser("list", help="List saved pivot poses.")
    subparsers.add_parser("validate", help="Validate the sample document and its hash.")

    solve = subparsers.add_parser(
        "solve",
        help="Solve flange-to-TCP translation and write only an accepted result.",
    )
    solve.add_argument("--dry-run", action="store_true")
    solve.add_argument("--replace", action="store_true")
    solve.add_argument("--expected-output-sha256")
    solve.add_argument("--min-samples", type=int, default=8)
    solve.add_argument("--max-rms-mm", type=float, default=2.0)
    solve.add_argument("--max-residual-mm", type=float, default=5.0)
    solve.add_argument("--max-condition-number", type=float, default=100.0)
    solve.add_argument("--min-normalized-singular-value", type=float, default=0.10)
    solve.add_argument("--min-orientation-span-deg", type=float, default=25.0)
    solve.add_argument("--max-tcp-offset-mm", type=float, default=300.0)
    return parser


def _resolve_arm_defaults(args: argparse.Namespace) -> None:
    if args.port is None:
        args.port = 50051 if args.arm == "left" else 50053
    if args.samples is None:
        args.samples = (
            DEFAULT_LEFT_SAMPLES if args.arm == "left" else DEFAULT_RIGHT_SAMPLES
        )
    if args.output is None:
        args.output = (
            DEFAULT_LEFT_OUTPUT if args.arm == "left" else DEFAULT_RIGHT_OUTPUT
        )


def _reader(args: argparse.Namespace) -> FlangeReadOnly:
    return FlangeReadOnly(
        arm_name=args.arm,
        host=args.host,
        port=args.port,
    )


def _capture(args: argparse.Namespace) -> int:
    if not args.no_prompt:
        response = input(
            f"让 {args.arm} 工具标定尖端保持在同一个固定点，只改变该臂姿态；"
            "停止 Follow、确认机械臂静止后按 Enter，输入 q 取消："
        ).strip()
        if response.lower() == "q":
            raise KeyboardInterrupt
    with _reader(args) as reader:
        raw_samples = reader.collect(
            count=args.sample_count,
            interval_s=args.sample_interval_s,
        )
    summary = summarize_flange_samples(raw_samples)
    if summary["capture_metrics"]["stable"] is not True:
        raise PivotCalibrationError(
            f"{args.arm} flange moved during capture: {summary['capture_metrics']}"
        )
    joint_errors = validate_joint_position(summary["joint_position_rad"])
    if joint_errors:
        raise PivotCalibrationError(
            f"{args.arm} flange pose is outside the joint soft-limit guard: "
            + "; ".join(joint_errors)
        )
    saved = PivotSampleStore(args.samples, arm=args.arm).append(
        summary, label=args.label
    )
    print(
        json.dumps(
            {
                "ok": True,
                "motion_commands_issued": False,
                "sample_id": saved["sample_id"],
                "label": saved["label"],
                "flange_pose_in_base": saved["flange_pose_in_base"],
                "capture_metrics": saved["capture_metrics"],
                "samples": str(args.samples.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _solve(args: argparse.Namespace) -> int:
    store = PivotSampleStore(args.samples, arm=args.arm)
    document = store.load()
    result = solve_pivot_translation(
        document["samples"],
        robot_id=str(document["robot_id"]),
        arm=args.arm,
        tcp_frame=str(document["tcp_frame"]),
        source_samples_sha256=document["content_sha256"],
        min_samples=args.min_samples,
        max_rms_residual_m=args.max_rms_mm / 1000.0,
        max_residual_m=args.max_residual_mm / 1000.0,
        max_condition_number=args.max_condition_number,
        min_normalized_singular_value=args.min_normalized_singular_value,
        min_orientation_span_deg=args.min_orientation_span_deg,
        max_tcp_offset_m=args.max_tcp_offset_mm / 1000.0,
    )
    payload = {
        "ok": bool(result["acceptance"]["accepted"]),
        "motion_commands_issued": False,
        "dry_run": bool(args.dry_run),
        "output_written": False,
        "translation_flange_to_tcp_m": result["flange_to_tcp"]["translation_m"],
        "metrics": result["metrics"],
        "acceptance": result["acceptance"],
        "result_sha256": result["content_sha256"],
    }
    if result["acceptance"]["accepted"] and not args.dry_run:
        write_accepted_result(
            args.output,
            result,
            replace=bool(args.replace),
            expected_current_sha256=args.expected_output_sha256,
        )
        payload["output_written"] = True
        payload["output"] = str(args.output.resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["acceptance"]["accepted"] else 2


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _resolve_arm_defaults(args)
    try:
        if args.command == "doctor":
            with _reader(args) as reader:
                sample = reader.read()
            print(
                json.dumps(
                    {
                        "ok": True,
                        "motion_commands_issued": False,
                        "sample": sample,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "capture":
            return _capture(args)
        store = PivotSampleStore(args.samples, arm=args.arm)
        if args.command == "list":
            document = store.load()
            print(
                json.dumps(
                    {
                        "ok": True,
                        "motion_commands_issued": False,
                        "revision": document["revision"],
                        "content_sha256": document["content_sha256"],
                        "sample_count": len(document["samples"]),
                        "samples": [
                            {
                                "sample_id": sample["sample_id"],
                                "label": sample.get("label", ""),
                                "captured_at": sample.get("captured_at"),
                                "flange_pose_in_base": sample["flange_pose_in_base"],
                            }
                            for sample in document["samples"]
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "validate":
            document = store.load()
            errors = validate_sample_document(document)
            print(
                json.dumps(
                    {
                        "ok": not errors,
                        "motion_commands_issued": False,
                        "sample_count": len(document["samples"]),
                        "errors": errors,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0 if not errors else 2
        if args.command == "solve":
            return _solve(args)
        raise PivotCalibrationError(f"unsupported command: {args.command}")
    except KeyboardInterrupt:
        print("已取消；未写入样本。", file=sys.stderr)
        return 130
    except (
        ArmReadError,
        PivotCalibrationError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
