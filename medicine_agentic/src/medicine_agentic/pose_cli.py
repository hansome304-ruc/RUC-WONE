from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from medicine_agentic.airbot_readonly import (
    AirbotReadOnly,
    ArmReadError,
    summarize_arm_samples,
)
from medicine_agentic.pose_store import (
    PoseStore,
    PoseStoreError,
    file_sha256,
    utc_now,
    validate_joint_position,
    validate_pose_document,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE = PROJECT_ROOT / "configs" / "p0_poses.json"
DEFAULT_PLAN = PROJECT_ROOT / "configs" / "p0_pose_plan.json"


def _yes_no(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"yes", "y", "true", "1"}:
        return True
    if normalized in {"no", "n", "false", "0"}:
        return False
    raise argparse.ArgumentTypeError("expected yes or no")


def _pose_hash(pose: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            pose,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only AIRBOT pose teaching and pose-store validation."
    )
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--left-port", type=int, default=50051)
    parser.add_argument("--right-port", type=int, default=50053)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="Read one feedback sample from both arms.")
    subparsers.add_parser("list", help="List recorded paired poses.")

    show = subparsers.add_parser("show", help="Show one recorded pose.")
    show.add_argument("name")

    validate = subparsers.add_parser("validate", help="Validate the complete pose store.")
    validate.add_argument("--all", action="store_true", help="Accepted for CLI readability.")

    approve = subparsers.add_parser(
        "approve",
        help="Record an operator's physical clearance review for one captured pose.",
    )
    approve.add_argument("name")
    approve.add_argument("--note", default="")
    approve.add_argument(
        "--confirm",
        default="",
        help="Non-interactive form must equal 'VALIDATE <name>'.",
    )

    capture = subparsers.add_parser("capture", help="Capture a stable paired pose.")
    _add_capture_arguments(capture)

    teach = subparsers.add_parser("teach", help="Walk through a pose teaching plan.")
    teach.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    teach.add_argument("--replace-existing", action="store_true")
    teach.add_argument("--sample-count", type=int, default=20)
    teach.add_argument("--sample-interval-s", type=float, default=0.05)
    teach.add_argument(
        "--manual-validate",
        action="store_true",
        help=(
            "After each capture, ask the on-site operator to confirm physical "
            "clearance while the robot remains at that pose."
        ),
    )
    return parser


def _add_capture_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("name")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--scene-id", default="medicine_table_v1")
    parser.add_argument("--base-docked", type=_yes_no, default=True)
    parser.add_argument("--lift-height-mm", type=float, default=0.0)
    parser.add_argument("--left-tool-id", default="suction_v1")
    parser.add_argument("--right-tool-id", default="gripper_v1")
    parser.add_argument(
        "--left-tcp-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "calibration" / "left_suction_tcp.json",
    )
    parser.add_argument(
        "--right-tcp-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "calibration" / "right_gripper_tcp.json",
    )
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--sample-interval-s", type=float, default=0.05)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--expected-pose-sha256")
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Capture immediately; intended for scripted use after an external confirmation.",
    )


def _reader(args: argparse.Namespace) -> AirbotReadOnly:
    return AirbotReadOnly(
        host=args.host,
        ports={"left": args.left_port, "right": args.right_port},
    )


def _capture(
    args: argparse.Namespace,
    *,
    instruction: str | None = None,
) -> dict[str, Any]:
    message = instruction if instruction is not None else args.instruction
    if message:
        print(message)
    if not args.no_prompt:
        response = input(
            f"遥操到姿态 {args.name!r}，停止 Follow 并确认双臂静止后按 Enter；输入 q 取消："
        ).strip()
        if response.lower() == "q":
            raise KeyboardInterrupt
    with _reader(args) as reader:
        pairs = reader.collect_pairs(
            count=args.sample_count,
            interval_s=args.sample_interval_s,
        )
        arms: dict[str, Any] = {}
        validation_errors: list[str] = []
        for arm_name in ("left", "right"):
            samples = [pair["arms"][arm_name] for pair in pairs]
            summary = summarize_arm_samples(samples)
            arms[arm_name] = summary
            if not summary["capture_metrics"]["stable"]:
                validation_errors.append(
                    f"{arm_name} arm moved during capture: "
                    f"{summary['capture_metrics']}"
                )
            validation_errors.extend(
                f"{arm_name}: {error}"
                for error in validate_joint_position(summary["joint_position_rad"])
            )
        maximum_skew_ms = float(max(pair["paired_sample_skew_ms"] for pair in pairs))
        if maximum_skew_ms > 50.0:
            validation_errors.append(
                f"paired sample skew {maximum_skew_ms:.1f}ms exceeds 50ms"
            )
        if validation_errors:
            raise PoseStoreError("capture rejected: " + "; ".join(validation_errors))

        pose = {
            "kind": "paired_joint_pose",
            "status": "draft",
            "captured_at": utc_now(),
            "source": "teleop_snapshot",
            "instruction": message or "",
            "scene_id": args.scene_id,
            "context": {
                "base_docked": bool(args.base_docked),
                "lift_height_mm": float(args.lift_height_mm),
            },
            "tooling": {
                "left_tool_id": args.left_tool_id,
                "right_tool_id": args.right_tool_id,
                "left_tcp_config": str(args.left_tcp_config.expanduser()),
                "right_tcp_config": str(args.right_tcp_config.expanduser()),
                "left_tcp_config_sha256": file_sha256(args.left_tcp_config),
                "right_tcp_config_sha256": file_sha256(args.right_tcp_config),
            },
            "arms": arms,
            "validation": {
                "stable": True,
                "inside_joint_soft_limits": True,
                "maximum_paired_sample_skew_ms": maximum_skew_ms,
                "validation_level": "feedback_and_soft_limits_only",
                "collision_free": "unproven",
                "manual_transition_validation_required": True,
            },
        }
        store = PoseStore(args.store)
        saved = store.upsert_pose(
            args.name,
            pose,
            replace=bool(args.replace),
            expected_pose_sha256=args.expected_pose_sha256,
        )
        saved_pose = saved["poses"][args.name]
        print(
            json.dumps(
                {
                    "ok": True,
                    "name": args.name,
                    "revision": saved["revision"],
                    "pose_sha256": _pose_hash(saved_pose),
                    "store_sha256": saved["content_sha256"],
                    "store": str(args.store.resolve()),
                    "left_joints_rad": saved_pose["arms"]["left"]["joint_position_rad"],
                    "right_joints_rad": saved_pose["arms"]["right"]["joint_position_rad"],
                    "maximum_paired_sample_skew_ms": maximum_skew_ms,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return saved_pose


def _approve_pose(
    store: PoseStore,
    name: str,
    *,
    note: str = "",
    confirmation: str = "",
) -> dict[str, Any]:
    payload = store.load()
    existing = payload["poses"].get(name)
    if existing is None:
        raise PoseStoreError(f"pose not found: {name}")
    if existing.get("validation", {}).get("stable") is not True:
        raise PoseStoreError(
            f"pose {name!r} was not captured as stable and cannot be approved"
        )
    expected_phrase = f"VALIDATE {name}"
    if not confirmation:
        print(
            "该操作只记录人工验收，不做碰撞计算也不移动机械臂。\n"
            "请在现场确认：双臂/工具与桌面、箱体、机器人本体均有安全间隙，"
            "底盘已固定，且该姿态与当前工具配置一致。"
        )
        confirmation = input(f"确认后输入 {expected_phrase}；直接回车取消：").strip()
    if confirmation != expected_phrase:
        raise PoseStoreError("manual validation was not confirmed; pose remains draft")

    updated_pose = copy.deepcopy(existing)
    updated_pose["status"] = "validated"
    validation = updated_pose.setdefault("validation", {})
    validation.update(
        {
            "collision_free": True,
            "validation_level": "manual_scene_clearance_at_taught_pose",
            "manual_transition_validation_required": True,
            "manual_validated_at": utc_now(),
            "manual_validation_note": note,
            "automatic_full_link_collision_check": False,
        }
    )
    saved = store.upsert_pose(
        name,
        updated_pose,
        replace=True,
        expected_pose_sha256=_pose_hash(existing),
    )
    approved = saved["poses"][name]
    result = {
        "ok": True,
        "name": name,
        "status": approved["status"],
        "pose_sha256": _pose_hash(approved),
        "store_revision": saved["revision"],
        "store_sha256": saved["content_sha256"],
        "validation": approved["validation"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return approved


def _teach(args: argparse.Namespace) -> int:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    poses = plan.get("poses")
    if not isinstance(poses, list) or not poses:
        raise PoseStoreError(f"teaching plan has no poses: {args.plan}")
    print(f"教学计划：{plan.get('name', args.plan.stem)}，共 {len(poses)} 个双臂姿态")
    for index, entry in enumerate(poses, start=1):
        name = str(entry["name"])
        instruction = str(entry.get("instruction", ""))
        existing = PoseStore(args.store).load()["poses"].get(name)
        if existing is not None and not args.replace_existing:
            print(f"[{index}/{len(poses)}] {name}: 已存在，跳过")
            continue
        print(f"\n[{index}/{len(poses)}] {name}")
        capture_args = argparse.Namespace(**vars(args))
        capture_args.name = name
        capture_args.instruction = instruction
        capture_args.scene_id = str(entry.get("scene_id", plan.get("scene_id", "medicine_table_v1")))
        capture_args.base_docked = bool(entry.get("base_docked", True))
        capture_args.lift_height_mm = float(entry.get("lift_height_mm", 0.0))
        capture_args.left_tool_id = str(entry.get("left_tool_id", "suction_v1"))
        capture_args.right_tool_id = str(entry.get("right_tool_id", "gripper_v1"))
        capture_args.left_tcp_config = PROJECT_ROOT / "configs" / "calibration" / "left_suction_tcp.json"
        capture_args.right_tcp_config = PROJECT_ROOT / "configs" / "calibration" / "right_gripper_tcp.json"
        capture_args.replace = existing is not None
        capture_args.expected_pose_sha256 = (
            _pose_hash(existing) if existing is not None else None
        )
        capture_args.no_prompt = False
        _capture(capture_args, instruction=instruction)
        if args.manual_validate:
            _approve_pose(
                PoseStore(args.store),
                name,
                note="approved interactively during pose teaching",
            )
    return 0


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            with _reader(args) as reader:
                pair = reader.capture_pair()
            print(json.dumps({"ok": True, **pair}, ensure_ascii=False, indent=2))
            return 0
        store = PoseStore(args.store)
        if args.command == "list":
            payload = store.load()
            rows = []
            for name, pose in sorted(payload["poses"].items()):
                rows.append(
                    {
                        "name": name,
                        "status": pose.get("status"),
                        "captured_at": pose.get("captured_at"),
                        "pose_sha256": _pose_hash(pose),
                        "collision_free": pose.get("validation", {}).get("collision_free"),
                    }
                )
            print(
                json.dumps(
                    {
                        "ok": True,
                        "store": str(args.store.resolve()),
                        "revision": payload["revision"],
                        "poses": rows,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "show":
            payload = store.load()
            pose = payload["poses"].get(args.name)
            if pose is None:
                raise PoseStoreError(f"pose not found: {args.name}")
            print(
                json.dumps(
                    {
                        "name": args.name,
                        "pose_sha256": _pose_hash(pose),
                        "pose": pose,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "validate":
            payload = store.load()
            errors = validate_pose_document(payload)
            print(
                json.dumps(
                    {
                        "ok": not errors,
                        "store": str(args.store.resolve()),
                        "pose_count": len(payload["poses"]),
                        "errors": errors,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0 if not errors else 2
        if args.command == "approve":
            _approve_pose(
                store,
                args.name,
                note=args.note,
                confirmation=args.confirm,
            )
            return 0
        if args.command == "capture":
            _capture(args)
            return 0
        if args.command == "teach":
            return _teach(args)
        raise PoseStoreError(f"unsupported command: {args.command}")
    except KeyboardInterrupt:
        print("已取消；未写入姿态。", file=sys.stderr)
        return 130
    except (ArmReadError, PoseStoreError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
