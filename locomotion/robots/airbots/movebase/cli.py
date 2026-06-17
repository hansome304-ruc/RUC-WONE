from __future__ import annotations

import argparse
import json
import sys

from robots.airbots.movebase import MoveBase, MoveBaseConfig, Velocity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Control AIRBOT mobile base via /cmd_vel.")
    parser.add_argument(
        "command",
        choices=[
            "doctor",
            "state",
            "stop",
            "forward",
            "backward",
            "left",
            "right",
            "strafe-left",
            "strafe-right",
            "turn-left",
            "turn-right",
            "raw",
        ],
    )
    parser.add_argument(
        "--backend",
        default="ros1_cmd_vel",
        choices=["ros1_cmd_vel", "ros2_cmd_vel"],
    )
    parser.add_argument("--ros-master-uri", default="http://192.168.31.7:11311/")
    parser.add_argument("--host-ip", default=None)
    parser.add_argument("--topic", default="/cmd_vel")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--speed", type=float, default=0.05)
    parser.add_argument("--yaw-speed", type=float, default=0.15)
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--max-linear", type=float, default=0.20)
    parser.add_argument("--max-angular", type=float, default=0.50)
    parser.add_argument("--wait-subscriber", type=float, default=2.0)
    parser.add_argument(
        "--no-require-subscriber",
        action="store_true",
        help="Publish even if no /cmd_vel subscriber is discovered.",
    )
    parser.add_argument("--yes", action="store_true", help="Required for non-zero motion.")
    return parser


def velocity_from_args(args: argparse.Namespace) -> Velocity:
    if args.command in {"doctor", "state", "stop"}:
        return Velocity()
    if args.command == "forward":
        return Velocity(x=args.speed)
    if args.command == "backward":
        return Velocity(x=-args.speed)
    if args.command in {"left", "strafe-left"}:
        return Velocity(y=args.speed)
    if args.command in {"right", "strafe-right"}:
        return Velocity(y=-args.speed)
    if args.command == "turn-left":
        return Velocity(yaw=args.yaw_speed)
    if args.command == "turn-right":
        return Velocity(yaw=-args.yaw_speed)
    return Velocity(x=args.x, y=args.y, yaw=args.yaw)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.duration < 0:
        raise SystemExit("--duration must be non-negative")
    moving = args.command != "stop" and any(
        abs(value) > 1e-9 for value in (velocity_from_args(args).x, velocity_from_args(args).y, velocity_from_args(args).yaw)
    )
    if moving and not args.yes:
        raise SystemExit(
            "Refusing to move without --yes. Keep E-stop reachable, then rerun with --yes."
        )

    config = MoveBaseConfig(
        backend=args.backend,
        topic=args.topic,
        odom_topic=args.odom_topic,
        ros_master_uri=args.ros_master_uri,
        host_ip=args.host_ip,
        publish_rate_hz=args.rate,
        wait_subscriber_timeout_s=args.wait_subscriber,
        require_subscriber=not args.no_require_subscriber,
        max_linear_mps=args.max_linear,
        max_angular_radps=args.max_angular,
    )
    with MoveBase(config) as base:
        if args.command == "doctor":
            print(json.dumps(summarize_diagnostics(base.get_diagnostics()), ensure_ascii=True, indent=2))
        elif args.command == "state":
            state = base.get_odometry()
            print(
                json.dumps(
                    {
                        "stamp": state.stamp,
                        "frame_id": state.frame_id,
                        "child_frame_id": state.child_frame_id,
                        "pose": {
                            "x": state.x,
                            "y": state.y,
                            "z": state.z,
                            "yaw": state.yaw,
                        },
                        "velocity": {
                            "x": state.vx,
                            "y": state.vy,
                            "yaw": state.wz,
                        },
                    },
                    ensure_ascii=True,
                    indent=2,
                )
            )
        elif args.command == "stop":
            base.stop()
        else:
            base.move_at_velocity(velocity_from_args(args), duration_s=args.duration)
    return 0


def summarize_diagnostics(raw: dict[str, str]) -> dict[str, object]:
    interesting_keys = {
        "current_map_name",
        "map_state",
        "initialed",
        "confidence",
        "get_warn",
        "laser_hz_low",
        "nav_laser_low_pub_hz",
        "no_laser_received",
        "longtime_no_valid_match",
        "moveAccess",
        "stop_reason",
        "nav_model",
        "collision_shutdown",
        "collision_shutdown_",
        "collision_type",
        "CollisionWarning",
    }
    summary: dict[str, object] = {}
    for topic, value in raw.items():
        if value.startswith("ERROR:"):
            summary[topic] = value
            continue
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            summary[topic] = value
            continue
        if isinstance(parsed, dict):
            compact = {key: parsed[key] for key in interesting_keys if key in parsed}
            if topic == "/collision_state" and "collision_info" in parsed:
                compact["collision_info"] = parsed["collision_info"]
            summary[topic] = compact or parsed
        else:
            summary[topic] = parsed
    return summary


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
