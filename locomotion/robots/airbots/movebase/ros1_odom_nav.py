from __future__ import annotations

import argparse
import math
import sys
import time

from robots.airbots.movebase import MoveBase, MoveBaseConfig, Velocity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Short-range odometry waypoint control. No map, no obstacle avoidance."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ros-master-uri", default="http://192.168.31.7:11311/")
    common.add_argument("--host-ip", default=None)
    common.add_argument("--topic", default="/cmd_vel")
    common.add_argument("--odom-topic", default="/odom")
    common.add_argument("--rate", type=float, default=10.0)
    common.add_argument("--timeout", type=float, default=20.0)
    common.add_argument("--pos-tol", type=float, default=0.03)
    common.add_argument("--yaw-tol", type=float, default=0.05)
    common.add_argument("--max-speed", type=float, default=0.08)
    common.add_argument("--max-yaw-speed", type=float, default=0.25)
    common.add_argument("--kp-linear", type=float, default=0.6)
    common.add_argument("--kp-angular", type=float, default=1.2)
    common.add_argument(
        "--holonomic",
        action="store_true",
        help="Allow linear.y commands. Leave off for differential/non-holonomic bases.",
    )
    common.add_argument("--yes", action="store_true", help="Required for motion.")

    rel = sub.add_parser("relative", parents=[common])
    rel.add_argument("--dx", type=float, default=0.0, help="Forward displacement in current base frame, metres.")
    rel.add_argument("--dy", type=float, default=0.0, help="Left displacement in current base frame, metres.")
    rel.add_argument("--dyaw", type=float, default=0.0, help="Yaw change. Radians unless --deg is set.")
    rel.add_argument("--deg", action="store_true", help="Interpret --dyaw as degrees.")

    absolute = sub.add_parser("to-odom", parents=[common])
    absolute.add_argument("--x", type=float, required=True)
    absolute.add_argument("--y", type=float, required=True)
    absolute.add_argument("--yaw", type=float, required=True, help="Yaw in odom. Radians unless --deg is set.")
    absolute.add_argument("--deg", action="store_true", help="Interpret --yaw as degrees.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.yes:
        raise SystemExit("Refusing to move without --yes. This controller has no map or obstacle avoidance.")
    if args.rate <= 0:
        raise SystemExit("--rate must be positive")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")

    config = MoveBaseConfig(
        backend="ros1_cmd_vel",
        topic=args.topic,
        odom_topic=args.odom_topic,
        ros_master_uri=args.ros_master_uri,
        host_ip=args.host_ip,
        publish_rate_hz=args.rate,
        max_linear_mps=args.max_speed,
        max_angular_radps=args.max_yaw_speed,
    )
    with MoveBase(config) as base:
        start = base.get_odometry()
        if args.command == "relative":
            dyaw = math.radians(args.dyaw) if args.deg else args.dyaw
            target_x, target_y = transform_relative(start.x, start.y, start.yaw, args.dx, args.dy)
            target_yaw = normalize_angle(start.yaw + dyaw)
        else:
            target_x = args.x
            target_y = args.y
            target_yaw = normalize_angle(math.radians(args.yaw) if args.deg else args.yaw)

        print(
            "start odom: "
            f"x={start.x:.3f} y={start.y:.3f} yaw={start.yaw:.3f}rad"
        )
        print(
            "target odom: "
            f"x={target_x:.3f} y={target_y:.3f} yaw={target_yaw:.3f}rad"
        )
        try:
            run_to_target(base, args, target_x, target_y, target_yaw)
        finally:
            base.stop()

        end = base.get_odometry()
        print(f"end odom: x={end.x:.3f} y={end.y:.3f} yaw={end.yaw:.3f}rad")
    return 0


def run_to_target(base: MoveBase, args: argparse.Namespace, target_x: float, target_y: float, target_yaw: float) -> None:
    deadline = time.monotonic() + args.timeout
    period = 1.0 / args.rate

    while time.monotonic() < deadline:
        state = base.get_odometry()
        dx = target_x - state.x
        dy = target_y - state.y
        dist = math.hypot(dx, dy)
        if dist <= args.pos_tol:
            break

        if args.holonomic:
            body_x = math.cos(state.yaw) * dx + math.sin(state.yaw) * dy
            body_y = -math.sin(state.yaw) * dx + math.cos(state.yaw) * dy
            vx = clamp(args.kp_linear * body_x, -args.max_speed, args.max_speed)
            vy = clamp(args.kp_linear * body_y, -args.max_speed, args.max_speed)
            desired_heading = math.atan2(dy, dx)
            yaw_err = normalize_angle(desired_heading - state.yaw)
            wz = clamp(args.kp_angular * yaw_err, -args.max_yaw_speed, args.max_yaw_speed)
        else:
            desired_heading = math.atan2(dy, dx)
            yaw_err = normalize_angle(desired_heading - state.yaw)
            if abs(yaw_err) > 0.35:
                vx = 0.0
            else:
                vx = clamp(args.kp_linear * dist, 0.0, args.max_speed)
            vy = 0.0
            wz = clamp(args.kp_angular * yaw_err, -args.max_yaw_speed, args.max_yaw_speed)

        base.send_velocity(Velocity(x=vx, y=vy, yaw=wz))
        time.sleep(period)

    rotate_to_yaw(base, args, target_yaw, deadline)


def rotate_to_yaw(base: MoveBase, args: argparse.Namespace, target_yaw: float, deadline: float) -> None:
    period = 1.0 / args.rate
    while time.monotonic() < deadline:
        state = base.get_odometry()
        yaw_err = normalize_angle(target_yaw - state.yaw)
        if abs(yaw_err) <= args.yaw_tol:
            return
        wz = clamp(args.kp_angular * yaw_err, -args.max_yaw_speed, args.max_yaw_speed)
        base.send_velocity(Velocity(yaw=wz))
        time.sleep(period)
    raise TimeoutError("Timed out before reaching odom waypoint")


def transform_relative(x: float, y: float, yaw: float, dx: float, dy: float) -> tuple[float, float]:
    return (
        x + dx * math.cos(yaw) - dy * math.sin(yaw),
        y + dx * math.sin(yaw) + dy * math.cos(yaw),
    )


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
