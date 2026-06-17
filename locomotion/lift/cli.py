from __future__ import annotations

import argparse
import json
import sys

from locomotion.lift.lift import (
    DEFAULT_BAUD,
    DEFAULT_MAX_POSITION_MM,
    DEFAULT_MIN_POSITION_MM,
    Lift,
    LiftConfig,
    format_packet,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Control the SJJ/TYC lift over serial.")
    parser.add_argument(
        "command",
        choices=["list", "status", "sniff", "stop", "up", "down", "goto"],
    )
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--position", type=int, help="Target position in mm for goto.")
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--check-timeout", type=float, default=1.0)
    parser.add_argument("--min-position", type=int, default=DEFAULT_MIN_POSITION_MM)
    parser.add_argument("--max-position", type=int, default=DEFAULT_MAX_POSITION_MM)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--skip-feedback-check", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Required for motion commands.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        ports = Lift.list_ports()
        if args.json:
            print(json.dumps([{"device": p[0], "description": p[1], "hwid": p[2]} for p in ports], indent=2))
        elif not ports:
            print("No serial ports found.")
            return 1
        else:
            for device, description, hwid in ports:
                print(f"{device}\t{description}\t{hwid}")
        return 0

    config = LiftConfig(
        port=args.port,
        baud=args.baud,
        min_position_mm=args.min_position,
        max_position_mm=args.max_position,
        feedback_timeout_s=args.check_timeout,
    )
    lift = Lift(config)
    if args.port is None:
        print(f"using default port: {lift.port}")

    if args.command == "status":
        packets = lift.read_status(timeout_s=args.timeout)
        if args.json:
            print(
                json.dumps(
                    [
                        {"position_mm": p.position_mm, "raw": format_packet(p.raw)}
                        for p in packets
                    ],
                    indent=2,
                )
            )
        else:
            for packet in packets:
                print(f"position={packet.position_mm}mm raw={format_packet(packet.raw)}")
        if not packets:
            print(f"No position packet received from {lift.port} within {args.timeout:.1f}s.")
            return 1
        return 0

    if args.command == "sniff":
        chunks = lift.sniff_raw(timeout_s=args.timeout)
        if args.json:
            print(json.dumps([format_packet(chunk) for chunk in chunks], indent=2))
        else:
            for chunk in chunks:
                print("raw:", format_packet(chunk))
        if not chunks:
            print(f"No raw bytes received from {lift.port} within {args.timeout:.1f}s.")
            return 1
        return 0

    if args.command != "stop" and not args.yes:
        raise SystemExit(
            "Refusing to move without --yes. Check clearance and keep stop reachable."
        )

    check_feedback = not args.skip_feedback_check
    if args.command == "stop":
        packet = lift.stop()
    elif args.command == "up":
        packet = lift.up(check_feedback=check_feedback)
    elif args.command == "down":
        packet = lift.down(check_feedback=check_feedback)
    else:
        if args.position is None:
            raise SystemExit("--position is required for goto")
        packet = lift.goto(args.position, check_feedback=check_feedback)

    print("sent:", format_packet(packet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
