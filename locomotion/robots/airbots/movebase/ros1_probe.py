from __future__ import annotations

import argparse
import json
import socket
import sys
import xmlrpc.client
from dataclasses import dataclass
from typing import Any

from robots.airbots.movebase.backends.ros1_cmd_vel import (
    read_odom_once,
    read_string_once,
)


DEFAULT_MASTER_URI = "http://192.168.31.7:11311/"
INTERESTING_WORDS = (
    "scan",
    "laser",
    "lidar",
    "map",
    "slam",
    "amcl",
    "localiz",
    "nav",
    "goal",
    "move_base",
    "cmd_vel",
    "odom",
    "tf",
    "costmap",
    "path",
    "pose",
    "collision",
)
DIAGNOSTIC_TOPICS = (
    "/map_info_s",
    "/localization_state",
    "/localization_warn",
    "/sensor_warn",
    "/collision_state",
    "/nav_state_info",
    "/func_state",
    "/autocharge_state",
)


@dataclass(frozen=True)
class TopicInfo:
    name: str
    type: str | None = None
    publishers: tuple[str, ...] = ()
    subscribers: tuple[str, ...] = ()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    socket.setdefaulttimeout(args.timeout)

    try:
        report = probe_ros1_master(args.ros_master_uri, timeout_s=args.timeout)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human_report(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the base ROS1 master without rostopic.")
    parser.add_argument("--ros-master-uri", default=DEFAULT_MASTER_URI)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--json", action="store_true")
    return parser


def probe_ros1_master(master_uri: str, timeout_s: float = 2.0) -> dict[str, Any]:
    master = xmlrpc.client.ServerProxy(master_uri, allow_none=True)
    caller_id = "/ruc_wone_base_probe"

    code, msg, state = master.getSystemState(caller_id)
    if code != 1:
        raise RuntimeError(f"getSystemState failed: {msg}")
    publishers_raw, subscribers_raw, services_raw = state

    type_by_topic: dict[str, str] = {}
    try:
        code, msg, topics = master.getPublishedTopics(caller_id, "/")
        if code == 1:
            type_by_topic = {name: typ for name, typ in topics}
    except Exception:
        type_by_topic = {}

    publishers = {name: tuple(nodes) for name, nodes in publishers_raw}
    subscribers = {name: tuple(nodes) for name, nodes in subscribers_raw}
    all_topic_names = sorted(set(publishers) | set(subscribers) | set(type_by_topic))
    topics = [
        TopicInfo(
            name=name,
            type=type_by_topic.get(name),
            publishers=publishers.get(name, ()),
            subscribers=subscribers.get(name, ()),
        )
        for name in all_topic_names
    ]

    diagnostics: dict[str, Any] = {}
    for topic in DIAGNOSTIC_TOPICS:
        if topic not in publishers:
            continue
        try:
            value = read_string_once(master_uri, topic=topic, caller_id=caller_id + "_diag", timeout_s=timeout_s)
            diagnostics[topic] = _parse_jsonish(value)
        except Exception as exc:
            diagnostics[topic] = f"ERROR: {exc}"

    odom: dict[str, Any] | str | None = None
    if "/odom" in publishers:
        try:
            state = read_odom_once(master_uri, topic="/odom", caller_id=caller_id + "_odom", timeout_s=timeout_s)
            odom = {
                "frame_id": state.frame_id,
                "child_frame_id": state.child_frame_id,
                "pose": {"x": state.x, "y": state.y, "yaw": state.yaw},
                "velocity": {"x": state.vx, "y": state.vy, "yaw": state.wz},
            }
        except Exception as exc:
            odom = f"ERROR: {exc}"

    return {
        "ros_master_uri": master_uri,
        "summary": summarize_topics(topics, services_raw),
        "topics": [topic.__dict__ for topic in topics],
        "services": [{"name": name, "providers": providers} for name, providers in sorted(services_raw)],
        "diagnostics": diagnostics,
        "odom": odom,
    }


def summarize_topics(topics: list[TopicInfo], services_raw: list[list[Any]]) -> dict[str, list[str]]:
    by_name = {topic.name: topic for topic in topics}

    def names_matching(*words: str) -> list[str]:
        out = []
        for topic in topics:
            haystack = f"{topic.name} {topic.type or ''}".lower()
            if any(word in haystack for word in words):
                out.append(topic.name)
        return out

    services = sorted(name for name, _providers in services_raw)
    return {
        "laser_candidates": names_matching("scan", "laser", "lidar"),
        "map_candidates": names_matching("map", "grid"),
        "localization_candidates": names_matching("amcl", "localiz", "pose"),
        "navigation_candidates": names_matching("nav", "goal", "move_base", "path", "costmap"),
        "control_topics": [name for name in ("/cmd_vel", "/odom", "/tf", "/tf_static") if name in by_name],
        "interesting_services": [
            name
            for name in services
            if any(word in name.lower() for word in INTERESTING_WORDS)
        ],
    }


def print_human_report(report: dict[str, Any]) -> None:
    print(f"ROS master: {report['ros_master_uri']}")
    print()
    print("Summary:")
    for key, values in report["summary"].items():
        print(f"  {key}:")
        if values:
            for value in values:
                print(f"    - {value}")
        else:
            print("    - <none>")

    print()
    print("Diagnostics:")
    if report["diagnostics"]:
        for topic, value in report["diagnostics"].items():
            print(f"  {topic}: {json.dumps(value, ensure_ascii=False)}")
    else:
        print("  <none>")

    print()
    print("Odom:")
    print(f"  {json.dumps(report['odom'], ensure_ascii=False)}")

    print()
    print("Interesting topics:")
    for topic in report["topics"]:
        if _is_interesting(topic["name"], topic.get("type")):
            typ = topic.get("type") or "unknown"
            pubs = ",".join(topic.get("publishers") or [])
            subs = ",".join(topic.get("subscribers") or [])
            print(f"  {topic['name']} [{typ}] pubs=[{pubs}] subs=[{subs}]")


def _is_interesting(name: str, typ: str | None) -> bool:
    haystack = f"{name} {typ or ''}".lower()
    return any(word in haystack for word in INTERESTING_WORDS)


def _parse_jsonish(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
