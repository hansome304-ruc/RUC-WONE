"""Feedback-only four-endpoint timing smoke test; never sends arm commands."""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from medicine_agentic.airbot_readonly import AirbotReadOnly


def main() -> int:
    runtime = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    leader_host = str(runtime["remote_host"])
    leader_ports = runtime.get("leader_ports", {"left": 50050, "right": 50052})
    follower = AirbotReadOnly(
        host="localhost",
        ports={"left": 50051, "right": 50053},
        arm_names=("left", "right"),
    )
    leader = AirbotReadOnly(
        host=leader_host,
        ports=leader_ports,
        arm_names=("left", "right"),
    )
    pair_skews = []
    source_skews = []
    cycle_timestamps = []
    try:
        follower.connect()
        leader.connect()
        next_tick = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            for _index in range(250):
                left = pool.submit(follower.capture_selected_fast)
                right = pool.submit(leader.capture_selected_fast)
                follower_sample = left.result()
                leader_sample = right.result()
                cycle_timestamps.append(time.time())
                pair_skews.extend(
                    [
                        float(follower_sample["paired_sample_skew_ms"]),
                        float(leader_sample["paired_sample_skew_ms"]),
                    ]
                )
                source_skews.append(
                    abs(
                        int(follower_sample["timestamp_ns"])
                        - int(leader_sample["timestamp_ns"])
                    )
                    / 1e6
                )
                next_tick += 0.02
                time.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        follower.close()
        leader.close()
    gaps = np.diff(np.asarray(cycle_timestamps, dtype=np.float64)) * 1000.0
    result = {
        "sample_cycles": len(cycle_timestamps),
        "max_arm_pair_skew_ms": max(pair_skews),
        "p95_arm_pair_skew_ms": float(np.percentile(pair_skews, 95)),
        "max_source_capture_skew_ms": max(source_skews),
        "p95_source_capture_skew_ms": float(np.percentile(source_skews, 95)),
        "max_cycle_gap_ms": float(np.max(gaps)),
        "p95_cycle_gap_ms": float(np.percentile(gaps, 95)),
        "passes": bool(
            max(pair_skews) <= 25.0
            and max(source_skews) <= 25.0
            and float(np.max(gaps)) <= 60.0
        ),
    }
    print(json.dumps(result, indent=2))
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
