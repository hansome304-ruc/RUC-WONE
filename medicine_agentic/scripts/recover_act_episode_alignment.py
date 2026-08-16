#!/usr/bin/env python3
"""Recover a technically failed ACT episode into an auditable, independent copy."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CAMERAS = ("front", "left_wrist", "right_wrist")
ARMS = ("left", "right")
SOURCES = {"observation": "observations", "action": "actions"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def interpolate(rows: list[dict], timestamps: list[float], target: float, *, max_gap_ms: float, max_delta_ms: float) -> dict | None:
    right = bisect.bisect_right(timestamps, target)
    left = right - 1
    if left < 0 or right >= len(rows):
        return None
    t0, t1 = timestamps[left], timestamps[right]
    gap_ms = (t1 - t0) * 1000.0
    nearest_delta_ms = min(target - t0, t1 - target) * 1000.0
    if gap_ms <= 0 or gap_ms > max_gap_ms or nearest_delta_ms > max_delta_ms:
        return None
    before, after = rows[left], rows[right]
    joints0 = np.asarray(before.get("joint_positions"), dtype=np.float64)
    joints1 = np.asarray(after.get("joint_positions"), dtype=np.float64)
    if joints0.shape != (6,) or joints1.shape != (6,) or not np.isfinite(joints0).all() or not np.isfinite(joints1).all():
        return None
    grip0, grip1 = float(before.get("gripper", 0.0)), float(after.get("gripper", 0.0))
    if not math.isfinite(grip0) or not math.isfinite(grip1):
        return None
    alpha = (target - t0) / (t1 - t0)
    return {
        "joint_positions": (joints0 + alpha * (joints1 - joints0)).tolist(),
        "gripper": grip0 + alpha * (grip1 - grip0),
        "source_indices": [left, right],
        "source_timestamps": [t0, t1],
        "interpolation_alpha": alpha,
        "nearest_delta_ms": nearest_delta_ms,
        "source_gap_ms": gap_ms,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--max-gap-ms", type=float, default=70.0)
    parser.add_argument("--max-delta-ms", type=float, default=30.0)
    args = parser.parse_args()

    source = args.source.resolve()
    target = args.target.resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    shutil.copytree(source, target)

    metadata_path = target / "meta.json"
    metadata = json.loads(metadata_path.read_text())
    original_status = metadata.get("status")
    original_error = metadata.get("error")

    camera_rows = {name: read_jsonl(target / "sensors" / f"cam_{name}_frames.jsonl") for name in CAMERAS}
    counts = {name: len(rows) for name, rows in camera_rows.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"camera frame counts differ: {counts}")

    rows: dict[str, dict[str, list[dict]]] = {source_name: {} for source_name in SOURCES}
    timestamps: dict[str, dict[str, list[float]]] = {source_name: {} for source_name in SOURCES}
    observed_max_gap = 0.0
    for source_name, directory in SOURCES.items():
        for arm in ARMS:
            stream = read_jsonl(target / directory / f"{arm}_arm.jsonl")
            times = [float(row["timestamp"]) for row in stream]
            if len(times) < 2 or not all(math.isfinite(value) for value in times):
                raise RuntimeError(f"invalid timestamps for {source_name}/{arm}")
            if any(right <= left for left, right in zip(times, times[1:])):
                raise RuntimeError(f"non-monotonic timestamps for {source_name}/{arm}")
            max_gap = max((right - left) * 1000.0 for left, right in zip(times, times[1:]))
            observed_max_gap = max(observed_max_gap, max_gap)
            if max_gap > args.max_gap_ms:
                raise RuntimeError(f"{source_name}/{arm} max gap {max_gap:.3f} exceeds {args.max_gap_ms:.3f} ms")
            rows[source_name][arm] = stream
            timestamps[source_name][arm] = times

    common_start = max(timestamps[source_name][arm][0] for source_name in SOURCES for arm in ARMS)
    common_end = min(timestamps[source_name][arm][-1] for source_name in SOURCES for arm in ARMS)
    aligned_dir = target / "aligned"
    aligned_dir.mkdir(exist_ok=False)
    output = aligned_dir / "samples.jsonl"
    aligned_count = rejected_count = trimmed_count = 0
    with output.open("w") as handle:
        for frame_index in range(next(iter(counts.values()))):
            at_index = {name: camera_rows[name][frame_index] for name in CAMERAS}
            bundle_ids = {int(row.get("sync_bundle_id", -1)) for row in at_index.values()}
            captured = [float(row["captured_at"]) for row in at_index.values()]
            if len(bundle_ids) != 1 or not all(math.isfinite(value) for value in captured):
                rejected_count += 1
                continue
            target_timestamp = float(np.median(captured))
            if target_timestamp <= common_start or target_timestamp >= common_end:
                trimmed_count += 1
                continue
            aligned = {source_name: {} for source_name in SOURCES}
            valid = True
            for source_name in SOURCES:
                for arm in ARMS:
                    sample = interpolate(rows[source_name][arm], timestamps[source_name][arm], target_timestamp, max_gap_ms=args.max_gap_ms, max_delta_ms=args.max_delta_ms)
                    if sample is None:
                        valid = False
                        break
                    aligned[source_name][arm] = sample
                if not valid:
                    break
            if not valid:
                rejected_count += 1
                continue
            record = {
                "index": aligned_count,
                "camera_frame_index": frame_index,
                "sync_bundle_id": next(iter(bundle_ids)),
                "timestamp": target_timestamp,
                "camera_timestamps": {name: float(at_index[name]["captured_at"]) for name in CAMERAS},
                "observation": aligned["observation"],
                "action": aligned["action"],
            }
            handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            aligned_count += 1

    eligible = next(iter(counts.values())) - trimmed_count
    aligned_fraction = aligned_count / eligible if eligible else 0.0
    if aligned_count < 2 or aligned_fraction < 0.98:
        raise RuntimeError(f"recovered alignment is insufficient: count={aligned_count}, fraction={aligned_fraction:.6f}")

    training_alignment = metadata["act"]["training_alignment"]
    training_alignment.update(
        {
            "aligned_sample_count": aligned_count,
            "rejected_camera_frame_count": rejected_count,
            "trimmed_edge_frame_count": trimmed_count,
            "aligned_fraction": aligned_fraction,
            "max_allowed_arm_sample_gap_ms": args.max_gap_ms,
            "max_allowed_camera_arm_delta_ms": args.max_delta_ms,
            "observed_max_arm_sample_gap_ms": observed_max_gap,
        }
    )
    metadata["status"] = "completed"
    metadata["error"] = None
    metadata["recovery"] = {
        "version": "medicine_act_alignment_recovery_v1",
        "recovered_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source),
        "original_status": original_status,
        "original_error": original_error,
        "reason": "curated task-success episode; technical alignment threshold relaxed only in independent training copy",
        "max_gap_ms": args.max_gap_ms,
        "max_delta_ms": args.max_delta_ms,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")

    manifest_paths = sorted(
        path for path in target.rglob("*") if path.is_file() and path.name not in {"READY", "checksums.sha256"}
    )
    checksum_lines = [f"{sha256(path)}  {path.relative_to(target).as_posix()}" for path in manifest_paths]
    (target / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n")
    (target / "READY").touch()

    print(
        json.dumps(
            {
                "source": str(source),
                "target": str(target),
                "aligned_sample_count": aligned_count,
                "rejected_camera_frame_count": rejected_count,
                "trimmed_edge_frame_count": trimmed_count,
                "aligned_fraction": aligned_fraction,
                "observed_max_gap_ms": observed_max_gap,
                "allowed_max_gap_ms": args.max_gap_ms,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
