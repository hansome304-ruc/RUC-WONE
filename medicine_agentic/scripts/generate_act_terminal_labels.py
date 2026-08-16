#!/usr/bin/env python3
"""Generate auditable per-episode terminal labels without conflating padding."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--terminal-window-frames", type=int, default=10)
    parser.add_argument("--hard-negative-seconds", type=float, default=1.5)
    parser.add_argument("--assume-curated-success", action="store_true")
    args = parser.parse_args()
    if args.terminal_window_frames <= 0:
        raise ValueError("terminal window must contain at least one valid control frame")

    root = args.root.resolve()
    episodes = sorted(path for path in root.iterdir() if path.is_dir() and (path / "READY").is_file())
    if not episodes:
        raise RuntimeError(f"no READY episodes under {root}")

    records: list[dict] = []
    total_samples = total_positive = total_hard_negative = 0
    for episode in episodes:
        meta = json.loads((episode / "meta.json").read_text())
        alignment = meta.get("act", {}).get("training_alignment", {})
        count = int(alignment.get("aligned_sample_count", 0))
        aligned_file = episode / meta["files"]["aligned_samples"]
        rows = read_jsonl(aligned_file)
        if count != len(rows) or count < args.terminal_window_frames + 2:
            raise RuntimeError(f"invalid aligned length for {episode.name}: meta={count}, rows={len(rows)}")
        if [int(row["index"]) for row in rows] != list(range(count)):
            raise RuntimeError(f"non-contiguous aligned indices for {episode.name}")
        timestamps = [float(row["timestamp"]) for row in rows]
        if not all(math.isfinite(value) for value in timestamps) or any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            raise RuntimeError(f"invalid aligned timestamps for {episode.name}")

        status = str(meta.get("status", ""))
        success = status == "completed" and args.assume_curated_success
        success_source = "user_curated_high_quality_dataset" if success else "recording_status_not_success"
        fps = float(meta.get("recorder", {}).get("nominal_camera_fps") or 30.0)
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError(f"invalid fps for {episode.name}")
        terminal_start = count - args.terminal_window_frames if success else None
        terminal_index = count - 1 if success else None
        hard_negative_frames = max(1, round(args.hard_negative_seconds * fps))
        hard_negative_end = terminal_start if success else count
        hard_negative_start = max(0, hard_negative_end - hard_negative_frames)
        terminal_duration_s = (
            timestamps[-1] - timestamps[terminal_start] if success and terminal_start is not None else 0.0
        )
        positive_count = args.terminal_window_frames if success else 0
        record = {
            "recording_id": meta["recording_id"],
            "label": meta.get("label"),
            "episode_path": str(episode),
            "status": status,
            "success": success,
            "success_source": success_source,
            "fps": fps,
            "valid_length": count,
            "terminal_index": terminal_index,
            "terminal_window": None if not success else [terminal_start, count],
            "terminal_window_frames": positive_count,
            "terminal_window_duration_s": terminal_duration_s,
            "hard_negative_window": [hard_negative_start, hard_negative_end],
            "hard_negative_frames": hard_negative_end - hard_negative_start,
            "padding_is_terminal": False,
            "recovery": meta.get("recovery"),
        }
        records.append(record)
        total_samples += count
        total_positive += positive_count
        total_hard_negative += hard_negative_end - hard_negative_start

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records))
    report = {
        "version": "medicine_act_terminal_labels_v1",
        "root": str(root),
        "episode_count": len(records),
        "success_count": sum(record["success"] for record in records),
        "failed_or_aborted_count": sum(not record["success"] for record in records),
        "terminal_window_frames": args.terminal_window_frames,
        "hard_negative_seconds": args.hard_negative_seconds,
        "total_valid_samples": total_samples,
        "done_positive_samples": total_positive,
        "done_negative_samples": total_samples - total_positive,
        "done_positive_ratio": total_positive / total_samples,
        "hard_negative_samples": total_hard_negative,
        "padding_is_terminal": False,
        "checks": {
            "terminal_inside_valid_length": True,
            "failed_or_aborted_have_no_done_positive": all(record["success"] or record["terminal_index"] is None for record in records),
            "hard_negatives_precede_terminal": all(
                (not record["success"]) or record["hard_negative_window"][1] == record["terminal_window"][0]
                for record in records
            ),
        },
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
