#!/usr/bin/env python3
"""Analyze a data-derived executed-edge termination rule for ACT action index 13."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def read_actions(episode: Path) -> tuple[dict, np.ndarray]:
    meta = json.loads((episode / "meta.json").read_text())
    aligned = episode / meta["files"]["aligned_samples"]
    values = []
    for line in aligned.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        values.append(float(row["action"]["right"]["gripper"]))
    array = np.asarray(values, dtype=np.float64)
    if array.size < 4 or not np.isfinite(array).all():
        raise RuntimeError(f"invalid right-gripper actions in {episode}")
    return meta, array


def kmeans_two(values: np.ndarray) -> tuple[float, float, np.ndarray]:
    centers = np.quantile(values, [0.2, 0.8]).astype(np.float64)
    for _ in range(100):
        distances = np.abs(values[:, None] - centers[None, :])
        labels = distances.argmin(axis=1)
        updated = np.asarray([values[labels == index].mean() for index in range(2)])
        if np.allclose(updated, centers, rtol=0, atol=1e-12):
            break
        centers = updated
    centers.sort()
    threshold = float(centers.mean())
    return float(centers[0]), float(centers[1]), threshold


def stable_edges(values: np.ndarray, threshold: float, direction: str, consecutive: int) -> list[dict]:
    terminal = values >= threshold if direction == "rising" else values <= threshold
    source = ~terminal
    edges: list[dict] = []
    source_seen = False
    index = 0
    while index < values.size:
        if source[index]:
            source_seen = True
            index += 1
            continue
        end = index + consecutive
        if source_seen and end <= values.size and bool(terminal[index:end].all()):
            edges.append(
                {
                    "edge_start_index": index,
                    "trigger_index": end - 1,
                    "trigger_commands": end,
                    "edge_fraction": index / values.size,
                    "tail_frames_from_edge": values.size - index,
                }
            )
            source_seen = False
            index = end
            continue
        index += 1
    return edges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--consecutive", type=int, default=3)
    args = parser.parse_args()

    episodes = sorted(path for path in args.root.resolve().iterdir() if path.is_dir() and (path / "READY").is_file())
    data = [(*read_actions(episode), episode) for episode in episodes]
    all_values = np.concatenate([values for _, values, _ in data])
    low, high, threshold = kmeans_two(all_values)
    separation = high - low
    within_low = all_values[all_values < threshold]
    within_high = all_values[all_values >= threshold]

    direction_votes = []
    for _, values, _ in data:
        window = min(10, max(3, values.size // 20))
        start_high = float(np.median(values[:window])) >= threshold
        end_high = float(np.median(values[-window:])) >= threshold
        if start_high != end_high:
            direction_votes.append("rising" if end_high else "falling")
        else:
            direction_votes.append("none")
    rising_votes = direction_votes.count("rising")
    falling_votes = direction_votes.count("falling")
    direction = "rising" if rising_votes >= falling_votes else "falling"

    records = []
    for (meta, values, episode), vote in zip(data, direction_votes):
        edges = stable_edges(values, threshold, direction, args.consecutive)
        opposite = stable_edges(values, threshold, "falling" if direction == "rising" else "rising", args.consecutive)
        terminal_side = values >= threshold if direction == "rising" else values <= threshold
        first = edges[0] if edges else None
        records.append(
            {
                "recording_id": meta["recording_id"],
                "label": meta.get("label"),
                "length": int(values.size),
                "initial_median": float(np.median(values[: min(10, values.size)])),
                "final_median": float(np.median(values[-min(10, values.size) :])),
                "direction_vote": vote,
                "same_direction_edge_count": len(edges),
                "opposite_direction_edge_count": len(opposite),
                "first_edge": first,
                "all_same_direction_edges": edges,
                "final_three_on_terminal_side": bool(terminal_side[-args.consecutive :].all()),
                "has_early_same_direction_edge": bool(first and first["edge_fraction"] < 0.70),
                "recovered": bool(meta.get("recovery")),
            }
        )

    first_edges = [record["first_edge"] for record in records if record["first_edge"]]
    trigger_commands = np.asarray([edge["trigger_commands"] for edge in first_edges], dtype=np.int64)
    lengths = np.asarray([record["length"] for record in records], dtype=np.int64)
    tail_frames = np.asarray([edge["tail_frames_from_edge"] for edge in first_edges], dtype=np.int64)
    min_commands = int(trigger_commands.min()) if trigger_commands.size else None
    length_spread = int(math.ceil(float(np.quantile(lengths, 0.95) - np.quantile(lengths, 0.50))))
    max_commands = int(lengths.max() + max(args.consecutive, length_spread))
    quantile_probs = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    report = {
        "version": "medicine_act_gripper_edge_analysis_v1",
        "termination_dimension": 13,
        "episode_count": len(records),
        "sample_count": int(all_values.size),
        "cluster_low_center": low,
        "cluster_high_center": high,
        "cluster_separation": separation,
        "threshold": threshold,
        "low_cluster_count": int(within_low.size),
        "high_cluster_count": int(within_high.size),
        "quantiles": {str(prob): float(np.quantile(all_values, prob)) for prob in quantile_probs},
        "direction_votes": {"rising": rising_votes, "falling": falling_votes, "none": direction_votes.count("none")},
        "recommended": {
            "termination_edge": direction,
            "termination_threshold": threshold,
            "termination_consecutive_steps": args.consecutive,
            "termination_min_commands": min_commands,
            "max_commands": max_commands,
            "max_commands_margin": max(args.consecutive, length_spread),
        },
        "coverage": {
            "episodes_with_edge": len(first_edges),
            "episodes_with_exactly_one_same_direction_edge": sum(record["same_direction_edge_count"] == 1 for record in records),
            "episodes_with_multiple_same_direction_edges": sum(record["same_direction_edge_count"] > 1 for record in records),
            "episodes_with_no_same_direction_edge": sum(record["same_direction_edge_count"] == 0 for record in records),
            "episodes_final_on_terminal_side": sum(record["final_three_on_terminal_side"] for record in records),
            "episodes_with_early_same_direction_edge_before_70pct": sum(record["has_early_same_direction_edge"] for record in records),
            "rule_covers_all_without_early_trigger": all(
                record["same_direction_edge_count"] == 1
                and record["final_three_on_terminal_side"]
                and not record["has_early_same_direction_edge"]
                for record in records
            ),
        },
        "trigger_commands": None if not trigger_commands.size else {
            "min": int(trigger_commands.min()),
            "p05": float(np.quantile(trigger_commands, 0.05)),
            "median": float(np.median(trigger_commands)),
            "p95": float(np.quantile(trigger_commands, 0.95)),
            "max": int(trigger_commands.max()),
        },
        "trajectory_lengths": {
            "min": int(lengths.min()),
            "median": float(np.median(lengths)),
            "p95": float(np.quantile(lengths, 0.95)),
            "max": int(lengths.max()),
        },
        "tail_frames_after_edge": None if not tail_frames.size else {
            "min": int(tail_frames.min()),
            "median": float(np.median(tail_frames)),
            "p95": float(np.quantile(tail_frames, 0.95)),
            "max": int(tail_frames.max()),
        },
        "episodes": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("episode_count", "sample_count", "cluster_low_center", "cluster_high_center", "threshold", "direction_votes", "recommended", "coverage", "trigger_commands", "trajectory_lengths", "tail_frames_after_edge")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
