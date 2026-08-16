#!/usr/bin/env python3
"""Rebuild deduplicated ACT HDF5 episodes on a strict fixed-rate time grid."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")


class ResampleError(RuntimeError):
    """The source dataset cannot be safely resampled."""


def _nearest_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source, target, side="left")
    right = np.clip(right, 0, len(source) - 1)
    left = np.maximum(0, right - 1)
    choose_right = np.abs(source[right] - target) < np.abs(source[left] - target)
    return np.where(choose_right, right, left).astype(np.int64)


def _interpolate(source_t: np.ndarray, values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[0] != len(source_t):
        raise ResampleError("state/action arrays must have shape (time, features)")
    result = np.empty((len(target_t), values.shape[1]), dtype=np.float32)
    for dimension in range(values.shape[1]):
        result[:, dimension] = np.interp(
            target_t,
            source_t,
            values[:, dimension].astype(np.float64),
        )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_grid(source_t: np.ndarray, hz: float) -> np.ndarray:
    if source_t.ndim != 1 or len(source_t) < 2:
        raise ResampleError("episode needs at least two timestamps")
    if not np.all(np.isfinite(source_t)) or np.any(np.diff(source_t) <= 0):
        raise ResampleError("episode timestamps must be finite and strictly increasing")
    period = 1.0 / hz
    count = int(np.floor((source_t[-1] - source_t[0]) / period + 1e-7)) + 1
    target = source_t[0] + np.arange(count, dtype=np.float64) * period
    return target[target <= source_t[-1] + 1e-7]


def resample_episode(source: Path, output: Path, *, hz: float = 30.0) -> dict[str, Any]:
    if output.exists():
        raise ResampleError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.inprogress-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    try:
        with h5py.File(source, "r") as src:
            timestamps = src["timestamps/aligned"][:].astype(np.float64)
            qpos = src["observations/qpos"][:].astype(np.float32)
            action = src["action"][:].astype(np.float32)
            if qpos.shape != action.shape or qpos.shape[1:] != (14,):
                raise ResampleError("source qpos/action must both have shape (time, 14)")
            target = _target_grid(timestamps, hz)
            nearest = _nearest_indices(timestamps, target)
            resampled_qpos = _interpolate(timestamps, qpos, target)
            resampled_action = _interpolate(timestamps, action, target)
            with h5py.File(temporary, "w") as dst:
                for key, value in src.attrs.items():
                    dst.attrs[key] = value
                details = {
                    "version": "medicine_act_fixed_rate_v1",
                    "hz": hz,
                    "source_file": source.name,
                    "source_sample_count": int(len(timestamps)),
                    "sample_count": int(len(target)),
                    "state_action_resampling": "linear",
                    "image_resampling": "nearest_exposure",
                    "start_timestamp": float(target[0]),
                    "end_timestamp": float(target[-1]),
                }
                dst.attrs["temporal_resampling_json"] = json.dumps(
                    details, sort_keys=True
                )
                observations = dst.create_group("observations")
                observations.create_dataset("qpos", data=resampled_qpos)
                image_group = observations.create_group("images")
                for camera_name in CAMERA_NAMES:
                    source_images = src[f"observations/images/{camera_name}"]
                    if len(source_images) != len(timestamps):
                        raise ResampleError(f"{camera_name} length does not match timestamps")
                    shape = tuple(source_images.shape[1:])
                    images = image_group.create_dataset(
                        camera_name,
                        shape=(len(target), *shape),
                        dtype=np.uint8,
                        chunks=(1, *shape),
                        compression="gzip",
                        compression_opts=1,
                        shuffle=True,
                    )
                    for target_index, source_index in enumerate(nearest):
                        images[target_index] = source_images[int(source_index)]
                dst.create_dataset("action", data=resampled_action)
                dst.create_dataset("resample_source_index", data=nearest)
                for name in ("camera_frame_index", "source_aligned_index"):
                    if name in src:
                        dst.create_dataset(name, data=src[name][:][nearest])
                timestamp_group = dst.create_group("timestamps")
                timestamp_group.create_dataset("aligned", data=target)
                if "timestamps/cameras" in src:
                    cameras = timestamp_group.create_dataset(
                        "cameras", data=src["timestamps/cameras"][:][nearest]
                    )
                    for key, value in src["timestamps/cameras"].attrs.items():
                        cameras.attrs[key] = value
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    details["sha256"] = _sha256(output)
    return details


def resample_dataset(source: Path, output: Path, *, hz: float) -> dict[str, Any]:
    if output.exists():
        raise ResampleError(f"refusing to overwrite output directory {output}")
    episodes = sorted(
        source.glob("episode_*.hdf5"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    if not episodes:
        raise ResampleError(f"no episode_*.hdf5 files found in {source}")
    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    try:
        for index, episode in enumerate(episodes):
            details = resample_episode(episode, output / episode.name, hz=hz)
            details["episode_index"] = index
            details["file"] = episode.name
            rows.append(details)
        source_manifest = source / "dataset_manifest.json"
        manifest: dict[str, Any] = {}
        if source_manifest.is_file():
            payload = json.loads(source_manifest.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                manifest.update(payload)
        manifest.update(
            {
                "version": "medicine_act_fixed_rate_bundle_v1",
                "source_dataset": str(source.resolve()),
                "episode_count": len(rows),
                "sample_count": sum(row["sample_count"] for row in rows),
                "temporal_resampling": {
                    "enabled": True,
                    "hz": hz,
                    "state_action": "linear",
                    "images": "nearest_exposure",
                },
                "episodes": rows,
            }
        )
        (output / "dataset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--hz", type=float, default=30.0)
    args = parser.parse_args()
    if not 1.0 <= args.hz <= 120.0:
        parser.error("--hz must be between 1 and 120")
    manifest = resample_dataset(args.source.resolve(), args.output.resolve(), hz=args.hz)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output.resolve()),
                "episode_count": manifest["episode_count"],
                "sample_count": manifest["sample_count"],
                "hz": args.hz,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
