"""Convert strict medicine ACT episodes into canonical training HDF5 files."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from medicine_agentic.act_dataset import discover_ready, validate_episode


CAMERA_MAPPING = {
    "front": "cam_high",
    "left_wrist": "cam_left_wrist",
    "right_wrist": "cam_right_wrist",
}
ARM_ORDER = ("left", "right")
JOINT_INDICES = np.asarray([*range(6), *range(7, 13)], dtype=np.int64)
GRIPPER_INDICES = np.asarray([6, 13], dtype=np.int64)


class ActExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class StationaryDedupConfig:
    """Conservative compression settings for sustained stationary intervals."""

    enabled: bool = True
    joint_tolerance_rad: float = 2.5e-3
    gripper_tolerance: float = 5e-4
    min_duration_seconds: float = 0.3
    keep_every_n_frames: int = 5
    anchor_multiplier: float = 2.0

    def validate(self) -> None:
        if self.joint_tolerance_rad < 0:
            raise ValueError("joint_tolerance_rad must be nonnegative")
        if self.gripper_tolerance < 0:
            raise ValueError("gripper_tolerance must be nonnegative")
        if self.min_duration_seconds < 0:
            raise ValueError("min_duration_seconds must be nonnegative")
        if (
            isinstance(self.keep_every_n_frames, bool)
            or not isinstance(self.keep_every_n_frames, int)
            or self.keep_every_n_frames <= 0
        ):
            raise ValueError("keep_every_n_frames must be a positive integer")
        if self.anchor_multiplier < 1:
            raise ValueError("anchor_multiplier must be at least 1")


def _vectors_within_tolerance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    joint_tolerance_rad: float,
    gripper_tolerance: float,
) -> bool:
    return bool(
        np.max(np.abs(left[JOINT_INDICES] - right[JOINT_INDICES]))
        <= joint_tolerance_rad
        and np.max(np.abs(left[GRIPPER_INDICES] - right[GRIPPER_INDICES]))
        <= gripper_tolerance
    )


def _select_stationary_dedup_indices(
    qpos: np.ndarray,
    action: np.ndarray,
    timestamps: np.ndarray,
    config: StationaryDedupConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select synchronized samples while conservatively compressing static runs.

    A run is stationary only when both follower state and leader action stay within
    tolerance from the previous sample and from the run anchor. The anchor check
    prevents slow motion from being mistaken for encoder noise. Short runs remain
    untouched; long runs retain their boundaries and every Nth stationary frame.
    """

    config.validate()
    sample_count = int(qpos.shape[0])
    if qpos.shape != action.shape or qpos.shape != (sample_count, 14):
        raise ActExportError("stationary dedup expects matching [N, 14] state/action arrays")
    if timestamps.shape != (sample_count,):
        raise ActExportError("stationary dedup expects one timestamp per sample")
    if sample_count > 1 and np.any(np.diff(timestamps) <= 0):
        raise ActExportError("stationary dedup requires strictly increasing timestamps")

    keep = np.ones(sample_count, dtype=bool)
    compressed_runs = 0
    if config.enabled and sample_count > 2:
        index = 0
        anchor_joint_tolerance = config.joint_tolerance_rad * config.anchor_multiplier
        anchor_gripper_tolerance = config.gripper_tolerance * config.anchor_multiplier
        while index < sample_count - 1:
            start = index
            end = start
            while end + 1 < sample_count:
                candidate = end + 1
                locally_static = (
                    _vectors_within_tolerance(
                        qpos[end],
                        qpos[candidate],
                        joint_tolerance_rad=config.joint_tolerance_rad,
                        gripper_tolerance=config.gripper_tolerance,
                    )
                    and _vectors_within_tolerance(
                        action[end],
                        action[candidate],
                        joint_tolerance_rad=config.joint_tolerance_rad,
                        gripper_tolerance=config.gripper_tolerance,
                    )
                )
                within_anchor = (
                    _vectors_within_tolerance(
                        qpos[start],
                        qpos[candidate],
                        joint_tolerance_rad=anchor_joint_tolerance,
                        gripper_tolerance=anchor_gripper_tolerance,
                    )
                    and _vectors_within_tolerance(
                        action[start],
                        action[candidate],
                        joint_tolerance_rad=anchor_joint_tolerance,
                        gripper_tolerance=anchor_gripper_tolerance,
                    )
                )
                if not locally_static or not within_anchor:
                    break
                end = candidate

            duration = float(timestamps[end] - timestamps[start])
            if end > start and duration >= config.min_duration_seconds:
                keep[start : end + 1] = False
                keep[start : end + 1 : config.keep_every_n_frames] = True
                keep[end] = True
                compressed_runs += 1

            index = end if end > start else start + 1

    selected = np.flatnonzero(keep).astype(np.int64)
    removed_count = sample_count - int(selected.size)
    return selected, {
        "enabled": config.enabled,
        "joint_tolerance_rad": config.joint_tolerance_rad,
        "gripper_tolerance": config.gripper_tolerance,
        "min_duration_seconds": config.min_duration_seconds,
        "keep_every_n_frames": config.keep_every_n_frames,
        "anchor_multiplier": config.anchor_multiplier,
        "sample_count_before": sample_count,
        "sample_count_after": int(selected.size),
        "removed_sample_count": removed_count,
        "retention_ratio": (float(selected.size) / sample_count if sample_count else 1.0),
        "compressed_run_count": compressed_runs,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _state_vector(row: dict[str, Any], group: str) -> list[float]:
    values: list[float] = []
    for arm in ARM_ORDER:
        sample = row[group][arm]
        values.extend(float(value) for value in sample["joint_positions"])
        values.append(float(sample["gripper"]))
    return values


def _letterbox_rgb(frame_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ActExportError("decoded camera frame is not BGR8")
    source_height, source_width = frame_bgr.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        frame_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - resized_width) // 2
    y0 = (height - resized_height) // 2
    canvas[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _decode_selected_frames(
    video_path: Path,
    frame_indices: list[int],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    if not frame_indices or any(right <= left for left, right in zip(frame_indices, frame_indices[1:])):
        raise ActExportError("camera frame indices must be nonempty and increasing")
    wanted = set(frame_indices)
    decoded: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(video_path))
    try:
        index = 0
        while index <= frame_indices[-1]:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                decoded[index] = _letterbox_rgb(frame, width, height)
            index += 1
    finally:
        capture.release()
    missing = [index for index in frame_indices if index not in decoded]
    if missing:
        raise ActExportError(
            f"video {video_path.name} is missing selected frames: {missing[:8]}"
        )
    return np.stack([decoded[index] for index in frame_indices], axis=0)


def export_episode(
    episode_path: str | Path,
    output_path: str | Path,
    *,
    image_width: int = 640,
    image_height: int = 480,
    stationary_dedup: StationaryDedupConfig | None = None,
) -> dict[str, Any]:
    episode = Path(episode_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    validation = validate_episode(episode, verify_video=True)
    if not validation.valid:
        raise ActExportError("episode validation failed: " + "; ".join(validation.errors))
    metadata = validation.metadata or {}
    if metadata.get("purpose") != "act_bimanual":
        raise ActExportError("canonical training export requires act_bimanual episodes")
    if tuple(metadata.get("selected_arms", [])) != ARM_ORDER:
        raise ActExportError("canonical training export requires left and right arms")
    camera_names = tuple(metadata.get("camera_names", []))
    if camera_names != tuple(CAMERA_MAPPING):
        raise ActExportError(
            "canonical training export requires front, left_wrist and right_wrist cameras"
        )
    if image_width < 64 or image_height < 64:
        raise ValueError("training image dimensions are too small")
    timestamp_basis = (
        metadata.get("act", {})
        .get("training_alignment", {})
        .get("timestamp_basis")
    )
    if timestamp_basis != "device_global_time":
        raise ActExportError("canonical export requires device-global timestamps")

    aligned_relative = metadata["files"]["aligned_samples"]
    rows = _read_jsonl(episode / aligned_relative)
    frame_indices = [int(row["camera_frame_index"]) for row in rows]
    qpos = np.asarray([_state_vector(row, "observation") for row in rows], dtype=np.float32)
    action = np.asarray([_state_vector(row, "action") for row in rows], dtype=np.float32)
    timestamps = np.asarray([float(row["timestamp"]) for row in rows], dtype=np.float64)
    camera_timestamps = np.asarray(
        [
            [float(row["camera_timestamps"][name]) for name in camera_names]
            for row in rows
        ],
        dtype=np.float64,
    )
    if qpos.shape != (len(rows), 14) or action.shape != (len(rows), 14):
        raise ActExportError("ACT state/action vectors must have 14 values")

    dedup_config = stationary_dedup or StationaryDedupConfig()
    selected_indices, deduplication = _select_stationary_dedup_indices(
        qpos,
        action,
        timestamps,
        dedup_config,
    )
    source_aligned_indices = selected_indices.copy()
    frame_indices = [frame_indices[index] for index in selected_indices]
    qpos = qpos[selected_indices]
    action = action[selected_indices]
    timestamps = timestamps[selected_indices]
    camera_timestamps = camera_timestamps[selected_indices]

    images: dict[str, np.ndarray] = {}
    for source_name, training_name in CAMERA_MAPPING.items():
        video_relative = metadata["files"]["images"][source_name]["video"]
        images[training_name] = _decode_selected_frames(
            episode / video_relative,
            frame_indices,
            width=image_width,
            height=image_height,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.inprogress-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    try:
        with h5py.File(temporary, "w") as root:
            root.attrs["sim"] = False
            root.attrs["compress"] = False
            root.attrs["format"] = "medicine_act_hdf5_v1"
            root.attrs["source_recording_id"] = str(metadata["recording_id"])
            root.attrs["source_label"] = str(metadata.get("label", ""))
            root.attrs["color_order"] = "RGB"
            root.attrs["resize_mode"] = "letterbox"
            root.attrs["timestamp_basis"] = timestamp_basis
            root.attrs["camera_mapping_json"] = json.dumps(CAMERA_MAPPING, sort_keys=True)
            root.attrs["stationary_dedup_json"] = json.dumps(
                deduplication,
                sort_keys=True,
            )
            root.attrs["state_order_json"] = json.dumps(
                [
                    *(f"left_joint_{index}" for index in range(1, 7)),
                    "left_gripper_raw",
                    *(f"right_joint_{index}" for index in range(1, 7)),
                    "right_gripper_raw",
                ]
            )
            observations = root.create_group("observations")
            observations.create_dataset("qpos", data=qpos)
            image_group = observations.create_group("images")
            for camera_name, array in images.items():
                image_group.create_dataset(
                    camera_name,
                    data=array,
                    chunks=(1, image_height, image_width, 3),
                    compression="gzip",
                    compression_opts=1,
                    shuffle=True,
                )
            root.create_dataset("action", data=action)
            root.create_dataset(
                "camera_frame_index",
                data=np.asarray(frame_indices, dtype=np.int64),
            )
            root.create_dataset("source_aligned_index", data=source_aligned_indices)
            timestamp_group = root.create_group("timestamps")
            timestamp_group.create_dataset("aligned", data=timestamps)
            camera_dataset = timestamp_group.create_dataset("cameras", data=camera_timestamps)
            camera_dataset.attrs["source_camera_names_json"] = json.dumps(camera_names)
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    return {
        "source_recording_id": metadata["recording_id"],
        "source_label": metadata.get("label"),
        "output": str(output),
        "sample_count": int(selected_indices.size),
        "source_sample_count": len(rows),
        "deduplication": deduplication,
        "state_dim": 14,
        "camera_names": list(CAMERA_MAPPING.values()),
        "image_size": [image_height, image_width],
        "sha256": _sha256(output),
    }


def _task_name(label: str) -> str:
    match = re.match(r"^act0?([123])(?:[_\-.]|$)", label.strip().lower())
    if match:
        return f"act{match.group(1)}"
    normalized = re.sub(r"[^0-9a-z]+", "_", label.strip().lower()).strip("_")
    return normalized or "unassigned"


def prepare_training_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    image_width: int = 640,
    image_height: int = 480,
    replace: bool = False,
    stationary_dedup: StationaryDedupConfig | None = None,
) -> dict[str, Any]:
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    episodes = discover_ready(source)
    if not episodes:
        raise ActExportError(f"no READY episodes found under {source}")
    if output.exists() and not replace:
        raise ActExportError(f"output already exists: {output}; pass --replace to rebuild it")
    if output == Path(output.anchor) or output == source or source in output.parents:
        raise ActExportError("unsafe output directory")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.inprogress-", dir=str(output.parent))
    )
    entries_by_task: dict[str, list[dict[str, Any]]] = {}
    try:
        for episode in episodes:
            meta = json.loads((episode / "meta.json").read_text(encoding="utf-8"))
            task = _task_name(str(meta.get("label", "")))
            task_entries = entries_by_task.setdefault(task, [])
            task_dir = staging / task
            task_dir.mkdir(parents=True, exist_ok=True)
            entry = export_episode(
                episode,
                task_dir / f"episode_{len(task_entries)}.hdf5",
                image_width=image_width,
                image_height=image_height,
                stationary_dedup=stationary_dedup,
            )
            entry["episode_index"] = len(task_entries)
            entry["file"] = f"episode_{len(task_entries)}.hdf5"
            task_entries.append(entry)
        for task, entries in entries_by_task.items():
            manifest = {
                "version": "medicine_act_training_bundle_v1",
                "task": task,
                "episode_count": len(entries),
                "state_dim": 14,
                "state_order": [
                    *(f"left_joint_{index}" for index in range(1, 7)),
                    "left_gripper_raw",
                    *(f"right_joint_{index}" for index in range(1, 7)),
                    "right_gripper_raw",
                ],
                "camera_mapping": CAMERA_MAPPING,
                "camera_names": list(CAMERA_MAPPING.values()),
                "image_size": [image_height, image_width],
                "image_color_order": "RGB",
                "resize_mode": "letterbox",
                "timestamp_basis": "device_global_time",
                "stationary_dedup": asdict(stationary_dedup or StationaryDedupConfig()),
                "episodes": entries,
            }
            (staging / task / "dataset_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        previous = output.with_name(f".{output.name}.previous-{os.getpid()}")
        if output.exists():
            output.replace(previous)
        try:
            staging.replace(output)
        except Exception:
            if previous.exists() and not output.exists():
                previous.replace(output)
            raise
        if previous.exists():
            shutil.rmtree(previous)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return {
        "source_root": str(source),
        "output_root": str(output),
        "task_count": len(entries_by_task),
        "episode_count": sum(len(entries) for entries in entries_by_task.values()),
        "tasks": {task: len(entries) for task, entries in entries_by_task.items()},
    }
