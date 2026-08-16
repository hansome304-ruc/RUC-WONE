"""Validation helpers for finalized ACT collection episodes."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationResult:
    path: Path
    valid: bool
    errors: tuple[str, ...]
    metadata: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "valid": self.valid,
            "errors": list(self.errors),
            "recording_id": (self.metadata or {}).get("recording_id"),
            "label": (self.metadata or {}).get("label"),
            "frame_count": (self.metadata or {}).get("frame_count"),
            "aligned_sample_count": (self.metadata or {})
            .get("act", {})
            .get("training_alignment", {})
            .get("aligned_sample_count"),
            "camera_names": (self.metadata or {}).get("camera_names"),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number} is not an object")
            rows.append(payload)
    return rows


def _json_lines(path: Path) -> int:
    return len(_read_json_lines(path))


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_aligned_rows(
    root: Path,
    path: Path,
    metadata: dict[str, Any],
    *,
    require_training_timing: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    rows = _read_json_lines(path)
    arms = [str(name) for name in metadata.get("selected_arms", [])]
    camera_names = [str(name) for name in metadata.get("camera_names", [])]
    camera_counts = metadata.get("camera_frame_counts", {})
    alignment = metadata.get("act", {}).get("training_alignment", {})
    synchronization = metadata.get("act", {}).get("synchronization", {})
    if (
        require_training_timing
        and alignment.get("timestamp_basis") != "device_global_time"
    ):
        errors.append(
            "training alignment does not declare device_global_time as its timestamp basis"
        )
    image_streams = metadata.get("files", {}).get("images", {})
    frame_metadata: dict[str, list[dict[str, Any]]] = {}
    source_timing_ready = True
    for name in camera_names:
        details = image_streams.get(name, {}) if isinstance(image_streams, dict) else {}
        relative = details.get("frame_metadata") if isinstance(details, dict) else None
        try:
            source_rows = _read_json_lines(root / str(relative)) if relative else []
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read {name} source frame timing: {exc}")
            source_rows = []
        frame_metadata[name] = source_rows
        if require_training_timing and (not source_rows or any(
            not _finite_number(row.get("sync_timestamp_ms"))
            or not _finite_number(row.get("device_timestamp_ms"))
            or "global" not in str(row.get("timestamp_domain", "")).lower()
            or row.get("timestamp_source") != "device_global_time"
            for row in source_rows
        )):
            errors.append(
                f"{name} frame metadata lacks strict global device timing provenance"
            )
            source_timing_ready = False
    max_camera_skew_ms = float(synchronization.get("max_allowed_skew_ms", math.inf))
    max_camera_arm_delta_ms = float(
        alignment.get("max_allowed_camera_arm_delta_ms", math.inf)
    )
    max_arm_gap_ms = float(alignment.get("max_allowed_arm_sample_gap_ms", math.inf))
    previous_timestamp = -math.inf
    previous_frame_index = -1
    previous_bundle_id = -1

    for expected_index, row in enumerate(rows):
        prefix = f"aligned row {expected_index}"
        if row.get("index") != expected_index:
            errors.append(f"{prefix} has non-sequential index")
        timestamp = row.get("timestamp")
        if not _finite_number(timestamp) or float(timestamp) <= previous_timestamp:
            errors.append(f"{prefix} timestamp is not finite and increasing")
        else:
            previous_timestamp = float(timestamp)

        frame_index = row.get("camera_frame_index")
        if (
            not isinstance(frame_index, int)
            or isinstance(frame_index, bool)
            or frame_index <= previous_frame_index
            or any(frame_index >= int(camera_counts.get(name, -1)) for name in camera_names)
        ):
            errors.append(f"{prefix} camera_frame_index is invalid")
        else:
            previous_frame_index = frame_index

        bundle_id = row.get("sync_bundle_id")
        if (
            not isinstance(bundle_id, int)
            or isinstance(bundle_id, bool)
            or bundle_id <= previous_bundle_id
        ):
            errors.append(f"{prefix} sync_bundle_id is not increasing")
        else:
            previous_bundle_id = bundle_id

        camera_timestamps = row.get("camera_timestamps")
        if not isinstance(camera_timestamps, dict) or list(camera_timestamps) != camera_names:
            errors.append(f"{prefix} camera timestamp keys differ from camera_names")
        else:
            values = [camera_timestamps.get(name) for name in camera_names]
            if not all(_finite_number(value) for value in values):
                errors.append(f"{prefix} has invalid camera timestamps")
            else:
                numeric = [float(value) for value in values]
                spread_ms = (max(numeric) - min(numeric)) * 1000.0
                if (
                    require_training_timing
                    and spread_ms > max_camera_skew_ms + 1e-6
                ):
                    errors.append(
                        f"{prefix} camera skew {spread_ms:.3f} ms exceeds "
                        f"{max_camera_skew_ms:.3f} ms"
                    )
                if _finite_number(timestamp) and abs(
                    float(timestamp) - statistics.median(numeric)
                ) > 1e-6:
                    errors.append(f"{prefix} timestamp is not the camera median")
                if (
                    require_training_timing
                    and source_timing_ready
                    and isinstance(frame_index, int)
                ):
                    source_timestamps: list[float] = []
                    for name, aligned_timestamp in zip(camera_names, numeric):
                        try:
                            source = frame_metadata[name][frame_index]
                        except IndexError:
                            errors.append(
                                f"{prefix} {name} source frame index is out of range"
                            )
                            continue
                        sync_timestamp_ms = float(source["sync_timestamp_ms"])
                        device_timestamp_ms = float(source["device_timestamp_ms"])
                        source_timestamp = sync_timestamp_ms / 1000.0
                        source_timestamps.append(source_timestamp)
                        if source.get("index") != frame_index:
                            errors.append(f"{prefix} {name} source index differs")
                        if source.get("sync_bundle_id") != bundle_id:
                            errors.append(f"{prefix} {name} source bundle differs")
                        if abs(sync_timestamp_ms - device_timestamp_ms) > 1e-3:
                            errors.append(
                                f"{prefix} {name} sync and device timestamps differ"
                            )
                        if abs(float(source.get("captured_at", math.nan)) - source_timestamp) > 1e-6:
                            errors.append(
                                f"{prefix} {name} captured_at is not exposure time"
                            )
                        if abs(aligned_timestamp - source_timestamp) > 1e-6:
                            errors.append(
                                f"{prefix} {name} timestamp differs from source exposure"
                            )
                    if len(source_timestamps) == len(camera_names):
                        source_spread_ms = (
                            max(source_timestamps) - min(source_timestamps)
                        ) * 1000.0
                        if source_spread_ms > max_camera_skew_ms + 1e-6:
                            errors.append(
                                f"{prefix} source exposure skew {source_spread_ms:.3f} ms "
                                f"exceeds {max_camera_skew_ms:.3f} ms"
                            )

        for group in ("observation", "action"):
            samples = row.get(group)
            if not isinstance(samples, dict) or list(samples) != arms:
                errors.append(f"{prefix} {group} arms differ from selected_arms")
                continue
            for arm in arms:
                sample = samples.get(arm)
                sample_prefix = f"{prefix} {group}/{arm}"
                if not isinstance(sample, dict):
                    errors.append(f"{sample_prefix} is not an object")
                    continue
                joints = sample.get("joint_positions")
                if (
                    not isinstance(joints, list)
                    or len(joints) != 6
                    or not all(_finite_number(value) for value in joints)
                ):
                    errors.append(f"{sample_prefix} joints are invalid")
                if not _finite_number(sample.get("gripper")):
                    errors.append(f"{sample_prefix} gripper is invalid")
                source_indices = sample.get("source_indices")
                source_timestamps = sample.get("source_timestamps")
                alpha = sample.get("interpolation_alpha")
                nearest_delta_ms = sample.get("nearest_delta_ms")
                source_gap_ms = sample.get("source_gap_ms")
                if (
                    not isinstance(source_indices, list)
                    or len(source_indices) != 2
                    or not all(isinstance(value, int) and value >= 0 for value in source_indices)
                ):
                    errors.append(f"{sample_prefix} source indices are invalid")
                if (
                    not isinstance(source_timestamps, list)
                    or len(source_timestamps) != 2
                    or not all(_finite_number(value) for value in source_timestamps)
                    or float(source_timestamps[0]) >= float(source_timestamps[1])
                    or (_finite_number(timestamp) and not (
                        float(source_timestamps[0]) <= float(timestamp) <= float(source_timestamps[1])
                    ))
                ):
                    errors.append(f"{sample_prefix} source timestamps are invalid")
                if not _finite_number(alpha) or not 0.0 <= float(alpha) <= 1.0:
                    errors.append(f"{sample_prefix} interpolation alpha is invalid")
                if (
                    not _finite_number(nearest_delta_ms)
                    or float(nearest_delta_ms) > max_camera_arm_delta_ms + 1e-6
                ):
                    errors.append(f"{sample_prefix} camera/arm delta exceeds limit")
                if (
                    not _finite_number(source_gap_ms)
                    or float(source_gap_ms) > max_arm_gap_ms + 1e-6
                ):
                    errors.append(f"{sample_prefix} source gap exceeds limit")
    return rows, errors


def _read_manifest(root: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    manifest = root / "checksums.sha256"
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        try:
            digest, relative = raw.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid checksum line {line_number}") from exc
        relative_path = Path(relative)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise ValueError(f"unsafe checksum line {line_number}")
        if relative in entries:
            raise ValueError(f"duplicate checksum entry: {relative}")
        entries[relative] = digest
    return entries


def validate_episode(
    path: str | Path,
    *,
    verify_video: bool = True,
    require_training_timing: bool = True,
) -> ValidationResult:
    root = Path(path).expanduser().resolve()
    errors: list[str] = []
    metadata: dict[str, Any] | None = None
    if not root.is_dir():
        return ValidationResult(root, False, ("episode directory does not exist",))

    for required in ("READY", "meta.json", "checksums.sha256"):
        if not (root / required).is_file():
            errors.append(f"missing {required}")
    try:
        metadata = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("metadata is not an object")
        if metadata.get("version") != "medicine_act_episode_v1":
            errors.append("unsupported metadata version")
        if metadata.get("status") != "completed":
            errors.append("episode status is not completed")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid meta.json: {exc}")
        metadata = None

    manifest: dict[str, str] = {}
    try:
        manifest = _read_manifest(root)
    except (OSError, ValueError) as exc:
        errors.append(f"invalid checksums.sha256: {exc}")
    for relative, expected in manifest.items():
        candidate = root / relative
        if not candidate.is_file():
            errors.append(f"manifest file missing: {relative}")
        elif _sha256(candidate) != expected:
            errors.append(f"checksum mismatch: {relative}")

    actual_files = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
        and item.relative_to(root).as_posix() not in {"READY", "checksums.sha256"}
    }
    untracked = sorted(actual_files - set(manifest))
    if untracked:
        errors.append("files absent from checksum manifest: " + ", ".join(untracked))

    if metadata is not None:
        files = metadata.get("files", {})
        aligned_relative = files.get("aligned_samples") if isinstance(files, dict) else None
        alignment_meta = metadata.get("act", {}).get("training_alignment", {})
        if not aligned_relative:
            errors.append("metadata has no training-authoritative aligned_samples")
        else:
            aligned_path = root / str(aligned_relative)
            if not aligned_path.is_file():
                errors.append("missing aligned training samples")
            else:
                try:
                    aligned_rows, aligned_errors = _validate_aligned_rows(
                        root,
                        aligned_path,
                        metadata,
                        require_training_timing=require_training_timing,
                    )
                    errors.extend(aligned_errors)
                    aligned_count = len(aligned_rows)
                    expected_aligned = int(
                        alignment_meta.get("aligned_sample_count", -1)
                    )
                    if aligned_count != expected_aligned:
                        errors.append(
                            f"aligned sample count mismatch: {aligned_count} != "
                            f"{expected_aligned}"
                        )
                    if aligned_count < 2:
                        errors.append("aligned training timeline is too short")
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"invalid aligned samples JSONL: {exc}")
        for group in ("observations", "actions"):
            paths = files.get(group, {}) if isinstance(files, dict) else {}
            if not isinstance(paths, dict) or not paths:
                errors.append(f"metadata has no {group} streams")
                continue
            expected_counts = metadata.get(
                "observation_sample_counts" if group == "observations" else "action_sample_counts",
                {},
            )
            for arm, relative in paths.items():
                stream_path = root / str(relative)
                if not stream_path.is_file():
                    errors.append(f"missing {group} stream: {arm}")
                    continue
                try:
                    count = _json_lines(stream_path)
                    if count != int(expected_counts.get(arm, -1)):
                        errors.append(
                            f"{group}/{arm} count mismatch: {count} != "
                            f"{expected_counts.get(arm)}"
                        )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"invalid {group}/{arm} JSONL: {exc}")

        image_streams = files.get("images") if isinstance(files, dict) else None
        if isinstance(image_streams, dict) and image_streams:
            expected_camera_names = metadata.get("camera_names", list(image_streams))
            if list(image_streams) != list(expected_camera_names):
                errors.append("camera_names and files/images order differ")
            camera_counts = metadata.get("camera_frame_counts", {})
            for camera_name, camera_files in image_streams.items():
                if not isinstance(camera_files, dict):
                    errors.append(f"invalid image stream metadata: {camera_name}")
                    continue
                video_relative = camera_files.get("video")
                timestamps_relative = camera_files.get("timestamps")
                frames_relative = camera_files.get("frame_metadata")
                for kind, relative in (
                    ("video", video_relative),
                    ("timestamps", timestamps_relative),
                    ("frame metadata", frames_relative),
                ):
                    if not relative or not (root / str(relative)).is_file():
                        errors.append(f"missing {camera_name} {kind}")
                if frames_relative and (root / str(frames_relative)).is_file():
                    try:
                        metadata_count = _json_lines(root / str(frames_relative))
                        if metadata_count != int(camera_counts.get(camera_name, -1)):
                            errors.append(
                                f"{camera_name} frame metadata count mismatch: "
                                f"{metadata_count} != {camera_counts.get(camera_name)}"
                            )
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        errors.append(f"invalid {camera_name} frame metadata: {exc}")
                if verify_video and video_relative and (root / str(video_relative)).is_file():
                    try:
                        import cv2

                        capture = cv2.VideoCapture(str(root / str(video_relative)))
                        try:
                            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                        finally:
                            capture.release()
                        if frame_count != int(camera_counts.get(camera_name, -1)):
                            errors.append(
                                f"{camera_name} RGB frame count mismatch: {frame_count} != "
                                f"{camera_counts.get(camera_name)}"
                            )
                    except ImportError:
                        errors.append(
                            "opencv is unavailable; use --no-video to skip decoding"
                        )
                        break
        else:
            # Backward compatibility for finalized single-front-camera episodes.
            rgb_relative = files.get("rgb") if isinstance(files, dict) else None
            rgb_path = root / str(rgb_relative or "")
            if not rgb_relative or not rgb_path.is_file():
                errors.append("missing RGB video")
            elif verify_video:
                try:
                    import cv2

                    capture = cv2.VideoCapture(str(rgb_path))
                    try:
                        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    finally:
                        capture.release()
                    if frame_count != int(metadata.get("frame_count", -1)):
                        errors.append(
                            f"RGB frame count mismatch: {frame_count} != "
                            f"{metadata.get('frame_count')}"
                        )
                except ImportError:
                    errors.append(
                        "opencv is unavailable; use --no-video to skip decoding"
                    )

    return ValidationResult(root, not errors, tuple(errors), metadata)


def discover_ready(root: str | Path) -> list[Path]:
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        return []
    return sorted(marker.parent for marker in base.rglob("READY") if marker.is_file())
