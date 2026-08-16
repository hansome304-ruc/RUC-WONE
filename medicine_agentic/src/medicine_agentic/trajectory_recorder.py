"""Hardware-read-only trajectory and calibration episode recorder.

The recorder is deliberately one-way: it reads camera frames and AIRBOT
feedback, and writes an episode to one configured directory.  It has no motion,
mode-switching, suction, chassis, or playback capability. Completed episodes can
be moved to a recorder-owned recovery directory by exact recording ID.
"""
from __future__ import annotations

import json
import hashlib
import bisect
import queue
import re
import shutil
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import cv2
import numpy as np

from medicine_agentic.airbot_readonly import AirbotReadOnly
from medicine_agentic.official_data_collection import OfficialEpisodeRecorder
from medicine_agentic.packaging_camera import CameraFrame, PackagingCamera


PURPOSE_ARMS: dict[str, tuple[str, ...]] = {
    # Existing calibration converters read both action files even when the
    # board is mounted on only one arm. Keep both streams for compatibility.
    "calibration_left": ("left", "right"),
    "calibration_right": ("left", "right"),
    "projection_validation": ("left", "right"),
    "trajectory_left": ("left",),
    "trajectory_right": ("right",),
    "trajectory_both": ("left", "right"),
    "act_left": ("left",),
    "act_right": ("right",),
    "act_bimanual": ("left", "right"),
}
ACTIVE_STATES = frozenset({"starting", "recording", "stopping"})
TSF_HEADER = struct.Struct("<4c4cQ")
TSF_VALUE = struct.Struct("<Q")
SAFE_LABEL = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff._-]+")


class RecordingError(RuntimeError):
    """Base class for recorder failures exposed by the local API."""


class RecordingConflict(RecordingError):
    """Raised when start/stop conflicts with the current state."""


class RecordingUnavailable(RecordingError):
    """Raised when the camera, selected arm, codec, or disk is unavailable."""


class HttpJpegCamera:
    """Read one loopback wrist-camera JPEG stream without controlling hardware."""

    def __init__(
        self,
        *,
        name: str,
        url: str,
        timeout_s: float,
        fps: float,
        max_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ValueError(f"ACT camera {name} must use a loopback HTTP URL")
        self.name = name
        self.url = url
        self.timeout_s = max(0.1, min(float(timeout_s), 10.0))
        self.fps = max(1.0, min(float(fps), 60.0))
        self.max_bytes = max(1024, int(max_bytes))
        self._lock = threading.Lock()
        self._frame_number = 0

    def profile(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": "http_jpeg",
            "source_url": self.url,
            "color": {"fps": self.fps, "format": "bgr8"},
        }

    def capture(self) -> CameraFrame:
        request = Request(self.url, headers={"Accept": "image/jpeg"})
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                content_type = str(response.headers.get("Content-Type", ""))
                if "image/jpeg" not in content_type.lower():
                    raise RecordingUnavailable(
                        f"{self.name} camera returned {content_type or 'non-JPEG data'}"
                    )
                payload = response.read(self.max_bytes + 1)
        except RecordingUnavailable:
            raise
        except Exception as exc:
            raise RecordingUnavailable(
                f"cannot read {self.name} camera: {type(exc).__name__}: {exc}"
            ) from exc
        if not payload or len(payload) > self.max_bytes:
            raise RecordingUnavailable(
                f"{self.name} camera returned an empty or oversized JPEG"
            )
        bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise RecordingUnavailable(f"{self.name} camera JPEG cannot be decoded")
        captured_at = time.time()
        with self._lock:
            self._frame_number += 1
            frame_number = self._frame_number
        return CameraFrame(
            bgr=bgr,
            depth_z16=None,
            depth_scale_m=None,
            captured_at=captured_at,
            frame_number=frame_number,
            device_timestamp_ms=None,
        )


class SharedCameraBundle:
    """Read one atomic three-view bundle from the video gateway seqlock."""

    def __init__(
        self,
        *,
        root: Path,
        camera_names: tuple[str, ...],
        fps: float,
        timeout_s: float,
        max_skew_ms: float,
    ) -> None:
        self.directory = root / "sync"
        self.camera_names = camera_names
        self.fps = max(1.0, min(float(fps), 60.0))
        self.timeout_s = max(0.1, min(float(timeout_s), 10.0))
        self.max_skew_ms = max(1.0, min(float(max_skew_ms), 1000.0))
        self.last_bundle_id: int | None = None
        self.last_metadata: dict[str, Any] | None = None
        self.rejected_bundle_count = 0
        self._maps: dict[str, np.memmap] = {}
        self._map_shapes: dict[str, tuple[int, ...]] = {}

    def profiles(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "name": name,
                "mode": "shared_synchronized_bundle",
                "source": str(self.directory),
                "color": {"fps": self.fps, "format": "bgr8"},
                "max_sync_skew_ms": self.max_skew_ms,
            }
            for name in self.camera_names
        }

    def begin_session(self) -> None:
        """Start a new episode at the latest bundle, then enforce continuity."""
        self.last_bundle_id = None
        self.last_metadata = None
        self.rejected_bundle_count = 0

    def _metadata_path(self, path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != "ruc_video_sync_bundle_v1":
            raise RecordingUnavailable("unsupported synchronized camera bundle")
        if payload.get("state") != "ready":
            raise RecordingUnavailable("synchronized camera bundle is being written")
        if list(payload.get("camera_names", [])) != list(self.camera_names):
            raise RecordingUnavailable("synchronized camera names do not match ACT config")
        return payload

    def _metadata(self) -> dict[str, Any]:
        return self._metadata_path(self.directory / "meta.json")

    def _color(self, details: dict[str, Any]) -> np.ndarray:
        shape = tuple(int(value) for value in details["color_shape"])
        if len(shape) != 3 or shape[2] != 3:
            raise RecordingUnavailable("invalid synchronized camera shape")
        relative = str(details["file"])
        if self._map_shapes.get(relative) != shape:
            self._maps[relative] = np.memmap(
                self.directory / relative,
                dtype=np.uint8,
                mode="r",
                shape=shape,
            )
            self._map_shapes[relative] = shape
        return np.asarray(self._maps[relative]).copy()

    def capture(
        self,
        *,
        latest_only: bool = False,
        captured_after: float | None = None,
    ) -> dict[str, CameraFrame]:
        """Capture a synchronized bundle.

        Recordings use the default contiguous mode so a missed bundle remains a
        hard data-quality failure. Closed-loop rollout uses ``latest_only``:
        inference is slower than the camera producer and must replan from the
        freshest observation instead of replaying stale bundles from the ring.
        ``captured_after`` additionally enforces that every camera exposure is
        newer than a completed action chunk. This prevents the next inference
        from consuming a bundle that was already in flight during execution.
        """
        deadline = time.monotonic() + self.timeout_s
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                latest = self._metadata()
                latest_bundle_id = int(latest["bundle_id"])
                ring_size = int(latest.get("ring_size", 1))
                if ring_size >= 2:
                    if latest_only:
                        if latest_bundle_id == self.last_bundle_id:
                            time.sleep(0.002)
                            continue
                        # A lower id means the producer restarted. For rollout,
                        # the new producer's freshest bundle is still the right
                        # observation; recording continuity remains strict below.
                        bundle_id = latest_bundle_id
                    else:
                        bundle_id = (
                            latest_bundle_id
                            if self.last_bundle_id is None
                            else self.last_bundle_id + 1
                        )
                    if bundle_id > latest_bundle_id:
                        time.sleep(0.002)
                        continue
                    if latest_bundle_id - bundle_id >= ring_size:
                        raise RecordingUnavailable(
                            "synchronized camera ring buffer overrun"
                        )
                    metadata_path = (
                        self.directory
                        / "slots"
                        / str(bundle_id % ring_size)
                        / "meta.json"
                    )
                    first = self._metadata_path(metadata_path)
                    if int(first["bundle_id"]) != bundle_id:
                        time.sleep(0.002)
                        continue
                else:
                    first = latest
                    bundle_id = latest_bundle_id
                    metadata_path = self.directory / "meta.json"
                    if bundle_id == self.last_bundle_id:
                        time.sleep(0.002)
                        continue
                sync_skew_ms = float(first.get("sync_skew_ms", float("inf")))
                if (
                    not np.isfinite(sync_skew_ms)
                    or sync_skew_ms > self.max_skew_ms
                ):
                    self.last_bundle_id = bundle_id
                    self.rejected_bundle_count += 1
                    time.sleep(0.002)
                    continue
                camera_meta = first.get("cameras")
                if not isinstance(camera_meta, dict):
                    raise RecordingUnavailable("synchronized camera metadata is missing")
                sync_timestamps: dict[str, float] = {}
                for name in self.camera_names:
                    details = camera_meta.get(name)
                    if not isinstance(details, dict):
                        raise RecordingUnavailable(
                            f"synchronized {name} metadata is missing"
                        )
                    sync_timestamp = details.get("sync_timestamp_ms")
                    if sync_timestamp is None or not np.isfinite(
                        float(sync_timestamp)
                    ):
                        raise RecordingUnavailable(
                            f"strict ACT requires a synchronized timestamp for {name}"
                        )
                    sync_timestamps[name] = float(sync_timestamp) / 1000.0
                if captured_after is not None and min(
                    sync_timestamps.values()
                ) <= float(captured_after):
                    # Mark this bundle consumed so latest_only waits for a
                    # genuinely post-command exposure instead of rereading it.
                    self.last_bundle_id = bundle_id
                    time.sleep(0.002)
                    continue
                frames: dict[str, CameraFrame] = {}
                for name in self.camera_names:
                    details = camera_meta.get(name)
                    if not isinstance(details, dict):
                        raise RecordingUnavailable(f"synchronized {name} metadata is missing")
                    bgr = self._color(details)
                    timestamp = details.get("device_timestamp_ms")
                    sync_timestamp = details.get("sync_timestamp_ms")
                    timestamp_domain = str(details.get("timestamp_domain", ""))
                    if (
                        timestamp is None
                        or sync_timestamp is None
                        or "global" not in timestamp_domain.lower()
                        or not np.isfinite(float(timestamp))
                        or not np.isfinite(float(sync_timestamp))
                        or abs(float(timestamp) - float(sync_timestamp)) > 1e-3
                    ):
                        raise RecordingUnavailable(
                            f"strict ACT requires a global device timestamp for {name}"
                        )
                    frames[name] = CameraFrame(
                        bgr=bgr,
                        depth_z16=None,
                        depth_scale_m=None,
                        # This field is the training timestamp.  Arrival time is
                        # intentionally not used because USB/encoder scheduling
                        # jitter occurs after sensor exposure.
                        captured_at=sync_timestamps[name],
                        frame_number=bundle_id,
                        device_timestamp_ms=float(timestamp),
                    )
                second = self._metadata_path(metadata_path)
                if int(second["bundle_id"]) != bundle_id:
                    continue
                self.last_bundle_id = bundle_id
                self.last_metadata = second
                return frames
            except (OSError, ValueError, KeyError, json.JSONDecodeError, RecordingError) as exc:
                last_error = exc
                time.sleep(0.002)
        raise RecordingUnavailable(
            "no fresh synchronized camera bundle: "
            f"{last_error or 'timed out waiting for a new bundle'}"
        )


def _safe_label(value: str) -> str:
    normalized = SAFE_LABEL.sub("_", value.strip()).strip("._-")
    return normalized[:48] or "recording"


def _display_label(value: str) -> str:
    normalized = " ".join(value.strip().split())
    return normalized[:80] or "recording"


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _is_act_purpose(purpose: str) -> bool:
    return purpose.startswith("act_")


def _write_checksums(root: Path) -> None:
    """Write a deterministic manifest after every episode stream is closed."""
    entries: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in {"READY", "checksums.sha256"}:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        entries.append(f"{digest.hexdigest()}  {relative}\n")
    (root / "checksums.sha256").write_text("".join(entries), encoding="utf-8")


def _write_tsf_header(stream: Any, *, video_size: int) -> None:
    # On first write reserve the complete header before appending timestamps;
    # on final rewrite preserve the existing end position.
    offset = max(stream.tell(), TSF_HEADER.size)
    stream.seek(0)
    stream.write(
        TSF_HEADER.pack(
            b"T",
            b"S",
            b"F",
            b"1",
            b"Q",
            b" ",
            b" ",
            b" ",
            int(video_size),
        )
    )
    stream.seek(offset)


@dataclass
class RecordingSession:
    recording_id: str
    label: str
    purpose: str
    arms: tuple[str, ...]
    camera_names: tuple[str, ...]
    temporary_dir: Path
    final_dir: Path
    requested_at: float
    state: str = "starting"
    started_at: float | None = None
    finished_at: float | None = None
    frame_count: int = 0
    duplicate_frame_count: int = 0
    camera_frame_counts: dict[str, int] = field(default_factory=dict)
    camera_duplicate_frame_counts: dict[str, int] = field(default_factory=dict)
    arm_sample_counts: dict[str, int] = field(default_factory=dict)
    observation_sample_counts: dict[str, int] = field(default_factory=dict)
    action_sample_counts: dict[str, int] = field(default_factory=dict)
    last_frame_number: int | None = None
    last_camera_frame_numbers: dict[str, int | None] = field(default_factory=dict)
    error: str | None = None
    saved_path: str | None = None
    camera_profile: dict[str, Any] | None = None
    camera_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    sync_skew_ms_samples: list[float] = field(default_factory=list)
    sync_rejected_bundle_count: int = 0
    sync_missed_bundle_count: int = 0
    encoder_backpressure_drop_count: int = 0
    aligned_sample_count: int = 0
    alignment_rejected_count: int = 0
    alignment_trimmed_edge_count: int = 0
    arm_pair_skew_ms_samples: list[float] = field(default_factory=list)
    source_capture_skew_ms_samples: list[float] = field(default_factory=list)
    max_arm_sample_gap_ms: float = 0.0
    last_sync_bundle_id: int | None = None
    leader_host: str | None = None
    leader_ports: dict[str, int] = field(default_factory=dict)
    action_from_observation_arms: tuple[str, ...] = ()
    preview_frame: CameraFrame | None = None
    preview_captured_at: float | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    worker: threading.Thread | None = None

    def snapshot(self) -> dict[str, Any]:
        now = self.finished_at or time.time()
        start = self.started_at or self.requested_at
        return {
            "active": self.state in ACTIVE_STATES,
            "id": self.recording_id,
            "label": self.label,
            "purpose": self.purpose,
            "arms": list(self.arms),
            "state": self.state,
            "requested_at": self.requested_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": max(0.0, now - start),
            "frame_count": self.frame_count,
            "duplicate_frame_count": self.duplicate_frame_count,
            "camera_names": list(self.camera_names),
            "camera_frame_counts": dict(self.camera_frame_counts),
            "camera_duplicate_frame_counts": dict(
                self.camera_duplicate_frame_counts
            ),
            "arm_sample_counts": dict(self.arm_sample_counts),
            "observation_sample_counts": dict(self.observation_sample_counts),
            "action_sample_counts": dict(self.action_sample_counts),
            "action_from_observation_arms": list(
                self.action_from_observation_arms
            ),
            "sync_rejected_bundle_count": self.sync_rejected_bundle_count,
            "sync_missed_bundle_count": self.sync_missed_bundle_count,
            "encoder_backpressure_drop_count": (
                self.encoder_backpressure_drop_count
            ),
            "aligned_sample_count": self.aligned_sample_count,
            "alignment_rejected_count": self.alignment_rejected_count,
            "alignment_trimmed_edge_count": self.alignment_trimmed_edge_count,
            "last_frame_number": self.last_frame_number,
            "error": self.error,
            "saved_path": self.saved_path,
            "target_path": str(self.final_dir),
            "preview_ready": self.preview_frame is not None,
            "preview_captured_at": self.preview_captured_at,
        }


class TrajectoryRecorder:
    """Record one calibration-compatible or multi-camera ACT episode at a time."""

    def __init__(
        self,
        camera: PackagingCamera,
        config: dict[str, Any],
        *,
        config_dir: Path,
        arm_factory: Callable[..., Any] = AirbotReadOnly,
        act_cameras: dict[str, Any] | None = None,
        act_bundle: Any | None = None,
    ) -> None:
        self.camera = camera
        self.enabled = bool(config.get("enabled", False))
        configured_root = Path(
            str(config.get("output_dir", "../recordings/trajectories"))
        ).expanduser()
        self.output_dir = (
            configured_root
            if configured_root.is_absolute()
            else config_dir / configured_root
        ).resolve()
        configured_act_root = Path(
            str(config.get("act_output_dir", "../recordings/act/finalized"))
        ).expanduser()
        self.act_output_dir = (
            configured_act_root
            if configured_act_root.is_absolute()
            else config_dir / configured_act_root
        ).resolve()
        configured_act_failed = Path(
            str(config.get("act_failed_dir", "../recordings/act/failed"))
        ).expanduser()
        self.act_failed_dir = (
            configured_act_failed
            if configured_act_failed.is_absolute()
            else config_dir / configured_act_failed
        ).resolve()
        self.arm_host = str(config.get("arm_host", "localhost"))
        raw_ports = config.get("arm_ports", {"left": 50051, "right": 50053})
        if not isinstance(raw_ports, dict):
            raise ValueError("trajectory_recorder.arm_ports must be an object")
        self.arm_ports = {
            "left": int(raw_ports.get("left", 50051)),
            "right": int(raw_ports.get("right", 50053)),
        }
        raw_act = config.get("act", {})
        if not isinstance(raw_act, dict):
            raise ValueError("trajectory_recorder.act must be an object")
        self.leader_host = str(raw_act.get("leader_host", "10.47.157.9"))
        raw_leader_ports = raw_act.get(
            "leader_ports", {"left": 50050, "right": 50052}
        )
        if not isinstance(raw_leader_ports, dict):
            raise ValueError("trajectory_recorder.act.leader_ports must be an object")
        self.leader_ports = {
            "left": int(raw_leader_ports.get("left", 50050)),
            "right": int(raw_leader_ports.get("right", 50052)),
        }
        raw_camera_names = raw_act.get("camera_names", ["front"])
        if not isinstance(raw_camera_names, list) or not raw_camera_names:
            raise ValueError("trajectory_recorder.act.camera_names must be a list")
        camera_names = tuple(str(name).strip() for name in raw_camera_names)
        if (
            camera_names[0] != "front"
            or any(not name for name in camera_names)
            or len(set(camera_names)) != len(camera_names)
        ):
            raise ValueError(
                "trajectory_recorder.act.camera_names must start with front "
                "and contain unique non-empty names"
            )
        self.act_camera_names = camera_names
        raw_bundle = raw_act.get("synchronized_bundle", {})
        if not isinstance(raw_bundle, dict):
            raise ValueError(
                "trajectory_recorder.act.synchronized_bundle must be an object"
            )
        self.act_max_sync_skew_ms = max(
            1.0, min(float(raw_bundle.get("max_skew_ms", 30.0)), 1000.0)
        )
        self.act_encoder_queue_size = max(
            2, min(int(raw_bundle.get("encoder_queue_size", 12)), 120)
        )
        self.act_require_contiguous_bundles = bool(
            raw_bundle.get("require_contiguous_bundles", True)
        )
        raw_alignment = raw_act.get("alignment", {})
        if not isinstance(raw_alignment, dict):
            raise ValueError("trajectory_recorder.act.alignment must be an object")
        self.act_max_arm_pair_skew_ms = max(
            1.0, min(float(raw_alignment.get("max_arm_pair_skew_ms", 25.0)), 500.0)
        )
        self.act_max_source_capture_skew_ms = max(
            1.0,
            min(float(raw_alignment.get("max_source_capture_skew_ms", 25.0)), 500.0),
        )
        self.act_max_arm_sample_gap_ms = max(
            10.0, min(float(raw_alignment.get("max_arm_sample_gap_ms", 60.0)), 1000.0)
        )
        self.act_max_camera_arm_delta_ms = max(
            1.0,
            min(float(raw_alignment.get("max_camera_arm_delta_ms", 30.0)), 500.0),
        )
        self.act_preview_max_camera_arm_delta_ms = max(
            self.act_max_camera_arm_delta_ms,
            min(
                float(raw_alignment.get("preview_max_camera_arm_delta_ms", 120.0)),
                500.0,
            ),
        )
        self.act_preview_max_joint_delta_rad = max(
            1e-5,
            min(float(raw_alignment.get("preview_max_joint_delta_rad", 0.002)), 0.1),
        )
        self.act_preview_max_eef_delta_m = max(
            1e-6,
            min(float(raw_alignment.get("preview_max_eef_delta_m", 0.001)), 0.05),
        )
        self.act_min_aligned_fraction = max(
            0.5, min(float(raw_alignment.get("min_aligned_fraction", 0.98)), 1.0)
        )
        self._act_bundle = act_bundle
        if self._act_bundle is None and raw_bundle.get("enabled") is True:
            configured_bundle_root = Path(
                str(raw_bundle.get("runtime_root", "/dev/shm/ruc-video"))
            ).expanduser()
            bundle_root = (
                configured_bundle_root
                if configured_bundle_root.is_absolute()
                else config_dir / configured_bundle_root
            ).resolve()
            self._act_bundle = SharedCameraBundle(
                root=bundle_root,
                camera_names=self.act_camera_names,
                fps=float(raw_bundle.get("fps", 30.0)),
                timeout_s=float(raw_bundle.get("timeout_s", 2.0)),
                max_skew_ms=self.act_max_sync_skew_ms,
            )
        provided_cameras = dict(act_cameras or {})
        raw_external_cameras = raw_act.get("external_cameras", {})
        if not isinstance(raw_external_cameras, dict):
            raise ValueError(
                "trajectory_recorder.act.external_cameras must be an object"
            )
        for name in self.act_camera_names[1:]:
            if self._act_bundle is not None:
                continue
            if name in provided_cameras:
                continue
            camera_config = raw_external_cameras.get(name)
            if not isinstance(camera_config, dict):
                raise ValueError(f"ACT camera {name} has no external_cameras config")
            url = str(camera_config.get("url", "")).strip()
            if not url:
                raise ValueError(f"ACT camera {name} has no URL")
            provided_cameras[name] = HttpJpegCamera(
                name=name,
                url=url,
                timeout_s=float(camera_config.get("timeout_s", 2.0)),
                fps=float(camera_config.get("fps", 30.0)),
            )
        unknown_cameras = set(provided_cameras) - set(self.act_camera_names[1:])
        if unknown_cameras:
            raise ValueError(
                "unexpected ACT cameras: " + ", ".join(sorted(unknown_cameras))
            )
        self._act_external_cameras = provided_cameras
        raw_runtime_config = str(raw_act.get("leader_runtime_config", "")).strip()
        if raw_runtime_config:
            configured_runtime = Path(raw_runtime_config).expanduser()
            self.leader_runtime_config = (
                configured_runtime
                if configured_runtime.is_absolute()
                else config_dir / configured_runtime
            ).resolve()
        else:
            self.leader_runtime_config = None
        self.camera_fps_limit = max(
            1.0, min(float(config.get("camera_fps_limit", 30.0)), 60.0)
        )
        self.arm_sample_hz = max(
            1.0, min(float(config.get("arm_sample_hz", 50.0)), 250.0)
        )
        self.max_duration_s = max(
            5.0, min(float(config.get("max_duration_s", 180.0)), 1800.0)
        )
        self.min_free_bytes = max(
            0, int(float(config.get("min_free_gb", 2.0)) * 1024**3)
        )
        self._arm_factory = arm_factory
        self._lock = threading.RLock()
        self._session: RecordingSession | None = None
        self._last_session: RecordingSession | None = None

    def _resolve_leader_endpoint(self) -> tuple[str, dict[str, int]]:
        """Resolve the active Leader host at the start of every ACT episode."""
        host = self.leader_host
        ports = dict(self.leader_ports)
        if self.leader_runtime_config is None:
            return host, ports
        try:
            payload = json.loads(
                self.leader_runtime_config.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RecordingUnavailable(
                f"cannot read active Leader config {self.leader_runtime_config}: {exc}"
            ) from exc
        runtime_host = str(payload.get("remote_host", "")).strip()
        if not runtime_host:
            raise RecordingUnavailable(
                f"active Leader config {self.leader_runtime_config} has no remote_host"
            )
        runtime_ports = payload.get("leader_ports")
        if runtime_ports is not None:
            if not isinstance(runtime_ports, dict):
                raise RecordingUnavailable("active Leader leader_ports must be an object")
            ports = {
                "left": int(runtime_ports.get("left", ports["left"])),
                "right": int(runtime_ports.get("right", ports["right"])),
            }
        return runtime_host, ports

    def _source_fps(self, camera: Any) -> float:
        profile = camera.profile()
        source = (
            profile.get("color")
            if profile.get("mode") == "realsense"
            else profile.get("source_color_profile")
        )
        if not isinstance(source, dict):
            source = profile.get("color")
        try:
            fps = float(source.get("fps", self.camera_fps_limit))
        except (AttributeError, TypeError, ValueError):
            fps = self.camera_fps_limit
        return max(1.0, min(fps, self.camera_fps_limit))

    def _session_cameras(self, session: RecordingSession) -> dict[str, Any]:
        cameras: dict[str, Any] = {"front": self.camera}
        if _is_act_purpose(session.purpose):
            for name in session.camera_names[1:]:
                cameras[name] = self._act_external_cameras[name]
        return cameras

    def capture_act_frames(
        self,
        begin_stream: bool = False,
        *,
        captured_after: float | None = None,
    ) -> dict[str, CameraFrame]:
        """Capture one synchronized ACT camera bundle without arm access.

        The rollout controller owns persistent follower-arm command connections,
        so it must not use capture_act_observation(), which opens a second pair
        of robot clients for each inference. Holding the recorder lock prevents
        a recording from starting during this bounded camera read.
        """

        with self._lock:
            if not self.enabled:
                raise RecordingUnavailable("trajectory recorder is disabled")
            if self._session is not None and self._session.state in ACTIVE_STATES:
                raise RecordingConflict(
                    "cannot capture ACT rollout frames while recording is active"
                )
            if tuple(self.act_camera_names) != (
                "front",
                "left_wrist",
                "right_wrist",
            ):
                raise RecordingUnavailable(
                    "ACT rollout requires front, left_wrist and right_wrist cameras"
                )
            if self._act_bundle is not None:
                if begin_stream:
                    self._act_bundle.begin_session()
                return self._act_bundle.capture(
                    latest_only=True,
                    captured_after=captured_after,
                )
            cameras: dict[str, Any] = {
                "front": self.camera,
                **self._act_external_cameras,
            }
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {
                    name: pool.submit(cameras[name].capture)
                    for name in self.act_camera_names
                }
                frames = {
                    name: futures[name].result() for name in self.act_camera_names
                }
            if captured_after is not None and min(
                float(frame.captured_at) for frame in frames.values()
            ) <= float(captured_after):
                raise RecordingUnavailable(
                    "ACT camera exposure is not newer than the completed action chunk"
                )
            return frames

    def capture_act_observation(self) -> dict[str, Any]:
        """Capture one synchronized, hardware-read-only ACT observation.

        The recorder lock prevents a recording from starting while the preview
        observation is being assembled. This method never opens a Leader link and
        has no path that can issue a robot command.
        """

        with self._lock:
            if not self.enabled:
                raise RecordingUnavailable("trajectory recorder is disabled")
            if self._session is not None and self._session.state in ACTIVE_STATES:
                raise RecordingConflict("cannot preview ACT while a recording is active")
            if tuple(self.act_camera_names) != (
                "front",
                "left_wrist",
                "right_wrist",
            ):
                raise RecordingUnavailable(
                    "ACT preview requires front, left_wrist and right_wrist cameras"
                )

            def capture_frames() -> dict[str, CameraFrame]:
                if self._act_bundle is not None:
                    self._act_bundle.begin_session()
                    return self._act_bundle.capture()
                cameras: dict[str, Any] = {
                    "front": self.camera,
                    **self._act_external_cameras,
                }
                with ThreadPoolExecutor(
                    max_workers=3,
                    thread_name_prefix="act-preview-cameras",
                ) as pool:
                    futures = {
                        name: pool.submit(camera.capture)
                        for name, camera in cameras.items()
                    }
                    return {name: futures[name].result() for name in self.act_camera_names}

            def state_from_feedback(sample: dict[str, Any]) -> list[float]:
                values: list[float] = []
                for name in ("left", "right"):
                    arm = sample.get("arms", {}).get(name, {})
                    joints = np.asarray(
                        arm.get("joint_position_rad"), dtype=np.float64
                    )
                    eef = np.asarray(
                        arm.get("eef_feedback_m", []), dtype=np.float64
                    ).reshape(-1)
                    if joints.shape != (6,) or not np.all(np.isfinite(joints)):
                        raise RecordingUnavailable(f"{name} ACT joints are invalid")
                    if not np.all(np.isfinite(eef)):
                        raise RecordingUnavailable(
                            f"{name} ACT gripper feedback is invalid"
                        )
                    values.extend(float(value) for value in joints)
                    values.append(float(eef[0]) if eef.size else 0.0)
                return values

            follower = self._arm_factory(
                host=self.arm_host,
                ports=self.arm_ports,
                arm_names=("left", "right"),
            )
            timing_validation = "strict"
            static_joint_delta_rad = 0.0
            static_eef_delta_m = 0.0
            try:
                follower.connect()
                try:
                    capture_feedback = getattr(
                        follower,
                        "capture_selected_fast",
                        follower.capture_selected,
                    )
                    with ThreadPoolExecutor(
                        max_workers=2,
                        thread_name_prefix="act-preview-observation",
                    ) as pool:
                        frames_future = pool.submit(capture_frames)
                        feedback_future = pool.submit(capture_feedback)
                        frames = frames_future.result()
                        feedback = feedback_future.result()
                    state = state_from_feedback(feedback)
                    if set(frames) != set(self.act_camera_names):
                        raise RecordingUnavailable("ACT camera bundle is incomplete")
                    camera_timestamps = np.asarray(
                        [
                            float(frames[name].captured_at)
                            for name in self.act_camera_names
                        ],
                        dtype=np.float64,
                    )
                    feedback_timestamp = float(feedback["timestamp_ns"]) / 1e9
                    camera_arm_delta_ms = float(
                        np.max(np.abs(camera_timestamps - feedback_timestamp))
                        * 1000.0
                    )
                    if (
                        not np.all(np.isfinite(camera_timestamps))
                        or not np.isfinite(feedback_timestamp)
                        or camera_arm_delta_ms
                        > self.act_preview_max_camera_arm_delta_ms
                    ):
                        raise RecordingUnavailable(
                            "ACT preview camera/arm timing is stale: "
                            f"{camera_arm_delta_ms:.3f} ms > "
                            f"{self.act_preview_max_camera_arm_delta_ms:.3f} ms"
                        )
                    if camera_arm_delta_ms > self.act_max_camera_arm_delta_ms:
                        confirmation = capture_feedback()
                        confirmation_state = state_from_feedback(confirmation)
                        initial = np.asarray(state, dtype=np.float64)
                        confirmed = np.asarray(confirmation_state, dtype=np.float64)
                        joint_indices = np.asarray(
                            [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
                        )
                        eef_indices = np.asarray([6, 13])
                        static_joint_delta_rad = float(
                            np.max(np.abs(confirmed[joint_indices] - initial[joint_indices]))
                        )
                        static_eef_delta_m = float(
                            np.max(np.abs(confirmed[eef_indices] - initial[eef_indices]))
                        )
                        if (
                            static_joint_delta_rad
                            > self.act_preview_max_joint_delta_rad
                            or static_eef_delta_m > self.act_preview_max_eef_delta_m
                        ):
                            raise RecordingUnavailable(
                                "ACT preview observation is delayed and the arms are moving: "
                                f"joint delta {static_joint_delta_rad:.6f} rad, "
                                f"EEF delta {static_eef_delta_m:.6f} m"
                            )
                        timing_validation = "static_delay_compensation"
                finally:
                    follower.close()
            except Exception as exc:
                raise RecordingUnavailable(f"cannot capture ACT observation: {exc}") from exc
            return {
                "state": state,
                "frames_bgr": {
                    name: np.asarray(frames[name].bgr).copy()
                    for name in self.act_camera_names
                },
                "captured_at": feedback_timestamp,
                "camera_arm_delta_ms": camera_arm_delta_ms,
                "arm_pair_skew_ms": float(feedback.get("paired_sample_skew_ms", 0.0)),
                "timing_validation": timing_validation,
                "static_joint_delta_rad": static_joint_delta_rad,
                "static_eef_delta_m": static_eef_delta_m,
                "hardware_access": "feedback_only",
            }

    def status(self) -> dict[str, Any]:
        with self._lock:
            current = self._session or self._last_session
            if current is None:
                return {
                    "enabled": self.enabled,
                    "active": False,
                    "state": "idle",
                    "output_dir": str(self.output_dir),
                    "act_output_dir": str(self.act_output_dir),
                    "allowed_purposes": sorted(PURPOSE_ARMS),
                }
            payload = current.snapshot()
        payload.update(
            {
                "enabled": self.enabled,
                "output_dir": str(self.output_dir),
                "act_output_dir": str(self.act_output_dir),
                "allowed_purposes": sorted(PURPOSE_ARMS),
            }
        )
        return payload

    def list_recordings(self, *, limit: int = 20) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        metadata_paths = list(self.output_dir.glob("*/meta.json"))
        metadata_paths.extend(self.act_output_dir.glob("*/meta.json"))
        metadata_paths.extend(self.act_output_dir.glob("*/episode_*/meta.json"))
        metadata_paths.extend(self.act_failed_dir.glob("*/meta.json"))
        for metadata_path in metadata_paths:
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            extra = payload.get("extra")
            if not isinstance(extra, dict):
                extra = {}
            task_meta = payload.get("task_meta")
            if not isinstance(task_meta, dict):
                task_meta = {}
            is_official = (
                extra.get("recording_strategy")
                in {
                    "official_open_pdd_shared_rgbd_v1",
                    "official_open_pdd_direct_rgbd_v1",
                }
            )
            recording_id = extra.get(
                "recording_id", payload.get("recording_id", metadata_path.parent.name)
            )
            label = extra.get(
                "label", payload.get("label", task_meta.get("prompt", "recording"))
            )
            purpose = extra.get(
                "purpose", payload.get("purpose", task_meta.get("task_name", "unknown"))
            )
            created_at = payload.get("created_at")
            finished_at = payload.get("finished_at")
            duration_s = payload.get("duration_s")
            if duration_s is None and created_at is not None and finished_at is not None:
                duration_s = max(0.0, float(finished_at) - float(created_at))
            official_action_counts: dict[str, int] = {}
            official_camera_counts: dict[str, int] = {}
            if is_official:
                for arm in ("left", "right"):
                    action_path = (
                        metadata_path.parent / "actions" / f"{arm}_arm.jsonl"
                    )
                    try:
                        with action_path.open("r", encoding="utf-8") as stream:
                            official_action_counts[arm] = sum(1 for _ in stream)
                    except OSError:
                        official_action_counts[arm] = 0
                for source, logical in (
                    ("front", "front"),
                    ("left", "left_wrist"),
                    ("right", "right_wrist"),
                ):
                    tsf_path = (
                        metadata_path.parent
                        / "sensors"
                        / f"cam_{source}_rgb.mp4.tsf"
                    )
                    try:
                        official_camera_counts[logical] = max(
                            0, (tsf_path.stat().st_size - TSF_HEADER.size) // 8
                        )
                    except OSError:
                        official_camera_counts[logical] = 0
            items.append(
                {
                    "id": recording_id,
                    "label": label,
                    "purpose": purpose,
                    "arms": (["left", "right"] if is_official else payload.get("selected_arms", [])),
                    "status": ("completed" if is_official else payload.get("status", "unknown")),
                    "created_at": created_at,
                    "finished_at": finished_at,
                    "duration_s": duration_s,
                    "frame_count": official_camera_counts.get(
                        "front", payload.get("frame_count", 0)
                    ),
                    "camera_names": (
                        ["front", "left_wrist", "right_wrist"]
                        if is_official
                        else payload.get("camera_names", ["front"])
                    ),
                    "camera_frame_counts": official_camera_counts or payload.get(
                        "camera_frame_counts", {"front": payload.get("frame_count", 0)}
                    ),
                    "arm_sample_counts": official_action_counts or payload.get(
                        "arm_sample_counts", {}
                    ),
                    "observation_sample_counts": payload.get(
                        "observation_sample_counts", {}
                    ),
                    "action_sample_counts": payload.get("action_sample_counts", {}),
                    "path": str(metadata_path.parent),
                    "error": None if is_official else payload.get("error"),
                    "replay_ready": bool(
                        (
                            is_official
                            and str(purpose).startswith("act_")
                            and bool(payload.get("finished_at"))
                            and bool(official_action_counts)
                            and all(
                                count >= 2
                                and (
                                    metadata_path.parent
                                    / "actions"
                                    / f"{arm}_arm.jsonl"
                                ).is_file()
                                for arm, count in official_action_counts.items()
                            )
                        )
                        or (
                            not is_official
                            and str(payload.get("purpose", "")).startswith("act_")
                            and payload.get("status") == "completed"
                            and isinstance(payload.get("files"), dict)
                            and isinstance(
                                payload.get("files", {}).get("aligned_samples"),
                                str,
                            )
                            and (
                                metadata_path.parent
                                / payload.get("files", {}).get("aligned_samples")
                            ).is_file()
                        )
                    ),
                }
            )
        items.sort(
            key=lambda item: float(item.get("created_at") or 0.0),
            reverse=True,
        )
        return items[: max(1, min(int(limit), 100))]

    def delete_recording(self, recording_id: str) -> dict[str, Any]:
        """Remove one completed episode from the UI and archive it for recovery."""
        if not isinstance(recording_id, str) or not recording_id.strip():
            raise ValueError("recording_id must be a non-empty string")
        recording_id = recording_id.strip()

        with self._lock:
            if self._session is not None and self._session.state in ACTIVE_STATES:
                raise RecordingConflict("stop recording before deleting an episode")

            matches: list[Path] = []
            roots = (self.output_dir, self.act_output_dir, self.act_failed_dir)
            for root in roots:
                metadata_paths = list(root.glob("*/meta.json"))
                metadata_paths.extend(root.glob("*/episode_*/meta.json"))
                for metadata_path in metadata_paths:
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    if not isinstance(metadata, dict):
                        continue
                    extra = metadata.get("extra")
                    if not isinstance(extra, dict):
                        extra = {}
                    candidate_id = extra.get(
                        "recording_id",
                        metadata.get("recording_id", metadata_path.parent.name),
                    )
                    if candidate_id == recording_id:
                        matches.append(metadata_path.parent)

            if not matches:
                raise FileNotFoundError(f"recording not found: {recording_id}")
            if len(matches) != 1:
                raise RecordingConflict(
                    f"recording ID is not unique and cannot be deleted: {recording_id}"
                )

            source = matches[0]
            recordings_root = self.output_dir.parent
            trash_root = recordings_root / ".trash" / "deleted-by-ui"
            trash_root.mkdir(parents=True, exist_ok=True)
            deleted_at = time.time()
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(deleted_at))
            destination = trash_root / (
                f"{stamp}_{uuid.uuid4().hex[:8]}_{source.name}"
            )
            shutil.move(str(source), str(destination))

            if (
                self._last_session is not None
                and self._last_session.recording_id == recording_id
            ):
                self._last_session = None

        return {
            "recording_id": recording_id,
            "deleted": True,
            "recoverable": True,
            "deleted_at": deleted_at,
            "archived_path": str(destination),
        }

    def active_preview_frame(self) -> tuple[bool, CameraFrame | None]:
        """Return the recorder-owned preview without acquiring another frame."""
        with self._lock:
            session = self._session
            if session is None or session.state not in ACTIVE_STATES:
                return False, None
            return True, session.preview_frame

    def start(
        self,
        *,
        label: str,
        purpose: str,
        action_from_observation_arms: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RecordingUnavailable("trajectory recording is disabled")
        if purpose not in PURPOSE_ARMS:
            raise ValueError(
                "purpose must be one of: " + ", ".join(sorted(PURPOSE_ARMS))
            )
        action_from_observation_arms = tuple(
            dict.fromkeys(str(arm) for arm in action_from_observation_arms)
        )
        invalid_action_sources = set(action_from_observation_arms) - set(
            PURPOSE_ARMS[purpose]
        )
        if invalid_action_sources:
            raise ValueError(
                "action_from_observation_arms must be selected recording arms: "
                + ", ".join(sorted(invalid_action_sources))
            )
        if action_from_observation_arms and not _is_act_purpose(purpose):
            raise ValueError(
                "action_from_observation_arms is supported only for ACT recordings"
            )
        with self._lock:
            if self._session is not None and self._session.state in ACTIVE_STATES:
                raise RecordingConflict("another recording is already active")
            target_root = self.act_output_dir if _is_act_purpose(purpose) else self.output_dir
            target_root.mkdir(parents=True, exist_ok=True)
            if _is_act_purpose(purpose):
                self.act_failed_dir.mkdir(parents=True, exist_ok=True)
            free_bytes = shutil.disk_usage(target_root).free
            if free_bytes < self.min_free_bytes:
                raise RecordingUnavailable(
                    f"free disk space is below {self.min_free_bytes / 1024**3:.1f} GiB"
                )
            now = time.time()
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
            display_label = _display_label(label)
            path_label = _safe_label(display_label)
            prefix = "act" if _is_act_purpose(purpose) else "trajectory"
            recording_id = f"{prefix}_{stamp}_{path_label}_{uuid.uuid4().hex[:8]}"
            temporary_dir = target_root / f".{recording_id}.inprogress"
            final_dir = target_root / recording_id
            camera_names = self.act_camera_names if _is_act_purpose(purpose) else ("front",)
            session = RecordingSession(
                recording_id=recording_id,
                label=display_label,
                purpose=purpose,
                arms=PURPOSE_ARMS[purpose],
                camera_names=camera_names,
                temporary_dir=temporary_dir,
                final_dir=final_dir,
                requested_at=now,
                action_from_observation_arms=action_from_observation_arms,
                arm_sample_counts={
                    name: 0 for name in PURPOSE_ARMS[purpose]
                },
                observation_sample_counts=(
                    {name: 0 for name in PURPOSE_ARMS[purpose]}
                    if _is_act_purpose(purpose)
                    else {}
                ),
                action_sample_counts=(
                    {name: 0 for name in PURPOSE_ARMS[purpose]}
                    if _is_act_purpose(purpose)
                    else {}
                ),
                camera_frame_counts={name: 0 for name in camera_names},
                camera_duplicate_frame_counts={name: 0 for name in camera_names},
                last_camera_frame_numbers={name: None for name in camera_names},
            )
            self._session = session
            worker = threading.Thread(
                target=self._run_session,
                args=(session,),
                name=f"trajectory-recorder-{recording_id}",
                daemon=True,
            )
            session.worker = worker
            worker.start()
            return session.snapshot()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            session = self._session
            if session is None or session.state not in ACTIVE_STATES:
                current = self._last_session.snapshot() if self._last_session else None
                if current is not None and current.get("state") in {"saved", "error"}:
                    return current
                raise RecordingConflict("no recording is active")
            if session.state != "stopping":
                session.state = "stopping"
                session.stop_event.set()
            return session.snapshot()

    def close(self, *, timeout_s: float = 8.0) -> None:
        with self._lock:
            session = self._session
            if session is None or session.state not in ACTIVE_STATES:
                return
            session.state = "stopping"
            session.stop_event.set()
            worker = session.worker
        if worker is not None:
            worker.join(timeout=max(0.1, timeout_s))

    def _set_error(self, session: RecordingSession, exc: BaseException) -> None:
        with self._lock:
            if session.error is None:
                session.error = f"{type(exc).__name__}: {exc}"
            session.stop_event.set()

    def _write_arm_sample(
        self,
        streams: dict[str, Any],
        sample: dict[str, Any],
        session: RecordingSession,
        *,
        counter_name: str,
        cycle_index: int | None = None,
        cycle_timestamp_ns: int | None = None,
        source_capture_skew_ms: float | None = None,
    ) -> None:
        for name in session.arms:
            arm = sample["arms"][name]
            eef = list(arm.get("eef_feedback_m", []))
            payload = {
                "joint_positions": list(arm["joint_position_rad"]),
                "ee_positions": (
                    list(arm["flange_position_m"])
                    + list(arm["flange_quaternion_xyzw"])
                ),
                "gripper": float(eef[0]) if eef else 0.0,
                "timestamp": float(arm["timestamp_ns"]) / 1e9,
                "driver_state": arm.get("driver_state"),
                "control_mode": arm.get("control_mode"),
                "paired_sample_skew_ms": sample.get("paired_sample_skew_ms", 0.0),
                "capture_duration_ms": sample.get("capture_duration_ms"),
                "capture_cycle_index": cycle_index,
                "capture_cycle_timestamp": (
                    float(cycle_timestamp_ns) / 1e9
                    if cycle_timestamp_ns is not None
                    else None
                ),
                "source_capture_skew_ms": source_capture_skew_ms,
            }
            streams[name].write(_json_line(payload))
            streams[name].flush()
            with self._lock:
                counters = getattr(session, counter_name)
                counters[name] += 1
                if _is_act_purpose(session.purpose):
                    session.arm_sample_counts[name] = min(
                        session.observation_sample_counts[name],
                        session.action_sample_counts[name],
                    )
                else:
                    session.arm_sample_counts[name] = counters[name]

    def _act_arm_loop(
        self,
        session: RecordingSession,
        follower: Any,
        leader: Any,
        observation_streams: dict[str, Any],
        action_streams: dict[str, Any],
    ) -> None:
        """Capture follower and leader pairs in one shared sampling cycle."""

        period_s = 1.0 / self.arm_sample_hz
        next_tick = time.monotonic()
        cycle_index = 0
        try:
            with ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="act-feedback-sources",
            ) as source_pool:
                while not session.stop_event.is_set():
                    cycle_started_ns = time.time_ns()
                    follower_capture = getattr(
                        follower,
                        "capture_selected_fast",
                        follower.capture_selected,
                    )
                    leader_capture = getattr(
                        leader,
                        "capture_selected_fast",
                        leader.capture_selected,
                    )
                    follower_future = source_pool.submit(follower_capture)
                    leader_future = source_pool.submit(leader_capture)
                    follower_sample = follower_future.result()
                    leader_sample = leader_future.result()
                    cycle_completed_ns = time.time_ns()
                    cycle_timestamp_ns = (cycle_started_ns + cycle_completed_ns) // 2
                    source_capture_skew_ms = abs(
                        int(follower_sample["timestamp_ns"])
                        - int(leader_sample["timestamp_ns"])
                    ) / 1e6
                    with self._lock:
                        session.arm_pair_skew_ms_samples.extend(
                            (
                                float(follower_sample.get("paired_sample_skew_ms", 0.0)),
                                float(leader_sample.get("paired_sample_skew_ms", 0.0)),
                            )
                        )
                        session.source_capture_skew_ms_samples.append(
                            source_capture_skew_ms
                        )
                    self._write_arm_sample(
                        observation_streams,
                        follower_sample,
                        session,
                        counter_name="observation_sample_counts",
                        cycle_index=cycle_index,
                        cycle_timestamp_ns=cycle_timestamp_ns,
                        source_capture_skew_ms=source_capture_skew_ms,
                    )
                    action_sample = leader_sample
                    if session.action_from_observation_arms:
                        action_sample = dict(leader_sample)
                        action_arms = dict(leader_sample["arms"])
                        for name in session.action_from_observation_arms:
                            action_arms[name] = follower_sample["arms"][name]
                        action_sample["arms"] = action_arms
                    self._write_arm_sample(
                        action_streams,
                        action_sample,
                        session,
                        counter_name="action_sample_counts",
                        cycle_index=cycle_index,
                        cycle_timestamp_ns=cycle_timestamp_ns,
                        source_capture_skew_ms=source_capture_skew_ms,
                    )
                    cycle_index += 1
                    next_tick += period_s
                    session.stop_event.wait(
                        max(0.0, next_tick - time.monotonic())
                    )
        except Exception as exc:
            self._set_error(session, exc)

    def _arm_loop(
        self,
        session: RecordingSession,
        robot: Any,
        streams: dict[str, Any],
        *,
        first_sample: dict[str, Any],
        counter_name: str = "arm_sample_counts",
    ) -> None:
        period_s = 1.0 / self.arm_sample_hz
        next_tick = time.monotonic()
        try:
            self._write_arm_sample(
                streams,
                first_sample,
                session,
                counter_name=counter_name,
            )
            while not session.stop_event.is_set():
                next_tick += period_s
                sample = robot.capture_selected()
                self._write_arm_sample(
                    streams,
                    sample,
                    session,
                    counter_name=counter_name,
                )
                session.stop_event.wait(max(0.0, next_tick - time.monotonic()))
        except Exception as exc:
            self._set_error(session, exc)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise RecordingUnavailable(
                        f"{path.name}:{line_number} is not an object"
                    )
                rows.append(payload)
        return rows

    def _interpolate_arm_sample(
        self,
        rows: list[dict[str, Any]],
        timestamps: list[float],
        target_timestamp: float,
    ) -> dict[str, Any] | None:
        right = bisect.bisect_left(timestamps, target_timestamp)
        if right <= 0 or right >= len(rows):
            return None
        left = right - 1
        before = rows[left]
        after = rows[right]
        t0 = timestamps[left]
        t1 = timestamps[right]
        gap_ms = (t1 - t0) * 1000.0
        nearest_delta_ms = min(target_timestamp - t0, t1 - target_timestamp) * 1000.0
        if (
            gap_ms <= 0.0
            or gap_ms > self.act_max_arm_sample_gap_ms
            or nearest_delta_ms > self.act_max_camera_arm_delta_ms
        ):
            return None
        alpha = (target_timestamp - t0) / (t1 - t0)
        before_joints = np.asarray(before.get("joint_positions"), dtype=np.float64)
        after_joints = np.asarray(after.get("joint_positions"), dtype=np.float64)
        if (
            before_joints.shape != (6,)
            or after_joints.shape != (6,)
            or not np.all(np.isfinite(before_joints))
            or not np.all(np.isfinite(after_joints))
        ):
            return None
        before_gripper = float(before.get("gripper", 0.0))
        after_gripper = float(after.get("gripper", 0.0))
        if not np.isfinite(before_gripper) or not np.isfinite(after_gripper):
            return None
        return {
            "joint_positions": (
                before_joints + alpha * (after_joints - before_joints)
            ).tolist(),
            "gripper": before_gripper + alpha * (after_gripper - before_gripper),
            "source_indices": [left, right],
            "source_timestamps": [t0, t1],
            "interpolation_alpha": alpha,
            "nearest_delta_ms": nearest_delta_ms,
            "source_gap_ms": gap_ms,
        }

    def _build_aligned_samples(self, session: RecordingSession) -> list[str]:
        """Create the only training-authoritative, equal-length ACT timeline."""

        camera_rows = {
            name: self._read_jsonl(
                session.temporary_dir / "sensors" / f"cam_{name}_frames.jsonl"
            )
            for name in session.camera_names
        }
        frame_counts = {name: len(rows) for name, rows in camera_rows.items()}
        if len(set(frame_counts.values())) != 1 or not frame_counts:
            return ["camera metadata counts differ during alignment"]
        stream_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
            "observation": {},
            "action": {},
        }
        stream_timestamps: dict[str, dict[str, list[float]]] = {
            "observation": {},
            "action": {},
        }
        quality_errors: list[str] = []
        for source, directory in (
            ("observation", "observations"),
            ("action", "actions"),
        ):
            for arm in session.arms:
                rows = self._read_jsonl(
                    session.temporary_dir / directory / f"{arm}_arm.jsonl"
                )
                timestamps = [float(row["timestamp"]) for row in rows]
                stream_rows[source][arm] = rows
                stream_timestamps[source][arm] = timestamps
                if len(timestamps) < 2 or any(
                    not np.isfinite(value) for value in timestamps
                ):
                    quality_errors.append(f"{source}/{arm} has invalid timestamps")
                    continue
                gaps_ms = [
                    (right - left) * 1000.0
                    for left, right in zip(timestamps, timestamps[1:])
                ]
                max_gap_ms = max(gaps_ms, default=0.0)
                session.max_arm_sample_gap_ms = max(
                    session.max_arm_sample_gap_ms,
                    max_gap_ms,
                )
                if max_gap_ms > self.act_max_arm_sample_gap_ms:
                    quality_errors.append(
                        f"{source}/{arm} sample gap {max_gap_ms:.3f} ms exceeds "
                        f"{self.act_max_arm_sample_gap_ms:.3f} ms"
                    )
                if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
                    quality_errors.append(f"{source}/{arm} timestamps are not monotonic")

        if quality_errors:
            return quality_errors
        aligned_dir = session.temporary_dir / "aligned"
        aligned_dir.mkdir()
        output = aligned_dir / "samples.jsonl"
        aligned_count = 0
        rejected_count = 0
        trimmed_edge_count = 0
        common_start = max(
            stream_timestamps[source][arm][0]
            for source in ("observation", "action")
            for arm in session.arms
        )
        common_end = min(
            stream_timestamps[source][arm][-1]
            for source in ("observation", "action")
            for arm in session.arms
        )
        with output.open("w", encoding="utf-8") as stream:
            for frame_index in range(next(iter(frame_counts.values()))):
                rows_at_index = {
                    name: camera_rows[name][frame_index]
                    for name in session.camera_names
                }
                bundle_ids = {
                    int(row.get("sync_bundle_id", -1))
                    for row in rows_at_index.values()
                }
                captured_at = [
                    float(row["captured_at"]) for row in rows_at_index.values()
                ]
                if len(bundle_ids) != 1 or any(
                    not np.isfinite(value) for value in captured_at
                ):
                    rejected_count += 1
                    continue
                target_timestamp = float(np.median(captured_at))
                if target_timestamp <= common_start or target_timestamp >= common_end:
                    trimmed_edge_count += 1
                    continue
                aligned: dict[str, dict[str, Any]] = {
                    "observation": {},
                    "action": {},
                }
                valid = True
                for source in ("observation", "action"):
                    for arm in session.arms:
                        sample = self._interpolate_arm_sample(
                            stream_rows[source][arm],
                            stream_timestamps[source][arm],
                            target_timestamp,
                        )
                        if sample is None:
                            valid = False
                            break
                        aligned[source][arm] = sample
                    if not valid:
                        break
                if not valid:
                    rejected_count += 1
                    continue
                stream.write(
                    _json_line(
                        {
                            "index": aligned_count,
                            "camera_frame_index": frame_index,
                            "sync_bundle_id": next(iter(bundle_ids)),
                            "timestamp": target_timestamp,
                            "camera_timestamps": {
                                name: float(rows_at_index[name]["captured_at"])
                                for name in session.camera_names
                            },
                            "observation": aligned["observation"],
                            "action": aligned["action"],
                        }
                    )
                )
                aligned_count += 1
        session.aligned_sample_count = aligned_count
        session.alignment_rejected_count = rejected_count
        session.alignment_trimmed_edge_count = trimmed_edge_count
        total_frames = next(iter(frame_counts.values()))
        eligible_frames = max(0, total_frames - trimmed_edge_count)
        aligned_fraction = aligned_count / eligible_frames if eligible_frames else 0.0
        if aligned_count < 2:
            quality_errors.append("fewer than two aligned training samples")
        if aligned_fraction < self.act_min_aligned_fraction:
            quality_errors.append(
                f"aligned fraction {aligned_fraction:.4f} is below "
                f"{self.act_min_aligned_fraction:.4f}"
            )
        return quality_errors

    def _write_frame(
        self,
        session: RecordingSession,
        camera_name: str,
        frame: CameraFrame,
        writer: Any,
        tsf_stream: Any,
        metadata_stream: Any,
        *,
        extra_metadata: dict[str, Any] | None = None,
    ) -> bool:
        if (
            frame.frame_number is not None
            and frame.frame_number == session.last_camera_frame_numbers[camera_name]
        ):
            with self._lock:
                session.camera_duplicate_frame_counts[camera_name] += 1
                if camera_name == "front":
                    session.duplicate_frame_count = (
                        session.camera_duplicate_frame_counts[camera_name]
                    )
            return False
        writer.write(frame.bgr)
        tsf_stream.write(
            TSF_VALUE.pack(int(round(float(frame.captured_at) * 1000.0)))
        )
        tsf_stream.flush()
        metadata_payload = {
            "index": session.camera_frame_counts[camera_name],
            "camera_name": camera_name,
            "captured_at": frame.captured_at,
            "frame_number": frame.frame_number,
            "device_timestamp_ms": frame.device_timestamp_ms,
            "has_depth": frame.depth_z16 is not None,
            "depth_scale_m": frame.depth_scale_m,
        }
        if extra_metadata:
            metadata_payload.update(extra_metadata)
        metadata_stream.write(_json_line(metadata_payload))
        metadata_stream.flush()
        with self._lock:
            session.camera_frame_counts[camera_name] += 1
            session.last_camera_frame_numbers[camera_name] = frame.frame_number
            if camera_name == "front":
                session.frame_count = session.camera_frame_counts[camera_name]
                session.last_frame_number = frame.frame_number
        if camera_name == "front":
            self._publish_preview(session, frame)
        return True

    def _publish_preview(
        self,
        session: RecordingSession,
        frame: CameraFrame,
    ) -> None:
        """Publish a frame that has already been committed to the episode."""
        with self._lock:
            if self._session is session and session.state in ACTIVE_STATES:
                session.preview_frame = frame
                session.preview_captured_at = float(frame.captured_at)

    def _camera_loop(
        self,
        session: RecordingSession,
        camera_name: str,
        camera: Any,
        writer: Any,
        tsf_stream: Any,
        metadata_stream: Any,
        *,
        first_frame: CameraFrame,
        fps: float,
    ) -> None:
        period_s = 1.0 / fps
        next_tick = time.monotonic()
        try:
            self._write_frame(
                session,
                camera_name,
                first_frame,
                writer,
                tsf_stream,
                metadata_stream,
            )
            while not session.stop_event.is_set():
                next_tick += period_s
                frame = camera.capture()
                if frame.bgr.shape != first_frame.bgr.shape:
                    raise RecordingUnavailable(
                        "camera resolution changed during recording"
                    )
                self._write_frame(
                    session,
                    camera_name,
                    frame,
                    writer,
                    tsf_stream,
                    metadata_stream,
                )
                session.stop_event.wait(max(0.0, next_tick - time.monotonic()))
        except Exception as exc:
            self._set_error(session, exc)

    def _bundle_camera_loop(
        self,
        session: RecordingSession,
        bundle: Any,
        writers: dict[str, Any],
        tsf_streams: dict[str, Any],
        metadata_streams: dict[str, Any],
        *,
        first_frames: dict[str, CameraFrame],
    ) -> None:
        expected_shapes = {
            name: frame.bgr.shape for name, frame in first_frames.items()
        }
        encoder_queues: dict[
            str, queue.Queue[tuple[CameraFrame, int, float, dict[str, Any]]]
        ] = {
            name: queue.Queue(maxsize=self.act_encoder_queue_size)
            for name in session.camera_names
        }
        producer_done = threading.Event()
        encoder_threads: list[threading.Thread] = []

        def encode_camera(camera_name: str) -> None:
            work_queue = encoder_queues[camera_name]
            try:
                while not producer_done.is_set() or not work_queue.empty():
                    try:
                        frame, bundle_id, sync_skew_ms, timing = work_queue.get(
                            timeout=0.05
                        )
                    except queue.Empty:
                        continue
                    try:
                        self._write_frame(
                            session,
                            camera_name,
                            frame,
                            writers[camera_name],
                            tsf_streams[camera_name],
                            metadata_streams[camera_name],
                            extra_metadata={
                                "sync_bundle_id": bundle_id,
                                "sync_skew_ms": sync_skew_ms,
                                **timing,
                            },
                        )
                    finally:
                        work_queue.task_done()
            except Exception as exc:
                self._set_error(session, exc)

        def enqueue_bundle(frames: dict[str, CameraFrame]) -> None:
            metadata = bundle.last_metadata or {}
            bundle_id = int(metadata.get("bundle_id", -1))
            sync_skew_ms = float(metadata.get("sync_skew_ms", float("inf")))
            camera_metadata = metadata.get("cameras")
            if set(frames) != set(session.camera_names):
                raise RecordingUnavailable("synchronized bundle camera set changed")
            if (
                bundle_id < 0
                or not np.isfinite(sync_skew_ms)
                or sync_skew_ms > self.act_max_sync_skew_ms
                or not isinstance(camera_metadata, dict)
            ):
                with self._lock:
                    session.sync_rejected_bundle_count += 1
                return
            for name in session.camera_names:
                details = camera_metadata.get(name)
                if not isinstance(details, dict):
                    raise RecordingUnavailable(
                        f"synchronized {name} timing metadata is missing"
                    )
                frame = frames[name]
                if frame.bgr.shape != expected_shapes[name]:
                    raise RecordingUnavailable(
                        f"synchronized {name} resolution changed during recording"
                    )
            with self._lock:
                if session.last_sync_bundle_id is not None:
                    session.sync_missed_bundle_count += max(
                        0, bundle_id - session.last_sync_bundle_id - 1
                    )
                session.last_sync_bundle_id = bundle_id
            if any(work_queue.full() for work_queue in encoder_queues.values()):
                with self._lock:
                    session.encoder_backpressure_drop_count += 1
                return
            for name in session.camera_names:
                details = camera_metadata[name]
                encoder_queues[name].put_nowait(
                    (
                        frames[name],
                        bundle_id,
                        sync_skew_ms,
                        {
                            "arrival_captured_at": float(details["captured_at"]),
                            "sync_timestamp_ms": float(details["sync_timestamp_ms"]),
                            "timestamp_domain": str(details["timestamp_domain"]),
                            "timestamp_source": "device_global_time",
                        },
                    )
                )
            with self._lock:
                session.sync_skew_ms_samples.append(sync_skew_ms)

        try:
            for name in session.camera_names:
                thread = threading.Thread(
                    target=encode_camera,
                    args=(name,),
                    name=f"{session.recording_id}-encoder-{name}",
                    daemon=True,
                )
                encoder_threads.append(thread)
                thread.start()
            enqueue_bundle(first_frames)
            while not session.stop_event.is_set():
                enqueue_bundle(bundle.capture())
        except Exception as exc:
            if not session.stop_event.is_set():
                self._set_error(session, exc)
        finally:
            producer_done.set()
            for thread in encoder_threads:
                thread.join(timeout=5.0)
            if any(thread.is_alive() for thread in encoder_threads):
                self._set_error(
                    session,
                    RecordingUnavailable("synchronized video encoder did not drain"),
                )

    def _meta_payload(
        self,
        session: RecordingSession,
        *,
        status: str,
        fps_by_camera: dict[str, float],
    ) -> dict[str, Any]:
        finished_at = session.finished_at or time.time()
        started_at = session.started_at or session.requested_at
        is_act = _is_act_purpose(session.purpose)
        files: dict[str, Any] = {
            "rgb": "sensors/cam_front_rgb.mp4",
            "rgb_timestamps": "sensors/cam_front_rgb.mp4.tsf",
            "frame_metadata": "sensors/cam_front_frames.jsonl",
            "actions": {
                name: f"actions/{name}_arm.jsonl" for name in session.arms
            },
            "images": {
                name: {
                    "video": f"sensors/cam_{name}_rgb.mp4",
                    "timestamps": f"sensors/cam_{name}_rgb.mp4.tsf",
                    "frame_metadata": f"sensors/cam_{name}_frames.jsonl",
                }
                for name in session.camera_names
            },
        }
        if is_act:
            files["observations"] = {
                name: f"observations/{name}_arm.jsonl" for name in session.arms
            }
            files["aligned_samples"] = "aligned/samples.jsonl"
        payload: dict[str, Any] = {
            "version": (
                "medicine_act_episode_v1"
                if is_act
                else "medicine_calibration_episode_v1"
            ),
            "recording_id": session.recording_id,
            "label": session.label,
            "purpose": session.purpose,
            "selected_arms": list(session.arms),
            "status": status,
            "created_at": session.requested_at,
            "started_at": session.started_at,
            "finished_at": finished_at,
            "duration_s": max(0.0, finished_at - started_at),
            "frame_count": session.frame_count,
            "duplicate_frame_count": session.duplicate_frame_count,
            "camera_names": list(session.camera_names),
            "camera_frame_counts": dict(session.camera_frame_counts),
            "camera_duplicate_frame_counts": dict(
                session.camera_duplicate_frame_counts
            ),
            "arm_sample_counts": dict(session.arm_sample_counts),
            "observation_sample_counts": dict(session.observation_sample_counts),
            "action_sample_counts": dict(session.action_sample_counts),
            "error": session.error,
            "safety": {
                "hardware_access": "feedback_only",
                "motion_commands": False,
                "mode_switching": False,
                "suction_commands": False,
                "chassis_commands": False,
                "playback": False,
            },
            "camera": session.camera_profile,
            "cameras": dict(session.camera_profiles),
            "recorder": {
                "nominal_camera_fps": fps_by_camera.get("front"),
                "nominal_camera_fps_by_name": dict(fps_by_camera),
                "nominal_arm_sample_hz": self.arm_sample_hz,
                "max_duration_s": self.max_duration_s,
                "rgb_codec": "mp4v",
                "video_encoder_mode": (
                    "per_camera_async" if self._act_bundle is not None else "inline"
                ),
                "depth_frames_saved": False,
            },
            "files": files,
        }
        if is_act:
            sync_samples = np.asarray(session.sync_skew_ms_samples, dtype=np.float64)
            payload["act"] = {
                "observation_source": "follower_joint_feedback",
                "action_source": (
                    "mixed_by_arm"
                    if session.action_from_observation_arms
                    else "leader_joint_feedback"
                ),
                "action_source_by_arm": {
                    name: (
                        "held_follower_joint_feedback"
                        if name in session.action_from_observation_arms
                        else "leader_joint_feedback"
                    )
                    for name in session.arms
                },
                "follower_endpoint": {
                    "host": self.arm_host,
                    "ports": {name: self.arm_ports[name] for name in session.arms},
                },
                "leader_endpoint": {
                    "host": session.leader_host or self.leader_host,
                    "ports": {
                        name: (session.leader_ports or self.leader_ports)[name]
                        for name in session.arms
                    },
                },
                "alignment": (
                    "source_synchronized_bundle"
                    if self._act_bundle is not None
                    else "offline_nearest_timestamp"
                ),
                "ready_marker": "READY",
                "checksum_manifest": "checksums.sha256",
                "camera_names": list(session.camera_names),
                "synchronization": {
                    "bundle_source": (
                        str(self._act_bundle.directory)
                        if self._act_bundle is not None
                        else None
                    ),
                    "bundle_count": int(sync_samples.size),
                    "max_allowed_skew_ms": self.act_max_sync_skew_ms,
                    "rejected_bundle_count": session.sync_rejected_bundle_count,
                    "missed_bundle_count": session.sync_missed_bundle_count,
                    "encoder_backpressure_drop_count": (
                        session.encoder_backpressure_drop_count
                    ),
                    "require_contiguous_bundles": (
                        self.act_require_contiguous_bundles
                    ),
                    "mean_skew_ms": (
                        float(sync_samples.mean()) if sync_samples.size else None
                    ),
                    "p95_skew_ms": (
                        float(np.percentile(sync_samples, 95))
                        if sync_samples.size
                        else None
                    ),
                    "max_skew_ms": (
                        float(sync_samples.max()) if sync_samples.size else None
                    ),
                },
                "training_alignment": {
                    "authority": "aligned/samples.jsonl",
                    "timeline": "median_synchronized_camera_exposure_timestamp",
                    "timestamp_basis": "device_global_time",
                    "method": "bounded_linear_interpolation",
                    "aligned_sample_count": session.aligned_sample_count,
                    "rejected_camera_frame_count": session.alignment_rejected_count,
                    "trimmed_edge_frame_count": session.alignment_trimmed_edge_count,
                    "aligned_fraction": (
                        session.aligned_sample_count
                        / max(
                            1,
                            session.frame_count - session.alignment_trimmed_edge_count,
                        )
                        if session.frame_count - session.alignment_trimmed_edge_count > 0
                        else 0.0
                    ),
                    "max_allowed_arm_pair_skew_ms": self.act_max_arm_pair_skew_ms,
                    "max_allowed_source_capture_skew_ms": (
                        self.act_max_source_capture_skew_ms
                    ),
                    "max_allowed_arm_sample_gap_ms": self.act_max_arm_sample_gap_ms,
                    "max_allowed_camera_arm_delta_ms": (
                        self.act_max_camera_arm_delta_ms
                    ),
                    "minimum_aligned_fraction": self.act_min_aligned_fraction,
                    "observed_max_arm_pair_skew_ms": (
                        max(session.arm_pair_skew_ms_samples)
                        if session.arm_pair_skew_ms_samples
                        else None
                    ),
                    "observed_max_source_capture_skew_ms": (
                        max(session.source_capture_skew_ms_samples)
                        if session.source_capture_skew_ms_samples
                        else None
                    ),
                    "observed_max_arm_sample_gap_ms": session.max_arm_sample_gap_ms,
                },
            }
        return payload

    def _run_official_act_session(self, session: RecordingSession) -> None:
        """Record ACT data exclusively through the official open_pdd backend."""

        backend = OfficialEpisodeRecorder(
            output_root=self.act_output_dir,
            runtime_root=Path("/dev/shm/ruc-video"),
            label=session.label,
            purpose=session.purpose,
            recording_id=session.recording_id,
            action_hz=self.arm_sample_hz,
            max_duration_s=self.max_duration_s,
        )

        def on_started(started_at: float) -> None:
            with self._lock:
                session.started_at = started_at
                session.state = "recording"

        def on_preview(
            bgr: np.ndarray,
            depth: np.ndarray | None,
            captured_at: float,
        ) -> None:
            frame = CameraFrame(
                bgr=np.asarray(bgr).copy(),
                depth_z16=(None if depth is None else np.asarray(depth).copy()),
                depth_scale_m=0.001,
                captured_at=captured_at,
            )
            with self._lock:
                session.preview_frame = frame
                session.preview_captured_at = captured_at

        successful = False
        try:
            saved_path = backend.run(
                session.stop_event,
                on_started=on_started,
                on_preview=on_preview,
            )
            session.saved_path = str(saved_path)
            metadata = json.loads(
                (saved_path / "meta.json").read_text(encoding="utf-8")
            )
            session.started_at = float(metadata.get("created_at", session.started_at))
            session.finished_at = float(metadata.get("finished_at", time.time()))
            session.action_sample_counts = {}
            for arm in ("left", "right"):
                path = saved_path / "actions" / f"{arm}_arm.jsonl"
                with path.open("r", encoding="utf-8") as stream:
                    count = sum(1 for _ in stream)
                session.action_sample_counts[arm] = count
                session.arm_sample_counts[arm] = count
            for source, logical in (
                ("front", "front"),
                ("left", "left_wrist"),
                ("right", "right_wrist"),
            ):
                tsf = saved_path / "sensors" / f"cam_{source}_rgb.mp4.tsf"
                count = max(0, (tsf.stat().st_size - TSF_HEADER.size) // 8)
                session.camera_frame_counts[logical] = count
            session.frame_count = session.camera_frame_counts.get("front", 0)
            successful = (
                all(value > 0 for value in session.action_sample_counts.values())
                and all(value > 0 for value in session.camera_frame_counts.values())
            )
            if not successful:
                raise RecordingUnavailable("official episode contains an empty stream")
        except Exception as exc:
            session.error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                session.finished_at = session.finished_at or time.time()
                session.preview_frame = None
                session.state = "saved" if successful else "error"
                self._last_session = session
                self._session = None

    def _run_session(self, session: RecordingSession) -> None:
        writers: dict[str, Any] = {}
        tsf_streams: dict[str, Any] = {}
        metadata_streams: dict[str, Any] = {}
        action_streams: dict[str, Any] = {}
        observation_streams: dict[str, Any] = {}
        robots: list[Any] = []
        arm_threads: list[threading.Thread] = []
        camera_threads: list[threading.Thread] = []
        is_act = _is_act_purpose(session.purpose)
        if is_act:
            self._run_official_act_session(session)
            return
        bundle = self._act_bundle if is_act else None
        cameras = {} if bundle is not None else self._session_cameras(session)
        fps_by_camera = (
            {name: float(bundle.fps) for name in session.camera_names}
            if bundle is not None
            else {
                name: self._source_fps(camera)
                for name, camera in cameras.items()
            }
        )
        try:
            session.temporary_dir.mkdir(parents=True, exist_ok=False)
            actions_dir = session.temporary_dir / "actions"
            sensors_dir = session.temporary_dir / "sensors"
            actions_dir.mkdir()
            sensors_dir.mkdir()
            observations_dir = session.temporary_dir / "observations"
            if is_act:
                observations_dir.mkdir()

            follower = self._arm_factory(
                host=self.arm_host,
                ports=self.arm_ports,
                arm_names=session.arms,
            )
            follower.connect()
            robots.append(follower)
            first_follower_sample = follower.capture_selected()
            leader = None
            first_leader_sample = None
            if is_act:
                leader_host, leader_ports = self._resolve_leader_endpoint()
                session.leader_host = leader_host
                session.leader_ports = leader_ports
                leader = self._arm_factory(
                    host=leader_host,
                    ports=leader_ports,
                    arm_names=session.arms,
                )
                leader.connect()
                robots.append(leader)
                first_leader_sample = leader.capture_selected()
            if bundle is not None:
                bundle.begin_session()
            first_frames: dict[str, CameraFrame] = (
                bundle.capture() if bundle is not None else {}
            )
            if bundle is None:
                for camera_name, camera in cameras.items():
                    first_frames[camera_name] = camera.capture()
            for camera_name in session.camera_names:
                first_frame = first_frames[camera_name]
                if first_frame.bgr.ndim != 3 or first_frame.bgr.shape[2] != 3:
                    raise RecordingUnavailable(
                        f"{camera_name} camera did not return a BGR image"
                    )
                first_frames[camera_name] = first_frame
                height, width = first_frame.bgr.shape[:2]
                rgb_path = sensors_dir / f"cam_{camera_name}_rgb.mp4"
                writer = cv2.VideoWriter(
                    str(rgb_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps_by_camera[camera_name],
                    (width, height),
                )
                if not writer.isOpened():
                    raise RecordingUnavailable(
                        f"OpenCV could not open the {camera_name} MP4 writer"
                    )
                writers[camera_name] = writer
                tsf_stream = (
                    sensors_dir / f"cam_{camera_name}_rgb.mp4.tsf"
                ).open("w+b")
                _write_tsf_header(tsf_stream, video_size=0)
                tsf_streams[camera_name] = tsf_stream
                metadata_streams[camera_name] = (
                    sensors_dir / f"cam_{camera_name}_frames.jsonl"
                ).open("w", encoding="utf-8")
            for name in session.arms:
                action_streams[name] = (actions_dir / f"{name}_arm.jsonl").open(
                    "w", encoding="utf-8"
                )
                if is_act:
                    observation_streams[name] = (
                        observations_dir / f"{name}_arm.jsonl"
                    ).open("w", encoding="utf-8")

            with self._lock:
                session.camera_profiles = (
                    bundle.profiles()
                    if bundle is not None
                    else {
                        name: camera.profile() for name, camera in cameras.items()
                    }
                )
                session.camera_profile = session.camera_profiles["front"]
                session.started_at = time.time()
                session.state = "recording"

            if is_act:
                arm_threads.append(
                    threading.Thread(
                        target=self._act_arm_loop,
                        args=(
                            session,
                            follower,
                            leader,
                            observation_streams,
                            action_streams,
                        ),
                        name=f"{session.recording_id}-synchronized-arm-feedback",
                        daemon=True,
                    )
                )
            else:
                arm_threads.append(
                    threading.Thread(
                        target=self._arm_loop,
                        args=(session, follower, action_streams),
                        kwargs={"first_sample": first_follower_sample},
                        name=f"{session.recording_id}-arms",
                        daemon=True,
                    )
                )
            if bundle is not None:
                camera_threads.append(
                    threading.Thread(
                        target=self._bundle_camera_loop,
                        args=(
                            session,
                            bundle,
                            writers,
                            tsf_streams,
                            metadata_streams,
                        ),
                        kwargs={"first_frames": first_frames},
                        name=f"{session.recording_id}-synchronized-cameras",
                        daemon=True,
                    )
                )
            else:
                for camera_name, camera in cameras.items():
                    camera_threads.append(
                        threading.Thread(
                            target=self._camera_loop,
                            args=(
                                session,
                                camera_name,
                                camera,
                                writers[camera_name],
                                tsf_streams[camera_name],
                                metadata_streams[camera_name],
                            ),
                            kwargs={
                                "first_frame": first_frames[camera_name],
                                "fps": fps_by_camera[camera_name],
                            },
                            name=f"{session.recording_id}-camera-{camera_name}",
                            daemon=True,
                        )
                    )
            for arm_thread in arm_threads:
                arm_thread.start()
            for camera_thread in camera_threads:
                camera_thread.start()

            while not session.stop_event.wait(0.1):
                if (
                    session.started_at is not None
                    and time.time() - session.started_at >= self.max_duration_s
                ):
                    session.stop_event.set()
                    break

            for arm_thread in arm_threads:
                arm_thread.join(timeout=5.0)
            for camera_thread in camera_threads:
                camera_thread.join(timeout=5.0)
            if any(thread.is_alive() for thread in arm_threads):
                raise RecordingUnavailable("arm recorder did not stop cleanly")
            if any(thread.is_alive() for thread in camera_threads):
                raise RecordingUnavailable("camera recorder did not stop cleanly")
        except Exception as exc:
            self._set_error(session, exc)
        finally:
            session.stop_event.set()
            for arm_thread in arm_threads:
                if arm_thread.is_alive():
                    arm_thread.join(timeout=2.0)
            for camera_thread in camera_threads:
                if camera_thread.is_alive():
                    camera_thread.join(timeout=2.0)
            for stream in (*action_streams.values(), *observation_streams.values()):
                try:
                    stream.close()
                except Exception:
                    pass
            for metadata_stream in metadata_streams.values():
                try:
                    metadata_stream.close()
                except Exception:
                    pass
            for writer in writers.values():
                try:
                    writer.release()
                except Exception:
                    pass
            for camera_name, tsf_stream in tsf_streams.items():
                try:
                    rgb_path = (
                        session.temporary_dir
                        / "sensors"
                        / f"cam_{camera_name}_rgb.mp4"
                    )
                    _write_tsf_header(
                        tsf_stream,
                        video_size=rgb_path.stat().st_size if rgb_path.exists() else 0,
                    )
                    tsf_stream.flush()
                    tsf_stream.close()
                except Exception as exc:
                    if session.error is None:
                        session.error = f"{type(exc).__name__}: {exc}"
            for robot in robots:
                try:
                    robot.close()
                except Exception:
                    pass

            if is_act and bundle is not None and session.error is None:
                preliminary_quality_errors: list[str] = []
                preliminary_counts = [
                    session.camera_frame_counts.get(name, 0)
                    for name in session.camera_names
                ]
                if len(set(preliminary_counts)) != 1:
                    preliminary_quality_errors.append(
                        "synchronized camera frame counts differ"
                    )
                if session.sync_rejected_bundle_count:
                    preliminary_quality_errors.append(
                        f"recorder rejected {session.sync_rejected_bundle_count} over-skew bundles"
                    )
                if session.encoder_backpressure_drop_count:
                    preliminary_quality_errors.append(
                        f"encoder dropped {session.encoder_backpressure_drop_count} bundles"
                    )
                if (
                    self.act_require_contiguous_bundles
                    and session.sync_missed_bundle_count
                ):
                    preliminary_quality_errors.append(
                        f"recorder missed {session.sync_missed_bundle_count} published bundles"
                    )
                if preliminary_quality_errors:
                    session.error = (
                        "strict ACT quality gate failed: "
                        + "; ".join(preliminary_quality_errors)
                    )

            if is_act and session.error is None:
                try:
                    alignment_errors = self._build_aligned_samples(session)
                except Exception as exc:
                    alignment_errors = [
                        f"alignment build failed: {type(exc).__name__}: {exc}"
                    ]
                if alignment_errors:
                    session.error = (
                        "strict ACT alignment gate failed: "
                        + "; ".join(alignment_errors)
                    )

            with self._lock:
                session.finished_at = time.time()
                if is_act and bundle is not None and session.error is None:
                    camera_counts = [
                        session.camera_frame_counts.get(name, 0)
                        for name in session.camera_names
                    ]
                    quality_errors: list[str] = []
                    if len(set(camera_counts)) != 1:
                        quality_errors.append(
                            "synchronized camera frame counts differ"
                        )
                    if session.sync_rejected_bundle_count:
                        quality_errors.append(
                            "recorder rejected "
                            f"{session.sync_rejected_bundle_count} over-skew bundles"
                        )
                    if session.encoder_backpressure_drop_count:
                        quality_errors.append(
                            "encoder dropped "
                            f"{session.encoder_backpressure_drop_count} bundles"
                        )
                    if (
                        self.act_require_contiguous_bundles
                        and session.sync_missed_bundle_count
                    ):
                        quality_errors.append(
                            "recorder missed "
                            f"{session.sync_missed_bundle_count} published bundles"
                        )
                    if (
                        session.sync_skew_ms_samples
                        and max(session.sync_skew_ms_samples)
                        > self.act_max_sync_skew_ms
                    ):
                        quality_errors.append("synchronized bundle skew exceeded limit")
                    if (
                        session.arm_pair_skew_ms_samples
                        and max(session.arm_pair_skew_ms_samples)
                        > self.act_max_arm_pair_skew_ms
                    ):
                        quality_errors.append(
                            "left/right arm feedback skew exceeded limit"
                        )
                    if (
                        session.source_capture_skew_ms_samples
                        and max(session.source_capture_skew_ms_samples)
                        > self.act_max_source_capture_skew_ms
                    ):
                        quality_errors.append(
                            "observation/action capture skew exceeded limit"
                        )
                    if quality_errors:
                        session.error = "strict ACT quality gate failed: " + "; ".join(
                            quality_errors
                        )
                successful = (
                    session.error is None
                    and all(
                        session.camera_frame_counts.get(name, 0) > 0
                        for name in session.camera_names
                    )
                    and all(
                        session.arm_sample_counts.get(name, 0) > 0
                        for name in session.arms
                    )
                )
                if not successful and session.error is None:
                    session.error = "recording has no complete camera/arm samples"

            failed_destination = (
                self.act_failed_dir / session.final_dir.name
                if is_act
                else session.final_dir.with_name(session.final_dir.name + ".failed")
            )
            destination = session.final_dir if successful else failed_destination
            try:
                if session.temporary_dir.exists():
                    _atomic_json(
                        session.temporary_dir / "meta.json",
                        self._meta_payload(
                            session,
                            status="completed" if successful else "failed",
                            fps_by_camera=fps_by_camera,
                        ),
                    )
                    if successful and is_act:
                        _write_checksums(session.temporary_dir)
                        _atomic_json(
                            session.temporary_dir / "READY",
                            {
                                "version": "medicine_act_ready_v1",
                                "recording_id": session.recording_id,
                            },
                        )
                    session.temporary_dir.replace(destination)
                    session.saved_path = str(destination)
            except Exception as exc:
                successful = False
                if session.error is None:
                    session.error = (
                        f"finalization failed: {type(exc).__name__}: {exc}"
                    )
                try:
                    if session.temporary_dir.exists():
                        _atomic_json(
                            session.temporary_dir / "meta.json",
                            self._meta_payload(
                                session,
                                status="failed",
                                fps_by_camera=fps_by_camera,
                            ),
                        )
                        session.temporary_dir.replace(failed_destination)
                        session.saved_path = str(failed_destination)
                except Exception as recovery_exc:
                    session.error += (
                        "; recovery failed: "
                        f"{type(recovery_exc).__name__}: {recovery_exc}"
                    )
                    if session.temporary_dir.exists():
                        session.saved_path = str(session.temporary_dir)
            finally:
                with self._lock:
                    session.preview_frame = None
                    session.state = "saved" if successful else "error"
                    self._last_session = session
                    self._session = None
