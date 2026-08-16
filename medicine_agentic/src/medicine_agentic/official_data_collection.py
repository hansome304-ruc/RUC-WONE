"""Thin 8899 adapter around the stock open_pdd data collector."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from open_pdd import DOSW1Controller, MultiprocessRecorder
from open_pdd.base import CameraType
from open_pdd.collect import build_root_dir, prepare_meta
from open_pdd.sensor.realsense import RealSense


class OfficialEpisodeRecorder:
    """Run one stock open_pdd MultiprocessRecorder episode."""

    CAMERA_SOURCES = {
        "cam_front": ("front", "420222072569"),
        "cam_left": ("left", "347622072392"),
        "cam_right": ("right", "347622071407"),
    }
    VIDEO_UNITS = (
        "ruc-video-sync.service",
        "ruc-video-camera@front.service",
        "ruc-video-camera@left.service",
        "ruc-video-camera@right.service",
    )

    @staticmethod
    def _systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("systemctl", "--user", *args),
            check=check,
            capture_output=True,
            text=True,
            timeout=20,
        )

    def _pause_video_gateway(self) -> tuple[str, ...]:
        active = tuple(
            unit
            for unit in self.VIDEO_UNITS
            if self._systemctl("is-active", "--quiet", unit, check=False).returncode
            == 0
        )
        # Stop the synchronizer before its source camera owners.
        for unit in self.VIDEO_UNITS:
            if unit in active:
                self._systemctl("stop", unit)
        time.sleep(1.0)
        return active

    def _resume_video_gateway(self, active: tuple[str, ...]) -> None:
        # Restore camera owners before the synchronizer that consumes them.
        for unit in reversed(self.VIDEO_UNITS):
            if unit in active:
                self._systemctl("start", unit)

    def __init__(
        self,
        *,
        output_root: Path,
        runtime_root: Path,
        label: str,
        purpose: str,
        recording_id: str,
        action_hz: float,
        max_duration_s: float,
    ) -> None:
        self.output_root = Path(output_root)
        self.runtime_root = Path(runtime_root)
        self.label = label
        self.purpose = purpose
        self.recording_id = recording_id
        self.action_hz = max(1.0, min(float(action_hz), 250.0))
        self.max_duration_s = max_duration_s
        self.recorder: MultiprocessRecorder | None = None

    def run(
        self,
        stop_event: Any,
        *,
        on_started: Callable[[float], None],
        on_preview: Callable[[np.ndarray, np.ndarray | None, float], None],
    ) -> Path:
        active_video_units = self._pause_video_gateway()
        recorder: MultiprocessRecorder | None = None
        try:
            root = build_root_dir(str(self.output_root))
            sensors = {
                key: RealSense(
                    serial_number=serial,
                    width=640,
                    height=480,
                    depth=True,
                    fps=30,
                )
                for key, (_source, serial) in self.CAMERA_SOURCES.items()
            }
            controller = DOSW1Controller(fps=self.action_hz)
            meta = prepare_meta(
                root,
                sensor_map=sensors,
                controller=controller,
                serial="001",
                task_name=self.purpose,
                prompt=self.label,
                extra={
                    "recording_id": self.recording_id,
                    "label": self.label,
                    "purpose": self.purpose,
                    "recording_strategy": "official_open_pdd_direct_rgbd_v1",
                    "nominal_action_hz": self.action_hz,
                    "camera_owner": "official_open_pdd",
                },
            )
            recorder = MultiprocessRecorder(
                meta=meta,
                root=str(root),
                sensor_map=sensors,
                controller=controller,
                sensor_frame_timeout=2,
                child_ready_timeout_seconds=30,
                process_join_timeout_seconds=10,
                child_poll_interval_seconds=0.05,
            )
            self.recorder = recorder
            recorder.init(warmup_seconds=0.2)
            started_at = time.time()
            recorder.start(created_at=started_at)
            on_started(started_at)
            next_preview = 0.0
            while not stop_event.wait(0.05):
                recorder.raise_if_failed()
                now = time.time()
                if now - started_at >= self.max_duration_s:
                    break
                if now >= next_preview:
                    next_preview = now + 0.2
                    rgb = recorder.get_last_frame_cache("cam_front", CameraType.RGB)
                    depth = recorder.get_last_frame_cache(
                        "cam_front", CameraType.DEPTH
                    )
                    if rgb is not None:
                        on_preview(rgb, depth, now)
            recorder.commit(finished_at=time.time())
            return Path(recorder.meta.episode_dir)
        finally:
            if recorder is not None:
                recorder.close()
            self.recorder = None
            self._resume_video_gateway(active_video_units)
