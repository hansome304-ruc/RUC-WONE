"""RealSense RGB-D capture + calibration loader.

Note: only one process can hold a RealSense pipeline at a time. Don't run
this while ``client_collection.py`` / ``main.py`` is recording.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyrealsense2 as rs


@dataclass
class Frame:
    rgb: np.ndarray         # (H, W, 3) uint8, RGB order
    depth_mm: np.ndarray    # (H, W) uint16, millimetres
    K: np.ndarray           # (3, 3) intrinsics (from RealSense, NOT from calib file)
    depth_scale_m: float    # metres per depth unit; usually ~0.001
    timestamp: float


log = logging.getLogger(__name__)


class RealSenseRGBD:
    """Thin wrapper around a single RealSense pipeline with depth→color align.

    Lifecycle model: pipeline is **lazily started** on the first `capture()`
    and kept open across calls. If a `wait_for_frames` ever times out
    (typical symptom when nobody consumed frames for several seconds while
    the arm was doing something else), we transparently `stop()` →
    `hardware_reset()` → `start()` and retry once. This avoids the
    "have-to-restart-the-script-every-time" pain.

    Why not start eagerly in `__init__`?
      - If a later step in GraspSkills crashes (calib missing, gRPC busy,
        etc.) the pipeline never gets `stop()`'d and the USB device stays
        held → next run gets `Device or resource busy`.
    Why not stop after every capture?
      - RealSense USB firmware needs ~seconds of cool-down between rapid
        stop/start cycles. Tested empirically: stop→sleep 3s→start fails
        with `Frame didn't arrive within 5000`. Always-on is more reliable.
    """

    _WAIT_TIMEOUT_MS = 5000

    def __init__(self, serial: str, width: int = 640, height: int = 480, fps: int = 30):
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self._pipe: rs.pipeline | None = None
        self._align: rs.align | None = None
        self._lock = threading.RLock()
        self.K_realsense: np.ndarray | None = None
        self.depth_scale: float | None = None
        # Fail fast if the camera isn't there — but DON'T touch its USB
        # streaming endpoint yet.
        ctx = rs.context()
        serials = [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]
        if serial not in serials:
            raise RuntimeError(
                f"RealSense serial {serial!r} not found. "
                f"Connected devices: {serials}"
            )

    def _start(self) -> None:
        if self._pipe is not None:
            return
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        prof = pipe.start(cfg)
        self._pipe = pipe
        self._align = rs.align(rs.stream.color)
        if self.K_realsense is None:
            cprof = prof.get_stream(rs.stream.color).as_video_stream_profile()
            intr = cprof.get_intrinsics()
            self.K_realsense = np.array(
                [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            self.depth_scale = float(prof.get_device().first_depth_sensor().get_depth_scale())

    def _stop(self) -> None:
        if self._pipe is None:
            return
        try:
            self._pipe.stop()
        except Exception:
            pass
        self._pipe = None
        self._align = None

    def _hard_reset(self) -> None:
        """Hardware-reset the device. Used when the pipeline has stalled."""
        try:
            for d in rs.context().query_devices():
                if d.get_info(rs.camera_info.serial_number) == self.serial:
                    d.hardware_reset()
                    break
        except Exception as exc:
            log.warning("hardware_reset on %s raised: %s", self.serial, exc)
        # Give USB a few seconds to re-enumerate.
        time.sleep(4)

    def _capture_once(self, warmup_frames: int) -> Frame:
        # Drain anything stale in the queue first so we return a fresh frame
        # rather than one that's seconds old (RealSense buffers up a few).
        for _ in range(max(0, warmup_frames)):
            self._pipe.wait_for_frames(self._WAIT_TIMEOUT_MS)
        frames = self._align.process(self._pipe.wait_for_frames(self._WAIT_TIMEOUT_MS))
        cf = frames.get_color_frame()
        df = frames.get_depth_frame()
        if not cf or not df:
            raise RuntimeError("RealSense returned empty frame")
        bgr = np.asanyarray(cf.get_data())
        rgb = bgr[..., ::-1].copy()
        depth_u16 = np.asanyarray(df.get_data())
        depth_mm = (depth_u16.astype(np.float32) * self.depth_scale * 1000.0).astype(np.uint16)
        return Frame(
            rgb=rgb, depth_mm=depth_mm, K=self.K_realsense.copy(),
            depth_scale_m=self.depth_scale, timestamp=time.time(),
        )

    def capture(self, warmup_frames: int = 5) -> Frame:
        with self._lock:
            self._start()
            try:
                return self._capture_once(warmup_frames)
            except RuntimeError as exc:
                msg = str(exc)
                if "Frame didn't arrive" not in msg and "empty frame" not in msg:
                    raise
                log.warning(
                    "RealSense %s stalled (%s); doing stop→hw_reset→start and retrying once.",
                    self.serial, msg,
                )
                self._stop()
                self._hard_reset()
                self._start()
                return self._capture_once(warmup_frames)

    def close(self):
        with self._lock:
            self._stop()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------- calibration ----------

@dataclass(frozen=True)
class CalibBundle:
    K_calib: np.ndarray           # (3, 3) intrinsics from calib (preferred over RealSense one)
    distortion: np.ndarray        # (k,)
    T_cam_to_left: np.ndarray     # (4, 4) cam frame → left-arm base frame
    T_cam_to_right: np.ndarray    # (4, 4) cam frame → right-arm base frame
    tcp_left: np.ndarray          # (3,) calibrated TCP offset (often [0,0,0])
    tcp_right: np.ndarray


def load_calib(left_path: Path, right_path: Path, intr_path: Path) -> CalibBundle:
    """Load the JSONs produced by ``calib_for_dos_w1_v3/run_calibration.sh``."""
    if not (left_path.exists() and right_path.exists() and intr_path.exists()):
        raise FileNotFoundError(
            "Calibration files not found. Run "
            "`bash ~/dos_w1/data_collection_client/calib_for_dos_w1_v3/run_calibration.sh "
            "--episode-left ... --episode-right ... --episode-test ... --output-dir "
            "/home/ubuntu/calib` first."
        )
    intr = json.loads(intr_path.read_text())
    L = json.loads(left_path.read_text())
    R = json.loads(right_path.read_text())
    return CalibBundle(
        K_calib=np.array(intr["camera"], dtype=np.float64),
        distortion=np.array(intr["distortion"], dtype=np.float64).ravel(),
        T_cam_to_left=np.array(L["cam_to_base"], dtype=np.float64),
        T_cam_to_right=np.array(R["cam_to_base"], dtype=np.float64),
        tcp_left=np.array(L.get("tcp", [0.0, 0.0, 0.0]), dtype=np.float64),
        tcp_right=np.array(R.get("tcp", [0.0, 0.0, 0.0]), dtype=np.float64),
    )
