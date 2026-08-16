#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


def log(camera: str, message: str) -> None:
    print(f"[ruc-video:{camera}] {message}", flush=True)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


class SharedFrameWriter:
    def __init__(self, root: Path, camera: dict[str, Any]):
        self.camera = camera
        self.directory = root / camera["name"]
        self.directory.mkdir(parents=True, exist_ok=True)
        self.color_map: np.memmap | None = None
        self.depth_map: np.memmap | None = None
        self.color_shape: tuple[int, ...] | None = None
        self.depth_shape: tuple[int, ...] | None = None
        self.frame_id = 0

    def _ensure_maps(self, color: np.ndarray, depth: np.ndarray | None) -> None:
        if self.color_shape != color.shape:
            self.color_map = np.memmap(
                self.directory / "color.bgr",
                dtype=np.uint8,
                mode="w+",
                shape=color.shape,
            )
            self.color_shape = color.shape
        if depth is not None and self.depth_shape != depth.shape:
            self.depth_map = np.memmap(
                self.directory / "depth.z16",
                dtype=np.uint16,
                mode="w+",
                shape=depth.shape,
            )
            self.depth_shape = depth.shape

    def publish(
        self,
        color: np.ndarray,
        depth: np.ndarray | None,
        depth_scale_m: float | None,
        device_timestamp_ms: float | None,
        *,
        source_frame_number: int | None = None,
        timestamp_domain: str | None = None,
    ) -> None:
        self._ensure_maps(color, depth)
        self.frame_id += 1
        atomic_json(
            self.directory / "meta.json",
            {"state": "writing", "frame_id": self.frame_id},
        )
        assert self.color_map is not None
        self.color_map[:] = color
        if depth is not None:
            assert self.depth_map is not None
            self.depth_map[:] = depth
        captured_at = time.time()
        atomic_json(
            self.directory / "meta.json",
            {
                "state": "ready",
                "camera": self.camera["name"],
                "label": self.camera.get("label", self.camera["name"]),
                "source": self.camera.get("source", ""),
                "serial": self.camera.get("serial", ""),
                "frame_id": self.frame_id,
                "captured_at": captured_at,
                "captured_monotonic_ns": time.monotonic_ns(),
                "device_timestamp_ms": device_timestamp_ms,
                "source_frame_number": source_frame_number,
                "timestamp_domain": timestamp_domain,
                "color_width": int(color.shape[1]),
                "color_height": int(color.shape[0]),
                "depth_width": int(depth.shape[1]) if depth is not None else 0,
                "depth_height": int(depth.shape[0]) if depth is not None else 0,
                "depth_scale_m": depth_scale_m,
                "capture_fps": int(self.camera["capture_fps"]),
                "publish_fps": int(self.camera["publish_fps"]),
            },
        )

    def status(self, state: str, message: str, **extra: Any) -> None:
        atomic_json(
            self.directory / "status.json",
            {
                "camera": self.camera["name"],
                "state": state,
                "message": message,
                "updated_at": time.time(),
                "pid": os.getpid(),
                **extra,
            },
        )


class LatestFrameEncoder:
    def __init__(
        self,
        camera: dict[str, Any],
        encoder_config: dict[str, Any],
        rtsp_base: str,
        runtime_root: Path,
    ):
        self.camera = camera
        self.encoder_config = encoder_config
        self.rtsp_url = f"{rtsp_base.rstrip('/')}/{camera['name']}"
        self.runtime_root = runtime_root
        self.frames: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.process: subprocess.Popen[bytes] | None = None
        self.process_shape: tuple[int, int] | None = None
        self.hardware_failed = False
        self.current_bitrate = int(camera["bitrate_kbps"])
        self.last_control_check = 0.0
        self.thread = threading.Thread(target=self._run, name="encoder", daemon=True)
        self.thread.start()

    def submit(self, frame: np.ndarray) -> None:
        try:
            self.frames.put_nowait(frame.copy())
        except queue.Full:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frames.put_nowait(frame.copy())
            except queue.Full:
                pass

    def _desired_bitrate(self) -> int:
        if not bool(self.encoder_config.get("adaptive_bitrate", True)):
            return int(self.camera["bitrate_kbps"])
        now = time.monotonic()
        if now - self.last_control_check < 1.0:
            return self.current_bitrate
        self.last_control_check = now
        try:
            control = json.loads((self.runtime_root / "control.json").read_text(encoding="utf-8"))
            main = str(control.get("main_camera", "front"))
        except Exception:
            main = "front"
        return int(
            self.camera["main_bitrate_kbps"]
            if main == self.camera["name"]
            else self.camera["aux_bitrate_kbps"]
        )

    def _command(self, width: int, height: int, bitrate: int) -> list[str]:
        ffmpeg = str(self.encoder_config.get("ffmpeg", "/usr/bin/ffmpeg"))
        fps = int(self.camera["publish_fps"])
        common = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
        ]
        if not self.hardware_failed and self.encoder_config.get("preferred") == "vaapi":
            common += ["-vaapi_device", str(self.encoder_config["vaapi_device"])]
        common += [
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
        ]
        if not self.hardware_failed and self.encoder_config.get("preferred") == "vaapi":
            common += [
                "-vf",
                "format=nv12,hwupload",
                "-c:v",
                "h264_vaapi",
                "-profile:v",
                "main",
                "-level:v",
                "3.1",
            ]
        else:
            common += [
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-profile:v",
                "baseline",
                "-x264-params",
                f"keyint={fps}:min-keyint={fps}:scenecut=0:bframes=0:rc-lookahead=0",
            ]
        common += [
            "-g",
            str(fps),
            "-bf",
            "0",
            "-b:v",
            f"{bitrate}k",
            "-maxrate",
            f"{bitrate}k",
            "-bufsize",
            f"{max(bitrate, 1000)}k",
            "-muxdelay",
            "0",
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            self.rtsp_url,
        ]
        return common

    def _stop_process(self) -> None:
        process, self.process = self.process, None
        self.process_shape = None
        if process is None:
            return
        if process.stdin:
            try:
                process.stdin.close()
            except Exception:
                pass
        try:
            process.terminate()
            process.wait(timeout=2)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def _start_process(self, width: int, height: int, bitrate: int) -> None:
        self._stop_process()
        log_path = self.runtime_root / self.camera["name"] / "ffmpeg.log"
        log_file = open(log_path, "ab", buffering=0)
        command = self._command(width, height, bitrate)
        log(self.camera["name"], f"encoder start {' '.join(command)}")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=log_file,
            bufsize=0,
        )
        log_file.close()
        self.process_shape = (width, height)
        self.current_bitrate = bitrate

    def _write(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        desired = self._desired_bitrate()
        if (
            self.process is None
            or self.process.poll() is not None
            or self.process_shape != (width, height)
            or desired != self.current_bitrate
        ):
            self._start_process(width, height, desired)
        assert self.process is not None and self.process.stdin is not None
        try:
            self.process.stdin.write(memoryview(np.ascontiguousarray(frame)))
        except (BrokenPipeError, OSError) as exc:
            if not self.hardware_failed and self.encoder_config.get("preferred") == "vaapi":
                self.hardware_failed = True
                log(self.camera["name"], f"VAAPI failed, falling back to x264: {exc}")
            self._stop_process()
            time.sleep(0.2)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame = self.frames.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame is None:
                break
            self._write(frame)
        self._stop_process()

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.frames.put_nowait(None)
        except queue.Full:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frames.put_nowait(None)
            except queue.Full:
                pass
        self.thread.join(timeout=4)
        self._stop_process()


class CameraRuntime:
    def __init__(self, config: dict[str, Any], camera: dict[str, Any]):
        self.config = config
        self.camera = camera
        self.name = str(camera["name"])
        self.stop_event = threading.Event()
        self.runtime_root = Path(config["runtime_dir"])
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.writer = SharedFrameWriter(self.runtime_root, camera)
        self.encoder = LatestFrameEncoder(
            camera,
            config["encoder"],
            config["rtsp_base"],
            self.runtime_root,
        )
        self.last_encoded_at = 0.0

    def stop(self, *_args: Any) -> None:
        self.stop_event.set()

    def _publish(
        self,
        color: np.ndarray,
        depth: np.ndarray | None,
        depth_scale_m: float | None,
        device_timestamp_ms: float | None,
        *,
        source_frame_number: int | None = None,
        timestamp_domain: str | None = None,
    ) -> None:
        self.writer.publish(
            color,
            depth,
            depth_scale_m,
            device_timestamp_ms,
            source_frame_number=source_frame_number,
            timestamp_domain=timestamp_domain,
        )
        interval = 1.0 / max(1, int(self.camera["publish_fps"]))
        now = time.monotonic()
        if now - self.last_encoded_at >= interval * 0.90:
            self.last_encoded_at = now
            self.encoder.submit(color)

    def _apply_view(self, color: np.ndarray, depth: np.ndarray | None):
        values = self.camera.get("view_homography") or []
        if len(values) != 9:
            return color, depth
        matrix = np.asarray(values, dtype=np.float64).reshape(3, 3)
        size = (color.shape[1], color.shape[0])
        corrected_color = cv2.warpPerspective(color, matrix, size, flags=cv2.INTER_LINEAR)
        corrected_depth = None
        if depth is not None:
            corrected_depth = cv2.warpPerspective(depth, matrix, size, flags=cv2.INTER_NEAREST)
        return corrected_color, corrected_depth

    def _run_realsense(self) -> None:
        if rs is None:
            raise RuntimeError("pyrealsense2 is unavailable")
        pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_device(str(self.camera["serial"]))
        rs_config.enable_stream(
            rs.stream.color,
            int(self.camera["color_width"]),
            int(self.camera["color_height"]),
            rs.format.bgr8,
            int(self.camera["capture_fps"]),
        )
        if self.camera.get("depth", False):
            rs_config.enable_stream(
                rs.stream.depth,
                int(self.camera["depth_width"]),
                int(self.camera["depth_height"]),
                rs.format.z16,
                int(self.camera["depth_fps"]),
            )
        started = False
        try:
            profile = pipeline.start(rs_config)
            started = True
            global_time_enabled = False
            for sensor in profile.get_device().query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1.0)
                    global_time_enabled = True
            align = rs.align(rs.stream.color) if self.camera.get("depth", False) else None
            depth_scale_m = None
            if align is not None:
                depth_scale_m = float(profile.get_device().first_depth_sensor().get_depth_scale())
            self.writer.status(
                "running",
                "RealSense opened",
                encoder="starting",
                global_time_enabled=global_time_enabled,
            )
            warmup = 4
            while not self.stop_event.is_set():
                frames = pipeline.wait_for_frames(1500)
                if align is not None:
                    frames = align.process(frames)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame() if align is not None else None
                if not color_frame:
                    continue
                color = np.asanyarray(color_frame.get_data()).copy()
                depth = np.asanyarray(depth_frame.get_data()).copy() if depth_frame else None
                color, depth = self._apply_view(color, depth)
                if warmup:
                    warmup -= 1
                    continue
                self._publish(
                    color,
                    depth,
                    depth_scale_m,
                    float(color_frame.get_timestamp()),
                    source_frame_number=int(color_frame.get_frame_number()),
                    timestamp_domain=str(color_frame.get_frame_timestamp_domain()),
                )
        finally:
            if started:
                pipeline.stop()

    def _run_v4l2(self) -> None:
        capture = cv2.VideoCapture(str(self.camera["source"]), cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.camera["color_width"]))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.camera["color_height"]))
        capture.set(cv2.CAP_PROP_FPS, int(self.camera["capture_fps"]))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {self.camera['source']}")
        try:
            self.writer.status("running", "V4L2 opened", encoder="starting")
            while not self.stop_event.is_set():
                ok, color = capture.read()
                if not ok or color is None:
                    raise RuntimeError("V4L2 frame read failed")
                color, _ = self._apply_view(color, None)
                self._publish(color, None, None, None)
        finally:
            capture.release()

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        failures = 0
        try:
            while not self.stop_event.is_set():
                started_at = time.monotonic()
                try:
                    if self.camera["source_type"] == "realsense":
                        self._run_realsense()
                    else:
                        self._run_v4l2()
                    failures = 0
                except Exception as exc:
                    failures += 1
                    self.writer.status("error", str(exc), failures=failures)
                    log(self.name, f"capture error: {type(exc).__name__}: {exc}")
                    if time.monotonic() - started_at > 30:
                        failures = 1
                    self.stop_event.wait(min(10.0, 0.5 * (2 ** min(failures, 4))))
            return 0
        finally:
            self.writer.status("stopped", "worker stopped")
            self.encoder.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--camera", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    camera = next((item for item in config["cameras"] if item["name"] == args.camera), None)
    if camera is None:
        raise SystemExit(f"unknown camera: {args.camera}")
    return CameraRuntime(config, camera).run()


if __name__ == "__main__":
    raise SystemExit(main())
