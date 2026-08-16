"""Camera sources for the private medicine-packaging console.

The camera layer is perception-only.  It has no dependency on any existing
web console and deliberately exposes no robot or suction controls.
"""
from __future__ import annotations

import io
import importlib.metadata
import json
import threading
import time
from dataclasses import dataclass
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPCookieProcessor, Request, build_opener

import cv2
import numpy as np


class CameraUnavailable(RuntimeError):
    """Raised when the selected camera cannot provide a color frame."""


STANDARD_WIDTH = 1280
STANDARD_HEIGHT = 720
STANDARD_FPS = 30


@dataclass(frozen=True)
class CameraFrame:
    """One color frame and its synchronized depth, when available."""

    bgr: np.ndarray
    depth_z16: np.ndarray | None
    captured_at: float
    frame_number: int | None = None
    device_timestamp_ms: float | None = None
    depth_scale_m: float | None = None


def _stream_config(camera_cfg: dict[str, Any], key: str) -> dict[str, Any]:
    stream = camera_cfg.get(key, {})
    if not isinstance(stream, dict):
        raise ValueError(f"camera.{key} must be an object")
    result = {
        "width": int(stream.get("width", STANDARD_WIDTH)),
        "height": int(stream.get("height", STANDARD_HEIGHT)),
        "fps": int(stream.get("fps", STANDARD_FPS)),
        "format": str(stream.get("format", "bgr8" if key == "color" else "z16")),
    }
    if result["width"] <= 0 or result["height"] <= 0 or result["fps"] <= 0:
        raise ValueError(f"camera.{key} width, height and fps must be positive")
    return result


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", str(value).split(".")[-1]))


def _intrinsics_payload(value: Any) -> dict[str, Any]:
    return {
        "width": int(value.width),
        "height": int(value.height),
        "fx": float(value.fx),
        "fy": float(value.fy),
        "cx": float(value.ppx),
        "cy": float(value.ppy),
        "distortion_model": _enum_name(value.model),
        "distortion_coefficients": [float(item) for item in value.coeffs],
    }


def _intrinsics_matrix(payload: dict[str, Any]) -> list[list[float]]:
    return [
        [float(payload["fx"]), 0.0, float(payload["cx"])],
        [0.0, float(payload["fy"]), float(payload["cy"])],
        [0.0, 0.0, 1.0],
    ]


class PackagingCamera:
    """Small common interface shared by offline and RealSense sources."""

    def __init__(self, camera_cfg: dict[str, Any]) -> None:
        self.name = str(camera_cfg.get("name", "front"))
        self.mode = str(camera_cfg.get("mode", "offline"))
        self.serial = str(camera_cfg.get("serial", "420222072569"))
        self.color = _stream_config(camera_cfg, "color")
        self.depth = _stream_config(camera_cfg, "depth")
        expected = (STANDARD_WIDTH, STANDARD_HEIGHT, STANDARD_FPS)
        for stream_name, stream in (("color", self.color), ("depth", self.depth)):
            actual = (stream["width"], stream["height"], stream["fps"])
            if actual != expected:
                raise ValueError(
                    f"camera.{stream_name} must use the packaging profile "
                    f"{STANDARD_WIDTH}x{STANDARD_HEIGHT}@{STANDARD_FPS}, got "
                    f"{actual[0]}x{actual[1]}@{actual[2]}"
                )
        if self.color["format"] != "bgr8":
            raise ValueError("camera.color.format must be 'bgr8'")
        if self.depth["format"] != "z16":
            raise ValueError("camera.depth.format must be 'z16'")
        self.aligned_to = "color"
        self.profile_approved = False
        configured_intrinsics = camera_cfg.get("intrinsics")
        if configured_intrinsics is None:
            self.intrinsics: list[list[float]] | None = None
        else:
            matrix = np.asarray(configured_intrinsics, dtype=np.float64)
            if (
                matrix.shape != (3, 3)
                or not np.isfinite(matrix).all()
                or matrix[0, 0] <= 0.0
                or matrix[1, 1] <= 0.0
                or not np.allclose(matrix[2], [0.0, 0.0, 1.0])
            ):
                raise ValueError("camera.intrinsics must be a valid 3x3 matrix")
            self.intrinsics = matrix.tolist()

        configured_resolution = camera_cfg.get("intrinsics_resolution")
        if configured_resolution is None:
            self.intrinsics_resolution: list[int] | None = (
                None
                if self.intrinsics is None
                else [self.color["width"], self.color["height"]]
            )
        else:
            try:
                self.intrinsics_resolution = [
                    int(configured_resolution[0]),
                    int(configured_resolution[1]),
                ]
            except (IndexError, TypeError, ValueError) as exc:
                raise ValueError(
                    "camera.intrinsics_resolution must be [width, height]"
                ) from exc
            if self.intrinsics_resolution != [
                self.color["width"],
                self.color["height"],
            ]:
                raise ValueError(
                    "camera.intrinsics_resolution must match camera.color"
                )

        source_profile = camera_cfg.get("source_color_profile")
        self.source_color_profile = (
            dict(source_profile) if isinstance(source_profile, dict) else None
        )
        source_intrinsics = camera_cfg.get("source_intrinsics")
        if source_intrinsics is None:
            self.source_intrinsics: list[list[float]] | None = None
        else:
            source_matrix = np.asarray(source_intrinsics, dtype=np.float64)
            if source_matrix.shape != (3, 3) or not np.isfinite(
                source_matrix
            ).all():
                raise ValueError(
                    "camera.source_intrinsics must be a finite 3x3 matrix"
                )
            self.source_intrinsics = source_matrix.tolist()

        self.intrinsics_source = (
            str(camera_cfg["intrinsics_source"])
            if camera_cfg.get("intrinsics_source") is not None
            else None
        )
        self.intrinsics_calibration = (
            str(camera_cfg["intrinsics_calibration"])
            if camera_cfg.get("intrinsics_calibration") is not None
            else None
        )
        self.distortion_model = (
            str(camera_cfg["distortion_model"])
            if camera_cfg.get("distortion_model") is not None
            else None
        )
        configured_coefficients = camera_cfg.get("distortion_coefficients")
        if configured_coefficients is None:
            self.distortion_coefficients: list[float] | None = None
        else:
            coefficients = np.asarray(
                configured_coefficients, dtype=np.float64
            ).reshape(-1)
            if coefficients.size != 5 or not np.isfinite(coefficients).all():
                raise ValueError(
                    "camera.distortion_coefficients must contain 5 finite values"
                )
            self.distortion_coefficients = coefficients.tolist()
        self.color_intrinsics = (
            dict(camera_cfg["color_intrinsics"])
            if isinstance(camera_cfg.get("color_intrinsics"), dict)
            else None
        )
        self.depth_intrinsics = (
            dict(camera_cfg["depth_intrinsics"])
            if isinstance(camera_cfg.get("depth_intrinsics"), dict)
            else None
        )
        self.aligned_depth_intrinsics = (
            dict(camera_cfg["aligned_depth_intrinsics"])
            if isinstance(camera_cfg.get("aligned_depth_intrinsics"), dict)
            else None
        )
        self.depth_to_color_extrinsics = (
            dict(camera_cfg["depth_to_color_extrinsics"])
            if isinstance(camera_cfg.get("depth_to_color_extrinsics"), dict)
            else None
        )
        self.device_name = (
            str(camera_cfg["device_name"])
            if camera_cfg.get("device_name") is not None
            else None
        )
        self.firmware_version = (
            str(camera_cfg["firmware_version"])
            if camera_cfg.get("firmware_version") is not None
            else None
        )
        self.librealsense_version = (
            str(camera_cfg["librealsense_version"])
            if camera_cfg.get("librealsense_version") is not None
            else None
        )
        configured_depth_scale = camera_cfg.get("depth_scale_m")
        self.depth_scale_m: float | None = (
            None
            if configured_depth_scale is None
            else float(configured_depth_scale)
        )
        self.last_valid_frame_at: float | None = None
        self.last_frame_number: int | None = None
        self.last_device_timestamp_ms: float | None = None
        self.last_depth_valid_ratio: float | None = None
        self.profile_approved = bool(camera_cfg.get("profile_approved", False))
        self.state = "starting"
        self.error: str | None = None

    def capture(self) -> CameraFrame:
        raise NotImplementedError

    def profile(self) -> dict[str, Any]:
        valid_frame_age_s = (
            None
            if self.last_valid_frame_at is None
            else max(0.0, time.time() - self.last_valid_frame_at)
        )
        return {
            "name": self.name,
            "mode": self.mode,
            "state": self.state,
            "serial": self.serial,
            "color": dict(self.color),
            "depth": dict(self.depth),
            "aligned_to": self.aligned_to,
            "intrinsics": self.intrinsics,
            "intrinsics_resolution": self.intrinsics_resolution,
            "intrinsics_source": self.intrinsics_source,
            "intrinsics_calibration": self.intrinsics_calibration,
            "source_color_profile": self.source_color_profile,
            "source_intrinsics": self.source_intrinsics,
            "distortion_model": self.distortion_model,
            "distortion_coefficients": self.distortion_coefficients,
            "color_intrinsics": self.color_intrinsics,
            "depth_intrinsics": self.depth_intrinsics,
            "aligned_depth_intrinsics": self.aligned_depth_intrinsics,
            "depth_to_color_extrinsics": self.depth_to_color_extrinsics,
            "depth_scale_m": self.depth_scale_m,
            "device_name": self.device_name,
            "firmware_version": self.firmware_version,
            "librealsense_version": self.librealsense_version,
            "last_valid_frame_at": self.last_valid_frame_at,
            "last_valid_frame_age_s": valid_frame_age_s,
            "last_frame_number": self.last_frame_number,
            "last_device_timestamp_ms": self.last_device_timestamp_ms,
            "last_depth_valid_ratio": self.last_depth_valid_ratio,
            "profile_approved": self.profile_approved,
            "error": self.error,
        }

    def live_rgbd_is_fresh(self, *, max_age_s: float = 1.0) -> bool:
        if (
            self.mode not in {"realsense", "shared", "shared_memory"}
            or self.state != "ready"
            or self.last_valid_frame_at is None
            or self.depth_scale_m is None
        ):
            return False
        return 0.0 <= time.time() - self.last_valid_frame_at <= max_age_s

    def close(self) -> None:
        """Release resources. Offline sources have nothing to release."""


class OfflinePackagingCamera(PackagingCamera):
    """Repeatable, read-only source backed by one checked-in JPEG."""

    def __init__(self, camera_cfg: dict[str, Any], image_path: Path) -> None:
        super().__init__(camera_cfg)
        self.mode = "offline"
        self.image_path = image_path
        self._frame: np.ndarray | None = None
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"cannot read offline image: {image_path}")
            size = (self.color["width"], self.color["height"])
            self._frame = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            self.state = "ready"
        except Exception as exc:
            self.state = "error"
            self.error = str(exc)

    def capture(self) -> CameraFrame:
        if self._frame is None:
            raise CameraUnavailable(self.error or "offline camera is unavailable")
        return CameraFrame(
            bgr=self._frame.copy(),
            depth_z16=None,
            captured_at=time.time(),
        )


class SharedWebConsoleCamera(PackagingCamera):
    """Read color frames from the existing web-console camera owner.

    Only the web console opens the RealSense device. This client keeps its own
    browser session and decodes the console's cached JPEG frames, so a second
    process never attempts to claim the USB camera.
    """

    MAX_RGBD_BYTES = 64 * 1024 * 1024
    LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

    def __init__(self, camera_cfg: dict[str, Any]) -> None:
        super().__init__(camera_cfg)
        self.mode = "shared"
        self.shared_base_url = str(
            camera_cfg.get("shared_base_url", "http://127.0.0.1:8888")
        ).strip().rstrip("/")
        self.shared_camera_name = str(
            camera_cfg.get("shared_camera_name", self.name)
        ).strip()
        self.shared_timeout_s = max(
            0.2, min(float(camera_cfg.get("shared_timeout_s", 2.0)), 10.0)
        )
        parsed = urlsplit(self.shared_base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in self.LOOPBACK_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "camera.shared_base_url must be a plain loopback HTTP URL"
            )
        if not self.shared_camera_name:
            raise ValueError("camera.shared_camera_name must not be empty")

        self._lock = threading.Lock()
        self._cookie_jar = CookieJar()
        self._opener = build_opener(HTTPCookieProcessor(self._cookie_jar))
        self._authenticated = False
        self._frame_sequence = 0
        self.state = "starting"
        try:
            with self._lock:
                self._capture_locked()
        except Exception as exc:
            # Keep the packaging HTTP service available while the camera owner
            # starts. Later capture calls retry the shared session automatically.
            self._authenticated = False
            self.state = "error"
            self.error = f"shared web-console camera is not ready: {exc}"

    def _url(self, path: str) -> str:
        return f"{self.shared_base_url}{path}"

    def _authenticate(self) -> None:
        request = Request(
            self._url("/api/auth/session"),
            headers={"Accept": "application/json", "Connection": "close"},
        )
        with self._opener.open(request, timeout=self.shared_timeout_s) as response:
            if response.status != 200:
                raise CameraUnavailable(
                    f"web-console session returned HTTP {response.status}"
                )
            response.read(1024 * 1024)
        self._authenticated = True

    def _fetch_rgbd(
        self,
    ) -> tuple[bytes, np.ndarray, float, float, int]:
        camera_name = quote(self.shared_camera_name, safe="")
        request = Request(
            self._url(f"/api/cameras/{camera_name}/rgbd.npz"),
            headers={"Accept": "application/x-npz", "Connection": "close"},
        )
        with self._opener.open(request, timeout=self.shared_timeout_s) as response:
            if response.status != 200:
                raise CameraUnavailable(
                    f"shared camera returned HTTP {response.status}"
                )
            content_type = response.headers.get_content_type()
            if content_type not in {"application/x-npz", "application/octet-stream"}:
                raise CameraUnavailable(
                    f"shared camera returned unexpected type {content_type!r}"
                )
            payload = response.read(self.MAX_RGBD_BYTES + 1)
        if not payload:
            raise CameraUnavailable("shared camera returned an empty RGB-D packet")
        if len(payload) > self.MAX_RGBD_BYTES:
            raise CameraUnavailable("shared RGB-D packet exceeds the size limit")
        try:
            with np.load(io.BytesIO(payload), allow_pickle=False) as packet:
                color_jpeg = packet["color_jpeg"].astype(np.uint8, copy=False).tobytes()
                depth_z16 = packet["depth_z16"].astype(np.uint16, copy=True)
                depth_scale_m = float(packet["depth_scale_m"].item())
                captured_at = float(packet["captured_at"].item())
                frame_number = int(packet["frame_number"].item())
        except Exception as exc:
            raise CameraUnavailable(f"invalid shared RGB-D packet: {exc}") from exc
        if not color_jpeg:
            raise CameraUnavailable("shared RGB-D packet has no color JPEG")
        if depth_z16.ndim != 2 or depth_z16.size == 0:
            raise CameraUnavailable("shared RGB-D packet has invalid depth data")
        if not 0.0 < depth_scale_m < 1.0:
            raise CameraUnavailable("shared RGB-D packet has invalid depth scale")
        return color_jpeg, depth_z16, depth_scale_m, captured_at, frame_number

    def _capture_locked(self) -> CameraFrame:
        if not self._authenticated:
            self._authenticate()
        try:
            payload, depth, depth_scale_m, captured_at, frame_number = (
                self._fetch_rgbd()
            )
        except HTTPError as exc:
            if exc.code not in (401, 403):
                raise
            self._authenticated = False
            self._authenticate()
            payload, depth, depth_scale_m, captured_at, frame_number = (
                self._fetch_rgbd()
            )

        encoded = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise CameraUnavailable("OpenCV could not decode the shared JPEG")
        target_size = (self.color["width"], self.color["height"])
        if (image.shape[1], image.shape[0]) != target_size:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        if (depth.shape[1], depth.shape[0]) != target_size:
            depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)

        self._frame_sequence = max(self._frame_sequence + 1, frame_number)
        self.last_valid_frame_at = captured_at
        self.last_frame_number = frame_number
        self.last_depth_valid_ratio = float(
            np.count_nonzero(depth) / depth.size
        )
        self.depth_scale_m = depth_scale_m
        self.state = "ready"
        self.error = None
        return CameraFrame(
            bgr=image,
            depth_z16=depth,
            captured_at=captured_at,
            frame_number=frame_number,
            depth_scale_m=depth_scale_m,
        )

    def capture(self) -> CameraFrame:
        with self._lock:
            try:
                return self._capture_locked()
            except CameraUnavailable as exc:
                self.state = "error"
                self.error = str(exc)
                raise
            except Exception as exc:
                self._authenticated = False
                self.state = "error"
                self.error = f"shared web-console capture failed: {exc}"
                raise CameraUnavailable(self.error) from exc

    def profile(self) -> dict[str, Any]:
        result = super().profile()
        result["shared_base_url"] = self.shared_base_url
        result["shared_camera_name"] = self.shared_camera_name
        result["exclusive_device_access"] = False
        return result

    def close(self) -> None:
        self.state = "stopped"


class SharedMemoryPackagingCamera(PackagingCamera):
    """Read the latest synchronized RGB-D frame from the video owner.

    A two-phase metadata marker acts as a sequence lock.  The reader retries
    whenever the camera worker starts publishing the next frame while a copy
    is in progress, so color and aligned depth always share one frame number.
    """

    def __init__(self, camera_cfg: dict[str, Any]) -> None:
        super().__init__(camera_cfg)
        self.mode = "shared_memory"
        self.shared_camera_name = str(
            camera_cfg.get("shared_camera_name", self.name)
        ).strip()
        root = Path(str(camera_cfg.get("shared_memory_root", "/dev/shm/ruc-video")))
        self.shared_directory = root / self.shared_camera_name
        self._lock = threading.Lock()
        self.state = "starting"
        try:
            with self._lock:
                self._capture_locked()
        except Exception as exc:
            self.state = "error"
            self.error = f"shared-memory camera is not ready: {exc}"

    def _metadata(self) -> dict[str, Any]:
        return json.loads(
            (self.shared_directory / "meta.json").read_text(encoding="utf-8")
        )

    def _read_consistent(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        last_error: Exception | None = None
        for _ in range(8):
            try:
                before = self._metadata()
                if before.get("state") != "ready":
                    time.sleep(0.004)
                    continue
                color = np.memmap(
                    self.shared_directory / "color.bgr",
                    dtype=np.uint8,
                    mode="r",
                    shape=(
                        int(before["color_height"]),
                        int(before["color_width"]),
                        3,
                    ),
                ).copy()
                if int(before.get("depth_width", 0)) <= 0:
                    raise CameraUnavailable("shared frame has no aligned depth")
                depth = np.memmap(
                    self.shared_directory / "depth.z16",
                    dtype=np.uint16,
                    mode="r",
                    shape=(
                        int(before["depth_height"]),
                        int(before["depth_width"]),
                    ),
                ).copy()
                after = self._metadata()
                if (
                    after.get("state") == "ready"
                    and after.get("frame_id") == before.get("frame_id")
                ):
                    return color, depth, after
            except Exception as exc:
                last_error = exc
            time.sleep(0.004)
        raise CameraUnavailable(
            f"could not read a consistent shared RGB-D frame: {last_error or 'writer busy'}"
        )

    def _capture_locked(self) -> CameraFrame:
        color, depth, meta = self._read_consistent()
        target_size = (self.color["width"], self.color["height"])
        if (color.shape[1], color.shape[0]) != target_size:
            color = cv2.resize(color, target_size, interpolation=cv2.INTER_AREA)
        if (depth.shape[1], depth.shape[0]) != target_size:
            depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)
        captured_at = float(meta["captured_at"])
        frame_number = int(meta["frame_id"])
        depth_scale_m = float(meta["depth_scale_m"])
        self.last_valid_frame_at = captured_at
        self.last_frame_number = frame_number
        self.last_depth_valid_ratio = float(np.count_nonzero(depth) / depth.size)
        self.depth_scale_m = depth_scale_m
        self.state = "ready"
        self.error = None
        return CameraFrame(
            bgr=color,
            depth_z16=depth,
            captured_at=captured_at,
            frame_number=frame_number,
            device_timestamp_ms=meta.get("device_timestamp_ms"),
            depth_scale_m=depth_scale_m,
        )

    def capture(self) -> CameraFrame:
        with self._lock:
            try:
                return self._capture_locked()
            except CameraUnavailable as exc:
                self.state = "error"
                self.error = str(exc)
                raise
            except Exception as exc:
                self.state = "error"
                self.error = f"shared-memory capture failed: {exc}"
                raise CameraUnavailable(self.error) from exc

    def profile(self) -> dict[str, Any]:
        result = super().profile()
        result["shared_memory_root"] = str(self.shared_directory.parent)
        result["shared_camera_name"] = self.shared_camera_name
        result["exclusive_device_access"] = False
        return result

    def close(self) -> None:
        self.state = "stopped"


class RealSensePackagingCamera(PackagingCamera):
    """Direct D435I RGB-D source with depth aligned to color.

    Startup failures, including a device already owned by another process, are
    retained as camera state instead of preventing the HTTP console from
    starting.
    """

    def __init__(self, camera_cfg: dict[str, Any]) -> None:
        super().__init__(camera_cfg)
        self.mode = "realsense"
        # Keep the projection model supplied by an offline checkerboard
        # calibration.  The active RealSense profile is still queried below
        # and exposed as ``source_intrinsics``, but it must not silently
        # replace the calibrated RGB model used by perception/deprojection.
        configured_intrinsics = self.intrinsics
        configured_intrinsics_resolution = self.intrinsics_resolution
        configured_intrinsics_source = self.intrinsics_source
        configured_intrinsics_calibration = self.intrinsics_calibration
        configured_distortion_model = self.distortion_model
        configured_distortion_coefficients = self.distortion_coefficients
        self._lock = threading.Lock()
        self._pipeline: Any = None
        self._align: Any = None
        pipeline: Any = None
        pipeline_started = False
        try:
            import pyrealsense2 as rs

            requested_color = dict(self.color)
            requested_depth = dict(self.depth)
            pipeline = rs.pipeline()
            rs_config = rs.config()
            rs_config.enable_device(self.serial)
            rs_config.enable_stream(
                rs.stream.color,
                self.color["width"],
                self.color["height"],
                rs.format.bgr8,
                self.color["fps"],
            )
            rs_config.enable_stream(
                rs.stream.depth,
                self.depth["width"],
                self.depth["height"],
                rs.format.z16,
                self.depth["fps"],
            )
            profile = pipeline.start(rs_config)
            pipeline_started = True
            self._pipeline = pipeline
            device = profile.get_device()
            color_profile = (
                profile.get_stream(rs.stream.color).as_video_stream_profile()
            )
            depth_profile = (
                profile.get_stream(rs.stream.depth).as_video_stream_profile()
            )
            color_intrinsics = _intrinsics_payload(
                color_profile.get_intrinsics()
            )
            depth_intrinsics = _intrinsics_payload(
                depth_profile.get_intrinsics()
            )
            depth_to_color = depth_profile.get_extrinsics_to(color_profile)

            active_color = {
                "width": int(color_profile.width()),
                "height": int(color_profile.height()),
                "fps": int(color_profile.fps()),
                "format": _enum_name(color_profile.format()),
            }
            active_depth = {
                "width": int(depth_profile.width()),
                "height": int(depth_profile.height()),
                "fps": int(depth_profile.fps()),
                "format": _enum_name(depth_profile.format()),
            }
            if active_color != requested_color:
                raise CameraUnavailable(
                    "resolved color profile does not match the request: "
                    f"requested={requested_color}, active={active_color}"
                )
            if active_depth != requested_depth:
                raise CameraUnavailable(
                    "resolved depth profile does not match the request: "
                    f"requested={requested_depth}, active={active_depth}"
                )
            self.color = active_color
            self.depth = active_depth
            self.color_intrinsics = color_intrinsics
            self.depth_intrinsics = depth_intrinsics
            self.source_color_profile = {
                "serial": self.serial,
                **self.color,
                "stream_index": int(color_profile.stream_index()),
                "unique_id": int(color_profile.unique_id()),
            }
            self.source_intrinsics = _intrinsics_matrix(color_intrinsics)
            if configured_intrinsics is None:
                self.intrinsics = _intrinsics_matrix(color_intrinsics)
                self.intrinsics_resolution = [
                    color_intrinsics["width"],
                    color_intrinsics["height"],
                ]
                self.intrinsics_source = "realsense_active_profile"
                self.intrinsics_calibration = None
                self.distortion_model = color_intrinsics["distortion_model"]
                self.distortion_coefficients = list(
                    color_intrinsics["distortion_coefficients"]
                )
            else:
                active_resolution = [
                    color_intrinsics["width"],
                    color_intrinsics["height"],
                ]
                if configured_intrinsics_resolution != active_resolution:
                    raise CameraUnavailable(
                        "configured projection intrinsics do not match the "
                        f"active color profile: configured="
                        f"{configured_intrinsics_resolution}, "
                        f"active={active_resolution}"
                    )
                self.intrinsics = configured_intrinsics
                self.intrinsics_resolution = configured_intrinsics_resolution
                self.intrinsics_source = configured_intrinsics_source
                self.intrinsics_calibration = configured_intrinsics_calibration
                self.distortion_model = (
                    configured_distortion_model
                    or color_intrinsics["distortion_model"]
                )
                self.distortion_coefficients = (
                    configured_distortion_coefficients
                    if configured_distortion_coefficients is not None
                    else list(color_intrinsics["distortion_coefficients"])
                )
            self.depth_to_color_extrinsics = {
                "rotation_row_major": [
                    float(value) for value in depth_to_color.rotation
                ],
                "translation_m": [
                    float(value) for value in depth_to_color.translation
                ],
            }
            self.depth_scale_m = float(
                device.first_depth_sensor().get_depth_scale()
            )
            self.device_name = (
                str(device.get_info(rs.camera_info.name))
                if device.supports(rs.camera_info.name)
                else None
            )
            self.firmware_version = (
                str(device.get_info(rs.camera_info.firmware_version))
                if device.supports(rs.camera_info.firmware_version)
                else None
            )
            try:
                self.librealsense_version = importlib.metadata.version(
                    "pyrealsense2"
                )
            except importlib.metadata.PackageNotFoundError:
                self.librealsense_version = str(
                    getattr(rs, "__version__", "unknown")
                )
            self._align = rs.align(rs.stream.color)
            self.state = "ready"
        except Exception as exc:
            self.state = "error"
            self.error = (
                "RealSense startup failed (the device may be busy or unavailable): "
                f"{exc}"
            )
            if pipeline is not None and pipeline_started:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            self._pipeline = None
            self._align = None

    def capture(self) -> CameraFrame:
        if self._pipeline is None or self._align is None:
            raise CameraUnavailable(self.error or "RealSense camera is unavailable")
        with self._lock:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=2000)
                aligned = self._align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    raise CameraUnavailable("aligned RGB-D frame is incomplete")
                color = np.asanyarray(color_frame.get_data()).copy()
                depth = np.asanyarray(depth_frame.get_data()).copy()
                expected_color = (self.color["height"], self.color["width"], 3)
                expected_depth = (self.color["height"], self.color["width"])
                if color.shape != expected_color:
                    raise CameraUnavailable(
                        f"unexpected color shape {color.shape}, expected {expected_color}"
                    )
                if depth.shape != expected_depth:
                    raise CameraUnavailable(
                        f"unexpected depth shape {depth.shape}, expected {expected_depth}"
                    )
                aligned_depth_profile = (
                    depth_frame.profile.as_video_stream_profile()
                )
                self.aligned_depth_intrinsics = _intrinsics_payload(
                    aligned_depth_profile.get_intrinsics()
                )
                captured_at = time.time()
                frame_number = int(color_frame.get_frame_number())
                device_timestamp_ms = float(color_frame.get_timestamp())
                self.last_valid_frame_at = captured_at
                self.last_frame_number = frame_number
                self.last_device_timestamp_ms = device_timestamp_ms
                self.last_depth_valid_ratio = float(
                    np.count_nonzero(depth) / depth.size
                )
                self.state = "ready"
                self.error = None
                return CameraFrame(
                    bgr=color,
                    depth_z16=depth,
                    captured_at=captured_at,
                    frame_number=frame_number,
                    device_timestamp_ms=device_timestamp_ms,
                    depth_scale_m=self.depth_scale_m,
                )
            except CameraUnavailable as exc:
                self.state = "error"
                self.error = str(exc)
                raise
            except Exception as exc:
                self.state = "error"
                self.error = f"RealSense capture failed: {exc}"
                raise CameraUnavailable(self.error) from exc

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None
                self._align = None
                self.state = "stopped"

    def profile(self) -> dict[str, Any]:
        result = super().profile()
        result["exclusive_device_access"] = True
        return result


def create_camera(
    camera_cfg: dict[str, Any],
    *,
    config_dir: Path,
) -> PackagingCamera:
    """Create the configured camera source."""

    mode = str(camera_cfg.get("mode", "offline")).strip().lower()
    if mode == "offline":
        configured = Path(
            str(camera_cfg.get("offline_image", "../artifacts/task1/scene_front.jpg"))
        ).expanduser()
        image_path = configured if configured.is_absolute() else config_dir / configured
        return OfflinePackagingCamera(camera_cfg, image_path.resolve())
    if mode in {"shared", "web_console"}:
        return SharedWebConsoleCamera(camera_cfg)
    if mode in {"shared_memory", "shm"}:
        return SharedMemoryPackagingCamera(camera_cfg)
    if mode == "realsense":
        if bool(camera_cfg.get("shared_when_available", False)):
            if camera_cfg.get("shared_memory_root"):
                return SharedMemoryPackagingCamera(camera_cfg)
            return SharedWebConsoleCamera(camera_cfg)
        return RealSensePackagingCamera(camera_cfg)
    raise ValueError(f"unsupported camera mode: {mode!r}")
