#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


class FrameStore:
    def __init__(self, root: Path, camera: dict[str, Any]):
        self.root = root
        self.camera = camera
        self.directory = root / camera["name"]
        self._meta_lock = threading.Lock()
        self._last_ready_meta: dict[str, Any] | None = None

    def _metadata(self) -> dict[str, Any]:
        return json.loads((self.directory / "meta.json").read_text(encoding="utf-8"))

    def _remember_ready(self, meta: dict[str, Any]) -> dict[str, Any]:
        if meta.get("state") == "ready":
            with self._meta_lock:
                self._last_ready_meta = dict(meta)
        return meta

    def _status_metadata(self) -> dict[str, Any]:
        """Never expose the writer's sub-frame `writing` window as a disconnect."""
        last_error: Exception | None = None
        for _ in range(3):
            try:
                meta = self._metadata()
                if meta.get("state") == "ready":
                    return self._remember_ready(meta)
            except Exception as exc:
                last_error = exc
            with self._meta_lock:
                cached = dict(self._last_ready_meta) if self._last_ready_meta else None
            if cached is not None:
                return cached
            time.sleep(0.002)
        raise RuntimeError(last_error or "writer is publishing its next frame")

    def read(self, *, color: bool, depth: bool) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]:
        last_error: Exception | None = None
        for _ in range(5):
            try:
                before = self._metadata()
                if before.get("state") != "ready":
                    time.sleep(0.004)
                    continue
                color_frame = None
                depth_frame = None
                if color:
                    color_frame = np.memmap(
                        self.directory / "color.bgr",
                        dtype=np.uint8,
                        mode="r",
                        shape=(int(before["color_height"]), int(before["color_width"]), 3),
                    ).copy()
                if depth and int(before.get("depth_width", 0)) > 0:
                    depth_frame = np.memmap(
                        self.directory / "depth.z16",
                        dtype=np.uint16,
                        mode="r",
                        shape=(int(before["depth_height"]), int(before["depth_width"])),
                    ).copy()
                after = self._metadata()
                if (
                    after.get("state") == "ready"
                    and after.get("frame_id") == before.get("frame_id")
                ):
                    return self._remember_ready(after), color_frame, depth_frame
            except Exception as exc:
                last_error = exc
            time.sleep(0.004)
        raise RuntimeError(f"no consistent frame: {last_error or 'writer busy'}")

    def status(self) -> dict[str, Any]:
        try:
            meta = self._status_metadata()
        except Exception as exc:
            return {
                "running": False,
                "live": False,
                "message": f"waiting for worker: {exc}",
                "frame_age_s": None,
            }
        age = max(0.0, time.time() - float(meta.get("captured_at", 0.0)))
        try:
            worker = json.loads((self.directory / "status.json").read_text(encoding="utf-8"))
        except Exception:
            worker = {}
        return {
            "running": worker.get("state") == "running",
            "live": meta.get("state") == "ready" and age < 2.0,
            "message": worker.get("message", meta.get("state", "unknown")),
            "frame_age_s": round(age, 3),
            "last_frame_at": meta.get("captured_at"),
            "frame_number": meta.get("frame_id"),
            "depth_available": int(meta.get("depth_width", 0)) > 0,
            "transport": "webrtc",
        }


class VideoServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], config: dict[str, Any]):
        super().__init__(address, VideoHandler)
        self.config = config
        self.root = Path(config["runtime_dir"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.cameras = {item["name"]: item for item in config["cameras"]}
        self.stores = {name: FrameStore(self.root, item) for name, item in self.cameras.items()}
        control = self.root / "control.json"
        if not control.exists():
            atomic_json(control, {"main_camera": "front", "updated_at": time.time()})


class VideoHandler(BaseHTTPRequestHandler):
    server_version = "RUCVideoAPI/2.0"
    protocol_version = "HTTP/1.1"

    @property
    def app(self) -> VideoServer:
        return self.server  # type: ignore[return-value]

    def log_message(self, fmt: str, *args: object) -> None:
        rendered = fmt % args
        if "/depth?" not in rendered:
            print(f"[ruc-video-api] {self.address_string()} - {rendered}", flush=True)

    def _json(self, status: int, payload: dict[str, Any], *, head: bool = False) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _camera(self, name: str) -> tuple[dict[str, Any], FrameStore] | None:
        camera = self.app.cameras.get(name)
        store = self.app.stores.get(name)
        if camera is None or store is None:
            self._json(404, {"ok": False, "error": f"unknown camera: {name}"})
            return None
        return camera, store

    def do_HEAD(self) -> None:
        self._dispatch(head=True)

    def do_GET(self) -> None:
        self._dispatch(head=False)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/video/main":
            self._json(404, {"ok": False, "error": "not found"})
            return
        try:
            length = min(int(self.headers.get("Content-Length", "0") or 0), 4096)
            payload = json.loads(self.rfile.read(length) or b"{}")
            name = str(payload.get("name", ""))
            if name not in self.app.cameras:
                raise ValueError("invalid camera")
            atomic_json(
                self.app.root / "control.json",
                {"main_camera": name, "updated_at": time.time()},
            )
            self._json(200, {"ok": True, "main_camera": name})
        except Exception as exc:
            self._json(400, {"ok": False, "error": str(exc)})

    def _dispatch(self, *, head: bool) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in {"/healthz", "/api/health"}:
            statuses = {name: store.status() for name, store in self.app.stores.items()}
            self._json(
                200,
                {
                    "ok": True,
                    "service": "ruc-video-api",
                    "cameras": statuses,
                    "live_count": sum(1 for value in statuses.values() if value["live"]),
                },
                head=head,
            )
            return
        if parsed.path == "/api/cameras":
            self._json(200, {"ok": True, "cameras": self._camera_list()}, head=head)
            return
        parts = parsed.path.strip("/").split("/")
        if len(parts) != 4 or parts[:2] != ["api", "cameras"]:
            self._json(404, {"ok": False, "error": "not found"}, head=head)
            return
        name, action = parts[2], parts[3]
        found = self._camera(name)
        if found is None:
            return
        camera, store = found
        try:
            if action == "depth":
                query = urllib.parse.parse_qs(parsed.query)
                x = float(query.get("x", ["0.5"])[0])
                y = float(query.get("y", ["0.5"])[0])
                self._json(200, {"ok": True, "depth": self._depth(store, x, y)}, head=head)
                return
            if action == "frame.jpg":
                self._frame_jpeg(store, head=head)
                return
            if action == "rgbd.npz":
                self._rgbd(store, head=head)
                return
            if action == "status":
                self._json(200, {"ok": True, "camera": camera, "runtime": store.status()}, head=head)
                return
        except Exception as exc:
            self._json(503, {"ok": False, "error": str(exc)}, head=head)
            return
        self._json(404, {"ok": False, "error": "not found"}, head=head)

    def _camera_list(self) -> list[dict[str, Any]]:
        result = []
        for name, camera in self.app.cameras.items():
            runtime = self.app.stores[name].status()
            result.append(
                {
                    "name": name,
                    "label": camera.get("label", name.title()),
                    "source": camera.get("source", ""),
                    "serial": camera.get("serial", ""),
                    "width": camera["color_width"],
                    "height": camera["color_height"],
                    "fps": camera["publish_fps"],
                    "enabled": True,
                    "transport": "webrtc",
                    "webrtc_path": f"/{name}",
                    "frame_url": f"/api/cameras/{name}/frame.jpg",
                    "stream_url": f"/api/cameras/{name}/stream.mjpg",
                    "depth_available": bool(camera.get("depth", False)),
                    "runtime": runtime,
                }
            )
        return result

    def _frame_jpeg(self, store: FrameStore, *, head: bool) -> None:
        meta, color, _ = store.read(color=True, depth=False)
        assert color is not None
        ok, encoded = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        body = encoded.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Camera-Version", str(meta["frame_id"]))
        self.send_header("X-Camera-Frame-Time", str(meta["captured_at"]))
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _rgbd(self, store: FrameStore, *, head: bool) -> None:
        meta, color, depth = store.read(color=True, depth=True)
        if color is None or depth is None:
            raise RuntimeError("RGB-D is unavailable")
        ok, encoded = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        output = io.BytesIO()
        np.savez_compressed(
            output,
            color_jpeg=encoded,
            depth_z16=depth.astype(np.uint16, copy=False),
            depth_scale_m=np.asarray(float(meta["depth_scale_m"]), dtype=np.float64),
            captured_at=np.asarray(float(meta["captured_at"]), dtype=np.float64),
            frame_number=np.asarray(int(meta["frame_id"]), dtype=np.int64),
        )
        body = output.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "application/x-npz")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _depth(self, store: FrameStore, x_norm: float, y_norm: float) -> dict[str, Any]:
        meta, _, depth = store.read(color=False, depth=True)
        if depth is None:
            return {"available": False, "reason": "camera has no depth"}
        x_norm = min(1.0, max(0.0, x_norm))
        y_norm = min(1.0, max(0.0, y_norm))
        height, width = depth.shape
        scale_mm = float(meta["depth_scale_m"]) * 1000.0

        def region(bounds: tuple[float, float, float, float], percentile: float):
            x0, y0, x1, y1 = bounds
            ix0 = min(width - 1, max(0, int(x0 * width)))
            ix1 = min(width, max(ix0 + 1, int(np.ceil(x1 * width))))
            iy0 = min(height - 1, max(0, int(y0 * height)))
            iy1 = min(height, max(iy0 + 1, int(np.ceil(y1 * height))))
            sample = depth[iy0:iy1, ix0:ix1]
            valid = sample[sample > 0].astype(np.float32) * scale_mm
            valid = valid[(valid >= 50.0) & (valid <= 5000.0)]
            if valid.size == 0:
                return None, 0.0
            return int(round(float(np.percentile(valid, percentile)))), round(float(valid.size / sample.size), 3)

        rx, ry = 0.012, 0.012
        target, ratio = region((x_norm - rx, y_norm - ry, x_norm + rx, y_norm + ry), 50.0)
        zones = {}
        for zone, bounds in {
            "left": (0.04, 0.18, 0.30, 0.90),
            "right": (0.70, 0.18, 0.96, 0.90),
            "top": (0.25, 0.04, 0.75, 0.30),
        }.items():
            distance, valid_ratio = region(bounds, 10.0)
            zones[zone] = {"distance_mm": distance, "valid_ratio": valid_ratio}
        age = max(0.0, time.time() - float(meta["captured_at"]))
        return {
            "available": target is not None,
            "reason": "" if target is not None else "no valid depth near target",
            "target_mm": target,
            "target_valid_ratio": ratio,
            "x": round(x_norm, 4),
            "y": round(y_norm, 4),
            "width": width,
            "height": height,
            "frame_number": int(meta["frame_id"]),
            "captured_at": float(meta["captured_at"]),
            "age_s": round(age, 3),
            "stale": age > 1.0,
            "zones": zones,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    server = VideoServer((args.host, args.port), config)
    print(f"[ruc-video-api] listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
