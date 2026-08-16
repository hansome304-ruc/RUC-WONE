#!/usr/bin/env python3
"""Render compact three-camera previews from processed ACT HDF5 episodes."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERA_ORDER = ("cam_high", "cam_left_wrist", "cam_right_wrist")
CAMERA_LABELS = ("front", "left wrist", "right wrist")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "recordings" / "act" / "processed",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "act_dedup_previews",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--tile-height", type=int, default=240)
    parser.add_argument("--crf", type=int, default=24)
    return parser


def _writer(output: Path, *, width: int, height: int, fps: float, crf: int) -> subprocess.Popen:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to render preview videos")
    return subprocess.Popen(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )


def _render_episode(
    hdf5_path: Path,
    output_path: Path,
    entry: dict[str, Any],
    *,
    fps: float,
    tile_width: int,
    tile_height: int,
    crf: int,
) -> dict[str, Any]:
    dedup = entry.get("deduplication", {})
    kept = int(entry.get("sample_count", 0))
    source = int(entry.get("source_sample_count", kept))
    removed = source - kept
    output_width = tile_width * len(CAMERA_ORDER)
    output_height = tile_height + 44
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.stem}.inprogress{output_path.suffix}"
    )
    if temporary.exists():
        temporary.unlink()
    process = _writer(
        temporary,
        width=output_width,
        height=output_height,
        fps=fps,
        crf=crf,
    )
    try:
        if process.stdin is None:
            raise RuntimeError("failed to open ffmpeg input pipe")
        with h5py.File(hdf5_path, "r") as root:
            image_group = root["observations/images"]
            source_indices = root["source_aligned_index"][:]
            if any(image_group[name].shape[0] != kept for name in CAMERA_ORDER):
                raise RuntimeError(f"camera length mismatch in {hdf5_path}")
            for index in range(kept):
                tiles: list[np.ndarray] = []
                for camera_name, camera_label in zip(CAMERA_ORDER, CAMERA_LABELS):
                    rgb = image_group[camera_name][index]
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    tile = cv2.resize(bgr, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
                    cv2.rectangle(tile, (0, 0), (tile_width, 28), (0, 0, 0), -1)
                    cv2.putText(
                        tile,
                        camera_label,
                        (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    tiles.append(tile)
                canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                canvas[:tile_height] = np.concatenate(tiles, axis=1)
                summary = (
                    f"{entry.get('source_label', hdf5_path.stem)}  "
                    f"kept {kept}/{source}  removed {removed} ({removed / source:.1%})  "
                    f"source frame {int(source_indices[index])}"
                )
                cv2.putText(
                    canvas,
                    summary,
                    (10, tile_height + 29),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                process.stdin.write(canvas.tobytes())
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")
        temporary.replace(output_path)
    except Exception:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "label": entry.get("source_label"),
        "output": str(output_path),
        "source_sample_count": source,
        "sample_count": kept,
        "removed_sample_count": removed,
        "retention_ratio": dedup.get("retention_ratio", kept / source),
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.fps <= 0 or args.tile_width < 64 or args.tile_height < 64:
        raise ValueError("fps must be positive and preview tiles must be at least 64 pixels")
    if not 0 <= args.crf <= 51:
        raise ValueError("crf must be between 0 and 51")
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    results: list[dict[str, Any]] = []
    for manifest_path in sorted(input_root.glob("*/dataset_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        task = str(manifest["task"])
        for entry in manifest["episodes"]:
            label = str(entry.get("source_label") or Path(entry["file"]).stem)
            safe_label = "".join(character if character.isalnum() else "_" for character in label)
            results.append(
                _render_episode(
                    manifest_path.parent / entry["file"],
                    output_root / f"{task}_{safe_label}_dedup_preview.mp4",
                    entry,
                    fps=args.fps,
                    tile_width=args.tile_width,
                    tile_height=args.tile_height,
                    crf=args.crf,
                )
            )
    if not results:
        raise RuntimeError(f"no processed ACT manifests found under {input_root}")
    print(json.dumps({"preview_count": len(results), "previews": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
