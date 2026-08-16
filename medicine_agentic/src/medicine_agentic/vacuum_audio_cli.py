"""CLI for read-only suction-acoustics collection and analysis."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medicine_agentic.vacuum_audio import (
    EMPTY_LABEL,
    SEALED_LABEL,
    VALID_LABELS,
    VacuumAudioModel,
    analyze_dataset,
    analyze_wav,
    evaluate_model,
    train_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "artifacts" / "vacuum_audio" / "samples"
DEFAULT_MODEL = PROJECT_ROOT / "configs" / "vacuum_audio_model.json"
DEFAULT_DOSW1_DEVICE = "plughw:U2K,0"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _record_wav(
    output: Path,
    *,
    seconds: int,
    sample_rate: int,
    channels: int,
    device: str,
) -> None:
    recorder = shutil.which("arecord")
    if recorder is None:
        raise RuntimeError(
            "arecord was not found; install ALSA utilities or record PCM WAV files externally"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        recorder,
        "-q",
        "-D",
        device,
        "-t",
        "wav",
        "-f",
        "S16_LE",
        "-r",
        str(sample_rate),
        "-c",
        str(channels),
        "-d",
        str(seconds),
        str(output),
    ]
    # Deliberately no shell and no pump/robot command.
    subprocess.run(command, check=True)


def _record_name(label: str, index: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"{label}_{stamp}_{index:03d}.wav"


def _condition_text(label: str) -> str:
    if label == EMPTY_LABEL:
        return "气泵保持开启，吸盘悬空且没有形成密封"
    return "气泵保持开启，吸盘已稳定吸住代表性药盒"


def _record_command(args: argparse.Namespace) -> int:
    print("安全声明：本命令只调用 arecord 录音，不控制气泵、不连接机械臂、不发送运动。")
    for index in range(1, args.count + 1):
        if not args.no_prompt:
            input(
                f"[{index}/{args.count}] 请确认：{_condition_text(args.label)}；"
                "状态稳定至少1秒后按 Enter 开始录音。"
            )
        output = args.out_dir / args.label / _record_name(args.label, index)
        _record_wav(
            output,
            seconds=args.seconds,
            sample_rate=args.sample_rate,
            channels=args.channels,
            device=args.device,
        )
        analysis = analyze_wav(output)
        metadata = {
            "label": args.label,
            "label_definition": _condition_text(args.label),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "read_only_capture": True,
            "pump_controlled_by_tool": False,
            "robot_controlled_by_tool": False,
            "device": args.device,
            "clip": analysis.to_dict(),
        }
        _write_json(output.with_suffix(".json"), metadata)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


def _features_command(args: argparse.Namespace) -> int:
    print(json.dumps(analyze_wav(args.input).to_dict(), ensure_ascii=False, indent=2))
    return 0


def _calibrate_command(args: argparse.Namespace) -> int:
    clips = analyze_dataset(args.samples_dir)
    model, report = train_model(
        clips,
        uncertainty_fraction=args.uncertainty_fraction,
        false_positive_cost=args.false_positive_cost,
        max_clipped_fraction=args.max_clipped_fraction,
    )
    model.save(args.model_out)
    report = {
        "model_path": str(args.model_out.resolve()),
        "sample_dir": str(args.samples_dir.resolve()),
        **report,
    }
    _write_json(args.report_out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _evaluate_command(args: argparse.Namespace) -> int:
    model = VacuumAudioModel.load(args.model)
    report = {
        "model_path": str(args.model.resolve()),
        "sample_dir": str(args.samples_dir.resolve()),
        **evaluate_model(model, analyze_dataset(args.samples_dir)),
    }
    if args.report_out is not None:
        _write_json(args.report_out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    metrics = report["metrics"]
    passed = (
        metrics["false_positive_rate"] <= args.max_false_positive_rate
        and metrics["conservative_false_negative_rate"]
        <= args.max_conservative_false_negative_rate
        and metrics["uncertain_rate"] <= args.max_uncertain_rate
    )
    return 0 if passed else 4


def _predict_command(args: argparse.Namespace) -> int:
    model = VacuumAudioModel.load(args.model)
    decision = model.predict(analyze_wav(args.input))
    print(json.dumps(decision.to_dict(), ensure_ascii=False, indent=2))
    if decision.state == SEALED_LABEL:
        return 0
    if decision.state == EMPTY_LABEL:
        return 2
    return 3


def _verify_command(args: argparse.Namespace) -> int:
    print("安全声明：本命令只录音并分析，不控制气泵、不连接机械臂、不发送运动。")
    if not args.no_prompt:
        input("请由外部系统建立待验证状态，声音稳定至少1秒后按 Enter 开始录音。")
    if args.output_wav is None:
        with tempfile.TemporaryDirectory(prefix="vacuum_audio_") as directory:
            output = Path(directory) / "verification.wav"
            _record_wav(
                output,
                seconds=args.seconds,
                sample_rate=args.sample_rate,
                channels=1,
                device=args.device,
            )
            decision = VacuumAudioModel.load(args.model).predict(analyze_wav(output))
    else:
        output = args.output_wav
        _record_wav(
            output,
            seconds=args.seconds,
            sample_rate=args.sample_rate,
            channels=1,
            device=args.device,
        )
        decision = VacuumAudioModel.load(args.model).predict(analyze_wav(output))
    print(json.dumps(decision.to_dict(), ensure_ascii=False, indent=2))
    if decision.state == SEALED_LABEL:
        return 0
    if decision.state == EMPTY_LABEL:
        return 2
    return 3


def _devices_command(_args: argparse.Namespace) -> int:
    recorder = shutil.which("arecord")
    if recorder is None:
        raise RuntimeError("arecord was not found")
    result = subprocess.run(
        [recorder, "-l"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, end="")
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only suction acoustics: records/analyses audio only; "
            "never controls pump or robot."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    devices = subparsers.add_parser("devices", help="List ALSA capture devices.")
    devices.set_defaults(handler=_devices_command)

    record = subparsers.add_parser(
        "record",
        help="Record manually established empty or sealed condition.",
    )
    record.add_argument("--label", choices=VALID_LABELS, required=True)
    record.add_argument("--out-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    record.add_argument("--seconds", type=int, default=3)
    record.add_argument("--count", type=int, default=1)
    record.add_argument("--sample-rate", type=int, default=16000)
    record.add_argument("--channels", type=int, choices=(1, 2), default=1)
    record.add_argument(
        "--device",
        default=DEFAULT_DOSW1_DEVICE,
        help="ALSA source; dosw1's UGREEN Camera 2K microphone is plughw:U2K,0.",
    )
    record.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not wait for Enter; still does not control pump or robot.",
    )
    record.set_defaults(handler=_record_command)

    features = subparsers.add_parser("features", help="Inspect one WAV clip.")
    features.add_argument("--input", type=Path, required=True)
    features.set_defaults(handler=_features_command)

    calibrate = subparsers.add_parser(
        "calibrate",
        help="Fit a conservative threshold from labeled clips.",
    )
    calibrate.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    calibrate.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    calibrate.add_argument(
        "--report-out",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "vacuum_audio" / "calibration_report.json",
    )
    calibrate.add_argument("--uncertainty-fraction", type=float, default=0.10)
    calibrate.add_argument("--false-positive-cost", type=float, default=2.0)
    calibrate.add_argument("--max-clipped-fraction", type=float, default=0.02)
    calibrate.set_defaults(handler=_calibrate_command)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate a fixed model on an independent labeled directory.",
    )
    evaluate.add_argument("--samples-dir", type=Path, required=True)
    evaluate.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    evaluate.add_argument("--report-out", type=Path)
    evaluate.add_argument("--max-false-positive-rate", type=float, default=0.0)
    evaluate.add_argument(
        "--max-conservative-false-negative-rate",
        type=float,
        default=0.05,
    )
    evaluate.add_argument("--max-uncertain-rate", type=float, default=0.10)
    evaluate.set_defaults(handler=_evaluate_command)

    predict = subparsers.add_parser("predict", help="Classify an existing WAV file.")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    predict.set_defaults(handler=_predict_command)

    verify = subparsers.add_parser(
        "verify",
        help="Record one temporary window, then classify it; no actuator control.",
    )
    verify.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    verify.add_argument("--seconds", type=int, default=2)
    verify.add_argument("--sample-rate", type=int, default=16000)
    verify.add_argument(
        "--device",
        default=DEFAULT_DOSW1_DEVICE,
        help="ALSA source; dosw1's UGREEN Camera 2K microphone is plughw:U2K,0.",
    )
    verify.add_argument("--output-wav", type=Path)
    verify.add_argument("--no-prompt", action="store_true")
    verify.set_defaults(handler=_verify_command)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("seconds", "count", "sample_rate"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "max_false_positive_rate",
        "max_conservative_false_negative_rate",
        "max_uncertain_rate",
    ):
        if hasattr(args, name):
            value = getattr(args, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if hasattr(args, "sample_rate") and args.sample_rate < 8000:
        raise ValueError("--sample-rate must be at least 8000")
    if hasattr(args, "seconds") and args.seconds < 1:
        raise ValueError("--seconds must be at least 1 because ALSA uses whole seconds")
    if hasattr(args, "count") and args.count < 1:
        raise ValueError("--count must be at least 1")


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        _validate_args(args)
        return int(args.handler(args))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"vacuum-audio error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
