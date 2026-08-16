"""Read-only acoustic evidence for suction-cup seal verification.

This module only reads WAV files and performs numerical analysis.  It has no
dependency on robot, pump, GPIO, fieldbus, or motion-control code.  Acoustic
evidence is intentionally fail-closed: only a confident ``sealed`` decision
sets ``sealed_ok`` to true; ``uncertain`` never does.
"""
from __future__ import annotations

import json
import math
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


MODEL_SCHEMA_VERSION = 1
EMPTY_LABEL = "empty"
SEALED_LABEL = "sealed"
VALID_LABELS = (EMPTY_LABEL, SEALED_LABEL)
FEATURE_NAMES = (
    "rms_dbfs",
    "rms_iqr_db",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_flatness",
    "rolloff85_hz",
    "low_band_ratio",
    "mid_band_ratio",
    "high_band_ratio",
    "peak_ratio",
    "spectral_flux",
)


@dataclass(frozen=True)
class ClipAnalysis:
    path: str
    sample_rate_hz: int
    duration_s: float
    clipped_fraction: float
    features: dict[str, float]

    def vector(self, feature_names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
        missing = [name for name in feature_names if name not in self.features]
        if missing:
            raise ValueError(f"clip is missing model features: {missing}")
        return np.asarray([self.features[name] for name in feature_names], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AcousticDecision:
    state: str
    sealed_ok: bool
    score: float
    threshold: float
    uncertainty_margin: float
    signed_margin: float
    quality_warnings: tuple[str, ...]
    clip: ClipAnalysis

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quality_warnings"] = list(self.quality_warnings)
        return payload


@dataclass(frozen=True)
class VacuumAudioModel:
    feature_names: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    direction: tuple[float, ...]
    threshold: float
    uncertainty_margin: float
    max_clipped_fraction: float
    training_sample_rates_hz: tuple[int, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        size = len(self.feature_names)
        if size == 0:
            raise ValueError("model must contain at least one feature")
        if len(self.center) != size or len(self.scale) != size or len(self.direction) != size:
            raise ValueError("model vectors must have the same length as feature_names")
        if any(value <= 0.0 or not math.isfinite(value) for value in self.scale):
            raise ValueError("all model scales must be finite and positive")
        if self.uncertainty_margin < 0.0:
            raise ValueError("uncertainty_margin cannot be negative")

    def score_vector(self, vector: np.ndarray) -> float:
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (len(self.feature_names),):
            raise ValueError(
                f"expected feature vector shape {(len(self.feature_names),)}, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("feature vector contains non-finite values")
        center = np.asarray(self.center, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        direction = np.asarray(self.direction, dtype=np.float64)
        return float(np.dot((values - center) / scale, direction))

    def predict(self, clip: ClipAnalysis) -> AcousticDecision:
        score = self.score_vector(clip.vector(self.feature_names))
        signed_margin = score - self.threshold
        warnings: list[str] = []
        if clip.clipped_fraction > self.max_clipped_fraction:
            warnings.append(
                "audio clipping exceeds calibrated quality limit "
                f"({clip.clipped_fraction:.4f} > {self.max_clipped_fraction:.4f})"
            )
        if (
            self.training_sample_rates_hz
            and clip.sample_rate_hz not in self.training_sample_rates_hz
        ):
            warnings.append(
                f"sample rate {clip.sample_rate_hz} Hz was not present during calibration"
            )

        if warnings:
            state = "uncertain"
        elif signed_margin >= self.uncertainty_margin:
            state = SEALED_LABEL
        elif signed_margin <= -self.uncertainty_margin:
            state = EMPTY_LABEL
        else:
            state = "uncertain"
        return AcousticDecision(
            state=state,
            sealed_ok=state == SEALED_LABEL,
            score=score,
            threshold=self.threshold,
            uncertainty_margin=self.uncertainty_margin,
            signed_margin=signed_margin,
            quality_warnings=tuple(warnings),
            clip=clip,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "feature_names": list(self.feature_names),
            "center": list(self.center),
            "scale": list(self.scale),
            "direction": list(self.direction),
            "threshold": self.threshold,
            "uncertainty_margin": self.uncertainty_margin,
            "max_clipped_fraction": self.max_clipped_fraction,
            "training_sample_rates_hz": list(self.training_sample_rates_hz),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VacuumAudioModel":
        version = int(payload.get("schema_version", -1))
        if version != MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported vacuum-audio model schema {version}; "
                f"expected {MODEL_SCHEMA_VERSION}"
            )
        return cls(
            feature_names=tuple(str(value) for value in payload["feature_names"]),
            center=tuple(float(value) for value in payload["center"]),
            scale=tuple(float(value) for value in payload["scale"]),
            direction=tuple(float(value) for value in payload["direction"]),
            threshold=float(payload["threshold"]),
            uncertainty_margin=float(payload["uncertainty_margin"]),
            max_clipped_fraction=float(payload.get("max_clipped_fraction", 0.02)),
            training_sample_rates_hz=tuple(
                int(value) for value in payload.get("training_sample_rates_hz", [])
            ),
            metadata=dict(payload.get("metadata", {})),
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "VacuumAudioModel":
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("vacuum-audio model root must be a JSON object")
        return cls.from_dict(payload)


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8)
        if packed.size % 3:
            raise ValueError("invalid 24-bit PCM payload length")
        triples = packed.reshape(-1, 3).astype(np.int32)
        values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float64) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    raise ValueError(f"unsupported PCM sample width: {sample_width} bytes")


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    source = Path(path)
    with wave.open(str(source), "rb") as stream:
        if stream.getcomptype() != "NONE":
            raise ValueError(f"compressed WAV is unsupported: {stream.getcomptype()}")
        channels = stream.getnchannels()
        sample_rate = stream.getframerate()
        sample_width = stream.getsampwidth()
        frame_count = stream.getnframes()
        raw = stream.readframes(frame_count)
    if channels <= 0 or sample_rate <= 0:
        raise ValueError(f"invalid WAV metadata: channels={channels}, rate={sample_rate}")
    samples = _decode_pcm(raw, sample_width)
    if samples.size % channels:
        raise ValueError("PCM sample count is not divisible by channel count")
    samples = samples.reshape(-1, channels).mean(axis=1)
    if samples.size == 0:
        raise ValueError(f"WAV contains no samples: {source}")
    return np.clip(samples, -1.0, 1.0), int(sample_rate)


def _frame_signal(
    samples: np.ndarray,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    if samples.size < frame_length:
        samples = np.pad(samples, (0, frame_length - samples.size))
    frame_count = 1 + (samples.size - frame_length) // hop_length
    starts = np.arange(frame_count)[:, None] * hop_length
    offsets = np.arange(frame_length)[None, :]
    return samples[starts + offsets]


def _band_ratio(
    power: np.ndarray,
    frequencies: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    mask = (frequencies >= low_hz) & (frequencies < high_hz)
    denominator = np.sum(power, axis=1) + 1e-18
    if not np.any(mask):
        return np.zeros(power.shape[0], dtype=np.float64)
    return np.sum(power[:, mask], axis=1) / denominator


def extract_features(
    samples: np.ndarray,
    sample_rate_hz: int,
    *,
    trim_s: float = 0.15,
    frame_ms: float = 64.0,
    hop_ms: float = 32.0,
) -> dict[str, float]:
    """Extract robust frame-median acoustic features from a mono waveform."""
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if sample_rate_hz < 8000:
        raise ValueError("sample rate must be at least 8000 Hz")
    if values.size / sample_rate_hz < 0.5:
        raise ValueError("audio clip must be at least 0.5 seconds")
    if not np.all(np.isfinite(values)):
        raise ValueError("audio clip contains non-finite samples")

    trim = int(round(trim_s * sample_rate_hz))
    if trim > 0 and values.size > 2 * trim + sample_rate_hz // 4:
        values = values[trim:-trim]
    values = values - float(np.mean(values))
    overall_rms = float(np.sqrt(np.mean(values * values) + 1e-18))
    if overall_rms < 1e-6:
        raise ValueError("audio clip is effectively silent")

    frame_length = max(128, int(round(frame_ms * sample_rate_hz / 1000.0)))
    frame_length = 1 << int(round(math.log2(frame_length)))
    hop_length = max(1, int(round(hop_ms * sample_rate_hz / 1000.0)))
    frames = _frame_signal(values, frame_length, hop_length)
    frames = frames - np.mean(frames, axis=1, keepdims=True)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-18)
    rms_db = 20.0 * np.log10(rms + 1e-12)
    signs = np.signbit(frames)
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)

    windowed = frames * np.hanning(frame_length)[None, :]
    spectrum = np.fft.rfft(windowed, axis=1)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(frame_length, d=1.0 / sample_rate_hz)
    valid_spectrum = frequencies >= 50.0
    power = power[:, valid_spectrum] + 1e-18
    frequencies = frequencies[valid_spectrum]
    power_sum = np.sum(power, axis=1) + 1e-18

    centroid = np.sum(power * frequencies[None, :], axis=1) / power_sum
    flatness = np.exp(np.mean(np.log(power), axis=1)) / np.mean(power, axis=1)
    cumulative = np.cumsum(power, axis=1)
    rolloff_indices = np.argmax(cumulative >= 0.85 * power_sum[:, None], axis=1)
    rolloff = frequencies[rolloff_indices]
    nyquist = sample_rate_hz / 2.0
    low = _band_ratio(power, frequencies, 80.0, min(500.0, nyquist))
    mid = _band_ratio(power, frequencies, 500.0, min(2000.0, nyquist))
    high = _band_ratio(power, frequencies, 2000.0, min(6000.0, nyquist))
    peak_ratio = np.max(power, axis=1) / power_sum

    normalized_power = power / power_sum[:, None]
    if normalized_power.shape[0] > 1:
        flux = np.sqrt(
            np.mean(np.diff(normalized_power, axis=0) ** 2, axis=1)
        )
        spectral_flux = float(np.median(flux))
    else:
        spectral_flux = 0.0

    q75, q25 = np.percentile(rms_db, [75.0, 25.0])
    result = {
        "rms_dbfs": float(np.median(rms_db)),
        "rms_iqr_db": float(q75 - q25),
        "zero_crossing_rate": float(np.median(zcr)),
        "spectral_centroid_hz": float(np.median(centroid)),
        "spectral_flatness": float(np.median(flatness)),
        "rolloff85_hz": float(np.median(rolloff)),
        "low_band_ratio": float(np.median(low)),
        "mid_band_ratio": float(np.median(mid)),
        "high_band_ratio": float(np.median(high)),
        "peak_ratio": float(np.median(peak_ratio)),
        "spectral_flux": spectral_flux,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("feature extraction produced non-finite values")
    return result


def analyze_wav(path: str | Path) -> ClipAnalysis:
    samples, sample_rate = read_wav_mono(path)
    return ClipAnalysis(
        path=str(Path(path).resolve()),
        sample_rate_hz=sample_rate,
        duration_s=float(samples.size / sample_rate),
        clipped_fraction=float(np.mean(np.abs(samples) >= 0.999)),
        features=extract_features(samples, sample_rate),
    )


def discover_labeled_wavs(root: str | Path) -> list[tuple[str, Path]]:
    """Find ``empty`` and ``sealed`` WAVs by subdirectory or filename prefix."""
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"sample directory does not exist: {base}")
    found: dict[Path, str] = {}
    for label in VALID_LABELS:
        label_dir = base / label
        if label_dir.is_dir():
            for path in label_dir.rglob("*.wav"):
                found[path.resolve()] = label
        for path in base.glob(f"{label}_*.wav"):
            found[path.resolve()] = label
    return sorted(((label, path) for path, label in found.items()), key=lambda item: str(item[1]))


def _robust_center_scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(matrix, axis=0)
    q75, q25 = np.percentile(matrix, [75.0, 25.0], axis=0)
    scale = (q75 - q25) / 1.349
    standard_deviation = np.std(matrix, axis=0)
    scale = np.where(scale > 1e-8, scale, standard_deviation)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return center, scale


def _select_threshold(
    scores: np.ndarray,
    labels: Sequence[str],
    false_positive_cost: float,
) -> float:
    unique = np.unique(scores)
    if unique.size == 1:
        raise ValueError("empty and sealed samples produce indistinguishable scores")
    epsilon = max(1e-9, float(np.ptp(unique)) * 1e-6)
    candidates = [float(unique[0] - epsilon), float(unique[-1] + epsilon)]
    candidates.extend(float((left + right) / 2.0) for left, right in zip(unique[:-1], unique[1:]))
    label_array = np.asarray(labels)
    empty_mask = label_array == EMPTY_LABEL
    sealed_mask = label_array == SEALED_LABEL
    best: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        predicted_sealed = scores >= threshold
        fpr = float(np.mean(predicted_sealed[empty_mask]))
        fnr = float(np.mean(~predicted_sealed[sealed_mask]))
        loss = false_positive_cost * fpr + fnr
        distance = float(np.min(np.abs(scores - threshold)))
        candidate = (loss, fpr, -distance, threshold)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best[3]


def _classification_metrics(
    decisions: Sequence[str],
    labels: Sequence[str],
) -> dict[str, Any]:
    if len(decisions) != len(labels):
        raise ValueError("decision and label counts differ")
    empty_count = sum(label == EMPTY_LABEL for label in labels)
    sealed_count = sum(label == SEALED_LABEL for label in labels)
    if not empty_count or not sealed_count:
        raise ValueError("metrics require both empty and sealed samples")
    false_positive = sum(
        label == EMPTY_LABEL and decision == SEALED_LABEL
        for label, decision in zip(labels, decisions)
    )
    false_negative = sum(
        label == SEALED_LABEL and decision == EMPTY_LABEL
        for label, decision in zip(labels, decisions)
    )
    uncertain_empty = sum(
        label == EMPTY_LABEL and decision == "uncertain"
        for label, decision in zip(labels, decisions)
    )
    uncertain_sealed = sum(
        label == SEALED_LABEL and decision == "uncertain"
        for label, decision in zip(labels, decisions)
    )
    return {
        "sample_count": len(labels),
        "empty_count": empty_count,
        "sealed_count": sealed_count,
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
        "uncertain_empty_count": uncertain_empty,
        "uncertain_sealed_count": uncertain_sealed,
        "false_positive_rate": false_positive / empty_count,
        "false_negative_rate": false_negative / sealed_count,
        # For safe suction gating, an uncertain sealed sample is also not accepted.
        "conservative_false_negative_rate": (
            false_negative + uncertain_sealed
        )
        / sealed_count,
        "uncertain_rate": (uncertain_empty + uncertain_sealed) / len(labels),
    }


def train_model(
    labeled_clips: Sequence[tuple[str, ClipAnalysis]],
    *,
    uncertainty_fraction: float = 0.10,
    false_positive_cost: float = 2.0,
    max_clipped_fraction: float = 0.02,
) -> tuple[VacuumAudioModel, dict[str, Any]]:
    if not 0.0 <= uncertainty_fraction < 0.5:
        raise ValueError("uncertainty_fraction must be in [0, 0.5)")
    if false_positive_cost <= 0.0:
        raise ValueError("false_positive_cost must be positive")
    labels = [label for label, _ in labeled_clips]
    if any(label not in VALID_LABELS for label in labels):
        raise ValueError(f"labels must be one of {VALID_LABELS}")
    counts = {label: labels.count(label) for label in VALID_LABELS}
    if min(counts.values(), default=0) < 3:
        raise ValueError("calibration requires at least 3 empty and 3 sealed clips")

    matrix = np.stack([clip.vector(FEATURE_NAMES) for _, clip in labeled_clips])
    center, scale = _robust_center_scale(matrix)
    standardized = (matrix - center) / scale
    label_array = np.asarray(labels)
    empty_center = np.median(standardized[label_array == EMPTY_LABEL], axis=0)
    sealed_center = np.median(standardized[label_array == SEALED_LABEL], axis=0)
    direction = sealed_center - empty_center
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        raise ValueError("empty and sealed feature centers are indistinguishable")
    direction /= norm
    scores = standardized @ direction
    threshold = _select_threshold(scores, labels, false_positive_cost)
    empty_median = float(np.median(scores[label_array == EMPTY_LABEL]))
    sealed_median = float(np.median(scores[label_array == SEALED_LABEL]))
    if sealed_median <= empty_median:
        raise ValueError("internal score orientation error")
    uncertainty_margin = uncertainty_fraction * (sealed_median - empty_median)

    model = VacuumAudioModel(
        feature_names=FEATURE_NAMES,
        center=tuple(float(value) for value in center),
        scale=tuple(float(value) for value in scale),
        direction=tuple(float(value) for value in direction),
        threshold=float(threshold),
        uncertainty_margin=float(uncertainty_margin),
        max_clipped_fraction=float(max_clipped_fraction),
        training_sample_rates_hz=tuple(
            sorted({clip.sample_rate_hz for _, clip in labeled_clips})
        ),
        metadata={
            "empty_label_definition": "pump on, suction cup open/unsealed",
            "sealed_label_definition": "pump on, suction cup sealed on representative carton",
            "false_positive_cost": false_positive_cost,
            "uncertainty_fraction": uncertainty_fraction,
            "training_clip_count": len(labeled_clips),
        },
    )
    decisions = [model.predict(clip).state for _, clip in labeled_clips]
    report = {
        "labels": counts,
        "score_summary": {
            EMPTY_LABEL: {
                "median": empty_median,
                "min": float(np.min(scores[label_array == EMPTY_LABEL])),
                "max": float(np.max(scores[label_array == EMPTY_LABEL])),
            },
            SEALED_LABEL: {
                "median": sealed_median,
                "min": float(np.min(scores[label_array == SEALED_LABEL])),
                "max": float(np.max(scores[label_array == SEALED_LABEL])),
            },
            "threshold": threshold,
            "uncertainty_margin": uncertainty_margin,
        },
        "training_metrics": _classification_metrics(decisions, labels),
        "samples": [
            {
                "path": clip.path,
                "label": label,
                "score": float(score),
                "decision": decision,
                "clipped_fraction": clip.clipped_fraction,
            }
            for (label, clip), score, decision in zip(labeled_clips, scores, decisions)
        ],
    }
    return model, report


def evaluate_model(
    model: VacuumAudioModel,
    labeled_clips: Sequence[tuple[str, ClipAnalysis]],
) -> dict[str, Any]:
    labels = [label for label, _ in labeled_clips]
    decisions = [model.predict(clip) for _, clip in labeled_clips]
    report = {
        "metrics": _classification_metrics(
            [decision.state for decision in decisions],
            labels,
        ),
        "samples": [
            {
                "path": clip.path,
                "label": label,
                "decision": decision.state,
                "sealed_ok": decision.sealed_ok,
                "score": decision.score,
                "signed_margin": decision.signed_margin,
                "quality_warnings": list(decision.quality_warnings),
            }
            for (label, clip), decision in zip(labeled_clips, decisions)
        ],
    }
    return report


def analyze_dataset(root: str | Path) -> list[tuple[str, ClipAnalysis]]:
    discovered = discover_labeled_wavs(root)
    if not discovered:
        raise ValueError(
            f"no labeled WAV files found below {root}; expected empty/ and sealed/"
        )
    return [(label, analyze_wav(path)) for label, path in discovered]

