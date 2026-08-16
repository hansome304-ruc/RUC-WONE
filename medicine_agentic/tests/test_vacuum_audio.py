from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medicine_agentic.vacuum_audio import (
    EMPTY_LABEL,
    SEALED_LABEL,
    VacuumAudioModel,
    analyze_dataset,
    analyze_wav,
    discover_labeled_wavs,
    evaluate_model,
    train_model,
)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1.0, 1.0)
    encoded = np.round(pcm * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(encoded)


def _synthetic_clip(
    label: str,
    seed: int,
    *,
    sample_rate: int = 16000,
    seconds: float = 1.5,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(int(sample_rate * seconds), dtype=np.float64) / sample_rate
    if label == EMPTY_LABEL:
        # Open-air hiss: broad high-frequency content and a weak pump tone.
        noise = rng.normal(0.0, 0.09, time.size)
        high = np.concatenate(([0.0], np.diff(noise)))
        return 0.55 * high + 0.015 * np.sin(2.0 * math.pi * 430.0 * time)
    # Sealed cup: stronger stable pump tone and much less broadband airflow.
    return (
        0.12 * np.sin(2.0 * math.pi * (430.0 + seed % 5) * time)
        + 0.035 * np.sin(2.0 * math.pi * 860.0 * time)
        + rng.normal(0.0, 0.008, time.size)
    )


class VacuumAudioTests(unittest.TestCase):
    def _make_dataset(self, root: Path, count: int = 6, seed_offset: int = 0) -> None:
        for index in range(count):
            for label in (EMPTY_LABEL, SEALED_LABEL):
                _write_wav(
                    root / label / f"{label}_{index:02d}.wav",
                    _synthetic_clip(label, seed_offset + index),
                )

    def test_discovers_directory_labels_and_extracts_finite_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_dataset(root, count=3)
            discovered = discover_labeled_wavs(root)
            self.assertEqual(len(discovered), 6)
            clip = analyze_wav(discovered[0][1])
            self.assertGreater(clip.duration_s, 1.0)
            self.assertEqual(clip.sample_rate_hz, 16000)
            self.assertTrue(all(math.isfinite(value) for value in clip.features.values()))

    def test_calibrates_round_trips_and_separates_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_dir = root / "train"
            test_dir = root / "test"
            self._make_dataset(train_dir, count=7)
            self._make_dataset(test_dir, count=5, seed_offset=100)

            model, calibration = train_model(analyze_dataset(train_dir))
            self.assertEqual(
                calibration["training_metrics"]["false_positive_count"],
                0,
            )
            model_path = root / "model.json"
            model.save(model_path)
            loaded = VacuumAudioModel.load(model_path)
            report = evaluate_model(loaded, analyze_dataset(test_dir))
            self.assertEqual(report["metrics"]["false_positive_count"], 0)
            self.assertEqual(
                report["metrics"]["conservative_false_negative_rate"],
                0.0,
            )

            payload = json.loads(model_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertIn("empty_label_definition", payload["metadata"])

    def test_uncertainty_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_dataset(root, count=5)
            model, _ = train_model(analyze_dataset(root))
            clip = analyze_dataset(root)[0][1]
            vector = clip.vector(model.feature_names)
            direction = np.asarray(model.direction)
            scale = np.asarray(model.scale)
            current_score = model.score_vector(vector)
            # Move the feature vector exactly onto the learned threshold.
            adjusted = vector + (model.threshold - current_score) * direction * scale
            altered = type(clip)(
                path=clip.path,
                sample_rate_hz=clip.sample_rate_hz,
                duration_s=clip.duration_s,
                clipped_fraction=clip.clipped_fraction,
                features={
                    name: float(value)
                    for name, value in zip(model.feature_names, adjusted)
                },
            )
            decision = model.predict(altered)
            self.assertEqual(decision.state, "uncertain")
            self.assertFalse(decision.sealed_ok)

    def test_clipped_audio_is_uncertain_even_if_spectrum_matches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_dataset(root, count=5)
            model, _ = train_model(analyze_dataset(root))
            sealed_clip = analyze_dataset(root)[-1][1]
            clipped_clip = type(sealed_clip)(
                path=sealed_clip.path,
                sample_rate_hz=sealed_clip.sample_rate_hz,
                duration_s=sealed_clip.duration_s,
                clipped_fraction=0.25,
                features=sealed_clip.features,
            )
            decision = model.predict(clipped_clip)
            self.assertEqual(decision.state, "uncertain")
            self.assertFalse(decision.sealed_ok)
            self.assertTrue(decision.quality_warnings)


if __name__ == "__main__":
    unittest.main()

