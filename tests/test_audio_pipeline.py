from __future__ import annotations

import csv
import math
from pathlib import Path

from teacher_style_identification.audio import (
    AudioDataset,
    AudioSample,
    HarmonicFeatureExtractor,
    NearestNeighborRecognizer,
    RecognitionReport,
    evaluate_recognizer,
)
from teacher_style_identification.audio.utils import save_wav_mono


SAMPLE_RATE = 16_000
DURATION = 0.5


def _generate_tone(frequency: float, phase: float = 0.0) -> list[float]:
    total_samples = int(SAMPLE_RATE * DURATION)
    return [0.5 * math.sin(2 * math.pi * frequency * (n / SAMPLE_RATE) + phase) for n in range(total_samples)]


def _write_manifest(tmp_path: Path, items: list[tuple[str, str]]) -> Path:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "transcript"])
        writer.writerows(items)
    return manifest


def _prepare_dataset(tmp_path: Path) -> AudioDataset:
    samples = []
    for idx, (freq, word) in enumerate(((300.0, "calm"), (600.0, "energetic"))):
        for j in range(3):
            signal = _generate_tone(freq, phase=j * math.pi / 6)
            file_path = tmp_path / f"sample_{word}_{idx}_{j}.wav"
            save_wav_mono(file_path, signal, SAMPLE_RATE)
            samples.append((file_path.name, word))

    manifest = _write_manifest(tmp_path, samples)
    return AudioDataset.from_csv(manifest, audio_root=tmp_path)


def test_nearest_neighbor_recognizer_fits_and_predicts(tmp_path: Path) -> None:
    dataset = _prepare_dataset(tmp_path)
    recognizer = NearestNeighborRecognizer(HarmonicFeatureExtractor())
    recognizer.fit(dataset)

    energetic_signal = _generate_tone(600.0, phase=math.pi / 5)
    sample_path = tmp_path / "probe.wav"
    save_wav_mono(sample_path, energetic_signal, SAMPLE_RATE)
    sample = AudioSample(sample_path, transcript="energetic")

    result = recognizer.predict_sample(sample)
    assert result.predicted_transcript == "energetic"
    assert 0 <= result.distance < 0.5


def test_evaluation_reports_accuracy(tmp_path: Path) -> None:
    dataset = _prepare_dataset(tmp_path)
    recognizer = NearestNeighborRecognizer()
    recognizer.fit(dataset)

    report = evaluate_recognizer(dataset, recognizer)
    assert isinstance(report, RecognitionReport)
    assert report.total_samples == len(dataset)
    assert report.accuracy == 1.0
    pretty = report.pretty()
    assert "Accuracy" in pretty and "Confusion matrix" in pretty
