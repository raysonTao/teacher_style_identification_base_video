"""Evaluation helpers for audio recognizers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List

from .data import AudioDataset, AudioSample
from .recognizer import NearestNeighborRecognizer, RecognitionResult


@dataclass(slots=True)
class RecognitionReport:
    """Stores aggregated evaluation statistics."""

    accuracy: float
    total_samples: int
    confusion: Dict[str, Dict[str, int]]

    def pretty(self) -> str:
        lines = [f"Accuracy: {self.accuracy:.3f} ({self.total_samples} samples)"]
        lines.append("Confusion matrix:")
        for truth, predictions in sorted(self.confusion.items()):
            preds = ", ".join(f"{pred}:{count}" for pred, count in sorted(predictions.items()))
            lines.append(f"  {truth}: {preds}")
        return "\n".join(lines)


def evaluate_recognizer(
    dataset: AudioDataset | Iterable[AudioSample], recognizer: NearestNeighborRecognizer
) -> RecognitionReport:
    """Evaluate ``recognizer`` against ``dataset`` and compute accuracy statistics."""

    results: List[RecognitionResult] = recognizer.transcribe(dataset)
    total = len(results)
    if total == 0:
        raise ValueError("Cannot evaluate on an empty dataset")

    correct = sum(result.predicted_transcript == result.sample.transcript for result in results)
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for result in results:
        truth = result.sample.transcript
        confusion[truth][result.predicted_transcript] += 1

    return RecognitionReport(
        accuracy=correct / total,
        total_samples=total,
        confusion={k: dict(v) for k, v in confusion.items()},
    )
