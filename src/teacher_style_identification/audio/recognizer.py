"""Simple nearest-neighbour audio recognizer using handcrafted features."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .data import AudioDataset, AudioSample
from .features import HarmonicFeatureExtractor
from .utils import load_wav_mono


@dataclass(slots=True)
class RecognitionResult:
    """Represents a transcription produced by the recognizer."""

    sample: AudioSample
    predicted_transcript: str
    distance: float


class NearestNeighborRecognizer:
    """Baseline speech recognizer using cosine similarity in feature space."""

    def __init__(self, feature_extractor: HarmonicFeatureExtractor | None = None) -> None:
        self.feature_extractor = feature_extractor or HarmonicFeatureExtractor()
        self._embeddings: List[List[float]] = []
        self._transcripts: List[str] = []

    def fit(self, dataset: AudioDataset | Sequence[AudioSample]) -> None:
        embeddings: List[List[float]] = []
        transcripts: List[str] = []
        for sample in dataset:
            signal, sample_rate = load_wav_mono(sample.path)
            features = self.feature_extractor.extract(signal, sample_rate)
            embeddings.append(self._normalize(features))
            transcripts.append(sample.transcript)

        if not embeddings:
            raise ValueError("Dataset must contain at least one sample")

        self._embeddings = embeddings
        self._transcripts = transcripts

    def is_fitted(self) -> bool:
        return bool(self._embeddings)

    def predict_sample(self, sample: AudioSample) -> RecognitionResult:
        if not self.is_fitted():
            raise RuntimeError("Recognizer must be fitted before calling predict_sample")

        signal, sample_rate = load_wav_mono(sample.path)
        features = self.feature_extractor.extract(signal, sample_rate)
        features = self._normalize(features)

        best_index = 0
        best_similarity = -1.0
        for index, reference in enumerate(self._embeddings):
            similarity = self._cosine_similarity(reference, features)
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = index

        predicted = self._transcripts[best_index]
        distance = 1.0 - best_similarity
        return RecognitionResult(sample=sample, predicted_transcript=predicted, distance=distance)

    def transcribe(self, samples: Iterable[AudioSample]) -> List[RecognitionResult]:
        return [self.predict_sample(sample) for sample in samples]

    @staticmethod
    def _normalize(vector: Sequence[float]) -> List[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return [0.0 for _ in vector]
        return [value / norm for value in vector]

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * y for x, y in zip(a, b))
