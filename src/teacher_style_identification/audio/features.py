"""Feature extraction utilities for audio recognition (pure Python)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence


@dataclass(slots=True)
class HarmonicFeatureExtractor:
    """Compute hand-crafted spectral and temporal descriptors.

    The extractor keeps dependencies minimal by operating purely on Python
    lists. It focuses on low-frequency harmonic content that differentiates
    distinct speaking styles or vocal energy levels, which is often sufficient
    for baseline experiments or academic prototypes.
    """

    sample_rate: int = 16_000
    analysis_window: float = 0.025  # seconds
    hop_length: float = 0.010  # seconds
    probe_frequencies: Sequence[float] = field(
        default_factory=lambda: (120.0, 240.0, 480.0, 960.0)
    )

    def extract(self, signal: Sequence[float], sample_rate: int) -> List[float]:
        if sample_rate != self.sample_rate:
            signal = self._resample(signal, sample_rate, self.sample_rate)

        if not signal:
            return [0.0] * (len(self.probe_frequencies) + 4)

        # Energy-based features
        mean = sum(signal) / len(signal)
        variance = sum((x - mean) ** 2 for x in signal) / len(signal)
        std = math.sqrt(variance)
        mean_abs = sum(abs(x) for x in signal) / len(signal)
        zero_crossings = self._zero_crossing_rate(signal)

        # Spectral probes at harmonic frequencies to capture timbre/intonation.
        powers = [self._average_power(signal, freq, self.sample_rate) for freq in self.probe_frequencies]

        return [mean_abs, std, variance, zero_crossings, *powers]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resample(self, signal: Sequence[float], orig_sr: int, target_sr: int) -> List[float]:
        if orig_sr == target_sr or not signal:
            return list(signal)
        duration = len(signal) / orig_sr
        target_length = max(1, int(round(duration * target_sr)))
        step = (len(signal) - 1) / max(target_length - 1, 1)
        resampled = []
        for i in range(target_length):
            pos = i * step
            left = int(math.floor(pos))
            right = min(left + 1, len(signal) - 1)
            frac = pos - left
            value = (1 - frac) * signal[left] + frac * signal[right]
            resampled.append(value)
        return resampled

    @staticmethod
    def _zero_crossing_rate(signal: Sequence[float]) -> float:
        if len(signal) < 2:
            return 0.0
        crossings = 0
        prev = signal[0]
        for value in signal[1:]:
            if (prev >= 0 > value) or (prev <= 0 < value):
                crossings += 1
            prev = value
        return crossings / (len(signal) - 1)

    @staticmethod
    def _average_power(signal: Sequence[float], frequency: float, sample_rate: int) -> float:
        if frequency <= 0:
            return 0.0
        angular = 2.0 * math.pi * frequency / sample_rate
        cos_sum = 0.0
        sin_sum = 0.0
        for index, value in enumerate(signal):
            angle = angular * index
            cos_sum += value * math.cos(angle)
            sin_sum += value * math.sin(angle)
        scale = 2.0 / len(signal)
        return scale * (cos_sum ** 2 + sin_sum ** 2)
