"""Utility helpers for audio processing (pure Python)."""

from __future__ import annotations

from array import array
from pathlib import Path
from typing import Sequence, Tuple


SUPPORTED_SAMPLE_WIDTHS = {1: "B", 2: "h", 4: "i"}


def load_wav_mono(path: Path | str) -> Tuple[list[float], int]:
    """Load a WAV file as a mono floating point signal in the range [-1, 1]."""

    import wave

    wav_path = Path(path)
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    with wave.open(str(wav_path), "rb") as handle:
        sample_width = handle.getsampwidth()
        if sample_width not in SUPPORTED_SAMPLE_WIDTHS:
            raise ValueError(f"Unsupported sample width: {sample_width} bytes")
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        channels = handle.getnchannels()
        raw = handle.readframes(frames)

    typecode = SUPPORTED_SAMPLE_WIDTHS[sample_width]
    audio = array(typecode)
    audio.frombytes(raw)

    if channels > 1:
        mono = []
        for i in range(0, len(audio), channels):
            chunk = audio[i : i + channels]
            mono.append(sum(chunk) / len(chunk))
        audio = array(typecode, (int(x) for x in mono))

    if sample_width == 1:
        result = [(value - 128) / 128.0 for value in audio]
    else:
        max_value = float(2 ** (8 * sample_width - 1))
        result = [value / max_value for value in audio]

    return result, sample_rate


def save_wav_mono(path: Path | str, signal: Sequence[float], sample_rate: int) -> None:
    """Save a mono floating point signal to a WAV file."""

    import wave

    wav_path = Path(path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    clipped = [min(1.0, max(-1.0, value)) for value in signal]
    int_samples = array("h", (int(value * 32767.0) for value in clipped))

    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(int_samples.tobytes())
