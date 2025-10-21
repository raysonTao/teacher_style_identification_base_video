"""Data structures and loaders for audio recognition datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import csv


@dataclass(frozen=True)
class AudioSample:
    """Representation of a single labelled audio example."""

    path: Path
    transcript: str
    metadata: Dict[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if not self.path.suffix:
            raise ValueError("AudioSample path must include a file extension")


class AudioDataset(Sequence[AudioSample]):
    """Container holding a sequence of :class:`AudioSample` instances."""

    def __init__(self, samples: Iterable[AudioSample]):
        self._samples: List[AudioSample] = list(samples)
        if not self._samples:
            raise ValueError("AudioDataset requires at least one sample")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._samples)

    def __getitem__(self, index: int) -> AudioSample:  # pragma: no cover - trivial
        return self._samples[index]

    def __iter__(self) -> Iterator[AudioSample]:
        return iter(self._samples)

    @classmethod
    def from_csv(
        cls,
        manifest_path: Path | str,
        audio_root: Path | str | None = None,
        encoding: str = "utf-8",
    ) -> "AudioDataset":
        """Load a dataset from a CSV manifest.

        The CSV file must contain at least two columns named ``path`` and
        ``transcript``. All additional columns are stored inside the
        :class:`AudioSample` metadata dictionary.
        """

        manifest = Path(manifest_path)
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest}")

        root = Path(audio_root) if audio_root is not None else manifest.parent

        with manifest.open("r", encoding=encoding, newline="") as handle:
            reader = csv.DictReader(handle)
            if "path" not in reader.fieldnames or "transcript" not in reader.fieldnames:
                raise ValueError("Manifest must contain 'path' and 'transcript' columns")

            samples = []
            for row in reader:
                rel_path = Path(row["path"]).expanduser()
                audio_path = (root / rel_path).resolve()
                metadata = {k: v for k, v in row.items() if k not in {"path", "transcript"}}
                samples.append(AudioSample(audio_path, row["transcript"], metadata or None))

        return cls(samples)

    def transcripts(self) -> List[str]:
        """Return a list of transcripts contained in the dataset."""

        return [sample.transcript for sample in self._samples]
