"""Audio recognition toolkit for teacher style identification."""

from .data import AudioSample, AudioDataset
from .features import HarmonicFeatureExtractor
from .recognizer import NearestNeighborRecognizer, RecognitionResult
from .evaluation import RecognitionReport, evaluate_recognizer

__all__ = [
    "AudioSample",
    "AudioDataset",
    "HarmonicFeatureExtractor",
    "NearestNeighborRecognizer",
    "RecognitionResult",
    "RecognitionReport",
    "evaluate_recognizer",
]
