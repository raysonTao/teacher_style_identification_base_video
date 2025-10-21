# teacher_style_identification_base_video

A research sandbox for building a teacher style identification system. The
project targets two major modalities: audio (speech) and video. This iteration
implements the full audio recognition stack, including dataset handling,
feature extraction, a lightweight recogniser, and automated evaluation.

## Features

- **Dataset manifest loader** for labelled audio clips with flexible metadata.
- **Pure Python harmonic feature extraction** with no heavy numerical
  dependencies, enabling offline experimentation in restricted environments.
- **Nearest neighbour recogniser** that projects audio into a compact embedding
  space for simple-yet-effective speech recognition baselines.
- **Evaluation utilities** to compute accuracy and confusion matrices.
- **End-to-end unit tests** that generate synthetic waveforms to validate the
  pipeline.

## Getting started

Install the project (and optional development dependencies for testing):

```bash
pip install -e .[dev]
```

Run the audio pipeline tests:

```bash
pytest
```

## Next steps

The repository will later be extended with the video understanding module. The
current codebase already exposes clear interfaces, making it straightforward to
plug in advanced models (e.g. neural CTC decoders or transformer-based
recognisers) once the dataset becomes available.
