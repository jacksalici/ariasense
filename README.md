# Aria Gen 2 Demo

A live demo for Meta Project Aria Gen 2. The dashboard shows:

- **What you're looking at:** YOLO detections on the RGB camera, with the object under your gaze highlighted.
- **How you're doing:** heart rate, HRV, blinks and fixations from the glasses' sensors.
- **An estimated state** (Focused, Startled, Stressed, Calm, Distracted, Fatigued), together with the reasons behind it.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`.

```bash
uv sync
```

## Usage

Inspect a recording, i.e. list its streams, rates and available fields:

```bash
uv run aria-demo inspect data/s01_desk.vrs
```

Append the report to the stream findings:

```bash
uv run aria-demo inspect data/s01_desk.vrs -o docs/streams.md
```

Render the gaze dot and trail on the upright RGB video. This is the M1 check that the dot lands on what you looked at:

```bash
uv run aria-demo overlay data/s01_desk.vrs -o runs/s01_overlay.mp4
```

Add `--show` for a live window and `--speed 1` to play back in real time.

## Development

```bash
uv run pytest
```

```bash
uv run ruff check . && uv run ruff format .
```

## Docs

- [docs/streams.md](docs/streams.md): which Gen 2 signals are available
- [docs/protocol.md](docs/protocol.md): recording and validation protocol
