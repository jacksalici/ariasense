"""Thin wrapper around Ultralytics YOLO that returns our `Detection` type."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ariasense.types import Detection

DEFAULT_MODEL = "yolo26s.pt"
MODELS_DIR = Path("models")


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def to_detections(boxes: Any, names: dict[int, str]) -> list[Detection]:
    """Convert an Ultralytics `Boxes` object (or anything shaped like it) to detections."""
    if boxes is None or len(boxes.xyxy) == 0:
        return []
    xyxy = np.asarray(boxes.xyxy.cpu() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy, dtype=float)
    conf = np.asarray(boxes.conf.cpu() if hasattr(boxes.conf, "cpu") else boxes.conf, dtype=float)
    cls = np.asarray(boxes.cls.cpu() if hasattr(boxes.cls, "cpu") else boxes.cls, dtype=int)
    ids = getattr(boxes, "id", None)
    ids = None if ids is None else np.asarray(ids.cpu() if hasattr(ids, "cpu") else ids, dtype=int)
    return [
        Detection(
            class_name=names[int(c)],
            confidence=float(p),
            box=tuple(float(x) for x in b),
            track_id=None if ids is None else int(ids[i]),
        )
        for i, (b, p, c) in enumerate(zip(xyxy, conf, cls, strict=True))
    ]


class YoloDetector:
    """YOLO with ByteTrack, so the same object keeps its `track_id` across frames."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        conf: float = 0.35,
        imgsz: int = 960,
        device: str | None = None,
        track: bool = True,
    ) -> None:
        from ultralytics import YOLO

        weights = Path(model)
        if not weights.exists() and weights.parent == Path("."):
            MODELS_DIR.mkdir(exist_ok=True)
            weights = MODELS_DIR / weights.name  # downloaded here on first use
        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device or pick_device()
        self.track = track

    def __call__(self, image_rgb: np.ndarray) -> list[Detection]:
        bgr = np.ascontiguousarray(image_rgb[..., ::-1])  # Ultralytics expects BGR arrays
        kwargs = dict(conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)
        if self.track:
            results = self.model.track(bgr, persist=True, tracker="bytetrack.yaml", **kwargs)
        else:
            results = self.model.predict(bgr, **kwargs)
        r = results[0]
        return to_detections(r.boxes, r.names)
