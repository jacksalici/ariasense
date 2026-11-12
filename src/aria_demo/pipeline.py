"""The whole demo in one loop: source -> (YOLO + attention) and (features -> state).

Every message the dashboard receives is built here as plain JSON-able dicts:

- `frame`    (~10 Hz): JPEG, gaze, trail, detections, attended object
- `features` (1 Hz):   raw values + z-scores
- `state`    (1 Hz):   state, confidence, reasons, axes, baseline progress, attention totals
- `baseline` (once ready): per-feature mean/std, for the chart bands
- `status`:            playback info (paused, speed, ended, loop count)

Coordinates are normalised to 0..1 so the browser can draw at any size.
"""

from __future__ import annotations

import base64
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from aria_demo.calibration.projection import GazeProjector
from aria_demo.overlay import gaze_frames
from aria_demo.perception.attention import AttentionState, AttentionTracker
from aria_demo.perception.pipeline import Detector, PerceptionFrame, perceive
from aria_demo.state.baseline import Baseline
from aria_demo.state.config import StateConfig
from aria_demo.state.estimator import StateEstimate, StateEstimator
from aria_demo.state.features import FeatureExtractor, FeatureVector
from aria_demo.types import Sample

Message = dict[str, Any]
Emit = Callable[[Message], None]

JPEG_WIDTH = 1280
JPEG_QUALITY = 75
EXPECTED_STREAMS = ("rgb", "eye_gaze", "ppg", "imu")
HEALTH_EVERY_S = 5.0


def dwell_pseudo_z(attention: AttentionState | None) -> float | None:
    """Map dwell on the attended object to a z-like score for the attention axis.

    0 s -> -0.5, 1 s -> 0, 3 s -> +1, capped at +2.5. None when nothing is attended.
    """
    if attention is None or attention.target is None:
        return None
    return float(np.clip(attention.dwell_s / 2.0 - 0.5, -0.5, 2.5))


def _r(x: float | None, nd: int = 3) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def encode_jpeg(image_rgb: np.ndarray, width: int = JPEG_WIDTH, quality: int = JPEG_QUALITY) -> str:
    h, w = image_rgb.shape[:2]
    if w > width:
        image_rgb = cv2.resize(image_rgb, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def frame_message(pf: PerceptionFrame, t_rel_s: float, jpeg_b64: str | None) -> Message:
    h, w = pf.frame.image.shape[:2]

    def norm(px: tuple[float, float]) -> list[float]:
        return [round(px[0] / w, 4), round(px[1] / h, 4)]

    att = pf.attention
    attended_idx = next((i for i, d in enumerate(pf.detections) if d is att.target), None)
    return {
        "type": "frame",
        "t": round(t_rel_s, 2),
        "jpeg": jpeg_b64,
        "aspect": round(w / h, 4),
        "gaze": norm(pf.frame.gaze_px) if pf.frame.gaze_px is not None else None,
        "trail": [norm(p) for p in pf.frame.trail],
        "blink": pf.frame.blinking,
        "detections": [
            {
                "cls": d.class_name,
                "conf": round(d.confidence, 2),
                "box": [
                    round(d.box[0] / w, 4),
                    round(d.box[1] / h, 4),
                    round(d.box[2] / w, 4),
                    round(d.box[3] / h, 4),
                ],
                "id": d.track_id,
            }
            for d in pf.detections
        ],
        "attended": None
        if att.target is None
        else {"cls": att.target.class_name, "dwell": round(att.dwell_s, 1), "index": attended_idx},
    }


def features_message(fv: FeatureVector, t_rel_s: float, z: dict[str, float]) -> Message:
    return {
        "type": "features",
        "t": round(t_rel_s, 2),
        "values": {k: _r(v) for k, v in fv.values.items()},
        "z": {k: _r(v, 2) for k, v in z.items()},
    }


def state_message(
    est: StateEstimate | None,
    t_rel_s: float,
    baseline_progress: float,
    attention: AttentionState | None,
) -> Message:
    msg: Message = {
        "type": "state",
        "t": round(t_rel_s, 2),
        "baseline_progress": round(baseline_progress, 3),
        "attention_totals": {k: round(v, 1) for k, v in (attention.totals_s if attention else {}).items()},
        "attention_switches": attention.switches if attention else 0,
    }
    if est is None:
        msg.update(
            state="Calibrating", confidence=0.0, reasons=[], arousal=None, attention=None, candidate=None
        )
    else:
        msg.update(
            state=est.state.value,
            confidence=round(est.confidence, 2),
            reasons=est.reasons,
            arousal=_r(est.arousal, 2),
            attention=_r(est.attention, 2),
            candidate=est.candidate.value,
        )
    return msg


def baseline_message(baseline: Baseline) -> Message:
    return {
        "type": "baseline",
        "mean": {k: _r(v) for k, v in baseline.mean.items()},
        "std": {k: _r(v) for k, v in baseline.std.items()},
    }


@dataclass
class Controls:
    """Set from the web server thread, read by the pipeline thread."""

    paused: bool = False
    speed: float | None = None  # pending speed change
    rebaseline: bool = False
    stop: threading.Event = field(default_factory=threading.Event)


class DemoPipeline:
    def __init__(
        self,
        source_factory: Callable[[], Any],
        detector: Detector,
        config: StateConfig | None = None,
        baseline_s: float = 60.0,
        step_s: float = 1.0,
        trail_s: float = 0.5,
        loop: bool = False,
        depth_m: float = 1.0,
        encode: Callable[[np.ndarray], str | None] = encode_jpeg,
    ) -> None:
        self.source_factory = source_factory
        self.detector = detector
        self.config = config or StateConfig()
        self.baseline_s = baseline_s
        self.step_s = step_s
        self.trail_s = trail_s
        self.loop = loop
        self.depth_m = depth_m
        self.encode = encode
        self.controls = Controls()
        self.baseline: Baseline | None = None
        self._baseline_t0: int | None = None
        self._speed: float | None = None
        self._source: Any = None

    def shutdown(self) -> None:
        """Stop from another thread, even if a live stream has stalled (no samples to wake us)."""
        self.controls.stop.set()
        self.controls.paused = False
        source = self._source
        if source is not None and hasattr(source, "close"):
            source.close()  # samples() then drains and returns

    # ---- helpers ----------------------------------------------------------
    def _new_baseline(self) -> None:
        self.baseline = Baseline(duration_s=self.baseline_s)
        self._baseline_t0 = None

    def _progress(self, t_ns: int) -> float:
        if self.baseline is None or self.baseline.ready:
            return 1.0
        if self._baseline_t0 is None:
            return 0.0
        return min(1.0, (t_ns - self._baseline_t0) / 1e9 / self.baseline_s)

    def _wait_if_paused(self, source: Any, emit: Emit) -> None:
        if not self.controls.paused:
            return
        emit({"type": "status", "paused": True})
        while self.controls.paused and not self.controls.stop.is_set():
            time.sleep(0.05)
        if hasattr(source, "pacer"):
            source.pacer.rebase()
        if hasattr(source, "flush"):
            source.flush()  # live: drop what arrived while paused, continue in the present
        emit({"type": "status", "paused": False})

    def _stream_health(self, source: Any, emit: Emit, elapsed_s: float) -> None:
        """Live sources: report stream rates and warn about missing streams."""
        stats = source.stats()
        missing = [k for k in EXPECTED_STREAMS if stats.counts.get(k, 0) == 0] if elapsed_s >= 5 else []
        emit(
            {
                "type": "status",
                "streams": {k: round(v, 1) for k, v in stats.rates_hz.items()},
                "missing_streams": missing,
                "dropped_rgb": stats.dropped_rgb,
                "idle_s": round(stats.idle_s, 1),
            }
        )

    # ---- main loop ----------------------------------------------------------
    def run(self, emit: Emit) -> None:
        self._new_baseline()
        loops = 0
        while not self.controls.stop.is_set():
            source = self._source = self.source_factory()
            try:
                if self._speed is not None and hasattr(source, "pacer"):
                    source.pacer.rebase(self._speed)  # keep a speed chosen in the UI across loops
                projector = GazeProjector.from_device_calibration(
                    source.device_calibration(), default_depth_m=self.depth_m
                )
                emit({"type": "status", "loop": loops, "ended": False, "paused": False})
                self._run_once(source, projector, emit)
            finally:
                if hasattr(source, "close"):
                    source.close()  # live: stop streaming and release the receiver port
            loops += 1
            if not self.loop or self.controls.stop.is_set():
                break
        emit({"type": "status", "ended": True})

    def _run_once(self, source: Any, projector: GazeProjector, emit: Emit) -> None:
        extractor = FeatureExtractor(step_s=self.step_s)
        tracker = AttentionTracker()
        estimator = StateEstimator(self.config, self.baseline if self.baseline.ready else None)
        last_attention: AttentionState | None = None
        t0: int | None = None

        def on_features(fv: FeatureVector) -> None:
            nonlocal estimator
            assert t0 is not None and self.baseline is not None
            t_rel = (fv.t_ns - t0) / 1e9
            if self.controls.rebaseline:
                self.controls.rebaseline = False
                self._new_baseline()
                estimator = StateEstimator(self.config, None)
            if not self.baseline.ready:
                self._baseline_t0 = self._baseline_t0 or fv.t_ns
                if self.baseline.add(fv):  # just froze
                    estimator = StateEstimator(self.config, self.baseline)
                    emit(baseline_message(self.baseline))
            z = self.baseline.z(fv) if self.baseline.ready else {}
            est = None
            if self.baseline.ready:
                dz = dwell_pseudo_z(last_attention)
                est = estimator.update(fv, extras=None if dz is None else {"object_dwell_s": dz})
            emit(features_message(fv, t_rel, {k: v for k, v in z.items() if v is not None}))
            emit(state_message(est, t_rel, self._progress(fv.t_ns), last_attention))

        def tapped(samples: Iterable[Sample]) -> Iterator[Sample]:
            nonlocal t0
            for s in samples:
                if self.controls.stop.is_set():
                    return
                if self.controls.speed is not None and hasattr(source, "pacer"):
                    self._speed, self.controls.speed = self.controls.speed, None
                    source.pacer.rebase(self._speed)
                    emit({"type": "status", "speed": self._speed})
                self._wait_if_paused(source, emit)
                t0 = s.t_ns if t0 is None else t0
                fv = extractor.add(s)
                if fv is not None:
                    on_features(fv)
                yield s

        frames = gaze_frames(tapped(source.samples()), projector, self.trail_s)
        # Stream health runs on its own timer: it must keep reporting when the stream stalls.
        done = threading.Event()
        if hasattr(source, "stats"):
            started = time.monotonic()

            def health_loop() -> None:
                while not done.wait(max(HEALTH_EVERY_S, 0.2)):
                    self._stream_health(source, emit, time.monotonic() - started)

            threading.Thread(target=health_loop, name="aria-health", daemon=True).start()

        latency = getattr(source, "latency_ms", None)
        try:
            for pf in perceive(frames, self.detector, tracker):
                last_attention = pf.attention
                msg = frame_message(pf, pf.frame.t_rel_s, self.encode(pf.frame.image))
                if latency is not None:
                    # Capture -> fully processed (decode, YOLO, encode), measured with TicSync.
                    msg["latency_ms"] = _r(latency(pf.frame.t_ns), 0)
                emit(msg)
        finally:
            done.set()


def vrs_source_factory(path: Path, speed: float) -> Callable[[], Any]:
    from aria_demo.sources.base import ALL_SIGNALS
    from aria_demo.sources.vrs_replay import VrsReplaySource

    return lambda: VrsReplaySource(path, signals=ALL_SIGNALS, speed=speed)


def live_source_factory(**kwargs: Any) -> Callable[[], Any]:
    from aria_demo.sources.live import LiveSource

    return lambda: LiveSource(**kwargs)


def sdk_replay_source_factory(path: Path, speed: float = 1.0) -> Callable[[], Any]:
    from aria_demo.sources.live import SdkReplaySource

    return lambda: SdkReplaySource(path, speed=speed)
