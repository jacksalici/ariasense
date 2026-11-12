"""Replay a VRS recording as a time-ordered stream of typed samples."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from aria_demo.calibration.projection import RGB_LABEL
from aria_demo.sources.base import ALL_SIGNALS, Pacer, Signal
from aria_demo.types import AlsSample, EyeGazeSample, ImuSample, PpgSample, RgbFrame, Sample

# SDK sensor-data type name -> our signal.
_TYPE_TO_SIGNAL = {
    "IMAGE": Signal.RGB,
    "EYE_GAZE": Signal.EYE_GAZE,
    "PPG": Signal.PPG,
    "ALS": Signal.ALS,
    "IMU": Signal.IMU,
}


def select_streams(
    provider: Any, signals: Iterable[Signal], rgb_label: str = RGB_LABEL
) -> dict[str, tuple[Signal, str]]:
    """Map str(stream id) -> (signal, label) for the streams that carry the wanted signals.

    Keys are strings because the SDK's StreamId is not hashable.

    Only the RGB camera is selected among IMAGE streams (not SLAM or eye-tracking cameras).
    """
    wanted = set(signals)
    selected = {}
    for stream_id in provider.get_all_streams():
        type_name = getattr(provider.get_sensor_data_type(stream_id), "name", "")
        signal = _TYPE_TO_SIGNAL.get(type_name)
        if signal is None or signal not in wanted:
            continue
        label = provider.get_label_from_stream_id(stream_id) or ""
        if signal is Signal.RGB and label != rgb_label:
            continue
        selected[str(stream_id)] = (signal, label)
    return selected


def _finite_or_none(x: float) -> float | None:
    return float(x) if np.isfinite(x) and x > 0 else None


def to_sample(sensor_data: Any, signal: Signal, label: str, t_ns: int) -> Sample | None:
    """Convert one SDK SensorData into our typed sample (None if unusable)."""
    if signal is Signal.RGB:
        image_data, _record = sensor_data.image_data_and_record()
        if not image_data.is_valid():
            return None
        image = image_data.to_numpy_array()
        if image.ndim == 2:  # grayscale fallback
            image = np.repeat(image[..., None], 3, axis=2)
        return RgbFrame(t_ns=t_ns, image=image)

    if signal is Signal.EYE_GAZE:
        g = sensor_data.eye_gaze_data()
        v = getattr(g, "vergence", None)
        return EyeGazeSample(
            t_ns=t_ns,
            yaw=float(g.yaw),
            pitch=float(g.pitch),
            depth_m=_finite_or_none(g.depth),
            valid=bool(getattr(g, "combined_gaze_valid", True)),
            # Gen 2 on-device flags are True while the eye is OPEN (verified on s01_desk:
            # True ~99% of samples, False in 3-4 sample bursts = blinks), so invert them.
            left_blink=(not v.left_blink) if v is not None and v.left_blink_valid else None,
            right_blink=(not v.right_blink) if v is not None and v.right_blink_valid else None,
        )

    if signal is Signal.PPG:
        return PpgSample(t_ns=t_ns, value=float(sensor_data.ppg_data().value))

    if signal is Signal.ALS:
        return AlsSample(t_ns=t_ns, lux=float(sensor_data.als_data().lux))

    if signal is Signal.IMU:
        m = sensor_data.imu_data()
        return ImuSample(
            t_ns=t_ns,
            label=label,
            accel=np.asarray(m.accel_msec2, dtype=float),
            gyro=np.asarray(m.gyro_radsec, dtype=float),
        )

    return None


class VrsReplaySource:
    """Replays a VRS file, optionally paced at `speed` x real time (0 = no pacing)."""

    def __init__(
        self,
        path: Path,
        signals: Iterable[Signal] = ALL_SIGNALS,
        speed: float = 1.0,
        rgb_label: str = RGB_LABEL,
    ) -> None:
        from projectaria_tools.core import data_provider

        self.path = Path(path)
        self._provider = data_provider.create_vrs_data_provider(str(self.path))
        if self._provider is None:
            raise ValueError(f"Could not open VRS file: {self.path}")
        self._streams = select_streams(self._provider, signals, rgb_label)
        self._stream_ids = [s for s in self._provider.get_all_streams() if str(s) in self._streams]
        if not self._streams:
            raise ValueError(f"No matching streams in {self.path} for {sorted(signals)}")
        self._pacer = Pacer(speed)

    @property
    def streams(self) -> dict[str, tuple[Signal, str]]:
        return dict(self._streams)

    def device_calibration(self) -> Any:
        calib = self._provider.get_device_calibration()
        if calib is None:
            raise ValueError(f"No device calibration in {self.path}")
        return calib

    def samples(self) -> Iterator[Sample]:
        from projectaria_tools.core.sensor_data import TimeDomain

        options = self._provider.get_default_deliver_queued_options()
        options.deactivate_stream_all()
        for stream_id in self._stream_ids:
            options.activate_stream(stream_id)

        for sensor_data in self._provider.deliver_queued_sensor_data(options):
            signal, label = self._streams[str(sensor_data.stream_id())]
            t_ns = sensor_data.get_time_ns(TimeDomain.DEVICE_TIME)
            sample = to_sample(sensor_data, signal, label, t_ns)
            if sample is None:
                continue
            self._pacer.wait_until(t_ns)
            yield sample
