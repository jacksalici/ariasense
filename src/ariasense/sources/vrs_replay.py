"""Replay a VRS recording as a time-ordered stream of typed samples."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from ariasense.calibration.projection import RGB_LABEL
from ariasense.sources import convert
from ariasense.sources.base import ALL_SIGNALS, Pacer, Signal
from ariasense.types import Sample

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


def to_sample(sensor_data: Any, signal: Signal, label: str, t_ns: int) -> Sample | None:
    """Convert one SDK SensorData into our typed sample (None if unusable)."""
    if signal is Signal.RGB:
        image_data, _record = sensor_data.image_data_and_record()
        return convert.rgb_frame(image_data, t_ns)
    if signal is Signal.EYE_GAZE:
        return convert.eye_gaze_sample(sensor_data.eye_gaze_data(), t_ns)
    if signal is Signal.PPG:
        return convert.ppg_sample(sensor_data.ppg_data(), t_ns)
    if signal is Signal.ALS:
        return convert.als_sample(sensor_data.als_data(), t_ns)
    if signal is Signal.IMU:
        return convert.imu_sample(sensor_data.imu_data(), label, t_ns)
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
    def pacer(self) -> Pacer:
        """Exposed so a UI can pause (rebase) or change the playback speed."""
        return self._pacer

    @property
    def streams(self) -> dict[str, tuple[Signal, str]]:
        return dict(self._streams)

    def duration_s(self) -> float:
        """Length of the recording, from the first to the last sample of the replayed streams."""
        from projectaria_tools.core.sensor_data import TimeDomain

        first = min(self._provider.get_first_time_ns(s, TimeDomain.DEVICE_TIME) for s in self._stream_ids)
        last = max(self._provider.get_last_time_ns(s, TimeDomain.DEVICE_TIME) for s in self._stream_ids)
        return max(0.0, (last - first) / 1e9)

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
