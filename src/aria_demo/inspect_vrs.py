"""Summarise the streams in an Aria VRS recording.

Used in milestone M0 to find out which signals a Gen 2 recording actually
contains (eye gaze, blinks, PPG, ALS, IMU, ...) before building on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# Data types whose first sample we open up to list the available fields.
PROBED_TYPES = {"EYE_GAZE", "PPG", "ALS", "HAND_POSE", "VIO"}


class _Provider(Protocol):
    """The subset of projectaria_tools' VrsDataProvider we rely on."""

    def get_all_streams(self) -> list[Any]: ...
    def get_label_from_stream_id(self, stream_id: Any) -> str | None: ...
    def get_sensor_data_type(self, stream_id: Any) -> Any: ...
    def get_num_data(self, stream_id: Any) -> int: ...
    def get_nominal_rate_hz(self, stream_id: Any) -> float: ...
    def get_first_time_ns(self, stream_id: Any, time_domain: Any) -> int: ...
    def get_last_time_ns(self, stream_id: Any, time_domain: Any) -> int: ...
    def get_sensor_data_by_index(self, stream_id: Any, index: int) -> Any: ...


@dataclass
class StreamSummary:
    stream_id: str
    label: str
    data_type: str
    num_samples: int
    nominal_rate_hz: float
    measured_rate_hz: float
    duration_s: float
    fields: list[str] = field(default_factory=list)


def public_fields(obj: Any) -> list[str]:
    """Public, non-callable attribute names of a pybind data object."""
    names = []
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:  # some pybind getters raise on invalid data
            continue
        if not callable(value):
            names.append(name)
    return sorted(names)


def _probe_fields(provider: _Provider, stream_id: Any, data_type: str) -> list[str]:
    sensor_data = provider.get_sensor_data_by_index(stream_id, 0)
    getter = {
        "EYE_GAZE": "eye_gaze_data",
        "PPG": "ppg_data",
        "ALS": "als_data",
        "HAND_POSE": "hand_pose_data",
        "VIO": "vio_data",
    }[data_type]
    record = getattr(sensor_data, getter)()
    fields = public_fields(record)
    # Eye gaze nests per-eye data (blinks, per-eye angles) under `vergence`.
    if data_type == "EYE_GAZE" and "vergence" in fields:
        fields += [f"vergence.{f}" for f in public_fields(record.vergence)]
    return fields


def summarize(provider: _Provider, time_domain: Any) -> list[StreamSummary]:
    summaries = []
    for stream_id in provider.get_all_streams():
        data_type = getattr(provider.get_sensor_data_type(stream_id), "name", "UNKNOWN")
        num = provider.get_num_data(stream_id)
        duration_s = 0.0
        if num > 1:
            first = provider.get_first_time_ns(stream_id, time_domain)
            last = provider.get_last_time_ns(stream_id, time_domain)
            duration_s = (last - first) / 1e9
        measured = (num - 1) / duration_s if duration_s > 0 else 0.0
        fields: list[str] = []
        if data_type in PROBED_TYPES and num > 0:
            try:
                fields = _probe_fields(provider, stream_id, data_type)
            except Exception as exc:  # keep the report going on a bad stream
                fields = [f"<probe failed: {exc}>"]
        summaries.append(
            StreamSummary(
                stream_id=str(stream_id),
                label=provider.get_label_from_stream_id(stream_id) or "?",
                data_type=data_type,
                num_samples=num,
                nominal_rate_hz=provider.get_nominal_rate_hz(stream_id),
                measured_rate_hz=measured,
                duration_s=duration_s,
                fields=fields,
            )
        )
    return sorted(summaries, key=lambda s: (s.data_type, s.label))


def render_markdown(summaries: list[StreamSummary], source: str, device: str) -> str:
    lines = [
        f"## Recording `{source}`",
        "",
        f"Device: {device}",
        "",
        "| Label | Type | Stream ID | Samples | Nominal Hz | Measured Hz | Duration s |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.label} | {s.data_type} | `{s.stream_id}` | {s.num_samples} | "
            f"{s.nominal_rate_hz:.1f} | {s.measured_rate_hz:.1f} | {s.duration_s:.1f} |"
        )
    probed = [s for s in summaries if s.fields]
    if probed:
        lines += ["", "### Fields per probed stream", ""]
        for s in probed:
            lines.append(f"- **{s.label}** ({s.data_type}): " + ", ".join(f"`{f}`" for f in s.fields))
    return "\n".join(lines) + "\n"


def inspect_vrs(path: Path) -> str:
    """Open a VRS file and return a Markdown report of its streams."""
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.sensor_data import TimeDomain

    provider = data_provider.create_vrs_data_provider(str(path))
    if provider is None:
        raise ValueError(f"Could not open VRS file: {path}")
    summaries = summarize(provider, TimeDomain.DEVICE_TIME)
    device = str(provider.get_device_version())
    return render_markdown(summaries, path.name, device)
