"""Streaming sources built on the Aria Gen 2 Client SDK callbacks.

- `LiveSource` streams from the glasses (USB, Wi-Fi, or the glasses' hotspot).
- `SdkReplaySource` pushes a VRS file through the *same* callbacks
  (`create_replay_source`), so the live path can be tested without a device.

The callbacks run on SDK threads and arrive slightly out of time order across
streams. They are queued, re-ordered with a small delay, and yielded as typed
samples. If processing falls behind, older RGB frames are dropped (the newest
one is kept); the small sensor samples are never dropped.
"""

from __future__ import annotations

import heapq
import itertools
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ariasense.sources import convert
from ariasense.types import RgbFrame, Sample

STREAMING_INTERFACES = ("usb", "wifi_sta", "wifi_sap")
RECEIVER_PORT = 6768


@dataclass
class StreamStats:
    counts: dict[str, int]
    rates_hz: dict[str, float]
    dropped_rgb: int
    late_dropped: int
    idle_s: float = 0.0  # wall-clock seconds since the last sample arrived


class Reorderer:
    """Merges the streams roughly in time order, keeping each stream strictly in order.

    Samples are held until they are `delay_ns` older than the newest seen, then
    released in timestamp order. A sample that arrives after newer samples of
    *other* streams were released is passed straight through instead of dropped:
    consumers only need each stream to be in order (feature windows and gaze
    lookups are per stream). On the glasses, PPG and IMU arrive in batches up to
    ~1 s behind the other streams, even over USB; a global order dropped them.
    Only a sample older than one already released *from its own stream* is
    dropped (counted in `late`).
    """

    def __init__(self, delay_ns: int) -> None:
        self.delay_ns = delay_ns
        self._heap: list[tuple[int, int, Sample]] = []
        self._seq = itertools.count()
        self._newest = -1
        self._released = -1
        self._released_by_type: dict[type, int] = {}
        self._passthrough: list[Sample] = []
        self.late = 0

    def push(self, sample: Sample) -> None:
        if sample.t_ns < self._released:
            kind = type(sample)
            if sample.t_ns < self._released_by_type.get(kind, -1):
                self.late += 1  # out of order within its own stream
                return
            self._released_by_type[kind] = sample.t_ns
            self._passthrough.append(sample)
            return
        heapq.heappush(self._heap, (sample.t_ns, next(self._seq), sample))
        self._newest = max(self._newest, sample.t_ns)

    def pop_ready(self, flush: bool = False) -> Iterator[Sample]:
        passthrough, self._passthrough = self._passthrough, []
        yield from passthrough
        while self._heap and (flush or self._heap[0][0] <= self._newest - self.delay_ns):
            t, _, sample = heapq.heappop(self._heap)
            kind = type(sample)
            if t < self._released_by_type.get(kind, -1):
                self.late += 1  # a passed-through sample of this stream was newer
                continue
            self._released = t
            self._released_by_type[kind] = t
            yield sample

    def __len__(self) -> int:
        return len(self._heap)


class StreamSource:
    """Common machinery: callbacks -> thread-safe queue -> re-ordered sample iterator."""

    def __init__(
        self,
        reorder_delay_s: float = 0.1,
        imu_label: str | None = None,
        imu_decimate: int = 4,
        calib_timeout_s: float = 15.0,
    ) -> None:
        self.reorder_delay_ns = int(reorder_delay_s * 1e9)
        self.imu_label = imu_label
        self.imu_decimate = max(1, imu_decimate)
        self.calib_timeout_s = calib_timeout_s
        self._queue: queue.SimpleQueue[Sample] = queue.SimpleQueue()
        self._calib: Any = None
        self._calib_ready = threading.Event()
        self._closed = threading.Event()
        self._close_lock = threading.Lock()
        self._counts: dict[str, int] = defaultdict(int)
        self._first_t: dict[str, int] = {}
        self._last_t: dict[str, int] = {}
        self._imu_seen = 0
        self._last_arrival = time.monotonic()
        self.dropped_rgb = 0
        self._reorder = Reorderer(self.reorder_delay_ns)

    # ---- callbacks (SDK threads) ------------------------------------------
    def _put(self, kind: str, sample: Sample | None) -> None:
        if sample is None or self._closed.is_set():
            return
        self._counts[kind] += 1
        self._last_arrival = time.monotonic()
        self._first_t.setdefault(kind, sample.t_ns)
        self._last_t[kind] = sample.t_ns
        self._queue.put(sample)

    def on_calib(self, device_calibration: Any) -> None:
        if device_calibration is not None:
            self._calib = device_calibration
            self._calib_ready.set()

    def on_rgb(self, image_data: Any, image_record: Any) -> None:
        self._put("rgb", convert.rgb_frame(image_data, int(image_record.capture_timestamp_ns)))

    def on_eye_gaze(self, eye_gaze: Any) -> None:
        self._put("eye_gaze", convert.eye_gaze_sample(eye_gaze, convert.eye_gaze_time_ns(eye_gaze)))

    def on_ppg(self, ppg: Any) -> None:
        self._put("ppg", convert.ppg_sample(ppg, int(ppg.capture_timestamp_ns)))

    def on_imu(self, imu: Any, label: str = "imu") -> None:
        # Batches arrive as lists on some transports; single samples on others.
        for m in imu if isinstance(imu, list) else [imu]:
            self.imu_label = self.imu_label or label
            if label != self.imu_label:
                continue
            self._imu_seen += 1
            if self._imu_seen % self.imu_decimate == 0:
                self._put("imu", convert.imu_sample(m, label, int(m.capture_timestamp_ns)))

    def register(self, target: Any) -> None:
        """Register our callbacks on a StreamReceiver / StreamDataInterface."""
        target.register_device_calib_callback(self.on_calib)
        target.register_rgb_callback(self.on_rgb)
        target.register_eye_gaze_callback(self.on_eye_gaze)
        target.register_ppg_callback(self.on_ppg)
        target.register_imu_callback(self.on_imu)

    # ---- consumer side (pipeline thread) -----------------------------------
    def device_calibration(self) -> Any:
        if not self._calib_ready.wait(self.calib_timeout_s):
            raise TimeoutError(
                f"No device calibration received within {self.calib_timeout_s:.0f} s: "
                "is the device streaming?"
            )
        return self._calib

    def finished(self) -> bool:
        """True when no more data will come (replay ended); live sources only end on close()."""
        return self._closed.is_set()

    def samples(self) -> Iterator[Sample]:
        while True:
            try:
                self._reorder.push(self._queue.get(timeout=0.05))
                while True:  # drain everything already queued
                    self._reorder.push(self._queue.get_nowait())
            except queue.Empty:
                pass
            done = self.finished() and self._queue.empty()
            ready = list(self._reorder.pop_ready(flush=done))
            # Several frames ready at once means we fell behind: keep only the newest.
            last_rgb = max((i for i, s in enumerate(ready) if isinstance(s, RgbFrame)), default=-1)
            for i, sample in enumerate(ready):
                if isinstance(sample, RgbFrame) and i != last_rgb:
                    self.dropped_rgb += 1
                    continue
                yield sample
            if done:
                return

    def flush(self) -> None:
        """Discard everything queued (e.g. after a pause) so processing resumes in the present."""
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self._reorder = Reorderer(self.reorder_delay_ns)

    def stats(self) -> StreamStats:
        rates = {}
        for kind, n in self._counts.items():
            span = (self._last_t[kind] - self._first_t[kind]) / 1e9
            rates[kind] = (n - 1) / span if span > 0 else 0.0
        idle = time.monotonic() - self._last_arrival
        return StreamStats(dict(self._counts), rates, self.dropped_rgb, self._reorder.late, idle)

    def latency_ms(self, t_ns: int) -> float | None:
        return None

    def close(self) -> None:
        self._closed.set()


class SdkReplaySource(StreamSource):
    """Replays a VRS through the SDK streaming callbacks (tests the live path offline)."""

    def __init__(self, path: Path, speed: float = 1.0, **kw: Any) -> None:
        super().__init__(**kw)
        import platform

        import aria.sdk_gen2 as sdk_gen2
        from aria.image_decoder import ImageDecoder

        # Unlike the live StreamReceiver, the raw StreamDataInterface hands over
        # H.265 access units: decode RGB ourselves, as the receiver does.
        self._decoder = ImageDecoder(hw_accel=platform.system() == "Darwin")
        self._stream_data = sdk_gen2.StreamDataInterface(enable_image_decoding=True)
        self.register(self._stream_data)
        config = sdk_gen2.ReplaySourceConfig()
        config.filename = str(path)
        config.speed_multiplier = speed
        self._session = sdk_gen2.create_replay_source(config, self._stream_data)
        if self._session is None:
            raise ValueError(f"Could not replay {path} through the SDK")

    def on_rgb(self, image_data: Any, image_record: Any) -> None:
        if self._decoder.decode_image(image_data, image_record):
            super().on_rgb(image_data, image_record)

    def finished(self) -> bool:
        return super().finished() or self._session.is_finished()

    def close(self) -> None:
        super().close()
        if not self._session.is_finished():
            self._session.stop()
        self._decoder.cleanup()


class LiveSource(StreamSource):
    """Streams from the glasses through the SDK's HTTP stream receiver.

    With `control_device=True` it connects to the glasses over USB, configures
    streaming and starts it; otherwise it only listens (start the stream with
    `aria_gen2 streaming start`). USB is needed to start/stop streaming, even
    when the data flows over Wi-Fi.
    """

    def __init__(
        self,
        interface: str = "usb",
        profile: str = "mp_streaming_demo",
        serial: str | None = None,
        control_device: bool = True,
        record_to: Path | None = None,
        port: int = RECEIVER_PORT,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        import aria.sdk_gen2 as sdk_gen2
        import aria.stream_receiver as receiver

        if interface not in STREAMING_INTERFACES:
            raise ValueError(f"interface must be one of {STREAMING_INTERFACES}")
        self._device = None
        if control_device:
            # Keep the client alive for the whole session: if it is garbage-collected the
            # device handle loses its session and stop_streaming() fails ("state: 1").
            self._client = sdk_gen2.DeviceClient()
            self._client.set_client_config(sdk_gen2.DeviceClientConfig())
            # Without adb, only the default target or an IP works (a serial needs adb).
            target = sdk_gen2.DeviceTarget(serial=serial) if serial else None
            self._device = self._client.connect(target) if target else self._client.connect()
            cfg = sdk_gen2.HttpStreamingConfig()
            cfg.profile_name = profile
            cfg.streaming_interface = {
                "usb": sdk_gen2.StreamingInterface.USB_NCM,
                "wifi_sta": sdk_gen2.StreamingInterface.WIFI_STA,
                "wifi_sap": sdk_gen2.StreamingInterface.WIFI_SAP,
            }[interface]
            if interface != "usb":
                # Keep streaming when the USB cable (needed to start) is pulled out.
                # While the cable is in, the glasses stream over USB whatever interface
                # is requested (seen on hermes: 16 MB over USB, 0.1 MB over Wi-Fi).
                cfg.keep_streaming_on_disconnection = True
            self._device.set_streaming_config(cfg)
            # Start the device *before* the receiver, as the SDK sample does: starting
            # streaming (re)writes the streaming certificates the receiver then loads.
            # The other order aborted the process ("couldn't read cert file").
            self._device.start_streaming()

        server_config = sdk_gen2.HttpServerConfig()
        server_config.address = "0.0.0.0"
        server_config.port = port
        self._receiver = receiver.StreamReceiver(enable_image_decoding=True)
        self._receiver.set_server_config(server_config)
        if record_to is not None:
            record_to.parent.mkdir(parents=True, exist_ok=True)
            self._receiver.record_to_vrs(str(record_to))
        self.register(self._receiver)
        self._receiver.start_server()
        self._started = time.monotonic()

    def latency_ms(self, t_ns: int) -> float | None:
        """Capture-to-host latency via TicSync, once the clock sync is valid."""
        sync = self._receiver.get_time_sync()
        if sync is None or not sync.is_valid():
            return None
        ms = sync.compute_latency_ms(t_ns)
        return ms if 0 <= ms < 10_000 else None

    def close(self) -> None:
        # Called by the pipeline and, on server shutdown, from another thread: run once.
        with self._close_lock:
            if self.finished():
                return
            super().close()
            try:
                if self._device is not None:
                    self._device.stop_streaming()
            except RuntimeError as exc:
                # Typically the USB cable was pulled after a Wi-Fi start: the glasses
                # keep streaming until stopped over USB.
                print(
                    f"[LiveSource] Could not stop streaming on the glasses ({exc}). "
                    "Plug in USB and run: uv run aria_gen2 streaming stop"
                )
            finally:
                time.sleep(0.5)  # let in-flight messages land (and the VRS recorder flush)
                self._receiver.stop_server()
