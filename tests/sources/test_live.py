"""Streaming source logic with fake SDK objects (no device, no SDK threads)."""

import datetime
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from aria_demo.pipeline import DemoPipeline
from aria_demo.sources.live import Reorderer, StreamSource
from aria_demo.types import EyeGazeSample, ImuSample, PpgSample, RgbFrame
from tests.fakes import FakeCalibration

MS = 1_000_000


def ppg(t_ms):
    return PpgSample(t_ms * MS, 1.0)


def frame(t_ms):
    return RgbFrame(t_ms * MS, np.zeros((2, 2, 3), np.uint8))


# ---- Reorderer ----------------------------------------------------------------


def test_reorderer_releases_in_order_after_delay():
    r = Reorderer(delay_ns=100 * MS)
    for t in (0, 50, 30, 120):
        r.push(ppg(t))
    assert [s.t_ns // MS for s in r.pop_ready()] == [0]  # newest 120 -> release <= 20
    r.push(ppg(200))
    assert [s.t_ns // MS for s in r.pop_ready()] == [30, 50]
    assert [s.t_ns // MS for s in r.pop_ready(flush=True)] == [120, 200]


def test_reorderer_passes_batches_late_against_other_streams():
    """PPG/IMU arrive in batches up to ~1 s behind the other streams: keep them."""
    r = Reorderer(delay_ns=0)
    r.push(frame(100))
    list(r.pop_ready())
    r.push(ppg(50))  # behind the frame already released, but first of its own stream
    r.push(ppg(60))
    r.push(frame(60))  # older than a frame already shown: useless, dropped
    assert r.late == 1
    assert [(type(s).__name__, s.t_ns // MS) for s in r.pop_ready()] == [
        ("PpgSample", 50),
        ("PpgSample", 60),
    ]


def test_reorderer_drops_out_of_order_within_a_stream():
    r = Reorderer(delay_ns=0)
    r.push(ppg(100))
    list(r.pop_ready())
    r.push(ppg(50))  # older than what this stream already released: would break windows
    assert r.late == 1
    assert list(r.pop_ready()) == []


# ---- StreamSource callbacks ----------------------------------------------------------


def fake_image(valid=True):
    return SimpleNamespace(is_valid=lambda: valid, to_numpy_array=lambda: np.zeros((4, 6, 3), np.uint8))


def fake_gaze(seconds, blink_open=True):
    vergence = SimpleNamespace(
        left_blink=blink_open, left_blink_valid=True, right_blink=blink_open, right_blink_valid=True
    )
    return SimpleNamespace(
        tracking_timestamp=datetime.timedelta(seconds=seconds),
        yaw=0.1,
        pitch=0.0,
        depth=0.5,
        combined_gaze_valid=True,
        vergence=vergence,
    )


def motion(t_ms):
    return SimpleNamespace(capture_timestamp_ns=t_ms * MS, accel_msec2=[0, 0, 9.8], gyro_radsec=[0, 0, 0.1])


def drained(src: StreamSource):
    src.close()  # finished -> samples() flushes the reorder buffer and returns
    return list(src.samples())


def test_callbacks_convert_and_order_samples():
    src = StreamSource(reorder_delay_s=0.0, imu_decimate=1)
    src.on_ppg(SimpleNamespace(capture_timestamp_ns=30 * MS, value=5.0))
    src.on_eye_gaze(fake_gaze(0.010, blink_open=False))
    src.on_rgb(fake_image(), SimpleNamespace(capture_timestamp_ns=20 * MS))
    src.on_rgb(fake_image(valid=False), SimpleNamespace(capture_timestamp_ns=25 * MS))  # ignored
    out = drained(src)
    assert [s.t_ns // MS for s in out] == [10, 20, 30]
    gaze = out[0]
    assert isinstance(gaze, EyeGazeSample) and gaze.left_blink is True  # SDK flag inverted
    assert isinstance(out[1], RgbFrame) and isinstance(out[2], PpgSample)


def test_imu_uses_one_label_decimates_and_accepts_batches():
    src = StreamSource(reorder_delay_s=0.0, imu_decimate=2)
    src.on_imu(motion(1), "imu-right")
    src.on_imu(motion(2), "imu-left")  # other IMU ignored
    src.on_imu([motion(3), motion(4), motion(5)], "imu-right")  # batch
    imu = [s for s in drained(src) if isinstance(s, ImuSample)]
    assert [s.t_ns // MS for s in imu] == [3, 5]  # every 2nd of 1, 3, 4, 5... -> 3 and 5
    assert all(s.label == "imu-right" for s in imu)


def test_only_newest_frame_kept_when_behind():
    src = StreamSource(reorder_delay_s=0.0)
    for t in (100, 200, 300):
        src.on_rgb(fake_image(), SimpleNamespace(capture_timestamp_ns=t * MS))
    src.on_ppg(SimpleNamespace(capture_timestamp_ns=250 * MS, value=1.0))
    out = drained(src)
    frames = [s.t_ns // MS for s in out if isinstance(s, RgbFrame)]
    assert frames == [300]
    assert len([s for s in out if isinstance(s, PpgSample)]) == 1  # sensors never dropped
    assert src.stats().dropped_rgb == 2


def test_flush_stats_and_calibration_timeout():
    src = StreamSource(calib_timeout_s=0.05)
    with pytest.raises(TimeoutError, match="calibration"):
        src.device_calibration()
    src.on_calib("CALIB")
    assert src.device_calibration() == "CALIB"

    for t in range(0, 1001, 100):
        src.on_ppg(SimpleNamespace(capture_timestamp_ns=t * MS, value=1.0))
    stats = src.stats()
    assert stats.counts == {"ppg": 11} and stats.rates_hz["ppg"] == pytest.approx(10)
    src.flush()
    assert drained(src) == []


def test_register_wires_all_callbacks():
    registered = {}

    class Target:
        def __getattr__(self, name):
            return lambda cb: registered.__setitem__(name, cb)

    src = StreamSource()
    src.register(Target())
    assert set(registered) == {
        "register_device_calib_callback",
        "register_rgb_callback",
        "register_eye_gaze_callback",
        "register_ppg_callback",
        "register_imu_callback",
    }


# ---- pipeline integration --------------------------------------------------------------


class FakeLive(StreamSource):
    """A StreamSource fed by a thread, like the SDK would, with a fake calibration."""

    def __init__(self):
        super().__init__(reorder_delay_s=0.05)
        self.closed = False
        self.on_calib(FakeCalibration())
        threading.Thread(target=self._feed, daemon=True).start()

    def _feed(self):
        for i in range(30):  # 3 s of frames and gaze, no PPG
            self.on_rgb(fake_image(), SimpleNamespace(capture_timestamp_ns=i * 100 * MS))
            self.on_eye_gaze(fake_gaze(i * 0.1))
            time.sleep(0.005)
        # ...then the stream stalls, as a live stream can: only shutdown() ends it.

    def latency_ms(self, t_ns):
        return 123.0

    def close(self):
        self.closed = True
        super().close()


def test_pipeline_closes_live_source_reports_latency_and_missing_streams(monkeypatch):
    import aria_demo.pipeline as pl

    monkeypatch.setattr(pl, "HEALTH_EVERY_S", 0.0)
    source = FakeLive()
    p = DemoPipeline(lambda: source, lambda img: [], baseline_s=1.0, encode=lambda img: None)
    msgs = []

    runner = threading.Thread(target=p.run, args=(msgs.append,), daemon=True)
    runner.start()
    time.sleep(1.0)  # feeding is done after ~0.2 s; the stream is now stalled
    p.shutdown()
    runner.join(timeout=5)

    assert not runner.is_alive(), "pipeline must stop even when the stream stalls"
    assert source.closed
    frames = [m for m in msgs if m["type"] == "frame"]
    assert frames and all(m["latency_ms"] == 123 for m in frames)
    health = [m for m in msgs if m["type"] == "status" and "streams" in m]
    assert health and "eye_gaze" in health[-1]["streams"]
    assert health[-1]["idle_s"] >= 0.5  # reported although no samples arrive any more
    assert msgs[-1] == {"type": "status", "ended": True}
