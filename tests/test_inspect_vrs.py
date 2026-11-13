from types import SimpleNamespace

from ariasense.inspect_vrs import public_fields, render_markdown, summarize


class FakeVergence:
    left_blink = True
    right_blink = False


class FakeEyeGaze:
    yaw = 0.1
    pitch = -0.2
    vergence = FakeVergence()

    def method(self):  # callables must not be reported as fields
        return None


class FakeSensorData:
    def eye_gaze_data(self):
        return FakeEyeGaze()


class FakeProvider:
    """Two streams: 30 Hz RGB over 2 s, and an empty eye-gaze stream."""

    streams = {
        "214-1": dict(label="camera-rgb", type="IMAGE", n=61, t0=0, t1=2_000_000_000, hz=30.0),
        "373-1": dict(label="eyegaze", type="EYE_GAZE", n=0, t0=0, t1=0, hz=60.0),
        "999-1": dict(label="gaze-live", type="EYE_GAZE", n=11, t0=0, t1=100_000_000, hz=90.0),
    }

    def get_all_streams(self):
        return list(self.streams)

    def get_label_from_stream_id(self, sid):
        return self.streams[sid]["label"]

    def get_sensor_data_type(self, sid):
        return SimpleNamespace(name=self.streams[sid]["type"])

    def get_num_data(self, sid):
        return self.streams[sid]["n"]

    def get_nominal_rate_hz(self, sid):
        return self.streams[sid]["hz"]

    def get_first_time_ns(self, sid, _domain):
        return self.streams[sid]["t0"]

    def get_last_time_ns(self, sid, _domain):
        return self.streams[sid]["t1"]

    def get_sensor_data_by_index(self, sid, _index):
        return FakeSensorData()


def test_public_fields_skips_private_and_callables():
    assert public_fields(FakeEyeGaze()) == ["pitch", "vergence", "yaw"]


def test_summarize_rates_and_fields():
    by_label = {s.label: s for s in summarize(FakeProvider(), time_domain=None)}

    rgb = by_label["camera-rgb"]
    assert rgb.measured_rate_hz == 30.0
    assert rgb.duration_s == 2.0
    assert rgb.fields == []  # images are not probed

    assert by_label["eyegaze"].measured_rate_hz == 0.0  # empty stream does not crash
    assert by_label["eyegaze"].fields == []

    live = by_label["gaze-live"]
    assert live.measured_rate_hz == 100.0
    assert "vergence.left_blink" in live.fields


def test_render_markdown_contains_table_and_fields():
    md = render_markdown(summarize(FakeProvider(), time_domain=None), "s.vrs", "Gen2")
    assert "| camera-rgb | IMAGE | `214-1` | 61 | 30.0 | 30.0 | 2.0 |" in md
    assert "`vergence.right_blink`" in md
