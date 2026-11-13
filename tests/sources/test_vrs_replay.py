from types import SimpleNamespace

import numpy as np

from ariasense.sources.base import Pacer, Signal
from ariasense.sources.vrs_replay import select_streams, to_sample
from ariasense.types import EyeGazeSample, ImuSample, PpgSample, RgbFrame


class FakeProvider:
    streams = {
        "214-1": ("IMAGE", "camera-rgb"),
        "1201-1": ("IMAGE", "slam-front-left"),
        "373-1": ("EYE_GAZE", "eyegaze"),
        "1202-1": ("IMU", "imu-left"),
        "248-1": ("PPG", "ppg"),
        "500-1": ("AUDIO", "mic"),
    }

    def get_all_streams(self):
        return list(self.streams)

    def get_sensor_data_type(self, sid):
        return SimpleNamespace(name=self.streams[sid][0])

    def get_label_from_stream_id(self, sid):
        return self.streams[sid][1]


def test_select_streams_keeps_only_rgb_camera_and_wanted_signals():
    selected = select_streams(FakeProvider(), {Signal.RGB, Signal.EYE_GAZE, Signal.IMU})
    assert selected == {
        "214-1": (Signal.RGB, "camera-rgb"),
        "373-1": (Signal.EYE_GAZE, "eyegaze"),
        "1202-1": (Signal.IMU, "imu-left"),
    }


def _image_sensor_data(array, valid=True):
    image = SimpleNamespace(is_valid=lambda: valid, to_numpy_array=lambda: array)
    return SimpleNamespace(image_data_and_record=lambda: (image, None))


def test_rgb_conversion_and_grayscale_fallback():
    rgb = to_sample(_image_sensor_data(np.zeros((4, 6, 3), np.uint8)), Signal.RGB, "camera-rgb", 5)
    assert isinstance(rgb, RgbFrame) and rgb.t_ns == 5 and rgb.image.shape == (4, 6, 3)

    gray = to_sample(_image_sensor_data(np.zeros((4, 6), np.uint8)), Signal.RGB, "camera-rgb", 5)
    assert gray.image.shape == (4, 6, 3)

    assert to_sample(_image_sensor_data(None, valid=False), Signal.RGB, "camera-rgb", 5) is None


def test_eye_gaze_conversion():
    vergence = SimpleNamespace(
        left_blink=False, left_blink_valid=True, right_blink=False, right_blink_valid=False
    )
    gaze = SimpleNamespace(
        yaw=0.1, pitch=-0.2, depth=float("nan"), combined_gaze_valid=True, vergence=vergence
    )
    # SDK flag False = eye closed -> our left_blink True
    s = to_sample(SimpleNamespace(eye_gaze_data=lambda: gaze), Signal.EYE_GAZE, "eyegaze", 7)
    assert s == EyeGazeSample(
        t_ns=7, yaw=0.1, pitch=-0.2, depth_m=None, valid=True, left_blink=True, right_blink=None
    )


def test_ppg_and_imu_conversion():
    ppg = to_sample(SimpleNamespace(ppg_data=lambda: SimpleNamespace(value=123)), Signal.PPG, "ppg", 1)
    assert ppg == PpgSample(t_ns=1, value=123.0)

    motion = SimpleNamespace(accel_msec2=[0, 0, 9.81], gyro_radsec=[0.1, 0, 0])
    imu = to_sample(SimpleNamespace(imu_data=lambda: motion), Signal.IMU, "imu-left", 2)
    assert isinstance(imu, ImuSample) and imu.label == "imu-left"
    np.testing.assert_allclose(imu.accel, [0, 0, 9.81])


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.slept: list[float] = []

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.slept.append(s)
        self.now += s


def test_pacer_real_time_and_double_speed():
    clock = FakeClock()
    pacer = Pacer(1.0, clock=clock, sleep=clock.sleep)
    pacer.wait_until(0)
    pacer.wait_until(500_000_000)
    assert clock.slept == [0.5]

    clock = FakeClock()
    pacer = Pacer(2.0, clock=clock, sleep=clock.sleep)
    pacer.wait_until(0)
    pacer.wait_until(1_000_000_000)
    assert clock.slept == [0.5]


def test_pacer_disabled_and_no_sleep_when_late():
    clock = FakeClock()
    Pacer(0, clock=clock, sleep=clock.sleep).wait_until(10**12)
    assert clock.slept == []

    pacer = Pacer(1.0, clock=clock, sleep=clock.sleep)
    pacer.wait_until(0)
    clock.now += 2.0  # processing took longer than the gap
    pacer.wait_until(1_000_000_000)
    assert clock.slept == []
