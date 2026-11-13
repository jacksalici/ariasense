import numpy as np

from ariasense.calibration.projection import GazeProjector
from ariasense.overlay import DOT_COLOR, OverlayStats, render, write_video
from ariasense.sync import History
from ariasense.types import EyeGazeSample, PpgSample, RgbFrame

W, H = 64, 48
MS = 1_000_000


def gaze(t_ms, valid=True, blink=False):
    return EyeGazeSample(
        t_ns=t_ms * MS, yaw=0.0, pitch=0.0, depth_m=1.0, valid=valid, left_blink=blink, right_blink=blink
    )


def frame(t_ms):
    return RgbFrame(t_ns=t_ms * MS, image=np.zeros((H, W, 3), np.uint8))


def center_projector():
    # Gaze lands on raw pixel (40, 5); after upright rotation -> (H-1-5, 40) = (42, 40),
    # below the status label in the top-left corner.
    return GazeProjector(np.eye(4), lambda p: np.array([40.0, 5.0]), (W, H), rotate_upright=True)


def test_history_lookup_and_window():
    h: History[EyeGazeSample] = History(window_ns=100 * MS)
    for t in (0, 50, 120, 200):
        h.add(gaze(t))
    assert len(h) == 2  # 0 and 50 fell out of the 100 ms window
    assert h.latest_before(130 * MS, 20 * MS).t_ns == 120 * MS
    assert h.latest_before(190 * MS, 20 * MS) is None  # too old
    assert h.latest_before(100 * MS, 1000 * MS) is None  # nothing at or before
    assert [g.t_ns for g in h.since(150 * MS)] == [200 * MS]


def test_render_draws_dot_at_projected_pixel_and_counts():
    samples = [gaze(0), frame(10), PpgSample(15 * MS, 1.0), gaze(20, valid=False), frame(30), frame(200)]
    stats = OverlayStats()
    out = list(render(samples, center_projector(), stats=stats))

    assert len(out) == 3
    t, img = out[0]
    assert t == 10 * MS and img.shape == (W, H, 3)  # upright
    assert tuple(img[40, 42]) == DOT_COLOR
    assert stats.frames == 3
    assert stats.frames_with_gaze == 2  # frame(200) has no gaze within 60 ms
    assert stats.frames_gaze_in_image == 1  # frame(30): latest gaze is invalid


def test_blink_counted():
    stats = OverlayStats()
    list(render([gaze(0, blink=True), frame(5)], center_projector(), stats=stats))
    assert stats.blinks == 1


def test_write_video(tmp_path):
    frames = [(i * 33 * MS, np.full((H, W, 3), 50, np.uint8)) for i in range(5)]
    out = tmp_path / "v.mp4"
    assert write_video(frames, out) == 5
    assert out.stat().st_size > 0
