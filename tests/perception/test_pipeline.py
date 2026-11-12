from types import SimpleNamespace

import numpy as np

from aria_demo.overlay import ATTENDED_COLOR, GazeFrame
from aria_demo.perception.detector import to_detections
from aria_demo.perception.pipeline import draw, perceive
from aria_demo.types import Detection

MS = 1_000_000
CUP = Detection("cup", 0.9, (10, 10, 40, 40), track_id=7)


def frame(t_ms, gaze_px, blinking=False):
    return GazeFrame(
        t_ns=t_ms * MS,
        t_rel_s=t_ms / 1000,
        image=np.zeros((100, 200, 3), np.uint8),
        gaze=None,
        gaze_px=gaze_px,
        trail=[],
        blinking=blinking,
    )


def test_perceive_attends_and_ignores_blinks():
    frames = [frame(0, (20, 20)), frame(100, (25, 25)), frame(200, (25, 25), blinking=True)]
    out = list(perceive(frames, lambda img: [CUP]))
    assert [pf.attention.target for pf in out[:2]] == [CUP, CUP]
    assert out[1].attention.dwell_s == 0.1
    # blinking frame: no new target, but kept within the grace period
    assert out[2].attention.target is CUP and out[2].attention.dwell_s == 0.1


def test_wearer_arm_is_ignored_but_other_people_are_kept():
    arm = Detection("person", 0.8, (0, 30, 200, 99), track_id=1)  # touches the bottom edge
    other = Detection("person", 0.8, (100, 10, 150, 60), track_id=2)
    (pf,) = perceive([frame(0, (20, 50))], lambda img: [arm, other])
    assert pf.detections == [other]
    assert pf.attention.target is None  # gaze is on the arm only

    (pf,) = perceive([frame(0, (20, 50))], lambda img: [arm], ignore_wearer_body=False)
    assert pf.attention.target is arm


def test_draw_highlights_attended_box():
    pf = next(perceive([frame(0, (20, 20))], lambda img: [CUP]))
    img = draw(pf)
    assert tuple(img[30, 40]) == ATTENDED_COLOR  # right edge of the cup box (below its label)


def test_to_detections_from_boxes_like_object():
    boxes = SimpleNamespace(
        xyxy=np.array([[1, 2, 3, 4.0]]), conf=np.array([0.8]), cls=np.array([41]), id=np.array([5])
    )
    (d,) = to_detections(boxes, {41: "cup"})
    assert d == Detection("cup", 0.8, (1.0, 2.0, 3.0, 4.0), track_id=5)
    no_ids = SimpleNamespace(xyxy=boxes.xyxy, conf=boxes.conf, cls=boxes.cls, id=None)
    assert to_detections(no_ids, {41: "cup"})[0].track_id is None
    assert to_detections(None, {}) == []
