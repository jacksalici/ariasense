import pytest

from ariasense.perception.attention import AttentionTracker, box_distance, iou, pick_attended
from ariasense.types import Detection

MS = 1_000_000


def det(name, box, track_id=None, conf=0.9):
    return Detection(class_name=name, confidence=conf, box=box, track_id=track_id)


TABLE = det("dining table", (0, 0, 1000, 800), 1)
CUP = det("cup", (100, 100, 200, 220), 2)
LAPTOP = det("laptop", (400, 100, 700, 400), 3)


def test_box_distance_and_iou():
    assert box_distance((150, 150), CUP.box) == 0
    assert box_distance((210, 100), CUP.box) == pytest.approx(10)
    assert box_distance((203, 224), CUP.box) == pytest.approx(5)
    assert iou(CUP.box, CUP.box) == pytest.approx(1)
    assert iou(CUP.box, LAPTOP.box) == 0


def test_pick_smallest_containing_box():
    assert pick_attended([TABLE, CUP, LAPTOP], (150, 150), 20) is CUP
    assert pick_attended([TABLE, CUP, LAPTOP], (300, 600), 20) is TABLE


def test_pick_snaps_to_nearby_box_only_within_radius():
    assert pick_attended([CUP, LAPTOP], (210, 150), 20) is CUP
    assert pick_attended([CUP, LAPTOP], (300, 600), 20) is None
    assert pick_attended([CUP], None, 20) is None
    assert pick_attended([], (1, 1), 20) is None


def test_dwell_accumulates_on_same_track_and_resets_on_switch():
    tr = AttentionTracker(grace_s=0.3)
    assert tr.update(0, CUP).dwell_s == 0
    moved_cup = det("cup", (110, 105, 210, 225), 2)  # same track id, box moved
    s = tr.update(100 * MS, moved_cup)
    assert s.dwell_s == pytest.approx(0.1) and s.target is moved_cup
    s = tr.update(200 * MS, LAPTOP)
    assert s.target is LAPTOP and s.dwell_s == 0 and s.switches == 2
    assert s.totals_s == {"cup": pytest.approx(0.1)}


def test_short_gap_keeps_target_long_gap_drops_it():
    tr = AttentionTracker(grace_s=0.3)
    tr.update(0, CUP)
    tr.update(100 * MS, CUP)
    s = tr.update(300 * MS, None)  # 200 ms gap, within grace
    assert s.target is CUP and s.dwell_s == pytest.approx(0.1)
    s = tr.update(400 * MS, CUP)  # back on the cup: dwell continues
    assert s.dwell_s == pytest.approx(0.4) and s.switches == 1
    s = tr.update(800 * MS, None)  # 400 ms gap: dropped
    assert s.target is None and s.dwell_s == 0


def test_match_without_track_ids_uses_class_and_iou():
    tr = AttentionTracker()
    a = det("cup", (100, 100, 200, 200))
    b = det("cup", (105, 100, 205, 200))
    tr.update(0, a)
    assert tr.update(50 * MS, b).dwell_s == pytest.approx(0.05)
    c = det("cup", (500, 500, 600, 600))  # another cup elsewhere
    assert tr.update(100 * MS, c).dwell_s == 0
