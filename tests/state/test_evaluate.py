import pytest

from aria_demo.state.estimator import State, StateEstimate
from aria_demo.state.evaluate import Segment, evaluate, load_labels, parse_state

SEC = 1_000_000_000


def est(t_s, state):
    return StateEstimate(int(t_s * SEC), state, 0.8, [], 0.0, 0.0, state)


def test_parse_and_load_labels(tmp_path):
    assert parse_state(" focused ") is State.FOCUSED
    assert parse_state("STARTLED") is State.STARTLED
    with pytest.raises(ValueError, match="Unknown state"):
        parse_state("happy")
    p = tmp_path / "l.csv"
    p.write_text("start_s,end_s,label\n0,60,Calm\n60,150,Focused\n150,160,\n")
    assert load_labels(p) == [Segment(0, 60, State.CALM), Segment(60, 150, State.FOCUSED)]


def test_confusion_accuracy_and_startle():
    labels = [Segment(0, 4, State.CALM), Segment(4, 8, State.FOCUSED), Segment(8, 10, State.STARTLED)]
    states = [State.CALM, State.CALM, State.FOCUSED, State.CALM]  # calm segment: 3/4
    states += [State.FOCUSED] * 4  # focused segment: 4/4
    states += [State.NEUTRAL, State.STARTLED]  # startle hit
    states += [State.STARTLED]  # t=10, outside all segments, but same event: not a new onset
    ev = evaluate([est(i, s) for i, s in enumerate(states)], labels, t0_ns=0)

    assert ev.confusion[(State.CALM, State.CALM)] == 3
    assert ev.confusion[(State.CALM, State.FOCUSED)] == 1
    assert ev.recall(State.FOCUSED) == 1.0
    assert ev.accuracy == pytest.approx(7 / 8)
    assert (ev.startle_hits, ev.startle_segments, ev.false_startles) == (1, 1, 0)
    table = ev.table()
    assert "75%" in table and "Startle segments detected: 1/1" in table


def test_false_startle_and_calibrating_ignored():
    labels = [Segment(0, 5, State.CALM)]
    states = [State.UNKNOWN, State.CALM, State.STARTLED, State.STARTLED, State.CALM]
    ev = evaluate([est(i, s) for i, s in enumerate(states)], labels, t0_ns=0)
    assert ev.false_startles == 1
    assert sum(ev.confusion.values()) == 4  # the Calibrating second is not scored
