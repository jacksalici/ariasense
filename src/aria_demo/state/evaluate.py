"""Compare estimated states with labelled protocol segments (see docs/protocol.md).

Labels CSV: `start_s,end_s,label`, with times relative to the recording start and a
label naming a state (e.g. Calm, Focused, Stressed, Startled, Distracted).

Every second inside a segment is scored, except Startled, which is an event:
its segment counts as a hit if a Startled estimate occurs anywhere inside it.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from aria_demo.state.estimator import State, StateEstimate


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float
    label: State


@dataclass
class Evaluation:
    confusion: Counter[tuple[State, State]] = field(default_factory=Counter)  # (expected, predicted)
    startle_hits: int = 0
    startle_segments: int = 0
    false_startles: int = 0  # startle onsets outside Startled segments

    @property
    def accuracy(self) -> float | None:
        total = sum(self.confusion.values())
        if not total:
            return None
        return sum(n for (e, p), n in self.confusion.items() if e == p) / total

    def recall(self, state: State) -> float | None:
        row = {p: n for (e, p), n in self.confusion.items() if e == state}
        total = sum(row.values())
        return row.get(state, 0) / total if total else None

    def table(self) -> str:
        expected = sorted({e for e, _ in self.confusion}, key=list(State).index)
        predicted = sorted({p for _, p in self.confusion}, key=list(State).index)
        w = max([len(s.value) for s in [*expected, *predicted]] + [8]) + 2
        lines = ["expected \\ predicted".ljust(w + 4) + "".join(p.value[:w].rjust(w) for p in predicted)]
        lines[0] += "recall".rjust(8)
        for e in expected:
            row = e.value.ljust(w + 4) + "".join(str(self.confusion[(e, p)]).rjust(w) for p in predicted)
            r = self.recall(e)
            lines.append(row + (f"{r:8.0%}" if r is not None else "-".rjust(8)))
        acc = self.accuracy
        lines.append(f"\nPer-second accuracy (excl. Startled): {acc:.0%}" if acc is not None else "")
        if self.startle_segments:
            lines.append(f"Startle segments detected: {self.startle_hits}/{self.startle_segments}")
        lines.append(f"Startle onsets outside Startled segments: {self.false_startles}")
        return "\n".join(lines)


def parse_state(name: str) -> State:
    key = name.strip().lower()
    for s in State:
        if key in (s.value.lower(), s.name.lower()):
            return s
    raise ValueError(f"Unknown state label '{name}'. Use one of: {', '.join(s.value for s in State)}")


def load_labels(path: Path) -> list[Segment]:
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("label", "").strip()]
    return [Segment(float(r["start_s"]), float(r["end_s"]), parse_state(r["label"])) for r in rows]


def evaluate(estimates: Sequence[StateEstimate], labels: Sequence[Segment], t0_ns: int) -> Evaluation:
    ev = Evaluation()
    times = [(e.t_ns - t0_ns) / 1e9 for e in estimates]

    def segment_at(t: float) -> Segment | None:
        return next((s for s in labels if s.start_s <= t < s.end_s), None)

    previous = None
    for t, est in zip(times, estimates, strict=True):
        seg = segment_at(t)
        onset = est.state is State.STARTLED and previous is not State.STARTLED
        previous = est.state
        if onset and (seg is None or seg.label is not State.STARTLED):
            ev.false_startles += 1
        if seg is None or seg.label is State.STARTLED or est.state is State.UNKNOWN:
            continue
        ev.confusion[(seg.label, est.state)] += 1

    for seg in labels:
        if seg.label is State.STARTLED:
            ev.startle_segments += 1
            ev.startle_hits += any(
                seg.start_s <= t < seg.end_s and e.state is State.STARTLED
                for t, e in zip(times, estimates, strict=True)
            )
    return ev
