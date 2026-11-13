"""Small helpers to align streams that arrive at different rates."""

from __future__ import annotations

from collections import deque
from typing import Protocol


class _Timed(Protocol):
    t_ns: int


class History[T: _Timed]:
    """Keeps the samples of the last `window_ns` and looks them up by time."""

    def __init__(self, window_ns: int) -> None:
        self.window_ns = window_ns
        self._items: deque[T] = deque()

    def add(self, item: T) -> None:
        self._items.append(item)
        while self._items and item.t_ns - self._items[0].t_ns > self.window_ns:
            self._items.popleft()

    def latest_before(self, t_ns: int, max_age_ns: int) -> T | None:
        """Most recent sample at or before `t_ns`, if it is at most `max_age_ns` old."""
        for item in reversed(self._items):
            if item.t_ns <= t_ns:
                return item if t_ns - item.t_ns <= max_age_ns else None
        return None

    def since(self, t_ns: int) -> list[T]:
        """Samples with timestamp >= `t_ns`, oldest first."""
        return [item for item in self._items if item.t_ns >= t_ns]

    def __len__(self) -> int:
        return len(self._items)
