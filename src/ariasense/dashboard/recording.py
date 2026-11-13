"""Disk-backed record of every camera frame shown, so the browser can scrub back through it.

The JPEGs stay out of memory (about 1 MB/s at 10 Hz): they are appended to a temp file and
looked up by time. Only a small index (time, offset, sizes) lives in RAM.
"""

from __future__ import annotations

import base64
import bisect
import json
import os
import tempfile
import threading
from typing import Any

import cv2
import numpy as np

Message = dict[str, Any]


class Recording:
    def __init__(self) -> None:
        self._file = tempfile.TemporaryFile(prefix="ariasense-frames-")  # removed on close/exit
        self._fd = self._file.fileno()
        self._lock = threading.Lock()
        self._size = 0
        self._t: list[float] = []
        self._entries: list[tuple[int, int, int]] = []  # offset, meta bytes, jpeg bytes

    def close(self) -> None:
        self._file.close()

    @property
    def duration(self) -> float:
        with self._lock:
            return self._t[-1] if self._t else 0.0

    def __len__(self) -> int:
        return len(self._t)

    def clear(self) -> None:
        with self._lock:
            self._reset()

    def _reset(self) -> None:
        # The file is reused from the start (no truncate): cheap, and it never outgrows the longest run.
        self._size = 0
        self._t.clear()
        self._entries.clear()

    def add(self, msg: Message) -> None:
        """Store a `frame` message (its JPEG is moved out of the metadata into the file)."""
        b64 = msg.get("jpeg")
        jpeg = base64.b64decode(b64) if b64 else b""
        meta = json.dumps({k: v for k, v in msg.items() if k != "jpeg"}).encode()
        t = float(msg["t"])
        with self._lock:
            if self._t and t < self._t[-1] - 1:  # time went backwards: a new run started
                self._reset()
            os.pwrite(self._fd, meta + jpeg, self._size)
            self._entries.append((self._size, len(meta), len(jpeg)))
            self._t.append(max(t, self._t[-1]) if self._t else t)  # keep the index sorted
            self._size += len(meta) + len(jpeg)

    def _read(self, t: float) -> tuple[Message, bytes] | None:
        """The frame at or just before `t` (the first one if `t` precedes them all)."""
        with self._lock:
            if not self._t:
                return None
            i = max(0, bisect.bisect_right(self._t, t) - 1)
            offset, n_meta, n_jpeg = self._entries[i]
            raw = os.pread(self._fd, n_meta + n_jpeg, offset)
        return json.loads(raw[:n_meta]), raw[n_meta:]

    def frame(self, t: float) -> Message | None:
        """A full `frame` message, as the live stream would have sent it."""
        found = self._read(t)
        if found is None:
            return None
        meta, jpeg = found
        return {**meta, "jpeg": base64.b64encode(jpeg).decode("ascii") if jpeg else None}

    def thumbnail(self, t: float, width: int = 160) -> bytes | None:
        found = self._read(t)
        if found is None or not found[1]:
            return None
        image = cv2.imdecode(np.frombuffer(found[1], np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None
        h, w = image.shape[:2]
        if w > width:
            image = cv2.resize(image, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes() if ok else None
