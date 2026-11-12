"""FastAPI app: serves the web UI, streams pipeline messages over a WebSocket, takes controls."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import traceback
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aria_demo.pipeline import DemoPipeline, Message

STATIC_DIR = Path(__file__).parent / "static"
HISTORY = 300  # seconds of features/state replayed to a newly connected browser
MAX_QUEUED_FRAMES = 2  # a slow client skips frames instead of lagging behind


class Hub:
    """Fans messages out from the pipeline thread to every connected browser."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[asyncio.Queue[Message]] = set()
        self.history: deque[Message] = deque(maxlen=2 * HISTORY)
        self.latest: dict[str, Message] = {}

    def emit(self, msg: Message) -> None:
        """Thread-safe: called from the pipeline thread."""
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._fanout, msg)

    def _fanout(self, msg: Message) -> None:
        kind = msg["type"]
        if kind in ("features", "state"):
            self.history.append(msg)
        elif kind == "status":
            if "loop" in msg:  # replay restarted: time starts again from 0
                self.history.clear()
            self.latest["status"] = {**self.latest.get("status", {}), **msg}
        else:
            self.latest[kind] = msg
        for q in self.clients:
            if kind == "frame" and q.qsize() > MAX_QUEUED_FRAMES:
                continue
            q.put_nowait(msg)

    def snapshot(self) -> list[Message]:
        """What a new client needs to catch up: baseline, status, history, last frame."""
        out = [self.latest[k] for k in ("status", "baseline") if k in self.latest]
        out += list(self.history)
        if "frame" in self.latest:
            out.append(self.latest["frame"])
        return out


class Control(BaseModel):
    action: str  # pause | resume | rebaseline | speed
    value: float | None = None


def create_app(pipeline: DemoPipeline, info: dict[str, Any] | None = None) -> FastAPI:
    hub = Hub()

    def run_pipeline() -> None:
        try:
            pipeline.run(hub.emit)
        except Exception as exc:  # surface crashes in the UI instead of dying silently
            traceback.print_exc()
            hub.emit({"type": "status", "error": f"{type(exc).__name__}: {exc}", "ended": True})

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        hub.loop = asyncio.get_running_loop()
        thread = threading.Thread(target=run_pipeline, name="aria-pipeline", daemon=True)
        thread.start()
        yield
        pipeline.controls.stop.set()
        pipeline.controls.paused = False
        thread.join(timeout=5)

    app = FastAPI(title="Aria Gen 2 Demo", lifespan=lifespan)
    app.state.hub = hub
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/info")
    async def get_info() -> dict[str, Any]:
        return {
            **(info or {}),
            "baseline_s": pipeline.baseline_s,
            "loop": pipeline.loop,
            "paused": pipeline.controls.paused,
        }

    @app.post("/api/control")
    async def control(c: Control) -> dict[str, Any]:
        ctl = pipeline.controls
        if c.action == "pause":
            ctl.paused = True
        elif c.action == "resume":
            ctl.paused = False
        elif c.action == "rebaseline":
            ctl.rebaseline = True
        elif c.action == "speed" and c.value is not None and c.value > 0:
            ctl.speed = c.value
        else:
            return {"ok": False, "error": f"unknown action {c.action!r}"}
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        queue: asyncio.Queue[Message] = asyncio.Queue()
        for msg in hub.snapshot():
            queue.put_nowait(msg)
        hub.clients.add(queue)
        try:
            while True:
                await socket.send_json(await queue.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.clients.discard(queue)

    return app
