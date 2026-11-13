"""FastAPI app: serves the web UI, streams pipeline messages over a WebSocket, takes controls."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import traceback
from collections import deque
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ariasense.dashboard.recording import Recording
from ariasense.pipeline import DemoPipeline, Message

STATIC_DIR = Path(__file__).parent / "static"
HISTORY = 6 * 3600  # seconds of features/state replayed to a newly connected browser
MAX_QUEUED_FRAMES = 2  # a slow client skips frames instead of lagging behind


class Hub:
    """Fans messages out from the pipeline thread to every connected browser."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[asyncio.Queue[Message]] = set()
        self.history: deque[Message] = deque(maxlen=2 * HISTORY)
        self.latest: dict[str, Message] = {}
        self.recording = Recording()  # every frame, for the timeline scrubber
        self.session = 0  # bumped each time a new recording is opened

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
                self.recording.clear()
            self.latest["status"] = {**self.latest.get("status", {}), **msg}
        else:
            self.latest[kind] = msg
            if kind == "frame":
                self.recording.add(msg)
        for q in self.clients:
            if kind == "frame" and q.qsize() > MAX_QUEUED_FRAMES:
                continue
            q.put_nowait(msg)

    def reset(self, status: Message) -> None:
        """A new recording was opened: forget everything and tell the browsers to start afresh."""
        self.session += 1
        self.history.clear()
        self.latest.clear()
        self.recording.clear()
        self._fanout({"type": "status", "session": self.session, **status})

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


class OpenFile(BaseModel):
    path: str
    live: bool = True  # False: analyse the whole file as fast as possible, then review it


# Builds a pipeline for a recording picked in the UI, played live or analysed in one go:
# (pipeline, info shown by /api/info).
Opener = Callable[[Path, bool], tuple[DemoPipeline, dict[str, Any]]]


class Runner:
    """Runs one pipeline at a time on its own thread; a new file replaces the current one."""

    def __init__(self, hub: Hub) -> None:
        self.hub = hub
        self.pipeline: DemoPipeline | None = None
        self.info: dict[str, Any] = {}
        self.thread: threading.Thread | None = None

    def start(self, pipeline: DemoPipeline, info: dict[str, Any]) -> None:
        self.pipeline, self.info = pipeline, info

        def run() -> None:
            try:
                pipeline.run(self.hub.emit)
            except Exception as exc:  # surface crashes in the UI instead of dying silently
                traceback.print_exc()
                self.hub.emit({"type": "status", "error": f"{type(exc).__name__}: {exc}", "ended": True})

        self.thread = threading.Thread(target=run, name="aria-pipeline", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.shutdown()  # also stops the glasses' stream in live mode
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.thread = None


def find_recordings(root: Path) -> list[dict[str, Any]]:
    """VRS files under `root`, newest first."""
    if not root.is_dir():
        return []
    files = sorted(root.rglob("*.vrs"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {"path": str(p), "name": str(p.relative_to(root)), "size_mb": round(p.stat().st_size / 1e6, 1)}
        for p in files
    ]


def create_app(
    pipeline: DemoPipeline | None,
    info: dict[str, Any] | None = None,
    opener: Opener | None = None,
    data_dir: Path = Path("data"),
) -> FastAPI:
    """`pipeline` None starts an empty dashboard; `opener` lets the UI open recordings."""
    hub = Hub()
    runner = Runner(hub)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        hub.loop = asyncio.get_running_loop()
        if pipeline is not None:
            runner.start(pipeline, info or {})
        yield
        runner.stop()
        hub.recording.close()

    app = FastAPI(title="Aria Gen 2 Demo", lifespan=lifespan)
    app.state.hub = hub
    app.state.runner = runner
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/info")
    async def get_info() -> dict[str, Any]:
        p = runner.pipeline
        return {
            **runner.info,
            "can_open": opener is not None,
            "baseline_s": p.baseline_s if p else None,
            "loop": p.loop if p else False,
            "paused": p.controls.paused if p else False,
        }

    @app.get("/api/files")
    async def list_files() -> dict[str, Any]:
        return {"dir": str(data_dir), "files": find_recordings(data_dir)}

    @app.post("/api/open")
    async def open_file(f: OpenFile) -> dict[str, Any]:
        if opener is None:
            return {"ok": False, "error": "opening files is not available here"}
        path = Path(f.path).expanduser()
        if not path.is_file() or path.suffix.lower() != ".vrs":
            return {"ok": False, "error": f"not a VRS file: {path}"}
        try:
            new_pipeline, new_info = opener(path, f.live)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        await asyncio.to_thread(runner.stop)
        hub.reset({"source": new_info.get("source", path.name), "live": new_info.get("live", True)})
        runner.start(new_pipeline, new_info)
        return {"ok": True, "source": new_info.get("source", path.name)}

    @app.get("/api/frame")
    def get_frame(t: float) -> Response:
        """The recorded camera frame (JPEG, gaze, detections) at or just before time `t`."""
        msg = hub.recording.frame(t)
        if msg is None:
            return Response(status_code=404)
        return JSONResponse(msg)

    @app.get("/api/thumb")
    def get_thumb(t: float, w: int = 160) -> Response:
        data = hub.recording.thumbnail(t, min(max(w, 16), 640))
        if data is None:
            return Response(status_code=404)
        return Response(data, media_type="image/jpeg")

    @app.post("/api/control")
    async def control(c: Control) -> dict[str, Any]:
        if runner.pipeline is None:
            return {"ok": False, "error": "no recording open"}
        ctl = runner.pipeline.controls
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
