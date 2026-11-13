"""End-to-end pipeline and server tests with a fake source and detector (no SDK, no YOLO)."""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ariasense.dashboard.server import Hub, create_app
from ariasense.perception.attention import AttentionState
from ariasense.pipeline import DemoPipeline, dwell_pseudo_z, encode_jpeg
from ariasense.sources.base import Pacer
from ariasense.types import Detection, EyeGazeSample, PpgSample, RgbFrame
from tests.fakes import FakeCalibration

SEC = 1_000_000_000
W, H = 64, 48
CUP = Detection("cup", 0.9, (10, 10, 50, 40), track_id=1)


class FakeSource:
    """6 s of PPG (128 Hz), gaze (30 Hz) and RGB (10 Hz)."""

    def __init__(self, seconds: float = 6.0):
        self.seconds = seconds
        self.pacer = Pacer(0)

    def device_calibration(self):
        return FakeCalibration()

    def duration_s(self):
        return self.seconds

    def samples(self):
        out = []
        t = np.arange(0, self.seconds, 1 / 128)
        out += [PpgSample(int(ti * SEC), 166_000 + 800 * np.sin(2 * np.pi * 1.2 * ti)) for ti in t]
        out += [
            EyeGazeSample(int(ti * SEC), 0.0, 0.0, None, True, False, False)
            for ti in np.arange(0, self.seconds, 1 / 30)
        ]
        out += [
            RgbFrame(int(ti * SEC), np.zeros((H, W, 3), np.uint8)) for ti in np.arange(0, self.seconds, 0.1)
        ]
        yield from sorted(out, key=lambda s: s.t_ns)


def make_pipeline(**kw) -> DemoPipeline:
    return DemoPipeline(
        lambda: FakeSource(), lambda img: [CUP], baseline_s=2.0, encode=lambda img: None, **kw
    )


def test_dwell_pseudo_z():
    assert dwell_pseudo_z(None) is None
    assert dwell_pseudo_z(AttentionState(None, 0.0)) is None
    assert dwell_pseudo_z(AttentionState(CUP, 1.0)) == 0.0
    assert dwell_pseudo_z(AttentionState(CUP, 100.0)) == 2.5


def test_pipeline_emits_frames_features_state_and_baseline():
    msgs = []
    make_pipeline().run(msgs.append)
    kinds = [m["type"] for m in msgs]
    assert kinds.count("frame") == 60
    assert kinds.count("baseline") == 1
    assert kinds.count("features") == kinds.count("state") >= 4
    assert msgs[-1] == {"type": "status", "ended": True}
    assert {"type": "status", "duration": 6.0} in msgs  # the timeline can span the file up front

    frame = next(m for m in msgs if m["type"] == "frame" and m["attended"])
    assert frame["gaze"] == [0.5, 0.5]  # straight gaze -> image centre
    assert frame["attended"]["cls"] == "cup" and frame["attended"]["index"] == 0
    assert frame["detections"][0]["box"] == [
        round(10 / W, 4),
        round(10 / H, 4),
        round(50 / W, 4),
        round(40 / H, 4),
    ]

    states = [m for m in msgs if m["type"] == "state"]
    assert states[0]["state"] == "Calibrating" and states[0]["baseline_progress"] < 1
    assert states[-1]["state"] != "Calibrating" and states[-1]["baseline_progress"] == 1
    assert states[-1]["attention_totals"]["cup"] > 1


def test_rebaseline_restarts_calibration():
    p = make_pipeline()
    msgs = []

    def emit(m):
        msgs.append(m)
        if m["type"] == "baseline":
            p.controls.rebaseline = True  # ask for a new baseline right after the first one

    p.run(emit)
    states = [m["state"] for m in msgs if m["type"] == "state"]
    first_ready = states.index(next(s for s in states if s != "Calibrating"))
    assert "Calibrating" in states[first_ready:]


def test_loop_restarts_source_until_stopped():
    p = make_pipeline(loop=True)
    loops = []

    def emit(m):
        if m["type"] == "status" and "loop" in m:
            loops.append(m["loop"])
            if m["loop"] == 2:
                p.controls.stop.set()

    p.run(emit)
    assert loops == [0, 1, 2]


def test_hub_drops_frames_for_slow_clients_and_keeps_history():
    import asyncio

    hub = Hub()
    q: asyncio.Queue = asyncio.Queue()
    hub.clients.add(q)
    for i in range(10):
        hub._fanout({"type": "frame", "i": i, "t": i / 10, "jpeg": None})
    assert q.qsize() == 3  # MAX_QUEUED_FRAMES + 1
    hub._fanout({"type": "state", "t": 1})
    hub._fanout({"type": "status", "paused": True})
    hub._fanout({"type": "status", "speed": 2})
    snap = hub.snapshot()
    assert snap[0] == {"type": "status", "paused": True, "speed": 2}
    assert {"type": "state", "t": 1} in snap and snap[-1]["type"] == "frame"
    assert len(hub.recording) == 10  # every frame is recorded, even those a slow client skipped

    hub._fanout({"type": "status", "loop": 1})  # replay restarted: old history is dropped
    assert not any(m["type"] == "state" for m in hub.snapshot())
    assert len(hub.recording) == 0


def test_server_serves_ui_streams_messages_and_controls():
    p = make_pipeline(loop=True)
    app = create_app(p, info={"source": "fake.vrs"})
    with TestClient(app) as client:
        assert "AriaSense" in client.get("/").text
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/api/info").json()["source"] == "fake.vrs"

        with client.websocket_connect("/ws") as ws:
            kinds = set()
            for _ in range(2000):  # the snapshot replays history first; frames follow
                kinds.add(ws.receive_json()["type"])
                if {"frame", "state"} <= kinds:
                    break
        assert {"frame", "state"} <= kinds

        assert client.post("/api/control", json={"action": "pause"}).json() == {"ok": True}
        assert p.controls.paused
        client.post("/api/control", json={"action": "speed", "value": 2})
        assert p.controls.speed == 2
        assert client.post("/api/control", json={"action": "explode"}).json()["ok"] is False
        client.post("/api/control", json={"action": "resume"})
        time.sleep(0.1)
    assert p.controls.stop.is_set()  # lifespan shutdown stops the pipeline


def _frame(t: float, value: int = 128) -> dict:
    return {"type": "frame", "t": t, "jpeg": encode_jpeg(np.full((H, W, 3), value, np.uint8)), "gaze": None}


def test_recording_returns_the_frame_at_or_before_a_time_and_thumbnails():
    import base64

    from ariasense.dashboard.recording import Recording

    rec = Recording()
    assert rec.frame(1.0) is None and rec.thumbnail(1.0) is None
    for t in (0.0, 0.1, 0.2, 5.0):
        rec.add(_frame(t))
    assert len(rec) == 4 and rec.duration == 5.0
    assert rec.frame(0.15)["t"] == 0.1
    assert rec.frame(99)["t"] == 5.0 and rec.frame(-3)["t"] == 0.0  # clamped to what exists
    full = rec.frame(0.2)
    assert full["gaze"] is None and base64.b64decode(full["jpeg"])[:2] == b"\xff\xd8"
    thumb = rec.thumbnail(0.2, width=32)
    assert thumb is not None and thumb[:2] == b"\xff\xd8"

    rec.add(_frame(0.0))  # time went backwards: a new run replaces the old recording
    assert len(rec) == 1
    rec.clear()
    assert rec.frame(0) is None and rec.duration == 0
    rec.add({"type": "frame", "t": 1.0, "jpeg": None})  # no JPEG (e.g. headless): metadata only
    assert rec.frame(1.0)["jpeg"] is None and rec.thumbnail(1.0) is None
    rec.close()


def test_server_serves_recorded_frames_for_the_timeline():
    p = make_pipeline()
    p.encode = lambda img: encode_jpeg(img)
    with TestClient(create_app(p)) as client:
        for _ in range(100):
            if client.get("/api/frame", params={"t": 5}).status_code == 200:
                break
            time.sleep(0.05)
        r = client.get("/api/frame", params={"t": 0.0})
        assert r.status_code == 200 and r.json()["type"] == "frame" and r.json()["jpeg"]
        thumb = client.get("/api/thumb", params={"t": 1.0, "w": 32})
        assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"


@pytest.mark.parametrize("width", [64, 2000])
def test_encode_jpeg_downscales(width):
    import base64

    import cv2

    img = np.full((width // 2, width, 3), 128, np.uint8)
    data = base64.b64decode(encode_jpeg(img, width=1280))
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == min(width, 1280)


class UnwornSource(FakeSource):
    """Glasses lying on a desk: gaze invalid, PPG flat, video still streaming."""

    def samples(self):
        for s in super().samples():
            if isinstance(s, EyeGazeSample):
                yield EyeGazeSample(s.t_ns, 0.0, 0.0, None, False, None, None)
            elif isinstance(s, PpgSample):
                yield PpgSample(s.t_ns, 166_000.0)
            else:
                yield s


def test_pipeline_reports_no_signal_and_does_not_calibrate_when_not_worn():
    msgs = []
    DemoPipeline(lambda: UnwornSource(), lambda img: [CUP], baseline_s=2.0, encode=lambda img: None).run(
        msgs.append
    )
    states = [m for m in msgs if m["type"] == "state"]
    assert states and all(m["state"] == "No signal" for m in states)
    assert "no pulse on the nose pad" in states[-1]["reasons"]
    assert not any(m["type"] == "baseline" for m in msgs)  # baseline never built from unworn data
    assert states[-1]["baseline_progress"] == 0


def test_empty_dashboard_opens_a_recording_from_the_ui(tmp_path):
    from ariasense.dashboard.server import find_recordings

    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.vrs").write_bytes(b"x" * 10)
    (tmp_path / "notes.txt").write_text("not a recording")
    opened = []

    def opener(path, live):
        opened.append((path, live))
        return make_pipeline(), {"source": path.name, "mode": "replay", "live": live}

    app = create_app(None, opener=opener, data_dir=tmp_path)
    with TestClient(app) as client:
        info = client.get("/api/info").json()
        assert info["can_open"] and "source" not in info and info["baseline_s"] is None
        assert client.post("/api/control", json={"action": "pause"}).json()["ok"] is False

        files = client.get("/api/files").json()["files"]
        assert [f["name"] for f in files] == ["sub/a.vrs"]
        assert client.post("/api/open", json={"path": str(tmp_path / "notes.txt")}).json()["ok"] is False

        r = client.post("/api/open", json={"path": files[0]["path"]}).json()
        assert r == {"ok": True, "source": "a.vrs"} and opened == [(tmp_path / "sub" / "a.vrs", True)]
        assert client.get("/api/info").json()["source"] == "a.vrs"
        hub = app.state.hub
        assert hub.session == 1 and hub.latest["status"]["session"] == 1

        # Replaces the running pipeline; live=False analyses the whole file instead.
        client.post("/api/open", json={"path": files[0]["path"], "live": False})
        assert hub.session == 2 and opened[-1][1] is False
        assert client.get("/api/info").json()["live"] is False
    assert find_recordings(tmp_path / "missing") == []


def test_dashboard_without_opener_cannot_open_files():
    with TestClient(create_app(make_pipeline())) as client:
        assert client.get("/api/info").json()["can_open"] is False
        assert client.post("/api/open", json={"path": "x.vrs"}).json()["ok"] is False
