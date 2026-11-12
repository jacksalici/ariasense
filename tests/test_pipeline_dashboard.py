"""End-to-end pipeline and server tests with a fake source and detector (no SDK, no YOLO)."""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from aria_demo.dashboard.server import Hub, create_app
from aria_demo.perception.attention import AttentionState
from aria_demo.pipeline import DemoPipeline, dwell_pseudo_z, encode_jpeg
from aria_demo.sources.base import Pacer
from aria_demo.types import Detection, EyeGazeSample, PpgSample, RgbFrame
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
        hub._fanout({"type": "frame", "i": i})
    assert q.qsize() == 3  # MAX_QUEUED_FRAMES + 1
    hub._fanout({"type": "state", "t": 1})
    hub._fanout({"type": "status", "paused": True})
    hub._fanout({"type": "status", "speed": 2})
    snap = hub.snapshot()
    assert snap[0] == {"type": "status", "paused": True, "speed": 2}
    assert {"type": "state", "t": 1} in snap and snap[-1]["type"] == "frame"

    hub._fanout({"type": "status", "loop": 1})  # replay restarted: old history is dropped
    assert not any(m["type"] == "state" for m in hub.snapshot())


def test_server_serves_ui_streams_messages_and_controls():
    p = make_pipeline(loop=True)
    app = create_app(p, info={"source": "fake.vrs"})
    with TestClient(app) as client:
        assert "Aria Gen 2" in client.get("/").text
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


@pytest.mark.parametrize("width", [64, 2000])
def test_encode_jpeg_downscales(width):
    import base64

    import cv2

    img = np.full((width // 2, width, 3), 128, np.uint8)
    data = base64.b64decode(encode_jpeg(img, width=1280))
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == min(width, 1280)
