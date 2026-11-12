"use strict";

// ---- configuration ---------------------------------------------------------
const STATE_COLORS = {
  Calibrating: "#64748b",
  Neutral: "#94a3b8",
  Calm: "#34d399",
  Focused: "#60a5fa",
  Distracted: "#fbbf24",
  Stressed: "#f87171",
  Fatigued: "#a78bfa",
  Startled: "#f472b6",
};

const SIGNALS = [
  { key: "hr_bpm", name: "Heart rate", unit: "bpm", digits: 0, color: "#f87171" },
  { key: "rmssd_ms", name: "HRV (RMSSD)", unit: "ms", digits: 0, color: "#fb923c" },
  { key: "blink_rate_pm", name: "Blinks", unit: "/min", digits: 0, color: "#a78bfa" },
  { key: "fix_dur_ms", name: "Fixation", unit: "ms", digits: 0, color: "#60a5fa" },
  { key: "gaze_disp_deg", name: "Gaze spread", unit: "°", digits: 1, color: "#38bdf8" },
  { key: "head_gyro_rms", name: "Head motion", unit: "rad/s", digits: 2, color: "#34d399" },
];

// Thresholds mirrored from configs/state.yaml, for the guide lines on the plane.
const PLANE = { range: 3, focused: 0.8, distracted: -0.8, stressed: 1.2, calm: 0.5 };
const SERIES_SECONDS = 120;
const TIMELINE_SECONDS = 300;
const PLANE_TRAIL_SECONDS = 30;

// ---- state -----------------------------------------------------------------
const S = {
  frame: null,
  image: new Image(),
  imageReady: false,
  series: Object.fromEntries(SIGNALS.map((s) => [s.key, []])),
  baseline: null,
  states: [], // {t, state}
  plane: [], // {t, arousal, attention}
  paused: false,
  showBoxes: true,
  lastT: 0,
};

const $ = (id) => document.getElementById(id);

// ---- canvas helpers ----------------------------------------------------------
function fit(canvas, aspect) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = aspect ? w / aspect : canvas.clientHeight;
  if (aspect) canvas.style.height = `${h}px`;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h);
}

// ---- video -------------------------------------------------------------------
S.image.onload = () => {
  S.imageReady = true;
  drawVideo();
};

function onFrame(m) {
  S.frame = m;
  S.lastT = m.t;
  $("clock").textContent = `t = ${m.t.toFixed(1)} s`;
  if (m.jpeg) S.image.src = `data:image/jpeg;base64,${m.jpeg}`;
  else drawVideo();

  const a = m.attended;
  $("lookingObj").textContent = a ? a.cls : "—";
  $("lookingDwell").textContent = a ? `${a.dwell.toFixed(1)} s` : m.blink ? "blink" : "";
}

function drawVideo() {
  const m = S.frame;
  if (!m || !S.imageReady) return;
  $("stageEmpty").hidden = true;
  const { ctx, w, h } = fit($("video"), m.aspect);
  ctx.drawImage(S.image, 0, 0, w, h);

  if (S.showBoxes) {
    m.detections.forEach((d, i) => {
      const attended = m.attended && m.attended.index === i;
      const [x1, y1, x2, y2] = [d.box[0] * w, d.box[1] * h, d.box[2] * w, d.box[3] * h];
      ctx.lineWidth = attended ? 3.5 : 1.2;
      ctx.strokeStyle = attended ? css("--attended") : css("--box");
      roundRect(ctx, x1, y1, x2 - x1, y2 - y1, 6);
      ctx.stroke();
      const label = attended ? `${d.cls}  ${m.attended.dwell.toFixed(1)}s` : `${d.cls} ${Math.round(d.conf * 100)}%`;
      ctx.font = `${attended ? 600 : 500} ${attended ? 14 : 11}px Inter, system-ui, sans-serif`;
      const tw = ctx.measureText(label).width + 10;
      const th = attended ? 22 : 17;
      ctx.fillStyle = attended ? css("--attended") : "rgba(20,26,40,0.78)";
      roundRect(ctx, x1, Math.max(0, y1 - th), tw, th, 5);
      ctx.fill();
      ctx.fillStyle = attended ? "#04130a" : "#dbe3f3";
      ctx.fillText(label, x1 + 5, Math.max(0, y1 - th) + th - 6);
    });
  }

  if (m.trail.length > 1) {
    ctx.strokeStyle = css("--trail");
    ctx.lineWidth = 3;
    ctx.lineJoin = "round";
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    m.trail.forEach(([x, y], i) => (i ? ctx.lineTo(x * w, y * h) : ctx.moveTo(x * w, y * h)));
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  if (m.gaze) {
    const [gx, gy] = [m.gaze[0] * w, m.gaze[1] * h];
    const r = Math.max(7, w / 110);
    ctx.strokeStyle = css("--gaze");
    ctx.fillStyle = css("--gaze");
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(gx, gy, r * 2.8, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(gx, gy, r, 0, Math.PI * 2);
    m.blink ? ctx.stroke() : ctx.fill();
  }
}

// ---- state card ----------------------------------------------------------------
function onState(m) {
  checkTime(m.t);
  const color = STATE_COLORS[m.state] || STATE_COLORS.Neutral;
  const card = $("stateCard");
  card.style.setProperty("--state", color);
  $("stateName").textContent = m.state;
  $("stateName").style.color = color; // set directly: a var()-driven transition lagged behind in Chrome
  $("confFill").style.width = `${Math.round((m.confidence || 0) * 100)}%`;
  $("confText").textContent = m.state === "Calibrating" ? "—" : `confidence ${Math.round(m.confidence * 100)}%`;
  $("candidate").textContent = m.candidate && m.candidate !== m.state && m.state !== "Startled" ? `→ ${m.candidate}?` : "";
  $("reasons").innerHTML = (m.reasons || []).map((r) => `<li>${escapeHtml(r)}</li>`).join("");

  const calibrating = m.baseline_progress < 1;
  $("calib").hidden = !calibrating;
  $("calibFill").style.width = `${Math.round(m.baseline_progress * 100)}%`;

  S.states.push({ t: m.t, state: m.state });
  trim(S.states, m.t, TIMELINE_SECONDS);
  if (m.arousal !== null && m.attention !== null) {
    S.plane.push({ t: m.t, arousal: m.arousal, attention: m.attention });
    trim(S.plane, m.t, PLANE_TRAIL_SECONDS);
  }
  renderBars(m.attention_totals || {}, m.attention_switches || 0);
  drawPlane();
  drawTimeline();
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function trim(arr, now, seconds) {
  while (arr.length && now - arr[0].t > seconds) arr.shift();
}

function renderBars(totals, switches) {
  const entries = Object.entries(totals).sort((a, b) => b[1] - a[1]).slice(0, 6);
  $("switches").textContent = switches ? `${switches} switches` : "";
  if (!entries.length) return;
  const max = entries[0][1] || 1;
  $("attnBars").innerHTML = entries
    .map(
      ([cls, secs]) => `<div class="bar-row"><span>${escapeHtml(cls)}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${(100 * secs) / max}%"></div></div>
        <span class="bar-val">${secs.toFixed(1)} s</span></div>`,
    )
    .join("");
}

// ---- arousal × attention plane -------------------------------------------------
function drawPlane() {
  const { ctx, w, h } = fit($("plane"));
  const pad = 26;
  const R = PLANE.range;
  const X = (v) => pad + ((Math.max(-R, Math.min(R, v)) + R) / (2 * R)) * (w - 2 * pad);
  const Y = (v) => h - pad - ((Math.max(-R, Math.min(R, v)) + R) / (2 * R)) * (h - 2 * pad);
  ctx.clearRect(0, 0, w, h);

  // Rule regions.
  const region = (x1, x2, y1, y2, state) => {
    ctx.fillStyle = STATE_COLORS[state] + "1c";
    ctx.fillRect(X(x1), Y(y2), X(x2) - X(x1), Y(y1) - Y(y2));
  };
  region(PLANE.focused, R, -R, 1.5, "Focused");
  region(-R, PLANE.distracted, -R, R, "Distracted");
  region(PLANE.distracted, PLANE.focused, PLANE.stressed, R, "Stressed");
  region(PLANE.distracted, PLANE.focused, -R, PLANE.calm, "Calm");

  ctx.strokeStyle = css("--line");
  ctx.lineWidth = 1;
  ctx.strokeRect(pad, pad, w - 2 * pad, h - 2 * pad);
  ctx.setLineDash([3, 4]);
  ctx.beginPath();
  ctx.moveTo(X(0), pad);
  ctx.lineTo(X(0), h - pad);
  ctx.moveTo(pad, Y(0));
  ctx.lineTo(w - pad, Y(0));
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = css("--muted");
  ctx.font = "11px Inter, system-ui, sans-serif";
  ctx.fillText("attention →", w - pad - 66, h - 8);
  ctx.save();
  ctx.translate(12, pad + 60);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("arousal →", 0, 0);
  ctx.restore();
  const tag = (txt, x, y) => {
    ctx.fillStyle = STATE_COLORS[txt];
    ctx.fillText(txt, x, y);
  };
  tag("Focused", X(PLANE.focused) + 6, Y(-R) - 6);
  tag("Distracted", X(-R) + 6, Y(-R) - 6);
  tag("Stressed", X(PLANE.distracted) + 6, Y(R) + 14);
  tag("Calm", X(PLANE.distracted) + 6, Y(-R) - 6);

  if (!S.plane.length) return;
  ctx.strokeStyle = "rgba(230,235,245,0.35)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  S.plane.forEach((p, i) => (i ? ctx.lineTo(X(p.attention), Y(p.arousal)) : ctx.moveTo(X(p.attention), Y(p.arousal))));
  ctx.stroke();
  const last = S.plane[S.plane.length - 1];
  const cur = S.states.length ? S.states[S.states.length - 1].state : "Neutral";
  ctx.fillStyle = STATE_COLORS[cur] || "#fff";
  ctx.strokeStyle = "#0b0f17";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(X(last.attention), Y(last.arousal), 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}

// ---- signals ---------------------------------------------------------------------
function buildSignals() {
  $("signals").innerHTML = SIGNALS.map(
    (s) => `<div class="sig"><div><div class="sig-name">${s.name}</div>
      <div class="sig-val" id="v_${s.key}">—<small>${s.unit}</small></div></div>
      <canvas id="c_${s.key}"></canvas></div>`,
  ).join("");
}

function onFeatures(m) {
  checkTime(m.t);
  for (const s of SIGNALS) {
    const v = m.values[s.key];
    const arr = S.series[s.key];
    arr.push({ t: m.t, v });
    trim(arr, m.t, SERIES_SECONDS);
    $(`v_${s.key}`).innerHTML = `${v === null || v === undefined ? "—" : v.toFixed(s.digits)}<small>${s.unit}</small>`;
  }
  drawSignals();
}

function drawSignals() {
  for (const s of SIGNALS) {
    const { ctx, w, h } = fit($(`c_${s.key}`));
    ctx.clearRect(0, 0, w, h);
    const pts = S.series[s.key].filter((p) => p.v !== null && p.v !== undefined);
    const mean = S.baseline?.mean?.[s.key];
    const sd = S.baseline?.std?.[s.key];
    const vals = pts.map((p) => p.v);
    if (mean !== undefined && mean !== null) vals.push(mean - 1.5 * sd, mean + 1.5 * sd);
    if (vals.length < 2) continue;
    let lo = Math.min(...vals);
    let hi = Math.max(...vals);
    if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
    const tEnd = pts[pts.length - 1]?.t ?? 0;
    const span = Math.max(30, Math.min(SERIES_SECONDS, tEnd - (pts[0]?.t ?? tEnd)));
    const X = (t) => w - ((tEnd - t) / span) * w;
    const Y = (v) => h - 3 - ((v - lo) / (hi - lo)) * (h - 6);

    if (mean !== undefined && mean !== null) {
      ctx.fillStyle = "rgba(148,163,184,0.16)";
      ctx.fillRect(0, Y(mean + sd), w, Y(mean - sd) - Y(mean + sd));
      ctx.strokeStyle = "rgba(148,163,184,0.45)";
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(0, Y(mean));
      ctx.lineTo(w, Y(mean));
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach((p, i) => (i ? ctx.lineTo(X(p.t), Y(p.v)) : ctx.moveTo(X(p.t), Y(p.v))));
    ctx.stroke();
  }
}

// ---- timeline ---------------------------------------------------------------------
function buildLegend() {
  $("legend").innerHTML = Object.entries(STATE_COLORS)
    .map(([k, c]) => `<span><i style="background:${c}"></i>${k}</span>`)
    .join("");
}

function drawTimeline() {
  const { ctx, w, h } = fit($("timeline"));
  ctx.clearRect(0, 0, w, h);
  if (!S.states.length) return;
  const tEnd = S.states[S.states.length - 1].t;
  const span = Math.max(30, Math.min(TIMELINE_SECONDS, tEnd - S.states[0].t + 1));
  const X = (t) => w - ((tEnd - t) / span) * w;
  const barH = h - 18;
  S.states.forEach((s, i) => {
    const next = S.states[i + 1]?.t ?? s.t + 1;
    ctx.fillStyle = STATE_COLORS[s.state] || "#666";
    const x = Math.floor(X(s.t));
    ctx.fillRect(x, 0, Math.max(1, Math.ceil(X(next)) - x), barH); // integer edges: no seams
  });
  ctx.fillStyle = css("--muted");
  ctx.font = "11px Inter, system-ui, sans-serif";
  const step = span > 120 ? 30 : 10;
  for (let t = Math.max(0, Math.ceil((tEnd - span) / step) * step); t <= tEnd; t += step) {
    ctx.fillText(`${t}s`, X(t) - 8, h - 3);
  }
}

// ---- status & controls ---------------------------------------------------------------
function onStatus(m) {
  if ("paused" in m) {
    S.paused = m.paused;
    $("pause").textContent = m.paused ? "Resume" : "Pause";
  }
  if ("speed" in m) $("speed").value = String(m.speed);
  if (m.error) toast(`Pipeline error: ${m.error}`);
  else if (m.ended) toast("Replay finished.", 4000);
  if ("loop" in m && m.loop > 0) resetHistory();
}

function resetHistory() {
  S.states = [];
  S.plane = [];
  for (const k in S.series) S.series[k] = [];
}

// Time going backwards means the replay restarted: start the charts afresh.
function checkTime(t) {
  const last = S.states.length ? S.states[S.states.length - 1].t : -Infinity;
  if (t < last - 1) resetHistory();
}

function toast(text, ms) {
  const el = $("toast");
  el.textContent = text;
  el.hidden = false;
  if (ms) setTimeout(() => (el.hidden = true), ms);
}

async function control(action, value) {
  try {
    await fetch("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, value }),
    });
  } catch (e) {
    toast(`Control failed: ${e}`, 4000);
  }
}

$("pause").onclick = () => control(S.paused ? "resume" : "pause");
$("rebaseline").onclick = () => control("rebaseline");
$("speed").onchange = (e) => control("speed", parseFloat(e.target.value));
$("showBoxes").onchange = (e) => {
  S.showBoxes = e.target.checked;
  drawVideo();
};
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && e.target === document.body) {
    e.preventDefault();
    $("pause").click();
  }
});
window.addEventListener("resize", () => {
  drawVideo();
  drawPlane();
  drawSignals();
  drawTimeline();
});

// ---- websocket ------------------------------------------------------------------------
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => {
    $("conn").textContent = "live";
    $("conn").className = "pill ok";
  };
  ws.onclose = () => {
    $("conn").textContent = "reconnecting…";
    $("conn").className = "pill warn";
    setTimeout(connect, 1000);
  };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "frame") onFrame(m);
    else if (m.type === "state") onState(m);
    else if (m.type === "features") onFeatures(m);
    else if (m.type === "baseline") {
      S.baseline = m;
      drawSignals();
    } else if (m.type === "status") onStatus(m);
  };
}

async function loadInfo() {
  try {
    const info = await (await fetch("/api/info")).json();
    $("source").textContent = `${info.source || "source"} · baseline ${info.baseline_s}s${info.loop ? " · looping" : ""}`;
    S.paused = info.paused;
  } catch {
    $("source").textContent = "info unavailable";
  }
}

buildSignals();
buildLegend();
drawPlane();
loadInfo();
connect();
