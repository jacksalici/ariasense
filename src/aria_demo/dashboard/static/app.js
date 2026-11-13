"use strict";

// ---- configuration ---------------------------------------------------------
const STATE_COLORS = {
  Calibrating: "#a1a1aa",
  "No signal": "#a1a1aa",
  Neutral: "#71717a",
  Calm: "#059669",
  Focused: "#2563eb",
  Distracted: "#d97706",
  Stressed: "#dc2626",
  Fatigued: "#7c3aed",
  Startled: "#db2777",
};

const SIGNALS = [
  { key: "hr_bpm", name: "Heart rate", unit: "bpm", digits: 0, color: "#dc2626" },
  { key: "rmssd_ms", name: "HRV (RMSSD)", unit: "ms", digits: 0, color: "#ea580c" },
  { key: "blink_rate_pm", name: "Blinks", unit: "/min", digits: 0, color: "#7c3aed" },
  { key: "fix_dur_ms", name: "Fixation", unit: "ms", digits: 0, color: "#2563eb" },
  { key: "gaze_disp_deg", name: "Gaze spread", unit: "°", digits: 1, color: "#0284c7" },
  { key: "head_gyro_rms", name: "Head motion", unit: "rad/s", digits: 2, color: "#059669" },
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
  sourceLabel: "",
  lastT: 0,
};

const $ = (id) => document.getElementById(id);

// ---- canvas helpers ----------------------------------------------------------
function fit(canvas, aspect) {
  const dpr = window.devicePixelRatio || 1;
  let w;
  let h;
  if (aspect) {
    // Contain the image inside its box, keeping the aspect ratio.
    const box = canvas.parentElement;
    w = box.clientWidth;
    h = box.clientHeight;
    if (w / aspect > h) w = h * aspect;
    w = Math.max(1, Math.floor(w));
    h = Math.max(1, Math.floor(w / aspect));
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
  } else {
    w = canvas.clientWidth;
    h = canvas.clientHeight;
  }
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

  if (m.latency_ms !== undefined) {
    const el = $("latency");
    el.hidden = false;
    el.textContent = m.latency_ms === null ? "latency —" : `latency ${m.latency_ms} ms`;
    el.className = `pill ${m.latency_ms === null ? "" : m.latency_ms < 400 ? "ok" : m.latency_ms < 1000 ? "warn" : "bad"}`;
  }

  const a = m.attended;
  $("lookingObj").textContent = a ? a.cls : "—";
  $("lookingDwell").textContent = a ? `${a.dwell.toFixed(1)} s` : m.blink ? "blink" : "—";
}

function drawVideo() {
  const m = S.frame;
  if (!m || !S.imageReady) return;
  $("stageEmpty").textContent = "";
  const { ctx, w, h } = fit($("video"), m.aspect);
  ctx.drawImage(S.image, 0, 0, w, h);

  if (S.showBoxes) {
    m.detections.forEach((d, i) => {
      const attended = m.attended && m.attended.index === i;
      const [x1, y1, x2, y2] = [d.box[0] * w, d.box[1] * h, d.box[2] * w, d.box[3] * h];
      ctx.lineWidth = attended ? 2.5 : 1;
      ctx.strokeStyle = attended ? css("--attended") : css("--box");
      roundRect(ctx, x1, y1, x2 - x1, y2 - y1, 6);
      ctx.stroke();
      const label = attended ? `${d.cls}  ${m.attended.dwell.toFixed(1)}s` : `${d.cls} ${Math.round(d.conf * 100)}%`;
      ctx.font = `${attended ? 600 : 500} ${attended ? 12 : 10}px Inter, system-ui, sans-serif`;
      const tw = ctx.measureText(label).width + 10;
      const th = attended ? 18 : 15;
      ctx.fillStyle = attended ? css("--attended") : "rgba(20,26,40,0.78)";
      roundRect(ctx, x1, Math.max(0, y1 - th), tw, th, 5);
      ctx.fill();
      ctx.fillStyle = attended ? "#04130a" : "#dbe3f3";
      ctx.fillText(label, x1 + 5, Math.max(0, y1 - th) + th - 5);
    });
  }

  if (m.trail.length > 1) {
    ctx.strokeStyle = css("--trail");
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    m.trail.forEach(([x, y], i) => (i ? ctx.lineTo(x * w, y * h) : ctx.moveTo(x * w, y * h)));
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  if (m.gaze) {
    const [gx, gy] = [m.gaze[0] * w, m.gaze[1] * h];
    const r = Math.max(5, w / 150);
    ctx.strokeStyle = css("--gaze");
    ctx.fillStyle = css("--gaze");
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(gx, gy, r * 2.4, 0, Math.PI * 2);
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
  $("confFill").style.background = color;
  $("confText").textContent = m.state === "Calibrating" ? "—" : `${Math.round(m.confidence * 100)}%`;
  $("candidate").textContent =
    m.candidate && m.candidate !== m.state && m.state !== "Startled" ? `→ ${m.candidate}?` : "not a measured emotion";
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
  const pad = 18;
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
  ctx.font = "10px Inter, system-ui, sans-serif";
  ctx.fillText("attention →", w - pad - 56, h - 5);
  ctx.save();
  ctx.translate(10, pad + 52);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("arousal →", 0, 0);
  ctx.restore();
  const tag = (txt, x, y) => {
    ctx.fillStyle = STATE_COLORS[txt];
    ctx.fillText(txt, x, y);
  };
  tag("Focused", X(PLANE.focused) + 6, Y(-R) - 6);
  tag("Distracted", X(-R) + 6, Y(-R) - 6);
  tag("Stressed", X(PLANE.distracted) + 6, Y(R) + 12);
  tag("Calm", X(PLANE.distracted) + 6, Y(-R) - 6);

  if (!S.plane.length) return;
  ctx.strokeStyle = "rgba(17,17,20,0.3)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  S.plane.forEach((p, i) => (i ? ctx.lineTo(X(p.attention), Y(p.arousal)) : ctx.moveTo(X(p.attention), Y(p.arousal))));
  ctx.stroke();
  const last = S.plane[S.plane.length - 1];
  const cur = S.states.length ? S.states[S.states.length - 1].state : "Neutral";
  ctx.fillStyle = STATE_COLORS[cur] || "#fff";
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(X(last.attention), Y(last.arousal), 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}

// ---- signals (uPlot sparklines) ---------------------------------------------------
const SPARKS = {}; // key -> { plot, win }
const MUTED = "#71717a";

function drawBand(u, key) {
  const mean = S.baseline?.mean?.[key];
  const sd = S.baseline?.std?.[key];
  if (mean === undefined || mean === null || sd === undefined || sd === null) return;
  const { ctx } = u;
  const { left, width } = u.bbox;
  const dpr = window.devicePixelRatio || 1;
  const y1 = u.valToPos(mean + sd, "y", true);
  const y2 = u.valToPos(mean - sd, "y", true);
  const ym = u.valToPos(mean, "y", true);
  ctx.save();
  ctx.fillStyle = "rgba(113,113,122,0.12)";
  ctx.fillRect(left, y1, width, y2 - y1);
  ctx.strokeStyle = "rgba(113,113,122,0.4)";
  ctx.lineWidth = dpr;
  ctx.setLineDash([3 * dpr, 3 * dpr]);
  ctx.beginPath();
  ctx.moveTo(left, ym);
  ctx.lineTo(left + width, ym);
  ctx.stroke();
  ctx.restore();
}

function makeSpark(s) {
  const el = $(`c_${s.key}`);
  const win = { t0: 0, t1: 1, lo: 0, hi: 1 };
  const plot = new uPlot(
    {
      width: Math.max(40, el.clientWidth),
      height: Math.max(16, el.clientHeight),
      padding: [2, 0, 2, 0],
      legend: { show: false },
      cursor: { show: false },
      select: { show: false },
      scales: { x: { time: false, range: () => [win.t0, win.t1] }, y: { range: () => [win.lo, win.hi] } },
      axes: [{ show: false }, { show: false }],
      series: [{}, { stroke: s.color, width: 1.5, points: { show: false } }],
      hooks: { drawClear: [(u) => drawBand(u, s.key)] },
    },
    [[], []],
    el,
  );
  SPARKS[s.key] = { plot, win, el };
}

function buildSignals() {
  $("signals").innerHTML = SIGNALS.map(
    (s) => `<div class="sig"><div><div class="sig-name">${s.name}</div>
      <div class="sig-val" id="v_${s.key}">—<small>${s.unit}</small></div></div>
      <div class="spark" id="c_${s.key}"></div></div>`,
  ).join("");
  SIGNALS.forEach(makeSpark);
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
    const { plot, win } = SPARKS[s.key];
    const pts = S.series[s.key].filter((p) => p.v !== null && p.v !== undefined);
    const mean = S.baseline?.mean?.[s.key];
    const sd = S.baseline?.std?.[s.key];
    const vals = pts.map((p) => p.v);
    if (mean !== undefined && mean !== null) vals.push(mean - 1.5 * sd, mean + 1.5 * sd);
    if (pts.length < 2) {
      plot.setData([[], []]);
      continue;
    }
    let lo = Math.min(...vals);
    let hi = Math.max(...vals);
    if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
    const tEnd = pts[pts.length - 1].t;
    const span = Math.max(30, Math.min(SERIES_SECONDS, tEnd - pts[0].t));
    Object.assign(win, { t0: tEnd - span, t1: tEnd, lo, hi });
    plot.setData([pts.map((p) => p.t), pts.map((p) => p.v)]);
  }
}

// ---- timeline (uPlot axis, state bars drawn under it) -----------------------------
const TL = { plot: null, win: { t0: 0, t1: 1 } };

function drawStates(u) {
  if (!S.states.length) return;
  const { ctx } = u;
  const { left, top, width, height } = u.bbox;
  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, width, height);
  ctx.clip();
  S.states.forEach((st, i) => {
    const next = S.states[i + 1]?.t ?? st.t + 1;
    const x = Math.floor(u.valToPos(st.t, "x", true));
    const x2 = Math.ceil(u.valToPos(next, "x", true));
    ctx.fillStyle = STATE_COLORS[st.state] || "#666";
    ctx.fillRect(x, top, Math.max(1, x2 - x), height); // integer edges: no seams
  });
  ctx.restore();
}

function buildTimeline() {
  const el = $("timeline");
  TL.plot = new uPlot(
    {
      width: Math.max(100, el.clientWidth),
      height: Math.max(30, el.clientHeight),
      padding: [0, 0, 0, 0],
      legend: { show: false },
      cursor: { show: false },
      select: { show: false },
      scales: { x: { time: false, range: () => [TL.win.t0, TL.win.t1] }, y: { range: () => [0, 1] } },
      axes: [
        {
          stroke: MUTED,
          font: "10px Inter, system-ui, sans-serif",
          size: 18,
          space: 50,
          incrs: [10, 30, 60],
          ticks: { show: false },
          grid: { show: false },
          values: (u, vals) => vals.map((v) => (v < 0 ? "" : `${Math.round(v)}s`)),
        },
        { show: false },
      ],
      series: [{}, { show: false }],
      hooks: { drawClear: [drawStates] },
    },
    [[], []],
    el,
  );
}

function buildLegend() {
  $("legend").innerHTML = Object.entries(STATE_COLORS)
    .map(([k, c]) => `<span><i style="background:${c}"></i>${k}</span>`)
    .join("");
}

function drawTimeline() {
  if (!S.states.length) return;
  const tEnd = S.states[S.states.length - 1].t;
  const span = Math.max(30, Math.min(TIMELINE_SECONDS, tEnd - S.states[0].t + 1));
  TL.win = { t0: tEnd - span, t1: tEnd };
  TL.plot.setData([[TL.win.t0, TL.win.t1], [null, null]]);
}

// ---- status & controls ---------------------------------------------------------------
function onStatus(m) {
  if ("paused" in m) {
    S.paused = m.paused;
    $("pause").textContent = m.paused ? "Resume" : "Pause";
  }
  if ("speed" in m) $("speed").value = String(m.speed);
  if ("missing_streams" in m) {
    const effects = {
      ppg: "heart rate & HRV unavailable",
      eye_gaze: "no gaze: attention & blinks unavailable",
      rgb: "no camera video",
      imu: "head motion & startle unavailable",
    };
    const missing = m.missing_streams.map((s) => `${s} (${effects[s] || "missing"})`);
    const notes = [];
    if (m.idle_s >= 3) notes.push(`No data from the glasses for ${Math.round(m.idle_s)} s — check the connection.`);
    if (missing.length) notes.push(`Not receiving: ${missing.join(" · ")}. Check the streaming profile.`);
    $("banner").hidden = !notes.length;
    $("banner").textContent = notes.join("  ");
  }
  if (m.streams) {
    const rates = Object.entries(m.streams).map(([k, v]) => `${k} ${Math.round(v)} Hz`).join(" · ");
    $("source").textContent = `${S.sourceLabel} · ${rates}${m.dropped_rgb ? ` · ${m.dropped_rgb} frames skipped` : ""}`;
  }
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
    S.sourceLabel = `${info.source || "source"} · baseline ${info.baseline_s}s${info.loop ? " · looping" : ""}`;
    $("source").textContent = S.sourceLabel;
    S.paused = info.paused;
    $("speedBox").hidden = info.mode !== "replay"; // speed only applies to file replay
  } catch {
    $("source").textContent = "info unavailable";
  }
}

// ---- grid layout (GridStack) --------------------------------------------------------
const GRID_ROWS = 20; // default layout height, in rows: the cell height scales so it fills the window
let grid = null;

function resizePlots() {
  for (const { plot, el } of Object.values(SPARKS)) {
    const w = Math.max(40, el.clientWidth);
    const h = Math.max(16, el.clientHeight);
    if (plot.width !== w || plot.height !== h) plot.setSize({ width: w, height: h });
  }
  const tl = $("timeline");
  const w = Math.max(100, tl.clientWidth);
  const h = Math.max(30, tl.clientHeight);
  if (TL.plot.width !== w || TL.plot.height !== h) TL.plot.setSize({ width: w, height: h });
}

function redrawAll() {
  resizePlots();
  drawVideo();
  drawPlane();
  drawSignals();
  drawTimeline();
}

function fitRows() {
  const top = $("grid").getBoundingClientRect().top + window.scrollY;
  const cell = Math.max(18, Math.min(64, Math.floor((window.innerHeight - top - 10) / GRID_ROWS)));
  grid.cellHeight(cell);
}

function setupGrid() {
  const KEY = "aria-demo-grid-v2";
  grid = GridStack.init(
    { column: 12, cellHeight: 24, margin: 3, float: false, handle: ".panel-head", resizable: { handles: "se,e,s" } },
    "#grid",
  );
  const defaults = grid.save(false);
  const known = new Set(defaults.map((n) => n.id));
  // load() resolves collisions for the whole set at once; only ids that exist are touched.
  const apply = (layout) => grid.load(layout.filter((n) => known.has(n.id)), false);
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) || "null");
    if (Array.isArray(saved)) apply(saved);
  } catch {}
  fitRows();
  grid.on("change", () => {
    try {
      localStorage.setItem(KEY, JSON.stringify(grid.save(false)));
    } catch {}
  });
  $("resetLayout").onclick = () => {
    try {
      localStorage.removeItem(KEY);
    } catch {}
    apply(defaults);
  };

  let queued = false;
  const schedule = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      redrawAll();
    });
  };
  const ro = new ResizeObserver(schedule);
  document.querySelectorAll(".fill, .spark").forEach((el) => ro.observe(el));
  window.addEventListener("resize", () => {
    fitRows();
    schedule();
  });
}

buildSignals();
buildTimeline();
buildLegend();
setupGrid();
drawPlane();
loadInfo();
connect();
