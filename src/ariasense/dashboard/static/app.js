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
const PLANE_TRAIL_SECONDS = 30;
const THUMB_W = 160; // thumbnail width requested from the server, px
const THUMB_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600];

// ---- state -----------------------------------------------------------------
const S = {
  frame: null,
  image: new Image(),
  imageReady: false,
  baseline: null,
  states: [], // full `state` messages of the whole session
  features: [], // full `features` messages of the whole session
  liveFrame: null, // latest frame received, shown while following live
  view: { live: true, t: 0, playing: false }, // what the panels show: the live edge or a chosen time
  epoch: null, // replay run; thumbnails are only valid within one
  paused: false,
  showBoxes: true,
  session: 0, // bumped by the server when another recording is opened
  follow: true, // false: the file is analysed as fast as possible and reviewed, never followed live
  duration: 0, // length of the file being replayed (s), when the source knows it
  ended: false,
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

// A frame arrived from the stream. It is always recorded server-side; shown only when following live.
function onFrame(m) {
  S.liveFrame = m;
  S.lastT = m.t;
  if (!S.view.live) {
    if (!S.frame) fetchFrame(S.view.t); // analysis just started: show its first frame
    drawTimelineThrottled(); // the session grew; the picture stays on the chosen moment
    return;
  }
  showFrame(m);
  drawTimelineThrottled();
}

function showFrame(m) {
  S.frame = m;
  if (m.jpeg) S.image.src = `data:image/jpeg;base64,${m.jpeg}`;
  else drawVideo();

  if (m.latency_ms !== undefined && S.view.live) {
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
  S.states.push(m);
  if (S.view.live) renderState(m);
  if (S.view.live) drawPlane();
  drawTimeline();
}

function renderState(m) {
  const color = STATE_COLORS[m.state] || STATE_COLORS.Neutral;
  const card = $("stateCard");
  card.style.setProperty("--state", color);
  $("stateName").textContent = m.state;
  $("stateName").style.color = color; // set directly: a var()-driven transition lagged behind in Chrome
  $("confFill").style.width = `${Math.round((m.confidence || 0) * 100)}%`;
  $("confFill").style.background = color;
  $("confText").textContent = m.state === "Calibrating" ? "—" : `${Math.round(m.confidence * 100)}%`;
  $("candidate").textContent =
    m.candidate && m.candidate !== m.state && m.state !== "Startled" ? `→ ${m.candidate}?` : "";
  $("reasons").innerHTML = (m.reasons || []).map((r) => `<li>${escapeHtml(r)}</li>`).join("");

  const calibrating = m.baseline_progress < 1;
  $("calib").hidden = !calibrating;
  $("calibFill").style.width = `${Math.round(m.baseline_progress * 100)}%`;

  renderBars(m.attention_totals || {});
  $("switches").textContent = String(m.attention_switches || 0);
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// ---- view time: what the panels show --------------------------------------------
// Live: the newest message. Review: the moment picked on the timeline.
function viewEnd(arr) {
  return S.view.live ? (arr.length ? arr[arr.length - 1].t : 0) : S.view.t;
}

// Index of the last message at or before t (-1 if none).
function indexAt(arr, t) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid].t <= t) lo = mid + 1;
    else hi = mid;
  }
  return lo - 1;
}

// Messages in the `seconds` before the view time.
function windowOf(arr, seconds) {
  const end = viewEnd(arr);
  return arr.slice(indexAt(arr, end - seconds) + 1, indexAt(arr, end) + 1);
}

// The timeline's span: the whole file when it is being analysed, else what has been seen so far.
function timelineEnd() {
  return S.follow || S.ended ? sessionEnd() : Math.max(sessionEnd(), S.duration || 0);
}

function sessionEnd() {
  return Math.max(S.lastT, S.states.length ? S.states[S.states.length - 1].t : 0, 1);
}

// Redraw every panel for the current view time.
function renderView() {
  const st = S.states[indexAt(S.states, viewEnd(S.states))];
  if (st) renderState(st);
  else resetKpis();
  renderSignalValues();
  drawPlane();
  drawSignals();
  drawTimeline();
}

function renderBars(totals) {
  const entries = Object.entries(totals).sort((a, b) => b[1] - a[1]).slice(0, 6);
  if (!entries.length) {
    $("attnBars").innerHTML = '<p class="muted">No attended objects yet.</p>';
    return;
  }
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

  const plane = windowOf(S.states, PLANE_TRAIL_SECONDS).filter((p) => p.arousal !== null && p.attention !== null);
  if (!plane.length) return;
  ctx.strokeStyle = "rgba(17,17,20,0.3)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  plane.forEach((p, i) => (i ? ctx.lineTo(X(p.attention), Y(p.arousal)) : ctx.moveTo(X(p.attention), Y(p.arousal))));
  ctx.stroke();
  const last = plane[plane.length - 1];
  const cur = S.states[indexAt(S.states, viewEnd(S.states))]?.state ?? "Neutral";
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
  S.features.push(m);
  if (!S.view.live) return;
  renderSignalValues();
  drawSignals();
}

function renderSignalValues() {
  const m = S.features[indexAt(S.features, viewEnd(S.features))];
  for (const s of SIGNALS) {
    const v = m?.values[s.key];
    $(`v_${s.key}`).innerHTML = `${v === null || v === undefined ? "—" : v.toFixed(s.digits)}<small>${s.unit}</small>`;
  }
}

function drawSignals() {
  const recent = windowOf(S.features, SERIES_SECONDS);
  for (const s of SIGNALS) {
    const { plot, win } = SPARKS[s.key];
    const pts = recent.map((f) => ({ t: f.t, v: f.values[s.key] })).filter((p) => p.v !== null && p.v !== undefined);
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

// ---- timeline: the whole session, thumbnails + state bars, click or drag to scrub ------------
const TL = { hoverX: null, drawnAt: 0, inflight: 0, drawing: false };
const THUMBS = new Map(); // `${epoch}:${t}` -> { img, ready }
const TICKS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];

function fmtTime(t) {
  const m = Math.floor(t / 60);
  const sec = Math.floor(t % 60);
  return m ? `${m}:${String(sec).padStart(2, "0")}` : `${sec}s`;
}

function thumbFor(t) {
  const key = `${S.session}:${S.epoch}:${t}`;
  let th = THUMBS.get(key);
  if (!th && TL.inflight < 6) {
    const img = new Image();
    th = { img, ready: false };
    THUMBS.set(key, th);
    TL.inflight += 1;
    img.onload = () => {
      th.ready = true;
      TL.inflight -= 1;
      drawTimelineThrottled(true);
    };
    img.onerror = () => {
      TL.inflight -= 1;
      setTimeout(() => THUMBS.delete(key), 2000); // not recorded yet: retry later
    };
    img.src = `/api/thumb?t=${t}&w=${THUMB_W}`;
  }
  return th && th.ready ? th.img : null;
}

function drawTimeline() {
  TL.drawnAt = performance.now();
  const { ctx, w, h } = fit($("timeline"));
  ctx.clearRect(0, 0, w, h);
  const done = sessionEnd(); // processed so far
  const end = timelineEnd();
  const pps = w / end; // pixels per second
  const axisH = 15;
  const barH = 10;
  const stripH = h - axisH - barH - 4;
  const barY = Math.max(0, stripH) + 2;

  // Thumbnails: one every `step` seconds, wide enough not to repeat the same picture.
  if (stripH >= 18) {
    const aspect = S.liveFrame?.aspect || S.frame?.aspect || 1;
    const tw = stripH * aspect;
    const step = THUMB_STEPS.find((v) => v * pps >= tw) || THUMB_STEPS[THUMB_STEPS.length - 1];
    ctx.fillStyle = S.lastT > 0 ? "#000" : css("--panel-2");
    ctx.fillRect(0, 0, w, stripH);
    for (let t = 0; t < done; t += step) {
      const x = t * pps;
      const cw = Math.min(step * pps, w - x);
      const img = thumbFor(t);
      if (!img) continue;
      // Cover-crop the thumbnail into its cell.
      const cellAspect = cw / stripH;
      const iw = img.naturalWidth;
      const ih = img.naturalHeight;
      let sw = iw;
      let sh = ih;
      if (iw / ih > cellAspect) sw = ih * cellAspect;
      else sh = iw / cellAspect;
      ctx.drawImage(img, (iw - sw) / 2, (ih - sh) / 2, sw, sh, x, 0, cw, stripH);
    }
  }

  // State bars.
  ctx.fillStyle = css("--panel-2");
  ctx.fillRect(0, barY, w, barH);
  S.states.forEach((st, i) => {
    const next = S.states[i + 1]?.t ?? st.t + 1;
    const x = Math.floor(st.t * pps);
    const x2 = Math.ceil(next * pps);
    ctx.fillStyle = STATE_COLORS[st.state] || "#666";
    ctx.fillRect(x, barY, Math.max(1, x2 - x), barH); // integer edges: no seams
  });

  // Not analysed yet.
  if (S.lastT > 0 && done < end - 0.05) {
    const x = Math.round(done * pps);
    ctx.fillStyle = css("--panel-2");
    ctx.fillRect(x, 0, w - x, barY + barH);
    if (w - x > 120 && stripH >= 18) {
      ctx.fillStyle = MUTED;
      ctx.font = "11px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(`Analysing… ${Math.floor((100 * done) / end)}%`, (x + w) / 2, stripH / 2);
    }
  }

  // Axis.
  const tick = TICKS.find((v) => v * pps >= 70) || TICKS[TICKS.length - 1];
  ctx.fillStyle = MUTED;
  ctx.font = "10px Inter, system-ui, sans-serif";
  ctx.textBaseline = "alphabetic";
  for (let t = 0; t <= end; t += tick) {
    const x = Math.round(t * pps);
    ctx.fillRect(x, barY + barH, 1, 3);
    ctx.textAlign = t === 0 ? "left" : "center";
    ctx.fillText(fmtTime(t), t === 0 ? 1 : Math.min(x, w - 14), h - 2);
  }

  // Hover time.
  if (TL.hoverX !== null) {
    const x = Math.max(0, Math.min(w, TL.hoverX));
    ctx.fillStyle = "rgba(17,17,20,0.35)";
    ctx.fillRect(Math.round(x), 0, 1, barY + barH);
    const label = fmtTime((x / w) * end);
    const lw = ctx.measureText(label).width + 8;
    ctx.fillStyle = "rgba(17,17,20,0.85)";
    roundRect(ctx, Math.min(Math.max(0, x - lw / 2), w - lw), 2, lw, 15, 4);
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.textAlign = "left";
    ctx.fillText(label, Math.min(Math.max(0, x - lw / 2), w - lw) + 4, 13);
  }

  // Playhead: at the right edge while live.
  const px = Math.max(1, Math.min(w - 1, (S.view.live ? end : S.view.t) * pps));
  ctx.fillStyle = S.view.live ? "#16a34a" : "#2563eb";
  ctx.fillRect(Math.round(px) - 1, 0, 2, barY + barH);
  ctx.beginPath();
  ctx.moveTo(px - 5, 0);
  ctx.lineTo(px + 5, 0);
  ctx.lineTo(px, 7);
  ctx.closePath();
  ctx.fill();
}

// The session grows ~10 times a second; redrawing the strip that often is wasted work.
function drawTimelineThrottled(now) {
  if (TL.drawing) return;
  const wait = now ? 0 : Math.max(0, 500 - (performance.now() - TL.drawnAt));
  TL.drawing = true;
  setTimeout(() => {
    TL.drawing = false;
    drawTimeline();
  }, wait);
}

function buildLegend() {
  $("legend").innerHTML = Object.entries(STATE_COLORS)
    .map(([k, c]) => `<span><i style="background:${c}"></i>${k}</span>`)
    .join("");
}

// ---- review: scrub back through the recording, or follow live --------------------------
let frameBusy = false;
let framePending = null;

async function fetchFrame(t) {
  if (frameBusy) {
    framePending = t; // only the newest request matters while scrubbing
    return;
  }
  frameBusy = true;
  try {
    const r = await fetch(`/api/frame?t=${t.toFixed(2)}`);
    if (r.ok && !S.view.live) showFrame(await r.json());
  } catch {}
  frameBusy = false;
  if (framePending !== null) {
    const next = framePending;
    framePending = null;
    if (!S.view.live) fetchFrame(next);
  }
}

function updateModeUi() {
  const live = S.view.live;
  $("tlLive").hidden = !S.follow;
  $("tlLive").classList.toggle("primary", live);
  $("tlLive").textContent = live ? "● Live" : "Go live";
  $("tlPlay").disabled = live;
  $("tlPlay").textContent = S.view.playing ? "Pause" : "Play";
  updatePauseButton();
}

function seek(t) {
  S.view.live = false;
  S.view.t = Math.max(0, Math.min(t, sessionEnd()));
  $("latency").hidden = true;
  updateModeUi();
  fetchFrame(S.view.t);
  renderView();
}

// Analysis mode: playback caught up with what has been processed (or the file ended).
function stopAtEnd() {
  S.view.t = sessionEnd();
  S.view.playing = false;
  updateModeUi();
  renderView();
}

// Analysis mode: never follow live; start reviewing from the beginning.
function enterReview() {
  S.view.live = false;
  S.view.playing = false;
  S.view.t = 0;
  updateModeUi();
  if (S.lastT > 0) fetchFrame(0);
  renderView();
}

function goLive() {
  if (!S.follow) return;
  S.view.live = true;
  S.view.playing = false;
  framePending = null;
  updateModeUi();
  if (S.liveFrame) showFrame(S.liveFrame);
  renderView();
}

function togglePlay() {
  if (S.view.live) return;
  S.view.playing = !S.view.playing;
  updateModeUi();
}

// Playback of the recording at its original speed, fetched frame by frame.
let lastTick = 0;
let lastKey = "";
function playbackTick(now) {
  const dt = lastTick ? (now - lastTick) / 1000 : 0;
  lastTick = now;
  if (S.view.playing && !S.view.live) {
    S.view.t += Math.min(dt, 0.25);
    if (S.view.t >= sessionEnd() - 0.1) {
      if (S.follow) goLive();
      else stopAtEnd();
    } else {
      fetchFrame(S.view.t);
      const key = `${indexAt(S.states, S.view.t)}:${indexAt(S.features, S.view.t)}`;
      if (key !== lastKey) {
        lastKey = key;
        renderView();
      } else {
        drawTimeline();
      }
    }
  }
  requestAnimationFrame(playbackTick);
}

function setupTimeline() {
  const canvas = $("timeline");
  const timeAt = (e) => {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, t: ((e.clientX - r.left) / r.width) * timelineEnd(), w: r.width };
  };
  let dragging = false;
  canvas.addEventListener("pointerdown", (e) => {
    dragging = true;
    canvas.setPointerCapture(e.pointerId);
    const { x, t, w } = timeAt(e);
    if (x >= w - 3 && S.follow) goLive();
    else seek(t);
  });
  canvas.addEventListener("pointermove", (e) => {
    const { x, t } = timeAt(e);
    TL.hoverX = x;
    if (dragging) seek(t);
    else drawTimelineThrottled(true);
  });
  const end = () => (dragging = false);
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
  canvas.addEventListener("pointerleave", () => {
    TL.hoverX = null;
    drawTimelineThrottled(true);
  });
  // Keep the panel header's buttons from starting a panel drag.
  for (const id of ["tlPlay", "tlLive"]) {
    for (const ev of ["mousedown", "touchstart", "pointerdown"]) {
      $(id).addEventListener(ev, (e) => e.stopPropagation());
    }
  }
  $("tlLive").onclick = goLive;
  $("tlPlay").onclick = togglePlay;
  updateModeUi();
  requestAnimationFrame(playbackTick);
}

// ---- status & controls ---------------------------------------------------------------
function onStatus(m) {
  if ("session" in m && m.session !== S.session) {
    S.session = m.session; // another recording was opened: start from scratch
    resetSession(m.live !== false);
    loadInfo();
  }
  if ("paused" in m) {
    S.paused = m.paused;
    updatePauseButton();
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
  if ("duration" in m) S.duration = m.duration;
  if ("ended" in m) S.ended = m.ended;
  if (m.error) toast(`Pipeline error: ${m.error}`);
  else if (m.ended && !S.follow) {
    toast("Analysis finished: the whole recording is on the timeline.", 5000);
    drawTimeline();
  } else if (m.ended) toast("Replay finished.", 4000);
  if ("loop" in m && m.loop !== S.epoch) {
    S.epoch = m.loop;
    S.lastT = 0; // a new run: old thumbnails belong to another recording
    THUMBS.clear();
    if (m.loop > 0) {
      resetHistory();
      goLive();
    }
  }
}

function resetKpis() {
  $("stateName").textContent = "—";
  $("stateName").style.color = "";
  $("confText").textContent = "—";
  $("confFill").style.width = "0%";
  $("candidate").textContent = "";
  $("reasons").innerHTML = "";
  $("switches").textContent = "—";
  renderBars({});
}

function resetSession(follow = true) {
  resetHistory();
  Object.assign(S, { lastT: 0, frame: null, liveFrame: null, baseline: null, imageReady: false, epoch: null });
  Object.assign(S, { duration: 0, ended: false });
  THUMBS.clear();
  $("toast").hidden = true;
  $("lookingObj").textContent = "—";
  $("lookingDwell").textContent = "—";
  const { ctx, w, h } = fit($("video"), 16 / 9);
  ctx.clearRect(0, 0, w, h);
  $("stageEmpty").textContent = "Waiting for the first frame…";
  setFollow(follow, true);
}

// Live replay follows the stream; analysis mode reviews it from the start while it fills in.
function setFollow(follow, force = false) {
  if (follow === S.follow && !force) return;
  S.follow = follow;
  $("speedBox").hidden = $("speedBox").hidden || !follow;
  if (follow) goLive();
  else enterReview();
}

function resetHistory() {
  S.states = [];
  S.features = [];
}

// Time going backwards means the replay restarted (or the page reconnected and the server
// is replaying the history): start the charts afresh.
function checkTime(t) {
  const last = Math.max(
    S.states.length ? S.states[S.states.length - 1].t : -Infinity,
    S.features.length ? S.features[S.features.length - 1].t : -Infinity,
  );
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

// Live: pauses the pipeline. Reviewing: plays/pauses the recording, like the timeline's button.
function updatePauseButton() {
  if (S.view.live) $("pause").textContent = S.paused ? "Resume" : "Pause";
  else $("pause").textContent = S.view.playing ? "Pause" : "Play";
}

$("pause").onclick = () => {
  if (S.view.live) control(S.paused ? "resume" : "pause");
  else togglePlay();
};
$("rebaseline").onclick = () => control("rebaseline");
$("speed").onchange = (e) => control("speed", parseFloat(e.target.value));
$("showBoxes").onchange = (e) => {
  S.showBoxes = e.target.checked;
  drawVideo();
};
document.addEventListener("keydown", (e) => {
  if (e.target !== document.body) return;
  if (e.code === "Space") {
    e.preventDefault();
    $("pause").click();
  } else if (e.code === "ArrowLeft" || e.code === "ArrowRight") {
    e.preventDefault();
    const step = (e.shiftKey ? 10 : 1) * (e.code === "ArrowLeft" ? -1 : 1);
    const from = S.view.live ? sessionEnd() : S.view.t;
    if (step > 0 && !S.view.live && from + step >= sessionEnd() && S.follow) goLive();
    else seek(from + step);
  } else if (e.code === "Escape" && !S.view.live) {
    goLive();
  }
});
// ---- websocket ------------------------------------------------------------------------
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => {
    if (S.reconnecting) $("toast").hidden = true;
    S.reconnecting = false;
  };
  ws.onclose = () => {
    if (!S.reconnecting) toast("Connection lost — reconnecting…");
    S.reconnecting = true;
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
    S.paused = info.paused;
    updatePauseButton();
    $("speedBox").hidden = info.mode !== "replay" || info.live === false; // only for real-time file replay
    if (info.source) setFollow(info.live !== false);
    $("openFile").hidden = !info.can_open;
    $("openFile").title = info.source ? `Open a VRS recording (now: ${info.source})` : "Open a VRS recording";
    document.title = info.source ? `AriaSense · ${info.source}` : "AriaSense";
    if (!info.source && !S.liveFrame) {
      resetKpis();
      $("calib").hidden = true;
      $("stageEmpty").textContent = info.can_open ? "No recording open. Use Open file to start." : "No source.";
    }
  } catch {}
}

// ---- open a recording ---------------------------------------------------------------
let chosenPath = "";

async function showOpenDialog() {
  const dlg = $("openDialog");
  $("openError").hidden = true;
  $("openPath").value = "";
  chosenPath = "";
  $("fileList").innerHTML = "";
  dlg.showModal();
  try {
    const { dir, files } = await (await fetch("/api/files")).json();
    $("openDir").textContent = `Recordings in ${dir}/`;
    $("fileList").innerHTML = files
      .map(
        (f) => `<li data-path="${escapeHtml(f.path)}"><span>${escapeHtml(f.name)}</span><span>${f.size_mb} MB</span></li>`,
      )
      .join("");
  } catch {
    $("openDir").textContent = "Could not list recordings.";
  }
}

async function openChosen() {
  const path = $("openPath").value.trim() || chosenPath;
  if (!path) {
    $("openError").textContent = "Pick a recording or type a path.";
    $("openError").hidden = false;
    return;
  }
  $("openGo").disabled = true;
  try {
    const r = await (
      await fetch("/api/open", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path, live: $("openLive").checked }),
      })
    ).json();
    if (!r.ok) throw new Error(r.error);
    $("openDialog").close();
  } catch (e) {
    $("openError").textContent = String(e.message || e);
    $("openError").hidden = false;
  } finally {
    $("openGo").disabled = false;
  }
}

$("openFile").onclick = showOpenDialog;
$("openCancel").onclick = () => $("openDialog").close();
$("openForm").onsubmit = (e) => {
  e.preventDefault();
  openChosen();
};
$("fileList").onclick = (e) => {
  const li = e.target.closest("li");
  if (!li) return;
  chosenPath = li.dataset.path;
  $("openPath").value = "";
  document.querySelectorAll("#fileList li").forEach((el) => el.classList.toggle("sel", el === li));
};
$("fileList").ondblclick = (e) => {
  if (e.target.closest("li")) openChosen();
};

// ---- grid layout (GridStack) --------------------------------------------------------
const GRID_ROWS = 22; // default layout height, in rows: the cell height scales so it fills the window
let grid = null;

function resizePlots() {
  for (const { plot, el } of Object.values(SPARKS)) {
    const w = Math.max(40, el.clientWidth);
    const h = Math.max(16, el.clientHeight);
    if (plot.width !== w || plot.height !== h) plot.setSize({ width: w, height: h });
  }
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
  const KEY = "ariasense-grid-v3";
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
buildLegend();
setupTimeline();
setupGrid();
drawPlane();
loadInfo();
connect();
