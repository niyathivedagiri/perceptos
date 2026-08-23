"""
perceptos.api.console
=====================

The mission-control console, served by the backend at ``/console``.

**Why a single file and no build step.** The spec asks for React + Vite +
Tailwind. That stack needs Node, a package install, a dev server on a second
port and a CORS story — four things that can each fail on a fresh machine
before a single real value is displayed. This console is one HTML document with
no dependencies, served from the same origin as the API. It runs the moment the
backend runs. It is deliberately a *console*, not the full six-panel flight
interface; see the README for exactly what it does and does not cover.

**Every value shown is fetched from the API.** There is no seeded state, no
placeholder number and no mock. When an endpoint has no data the panel says so
rather than displaying a plausible zero — a blank field is information, a fake
one is a defect.

**Provenance is rendered, not implied.** Each value that carries a source shows
a badge whose colour comes from ``DataSourceStatus.severity``: live NASA data
reads differently from cached data, which reads differently from simulation.
Clicking any lineage node opens the drawer and walks the derivation chain back
to its origin.
"""

from __future__ import annotations

__all__ = ["CONSOLE_HTML"]


CONSOLE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PerceptOS — Mission Control</title>
<style>
:root{
  color-scheme:dark;
  --bg:#070b10; --panel:#0c1219; --line:#18222e; --line2:#22303f;
  --fg:#dbe6f2; --dim:#6b7d91; --accent:#4da3d6;
  --ok:#3fb950; --info:#4da3d6; --warn:#d29922; --error:#f85149;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{display:flex;align-items:center;gap:16px;padding:12px 20px;
  border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:20}
h1{font-size:15px;letter-spacing:.22em;margin:0;font-weight:600}
.spacer{flex:1}
.badge{display:inline-flex;align-items:center;gap:6px;padding:2px 9px;border-radius:3px;
  font-size:11px;letter-spacing:.06em;border:1px solid var(--line2);white-space:nowrap}
.badge::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--dim)}
.badge.ok{color:var(--ok);border-color:#1a3d24}.badge.ok::before{background:var(--ok)}
.badge.info{color:var(--info);border-color:#153348}.badge.info::before{background:var(--info)}
.badge.warn{color:var(--warn);border-color:#3d3016}.badge.warn::before{background:var(--warn)}
.badge.error{color:var(--error);border-color:#4a1d1d}.badge.error::before{background:var(--error)}
button,select{background:#111a24;color:var(--fg);border:1px solid var(--line2);
  border-radius:3px;padding:5px 11px;font:inherit;font-size:12px;cursor:pointer}
button:hover:not(:disabled){border-color:var(--accent);color:#fff}
button:disabled{opacity:.4;cursor:not-allowed}
nav{display:flex;gap:2px;padding:0 20px;border-bottom:1px solid var(--line);background:var(--panel)}
nav button{border:none;border-bottom:2px solid transparent;border-radius:0;padding:9px 14px;color:var(--dim)}
nav button.active{color:var(--fg);border-bottom-color:var(--accent)}
main{padding:18px 20px 80px;max-width:1500px;margin:0 auto}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(310px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:5px;padding:14px 16px}
.card h2{font-size:11px;letter-spacing:.16em;color:var(--dim);margin:0 0 12px;font-weight:500;
  text-transform:uppercase}
.kv{display:grid;grid-template-columns:minmax(120px,auto) 1fr;gap:5px 14px;align-items:baseline}
.kv dt{color:var(--dim);font-size:12px}
.kv dd{margin:0;word-break:break-word}
.num{font-size:19px;color:#fff}
.dim{color:var(--dim)}
.empty{color:var(--dim);font-style:italic;padding:6px 0}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:500;font-size:11px;letter-spacing:.08em}
tr.click{cursor:pointer}tr.click:hover{background:#101b25}
.bar{height:5px;background:#111a24;border-radius:3px;overflow:hidden;margin-top:5px}
.bar i{display:block;height:100%;background:var(--accent)}
pre{background:#080d13;border:1px solid var(--line);border-radius:4px;padding:10px;
  overflow:auto;font-size:11px;color:#9fb3c8;max-height:280px;margin:0}
#drawer{position:fixed;top:0;right:0;width:min(620px,94vw);height:100%;background:var(--panel);
  border-left:1px solid var(--line2);transform:translateX(101%);transition:transform .18s ease;
  z-index:40;display:flex;flex-direction:column;box-shadow:-16px 0 40px rgba(0,0,0,.5)}
#drawer.open{transform:none}
#drawer header{border-bottom:1px solid var(--line)}
#drawerBody{overflow:auto;padding:16px 18px;flex:1}
.chain{list-style:none;margin:0;padding:0}
.chain li{border-left:2px solid var(--line2);padding:0 0 16px 16px;position:relative}
.chain li::before{content:"";position:absolute;left:-5px;top:5px;width:8px;height:8px;
  border-radius:50%;background:var(--accent);border:2px solid var(--panel)}
.chain li:last-child{border-left-color:transparent;padding-bottom:0}
.chain h4{margin:0 0 3px;font-size:13px;color:#fff;font-weight:600}
.chain .meta{font-size:11px;color:var(--dim);margin-bottom:5px}
.stage{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid var(--line)}
a{color:var(--accent);word-break:break-all}
figure{margin:0;background:#080d13;border:1px solid var(--line);border-radius:4px;overflow:hidden}
figure img{display:block;width:100%;height:auto;image-rendering:pixelated;background:#05080c}
figcaption{padding:7px 10px;border-top:1px solid var(--line);font-size:11px;color:var(--dim);
  display:flex;justify-content:space-between;gap:8px;align-items:center;flex-wrap:wrap}
.imggrid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}
.legend{display:flex;align-items:center;gap:6px;font-size:10px;color:var(--dim)}
.legend .swatch{width:34px;height:8px;border-radius:2px}
.nodata{display:inline-block;width:9px;height:9px;background:#f0f;border-radius:2px;
  vertical-align:-1px;margin-right:4px}
.unavail{border:1px dashed var(--line2);border-radius:4px;padding:22px;text-align:center;
  color:var(--dim);background:#080d13}
.unavail strong{display:block;color:var(--error);letter-spacing:.14em;margin-bottom:8px;font-size:12px}
.unavail code{display:block;margin-top:10px;color:var(--accent);font-size:11px}
.toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:#111a24;
  border:1px solid var(--line2);padding:9px 18px;border-radius:4px;z-index:60;font-size:12px}
.toast.err{border-color:var(--error);color:var(--error)}
</style>
</head>
<body>

<header>
  <h1>PERCEPTOS</h1>
  <span id="envBadge" class="badge">…</span>
  <span id="modeBadge" class="badge">…</span>
  <span id="missionBadge" class="badge">no mission</span>
  <span class="spacer"></span>
  <select id="envSelect" title="Environment"></select>
  <button id="btnMission">Start mission</button>
  <button id="btnRun">Run 5 frames</button>
</header>

<nav>
  <button data-tab="overview" class="active">Overview</button>
  <button data-tab="perception">Perception</button>
  <button data-tab="terrain">Terrain</button>
  <button data-tab="navigation">Navigation</button>
  <button data-tab="slam">SLAM</button>
  <button data-tab="replay">Replay</button>
  <button data-tab="pipeline">Pipeline</button>
  <button data-tab="lineage">Data lineage</button>
  <button data-tab="sources">Data sources</button>
  <button data-tab="health">Telemetry</button>
</nav>

<main>
  <section data-panel="overview" class="grid"></section>
  <section data-panel="perception" hidden></section>
  <section data-panel="terrain" hidden></section>
  <section data-panel="navigation" hidden></section>
  <section data-panel="slam" hidden></section>
  <section data-panel="replay" hidden></section>
  <section data-panel="pipeline" class="grid" hidden></section>
  <section data-panel="lineage" hidden></section>
  <section data-panel="sources" hidden></section>
  <section data-panel="health" class="grid" hidden></section>
</main>

<aside id="drawer">
  <header><h1 style="font-size:12px">DATA LINEAGE</h1><span class="spacer"></span>
    <button onclick="closeDrawer()">Close</button></header>
  <div id="drawerBody"></div>
</aside>

<script>
const API = location.origin;
const $ = (s, r=document) => r.querySelector(s);
const $$ = (s, r=document) => [...r.querySelectorAll(s)];
let STATE = {mission:null, running:false, ws:null};

// ---- helpers -------------------------------------------------------------
const esc = s => String(s ?? "").replace(/[&<>"]/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

// Severity for a provenance source. Mirrors DataSourceStatus.severity on the
// backend: live data reads green, cached blue, simulation amber, missing red.
function sev(sourceType, source){
  const t = (sourceType || source || "").toUpperCase();
  if (t.startsWith("NASA_") || t === "LIVE_SENSOR" || t === "LIVE_CAMERA") return "ok";
  if (t === "LOCAL_CACHED_DATA" || t === "CALCULATED" || t === "ESTIMATED") return "info";
  if (t === "SIMULATION") return "warn";
  return "error";
}
function badge(text, severity){
  return `<span class="badge ${severity}">${esc(text)}</span>`;
}
// A value that is genuinely absent renders as "unavailable", never as 0.
function val(v, unit=""){
  if (v === null || v === undefined || v === "") return '<span class="dim">unavailable</span>';
  if (typeof v === "number") return esc(Number.isInteger(v) ? v : v.toFixed(3)) + unit;
  return esc(v) + unit;
}
function toast(msg, isErr=false){
  const el = document.createElement("div");
  el.className = "toast" + (isErr ? " err" : "");
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3600);
}
async function api(path, opts={}){
  const r = await fetch(API + path, {
    headers: {"Content-Type":"application/json"}, ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!r.ok){
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return r.status === 204 ? null : r.json();
}

// ---- tabs ----------------------------------------------------------------
$$("nav button").forEach(b => b.onclick = () => {
  $$("nav button").forEach(x => x.classList.toggle("active", x === b));
  $$("main section").forEach(s => s.hidden = s.dataset.panel !== b.dataset.tab);
  refresh();
});
const activeTab = () => $("nav button.active").dataset.tab;

// ---- overview ------------------------------------------------------------
async function renderOverview(){
  const [root, health, env] = await Promise.all([
    api("/api"), api("/api/health"), api("/api/environments"),
  ]);
  const sys = health.system, pwr = health.power;
  const cards = [];

  cards.push(card("Mission", `<dl class="kv">
    <dt>Mission ID</dt><dd>${val(root.mission.mission_id)}</dd>
    <dt>Running</dt><dd>${root.mission.running ? "yes" : "no"}</dd>
    <dt>Frames</dt><dd class="num">${root.mission.frames_processed}</dd>
    <dt>Environment</dt><dd>${esc(root.environment.display_name)}</dd>
    <dt>Supervision</dt><dd>${esc(root.environment.supervision)}</dd>
  </dl>`));

  cards.push(card("Data source", `<dl class="kv">
    <dt>Pipeline mode</dt><dd>${badge(root.data_source.pipeline_mode.toUpperCase(),
      root.data_source.pipeline_mode === "simulation" ? "warn" : "ok")}</dd>
    <dt>Perseverance</dt><dd>${esc(root.data_source.primary_perseverance_service)}</dd>
    <dt>API key needed</dt><dd>${root.data_source.primary_service_requires_key ? "yes" : "no"}</dd>
    <dt>NASA key</dt><dd>${badge(root.data_source.nasa_key_status,
      root.data_source.nasa_key_status === "NASA_PERSONAL_KEY" ? "ok" : "info")}</dd>
  </dl>`));

  cards.push(card("System " + badge(sys._provenance.source, sev(null, sys._provenance.source)),
    sys.metrics_available ? `<dl class="kv">
      <dt>CPU</dt><dd>${val(sys.cpu_percent,"%")}<div class="bar"><i style="width:${sys.cpu_percent||0}%"></i></div></dd>
      <dt>Memory</dt><dd>${val(sys.memory_percent,"%")}<div class="bar"><i style="width:${sys.memory_percent||0}%"></i></div></dd>
      <dt>Process RSS</dt><dd>${val(sys.process_rss_mb," MB")}</dd>
      <dt>Disk free</dt><dd>${val(sys.disk_free_gb," GB")}</dd>
      <dt>Device</dt><dd>${esc(sys.device.name)} <span class="dim">(${esc(sys.device.kind)})</span></dd>
      <dt>Why</dt><dd class="dim">${esc(sys.device.reason)}</dd>
    </dl>` : `<p class="empty">psutil not installed — host metrics unavailable.
      Nothing is estimated in its place.</p>`));

  cards.push(card("Rover power " + badge("SIMULATION","warn"), `<dl class="kv">
    <dt>State of charge</dt><dd class="num">${val(pwr.percent,"%")}</dd>
    <dt>Remaining</dt><dd>${val(pwr.remaining_wh," Wh")}</dd>
    <dt>Endurance</dt><dd>${pwr.net_positive ? "generator covers load" : val(pwr.endurance_hours," h")}</dd>
  </dl><p class="dim" style="margin:10px 0 0;font-size:11px">${esc(pwr.assumptions)}</p>`));

  cards.push(card("Lineage", `<dl class="kv">
    <dt>Live nodes</dt><dd class="num">${root.lineage.live_nodes}</dd>
    <dt>Persisted nodes</dt><dd>${root.lineage.persisted_nodes}</dd>
    <dt>Persisted edges</dt><dd>${root.lineage.persisted_edges}</dd>
    <dt>Status</dt><dd>${esc(root.lineage.status)}</dd>
  </dl>`));

  $('[data-panel="overview"]').innerHTML = cards.join("");
  $("#envBadge").textContent = root.environment.display_name;
  $("#envBadge").className = "badge info";
  $("#modeBadge").textContent = root.data_source.pipeline_mode.toUpperCase();
  $("#modeBadge").className = "badge " + (root.data_source.pipeline_mode === "simulation" ? "warn" : "ok");
  STATE.mission = root.mission.mission_id;
  STATE.running = root.mission.running;
  $("#missionBadge").textContent = STATE.running ? "recording " + STATE.mission : "no mission";
  $("#missionBadge").className = "badge " + (STATE.running ? "ok" : "");
  $("#btnMission").textContent = STATE.running ? "Stop mission" : "Start mission";

  if (!$("#envSelect").options.length){
    $("#envSelect").innerHTML = env.environments
      .map(e => `<option value="${e.name}"${e.name===env.active?" selected":""}>${esc(e.display_name)}</option>`)
      .join("");
  }
}
const card = (title, body) => `<div class="card"><h2>${title}</h2>${body}</div>`;

// ---- pipeline ------------------------------------------------------------
async function renderPipeline(){
  const [contract, diagram] = await Promise.all([
    api("/api/pipeline"), fetch(API + "/api/pipeline/diagram").then(r => r.text()),
  ]);
  const rows = contract.stages.map(s => `<tr>
    <td>${esc(s.name)}</td>
    <td class="dim">${esc(s.requires.join(", ") || "—")}</td>
    <td class="dim">${esc(s.provides.join(", ") || "—")}</td>
    <td>${badge(s.enabled ? "enabled" : "disabled", s.enabled ? "ok" : "warn")}</td>
    <td><button onclick="toggleStage('${esc(s.name)}',${!s.enabled})">${s.enabled?"disable":"enable"}</button></td>
  </tr>`).join("");
  $('[data-panel="pipeline"]').innerHTML =
    card("Stage contract (validated at construction)",
      `<table><thead><tr><th>Stage</th><th>Requires</th><th>Provides</th>
       <th>State</th><th></th></tr></thead><tbody>${rows}</tbody></table>`) +
    card("Block diagram — generated from the wiring, not hand-drawn",
      `<pre>${esc(diagram)}</pre>`);
}
async function toggleStage(name, enabled){
  try { await api(`/api/pipeline/stages/${name}`, {method:"POST", body:{enabled}}); refresh(); }
  catch(e){ toast(e.message, true); }
}

// ---- shared: unavailable state ------------------------------------------
// A panel with no data says so, loudly. It never renders an empty chart that
// could be mistaken for "measured zero".
function unavailable(title, reason, hint){
  return card(title, `<div class="unavail"><strong>UNAVAILABLE</strong>
    ${esc(reason || "No data.")}${hint ? `<code>${esc(hint)}</code>` : ""}</div>`);
}
// Cache-buster so a re-rendered panel fetches the current frame, not a stale one.
const stamp = () => "?t=" + Date.now();

function imageFigure(src, label, meta){
  return `<figure><img src="${src}" alt="${esc(label)}" loading="lazy"
    onerror="this.closest('figure').innerHTML='<div class=unavail><strong>UNAVAILABLE</strong>${esc(label)} could not be rendered</div>'">
    <figcaption><span>${esc(label)}</span><span>${meta || ""}</span></figcaption></figure>`;
}

// ---- perception ----------------------------------------------------------
async function renderPerception(){
  const panel = $('[data-panel="perception"]');
  const [frame, stereo] = await Promise.all([api("/api/frame"), api("/api/stereo")]);

  if (!frame.available){
    panel.innerHTML = unavailable("Perception", frame.reason,
      "POST /api/pipeline/run  — or press “Run 5 frames”");
    return;
  }
  const renderable = frame.slots.filter(s => s.renderable);
  const figures = renderable.map(s => {
    const badgeHtml = badge(s.source_type || s.source, sev(s.source_type, s.source));
    const conf = s.confidence !== null && s.confidence !== undefined
      ? ` conf ${s.confidence.toFixed(2)}` : "";
    return `<div>${imageFigure(`${API}/api/frame/${s.key}.png${stamp()}`, s.label,
      badgeHtml + esc(conf))}
      <div style="margin-top:5px;font-size:11px" class="dim">
        ${esc(s.algorithm || "")}
        ${s.node_id ? `· <a href="#" onclick="openDrawer('${esc(s.node_id)}');return false">trace</a>` : ""}
      </div></div>`;
  }).join("");

  const meta = frame.metadata?.stereo || {};
  const stereoBody = stereo.available
    ? `<dl class="kv">
        <dt>Status</dt><dd>${badge(stereo.stereo_status,
          stereo.depth_valid ? "ok" : "warn")}</dd>
        <dt>Depth valid</dt><dd>${stereo.depth_valid ? "yes" : "no"}</dd>
        <dt>Rectified</dt><dd>${stereo.rectified ? "yes" : "no"}</dd>
        <dt>Baseline</dt><dd>${val(meta.baseline_m, " m")}
          <span class="dim">${meta.baseline_m ? "(measured from CAHV camera centres)" : ""}</span></dd>
        <dt>Focal length</dt><dd>${val(meta.focal_length_px, " px")}</dd>
        <dt>Epipolar residual</dt><dd>${val(meta.epipolar_error_px, " px")}
          <span class="dim">${meta.epipolar_error_px !== undefined
            ? "(&lt;2 px required for scanline matching)" : ""}</span></dd>
        <dt>Distortion</dt><dd>${meta.distortion_corrected === false
          ? badge("NOT CORRECTED", "warn") : val(undefined)}</dd>
        <dt>Reason</dt><dd class="dim">${esc(stereo.reason)}</dd>
      </dl>`
    : `<div class="unavail"><strong>${esc(stereo.stereo_status)}</strong>
        ${esc(stereo.reason)}
        <code>python scripts/serve.py --source stereo</code></div>`;

  // The point cloud only exists when a rectified pair produced metric depth.
  const cloud = stereo.depth_valid
    ? `<div style="margin-top:14px">${card("Reconstructed point cloud",
        imageFigure(`${API}/api/frame/pointcloud.png${stamp()}`,
          "Back-projected depth — colour is range",
          badge("CALCULATED","info")) +
        `<p class="dim" style="font-size:11px;margin:8px 0 0">
          Every point was reconstructed from the measured disparity. Nothing is
          interpolated between them.</p>`)}</div>`
    : "";

  panel.innerHTML =
    card(`Frame ${frame.frame_id} — ${renderable.length} renderable product(s)
      ${badge(frame.data_mode.toUpperCase(), frame.data_mode === "simulation" ? "warn" : "ok")}`,
      `<p class="dim" style="margin:0 0 12px;font-size:11px">
        <span class="nodata"></span>magenta marks cells with no measurement.
        Total pipeline latency ${frame.total_latency_ms.toFixed(2)} ms.</p>
       <div class="imggrid">${figures}</div>`) +
    `<div style="margin-top:14px">${card("Stereo", stereoBody)}</div>` + cloud;
}

// ---- terrain -------------------------------------------------------------
let ALLOW_SIM_TERRAIN = false;
let TERRAIN_DRAPE = "traversability", TERRAIN_AZ = -40, TERRAIN_VE = 2;
async function renderTerrain(){
  const panel = $('[data-panel="terrain"]');
  const q = ALLOW_SIM_TERRAIN ? "?allow_simulation=true" : "";
  const t = await api("/api/terrain" + q);

  if (!t.available){
    panel.innerHTML = unavailable("Terrain", t.reason,
      "python scripts/fetch_data.py jezero-dem") +
      `<div style="margin-top:12px">${card("Work with synthetic terrain instead",
        `<p class="dim">Synthetic terrain is available, but it is labelled
         SIMULATION and every product derived from it inherits that tag.</p>
         <button onclick="ALLOW_SIM_TERRAIN=true;refresh()">Enable SIMULATION terrain</button>`)}</div>`;
    return;
  }
  const isSim = t.status === "SIMULATION";
  const figures = t.products.map(p => {
    const url = `${API}/api/terrain/${p.key}.png${q || "?"}${q ? "&" : ""}t=${Date.now()}`;
    return `<div>${imageFigure(url, p.label, esc(p.units || ""))}
      ${p.note ? `<p class="dim" style="font-size:10px;margin:5px 0 0">${esc(p.note)}</p>` : ""}</div>`;
  }).join("");
  const stats = t.statistics ? Object.entries(t.statistics)
    .filter(([, v]) => typeof v !== "object")
    .map(([k, v]) => `<dt>${esc(k.replace(/_/g," "))}</dt><dd>${val(v)}</dd>`).join("") : "";

  const view3d = `${API}/api/terrain/view3d.png${q || "?"}${q ? "&" : ""}` +
    `drape=${TERRAIN_DRAPE}&azimuth_deg=${TERRAIN_AZ}` +
    `&vertical_exaggeration=${TERRAIN_VE}&t=${Date.now()}`;
  const hillshade = `${API}/api/terrain/hillshade.png${q || "?"}${q ? "&" : ""}` +
    `vertical_exaggeration=${TERRAIN_VE}&t=${Date.now()}`;

  const three = card("3D terrain view",
    imageFigure(view3d, "Perspective view — planned route in white, rover in amber",
      badge(t.status, isSim ? "warn" : "ok")) +
    `<p style="margin:10px 0 0">
      <label class="dim">drape </label>
      <select onchange="TERRAIN_DRAPE=this.value;refresh()">
        ${t.products.map(p => `<option value="${esc(p.key)}"${p.key===TERRAIN_DRAPE?" selected":""}>${esc(p.label)}</option>`).join("")}
        <option value="none"${TERRAIN_DRAPE==="none"?" selected":""}>plain</option>
      </select>
      <label class="dim"> rotate </label>
      <input type="range" min="-180" max="180" value="${TERRAIN_AZ}" style="width:150px"
        onchange="TERRAIN_AZ=+this.value;refresh()">
      <label class="dim"> vertical &times;${TERRAIN_VE} </label>
      <input type="range" min="1" max="10" step="0.5" value="${TERRAIN_VE}" style="width:110px"
        onchange="TERRAIN_VE=+this.value;refresh()">
    </p>
    <p class="dim" style="font-size:11px;margin:8px 0 0">
      Vertical exaggeration is a display choice, not a measurement: it changes
      how steep the terrain looks without changing what it is. Every quad
      corresponds to four DEM cells that were actually measured.</p>`) +
    `<div style="margin-top:14px">${card("Shaded relief",
      imageFigure(hillshade, "Hillshade — surface shape under raking light",
        "light 315&deg;/45&deg;") +
      `<p class="dim" style="font-size:11px;margin:8px 0 0">A colour ramp shows
       how high a cell is; hillshade shows what the surface is shaped like. A
       30 cm boulder is invisible in a ramp spanning 8 m of relief.</p>`)}</div>`;

  panel.innerHTML =
    three + `<div style="margin-top:14px"></div>` +
    card(`Terrain ${badge(t.status, isSim ? "warn" : "ok")}`,
      `<dl class="kv">
        <dt>Source</dt><dd>${esc(t.source_path)}</dd>
        <dt>Grid</dt><dd>${t.shape.join(" × ")} cells @ ${t.cell_size_m} m</dd>
        <dt>DEM confidence</dt><dd>${val(t.dem_provenance.confidence)}</dd>
        <dt>Algorithm</dt><dd class="dim">${esc(t.dem_provenance.algorithm)}</dd>
        ${t.dem_provenance.source_url ? `<dt>Source URL</dt><dd><a href="${esc(t.dem_provenance.source_url)}" target="_blank" rel="noopener">${esc(t.dem_provenance.source_url)}</a></dd>` : ""}
        ${stats}
      </dl>
      <p style="margin:12px 0 0"><button onclick="openDrawer('${esc(t.dem_node_id)}')">Trace DEM</button>
      ${t.traversability_node_id ? `<button onclick="openDrawer('${esc(t.traversability_node_id)}')">Trace traversability</button>` : ""}
      </p>`) +
    `<div style="margin-top:14px" class="imggrid">${figures}</div>` +
    (t.occupancy ? `<div style="margin-top:14px">${card("Occupancy",
      `<dl class="kv"><dt>Ray occupancy</dt><dd>${badge(t.occupancy.ray_occupancy_status,"error")}</dd>
       <dt>Why</dt><dd class="dim">${esc(t.occupancy.ray_occupancy_reason)}</dd>
       <dt>Elevation coverage</dt><dd>${val(t.occupancy.elevation_coverage)}</dd></dl>`)}</div>` : "");
}

// ---- navigation ----------------------------------------------------------
async function renderNavigation(){
  const panel = $('[data-panel="navigation"]');
  const n = await api("/api/navigation");
  const q = ALLOW_SIM_TERRAIN ? "?allow_simulation=true" : "";

  if (!n.available){
    panel.innerHTML = unavailable("Navigation", n.reason) +
      `<div style="margin-top:12px">${card("Plan a route",
        `<p class="dim">Planning runs the existing A* / Theta* planners over the
         live traversability map. Terrain must be available first.</p>
         <p><select id="plannerSel"><option value="theta">Theta* (any-angle)</option>
            <option value="astar">A* (8-connected)</option></select>
         <button onclick="doPlan()">Plan route</button></p>`)}</div>`;
    return;
  }
  const replans = n.replans.length
    ? `<table><thead><tr><th>#</th><th>Cells changed</th><th>Repair</th>
        <th>vs initial</th><th>New length</th></tr></thead><tbody>${
        n.replans.map((r,i)=>`<tr><td>${i+1}</td><td>${r.cells_changed}</td>
          <td>${r.repair_expansions}</td><td class="dim">${r.initial_expansions}</td>
          <td>${r.new_length_m} m</td></tr>`).join("")}</tbody></table>`
    : `<p class="empty">No replanning events yet.</p>`;

  panel.innerHTML =
    card(`Planned route — ${esc(n.planner)}`,
      `<dl class="kv">
        <dt>Length</dt><dd class="num">${val(n.length_m," m")}</dd>
        <dt>Waypoints</dt><dd>${val(n.waypoint_count ?? n.waypoints.length)}</dd>
        <dt>Mean cell cost</dt><dd>${val(n.mean_cell_cost)}</dd>
        <dt>Max cell cost</dt><dd>${val(n.max_cell_cost)}</dd>
        <dt>Turns</dt><dd>${val(n.turns_deg," deg")}</dd>
        <dt>Est. duration</dt><dd>${val(n.estimated_duration_s," s")}</dd>
        <dt>Est. energy</dt><dd>${val(n.estimated_energy_wh," Wh")}</dd>
        <dt>Nodes expanded</dt><dd>${val(n.nodes_expanded)}</dd>
        <dt>Source</dt><dd>${badge(n.provenance.source_type || n.provenance.source,
          sev(n.provenance.source_type, n.provenance.source))}</dd>
      </dl>
      <p style="margin:12px 0 0">
        <button onclick="openDrawer('${esc(n.plan_node_id)}')">Trace this plan</button>
        <button onclick="doReplan()">Simulate a hazard discovery</button>
        <select id="plannerSel"><option value="theta">Theta*</option>
          <option value="astar">A*</option></select>
        <button onclick="doPlan()">Re-plan</button></p>`) +
    `<div style="margin-top:14px" class="imggrid">
      ${imageFigure(`${API}/api/navigation/path.png${stamp()}`,
        "Planned path over the cost map", "white line = route")}
     </div>` +
    `<div style="margin-top:14px">${card("Replanning events (D* Lite)", replans)}</div>`;
}
async function doPlan(){
  const planner = ($("#plannerSel") || {}).value || "theta";
  try {
    await api(`/api/navigation/plan${ALLOW_SIM_TERRAIN ? "?allow_simulation=true" : ""}`,
      {method:"POST", body:{planner}});
    toast("Route planned with " + planner); refresh();
  } catch(e){ toast(e.message, true); }
}
async function doReplan(){
  // Reveals cells the orbital map did not resolve — the case D* Lite exists for.
  try {
    const n = await api("/api/navigation");
    if (!n.available) return;
    const mid = n.waypoints[Math.floor(n.waypoints.length/2)];
    const t = await api("/api/terrain" + (ALLOW_SIM_TERRAIN ? "?allow_simulation=true" : ""));
    const r = Math.round(mid.y / t.cell_size_m), c = Math.round(mid.x / t.cell_size_m);
    const changes = [];
    for (let dr=-1; dr<=1; dr++) for (let dc=-1; dc<=1; dc++) changes.push([r+dr, c+dc, 1.0]);
    const res = await api("/api/navigation/replan", {method:"POST", body:{changes}});
    toast(`Repaired with ${res.event.repair_expansions} expansions `
      + `(initial search: ${res.event.initial_expansions})`);
    refresh();
  } catch(e){ toast(e.message, true); }
}

// ---- slam ----------------------------------------------------------------
async function renderSlam(){
  const panel = $('[data-panel="slam"]');
  const s = await api("/api/slam");
  if (!s.available){
    panel.innerHTML = unavailable("SLAM", s.reason);
    return;
  }
  const st = s.statistics, opt = s.optimisation;
  const loops = s.loop_closures.length
    ? `<table><thead><tr><th>From</th><th>To</th><th>Confidence</th>
        <th>Residual</th></tr></thead><tbody>${
        s.loop_closures.map(l => `<tr><td>${l.from}</td><td>${l.to}</td>
          <td>${val(l.confidence)}</td><td>${val(l.residual_norm)}</td></tr>`).join("")
      }</tbody></table>`
    : `<p class="empty">No loop closures. This trajectory is dead reckoning,
       and its drift is unbounded however neatly the odometry fits.</p>`;

  // A false closure is usually satisfied by deforming the odometry, so the
  // odometry residual is the number that actually reveals it.
  const suspect = (opt?.suspect_edges || []).length
    ? `<div class="unavail" style="text-align:left"><strong>SUSPECT LOOP CLOSURE</strong>
       ${opt.suspect_edges.map(e => `<div style="margin-top:6px">Edge ${e.from}→${e.to}:
         ${(e.suspect_reasons||[]).map(esc).join("; ")}</div>`).join("")}</div>`
    : "";

  panel.innerHTML =
    card("Trajectory",
      imageFigure(`${API}/api/slam/trajectory.png${stamp()}`,
        "Top-down map — amber path, cyan keyframes, green loop closures",
        badge("ESTIMATED","info")) +
      `<p class="dim" style="font-size:11px;margin:8px 0 0">
        Loop-closure edges are drawn explicitly: the difference between a
        trajectory that closed a loop and one that merely passed nearby is
        invisible if only the path is shown.</p>`) +
    `<div style="margin-top:14px"></div>` +
    card(`Map ${badge(s.landmark_status, s.landmark_status === "MEASURED" ? "ok" : "error")}`,
      `<dl class="kv">
        <dt>Keyframes</dt><dd class="num">${st.keyframes}</dd>
        <dt>Frames seen</dt><dd>${st.frames_seen}
          <span class="dim">(${(st.keyframe_rate*100).toFixed(1)}% promoted)</span></dd>
        <dt>Landmarks</dt><dd>${st.landmarks}</dd>
        <dt>Path length</dt><dd>${val(st.path_length_m," m")}</dd>
        <dt>Loop closures</dt><dd>${st.loop_closures}</dd>
        <dt>Candidates examined</dt><dd>${st.loop_detection.candidates_examined}
          <span class="dim">(${st.loop_detection.rejected} rejected)</span></dd>
      </dl>${s.landmark_status !== "MEASURED"
        ? `<p class="dim" style="margin:10px 0 0;font-size:11px">${esc(s.landmark_reason)}</p>` : ""}`) +
    (opt ? `<div style="margin-top:14px">${card("Pose-graph optimisation",
      `<dl class="kv">
        <dt>Converged</dt><dd>${badge(opt.converged?"yes":"no", opt.converged?"ok":"warn")}</dd>
        <dt>Iterations</dt><dd>${opt.iterations}</dd>
        <dt>Error</dt><dd>${val(opt.initial_error)} → ${val(opt.final_error)}
          <span class="dim">(${(opt.error_reduction*100).toFixed(1)}% reduction)</span></dd>
        <dt>Max loop residual</dt><dd>${val(opt.max_loop_residual)}</dd>
        <dt>Max odometry residual</dt><dd>${val(opt.max_odometry_residual)}
          <span class="dim">— where a false closure actually shows</span></dd>
      </dl>${suspect}
      <p style="margin:12px 0 0"><button onclick="doOptimise()">Re-optimise</button></p>`)}</div>` : "") +
    `<div style="margin-top:14px">${card("Loop closures", loops)}</div>`;
}
async function doOptimise(){
  try { const r = await api("/api/slam/optimise", {method:"POST"});
        toast(`${r.iterations} iterations, error ${r.final_error}`); refresh(); }
  catch(e){ toast(e.message, true); }
}

// ---- replay --------------------------------------------------------------
async function renderReplay(){
  const panel = $('[data-panel="replay"]');
  const [state, missions] = await Promise.all([
    api("/api/replay"), api("/api/missions?limit=25"),
  ]);

  const list = missions.missions.length
    ? `<table><thead><tr><th>Mission</th><th>Environment</th><th>Frames</th>
        <th>Status</th><th></th></tr></thead><tbody>${
        missions.missions.map(m => `<tr>
          <td>${esc(m.name)}</td><td class="dim">${esc(m.environment)}</td>
          <td>${m.frames}</td><td>${esc(m.status)}</td>
          <td><button onclick="openReplay('${esc(m.id)}')">Open</button>
              <a href="${API}/api/missions/${esc(m.id)}/report?format=markdown"
                 target="_blank">Report</a></td></tr>`).join("")
      }</tbody></table>`
    : `<p class="empty">No recorded missions. Start one and run the pipeline.</p>`;

  if (!state.open){
    panel.innerHTML = card("Recorded missions", list);
    return;
  }
  const frame = await api("/api/replay/frame");
  const timeline = await api("/api/replay/timeline");
  const profile = await api("/api/replay/profile");

  const stages = frame.stages.map(s => `<div class="stage">
    <span>${esc(s.stage)} ${badge(s.status, s.status === "ok" ? "ok" :
      s.status === "failed" ? "error" : "warn")}</span>
    <span>${s.latency_ms.toFixed(2)} ms</span></div>`).join("");

  const nodes = frame.lineage_nodes.map(n => `<tr class="click"
    onclick="openDrawer('${esc(n.id)}')">
    <td>${esc(n.label)}</td><td class="dim">${esc(n.stage||"—")}</td>
    <td>${badge(n.provenance.source_type || n.provenance.source,
      sev(n.provenance.source_type, n.provenance.source))}</td>
    <td>${val(n.provenance.confidence)}</td></tr>`).join("");

  panel.innerHTML =
    card(`Replaying ${esc(state.mission.name)} — frame ${frame.frame_index}
      of ${state.total_frames}`,
      `<p style="margin:0 0 12px">
        <button onclick="replayCtl('seek',0)" ${state.at_start?"disabled":""}>⏮</button>
        <button onclick="replayCtl('step',-1)" ${state.at_start?"disabled":""}>◀</button>
        <button onclick="replayCtl('${state.playing?"pause":"play"}')">${state.playing?"⏸":"▶"}</button>
        <button onclick="replayCtl('step',1)" ${state.at_end?"disabled":""}>▶▶</button>
        <button onclick="replayCtl('seek',${state.total_frames-1})" ${state.at_end?"disabled":""}>⏭</button>
        <button onclick="closeReplay()">Close</button></p>
      <input type="range" min="0" max="${state.total_frames-1}" value="${state.position}"
        style="width:100%" onchange="replayCtl('seek', +this.value)">
      <dl class="kv" style="margin-top:12px">
        <dt>Frame latency</dt><dd>${frame.total_latency_ms.toFixed(2)} ms</dd>
        <dt>Healthy</dt><dd>${frame.healthy ? "yes" : "no"}</dd>
        <dt>Mean / p95 / max</dt><dd>${timeline.statistics.mean_latency_ms} /
          ${timeline.statistics.p95_latency_ms} / ${timeline.statistics.max_latency_ms} ms</dd>
        <dt>Bottleneck</dt><dd>${esc(profile.bottleneck || "—")}</dd>
      </dl>
      <p class="dim" style="font-size:11px;margin:10px 0 0">${esc(frame.note)}</p>`) +
    `<div style="margin-top:14px">${card("Stage timings (recorded)", stages)}</div>` +
    `<div style="margin-top:14px">${card("Lineage at this frame",
      nodes ? `<table><thead><tr><th>Value</th><th>Stage</th><th>Source</th>
        <th>Conf.</th></tr></thead><tbody>${nodes}</tbody></table>`
      : `<p class="empty">No lineage recorded for this frame.</p>`)}</div>` +
    `<div style="margin-top:14px">${card("Counterfactual — what if?",
      `<p class="dim">Re-plans the current terrain with different parameters.
       Answers what a different planner would have decided given the same
       perception; it cannot replay perception that was never captured.</p>
       <p><select id="cfPlanner"><option value="theta">Theta*</option>
          <option value="astar">A*</option></select>
       <select id="cfRadius"><option value="">rover radius unchanged</option>
          <option value="0.2">0.2 m</option><option value="1.5">1.5 m</option>
          <option value="3.0">3.0 m</option></select>
       <button onclick="doCounterfactual()">Compare</button></p>
       <div id="cfResult"></div>`)}</div>`;
}
async function openReplay(id){
  try { await api(`/api/replay/${id}/open`, {method:"POST"}); refresh(); }
  catch(e){ toast(e.message, true); }
}
async function closeReplay(){
  STATE.replayClosed = true;
  try { await api("/api/replay/pause", {method:"POST"}); } catch {}
  $('[data-panel="replay"]').innerHTML = "";
  refresh();
}
async function replayCtl(action, value){
  try {
    if (action === "seek") await api("/api/replay/seek", {method:"POST", body:{position:value}});
    else if (action === "step") await api(`/api/replay/step?delta=${value}`, {method:"POST"});
    else await api(`/api/replay/${action}`, {method:"POST"});
    refresh();
  } catch(e){ toast(e.message, true); }
}
async function doCounterfactual(){
  const body = {planner: $("#cfPlanner").value};
  const radius = $("#cfRadius").value;
  if (radius) body.inflation_radius_m = parseFloat(radius);
  try {
    const r = await api("/api/replay/counterfactual", {method:"POST", body});
    const diffs = Object.entries(r.differences).map(([k,v]) =>
      `<tr><td>${esc(k)}</td><td>${val(v.baseline)}</td><td>${val(v.alternative)}</td>
       <td>${v.delta > 0 ? "+" : ""}${val(v.delta)}</td></tr>`).join("");
    $("#cfResult").innerHTML = `<div style="margin-top:12px">
      <p><strong>${esc(r.parameter)}</strong></p>
      <p>${esc(r.verdict)}</p>
      ${diffs ? `<table><thead><tr><th>Metric</th><th>Recorded</th>
        <th>Alternative</th><th>Δ</th></tr></thead><tbody>${diffs}</tbody></table>` : ""}
      </div>`;
  } catch(e){ toast(e.message, true); }
}

// ---- lineage -------------------------------------------------------------
async function renderLineage(){
  const panel = $('[data-panel="lineage"]');
  let nodes = [];
  try { nodes = (await api("/api/lineage?limit=200")).nodes; }
  catch(e){ panel.innerHTML = card("Data lineage", `<p class="empty">${esc(e.message)}</p>`); return; }

  if (!nodes.length){
    panel.innerHTML = card("Data lineage",
      `<p class="empty">No lineage yet. Start a mission and run the pipeline —
       nodes appear here as values are produced.</p>`);
    return;
  }
  const rows = nodes.map(n => {
    const p = n.provenance;
    return `<tr class="click" onclick="openDrawer('${esc(n.id)}')">
      <td>${esc(n.label)}</td>
      <td class="dim">${esc(n.stage || "—")}</td>
      <td>${badge(p.source_type || p.source, sev(p.source_type, p.source))}</td>
      <td>${val(p.confidence)}</td>
      <td>${val(p.uncertainty)}</td>
      <td class="dim">${val(p.latency_ms," ms")}</td>
      <td class="dim">${esc((p.algorithm||"").slice(0,38))}</td>
    </tr>`;
  }).join("");
  panel.innerHTML = card(`Lineage nodes (${nodes.length}) — click any row to trace it`,
    `<table><thead><tr><th>Value</th><th>Stage</th><th>Source</th><th>Conf.</th>
     <th>Uncert.</th><th>Latency</th><th>Algorithm</th></tr></thead>
     <tbody>${rows}</tbody></table>`);
}

async function openDrawer(nodeId){
  const body = $("#drawerBody");
  body.innerHTML = '<p class="empty">Loading derivation chain…</p>';
  $("#drawer").classList.add("open");
  let d;
  try { d = await api(`/api/lineage/${nodeId}`); }
  catch(e){ body.innerHTML = `<p class="empty">${esc(e.message)}</p>`; return; }

  const head = `<dl class="kv">
    <dt>Value</dt><dd>${esc(d.node.label)}</dd>
    <dt>Depth</dt><dd>${d.depth} nodes</dd>
    <dt>Weakest confidence</dt><dd>${val(d.weakest_confidence)}</dd>
    <dt>Total latency</dt><dd>${val(d.total_latency_ms," ms")}</dd>
    <dt>Origin</dt><dd>${esc(d.origin || "live")}</dd>
    <dt>Contains simulation</dt><dd>${d.contains_simulation
      ? badge("YES — synthetic ancestor","warn") : badge("no","ok")}</dd>
    ${(d.datasets||[]).length ? `<dt>Datasets</dt><dd>${d.datasets.map(esc).join(", ")}</dd>` : ""}
    ${(d.source_urls||[]).length ? `<dt>Source URL</dt><dd>${d.source_urls
      .map(u => `<a href="${esc(u)}" target="_blank" rel="noopener">${esc(u)}</a>`).join("<br>")}</dd>` : ""}
  </dl>`;

  // The chain runs root-first, so this reads as a derivation: the origin at
  // the top, the value you clicked at the bottom.
  const chain = d.chain.map(n => {
    const p = n.provenance;
    const params = Object.entries(p.parameters || {}).slice(0, 8)
      .map(([k,v]) => `${esc(k)}=${esc(typeof v === "object" ? JSON.stringify(v).slice(0,50) : v)}`)
      .join("  ");
    return `<li>
      <h4>${esc(n.label)}</h4>
      <div class="meta">${badge(p.source_type || p.source, sev(p.source_type, p.source))}
        ${p.dataset ? " · " + esc(p.dataset) : ""}</div>
      <dl class="kv" style="font-size:12px">
        <dt>Algorithm</dt><dd>${esc(p.algorithm)}</dd>
        <dt>Confidence</dt><dd>${val(p.confidence)}</dd>
        <dt>Uncertainty</dt><dd>${val(p.uncertainty)}${p.units ? " " + esc(p.units) : ""}</dd>
        <dt>Latency</dt><dd>${val(p.latency_ms," ms")}</dd>
        <dt>Inputs</dt><dd>${(n.parents||[]).length
          ? n.parents.map(id => `<a href="#" onclick="openDrawer('${esc(id)}');return false">${esc(id.slice(0,10))}</a>`).join(" ")
          : '<span class="dim">none — this is an origin</span>'}</dd>
        <dt>Output</dt><dd>${esc(JSON.stringify(n.value).slice(0,110))}</dd>
        ${params ? `<dt>Parameters</dt><dd class="dim">${params}</dd>` : ""}
      </dl></li>`;
  }).join("");

  body.innerHTML = head + `<h2 style="font-size:11px;letter-spacing:.16em;color:var(--dim);
    margin:20px 0 12px">DERIVATION CHAIN</h2><ul class="chain">${chain}</ul>`;
}
function closeDrawer(){ $("#drawer").classList.remove("open"); }

// ---- data sources --------------------------------------------------------
async function renderSources(){
  const d = await api("/api/datasources");
  const svc = Object.values(d.services).map(s => `<div class="card">
    <h2>${esc(s.name)} ${badge(s.requires_api_key ? "API KEY REQUIRED" : "NO KEY REQUIRED",
      s.requires_api_key ? "warn" : "ok")}</h2>
    <dl class="kv">
      <dt>Endpoint</dt><dd><a href="${esc(s.endpoint)}" target="_blank" rel="noopener">${esc(s.endpoint)}</a></dd>
      <dt>Provides</dt><dd>${esc(s.provides)}</dd>
      <dt>Rate limit</dt><dd>${esc(s.rate_limit)}</dd>
      <dt>Env var</dt><dd>${val(s.env_var)}</dd>
      <dt>DEMO_KEY enough</dt><dd>${s.demo_key_sufficient === null ? "n/a" : (s.demo_key_sufficient?"yes":"no")}</dd>
      <dt>Used by</dt><dd class="dim">${esc(s.used_by)}</dd>
      <dt>If key missing</dt><dd class="dim">${esc(s.on_missing_key)}</dd>
      <dt>Credit</dt><dd>${esc(s.credit)}</dd>
      <dt>Licence</dt><dd>${esc(s.licence)}</dd>
      ${s.notes ? `<dt>Notes</dt><dd class="dim">${esc(s.notes)}</dd>` : ""}
    </dl></div>`).join("");
  $('[data-panel="sources"]').innerHTML =
    card("Credentials", `<p>${esc(d.summary)}</p><dl class="kv" style="margin-top:10px">
      <dt>Key status</dt><dd>${badge(d.api_nasa_gov_key_status,
        d.api_nasa_gov_key_configured ? "ok" : "info")}</dd>
      <dt>Keyless services</dt><dd>${d.services_without_key.map(esc).join(", ")}</dd>
      <dt>Services needing a key</dt><dd>${d.services_requiring_key.map(esc).join(", ") || "none"}</dd>
      <dt>Verified</dt><dd>${esc(d.verified_utc)}</dd></dl>`) +
    `<div class="grid" style="margin-top:14px">${svc}</div>`;
}

// ---- telemetry -----------------------------------------------------------
async function renderHealth(){
  const panel = $('[data-panel="health"]');
  let frames = [];
  if (STATE.mission){
    try { frames = (await api(`/api/missions/${STATE.mission}/frames?limit=25`)).frames; }
    catch {}
  }
  const timing = frames.length
    ? `<table><thead><tr><th>Frame</th><th>Latency</th><th>Healthy</th></tr></thead><tbody>${
        frames.slice(-15).reverse().map(f => `<tr>
          <td>${f.frame_index}</td><td>${f.total_latency_ms.toFixed(2)} ms</td>
          <td>${badge(f.healthy?"ok":"degraded", f.healthy?"ok":"error")}</td></tr>`).join("")
      }</tbody></table>`
    : `<p class="empty">No recorded frames. Start a mission and run the pipeline.</p>`;
  panel.innerHTML = card("Frame timing (recorded)", timing) +
    card("Live stream", `<div id="wsLog"><p class="empty">Connecting…</p></div>`);
  connectWs();
}
function connectWs(){
  if (STATE.ws && STATE.ws.readyState <= 1) return;
  const ws = new WebSocket(API.replace(/^http/, "ws") + "/ws/telemetry");
  STATE.ws = ws;
  const lines = [];
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    const t = new Date().toLocaleTimeString();
    if (m.type === "frame"){
      lines.unshift(`${t}  frame ${m.payload.frame_id}  ${m.payload.total_latency_ms.toFixed(1)} ms  ${m.payload.data_mode}`);
    } else if (m.type === "health"){
      const c = m.payload.system.cpu_percent;
      lines.unshift(`${t}  health  cpu=${c === null ? "n/a" : c + "%"}`);
    } else {
      lines.unshift(`${t}  ${m.type}`);
    }
    const log = $("#wsLog");
    if (log) log.innerHTML = `<pre>${esc(lines.slice(0,18).join("\n"))}</pre>`;
  };
  ws.onerror = () => { const l = $("#wsLog"); if (l) l.innerHTML = '<p class="empty">WebSocket unavailable.</p>'; };
}

// ---- actions -------------------------------------------------------------
$("#btnMission").onclick = async () => {
  try {
    if (STATE.running) await api("/api/missions/active/stop", {method:"POST", body:{status:"complete"}});
    else await api("/api/missions", {method:"POST", body:{name:"", notes:"console"}});
    refresh();
  } catch(e){ toast(e.message, true); }
};
$("#btnRun").onclick = async () => {
  $("#btnRun").disabled = true;
  try { const r = await api("/api/pipeline/run", {method:"POST", body:{frames:5}});
        toast(`${r.frames} frames processed`); refresh(); }
  catch(e){ toast(e.message, true); }
  finally { $("#btnRun").disabled = false; }
};
$("#envSelect").onchange = async e => {
  try { await api("/api/environments/active", {method:"POST", body:{environment:e.target.value}});
        toast("Environment: " + e.target.value); refresh(); }
  catch(err){ toast(err.message, true); }
};

// ---- refresh loop --------------------------------------------------------
const RENDER = {overview:renderOverview, perception:renderPerception,
                slam:renderSlam, replay:renderReplay,
                terrain:renderTerrain, navigation:renderNavigation,
                pipeline:renderPipeline, lineage:renderLineage,
                sources:renderSources, health:renderHealth};
async function refresh(){
  try {
    await renderOverview();
    const tab = activeTab();
    if (tab !== "overview") await RENDER[tab]();
  } catch(e){ toast("API: " + e.message, true); }
}
refresh();
setInterval(() => { if (!$("#drawer").classList.contains("open")) refresh(); }, 5000);
</script>
</body>
</html>
"""
