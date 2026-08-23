# PerceptOS — Build Plan

Version 1.0 · Phases 0–8 · data sourcing, gates, effort, risk

---

## How to read this

Each phase has five fixed sections:

| Section | Meaning |
|---|---|
| **Goal** | The one sentence that says why this phase exists |
| **Data** | Where the bytes come from. No phase starts without this answered |
| **Build** | Modules, in dependency order |
| **Gate** | Objective, testable conditions. You do not start the next phase until these pass |
| **Effort** | Focused hours, and calendar weeks at ~10 hrs/week |

Effort is honest rather than optimistic. It assumes you write the code yourself
with help, not that it appears fully formed. Phases 1 and 2 took roughly the
hours listed below when built with assistance.

**Total: ~330 focused hours, ~33 weeks at 10 hrs/week.** Phases 0–5 give you a
portfolio-complete system at ~200 hours; 6–8 are the polish that separates
"impressive project" from "looks like a product".

---

# Part 1 — The data strategy

This is the part your original spec left open, and it decides what is
buildable. Everything else follows from it.

## The three-tier rule

Every module must work at all three tiers, and must declare which tier it is
running at. This is already enforced by the provenance layer — a value derived
from Tier 1 data carries `SIMULATION` and cannot be laundered into anything
else.

| Tier | Source | Provenance tag | Purpose |
|---|---|---|---|
| **1 — Synthetic** | Generated in-process | `SIMULATION` | CI, tests, development without hardware. Already built (`SyntheticSource`, `synthetic_terrain`) |
| **2 — Recorded** | Public mission and benchmark datasets | `LIVE_CAMERA` / `CALCULATED` | The demo you show people. Real Mars imagery, real elevation, real ground truth |
| **3 — Live** | Your own camera or stereo rig | `LIVE_CAMERA` / `LIVE_SENSOR` | Proves the acquisition path works end to end on real hardware |

Tier 2 is where the credibility comes from. A recruiter who sees your
traversability map running on the actual Jezero crater DEM that Perseverance
navigates by reacts differently to one running on `numpy.random`.

## Where the data comes from

### Mars — the flagship mode

**Rover surface imagery (Tier 2).** NASA publishes Perseverance's raw images
continuously, including left/right Navcam and Hazcam frames. Those pairs are
genuine stereo — the same imagery the rover's own visodom and autonav use.
<cite index="11-1">Perseverance's wider Navcam field of view means the onboard navigation software needs only a single stereo Navcam acquisition rather than the three required on Curiosity.</cite>

- Browse: `https://mars.nasa.gov/mars2020/multimedia/raw-images/`
- JSON API: `https://api.nasa.gov/mars-photos/api/v1/rovers/perseverance/photos?sol=N&camera=NAVCAM_LEFT`
  (free API key from api.nasa.gov; `DEMO_KEY` works for light use)
- Camera codes you want: `NAVCAM_LEFT` / `NAVCAM_RIGHT` (stereo pairs),
  `FRONT_HAZCAM_LEFT_A` / `_RIGHT_A` (near-field hazards), `MCZ_LEFT` / `MCZ_RIGHT`
  (Mastcam-Z, science targets)
- Licence: NASA imagery is public domain. Credit "NASA/JPL-Caltech".

**Terrain labels (Tier 2).** AI4Mars is the dataset for terrain
classification, built by JPL for exactly this problem.
<cite index="10-1">Version 0.6 contains roughly 425,000 semantic segmentation labels across about 50,000 images from Perseverance, Curiosity, Opportunity and Spirit, crowdsourced with each image labelled by five to ten people, plus around 1,500 validation labels annotated by NASA rover planners and mission scientists.</cite>
<cite index="10-1">The labels are greyscale images with class pixel values 0–3.</cite>

- Zenodo record `15995036`, file `ai4mars-dataset-merged-0.6.zip`, ~17.8 GB
- Classes: soil, bedrock, sand, big rock — the last two being the ones that
  actually threaten a rover
- This is your training set for the terrain classification module in Phase 5

**Elevation (Tier 2) — the best find.** USGS publishes HiRISE-derived digital
terrain models of Jezero crater at 1 m/pixel, produced specifically to support
Perseverance operations.
<cite index="21-1">The 2024 Mars Sample Return mosaics are aligned to the existing Jezero DEMs and are fit for rover navigation and hazard mapping.</cite>
<cite index="27-1">A wider archive of HiRISE DTMs is available as Cloud Optimized GeoTIFFs on the AWS Registry of Open Data, controlled to MOLA so they act as a geodetic reference frame.</cite>

- Jezero 1 m DEM: `astrogeology.usgs.gov/search/map/Mars/Mars2020/JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40`
- Matching 25 cm orthoimage mosaic (the visual layer for your 3D view):
  `.../JEZ_hirise_soc_006_orthoMosaic_25cm_Eqc_latTs0_lon0_first`
- Wider archive: `registry.opendata.aws/nasa-usgs-mars-hirise-dtms/` — COGs
  support ranged reads, so you can pull a 500 m tile without the whole scene
- Coarser fallback: CTX DEM mosaic at 20 m/pixel, MOLA global at 463 m/pixel

Feed a Jezero DEM tile into `TraversabilityAnalyser` and the slope, step and
roughness hazards you get out are the real hazards on the real crater floor.
That single substitution converts the Mars mode from a simulation into an
analysis.

### Stereo — the module with no free lunch

You have no stereo rig, and stereo is load-bearing for depth, reconstruction
and visual odometry scale. Three options, in increasing order of cost:

| Option | Cost | What you get | Verdict |
|---|---|---|---|
| Middlebury Stereo 2014/2021 | Free | Rectified pairs with ground-truth disparity | **Start here.** Ground truth lets you *measure* your matcher's error, not just eyeball it |
| Perseverance Navcam L/R pairs | Free | Real Mars stereo, no ground truth, needs rectification | Second. The visual payoff is high |
| ETH3D / KITTI | Free, registration | Outdoor stereo + odometry ground truth | KITTI is CC BY-NC-SA — fine for a portfolio, check before any commercial use |
| Two USB webcams on a rigid bar | ~₹3,000 | Your own live stereo, poor baseline stability | Cheap but the calibration will fight you |
| OAK-D Lite | ~₹12,000 | Calibrated stereo + depth, USB-C, macOS support | **The right buy if you buy one.** Removes the calibration problem entirely |
| Intel RealSense D435i | ~₹25,000 | Calibrated stereo + IMU | Better, more expensive, macOS support is patchier |

My recommendation: Middlebury for correctness testing, Perseverance pairs for
the demo, and an OAK-D Lite only if you want Tier 3 to be genuinely live. The
project is complete and defensible without buying anything.

### Calibration (Tier 3, free)

Print a chessboard or ChArUco target on A4, glue it to something rigid — a
clipboard or a sheet of foam board. Warping is the single biggest source of
calibration error, and paper on a desk warps.

- Pattern: 9×6 inner corners, 25 mm squares. Not square — a square
  inner-corner grid is rotationally ambiguous and your calibrator already
  rejects it
- Capture 15–20 views, deliberately including the frame corners and strong
  tilts. Your quality gate checks coverage and orientation spread precisely
  because centred fronto-parallel views produce a beautiful RMS and wrong
  distortion coefficients

### Other modes

| Mode | Data source | Notes |
|---|---|---|
| Earth Inspection | Your webcam + COCO-pretrained YOLOv11 | Zero setup, works today |
| Lunar Exploration | LOLA global DEM (~118 m/px), LRO NAC DTMs (~2 m/px) via PDS Geosciences | Same pipeline as Mars, different `RoverGeometry` and gravity |
| Warehouse Robotics | LOCO (logistics objects) or your own phone footage | Flat floors, so traversability off, tracking on |
| Industrial Inspection | Your own footage of pipes/panels/machinery | Honest and easy to record |
| Drone Inspection | VisDrone or UAVid | Aerial viewpoint stresses the tracker differently |

## Data engineering rules

These are worth fixing now, before there are 40 GB on your disk.

1. **Nothing above 10 MB goes in git.** Not one image. `datasets/` and
   `models/` are already in `.gitignore`.
2. **Write a downloader, not a folder.** `scripts/fetch_data.py <name>` pulls
   a named dataset, verifies its checksum, and writes a manifest. The repo
   ships the *recipe*; anyone cloning it can reproduce your data.
3. **Ship a tiny curated sample.** ~20 MB in `datasets/samples/` — three
   Navcam stereo pairs, one 200×200 Jezero DEM tile, ten AI4Mars images. The
   demos run from this out of the box, with no download.
4. **Every dataset gets a `SOURCE.md`** recording the URL, licence, download
   date, checksum and citation. A portfolio project that can't say where its
   data came from undercuts the entire provenance argument it is making.
5. **Cache derived products.** Rectifying a Navcam pair or reprojecting a DEM
   tile is slow and deterministic — do it once into `datasets/derived/`.
6. **Storage budget:** ~60 GB if you take AI4Mars in full, ~15 GB if you take
   the Perseverance subset only. Plan for an external drive or a partial pull.

---

# Part 2 — The phases

## Phase 0 — Foundations ✅ complete

**Goal.** Decide the non-negotiables before writing modules against them.

Provenance contract, five source classes, confidence propagation by minimum,
simulation contamination, lineage DAG. These are architectural: bolting them on
after sixteen modules means rewriting sixteen modules.

**Effort.** ~12 h.

---

## Phase 1 — Perception kernel ✅ complete

**Goal.** Everything that must be right before any pixel is touched.

**Data.** Tier 1 only — synthetic DEM, synthetic correspondences.

**Built.** `core/provenance`, `core/lineage`, `core/stage`, `core/pipeline`,
`core/compute`, `estimation/kalman`, `stereo/depth`, `servoing/ibvs`,
`mars/occupancy`, `mars/traversability`, `mars/planner`, `mars/slip`.

**Gate.** ✅ 74 tests. Runs on NumPy alone. Autonomy chain executes end to end.

**Effort.** ~45 h.

---

## Phase 2 — Vision layer ✅ complete

**Goal.** Real pixels through real OpenCV, as thin stages over the kernel.

**Data.** Tier 1 synthetic camera with ego-motion; synthetic chessboards.

**Built.** `core/cv_compat`, `sensors/camera`, `calibration/chessboard`,
`imageproc/filters`, `features/extractors`, `stereo/matcher`, `detection/yolo`,
`tracking/trackers`, `mars/odometry`.

**Gate.** ✅ 148 tests. Full pipeline at ~27 fps ceiling on CPU. Missing YOLO
degrades to SKIPPED rather than crashing.

**Effort.** ~55 h.

---

## Phase 3 — Data acquisition and the service layer

**Goal.** Get real data in, and get the pipeline out over HTTP.

This phase is deliberately split from the UI. Building the frontend against an
API that doesn't exist yet is how projects end up with beautiful mocked
dashboards, which is the exact failure mode your spec forbids.

### Data

Tier 2 arrives here. Build `scripts/fetch_data.py` first — before any API
code — with three fetchers:

- `perseverance` — hits the NASA API for a given sol range and camera set,
  downloads Navcam L/R pairs, writes a manifest with sol, LMST, camera,
  and image ID
- `jezero-dem` — pulls a bounded tile from the HiRISE DTM, reprojects to a
  local metric grid, writes a `.npy` plus a `SOURCE.md`
- `ai4mars` — Zenodo download with checksum verification, then a subset
  extractor so you can work with 2 GB instead of 18 GB

Then `datasets/samples/` — the ~20 MB curated set the demos run from.

### Build

1. **`sensors/dataset_source.py`** — a `FrameSource` that replays a downloaded
   image sequence. This is the bridge: with it, every Phase 2 module runs on
   real Mars imagery without a single change.
2. **`stereo/rectify.py`** — Navcam pairs are not rectified. You need
   `stereoCalibrate` / `stereoRectify`, or epipolar rectification from matched
   features when you lack the rig geometry. **This is the hardest piece in
   Phase 3.** Budget for it.
3. **`db/store.py`** — SQLite schema: missions, frames, detections, tracks,
   lineage nodes, telemetry samples. Lineage persistence matters: a mission
   you can't replay the reasoning of is a mission you can't review.
4. **`health/monitor.py`** — CPU, memory, storage via `psutil`; GPU via the
   compute module; battery as a simulated model tagged `SIMULATION`, because
   your laptop is not a rover and pretending otherwise would violate the
   credibility rule your own spec sets.
5. **`api/`** — FastAPI. Routes: modes, pipeline control, frame streaming
   (WebSocket, JPEG-encoded), lineage (`GET /api/lineage/{node_id}` — the
   payload already exists), health, missions, reports.
6. **`mission/planner.py` + `mission/timeline.py`** — waypoints, coverage,
   battery estimation, event log.

### Gate

- `fetch_data.py perseverance --sol 200-210` downloads real pairs and the
  existing pipeline runs on them unmodified
- A Jezero DEM tile produces a traversability map and a planned path
- FastAPI serves a live frame stream and a lineage payload; OpenAPI docs
  render at `/docs`
- Restarting the server replays a stored mission from SQLite
- ~60 new tests, including API contract tests

### Effort

~55 h · 6 weeks. Rectification alone is ~15 h; do not underestimate it.

---

## Phase 4 — Mission control UI

**Goal.** The thing people actually look at.

### Data

None new. The UI consumes only the API. If a number cannot be obtained from an
endpoint, it does not appear on screen — that rule is what keeps the dashboard
honest.

### Build

1. Vite + React + TypeScript + Tailwind. Generate the API client from the
   OpenAPI schema so types can't drift.
2. **Shell** — dockable, resizable panels; mission timer; mode switcher wired
   to the live endpoint.
3. **Source badges everywhere.** Every displayed value renders its
   `Source` as a coloured chip, with `SIMULATION` visually distinct. Build
   this as one `<TracedValue>` component early; retrofitting it is painful.
4. **The Data Lineage drawer.** Click any value → the derivation chain,
   algorithms, confidences, latencies, and the raw frame behind it. Render the
   graph from the Mermaid the backend already emits. This is the centrepiece —
   give it the time it deserves.
5. Camera view with detection and track overlays; timing waterfall; health
   panel; image-processing sliders generated from `ImageChain.control_schema()`;
   Plotly telemetry.
6. **Three.js / R3F 3D view** — point cloud, terrain mesh from the DEM,
   rover model, planned path, traversability heatmap.

### Gate

- Every panel is fed by a real endpoint. Zero mocked data
- Clicking a planned waypoint opens the drawer and traces back to the source
  frame
- Mode switching changes the pipeline without a reload
- Lighthouse accessibility ≥ 90; UI stays responsive at 15+ fps streaming

### Effort

~70 h · 7 weeks. The largest single phase. The 3D view is ~20 h of it.

---

## Phase 5 — Mars flagship

**Goal.** The mode that makes the project memorable.

### Data

AI4Mars for terrain classification; Perseverance Navcam and Hazcam for
imagery; Jezero HiRISE DTM for elevation; Mastcam-Z for science targets.

### Build

1. **`mars/terrain.py`** — semantic segmentation over AI4Mars' four classes.
   Fine-tune a small segmentation model (DeepLabv3-MobileNet or a compact
   U-Net) on the merged labels. Train on CPU/MPS overnight or rent an hour of
   cloud GPU. Export ONNX for fast inference.
2. **Fuse semantics into traversability.** Right now cost is purely
   geometric. Sand that *looks* flat is a slip hazard — this is precisely what
   trapped Spirit. Add a semantic cost layer so `sand` raises cost even at
   zero slope. This is the single most defensible improvement in the whole
   project.
3. **`mars/hazards.py`** — rock and crater detection. Rocks from the `big
   rock` class plus DEM step response; craters from radial rim symmetry on the
   DEM.
4. **`mars/science.py`** — science target triage: layered outcrop, vein fill,
   float rock. Scoring plus a sample-collection planner that maximises science
   value under an energy budget.
5. **Camera assignment** — Navcam for far-field planning, Hazcam for
   near-field safety, stereo for depth. Different frame rates, different roles.
6. **`mars/rover_state.py`** — pose, wheel odometry, slip integration, an
   energy model, and a one-way light time model so the UI can show *why* the
   rover must decide locally.
7. **AI scientist assistant** — Claude API over the mission state. Structured
   JSON out, rendered as UI, not free text. Grounded strictly in lineage data
   so it cannot invent observations.

### Gate

- Terrain classifier ≥ 85% mean IoU on the AI4Mars validation split
- A Perseverance sol's real Navcam imagery produces a semantic traversability
  map and a safe path
- Every autonomous decision traces back to source imagery through the drawer
- The assistant refuses to answer beyond what the lineage supports

### Effort

~60 h · 6 weeks. Training and data wrangling is ~25 h of it.

---

## Phase 6 — Dataset manager and training loop

**Goal.** Close the loop: the platform improves its own models.

### Data

Your own annotations, plus AI4Mars as the base.

### Build

Upload and ingest · annotation UI (bounding box + polygon, on canvas) ·
dataset versioning with content-addressed manifests · train/val/test splits
with leakage checks · training runs as background jobs with live loss curves ·
model registry with metrics, and provenance linking each model to the exact
dataset version that produced it · evaluation against a held-out set.

Provenance extends naturally here: a detection's lineage should reach past the
model to the dataset version and the annotator.

### Gate

- Annotate 100 images, train, and the new model appears in the registry and is
  selectable in the running pipeline
- Dataset versions are reproducible from manifest alone
- Model cards record training data, metrics and limitations

### Effort

~40 h · 4 weeks.

---

## Phase 7 — Digital twin, replay and reporting

**Goal.** Everything that makes a mission reviewable after the fact.

### Data

Missions recorded in Phases 3–5.

### Build

Mission recorder (frames + telemetry + full lineage) · replay with pause,
scrub, frame-step and variable speed · counterfactual replay — re-run a stored
mission with different parameters and diff the decisions, which is the feature
that turns replay into an engineering tool · PDF/CSV/JSON reports with embedded
lineage · a comparison view for two runs.

### Gate

- A recorded mission replays deterministically, byte-identical decisions
- Changing `cost_weight` on replay produces a visible, explained path diff
- Generated PDF includes the lineage for every headline claim

### Effort

~35 h · 3.5 weeks.

---

## Phase 8 — Hardening and portfolio packaging

**Goal.** Make it look like a product, and make it survivable in an interview.

CI on GitHub Actions (macOS + Linux, matrix over OpenCV 4.x and 5.x — your
capability layer earns its keep here) · ruff + mypy clean · coverage ≥ 85% ·
performance benchmarks with regression gates · architecture, sequence and class
diagrams, mostly generated from code · deployment guide · optional Docker ·
a 3–5 minute demo video · a written architecture-decision record for each of
the trade-offs in the README.

**Gate.** Fresh clone → `uv sync` → demo runs, on a machine that is not yours.
Get someone else to try it; this always finds something.

**Effort.** ~30 h · 3 weeks.

---

# Part 3 — Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Navcam stereo rectification is harder than expected | High | Blocks depth on real data | Fall back to Middlebury for depth correctness; use Mars pairs for detection and terrain only |
| AI4Mars training is too slow on CPU/MPS | Medium | Delays Phase 5 | Train on a subset; rent one cloud GPU hour; or use a smaller backbone |
| Scope creep across 16 modules | High | Nothing finishes | Phase gates. Do not start N+1 until N's gate passes |
| Disk fills up | Medium | Annoying | Subset extractors; external drive; never commit data |
| Dataset URLs move | Medium | Downloader breaks | Checksums + `SOURCE.md`; fail loudly with the archive page URL |
| OpenCV 4 vs 5 differences on your Mac | Low | Already handled | `cv_compat` + CI matrix |
| Burnout at ~200 hours | **High** | Project abandoned at 80% | Ship Phase 4 as a complete, presentable v1.0. Phases 6–8 are genuinely optional |

That last row is the real risk. A finished five-phase project beats an
abandoned eight-phase one, and the gap between them is entirely about deciding
in advance where "done" is allowed to be.

---

# Part 4 — Suggested order of work

If you want the shortest path to something presentable:

1. `fetch_data.py` + `dataset_source.py` — **do this first.** Two days of work
   that converts everything already built from a simulation into an analysis
   of real Mars data. Highest return in the entire plan.
2. Jezero DEM → traversability → path plan. A screenshot of a safe path
   planned across the real crater floor is your single best portfolio asset,
   and it needs nothing from Phases 3–8 except the downloader.
3. FastAPI + the lineage endpoint.
4. UI shell + the lineage drawer.
5. Everything else.

Steps 1 and 2 are roughly 15 hours and produce most of the credibility.

---

## Data source quick reference

| What | Where | Size | Licence |
|---|---|---|---|
| Perseverance raw images | `api.nasa.gov/mars-photos/api/v1/rovers/perseverance/photos` | Per-request | Public domain |
| Perseverance browser | `mars.nasa.gov/mars2020/multimedia/raw-images/` | — | Public domain |
| AI4Mars v0.6 | Zenodo record `15995036` | 17.8 GB | Open, cite the paper |
| Jezero 1 m DEM | `astrogeology.usgs.gov/search/map/Mars/Mars2020/JEZ_hirise_soc_006_DTM_...` | ~GB | USGS public |
| HiRISE DTM archive (COG) | `registry.opendata.aws/nasa-usgs-mars-hirise-dtms/` | Ranged reads | Public |
| MSR TRN DTM mosaics | `astrogeology.usgs.gov/search/map/mars-sample-return-terrain-relative-navigation-hirise-dtm-mosaic` | 13.2 GB | USGS public |
| Middlebury stereo | `vision.middlebury.edu/stereo/data/` | ~2 GB | Check terms |
| KITTI | `cvlibs.net/datasets/kitti/` | ~20 GB | CC BY-NC-SA |
| COCO YOLOv11 weights | Ultralytics | ~6 MB (n) | AGPL-3.0 — check before commercial use |

The Ultralytics licence is worth a moment's thought: AGPL-3.0 is fine for a
public portfolio repository, and would matter if this ever became commercial.
