# PerceptOS

**Enterprise Robotic Perception, Autonomy & Planetary Exploration Platform**

---

## Status: v1.0 — Phase 4 complete (19/19), Phase 5 & 6 delivered

This repository contains the **backend of PerceptOS**, built and tested end to
end: the perception kernel (Phase 1) and the full OpenCV vision layer on top of
it (Phase 2). The service layer and mission-control UI are next; see
[Build sequence](#build-sequence).

What runs today:

| Layer | Status | Module |
|---|---|---|
| Provenance & credibility contract | ✅ tested | `core/provenance.py` |
| Data lineage graph (the drawer, server side) | ✅ tested | `core/lineage.py` |
| Stage abstraction + failure isolation | ✅ tested | `core/stage.py` |
| Pipeline with static wiring validation | ✅ tested | `core/pipeline.py` |
| Device selection (MPS → CUDA → CPU) | ✅ tested | `core/compute.py` |
| Constant-acceleration Kalman filter | ✅ tested | `estimation/kalman.py` |
| Stereo depth + uncertainty propagation | ✅ tested | `stereo/depth.py` |
| Image-based visual servoing | ✅ tested | `servoing/ibvs.py` |
| Occupancy + elevation mapping | ✅ tested | `mars/occupancy.py` |
| Traversability (slope / step / roughness) | ✅ tested | `mars/traversability.py` |
| A* safe path planning | ✅ tested | `mars/planner.py` |
| Wheel slip estimation | ✅ tested | `mars/slip.py` |
| **Vision layer (Phase 2)** | | |
| OpenCV capability detection (4.x/5.x) | ✅ tested | `core/cv_compat.py` |
| Camera acquisition + health telemetry | ✅ tested | `sensors/camera.py` |
| Chessboard calibration + quality gating | ✅ tested | `calibration/chessboard.py` |
| Image processing chain | ✅ tested | `imageproc/filters.py` |
| Feature extraction + matching | ✅ tested | `features/extractors.py` |
| Stereo matching (BM / SGBM) | ✅ tested | `stereo/matcher.py` |
| YOLOv11 detection | ✅ tested | `detection/yolo.py` |
| Multi-object tracking | ✅ tested | `tracking/trackers.py` |
| Visual odometry | ✅ tested | `mars/odometry.py` |
| **Real-data layer (Phase 3a)** | | |
| HTTP transport + provenance cache | ✅ tested | `data/cache.py` |
| NASA Mars 2020 client + stereo pairing | ✅ tested | `data/mars2020.py` |
| DEM / AI4Mars loaders + semantic costs | ✅ tested | `data/terrain.py` |
| Environment abstraction (5 environments) | ✅ tested | `core/environment.py` |
| Dataset frame sources | ✅ tested | `sensors/dataset_source.py` |
| **Fusion & planning (Phase 3c)** | | |
| EKF sensor fusion, 5 sensor models | ✅ tested | `fusion/ekf.py` |
| Theta\* any-angle planning | ✅ tested | `mars/planners.py` |
| D\* Lite incremental replanning | ✅ tested | `mars/planners.py` |
| **Service layer (Phase 3b)** | | |
| SQLite persistence incl. lineage | ✅ tested | `db/store.py` |
| Health telemetry (measured vs modelled) | ✅ tested | `health/monitor.py` |
| Runtime service (missions, modes) | ✅ tested | `api/service.py` |
| FastAPI + WebSocket + OpenAPI | ✅ tested | `api/app.py` |
| **NASA sources & console (Phase 3d/4a)** | | |
| Data-source & stereo status vocabulary | ✅ tested | `data/sources.py` |
| JPL raw-image client (keyless, CAHVOR) | ✅ tested | `data/mars2020_raw.py` |
| Root status page + `/api/datasources` | ✅ tested | `api/app.py` |
| Mission-control console | ✅ tested | `api/console.py` |
| **Phase 4 visualisation** | | |
| Array→PNG rendering, no-data honest | ✅ tested | `api/render.py` |
| Terrain & navigation sessions | ✅ tested | `api/analysis.py` |
| Frame / stereo / terrain / nav endpoints | ✅ tested | `api/app.py` |
| Perception, Terrain, Navigation panels | ✅ tested | `api/console.py` |
| **Stereo rectification (Phase 4c)** | | |
| CAHV→pinhole + epipolar rectification | ✅ tested | `stereo/rectify.py` |
| Rectified stereo source & stage | ✅ tested | `sensors/stereo_source.py` |
| Point cloud projection | ✅ tested | `api/render.py` |
| **Phase 5 — SLAM, replay, reports** | | |
| SE(2) pose graph, Gauss-Newton, Huber | ✅ tested | `slam/graph.py` |
| Keyframes + loop closure verification | ✅ tested | `slam/frontend.py` |
| SLAM system + pipeline stage | ✅ tested | `slam/system.py` |
| Mission replay + counterfactuals | ✅ tested | `api/replay.py` |
| JSON / CSV / Markdown reports | ✅ tested | `reports/mission.py` |
| 3D terrain view + hillshade | ✅ tested | `api/render.py` |
| SLAM trajectory plot | ✅ tested | `api/render.py` |
| **Phase 6 — datasets & models** | | |
| Content-addressed dataset versions | ✅ tested | `datasets/manager.py` |
| Group-aware splits + leakage audit | ✅ tested | `datasets/manager.py` |
| Model registry, evaluation, cards | ✅ tested | `datasets/registry.py` |

**585 tests, all passing.** 54 documented API routes. All 19 Phase 4 UI spec
items present. The backend is now feature-complete; what remains
is the mission-control UI. The Phase 1 kernel runs on NumPy alone; the vision
layer needs OpenCV, and PyTorch/Ultralytics stay optional throughout — a
missing detector is a degraded mode, not a crash.

---

## Quick start

```bash
git clone <your-repo> perceptos && cd perceptos

# Option A — uv (recommended)
uv sync

# Option B — pip
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# The core needs nothing but NumPy:
pip install numpy pytest

python demo_mars_autonomy.py       # autonomy chain + lineage output
python demo_perception_pipeline.py # full vision pipeline, needs OpenCV
python demo_real_data.py           # real-vs-simulation mode switch
python demo_fusion_navigation.py   # EKF fusion + dynamic replanning
python scripts/fetch_data.py list  # available NASA/USGS datasets
python -m pytest tests -q          # 208 tests
```

Verified on Python 3.12. No CUDA, no TensorRT, no ROS, no Docker, no NVIDIA GPU.

---

## NASA credentials — the short answer

**You almost certainly do not need a NASA API key.**

PerceptOS's primary Perseverance source is the **JPL Mars 2020 Raw Image API**,
which is keyless. Ask the running system yourself:

```bash
python scripts/fetch_data.py credentials     # or GET /api/datasources
```

| Service | Endpoint | Key? | Used by |
|---|---|---|---|
| **JPL Mars 2020 Raw Image API** | `mars.nasa.gov/rss/api/?feed=raw_images&category=mars2020&feedtype=json` | **No** | `data/mars2020_raw.py` — the default |
| NASA Mars Rover Photos | `api.nasa.gov/mars-photos/api/v1/rovers/perseverance/photos` | Yes | `data/mars2020.py` — fallback only |
| USGS Jezero HiRISE DTM | `astrogeology.usgs.gov/search/map/Mars/Mars2020/JEZ_hirise_soc_006_DTM_...` | **No** | `data/terrain.py` |
| AI4Mars v0.6 | `zenodo.org/records/15995036` | **No** | `data/terrain.py` |

Rate limits for the one service that *does* take a key: `DEMO_KEY` allows 30
requests/hour and 50/day per IP; a free personal key from api.nasa.gov raises
that to 1,000/hour. Set it with `export NASA_API_KEY=...` if you want the
fallback. Without it, that one fetcher warns and uses `DEMO_KEY`; everything
else is unaffected.

### Why the keyless source is the better one anyway

The Mars Rover Photos API returns `id`, `sol`, `earth_date`, a camera name and
a URL. The JPL feed additionally returns the spacecraft clock, UTC and Mars
local time, rover attitude, site/drive indices — and the **CAHV/CAHVOR camera
model**.

That camera model is why PerceptOS can do stereo honestly. The distance between
the left and right `C` vectors *is* the physical baseline, and `|H − (H·A)A|`
*is* the focal length in pixels. Both measured, neither assumed. From the Mars
Rover Photos API those numbers do not exist, so any depth computed from it would
rest on invented geometry.

---

## Data status vocabulary

Every value the API returns declares one of these, and the console badges each
one differently:

| Status | Meaning |
|---|---|
| `NASA_PUBLIC_NO_KEY` | Fetched live from a keyless NASA/USGS endpoint |
| `NASA_DEMO_KEY` / `NASA_PERSONAL_KEY` | Fetched live via api.nasa.gov |
| `LOCAL_CACHED_DATA` | Real NASA data, previously downloaded — real, but not live |
| `SIMULATION` | Synthetic. Propagates to everything derived from it |
| `UNAVAILABLE` | Genuinely missing. **Never** silently replaced by simulation |

Stereo has its own vocabulary, because "these two images are a stereo pair" is
several separate claims:

| Status | Depth valid? | Meaning |
|---|---|---|
| `RECTIFIED_STEREO` | **yes** | Calibrated, simultaneous, epipolar-rectified |
| `VALID_STEREO` | no | Real camera model and one exposure, but not yet rectified |
| `UNRECTIFIED_PAIR` | no | Failed a precondition — not actually a pair |
| `CALIBRATION_UNAVAILABLE` | no | No camera model published; baseline unknown |
| `SIMULATION` | yes | Synthetic pair with a known synthetic geometry |

And terrain labels:

| Status | Meaning |
|---|---|
| `GROUND_TRUTH_LABEL` | AI4Mars mask drawn by a human. **Not** model output |
| `MODEL_PREDICTION` | Produced by a trained model at inference time |
| `SIMULATION` / `UNAVAILABLE` | Synthetic, or absent |

---

## The two ideas that make this more than a demo

### 1. No number exists without provenance

A bare `float` is not a valid output of any PerceptOS stage. Every value is a
`Traced` object binding the number to its `Source`, the algorithm that produced
it, its timestamp, latency, units, confidence, and the exact parameters in force
at the time.

Five sources, and nothing else is permitted:
`LIVE_SENSOR` · `LIVE_CAMERA` · `CALCULATED` · `ESTIMATED` · `SIMULATION`

Two invariants are enforced in code, not in documentation:

```python
# An estimate without an uncertainty is an unfalsifiable claim.
Provenance(source=Source.ESTIMATED, algorithm="yolo")
# -> ProvenanceError: ESTIMATED value must carry a confidence

# Simulated data cannot be laundered into something that looks measured.
sim   = traced(1.0, source=Source.SIMULATION, algorithm="synthetic_terrain")
child = sim.derive(2.0, algorithm="fuse", source=Source.LIVE_SENSOR)
child.source  # -> Source.SIMULATION   (the request is overridden)
```

Confidence propagates by taking the **minimum** across parents: a derived value
is never more certain than the weakest thing it was computed from.

### 2. Every decision is walkable backwards

`LineageGraph.explain(node_id)` returns the complete derivation chain behind any
displayed value — every algorithm, every intermediate, cumulative latency, the
weakest confidence anywhere upstream, and whether simulation touched it. That is
the payload the Data Lineage drawer renders. Real output from the demo:

```
derivation depth    : 4 nodes
weakest confidence  : 1.000
simulation in chain : True

1. stereo_point_cloud    simulation   conf 1.00   synthetic_terrain(Jezero-analogue DEM)
2. elevation_map         simulation   conf 1.00   OccupancyGrid.elevation_map(running-mean fusion)
3. traversability_map    simulation   conf 1.00   TraversabilityAnalyser(slope|step|roughness)
4. safe_path_plan        simulation   conf 1.00   AStarPlanner(octile, inflated, turn-penalised)
```

---

## Architecture

```
                    ┌──────────────────────────────────────┐
   sensors ────────▶│  Pipeline  (ordered, validated)      │
                    │                                      │
                    │  ┌────────────────────────────────┐  │
                    │  │ Stage: requires / provides     │  │
                    │  │  • auto-timed                  │  │
                    │  │  • auto-recorded to lineage    │  │
                    │  │  • failures isolated           │  │
                    │  └────────────────────────────────┘  │
                    └──────────┬───────────────┬───────────┘
                               │               │
                     FrameContext         LineageGraph
                     (blackboard)         (append-only DAG)
                               │               │
                               ▼               ▼
                         mode profile     lineage drawer
                       (YAML, hot-swap)   reports / audit
```

Stages know nothing about each other. They read named keys from a
`FrameContext` and write named keys back, which is why the same Kalman filter
serves Warehouse mode and Mars mode without a single conditional.

`Pipeline.validate()` runs at construction and refuses a mis-wired graph:

```python
Pipeline([DownstreamStage()])
# -> WiringError: stage 'downstream' requires ['doubled'] which no earlier stage provides
```

`Pipeline.to_mermaid()` generates the system block diagram **from the real
wiring**, so the diagram cannot drift out of date with the code.

### Folder structure

```
perceptos/
├── backend/perceptos/
│   ├── core/          provenance, lineage, stage, pipeline, compute
│   ├── sensors/       camera acquisition, FPS / latency / dropped frames
│   ├── calibration/   chessboard intrinsics, distortion, undistortion
│   ├── imageproc/     filters, CLAHE, thresholding, morphology, contours
│   ├── features/      ORB, FAST, AKAZE, SIFT, Harris
│   ├── stereo/        matcher (BM/SGBM) + depth geometry & uncertainty
│   ├── detection/     YOLOv11 inference
│   ├── tracking/      CSRT / KCF / MOSSE / optical flow, track IDs
│   ├── estimation/    Kalman filtering, velocity, future prediction
│   ├── servoing/      IBVS / PBVS, robot motion commands
│   ├── mars/          occupancy, traversability, planner, slip, odometry
│   ├── mission/       waypoints, timeline, coverage, replay
│   ├── health/        battery, CPU, GPU, storage, camera health
│   ├── reports/       PDF / CSV / JSON mission summaries
│   ├── api/           FastAPI service layer
│   └── db/            SQLite persistence
├── configs/modes/     operational mode profiles (hot-swappable)
├── frontend/          React + TypeScript + Three.js mission control
├── tests/             74 tests
└── demo_mars_autonomy.py
```

---

## Engineering decisions worth defending in an interview

These are the places where the obvious implementation is wrong, and this one
isn't. Each is covered by a test that fails if the shortcut is reintroduced.

**Hazards combine by maximum, not by sum or average.** Summing lets very smooth
ground average away a cliff beside it. One lethal hazard is lethal regardless of
how benign the other two are.

**Unobserved terrain is never traversable.** A NaN elevation cell is a hazard,
not a zero. Zero elevation is a claim; NaN is an honest absence of one.

**Hazards are inflated by the rover's circumscribed radius before planning.**
The rover is not a point. Without inflation the planner routes the *centre*
safely past a rock the *wheels* would hit.

**But inflation pads with `False` at the map border.** Padding with `True`
dilates the boundary inwards and makes every goal within one rover-radius of the
edge unreachable — the map edge is an arbitrary crop, not a wall.

**And a rover already inside an inflated footprint can back out through it.**
Freeing only its own cell leaves it boxed in with no legal move. The whole
connected inflated-but-not-lethal region is freed; genuinely lethal cells stay
blocked.

**The A\* heuristic is scaled by the *minimum* achievable per-metre cost.**
Scaling by the nominal cost makes the heuristic inadmissible, A\* stops being
optimal, and it silently returns plausible-looking but needlessly dangerous
paths.

**The Kalman update uses Joseph form.** The textbook `(I − KH)P` update
accumulates asymmetry over a long mission until the covariance stops being
positive-definite. Tested over 500 iterations.

**Depth carries `sigma_Z = Z² · sigma_d / (f · B)`.** Stereo error grows with the
*square* of range. A rig accurate to millimetres at 1 m is accurate to metres at
40 m, and `max_reliable_range()` tells the planner where to stop believing its
own stereo.

**IBVS uses damped least squares, not the pseudo-inverse.** Near a singular
configuration the pseudo-inverse commands enormous velocities — on real hardware
that is a violent lurch. Damping bounds the command at the cost of slightly
slower convergence, which is the right trade for anything with mass.

**Slip ignores commanded steps below 1 cm.** Dividing by a near-zero commanded
distance is how naive slip estimators emit 900% alarms while stationary.

**Slip embedding requires a sustained window, not one bad sample.** A single
high-slip sample is normal when crossing a rock. Sustained high slip means the
wheels are digging in — the failure that immobilised Spirit at Troy in 2009.

**Every OpenCV factory is resolved through a capability layer.** `cv2.ORB_create`
became `cv2.ORB.create` between 4.x and 5.x; AKAZE, BRISK and KAZE are absent
from several wheels; MOSSE only ever exists under `cv2.legacy`. Calling them
directly means an `AttributeError` on a machine that differs by one wheel, so
they are resolved once at import and a missing one names the wheel that
provides it.

**Disparity pinned at the search minimum is not coverage.** It is either an
unmatched pixel or a point at infinity, and neither converts to a finite range.
On a textureless surface SGBM returns 0 nearly everywhere; counting those as
valid reports 90% coverage for a map containing no distance information at all.

**Planar scenes are degenerate for the essential matrix.** When every tracked
feature lies on one plane, `recoverPose` returns a confident, wrong answer —
classically a spurious 180-degree yaw. Visual odometry scores a homography
against the fundamental matrix (the ORB-SLAM model-selection test, with the
different chi-square thresholds the two models' degrees of freedom require) and
refuses to emit a pose when the homography wins.

**A track is not reported until it has been seen `min_hits` times.** Without
confirmation, every one-frame detector false positive becomes a track with an
ID and the mission log fills with phantom objects.

**Association runs against the Kalman prediction, not the last known box.**
Matching on raw proximity swaps the identities of two crossing objects at the
crossing point, every time.

**Live drop counts are `ESTIMATED`, file drop counts are measured.** A decoder
knows its true frame index; a webcam does not. Inferring a live drop count from
elapsed time is fine — presenting it as measured is the standard lie in camera
dashboards.

**Camera health reports `starting` until the rolling window fills.** A rate
computed from one interval is noise; grading on it makes every freshly opened
camera flash `degraded`, which trains operators to ignore the indicator.

**Calibration gates on coverage and view diversity, not just RMS.** Twenty
views from the same angle in the middle of the frame give an excellent RMS and
badly wrong distortion coefficients at the edges, which is exactly where
distortion matters.

**Stereo pairs are matched on spacecraft clock, not list order.** NASA's API
returns left and right Navcam frames as unrelated records. The pairing key is
buried in the product ID (`NLF_0203_0684970679_...`): partners share the SCLK
value because they are exposed together. Pairing by adjacency or timestamp
proximity mismatches frames whenever the rover shoots a mosaic, and a
mismatched pair yields a dense, confident, entirely wrong depth map.

**Unrectified pairs are refused, not silently matched.** Raw Navcam pairs have
non-horizontal epipolar lines, so block matching along scanlines is invalid.
`StereoDatasetSource.read_for_matching()` raises rather than serving them.

**DEM no-data sentinels become NaN before anything reads them.** HiRISE DTMs
mark missing data with a large negative value, not NaN. Load one naively and
the hazard analyser correctly reports a -3.4e38 metre cliff as the most
dangerous surface in the solar system.

**Segmentation speckle is opened before lethal classes are inflated.** A single
stray `big_rock` pixel becomes a metre-wide no-go disc after rover-radius
inflation. At 5% speckle the map becomes entirely unplannable — measured, not
theorised, and covered by a test.

**Semantic and geometric costs fuse by maximum.** Sand is geometrically flat
and a sinkage hazard. Averaging lets smooth geometry cancel the sand warning,
which is the failure that ended Spirit's mission.

**Gravity does not change the geometric slope limit.** Static tip-over is
governed by centre of mass versus wheelbase — pure geometry. A rover that tips
at 45° on Earth tips at 45° on the Moon. Gravity enters the energy model and
the dynamics, and that is the only place `Environment.gravity_ms2` is used.

**Angular innovations are wrapped.** A heading measurement of +179° against a
prediction of −179° is a 2° error, not 358°. Without wrapping, every crossing
of the ±180 boundary injects a ~6.28 rad innovation and the filter slews — and
it only shows up on the specific traverse that happens to cross it.

**Measurement gating is suspended during warm-up.** A chi-square gate is a
statement about a *converged* filter. Applied to an uninitialised one it
rejects exactly the large, correct corrections that would converge it, and the
filter then sits at its initial guess rejecting every measurement forever.

**Consistency is judged on rejection rate, not just NIS.** Mean NIS over
accepted measurements is a censored statistic — the gate has already discarded
the high-NIS samples. A filter whose R is far too small rejects nearly
everything and the survivors look beautifully consistent.

**Gyro bias is only estimated when it is observable.** With a gyro alone,
`omega` and `b_gyro` enter the measurement identically and no data separates
them. Without a recent absolute-heading fix the bias state is frozen; letting
it random-walk produces a confident, wrong bias that heading integration then
follows.

**Wheel encoders are down-weighted by measured slip.** Fusing wheel and visual
odometry as if both measured ground speed is wrong whenever the wheels slip —
whichever updates first captures the state and the gate rejects the other as an
outlier, so a rover digging into a sand trap reports a healthy filter.

**Theta\*'s line-of-sight cost uses straight-line distance, not the supercover
cell sum.** Supercover deliberately emits both corner cells at every diagonal
crossing, so summing steps over it roughly doubles the measured length and
Theta\* concludes the straight segment is more expensive than the staircase it
exists to replace.

**D\* Lite key comparison needs a floating-point tolerance.** Keys are
`g + h + km` sums accumulated over long paths, so mathematically equal keys
differ by a few ulps. Compared exactly, a vertex tying with the start key reads
as *greater*, `ComputeShortestPath` terminates on its first iteration, and the
planner silently returns the stale pre-change path — no crash, no warning, just
a replanner that has stopped replanning.

**D\* Lite's advantage is conditional, and the README says so.** Measured on a
120×120 grid: a change off the optimal path costs 0 expansions; a change on the
path in open terrain costs about the same as a fresh search (114 vs 787,
because a perfect octile heuristic made the initial search trivially cheap); on
cluttered terrain — the real rover case — repair was 63× cheaper (36 vs 2 284).

**SQLite foreign keys are switched on explicitly.** They are disabled per
connection by default, so a schema full of `REFERENCES` clauses that nobody
enabled enforces precisely nothing. There is a test that inserts an orphan row
and requires it to fail.

**The lineage graph is a first-class table, not a JSON blob.** Storing frames
and detections is easy and mostly useless alone: six months later the question
is never "what did the rover see?" but "why did it turn left?". Ancestors are
resolved with one recursive CTE rather than a round trip per level, and the
persisted payload is asserted equal to the live one — that equivalence is what
lets the drawer answer about a mission recorded months ago.

**The battery is labelled SIMULATION and always will be.** There is no rover
battery in a laptop — or rather there is, and it is the laptop's, which is not
what the tile claims to show. Host CPU, memory and disk are `LIVE_SENSOR`;
battery, endurance and motor state come from an explicit power model and are
badged as modelled. A blank tile is better than a confident wrong one.

**Data mode is inspected, not declared.** `/api/status` reports what the
pipeline is actually fed by walking the frame source, not what a config file
claims — so the mode badge cannot drift from reality.

**WebSocket back-pressure drops messages rather than blocking.** A slow browser
tab must never stall frame processing, and a subscriber that raises is removed
rather than allowed to take down the pipeline.

**The stereo baseline is measured from the camera model, never assumed.** JPL
publishes CAHV/CAHVOR components; the distance between the two `C` vectors is
the physical baseline. When the model is absent the pair is reported
`CALIBRATION_UNAVAILABLE` and no depth is offered. There is no code path where
a plausible default gets filled in.

**Simultaneity is a hard precondition for a stereo pair, not a score.** Two
Navcam frames an hour apart show different scenes under different lighting from
a possibly different pose; a perfect camera model does not rescue them. A
calibrated non-simultaneous pair scores *below* an uncalibrated simultaneous
one, and there is a test asserting that ordering.

**A `SIMULATION` value cannot declare an external `source_type`.** The
constructor raises. Synthetic data has no archive URL, and the type system
should not let one be attached.

**A missing dataset is `UNAVAILABLE`, not `SIMULATION`.** `serve.py --source
nasa` with no cached frames exits with fetch instructions rather than quietly
falling back to synthetic terrain and continuing to look healthy.

**`confidence` and `uncertainty` are separate fields.** Confidence is "how much
do we trust this at all" in [0,1]; uncertainty is "plus or minus what" in
physical units. A depth of 4.0 m can be confidence 0.9 and uncertainty 0.35 m,
and neither substitutes for the other.

**Random splitting of rover imagery measures memorisation, not accuracy.**
Consecutive Navcam frames from one sol show the same rocks from almost the same
viewpoint, seconds apart. Split at random and near-duplicates land in both
train and test. Measured on 4 sols × 5 frames: a random split leaked 3 of 4
groups; a grouped split leaked none. `DatasetVersion.split` therefore requires
a group key, or an explicit `allow_random=True` asserting the samples really
are independent.

**A dataset version is the SHA-256 of its manifest.** `v3` names nothing — the
folder may have been edited since. A content hash names exact bytes, so a
model's provenance identifies precisely what it was trained on, and an edited
manifest is detected on load.

**An empty partition is refused.** Four groups at 70/15/15 rounds to 3/1/0, and
evaluating on nothing returns metrics that look like results. The error names
the real cause: too few groups, not a bad ratio.

**A single accuracy figure is never reported alone.** On AI4Mars, soil covers
most of most frames while `big rock` — the class that threatens the vehicle —
covers a few percent. A model predicting "soil" everywhere scores 97% and finds
no hazards at all. Evaluation reports per-class IoU, precision and recall
alongside the majority-class baseline, and flags any class with under 20%
recall as unsafe regardless of its overall score.

**An absent class has undefined metrics, not zero ones.** Reporting 0.0 drags
the mean IoU down and implies the model failed at something it was never asked
to do — and it made `weakest_class` point at a class the split never contained.

**Sample ids are paths, not filename stems.** Rover archives name files
identically inside each sol directory, so `sol00203/img0.png` and
`sol00204/img0.png` collided and the leakage audit reported every collision as
a duplicate across partitions.

**Vertical exaggeration is a display choice and travels in the headers.** Real
planetary terrain is far flatter than intuition expects, so the 3D view
exaggerates by 2× to make features legible — which changes how steep the
terrain looks without changing what it is.

**A false loop closure does not leave a large residual on its own edge.** This
is the finding that shaped the SLAM back end. One might expect a wrong closure
to fit badly; it usually fits *better* than the good edges, because the
optimiser satisfies it by deforming everything around it. Measured on a drifted
square traverse: the false closure's residual was 0.096 while the odometry it
had corrupted sat at 0.27, and without a robust kernel at 1.03. **Checking
loop-closure residuals alone therefore misses precisely the worst case.** What
signals a false closure is odometry that fitted well before and does not after.
Both signals are monitored, and the suspect edge is reported — never silently
dropped, since deleting it would hide a front-end failure.

**Loop closure needs geometric verification, not appearance.** Martian regolith
is visually self-similar for kilometres, so descriptor matching alone proposes
false closures freely. Candidates must additionally agree on a single rigid
transform under RANSAC, and the implied correction must be plausible against
the accumulated drift — a closure demanding a 40 m correction after a 2 m
traverse is proposing that the rover teleported.

**Landmarks require measured depth.** A monocular front end can triangulate
them, but only up to unknown scale, and an unscaled cloud drawn beside a metric
trajectory invites exactly the wrong reading. Without stereo the map is a
trajectory and the panel says `UNAVAILABLE` with the reason.

**Replay replays; it does not recompute.** Every value comes from the mission
record, so the timings are the ones actually observed. The counterfactual is
scoped honestly: it answers what a different *planner* would have decided given
the same perception, and cannot answer what a different perception pipeline
would have seen, because those frames were never captured.

**Report limitations are generated from what was recorded**, not written by
hand, so they cannot drift out of step with the run they describe. A synthetic
mission is labelled `SIMULATION` in the header, the summary and every affected
figure — a report is the artefact most likely to be read out of context.

**The stereo baseline comes from the CAHV camera centres, and rectification
is verified before it is claimed.** Decomposing the published model gives
`K`, `R` and `C` per camera; the distance between the two `C` vectors is the
physical baseline and `|H − (H·A)A|` the focal length. After rectification the
median vertical disparity of matched features is *measured*, and the pair only
becomes `RECTIFIED_STEREO` — the sole state permitting depth — if that residual
is under 2 px. Skip the measurement and the status stays `VALID_STEREO` with
depth withheld: a claim without evidence is not a status.

Verified against known geometry: a scene at a true 32.81 m reads 32.81 m, and
at 16.41 m reads 16.41 m.

**CAHVOR distortion is never claimed to be corrected.** Its `O`/`R` terms are
not OpenCV's `k1..k3`, and converting approximately would introduce error while
asserting a correction. `distortion_corrected` is always `False` and the
residual is reported instead.

**A measured rig beats a configured one.** `DepthStage` prefers the rig carried
on the context over any passed at construction: a stale baseline silently
rescales every depth in the frame. With neither, it raises rather than assuming.

**Contradictory source labelling is rejected at construction.** A source cannot
be `SIMULATION` and `NASA_PUBLIC_NO_KEY` at once. The provenance layer already
refused to build such a value, but mid-frame — several steps from the mistake.

**`traced()` inherits `source_type` exactly as `derive()` does.** The two
construction paths disagreed, so depth derived from cached NASA imagery lost
its origin and the UI badge fell back to unknown.

**A literal route declared after a matching parameterised one is unreachable.**
`/api/frame/pointcloud.png` was being swallowed by `/api/frame/{slot}.png`;
Starlette matches in declaration order.

**Point clouds are subsampled, never synthesised.** The drawn count and the
total are both reported, and the metric extent travels in the render note so
density is not mistaken for coverage.

**No-data is drawn as no-data.** NaN, infinity and masked cells render in a
reserved magenta that appears in no colormap, and the valid fraction ships as
an HTTP header. Clamping NaN to zero — the usual shortcut — produces a smooth,
complete-looking raster that claims measurements were made where none were.

**Rendering never interpolates.** Upscaling uses nearest neighbour only.
Bilinear on a hazard map invents cells exactly on the safe/lethal boundary, so
a path can appear to thread a gap that does not exist.

**`hazard_mask` and the planner's `cost >= 0.95` are different sets.** The
analyser's mask marks outright limit breaches (~2% of a real Jezero tile); the
planners additionally block cells whose *weighted severity* saturates (~6%).
Anything reasoning about reachability must use the planner's criterion, or it
selects goals the planner then refuses.

**Reachability uses the planners' no-corner-cutting rule.** A naive
8-connected flood fill treats the diagonal gap between two rocks as passable
and reports cells as reachable that the planner rejects.

**An auto-selected goal is chosen from the reachable component.** The farthest
*safe* cell is often behind a ridge in a separate component; the farthest
*reachable* cell is a route. An explicitly commanded goal is never relocated —
it is honoured exactly or refused.

**NaN is not valid JSON.** `float(nan)` serialises to the bare token `NaN`,
which no strict parser accepts, so every lineage request touching a real survey
returned 500. Array summaries now use nanmin/nanmax and report non-finite
results as `null`. Synthetic data is always finite, so only real data exposed
this.

**D\* Lite is built lazily.** Its initialisation is a full backward search over
the grid — about 2.5 s on 160×160. Paying that on every plan request when most
callers never replan made planning feel broken; plan latency dropped ~4× when
it moved to first-replan.

**Ray occupancy from a DEM is `UNAVAILABLE`, not a flat 0.5 map.** An elevation
model carries no sensor origin, so log-odds stay at the prior everywhere.
Publishing that as "occupancy" would look like a computed result rather than
the absence of one; elevation *coverage*, which is real, is offered instead.

**Undistortion caches its remap tables.** Calling `cv2.undistort` per frame
rebuilds them every time and costs more than the remap itself — a common reason
a pipeline mysteriously runs at half speed.

**Occupancy log-odds are clamped.** An unclamped grid becomes permanently
certain and can never recover from a spurious reading. Tested by saturating a
cell as occupied and then recovering it.

**Disabled and skipped stages still appear in the report.** A stage that
silently vanishes is indistinguishable from one that ran and found nothing.

---

## Build sequence

**Phase 1 — perception kernel** ✅ *complete, this repository*
Provenance, lineage, stage/pipeline, device selection, Kalman, stereo depth,
IBVS, occupancy, traversability, planner, slip.

**Phase 2 — vision layer** ✅ *complete*
Capability detection across OpenCV 4.x/5.x · camera manager with FPS, latency
and honest drop accounting · chessboard calibration with coverage and
diversity gating · declarative image processing chain · ORB/FAST/SIFT/AKAZE/
Harris extraction and matching · StereoBM/SGBM with disparity validation ·
YOLOv11 detection (optional dependency, degrades to SKIPPED) · CSRT/KCF/MOSSE
tracking with Kalman association · visual odometry with planar-degeneracy
rejection.

**Phase 3a — real-data layer** ✅ *complete*
Injectable HTTP transport (so NASA clients are testable offline) · content-
addressed cache with checksums, manifests and auto-generated `SOURCE.md` ·
Perseverance client with SCLK-based stereo pairing · HiRISE DEM loader with
sentinel handling · AI4Mars label loader, speckle removal and semantic cost
fusion · five-environment abstraction · dataset frame sources · `fetch_data.py`.

**Phase 3c — fusion and planning** ✅ *complete*
EKF over a unicycle model with gyro bias, five pluggable sensor models with
environment-aware availability, chi-square gating with warm-up, observability
tracking and per-sensor consistency diagnostics · Theta\* any-angle planning ·
D\* Lite incremental replanning.

**Phase 3b — service layer** ✅ *complete*
SQLite persistence with first-class lineage tables and recursive ancestor
queries · health telemetry separating measured host metrics from a modelled
rover power system · runtime service owning missions, modes and subscribers ·
FastAPI with typed routes, OpenAPI docs, and a WebSocket telemetry stream ·
`GET /api/lineage/{node_id}` answering from live memory or the database with an
identical payload.

**Phase 3d — NASA sources** ✅ *complete*
Keyless JPL raw-image client with CAHVOR camera models · evidence-based stereo
verification · data-source, stereo and label status vocabularies ·
`/` status page, `/api/datasources`, `fetch_data.py credentials`.

**Phase 4a — mission-control console** ✅ *partial, see limitations*
Single-file dark console at `/console` consuming the live API: overview,
pipeline contract and block diagram, lineage table, interactive lineage drawer,
data-source panel, telemetry with live WebSocket. **Not yet built:** the camera
/ stereo / depth image panels, terrain map rendering, 3D navigation view and
SLAM view. Those need image endpoints the API does not expose yet.

**Phase 4b — visualisation** ✅ *complete*
Array→PNG rendering with reserved no-data colour and audit headers · frame
slot introspection and per-slot rendering · stereo status endpoint · terrain
sessions over real cached DEMs with `UNAVAILABLE` when absent · navigation
sessions with A\*/Theta\* planning, D\* Lite replanning and path rendering ·
Perception, Terrain and Navigation console panels, all consuming live endpoints.

**Phase 4c — stereo rectification** ✅ *complete*
CAHV→pinhole decomposition · epipolar rectification with measured residual ·
per-pair status earned by verification · rectified stereo source and pipeline
stage · measured-rig precedence in `DepthStage` · point cloud projection ·
`serve.py --source stereo`.

**Phase 5 — SLAM, replay, reports** ✅ *complete*
SE(2) pose graph with Gauss-Newton and a Huber kernel · keyframe selection ·
loop closure with appearance, geometric and spatial filters · odometry-residual
monitoring for false closures · mission replay with scrubbing, stage profiling
and counterfactual planning · JSON/CSV/Markdown reports with embedded lineage
and generated limitations · SLAM and Replay console panels.

**Phase 4 — mission control UI** ✅ *complete — 19/19 spec items*
Perception, terrain, navigation, SLAM, replay, lineage, telemetry and data
source panels · 3D perspective terrain with draped cost maps and the planned
route · shaded relief · point cloud · SLAM trajectory plot.

**Phase 6 — dataset manager and model registry** ✅ *complete*
Content-addressed dataset versions · directory ingestion with automatic
grouping · group-aware splitting with a leakage audit · empty-partition refusal
· model registry linking each model to an exact dataset version · per-class
evaluation against the majority baseline · Markdown model cards stamping leaked
splits. **Training is not included** — PerceptOS records and evaluates models;
fitting them belongs to the model's own framework.

**Phase 7 — hardening, CI, deployment** *(next)*
React + TypeScript + Vite + Tailwind, Three.js / R3F 3D scene, Plotly
telemetry, dockable resizable panels, and the Data Lineage drawer wired to
`GET /api/lineage/{node_id}` — which already returns the correct payload today.

**Phase 5 — Mars flagship**
Terrain classification, rock/crater detection, science target triage, sample
collection planner, 3D rover visualisation, AI scientist assistant.

---

## Testing

```bash
python -m pytest tests -q                              # 148 tests
python -m pytest tests/test_core.py -q                 # kernel only, no OpenCV
python -m pytest tests -q --cov=backend/perceptos       # with coverage
```

The suite deliberately tests invariants rather than happy paths: that simulated
data cannot be laundered, that an estimate cannot exist without an uncertainty,
that the planner never enters an inflated hazard, that the Kalman covariance
stays symmetric positive-definite over 500 iterations, that stereo depth error
scales as Z², that a known stereo shift is recovered as the right disparity,
that two crossing tracks do not swap identities, and that a planar scene is
refused rather than given a spurious pose.

`tests/test_vision.py` skips cleanly when OpenCV is absent, so a geometry-only
install still runs the kernel suite.

---

## Licence

Choose one before publishing. Apache-2.0 is the usual choice for robotics work
intended as a portfolio piece.
