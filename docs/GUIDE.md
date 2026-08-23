# PerceptOS — Complete Guide

**v1.0.1 · 586 tests · 57 API routes · ~25,800 lines · Python 3.12+ · Apple Silicon native**

A robotic perception, autonomy and planetary exploration platform. This document
takes you from a fresh clone to understanding every layer.

---

## Table of contents

1. [What this is, in one page](#1-what-this-is-in-one-page)
2. [Install and first run](#2-install-and-first-run)
3. [The two ideas everything rests on](#3-the-two-ideas-everything-rests-on)
4. [Running everything — every command](#4-running-everything--every-command)
5. [The console, panel by panel](#5-the-console-panel-by-panel)
6. [Real NASA data](#6-real-nasa-data)
7. [Layer-by-layer code tour](#7-layer-by-layer-code-tour)
8. [Complete API reference](#8-complete-api-reference)
9. [Writing your own code against it](#9-writing-your-own-code-against-it)
10. [Testing](#10-testing)
11. [File map](#11-file-map)
12. [Troubleshooting](#12-troubleshooting)
13. [Engineering decisions worth defending](#13-engineering-decisions-worth-defending)

---

## 1. What this is, in one page

PerceptOS turns camera frames and elevation data into rover driving decisions,
and makes every step of that reasoning inspectable.

```
camera / NASA archive
        ↓
   image processing  ─────┐
        ↓                 │
   feature extraction     │  each step records:
        ↓                 │    where its input came from
   stereo → depth         ├─   what algorithm ran
        ↓                 │    how confident it is
   terrain analysis       │    how long it took
        ↓                 │    which values it derived from
   path planning  ────────┘
        ↓
   drive decision  ← click this in the UI and walk it all the way back
```

**The distinguishing property is not the algorithms.** They are standard: ORB,
SGBM, Kalman, A\*, pose-graph SLAM. What is unusual is that no number can be
displayed without declaring its origin, and every displayed number can be
traced back to the bytes it came from.

### The five source classes

Every value in the system carries exactly one:

| Class | Meaning |
|---|---|
| `LIVE_CAMERA` / `LIVE_SENSOR` | Measured by a real instrument |
| `CALCULATED` | Deterministic arithmetic on traced inputs |
| `ESTIMATED` | Output of a statistical or learned model — must carry a confidence |
| `SIMULATION` | Synthetic. Propagates to everything derived from it |
| `UNAVAILABLE` | Genuinely missing. **Never** silently replaced by simulation |

---

## 2. Install and first run

### Check your Python

macOS ships 3.9 as `python3`. You need 3.12+.

```bash
python3 --version
brew install python@3.12        # if needed
```

### Set up

```bash
unzip perceptos-v1.0.zip && cd perceptos
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Minimal install** (kernel, Mars autonomy, EKF, planners, SLAM — all pure NumPy):

```bash
pip install numpy pytest
python -m pytest tests/test_core.py -q      # 74 tests, no OpenCV needed
```

**Full install** adds OpenCV, PyTorch with Metal, FastAPI:

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests -q                   # expect: 585 passed, 1 skipped
```

> PyTorch resolves the Apple Silicon build automatically. **Do not** add
> `--index-url` — that is the CUDA instruction and gives you a broken install.

### Three-command smoke test

```bash
python -m pytest tests -q          # 585 passed
python demo_mars_autonomy.py       # terrain → hazards → A* → lineage
python scripts/serve.py            # then open http://localhost:8000/console
```

If the tests pass, the backend is sound; anything after that is environment.

---

## 3. The two ideas everything rests on

### 3.1 Provenance: no bare numbers

A `float` is not a valid output of any stage. Values are `Traced` objects.

```python
from perceptos.core.provenance import traced, Source

depth = traced(
    4.2,
    source=Source.CALCULATED,
    algorithm="disparity_to_depth(Z = f*B/d)",
    confidence=0.87,
    uncertainty=0.35,              # ± metres — NOT the same as confidence
    units="m",
    dataset="perseverance_jpl",
    source_url="https://mars.nasa.gov/rss/api/",
)

depth.value                        # 4.2
depth.provenance.confidence        # 0.87
depth.provenance.uncertainty       # 0.35
```

**`confidence` and `uncertainty` are different fields.** Confidence is "how much
do we trust this at all" in [0,1]. Uncertainty is "plus or minus what" in
physical units. A depth of 4.0 m can be confidence 0.9 and uncertainty 0.35 m,
and neither substitutes for the other.

Two rules are enforced by the constructor, not by documentation:

```python
# An estimate without an uncertainty is an unfalsifiable claim.
Provenance(source=Source.ESTIMATED, algorithm="yolo")
# -> ProvenanceError: ESTIMATED value must carry a confidence

# Simulated data cannot be laundered into something that looks measured.
sim   = traced(1.0, source=Source.SIMULATION, algorithm="synthetic_terrain")
child = sim.derive(2.0, algorithm="fuse", source=Source.LIVE_SENSOR)
child.source        # -> Source.SIMULATION   (the request is overridden)
```

Confidence propagates by **minimum**: a derived value is never more certain than
the weakest thing it was computed from.

```python
a = traced(1.0, source=Source.ESTIMATED, algorithm="a", confidence=0.9)
b = traced(2.0, source=Source.ESTIMATED, algorithm="b", confidence=0.4)
a.derive(3.0, algorithm="fuse", extra_parents=[b]).confidence     # 0.4
```

### 3.2 Lineage: walk any decision backwards

Every traced value knows its parents, forming a DAG.

```python
from perceptos.core.lineage import LineageGraph

graph = LineageGraph()
frame     = traced(image, source=Source.LIVE_CAMERA, algorithm="navcam",
                   label="raw_frame")
processed = frame.derive(blurred, algorithm="GaussianBlur", label="processed")
decision  = processed.derive({"steer": "left"}, algorithm="Planner",
                             source=Source.ESTIMATED, confidence=0.8,
                             label="decision")

for value in (frame, processed, decision):
    graph.record(value)

explanation = graph.explain(decision.id)
explanation["depth"]                 # 3
explanation["algorithms"]            # ['navcam', 'GaussianBlur', 'Planner']
explanation["weakest_confidence"]    # 0.8
explanation["contains_simulation"]   # False
explanation["datasets"]              # ['perseverance_jpl']
```

Real output from `demo_mars_autonomy.py`:

```
derivation depth    : 4 nodes
weakest confidence  : 1.000
simulation in chain : True

1. stereo_point_cloud    simulation   conf 1.00   synthetic_terrain(Jezero-analogue DEM)
2. elevation_map         simulation   conf 1.00   OccupancyGrid.elevation_map
3. traversability_map    simulation   conf 1.00   TraversabilityAnalyser(slope|step|roughness)
4. safe_path_plan        simulation   conf 1.00   AStarPlanner(octile, inflated, turn-penalised)
```

And on **real cached NASA elevation**:

```
depth 3 | datasets ['jezero_dem'] | contains_simulation False
  1. dem_elevation        live_sensor  LOCAL_CACHED_DATA  conf 0.98875
  2. traversability_map   calculated   LOCAL_CACHED_DATA  conf 0.98875
  3. safe_path_plan       calculated   LOCAL_CACHED_DATA  conf 0.98875
```

### 3.3 The pipeline: stages that know nothing about each other

```python
from perceptos.core.stage import PipelineStage, FrameContext
from perceptos.core.pipeline import Pipeline

class BlurStage(PipelineStage):
    name     = "blur"
    requires = ("gray",)              # read from the context
    provides = ("blurred",)           # promise to write

    def run(self, ctx: FrameContext) -> str:
        (gray,) = ctx.require("gray")
        out = cv2.GaussianBlur(gray.value, (5, 5), 0)
        self.emit(ctx, "blurred", out, algorithm="cv2.GaussianBlur",
                  parents=[gray], units="px")
        return "blurred"              # message shown in the timing waterfall
```

The base class handles timing, lineage registration and **failure isolation** —
a stage that raises is marked `FAILED` and the pipeline continues. On a rover, a
dead feature extractor must not take down hazard avoidance.

Wiring is validated at construction, not discovered mid-mission:

```python
Pipeline([BlurStage()])
# -> WiringError: stage 'blur' requires ['gray'] which no earlier stage provides
```

And the block diagram is generated **from the real wiring**, so it cannot drift:

```python
print(pipeline.to_mermaid())
# graph LR
#   acquisition["acquisition"]
#   acquisition -->|gray| blur
```

---

## 4. Running everything — every command

### Demos (no server needed)

```bash
python demo_mars_autonomy.py        # terrain → traversability → A* → lineage
python demo_perception_pipeline.py  # camera → features → tracking (needs OpenCV)
python demo_real_data.py            # real-vs-simulation mode switch
python demo_fusion_navigation.py    # EKF fusion + D* Lite replanning
```

Each prints its own explanation. `demo_mars_autonomy.py` and
`demo_fusion_navigation.py` need only NumPy.

### The server

```bash
python scripts/serve.py                      # synthetic camera, Mars mode
python scripts/serve.py --source nasa        # cached Perseverance frames
python scripts/serve.py --source stereo      # cached pairs → real metric depth
python scripts/serve.py --source webcam      # your camera
python scripts/serve.py --environment warehouse
python scripts/serve.py --port 8080 --db /tmp/missions.db
```

| URL | What |
|---|---|
| `http://localhost:8000/` | Status page (HTML in a browser, JSON to `curl`) |
| `http://localhost:8000/console` | **Mission control — start here** |
| `http://localhost:8000/docs` | Interactive OpenAPI, all 57 routes |
| `ws://localhost:8000/ws/telemetry` | Live frame + health push |

> Port defaults to **8000**. On macOS avoid 5000 and 7000 — AirPlay Receiver
> takes them.

### Data

```bash
python scripts/fetch_data.py list           # what's available
python scripts/fetch_data.py credentials    # which NASA services need a key
python scripts/fetch_data.py status         # what's already cached
python scripts/fetch_data.py perseverance --sol 203     # keyless JPL archive
python scripts/fetch_data.py jezero-dem     # DEM download instructions
python scripts/fetch_data.py ai4mars        # AI4Mars instructions
```

### Tests

```bash
python -m pytest tests -q                      # all 585
python -m pytest tests/test_core.py -q         # NumPy only, no OpenCV
python -m pytest tests -q --cov=backend/perceptos
python -m pytest tests --durations=10          # slowest tests
```

### Command-line API walkthrough

```bash
# 1. Start recording
curl -s localhost:8000/api/missions -X POST \
     -H 'content-type: application/json' -d '{"name":"demo"}'

# 2. Run frames
curl -s localhost:8000/api/pipeline/run -X POST \
     -H 'content-type: application/json' -d '{"frames":5}'

# 3. List lineage nodes, grab an id
curl -s 'localhost:8000/api/lineage?label=features' | python3 -m json.tool

# 4. Trace one back to its origin
curl -s localhost:8000/api/lineage/<node_id> | python3 -m json.tool

# 5. Audit headers on a rendered image
curl -sI localhost:8000/api/frame/gray.png | grep X-PerceptOS

# 6. Terrain (returns UNAVAILABLE without a DEM — by design)
curl -s localhost:8000/api/terrain | python3 -m json.tool

# 7. Plan a route over explicitly-labelled synthetic terrain
curl -s 'localhost:8000/api/navigation/plan?allow_simulation=true' \
     -X POST -H 'content-type: application/json' -d '{"planner":"theta"}'

# 8. Mission report
curl -s 'localhost:8000/api/missions/<id>/report?format=markdown'
```

Step 5 is worth running — colormap, valid fraction, source type and node id
travel with the image, so an `<img>` tag stays auditable.

---

## 5. The console, panel by panel

Open `/console`. Press **Start mission**, then **Run 5 frames**.

| Panel | What it shows | Notes |
|---|---|---|
| **Overview** | Mission, data mode, system health, rover power, lineage counts | Battery is always badged `SIMULATION` |
| **Perception** | Camera, greyscale, processed, features, tracks, depth, point cloud | Every image has a **trace** link |
| **Terrain** | 3D perspective view, hillshade, elevation, slope, roughness, step, traversability, hazard, coverage | Says `UNAVAILABLE` until a DEM exists |
| **Navigation** | Planned route, cost/risk, replanning events, path over cost map | A\* and Theta\*, D\* Lite repair |
| **SLAM** | Trajectory plot, keyframes, landmarks, loop closures, optimisation | Landmarks need stereo depth |
| **Replay** | Scrub recorded missions, stage timings, counterfactuals | Replays; never recomputes |
| **Data lineage** | Every node; click to open the drawer | The centrepiece |
| **Data sources** | Every NASA service and its key requirement | Answers "do I need a key?" |
| **Telemetry** | Frame timings, live WebSocket stream | |

### Why Terrain is empty on first load

**This is deliberate.** No DEM is on disk, and PerceptOS will not substitute
synthetic terrain for a missing survey. Two honest options:

- **Click "Enable SIMULATION terrain"** — instant, everything badged amber.
- **Fetch a real DEM** — see §6.

### The lineage drawer

Click any value anywhere. The drawer walks the chain **root-first**, so it reads
as a derivation: origin at the top, the value you clicked at the bottom. Each
hop shows algorithm, confidence, uncertainty, latency, parameters, and clickable
parent ids.

---

## 6. Real NASA data

### You almost certainly do not need an API key

Verified against NASA documentation. Ask the running system yourself:

```bash
python scripts/fetch_data.py credentials
curl -s localhost:8000/api/datasources | python3 -m json.tool
```

| Service | Endpoint | Key? |
|---|---|---|
| **JPL Mars 2020 Raw Image API** | `mars.nasa.gov/rss/api/?feed=raw_images&category=mars2020&feedtype=json` | **No** |
| NASA Mars Rover Photos | `api.nasa.gov/mars-photos/api/v1/rovers/perseverance/photos` | Yes — fallback only |
| USGS Jezero HiRISE DTM | `astrogeology.usgs.gov/search/map/Mars/Mars2020/JEZ_hirise_soc_006_DTM_...` | **No** |
| AI4Mars v0.6 | `zenodo.org/records/15995036` | **No** |

Three of four need nothing. If you do want the fallback:
`export NASA_API_KEY=...` (free from api.nasa.gov; `DEMO_KEY` allows 30
requests/hour, a personal key 1,000).

### Why the keyless source is also the better one

The Mars Rover Photos API returns `id`, `sol`, `earth_date`, a camera name and a
URL. The JPL feed additionally returns spacecraft clock, rover attitude,
site/drive — and the **CAHV/CAHVOR camera model**.

That model is what makes stereo honest:

```
hs = |H − (H·A)A|      → focal length in pixels
|C_left − C_right|     → physical baseline in metres
```

Both **measured**. From the keyed API those numbers do not exist, so any depth
would rest on invented geometry.

### Fetching imagery

```bash
python scripts/fetch_data.py perseverance --sol 203
python scripts/serve.py --source nasa       # imagery, no depth
python scripts/serve.py --source stereo     # rectified pairs → metric depth
```

If nothing is cached, `--source nasa` **exits with fetch instructions** rather
than silently falling back to synthetic.

### Fetching elevation

```bash
python scripts/fetch_data.py jezero-dem     # prints URL + steps
```

Then crop a tile:

```python
import rasterio, numpy as np
with rasterio.open("DTEEC_....IMG") as ds:
    np.save("datasets/jezero_dem/tile.npy", ds.read(1)[5000:5600, 5000:5600])
```

Restart the server. Terrain now reads `LOCAL_CACHED_DATA`, and you will see
**magenta** regions where the survey had no data — the no-data sentinel
converted to NaN, exactly as it should be.

### The stereo status vocabulary

"These two images are a stereo pair" is several separate claims:

| Status | Depth valid? | Meaning |
|---|---|---|
| `RECTIFIED_STEREO` | **yes** | Calibrated, simultaneous, rectified, residual verified < 2 px |
| `VALID_STEREO` | no | Real camera model and one exposure, not yet rectified |
| `UNRECTIFIED_PAIR` | no | Failed a precondition — not actually a pair |
| `CALIBRATION_UNAVAILABLE` | no | No camera model published; baseline unknown |
| `SIMULATION` | yes | Synthetic pair with known synthetic geometry |

**Status is earned by measurement.** After rectifying, the median vertical
disparity of matched features is measured. Under 2 px → `RECTIFIED_STEREO`.
Unmeasurable or above → stays `VALID_STEREO`, depth withheld.

Verified against known geometry: a scene at a true **32.81 m reads 32.81 m**.

---

## 7. Layer-by-layer code tour

### 7.1 Camera acquisition — `sensors/camera.py`

```python
from perceptos.sensors.camera import CameraManager, WebcamSource, SyntheticSource

with CameraManager(WebcamSource(0), name="navcam") as cam:
    frame = cam.read()               # a Traced[np.ndarray]
    health = cam.health()
    print(health.fps, health.latency_ms, health.grade)
```

**Live drop counts are `ESTIMATED`, file drop counts are measured.** A decoder
knows its true frame index; a webcam does not. Inferring a live count from
elapsed time is fine — presenting it as measured is the standard camera-dashboard
lie.

Health reports `starting` until the rolling window fills: a rate from one
interval is noise, and an indicator that flashes red on every startup is one
operators learn to ignore.

### 7.2 Calibration — `calibration/chessboard.py`

```python
from perceptos.calibration.chessboard import ChessboardCalibrator

cal = ChessboardCalibrator(pattern=(9, 6), square_size_m=0.025)
for image in views:
    cal.add_view(image)
result = cal.calibrate()

result.grade            # excellent | good | marginal | reject
result.warnings()       # specific and actionable, not "quality is low"
result.save("calibration/navcam")
```

Gates on **three** things, not just RMS: reprojection error, corner **coverage**,
and **view diversity**. Twenty views from the same angle in the middle of the
frame give an excellent RMS and badly wrong distortion at the edges — which is
exactly where distortion matters.

> Pattern is **inner corners**. A 10×7-square board is 9×6. A square pattern is
> rotationally ambiguous and is rejected.

### 7.3 Image processing — `imageproc/filters.py`

```python
from perceptos.imageproc.filters import ImageChain

chain = ImageChain().add("clahe", clip_limit=2.0).add("gaussian_blur", ksize=5)
out = chain.apply(gray)
chain.timings       # per-operation latency, not one opaque number
chain.slowest       # '0:clahe'
```

Ordering is validated — `canny` after `otsu` does not raise in OpenCV, it just
produces something plausible and wrong:

```python
ImageChain().add("otsu").add("canny").validate(input_is_gray=True)
# ['step 1 (canny) runs on an already-binarised image; move it before the threshold']
```

CLAHE on colour runs in LAB so only luminance changes; per-channel BGR shifts hue.

### 7.4 Features — `features/extractors.py`

```python
from perceptos.features.extractors import FeatureExtractor

FeatureExtractor.available()        # what THIS OpenCV build has
extractor = FeatureExtractor("ORB", max_features=600)
features  = extractor.detect(gray)

features.spatial_spread     # fraction of an 8×8 grid occupied
features.quality            # count AND spread — 2000 keypoints on one rock
                            # is weaker than 200 spread across the frame
```

Norm follows the **descriptor**, not the detector: ORB/AKAZE/BRISK are binary
(Hamming), SIFT/KAZE are float (L2). Matching binary with L2 "works" — the
matches are just wrong.

### 7.5 Stereo — `stereo/`

```python
from perceptos.stereo.rectify import rectify_pair
from perceptos.stereo.matcher import StereoMatcher
from perceptos.stereo.depth import disparity_to_depth, reproject_to_3d

rect  = rectify_pair(left, right, left_camera_model, right_camera_model)
rect.status              # StereoStatus.RECTIFIED_STEREO
rect.baseline_m          # 0.42 — measured from the two C vectors
rect.epipolar_error_px   # 0.0 — verified, not assumed

disp  = StereoMatcher(algorithm="SGBM", num_disparities=128).compute(rect.left, rect.right)
depth = disparity_to_depth(disp.disparity, rect.rig)
cloud = reproject_to_3d(depth.depth_m, rect.rig)
```

**Depth error grows with the square of range:**

```
sigma_Z = Z² · sigma_d / (f · B)
```

A rig accurate to millimetres at 1 m is accurate to metres at 40 m.
`rig.max_reliable_range()` tells the planner where to stop believing its own
stereo.

Two traps handled: disparity comes back as `CV_16S` **scaled by 16**, and
invalid pixels are a large negative sentinel, not zero. Disparity pinned at the
search minimum is also excluded — it converts to infinite depth, and counting it
reports 90% coverage for a map with no range information in it.

### 7.6 Tracking — `tracking/trackers.py`

```python
from perceptos.tracking.trackers import MultiObjectTracker

tracker = MultiObjectTracker(tracker_type="CSRT", min_hits=3, max_misses=12)
tracks  = tracker.update(frame, detections)
```

**Association runs against the Kalman *prediction*, not the last known box.**
Matching on raw proximity swaps the identities of two crossing objects at the
crossing point, every time.

A track is not reported until seen `min_hits` times — without confirmation,
every one-frame detector false positive becomes a track with an ID.

### 7.7 State estimation — `estimation/kalman.py` and `fusion/ekf.py`

```python
from perceptos.fusion.ekf import PoseEKF, WheelOdometry, Gyroscope, StarTracker, GNSS

ekf = PoseEKF(environment="mars", initial_pose=(0, 0, 0))
ekf.register(WheelOdometry(speed_noise_std=0.02))
ekf.register(Gyroscope(rate_noise_std=0.004))
ekf.register(StarTracker(heading_noise_std_rad=0.003))

ekf.register(GNSS())
# -> ValueError: 'gnss' is not available in 'mars'.
#    There is no satellite navigation constellation at Mars.

ekf.predict(dt=1.0)
ekf.update("wheel_odometry", [0.5], slip_ratio=0.3)   # down-weighted by slip
estimate = ekf.estimate()
estimate.position_error_ellipse    # 95% ellipse, not a circle
estimate.bias_observable           # False without an absolute heading fix
```

Four things that separate a working EKF from one that looks like it works:

1. **Angular innovations are wrapped.** +179° against −179° is a 2° error, not
   358°. Without wrapping, every crossing of ±180 slews the filter — and it only
   shows up on the traverse that happens to cross it.
2. **Gating is suspended during warm-up.** A chi-square gate describes a
   *converged* filter. Applied to an uninitialised one it rejects exactly the
   large correct corrections that would converge it, and the filter deadlocks.
3. **Consistency is judged on rejection rate, not just NIS.** Mean NIS over
   accepted measurements is *censored* — the gate already discarded the high-NIS
   samples, so a filter with R far too small rejects nearly everything and the
   survivors look beautiful.
4. **Gyro bias is frozen when unobservable.** With a gyro alone, `omega` and
   `b_gyro` enter the measurement identically and no data separates them.

### 7.8 Terrain — `mars/traversability.py`, `mars/occupancy.py`

```python
from perceptos.mars.traversability import TraversabilityAnalyser, RoverGeometry

analyser = TraversabilityAnalyser(RoverGeometry(), cell_size_m=0.25)
result   = analyser.analyse(dem)

result.cost              # 0..1
result.hazard_mask       # bool
result.slope_deg, result.step_m, result.roughness_m
result.statistics()
```

**Hazards combine by maximum, not sum or average.** Summing lets very smooth
ground average away a cliff beside it. One lethal hazard is lethal regardless of
how benign the other two are.

**Unobserved terrain is never traversable.** A NaN elevation cell is a hazard,
not a zero. Zero elevation is a claim; NaN is an honest absence of one.

### 7.9 Planning — `mars/planner.py`, `mars/planners.py`

```python
from perceptos.mars.planner import AStarPlanner
from perceptos.mars.planners import ThetaStarPlanner, DStarLitePlanner

plan = ThetaStarPlanner(cost_weight=4.0, inflation_radius_m=0.5) \
         .plan(cost_map, start, goal, cell_size_m=0.25)

plan.length_m, plan.max_cell_cost, plan.turns_deg
plan.estimate_duration_s(), plan.estimate_energy_wh()

dstar = DStarLitePlanner()
dstar.initialise(cost_map, start, goal, cell_size_m=0.25)
dstar.update_costs([((60, 60), 1.0)])     # a rock resolves in the near field
dstar.extract_path()
```

> Every code snippet in this document was executed against the shipped build
> before publication. Doing so found a real defect: D\* Lite's greedy g-descent
> could oscillate between two mutually-cheapest cells after a repair, failing
> extraction on fully routable terrain. Fixed, with a regression test.

**D\* Lite's advantage is conditional.** Measured on a 120×120 grid:

| change | initial | repair | fresh search |
|---|---|---|---|
| off the optimal path | 114 | **0** | 114 |
| on the path, open terrain | 114 | 787 | 783 |
| on the path, cluttered 12% | 2,284 | **36** | 2,280 |

On empty terrain the octile heuristic is perfect and there is nothing to save.
The 63× win appears on cluttered ground — the actual rover case.

**Three inflation subtleties**, all of which produced wrong answers before being
handled: the rover is not a point so hazards are dilated; padding the map border
with `True` makes every goal near the edge unreachable; and a rover already
inside an inflated footprint must be able to back out.

### 7.10 SLAM — `slam/`

```python
from perceptos.slam.system import SlamSystem
from perceptos.slam.graph import Pose2D

slam = SlamSystem()
for pose, descriptors, keypoints in stream:
    slam.process(pose, descriptors=descriptors, keypoints=keypoints)

result = slam.optimise()
result.max_loop_residual        # small even when a closure is FALSE
result.max_odometry_residual    # ← this is where the damage shows
result.suspect_edges
```

**The finding that shaped this module.** A false loop closure usually does *not*
leave a large residual on its own edge — the optimiser satisfies it by deforming
everything else, so the bad edge fits **better** than the good ones. Measured on
a drifted square traverse:

| | closure residual | worst odometry residual |
|---|---|---|
| clean graph | — | **0.000** |
| false closure, no Huber | 0.391 | **1.033** |
| false closure, Huber 0.5 | 0.096 | **0.268** |

Checking loop-closure residuals — the obvious approach — misses precisely the
worst case. Both signals are monitored, and the suspect edge is **reported, never
dropped**: deleting it would hide a front-end failure.

Loop closure passes three filters in series — appearance (Lowe's ratio),
**geometric** (RANSAC rigid transform), and **spatial plausibility** (implied
correction vs accumulated drift). Martian regolith is self-similar for
kilometres, so appearance alone proposes false closures freely.

### 7.11 Datasets — `datasets/`

```python
from perceptos.datasets.manager import DatasetManager
from perceptos.datasets.registry import ModelRegistry

manager = DatasetManager("datasets/versions")
version = manager.ingest_directory("datasets/perseverance_jpl",
                                   name="perseverance", group_from="parent")
manager.save(version)
version.version_id          # SHA-256 of the manifest — names exact bytes

split = version.split(group_key="group", seed=0)
split.leakage.clean         # True

version.split(group_key=None)
# -> LeakageError: random split puts near-duplicate frames in both
#    train and test; accuracy would measure memorisation
```

**Random splitting of rover imagery measures memorisation.** Consecutive Navcam
frames from one sol show the same rocks seconds apart. Measured on 4 sols × 5
frames: random leaked **3 of 4 groups**; grouped leaked **none**.

```python
evaluation = ModelRegistry.evaluate(truth_maps, predictions,
                                    ["soil", "bedrock", "sand", "big_rock"])
evaluation.pixel_accuracy       # 0.97
evaluation.beats_baseline       # False  ← predicting "soil" everywhere
evaluation.weakest_class()      # 'big_rock'
evaluation.warnings()           # ['...0.0% recall: the model finds almost
                                #   none of it... the model is unsafe...']
```

**No single accuracy figure is reported alone.** On AI4Mars, soil covers most of
most frames while `big rock` — the class that threatens the vehicle — covers a
few percent.

> **Training is not included.** PerceptOS records, versions and evaluates
> models; fitting them belongs to the model's own framework.

---

## 8. Complete API reference

**57 routes.** Full interactive docs at `/docs`.

### Status
```
GET    /                                   Status page (HTML or JSON)
GET    /api                                Same, always JSON
GET    /api/status                         Environment + data mode
GET    /api/datasources                    NASA services and key requirements
GET    /api/health          /api/health/live
```

### Environments
```
GET    /api/environments                   All five
GET    /api/environments/{name}
POST   /api/environments/active            Switch without restarting
```

### Pipeline
```
GET    /api/pipeline                       Stage contract
GET    /api/pipeline/diagram               Mermaid, from real wiring
POST   /api/pipeline/run                   {"frames": 5}
POST   /api/pipeline/stages/{stage}        {"enabled": false}
```

### Perception
```
GET    /api/frame                          Slots + node ids
GET    /api/frame/{slot}.png               Rendered, with audit headers
GET    /api/frame/pointcloud.png           Needs RECTIFIED_STEREO
GET    /api/stereo                         Status + measured geometry
```

### Terrain
```
GET    /api/terrain                        Products, or UNAVAILABLE
GET    /api/terrain/{layer}.png            elevation|slope|roughness|step|
                                           traversability|hazard|coverage
GET    /api/terrain/hillshade.png          ?azimuth_deg&altitude_deg
GET    /api/terrain/view3d.png             ?drape&azimuth_deg&vertical_exaggeration
```

### Navigation
```
GET    /api/navigation
POST   /api/navigation/plan                {"planner": "theta"|"astar"}
POST   /api/navigation/replan              {"changes": [[r,c,cost], ...]}
GET    /api/navigation/path.png
```

### SLAM
```
GET    /api/slam                           Trajectory, keyframes, closures
GET    /api/slam/trajectory.png
POST   /api/slam/optimise
```

### Missions, lineage, replay, reports
```
GET    /api/missions                       POST /api/missions
POST   /api/missions/active/stop
GET    /api/missions/{id}                  DELETE /api/missions/{id}
GET    /api/missions/{id}/frames           /frames/{index}
GET    /api/missions/{id}/telemetry/{channel}
GET    /api/missions/{id}/report           ?format=json|csv|markdown

GET    /api/lineage                        ?label&source&frame_index
GET    /api/lineage/{node_id}              ?mission_id  ← the drawer's backend

POST   /api/replay/{mission_id}/open
GET    /api/replay        /frame  /timeline  /profile
POST   /api/replay/seek   /step   /play      /pause
POST   /api/replay/counterfactual
```

### Datasets and models
```
GET    /api/datasets                       POST /api/datasets/ingest
GET    /api/datasets/{version_id}
POST   /api/datasets/{version_id}/split
GET    /api/models    /api/models/compare?ids=a,b
GET    /api/models/{id}   /api/models/{id}/card
```

### Image response headers

Every rendered PNG carries:

```
X-PerceptOS-Colormap          viridis | inferno | hazard | grey | rgb
X-PerceptOS-Vmin / Vmax       normalisation range, in the value's units
X-PerceptOS-Units             m | deg | cost | px
X-PerceptOS-Valid-Fraction    fraction of cells with real data
X-PerceptOS-No-Data-RGB       255,0,255 — magenta, in no colormap
X-PerceptOS-Node-Id           resolves in /api/lineage/{id}
X-PerceptOS-Source-Type       LOCAL_CACHED_DATA | SIMULATION | ...
```

**No-data is drawn as no-data.** Clamping NaN to zero produces a smooth,
complete-looking raster that claims measurements were made where none were.

---

## 9. Writing your own code against it

### A custom stage

```python
import numpy as np
from perceptos.core.stage import PipelineStage, FrameContext
from perceptos.core.provenance import Source

class RockCounter(PipelineStage):
    name     = "rock_counter"
    requires = ("processed",)
    provides = ("rock_count",)
    source   = Source.ESTIMATED

    def run(self, ctx: FrameContext) -> str:
        (image,) = ctx.require("processed")
        count = int((image.value > 200).sum() // 100)
        self.emit(ctx, "rock_count", count,
                  algorithm="ThresholdCounter(>200, /100)",
                  confidence=0.6, units="rocks", parents=[image])
        return f"{count} candidate rocks"
```

### A custom environment

```python
from perceptos.core.environment import Environment, TerrainClass, DataSourceSpec, register_environment
from perceptos.mars.traversability import RoverGeometry

register_environment(Environment(
    name="europa", display_name="Europa Ice Shell",
    gravity_ms2=1.315,
    rover=RoverGeometry(max_slope_deg=15.0, critical_slope_deg=20.0,
                        max_step_m=0.20, max_roughness_m=0.08),
    terrain_classes=(
        TerrainClass(0, "smooth_ice", mobility_cost=0.15, slip_risk=0.40),
        TerrainClass(1, "chaos_terrain", mobility_cost=0.90, slip_risk=0.20),
    ),
    data=DataSourceSpec(notes="No public surface imagery exists."),
    cell_size_m=0.5, cruise_speed_mps=0.05,
))
```

**Gravity does not change the geometric slope limit.** Static tip-over is centre
of mass versus wheelbase — pure geometry. A rover that tips at 45° on Earth tips
at 45° on Europa. Gravity enters the *energy model*, and that is the only place
`gravity_ms2` is used.

### A custom sensor

```python
from perceptos.fusion.ekf import SensorModel, THETA
import numpy as np

class SunSensor(SensorModel):
    name = "sun_sensor"
    angle_indices = (0,)                    # wrap this component
    available_in = frozenset({"mars", "moon"})
    provides_absolute_heading = True        # makes gyro bias observable

    def __init__(self, *, noise_std_rad=0.01, **kw):
        super().__init__(noise_std=[noise_std_rad], **kw)

    def predict_measurement(self, x): return np.array([x[THETA]])

    def jacobian(self, x):
        H = np.zeros((1, 6)); H[0, THETA] = 1.0; return H
```

### Embedding the service

```python
from perceptos.api.service import PerceptOSService
from perceptos.db.store import MissionStore

service = PerceptOSService(store=MissionStore("missions.db"),
                           pipeline_factory=my_factory, environment="mars")
service.start_mission(name="traverse-1")
for _ in range(100):
    service.process_frame()
service.stop_mission()

node = service.search_lineage(label="safe_path_plan")[0]
service.explain(node["id"])
```

---

## 10. Testing

**585 tests, 1 skipped**, ~30 s.

| File | Tests | Covers |
|---|---|---|
| `test_core.py` | 74 | Provenance, lineage, pipeline, Kalman, depth, IBVS, planner |
| `test_vision.py` | 75 | Capability layer, camera, calibration, filters, features, stereo, tracking, VO |
| `test_visualisation.py` | 71 | Rendering, terrain, navigation, 3D views, console |
| `test_data.py` | 60 | Cache, NASA client, environments, DEM, semantic costs |
| `test_nasa_sources.py` | 54 | Source vocabulary, JPL client, stereo verification |
| `test_datasets.py` | 48 | Versioning, leakage, evaluation, registry |
| `test_service.py` | 47 | Store, health, runtime, HTTP API |
| `test_fusion_planning.py` | 46 | EKF, Theta\*, D\* Lite |
| `test_slam.py` | 39 | Pose graph, false closures, keyframes, loop detection |
| `test_rectify.py` | 37 | CAHV decomposition, rectification, stereo pipeline |
| `test_replay_reports.py` | 35 | Replay, counterfactuals, reports |

The suite tests **invariants**, not happy paths:

```python
def test_simulation_cannot_claim_an_external_source():
    with pytest.raises(ProvenanceError, match="cannot declare source_type"):
        Provenance(source=Source.SIMULATION, algorithm="synthetic",
                   source_type="NASA_PUBLIC_NO_KEY")

def test_missing_camera_model_refuses_to_guess():
    assert pair.status is StereoStatus.CALIBRATION_UNAVAILABLE
    assert pair.baseline_m is None

def test_a_majority_class_predictor_is_exposed():
    assert result.pixel_accuracy > 0.95
    assert result.beats_baseline is False
```

Each would pass trivially if the code were willing to fill in a plausible
default. They fail if it ever starts doing so.

---

## 11. File map

```
perceptos/
├── backend/perceptos/
│   ├── core/          1,883  provenance, lineage, stage, pipeline,
│   │                         compute (MPS→CUDA→CPU), cv_compat, environment
│   ├── sensors/       1,153  camera, dataset_source, stereo_source
│   ├── calibration/     411  chessboard
│   ├── imageproc/       333  filters
│   ├── features/        302  extractors
│   ├── stereo/          874  depth, matcher, rectify
│   ├── detection/       258  yolo (optional dependency)
│   ├── tracking/        357  trackers
│   ├── estimation/      270  kalman
│   ├── servoing/        266  ibvs
│   ├── fusion/          694  ekf
│   ├── mars/          2,045  traversability, occupancy, planner, planners,
│   │                         slip, odometry
│   ├── slam/          1,246  graph, frontend, system
│   ├── data/          2,106  cache, mars2020, mars2020_raw, terrain, sources
│   ├── datasets/        872  manager, registry
│   ├── api/           4,006  app, service, analysis, render, replay, console
│   ├── db/              554  store (SQLite + lineage tables)
│   ├── health/          253  monitor
│   └── reports/         297  mission
├── scripts/                  serve.py, fetch_data.py
├── tests/                    11 files, 585 tests
├── configs/modes/            mars_exploration.yaml, warehouse_robotics.yaml
├── docs/BUILD_PLAN.md
└── demo_*.py                 4 runnable demos
```

---

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `python3 --version` shows 3.9 | macOS default. `brew install python@3.12`, use `python3.12` |
| `ModuleNotFoundError: fastapi` | `pip install -r requirements.txt`, venv activated? |
| Terrain says `UNAVAILABLE` | **Working as designed.** Click "Enable SIMULATION terrain", or fetch a DEM |
| Stereo says `UNAVAILABLE` | Default pipeline has no right eye. `--source stereo` |
| Point cloud 404 | Needs `RECTIFIED_STEREO`. Same fix |
| `--source nasa` exits | Nothing cached. `fetch_data.py perseverance --sol 203` |
| Webcam not found (macOS) | System Settings → Privacy & Security → Camera → add Terminal |
| Port in use | Avoid 5000/7000 (AirPlay). `--port 8080` |
| `AKAZE is not available` | OpenCV 5 dropped it. `FeatureExtractor.available()` lists what you have |
| Open3D won't install | No 3.13 wheels. Use 3.12 |
| `schema v1, expects v2` | Old database. Delete `missions.db` or point `--db` elsewhere |
| Tests fail on OpenCV only | `python -m pytest tests/test_core.py -q` — NumPy only |

**Diagnostic order:**

```bash
python -m pytest tests/test_core.py -q     # NumPy only — passes? code is fine
python -m pytest tests -q                  # full — fails? dependency issue
python scripts/fetch_data.py credentials   # NASA config
curl -s localhost:8000/api/status          # server reachable?
```

---

## 13. Engineering decisions worth defending

If you present this, these are the answers to "why did you do it that way?"

**Hazards combine by maximum.** Summing lets smooth ground average away a cliff.

**Unobserved terrain is never traversable.** NaN is a hazard, not a zero.

**Hazards are inflated by the rover radius — but the map border pads with
`False`.** The edge is an arbitrary crop, not a wall.

**The A\* heuristic scales by the *minimum* per-metre cost.** Anything higher is
inadmissible and silently returns plausible-but-dangerous paths.

**Kalman uses Joseph form.** The textbook update accumulates asymmetry until the
covariance stops being positive-definite.

**Depth carries `sigma_Z = Z²·sigma_d/(f·B)`.** Error grows with the *square* of
range.

**IBVS uses damped least squares.** Near singularity the pseudo-inverse commands
a violent lurch.

**Slip embedding needs a sustained window.** One high-slip sample is normal
crossing a rock; sustained slip is the failure that immobilised Spirit at Troy.

**Occupancy log-odds are clamped.** Unclamped, a cell becomes permanently certain
and never recovers from a spurious reading.

**Simultaneity is a hard precondition for stereo, not a score.** A calibrated
pair taken an hour apart scores *below* an uncalibrated simultaneous one.

**`hazard_mask` and `cost >= 0.95` are different sets** — ~2% vs ~6% of a real
Jezero tile. Reachability must use the planner's criterion.

**Rendering never interpolates.** Bilinear on a hazard map invents cells exactly
on the safe/lethal boundary.

**NaN is not valid JSON.** `float(nan)` emits a bare `NaN` token no strict parser
accepts — every lineage request touching a real survey returned 500.

**SQLite foreign keys are switched on explicitly.** Disabled per connection by
default, so a schema full of `REFERENCES` enforces nothing.

**A false loop closure does not leave a large residual on its own edge.** It
corrupts the odometry instead. Checking the obvious thing misses the worst case.

**Greedy g-descent needs cycle protection.** After a D\* Lite repair, g-values
are only guaranteed consistent along the optimal path; elsewhere two neighbours
can each name the other as cheapest. An optimal path never revisits a cell, so
refusing to do so breaks the tie without excluding any valid route.

**Random splitting of rover imagery measures memorisation.** 3 of 4 groups leaked
in the measured case.

**A single accuracy figure is never reported alone.** 97% while finding zero
hazards.

**Vertical exaggeration is a display choice and ships in the headers.** It
changes how steep terrain looks without changing what it is.

---

## Honest limitations

- **Not verified against live NASA endpoints.** The development sandbox has no
  route to `mars.nasa.gov`. Clients are built against documented schemas and
  tested with fixtures; parsing is defensive, but the first live call happens on
  your machine.
- **CAHVOR distortion is not corrected.** Its `O`/`R` terms are not OpenCV's
  `k1..k3`; approximating would introduce error while claiming correction.
  `distortion_corrected` is permanently `False`.
- **SLAM is SE(2), not SE(3).** Well-conditioned for a surface rover; cannot
  represent driving over genuinely 3D structure. No bundle adjustment.
- **No model training.** Records, versions and evaluates only.
- **No annotation UI.** Ingests and versions existing labels; does not draw them.
- **The console is one HTML file, not React.** Deliberate: no Node, no second
  port, no CORS story. The OpenAPI schema is complete if you want to generate a
  React client.
- **Not verified on Apple Silicon.** Device detection is tested; wheel
  resolution is not.

---

*Generated for PerceptOS v1.0. Values labelled `SIMULATION` are synthetic;
values labelled `UNAVAILABLE` were not measured and have not been substituted.*
