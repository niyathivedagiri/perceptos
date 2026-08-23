# PerceptOS

**A robotic perception and autonomy framework that can explain every decision it makes.**

Turns camera images and terrain elevation data into safe driving routes — and lets you
click any number on screen and trace it back to the raw data it came from.

Built as a self-directed project to learn robotic perception properly.

![Python](https://img.shields.io/badge/python-3.12+-blue)
![Tests](https://img.shields.io/badge/tests-586%20passing-brightgreen)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## What it does

```
camera images  ──→  corners  ──→  stereo  ──→  height map
                                                    │
                                                    ▼
     route  ◄──  path planning  ◄──  hazard analysis
```

Given two camera images or a terrain elevation file, PerceptOS:

1. Finds trackable features and measures how the camera moved
2. Computes distance to everything using stereo geometry
3. Builds a height map, then measures slope, ledges and roughness per cell
4. Plans a route that avoids hazards rather than just taking the shortest line
5. Records where every single number came from

---

## The distinguishing feature

Most perception code displays numbers with no indication of whether they were
measured, computed, or invented. PerceptOS refuses to.

Every value carries its origin. Click a planned route and you get:

```
safe path plan          calculated     conf 0.986
    ↑ derived from
traversability map      calculated     conf 0.986
    ↑ derived from
elevation map           live_sensor    conf 0.986
    ↑
Jezero crater DEM       NASA/USGS survey data
```

Six labels, and they cannot be faked:

| Label | Meaning |
|---|---|
| `LIVE_CAMERA` / `LIVE_SENSOR` | Measured by a real instrument |
| `LOCAL_CACHED_DATA` | Real data, downloaded earlier |
| `CALCULATED` | Arithmetic on traced inputs |
| `ESTIMATED` | Statistical model — **must** carry a confidence |
| `SIMULATION` | Synthetic. Spreads to everything derived from it |
| `UNAVAILABLE` | Genuinely missing. **Never** silently replaced |

The rules are enforced by the code, not by discipline:

```python
# An estimate without an uncertainty is an unfalsifiable claim
Provenance(source=Source.ESTIMATED, algorithm="detector")
# → ProvenanceError: ESTIMATED value must carry a confidence

# Synthetic data cannot be relabelled as measured
sim   = traced(1.0, source=Source.SIMULATION, algorithm="synthetic_terrain")
child = sim.derive(2.0, algorithm="fuse", source=Source.LIVE_SENSOR)
child.source      # → Source.SIMULATION   (the request is overridden)
```

If terrain data is missing, the UI shows `UNAVAILABLE` rather than substituting a
synthetic map so the screen looks full.

---

## Is this an AI project?

**No, and this README won't pretend otherwise.**

Every algorithm is classical: gradients, stereo geometry, Kalman filtering, graph
search, least-squares optimisation. **Nothing is trained. No model weights are
shipped. The default pipeline contains no neural network.**

There is an optional YOLOv11 object-detection stage. It is a pretrained model, not
one I trained, it is not in the default pipeline, and it degrades to `SKIPPED`
when PyTorch is absent — which is the normal case.

**What it actually is:** computer vision, robotics algorithms, applied mathematics
and software engineering.

---

## Installation

Requires **Python 3.12+**. macOS ships 3.9 as `python3`.

```bash
git clone https://github.com/<your-username>/perceptos.git
cd perceptos

python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**Minimal install** — the geometry, planning, filtering and SLAM run on NumPy alone:

```bash
pip install numpy pytest
python -m pytest tests/test_core.py -q      # 74 tests, no OpenCV needed
```

> PyTorch resolves the Apple Silicon build automatically. Do **not** add
> `--index-url` — that is the CUDA instruction and gives a broken install.

---

## Running it

### Verify the install

```bash
python -m pytest tests -q          # expect: 586 passed, 1 skipped
```

### Four demos, no server needed

```bash
python demo_mars_autonomy.py        # terrain → hazards → route → audit trail
python demo_perception_pipeline.py  # camera → features → tracking
python demo_real_data.py            # real-vs-simulation data switching
python demo_fusion_navigation.py    # sensor fusion + dynamic replanning
```

### The web interface

```bash
python scripts/serve.py
```

| URL | What |
|---|---|
| `http://localhost:8000/console` | **Mission control — start here** |
| `http://localhost:8000/docs` | Interactive API, all 57 routes |
| `http://localhost:8000/` | Service status |

In the console: press **Start mission**, then **Run 5 frames**, then explore the
Perception, Terrain and Data Lineage tabs.

> The Terrain tab says `UNAVAILABLE` on first run. **That is deliberate** — no
> elevation data is present, and synthetic terrain is not substituted silently.
> Click *"Enable SIMULATION terrain"*, or fetch a real DEM (below).

### Other input sources

```bash
python scripts/serve.py --source webcam         # your camera
python scripts/serve.py --source nasa           # cached Perseverance frames
python scripts/serve.py --source stereo         # cached pairs → metric depth
python scripts/serve.py --environment warehouse # different vehicle profile
python scripts/serve.py --port 8080
```

---

## Real NASA data

**No API key is required for the primary sources.** Ask the running system:

```bash
python scripts/fetch_data.py credentials
```

| Source | Endpoint | Key needed? |
|---|---|---|
| **JPL Mars 2020 Raw Images** | `mars.nasa.gov/rss/api/` | **No** |
| USGS Jezero HiRISE DEM | `astrogeology.usgs.gov` | **No** |
| AI4Mars labels | `zenodo.org/records/15995036` | **No** |
| NASA Mars Rover Photos *(fallback)* | `api.nasa.gov` | Yes |

The keyless JPL feed is also the *better* source: it publishes the **CAHV camera
model**, which gives the real baseline and focal length. Without those, stereo depth
would rest on invented geometry — so PerceptOS reports
`CALIBRATION_UNAVAILABLE` rather than guessing.

```bash
export NASA_API_KEY=...                              # only for the fallback
python scripts/fetch_data.py perseverance --sol 203  # keyless
python scripts/fetch_data.py jezero-dem              # prints instructions
```

---

## Algorithms

| Algorithm | Purpose | Type |
|---|---|---|
| Gradient slope | Steepness from elevation | Mathematics |
| Hazard fusion | Slope + step + roughness → one cost | Mathematics |
| Obstacle inflation | Grow hazards by rover radius | Robotics |
| **A\*** | Baseline route search | Graph search |
| **Theta\*** | Any-angle routes, fewer turns | Graph search |
| **D\* Lite** | Repair a plan instead of replanning | Incremental search |
| **Kalman filter** | Track a moving target | Estimation |
| **EKF** | Fuse wheel, gyro, visual, star tracker | Estimation |
| **Pose-graph SLAM** | Remove drift via loop closure | Least-squares optimisation |
| ORB features | Find and match corners | Computer vision |
| SGBM stereo | Pixel matching between two images | Computer vision |
| CAHV decomposition | NASA camera model → geometry | Computer vision |
| IBVS | Keep a camera on target | Control theory |

**None of these are machine learning.**

### Measured comparisons

Same 30 × 30 m terrain, same start and goal:

| Planner | Length | Turning | Waypoints | Nodes searched |
|---|---|---|---|---|
| A\* | 43.24 m | 450° | 128 | 55,607 |
| **Theta\*** | **42.92 m** | **240°** | **14** | **9,797** |

D\* Lite, after a rock appears mid-route:

| | Nodes searched |
|---|---|
| Replan from scratch | 9,152 |
| **D\* Lite repair** | **4** |

Stereo depth against known ground truth:

| Pixel shift | True distance | Measured |
|---|---|---|
| 16 px | 32.81 m | **32.81 m** |
| 32 px | 16.41 m | **16.41 m** |
| 64 px | 8.20 m | **8.20 m** |

---

## Project structure

```
perceptos/
├── backend/perceptos/
│   ├── core/          provenance, lineage, pipeline, device selection
│   ├── sensors/       camera, dataset replay, stereo sources
│   ├── calibration/   chessboard calibration with quality gating
│   ├── imageproc/     filters (CLAHE, blur, threshold, edges)
│   ├── features/      ORB / FAST / SIFT / Harris extraction and matching
│   ├── stereo/        rectification, block matching, depth geometry
│   ├── tracking/      multi-object tracking with Kalman association
│   ├── estimation/    Kalman filter
│   ├── fusion/        EKF sensor fusion
│   ├── mars/          traversability, occupancy, A*/Theta*/D* Lite, slip
│   ├── slam/          pose graph, keyframes, loop closure
│   ├── data/          NASA clients, DEM loading, provenance cache
│   ├── datasets/      versioning, leakage-aware splitting, model registry
│   ├── api/           FastAPI service, rendering, replay, console UI
│   ├── db/            SQLite with first-class lineage tables
│   └── reports/       JSON / CSV / Markdown mission reports
├── scripts/           serve.py, fetch_data.py
├── tests/             586 tests across 11 files
├── docs/              GUIDE.md, BUILD_PLAN.md
└── demo_*.py          4 runnable demos
```

---

## Testing

```bash
python -m pytest tests -q                      # all 586
python -m pytest tests/test_core.py -q         # NumPy only
python -m pytest tests -q --cov=backend/perceptos
```

The suite tests **invariants**, not happy paths:

```python
def test_simulation_cannot_claim_an_external_source(): ...
def test_missing_camera_model_refuses_to_guess(): ...
def test_unobserved_terrain_is_never_traversable(): ...
def test_a_majority_class_predictor_is_exposed(): ...
```

Each would pass trivially if the code were willing to fill in a plausible default.
They fail if it ever starts doing so.

---

## Limitations

Stated plainly, because a project that oversells is worse than one that doesn't.

- **No hardware.** No motors, no actuators, no vehicle. Routes and velocity
  commands are computed and stored, never executed.
- **No machine learning.** No trained models, no shipped weights. The optional
  YOLO stage is not in the default pipeline.
- **Not tested against live NASA endpoints.** The clients are built against
  documented schemas and tested with fixtures; parsing is defensive, but the
  first live call happens on your machine.
- **CAHVOR lens distortion is not corrected.** Its parameters are not OpenCV's
  distortion model, and approximating would introduce error while claiming a
  correction. `distortion_corrected` is permanently `False`.
- **SLAM is 2D** (position + heading). Suitable for a surface rover; cannot
  represent driving over genuinely 3D structure. No bundle adjustment.
- **The console is a single HTML file, not React.** Deliberate: no Node, no build
  step, no second server. The OpenAPI schema is complete if you want to generate
  a proper frontend.
- **No control loop.** IBVS is implemented and tested but nothing consumes it.

---

## Possible next steps

- Rectify real Perseverance stereo pairs end to end and validate depth against
  published range data
- Train a terrain classifier on AI4Mars and fuse semantic cost with geometric cost
- Extend SLAM to 3D with bundle adjustment
- Connect a real robot and close the control loop
- Continuous integration across macOS and Linux, OpenCV 4.x and 5.x

---

## Documentation

- **[docs/GUIDE.md](docs/GUIDE.md)** — full guide: concepts, code tour, API reference,
  troubleshooting
- **[docs/BUILD_PLAN.md](docs/BUILD_PLAN.md)** — how the project was phased, with data
  sourcing

---

## Licence

MIT — see [LICENSE](LICENSE).

NASA/JPL/USGS data is public domain and is **not redistributed here** — only the
instructions to fetch it. Credit: NASA/JPL-Caltech/USGS.
