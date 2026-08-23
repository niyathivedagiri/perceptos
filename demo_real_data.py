#!/usr/bin/env python3
"""
PerceptOS — Real Data Mode
==========================

Demonstrates the switch the spec asks for::

    Mode: SIMULATION  |  REAL

The mode is not a configuration flag you set by hand and hope is true. It is
*detected* from what is actually on disk, and announced. If real NASA data is
cached, the pipeline uses it and every value carries ``LIVE_SENSOR`` or
``LIVE_CAMERA``. If it is not, the pipeline runs synthetic terrain and every
value carries ``SIMULATION`` — propagated automatically through the whole
dependency chain, with no way to launder it back.

The algorithms are identical in both modes. Only the data source changes.

    python scripts/fetch_data.py perseverance --sol 203
    python demo_real_data.py

Run:  python demo_real_data.py [--environment mars|moon|earth|warehouse|factory]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from perceptos.core.environment import get_environment          # noqa: E402
from perceptos.core.lineage import LineageGraph                 # noqa: E402
from perceptos.core.provenance import Source, traced            # noqa: E402
from perceptos.data.cache import DatasetCache                   # noqa: E402
from perceptos.data.terrain import (                            # noqa: E402
    denoise_labels,
    fuse_costs,
    load_dem_traced,
    semantic_cost_layer,
)
from perceptos.mars.planner import AStarPlanner, PlanningFailure  # noqa: E402
from perceptos.mars.traversability import TraversabilityAnalyser  # noqa: E402

ROOT = Path(__file__).parent
DATASETS = ROOT / "datasets"


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def find_real_dem() -> Path | None:
    """Look for a cached DEM tile. Presence of the file *is* the mode switch."""
    for candidate in sorted(DATASETS.glob("*_dem/*.npy")) + sorted(DATASETS.glob("*dem*/*.npy")):
        return candidate
    return None


def synthetic_dem(cell_size_m: float, size: int = 120, seed: int = 42) -> np.ndarray:
    """Fallback terrain. SIMULATION ONLY — tagged as such by the caller."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) * cell_size_m
    dem = 0.04 * xx
    r = np.hypot(yy - 22.0, xx - 20.0)
    dem += np.where(r < 6.0, -1.4 * (1 - (r / 6.0) ** 2), 0.0)
    dem += 0.7 * np.exp(-((r - 6.0) ** 2) / 1.2)
    for _ in range(28):
        by, bx = rng.uniform(0, size * cell_size_m, 2)
        s, h = rng.uniform(0.3, 0.9), rng.uniform(0.25, 0.8)
        dem += h * np.exp(-(((yy - by) ** 2 + (xx - bx) ** 2) / (2 * s ** 2)))
    return dem + rng.normal(0.0, 0.012, dem.shape)


def synthetic_labels(shape: tuple[int, int], seed: int = 7) -> np.ndarray:
    """Stand-in for AI4Mars output. SIMULATION ONLY.

    Includes a sand patch on flat ground, which is the case that proves why
    the semantic layer matters.
    """
    rng = np.random.default_rng(seed)
    labels = np.zeros(shape, dtype=np.uint8)                    # soil
    labels[rng.random(shape) < 0.18] = 1                        # bedrock
    labels[70:95, 20:60] = 2                                    # a sand sheet
    labels[30:36, 75:82] = 3                                    # a genuine boulder
    labels[55:59, 40:45] = 3                                    # and another
    labels[rng.random(shape) < 0.02] = 3                        # segmentation speckle
    return labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="mars",
                        choices=["mars", "moon", "earth", "warehouse", "factory"])
    args = parser.parse_args()

    env = get_environment(args.environment)
    lineage = LineageGraph()

    # ------------------------------------------------------ 0. mode detection
    banner("0  MODE DETECTION")
    real_dem = find_real_dem()
    mode = "REAL" if real_dem else "SIMULATION"
    print(f"  environment      : {env.display_name} ({env.name})")
    print(f"  gravity          : {env.gravity_ms2} m/s^2")
    print(f"  supervision      : {env.supervision}"
          + (f"  (light time {env.one_way_light_time_s[0]:.0f}-"
             f"{env.one_way_light_time_s[1]:.0f} s)" if env.is_remote() else ""))
    print(f"  cached DEM       : {real_dem if real_dem else 'none'}")
    print(f"\n  >>> MODE: {mode} <<<")
    if mode == "SIMULATION":
        print("  No cached NASA elevation found, so terrain is synthetic and")
        print("  every derived value below will carry the SIMULATION tag.")
        print("  To switch to REAL:  python scripts/fetch_data.py jezero-dem")
    else:
        print("  Real HiRISE elevation found. Values carry LIVE_SENSOR.")

    stats = DatasetCache(DATASETS).statistics()
    if stats["datasets"]:
        print("\n  cached datasets:")
        for name, info in stats["datasets"].items():
            print(f"    {name:<22} {info['files']:>5} files  {info['bytes'] / 1e6:>8.1f} MB")

    # ------------------------------------------------------------ 1. elevation
    banner("1  ELEVATION")
    if real_dem:
        dem_tv = load_dem_traced(real_dem, cell_size_m=env.cell_size_m)
    else:
        dem_tv = traced(
            synthetic_dem(env.cell_size_m),
            source=Source.SIMULATION,
            algorithm="synthetic_terrain(crater + boulder field)",
            confidence=1.0, units="m", label="dem_elevation",
            parameters={"cell_size_m": env.cell_size_m, "seed": 42},
        )
    lineage.record(dem_tv, stage="terrain")
    dem = dem_tv.value
    print(f"  shape            : {dem.shape}")
    print(f"  cell size        : {env.cell_size_m} m")
    print(f"  relief           : {np.nanmin(dem):.2f} .. {np.nanmax(dem):.2f} m")
    print(f"  SOURCE           : {dem_tv.source.value.upper()}")
    print(f"  algorithm        : {dem_tv.provenance.algorithm}")
    print(f"  confidence       : {dem_tv.confidence}")

    # ---------------------------------------------------- 2. geometric hazards
    banner("2  GEOMETRIC TRAVERSABILITY  (unchanged module, new input)")
    analyser = TraversabilityAnalyser(env.rover, cell_size_m=env.cell_size_m,
                                      window_cells=3)
    trav_tv = analyser.analyse_traced(dem_tv)
    lineage.record(trav_tv, stage="traversability")
    for key, value in trav_tv.value.statistics().items():
        print(f"  {key:<22}: {value}")

    # ---------------------------------------------------- 3. semantic hazards
    banner("3  SEMANTIC LAYER  (why segmentation earns its place)")
    labels = synthetic_labels(dem.shape)
    labels_tv = traced(
        labels, source=Source.SIMULATION,
        algorithm="stand-in for AI4Mars segmentation model",
        confidence=0.84, units="class_id", label="terrain_labels",
        parameters={"classes": list(env.class_names),
                    "note": "replace with a model trained on AI4Mars v0.6"},
    )
    lineage.record(labels_tv, stage="segmentation")

    # Segmentation speckle must be cleaned before the lethal classes are
    # inflated by the rover radius, or a few stray pixels block the whole map.
    raw_lethal = int((labels == env.terrain_class("big_rock").id).sum())
    labels = denoise_labels(labels, env, min_blob_cells=3)
    kept_lethal = int((labels == env.terrain_class("big_rock").id).sum())
    print(f"  speckle removed  : {raw_lethal - kept_lethal} isolated 'big_rock' "
          f"pixels of {raw_lethal}")

    semantic = semantic_cost_layer(labels, env)
    geometric = trav_tv.value.cost
    fused = fuse_costs(geometric, semantic)

    sand_id = env.terrain_class("sand").id
    sand_mask = labels == sand_id
    print(f"  classes          : {', '.join(env.class_names)}")
    print(f"  sand coverage    : {sand_mask.mean():.1%}")
    if sand_mask.any():
        print(f"  over the sand sheet:")
        print(f"    geometric cost : {geometric[sand_mask].mean():.3f}  "
              "(flat ground looks free)")
        print(f"    semantic cost  : {semantic[sand_mask].mean():.3f}  "
              "(slip hazard)")
        print(f"    fused (max)    : {fused[sand_mask].mean():.3f}")
        print("  Averaging instead of taking the maximum would hide the sand.")

    fused_tv = trav_tv.derive(
        fused, algorithm="fuse_costs(max(geometric, semantic))",
        source=Source.CALCULATED, units="cost",
        extra_parents=[labels_tv], label="fused_cost_map",
        parameters={"fusion": "elementwise maximum"},
    )
    lineage.record(fused_tv, stage="fusion")

    # -------------------------------------------------------------- 4. planning
    banner("4  PATH PLANNING OVER THE FUSED COST MAP")
    planner = AStarPlanner(cost_weight=6.0, turn_penalty=0.3,
                           inflation_radius_m=env.rover.wheel_radius_m * 2)
    rows, cols = fused.shape
    start, goal = (5, 5), (rows - 6, cols - 6)
    try:
        plan = planner.plan(fused, start, goal, cell_size_m=env.cell_size_m)
    except PlanningFailure as exc:
        print(f"  PLANNING FAILED: {exc}")
        print(f"  diagnostics    : {exc.diagnostics}")
        return 1

    plan_tv = fused_tv.derive(
        plan, algorithm="AStarPlanner(octile, inflated, turn-penalised)",
        source=Source.CALCULATED, units="m", label="safe_path_plan",
    )
    lineage.record(plan_tv, stage="planning")

    for key, value in plan.to_dict().items():
        if key in ("waypoints", "cells"):
            continue
        print(f"  {key:<22}: {value}")
    print(f"  {'energy (env model)':<22}: "
          f"{plan.length_m * env.energy_per_metre_wh(1025):.2f} Wh")

    # --------------------------------------------------------------- 5. lineage
    banner("5  DATA LINEAGE OF THE DRIVE DECISION")
    explanation = lineage.explain(plan_tv.id)
    print(f"  depth                 : {explanation['depth']} nodes")
    print(f"  weakest confidence    : {explanation['weakest_confidence']}")
    print(f"  simulation in chain   : {explanation['contains_simulation']}")
    print()
    for i, node in enumerate(explanation["chain"]):
        prov = node["provenance"]
        conf = prov["confidence"]
        conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else " -- "
        print(f"    {i + 1}. {node['label']:<18} {prov['source']:<11} "
              f"conf {conf_s}  {prov['algorithm'][:42]}")

    print()
    if explanation["contains_simulation"]:
        print("  This path is SIMULATED. One synthetic ancestor contaminates the")
        print("  entire chain, by design — there is no code path that produces a")
        print("  value claiming to be measured when any input was not.")
    else:
        print("  Every ancestor of this path is real NASA/USGS measurement.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
