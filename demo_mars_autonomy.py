#!/usr/bin/env python3
"""
PerceptOS — Mars Rover Autonomy demonstration
=============================================

Runs the full autonomy chain end to end on synthetic terrain and prints the
data lineage for the resulting drive decision.

    stereo point cloud  ->  occupancy + elevation fusion
                        ->  traversability (slope / step / roughness)
                        ->  A* safe path planning
                        ->  wheel slip monitoring
                        ->  drive decision

Every value produced here is tagged ``SIMULATION`` because the terrain is
synthetic.  That tag propagates automatically through every downstream value —
there is no way to launder simulated data into a value that looks measured.
Swap :func:`synthetic_terrain` for a real stereo point cloud and the same
pipeline produces ``CALCULATED`` values with no other change.

Run:  python demo_mars_autonomy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from perceptos.core.lineage import LineageGraph          # noqa: E402
from perceptos.core.provenance import Source, traced      # noqa: E402
from perceptos.mars.occupancy import GridConfig, OccupancyGrid  # noqa: E402
from perceptos.mars.planner import AStarPlanner, PlanningFailure  # noqa: E402
from perceptos.mars.slip import WheelSlipEstimator        # noqa: E402
from perceptos.mars.traversability import (               # noqa: E402
    RoverGeometry,
    TraversabilityAnalyser,
)

CELL = 0.25          # metres per cell
GRID = 120           # cells per side  -> 30 m x 30 m of terrain


def synthetic_terrain(seed: int = 42) -> np.ndarray:
    """A Jezero-flavoured DEM: regional tilt, a crater, scattered rocks, ripples.

    SIMULATION ONLY.  Returned in metres.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:GRID, 0:GRID].astype(np.float64) * CELL

    dem = 0.04 * xx                                   # gentle regional slope
    # Impact crater: raised rim, depressed floor.
    cr, cc, radius = 22.0, 20.0, 6.0
    r = np.hypot(yy - cr, xx - cc)
    dem += np.where(r < radius, -1.4 * (1 - (r / radius) ** 2), 0.0)
    dem += 0.7 * np.exp(-((r - radius) ** 2) / 1.2)    # rim

    # Scattered boulders — the primary step hazard.
    for _ in range(28):
        by, bx = rng.uniform(0, GRID * CELL, 2)
        size = rng.uniform(0.3, 0.9)
        height = rng.uniform(0.25, 0.8)
        dem += height * np.exp(-(((yy - by) ** 2 + (xx - bx) ** 2) / (2 * size ** 2)))

    dem += 0.03 * np.sin(xx * 2.2) * np.cos(yy * 1.7)  # aeolian ripples
    dem += rng.normal(0.0, 0.012, dem.shape)           # sensor noise
    return dem


def dem_to_point_cloud(dem: np.ndarray) -> np.ndarray:
    """Flatten a DEM into an (N, 3) world-frame cloud, as stereo would deliver."""
    rows, cols = dem.shape
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64) * CELL
    return np.stack([xx.ravel(), yy.ravel(), dem.ravel()], axis=1)


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def main() -> int:
    lineage = LineageGraph()

    # ---------------------------------------------------------- 1. perception
    banner("1  SENSOR ACQUISITION  (simulated stereo point cloud)")
    dem_truth = synthetic_terrain()
    cloud = dem_to_point_cloud(dem_truth)

    cloud_tv = traced(
        cloud,
        source=Source.SIMULATION,
        algorithm="synthetic_terrain(Jezero-analogue DEM)",
        confidence=1.0,
        units="m",
        label="stereo_point_cloud",
        parameters={"seed": 42, "cell_size_m": CELL},
    )
    lineage.record(cloud_tv, stage="acquisition")
    print(f"  points          : {len(cloud):,}")
    print(f"  source          : {cloud_tv.source.value.upper()}  (badged in UI)")
    print(f"  elevation range : {cloud[:, 2].min():.2f} .. {cloud[:, 2].max():.2f} m")

    # -------------------------------------------------------- 2. mapping
    banner("2  OCCUPANCY + ELEVATION MAPPING")
    grid = OccupancyGrid(
        GridConfig(width_cells=GRID, height_cells=GRID, cell_size_m=CELL)
    )
    fused = grid.integrate_points(cloud)

    # Hazard-camera style range scan from the rover's current pose, so the
    # occupancy layer carries free-space evidence as well as elevation.
    sensor_xy = (1.5, 1.5)
    angles = np.linspace(-np.pi / 3, np.pi / 3, 90)
    ranges = 6.0 + 4.0 * np.abs(np.sin(angles * 3.0))
    rays = grid.integrate_scan(sensor_xy, ranges.tolist(), angles.tolist(),
                               max_range_m=12.0)

    dem_tv = grid.traced_elevation(parents=[cloud_tv])
    lineage.record(dem_tv, stage="mapping")
    for key, value in grid.statistics().items():
        print(f"  {key:<22}: {value}")
    print(f"  {'points fused':<22}: {fused:,}")
    print(f"  {'ray-cast cells':<22}: {rays:,}")

    # ------------------------------------------------------ 3. traversability
    banner("3  TRAVERSABILITY ANALYSIS")
    analyser = TraversabilityAnalyser(
        RoverGeometry(), cell_size_m=CELL, window_cells=3
    )
    trav_tv = analyser.analyse_traced(dem_tv)
    lineage.record(trav_tv, stage="traversability")
    trav = trav_tv.value
    for key, value in trav.statistics().items():
        print(f"  {key:<22}: {value}")
    print(f"  {'confidence':<22}: {trav_tv.confidence:.3f}")

    # ------------------------------------------------------------ 4. planning
    banner("4  SAFE PATH PLANNING")
    planner = AStarPlanner(
        cost_weight=6.0, turn_penalty=0.3, inflation_radius_m=0.6
    )
    start, goal = (5, 5), (GRID - 6, GRID - 6)
    try:
        plan_tv = planner.plan_traced(
            trav_tv, start, goal, origin=(0.0, 0.0)
        )
    except PlanningFailure as exc:
        print(f"  PLANNING FAILED: {exc}")
        print(f"  diagnostics    : {exc.diagnostics}")
        return 1
    lineage.record(plan_tv, stage="planning")
    plan = plan_tv.value
    for key, value in plan.to_dict().items():
        if key in ("waypoints", "cells"):
            continue
        print(f"  {key:<22}: {value}")
    print(f"  {'waypoints':<22}: {len(plan.world_points)}")
    print(f"  {'source (inherited)':<22}: {plan_tv.source.value.upper()}")

    # ------------------------------------------------------ 5. slip monitoring
    banner("5  WHEEL SLIP MONITORING  (first 12 drive steps)")
    slip_estimator = WheelSlipEstimator(window=6)
    rng = np.random.default_rng(7)
    estimate = None
    for i in range(12):
        commanded = 0.5
        # Slip rises as the rover climbs the crater rim around steps 5-9.
        severity = 0.55 if 5 <= i <= 9 else 0.06
        measured = commanded * (1.0 - severity - rng.normal(0, 0.02))
        estimate = slip_estimator.update(
            commanded, measured, timestamp=float(i), slope_deg=18.0, vo_confidence=0.82
        )
        flag = "EMBEDDING" if estimate.is_embedding else ("SLIP" if estimate.is_slipping else "ok")
        print(f"  step {i:>2}  slip {estimate.slip_ratio:>6.1%}  [{flag}]")
    slip_tv = slip_estimator.traced_estimate(estimate, parents=[plan_tv])
    lineage.record(slip_tv, stage="slip")
    print(f"\n  recommendation : {estimate.recommendation}")

    # ------------------------------------------------------- 6. data lineage
    banner("6  DATA LINEAGE  —  why did the rover choose this path?")
    explanation = lineage.explain(plan_tv.id)
    print(f"  derivation depth      : {explanation['depth']} nodes")
    print(f"  weakest confidence    : {explanation['weakest_confidence']:.3f}")
    print(f"  simulation in chain   : {explanation['contains_simulation']}")
    print(f"  total latency         : {explanation['total_latency_ms']} ms")
    print("\n  algorithm chain:")
    for i, node in enumerate(explanation["chain"]):
        prov = node["provenance"]
        conf = prov["confidence"]
        conf_s = f"{conf:.2f}" if conf is not None else " -- "
        print(
            f"    {i + 1}. {node['label']:<22} "
            f"{prov['source']:<12} conf {conf_s}  {prov['algorithm'][:44]}"
        )

    banner("MERMAID LINEAGE GRAPH  (rendered in the UI drawer)")
    print(explanation["mermaid"])

    print(
        "\nEvery number above is traceable to its origin. "
        "Nothing is displayed without a source tag.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
