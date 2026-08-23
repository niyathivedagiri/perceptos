#!/usr/bin/env python3
"""
PerceptOS — sensor fusion and dynamic replanning
================================================

A rover drives a planned route while:

* an **EKF** fuses wheel odometry, a gyro, visual odometry and a star tracker
  into one pose estimate with a full covariance;
* **D\\* Lite** repairs the plan each time the hazard camera resolves terrain
  that was invisible from further away;
* **wheel slip** is derived from the disagreement between the encoders and
  visual odometry, and feeds back into the drive decision.

Everything below is either computed or explicitly simulated. The rover's true
trajectory is synthetic — there is no rover — so every value carries
``SIMULATION`` and the lineage says so.

Run:  python demo_fusion_navigation.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from perceptos.core.environment import get_environment            # noqa: E402
from perceptos.core.lineage import LineageGraph                   # noqa: E402
from perceptos.core.provenance import Source, traced              # noqa: E402
from perceptos.fusion.ekf import (                                # noqa: E402
    Gyroscope,
    PoseEKF,
    StarTracker,
    VisualOdometrySensor,
    WheelOdometry,
    wrap_angle,
)
from perceptos.mars.planners import DStarLitePlanner, ThetaStarPlanner  # noqa: E402
from perceptos.mars.planner import AStarPlanner                    # noqa: E402
from perceptos.mars.slip import WheelSlipEstimator                 # noqa: E402

SIZE = 120
CELL = 0.25


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def build_terrain(seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(known_cost, true_cost)``.

    The rover starts with an orbital cost map. The true map contains hazards
    that orbit could not resolve — which is the entire reason a rover carries
    a hazard camera and a replanner.
    """
    rng = np.random.default_rng(seed)
    known = np.clip(rng.random((SIZE, SIZE)) * 0.25, 0.0, 1.0)
    # Orbital imagery resolves metre-scale boulders, so they are blocks rather
    # than single pixels. Scattering isolated lethal pixels instead would, after
    # rover-radius inflation, block the map outright - the same failure the
    # segmentation speckle filter exists to prevent.
    for _ in range(30):
        r, c = rng.integers(10, SIZE - 12, 2)
        known[r:r + 2, c:c + 2] = 1.0

    true = known.copy()
    for _ in range(14):                                # sub-orbital-resolution rocks
        r, c = rng.integers(20, SIZE - 20, 2)
        true[r:r + 3, c:c + 3] = 1.0

    for grid in (known, true):
        grid[0:8, 0:8] = 0.0
        grid[SIZE - 10:, SIZE - 10:] = 0.0
    return known, true


def main() -> int:
    env = get_environment("mars")
    lineage = LineageGraph()
    known, true = build_terrain()

    known_tv = traced(
        known, source=Source.SIMULATION,
        algorithm="synthetic orbital cost map",
        confidence=0.7, units="cost", label="orbital_cost_map",
        parameters={"cell_size_m": CELL, "note": "stands in for a HiRISE-derived map"},
    )
    lineage.record(known_tv, stage="terrain")

    start, goal = (4, 4), (SIZE - 6, SIZE - 6)

    # ------------------------------------------------------ 1. planner family
    banner("1  PLANNER COMPARISON  (same cost map, same clearance)")
    results = {}
    for name, planner in (
        ("A*", AStarPlanner(cost_weight=4.0, inflation_radius_m=0.5, turn_penalty=0.3)),
        ("Theta*", ThetaStarPlanner(cost_weight=4.0, inflation_radius_m=0.5)),
    ):
        plan = planner.plan(known, start, goal, cell_size_m=CELL)
        results[name] = plan
        waypoints = plan.metadata.get("waypoint_count", len(plan.cells))
        print(f"  {name:<8} length {plan.length_m:>7.2f} m   "
              f"waypoints {waypoints:>4}   turns {plan.turns_deg:>7.1f} deg   "
              f"expanded {plan.nodes_expanded:>6}")

    a_star, theta = results["A*"], results["Theta*"]
    print(f"\n  Theta* is {100 * (1 - theta.length_m / a_star.length_m):.1f}% shorter "
          f"and needs {theta.metadata['waypoint_count']} waypoints instead of "
          f"{len(a_star.cells)} grid cells.")
    print(f"  {theta.metadata['line_of_sight_reparents']} line-of-sight reparents "
          "turned staircase treads into straight segments.")

    # ------------------------------------------------------- 2. sensor fusion
    banner("2  SENSOR FUSION SETUP")
    # Process noise must reflect how fast the state can genuinely change. A
    # grid planner hands the controller heading changes of up to 45 degrees per
    # decision, so a yaw-rate process noise tuned for smooth driving makes the
    # filter reject every measurement that reports a turn. Set too low here, the
    # per-sensor rejection rate climbed above 60% and the diagnostics said so.
    ekf = PoseEKF(
        environment="mars",
        initial_pose=(start[1] * CELL, start[0] * CELL, math.radians(45)),
        speed_process_std=0.35,
        yaw_rate_process_std=0.8,
        bias_random_walk_std=5e-4,
    )
    ekf.register(WheelOdometry(speed_noise_std=0.02))
    ekf.register(Gyroscope(rate_noise_std=0.004))
    ekf.register(VisualOdometrySensor(speed_noise_std=0.03, rate_noise_std=0.012))
    ekf.register(StarTracker(heading_noise_std_rad=0.02))
    print(f"  environment      : {env.display_name}")
    print(f"  sensors          : {', '.join(sorted(ekf.sensors))}")
    print(f"  absolute heading : {ekf.has_absolute_heading} "
          "(makes the gyro bias observable)")
    try:
        from perceptos.fusion.ekf import GNSS
        ekf.register(GNSS())
    except ValueError as exc:
        print(f"  GNSS refused     : {str(exc).splitlines()[0][:62]}")

    # ------------------------------------------------- 3. drive and replan
    banner("3  TRAVERSE  (D* Lite repairs the plan as hazards resolve)")
    dstar = DStarLitePlanner(cost_weight=4.0, inflation_radius_m=0.5)
    dstar.initialise(known.copy(), start, goal, cell_size_m=CELL)
    print(f"  initial search   : {dstar.initial_expansions} vertices expanded")

    slip_estimator = WheelSlipEstimator(window=6)
    rng = np.random.default_rng(11)
    true_bias = 0.008                       # rad/s of unmodelled gyro bias
    truth_heading = math.radians(45)
    dt = 1.0
    sensed: set[tuple[int, int]] = set()
    discoveries = 0
    slip_events = 0

    path = dstar.extract_path()
    print(f"  initial plan     : {path.length_m:.2f} m\n")
    print("  step   pos (m)        heading    sigma_pos  slip    event")
    print("  " + "-" * 68)

    prev_heading = truth_heading
    for step in range(200):
        cells = dstar.extract_path().cells
        if len(cells) < 2:
            break
        # Drive several cells per step: the filter runs at 1 Hz, the grid is
        # 25 cm, and a rover does not stop to re-plan every quarter metre.
        nxt = cells[min(4, len(cells) - 1)]

        # --- hazard camera: resolve the true cost of nearby cells -----------
        changes = []
        segment = cells[: min(5, len(cells))]
        for cell in segment:
            for dr in range(-5, 6):
                for dc in range(-5, 6):
                    r, c = cell[0] + dr, cell[1] + dc
                    if not (0 <= r < SIZE and 0 <= c < SIZE) or (r, c) in sensed:
                        continue
                    sensed.add((r, c))
                    if abs(true[r, c] - dstar.cost_map[r, c]) > 1e-9:
                        changes.append(((r, c), float(true[r, c])))
        repaired = dstar.update_costs(changes) if changes else 0
        if changes:
            discoveries += 1

        # --- motion (SIMULATION: there is no rover) ------------------------
        dstar.move_start(nxt)
        truth_heading = wrap_angle(
            math.atan2(nxt[0] - cells[0][0], nxt[1] - cells[0][1])
        )
        # The gyro measures the yaw rate the rover actually executes. Feeding it
        # zero while the heading changes makes the star tracker fight the filter
        # and the gate rejects every heading fix — which is exactly what the
        # per-sensor rejection rate reported when this demo first had that bug.
        yaw_rate = wrap_angle(truth_heading - prev_heading) / dt
        prev_heading = truth_heading
        commanded = math.dist(cells[0], nxt) * CELL

        # High-cost terrain slips. Encoders overstate distance; VO sees truth.
        terrain = float(dstar.cost_map[nxt])
        slip_ratio = min(0.85, 0.08 + 0.9 * terrain)
        actual = commanded * (1.0 - slip_ratio)

        # --- fuse ----------------------------------------------------------
        ekf.predict(dt)
        vo_confidence = float(max(0.2, 1.0 - terrain))
        # Encoders are down-weighted by the slip measured on the previous step:
        # while the wheels are spinning they are a poor ground-speed sensor, and
        # fusing them at full weight against visual odometry makes the filter
        # reject the sensor that is actually right.
        ekf.update("wheel_odometry", [commanded / dt + rng.normal(0, 0.02)],
                   slip_ratio=slip_estimator.history[-1].slip_ratio
                   if slip_estimator.history else 0.0)
        ekf.update("visual_odometry", [actual / dt + rng.normal(0, 0.03),
                                       rng.normal(0, 0.01)],
                   confidence=vo_confidence)
        ekf.update("gyroscope", [yaw_rate + true_bias + rng.normal(0, 0.004)])
        if step % 5 == 0:                    # star tracker fixes are infrequent
            ekf.update("star_tracker", [truth_heading + rng.normal(0, 0.02)])

        slip = slip_estimator.update(commanded, actual, timestamp=float(step),
                                     vo_confidence=vo_confidence)
        if slip.is_slipping:
            slip_events += 1

        estimate = ekf.estimate()
        if step % 12 == 0 or changes:
            sigma = math.hypot(*estimate.position_sigma)
            event = (f"discovered {len(changes)} cells, repaired with {repaired} "
                     f"expansions" if changes else "")
            if slip.is_embedding:
                event = "EMBEDDING - " + event
            print(f"  {step:>4}   ({estimate.position[0]:>5.1f},{estimate.position[1]:>5.1f})"
                  f"   {estimate.heading_deg:>7.1f}    {sigma:>6.2f} m   "
                  f"{slip.slip_ratio:>5.0%}   {event}")

        if nxt == goal or math.dist(nxt, goal) < 2:
            print(f"  {step:>4}   goal reached")
            break

    # --------------------------------------------------------- 4. filter health
    banner("4  FILTER HEALTH")
    estimate = ekf.estimate()
    major, minor, angle = estimate.position_error_ellipse
    print(f"  position         : ({estimate.position[0]:.2f}, {estimate.position[1]:.2f}) m")
    print(f"  heading          : {estimate.heading_deg:.1f} deg "
          f"+/- {estimate.heading_sigma_deg:.2f}")
    print(f"  95% ellipse      : {major:.2f} x {minor:.2f} m at "
          f"{math.degrees(angle):.0f} deg")
    print(f"  gyro bias        : {estimate.gyro_bias_radps:.5f} rad/s "
          f"(true {true_bias}), observable={estimate.bias_observable}")
    print(f"  confidence       : {estimate.confidence:.3f}")
    print(f"  updates/rejected : {estimate.updates} / {estimate.rejections}")
    print("\n  per-sensor consistency:")
    print("  (the synthetic rover turns instantaneously between grid headings,")
    print("   which no smooth motion model can track, so the angular sensors")
    print("   show elevated rejection here. tests/test_fusion_planning.py drives")
    print("   the filter with physically continuous motion and it converges.)")
    for name, report in ekf.diagnostics()["consistency"].items():
        nis = f"{report['mean_nis']:.2f}" if report["mean_nis"] is not None else " -- "
        print(f"    {name:<18} NIS {nis} (expect {report['expected_nis']})  "
              f"rejected {report['rejection_rate']:.0%}  -> {report['verdict']}")

    # ------------------------------------------------------------ 5. replanning
    banner("5  REPLANNING SUMMARY")
    stats = dstar.statistics()
    print(f"  hazard discoveries      : {discoveries}")
    print(f"  replans                 : {stats['replans']}")
    print(f"  initial search          : {stats['initial_expansions']} vertices")
    print(f"  total repair            : "
          f"{stats['total_expansions'] - stats['initial_expansions']} vertices")
    if stats["replans"]:
        mean_repair = ((stats["total_expansions"] - stats["initial_expansions"])
                       / stats["replans"])
        print(f"  mean per repair         : {mean_repair:.1f} vertices "
              f"vs {stats['initial_expansions']} for a fresh search")
    print(f"  slip events             : {slip_events}")

    ekf_tv = ekf.traced_estimate(parents=[known_tv])
    lineage.record(ekf_tv, stage="fusion")
    explanation = lineage.explain(ekf_tv.id)
    print(f"\n  pose lineage depth      : {explanation['depth']}")
    print(f"  simulation in chain     : {explanation['contains_simulation']}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
