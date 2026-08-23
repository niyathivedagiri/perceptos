"""
PerceptOS fusion and planner tests.

The EKF tests drive the filter with simulated ground truth and check it
*recovers* that truth — including the failure modes that a filter can hide:
angle wrapping across the +/-180 boundary, outlier measurements, and a gyro
bias that is not actually observable.

The planner tests check the properties that justify each algorithm's existence:
Theta* must produce fewer waypoints and a shorter path than A* on open terrain,
and D* Lite must repair a plan with far fewer expansions than a fresh search.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from perceptos.core.provenance import Source
from perceptos.fusion.ekf import (
    CHI2_95,
    GNSS,
    Gyroscope,
    PoseEKF,
    StarTracker,
    VisualOdometrySensor,
    WheelOdometry,
    wrap_angle,
)
from perceptos.mars.planner import AStarPlanner, PlanningFailure
from perceptos.mars.planners import (
    DStarLitePlanner,
    ThetaStarPlanner,
    supercover_line,
)


# ===========================================================================
# Angle handling — the classic EKF bug
# ===========================================================================

class TestAngleWrapping:
    def test_wrap_maps_into_the_principal_range(self):
        assert wrap_angle(3 * math.pi) == pytest.approx(-math.pi, abs=1e-9)
        assert wrap_angle(-3 * math.pi) == pytest.approx(-math.pi, abs=1e-9)
        assert wrap_angle(0.5) == pytest.approx(0.5)

    def test_innovation_across_the_boundary_is_small_not_two_pi(self):
        """+179 measured against -179 predicted is a 2 degree error, not 358."""
        ekf = PoseEKF(environment="mars", initial_pose=(0.0, 0.0, math.radians(-179)))
        tracker = StarTracker()
        innovation = tracker.innovation(ekf.x, [math.radians(179)])
        assert abs(innovation[0]) < math.radians(3), (
            "unwrapped innovation would be ~6.2 rad and slew the filter"
        )

    def test_filter_tracks_heading_through_the_discontinuity(self):
        ekf = PoseEKF(environment="mars", initial_pose=(0.0, 0.0, math.radians(170)))
        ekf.register(Gyroscope(rate_noise_std=0.01))
        ekf.register(StarTracker(heading_noise_std_rad=0.01))
        truth = math.radians(170)
        rate = math.radians(4.0)
        for _ in range(20):                     # rotate through +180
            truth = wrap_angle(truth + rate * 0.5)
            ekf.predict(0.5)
            ekf.update("gyroscope", [rate])
            ekf.update("star_tracker", [truth])
        error = abs(wrap_angle(ekf.estimate().heading_rad - truth))
        assert math.degrees(error) < 5.0, "filter lost heading across +/-180"


# ===========================================================================
# Sensor availability
# ===========================================================================

class TestSensorAvailability:
    def test_gnss_is_rejected_on_mars(self):
        """There is no satellite navigation constellation at Mars."""
        with pytest.raises(ValueError, match="not available in 'mars'"):
            PoseEKF(environment="mars").register(GNSS())

    def test_gnss_is_accepted_on_earth(self):
        ekf = PoseEKF(environment="earth").register(GNSS())
        assert "gnss" in ekf.sensors

    def test_star_tracker_is_rejected_in_a_warehouse(self):
        with pytest.raises(ValueError, match="not available"):
            PoseEKF(environment="warehouse").register(StarTracker())

    def test_no_environment_permits_everything(self):
        ekf = PoseEKF().register(GNSS()).register(StarTracker())
        assert len(ekf.sensors) == 2

    def test_unregistered_sensor_lists_what_is_registered(self):
        ekf = PoseEKF().register(WheelOdometry())
        with pytest.raises(KeyError, match="registered"):
            ekf.update("lidar", [1.0])

    def test_wrong_measurement_dimension_is_rejected(self):
        ekf = PoseEKF().register(VisualOdometrySensor())
        with pytest.raises(ValueError, match="expects 2"):
            ekf.update("visual_odometry", [1.0])


# ===========================================================================
# Filter behaviour
# ===========================================================================

class TestPoseEKF:
    def _rover(self, environment="mars"):
        ekf = PoseEKF(environment=environment, initial_pose=(0.0, 0.0, 0.0))
        ekf.register(WheelOdometry(speed_noise_std=0.02))
        ekf.register(Gyroscope(rate_noise_std=0.005))
        return ekf

    def test_straight_drive_recovers_distance_travelled(self):
        ekf = self._rover()
        speed, dt = 0.5, 0.1
        for _ in range(200):                     # 20 s at 0.5 m/s = 10 m
            ekf.predict(dt)
            ekf.update("wheel_odometry", [speed])
            ekf.update("gyroscope", [0.0])
        estimate = ekf.estimate()
        assert estimate.position[0] == pytest.approx(10.0, abs=0.5)
        assert abs(estimate.position[1]) < 0.3
        assert abs(estimate.heading_deg) < 5.0

    def test_constant_turn_traces_a_circle(self):
        ekf = self._rover()
        speed, rate, dt = 1.0, math.radians(18.0), 0.1
        for _ in range(200):                     # 20 s at 18 deg/s = one full turn
            ekf.predict(dt)
            ekf.update("wheel_odometry", [speed])
            ekf.update("gyroscope", [rate])
        estimate = ekf.estimate()
        # A closed circle returns near the origin.
        assert math.hypot(*estimate.position) < 1.5

    def test_covariance_stays_symmetric_positive_definite(self):
        ekf = self._rover()
        rng = np.random.default_rng(0)
        for _ in range(500):
            ekf.predict(0.1)
            ekf.update("wheel_odometry", [0.5 + rng.normal(0, 0.02)])
            ekf.update("gyroscope", [rng.normal(0, 0.01)])
        P = ekf.P
        assert np.allclose(P, P.T, atol=1e-9)
        assert np.all(np.linalg.eigvalsh(P) > -1e-9)

    def test_uncertainty_grows_without_measurements(self):
        ekf = self._rover()
        ekf.update("wheel_odometry", [0.5])
        before = ekf.P[0, 0]
        for _ in range(50):
            ekf.predict(0.1)
        assert ekf.P[0, 0] > before
        assert ekf.estimate().confidence < 1.0

    def test_error_ellipse_is_elongated_after_a_straight_drive(self):
        """Along-track uncertainty grows faster than cross-track. A circle lies."""
        ekf = self._rover()
        for _ in range(300):
            ekf.predict(0.1)
            ekf.update("wheel_odometry", [1.0])
            ekf.update("gyroscope", [0.0])
        major, minor, _ = ekf.estimate().position_error_ellipse
        assert major > minor * 1.5

    def test_gnss_pulls_position_back(self):
        ekf = PoseEKF(environment="earth", initial_pose=(0.0, 0.0, 0.0))
        ekf.register(WheelOdometry()).register(Gyroscope()).register(
            GNSS(position_noise_std=0.5)
        )
        for _ in range(100):
            ekf.predict(0.1)
            ekf.update("wheel_odometry", [1.0])
        drifted = ekf.estimate().position[0]
        for _ in range(20):
            ekf.predict(0.1)
            ekf.update("gnss", [20.0, 0.0])
        assert abs(ekf.estimate().position[0] - 20.0) < abs(drifted - 20.0)


# ===========================================================================
# Outlier gating
# ===========================================================================

class TestMeasurementGating:
    def _settled(self):
        ekf = PoseEKF(environment="earth", initial_pose=(0.0, 0.0, 0.0))
        ekf.register(WheelOdometry(speed_noise_std=0.02))
        ekf.register(GNSS(position_noise_std=0.5))
        for _ in range(100):
            ekf.predict(0.1)
            ekf.update("wheel_odometry", [1.0])
            ekf.update("gnss", [ekf.x[0], 0.0])
        return ekf

    def test_a_wild_outlier_is_rejected(self):
        """One GPS multipath spike must not drag the estimate."""
        ekf = self._settled()
        before = ekf.estimate().position
        result = ekf.update("gnss", [5000.0, 5000.0])
        assert result.accepted is False
        assert result.nis > result.gate
        assert "outlier" in result.reason
        assert ekf.estimate().position == pytest.approx(before, abs=1e-9)

    def test_a_plausible_measurement_is_accepted(self):
        ekf = self._settled()
        result = ekf.update("gnss", [ekf.x[0] + 0.3, 0.2])
        assert result.accepted is True
        assert result.nis <= result.gate

    def test_rejections_are_counted_per_sensor(self):
        ekf = self._settled()
        for _ in range(3):
            ekf.update("gnss", [9999.0, 9999.0])
        stats = ekf.diagnostics()["sensors"]["gnss"]
        assert stats["rejected"] == 3
        assert stats["rejection_rate"] > 0

    def test_gate_defaults_to_the_chi_square_bound_for_the_dimension(self):
        assert GNSS().gate_chi2 == CHI2_95[2]
        assert WheelOdometry().gate_chi2 == CHI2_95[1]

    def test_consistency_verdict_flags_an_over_trusted_sensor(self):
        """Mean NIS far above the measurement dimension means R is too small."""
        ekf = PoseEKF(environment="earth", initial_pose=(0.0, 0.0, 0.0))
        ekf.register(WheelOdometry(speed_noise_std=0.001))   # absurdly confident
        rng = np.random.default_rng(1)
        for _ in range(200):
            ekf.predict(0.1)
            ekf.update("wheel_odometry", [1.0 + rng.normal(0, 0.15)])
        verdict = ekf.diagnostics()["consistency"]["wheel_odometry"]["verdict"]
        assert "optimistic" in verdict


# ===========================================================================
# Observability — the thing a filter can hide
# ===========================================================================

class TestGyroBiasObservability:
    def test_bias_is_unobservable_with_gyro_alone(self):
        """omega and b_gyro enter the measurement identically; no data separates them."""
        ekf = PoseEKF(environment="mars")
        ekf.register(WheelOdometry()).register(Gyroscope())
        for _ in range(50):
            ekf.predict(0.1)
            ekf.update("gyroscope", [0.02])
        assert ekf.has_absolute_heading is False
        assert ekf.bias_observable is False
        assert ekf.estimate().to_dict()["bias_observable"] is False

    def test_an_absolute_heading_reference_makes_bias_observable(self):
        ekf = PoseEKF(environment="mars")
        ekf.register(Gyroscope()).register(StarTracker())
        ekf.predict(0.1)
        ekf.update("star_tracker", [0.0])
        assert ekf.bias_observable is True

    def test_observability_lapses_when_the_reference_goes_stale(self):
        ekf = PoseEKF(environment="mars", bias_observability_timeout_s=5.0)
        ekf.register(Gyroscope()).register(StarTracker())
        ekf.predict(0.1)
        ekf.update("star_tracker", [0.0])
        assert ekf.bias_observable
        for _ in range(70):
            ekf.predict(0.1)
        assert not ekf.bias_observable, "a stale heading fix must not keep bias trusted"

    def test_bias_growth_is_frozen_while_unobservable(self):
        """An unobservable state left as a free random walk produces a confident lie."""
        ekf = PoseEKF(environment="mars")
        ekf.register(Gyroscope())
        before = ekf.P[5, 5]
        for _ in range(100):
            ekf.predict(0.1)
        assert ekf.P[5, 5] <= before + 1e-9

    def test_bias_is_estimated_when_a_star_tracker_is_present(self):
        ekf = PoseEKF(environment="mars", bias_random_walk_std=1e-3)
        ekf.register(Gyroscope(rate_noise_std=0.002))
        ekf.register(StarTracker(heading_noise_std_rad=0.002))
        true_bias, dt = 0.01, 0.1
        heading = 0.0
        for _ in range(600):                     # 60 s stationary in heading
            ekf.predict(dt)
            ekf.update("gyroscope", [0.0 + true_bias])   # gyro reads pure bias
            ekf.update("star_tracker", [heading])
        assert ekf.estimate().gyro_bias_radps == pytest.approx(true_bias, abs=0.006)


# ===========================================================================
# Visual odometry confidence weighting
# ===========================================================================

class TestVisualOdometryWeighting:
    def test_low_confidence_inflates_the_noise(self):
        sensor = VisualOdometrySensor(speed_noise_std=0.03)
        strong = sensor.with_confidence(1.0).R.copy()
        weak = sensor.with_confidence(0.25).R.copy()
        assert weak[0, 0] > strong[0, 0] * 10

    def test_a_weak_vo_solution_moves_the_estimate_less(self):
        """A 0.2-confidence frame must not carry the weight of a 0.95 one."""
        def run(confidence):
            ekf = PoseEKF(environment="mars", initial_pose=(0.0, 0.0, 0.0))
            ekf.register(VisualOdometrySensor(speed_noise_std=0.03))
            ekf.predict(0.1)
            ekf.update("visual_odometry", [2.0, 0.0], confidence=confidence)
            return ekf.estimate().speed_mps

        assert run(0.15) < run(0.95)

    def test_traced_estimate_is_an_estimate_with_covariance(self):
        ekf = PoseEKF(environment="mars").register(WheelOdometry())
        ekf.predict(0.1)
        ekf.update("wheel_odometry", [0.5])
        tv = ekf.traced_estimate()
        assert tv.source is Source.ESTIMATED
        assert 0.0 <= tv.confidence <= 1.0
        assert tv.value.covariance.shape == (6, 6)


# ===========================================================================
# Theta* — any-angle planning
# ===========================================================================

class TestThetaStar:
    def test_supercover_includes_corner_clipped_cells(self):
        """Plain Bresenham would let a path slip through a diagonal gap."""
        cells = supercover_line((0, 0), (2, 2))
        assert (0, 1) in cells or (1, 0) in cells
        assert (2, 2) in cells

    def test_open_terrain_gives_a_two_waypoint_path(self):
        planner = ThetaStarPlanner(inflation_radius_m=0.0, cost_weight=0.0)
        plan = planner.plan(np.zeros((40, 40)), (2, 2), (35, 30), cell_size_m=0.1)
        assert plan.metadata["waypoint_count"] <= 3, (
            "a straight run across open ground needs no intermediate waypoints"
        )

    def test_shorter_and_smoother_than_a_star_on_open_terrain(self):
        cost = np.zeros((60, 60))
        astar = AStarPlanner(inflation_radius_m=0.0, cost_weight=0.0, turn_penalty=0.0)
        theta = ThetaStarPlanner(inflation_radius_m=0.0, cost_weight=0.0)
        a = astar.plan(cost, (5, 5), (50, 20), cell_size_m=0.1)
        t = theta.plan(cost, (5, 5), (50, 20), cell_size_m=0.1)
        assert t.length_m <= a.length_m + 1e-6, "any-angle must not be longer"
        assert t.metadata["waypoint_count"] < len(a.cells)

    def test_respects_the_same_hazard_clearance_as_a_star(self):
        cost = np.zeros((60, 60))
        cost[28:32, 28:32] = 1.0
        plan = ThetaStarPlanner(inflation_radius_m=0.5).plan(
            cost, (5, 5), (55, 55), cell_size_m=0.1
        )
        for r, c in plan.cells:
            assert math.hypot(r - 30, c - 30) > 3.0

    def test_line_of_sight_is_blocked_by_a_wall(self):
        cost = np.zeros((40, 40))
        cost[10:30, 20] = 1.0
        planner = ThetaStarPlanner(inflation_radius_m=0.0)
        blocked = cost >= planner.hazard_threshold
        visible, _ = planner.line_of_sight(cost, blocked, (20, 5), (20, 35))
        assert visible is False

    def test_line_of_sight_charges_the_terrain_it_crosses(self):
        """Blockage-only LOS would cut straight through expensive-but-legal ground."""
        cheap = np.zeros((40, 40))
        pricey = np.full((40, 40), 0.8)
        planner = ThetaStarPlanner(inflation_radius_m=0.0, cost_weight=4.0)
        blocked = np.zeros((40, 40), dtype=bool)
        _, cost_cheap = planner.line_of_sight(cheap, blocked, (5, 5), (5, 25))
        _, cost_pricey = planner.line_of_sight(pricey, blocked, (5, 5), (5, 25))
        assert cost_pricey > cost_cheap * 3

    def test_routes_around_a_wall(self):
        cost = np.zeros((40, 40))
        cost[10:30, 20] = 1.0
        plan = ThetaStarPlanner(inflation_radius_m=0.0).plan(
            cost, (20, 5), (20, 35), cell_size_m=0.1
        )
        assert plan.max_cell_cost < 0.95

    def test_enclosed_goal_still_fails_cleanly(self):
        cost = np.zeros((30, 30))
        cost[10:20, 10] = cost[10:20, 20] = 1.0
        cost[10, 10:21] = cost[19, 10:21] = 1.0
        with pytest.raises(PlanningFailure):
            ThetaStarPlanner(inflation_radius_m=0.0).plan(
                cost, (2, 2), (15, 15), cell_size_m=0.1
            )


# ===========================================================================
# D* Lite — incremental replanning
# ===========================================================================

class TestDStarLite:
    def _planner(self, cost=None):
        cost = np.zeros((50, 50)) if cost is None else cost
        planner = DStarLitePlanner(inflation_radius_m=0.0, cost_weight=4.0)
        planner.initialise(cost, (2, 2), (45, 45), cell_size_m=0.1)
        return planner

    def test_initial_plan_reaches_the_goal(self):
        plan = self._planner().extract_path()
        assert plan.cells[0] == (2, 2)
        assert plan.cells[-1] == (45, 45)

    def test_agrees_with_a_star_on_an_open_map(self):
        cost = np.zeros((50, 50))
        dstar = self._planner(cost.copy()).extract_path()
        astar = AStarPlanner(inflation_radius_m=0.0, cost_weight=4.0,
                             turn_penalty=0.0).plan(cost, (2, 2), (45, 45),
                                                    cell_size_m=0.1)
        assert dstar.length_m == pytest.approx(astar.length_m, rel=0.15)

    def test_a_change_off_the_optimal_path_costs_nothing(self):
        """The common case as a rover observes terrain it will never cross."""
        planner = self._planner()
        assert planner.update_costs([((5, 45), 1.0)]) == 0

    def test_repair_beats_a_fresh_search_on_cluttered_terrain(self):
        """The advantage is real but conditional — this is where it shows up.

        On empty terrain the octile heuristic is perfect, the initial search
        expands almost nothing, and there is nothing for incremental repair to
        save. On cluttered terrain, which is the actual rover case, repair was
        measured 60x cheaper.
        """
        rng = np.random.default_rng(4)
        clutter = (rng.random((120, 120)) < 0.12).astype(float)
        clutter[0:6, 0:6] = 0.0
        clutter[112:, 112:] = 0.0

        def build(cost):
            planner = DStarLitePlanner(inflation_radius_m=0.0, cost_weight=4.0)
            planner.initialise(cost.copy(), (2, 2), (115, 115), cell_size_m=0.1)
            return planner

        planner = build(clutter)
        changes = [((60, 60), 1.0), ((60, 61), 1.0), ((61, 60), 1.0)]
        repair = planner.update_costs(changes)

        changed = clutter.copy()
        for (r, c), v in changes:
            changed[r, c] = v
        fresh = build(changed)

        assert repair < fresh.initial_expansions * 0.25, (
            f"repair {repair} vs fresh search {fresh.initial_expansions}"
        )

    def test_key_comparison_tolerates_floating_point_ties(self):
        """Exact key comparison makes ComputeShortestPath stop on iteration one.

        The keys are g + h + km sums accumulated over a long path, so two
        mathematically equal keys differ by a few ulps. Without a tolerance,
        D* Lite silently returns the stale pre-change path — no crash, no
        warning, just a planner that has quietly stopped replanning.
        """
        planner = self._planner()
        before = planner.extract_path()
        # A wall directly across the diagonal path this planner just produced.
        planner.update_costs([((r, 25), 1.0) for r in range(0, 40)])
        after = planner.extract_path()
        assert after.length_m > before.length_m, (
            "the plan did not change: key comparison terminated the search early"
        )

    def test_a_newly_discovered_hazard_changes_the_path(self):
        cost = np.zeros((50, 50))
        planner = self._planner(cost)
        before = planner.extract_path()
        # Drop a wall across the corridor the first path used.
        changes = [((r, 25), 1.0) for r in range(0, 40)]
        planner.update_costs(changes)
        after = planner.extract_path()
        assert after.length_m > before.length_m
        assert after.max_cell_cost < 0.95

    def test_moving_the_start_does_not_require_reinitialisation(self):
        planner = self._planner()
        plan = planner.extract_path()
        planner.move_start(plan.cells[10])
        planner.update_costs([((30, 30), 1.0)])
        assert planner.extract_path().cells[0] == plan.cells[10]

    def test_km_accumulates_as_the_rover_advances(self):
        """Without km the priority ordering goes stale and paths get worse silently."""
        planner = self._planner()
        assert planner.km == 0.0
        planner.move_start((10, 10))
        planner.update_costs([((30, 30), 1.0)])
        assert planner.km > 0.0

    def test_fully_blocked_goal_reports_failure_with_diagnostics(self):
        planner = self._planner()
        planner.update_costs([((r, c), 1.0)
                              for r in range(40, 50) for c in range(40, 50)])
        with pytest.raises(PlanningFailure) as exc:
            planner.extract_path()
        assert "blocked_fraction" in exc.value.diagnostics

    def test_extraction_does_not_cycle_after_a_repair(self):
        """Greedy g-descent can oscillate between two mutually-cheapest cells.

        After a repair the g-values are only guaranteed consistent along the
        optimal path. Elsewhere two neighbours can each name the other as
        their cheapest successor. Found on an 80x80 ramp, where (55,55) and
        (55,56) pointed at each other and extraction failed on terrain that
        was fully routable.
        """
        from perceptos.mars.traversability import (
            RoverGeometry,
            TraversabilityAnalyser,
        )

        dem = np.tile(np.linspace(0, 2, 80), (80, 1))
        result = TraversabilityAnalyser(RoverGeometry(),
                                        cell_size_m=0.25).analyse(dem)
        planner = DStarLitePlanner()
        planner.initialise(result.cost, (2, 2), (75, 75), cell_size_m=0.25)
        assert planner.extract_path().cells[-1] == (75, 75)

        planner.update_costs([((60, 60), 1.0)])
        plan = planner.extract_path()
        assert plan.cells[-1] == (75, 75), "extraction failed after a repair"
        assert len(plan.cells) == len(set(plan.cells)), (
            "an optimal path never visits the same cell twice"
        )

    def test_statistics_expose_the_replanning_story(self):
        planner = self._planner()
        planner.update_costs([((20, 20), 1.0)])
        planner.update_costs([((25, 25), 1.0)])
        stats = planner.statistics()
        assert stats["replans"] == 2
        assert stats["initial_expansions"] > 0
        assert stats["total_expansions"] >= stats["initial_expansions"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
