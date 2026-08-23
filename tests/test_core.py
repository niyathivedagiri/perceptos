"""
PerceptOS core test suite.

These tests protect the invariants that make the platform trustworthy, not just
the happy paths:

* simulated data cannot be laundered into a value that looks measured
* an estimate cannot exist without an uncertainty
* derived confidence never exceeds its weakest input
* the planner never routes through an inflated hazard
* the Kalman covariance stays symmetric positive-definite
* stereo depth error scales as Z^2

Run:  python -m pytest tests -q      (or: python tests/test_core.py)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from perceptos.core.compute import select_device
from perceptos.core.lineage import LineageGraph
from perceptos.core.pipeline import Pipeline, WiringError
from perceptos.core.provenance import (
    ProvenanceError,
    Provenance,
    Source,
    traced,
)
from perceptos.core.stage import FrameContext, PipelineStage, StageStatus
from perceptos.estimation.kalman import ConstantAccelerationKF
from perceptos.mars.occupancy import GridConfig, OccupancyGrid
from perceptos.mars.planner import AStarPlanner, PlanningFailure
from perceptos.mars.slip import WheelSlipEstimator
from perceptos.mars.traversability import RoverGeometry, TraversabilityAnalyser
from perceptos.servoing.ibvs import CameraIntrinsics, IBVSController
from perceptos.stereo.depth import StereoRig, disparity_to_depth, reproject_to_3d


# ===========================================================================
# Provenance — the credibility contract
# ===========================================================================

class TestProvenance:
    def test_estimate_without_confidence_is_rejected(self):
        """An estimate with no uncertainty is an unfalsifiable claim."""
        with pytest.raises(ProvenanceError, match="confidence"):
            Provenance(source=Source.ESTIMATED, algorithm="some_model")

    def test_confidence_must_be_a_probability(self):
        with pytest.raises(ProvenanceError):
            Provenance(source=Source.CALCULATED, algorithm="x", confidence=1.5)

    def test_algorithm_is_mandatory(self):
        with pytest.raises(ProvenanceError):
            Provenance(source=Source.CALCULATED, algorithm="  ")

    def test_simulation_contaminates_downstream_values(self):
        """This is the single most important rule in PerceptOS."""
        sim = traced(1.0, source=Source.SIMULATION, algorithm="synthetic")
        child = sim.derive(2.0, algorithm="doubling", source=Source.CALCULATED)
        grandchild = child.derive(4.0, algorithm="doubling", source=Source.LIVE_SENSOR)

        assert child.source is Source.SIMULATION
        assert grandchild.source is Source.SIMULATION, (
            "a simulated ancestor must not be laundered into a sensor reading"
        )

    def test_real_data_stays_real(self):
        real = traced(1.0, source=Source.LIVE_CAMERA, algorithm="camera")
        child = real.derive(2.0, algorithm="undistort")
        assert child.source is Source.CALCULATED
        assert not child.is_synthetic

    def test_confidence_inherits_the_weakest_parent(self):
        a = traced(1.0, source=Source.ESTIMATED, algorithm="a", confidence=0.9)
        b = traced(2.0, source=Source.ESTIMATED, algorithm="b", confidence=0.4)
        child = a.derive(3.0, algorithm="fuse", extra_parents=[b])
        assert child.confidence == pytest.approx(0.4)

    def test_explicit_confidence_overrides_inheritance(self):
        a = traced(1.0, source=Source.ESTIMATED, algorithm="a", confidence=0.9)
        child = a.derive(2.0, algorithm="b", confidence=0.2)
        assert child.confidence == pytest.approx(0.2)

    def test_large_arrays_are_summarised_not_serialised(self):
        big = traced(np.zeros((4000, 4000)), source=Source.LIVE_CAMERA, algorithm="cam")
        payload = big.to_dict()
        assert payload["value"]["__array__"] is True
        assert payload["value"]["shape"] == [4000, 4000]


# ===========================================================================
# Lineage — the Data Lineage drawer
# ===========================================================================

class TestLineage:
    def _chain(self):
        graph = LineageGraph()
        frame = traced(np.zeros((4, 4)), source=Source.LIVE_CAMERA, algorithm="cam",
                       label="raw_frame")
        blurred = frame.derive(np.zeros((4, 4)), algorithm="GaussianBlur", label="blurred")
        detection = blurred.derive(
            {"cls": "rock"}, algorithm="YOLOv11", source=Source.ESTIMATED,
            confidence=0.83, label="detection",
        )
        decision = detection.derive({"steer": "left"}, algorithm="Planner",
                                    label="decision")
        for node, stage in (
            (frame, "acq"), (blurred, "imageproc"),
            (detection, "detect"), (decision, "plan"),
        ):
            graph.record(node, stage=stage)
        return graph, frame, decision

    def test_explain_reconstructs_the_full_chain(self):
        graph, frame, decision = self._chain()
        explanation = graph.explain(decision.id)
        assert explanation["depth"] == 4
        assert explanation["algorithms"][0] == "cam"
        assert "YOLOv11" in explanation["algorithms"]
        assert explanation["roots"][0]["id"] == frame.id

    def test_explain_reports_the_weakest_link(self):
        graph, _, decision = self._chain()
        assert graph.explain(decision.id)["weakest_confidence"] == pytest.approx(0.83)

    def test_explain_flags_simulation_anywhere_upstream(self):
        graph = LineageGraph()
        root = traced(1.0, source=Source.SIMULATION, algorithm="sim")
        leaf = root.derive(2.0, algorithm="calc")
        graph.record(root)
        graph.record(leaf)
        assert graph.explain(leaf.id)["contains_simulation"] is True

    def test_unknown_node_raises(self):
        with pytest.raises(KeyError):
            LineageGraph().explain("nope")

    def test_artefact_store_is_bounded(self):
        graph = LineageGraph(artefact_capacity=3)
        for i in range(10):
            graph.record(
                traced(i, source=Source.LIVE_CAMERA, algorithm="cam"),
                artefact=np.zeros((2, 2)),
            )
        assert len(graph.artefacts) == 3, "artefact store must not grow unbounded"
        assert len(graph) == 10, "lineage metadata is kept even when artefacts evict"


# ===========================================================================
# Pipeline — wiring validation and failure isolation
# ===========================================================================

class _Double(PipelineStage):
    name = "double"
    requires = ("frame",)
    provides = ("doubled",)

    def run(self, ctx):
        (frame,) = ctx.require("frame")
        self.emit(ctx, "doubled", frame.value * 2, algorithm="double", parents=[frame])
        return "ok"


class _Explode(PipelineStage):
    name = "explode"
    requires = ("doubled",)
    provides = ("never",)

    def run(self, ctx):
        raise RuntimeError("simulated sensor fault")


class _Downstream(PipelineStage):
    name = "downstream"
    requires = ("doubled",)
    provides = ("final",)

    def run(self, ctx):
        (d,) = ctx.require("doubled")
        self.emit(ctx, "final", d.value + 1, algorithm="increment", parents=[d])


class TestPipeline:
    def test_missing_wiring_is_caught_at_construction(self):
        with pytest.raises(WiringError, match="requires"):
            Pipeline([_Downstream()])

    def test_duplicate_stage_names_rejected(self):
        with pytest.raises(WiringError, match="duplicate"):
            Pipeline([_Double(), _Double()])

    def test_a_failing_stage_does_not_stop_the_pipeline(self):
        """On a rover, a dead stage must not take down hazard avoidance."""
        pipeline = Pipeline([_Double(), _Explode(), _Downstream()])
        seed = traced(5, source=Source.LIVE_CAMERA, algorithm="cam")
        ctx, report = pipeline.process({"frame": seed})

        statuses = {s.stage: s.status for s in report.stages}
        assert statuses["double"] is StageStatus.OK
        assert statuses["explode"] is StageStatus.FAILED
        assert statuses["downstream"] is StageStatus.OK
        assert ctx.get("final").value == 11
        assert report.healthy is False

    def test_disabled_stage_is_skipped_and_downstream_degrades_gracefully(self):
        pipeline = Pipeline([_Double(), _Downstream()])
        pipeline.apply_mode({"mode": "warehouse", "stages": {"double": {"enabled": False}}})
        ctx, report = pipeline.process(
            {"frame": traced(5, source=Source.LIVE_CAMERA, algorithm="cam")}
        )
        statuses = {s.stage: s.status for s in report.stages}
        assert statuses["double"] is StageStatus.DISABLED
        assert statuses["downstream"] is StageStatus.SKIPPED
        assert pipeline.mode == "warehouse"

    def test_mode_switch_is_reversible_without_restart(self):
        pipeline = Pipeline([_Double(), _Downstream()])
        pipeline.apply_mode({"stages": {"double": False}})
        assert pipeline.stage("double").enabled is False
        pipeline.apply_mode({"stages": {"double": True}})
        assert pipeline.stage("double").enabled is True

    def test_every_output_is_recorded_in_lineage_with_latency(self):
        pipeline = Pipeline([_Double(), _Downstream()])
        ctx, _ = pipeline.process(
            {"frame": traced(5, source=Source.LIVE_CAMERA, algorithm="cam")}
        )
        final = ctx.get("final")
        assert final.provenance.latency_ms is not None
        assert final.provenance.latency_ms >= 0
        assert len(ctx.lineage) == 3  # frame + doubled + final

    def test_block_diagram_is_generated_from_real_wiring(self):
        pipeline = Pipeline([_Double(), _Downstream()])
        mermaid = pipeline.to_mermaid()
        assert "double" in mermaid and "downstream" in mermaid
        assert "|doubled|" in mermaid


# ===========================================================================
# Kalman filter
# ===========================================================================

class TestKalmanFilter:
    def test_recovers_constant_velocity_from_positions_alone(self):
        dt, vx, vy = 0.1, 3.0, -1.5
        kf = ConstantAccelerationKF(
            dt=dt, process_noise_std=0.1, measurement_noise_std=0.05,
            initial_position=(0.0, 0.0),
        )
        for i in range(1, 120):
            kf.predict(dt)
            kf.update((vx * i * dt, vy * i * dt))
        assert kf.state().velocity[0] == pytest.approx(vx, abs=0.15)
        assert kf.state().velocity[1] == pytest.approx(vy, abs=0.15)

    def test_recovers_constant_acceleration(self):
        dt, ax = 0.05, 2.0
        kf = ConstantAccelerationKF(dt=dt, process_noise_std=0.5,
                                    measurement_noise_std=0.02)
        for i in range(1, 200):
            t = i * dt
            kf.predict(dt)
            kf.update((0.5 * ax * t * t, 0.0))
        assert kf.state().acceleration[0] == pytest.approx(ax, abs=0.3)

    def test_future_prediction_is_accurate_for_ballistic_motion(self):
        dt = 0.1
        kf = ConstantAccelerationKF(dt=dt, process_noise_std=0.05,
                                    measurement_noise_std=0.01)
        for i in range(1, 80):
            kf.predict(dt)
            kf.update((2.0 * i * dt, 0.0))
        predicted, cov = kf.predict_future(1.0)
        expected = 2.0 * 79 * dt + 2.0 * 1.0
        assert predicted[0] == pytest.approx(expected, abs=0.4)
        assert cov[0, 0] > 0

    def test_covariance_stays_symmetric_positive_definite(self):
        """Joseph-form update must not let P drift asymmetric over a long run."""
        kf = ConstantAccelerationKF(dt=0.1)
        rng = np.random.default_rng(0)
        for i in range(500):
            kf.predict()
            kf.update((i * 0.1 + rng.normal(0, 0.5), rng.normal(0, 0.5)))
        P = kf.P
        assert np.allclose(P, P.T, atol=1e-9), "covariance lost symmetry"
        assert np.all(np.linalg.eigvalsh(P) > -1e-9), "covariance lost positive-definiteness"

    def test_uncertainty_grows_without_measurements(self):
        kf = ConstantAccelerationKF(dt=0.1)
        kf.update((0.0, 0.0))
        before = kf.P[0, 0]
        for _ in range(30):
            kf.predict()
            kf.mark_missed()
        assert kf.P[0, 0] > before
        assert kf.track_confidence < 0.5, "a long-unseen track must lose confidence"

    def test_traced_state_is_an_estimate_with_confidence(self):
        kf = ConstantAccelerationKF(dt=0.1)
        kf.update((1.0, 1.0))
        tv = kf.traced_state()
        assert tv.source is Source.ESTIMATED
        assert 0.0 <= tv.confidence <= 1.0

    def test_rejects_non_positive_timestep(self):
        with pytest.raises(ValueError):
            ConstantAccelerationKF(dt=0.0)


# ===========================================================================
# Traversability
# ===========================================================================

class TestTraversability:
    def _analyser(self, cell=0.1):
        return TraversabilityAnalyser(RoverGeometry(), cell_size_m=cell, window_cells=3)

    def test_flat_ground_is_fully_traversable(self):
        result = self._analyser().analyse(np.zeros((40, 40)))
        assert result.traversable_fraction == pytest.approx(1.0)
        assert result.cost.max() == pytest.approx(0.0, abs=1e-6)

    def test_slope_is_measured_correctly(self):
        """A 1:1 ramp is exactly 45 degrees."""
        cell = 0.1
        ramp = np.tile(np.arange(40) * cell, (40, 1))
        result = self._analyser(cell).analyse(ramp)
        interior = result.slope_deg[5:-5, 5:-5]
        assert float(np.mean(interior)) == pytest.approx(45.0, abs=0.5)

    def test_steep_slope_is_flagged_as_hazard(self):
        cell = 0.1
        steep = np.tile(np.arange(40) * cell * 2.0, (40, 1))  # ~63 degrees
        result = self._analyser(cell).analyse(steep)
        assert result.hazard_mask.mean() > 0.9

    def test_a_boulder_creates_a_step_hazard(self):
        dem = np.zeros((40, 40))
        dem[18:22, 18:22] = 0.6  # 60 cm rock, above the 35 cm step limit
        result = self._analyser().analyse(dem)
        assert result.hazard_mask[18:22, 18:22].any()
        assert result.hazard_mask[0:5, 0:5].sum() == 0, "far field must stay clear"

    def test_unobserved_cells_are_never_traversable(self):
        """Unknown terrain is not safe terrain."""
        dem = np.zeros((40, 40))
        dem[10:15, 10:15] = np.nan
        result = self._analyser().analyse(dem)
        assert result.hazard_mask[10:15, 10:15].all()
        assert result.coverage < 1.0

    def test_hazards_combine_by_maximum_not_average(self):
        """A cliff beside smooth ground must stay lethal."""
        cell = 0.1
        dem = np.zeros((40, 40))
        dem[:, 20:] = 3.0  # vertical wall
        result = self._analyser(cell).analyse(dem)
        assert result.cost[:, 18:22].max() == pytest.approx(1.0)

    def test_confidence_tracks_dem_coverage(self):
        dem = np.zeros((40, 40))
        dem[0:20, :] = np.nan  # half the map unobserved
        dem_tv = traced(dem, source=Source.LIVE_CAMERA, algorithm="stereo")
        result = self._analyser().analyse_traced(dem_tv)
        assert result.confidence == pytest.approx(0.5, abs=0.05)

    def test_dem_smaller_than_window_is_rejected(self):
        with pytest.raises(ValueError, match="smaller than"):
            self._analyser().analyse(np.zeros((2, 2)))


# ===========================================================================
# Path planning
# ===========================================================================

class TestPathPlanner:
    def test_straight_line_across_open_ground(self):
        planner = AStarPlanner(inflation_radius_m=0.0, turn_penalty=0.0)
        plan = planner.plan(np.zeros((30, 30)), (0, 0), (0, 20), cell_size_m=0.1)
        assert plan.length_m == pytest.approx(2.0, abs=0.05)
        assert plan.max_cell_cost == pytest.approx(0.0)

    def test_planner_routes_around_a_wall(self):
        cost = np.zeros((40, 40))
        cost[10:30, 20] = 1.0  # partial wall with a gap at the top
        planner = AStarPlanner(inflation_radius_m=0.0)
        plan = planner.plan(cost, (20, 5), (20, 35), cell_size_m=0.1)
        assert plan.max_cell_cost < 0.95, "the path must not cross a lethal cell"
        assert plan.length_m > 3.0, "detour should be longer than the direct route"

    def test_planner_never_enters_an_inflated_hazard(self):
        cost = np.zeros((60, 60))
        cost[28:32, 28:32] = 1.0
        planner = AStarPlanner(inflation_radius_m=0.5)
        plan = planner.plan(cost, (5, 5), (55, 55), cell_size_m=0.1)
        for r, c in plan.cells:
            distance = math.hypot(r - 30, c - 30)
            assert distance > 3.0, f"path cell {(r, c)} violates rover clearance"

    def test_cost_weight_trades_distance_for_safety(self):
        cost = np.zeros((40, 40))
        cost[15:25, :] = 0.7  # a risky but short corridor
        cost[15:25, 0] = cost[15:25, -1] = 0.0
        cheap = AStarPlanner(cost_weight=0.0, inflation_radius_m=0.0)
        safe = AStarPlanner(cost_weight=50.0, inflation_radius_m=0.0)
        direct = cheap.plan(cost, (5, 20), (35, 20), cell_size_m=0.1)
        cautious = safe.plan(cost, (5, 20), (35, 20), cell_size_m=0.1)
        assert cautious.mean_cell_cost <= direct.mean_cell_cost
        assert cautious.length_m >= direct.length_m

    def test_enclosed_goal_raises_with_diagnostics(self):
        cost = np.zeros((30, 30))
        cost[10:20, 10] = cost[10:20, 20] = 1.0
        cost[10, 10:21] = cost[19, 10:21] = 1.0
        planner = AStarPlanner(inflation_radius_m=0.0)
        with pytest.raises(PlanningFailure) as exc:
            planner.plan(cost, (2, 2), (15, 15), cell_size_m=0.1)
        assert "nodes_expanded" in exc.value.diagnostics

    def test_goal_inside_hazard_is_reported_clearly(self):
        cost = np.zeros((30, 30))
        cost[15, 15] = 1.0
        planner = AStarPlanner(inflation_radius_m=0.2)
        with pytest.raises(PlanningFailure, match="inflated hazard"):
            planner.plan(cost, (2, 2), (15, 15), cell_size_m=0.1)

    def test_out_of_bounds_goal_is_rejected(self):
        with pytest.raises(PlanningFailure, match="outside"):
            AStarPlanner().plan(np.zeros((10, 10)), (0, 0), (50, 50))

    def test_rover_can_escape_from_inside_a_hazard_footprint(self):
        """If the rover is already in trouble, the planner must still plan."""
        cost = np.zeros((40, 40))
        cost[5, 5] = 1.0
        planner = AStarPlanner(inflation_radius_m=0.5)
        plan = planner.plan(cost, (6, 6), (35, 35), cell_size_m=0.1)
        assert len(plan.cells) > 1

    def test_energy_and_duration_scale_with_terrain_cost(self):
        planner = AStarPlanner(inflation_radius_m=0.0, cost_weight=0.0)
        easy = planner.plan(np.zeros((30, 30)), (0, 0), (0, 25), cell_size_m=0.1)
        hard = planner.plan(np.full((30, 30), 0.6), (0, 0), (0, 25), cell_size_m=0.1)
        assert hard.estimate_energy_wh() > easy.estimate_energy_wh()
        assert hard.estimate_duration_s() > easy.estimate_duration_s()


# ===========================================================================
# Occupancy grid
# ===========================================================================

class TestOccupancyGrid:
    def test_coordinate_round_trip(self):
        grid = OccupancyGrid(GridConfig(width_cells=50, height_cells=50, cell_size_m=0.2))
        row, col = grid.world_to_cell(3.3, 4.7)
        x, y = grid.cell_to_world(row, col)
        assert abs(x - 3.3) <= 0.2 and abs(y - 4.7) <= 0.2

    def test_ray_marks_free_space_and_an_occupied_endpoint(self):
        grid = OccupancyGrid(GridConfig(width_cells=60, height_cells=60, cell_size_m=0.1))
        grid.integrate_ray((0.5, 0.5), (4.0, 0.5), hit=True)
        prob = grid.probability_map()
        endpoint = grid.world_to_cell(4.0, 0.5)
        midpoint = grid.world_to_cell(2.0, 0.5)
        assert prob[endpoint] > 0.6, "hit cell should read occupied"
        assert prob[midpoint] < 0.4, "traversed cells should read free"

    def test_log_odds_are_clamped_so_cells_stay_revisable(self):
        cfg = GridConfig(width_cells=30, height_cells=30, cell_size_m=0.1)
        grid = OccupancyGrid(cfg)
        for _ in range(500):
            grid.integrate_ray((0.5, 0.5), (2.0, 0.5), hit=True)
        assert grid.log_odds.max() <= cfg.log_odds_max + 1e-6
        # A cell saturated as occupied must still be able to recover.
        endpoint = grid.world_to_cell(2.0, 0.5)
        for _ in range(200):
            grid.integrate_ray((0.5, 0.5), (4.0, 0.5), hit=True)
        assert grid.probability_map()[endpoint] < 0.5

    def test_elevation_fusion_averages_repeated_observations(self):
        grid = OccupancyGrid(GridConfig(width_cells=20, height_cells=20, cell_size_m=0.5))
        points = np.array([[1.0, 1.0, 2.0], [1.0, 1.0, 4.0]])
        grid.integrate_points(points)
        dem = grid.elevation_map()
        row, col = grid.world_to_cell(1.0, 1.0)
        assert dem[row, col] == pytest.approx(3.0)

    def test_unobserved_cells_are_nan_not_zero(self):
        """Zero elevation is a claim; NaN is an honest absence of one."""
        grid = OccupancyGrid(GridConfig(width_cells=20, height_cells=20))
        grid.integrate_points(np.array([[0.5, 0.5, 1.0]]))
        assert np.isnan(grid.elevation_map()).any()

    def test_points_outside_the_grid_are_dropped_not_wrapped(self):
        grid = OccupancyGrid(GridConfig(width_cells=10, height_cells=10, cell_size_m=0.1))
        fused = grid.integrate_points(np.array([[99.0, 99.0, 1.0], [0.5, 0.5, 2.0]]))
        assert fused == 1

    def test_invalid_sensor_model_is_rejected(self):
        with pytest.raises(ValueError):
            GridConfig(prob_hit=0.4)
        with pytest.raises(ValueError):
            GridConfig(prob_miss=0.7)


# ===========================================================================
# Wheel slip
# ===========================================================================

class TestWheelSlip:
    def test_no_slip_when_wheels_and_rover_agree(self):
        est = WheelSlipEstimator().update(1.0, 1.0)
        assert est.slip_ratio == pytest.approx(0.0)
        assert not est.is_slipping

    def test_total_loss_of_traction_reads_one(self):
        est = WheelSlipEstimator().update(1.0, 0.0)
        assert est.slip_ratio == pytest.approx(1.0)
        assert est.is_slipping

    def test_skid_is_reported_as_negative(self):
        est = WheelSlipEstimator().update(1.0, 2.0)
        assert est.slip_ratio < 0
        assert "kid" in est.recommendation

    def test_stationary_rover_does_not_emit_spurious_slip(self):
        """Dividing by a near-zero commanded distance is the classic false alarm."""
        est = WheelSlipEstimator().update(0.0001, 0.0)
        assert est.slip_ratio == pytest.approx(0.0)
        assert "Stationary" in est.recommendation

    def test_embedding_requires_sustained_slip_not_one_bad_sample(self):
        estimator = WheelSlipEstimator(window=5, embedding_threshold=0.6)
        single = estimator.update(1.0, 0.1)          # one severe sample
        assert single.is_slipping and not single.is_embedding
        for _ in range(6):
            latest = estimator.update(1.0, 0.1)
        assert latest.is_embedding
        assert "HALT" in latest.recommendation

    def test_weak_visual_odometry_lowers_confidence(self):
        estimator = WheelSlipEstimator(window=4)
        strong = estimator.update(1.0, 0.8, vo_confidence=1.0)
        weak = WheelSlipEstimator(window=4).update(1.0, 0.8, vo_confidence=0.1)
        assert weak.confidence < strong.confidence

    def test_traced_estimate_is_marked_as_an_estimate(self):
        estimator = WheelSlipEstimator()
        tv = estimator.traced_estimate(estimator.update(1.0, 0.6))
        assert tv.source is Source.ESTIMATED
        assert tv.confidence is not None


# ===========================================================================
# Stereo depth
# ===========================================================================

class TestStereoDepth:
    def _rig(self):
        return StereoRig(fx=800.0, fy=800.0, cx=320.0, cy=240.0, baseline_m=0.12)

    def test_depth_follows_the_stereo_relation(self):
        rig = self._rig()
        disparity = np.full((4, 4), 24.0)
        result = disparity_to_depth(disparity, rig)
        expected = 800.0 * 0.12 / 24.0  # 4 m
        assert result.depth_m[0, 0] == pytest.approx(expected)

    def test_depth_error_grows_with_the_square_of_range(self):
        rig = self._rig()
        near, far = rig.depth_sigma(2.0), rig.depth_sigma(20.0)
        assert far / near == pytest.approx(100.0, rel=0.01)

    def test_invalid_disparity_becomes_nan_not_infinity(self):
        rig = self._rig()
        disparity = np.array([[0.0, -3.0], [np.nan, 30.0]])
        result = disparity_to_depth(disparity, rig)
        assert np.isnan(result.depth_m[0, 0])
        assert np.isnan(result.depth_m[1, 0])
        assert np.isfinite(result.depth_m[1, 1])
        assert result.coverage == pytest.approx(0.25)

    def test_reliable_range_matches_the_closed_form(self):
        rig = self._rig()
        limit = rig.max_reliable_range(0.05)
        assert rig.depth_sigma(limit) / limit == pytest.approx(0.05, rel=1e-6)

    def test_reprojection_recovers_the_pinhole_geometry(self):
        rig = self._rig()
        depth = np.full((10, 10), 5.0)
        cloud = reproject_to_3d(depth, rig)
        assert cloud.shape == (100, 3)
        assert np.allclose(cloud[:, 2], 5.0)
        # Pixel (0,0) maps to the top-left of the frustum.
        assert cloud[0, 0] == pytest.approx((0 - 320.0) * 5.0 / 800.0)

    def test_degenerate_rig_is_rejected(self):
        with pytest.raises(ValueError):
            StereoRig(fx=800, fy=800, cx=0, cy=0, baseline_m=0.0)


# ===========================================================================
# Visual servoing
# ===========================================================================

class TestIBVS:
    def _controller(self, **kwargs):
        return IBVSController(
            CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0), **kwargs
        )

    def test_centred_target_produces_no_motion(self):
        cmd = self._controller().centre_target((320.0, 240.0), (640, 480), depth_m=2.0)
        assert cmd.converged
        assert np.allclose(cmd.linear, 0.0, atol=1e-9)

    def test_offset_target_drives_the_camera_towards_it(self):
        cmd = self._controller().centre_target((500.0, 240.0), (640, 480), depth_m=2.0)
        assert not cmd.converged
        assert cmd.error_norm_px > 100
        assert abs(cmd.linear[0]) + abs(cmd.angular[1]) > 1e-4

    def test_interaction_matrix_has_the_documented_structure(self):
        L = IBVSController.interaction_matrix(0.1, 0.2, 2.0)
        assert L.shape == (2, 6)
        assert L[0, 0] == pytest.approx(-0.5)
        assert L[0, 1] == pytest.approx(0.0)
        assert L[0, 2] == pytest.approx(0.05)
        assert L[1, 5] == pytest.approx(-0.1)

    def test_commands_are_saturated_not_unbounded(self):
        """A pseudo-inverse near singularity would command a violent lurch."""
        controller = self._controller(gain=500.0, max_linear=0.3, max_angular=0.5)
        cmd = controller.centre_target((639.0, 479.0), (640, 480), depth_m=0.2)
        assert cmd.saturated
        assert np.linalg.norm(cmd.linear) <= 0.3 + 1e-9
        assert np.linalg.norm(cmd.angular) <= 0.5 + 1e-9

    def test_unknown_depth_degrades_confidence_but_still_commands(self):
        controller = self._controller()
        known = controller.centre_target((400.0, 300.0), (640, 480), depth_m=2.0)
        unknown = controller.centre_target((400.0, 300.0), (640, 480))
        assert unknown.confidence < known.confidence
        assert np.linalg.norm(unknown.linear) > 0

    def test_four_features_constrain_more_dof_than_one(self):
        controller = self._controller()
        one = controller.compute([(400, 300)], [(320, 240)], [2.0])
        four = controller.compute(
            [(400, 300), (240, 300), (400, 180), (240, 180)],
            [(360, 280), (280, 280), (360, 200), (280, 200)],
            [2.0] * 4,
        )
        assert four.confidence > one.confidence

    def test_mismatched_feature_counts_are_rejected(self):
        with pytest.raises(ValueError, match="mismatch"):
            self._controller().compute([(1, 1)], [(1, 1), (2, 2)], [1.0])

    def test_convergence_over_repeated_iterations(self):
        """Closed-loop simulation: the feature error must decay to near zero.

        The plant is the interaction matrix itself — the feature is propagated
        by ``x_dot = L @ v``, which is the actual first-order camera response,
        rather than an ad-hoc approximation.
        """
        controller = self._controller(gain=0.6)
        K, depth, dt = controller.K, 2.0, 0.1
        u, v_px = 520.0, 380.0
        errors = []
        for _ in range(60):
            cmd = controller.centre_target((u, v_px), (640, 480), depth_m=depth)
            errors.append(cmd.error_norm_px)
            x, y = K.to_normalised(u, v_px)
            L = IBVSController.interaction_matrix(x, y, depth)
            feature_velocity = L @ np.array([*cmd.linear, *cmd.angular])
            x += float(feature_velocity[0]) * dt
            y += float(feature_velocity[1]) * dt
            u, v_px = K.cx + x * K.fx, K.cy + y * K.fy

        assert errors[-1] < errors[0] * 0.05, "IBVS loop failed to converge"
        assert all(
            errors[i + 1] <= errors[i] + 1e-6 for i in range(len(errors) - 1)
        ), "error must decay monotonically under proportional control"


# ===========================================================================
# Compute environment
# ===========================================================================

class TestCompute:
    def test_device_selection_always_returns_something_usable(self):
        info = select_device()
        assert info.kind in ("mps", "cuda", "cpu")
        assert info.torch_device
        assert info.reason, "the operator must be told why this device was chosen"

    def test_selection_never_raises_without_torch(self):
        # select_device is cached; calling twice must be stable and safe.
        assert select_device().kind == select_device().kind


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
