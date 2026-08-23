"""
PerceptOS SLAM tests.

Trajectories are generated with known ground truth, so drift and correction are
measured rather than eyeballed.

The most important test in this file is
:meth:`TestFalseLoopClosure.test_a_false_closure_destroys_an_unprotected_graph`.
It exists to demonstrate the failure the robust kernel guards against, by
building a graph without protection and measuring how far the trajectory is
dragged. If someone later removes the Huber kernel, that test records exactly
what was lost.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from perceptos.core.cv_compat import HAS_OPENCV
from perceptos.core.provenance import Source
from perceptos.slam.frontend import (
    Keyframe,
    KeyframeSelector,
    LoopDetector,
)
from perceptos.slam.graph import Pose2D, PoseGraph, wrap_angle
from perceptos.slam.system import SlamSystem


def square_loop(side_m: float = 10.0, per_side: int = 5) -> list[Pose2D]:
    """A closed square traverse. Ground truth: the end equals the start."""
    poses = [Pose2D(0.0, 0.0, 0.0)]
    step = side_m / per_side
    heading = 0.0
    x = y = 0.0
    for _ in range(4):
        for _ in range(per_side):
            x += step * math.cos(heading)
            y += step * math.sin(heading)
            poses.append(Pose2D(x, y, heading))
        heading = float(wrap_angle(heading + math.pi / 2))
        poses.append(Pose2D(x, y, heading))
    return poses


def build_graph(poses, *, drift_per_step: float = 0.0, huber=1.0,
                seed: int = 0) -> tuple[PoseGraph, list[Pose2D]]:
    """Build a chain graph whose odometry is exact but whose initial guess drifts."""
    rng = np.random.default_rng(seed)
    graph = PoseGraph(huber_delta=huber)
    estimate = poses[0]
    graph.add_node(estimate)
    for i in range(1, len(poses)):
        relative = poses[i - 1].between(poses[i])
        # The initial guess accumulates error; the measurement does not.
        noisy = Pose2D(relative.x + rng.normal(0, drift_per_step),
                       relative.y + rng.normal(0, drift_per_step),
                       relative.theta + rng.normal(0, drift_per_step * 0.1))
        estimate = estimate.compose(noisy)
        graph.add_node(estimate)
        graph.add_edge(i - 1, i, relative)
    return graph, poses


# ===========================================================================
# Pose algebra
# ===========================================================================

class TestPose2D:
    def test_compose_then_between_round_trips(self):
        a = Pose2D(1.0, 2.0, 0.5)
        delta = Pose2D(3.0, -1.0, 0.2)
        b = a.compose(delta)
        recovered = a.between(b)
        assert recovered.x == pytest.approx(delta.x)
        assert recovered.y == pytest.approx(delta.y)
        assert recovered.theta == pytest.approx(delta.theta)

    def test_inverse_undoes_compose(self):
        a = Pose2D(2.0, -3.0, 1.1)
        identity = a.compose(a.inverse().compose(a)).between(a)
        assert abs(identity.x) < 1e-9 and abs(identity.y) < 1e-9

    def test_between_is_expressed_in_the_source_frame(self):
        """Rotating the source frame changes the relative coordinates."""
        origin = Pose2D(0.0, 0.0, math.pi / 2)      # facing +y
        target = Pose2D(0.0, 5.0, math.pi / 2)      # 5 m ahead
        relative = origin.between(target)
        assert relative.x == pytest.approx(5.0)     # forward, not +y
        assert abs(relative.y) < 1e-9

    def test_heading_is_always_wrapped(self):
        pose = Pose2D.from_array([0.0, 0.0, 3 * math.pi])
        assert -math.pi <= pose.theta < math.pi


# ===========================================================================
# Optimisation
# ===========================================================================

class TestPoseGraphOptimisation:
    def test_a_perfect_chain_needs_no_correction(self):
        graph, poses = build_graph(square_loop(), drift_per_step=0.0)
        assert graph.total_error() == pytest.approx(0.0, abs=1e-9)
        result = graph.optimise()
        assert result.converged
        assert result.final_error == pytest.approx(0.0, abs=1e-8)

    def test_optimisation_reduces_error_on_a_drifted_guess(self):
        graph, _ = build_graph(square_loop(), drift_per_step=0.05, seed=1)
        result = graph.optimise()
        assert result.final_error < result.initial_error
        assert result.error_reduction > 0.9

    def test_a_loop_closure_pulls_the_trajectory_shut(self):
        """The whole point of SLAM: recognising a place removes drift."""
        poses = square_loop()
        graph, _ = build_graph(poses, drift_per_step=0.08, seed=2)
        last = max(graph.nodes)
        before = graph.nodes[0].pose.distance_to(graph.nodes[last].pose)

        # The rover has genuinely returned to its start.
        graph.add_edge(0, last, Pose2D(0.0, 0.0, 0.0),
                       information=np.diag([4.0, 4.0, 8.0]),
                       kind="loop_closure", confidence=0.8)
        graph.optimise()
        after = graph.nodes[0].pose.distance_to(graph.nodes[last].pose)
        assert after < before * 0.5, f"gap {before:.2f} m -> {after:.2f} m"

    def test_the_fixed_node_does_not_move(self):
        graph, _ = build_graph(square_loop(), drift_per_step=0.05, seed=3)
        before = graph.nodes[0].pose
        graph.optimise()
        assert graph.nodes[0].pose.as_array() == pytest.approx(before.as_array())

    def test_a_graph_with_no_fixed_node_is_refused(self):
        """Three unconstrained degrees of freedom make H singular."""
        graph = PoseGraph()
        graph.add_node(Pose2D(0, 0, 0), fixed=False)
        graph.add_node(Pose2D(1, 0, 0), fixed=False)
        graph.add_edge(0, 1, Pose2D(1, 0, 0))
        with pytest.raises(ValueError, match="no node is fixed"):
            graph.optimise()

    def test_an_empty_graph_optimises_trivially(self):
        result = PoseGraph().optimise()
        assert result.converged
        assert "no edges" in result.message

    def test_self_edges_are_rejected(self):
        graph = PoseGraph()
        graph.add_node(Pose2D(0, 0, 0))
        with pytest.raises(ValueError, match="cannot connect a node to itself"):
            graph.add_edge(0, 0, Pose2D(0, 0, 0))

    def test_asymmetric_information_is_rejected(self):
        graph = PoseGraph()
        graph.add_node(Pose2D(0, 0, 0))
        graph.add_node(Pose2D(1, 0, 0))
        with pytest.raises(ValueError, match="symmetric"):
            graph.add_edge(0, 1, Pose2D(1, 0, 0),
                           information=np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]]))

    def test_optimisation_works_across_the_angle_wrap(self):
        """A rover turning through south must not slew the whole graph."""
        poses = [Pose2D(0, 0, math.radians(170)),
                 Pose2D(1, 0, math.radians(179)),
                 Pose2D(2, 0, math.radians(-179)),
                 Pose2D(3, 0, math.radians(-170))]
        graph, _ = build_graph(poses, drift_per_step=0.02, seed=4)
        result = graph.optimise()
        assert result.converged
        assert result.final_error < 1.0


# ===========================================================================
# False loop closures — the failure that matters
# ===========================================================================

class TestFalseLoopClosure:
    def _drifted(self, huber):
        poses = square_loop(side_m=10.0, per_side=5)
        return build_graph(poses, drift_per_step=0.03, huber=huber, seed=5)

    def test_a_false_closure_destroys_an_unprotected_graph(self):
        """Least squares has no notion that a measurement can be nonsense.

        A wrong edge asserts two unrelated places are the same, and the
        optimiser folds the trajectory to satisfy it. This test measures how
        far, so removing the robust kernel has a visible cost.
        """
        graph, poses = self._drifted(huber=None)
        graph.optimise()
        honest = [p for p in graph.trajectory()]

        graph, _ = self._drifted(huber=None)
        mid = len(poses) // 2
        # A closure claiming the midpoint is the origin. It is not: the rover
        # is halfway round a 10 m square.
        graph.add_edge(0, mid, Pose2D(0.0, 0.0, 0.0),
                       information=np.diag([5.0, 5.0, 5.0]),
                       kind="loop_closure", confidence=1.0)
        graph.optimise()
        corrupted = graph.trajectory()

        displacement = max(a.distance_to(b) for a, b in zip(honest, corrupted))
        assert displacement > 1.0, (
            f"a false closure moved the trajectory by only {displacement:.2f} m; "
            "this test is meant to demonstrate real damage"
        )

    def _corrupt(self, huber):
        """Add a false closure and return the resulting optimisation."""
        graph, poses = self._drifted(huber=huber)
        graph.add_edge(0, len(poses) // 2, Pose2D(0.0, 0.0, 0.0),
                       information=np.diag([5.0, 5.0, 5.0]),
                       kind="loop_closure", confidence=1.0)
        return graph, graph.optimise()

    def test_the_damage_lands_on_the_odometry_not_the_bad_edge(self):
        """The obvious check — the closure's own residual — misses this entirely.

        The optimiser satisfies the false constraint by deforming everything
        around it, so the bad edge ends up fitting better than the good ones.
        Measured here: the closure residual is small while the odometry it
        corrupted is several times larger.
        """
        graph, result = self._corrupt(huber=0.5)
        assert result.converged, "the graph looks perfectly healthy"
        assert result.max_loop_residual < result.max_odometry_residual, (
            "if the bad edge had the larger residual, checking it would suffice"
        )

    def test_the_robust_kernel_limits_the_damage(self):
        """Measured on odometry corruption, which is where the harm appears."""
        _, unprotected = self._corrupt(huber=None)
        _, protected = self._corrupt(huber=0.5)
        assert protected.max_odometry_residual < unprotected.max_odometry_residual, (
            f"Huber: {protected.max_odometry_residual:.3f} vs "
            f"plain: {unprotected.max_odometry_residual:.3f}"
        )

    def test_clean_odometry_stays_clean_without_a_false_closure(self):
        """Establishes the baseline the alarm is measured against."""
        graph, _ = self._drifted(huber=0.5)
        result = graph.optimise()
        assert result.max_odometry_residual < 0.01

    def test_a_false_closure_is_flagged_even_though_it_fits(self):
        graph, result = self._corrupt(huber=0.5)
        assert result.suspect_edges, "a corrupting closure must surface"
        reasons = " ".join(r for e in result.suspect_edges
                           for r in e["suspect_reasons"])
        assert "deforming well-measured motion" in reasons
        assert len(graph.loop_closures) == 1, "the edge must not be silently dropped"

    def test_confidence_reflects_odometry_corruption(self):
        """A true closure and a false one, compared like with like.

        Both graphs have exactly one loop closure, so neither is penalised for
        dead reckoning. The difference is only whether the closure is true.
        Scoring the closure's own residual would rank the false one higher,
        because the optimiser fitted it better than the honest constraint.
        """
        honest, poses = self._drifted(huber=0.5)
        last = max(honest.nodes)
        honest.add_edge(0, last, Pose2D(0.0, 0.0, 0.0),
                        information=np.diag([5.0, 5.0, 5.0]),
                        kind="loop_closure", confidence=1.0)
        honest.last_result = honest.optimise()

        corrupt, corrupt_result = self._corrupt(huber=0.5)
        corrupt.last_result = corrupt_result

        assert (corrupt.traced_trajectory().confidence
                < honest.traced_trajectory().confidence), (
            f"false: {corrupt.traced_trajectory().confidence:.3f} vs "
            f"true: {honest.traced_trajectory().confidence:.3f}"
        )

    def test_dead_reckoning_never_claims_high_confidence(self):
        """No loop closures means unbounded drift, however neatly it fits."""
        graph, _ = build_graph(square_loop(), drift_per_step=0.0)
        graph.last_result = graph.optimise()
        value = graph.traced_trajectory()
        assert value.source is Source.ESTIMATED
        assert value.confidence <= 0.6


# ===========================================================================
# Keyframe selection
# ===========================================================================

class TestKeyframeSelector:
    def test_the_first_frame_is_always_a_keyframe(self):
        promote, reason = KeyframeSelector().should_promote(Pose2D(0, 0, 0))
        assert promote and "first" in reason

    def test_a_stationary_rover_produces_no_new_keyframes(self):
        """Otherwise loitering fills the graph with duplicates of one pose."""
        selector = KeyframeSelector(min_translation_m=0.5)
        selector.accept(Pose2D(0, 0, 0), 0.0, "first")
        promote, reason = selector.should_promote(Pose2D(0.01, 0.0, 0.0),
                                                 timestamp=1.0)
        assert not promote and "insufficient" in reason

    def test_translation_promotes(self):
        selector = KeyframeSelector(min_translation_m=0.5)
        selector.accept(Pose2D(0, 0, 0), 0.0, "first")
        promote, reason = selector.should_promote(Pose2D(1.0, 0, 0), timestamp=1.0)
        assert promote and "travelled" in reason

    def test_pure_rotation_promotes_without_translation(self):
        """A turn changes the view completely while moving nothing."""
        selector = KeyframeSelector(min_rotation_rad=math.radians(10))
        selector.accept(Pose2D(0, 0, 0), 0.0, "first")
        promote, reason = selector.should_promote(
            Pose2D(0, 0, math.radians(25)), timestamp=1.0)
        assert promote and "turned" in reason

    def test_degraded_tracking_promotes(self):
        selector = KeyframeSelector(min_tracked_ratio=0.5)
        selector.accept(Pose2D(0, 0, 0), 0.0, "first")
        promote, reason = selector.should_promote(Pose2D(0.01, 0, 0), timestamp=1.0,
                                                  tracked_ratio=0.2)
        assert promote and "tracking" in reason

    def test_a_time_bound_still_produces_keyframes_on_a_slow_traverse(self):
        selector = KeyframeSelector(max_interval_s=10.0)
        selector.accept(Pose2D(0, 0, 0), 0.0, "first")
        promote, reason = selector.should_promote(Pose2D(0.01, 0, 0), timestamp=60.0)
        assert promote and "since the last keyframe" in reason


# ===========================================================================
# Loop detection
# ===========================================================================

@pytest.mark.skipif(not HAS_OPENCV, reason="loop detection needs OpenCV")
class TestLoopDetector:
    def _keyframe(self, node_id, pose, seed, n=80):
        rng = np.random.default_rng(seed)
        return Keyframe(
            id=node_id, pose=pose,
            descriptors=rng.integers(0, 255, (n, 32), dtype=np.uint8),
            keypoints=rng.uniform(0, 640, (n, 2)),
        )

    def test_nearby_keyframes_are_neighbours_not_loops(self):
        detector = LoopDetector(min_separation=10)
        query = self._keyframe(5, Pose2D(5, 0, 0), seed=1)
        others = [self._keyframe(i, Pose2D(i, 0, 0), seed=1) for i in range(5)]
        assert detector.detect(query, others) is None

    def test_unrelated_scenes_are_rejected(self):
        """Different random descriptors must not match."""
        detector = LoopDetector(min_separation=2)
        query = self._keyframe(20, Pose2D(20, 0, 0), seed=99)
        others = [self._keyframe(i, Pose2D(i, 0, 0), seed=i) for i in range(5)]
        assert detector.detect(query, others) is None

    def test_a_teleporting_closure_is_rejected_on_spatial_grounds(self):
        """A 40 m correction after a 2 m traverse is not drift."""
        detector = LoopDetector(min_separation=2, min_matches=1, min_inliers=1,
                                min_inlier_ratio=0.0, max_drift_ratio=0.5)
        shared = np.random.default_rng(7).integers(0, 255, (80, 32), dtype=np.uint8)
        points = np.random.default_rng(7).uniform(0, 640, (80, 2))
        query = Keyframe(id=20, pose=Pose2D(50.0, 0, 0), descriptors=shared,
                         keypoints=points)
        target = Keyframe(id=1, pose=Pose2D(0.0, 0, 0), descriptors=shared,
                          keypoints=points)
        detector.detect(query, [target], odometry_distance_m={1: 2.0})
        assert detector.candidates
        assert any("exceeds" in r for c in detector.candidates
                   for r in c.rejections)

    def test_statistics_expose_the_rejection_rate(self):
        detector = LoopDetector(min_separation=2)
        query = self._keyframe(20, Pose2D(20, 0, 0), seed=99)
        detector.detect(query, [self._keyframe(i, Pose2D(i, 0, 0), seed=i)
                                for i in range(5)])
        stats = detector.statistics()
        assert "candidates_examined" in stats and "acceptance_rate" in stats

    def test_confidence_is_capped_below_one(self):
        """Even a well-verified closure on repetitive terrain can be wrong."""
        assert LoopDetector._confidence(500, 1.0, 500) <= 0.9


# ===========================================================================
# The system
# ===========================================================================

class TestSlamSystem:
    def test_only_moving_frames_become_keyframes(self):
        system = SlamSystem(selector=KeyframeSelector(min_translation_m=1.0))
        for i in range(20):
            system.process(Pose2D(i * 0.1, 0.0, 0.0), timestamp=float(i))
        assert system.frames_seen == 20
        assert 2 <= len(system.keyframes) <= 4

    def test_odometry_edges_link_consecutive_keyframes(self):
        system = SlamSystem(selector=KeyframeSelector(min_translation_m=0.5))
        for i in range(10):
            system.process(Pose2D(float(i), 0.0, 0.0), timestamp=float(i))
        assert len(system.graph.edges) == len(system.keyframes) - 1
        assert all(e.kind == "odometry" for e in system.graph.edges)

    def test_no_stereo_means_no_landmarks_and_says_why(self):
        """An unscaled cloud beside a metric trajectory would mislead."""
        system = SlamSystem(selector=KeyframeSelector(min_translation_m=0.5))
        for i in range(6):
            system.process(Pose2D(float(i), 0.0, 0.0), timestamp=float(i))
        snapshot = system.snapshot()
        assert snapshot.landmark_status == "UNAVAILABLE"
        assert snapshot.landmarks == []
        assert "unknown scale" in snapshot.landmark_reason

    def test_measured_depth_produces_landmarks(self):
        system = SlamSystem(selector=KeyframeSelector(min_translation_m=0.5))
        cloud = np.stack([np.linspace(-1, 1, 50), np.zeros(50),
                          np.full(50, 5.0)], axis=1)
        for i in range(4):
            system.process(Pose2D(float(i), 0.0, 0.0), timestamp=float(i),
                           points_3d=cloud)
        snapshot = system.snapshot()
        assert snapshot.landmark_status == "MEASURED"
        assert len(snapshot.landmarks) > 0

    def test_landmarks_are_placed_ahead_of_the_rover(self):
        """Camera +Z is forward; a mis-signed transform puts the map behind."""
        system = SlamSystem(selector=KeyframeSelector(min_translation_m=0.5))
        cloud = np.array([[0.0, 0.0, 5.0]])
        system.process(Pose2D(0.0, 0.0, 0.0), timestamp=0.0, points_3d=cloud)
        assert system.landmarks[0].position[0] == pytest.approx(5.0, abs=1e-6)

    def test_the_landmark_budget_is_respected(self):
        system = SlamSystem(selector=KeyframeSelector(min_translation_m=0.5),
                            max_landmarks=100)
        cloud = np.random.default_rng(0).uniform(1, 5, (5000, 3))
        for i in range(5):
            system.process(Pose2D(float(i), 0.0, 0.0), timestamp=float(i),
                           points_3d=cloud)
        assert len(system.landmarks) <= 100

    def test_snapshot_is_json_shaped(self):
        system = SlamSystem(selector=KeyframeSelector(min_translation_m=0.5))
        for i in range(5):
            system.process(Pose2D(float(i), 0.0, 0.0), timestamp=float(i))
        payload = system.snapshot().to_dict()
        for key in ("keyframes", "trajectory", "landmarks", "loop_closures",
                    "statistics", "landmark_status"):
            assert key in payload

    def test_statistics_report_the_keyframe_rate(self):
        system = SlamSystem(selector=KeyframeSelector(min_translation_m=1.0))
        for i in range(20):
            system.process(Pose2D(i * 0.1, 0.0, 0.0), timestamp=float(i))
        stats = system.statistics()
        assert stats["frames_seen"] == 20
        assert 0.0 < stats["keyframe_rate"] < 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
