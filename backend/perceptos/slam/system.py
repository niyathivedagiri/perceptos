"""
perceptos.slam.system
=====================

The SLAM system: keyframes in, an optimised trajectory and landmark map out.

Wires the front end (:mod:`perceptos.slam.frontend`) to the back end
(:mod:`perceptos.slam.graph`) and exposes both as a
:class:`~perceptos.core.stage.PipelineStage`, so SLAM runs alongside the
existing perception stages rather than as a parallel system.

Odometry input is taken from whatever the pipeline already produces. If an EKF
is present its fused pose is used, because it has already reconciled wheel
odometry with visual odometry and knows its own covariance. Otherwise the
visual odometry increment is used directly and the resulting edges are given
weaker information, which is the honest reflection of a less certain input.

**Landmarks require measured depth.** A monocular front end could triangulate
them from motion, but only up to an unknown scale, and an unscaled landmark
cloud drawn beside a metric trajectory invites exactly the wrong reading. When
no stereo depth is available the map is a trajectory with no landmarks, and the
system says so rather than producing a plausible-looking cloud at an invented
scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from ..core.provenance import Source, Traced, traced
from ..core.stage import FrameContext, PipelineStage
from .frontend import Keyframe, KeyframeSelector, Landmark, LoopDetector
from .graph import OptimisationResult, Pose2D, PoseGraph

__all__ = ["SlamSystem", "SlamStage", "SlamSnapshot"]


@dataclass(slots=True)
class SlamSnapshot:
    """Everything the SLAM panel shows, each item traceable."""

    keyframes: list[dict[str, Any]]
    trajectory: list[dict[str, Any]]
    landmarks: list[dict[str, Any]]
    loop_closures: list[dict[str, Any]]
    optimisation: Optional[dict[str, Any]]
    statistics: dict[str, Any]
    landmark_status: str
    landmark_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "keyframes": self.keyframes,
            "trajectory": self.trajectory,
            "landmarks": self.landmarks,
            "loop_closures": self.loop_closures,
            "optimisation": self.optimisation,
            "statistics": self.statistics,
            "landmark_status": self.landmark_status,
            "landmark_reason": self.landmark_reason,
        }


class SlamSystem:
    """Keyframe-based SLAM with loop closure and pose-graph optimisation."""

    def __init__(self, *, selector: Optional[KeyframeSelector] = None,
                 detector: Optional[LoopDetector] = None,
                 graph: Optional[PoseGraph] = None,
                 optimise_every: int = 1,
                 max_landmarks: int = 20_000) -> None:
        self.selector = selector or KeyframeSelector()
        self.detector = detector or LoopDetector()
        self.graph = graph or PoseGraph()
        self.optimise_every = max(1, int(optimise_every))
        self.max_landmarks = int(max_landmarks)

        self.keyframes: list[Keyframe] = []
        self.landmarks: list[Landmark] = []
        self.frames_seen = 0
        self._odometry_distance: dict[int, float] = {}
        self._cumulative_distance = 0.0
        self._pending_closures = 0
        self.last_optimisation: Optional[OptimisationResult] = None

    # ------------------------------------------------------------- ingestion

    def process(self, pose: Pose2D, *, timestamp: float = 0.0,
                descriptors: Optional[np.ndarray] = None,
                keypoints: Optional[np.ndarray] = None,
                points_3d: Optional[np.ndarray] = None,
                tracked_ratio: float = 1.0,
                odometry_information: Optional[np.ndarray] = None,
                frame_index: int = 0) -> Optional[Keyframe]:
        """Offer a frame. Returns the keyframe if one was created."""
        self.frames_seen += 1
        promote, reason = self.selector.should_promote(
            pose, timestamp=timestamp, tracked_ratio=tracked_ratio)
        if not promote:
            return None

        previous = self.keyframes[-1] if self.keyframes else None
        node_id = self.graph.add_node(pose, timestamp=timestamp,
                                      metadata={"reason": reason})
        keyframe = Keyframe(
            id=node_id, pose=pose, descriptors=descriptors, keypoints=keypoints,
            points_3d=points_3d, timestamp=timestamp, frame_index=frame_index,
            metadata={"reason": reason},
        )
        self.keyframes.append(keyframe)
        self.selector.accept(pose, timestamp, reason)

        if previous is not None:
            relative = previous.pose.between(pose)
            self.graph.add_edge(previous.id, node_id, relative,
                                information=odometry_information,
                                kind="odometry")
            self._cumulative_distance += previous.pose.distance_to(pose)
        self._odometry_distance[node_id] = self._cumulative_distance

        if points_3d is not None and len(points_3d):
            self._add_landmarks(keyframe, points_3d)

        self._try_close_loop(keyframe)
        if self._pending_closures and len(self.keyframes) % self.optimise_every == 0:
            self.optimise()
        return keyframe

    def _add_landmarks(self, keyframe: Keyframe, points_3d: np.ndarray) -> None:
        """Transform measured camera-frame points into the world frame."""
        points = np.asarray(points_3d, dtype=np.float64)
        finite = points[np.isfinite(points).all(axis=1)]
        if finite.size == 0:
            return
        budget = self.max_landmarks - len(self.landmarks)
        if budget <= 0:
            return
        if len(finite) > budget:
            finite = finite[:: int(math.ceil(len(finite) / budget))]

        # Camera frame (X right, Y down, Z forward) into the planar world frame
        # at this keyframe's pose. Elevation is carried through unchanged; the
        # pose graph does not estimate it, and pretending otherwise would
        # imply a vertical accuracy this back end does not have.
        rotation = keyframe.pose.rotation
        for point in finite:
            planar = rotation @ np.array([point[2], -point[0]])
            self.landmarks.append(Landmark(
                id=len(self.landmarks),
                position=np.array([keyframe.pose.x + planar[0],
                                   keyframe.pose.y + planar[1],
                                   -point[1]]),
                observations=[keyframe.id],
            ))

    def _try_close_loop(self, keyframe: Keyframe) -> None:
        candidate = self.detector.detect(
            keyframe, self.keyframes[:-1],
            odometry_distance_m={
                k: self._cumulative_distance - v
                for k, v in self._odometry_distance.items()
            },
        )
        if candidate is None or not candidate.accepted:
            return
        # Information scales with the closure's own confidence, so a marginal
        # match cannot outweigh a chain of good odometry.
        scale = 0.5 * candidate.confidence
        self.graph.add_edge(
            candidate.match_id, candidate.query_id,
            self.keyframes[candidate.match_id].pose.between(keyframe.pose),
            information=np.diag([scale, scale, scale * 2.0]),
            kind="loop_closure", confidence=candidate.confidence,
        )
        self._pending_closures += 1

    # ---------------------------------------------------------- optimisation

    def optimise(self, **kwargs: Any) -> OptimisationResult:
        """Run the back end and write the corrected poses back to keyframes."""
        result = self.graph.optimise(**kwargs)
        self.graph.last_result = result
        self.last_optimisation = result
        for keyframe in self.keyframes:
            node = self.graph.nodes.get(keyframe.id)
            if node is not None:
                keyframe.pose = node.pose
        self._pending_closures = 0
        return result

    # --------------------------------------------------------------- readout

    def snapshot(self, *, max_landmarks: int = 4000) -> SlamSnapshot:
        landmarks = self.landmarks
        if not landmarks:
            status = "UNAVAILABLE"
            reason = (
                "No landmarks: this run had no stereo depth. A monocular front "
                "end could triangulate points, but only up to an unknown "
                "scale, and PerceptOS will not draw an unscaled cloud beside a "
                "metric trajectory. Run with a rectified stereo source."
            )
            sample: list[dict[str, Any]] = []
        else:
            status, reason = "MEASURED", ""
            step = max(1, math.ceil(len(landmarks) / max_landmarks))
            sample = [lm.to_dict() for lm in landmarks[::step]]

        return SlamSnapshot(
            keyframes=[k.to_dict() for k in self.keyframes],
            trajectory=[p.to_dict() for p in self.graph.trajectory()],
            landmarks=sample,
            loop_closures=[e.to_dict() for e in self.graph.loop_closures],
            optimisation=(self.last_optimisation.to_dict()
                          if self.last_optimisation else None),
            statistics=self.statistics(),
            landmark_status=status,
            landmark_reason=reason,
        )

    def statistics(self) -> dict[str, Any]:
        return {
            "frames_seen": self.frames_seen,
            "keyframes": len(self.keyframes),
            "landmarks": len(self.landmarks),
            "keyframe_rate": (round(len(self.keyframes) / self.frames_seen, 4)
                              if self.frames_seen else 0.0),
            **self.graph.statistics(),
            "loop_detection": self.detector.statistics(),
            "keyframe_selection": self.selector.statistics(),
        }

    def traced_trajectory(self, *, parents: Sequence[Traced[Any]] = ()
                          ) -> Traced[Any]:
        return self.graph.traced_trajectory(parents=parents)


class SlamStage(PipelineStage):
    """Runs SLAM from whatever pose and features the pipeline already produces."""

    name = "slam"
    requires = ()
    provides = ("slam_pose",)
    source = Source.ESTIMATED

    def __init__(self, system: Optional[SlamSystem] = None, **params: Any) -> None:
        super().__init__(**params)
        self.system = system or SlamSystem()
        self._pose = Pose2D(0.0, 0.0, 0.0)

    def preconditions_met(self, ctx: FrameContext) -> bool:
        return True

    def skip_reason(self, ctx: FrameContext) -> Optional[str]:
        if not ctx.has("odometry") and not ctx.has("fused_pose"):
            return ("no odometry or fused pose on the context; SLAM needs a "
                    "motion estimate to seed keyframe poses")
        return None

    def _current_pose(self, ctx: FrameContext) -> tuple[Pose2D, Optional[np.ndarray]]:
        """Prefer a fused EKF pose; fall back to integrating visual odometry."""
        fused = ctx.get("fused_pose")
        if fused is not None:
            estimate = fused.value
            pose = Pose2D(estimate.position[0], estimate.position[1],
                          estimate.heading_rad)
            # The EKF already knows its own uncertainty, so use it directly.
            covariance = np.asarray(estimate.covariance)[:3, :3]
            try:
                information = np.linalg.inv(covariance + np.eye(3) * 1e-9)
            except np.linalg.LinAlgError:
                information = None
            return pose, information

        odometry = ctx.get("odometry")
        result = odometry.value
        if getattr(result, "degenerate", False) or not getattr(
                result, "scale_resolved", False):
            # Unscaled or degenerate VO cannot advance a metric pose. Hold
            # position rather than integrating a unit direction as metres.
            return self._pose, np.diag([0.05, 0.05, 0.05])
        step = float(result.distance_m)
        yaw = math.radians(float(result.yaw_deg))
        self._pose = self._pose.compose(Pose2D(step, 0.0, yaw))
        confidence = max(0.05, float(getattr(result, "confidence", 0.5)))
        scale = confidence * 4.0
        return self._pose, np.diag([scale, scale, scale * 2.0])

    def run(self, ctx: FrameContext) -> str:
        pose, information = self._current_pose(ctx)
        features = ctx.get("features")
        depth = ctx.get("depth")
        rig = ctx.get("stereo_rig")

        descriptors = keypoints = points = None
        if features is not None:
            feature_set = features.value
            descriptors = getattr(feature_set, "descriptors", None)
            points_2d = getattr(feature_set, "points", None)
            keypoints = np.asarray(points_2d) if points_2d is not None else None
        if depth is not None and rig is not None:
            from ..stereo.depth import reproject_to_3d
            points = reproject_to_3d(depth.value.depth_m, rig.value, subsample=8)

        keyframe = self.system.process(
            pose, timestamp=ctx.created_at, descriptors=descriptors,
            keypoints=keypoints, points_3d=points,
            odometry_information=information, frame_index=ctx.frame_id,
        )

        parents = [v for v in (features, depth) if v is not None]
        trajectory = self.system.traced_trajectory(parents=parents)
        ctx.put("slam_pose", trajectory, stage=self.name)
        ctx.metadata["slam"] = self.system.statistics()

        if keyframe is None:
            return f"no keyframe ({self.system.statistics()['keyframes']} so far)"
        loops = len(self.system.graph.loop_closures)
        return (f"keyframe {keyframe.id} ({keyframe.metadata['reason']}), "
                f"{loops} loop closure(s), "
                f"{len(self.system.landmarks)} landmark(s)")
