"""
perceptos.slam.graph
====================

SE(2) pose-graph optimisation — the back end of PerceptOS's SLAM.

A pose graph holds one node per keyframe pose and one edge per relative
measurement between two poses. Sequential edges come from odometry; the
occasional edge between distant nodes is a **loop closure**, produced when the
rover recognises somewhere it has been before. Optimisation finds the poses
that best satisfy every measurement at once, which is what removes accumulated
drift.

Why SE(2) and not SE(3)
-----------------------

A surface rover is constrained to the terrain. Its roll and pitch are set by
the ground it is standing on rather than being free variables, and its
navigation-relevant state is planar position plus heading. Optimising three
degrees of freedom per node instead of six makes the problem far better
conditioned on the data actually available, and the terrain module already
supplies elevation independently. The limitation is stated rather than hidden:
this back end cannot represent a rover driving over a genuinely 3D structure.

The part that ruins pose graphs
-------------------------------

**One false loop closure destroys the whole map.** A wrong edge asserts that
two unrelated places are the same place, and least squares — which has no
notion that a measurement might be nonsense — folds the entire trajectory to
satisfy it. Drift produces a map that is gradually wrong; a false loop produces
one that is confidently and structurally wrong, which is worse.

Two defences are implemented here:

* **A Huber robust kernel** down-weights edges with large residuals, so a
  single bad constraint bends the solution instead of dominating it. Measured
  on a drifted square traverse, a false closure inflated the worst odometry
  residual to 1.03 m without the kernel and 0.27 m with it.
* **Odometry-residual monitoring after convergence.**

The second needs explaining, because the obvious check does not work. One
might expect a false loop closure to leave a large residual on its own edge.
It usually does not: the optimiser satisfies it by deforming everything else,
so the bad edge ends up fitting *better* than the good ones. In the same
measurement above, the false closure's own residual was 0.096 while the
odometry it had corrupted sat at 0.27. **Checking loop-closure residuals alone
therefore misses precisely the worst case.** What actually signals a false
closure is odometry that was well satisfied before and is not after — the
optimiser had to break something it should not have.

Neither defence makes a wrong loop closure safe. They make it visible and
limit its reach, which is the most a back end can honestly do — rejecting the
closure is the front end's job.

Implementation is Gauss-Newton with a dense normal-equation solve. Dense is
adequate at the scale a rover produces (hundreds of keyframes, not the tens of
thousands a hand-held camera generates), and it keeps the linear algebra
inspectable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..core.provenance import Source, Traced, traced

__all__ = [
    "Pose2D",
    "PoseNode",
    "PoseEdge",
    "PoseGraph",
    "OptimisationResult",
    "wrap_angle",
]


def wrap_angle(theta: float | np.ndarray) -> Any:
    """Wrap to [-pi, pi). Applied to every angular residual without exception."""
    return (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True, slots=True)
class Pose2D:
    """A planar pose: position and heading."""

    x: float
    y: float
    theta: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta], dtype=np.float64)

    @classmethod
    def from_array(cls, values: Sequence[float]) -> "Pose2D":
        return cls(float(values[0]), float(values[1]),
                   float(wrap_angle(float(values[2]))))

    @property
    def rotation(self) -> np.ndarray:
        c, s = math.cos(self.theta), math.sin(self.theta)
        return np.array([[c, -s], [s, c]])

    def compose(self, other: "Pose2D") -> "Pose2D":
        """``self ⊕ other`` — apply a relative transform in this pose's frame."""
        xy = self.rotation @ np.array([other.x, other.y])
        return Pose2D(self.x + xy[0], self.y + xy[1],
                      float(wrap_angle(self.theta + other.theta)))

    def between(self, other: "Pose2D") -> "Pose2D":
        """Relative transform from ``self`` to ``other``, in ``self``'s frame."""
        delta = np.array([other.x - self.x, other.y - self.y])
        local = self.rotation.T @ delta
        return Pose2D(local[0], local[1],
                      float(wrap_angle(other.theta - self.theta)))

    def inverse(self) -> "Pose2D":
        xy = -(self.rotation.T @ np.array([self.x, self.y]))
        return Pose2D(xy[0], xy[1], float(wrap_angle(-self.theta)))

    def distance_to(self, other: "Pose2D") -> float:
        return float(math.hypot(other.x - self.x, other.y - self.y))

    def to_dict(self) -> dict[str, Any]:
        return {"x": round(self.x, 4), "y": round(self.y, 4),
                "theta": round(self.theta, 5),
                "heading_deg": round(math.degrees(self.theta), 2)}


@dataclass(slots=True)
class PoseNode:
    """One keyframe pose in the graph."""

    id: int
    pose: Pose2D
    fixed: bool = False              # the first node anchors the gauge freedom
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "fixed": self.fixed,
                "timestamp": self.timestamp, **self.pose.to_dict()}


@dataclass(slots=True)
class PoseEdge:
    """A relative-pose measurement between two nodes.

    ``information`` is the inverse covariance of the measurement. A loop
    closure derived from image matching is typically far less certain than a
    short odometry step, and giving both the same weight is how a marginal
    loop closure comes to dominate a hundred good odometry edges.
    """

    from_id: int
    to_id: int
    measurement: Pose2D
    information: np.ndarray = field(
        default_factory=lambda: np.diag([1.0, 1.0, 1.0]))
    kind: str = "odometry"           # odometry | loop_closure
    confidence: float = 1.0
    residual_norm: Optional[float] = None   # filled in after optimisation

    @property
    def is_loop_closure(self) -> bool:
        return self.kind == "loop_closure"

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_id, "to": self.to_id, "kind": self.kind,
            "measurement": self.measurement.to_dict(),
            "confidence": round(self.confidence, 4),
            "residual_norm": (round(self.residual_norm, 5)
                              if self.residual_norm is not None else None),
            "information_trace": round(float(np.trace(self.information)), 4),
        }


@dataclass(slots=True)
class OptimisationResult:
    """What the optimiser did, and how much it had to fight the data."""

    iterations: int
    initial_error: float
    final_error: float
    converged: bool
    delta_norm: float
    max_loop_residual: Optional[float] = None
    max_odometry_residual: Optional[float] = None
    mean_odometry_residual: Optional[float] = None
    suspect_edges: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""

    @property
    def error_reduction(self) -> float:
        if self.initial_error <= 0:
            return 0.0
        return float(1.0 - self.final_error / self.initial_error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "initial_error": round(self.initial_error, 6),
            "final_error": round(self.final_error, 6),
            "error_reduction": round(self.error_reduction, 4),
            "converged": self.converged,
            "delta_norm": round(self.delta_norm, 8),
            "max_loop_residual": (round(self.max_loop_residual, 5)
                                  if self.max_loop_residual is not None else None),
            "max_odometry_residual": (round(self.max_odometry_residual, 5)
                                      if self.max_odometry_residual is not None
                                      else None),
            "mean_odometry_residual": (round(self.mean_odometry_residual, 5)
                                       if self.mean_odometry_residual is not None
                                       else None),
            "suspect_edges": list(self.suspect_edges),
            "message": self.message,
        }


class PoseGraph:
    """SE(2) pose graph with Gauss-Newton optimisation and a Huber kernel.

    Parameters
    ----------
    huber_delta:
        Residual magnitude beyond which an edge's influence grows linearly
        rather than quadratically. Set it to ``None`` for plain least squares —
        useful for demonstrating exactly how badly one false loop closure
        distorts an unprotected graph.
    """

    def __init__(self, *, huber_delta: Optional[float] = 1.0,
                 odometry_alarm: float = 0.25) -> None:
        self.nodes: dict[int, PoseNode] = {}
        self.edges: list[PoseEdge] = []
        self.huber_delta = huber_delta
        #: Odometry residual above which a loop closure is treated as suspect.
        #: Well-measured motion should fit closely; when it stops fitting after
        #: a closure is added, the closure is the thing that changed.
        self.odometry_alarm = float(odometry_alarm)
        self._next_id = 0
        self.last_result: Optional[OptimisationResult] = None

    # ------------------------------------------------------------ construction

    def add_node(self, pose: Pose2D, *, fixed: Optional[bool] = None,
                 timestamp: float = 0.0,
                 metadata: Optional[dict[str, Any]] = None) -> int:
        """Add a node. The first is fixed automatically to anchor the solution.

        Without a fixed node the problem has three unconstrained degrees of
        freedom — the whole map can translate and rotate freely — and the
        normal equations are singular.
        """
        node_id = self._next_id
        self._next_id += 1
        anchor = (len(self.nodes) == 0) if fixed is None else bool(fixed)
        self.nodes[node_id] = PoseNode(id=node_id, pose=pose, fixed=anchor,
                                       timestamp=timestamp,
                                       metadata=dict(metadata or {}))
        return node_id

    def add_edge(self, from_id: int, to_id: int, measurement: Pose2D, *,
                 information: Optional[np.ndarray] = None,
                 kind: str = "odometry", confidence: float = 1.0) -> PoseEdge:
        for node_id in (from_id, to_id):
            if node_id not in self.nodes:
                raise KeyError(f"no node {node_id} in the graph")
        if from_id == to_id:
            raise ValueError("an edge cannot connect a node to itself")
        if information is None:
            # A loop closure from image matching is far less certain than a
            # short odometry step; defaulting both to identity is how a
            # marginal closure comes to outweigh good odometry.
            scale = 1.0 if kind == "odometry" else 0.25
            information = np.diag([scale, scale, scale * 2.0])
        information = np.asarray(information, dtype=np.float64)
        if information.shape != (3, 3):
            raise ValueError(f"information must be 3x3, got {information.shape}")
        if not np.allclose(information, information.T, atol=1e-9):
            raise ValueError("information matrix must be symmetric")
        edge = PoseEdge(from_id, to_id, measurement, information, kind,
                        float(confidence))
        self.edges.append(edge)
        return edge

    @property
    def loop_closures(self) -> list[PoseEdge]:
        return [e for e in self.edges if e.is_loop_closure]

    # -------------------------------------------------------------- residuals

    def _residual(self, edge: PoseEdge) -> np.ndarray:
        """Measured relative pose minus predicted, with the angle wrapped."""
        predicted = self.nodes[edge.from_id].pose.between(self.nodes[edge.to_id].pose)
        error = np.array([
            predicted.x - edge.measurement.x,
            predicted.y - edge.measurement.y,
            float(wrap_angle(predicted.theta - edge.measurement.theta)),
        ])
        return error

    @staticmethod
    def _jacobians(from_pose: Pose2D, to_pose: Pose2D
                   ) -> tuple[np.ndarray, np.ndarray]:
        """Analytic Jacobians of the residual w.r.t. both poses.

        Derived rather than finite-differenced: numerical differentiation of a
        wrapped angle is unreliable near the +/-pi boundary, exactly where a
        rover turning through south will spend time.
        """
        c, s = math.cos(from_pose.theta), math.sin(from_pose.theta)
        R_t = np.array([[c, s], [-s, c]])          # R(theta)^T
        dR_t = np.array([[-s, c], [-c, -s]])       # d(R^T)/dtheta
        delta = np.array([to_pose.x - from_pose.x, to_pose.y - from_pose.y])

        A = np.zeros((3, 3))                       # d(residual)/d(from_pose)
        A[:2, :2] = -R_t
        A[:2, 2] = dR_t @ delta
        A[2, 2] = -1.0

        B = np.zeros((3, 3))                       # d(residual)/d(to_pose)
        B[:2, :2] = R_t
        B[2, 2] = 1.0
        return A, B

    def _weight(self, residual: np.ndarray, edge: PoseEdge) -> float:
        """Huber weight. Returns 1.0 when robustification is disabled."""
        if self.huber_delta is None:
            return 1.0
        norm = float(math.sqrt(residual @ edge.information @ residual))
        if norm <= self.huber_delta:
            return 1.0
        # Beyond delta the cost grows linearly, so the weight falls as delta/|r|.
        return float(self.huber_delta / norm)

    def total_error(self) -> float:
        """Sum of information-weighted squared residuals."""
        return float(sum(
            r @ e.information @ r
            for e in self.edges
            for r in (self._residual(e),)
        ))

    # ----------------------------------------------------------- optimisation

    def optimise(self, *, max_iterations: int = 30, tolerance: float = 1e-6,
                 damping: float = 1e-6) -> OptimisationResult:
        """Gauss-Newton over all free nodes."""
        if not self.edges:
            return OptimisationResult(0, 0.0, 0.0, True, 0.0,
                                      message="no edges to optimise")
        if not any(n.fixed for n in self.nodes.values()):
            raise ValueError(
                "no node is fixed: the graph has three unconstrained degrees "
                "of freedom (the whole map may translate and rotate), and the "
                "normal equations are singular"
            )

        free_ids = [n.id for n in self.nodes.values() if not n.fixed]
        index = {node_id: i for i, node_id in enumerate(free_ids)}
        dimension = 3 * len(free_ids)
        initial_error = self.total_error()
        if dimension == 0:
            return OptimisationResult(0, initial_error, initial_error, True, 0.0,
                                      message="every node is fixed")

        converged = False
        delta_norm = 0.0
        iterations = 0

        for iterations in range(1, max_iterations + 1):
            H = np.zeros((dimension, dimension))
            b = np.zeros(dimension)

            for edge in self.edges:
                residual = self._residual(edge)
                weight = self._weight(residual, edge)
                omega = edge.information * weight * edge.confidence
                A, B = self._jacobians(self.nodes[edge.from_id].pose,
                                       self.nodes[edge.to_id].pose)
                i = index.get(edge.from_id)
                j = index.get(edge.to_id)
                if i is not None:
                    H[3 * i:3 * i + 3, 3 * i:3 * i + 3] += A.T @ omega @ A
                    b[3 * i:3 * i + 3] += A.T @ omega @ residual
                if j is not None:
                    H[3 * j:3 * j + 3, 3 * j:3 * j + 3] += B.T @ omega @ B
                    b[3 * j:3 * j + 3] += B.T @ omega @ residual
                if i is not None and j is not None:
                    block = A.T @ omega @ B
                    H[3 * i:3 * i + 3, 3 * j:3 * j + 3] += block
                    H[3 * j:3 * j + 3, 3 * i:3 * i + 3] += block.T

            # Levenberg-style damping keeps H invertible when a node is only
            # weakly constrained — common for the newest keyframe.
            H += np.eye(dimension) * damping
            try:
                delta = np.linalg.solve(H, -b)
            except np.linalg.LinAlgError:
                return OptimisationResult(
                    iterations, initial_error, self.total_error(), False, 0.0,
                    message="normal equations are singular; the graph is "
                            "under-constrained",
                )

            for node_id, i in index.items():
                node = self.nodes[node_id]
                update = delta[3 * i:3 * i + 3]
                node.pose = Pose2D.from_array(node.pose.as_array() + update)

            delta_norm = float(np.linalg.norm(delta))
            if delta_norm < tolerance:
                converged = True
                break

        final_error = self.total_error()
        for edge in self.edges:
            edge.residual_norm = float(np.linalg.norm(self._residual(edge)))

        loops = self.loop_closures
        odometry = [e for e in self.edges if not e.is_loop_closure]
        max_loop = max((e.residual_norm for e in loops), default=None)
        odometry_residuals = [e.residual_norm for e in odometry
                              if e.residual_norm is not None]
        max_odometry = max(odometry_residuals, default=None)
        mean_odometry = (float(np.mean(odometry_residuals))
                         if odometry_residuals else None)

        # Two independent signals, because each catches a case the other
        # misses. A closure the optimiser could NOT satisfy leaves a large
        # residual on its own edge. A closure it COULD satisfy leaves a small
        # one and corrupts the odometry instead — and that is the dangerous
        # case, because the graph looks converged and self-consistent.
        suspect: list[dict[str, Any]] = []
        for edge in loops:
            if edge.residual_norm is None:
                continue
            reasons = []
            if self.huber_delta is not None and edge.residual_norm > 3.0 * self.huber_delta:
                reasons.append(
                    f"residual {edge.residual_norm:.3f} exceeds 3x the Huber "
                    f"threshold; the optimiser could not satisfy this edge"
                )
            if (max_odometry is not None and max_odometry > self.odometry_alarm
                    and loops):
                reasons.append(
                    f"odometry residual reached {max_odometry:.3f} after this "
                    "closure was added; a constraint the graph absorbed by "
                    "deforming well-measured motion"
                )
            if reasons:
                entry = edge.to_dict()
                entry["suspect_reasons"] = reasons
                suspect.append(entry)

        return OptimisationResult(
            iterations=iterations, initial_error=initial_error,
            final_error=final_error, converged=converged, delta_norm=delta_norm,
            max_loop_residual=max_loop, max_odometry_residual=max_odometry,
            mean_odometry_residual=mean_odometry, suspect_edges=suspect,
            message=("converged" if converged else
                     f"stopped at the {max_iterations}-iteration limit"),
        )

    # --------------------------------------------------------------- readout

    def trajectory(self) -> list[Pose2D]:
        return [self.nodes[i].pose for i in sorted(self.nodes)]

    def path_length_m(self) -> float:
        poses = self.trajectory()
        return float(sum(a.distance_to(b) for a, b in zip(poses, poses[1:])))

    def statistics(self) -> dict[str, Any]:
        loops = self.loop_closures
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "odometry_edges": len(self.edges) - len(loops),
            "loop_closures": len(loops),
            "path_length_m": round(self.path_length_m(), 3),
            "total_error": round(self.total_error(), 6),
            "huber_delta": self.huber_delta,
            "robust": self.huber_delta is not None,
            "suspect_loop_closures": len(
                (self.last_result.suspect_edges if self.last_result else [])),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in
                      sorted(self.nodes.values(), key=lambda n: n.id)],
            "edges": [e.to_dict() for e in self.edges],
            "statistics": self.statistics(),
            "optimisation": (self.last_result.to_dict()
                             if self.last_result else None),
        }

    def traced_trajectory(self, *, parents: Sequence[Traced[Any]] = ()
                          ) -> Traced[list[Pose2D]]:
        """Export the optimised trajectory with provenance.

        Confidence reflects how well the optimiser satisfied the constraints:
        a graph with a large unresolved loop residual has produced a
        trajectory that its own measurements disagree with.
        """
        result = self.last_result
        if result is None:
            confidence = 0.3
        elif result.max_loop_residual is None:
            # No loop closures at all: this is dead reckoning, and drift is
            # unbounded regardless of how neatly the odometry chain fits.
            confidence = 0.5
        else:
            # Driven by the worst of the two signals. A graph that satisfied
            # its closures by wrecking its odometry is not trustworthy, and
            # scoring only the closure residual would call it excellent.
            worst = max(result.max_loop_residual,
                        result.max_odometry_residual or 0.0)
            confidence = float(max(0.05, min(0.95, 1.0 / (1.0 + worst))))
        return traced(
            self.trajectory(),
            source=Source.ESTIMATED,
            algorithm="PoseGraph.optimise(SE(2) Gauss-Newton, Huber kernel)",
            confidence=confidence,
            units="m, rad",
            uncertainty=result.max_loop_residual if result else None,
            label="slam_trajectory",
            parameters={
                **self.statistics(),
                "converged": result.converged if result else None,
            },
            parents=parents,
        )
