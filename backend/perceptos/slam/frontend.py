"""
perceptos.slam.frontend
=======================

The SLAM front end: deciding what to remember and what has been seen before.

The back end (:mod:`perceptos.slam.graph`) solves whatever constraints it is
given. Everything that determines whether those constraints are *correct*
happens here.

Keyframe selection
------------------

Not every frame becomes a node. A rover sitting still would otherwise add
hundreds of nodes describing the same pose, which inflates the problem without
adding information and biases the optimiser toward wherever it happened to
loiter. A frame is promoted when it has travelled far enough, turned far
enough, or when tracking quality has degraded enough that a fresh reference is
needed.

Loop closure — where SLAM goes wrong
------------------------------------

Recognising a previously-visited place is what makes SLAM better than dead
reckoning. Getting it wrong is what makes it much worse: a false loop closure
asserts two unrelated places are identical and folds the whole trajectory to
match. On Mars this failure is unusually easy to trigger, because regolith is
visually self-similar for kilometres — one patch of rocky sand matches another
patch of rocky sand very convincingly.

Three filters run in series, and a candidate must pass all of them:

1. **Appearance.** Descriptor matching with Lowe's ratio test. Necessary, and
   nowhere near sufficient on repetitive terrain.
2. **Geometric verification.** The matches must be explicable by a single rigid
   transform, found by RANSAC. Random matches between two similar-looking but
   different places do not agree on one transform, and this is the filter that
   rejects most false positives.
3. **Spatial plausibility.** The proposed closure must not require the rover to
   have teleported: the implied correction is compared against the accumulated
   odometry drift, and a closure demanding a correction far larger than the
   drift could plausibly be is rejected.

A candidate that passes all three is still reported with a confidence rather
than as a fact, and the back end weights it accordingly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..core.cv_compat import HAS_OPENCV, cv
from .graph import Pose2D, wrap_angle

__all__ = [
    "Keyframe",
    "LoopCandidate",
    "KeyframeSelector",
    "LoopDetector",
    "Landmark",
]


@dataclass(slots=True)
class Keyframe:
    """A remembered view: its pose estimate and what it looked like."""

    id: int
    pose: Pose2D
    descriptors: Optional[np.ndarray] = None
    keypoints: Optional[np.ndarray] = None       # (N, 2) pixel coordinates
    points_3d: Optional[np.ndarray] = None       # (N, 3) camera-frame, if stereo
    timestamp: float = 0.0
    frame_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def feature_count(self) -> int:
        return 0 if self.descriptors is None else int(len(self.descriptors))

    @property
    def has_depth(self) -> bool:
        return self.points_3d is not None and len(self.points_3d) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "frame_index": self.frame_index,
            "timestamp": self.timestamp, "features": self.feature_count,
            "has_depth": self.has_depth, **self.pose.to_dict(),
        }


@dataclass(slots=True)
class Landmark:
    """A 3D point observed from one or more keyframes.

    Landmarks are only created when stereo supplied a *measured* depth. A
    monocular front end can triangulate them from motion, but the scale would
    be unknown, and an unscaled landmark map drawn next to a metric trajectory
    invites exactly the wrong reading.
    """

    id: int
    position: np.ndarray                          # (3,) world frame, metres
    observations: list[int] = field(default_factory=list)
    descriptor: Optional[np.ndarray] = None

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id,
                "position": [round(float(v), 4) for v in self.position],
                "observations": self.observation_count}


@dataclass(slots=True)
class LoopCandidate:
    """A proposed loop closure and the evidence for it."""

    query_id: int
    match_id: int
    inliers: int
    total_matches: int
    inlier_ratio: float
    relative_pose: Pose2D
    confidence: float
    accepted: bool
    rejections: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id, "match_id": self.match_id,
            "inliers": self.inliers, "total_matches": self.total_matches,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "relative_pose": self.relative_pose.to_dict(),
            "confidence": round(self.confidence, 4),
            "accepted": self.accepted,
            "rejections": list(self.rejections),
            "evidence": list(self.evidence),
        }


class KeyframeSelector:
    """Decides which frames earn a node in the graph.

    Parameters
    ----------
    min_translation_m / min_rotation_rad:
        Motion thresholds. Either alone promotes a frame — a pure rotation
        changes the view completely while moving nothing.
    min_tracked_ratio:
        Promote when the fraction of features still tracked from the last
        keyframe falls below this, because the reference has gone stale even
        if the rover has barely moved.
    max_interval_s:
        Upper bound so a very slow traverse still produces keyframes.
    """

    def __init__(self, *, min_translation_m: float = 0.5,
                 min_rotation_rad: float = math.radians(10.0),
                 min_tracked_ratio: float = 0.5,
                 max_interval_s: float = 30.0) -> None:
        self.min_translation_m = float(min_translation_m)
        self.min_rotation_rad = float(min_rotation_rad)
        self.min_tracked_ratio = float(min_tracked_ratio)
        self.max_interval_s = float(max_interval_s)
        self.last_pose: Optional[Pose2D] = None
        self.last_timestamp: float = 0.0
        self.promotions: list[str] = []

    def should_promote(self, pose: Pose2D, *, timestamp: float = 0.0,
                       tracked_ratio: float = 1.0) -> tuple[bool, str]:
        """Return ``(promote, reason)``. The reason is recorded in the keyframe."""
        if self.last_pose is None:
            return True, "first keyframe"
        relative = self.last_pose.between(pose)
        translation = math.hypot(relative.x, relative.y)
        rotation = abs(float(wrap_angle(relative.theta)))
        elapsed = timestamp - self.last_timestamp

        if translation >= self.min_translation_m:
            return True, f"travelled {translation:.2f} m"
        if rotation >= self.min_rotation_rad:
            return True, f"turned {math.degrees(rotation):.1f} deg"
        if tracked_ratio < self.min_tracked_ratio:
            return True, f"tracking fell to {tracked_ratio:.0%}"
        if elapsed >= self.max_interval_s:
            return True, f"{elapsed:.0f} s since the last keyframe"
        return False, "insufficient motion"

    def accept(self, pose: Pose2D, timestamp: float, reason: str) -> None:
        self.last_pose = pose
        self.last_timestamp = timestamp
        self.promotions.append(reason)

    def statistics(self) -> dict[str, Any]:
        return {
            "promotions": len(self.promotions),
            "min_translation_m": self.min_translation_m,
            "min_rotation_deg": round(math.degrees(self.min_rotation_rad), 1),
            "recent_reasons": self.promotions[-5:],
        }


class LoopDetector:
    """Proposes loop closures, and refuses most of what it proposes.

    Parameters
    ----------
    min_separation:
        Keyframes closer than this many indices are neighbours, not a loop.
        Without it every frame "closes a loop" with its predecessor.
    min_inliers / min_inlier_ratio:
        Geometric verification thresholds. Appearance alone is not evidence on
        terrain that repeats.
    max_drift_ratio:
        The implied correction may not exceed this fraction of the distance
        travelled since the matched keyframe. A closure demanding a 40 m
        correction after a 10 m traverse is proposing that the rover teleported.
    """

    def __init__(self, *, min_separation: int = 10, min_matches: int = 25,
                 min_inliers: int = 15, min_inlier_ratio: float = 0.35,
                 ratio_test: float = 0.75,
                 max_drift_ratio: float = 0.5) -> None:
        self.min_separation = int(min_separation)
        self.min_matches = int(min_matches)
        self.min_inliers = int(min_inliers)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.ratio_test = float(ratio_test)
        self.max_drift_ratio = float(max_drift_ratio)
        self.candidates: list[LoopCandidate] = []

    # ------------------------------------------------------------ appearance

    def _match(self, query: Keyframe, target: Keyframe) -> list[Any]:
        """Descriptor matching with Lowe's ratio test."""
        if not HAS_OPENCV:
            return []
        if query.descriptors is None or target.descriptors is None:
            return []
        if len(query.descriptors) < 2 or len(target.descriptors) < 2:
            return []
        cv2 = cv()
        # Binary descriptors (ORB) need Hamming; float descriptors (SIFT) need
        # L2. Using the wrong norm still returns matches — wrong ones.
        norm = (cv2.NORM_HAMMING if query.descriptors.dtype == np.uint8
                else cv2.NORM_L2)
        matcher = cv2.BFMatcher(norm, crossCheck=False)
        raw = matcher.knnMatch(query.descriptors, target.descriptors, k=2)
        good = []
        for candidates in raw:
            if len(candidates) < 2:
                continue
            best, second = candidates
            if best.distance < self.ratio_test * second.distance:
                good.append(best)
        return good

    # -------------------------------------------------------------- geometry

    def _verify(self, query: Keyframe, target: Keyframe, matches: Sequence[Any]
                ) -> tuple[int, Optional[np.ndarray]]:
        """RANSAC a rigid 2D transform through the matched keypoints.

        This is the filter that rejects most false positives: matches between
        two different-but-similar places do not agree on any single transform,
        so the inlier count collapses.
        """
        if not HAS_OPENCV or len(matches) < self.min_matches:
            return 0, None
        if query.keypoints is None or target.keypoints is None:
            return 0, None
        cv2 = cv()
        src = np.array([query.keypoints[m.queryIdx] for m in matches],
                       dtype=np.float32).reshape(-1, 1, 2)
        dst = np.array([target.keypoints[m.trainIdx] for m in matches],
                       dtype=np.float32).reshape(-1, 1, 2)
        try:
            transform, inlier_mask = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                maxIters=2000, confidence=0.99,
            )
        except Exception:  # noqa: BLE001
            return 0, None
        if transform is None or inlier_mask is None:
            return 0, None
        return int(inlier_mask.sum()), transform

    # ------------------------------------------------------------- detection

    def detect(self, query: Keyframe, keyframes: Sequence[Keyframe], *,
               odometry_distance_m: Optional[dict[int, float]] = None
               ) -> Optional[LoopCandidate]:
        """Look for a loop closure for ``query``. Returns the best, if any."""
        best: Optional[LoopCandidate] = None
        for target in keyframes:
            if abs(query.id - target.id) < self.min_separation:
                continue
            matches = self._match(query, target)
            if len(matches) < self.min_matches:
                continue
            inliers, transform = self._verify(query, target, matches)
            ratio = inliers / len(matches) if matches else 0.0

            rejections: list[str] = []
            evidence = [f"{len(matches)} descriptor matches survived the ratio test"]
            if inliers < self.min_inliers:
                rejections.append(
                    f"only {inliers} geometric inliers (need {self.min_inliers}); "
                    "the matches do not agree on a single transform"
                )
            elif ratio < self.min_inlier_ratio:
                rejections.append(
                    f"inlier ratio {ratio:.0%} below {self.min_inlier_ratio:.0%}"
                )
            else:
                evidence.append(f"{inliers} inliers ({ratio:.0%}) agree on one "
                                "rigid transform")

            relative = query.pose.between(target.pose)
            implied = math.hypot(relative.x, relative.y)
            if odometry_distance_m is not None:
                travelled = odometry_distance_m.get(target.id)
                if travelled is not None and travelled > 0:
                    allowed = self.max_drift_ratio * travelled
                    if implied > allowed:
                        rejections.append(
                            f"implied correction {implied:.1f} m exceeds "
                            f"{allowed:.1f} m of plausible drift over "
                            f"{travelled:.1f} m travelled"
                        )
                    else:
                        evidence.append(
                            f"correction {implied:.1f} m is within plausible "
                            f"drift for {travelled:.1f} m travelled"
                        )

            accepted = not rejections
            confidence = self._confidence(inliers, ratio, len(matches)) if accepted else 0.0
            candidate = LoopCandidate(
                query_id=query.id, match_id=target.id, inliers=inliers,
                total_matches=len(matches), inlier_ratio=ratio,
                relative_pose=relative, confidence=confidence,
                accepted=accepted, rejections=rejections, evidence=evidence,
            )
            self.candidates.append(candidate)
            if accepted and (best is None or candidate.confidence > best.confidence):
                best = candidate
        return best

    @staticmethod
    def _confidence(inliers: int, ratio: float, matches: int) -> float:
        """Confidence in a proposed closure. Deliberately capped below 1.

        Even a well-verified closure on self-similar terrain can be wrong, and
        the back end's robust kernel exists precisely because this number is
        never a certainty.
        """
        inlier_score = min(1.0, inliers / 60.0)
        return float(max(0.05, min(0.9, 0.5 * ratio + 0.5 * inlier_score)))

    def statistics(self) -> dict[str, Any]:
        accepted = [c for c in self.candidates if c.accepted]
        return {
            "candidates_examined": len(self.candidates),
            "accepted": len(accepted),
            "rejected": len(self.candidates) - len(accepted),
            "acceptance_rate": (round(len(accepted) / len(self.candidates), 4)
                                if self.candidates else 0.0),
            "min_inliers": self.min_inliers,
            "min_inlier_ratio": self.min_inlier_ratio,
            "recent": [c.to_dict() for c in self.candidates[-5:]],
        }
