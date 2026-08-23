"""
perceptos.mars.odometry
=======================

Visual odometry: how far has the rover actually moved?

Sparse feature tracking with Lucas-Kanade optical flow, pose recovery from the
essential matrix, and metric scale from stereo depth. The output feeds
:mod:`perceptos.mars.slip` — comparing this against wheel encoders is what
detects sinkage.

Three things this handles that a textbook implementation does not:

* **Monocular VO has no scale.** ``recoverPose`` returns a *unit* translation
  vector: direction only. Integrating it as metres gives a trajectory with the
  right shape and meaningless magnitude. When stereo depth is available the
  scale is recovered from the median depth of the tracked features; when it is
  not, the result is explicitly flagged ``scale_resolved=False`` so no caller
  can accidentally treat a direction as a distance.

* **Degenerate motion breaks the essential matrix.** Pure rotation, or a
  stationary camera, leaves the epipolar geometry undetermined and
  ``recoverPose`` returns confident garbage. Translation magnitude is checked
  against the feature disparity before the pose is accepted.

* **Planar scenes are degenerate for the essential matrix.** When every
  tracked feature lies on one plane — a flat regolith pan, a warehouse floor,
  a wall — the epipolar geometry is under-determined and ``recoverPose``
  returns a confident, wrong answer, classically a spurious 180-degree yaw.
  A homography is the correct model there. This module therefore scores both
  models against the same correspondences (the model-selection test used in
  ORB-SLAM) and refuses to emit a pose when the homography wins, rather than
  reporting the artefact as a measurement.

* **Feature attrition.** Optical flow loses points every frame. Without
  re-detection the tracker silently converges on a handful of points and the
  pose becomes noise, so features are topped up whenever the count falls below
  a threshold — and the forward-backward flow check discards points that track
  to a different place on the way back, which is the cheapest reliable outlier
  filter available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from ..core.cv_compat import cv
from ..core.provenance import Source, Traced, traced
from ..core.stage import FrameContext, PipelineStage

__all__ = ["OdometryResult", "VisualOdometry", "VisualOdometryStage"]


@dataclass(slots=True)
class OdometryResult:
    """Incremental motion estimate for one frame pair."""

    translation: np.ndarray            # 3-vector, camera frame
    rotation: np.ndarray               # 3x3
    scale_resolved: bool
    tracked_points: int
    inliers: int
    confidence: float
    degenerate: bool = False
    note: str = ""
    #: Model-selection score in [0, 1]. Above ~0.45 a homography explains the
    #: correspondences better than epipolar geometry, i.e. the scene is planar.
    planarity: float = 0.0

    @property
    def distance_m(self) -> float:
        """Magnitude of the step. Meaningless unless ``scale_resolved``."""
        return float(np.linalg.norm(self.translation))

    @property
    def yaw_deg(self) -> float:
        """Heading change about the camera's vertical axis."""
        return float(math.degrees(math.atan2(self.rotation[0, 2], self.rotation[2, 2])))

    @property
    def inlier_ratio(self) -> float:
        return self.inliers / self.tracked_points if self.tracked_points else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "translation": [round(float(v), 5) for v in self.translation],
            "distance_m": round(self.distance_m, 4) if self.scale_resolved else None,
            "distance_unitless": round(self.distance_m, 4),
            "yaw_deg": round(self.yaw_deg, 3),
            "scale_resolved": self.scale_resolved,
            "tracked_points": self.tracked_points,
            "inliers": self.inliers,
            "inlier_ratio": round(self.inlier_ratio, 3),
            "confidence": round(self.confidence, 3),
            "degenerate": self.degenerate,
            "planarity": round(self.planarity, 3),
            "note": self.note,
        }


class VisualOdometry:
    """Sparse feature-based VO with optional stereo scale.

    Parameters
    ----------
    camera_matrix:
        3x3 intrinsics of the (undistorted) camera.
    min_features:
        Re-detect when tracked points fall below this. Running out of features
        is the usual reason a VO trajectory quietly goes wrong.
    min_translation_px:
        Median feature displacement below which motion is treated as
        degenerate rather than solved.
    """

    def __init__(self, camera_matrix: np.ndarray, *, max_features: int = 800,
                 min_features: int = 150, min_translation_px: float = 1.5,
                 fb_threshold_px: float = 1.0, planarity_threshold: float = 0.45) -> None:
        self.K = np.asarray(camera_matrix, dtype=np.float64)
        if self.K.shape != (3, 3):
            raise ValueError(f"camera_matrix must be 3x3, got {self.K.shape}")
        self.max_features = int(max_features)
        self.min_features = int(min_features)
        self.min_translation_px = float(min_translation_px)
        self.fb_threshold_px = float(fb_threshold_px)
        self.planarity_threshold = float(planarity_threshold)

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_points: Optional[np.ndarray] = None
        self.pose_R = np.eye(3)
        self.pose_t = np.zeros((3, 1))
        self.path: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
        self.frames = 0

    # ------------------------------------------------------------- detection

    def _detect(self, gray: np.ndarray) -> np.ndarray:
        cv2 = cv()
        points = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_features, qualityLevel=0.01,
            minDistance=12, blockSize=7,
        )
        return points if points is not None else np.zeros((0, 1, 2), np.float32)

    def _track(self, prev_gray: np.ndarray, gray: np.ndarray, points: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
        """LK flow with a forward-backward consistency check."""
        cv2 = cv()
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        forward, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, points, None, winSize=(21, 21), maxLevel=3, criteria=criteria
        )
        if forward is None:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        backward, status_b, _ = cv2.calcOpticalFlowPyrLK(
            gray, prev_gray, forward, None, winSize=(21, 21), maxLevel=3, criteria=criteria
        )
        if backward is None:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

        p0 = points.reshape(-1, 2)
        p1 = forward.reshape(-1, 2)
        pb = backward.reshape(-1, 2)
        ok = (status.ravel() == 1) & (status_b.ravel() == 1)
        # A point that tracks forward then back to somewhere else was never
        # tracked correctly. This one check removes most flow outliers.
        ok &= np.linalg.norm(p0 - pb, axis=1) < self.fb_threshold_px
        return p0[ok], p1[ok]

    # ------------------------------------------------------- model selection

    @staticmethod
    def _score_homography(H: np.ndarray, p0: np.ndarray, p1: np.ndarray,
                          sigma: float = 1.0) -> float:
        """Symmetric transfer error scored against the 2-DOF chi-square bound."""
        if H is None:
            return 0.0
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return 0.0
        threshold = 5.991           # chi-square 95%, 2 degrees of freedom
        inv_sigma2 = 1.0 / (sigma * sigma)
        score = 0.0
        for forward, a, b in ((True, p0, p1), (False, p1, p0)):
            M = H if forward else H_inv
            ones = np.ones((len(a), 1), np.float64)
            projected = (M @ np.hstack([a, ones]).T).T
            w = projected[:, 2:3]
            w[np.abs(w) < 1e-12] = 1e-12
            residual = np.sum((projected[:, :2] / w - b) ** 2, axis=1) * inv_sigma2
            score += float(np.sum(np.where(residual < threshold, threshold - residual, 0.0)))
        return score

    @staticmethod
    def _score_fundamental(F: np.ndarray, p0: np.ndarray, p1: np.ndarray,
                           sigma: float = 1.0) -> float:
        """Symmetric epipolar distance scored against the 1-DOF chi-square bound."""
        if F is None or F.shape != (3, 3):
            return 0.0
        # Different threshold from the homography on purpose: the models have
        # different degrees of freedom, so comparing raw residuals would always
        # favour the more flexible homography and declare every scene planar.
        threshold, score_cap = 3.841, 5.991
        inv_sigma2 = 1.0 / (sigma * sigma)
        ones = np.ones((len(p0), 1), np.float64)
        a = np.hstack([p0, ones])
        b = np.hstack([p1, ones])
        score = 0.0
        for lines, points in ((F @ a.T, b), (F.T @ b.T, a)):
            lines = lines.T
            norm = lines[:, 0] ** 2 + lines[:, 1] ** 2
            norm[norm < 1e-12] = 1e-12
            numerator = np.sum(lines * points, axis=1) ** 2
            residual = numerator / norm * inv_sigma2
            score += float(np.sum(np.where(residual < threshold, score_cap - residual, 0.0)))
        return score

    def _planarity(self, p0: np.ndarray, p1: np.ndarray) -> float:
        """Fraction of explanatory power held by the homography model.

        Returns ``S_H / (S_H + S_F)``. Near 1 the scene is a plane (or the
        motion is pure rotation) and the essential matrix must not be trusted.
        """
        cv2 = cv()
        p0 = np.asarray(p0, np.float64)
        p1 = np.asarray(p1, np.float64)
        H, _ = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
        F, _ = cv2.findFundamentalMat(p0, p1, cv2.FM_RANSAC, 3.0, 0.99)
        if F is not None and F.shape[0] > 3:
            F = F[:3]                      # RANSAC can return stacked solutions
        s_h = self._score_homography(H, p0, p1)
        s_f = self._score_fundamental(F, p0, p1)
        total = s_h + s_f
        return float(s_h / total) if total > 0 else 1.0

    # --------------------------------------------------------------- estimate

    def process(self, gray: np.ndarray,
                depth_m: Optional[np.ndarray] = None) -> Optional[OdometryResult]:
        """Estimate motion from the previous frame to this one.

        Returns ``None`` on the first frame, when there is nothing to compare
        against.
        """
        cv2 = cv()
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        self.frames += 1

        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_points = self._detect(gray)
            return None

        if self._prev_points is None or len(self._prev_points) < self.min_features:
            self._prev_points = self._detect(self._prev_gray)

        p0, p1 = self._track(self._prev_gray, gray, self._prev_points)
        tracked = len(p0)

        if tracked < 8:
            self._prev_gray = gray
            self._prev_points = self._detect(gray)
            return OdometryResult(
                np.zeros(3), np.eye(3), False, tracked, 0, 0.0, degenerate=True,
                note=f"only {tracked} points survived tracking; need 8 for the "
                     "essential matrix",
            )

        displacement = float(np.median(np.linalg.norm(p1 - p0, axis=1)))
        if displacement < self.min_translation_px:
            self._prev_gray = gray
            self._prev_points = p1.reshape(-1, 1, 2)
            return OdometryResult(
                np.zeros(3), np.eye(3), True, tracked, tracked, 0.9,
                degenerate=True,
                note=f"median displacement {displacement:.2f} px - rover is "
                     "effectively stationary; pose left unchanged",
            )

        planarity = self._planarity(p0, p1)
        if planarity > self.planarity_threshold:
            self._prev_gray = gray
            self._prev_points = p1.reshape(-1, 1, 2)
            return OdometryResult(
                np.zeros(3), np.eye(3), False, tracked, 0, 0.0, degenerate=True,
                planarity=planarity,
                note=f"planar scene (homography score {planarity:.2f} > "
                     f"{self.planarity_threshold}); the essential matrix is "
                     "under-determined here and would return a spurious pose",
            )

        E, mask = cv2.findEssentialMat(
            p1, p0, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None or E.shape != (3, 3):
            self._prev_gray = gray
            self._prev_points = self._detect(gray)
            return OdometryResult(
                np.zeros(3), np.eye(3), False, tracked, 0, 0.0, degenerate=True,
                note="essential matrix could not be estimated (degenerate geometry)",
            )

        inliers, R, t, pose_mask = cv2.recoverPose(E, p1, p0, self.K, mask=mask)
        inliers = int(inliers)

        # recoverPose returns a unit translation. Recover metric scale from the
        # depth of the inlier features if a depth map is available.
        scale, scale_resolved, note = self._resolve_scale(p1, pose_mask, depth_m)
        translation = (t.ravel() * scale) if scale_resolved else t.ravel()

        inlier_ratio = inliers / tracked if tracked else 0.0
        confidence = float(max(0.0, min(1.0,
            0.5 * inlier_ratio
            + 0.3 * min(1.0, tracked / self.min_features)
            + 0.2 * (1.0 if scale_resolved else 0.0)
        )))

        if scale_resolved:
            self.pose_t = self.pose_t + self.pose_R @ translation.reshape(3, 1)
            self.pose_R = R @ self.pose_R
            self.path.append(tuple(float(v) for v in self.pose_t.ravel()))

        self._prev_gray = gray
        self._prev_points = (p1[pose_mask.ravel() > 0].reshape(-1, 1, 2)
                             if pose_mask is not None else p1.reshape(-1, 1, 2))
        if len(self._prev_points) < self.min_features:
            self._prev_points = self._detect(gray)

        return OdometryResult(
            translation=translation, rotation=R, scale_resolved=scale_resolved,
            tracked_points=tracked, inliers=inliers, confidence=confidence,
            planarity=planarity, note=note,
        )

    def _resolve_scale(self, points: np.ndarray, pose_mask: Optional[np.ndarray],
                       depth_m: Optional[np.ndarray]) -> tuple[float, bool, str]:
        """Metric scale from the median depth of the inlier features."""
        if depth_m is None:
            return 1.0, False, ("no depth map: translation is a unit direction, "
                                "not a distance")
        h, w = depth_m.shape[:2]
        selected = points
        if pose_mask is not None and len(pose_mask) == len(points):
            keep = pose_mask.ravel() > 0
            if keep.any():
                selected = points[keep]
        cols = np.clip(selected[:, 0].astype(int), 0, w - 1)
        rows = np.clip(selected[:, 1].astype(int), 0, h - 1)
        depths = depth_m[rows, cols]
        depths = depths[np.isfinite(depths) & (depths > 0)]
        if depths.size < 5:
            return 1.0, False, (f"only {depths.size} features had valid depth; "
                                "scale unresolved")
        # The unit translation multiplied by a characteristic scene depth gives
        # the metric step. Median resists the outliers stereo always produces.
        return float(np.median(depths)) * 0.1, True, ""

    @property
    def total_distance_m(self) -> float:
        return float(sum(
            math.dist(self.path[i], self.path[i + 1]) for i in range(len(self.path) - 1)
        ))

    def statistics(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "path_points": len(self.path),
            "total_distance_m": round(self.total_distance_m, 4),
            "position": [round(float(v), 4) for v in self.pose_t.ravel()],
            "tracked_points": len(self._prev_points) if self._prev_points is not None else 0,
        }

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_points = None
        self.pose_R = np.eye(3)
        self.pose_t = np.zeros((3, 1))
        self.path = [(0.0, 0.0, 0.0)]


class VisualOdometryStage(PipelineStage):
    """Estimates frame-to-frame rover motion."""

    name = "visual_odometry"
    requires = ("gray",)
    provides = ("odometry",)
    source = Source.ESTIMATED

    def __init__(self, odometry: VisualOdometry, **params: Any) -> None:
        super().__init__(**params)
        self.odometry = odometry

    def run(self, ctx: FrameContext) -> str:
        (gray,) = ctx.require("gray")
        depth_tv = ctx.get("depth")
        depth_map = depth_tv.value.depth_m if depth_tv is not None else None

        result = self.odometry.process(gray.value, depth_map)
        if result is None:
            return "first frame: reference features seeded, no motion yet"

        parents = [gray] + ([depth_tv] if depth_tv is not None else [])
        self.emit(ctx, "odometry", result,
                  algorithm="VisualOdometry(LK flow + essential matrix)",
                  parents=parents, confidence=result.confidence, units="m",
                  parameters=self.odometry.statistics())
        if result.degenerate:
            return f"degenerate: {result.note}"
        scale = f"{result.distance_m:.3f} m" if result.scale_resolved else "unit direction"
        return (f"{scale}, yaw {result.yaw_deg:+.2f} deg, "
                f"{result.inliers}/{result.tracked_points} inliers")
