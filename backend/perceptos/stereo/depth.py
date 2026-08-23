"""
perceptos.stereo.depth
======================

Disparity -> depth -> metric 3D, with honest uncertainty.

The governing relation for a rectified stereo pair is::

    Z = f * B / d

where ``f`` is the focal length in pixels, ``B`` the baseline in metres and
``d`` the disparity in pixels.  The important consequence, and the reason this
module exists separately from the matcher, is that depth error grows with the
*square* of depth::

    sigma_Z = Z^2 * sigma_d / (f * B)

A stereo rig with 0.25 px matching precision is accurate to millimetres at 1 m
and to metres at 40 m.  A perception platform that reports a depth without that
error bar is inviting an operator to trust a number that does not deserve it,
so every depth this module produces is accompanied by ``sigma_Z``, and the
``max_reliable_range`` helper tells the mission planner where to stop believing
its own stereo.

Pure NumPy — the matcher (OpenCV) lives in :mod:`perceptos.stereo.matcher`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..core.provenance import Source, Traced

__all__ = ["StereoRig", "DepthResult", "disparity_to_depth", "reproject_to_3d"]


@dataclass(frozen=True, slots=True)
class StereoRig:
    """Calibrated rectified stereo geometry."""

    fx: float                      # focal length, pixels
    fy: float
    cx: float                      # principal point, pixels
    cy: float
    baseline_m: float
    disparity_sigma_px: float = 0.25   # sub-pixel matching precision
    min_disparity_px: float = 1.0      # below this, depth is unbounded

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("focal lengths must be positive")
        if self.baseline_m <= 0:
            raise ValueError("baseline must be positive")

    @classmethod
    def from_intrinsics(cls, K: np.ndarray, baseline_m: float, **kwargs: Any) -> "StereoRig":
        K = np.asarray(K, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"intrinsic matrix must be 3x3, got {K.shape}")
        return cls(
            fx=float(K[0, 0]), fy=float(K[1, 1]),
            cx=float(K[0, 2]), cy=float(K[1, 2]),
            baseline_m=float(baseline_m), **kwargs,
        )

    def depth_sigma(self, depth_m: float | np.ndarray) -> Any:
        """Depth standard deviation at a given range — grows as Z^2."""
        z = np.asarray(depth_m, dtype=np.float64)
        return z ** 2 * self.disparity_sigma_px / (self.fx * self.baseline_m)

    def max_reliable_range(self, max_relative_error: float = 0.05) -> float:
        """Range beyond which relative depth error exceeds the tolerance.

        From sigma_Z / Z = Z * sigma_d / (f * B), solving for Z.
        """
        if not 0 < max_relative_error < 1:
            raise ValueError("max_relative_error must lie in (0, 1)")
        return float(
            max_relative_error * self.fx * self.baseline_m / self.disparity_sigma_px
        )

    def depth_resolution_at(self, depth_m: float) -> float:
        """Smallest depth change resolvable by one disparity quantum at range."""
        return float(depth_m ** 2 / (self.fx * self.baseline_m))

    def to_dict(self) -> dict[str, Any]:
        return {
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "baseline_m": self.baseline_m,
            "disparity_sigma_px": self.disparity_sigma_px,
            "max_reliable_range_m": round(self.max_reliable_range(), 3),
        }


@dataclass(slots=True)
class DepthResult:
    depth_m: np.ndarray            # NaN where disparity was invalid
    sigma_m: np.ndarray
    valid_mask: np.ndarray
    rig: StereoRig

    @property
    def coverage(self) -> float:
        return float(self.valid_mask.mean())

    @property
    def mean_relative_error(self) -> Optional[float]:
        if not self.valid_mask.any():
            return None
        z = self.depth_m[self.valid_mask]
        s = self.sigma_m[self.valid_mask]
        return float(np.mean(s / z))

    def statistics(self) -> dict[str, Any]:
        if not self.valid_mask.any():
            return {"coverage": 0.0, "valid_pixels": 0}
        z = self.depth_m[self.valid_mask]
        return {
            "coverage": round(self.coverage, 4),
            "valid_pixels": int(self.valid_mask.sum()),
            "min_depth_m": float(z.min()),
            "max_depth_m": float(z.max()),
            "median_depth_m": float(np.median(z)),
            "mean_relative_error": round(self.mean_relative_error or 0.0, 4),
            "max_reliable_range_m": round(self.rig.max_reliable_range(), 3),
        }


def disparity_to_depth(disparity_px: np.ndarray, rig: StereoRig) -> DepthResult:
    """Convert a disparity map to metric depth with per-pixel uncertainty."""
    d = np.asarray(disparity_px, dtype=np.float64)
    valid = np.isfinite(d) & (d >= rig.min_disparity_px)
    depth = np.full(d.shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth[valid] = rig.fx * rig.baseline_m / d[valid]
    sigma = np.full(d.shape, np.nan, dtype=np.float64)
    sigma[valid] = rig.depth_sigma(depth[valid])
    return DepthResult(depth_m=depth, sigma_m=sigma, valid_mask=valid, rig=rig)


def reproject_to_3d(
    depth_m: np.ndarray, rig: StereoRig, *, subsample: int = 1
) -> np.ndarray:
    """Back-project a depth map to a camera-frame point cloud, shape (N, 3).

    Uses the pinhole model::

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth map must be 2D")
    step = max(1, int(subsample))
    depth = depth[::step, ::step]
    rows, cols = depth.shape
    v, u = np.mgrid[0:rows, 0:cols].astype(np.float64)
    u = u * step
    v = v * step

    valid = np.isfinite(depth) & (depth > 0)
    z = depth[valid]
    x = (u[valid] - rig.cx) * z / rig.fx
    y = (v[valid] - rig.cy) * z / rig.fy
    return np.stack([x, y, z], axis=1)


def depth_traced(
    disparity: Traced[np.ndarray], rig: StereoRig
) -> Traced[DepthResult]:
    """Depth conversion that carries the disparity map's lineage forward."""
    result = disparity_to_depth(disparity.value, rig)
    # Confidence combines how much of the frame matched with how precise the
    # depths in it actually are.
    rel_err = result.mean_relative_error
    precision = 1.0 / (1.0 + 10.0 * rel_err) if rel_err is not None else 0.0
    confidence = float(max(0.01, min(1.0, 0.5 * result.coverage + 0.5 * precision)))
    return disparity.derive(
        result,
        algorithm="disparity_to_depth(Z = f*B/d, sigma_Z = Z^2*sigma_d/(f*B))",
        source=Source.CALCULATED,
        confidence=confidence,
        units="m",
        parameters=rig.to_dict(),
        label="depth_map",
    )
