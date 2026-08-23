"""
perceptos.servoing.ibvs
=======================

Image-Based Visual Servoing (IBVS).

Closes the loop from pixels straight to body velocity, without ever estimating
the target's pose.  For a point feature at normalised image coordinates
``(x, y)`` at depth ``Z``, the interaction matrix (image Jacobian) relating
feature velocity to camera spatial velocity ``v = [vx, vy, vz, wx, wy, wz]`` is

::

    L = [ -1/Z    0    x/Z    x*y     -(1+x^2)   y  ]
        [   0   -1/Z   y/Z   1+y^2      -x*y    -x  ]

The control law is the damped least-squares (Levenberg-Marquardt) inverse::

    v = -lambda * (L^T L + mu I)^-1 L^T * e

Damping matters here rather than using a plain pseudo-inverse: near a
singular configuration the pseudo-inverse produces enormous velocity commands,
which on real hardware means a violent lurch. The damping term bounds the
command at the cost of a slightly slower convergence — the right trade for
anything with mass.

Depth ``Z`` appears in ``L``, so IBVS needs a depth estimate. PerceptOS feeds
it from the stereo module; when depth is unavailable the controller falls back
to a constant nominal depth, which still converges for the translational
components but degrades gracefully rather than failing — and reports lower
confidence so the operator knows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from ..core.provenance import Source, Traced, traced

__all__ = ["CameraIntrinsics", "ServoCommand", "IBVSController"]


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def to_normalised(self, u: float, v: float) -> tuple[float, float]:
        """Pixels -> normalised image plane coordinates."""
        return ((u - self.cx) / self.fx, (v - self.cy) / self.fy)


@dataclass(frozen=True, slots=True)
class ServoCommand:
    """Body velocity command plus the error that produced it."""

    linear: tuple[float, float, float]     # m/s in camera frame
    angular: tuple[float, float, float]    # rad/s in camera frame
    pixel_error: tuple[float, float]
    error_norm_px: float
    converged: bool
    confidence: float
    saturated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "linear_mps": [round(v, 5) for v in self.linear],
            "angular_radps": [round(w, 5) for w in self.angular],
            "pixel_error": [round(e, 2) for e in self.pixel_error],
            "error_norm_px": round(self.error_norm_px, 2),
            "converged": self.converged,
            "saturated": self.saturated,
            "confidence": round(self.confidence, 3),
        }


class IBVSController:
    """Point-feature IBVS with damped least-squares control.

    Parameters
    ----------
    gain:
        Proportional gain ``lambda``.  Higher converges faster and overshoots.
    damping:
        Levenberg-Marquardt term ``mu``.  Raise it if commands spike.
    tolerance_px:
        Error norm below which the target is considered centred.
    max_linear / max_angular:
        Hard command limits.  Exceeding them sets ``saturated``, which the
        health panel surfaces — a permanently saturated controller means the
        gain is wrong, not that the robot is working hard.
    nominal_depth_m:
        Fallback depth when stereo has no valid measurement at the feature.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        *,
        gain: float = 0.5,
        damping: float = 0.01,
        tolerance_px: float = 5.0,
        max_linear: float = 0.5,
        max_angular: float = 0.8,
        nominal_depth_m: float = 2.0,
    ) -> None:
        if gain <= 0:
            raise ValueError("gain must be positive")
        if damping < 0:
            raise ValueError("damping must be non-negative")
        self.K = intrinsics
        self.gain = float(gain)
        self.damping = float(damping)
        self.tolerance_px = float(tolerance_px)
        self.max_linear = float(max_linear)
        self.max_angular = float(max_angular)
        self.nominal_depth_m = float(nominal_depth_m)

    # ------------------------------------------------------------- Jacobian

    @staticmethod
    def interaction_matrix(x: float, y: float, z: float) -> np.ndarray:
        """2x6 image Jacobian for one point feature at normalised (x, y), depth z."""
        if z <= 0:
            raise ValueError("depth must be positive")
        return np.array(
            [
                [-1.0 / z, 0.0, x / z, x * y, -(1.0 + x * x), y],
                [0.0, -1.0 / z, y / z, 1.0 + y * y, -x * y, -x],
            ],
            dtype=np.float64,
        )

    def stacked_interaction(
        self, features_px: Sequence[Sequence[float]], depths_m: Sequence[float]
    ) -> np.ndarray:
        """Stack the Jacobians of several features into a (2N, 6) matrix.

        Four or more features are needed to constrain all six DOF; with fewer,
        the damped inverse still returns the minimum-norm solution.
        """
        blocks = []
        for (u, v), z in zip(features_px, depths_m):
            x, y = self.K.to_normalised(float(u), float(v))
            blocks.append(self.interaction_matrix(x, y, max(float(z), 1e-3)))
        return np.vstack(blocks)

    # -------------------------------------------------------------- control

    def compute(
        self,
        current_px: Sequence[Sequence[float]],
        desired_px: Sequence[Sequence[float]],
        depths_m: Optional[Sequence[float]] = None,
    ) -> ServoCommand:
        """Velocity command driving ``current_px`` towards ``desired_px``."""
        current = np.asarray(current_px, dtype=np.float64).reshape(-1, 2)
        desired = np.asarray(desired_px, dtype=np.float64).reshape(-1, 2)
        if current.shape != desired.shape:
            raise ValueError(
                f"feature count mismatch: {current.shape} vs {desired.shape}"
            )
        n = current.shape[0]
        if n == 0:
            raise ValueError("at least one feature is required")

        if depths_m is None:
            depths = [self.nominal_depth_m] * n
            depth_known = False
        else:
            depths = [
                float(z) if math.isfinite(z) and z > 0 else self.nominal_depth_m
                for z in depths_m
            ]
            depth_known = all(math.isfinite(z) and z > 0 for z in depths_m)

        # Error in normalised coordinates — the space the Jacobian is defined in.
        error_norm_space = np.array(
            [
                self.K.to_normalised(*cur) [i] - self.K.to_normalised(*des)[i]
                for cur, des in zip(current, desired)
                for i in (0, 1)
            ],
            dtype=np.float64,
        )
        pixel_error = (current - desired).ravel()
        error_norm_px = float(np.linalg.norm(pixel_error))

        L = self.stacked_interaction(current, depths)
        LtL = L.T @ L
        damped = LtL + self.damping * np.eye(6)
        velocity = -self.gain * np.linalg.solve(damped, L.T @ error_norm_space)

        linear = velocity[:3]
        angular = velocity[3:]
        saturated = bool(
            np.linalg.norm(linear) > self.max_linear
            or np.linalg.norm(angular) > self.max_angular
        )
        linear = _clip_norm(linear, self.max_linear)
        angular = _clip_norm(angular, self.max_angular)

        # Confidence: geometric conditioning of L, feature count, depth quality.
        cond = float(np.linalg.cond(LtL)) if np.isfinite(LtL).all() else math.inf
        conditioning = 1.0 / (1.0 + math.log10(max(cond, 1.0)))
        feature_score = min(1.0, n / 4.0)
        depth_score = 1.0 if depth_known else 0.5
        confidence = float(
            max(0.0, min(1.0, 0.4 * conditioning + 0.3 * feature_score + 0.3 * depth_score))
        )

        return ServoCommand(
            linear=tuple(float(v) for v in linear),
            angular=tuple(float(w) for w in angular),
            pixel_error=(float(pixel_error[0]), float(pixel_error[1])),
            error_norm_px=error_norm_px,
            converged=error_norm_px <= self.tolerance_px,
            confidence=confidence,
            saturated=saturated,
        )

    def centre_target(
        self,
        target_px: Sequence[float],
        image_size: tuple[int, int],
        depth_m: Optional[float] = None,
    ) -> ServoCommand:
        """Convenience: drive a single detected target to the image centre."""
        width, height = image_size
        return self.compute(
            [target_px], [(width / 2.0, height / 2.0)],
            [depth_m] if depth_m is not None else None,
        )

    def traced_command(
        self, command: ServoCommand, *, parents: Sequence[Traced[Any]] = ()
    ) -> Traced[ServoCommand]:
        return traced(
            command,
            source=Source.CALCULATED,
            algorithm="IBVSController(damped least-squares interaction matrix)",
            confidence=command.confidence,
            units="m/s, rad/s",
            parameters={
                "gain": self.gain,
                "damping": self.damping,
                "max_linear": self.max_linear,
                "max_angular": self.max_angular,
            },
            label="servo_command",
            parents=parents,
        )


def _clip_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    """Scale a vector so its norm does not exceed ``limit``, preserving direction."""
    norm = float(np.linalg.norm(vector))
    if norm <= limit or norm == 0.0:
        return vector
    return vector * (limit / norm)
