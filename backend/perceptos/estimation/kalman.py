"""
perceptos.estimation.kalman
===========================

Constant-acceleration Kalman filter for 2D target state estimation.

State vector (6D)::

    x = [px, py, vx, vy, ax, ay]^T

Measurement is position only (a detection centroid), so the filter recovers
velocity and acceleration purely from the position history — which is what
lets the tracker survive occlusion and lets the mission planner extrapolate
where a target *will* be.

Implemented directly in NumPy rather than via FilterPy so that:

* the process-noise model is explicit and reviewable,
* uncertainty is exported as a covariance the UI can draw as an error ellipse,
* it runs anywhere, including a stripped Apple Silicon environment.

The process noise uses the continuous white-noise-jerk model discretised over
``dt``, which is the standard choice for a manoeuvring target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from ..core.provenance import Source, Traced, traced

__all__ = ["KalmanState", "ConstantAccelerationKF"]


@dataclass(frozen=True, slots=True)
class KalmanState:
    """Snapshot of the filter, in units the rest of the system understands."""

    position: tuple[float, float]
    velocity: tuple[float, float]
    acceleration: tuple[float, float]
    covariance: np.ndarray
    timestamp: float
    innovation: Optional[tuple[float, float]] = None
    nis: Optional[float] = None  # normalised innovation squared — track health

    @property
    def speed(self) -> float:
        return float(math.hypot(*self.velocity))

    @property
    def heading_deg(self) -> float:
        return float(math.degrees(math.atan2(self.velocity[1], self.velocity[0])))

    @property
    def position_sigma(self) -> tuple[float, float]:
        return (
            float(math.sqrt(max(self.covariance[0, 0], 0.0))),
            float(math.sqrt(max(self.covariance[1, 1], 0.0))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": list(self.position),
            "velocity": list(self.velocity),
            "acceleration": list(self.acceleration),
            "speed": self.speed,
            "heading_deg": self.heading_deg,
            "position_sigma": list(self.position_sigma),
            "nis": self.nis,
            "timestamp": self.timestamp,
        }


class ConstantAccelerationKF:
    """6-state constant-acceleration Kalman filter with position measurements.

    Parameters
    ----------
    dt:
        Nominal timestep in seconds.  Each :meth:`predict` may override it,
        which matters because camera frame intervals jitter.
    process_noise_std:
        Standard deviation of the unmodelled jerk, in m/s^3 (or px/s^3 if the
        filter is run in image space).  Larger = trusts the measurements more.
    measurement_noise_std:
        Standard deviation of the position measurement, same units as position.
    """

    STATE_DIM = 6
    MEAS_DIM = 2

    def __init__(
        self,
        *,
        dt: float = 1.0 / 30.0,
        process_noise_std: float = 1.0,
        measurement_noise_std: float = 2.0,
        initial_position: Sequence[float] = (0.0, 0.0),
        initial_uncertainty: float = 500.0,
    ) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.q_std = float(process_noise_std)
        self.r_std = float(measurement_noise_std)

        self.x = np.zeros((self.STATE_DIM, 1), dtype=np.float64)
        self.x[0, 0], self.x[1, 0] = float(initial_position[0]), float(initial_position[1])

        # Position is known roughly (first detection); velocity/accel are not.
        self.P = np.eye(self.STATE_DIM) * float(initial_uncertainty)
        self.P[0, 0] = self.P[1, 1] = float(measurement_noise_std) ** 2

        self.H = np.zeros((self.MEAS_DIM, self.STATE_DIM))
        self.H[0, 0] = self.H[1, 1] = 1.0
        self.R = np.eye(self.MEAS_DIM) * float(measurement_noise_std) ** 2

        self.timestamp = 0.0
        self.updates = 0
        self.misses = 0
        self._last_innovation: Optional[np.ndarray] = None
        self._last_nis: Optional[float] = None

    # ------------------------------------------------------------- matrices

    def transition_matrix(self, dt: float) -> np.ndarray:
        """Newtonian constant-acceleration transition."""
        F = np.eye(self.STATE_DIM)
        half_dt2 = 0.5 * dt * dt
        F[0, 2] = F[1, 3] = dt
        F[2, 4] = F[3, 5] = dt
        F[0, 4] = F[1, 5] = half_dt2
        return F

    def process_noise(self, dt: float) -> np.ndarray:
        """Discretised white-noise-jerk process covariance.

        Per axis the continuous model integrates to::

            [ dt^5/20  dt^4/8  dt^3/6 ]
            [ dt^4/8   dt^3/3  dt^2/2 ]
            [ dt^3/6   dt^2/2  dt     ] * q^2
        """
        q2 = self.q_std ** 2
        d2, d3, d4, d5 = dt ** 2, dt ** 3, dt ** 4, dt ** 5
        block = np.array(
            [
                [d5 / 20.0, d4 / 8.0, d3 / 6.0],
                [d4 / 8.0, d3 / 3.0, d2 / 2.0],
                [d3 / 6.0, d2 / 2.0, dt],
            ]
        ) * q2
        Q = np.zeros((self.STATE_DIM, self.STATE_DIM))
        # State order is [px,py,vx,vy,ax,ay], so each axis touches indices i,i+2,i+4.
        for axis in (0, 1):
            idx = [axis, axis + 2, axis + 4]
            for r, i in enumerate(idx):
                for c, j in enumerate(idx):
                    Q[i, j] = block[r, c]
        return Q

    # -------------------------------------------------------------- filtering

    def predict(self, dt: Optional[float] = None) -> KalmanState:
        """Propagate the state forward.  Call once per frame, detection or not."""
        step = float(dt if dt is not None else self.dt)
        if step <= 0:
            raise ValueError("dt must be positive")
        F = self.transition_matrix(step)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.process_noise(step)
        self.timestamp += step
        return self.state()

    def update(self, measurement: Sequence[float]) -> KalmanState:
        """Fold in a position measurement (a detection centroid)."""
        z = np.asarray(measurement, dtype=np.float64).reshape(self.MEAS_DIM, 1)
        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + self.R      # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain

        self.x = self.x + K @ y
        # Joseph form keeps P symmetric positive-definite over long missions.
        I_KH = np.eye(self.STATE_DIM) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        self._last_innovation = y
        self._last_nis = float((y.T @ np.linalg.inv(S) @ y).item())
        self.updates += 1
        return self.state()

    def mark_missed(self) -> None:
        """Record that no detection was associated this frame."""
        self.misses += 1

    # ------------------------------------------------------------- prediction

    def predict_future(self, horizon_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Where will the target be in ``horizon_s`` seconds?

        Returns ``(position, covariance)`` **without** mutating the filter, so
        the mission planner can query multiple horizons for one frame.
        """
        F = self.transition_matrix(float(horizon_s))
        x_future = F @ self.x
        P_future = F @ self.P @ F.T + self.process_noise(float(horizon_s))
        return x_future[:2, 0].copy(), P_future[:2, :2].copy()

    def trajectory(self, horizon_s: float, steps: int = 10) -> list[tuple[float, float]]:
        """Sampled predicted path — drawn as the dashed line on the tactical view."""
        if steps < 1:
            raise ValueError("steps must be >= 1")
        return [
            tuple(self.predict_future(horizon_s * (i + 1) / steps)[0].tolist())
            for i in range(steps)
        ]

    # --------------------------------------------------------------- accessors

    @property
    def track_confidence(self) -> float:
        """Confidence in [0,1] from positional uncertainty and hit ratio.

        Two independent failure modes are combined: a covariance that has grown
        because the target has not been seen, and a poor association history.
        """
        sigma = math.sqrt(max(self.P[0, 0], 0.0) + max(self.P[1, 1], 0.0))
        geometric = 1.0 / (1.0 + sigma / max(self.r_std, 1e-6))
        total = self.updates + self.misses
        hit_ratio = self.updates / total if total else 0.0
        return float(max(0.0, min(1.0, 0.5 * geometric + 0.5 * hit_ratio)))

    def state(self) -> KalmanState:
        return KalmanState(
            position=(float(self.x[0, 0]), float(self.x[1, 0])),
            velocity=(float(self.x[2, 0]), float(self.x[3, 0])),
            acceleration=(float(self.x[4, 0]), float(self.x[5, 0])),
            covariance=self.P[:2, :2].copy(),
            timestamp=self.timestamp,
            innovation=(
                (float(self._last_innovation[0, 0]), float(self._last_innovation[1, 0]))
                if self._last_innovation is not None
                else None
            ),
            nis=self._last_nis,
        )

    def traced_state(self, *, parents: Sequence[Traced[Any]] = ()) -> Traced[KalmanState]:
        """Export the state as a traced value for the lineage graph."""
        return traced(
            self.state(),
            source=Source.ESTIMATED,
            algorithm="ConstantAccelerationKF(white-noise-jerk)",
            confidence=self.track_confidence,
            units="m",
            parameters={
                "dt": self.dt,
                "process_noise_std": self.q_std,
                "measurement_noise_std": self.r_std,
                "updates": self.updates,
                "misses": self.misses,
            },
            label="kalman_state",
            parents=parents,
        )
