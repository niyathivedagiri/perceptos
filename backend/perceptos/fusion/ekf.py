"""
perceptos.fusion.ekf
====================

Extended Kalman Filter for rover pose, with pluggable sensors.

State (6D)::

    x = [px, py, theta, v, omega, b_gyro]^T

A unicycle motion model with an explicit **gyro bias** state. The bias is not
decoration: an uncorrected MEMS gyro bias of 0.01 rad/s integrates to a
34-degree heading error in one minute, which on a rover means driving
confidently into terrain it believes is somewhere else.

Sensors are :class:`SensorModel` objects, each supplying its own measurement
function, Jacobian, noise and availability. Adding a sensor is a new class, not
a new branch in the filter.

Four things here are the difference between an EKF that works and one that
looks like it works:

**Angle wrapping in the innovation.** A heading measurement of +179 degrees
against a prediction of -179 degrees is a 2-degree error, not 358. Without
wrapping, every crossing of the +/-180 boundary injects a ~6.28 rad innovation
and the filter slews violently. This is the single most common EKF bug and it
only shows up on the specific traverse that happens to cross the discontinuity.

**Measurement gating.** Each update computes the normalised innovation squared
and rejects measurements beyond a chi-square bound. One GPS multipath spike or
one bad visual-odometry solution will otherwise drag the estimate and inflate
nothing, because the filter has no notion that a measurement can be wrong.

**Observability tracking.** Gyro bias is observable only when an absolute
heading reference is being fused. With gyro alone, bias and true yaw rate are
mathematically indistinguishable — the filter will happily report a tight
covariance on a bias it cannot actually determine. When no absolute heading
sensor has been seen recently, the bias state is frozen and the estimate says
so rather than quietly lying.

**Joseph-form covariance update**, so P stays symmetric positive-definite over
a long mission.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from ..core.provenance import Source, Traced, traced

__all__ = [
    "PoseEstimate",
    "SensorModel",
    "WheelOdometry",
    "Gyroscope",
    "VisualOdometrySensor",
    "GNSS",
    "StarTracker",
    "SensorUpdate",
    "PoseEKF",
    "wrap_angle",
    "CHI2_95",
]

#: Chi-square 95% critical values by degrees of freedom, for measurement gating.
CHI2_95: dict[int, float] = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070, 6: 12.592}

# State vector indices.
PX, PY, THETA, V, OMEGA, B_GYRO = range(6)
STATE_DIM = 6


def wrap_angle(angle: float | np.ndarray) -> Any:
    """Wrap to [-pi, pi). Applied to every angular innovation, without exception."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True, slots=True)
class PoseEstimate:
    """Filter state with its full uncertainty."""

    position: tuple[float, float]
    heading_rad: float
    speed_mps: float
    yaw_rate_radps: float
    gyro_bias_radps: float
    covariance: np.ndarray            # 6x6
    timestamp: float
    bias_observable: bool
    updates: int = 0
    rejections: int = 0

    @property
    def heading_deg(self) -> float:
        return float(math.degrees(self.heading_rad))

    @property
    def position_sigma(self) -> tuple[float, float]:
        return (float(math.sqrt(max(self.covariance[PX, PX], 0.0))),
                float(math.sqrt(max(self.covariance[PY, PY], 0.0))))

    @property
    def heading_sigma_deg(self) -> float:
        return float(math.degrees(math.sqrt(max(self.covariance[THETA, THETA], 0.0))))

    @property
    def position_error_ellipse(self) -> tuple[float, float, float]:
        """95% error ellipse as ``(semi_major, semi_minor, angle_rad)``.

        This is what the UI draws. A circle of radius sigma would understate a
        correlated position error, which is the normal case when a rover has
        driven in a straight line: along-track uncertainty grows far faster
        than cross-track.
        """
        block = self.covariance[:2, :2]
        values, vectors = np.linalg.eigh(block)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        scale = math.sqrt(CHI2_95[2])
        return (
            float(scale * math.sqrt(max(values[0], 0.0))),
            float(scale * math.sqrt(max(values[1], 0.0))),
            float(math.atan2(vectors[1, 0], vectors[0, 0])),
        )

    @property
    def confidence(self) -> float:
        """Scalar confidence in [0, 1] from positional and heading uncertainty."""
        sigma_pos = math.hypot(*self.position_sigma)
        sigma_head = math.radians(self.heading_sigma_deg)
        positional = 1.0 / (1.0 + sigma_pos)
        angular = 1.0 / (1.0 + 5.0 * sigma_head)
        return float(max(0.0, min(1.0, 0.6 * positional + 0.4 * angular)))

    def to_dict(self) -> dict[str, Any]:
        major, minor, angle = self.position_error_ellipse
        return {
            "position": [round(v, 4) for v in self.position],
            "heading_deg": round(self.heading_deg, 3),
            "speed_mps": round(self.speed_mps, 4),
            "yaw_rate_radps": round(self.yaw_rate_radps, 5),
            "gyro_bias_radps": round(self.gyro_bias_radps, 6),
            "position_sigma_m": [round(v, 4) for v in self.position_sigma],
            "heading_sigma_deg": round(self.heading_sigma_deg, 3),
            "error_ellipse_95": {"semi_major_m": round(major, 4),
                                 "semi_minor_m": round(minor, 4),
                                 "angle_rad": round(angle, 4)},
            "bias_observable": self.bias_observable,
            "confidence": round(self.confidence, 4),
            "updates": self.updates,
            "rejections": self.rejections,
            "timestamp": self.timestamp,
        }


class SensorModel(abc.ABC):
    """One measurement source.

    Subclasses declare what they measure, the Jacobian of that measurement with
    respect to the state, their noise, and where they physically work. The
    filter itself contains no sensor-specific logic.
    """

    name: str = "sensor"
    #: Indices of measurement components that are angles and need wrapping.
    angle_indices: tuple[int, ...] = ()
    #: Environments where this sensor exists at all.
    available_in: frozenset[str] = frozenset({"mars", "moon", "earth",
                                              "warehouse", "factory"})
    #: True if this sensor observes heading absolutely (makes gyro bias observable).
    provides_absolute_heading: bool = False

    #: Measurements accepted unconditionally before gating begins. A gate is a
    #: statement about a *converged* filter: applied to an uninitialised one it
    #: rejects exactly the large, correct corrections that would converge it,
    #: and the filter then sits forever at its initial guess rejecting every
    #: measurement as an outlier. This is a real and easily-missed deadlock.
    warmup_updates: int = 5

    def __init__(self, *, noise_std: Sequence[float] | float,
                 gate_chi2: Optional[float] = None,
                 warmup_updates: Optional[int] = None) -> None:
        std = np.atleast_1d(np.asarray(noise_std, dtype=np.float64))
        if np.any(std <= 0):
            raise ValueError(f"{self.name}: noise_std must be positive")
        self.R = np.diag(std ** 2)
        self.dim = int(std.size)
        self.gate_chi2 = float(gate_chi2) if gate_chi2 is not None else CHI2_95.get(
            self.dim, 12.592
        )
        if warmup_updates is not None:
            self.warmup_updates = int(warmup_updates)
        self.accepted = 0
        self.rejected = 0

    @abc.abstractmethod
    def predict_measurement(self, x: np.ndarray) -> np.ndarray:
        """h(x): what this sensor should read given the state."""

    @abc.abstractmethod
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """H = dh/dx, shape (dim, 6)."""

    def innovation(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """z - h(x), with angular components wrapped to [-pi, pi)."""
        y = np.asarray(z, dtype=np.float64).reshape(-1) - self.predict_measurement(x)
        for i in self.angle_indices:
            y[i] = float(wrap_angle(y[i]))
        return y

    @property
    def in_warmup(self) -> bool:
        return self.accepted < self.warmup_updates

    def is_available(self, environment: Optional[str]) -> bool:
        return environment is None or environment in self.available_in

    def statistics(self) -> dict[str, Any]:
        total = self.accepted + self.rejected
        return {
            "name": self.name,
            "dim": self.dim,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "rejection_rate": round(self.rejected / total, 4) if total else 0.0,
            "gate_chi2": self.gate_chi2,
            "in_warmup": self.in_warmup,
            "noise_std": [round(float(v), 6) for v in np.sqrt(np.diag(self.R))],
        }


class WheelOdometry(SensorModel):
    """Wheel encoders: forward speed only.

    Encoders measure **wheel** speed, which equals **ground** speed only when
    the wheels are not slipping. On regolith they routinely do.

    This matters more than it first appears. Fusing wheel odometry and visual
    odometry as if both measured ground speed is wrong whenever there is slip:
    the two sensors then genuinely disagree, and the filter has no way to know
    that one of them is right. What happens in practice is that whichever
    updates first captures the state and the gate rejects the other as an
    outlier — so a rover digging itself into a sand trap reports a healthy
    filter and a confident, wrong speed.

    The correct architecture is that visual odometry measures ground speed and
    the encoders are down-weighted in proportion to the measured slip.
    :meth:`with_slip` does that: at 50% slip the encoder reading is treated as
    carrying roughly ten times its nominal standard deviation, so it still
    contributes when traction is good and quietly stops dominating when it is
    not.
    """

    name = "wheel_odometry"

    def __init__(self, *, speed_noise_std: float = 0.02, **kwargs: Any) -> None:
        super().__init__(noise_std=[speed_noise_std], **kwargs)
        self._base_R = self.R.copy()
        self.slip_ratio = 0.0

    def with_slip(self, slip_ratio: float) -> "WheelOdometry":
        """Inflate R in proportion to measured wheel slip."""
        self.slip_ratio = float(max(0.0, min(1.0, abs(slip_ratio))))
        # Standard deviation scales as 1 + 20*slip: negligible at 1% slip,
        # a factor of 11 at 50%, a factor of 21 at total loss of traction.
        self.R = self._base_R * (1.0 + 20.0 * self.slip_ratio) ** 2
        return self

    def predict_measurement(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[V]])

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        H = np.zeros((1, STATE_DIM))
        H[0, V] = 1.0
        return H

    def statistics(self) -> dict[str, Any]:
        return {**super().statistics(), "slip_ratio": round(self.slip_ratio, 4)}


class Gyroscope(SensorModel):
    """MEMS gyro: measures true yaw rate **plus** an unknown slowly-varying bias.

    ``h(x) = omega + b_gyro``. Both terms are states, so from this measurement
    alone they are unidentifiable — the observability check in the filter exists
    precisely because of this.
    """

    name = "gyroscope"

    def __init__(self, *, rate_noise_std: float = 0.005, **kwargs: Any) -> None:
        super().__init__(noise_std=[rate_noise_std], **kwargs)

    def predict_measurement(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[OMEGA] + x[B_GYRO]])

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        H = np.zeros((1, STATE_DIM))
        H[0, OMEGA] = 1.0
        H[0, B_GYRO] = 1.0
        return H


class VisualOdometrySensor(SensorModel):
    """Visual odometry: forward speed and yaw rate from image motion.

    Feeds from :class:`~perceptos.mars.odometry.VisualOdometry`. Its noise is
    not constant — a low-texture scene or a planar-degenerate frame gives a far
    worse solution — so :meth:`with_confidence` scales R by the reported
    solution quality. A VO measurement with 0.2 confidence entering the filter
    with the same weight as a 0.95 one is how a good filter gets ruined by one
    bad frame.
    """

    name = "visual_odometry"
    available_in = frozenset({"mars", "moon", "earth", "warehouse", "factory"})

    def __init__(self, *, speed_noise_std: float = 0.03,
                 rate_noise_std: float = 0.01, **kwargs: Any) -> None:
        super().__init__(noise_std=[speed_noise_std, rate_noise_std], **kwargs)
        self._base_R = self.R.copy()

    def with_confidence(self, confidence: float) -> "VisualOdometrySensor":
        """Inflate R as solution quality drops. Returns self for chaining."""
        quality = max(0.05, min(1.0, float(confidence)))
        # Variance scales as 1/quality^2: a half-confidence solution is treated
        # as twice the standard deviation, not half the weight.
        self.R = self._base_R / (quality ** 2)
        return self

    def predict_measurement(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[V], x[OMEGA]])

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        H = np.zeros((2, STATE_DIM))
        H[0, V] = 1.0
        H[1, OMEGA] = 1.0
        return H


class GNSS(SensorModel):
    """Satellite positioning. Earth only — there is no GPS constellation at Mars."""

    name = "gnss"
    available_in = frozenset({"earth"})

    def __init__(self, *, position_noise_std: float = 1.5, **kwargs: Any) -> None:
        super().__init__(noise_std=[position_noise_std, position_noise_std], **kwargs)

    def predict_measurement(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[PX], x[PY]])

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        H = np.zeros((2, STATE_DIM))
        H[0, PX] = 1.0
        H[1, PY] = 1.0
        return H


class StarTracker(SensorModel):
    """Absolute heading from star field matching.

    The planetary analogue of a compass — Mars has no usable global magnetic
    field, so a magnetometer is not an option there. This is the sensor that
    makes gyro bias observable, and its measurement is an angle, so the
    innovation must be wrapped.
    """

    name = "star_tracker"
    angle_indices = (0,)
    available_in = frozenset({"mars", "moon"})
    provides_absolute_heading = True

    def __init__(self, *, heading_noise_std_rad: float = 0.002, **kwargs: Any) -> None:
        super().__init__(noise_std=[heading_noise_std_rad], **kwargs)

    def predict_measurement(self, x: np.ndarray) -> np.ndarray:
        return np.array([wrap_angle(x[THETA])])

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        H = np.zeros((1, STATE_DIM))
        H[0, THETA] = 1.0
        return H


@dataclass(slots=True)
class SensorUpdate:
    """Outcome of one measurement, including why it was rejected if it was."""

    sensor: str
    accepted: bool
    nis: float
    gate: float
    innovation: list[float]
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor": self.sensor, "accepted": self.accepted,
            "nis": round(self.nis, 4), "gate": self.gate,
            "innovation": [round(v, 6) for v in self.innovation],
            "reason": self.reason,
        }


class PoseEKF:
    """Extended Kalman Filter over a unicycle model with gyro bias.

    Parameters
    ----------
    environment:
        Environment name. Sensors declare where they exist, so registering a
        GNSS receiver for a Mars mission is rejected rather than silently
        contributing measurements from a constellation that is not there.
    bias_observability_timeout_s:
        How long after the last absolute-heading fix the gyro bias remains
        trusted. Beyond it the bias state is frozen.
    """

    def __init__(self, *, environment: Optional[str] = None,
                 initial_pose: Sequence[float] = (0.0, 0.0, 0.0),
                 speed_process_std: float = 0.05,
                 yaw_rate_process_std: float = 0.02,
                 bias_random_walk_std: float = 1e-4,
                 initial_position_std: float = 1.0,
                 initial_heading_std: float = 0.1,
                 bias_observability_timeout_s: float = 60.0) -> None:
        self.environment = environment
        self.x = np.zeros(STATE_DIM)
        self.x[PX], self.x[PY], self.x[THETA] = (float(v) for v in initial_pose[:3])

        self.P = np.diag([
            initial_position_std ** 2, initial_position_std ** 2,
            initial_heading_std ** 2, 0.5 ** 2, 0.2 ** 2, 0.01 ** 2,
        ])
        self.speed_process_std = float(speed_process_std)
        self.yaw_rate_process_std = float(yaw_rate_process_std)
        self.bias_random_walk_std = float(bias_random_walk_std)
        self.bias_timeout_s = float(bias_observability_timeout_s)

        self.sensors: dict[str, SensorModel] = {}
        self.timestamp = 0.0
        self.updates = 0
        self.rejections = 0
        self.history: list[SensorUpdate] = []
        self._last_absolute_heading_t: Optional[float] = None

    # ------------------------------------------------------------- sensors

    def register(self, sensor: SensorModel) -> "PoseEKF":
        """Add a sensor, refusing ones that cannot exist in this environment."""
        if not sensor.is_available(self.environment):
            raise ValueError(
                f"{sensor.name!r} is not available in {self.environment!r}; "
                f"it exists in {sorted(sensor.available_in)}. "
                "There is no satellite navigation constellation at Mars."
            )
        self.sensors[sensor.name] = sensor
        return self

    @property
    def has_absolute_heading(self) -> bool:
        return any(s.provides_absolute_heading for s in self.sensors.values())

    @property
    def bias_observable(self) -> bool:
        """Is the gyro bias currently determinable?

        Only when an absolute heading reference has been fused recently. With
        gyro alone, ``omega`` and ``b_gyro`` enter the measurement identically
        and no amount of data separates them.
        """
        if not self.has_absolute_heading:
            return False
        if self._last_absolute_heading_t is None:
            return False
        return (self.timestamp - self._last_absolute_heading_t) <= self.bias_timeout_s

    # ------------------------------------------------------------- prediction

    def transition(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Unicycle propagation."""
        out = x.copy()
        out[PX] = x[PX] + x[V] * math.cos(x[THETA]) * dt
        out[PY] = x[PY] + x[V] * math.sin(x[THETA]) * dt
        out[THETA] = wrap_angle(x[THETA] + x[OMEGA] * dt)
        return out

    def transition_jacobian(self, x: np.ndarray, dt: float) -> np.ndarray:
        F = np.eye(STATE_DIM)
        F[PX, THETA] = -x[V] * math.sin(x[THETA]) * dt
        F[PX, V] = math.cos(x[THETA]) * dt
        F[PY, THETA] = x[V] * math.cos(x[THETA]) * dt
        F[PY, V] = math.sin(x[THETA]) * dt
        F[THETA, OMEGA] = dt
        return F

    def process_noise(self, dt: float) -> np.ndarray:
        Q = np.zeros((STATE_DIM, STATE_DIM))
        Q[V, V] = (self.speed_process_std * dt) ** 2
        Q[OMEGA, OMEGA] = (self.yaw_rate_process_std * dt) ** 2
        # Bias is a random walk. When unobservable its growth is suppressed:
        # letting it wander freely produces a confident, wrong bias estimate
        # that the heading integration then follows.
        walk = self.bias_random_walk_std if self.bias_observable else 0.0
        Q[B_GYRO, B_GYRO] = (walk * dt) ** 2
        # Small position/heading noise keeps P positive-definite when the rover
        # is stationary and no other term contributes.
        Q[PX, PX] = Q[PY, PY] = (0.001 * dt) ** 2
        Q[THETA, THETA] = (0.001 * dt) ** 2
        return Q

    def predict(self, dt: float) -> PoseEstimate:
        if dt <= 0:
            raise ValueError("dt must be positive")
        F = self.transition_jacobian(self.x, dt)
        self.x = self.transition(self.x, dt)
        self.P = F @ self.P @ F.T + self.process_noise(dt)
        self.timestamp += dt
        return self.estimate()

    # ---------------------------------------------------------------- update

    def update(self, sensor_name: str, measurement: Sequence[float] | float,
               *, confidence: Optional[float] = None,
               slip_ratio: Optional[float] = None) -> SensorUpdate:
        """Fuse one measurement, gating it against a chi-square bound."""
        sensor = self.sensors.get(sensor_name)
        if sensor is None:
            raise KeyError(
                f"sensor {sensor_name!r} not registered; "
                f"registered: {sorted(self.sensors)}"
            )
        if confidence is not None and hasattr(sensor, "with_confidence"):
            sensor.with_confidence(confidence)
        if slip_ratio is not None and hasattr(sensor, "with_slip"):
            sensor.with_slip(slip_ratio)

        z = np.atleast_1d(np.asarray(measurement, dtype=np.float64))
        if z.size != sensor.dim:
            raise ValueError(
                f"{sensor_name!r} expects {sensor.dim} value(s), got {z.size}"
            )

        y = sensor.innovation(self.x, z)
        H = sensor.jacobian(self.x)
        S = H @ self.P @ H.T + sensor.R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            result = SensorUpdate(sensor_name, False, float("inf"), sensor.gate_chi2,
                                  y.tolist(), "singular innovation covariance")
            sensor.rejected += 1
            self.rejections += 1
            self.history.append(result)
            return result

        nis = float(y @ S_inv @ y)
        if nis > sensor.gate_chi2 and not sensor.in_warmup:
            sensor.rejected += 1
            self.rejections += 1
            result = SensorUpdate(
                sensor_name, False, nis, sensor.gate_chi2, y.tolist(),
                f"NIS {nis:.2f} exceeds the {sensor.gate_chi2} chi-square gate; "
                "measurement treated as an outlier",
            )
            self.history.append(result)
            return result

        was_warmup = sensor.in_warmup
        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        self.x[THETA] = wrap_angle(self.x[THETA])

        # Joseph form: keeps P symmetric positive-definite over long missions.
        I_KH = np.eye(STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ sensor.R @ K.T

        sensor.accepted += 1
        self.updates += 1
        if sensor.provides_absolute_heading:
            self._last_absolute_heading_t = self.timestamp

        result = SensorUpdate(
            sensor_name, True, nis, sensor.gate_chi2, y.tolist(),
            "accepted during warm-up (gate not yet applied)" if was_warmup else "",
        )
        self.history.append(result)
        return result

    def update_many(self, measurements: Mapping[str, Any]) -> list[SensorUpdate]:
        """Fuse several sensors at one timestep, in registration order."""
        return [self.update(name, value) for name, value in measurements.items()
                if name in self.sensors]

    # --------------------------------------------------------------- readout

    def estimate(self) -> PoseEstimate:
        return PoseEstimate(
            position=(float(self.x[PX]), float(self.x[PY])),
            heading_rad=float(self.x[THETA]),
            speed_mps=float(self.x[V]),
            yaw_rate_radps=float(self.x[OMEGA]),
            gyro_bias_radps=float(self.x[B_GYRO]),
            covariance=self.P.copy(),
            timestamp=self.timestamp,
            bias_observable=self.bias_observable,
            updates=self.updates,
            rejections=self.rejections,
        )

    def traced_estimate(self, *, parents: Sequence[Traced[Any]] = ()
                        ) -> Traced[PoseEstimate]:
        estimate = self.estimate()
        return traced(
            estimate,
            source=Source.ESTIMATED,
            algorithm="PoseEKF(unicycle + gyro bias, Joseph form, chi-square gated)",
            confidence=estimate.confidence,
            units="m, rad",
            label="fused_pose",
            parameters={
                "sensors": sorted(self.sensors),
                "environment": self.environment,
                "updates": self.updates,
                "rejections": self.rejections,
                "bias_observable": estimate.bias_observable,
            },
            parents=parents,
        )

    def diagnostics(self) -> dict[str, Any]:
        """Filter health, judged on NIS **and** rejection rate together.

        A well-tuned EKF has a mean NIS near its measurement dimension. But
        mean NIS over accepted measurements alone is a *censored* statistic:
        the gate has already discarded the high-NIS samples, so a filter whose
        R is far too small rejects almost everything and the survivors look
        beautifully consistent. In the extreme the sensor disappears from the
        statistics entirely, having had every measurement thrown away, and the
        filter reports perfect health while flying blind.

        Rejection rate is therefore part of the verdict, not a separate number
        for someone to notice.
        """
        scored = [u for u in self.history
                  if u.accepted and "warm-up" not in u.reason]
        by_sensor: dict[str, list[float]] = {}
        for update in scored:
            by_sensor.setdefault(update.sensor, []).append(update.nis)

        consistency = {}
        for name, sensor in self.sensors.items():
            values = by_sensor.get(name, [])
            total = sensor.accepted + sensor.rejected
            rejection_rate = sensor.rejected / total if total else 0.0
            expected = sensor.dim
            mean_nis = float(np.mean(values)) if values else None

            if total == 0:
                verdict = "no measurements"
            elif not values:
                verdict = "optimistic (R too small - every measurement rejected)"
            elif rejection_rate > 0.20:
                verdict = (f"optimistic (R too small - gate rejecting "
                           f"{rejection_rate:.0%} of measurements)")
            elif mean_nis > 2.0 * expected:
                verdict = "optimistic (R too small)"
            elif mean_nis < 0.4 * expected and rejection_rate < 0.05:
                verdict = "conservative (R too large)"
            else:
                verdict = "consistent"

            consistency[name] = {
                "mean_nis": round(mean_nis, 3) if mean_nis is not None else None,
                "expected_nis": expected,
                "rejection_rate": round(rejection_rate, 4),
                "samples": len(values),
                "verdict": verdict,
            }
        return {
            "environment": self.environment,
            "updates": self.updates,
            "rejections": self.rejections,
            "bias_observable": self.bias_observable,
            "has_absolute_heading": self.has_absolute_heading,
            "sensors": {n: s.statistics() for n, s in self.sensors.items()},
            "consistency": consistency,
        }
