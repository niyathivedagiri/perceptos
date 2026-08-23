"""
perceptos.mars.slip
===================

Wheel slip estimation by disagreement between wheel odometry and visual
odometry.

Wheel encoders measure how far the wheels *turned*.  Visual odometry measures
how far the rover actually *moved*.  On loose regolith these disagree, and the
difference is slip.  This is not a theoretical concern: Spirit was permanently
immobilised in soft soil at Troy in 2009, and slip monitoring is why
Perseverance's autonav backs off before that happens.

Definitions used here follow the standard terramechanics convention:

    slip ratio   i = (v_wheel - v_actual) / v_wheel      for v_wheel > v_actual
    skid ratio   i = (v_wheel - v_actual) / v_actual     for v_actual > v_wheel

so ``i`` is positive when the wheels spin faster than the rover advances
(sinkage, climbing) and negative when the rover outruns its wheels (descending,
sliding downslope).  ``i = 1.0`` means total loss of traction.

The estimator keeps a short history and flags an *embedding* condition when
slip stays above the threshold for a sustained window — a single high-slip
sample is normal when crossing a rock, whereas sustained high slip means the
wheels are digging in.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ..core.provenance import Source, Traced, traced

__all__ = ["SlipSample", "SlipEstimate", "WheelSlipEstimator"]


@dataclass(frozen=True, slots=True)
class SlipSample:
    commanded_m: float
    measured_m: float
    slip_ratio: float
    timestamp: float


@dataclass(frozen=True, slots=True)
class SlipEstimate:
    """Current slip condition and the recommendation that follows from it."""

    slip_ratio: float
    mean_slip_ratio: float
    is_slipping: bool
    is_embedding: bool
    confidence: float
    samples: int
    recommendation: str
    slope_deg: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "slip_ratio": round(self.slip_ratio, 4),
            "mean_slip_ratio": round(self.mean_slip_ratio, 4),
            "slip_percent": round(self.slip_ratio * 100.0, 2),
            "is_slipping": self.is_slipping,
            "is_embedding": self.is_embedding,
            "confidence": round(self.confidence, 3),
            "samples": self.samples,
            "slope_deg": self.slope_deg,
            "recommendation": self.recommendation,
        }


class WheelSlipEstimator:
    """Rolling slip estimator with an embedding (sinkage) alarm.

    Parameters
    ----------
    slip_threshold:
        Ratio above which a single step counts as slipping.
    embedding_threshold:
        Sustained mean ratio that triggers the stop-and-reassess response.
    window:
        Number of recent steps in the rolling mean.
    min_commanded_m:
        Steps shorter than this are ignored — dividing by a near-zero commanded
        distance produces meaningless ratios, the classic way slip estimators
        emit spurious 900% alarms while the rover is stationary.
    """

    def __init__(
        self,
        *,
        slip_threshold: float = 0.30,
        embedding_threshold: float = 0.60,
        window: int = 10,
        min_commanded_m: float = 0.01,
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.slip_threshold = float(slip_threshold)
        self.embedding_threshold = float(embedding_threshold)
        self.min_commanded_m = float(min_commanded_m)
        self.history: deque[SlipSample] = deque(maxlen=int(window))

    @staticmethod
    def slip_ratio(commanded_m: float, measured_m: float) -> float:
        """Signed slip/skid ratio, clamped to [-1, 1]."""
        if commanded_m == 0.0 and measured_m == 0.0:
            return 0.0
        if commanded_m > measured_m:
            denominator = abs(commanded_m)
        else:
            denominator = abs(measured_m)
        if denominator < 1e-9:
            return 0.0
        return float(max(-1.0, min(1.0, (commanded_m - measured_m) / denominator)))

    def update(
        self,
        commanded_m: float,
        measured_m: float,
        *,
        timestamp: float = 0.0,
        slope_deg: Optional[float] = None,
        vo_confidence: float = 1.0,
    ) -> SlipEstimate:
        """Fold in one drive step.

        ``vo_confidence`` is the visual odometry solution quality.  When VO is
        weak (featureless terrain, motion blur) the slip number inherits that
        weakness, so the operator does not act on a confident-looking figure
        derived from an unreliable measurement.
        """
        commanded_m, measured_m = float(commanded_m), float(measured_m)
        if abs(commanded_m) < self.min_commanded_m:
            return self._estimate(0.0, slope_deg, vo_confidence, stationary=True)

        ratio = self.slip_ratio(commanded_m, measured_m)
        self.history.append(SlipSample(commanded_m, measured_m, ratio, timestamp))
        return self._estimate(ratio, slope_deg, vo_confidence)

    def _estimate(
        self,
        ratio: float,
        slope_deg: Optional[float],
        vo_confidence: float,
        *,
        stationary: bool = False,
    ) -> SlipEstimate:
        samples = list(self.history)
        mean_ratio = (
            sum(s.slip_ratio for s in samples) / len(samples) if samples else 0.0
        )
        is_slipping = ratio >= self.slip_threshold
        # Embedding requires a *sustained* signal across a full window.
        is_embedding = (
            len(samples) == self.history.maxlen
            and mean_ratio >= self.embedding_threshold
        )
        # Confidence degrades with sample scarcity and with weak VO.
        sample_conf = min(1.0, len(samples) / max(1, self.history.maxlen or 1))
        confidence = float(
            max(0.0, min(1.0, 0.5 * sample_conf + 0.5 * max(0.0, min(1.0, vo_confidence))))
        )
        return SlipEstimate(
            slip_ratio=ratio,
            mean_slip_ratio=mean_ratio,
            is_slipping=is_slipping,
            is_embedding=is_embedding,
            confidence=confidence,
            samples=len(samples),
            slope_deg=slope_deg,
            recommendation=self._recommend(
                ratio, mean_ratio, is_embedding, slope_deg, stationary
            ),
        )

    def _recommend(
        self,
        ratio: float,
        mean_ratio: float,
        is_embedding: bool,
        slope_deg: Optional[float],
        stationary: bool,
    ) -> str:
        if stationary:
            return "Stationary — slip not evaluated."
        if is_embedding:
            return (
                "HALT. Sustained slip indicates wheel sinkage. Reverse along the "
                "inbound track and re-plan around this terrain unit."
            )
        if ratio >= self.slip_threshold:
            if slope_deg is not None and slope_deg > 15.0:
                return (
                    f"Reduce commanded step size; slip {ratio:.0%} on a "
                    f"{slope_deg:.0f} deg slope. Prefer a cross-slope traverse."
                )
            return f"Slip {ratio:.0%} above threshold — shorten drive steps and re-check."
        if ratio <= -self.slip_threshold:
            return "Skid detected (rover outrunning wheels) — likely downslope sliding."
        return "Nominal traction."

    def traced_estimate(
        self, estimate: SlipEstimate, *, parents: Sequence[Traced[Any]] = ()
    ) -> Traced[SlipEstimate]:
        return traced(
            estimate,
            source=Source.ESTIMATED,
            algorithm="WheelSlipEstimator(encoder vs visual odometry)",
            confidence=estimate.confidence,
            units="ratio",
            parameters={
                "slip_threshold": self.slip_threshold,
                "embedding_threshold": self.embedding_threshold,
                "window": self.history.maxlen,
            },
            label="wheel_slip",
            parents=parents,
        )
