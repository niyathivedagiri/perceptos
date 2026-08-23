"""
perceptos.core.provenance
=========================

The credibility layer of PerceptOS.

Design rule (non-negotiable): *no number reaches the operator without
declaring where it came from.*  A raw ``float`` is not a valid output of any
PerceptOS stage.  Stages emit :class:`Traced` values, which bind the value to
a :class:`Provenance` record describing its source, the algorithm that produced
it, when it was produced, how long that took, and how confident the system is.

The UI is contractually required to render the source badge next to every
value.  ``Source.SIMULATION`` values are rendered with a warning badge so an
operator can never mistake synthetic data for sensor data.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Generic, Iterable, Mapping, Optional, Sequence, TypeVar

T = TypeVar("T")

__all__ = [
    "Source",
    "Provenance",
    "Traced",
    "traced",
    "ProvenanceError",
    "utc_now",
]


class ProvenanceError(ValueError):
    """Raised when a value is constructed without valid provenance."""


def utc_now() -> float:
    """Wall-clock UTC seconds.  Single choke point so tests can monkeypatch it."""
    return time.time()


class Source(str, Enum):
    """The only five origins a displayed value may have.

    Anything that cannot be assigned one of these is a bug, not a feature.
    """

    #: Read directly off a non-imaging sensor (IMU, encoder, battery monitor).
    LIVE_SENSOR = "live_sensor"
    #: A frame or a value derived 1:1 from a live camera frame.
    LIVE_CAMERA = "live_camera"
    #: Deterministic arithmetic on inputs that are themselves traced.
    CALCULATED = "calculated"
    #: Output of a statistical/learned model — carries irreducible uncertainty.
    ESTIMATED = "estimated"
    #: Synthetic.  Must be badged in the UI.  Never silently mixed with real data.
    SIMULATION = "simulation"

    @property
    def is_synthetic(self) -> bool:
        return self is Source.SIMULATION

    @property
    def is_measured(self) -> bool:
        return self in (Source.LIVE_SENSOR, Source.LIVE_CAMERA)


@dataclass(frozen=True, slots=True)
class Provenance:
    """Immutable description of how a single value came to exist.

    Parameters
    ----------
    source:
        One of :class:`Source`.
    algorithm:
        Human-readable identifier of the producing algorithm, e.g.
        ``"cv2.StereoSGBM"``, ``"ConstantAccelerationKF.update"``.  This string
        is shown verbatim in the lineage drawer, so keep it precise.
    timestamp_utc:
        When the value was produced (UTC seconds).
    confidence:
        ``0.0``–``1.0``.  Required for :attr:`Source.ESTIMATED`, because an
        estimate without an uncertainty is an unfalsifiable claim.
    latency_ms:
        Wall-clock cost of producing this value.  Populated automatically by
        :class:`~perceptos.core.stage.PipelineStage`.
    units:
        SI units where applicable (``"m"``, ``"m/s"``, ``"deg"``, ``"px"``).
        Unitless quantities use ``None``; ratios use ``"ratio"``.
    parameters:
        The exact tuning knobs in force when the value was produced.  This is
        what makes a run reproducible six months later.
    """

    source: Source
    algorithm: str
    timestamp_utc: float = field(default_factory=utc_now)
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None
    units: Optional[str] = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    #: Where the underlying bytes came from, as a DataSourceStatus value
    #: ("NASA_PUBLIC_NO_KEY", "LOCAL_CACHED_DATA", "SIMULATION", ...). Kept as
    #: a plain string so this module stays free of a data-layer import.
    source_type: Optional[str] = None
    #: The exact URL the data was retrieved from, where one exists.
    source_url: Optional[str] = None
    #: Named dataset ("perseverance_jpl", "ai4mars", "jezero_dem").
    dataset: Optional[str] = None
    #: Uncertainty in the same units as the value. Distinct from confidence:
    #: confidence is "how much do we trust this at all" in [0,1]; uncertainty
    #: is "plus or minus what" in physical units. A depth of 4.0 m can have
    #: confidence 0.9 and uncertainty 0.35 m, and neither substitutes for the
    #: other.
    uncertainty: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.algorithm or not self.algorithm.strip():
            raise ProvenanceError("provenance.algorithm must be a non-empty string")
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise ProvenanceError(
                f"confidence must lie in [0, 1], got {self.confidence!r}"
            )
        if self.source is Source.ESTIMATED and self.confidence is None:
            raise ProvenanceError(
                f"ESTIMATED value from {self.algorithm!r} must carry a confidence; "
                "an estimate without uncertainty cannot be audited"
            )
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ProvenanceError("latency_ms cannot be negative")
        if self.uncertainty is not None and self.uncertainty < 0:
            raise ProvenanceError("uncertainty cannot be negative")
        if self.source is Source.SIMULATION and self.source_type not in (
            None, "SIMULATION"
        ):
            raise ProvenanceError(
                f"a SIMULATION value cannot declare source_type="
                f"{self.source_type!r}; synthetic data must not be labelled "
                "as having come from an external archive"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "algorithm": self.algorithm,
            "timestamp_utc": self.timestamp_utc,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "units": self.units,
            "parameters": dict(self.parameters),
            "is_synthetic": self.source.is_synthetic,
            "source_type": self.source_type or (
                "SIMULATION" if self.source.is_synthetic else None
            ),
            "source_url": self.source_url,
            "dataset": self.dataset,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True, slots=True)
class Traced(Generic[T]):
    """A value plus its provenance plus its parents.

    ``Traced`` objects form a directed acyclic graph.  Walking ``parents``
    backwards from any displayed value reconstructs the full derivation chain
    that the Data Lineage drawer renders.
    """

    value: T
    provenance: Provenance
    parents: tuple[str, ...] = ()
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    label: str = ""

    @property
    def source(self) -> Source:
        return self.provenance.source

    @property
    def confidence(self) -> Optional[float]:
        return self.provenance.confidence

    @property
    def is_synthetic(self) -> bool:
        """True if this value, by itself, is simulated."""
        return self.provenance.source.is_synthetic

    def derive(
        self,
        value: Any,
        *,
        algorithm: str,
        source: Source = Source.CALCULATED,
        confidence: Optional[float] = None,
        latency_ms: Optional[float] = None,
        units: Optional[str] = None,
        parameters: Optional[Mapping[str, Any]] = None,
        label: str = "",
        extra_parents: Sequence["Traced[Any]"] = (),
        source_type: Optional[str] = None,
        source_url: Optional[str] = None,
        dataset: Optional[str] = None,
        uncertainty: Optional[float] = None,
    ) -> "Traced[Any]":
        """Produce a child value that records *this* value as an input.

        Confidence propagation: if the child does not declare its own
        confidence, it inherits the *minimum* confidence of its parents.  A
        derived value can never be more certain than the weakest thing it was
        computed from.
        """
        parents = [self.id, *(p.id for p in extra_parents)]
        if confidence is None:
            known = [
                c
                for c in (
                    self.confidence,
                    *(p.confidence for p in extra_parents),
                )
                if c is not None
            ]
            confidence = min(known) if known else None
        # Simulated inputs contaminate downstream values — this is intentional.
        if source is not Source.SIMULATION and any(
            p.is_synthetic for p in (self, *extra_parents)
        ):
            source = Source.SIMULATION
            source_type = "SIMULATION"
        # Dataset lineage flows downstream: a slope map derived from a Jezero
        # DEM is still Jezero data, and the drawer should say so without the
        # caller having to restate it at every hop.
        if dataset is None:
            dataset = next((p.provenance.dataset for p in (self, *extra_parents)
                            if p.provenance.dataset), None)
        if source_type is None and source is not Source.SIMULATION:
            source_type = next((p.provenance.source_type
                                for p in (self, *extra_parents)
                                if p.provenance.source_type), None)
        return Traced(
            value=value,
            provenance=Provenance(
                source=source,
                algorithm=algorithm,
                confidence=confidence,
                latency_ms=latency_ms,
                units=units,
                parameters=dict(parameters or {}),
                source_type=source_type,
                source_url=source_url,
                dataset=dataset,
                uncertainty=uncertainty,
            ),
            parents=tuple(parents),
            label=label,
        )

    def with_latency(self, latency_ms: float) -> "Traced[T]":
        return replace(self, provenance=replace(self.provenance, latency_ms=latency_ms))

    def to_dict(self, *, include_value: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "parents": list(self.parents),
            "provenance": self.provenance.to_dict(),
        }
        if include_value:
            out["value"] = _summarise(self.value)
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Traced({_summarise(self.value)!r}, src={self.provenance.source.value}, "
            f"algo={self.provenance.algorithm!r})"
        )


def traced(
    value: T,
    *,
    source: Source,
    algorithm: str,
    confidence: Optional[float] = None,
    latency_ms: Optional[float] = None,
    units: Optional[str] = None,
    parameters: Optional[Mapping[str, Any]] = None,
    label: str = "",
    parents: Iterable["Traced[Any]"] = (),
    source_type: Optional[str] = None,
    source_url: Optional[str] = None,
    dataset: Optional[str] = None,
    uncertainty: Optional[float] = None,
) -> Traced[T]:
    """Convenience constructor for a root (or explicitly parented) traced value."""
    parent_list = list(parents)
    if any(p.is_synthetic for p in parent_list) and source is not Source.SIMULATION:
        source = Source.SIMULATION
        source_type = "SIMULATION"
    if source is Source.SIMULATION and source_type is None:
        source_type = "SIMULATION"
    if dataset is None:
        dataset = next((p.provenance.dataset for p in parent_list
                        if p.provenance.dataset), None)
    # Inherit source_type from parents exactly as Traced.derive does. Without
    # this the two construction paths disagree: a value built with traced(...,
    # parents=[x]) loses the origin that the same value built with x.derive(...)
    # would keep, so a depth map derived from cached NASA imagery reported a
    # blank source_type and the UI badge fell back to "unknown".
    if source_type is None and source is not Source.SIMULATION:
        source_type = next((p.provenance.source_type for p in parent_list
                            if p.provenance.source_type), None)
    return Traced(
        value=value,
        provenance=Provenance(
            source=source,
            algorithm=algorithm,
            confidence=confidence,
            latency_ms=latency_ms,
            units=units,
            parameters=dict(parameters or {}),
            source_type=source_type,
            source_url=source_url,
            dataset=dataset,
            uncertainty=uncertainty,
        ),
        parents=tuple(p.id for p in parent_list),
        label=label,
    )


def _finite(array: Any, *, want_max: bool = False) -> Optional[float]:
    """nanmin/nanmax as a JSON-safe float, or ``None`` if nothing is finite."""
    try:
        if array.size == 0:
            return None
        import numpy as _np  # noqa: PLC0415 - only needed for array inputs

        with _np.errstate(all="ignore"):
            result = float(_np.nanmax(array) if want_max else _np.nanmin(array))
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, AttributeError):
        return None


def _summarise(value: Any) -> Any:
    """JSON-safe, bounded summary of a value.

    Large arrays (images, point clouds, cost maps) are described by shape and
    range rather than serialised — the lineage drawer fetches the actual raster
    lazily from the artefact store.
    """
    shape = getattr(value, "shape", None)
    if shape is not None and hasattr(value, "dtype"):
        try:
            # nanmin/nanmax, and non-finite results reported as None. A real
            # DEM contains NaN wherever the survey had no data, and `float(nan)`
            # serialises to the bare token `NaN`, which is not valid JSON —
            # every lineage request touching that value would 500. This only
            # surfaces with real data; synthetic arrays are always finite.
            return {
                "__array__": True,
                "shape": list(shape),
                "dtype": str(value.dtype),
                "min": _finite(value),
                "max": _finite(value, want_max=True),
            }
        except (TypeError, ValueError):  # non-numeric arrays
            return {"__array__": True, "shape": list(shape), "dtype": str(value.dtype)}
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_summarise(v) for v in value[:32]]
    if isinstance(value, Mapping):
        return {str(k): _summarise(v) for k, v in list(value.items())[:32]}
    return repr(value)[:256]
