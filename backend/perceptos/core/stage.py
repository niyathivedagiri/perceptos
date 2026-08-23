"""
perceptos.core.stage
====================

The unit of composition in PerceptOS.

Every perception step — undistortion, feature extraction, disparity, detection,
tracking, planning — is a :class:`PipelineStage`.  Stages know nothing about
each other; they read named inputs from a :class:`FrameContext` and write named
outputs back.  This is what lets the same Kalman filter serve the warehouse
mode and the Mars mode without a single conditional.

The base class handles the cross-cutting concerns so subclasses cannot forget
them:

* wall-clock timing, stamped onto every produced value
* lineage registration
* graceful degradation — a stage that raises is marked ``FAILED`` and the
  pipeline continues with the inputs it already has, rather than dropping the
  frame.  On a rover, a dead feature extractor must not take down hazard
  avoidance.
"""

from __future__ import annotations

import abc
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional

from .lineage import LineageGraph
from .provenance import Source, Traced, traced

__all__ = ["StageStatus", "StageResult", "FrameContext", "PipelineStage"]


class StageStatus(str, Enum):
    OK = "ok"
    SKIPPED = "skipped"       # preconditions not met (e.g. no stereo pair)
    DEGRADED = "degraded"     # produced output, but below quality threshold
    FAILED = "failed"         # raised; pipeline continued without it
    DISABLED = "disabled"     # switched off by the active mode profile


@dataclass(slots=True)
class StageResult:
    stage: str
    status: StageStatus
    latency_ms: float
    outputs: dict[str, Traced[Any]] = field(default_factory=dict)
    message: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 3),
            "outputs": {k: v.to_dict() for k, v in self.outputs.items()},
            "message": self.message,
            "error": self.error,
        }


class FrameContext:
    """Mutable blackboard for one frame's journey through the pipeline.

    Stages read via :meth:`get` / :meth:`require` and write via :meth:`put`.
    All writes are recorded in the lineage graph automatically.
    """

    def __init__(
        self,
        *,
        frame_id: int,
        lineage: Optional[LineageGraph] = None,
        mode: str = "earth_inspection",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.frame_id = int(frame_id)
        self.mode = mode
        self.lineage = lineage if lineage is not None else LineageGraph()
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.created_at = time.time()
        self.results: list[StageResult] = []
        self._slots: dict[str, Traced[Any]] = {}

    # ---------------------------------------------------------------- storage

    def put(
        self,
        key: str,
        value: Traced[Any],
        *,
        stage: str = "",
        artefact: Any = None,
    ) -> Traced[Any]:
        self._slots[key] = value
        self.lineage.record(value, stage=stage, artefact=artefact, label=value.label or key)
        return value

    def get(self, key: str, default: Any = None) -> Optional[Traced[Any]]:
        return self._slots.get(key, default)

    def require(self, *keys: str) -> tuple[Traced[Any], ...]:
        missing = [k for k in keys if k not in self._slots]
        if missing:
            raise KeyError(f"missing required inputs: {', '.join(missing)}")
        return tuple(self._slots[k] for k in keys)

    def has(self, *keys: str) -> bool:
        return all(k in self._slots for k in keys)

    def keys(self) -> Iterable[str]:
        return self._slots.keys()

    # --------------------------------------------------------------- reporting

    @property
    def total_latency_ms(self) -> float:
        return round(sum(r.latency_ms for r in self.results), 3)

    def summary(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "mode": self.mode,
            "created_at": self.created_at,
            "total_latency_ms": self.total_latency_ms,
            "stages": [r.to_dict() for r in self.results],
            "slots": sorted(self._slots.keys()),
            "lineage_nodes": len(self.lineage),
        }


class PipelineStage(abc.ABC):
    """Base class for every perception stage.

    Subclasses implement :meth:`run` and declare :attr:`requires` /
    :attr:`provides`.  The pipeline uses those declarations to validate wiring
    at construction time rather than discovering a missing input mid-mission.
    """

    #: Stable identifier, shown in the lineage drawer and the timing waterfall.
    name: str = "unnamed_stage"
    #: Context keys this stage needs.  Missing key => SKIPPED, not an exception.
    requires: tuple[str, ...] = ()
    #: Context keys this stage promises to write when it returns OK.
    provides: tuple[str, ...] = ()
    #: Declared source class of this stage's outputs.
    source: Source = Source.CALCULATED

    def __init__(self, *, enabled: bool = True, **params: Any) -> None:
        self.enabled = bool(enabled)
        self.params: dict[str, Any] = dict(params)

    # ------------------------------------------------------------- subclass API

    @abc.abstractmethod
    def run(self, ctx: FrameContext) -> Optional[str]:
        """Do the work.  Write outputs with :meth:`emit`.

        Return an optional human-readable status message.  Raise to signal
        failure — the pipeline will catch it and mark the stage FAILED.
        """

    def emit(
        self,
        ctx: FrameContext,
        key: str,
        value: Any,
        *,
        algorithm: Optional[str] = None,
        source: Optional[Source] = None,
        confidence: Optional[float] = None,
        units: Optional[str] = None,
        parents: Iterable[Traced[Any]] = (),
        parameters: Optional[Mapping[str, Any]] = None,
        artefact: Any = None,
        label: str = "",
    ) -> Traced[Any]:
        """Write a traced output into the context.

        Latency is filled in by the pipeline once the stage returns, so callers
        never have to time themselves.
        """
        tv = traced(
            value,
            source=source or self.source,
            algorithm=algorithm or f"{type(self).__name__}",
            confidence=confidence,
            units=units,
            parameters=dict(parameters or self.params),
            label=label or key,
            parents=parents,
        )
        return ctx.put(key, tv, stage=self.name, artefact=artefact)

    def preconditions_met(self, ctx: FrameContext) -> bool:
        return ctx.has(*self.requires)

    def skip_reason(self, ctx: FrameContext) -> Optional[str]:
        """Why this stage cannot run, or ``None`` if it can.

        Subclasses override this to explain a self-imposed skip. Reporting
        "missing inputs" for a stage that skipped because its model failed to
        load sends whoever reads the log looking in entirely the wrong place.
        """
        missing = [k for k in self.requires if not ctx.has(k)]
        return f"missing inputs: {', '.join(missing)}" if missing else None

    # ------------------------------------------------------------ pipeline API

    def execute(self, ctx: FrameContext) -> StageResult:
        """Invoked by :class:`~perceptos.core.pipeline.Pipeline`.  Never raises."""
        # Disabled and skipped stages are still recorded.  A stage that silently
        # vanishes from the report is indistinguishable from one that ran and
        # found nothing — the operator must be able to tell those apart.
        if not self.enabled:
            result = StageResult(self.name, StageStatus.DISABLED, 0.0,
                                 message="disabled by mode profile")
            ctx.results.append(result)
            return result
        reason = self.skip_reason(ctx)
        if reason is not None:
            result = StageResult(self.name, StageStatus.SKIPPED, 0.0, message=reason)
            ctx.results.append(result)
            return result

        before = set(ctx.keys())
        start = time.perf_counter()
        try:
            message = self.run(ctx) or ""
            status = StageStatus.OK
            error = None
        except Exception as exc:  # noqa: BLE001 - deliberate isolation boundary
            message = f"{type(exc).__name__}: {exc}"
            error = traceback.format_exc(limit=6)
            status = StageStatus.FAILED
        latency_ms = (time.perf_counter() - start) * 1000.0

        produced: dict[str, Traced[Any]] = {}
        for key in set(ctx.keys()) - before:
            tv = ctx.get(key)
            if tv is not None:
                stamped = tv.with_latency(latency_ms)
                ctx._slots[key] = stamped  # noqa: SLF001 - same package, by design
                node = ctx.lineage.node(tv.id)
                if node is not None:
                    node.provenance["latency_ms"] = round(latency_ms, 3)
                produced[key] = stamped

        if status is StageStatus.OK:
            unmet = [k for k in self.provides if k not in ctx.keys()]
            if unmet:
                status = StageStatus.DEGRADED
                message = (message + f" | contract not met: {', '.join(unmet)}").strip(" |")

        result = StageResult(self.name, status, latency_ms, produced, message, error)
        ctx.results.append(result)
        return result

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r} enabled={self.enabled}>"
