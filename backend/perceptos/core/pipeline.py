"""
perceptos.core.pipeline
=======================

Ordered execution of stages, with static wiring validation.

:meth:`Pipeline.validate` runs at construction: it walks the stage list keeping
a set of keys that will exist by the time each stage runs, and reports any
stage whose ``requires`` can never be satisfied.  Catching a mis-wired pipeline
at import time beats discovering it on a rover 200 million km away.

The pipeline also owns the *mode profile*.  Switching from Mars Exploration to
Warehouse Robotics enables and disables stages in place — no restart, which is
what the spec requires.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from .lineage import LineageGraph
from .stage import FrameContext, PipelineStage, StageResult, StageStatus

__all__ = ["Pipeline", "PipelineReport", "WiringError"]


class WiringError(RuntimeError):
    """Raised when a stage requires a key no upstream stage can provide."""


@dataclass(slots=True)
class PipelineReport:
    """Timing waterfall + status roll-up for one frame."""

    frame_id: int
    mode: str
    total_latency_ms: float
    stages: list[StageResult] = field(default_factory=list)

    @property
    def failed(self) -> list[StageResult]:
        return [s for s in self.stages if s.status is StageStatus.FAILED]

    @property
    def healthy(self) -> bool:
        return not self.failed

    @property
    def bottleneck(self) -> Optional[StageResult]:
        active = [s for s in self.stages if s.latency_ms > 0]
        return max(active, key=lambda s: s.latency_ms) if active else None

    def to_dict(self) -> dict[str, Any]:
        bn = self.bottleneck
        return {
            "frame_id": self.frame_id,
            "mode": self.mode,
            "total_latency_ms": round(self.total_latency_ms, 3),
            "fps_ceiling": round(1000.0 / self.total_latency_ms, 2)
            if self.total_latency_ms > 0
            else None,
            "healthy": self.healthy,
            "bottleneck": bn.stage if bn else None,
            "stages": [s.to_dict() for s in self.stages],
        }


class Pipeline:
    """An ordered, validated, mode-aware chain of :class:`PipelineStage`."""

    def __init__(
        self,
        stages: Sequence[PipelineStage],
        *,
        name: str = "perception",
        mode: str = "earth_inspection",
        validate: bool = True,
        seed_keys: Iterable[str] = ("frame",),
    ) -> None:
        self.name = name
        self.mode = mode
        self.stages: list[PipelineStage] = list(stages)
        self.seed_keys = tuple(seed_keys)
        self._frame_counter = 0
        if validate:
            self.validate()

    # ------------------------------------------------------------- validation

    def validate(self) -> None:
        """Static check that every stage's inputs can exist when it runs."""
        available: set[str] = set(self.seed_keys)
        problems: list[str] = []
        seen: set[str] = set()
        for stage in self.stages:
            if stage.name in seen:
                problems.append(f"duplicate stage name {stage.name!r}")
            seen.add(stage.name)
            unmet = [k for k in stage.requires if k not in available]
            if unmet:
                problems.append(
                    f"stage {stage.name!r} requires {unmet} which no earlier "
                    f"stage provides (available: {sorted(available)})"
                )
            available.update(stage.provides)
        if problems:
            raise WiringError("; ".join(problems))

    def contract(self) -> list[dict[str, Any]]:
        """Machine-readable wiring, used to auto-generate the block diagram."""
        return [
            {
                "name": s.name,
                "class": type(s).__name__,
                "requires": list(s.requires),
                "provides": list(s.provides),
                "source": s.source.value,
                "enabled": s.enabled,
            }
            for s in self.stages
        ]

    def to_mermaid(self) -> str:
        """Auto-generated system block diagram — always matches the real code."""
        lines = ["graph LR"]
        producer: dict[str, str] = {k: "input" for k in self.seed_keys}
        lines.append('  input(["Sensor input"])')
        for s in self.stages:
            style = "" if s.enabled else ":::disabled"
            lines.append(f'  {s.name}["{s.name}"]{style}')
            for key in s.requires:
                src = producer.get(key)
                if src:
                    lines.append(f"  {src} -->|{key}| {s.name}")
            for key in s.provides:
                producer[key] = s.name
        lines.append("  classDef disabled fill:#222,stroke:#555,color:#777;")
        return "\n".join(lines)

    # ------------------------------------------------------------ mode control

    def apply_mode(self, profile: Mapping[str, Any]) -> dict[str, bool]:
        """Enable/disable stages for a mode without restarting the process.

        ``profile`` is the ``stages`` block of a mode YAML: a mapping of stage
        name to ``{"enabled": bool, "params": {...}}``.  Unlisted stages keep
        their current state, so a partial profile is a valid patch.
        """
        self.mode = str(profile.get("mode", self.mode))
        wanted: Mapping[str, Any] = profile.get("stages", {}) or {}
        applied: dict[str, bool] = {}
        for stage in self.stages:
            spec = wanted.get(stage.name)
            if spec is None:
                applied[stage.name] = stage.enabled
                continue
            if isinstance(spec, bool):
                stage.enabled = spec
            else:
                stage.enabled = bool(spec.get("enabled", stage.enabled))
                stage.params.update(spec.get("params", {}) or {})
            applied[stage.name] = stage.enabled
        return applied

    def stage(self, name: str) -> Optional[PipelineStage]:
        return next((s for s in self.stages if s.name == name), None)

    # ---------------------------------------------------------------- execution

    def process(
        self,
        seeds: Mapping[str, Any],
        *,
        lineage: Optional[LineageGraph] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> tuple[FrameContext, PipelineReport]:
        """Run one frame through every enabled stage.

        ``seeds`` maps context keys to already-traced root values (typically the
        camera frame).  Returns the finished context and a timing report.
        """
        self._frame_counter += 1
        ctx = FrameContext(
            frame_id=self._frame_counter,
            lineage=lineage,
            mode=self.mode,
            metadata=metadata,
        )
        for key, value in seeds.items():
            ctx.put(key, value, stage="acquisition")

        start = time.perf_counter()
        for stage in self.stages:
            stage.execute(ctx)
        total_ms = (time.perf_counter() - start) * 1000.0

        report = PipelineReport(
            frame_id=ctx.frame_id,
            mode=self.mode,
            total_latency_ms=total_ms,
            stages=list(ctx.results),
        )
        return ctx, report

    def __len__(self) -> int:
        return len(self.stages)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        on = sum(1 for s in self.stages if s.enabled)
        return f"<Pipeline {self.name!r} mode={self.mode!r} stages={on}/{len(self.stages)}>"
