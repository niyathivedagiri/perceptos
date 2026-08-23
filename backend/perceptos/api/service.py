"""
perceptos.api.service
=====================

The runtime that owns the pipeline, the store and the mission lifecycle.

Kept separate from the FastAPI routes on purpose. The routes translate HTTP to
method calls and nothing else; every behaviour lives here and is testable
without a web server. That separation is also what makes the same runtime
usable from a CLI, a notebook, or a future ROS bridge.

The service is deliberately explicit about **data mode**. It does not accept
"real" as a claim — it inspects the frame source and reports what is actually
attached, so the mode badge in the UI reflects the pipeline rather than a
configuration file someone forgot to update.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..core.environment import ENVIRONMENTS, Environment, get_environment
from ..core.lineage import LineageGraph
from ..core.pipeline import Pipeline
from ..core.provenance import Source
from ..core.stage import FrameContext
from ..db.store import MissionStore
from ..health.monitor import HealthMonitor
from .analysis import NavigationService, TerrainProducts, TerrainService

__all__ = ["PerceptOSService", "ServiceState", "ServiceError"]


class ServiceError(RuntimeError):
    """A request cannot be satisfied in the current state."""


@dataclass(slots=True)
class ServiceState:
    environment: str = "mars"
    mission_id: Optional[str] = None
    mission_name: str = ""
    running: bool = False
    frames_processed: int = 0
    started_utc: Optional[float] = None
    last_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "mission_id": self.mission_id,
            "mission_name": self.mission_name,
            "running": self.running,
            "frames_processed": self.frames_processed,
            "started_utc": self.started_utc,
            "uptime_s": round(time.time() - self.started_utc, 1)
            if self.started_utc else None,
            "last_error": self.last_error,
        }


class PerceptOSService:
    """Application runtime: pipeline + persistence + health, behind one object.

    Parameters
    ----------
    pipeline_factory:
        Called with an :class:`~perceptos.core.environment.Environment` and must
        return a configured :class:`~perceptos.core.pipeline.Pipeline`. Injected
        rather than constructed here so tests can supply a trivial pipeline and
        the API layer never depends on OpenCV being installed.
    """

    def __init__(self, *, store: Optional[MissionStore] = None,
                 pipeline_factory: Optional[Callable[[Environment], Pipeline]] = None,
                 monitor: Optional[HealthMonitor] = None,
                 environment: str = "mars",
                 datasets_root: str = "datasets",
                 lineage_capacity: int = 512) -> None:
        self.store = store if store is not None else MissionStore("missions.db")
        self.monitor = monitor if monitor is not None else HealthMonitor()
        self.pipeline_factory = pipeline_factory
        self.state = ServiceState(environment=environment,
                                  started_utc=time.time())
        self.environment: Environment = get_environment(environment)
        self.pipeline: Optional[Pipeline] = None
        self.lineage = LineageGraph(artefact_capacity=lineage_capacity)
        self._lock = threading.RLock()
        self._subscribers: list[Callable[[dict[str, Any]], None]] = []
        self.last_report: Optional[dict[str, Any]] = None
        # The most recent frame's context is retained so the console can render
        # its slots. One frame, not a history: holding every context would pin
        # every artefact and defeat the LRU bound on the artefact store.
        self.last_context: Optional[FrameContext] = None
        self.terrain = TerrainService(datasets_root=datasets_root)
        self.navigation = NavigationService()

    # ----------------------------------------------------------- environments

    def list_environments(self) -> list[dict[str, Any]]:
        return [env.to_dict() for env in ENVIRONMENTS.values()]

    def set_environment(self, name: str) -> dict[str, Any]:
        """Switch environment. Rebuilds the pipeline without restarting the process."""
        with self._lock:
            environment = get_environment(name)
            self.environment = environment
            self.state.environment = environment.name
            if self.pipeline is not None and self.pipeline_factory is not None:
                self.pipeline = self.pipeline_factory(environment)
            if self.pipeline is not None:
                self.pipeline.apply_mode({"mode": environment.name,
                                          "stages": dict(environment.stages)})
            return environment.to_dict()

    # --------------------------------------------------------------- pipeline

    def ensure_pipeline(self) -> Pipeline:
        if self.pipeline is None:
            if self.pipeline_factory is None:
                raise ServiceError(
                    "no pipeline factory configured; construct PerceptOSService "
                    "with pipeline_factory=... to enable frame processing"
                )
            self.pipeline = self.pipeline_factory(self.environment)
        return self.pipeline

    def pipeline_contract(self) -> list[dict[str, Any]]:
        return self.ensure_pipeline().contract()

    def pipeline_diagram(self) -> str:
        return self.ensure_pipeline().to_mermaid()

    def set_stage_enabled(self, stage: str, enabled: bool) -> dict[str, bool]:
        pipeline = self.ensure_pipeline()
        if pipeline.stage(stage) is None:
            raise ServiceError(
                f"unknown stage {stage!r}; stages: "
                f"{[s.name for s in pipeline.stages]}"
            )
        return pipeline.apply_mode({"stages": {stage: {"enabled": bool(enabled)}}})

    # --------------------------------------------------------------- missions

    @property
    def data_mode(self) -> str:
        """What the pipeline is *actually* fed, inspected rather than declared."""
        pipeline = self.pipeline
        if pipeline is None:
            return "unknown"
        for stage in pipeline.stages:
            source = getattr(getattr(stage, "manager", None), "source", None) \
                or getattr(stage, "stereo", None)
            kind = getattr(source, "source_kind", None)
            if kind is not None:
                return "simulation" if kind is Source.SIMULATION else "real"
        return "simulation"

    def start_mission(self, *, name: str = "", notes: str = "") -> dict[str, Any]:
        with self._lock:
            if self.state.running:
                raise ServiceError(
                    f"mission {self.state.mission_id} is already running; "
                    "stop it before starting another"
                )
            self.ensure_pipeline()
            mission_name = name or f"{self.environment.name}-{int(time.time())}"
            mission_id = self.store.start_mission(
                name=mission_name,
                environment=self.environment.name,
                mode=self.environment.name,
                data_mode=self.data_mode,
                notes=notes,
                config={"stages": self.pipeline_contract()},
            )
            self.state.mission_id = mission_id
            self.state.mission_name = mission_name
            self.state.running = True
            self.state.frames_processed = 0
            self.state.last_error = ""
            self.lineage = LineageGraph(artefact_capacity=512)
            return self.store.get_mission(mission_id).to_dict()

    def stop_mission(self, *, status: str = "complete") -> dict[str, Any]:
        with self._lock:
            if not self.state.running or self.state.mission_id is None:
                raise ServiceError("no mission is running")
            mission_id = self.state.mission_id
            self.store.end_mission(mission_id, status=status)
            self.state.running = False
            record = self.store.get_mission(mission_id)
            return record.to_dict() if record else {}

    # ------------------------------------------------------------- processing

    def process_frame(self, seeds: Optional[dict[str, Any]] = None
                      ) -> dict[str, Any]:
        """Run one frame and persist it if a mission is recording."""
        with self._lock:
            pipeline = self.ensure_pipeline()
            try:
                ctx, report = pipeline.process(seeds or {}, lineage=self.lineage)
            except Exception as exc:  # noqa: BLE001
                self.state.last_error = f"{type(exc).__name__}: {exc}"
                raise ServiceError(self.state.last_error) from exc

            self.state.frames_processed += 1
            self.last_context = ctx
            payload = report.to_dict()
            payload["mission_id"] = self.state.mission_id
            payload["data_mode"] = self.data_mode
            payload["lineage_nodes"] = len(ctx.lineage)
            payload["slots"] = sorted(ctx.keys())
            payload["metadata"] = _jsonable(ctx.metadata)
            self.last_report = payload

            if self.state.running and self.state.mission_id:
                self.store.record_frame(self.state.mission_id, ctx, report)

            self._publish({"type": "frame", "payload": payload})
            return payload

    def run(self, frames: int, *, seeds: Optional[dict[str, Any]] = None
            ) -> list[dict[str, Any]]:
        return [self.process_frame(seeds) for _ in range(int(frames))]

    # ------------------------------------------------------------- frame data

    #: Context keys the console can render, with how to render them.
    RENDERABLE: dict[str, dict[str, Any]] = {
        "frame": {"label": "Camera frame", "colormap": "rgb", "units": "px"},
        "undistorted": {"label": "Undistorted", "colormap": "rgb", "units": "px"},
        "gray": {"label": "Greyscale", "colormap": "grey", "units": "px"},
        "processed": {"label": "Processed", "colormap": "grey", "units": "px"},
        "gray_right": {"label": "Right eye", "colormap": "grey", "units": "px"},
        "disparity": {"label": "Disparity", "colormap": "inferno", "units": "px"},
        "depth": {"label": "Depth", "colormap": "viridis", "units": "m"},
        "features": {"label": "Features", "colormap": "rgb", "units": "keypoints"},
        "tracks": {"label": "Tracks", "colormap": "rgb", "units": "tracks"},
    }

    def frame_slots(self) -> dict[str, Any]:
        """What the last processed frame produced, and how to fetch each part.

        Returns UNAVAILABLE rather than an empty shell when no frame has run,
        so the console can say "no frame yet" instead of drawing blank panels.
        """
        ctx = self.last_context
        if ctx is None:
            return {
                "available": False,
                "status": "UNAVAILABLE",
                "reason": "No frame has been processed yet. Run the pipeline "
                          "(POST /api/pipeline/run) to populate this.",
                "slots": [],
            }
        slots = []
        for key in ctx.keys():
            value = ctx.get(key)
            if value is None:
                continue
            spec = self.RENDERABLE.get(key)
            slots.append({
                "key": key,
                "label": (spec or {}).get("label", key),
                "renderable": spec is not None,
                "colormap": (spec or {}).get("colormap"),
                "units": (spec or {}).get("units") or value.provenance.units,
                "node_id": value.id,
                "source": value.provenance.source.value,
                "source_type": value.provenance.source_type,
                "dataset": value.provenance.dataset,
                "confidence": value.provenance.confidence,
                "uncertainty": value.provenance.uncertainty,
                "algorithm": value.provenance.algorithm,
                "latency_ms": value.provenance.latency_ms,
            })
        slots.sort(key=lambda s: (not s["renderable"], s["key"]))
        return {
            "available": True,
            "status": "OK",
            "frame_id": ctx.frame_id,
            "mode": ctx.mode,
            "data_mode": self.data_mode,
            "created_at": ctx.created_at,
            "total_latency_ms": ctx.total_latency_ms,
            "metadata": _jsonable(ctx.metadata),
            "slots": slots,
        }

    def slot_value(self, key: str) -> Optional[Any]:
        """The traced value in a slot of the last frame, or ``None``."""
        if self.last_context is None:
            return None
        return self.last_context.get(key)

    # ------------------------------------------------------------ analysis

    def analyse_terrain(self, *, allow_simulation: bool = False,
                        force: bool = False) -> TerrainProducts:
        """Terrain products, computed once and reused until inputs change."""
        cached = self.terrain.products
        if cached is not None and not force and cached.available:
            return cached
        return self.terrain.analyse(self.environment,
                                    allow_simulation=allow_simulation)

    # ---------------------------------------------------------------- lineage

    def explain(self, node_id: str, *, mission_id: Optional[str] = None
                ) -> Optional[dict[str, Any]]:
        """Live lineage first, persisted lineage second.

        A node from the current run is answered from memory (it still has its
        artefact); one from an older mission is reconstructed from the database.
        The payload shape is identical either way, so the UI drawer does not
        care which it got.
        """
        for graph, origin in ((self.lineage, "live"),
                              (self.terrain.lineage, "terrain"),
                              (self.navigation.lineage, "navigation")):
            if mission_id is None and node_id in graph:
                payload = graph.explain(node_id)
                payload["mission_id"] = self.state.mission_id
                payload["origin"] = origin
                return payload
        target = mission_id or self.state.mission_id
        if target is None:
            return None
        payload = self.store.explain(target, node_id)
        if payload is not None:
            payload["origin"] = "persisted"
        return payload

    def search_lineage(self, *, mission_id: Optional[str] = None, **kwargs: Any
                       ) -> list[dict[str, Any]]:
        target = mission_id or self.state.mission_id
        if target is None:
            return []
        return self.store.find_nodes(target, **kwargs)

    # ----------------------------------------------------------------- health

    def health(self) -> dict[str, Any]:
        driving = self.state.running
        snapshot = self.monitor.snapshot(driving=driving)
        snapshot["service"] = self.state.to_dict()
        snapshot["store"] = self.store.statistics()
        snapshot["pipeline"] = {
            "configured": self.pipeline is not None,
            "stages": [s.name for s in self.pipeline.stages] if self.pipeline else [],
            "last_total_latency_ms": (self.last_report or {}).get("total_latency_ms"),
            "fps_ceiling": (self.last_report or {}).get("fps_ceiling"),
        }
        if self.state.running and self.state.mission_id:
            self.store.record_telemetry(
                self.state.mission_id, "health", snapshot["system"],
                source=snapshot["system"]["_provenance"]["source"],
            )
        return snapshot

    # ------------------------------------------------------------ subscribers

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        """Register a WebSocket-style listener. Returns an unsubscribe callable."""
        self._subscribers.append(callback)

        def unsubscribe() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return unsubscribe

    def _publish(self, message: dict[str, Any]) -> None:
        # A failing subscriber must not take down the pipeline: a browser tab
        # closing mid-frame is routine, not an incident.
        for callback in list(self._subscribers):
            try:
                callback(message)
            except Exception:  # noqa: BLE001
                self._subscribers.remove(callback)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _jsonable(to_dict())
        except Exception:  # noqa: BLE001
            pass
    return repr(value)[:256]
