"""
perceptos.api.app
=================

The HTTP and WebSocket surface.

Routes are thin: each one translates a request into a
:class:`~perceptos.api.service.PerceptOSService` call and shapes the response.
No behaviour lives here, which is why the service can be tested without a
server and the API can be tested without OpenCV.

Endpoint groups:

===========================  ==========================================
``/api/environments``        the five operating contexts, and switching
``/api/pipeline``            wiring contract, block diagram, stage toggles
``/api/missions``            start, stop, list, frames, delete
``/api/lineage``             the Data Lineage drawer's backend
``/api/health``              measured host metrics + modelled rover power
``/ws/telemetry``            live frame and health push
===========================  ==========================================

``GET /api/lineage/{node_id}`` is the one that matters. It answers "why did the
rover decide that?" for any value the system has ever displayed, from the live
run or from a mission recorded months ago, with an identical payload shape so
the frontend drawer does not branch on which.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
from typing import Any, Optional

from fastapi import Body, Depends, FastAPI, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect

from ..core.environment import ENVIRONMENTS
from ..data.sources import (
    NASA_SERVICES,
    DataSourceStatus,
    LabelProvenance,
    StereoStatus,
    credentials_report,
)
from ..mars.planner import PlanningFailure
from ..datasets.manager import DatasetManager, LeakageError
from ..datasets.registry import ModelRegistry
from ..reports.mission import ReportBuilder
from .analysis import PLANNERS
from .replay import ReplayService
from .console import CONSOLE_HTML
from .render import (
    render_array,
    render_hillshade,
    render_keypoints,
    render_path,
    render_point_cloud,
    render_terrain_3d,
    render_tracks,
    render_trajectory,
)
from .service import PerceptOSService, ServiceError

__all__ = ["create_app", "get_service"]


# --------------------------------------------------------------------------
# Request / response schemas
# --------------------------------------------------------------------------

class EnvironmentSwitch(BaseModel):
    environment: str = Field(..., description="mars | moon | earth | warehouse | factory")


class StageToggle(BaseModel):
    enabled: bool


class MissionStart(BaseModel):
    name: str = ""
    notes: str = ""


class MissionStop(BaseModel):
    status: str = "complete"


class RunRequest(BaseModel):
    frames: int = Field(1, ge=1, le=500)


class PlanRequest(BaseModel):
    planner: str = Field("theta", description="astar | theta")
    start: Optional[list[int]] = Field(None, description="[row, col]")
    goal: Optional[list[int]] = None
    cost_weight: float = Field(4.0, ge=0.0, le=100.0)
    rover_radius_m: float = Field(0.5, ge=0.0, le=10.0)


class IngestRequest(BaseModel):
    directory: str
    name: str
    pattern: str = "*.png"
    group_from: str = Field("parent", description="parent | prefix | none")


class SplitRequest(BaseModel):
    train: float = Field(0.7, gt=0.0, lt=1.0)
    validation: float = Field(0.15, ge=0.0, lt=1.0)
    test: float = Field(0.15, gt=0.0, lt=1.0)
    group_key: Optional[str] = "group"
    seed: int = 0
    allow_random: bool = False


class ReplaySeek(BaseModel):
    position: int = Field(0, ge=0)


class CounterfactualRequest(BaseModel):
    planner: str = "theta"
    cost_weight: Optional[float] = Field(None, ge=0.0, le=100.0)
    inflation_radius_m: Optional[float] = Field(None, ge=0.0, le=10.0)


class ReplanRequest(BaseModel):
    changes: list[list[float]] = Field(
        ..., description="[[row, col, cost], ...] newly observed cell costs",
    )


class ErrorPayload(BaseModel):
    detail: str
    hint: str = ""


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------

def create_app(service: Optional[PerceptOSService] = None, *,
               cors_origins: Optional[list[str]] = None) -> FastAPI:
    """Build the application around a service instance.

    The service is injected rather than constructed at import time so tests get
    an in-memory database and a trivial pipeline, and so a deployment can wire
    in a real camera without editing this module.
    """
    runtime = service if service is not None else PerceptOSService()

    app = FastAPI(
        title="PerceptOS",
        version="0.5.0",
        description=(
            "Robotic perception, autonomy and planetary exploration platform. "
            "Every value returned by this API carries its source, algorithm, "
            "confidence, latency and full derivation chain."
        ),
    )
    app.state.service = runtime
    app.state.replay = ReplayService(runtime.store)
    app.state.datasets = DatasetManager(
        Path(runtime.terrain.root) / "versions")
    app.state.models = ModelRegistry("models/registry")

    def _replay_guard(fn):
        try:
            return fn()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc).strip("'")) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    # The frontend runs on a different port in development.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_runtime() -> PerceptOSService:
        return app.state.service

    def _guard(callable_: Any, *args: Any, **kwargs: Any) -> Any:
        """Translate a ServiceError into a 409 with the message intact."""
        try:
            return callable_(*args, **kwargs)
        except ServiceError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc).strip("'")) from exc

    # ------------------------------------------------------------------ root

    @app.get("/", tags=["status"], response_class=HTMLResponse)
    def root(request: Request):
        """Service status. HTML for a browser, JSON for anything else.

        Content negotiation rather than two endpoints: a human opening the base
        URL gets a readable page, and `curl` or a client library gets JSON from
        the same address.
        """
        payload = _root_payload(app.state.service, app)
        accept = request.headers.get("accept", "")
        if "text/html" not in accept:
            return JSONResponse(payload)
        return HTMLResponse(_render_status_page(payload))

    @app.get("/console", include_in_schema=False, response_class=HTMLResponse)
    def console() -> str:
        """Mission control console. Served from the same origin as the API.

        No build step and no second dev server: the console is one document
        that runs the moment the backend does.
        """
        return CONSOLE_HTML

    @app.get("/api", tags=["status"])
    def api_root(svc: PerceptOSService = Depends(get_runtime)):
        return _root_payload(svc, app)

    @app.get("/api/datasources", tags=["status"])
    def datasources(svc: PerceptOSService = Depends(get_runtime)):
        """Which external services PerceptOS uses, and which need credentials.

        Answers "do I need a NASA API key?" from the running system rather than
        from a README that may have drifted from the code.
        """
        report = credentials_report()
        report["active_environment"] = svc.environment.name
        report["environment_data"] = svc.environment.to_dict()["data"]
        report["pipeline_data_mode"] = svc.data_mode
        return report

    @app.get("/api/status", tags=["status"])
    def status(svc: PerceptOSService = Depends(get_runtime)) -> dict[str, Any]:
        return {
            "service": svc.state.to_dict(),
            "environment": svc.environment.to_dict(),
            "data_mode": svc.data_mode,
            "schema": "perceptos/0.5.0",
        }

    # ---------------------------------------------------------- environments

    @app.get("/api/environments", tags=["environments"])
    def list_environments(svc: PerceptOSService = Depends(get_runtime)):
        return {"environments": svc.list_environments(),
                "active": svc.environment.name}

    @app.get("/api/environments/{name}", tags=["environments"])
    def get_environment_detail(name: str):
        if name not in ENVIRONMENTS:
            raise HTTPException(404, f"unknown environment {name!r}; "
                                     f"available: {sorted(ENVIRONMENTS)}")
        return ENVIRONMENTS[name].to_dict()

    @app.post("/api/environments/active", tags=["environments"])
    def switch_environment(body: EnvironmentSwitch,
                           svc: PerceptOSService = Depends(get_runtime)):
        return _guard(svc.set_environment, body.environment)

    # -------------------------------------------------------------- pipeline

    @app.get("/api/pipeline", tags=["pipeline"])
    def pipeline_contract(svc: PerceptOSService = Depends(get_runtime)):
        return {"stages": _guard(svc.pipeline_contract),
                "environment": svc.environment.name}

    @app.get("/api/pipeline/diagram", response_class=PlainTextResponse,
             tags=["pipeline"])
    def pipeline_diagram(svc: PerceptOSService = Depends(get_runtime)) -> str:
        """Mermaid block diagram generated from the real wiring."""
        return _guard(svc.pipeline_diagram)

    @app.post("/api/pipeline/stages/{stage}", tags=["pipeline"])
    def toggle_stage(stage: str, body: StageToggle,
                     svc: PerceptOSService = Depends(get_runtime)):
        return {"stages": _guard(svc.set_stage_enabled, stage, body.enabled)}

    @app.post("/api/pipeline/run", tags=["pipeline"])
    def run_frames(body: RunRequest, svc: PerceptOSService = Depends(get_runtime)):
        reports = _guard(svc.run, body.frames)
        return {"frames": len(reports), "reports": reports}

    # -------------------------------------------------------------- missions

    @app.get("/api/missions", tags=["missions"])
    def list_missions(limit: int = Query(50, ge=1, le=500),
                      svc: PerceptOSService = Depends(get_runtime)):
        return {"missions": [m.to_dict() for m in svc.store.list_missions(limit=limit)]}

    @app.post("/api/missions", status_code=201, tags=["missions"])
    def start_mission(body: MissionStart = Body(default=MissionStart()),
                      svc: PerceptOSService = Depends(get_runtime)):
        return _guard(svc.start_mission, name=body.name, notes=body.notes)

    @app.post("/api/missions/active/stop", tags=["missions"])
    def stop_mission(body: MissionStop = Body(default=MissionStop()),
                     svc: PerceptOSService = Depends(get_runtime)):
        return _guard(svc.stop_mission, status=body.status)

    @app.get("/api/missions/{mission_id}", tags=["missions"])
    def get_mission(mission_id: str, svc: PerceptOSService = Depends(get_runtime)):
        record = svc.store.get_mission(mission_id)
        if record is None:
            raise HTTPException(404, f"no mission {mission_id!r}")
        return record.to_dict()

    @app.get("/api/missions/{mission_id}/frames", tags=["missions"])
    def get_frames(mission_id: str, limit: int = Query(200, ge=1, le=2000),
                   offset: int = Query(0, ge=0),
                   svc: PerceptOSService = Depends(get_runtime)):
        if svc.store.get_mission(mission_id) is None:
            raise HTTPException(404, f"no mission {mission_id!r}")
        return {"frames": svc.store.get_frames(mission_id, limit=limit, offset=offset)}

    @app.get("/api/missions/{mission_id}/frames/{frame_index}", tags=["missions"])
    def get_frame(mission_id: str, frame_index: int,
                  svc: PerceptOSService = Depends(get_runtime)):
        detail = svc.store.get_frame_detail(mission_id, frame_index)
        if detail is None:
            raise HTTPException(404, f"no frame {frame_index} in mission {mission_id!r}")
        return detail

    @app.get("/api/missions/{mission_id}/telemetry/{channel}", tags=["missions"])
    def get_telemetry(mission_id: str, channel: str,
                      limit: int = Query(500, ge=1, le=5000),
                      svc: PerceptOSService = Depends(get_runtime)):
        return {"channel": channel,
                "samples": svc.store.get_telemetry(mission_id, channel, limit=limit)}

    @app.delete("/api/missions/{mission_id}", status_code=204, tags=["missions"])
    def delete_mission(mission_id: str, svc: PerceptOSService = Depends(get_runtime)):
        if not svc.store.delete_mission(mission_id):
            raise HTTPException(404, f"no mission {mission_id!r}")

    # ------------------------------------------------------------ perception

    def _png(result) -> Response:
        return Response(content=result.png, media_type="image/png",
                        headers=result.headers())

    @app.get("/api/frame", tags=["perception"])
    def frame_slots(svc: PerceptOSService = Depends(get_runtime)):
        """What the last processed frame produced, and how to render each part."""
        return svc.frame_slots()

    # Declared before the parameterised /api/frame/{slot}.png route:
    # Starlette matches in declaration order, so a literal path defined
    # after a matching parameterised one is unreachable.
    @app.get("/api/frame/pointcloud.png", tags=["perception"],
             responses={200: {"content": {"image/png": {}}}})
    def point_cloud(elevation_deg: float = Query(22.0, ge=-89.0, le=89.0),
                    azimuth_deg: float = Query(-35.0, ge=-180.0, le=180.0),
                    subsample: int = Query(2, ge=1, le=16),
                    svc: PerceptOSService = Depends(get_runtime)):
        """Back-project the current depth map into a metric point cloud."""
        from ..stereo.depth import reproject_to_3d

        depth = svc.slot_value("depth")
        if depth is None:
            raise HTTPException(
                404,
                "No depth map. A point cloud requires RECTIFIED_STEREO; run a "
                "stereo pipeline (serve.py --source stereo). PerceptOS will "
                "not reconstruct geometry it has not measured.",
            )
        rig_value = svc.slot_value("stereo_rig")
        if rig_value is None:
            raise HTTPException(
                409, "depth exists but no stereo rig is on the context, so the "
                     "reconstruction scale is unknown")
        cloud = reproject_to_3d(depth.value.depth_m, rig_value.value,
                                subsample=subsample)
        result = render_point_cloud(cloud, elevation_deg=elevation_deg,
                                    azimuth_deg=azimuth_deg)
        response = _png(result)
        response.headers["X-PerceptOS-Node-Id"] = depth.id
        response.headers["X-PerceptOS-Source"] = depth.provenance.source.value
        if depth.provenance.source_type:
            response.headers["X-PerceptOS-Source-Type"] = depth.provenance.source_type
        return response

    @app.get("/api/frame/{slot}.png", tags=["perception"],
             responses={200: {"content": {"image/png": {}}}})
    def frame_render(slot: str, max_dimension: int = Query(900, ge=64, le=4000),
                     svc: PerceptOSService = Depends(get_runtime)):
        """Render one slot of the last frame as a PNG.

        Response headers carry the colormap, the value range used for
        normalisation and the fraction of valid cells, so an ``<img>`` tag is
        still auditable rather than being decoration.
        """
        value = svc.slot_value(slot)
        if value is None:
            raise HTTPException(
                404,
                f"slot {slot!r} is UNAVAILABLE. Either no frame has been "
                "processed, or this pipeline does not produce that slot. "
                "GET /api/frame lists what exists.",
            )
        spec = svc.RENDERABLE.get(slot, {})
        try:
            payload = value.value
            if slot == "features":
                frame = svc.slot_value("frame") or svc.slot_value("gray")
                if frame is None:
                    raise HTTPException(404, "no frame to overlay features on")
                result = render_keypoints(frame.value, payload.points,
                                          max_dimension=max_dimension)
            elif slot == "tracks":
                frame = svc.slot_value("frame")
                if frame is None:
                    raise HTTPException(404, "no frame to overlay tracks on")
                result = render_tracks(frame.value, payload,
                                       max_dimension=max_dimension)
            elif slot == "depth":
                result = render_array(payload.depth_m, colormap="viridis",
                                      units="m", max_dimension=max_dimension,
                                      note="invalid pixels shown in magenta")
            elif slot == "disparity":
                result = render_array(payload.disparity, colormap="inferno",
                                      units="px", max_dimension=max_dimension)
            else:
                result = render_array(
                    np.asarray(payload),
                    colormap=spec.get("colormap", "grey") if spec.get("colormap") != "rgb"
                    else "grey",
                    units=spec.get("units"), max_dimension=max_dimension,
                )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(422, f"slot {slot!r} cannot be rendered: {exc}") from exc

        response = _png(result)
        response.headers["X-PerceptOS-Node-Id"] = value.id
        response.headers["X-PerceptOS-Source"] = value.provenance.source.value
        if value.provenance.source_type:
            response.headers["X-PerceptOS-Source-Type"] = value.provenance.source_type
        return response

    @app.get("/api/stereo", tags=["perception"])
    def stereo_status(svc: PerceptOSService = Depends(get_runtime)):
        """Stereo availability for the current frame, with an explicit verdict.

        Reports CALIBRATION_UNAVAILABLE rather than silently offering depth
        when no camera model backs the pair.
        """
        slots = svc.frame_slots()
        if not slots["available"]:
            return {"available": False, "stereo_status": "UNAVAILABLE",
                    "depth_valid": False, "reason": slots["reason"]}
        keys = {s["key"] for s in slots["slots"]}
        meta = slots.get("metadata", {}) or {}
        stereo_meta = meta.get("stereo", {}) or {}
        has_pair = "gray_right" in keys
        rectified = bool(stereo_meta.get("rectified", False))
        depth = svc.slot_value("depth")

        if not has_pair:
            status, reason = "UNAVAILABLE", (
                "This pipeline has no right-eye source, so there is no stereo "
                "pair. Configure a StereoAcquisitionStage to enable it.")
        elif not rectified:
            status, reason = "UNRECTIFIED_PAIR", (
                "A stereo pair is present but not epipolar-rectified. Block "
                "matching along scanlines would be invalid, so depth is not "
                "offered.")
        elif depth is None:
            status, reason = "VALID_STEREO", "Rectified pair; depth not computed."
        else:
            status, reason = "RECTIFIED_STEREO", "Rectified pair with depth."

        return {
            "available": has_pair,
            "stereo_status": status,
            "depth_valid": status == "RECTIFIED_STEREO",
            "rectified": rectified,
            "reason": reason,
            "left_slot": "gray" if has_pair else None,
            "right_slot": "gray_right" if has_pair else None,
            "depth_node_id": depth.id if depth is not None else None,
            "pair_metadata": stereo_meta,
        }

    # --------------------------------------------------------------- terrain

    @app.get("/api/terrain", tags=["terrain"])
    def terrain(allow_simulation: bool = False, force: bool = False,
                svc: PerceptOSService = Depends(get_runtime)):
        """Terrain products from a real DEM, or an explicit UNAVAILABLE."""
        return svc.analyse_terrain(allow_simulation=allow_simulation,
                                   force=force).to_dict()

    # Declared before /api/terrain/{layer}.png: Starlette matches routes
    # in declaration order, so these literal paths must come first or the
    # parameterised one swallows them. Same trap as pointcloud.png.
    @app.get("/api/terrain/hillshade.png", tags=["terrain"],
             responses={200: {"content": {"image/png": {}}}})
    def terrain_hillshade(allow_simulation: bool = False,
                          azimuth_deg: float = Query(315.0, ge=0.0, le=360.0),
                          altitude_deg: float = Query(45.0, ge=1.0, le=89.0),
                          vertical_exaggeration: float = Query(1.0, ge=0.1, le=20.0),
                          max_dimension: int = Query(900, ge=64, le=4000),
                          svc: PerceptOSService = Depends(get_runtime)):
        """Shaded relief. Shows surface *shape*, which a colour ramp cannot."""
        products = svc.analyse_terrain(allow_simulation=allow_simulation)
        if not products.available:
            raise HTTPException(404, products.reason)
        result = render_hillshade(
            products.dem.value, cell_size_m=products.cell_size_m,
            azimuth_deg=azimuth_deg, altitude_deg=altitude_deg,
            vertical_exaggeration=vertical_exaggeration,
            max_dimension=max_dimension)
        response = _png(result)
        response.headers["X-PerceptOS-Source-Type"] = products.status.value
        response.headers["X-PerceptOS-Node-Id"] = products.dem.id
        return response

    @app.get("/api/terrain/view3d.png", tags=["terrain"],
             responses={200: {"content": {"image/png": {}}}})
    def terrain_view_3d(allow_simulation: bool = False,
                        drape: str = Query("traversability"),
                        azimuth_deg: float = Query(-40.0, ge=-180.0, le=180.0),
                        elevation_deg: float = Query(28.0, ge=1.0, le=89.0),
                        vertical_exaggeration: float = Query(2.0, ge=0.1, le=20.0),
                        show_path: bool = True,
                        svc: PerceptOSService = Depends(get_runtime)):
        """Perspective terrain view, optionally draped and with the planned route.

        The vertical exaggeration is a display choice and travels in the
        response headers, because it changes how steep the terrain looks
        without changing what it is.
        """
        products = svc.analyse_terrain(allow_simulation=allow_simulation)
        if not products.available:
            raise HTTPException(404, products.reason)

        overlay, colormap = (None, "hazard")
        if drape and drape.lower() not in ("none", "elevation"):
            overlay, colormap, _ = products.layer(drape)
            if overlay is None:
                raise HTTPException(
                    404, f"drape layer {drape!r} is UNAVAILABLE. "
                         "GET /api/terrain lists the products that exist.")

        state = svc.navigation.state
        path = state.plan.cells if (show_path and state is not None) else None
        rover = state.start if (show_path and state is not None) else None
        try:
            result = render_terrain_3d(
                products.dem.value, cell_size_m=products.cell_size_m,
                overlay=np.asarray(overlay) if overlay is not None else None,
                overlay_colormap=colormap, path_cells=path, rover_cell=rover,
                azimuth_deg=azimuth_deg, elevation_deg=elevation_deg,
                vertical_exaggeration=vertical_exaggeration)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        response = _png(result)
        response.headers["X-PerceptOS-Source-Type"] = products.status.value
        response.headers["X-PerceptOS-Node-Id"] = products.dem.id
        response.headers["X-PerceptOS-Vertical-Exaggeration"] = str(vertical_exaggeration)
        return response

    @app.get("/api/terrain/{layer}.png", tags=["terrain"],
             responses={200: {"content": {"image/png": {}}}})
    def terrain_render(layer: str, allow_simulation: bool = False,
                       max_dimension: int = Query(900, ge=64, le=4000),
                       svc: PerceptOSService = Depends(get_runtime)):
        products = svc.analyse_terrain(allow_simulation=allow_simulation)
        if not products.available:
            raise HTTPException(404, products.reason)
        array, colormap, units = products.layer(layer)
        if array is None:
            raise HTTPException(
                404, f"terrain layer {layer!r} is UNAVAILABLE. "
                     "GET /api/terrain lists the products that exist.")
        result = render_array(np.asarray(array), colormap=colormap, units=units,
                              max_dimension=max_dimension)
        response = _png(result)
        response.headers["X-PerceptOS-Source-Type"] = products.status.value
        return response

    # ------------------------------------------------------------ navigation

    @app.get("/api/navigation", tags=["navigation"])
    def navigation(svc: PerceptOSService = Depends(get_runtime)):
        """The active plan, or UNAVAILABLE with the reason."""
        state = svc.navigation.state
        if state is None:
            terrain_ready = (svc.terrain.products is not None
                             and svc.terrain.products.available)
            return {
                "available": False, "status": "UNAVAILABLE",
                "reason": ("No route has been planned. POST /api/navigation/plan "
                           + ("to plan one." if terrain_ready
                              else "after terrain becomes available.")),
                "terrain_ready": terrain_ready,
            }
        return {"available": True, "status": "OK", **state.to_dict()}

    @app.post("/api/navigation/plan", tags=["navigation"])
    def navigation_plan(body: PlanRequest = Body(default=PlanRequest()),
                        allow_simulation: bool = False,
                        svc: PerceptOSService = Depends(get_runtime)):
        products = svc.analyse_terrain(allow_simulation=allow_simulation)
        if not products.available:
            raise HTTPException(409, products.reason)
        try:
            state = svc.navigation.plan(
                products, planner=body.planner,
                start=tuple(body.start) if body.start else None,
                goal=tuple(body.goal) if body.goal else None,
                cost_weight=body.cost_weight,
                rover_radius_m=body.rover_radius_m,
            )
        except PlanningFailure as exc:
            raise HTTPException(409, f"{exc} | {exc.diagnostics}") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"available": True, "status": "OK", **state.to_dict()}

    @app.post("/api/navigation/replan", tags=["navigation"])
    def navigation_replan(body: ReplanRequest,
                          svc: PerceptOSService = Depends(get_runtime)):
        """Apply newly observed cell costs and repair the plan incrementally."""
        changes = [((int(c[0]), int(c[1])), float(c[2])) for c in body.changes
                   if len(c) >= 3]
        if not changes:
            raise HTTPException(400, "changes must be [[row, col, cost], ...]")
        try:
            event = svc.navigation.replan(changes)
        except PlanningFailure as exc:
            raise HTTPException(409, f"{exc} | {exc.diagnostics}") from exc
        return {"event": event, "plan": svc.navigation.state.to_dict()}

    @app.get("/api/navigation/path.png", tags=["navigation"],
             responses={200: {"content": {"image/png": {}}}})
    def navigation_render(max_dimension: int = Query(900, ge=64, le=4000),
                          svc: PerceptOSService = Depends(get_runtime)):
        state = svc.navigation.state
        products = svc.terrain.products
        if state is None or products is None or not products.available:
            raise HTTPException(404, "No planned route to render. "
                                     "POST /api/navigation/plan first.")
        result = render_path(products.traversability.value.cost, state.plan.cells,
                             max_dimension=max_dimension)
        response = _png(result)
        response.headers["X-PerceptOS-Source-Type"] = products.status.value
        return response

    # ----------------------------------------------------------------- slam

    @app.get("/api/slam", tags=["slam"])
    def slam(svc: PerceptOSService = Depends(get_runtime)):
        """Trajectory, keyframes, landmarks and loop closures."""
        stage = None
        if svc.pipeline is not None:
            stage = svc.pipeline.stage("slam")
        if stage is None:
            return {
                "available": False, "status": "UNAVAILABLE",
                "reason": ("This pipeline has no SLAM stage. Add a SlamStage to "
                           "the pipeline; SLAM needs an odometry or fused-pose "
                           "input to seed keyframe poses."),
            }
        snapshot = stage.system.snapshot()
        return {"available": True, "status": "OK", **snapshot.to_dict()}

    @app.get("/api/slam/trajectory.png", tags=["slam"],
             responses={200: {"content": {"image/png": {}}}})
    def slam_trajectory(svc: PerceptOSService = Depends(get_runtime)):
        """Top-down trajectory with landmarks and loop-closure edges."""
        stage = svc.pipeline.stage("slam") if svc.pipeline else None
        if stage is None:
            raise HTTPException(
                404, "This pipeline has no SLAM stage, so there is no "
                     "trajectory to plot.")
        snapshot = stage.system.snapshot()
        if not snapshot.trajectory:
            raise HTTPException(404, "SLAM has produced no keyframes yet.")
        result = render_trajectory(snapshot.trajectory,
                                   landmarks=snapshot.landmarks,
                                   loop_closures=snapshot.loop_closures)
        return _png(result)

    @app.post("/api/slam/optimise", tags=["slam"])
    def slam_optimise(svc: PerceptOSService = Depends(get_runtime)):
        stage = svc.pipeline.stage("slam") if svc.pipeline else None
        if stage is None:
            raise HTTPException(409, "no SLAM stage in this pipeline")
        return stage.system.optimise().to_dict()

    # --------------------------------------------------------------- replay

    @app.post("/api/replay/{mission_id}/open", tags=["replay"])
    def replay_open(mission_id: str, svc: PerceptOSService = Depends(get_runtime)):
        """Open a recorded mission for playback."""
        try:
            return app.state.replay.open(mission_id).to_dict()
        except KeyError as exc:
            raise HTTPException(404, str(exc).strip("'")) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/replay", tags=["replay"])
    def replay_state():
        session = app.state.replay.session
        if session is None:
            return {"open": False, "status": "UNAVAILABLE",
                    "reason": "No replay session. POST /api/replay/{id}/open."}
        return {"open": True, "status": "OK", **session.to_dict()}

    @app.post("/api/replay/seek", tags=["replay"])
    def replay_seek(body: ReplaySeek):
        return _replay_guard(lambda: app.state.replay.seek(body.position).to_dict())

    @app.post("/api/replay/step", tags=["replay"])
    def replay_step(delta: int = Query(1, ge=-1000, le=1000)):
        return _replay_guard(lambda: app.state.replay.step(delta).to_dict())

    @app.post("/api/replay/play", tags=["replay"])
    def replay_play(speed: float = Query(1.0, gt=0.0, le=16.0)):
        return _replay_guard(lambda: app.state.replay.play(speed=speed).to_dict())

    @app.post("/api/replay/pause", tags=["replay"])
    def replay_pause():
        return _replay_guard(lambda: app.state.replay.pause().to_dict())

    @app.get("/api/replay/frame", tags=["replay"])
    def replay_frame(index: Optional[int] = None):
        return _replay_guard(lambda: app.state.replay.frame(index))

    @app.get("/api/replay/timeline", tags=["replay"])
    def replay_timeline():
        return _replay_guard(lambda: app.state.replay.timeline())

    @app.get("/api/replay/profile", tags=["replay"])
    def replay_profile():
        return _replay_guard(lambda: app.state.replay.stage_profile())

    @app.post("/api/replay/counterfactual", tags=["replay"])
    def replay_counterfactual(body: CounterfactualRequest,
                              svc: PerceptOSService = Depends(get_runtime)):
        """Re-plan the current terrain with different parameters and diff.

        Answers what a different *planner* would have decided given the same
        perception. It cannot answer what a different perception pipeline would
        have seen — those frames were never captured.
        """
        products = svc.terrain.products
        state = svc.navigation.state
        if products is None or not products.available or state is None:
            raise HTTPException(
                409, "A counterfactual needs a recorded terrain decision. "
                     "Plan a route first (POST /api/navigation/plan).")
        baseline = {**state.plan.to_dict(),
                    "planner": state.planner, "planned": True,
                    **state.plan_node.provenance.parameters}
        try:
            result = app.state.replay.counterfactual_plan(
                products.traversability.value.cost,
                start=state.start, goal=state.goal,
                cell_size_m=products.cell_size_m, baseline=baseline,
                planner=body.planner, cost_weight=body.cost_weight,
                inflation_radius_m=body.inflation_radius_m,
            )
        except ValueError as exc:
            # A bad planner name is a client error, not a server fault; letting
            # it escape produced a 500 with a stack trace instead of the list
            # of valid planners.
            raise HTTPException(400, str(exc)) from exc
        return result.to_dict()

    # --------------------------------------------------------------- reports

    @app.get("/api/missions/{mission_id}/report", tags=["reports"])
    def mission_report(mission_id: str, format: str = Query("json"),
                       svc: PerceptOSService = Depends(get_runtime)):
        """Mission report as JSON, CSV or Markdown, with lineage references."""
        try:
            report = ReportBuilder(svc.store).build(mission_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc).strip("'")) from exc
        fmt = format.lower()
        if fmt == "json":
            return report.payload
        if fmt == "csv":
            return PlainTextResponse(
                report.to_csv(), media_type="text/csv",
                headers={"Content-Disposition":
                         f'attachment; filename="mission-{mission_id}.csv"'})
        if fmt in ("md", "markdown"):
            return PlainTextResponse(
                report.to_markdown(), media_type="text/markdown",
                headers={"Content-Disposition":
                         f'attachment; filename="mission-{mission_id}.md"'})
        raise HTTPException(400, f"unknown format {format!r}; "
                                 "use json, csv or markdown")

    # --------------------------------------------------------------- datasets

    @app.get("/api/datasets", tags=["datasets"])
    def dataset_versions():
        """Every stored dataset version, newest first."""
        versions = app.state.datasets.list_versions()
        return {"versions": versions, "count": len(versions),
                "root": str(app.state.datasets.root)}

    @app.post("/api/datasets/ingest", status_code=201, tags=["datasets"])
    def dataset_ingest(body: IngestRequest):
        """Build a content-addressed version from a directory of images."""
        try:
            version = app.state.datasets.ingest_directory(
                body.directory, name=body.name, pattern=body.pattern,
                group_from=body.group_from)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        app.state.datasets.save(version)
        return {**version.statistics(),
                "group_counts": version.group_counts()}

    @app.get("/api/datasets/{version_id}", tags=["datasets"])
    def dataset_version(version_id: str):
        try:
            version = app.state.datasets.load(version_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc).strip("'")) from exc
        except ValueError as exc:
            # An edited manifest: the recorded id no longer names this data.
            raise HTTPException(409, str(exc)) from exc
        return {**version.statistics(), "group_counts": version.group_counts()}

    @app.post("/api/datasets/{version_id}/split", tags=["datasets"])
    def dataset_split(version_id: str, body: SplitRequest = Body(default=SplitRequest())):
        """Partition a version, grouped by default.

        A random split over sequential rover imagery is refused unless the
        caller explicitly asserts the samples are independent, because it puts
        near-duplicate frames in both train and test.
        """
        try:
            version = app.state.datasets.load(version_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc).strip("'")) from exc
        try:
            result = version.split(
                train=body.train, validation=body.validation, test=body.test,
                group_key=body.group_key, seed=body.seed,
                allow_random=body.allow_random)
        except LeakageError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return result.to_dict()

    # ----------------------------------------------------------------- models

    @app.get("/api/models", tags=["models"])
    def models():
        registered = app.state.models.list_models()
        return {"models": registered, "count": len(registered),
                "note": ("PerceptOS records and evaluates models; it does not "
                         "train them. Training belongs to the model's own "
                         "framework.")}

    @app.get("/api/models/{model_id}", tags=["models"])
    def model_detail(model_id: str):
        try:
            return app.state.models.get(model_id).to_dict()
        except KeyError as exc:
            raise HTTPException(404, str(exc).strip("'")) from exc

    @app.get("/api/models/{model_id}/card", tags=["models"],
             response_class=PlainTextResponse)
    def model_card(model_id: str) -> str:
        """Model card in Markdown: training data, metrics and limitations."""
        try:
            record = app.state.models.get(model_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc).strip("'")) from exc
        return record.model_card()

    @app.get("/api/models/compare", tags=["models"])
    def model_compare(ids: str = Query(..., description="comma-separated model ids")):
        model_ids = [i.strip() for i in ids.split(",") if i.strip()]
        if len(model_ids) < 2:
            raise HTTPException(400, "provide at least two model ids")
        try:
            return app.state.models.compare(model_ids)
        except KeyError as exc:
            raise HTTPException(404, str(exc).strip("'")) from exc

    # --------------------------------------------------------------- lineage

    @app.get("/api/lineage/{node_id}", tags=["lineage"])
    def explain(node_id: str, mission_id: Optional[str] = None,
                svc: PerceptOSService = Depends(get_runtime)):
        """The Data Lineage drawer. Why does this value have the value it has?"""
        payload = svc.explain(node_id, mission_id=mission_id)
        if payload is None:
            raise HTTPException(
                404,
                f"no lineage node {node_id!r}. It may belong to an earlier "
                "mission - pass ?mission_id= to search a recorded one.",
            )
        return payload

    @app.get("/api/lineage", tags=["lineage"])
    def search_lineage(mission_id: Optional[str] = None,
                       label: Optional[str] = None,
                       source: Optional[str] = None,
                       frame_index: Optional[int] = None,
                       limit: int = Query(100, ge=1, le=1000),
                       svc: PerceptOSService = Depends(get_runtime)):
        return {"nodes": svc.search_lineage(mission_id=mission_id, label=label,
                                            source=source, frame_index=frame_index,
                                            limit=limit)}

    # ---------------------------------------------------------------- health

    @app.get("/api/health", tags=["health"])
    def health(svc: PerceptOSService = Depends(get_runtime)):
        return svc.health()

    @app.get("/api/health/live", tags=["health"])
    def liveness() -> dict[str, str]:
        return {"status": "ok"}

    # ------------------------------------------------------------- websocket

    @app.websocket("/ws/telemetry")
    async def telemetry_socket(websocket: WebSocket) -> None:
        """Pushes each processed frame and periodic health to the UI."""
        await websocket.accept()
        svc: PerceptOSService = app.state.service
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)

        def on_message(message: dict[str, Any]) -> None:
            # The pipeline runs on a worker thread; hop back to the event loop.
            # A full queue drops the message rather than blocking the pipeline:
            # a slow client must never stall frame processing.
            loop.call_soon_threadsafe(_offer, message)

        def _offer(message: dict[str, Any]) -> None:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                pass

        unsubscribe = svc.subscribe(on_message)
        try:
            await websocket.send_json({"type": "hello",
                                       "payload": {"service": svc.state.to_dict(),
                                                   "data_mode": svc.data_mode}})
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    message = {"type": "health", "payload": svc.health()}
                await websocket.send_json(message)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            unsubscribe()

    return app


def _root_payload(svc: PerceptOSService, app: FastAPI) -> dict[str, Any]:
    """Service status. Every field is read from the running system."""
    import time

    routes = [r for r in app.routes if getattr(r, "include_in_schema", False)]
    lineage_live = len(svc.lineage)
    store_stats = svc.store.statistics()
    return {
        "service": "PerceptOS",
        "description": "Robotic perception, autonomy and planetary exploration platform",
        "version": app.version,
        "api_status": "ok",
        "timestamp_utc": time.time(),
        "environment": {
            "name": svc.environment.name,
            "display_name": svc.environment.display_name,
            "supervision": svc.environment.supervision,
        },
        "data_source": {
            "pipeline_mode": svc.data_mode,
            "primary_perseverance_service": "jpl_raw_images",
            "primary_service_requires_key": False,
            "nasa_key_status": credentials_report()["api_nasa_gov_key_status"],
        },
        "lineage": {
            "live_nodes": lineage_live,
            "persisted_nodes": store_stats["lineage_nodes"],
            "persisted_edges": store_stats["lineage_edges"],
            "status": "active" if lineage_live or store_stats["lineage_nodes"]
                      else "empty",
        },
        "mission": {
            "running": svc.state.running,
            "mission_id": svc.state.mission_id,
            "frames_processed": svc.state.frames_processed,
        },
        "routes": {
            "count": len(routes),
            "documentation": "/docs",
            "openapi": "/openapi.json",
            "redoc": "/redoc",
        },
        "links": {
            "status": "/api/status",
            "datasources": "/api/datasources",
            "environments": "/api/environments",
            "pipeline": "/api/pipeline",
            "missions": "/api/missions",
            "lineage": "/api/lineage",
            "health": "/api/health",
            "console": "/console",
            "telemetry_websocket": "/ws/telemetry",
        },
    }


def _render_status_page(payload: dict[str, Any]) -> str:
    """Minimal dark status page. Every value comes from ``payload``."""
    env = payload["environment"]
    data = payload["data_source"]
    lineage = payload["lineage"]
    mission = payload["mission"]
    rows = [
        ("Version", payload["version"]),
        ("API status", payload["api_status"]),
        ("Environment", f"{env['display_name']} ({env['supervision']})"),
        ("Pipeline data mode", data["pipeline_mode"]),
        ("Perseverance source", f"{data['primary_perseverance_service']} "
                                f"(API key required: "
                                f"{data['primary_service_requires_key']})"),
        ("NASA key status", data["nasa_key_status"]),
        ("Lineage", f"{lineage['live_nodes']} live / "
                    f"{lineage['persisted_nodes']} persisted nodes "
                    f"({lineage['status']})"),
        ("Mission", f"running={mission['running']} "
                    f"frames={mission['frames_processed']}"),
        ("Routes", payload["routes"]["count"]),
    ]
    body = "\n".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows
    )
    links = "\n".join(
        f'<a href="{href}">{name}</a>' for name, href in payload["links"].items()
        if not href.startswith("ws")
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>PerceptOS</title><style>
:root{{color-scheme:dark}}
body{{background:#0a0e14;color:#c8d3e0;font:14px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;
margin:0;padding:48px 24px;display:flex;justify-content:center}}
main{{max-width:760px;width:100%}}
h1{{font-size:22px;letter-spacing:.18em;color:#e8eef6;margin:0 0 4px;font-weight:600}}
p.sub{{color:#5d6b7d;margin:0 0 28px}}
table{{width:100%;border-collapse:collapse;margin-bottom:28px}}
th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid #1b2430}}
th{{color:#5d6b7d;font-weight:400;width:220px}}
td{{color:#e8eef6}}
nav{{display:flex;flex-wrap:wrap;gap:8px}}
a{{color:#4da3d6;text-decoration:none;border:1px solid #1b2430;padding:6px 12px;border-radius:4px}}
a:hover{{border-color:#4da3d6;background:#0f1620}}
</style></head><body><main>
<h1>PERCEPTOS</h1>
<p class="sub">{payload["description"]}</p>
<table>{body}</table>
<nav>{links}</nav>
</main></body></html>"""


def get_service(app: FastAPI) -> PerceptOSService:
    """Access the runtime attached to an application."""
    return app.state.service
