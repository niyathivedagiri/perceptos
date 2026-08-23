"""
perceptos.api.analysis
======================

Terrain and navigation sessions for the mission-control console.

The perception pipeline runs per frame; terrain analysis and route planning do
not. They operate on an elevation model that changes rarely and cost seconds
rather than milliseconds. So they live here as *sessions*: computed once on
request, cached with their lineage, and served to the console until the inputs
change.

Both services reuse the existing modules unchanged —
:class:`~perceptos.mars.traversability.TraversabilityAnalyser`,
:class:`~perceptos.mars.occupancy.OccupancyGrid`,
:class:`~perceptos.mars.planner.AStarPlanner` and the Theta*/D* family. Nothing
here reimplements analysis; it wires real inputs into algorithms that are
already tested.

**The availability rule is enforced here, not documented.** If no DEM is on
disk, :meth:`TerrainService.analyse` reports ``UNAVAILABLE`` and returns no
products. It does not quietly substitute synthetic terrain so the panel has
something to draw. Synthetic terrain is available, but only when the caller
asks for it explicitly, and everything derived from it is tagged
``SIMULATION`` by the provenance layer with no way to launder it back.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..core.environment import Environment
from ..core.lineage import LineageGraph
from ..core.provenance import Source, Traced, traced
from ..data.sources import DataSourceStatus
from ..data.terrain import (
    denoise_labels,
    fuse_costs,
    load_dem_traced,
    semantic_cost_layer,
)
from ..mars.occupancy import GridConfig, OccupancyGrid
from ..mars.planner import AStarPlanner, PathPlan, PlanningFailure
from ..mars.planners import DStarLitePlanner, ThetaStarPlanner
from ..mars.traversability import TraversabilityAnalyser, TraversabilityResult

__all__ = ["TerrainProducts", "TerrainService", "NavigationService", "PLANNERS"]

PLANNERS = {"astar": AStarPlanner, "theta": ThetaStarPlanner}


@dataclass(slots=True)
class TerrainProducts:
    """Everything the terrain panel shows, each item traceable."""

    status: DataSourceStatus
    dem: Optional[Traced[np.ndarray]] = None
    traversability: Optional[Traced[TraversabilityResult]] = None
    occupancy: Optional[OccupancyGrid] = None
    semantic_cost: Optional[np.ndarray] = None
    fused_cost: Optional[Traced[np.ndarray]] = None
    cell_size_m: float = 0.25
    source_path: str = ""
    reason: str = ""
    computed_utc: float = field(default_factory=time.time)

    @property
    def available(self) -> bool:
        return self.dem is not None

    def to_dict(self) -> dict[str, Any]:
        if not self.available:
            return {
                "status": self.status.value,
                "available": False,
                "reason": self.reason,
                "products": [],
            }
        result = self.traversability.value if self.traversability else None
        products = [
            {"key": "elevation", "label": "Elevation (DEM)",
             "node_id": self.dem.id, "units": "m", "colormap": "viridis"},
        ]
        if result is not None:
            products += [
                {"key": "slope", "label": "Slope", "node_id": self.traversability.id,
                 "units": "deg", "colormap": "inferno"},
                {"key": "roughness", "label": "Roughness",
                 "node_id": self.traversability.id, "units": "m",
                 "colormap": "inferno"},
                {"key": "step", "label": "Step height",
                 "node_id": self.traversability.id, "units": "m",
                 "colormap": "inferno"},
                {"key": "traversability", "label": "Traversability cost",
                 "node_id": self.traversability.id, "units": "cost",
                 "colormap": "hazard"},
                {"key": "hazard", "label": "Hazard mask",
                 "node_id": self.traversability.id, "units": "bool",
                 "colormap": "hazard"},
            ]
        if self.occupancy is not None:
            # A DEM carries no ray evidence, so log-odds occupancy stays at its
            # 0.5 prior everywhere. Publishing that flat map as "occupancy"
            # would look like a computed result rather than an absence of one,
            # so what is offered is the elevation *coverage* — which is real —
            # with the limitation stated.
            products.append({
                "key": "coverage", "label": "Elevation observation coverage",
                "node_id": self.dem.id, "units": "observations",
                "colormap": "grey",
                "note": ("Cells with at least one elevation observation. "
                         "Ray-based occupancy is UNAVAILABLE from a DEM alone: "
                         "it needs a sensor origin, which an elevation model "
                         "does not provide."),
            })
        if self.fused_cost is not None:
            products.append({"key": "fused", "label": "Fused cost (geometric+semantic)",
                             "node_id": self.fused_cost.id, "units": "cost",
                             "colormap": "hazard"})

        payload: dict[str, Any] = {
            "status": self.status.value,
            "available": True,
            "source_path": self.source_path,
            "cell_size_m": self.cell_size_m,
            "shape": list(self.dem.value.shape),
            "computed_utc": self.computed_utc,
            "dem_node_id": self.dem.id,
            "dem_provenance": self.dem.provenance.to_dict(),
            "products": products,
        }
        if result is not None:
            payload["statistics"] = result.statistics()
            payload["traversability_node_id"] = self.traversability.id
            payload["traversability_confidence"] = self.traversability.confidence
        if self.occupancy is not None:
            payload["occupancy"] = {
                **self.occupancy.statistics(),
                "ray_occupancy_status": "UNAVAILABLE",
                "ray_occupancy_reason": (
                    "No range scan integrated; occupancy log-odds remain at the "
                    "0.5 prior. Elevation coverage is reported instead."
                ),
            }
        return payload

    def layer(self, key: str) -> tuple[Optional[np.ndarray], str, Optional[str]]:
        """Return ``(array, colormap, units)`` for a product key."""
        result = self.traversability.value if self.traversability else None
        if key == "elevation" and self.dem is not None:
            return self.dem.value, "viridis", "m"
        if result is None:
            return None, "viridis", None
        if key == "slope":
            return result.slope_deg, "inferno", "deg"
        if key == "roughness":
            return result.roughness_m, "inferno", "m"
        if key == "step":
            return result.step_m, "inferno", "m"
        if key == "traversability":
            return result.cost, "hazard", "cost"
        if key == "hazard":
            return result.hazard_mask.astype(np.float32), "hazard", "bool"
        if key == "coverage" and self.occupancy is not None:
            return (self.occupancy.observation_count.astype(np.float32),
                    "grey", "observations")
        if key == "occupancy" and self.occupancy is not None:
            # Kept reachable for a pipeline that *does* cast rays; from a DEM
            # this is the uniform prior and the note above explains why.
            return self.occupancy.probability_map(), "grey", "probability"
        if key == "fused" and self.fused_cost is not None:
            return self.fused_cost.value, "hazard", "cost"
        return None, "viridis", None


class TerrainService:
    """Computes terrain products from a real DEM, or reports why it cannot."""

    #: Where a cached DEM tile is expected. Populated by scripts/fetch_data.py.
    DEM_GLOBS = ("*_dem/*.npy", "*dem*/*.npy", "jezero*/*.npy")

    def __init__(self, *, datasets_root: str | Path = "datasets") -> None:
        self.root = Path(datasets_root)
        self.products: Optional[TerrainProducts] = None
        self.lineage = LineageGraph(artefact_capacity=64)

    def find_dem(self) -> Optional[Path]:
        """Locate a cached DEM tile. Presence of the file is the mode switch."""
        if not self.root.exists():
            return None
        for pattern in self.DEM_GLOBS:
            for candidate in sorted(self.root.glob(pattern)):
                return candidate
        return None

    def analyse(self, environment: Environment, *,
                allow_simulation: bool = False,
                max_cells: int = 400) -> TerrainProducts:
        """Run the full terrain chain over the best available elevation data.

        ``allow_simulation`` must be passed explicitly. The default is to
        report UNAVAILABLE rather than draw a synthetic map that an operator
        might read as a survey.
        """
        self.lineage = LineageGraph(artefact_capacity=64)
        dem_path = self.find_dem()

        if dem_path is not None:
            dem_tv = load_dem_traced(dem_path, cell_size_m=environment.cell_size_m)
            status = DataSourceStatus.LOCAL_CACHED_DATA
            source_path = str(dem_path)
        elif allow_simulation:
            dem_tv = traced(
                self._synthetic_dem(environment.cell_size_m),
                source=Source.SIMULATION,
                algorithm="synthetic_terrain(crater + boulder field)",
                confidence=1.0, units="m", label="dem_elevation",
                parameters={"cell_size_m": environment.cell_size_m, "seed": 42},
            )
            status = DataSourceStatus.SIMULATION
            source_path = "(generated in-process)"
        else:
            self.products = TerrainProducts(
                status=DataSourceStatus.UNAVAILABLE,
                reason=(
                    "No cached digital elevation model found under "
                    f"{self.root}/. Fetch one with:\n"
                    "    python scripts/fetch_data.py jezero-dem\n"
                    "PerceptOS will not substitute synthetic terrain for a "
                    "missing survey. Request ?allow_simulation=true to work "
                    "with an explicitly labelled synthetic DEM instead."
                ),
            )
            return self.products

        # Large DEMs are decimated before analysis: a 4000x4000 tile is 16
        # million cells and the console needs an answer, not a coffee break.
        dem = dem_tv.value
        if max(dem.shape) > max_cells:
            factor = int(math.ceil(max(dem.shape) / max_cells))
            dem_tv = dem_tv.derive(
                _block_mean(dem, factor),
                algorithm=f"block_mean_downsample(factor={factor})",
                source=dem_tv.source, units="m", label="dem_elevation",
                parameters={"factor": factor,
                            "cell_size_m": environment.cell_size_m * factor},
            )
        cell_size = environment.cell_size_m * (
            int(math.ceil(max(dem.shape) / max_cells)) if max(dem.shape) > max_cells else 1
        )
        self.lineage.record(dem_tv, stage="terrain")

        analyser = TraversabilityAnalyser(environment.rover, cell_size_m=cell_size,
                                          window_cells=3)
        traversability = analyser.analyse_traced(dem_tv)
        self.lineage.record(traversability, stage="traversability")

        occupancy = self._build_occupancy(dem_tv.value, cell_size)

        self.products = TerrainProducts(
            status=status, dem=dem_tv, traversability=traversability,
            occupancy=occupancy, cell_size_m=cell_size, source_path=source_path,
        )
        return self.products

    def _build_occupancy(self, dem: np.ndarray, cell_size: float) -> OccupancyGrid:
        """Fuse the DEM into an occupancy grid using the existing module."""
        rows, cols = dem.shape
        grid = OccupancyGrid(GridConfig(width_cells=cols, height_cells=rows,
                                        cell_size_m=cell_size))
        yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64) * cell_size
        finite = np.isfinite(dem)
        cloud = np.stack([xx[finite], yy[finite], dem[finite]], axis=1)
        if cloud.size:
            grid.integrate_points(cloud)
        return grid

    @staticmethod
    def _synthetic_dem(cell_size_m: float, size: int = 160,
                       seed: int = 42) -> np.ndarray:
        """SIMULATION ONLY. Tagged by the caller, never presented as a survey."""
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) * cell_size_m
        dem = 0.04 * xx
        r = np.hypot(yy - size * cell_size_m * 0.35, xx - size * cell_size_m * 0.3)
        dem += np.where(r < 6.0, -1.4 * (1 - (r / 6.0) ** 2), 0.0)
        dem += 0.7 * np.exp(-((r - 6.0) ** 2) / 1.2)
        for _ in range(30):
            by, bx = rng.uniform(0, size * cell_size_m, 2)
            s, h = rng.uniform(0.3, 0.9), rng.uniform(0.25, 0.8)
            dem += h * np.exp(-(((yy - by) ** 2 + (xx - bx) ** 2) / (2 * s ** 2)))
        return dem + rng.normal(0.0, 0.012, dem.shape)


@dataclass(slots=True)
class NavigationState:
    """A planned traverse and the events that shaped it."""

    plan: PathPlan
    plan_node: Traced[PathPlan]
    planner: str
    start: tuple[int, int]
    goal: tuple[int, int]
    cell_size_m: float
    replans: list[dict[str, Any]] = field(default_factory=list)
    computed_utc: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        payload = self.plan.to_dict()
        payload.update({
            "planner": self.planner,
            "start_cell": list(self.start),
            "goal_cell": list(self.goal),
            "cell_size_m": self.cell_size_m,
            "plan_node_id": self.plan_node.id,
            "provenance": self.plan_node.provenance.to_dict(),
            "replans": self.replans,
            "computed_utc": self.computed_utc,
        })
        return payload


class NavigationService:
    """Plans routes over terrain products and records replanning events."""

    def __init__(self) -> None:
        self.state: Optional[NavigationState] = None
        self.dstar: Optional[DStarLitePlanner] = None
        self.lineage = LineageGraph(artefact_capacity=32)
        # Inputs needed to build the D* Lite instance, held until the first
        # replan actually asks for one.
        self._dstar_seed: Optional[dict[str, Any]] = None

    def plan(self, products: TerrainProducts, *, planner: str = "theta",
             start: Optional[tuple[int, int]] = None,
             goal: Optional[tuple[int, int]] = None,
             cost_weight: float = 4.0,
             inflation_radius_m: Optional[float] = None,
             rover_radius_m: float = 0.5) -> NavigationState:
        """Plan a route. Raises :class:`PlanningFailure` with diagnostics."""
        if not products.available or products.traversability is None:
            raise PlanningFailure(
                "cannot plan without terrain",
                diagnostics={"terrain_status": products.status.value,
                             "reason": products.reason},
            )
        key = planner.lower()
        if key not in PLANNERS:
            raise ValueError(f"unknown planner {planner!r}; "
                             f"available: {sorted(PLANNERS)} (and 'dstar' for replanning)")

        cost = products.traversability.value.cost
        rows, cols = cost.shape
        start = tuple(start) if start else self._default_start(products,
                                                                rover_radius_m)
        # Only auto-select when the caller did not specify. An explicit goal is
        # honoured even if it fails — silently moving a commanded destination
        # would be worse than reporting that it is unreachable.
        auto_goal = goal is None
        goal = tuple(goal) if goal else self._default_goal(products, start,
                                                            rover_radius_m)

        engine = PLANNERS[key](
            cost_weight=cost_weight,
            inflation_radius_m=(inflation_radius_m if inflation_radius_m is not None
                                else rover_radius_m),
        )
        plan = engine.plan(cost, start, goal, cell_size_m=products.cell_size_m)
        plan_node = products.traversability.derive(
            plan, algorithm=f"{type(engine).__name__}",
            source=Source.CALCULATED, units="m", label="safe_path_plan",
            parameters={"planner": key, "start": list(start), "goal": list(goal),
                        "cost_weight": cost_weight, "goal_auto_selected": auto_goal},
        )
        self.lineage.record(products.dem, stage="terrain")
        self.lineage.record(products.traversability, stage="traversability")
        self.lineage.record(plan_node, stage="planning")

        # D* Lite is built lazily. Its initialisation is a full backward
        # search over the whole grid — on a 160x160 map that is ~2.5 s, and
        # paying it on every plan request when most callers never replan makes
        # planning feel broken. The inputs are kept; the search waits until a
        # replan is genuinely requested.
        self.dstar = None
        self._dstar_seed = {
            "cost": cost.copy(), "start": start, "goal": goal,
            "cell_size_m": products.cell_size_m, "cost_weight": cost_weight,
            "rover_radius_m": rover_radius_m,
        }

        self.state = NavigationState(
            plan=plan, plan_node=plan_node, planner=key, start=start, goal=goal,
            cell_size_m=products.cell_size_m,
        )
        return self.state

    @staticmethod
    def _safe_cells(products: TerrainProducts, *, engine: Any = None,
                    rover_radius_m: float = 0.5) -> np.ndarray:
        """Cells the planner will actually accept.

        Two subtleties, both of which produced wrong answers before being
        handled:

        **The planner's definition of lethal is not the analyser's.**
        ``TraversabilityResult.hazard_mask`` marks cells that breach a vehicle
        limit outright — on a real Jezero tile about 2% of cells. The planners
        block on ``cost >= hazard_threshold`` (0.95), which additionally
        catches cells whose *weighted severity* saturates without any single
        limit being breached — about 6% of the same tile. Selecting a goal
        against the smaller set yields cells the planner then refuses. The
        planner's own criterion is authoritative here.

        **The rover is not a point.** Lethal cells are dilated by the
        circumscribed radius before searching, so a cell can be individually
        legal and still unreachable. Reusing the planner's ``inflate`` makes
        the two agree by construction rather than by coincidence.
        """
        result = products.traversability.value
        planner = engine or AStarPlanner(inflation_radius_m=rover_radius_m)
        lethal = (result.cost >= planner.hazard_threshold) | ~np.isfinite(result.cost)
        blocked, _ = planner.inflate(lethal, products.cell_size_m)
        return ~blocked

    @classmethod
    def _default_start(cls, products: TerrainProducts,
                       rover_radius_m: float = 0.5) -> tuple[int, int]:
        """A start inside the largest traversable region.

        Taking the cell nearest the corner is the obvious choice and a poor
        one: on fragmented terrain the corner often sits in a small pocket
        walled off from the rest of the map, and every auto-planned route is
        then a few metres long. Seeding inside the largest connected region
        gives a route that actually crosses the terrain.
        """
        safe = cls._safe_cells(products, rover_radius_m=rover_radius_m)
        rows, cols = safe.shape
        if not safe.any():
            return (2, 2)

        best_seed, best_size, unvisited = None, 0, safe.copy()
        # Probe on a coarse lattice: flooding from every safe cell is O(n^2)
        # on an open map, and a lattice step of ~1/16 of the map cannot miss a
        # component large enough to be worth driving across.
        step = max(1, min(rows, cols) // 16)
        for r in range(0, rows, step):
            for c in range(0, cols, step):
                if not unvisited[r, c]:
                    continue
                component = cls._reachable_from(safe, (r, c))
                size = int(component.sum())
                unvisited &= ~component
                if size > best_size:
                    best_seed, best_size = (r, c), size
        if best_seed is None:
            index = int(np.argmax(safe))
            return (index // cols, index % cols)

        # Within the winning region, start from the cell nearest its corner.
        component = cls._reachable_from(safe, best_seed)
        yy, xx = np.mgrid[0:rows, 0:cols]
        index = int(np.argmin(np.where(component, yy + xx, np.inf)))
        return (index // cols, index % cols)

    @classmethod
    def _default_goal(cls, products: TerrainProducts, start: tuple[int, int],
                      rover_radius_m: float = 0.5) -> tuple[int, int]:
        """Farthest safe cell from the start.

        Blindly taking the opposite corner picks lethal terrain on any real
        DEM — the corner of a Jezero tile is as likely to be a crater wall as
        open regolith, and the planner then refuses a route that was never
        reachable. This selects the most distant cell the rover could actually
        stand on, and the plan records that the goal was auto-selected.
        """
        safe = cls._safe_cells(products, rover_radius_m=rover_radius_m)
        rows, cols = safe.shape
        if not safe.any():
            return (rows - 3, cols - 3)

        # Farthest *reachable* cell, not farthest safe cell. On fragmented
        # terrain the two differ: the most distant traversable cell often sits
        # in a separate connected component behind a ridge, and the planner
        # then correctly reports "no traversable route" for a goal that was
        # never reachable. A flood fill from the start restricts the choice to
        # cells the rover can actually get to.
        reachable = cls._reachable_from(safe, start)
        if not reachable.any():
            return start
        yy, xx = np.mgrid[0:rows, 0:cols]
        distance = np.where(reachable, np.hypot(yy - start[0], xx - start[1]),
                            -np.inf)
        index = int(np.argmax(distance))
        return (index // cols, index % cols)

    @staticmethod
    def _reachable_from(safe: np.ndarray, start: tuple[int, int]) -> np.ndarray:
        """Flood fill using the planners' exact connectivity rule.

        Two details make this match the planner rather than merely approximate
        it:

        **Diagonal corner-cutting is forbidden.** The planners reject a
        diagonal step when either orthogonal neighbour is blocked, because a
        rover cannot squeeze through the corner gap between two rocks. A naive
        8-connected fill treats that gap as passable and therefore reports
        cells as reachable that the planner will refuse — which is exactly how
        an auto-selected goal ends up unroutable.

        **Iterative, not recursive.** A large open region on a 4000x4000 DEM
        would exhaust the interpreter's stack.
        """
        rows, cols = safe.shape
        seen = np.zeros_like(safe, dtype=bool)
        if not (0 <= start[0] < rows and 0 <= start[1] < cols) or not safe[start]:
            return seen
        stack = [start]
        seen[start] = True
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    if not safe[nr, nc] or seen[nr, nc]:
                        continue
                    # Same corner-cutting rule the planners apply.
                    if dr and dc and not (safe[r, nc] and safe[nr, c]):
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        return seen

    def replan(self, changes: list[tuple[tuple[int, int], float]]
               ) -> dict[str, Any]:
        """Apply newly observed costs and repair the existing plan.

        Returns the repair statistics, which are what make the case for
        incremental replanning visible in the console rather than asserted.
        """
        if self.state is None or self._dstar_seed is None:
            raise PlanningFailure("no active plan to repair",
                                  diagnostics={"hint": "call plan() first"})
        if self.dstar is None:
            seed = self._dstar_seed
            self.dstar = DStarLitePlanner(
                cost_weight=seed["cost_weight"],
                inflation_radius_m=seed["rover_radius_m"],
            )
            self.dstar.initialise(seed["cost"], seed["start"], seed["goal"],
                                  cell_size_m=seed["cell_size_m"])
        expansions = self.dstar.update_costs(changes)
        plan = self.dstar.extract_path()
        event = {
            "at_utc": time.time(),
            "cells_changed": len(changes),
            "repair_expansions": expansions,
            "initial_expansions": self.dstar.initial_expansions,
            "new_length_m": round(plan.length_m, 3),
            "previous_length_m": round(self.state.plan.length_m, 3),
            "max_cell_cost": round(plan.max_cell_cost, 4),
        }
        self.state.replans.append(event)
        self.state.plan = plan
        return event


def _block_mean(array: np.ndarray, factor: int) -> np.ndarray:
    """NaN-aware block-mean downsample. One missing cell must not poison a block."""
    factor = max(1, int(factor))
    if factor == 1:
        return array
    rows, cols = array.shape
    r2, c2 = (rows // factor) * factor, (cols // factor) * factor
    if r2 == 0 or c2 == 0:
        return array
    blocks = array[:r2, :c2].reshape(r2 // factor, factor, c2 // factor, factor)
    with np.errstate(invalid="ignore"):
        return np.nanmean(blocks, axis=(1, 3))
