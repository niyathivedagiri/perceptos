"""
perceptos.mars.planner
======================

Safe path planning over a traversability cost map.

A* on an 8-connected grid, with three additions that matter on a real rover
and are usually missing from textbook implementations:

1. **Hazard inflation.**  The rover is not a point.  Lethal cells are dilated
   by the vehicle's circumscribed radius before planning, so the planner cannot
   route the *centre* of the rover safely past a rock that the *wheels* would
   hit.
2. **Turn penalty.**  Point turns cost energy and induce slip.  Penalising
   heading changes produces the smooth arcs a rover actually drives instead of
   a staircase.
3. **Admissible heuristic under a cost multiplier.**  The heuristic is scaled
   by the *minimum* achievable per-metre cost, not the nominal one.  Scaling by
   anything higher makes the heuristic inadmissible and A* stops being optimal —
   a subtle bug that produces plausible-looking but needlessly dangerous paths.

Pure NumPy plus ``heapq``.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..core.provenance import Source, Traced

__all__ = ["PathPlan", "AStarPlanner", "PlanningFailure"]

Cell = tuple[int, int]

# (row delta, col delta, step length in cells)
_NEIGHBOURS: tuple[tuple[int, int, float], ...] = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
)


class PlanningFailure(RuntimeError):
    """No safe route exists.  Carries diagnostics for the operator."""

    def __init__(self, message: str, *, diagnostics: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass(slots=True)
class PathPlan:
    """A planned traverse, with everything needed to justify it."""

    cells: list[Cell]
    world_points: list[tuple[float, float]]
    total_cost: float
    length_m: float
    max_cell_cost: float
    mean_cell_cost: float
    turns_deg: float
    nodes_expanded: int
    cell_size_m: float
    inflation_cells: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def waypoints(self) -> list[dict[str, float]]:
        return [{"x": x, "y": y} for x, y in self.world_points]

    def estimate_duration_s(self, speed_mps: float = 0.042) -> float:
        """Traverse time.  Default speed is Perseverance's ~150 m/h cruise.

        Cost slows the rover: high-cost terrain is driven cautiously, so
        effective speed is scaled down by the mean cell cost.
        """
        if speed_mps <= 0:
            raise ValueError("speed_mps must be positive")
        derate = 1.0 / (1.0 + 2.0 * self.mean_cell_cost)
        return self.length_m / (speed_mps * derate)

    def estimate_energy_wh(self, wh_per_m: float = 1.2) -> float:
        """Energy budget, penalised by terrain cost and turning."""
        turn_penalty_wh = 0.02 * self.turns_deg
        return self.length_m * wh_per_m * (1.0 + self.mean_cell_cost) + turn_penalty_wh

    def to_dict(self) -> dict[str, Any]:
        return {
            "waypoints": self.waypoints,
            "cells": [list(c) for c in self.cells],
            "total_cost": round(self.total_cost, 4),
            "length_m": round(self.length_m, 3),
            "max_cell_cost": round(self.max_cell_cost, 4),
            "mean_cell_cost": round(self.mean_cell_cost, 4),
            "turns_deg": round(self.turns_deg, 2),
            "nodes_expanded": self.nodes_expanded,
            "estimated_duration_s": round(self.estimate_duration_s(), 1),
            "estimated_energy_wh": round(self.estimate_energy_wh(), 2),
            **self.metadata,
        }


class AStarPlanner:
    """A* over a 0–1 traversability cost grid.

    Parameters
    ----------
    cost_weight:
        How strongly terrain cost dominates distance.  ``0`` gives the shortest
        path regardless of danger; large values hug the safest terrain even if
        it triples the traverse.
    turn_penalty:
        Cost added per 90 degrees of heading change.
    hazard_threshold:
        Cells at or above this cost are untraversable.
    inflation_radius_m:
        Rover circumscribed radius used to dilate hazards.
    """

    def __init__(
        self,
        *,
        cost_weight: float = 4.0,
        turn_penalty: float = 0.35,
        hazard_threshold: float = 0.95,
        inflation_radius_m: float = 1.5,
    ) -> None:
        self.cost_weight = float(cost_weight)
        self.turn_penalty = float(turn_penalty)
        self.hazard_threshold = float(hazard_threshold)
        self.inflation_radius_m = float(inflation_radius_m)

    # ------------------------------------------------------------- inflation

    def inflate(self, blocked: np.ndarray, cell_size_m: float) -> tuple[np.ndarray, int]:
        """Dilate the hazard mask by the rover's circumscribed radius."""
        radius_cells = int(math.ceil(self.inflation_radius_m / cell_size_m))
        if radius_cells <= 0:
            return blocked.copy(), 0
        size = 2 * radius_cells + 1
        # Pad with False, not True.  Padding with True would dilate the map
        # border inwards and make every goal within one rover-radius of the
        # edge unreachable — the map boundary is an arbitrary crop, not a wall.
        # Genuinely unobserved terrain is already lethal via the NaN handling in
        # TraversabilityAnalyser, so it does not need encoding twice.
        padded = np.pad(blocked, radius_cells, mode="constant", constant_values=False)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
        # Circular structuring element — a square would over-inflate diagonals.
        yy, xx = np.mgrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
        disk = (yy ** 2 + xx ** 2) <= radius_cells ** 2
        return (windows & disk).any(axis=(-2, -1)), radius_cells

    # ---------------------------------------------------------------- planning

    def plan(
        self,
        cost_map: np.ndarray,
        start: Cell,
        goal: Cell,
        *,
        cell_size_m: float = 0.1,
        origin: tuple[float, float] = (0.0, 0.0),
        max_expansions: int = 2_000_000,
    ) -> PathPlan:
        """Find the lowest-cost safe route from ``start`` to ``goal`` (row, col)."""
        cost_map = np.asarray(cost_map, dtype=np.float64)
        if cost_map.ndim != 2:
            raise ValueError("cost_map must be 2D")
        rows, cols = cost_map.shape

        for label, cell in (("start", start), ("goal", goal)):
            if not (0 <= cell[0] < rows and 0 <= cell[1] < cols):
                raise PlanningFailure(f"{label} {cell} lies outside the {cost_map.shape} map")

        lethal = cost_map >= self.hazard_threshold
        blocked, inflation_cells = self.inflate(lethal, cell_size_m)

        # Escape handling.  If the rover is already standing inside an inflated
        # footprint, freeing only its own cell leaves it boxed in with no legal
        # move.  Instead, free the whole connected region of inflated-but-not-
        # lethal cells containing the start: the rover is demonstrably surviving
        # in that region, so it is permitted to back out through it.  Genuinely
        # lethal cells stay blocked, so this never authorises driving into rock.
        if blocked[start] and not lethal[start]:
            blocked = blocked.copy()
            escape: list[Cell] = [start]
            blocked[start] = False
            while escape:
                r, c = escape.pop()
                for dr, dc, _ in _NEIGHBOURS:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and blocked[nr, nc]
                        and not lethal[nr, nc]
                    ):
                        blocked[nr, nc] = False
                        escape.append((nr, nc))
        elif lethal[start]:
            # The rover reports being inside a lethal cell — trust the vehicle
            # over the map, but only for the single cell it occupies.
            blocked = blocked.copy()
            blocked[start] = False
        if blocked[goal]:
            raise PlanningFailure(
                "goal is inside an inflated hazard footprint",
                diagnostics={
                    "goal_cost": float(cost_map[goal]),
                    "inflation_cells": inflation_cells,
                    "hazard_threshold": self.hazard_threshold,
                },
            )

        # Admissible heuristic: cheapest possible cost to cross one metre.
        min_step_cost = 1.0 + self.cost_weight * float(cost_map[~blocked].min()) \
            if (~blocked).any() else 1.0

        def heuristic(cell: Cell) -> float:
            dr, dc = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
            # Octile distance — exact for 8-connected grids.
            octile = (max(dr, dc) - min(dr, dc)) + math.sqrt(2) * min(dr, dc)
            return octile * cell_size_m * min_step_cost

        start_state = (start, -1)  # (cell, incoming direction index)
        open_heap: list[tuple[float, float, Cell, int]] = [
            (heuristic(start), 0.0, start, -1)
        ]
        g_score: dict[tuple[Cell, int], float] = {start_state: 0.0}
        came_from: dict[tuple[Cell, int], tuple[Cell, int]] = {}
        closed: set[tuple[Cell, int]] = set()
        expanded = 0

        while open_heap:
            _, g, cell, direction = heapq.heappop(open_heap)
            state = (cell, direction)
            if state in closed:
                continue
            closed.add(state)
            expanded += 1
            if expanded > max_expansions:
                raise PlanningFailure(
                    "search budget exhausted",
                    diagnostics={"nodes_expanded": expanded},
                )

            if cell == goal:
                return self._reconstruct(
                    came_from, state, cost_map, cell_size_m, origin,
                    g, expanded, inflation_cells,
                )

            for idx, (dr, dc, step_cells) in enumerate(_NEIGHBOURS):
                nr, nc = cell[0] + dr, cell[1] + dc
                if not (0 <= nr < rows and 0 <= nc < cols) or blocked[nr, nc]:
                    continue
                neighbour = (nr, nc)
                # Forbid diagonal moves that cut a hazard corner.
                if dr and dc and (blocked[cell[0], nc] or blocked[nr, cell[1]]):
                    continue

                distance = step_cells * cell_size_m
                terrain = 0.5 * (cost_map[cell] + cost_map[neighbour])
                move_cost = distance * (1.0 + self.cost_weight * terrain)
                if direction >= 0:
                    move_cost += self.turn_penalty * _turn_cost(direction, idx)

                n_state = (neighbour, idx)
                tentative = g + move_cost
                if tentative < g_score.get(n_state, math.inf):
                    g_score[n_state] = tentative
                    came_from[n_state] = state
                    heapq.heappush(
                        open_heap, (tentative + heuristic(neighbour), tentative, neighbour, idx)
                    )

        raise PlanningFailure(
            "no traversable route to goal",
            diagnostics={
                "nodes_expanded": expanded,
                "blocked_fraction": float(blocked.mean()),
                "inflation_cells": inflation_cells,
                "hint": "reduce inflation_radius_m, or command a closer intermediate waypoint",
            },
        )

    def plan_traced(
        self,
        traversability: Traced[Any],
        start: Cell,
        goal: Cell,
        **kwargs: Any,
    ) -> Traced[PathPlan]:
        """Plan from a traced traversability result, preserving lineage.

        This is the link that lets the operator click "steer left" in the UI and
        walk all the way back to the stereo pair that revealed the rock.
        """
        result = traversability.value
        plan = self.plan(
            result.cost,
            start,
            goal,
            cell_size_m=getattr(result, "cell_size_m", kwargs.pop("cell_size_m", 0.1)),
            **kwargs,
        )
        return traversability.derive(
            plan,
            algorithm="AStarPlanner(octile, inflated, turn-penalised)",
            source=Source.CALCULATED,
            units="m",
            parameters={
                "cost_weight": self.cost_weight,
                "turn_penalty": self.turn_penalty,
                "hazard_threshold": self.hazard_threshold,
                "inflation_radius_m": self.inflation_radius_m,
                "start": list(start),
                "goal": list(goal),
            },
            label="safe_path_plan",
        )

    # ----------------------------------------------------------------- helpers

    def _reconstruct(
        self,
        came_from: dict[tuple[Cell, int], tuple[Cell, int]],
        state: tuple[Cell, int],
        cost_map: np.ndarray,
        cell_size_m: float,
        origin: tuple[float, float],
        total_cost: float,
        expanded: int,
        inflation_cells: int,
    ) -> PathPlan:
        cells: list[Cell] = []
        directions: list[int] = []
        cursor: Optional[tuple[Cell, int]] = state
        while cursor is not None:
            cells.append(cursor[0])
            directions.append(cursor[1])
            cursor = came_from.get(cursor)
        cells.reverse()
        directions.reverse()

        turns = sum(
            _turn_cost(directions[i], directions[i + 1]) * 90.0
            for i in range(len(directions) - 1)
            if directions[i] >= 0 and directions[i + 1] >= 0
        )
        costs = np.array([cost_map[c] for c in cells], dtype=np.float64)
        length = sum(
            math.dist(cells[i], cells[i + 1]) * cell_size_m for i in range(len(cells) - 1)
        )
        world = [
            (origin[0] + c[1] * cell_size_m, origin[1] + c[0] * cell_size_m) for c in cells
        ]
        return PathPlan(
            cells=cells,
            world_points=world,
            total_cost=float(total_cost),
            length_m=float(length),
            max_cell_cost=float(costs.max()),
            mean_cell_cost=float(costs.mean()),
            turns_deg=float(turns),
            nodes_expanded=expanded,
            cell_size_m=cell_size_m,
            inflation_cells=inflation_cells,
        )


def _turn_cost(from_idx: int, to_idx: int) -> float:
    """Heading change between two neighbour directions, in units of 90 degrees."""
    a = math.atan2(_NEIGHBOURS[from_idx][0], _NEIGHBOURS[from_idx][1])
    b = math.atan2(_NEIGHBOURS[to_idx][0], _NEIGHBOURS[to_idx][1])
    delta = abs(math.degrees(b - a)) % 360.0
    if delta > 180.0:
        delta = 360.0 - delta
    return delta / 90.0
