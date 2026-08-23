"""
perceptos.mars.planners
=======================

Two planners beyond A*, each solving a problem A* genuinely has.

**Theta\\*** — any-angle planning. A* on an 8-connected grid can only produce
paths made of 45-degree segments, so a straight-line traverse across open
terrain comes out as a staircase. The path is up to 8% longer than the true
optimum, and every step of the staircase is a heading change the rover has to
execute. Theta* fixes this by attempting to reparent each node to its
*grandparent* whenever line of sight allows, producing genuinely straight
segments and far fewer waypoints.

**D\\* Lite** — incremental replanning. A rover discovers hazards as it drives:
a rock resolves in the near field that was invisible at 20 m. Replanning from
scratch with A* every time costs a full search. D* Lite searches *backwards*
from the goal and repairs only the part of the search tree the new information
invalidates, which is typically a small fraction of the map. This is why real
planetary rovers use the D* family rather than re-running A*.

Both operate on the same 0-1 cost grid as
:class:`~perceptos.mars.planner.AStarPlanner`, and both reuse its hazard
inflation, so a path from any of the three respects the same rover clearance.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..core.provenance import Source, Traced
from .planner import _NEIGHBOURS, AStarPlanner, Cell, PathPlan, PlanningFailure

__all__ = ["ThetaStarPlanner", "DStarLitePlanner", "supercover_line"]


def supercover_line(a: Cell, b: Cell) -> list[Cell]:
    """Every cell a straight segment from ``a`` to ``b`` touches.

    Ordinary Bresenham skips cells that the line only clips at a corner. For a
    rendering routine that is invisible; for a line-of-sight check it means a
    path can be declared clear through the diagonal gap between two rocks the
    rover cannot actually fit through. The supercover variant includes them.
    """
    r0, c0 = a
    r1, c1 = b
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r1 >= r0 else -1
    sc = 1 if c1 >= c0 else -1
    r, c = r0, c0
    cells = [(r, c)]
    error = dc - dr
    while (r, c) != (r1, c1):
        e2 = 2 * error
        stepped_c = stepped_r = False
        if e2 > -dr:
            error -= dr
            c += sc
            stepped_c = True
        if e2 < dc:
            error += dc
            r += sr
            stepped_r = True
        if stepped_c and stepped_r:
            # Diagonal step: include the two cells the line passes between.
            cells.append((r - sr, c))
            cells.append((r, c - sc))
        cells.append((r, c))
    return cells


class ThetaStarPlanner(AStarPlanner):
    """Any-angle planning via grandparent reparenting.

    Inherits inflation, hazard thresholds and cost weighting from
    :class:`~perceptos.mars.planner.AStarPlanner`, so the safety envelope is
    identical and only the path shape changes.

    The line-of-sight test integrates terrain cost along the segment rather
    than only checking for blockage. Checking blockage alone — the textbook
    formulation — lets Theta* cut a straight line through expensive-but-legal
    terrain that A* would have routed around, so the "shorter" path is
    genuinely more dangerous. Charging the real cost keeps the comparison
    honest.
    """

    def line_of_sight(self, cost_map: np.ndarray, blocked: np.ndarray,
                      a: Cell, b: Cell) -> tuple[bool, float]:
        """Is ``b`` reachable from ``a`` in a straight line, and at what cost?"""
        cells = supercover_line(a, b)
        rows, cols = cost_map.shape
        terrain_sum = 0.0
        for r, c in cells:
            if not (0 <= r < rows and 0 <= c < cols) or blocked[r, c]:
                return False, math.inf
            terrain_sum += float(cost_map[r, c])

        # The segment's length is the straight-line distance, NOT the sum of
        # steps between supercover cells: supercover deliberately emits the two
        # corner cells at every diagonal crossing, so summing consecutive
        # distances over it roughly doubles the length and Theta* then concludes
        # that going straight is more expensive than the staircase it is meant
        # to replace. Terrain is charged as the mean cost along the cells the
        # segment actually touches.
        length_cells = math.dist(a, b)
        mean_terrain = terrain_sum / len(cells) if cells else 0.0
        return True, length_cells * (1.0 + self.cost_weight * mean_terrain)

    def plan(self, cost_map: np.ndarray, start: Cell, goal: Cell, *,
             cell_size_m: float = 0.1, origin: tuple[float, float] = (0.0, 0.0),
             max_expansions: int = 2_000_000) -> PathPlan:
        cost_map = np.asarray(cost_map, dtype=np.float64)
        if cost_map.ndim != 2:
            raise ValueError("cost_map must be 2D")
        rows, cols = cost_map.shape
        for label, cell in (("start", start), ("goal", goal)):
            if not (0 <= cell[0] < rows and 0 <= cell[1] < cols):
                raise PlanningFailure(f"{label} {cell} lies outside {cost_map.shape}")

        blocked = self._prepare_blocked(cost_map, start, cell_size_m)
        inflation_cells = int(math.ceil(self.inflation_radius_m / cell_size_m))
        if blocked[goal]:
            raise PlanningFailure(
                "goal is inside an inflated hazard footprint",
                diagnostics={"goal_cost": float(cost_map[goal]),
                             "inflation_cells": inflation_cells},
            )

        free = cost_map[~blocked]
        min_step = 1.0 + self.cost_weight * float(free.min()) if free.size else 1.0

        def heuristic(cell: Cell) -> float:
            return math.dist(cell, goal) * cell_size_m * min_step

        g: dict[Cell, float] = {start: 0.0}
        parent: dict[Cell, Cell] = {start: start}
        open_heap: list[tuple[float, Cell]] = [(heuristic(start), start)]
        closed: set[Cell] = set()
        expanded = 0
        los_reparents = 0

        while open_heap:
            _, cell = heapq.heappop(open_heap)
            if cell in closed:
                continue
            closed.add(cell)
            expanded += 1
            if expanded > max_expansions:
                raise PlanningFailure("search budget exhausted",
                                      diagnostics={"nodes_expanded": expanded})

            if cell == goal:
                return self._build_plan(parent, goal, start, cost_map, cell_size_m,
                                        origin, g[goal], expanded, inflation_cells,
                                        los_reparents)

            for dr, dc, step_cells in _NEIGHBOURS:
                nr, nc = cell[0] + dr, cell[1] + dc
                if not (0 <= nr < rows and 0 <= nc < cols) or blocked[nr, nc]:
                    continue
                if dr and dc and (blocked[cell[0], nc] or blocked[nr, cell[1]]):
                    continue
                neighbour = (nr, nc)
                if neighbour in closed:
                    continue

                # Path 2 (the Theta* step): can we skip `cell` entirely and go
                # straight from its parent? If so the path gets a real straight
                # segment instead of another staircase tread.
                grandparent = parent[cell]
                visible, los_cost = self.line_of_sight(cost_map, blocked,
                                                       grandparent, neighbour)
                if visible and g[grandparent] + los_cost * cell_size_m < g.get(
                    neighbour, math.inf
                ):
                    g[neighbour] = g[grandparent] + los_cost * cell_size_m
                    parent[neighbour] = grandparent
                    los_reparents += 1
                    heapq.heappush(open_heap, (g[neighbour] + heuristic(neighbour),
                                               neighbour))
                    continue

                # Path 1: ordinary A* edge.
                distance = step_cells * cell_size_m
                terrain = 0.5 * (cost_map[cell] + cost_map[neighbour])
                tentative = g[cell] + distance * (1.0 + self.cost_weight * terrain)
                if tentative < g.get(neighbour, math.inf):
                    g[neighbour] = tentative
                    parent[neighbour] = cell
                    heapq.heappush(open_heap, (tentative + heuristic(neighbour),
                                               neighbour))

        raise PlanningFailure(
            "no traversable route to goal",
            diagnostics={"nodes_expanded": expanded,
                         "blocked_fraction": float(blocked.mean()),
                         "inflation_cells": inflation_cells},
        )

    # ----------------------------------------------------------------- helpers

    def _prepare_blocked(self, cost_map: np.ndarray, start: Cell,
                         cell_size_m: float) -> np.ndarray:
        """Inflate hazards, then free the rover's own escape region (as A* does)."""
        lethal = cost_map >= self.hazard_threshold
        blocked, _ = self.inflate(lethal, cell_size_m)
        rows, cols = cost_map.shape
        if blocked[start] and not lethal[start]:
            blocked = blocked.copy()
            stack = [start]
            blocked[start] = False
            while stack:
                r, c = stack.pop()
                for dr, dc, _ in _NEIGHBOURS:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols
                            and blocked[nr, nc] and not lethal[nr, nc]):
                        blocked[nr, nc] = False
                        stack.append((nr, nc))
        elif lethal[start]:
            blocked = blocked.copy()
            blocked[start] = False
        return blocked

    def _build_plan(self, parent: dict[Cell, Cell], goal: Cell, start: Cell,
                    cost_map: np.ndarray, cell_size_m: float,
                    origin: tuple[float, float], total_cost: float,
                    expanded: int, inflation_cells: int,
                    los_reparents: int) -> PathPlan:
        waypoints: list[Cell] = []
        cursor = goal
        guard = 0
        while True:
            waypoints.append(cursor)
            if cursor == start or guard > len(parent) + 2:
                break
            cursor = parent[cursor]
            guard += 1
        waypoints.reverse()

        # Densify for cost statistics: the waypoints are sparse by design, but
        # the cells actually driven over are what determine risk.
        cells: list[Cell] = [waypoints[0]]
        for a, b in zip(waypoints, waypoints[1:]):
            cells.extend(supercover_line(a, b)[1:])

        costs = np.array([cost_map[c] for c in cells], dtype=np.float64)
        length = sum(math.dist(a, b) * cell_size_m
                     for a, b in zip(waypoints, waypoints[1:]))
        turns = 0.0
        for i in range(1, len(waypoints) - 1):
            v1 = (waypoints[i][0] - waypoints[i - 1][0],
                  waypoints[i][1] - waypoints[i - 1][1])
            v2 = (waypoints[i + 1][0] - waypoints[i][0],
                  waypoints[i + 1][1] - waypoints[i][1])
            delta = abs(math.degrees(math.atan2(v2[0], v2[1])
                                     - math.atan2(v1[0], v1[1]))) % 360.0
            turns += 360.0 - delta if delta > 180.0 else delta

        world = [(origin[0] + c[1] * cell_size_m, origin[1] + c[0] * cell_size_m)
                 for c in waypoints]
        return PathPlan(
            cells=cells, world_points=world, total_cost=float(total_cost),
            length_m=float(length), max_cell_cost=float(costs.max()),
            mean_cell_cost=float(costs.mean()), turns_deg=float(turns),
            nodes_expanded=expanded, cell_size_m=cell_size_m,
            inflation_cells=inflation_cells,
            metadata={"planner": "ThetaStar", "waypoint_count": len(waypoints),
                      "line_of_sight_reparents": los_reparents},
        )

    def plan_traced(self, traversability: Traced[Any], start: Cell, goal: Cell,
                    **kwargs: Any) -> Traced[PathPlan]:
        result = traversability.value
        plan = self.plan(
            result.cost, start, goal,
            cell_size_m=getattr(result, "cell_size_m", kwargs.pop("cell_size_m", 0.1)),
            **kwargs,
        )
        return traversability.derive(
            plan, algorithm="ThetaStarPlanner(any-angle, cost-integrated LOS)",
            source=Source.CALCULATED, units="m", label="safe_path_plan",
            parameters={"cost_weight": self.cost_weight,
                        "inflation_radius_m": self.inflation_radius_m,
                        "start": list(start), "goal": list(goal)},
        )


class DStarLitePlanner:
    """Incremental replanning for a rover discovering hazards as it drives.

    Searches backwards from the goal, so the start can move without
    invalidating the search tree. When cost changes arrive, only the affected
    vertices are repaired.

    The ``km`` accumulator is the part people get wrong: as the rover advances,
    every previously computed key becomes stale by the heuristic distance
    travelled. Rather than re-keying the whole queue, D* Lite adds that
    distance to a running offset. Omitting it does not crash — it just makes
    the priority ordering wrong, and the planner silently returns suboptimal
    paths that look plausible.
    """

    def __init__(self, *, cost_weight: float = 4.0, hazard_threshold: float = 0.95,
                 inflation_radius_m: float = 1.5) -> None:
        self.cost_weight = float(cost_weight)
        self.hazard_threshold = float(hazard_threshold)
        self.inflation_radius_m = float(inflation_radius_m)
        self._helper = AStarPlanner(cost_weight=cost_weight,
                                    hazard_threshold=hazard_threshold,
                                    inflation_radius_m=inflation_radius_m)
        self.reset_statistics()

    def reset_statistics(self) -> None:
        self.replans = 0
        self.total_expansions = 0
        self.last_expansions = 0
        self.initial_expansions = 0

    # ------------------------------------------------------------------ setup

    def initialise(self, cost_map: np.ndarray, start: Cell, goal: Cell, *,
                   cell_size_m: float = 0.1) -> None:
        self.cost_map = np.asarray(cost_map, dtype=np.float64)
        self.rows, self.cols = self.cost_map.shape
        self.cell_size_m = float(cell_size_m)
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.km = 0.0
        self._last_start = self.start
        self._refresh_blocked()

        self.g: dict[Cell, float] = {}
        self.rhs: dict[Cell, float] = {}
        self.queue: list[tuple[tuple[float, float], Cell]] = []
        self._in_queue: dict[Cell, tuple[float, float]] = {}
        self.rhs[self.goal] = 0.0
        self._push(self.goal)
        self.reset_statistics()
        self.initial_expansions = self._compute_shortest_path()

    def _refresh_blocked(self) -> None:
        lethal = self.cost_map >= self.hazard_threshold
        self.blocked, self.inflation_cells = self._helper.inflate(lethal, self.cell_size_m)
        if self.blocked[self.start] and not lethal[self.start]:
            self.blocked = self.blocked.copy()
            self.blocked[self.start] = False

    # --------------------------------------------------------------- internals

    def _h(self, a: Cell, b: Cell) -> float:
        dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
        octile = (max(dr, dc) - min(dr, dc)) + math.sqrt(2) * min(dr, dc)
        return octile * self.cell_size_m

    def _edge_cost(self, a: Cell, b: Cell) -> float:
        if self.blocked[a] or self.blocked[b]:
            return math.inf
        distance = math.dist(a, b) * self.cell_size_m
        terrain = 0.5 * (self.cost_map[a] + self.cost_map[b])
        return distance * (1.0 + self.cost_weight * terrain)

    def _neighbours(self, cell: Cell) -> Iterable[Cell]:
        for dr, dc, _ in _NEIGHBOURS:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and not self.blocked[nr, nc]:
                if dr and dc and (self.blocked[cell[0], nc] or self.blocked[nr, cell[1]]):
                    continue
                yield (nr, nc)

    #: Tolerance for key comparison. Keys are ``g + h + km`` sums accumulated
    #: over long paths, so two mathematically equal keys differ by a few ulps.
    #: Comparing them exactly makes a vertex that ties with the start key
    #: compare as *greater*, ComputeShortestPath terminates on its first
    #: iteration, and D* Lite silently returns the stale pre-change path. The
    #: algorithm does not crash or warn; it just stops replanning, which is the
    #: worst possible failure for the component whose entire job is replanning.
    KEY_EPS = 1e-9

    @classmethod
    def _key_less(cls, a: tuple[float, float], b: tuple[float, float]) -> bool:
        """Lexicographic key comparison with a floating-point tolerance."""
        if a[0] < b[0] - cls.KEY_EPS:
            return True
        if a[0] > b[0] + cls.KEY_EPS:
            return False
        return a[1] < b[1] - cls.KEY_EPS

    def _key(self, cell: Cell) -> tuple[float, float]:
        value = min(self.g.get(cell, math.inf), self.rhs.get(cell, math.inf))
        return (value + self._h(self.start, cell) + self.km, value)

    def _push(self, cell: Cell) -> None:
        key = self._key(cell)
        self._in_queue[cell] = key
        heapq.heappush(self.queue, (key, cell))

    def _update_vertex(self, cell: Cell) -> None:
        if cell != self.goal:
            self.rhs[cell] = min(
                (self._edge_cost(cell, n) + self.g.get(n, math.inf)
                 for n in self._neighbours(cell)),
                default=math.inf,
            )
        self._in_queue.pop(cell, None)
        g_value = self.g.get(cell, math.inf)
        rhs_value = self.rhs.get(cell, math.inf)
        inconsistent = (
            g_value != rhs_value
            and not (math.isfinite(g_value) and math.isfinite(rhs_value)
                     and abs(g_value - rhs_value) <= self.KEY_EPS)
        )
        if inconsistent:
            self._push(cell)

    def _compute_shortest_path(self, budget: int = 500_000) -> int:
        expanded = 0
        while self.queue:
            key, cell = self.queue[0]
            start_key = self._key(self.start)
            g_start = self.g.get(self.start, math.inf)
            rhs_start = self.rhs.get(self.start, math.inf)
            consistent = (g_start == rhs_start) or (
                math.isfinite(g_start) and math.isfinite(rhs_start)
                and abs(g_start - rhs_start) <= self.KEY_EPS
            )
            if not self._key_less(key, start_key) and consistent:
                break
            heapq.heappop(self.queue)
            # Lazy deletion: a stale queue entry is one whose recorded key no
            # longer matches the vertex's current key.
            if self._in_queue.get(cell) != key:
                continue
            self._in_queue.pop(cell, None)
            expanded += 1
            if expanded > budget:
                break

            new_key = self._key(cell)
            if self._key_less(key, new_key):
                self._push(cell)
            elif self.g.get(cell, math.inf) > self.rhs.get(cell, math.inf) + self.KEY_EPS:
                self.g[cell] = self.rhs.get(cell, math.inf)
                for n in self._neighbours(cell):
                    self._update_vertex(n)
            else:
                self.g[cell] = math.inf
                for n in list(self._neighbours(cell)) + [cell]:
                    self._update_vertex(n)
        self.total_expansions += expanded
        self.last_expansions = expanded
        return expanded

    # ------------------------------------------------------------------- API

    def update_costs(self, changes: Sequence[tuple[Cell, float]]) -> int:
        """Apply newly observed cell costs and repair only what they invalidate.

        Returns the number of vertices expanded — the number that makes the
        case for D* Lite over a fresh A* search.
        """
        if not changes:
            return 0
        self.km += self._h(self._last_start, self.start)
        self._last_start = self.start

        touched: set[Cell] = set()
        for (row, col), cost in changes:
            if not (0 <= row < self.rows and 0 <= col < self.cols):
                continue
            self.cost_map[row, col] = float(cost)
            touched.add((row, col))
        self._refresh_blocked()

        for cell in list(touched):
            for n in list(self._neighbours(cell)) + [cell]:
                touched.add(n)
        for cell in touched:
            self._update_vertex(cell)

        self.replans += 1
        return self._compute_shortest_path()

    def move_start(self, cell: Cell) -> None:
        """Advance the rover. Cheap — the search tree is rooted at the goal."""
        self.start = tuple(cell)

    def extract_path(self, *, max_steps: int = 100_000) -> PathPlan:
        """Greedily descend g-values from start to goal."""
        if math.isinf(self.g.get(self.start, math.inf)):
            raise PlanningFailure(
                "no traversable route to goal",
                diagnostics={"blocked_fraction": float(self.blocked.mean()),
                             "inflation_cells": self.inflation_cells,
                             "expansions": self.total_expansions},
            )
        cells = [self.start]
        cursor = self.start
        total = 0.0
        # Greedy descent alone can cycle. After a repair the g-values are only
        # guaranteed consistent along the optimal path; elsewhere two adjacent
        # cells can each name the other as their cheapest successor, and the
        # walk oscillates between them until the step budget runs out. Measured
        # case: a repair on an 80x80 ramp left (55,55) and (55,56) pointing at
        # each other, and extraction failed on terrain that was fully routable.
        #
        # Refusing to revisit a cell breaks the tie deterministically and
        # cannot exclude a valid route, because an optimal path never visits
        # the same cell twice.
        visited: set[Cell] = {self.start}
        for _ in range(max_steps):
            if cursor == self.goal:
                break
            candidates = [
                (self._edge_cost(cursor, n) + self.g.get(n, math.inf), n)
                for n in self._neighbours(cursor)
                if n not in visited
            ]
            candidates = [c for c in candidates if math.isfinite(c[0])]
            if not candidates:
                raise PlanningFailure(
                    "path extraction stalled: every onward cell is either "
                    "unreachable or already visited",
                    diagnostics={"at": list(cursor), "steps_taken": len(cells),
                                 "hint": "g-values are locally inconsistent; "
                                         "re-run update_costs or re-initialise"},
                )
            cost, nxt = min(candidates)
            total += self._edge_cost(cursor, nxt)
            cursor = nxt
            visited.add(cursor)
            cells.append(cursor)
        else:
            raise PlanningFailure(
                "path extraction exceeded step budget",
                diagnostics={"max_steps": max_steps, "reached": list(cursor)},
            )

        costs = np.array([self.cost_map[c] for c in cells])
        length = sum(math.dist(a, b) * self.cell_size_m
                     for a, b in zip(cells, cells[1:]))
        world = [(c[1] * self.cell_size_m, c[0] * self.cell_size_m) for c in cells]
        return PathPlan(
            cells=cells, world_points=world, total_cost=float(total),
            length_m=float(length), max_cell_cost=float(costs.max()),
            mean_cell_cost=float(costs.mean()), turns_deg=0.0,
            nodes_expanded=self.total_expansions, cell_size_m=self.cell_size_m,
            inflation_cells=self.inflation_cells,
            metadata={"planner": "DStarLite", "replans": self.replans,
                      "initial_expansions": self.initial_expansions,
                      "last_repair_expansions": self.last_expansions},
        )

    def statistics(self) -> dict[str, Any]:
        return {
            "planner": "DStarLite",
            "replans": self.replans,
            "initial_expansions": self.initial_expansions,
            "last_repair_expansions": self.last_expansions,
            "total_expansions": self.total_expansions,
            "km": round(self.km, 4),
            "start": list(self.start),
            "goal": list(self.goal),
        }
