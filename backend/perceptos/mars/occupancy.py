"""
perceptos.mars.occupancy
========================

Log-odds occupancy grid mapping with an elevation layer.

Occupancy is stored as log-odds rather than probability for two reasons that
matter over a long traverse: repeated observations accumulate by addition
(numerically stable, no repeated multiplication of small floats), and the
representation is symmetric about "unknown", so evidence for free space cancels
evidence for occupancy correctly.

An inverse sensor model updates cells along each ray: everything between the
sensor and the hit is evidence of free space, the hit cell itself is evidence
of occupancy, and beyond the hit nothing is observed.  Log-odds are clamped so
a cell can always be revised — an unclamped grid becomes permanently certain
and can never recover from a spurious reading.

The elevation layer keeps a running mean and variance per cell, fed by stereo
point clouds, and is what :mod:`perceptos.mars.traversability` consumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from ..core.provenance import Source, Traced, traced

__all__ = ["OccupancyGrid", "GridConfig"]


@dataclass(frozen=True, slots=True)
class GridConfig:
    width_cells: int = 200
    height_cells: int = 200
    cell_size_m: float = 0.1
    origin_x_m: float = 0.0          # world coords of cell (0, 0)
    origin_y_m: float = 0.0
    prob_hit: float = 0.75           # inverse sensor model: P(occupied | hit)
    prob_miss: float = 0.35          # P(occupied | ray passed through)
    log_odds_min: float = -4.0       # ~ P = 0.018
    log_odds_max: float = 4.0        # ~ P = 0.982
    occupied_threshold: float = 0.65

    def __post_init__(self) -> None:
        if not 0.5 < self.prob_hit < 1.0:
            raise ValueError("prob_hit must lie in (0.5, 1.0)")
        if not 0.0 < self.prob_miss < 0.5:
            raise ValueError("prob_miss must lie in (0.0, 0.5)")


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class OccupancyGrid:
    """2D log-odds occupancy grid with a fused elevation layer."""

    def __init__(self, config: Optional[GridConfig] = None) -> None:
        self.config = config or GridConfig()
        c = self.config
        self.log_odds = np.zeros((c.height_cells, c.width_cells), dtype=np.float32)
        self.elevation_sum = np.zeros_like(self.log_odds, dtype=np.float64)
        self.elevation_sq_sum = np.zeros_like(self.log_odds, dtype=np.float64)
        self.observation_count = np.zeros_like(self.log_odds, dtype=np.int32)
        self.l_hit = _logit(c.prob_hit)
        self.l_miss = _logit(c.prob_miss)
        self.updates = 0

    # ----------------------------------------------------------- coordinates

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """World metres -> (row, col).  Row indexes +y, col indexes +x."""
        c = self.config
        col = int(math.floor((x - c.origin_x_m) / c.cell_size_m))
        row = int(math.floor((y - c.origin_y_m) / c.cell_size_m))
        return row, col

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Cell centre in world metres."""
        c = self.config
        return (
            c.origin_x_m + (col + 0.5) * c.cell_size_m,
            c.origin_y_m + (row + 0.5) * c.cell_size_m,
        )

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.config.height_cells and 0 <= col < self.config.width_cells

    # ------------------------------------------------------------- ray casting

    @staticmethod
    def bresenham(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        """Integer line from ``start`` to ``end`` inclusive."""
        r0, c0 = start
        r1, c1 = end
        cells = []
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        sr = 1 if r1 >= r0 else -1
        sc = 1 if c1 >= c0 else -1
        err = dc - dr
        r, c = r0, c0
        while True:
            cells.append((r, c))
            if (r, c) == (r1, c1):
                return cells
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                c += sc
            if e2 < dc:
                err += dc
                r += sr

    def integrate_ray(
        self, sensor_xy: tuple[float, float], hit_xy: tuple[float, float], *, hit: bool = True
    ) -> int:
        """Update one ray.  Returns the number of cells touched."""
        start = self.world_to_cell(*sensor_xy)
        end = self.world_to_cell(*hit_xy)
        cells = self.bresenham(start, end)
        c = self.config
        touched = 0
        for i, (row, col) in enumerate(cells):
            if not self.in_bounds(row, col):
                continue
            is_endpoint = i == len(cells) - 1
            delta = self.l_hit if (is_endpoint and hit) else self.l_miss
            self.log_odds[row, col] = np.clip(
                self.log_odds[row, col] + delta, c.log_odds_min, c.log_odds_max
            )
            touched += 1
        self.updates += 1
        return touched

    def integrate_scan(
        self,
        sensor_xy: tuple[float, float],
        ranges_m: Sequence[float],
        angles_rad: Sequence[float],
        *,
        max_range_m: float = 20.0,
    ) -> int:
        """Integrate a full range scan (hazard-camera derived or simulated LIDAR)."""
        if len(ranges_m) != len(angles_rad):
            raise ValueError("ranges_m and angles_rad must have equal length")
        touched = 0
        for r, theta in zip(ranges_m, angles_rad):
            if not math.isfinite(r) or r <= 0:
                continue
            hit = r < max_range_m
            reach = min(float(r), max_range_m)
            end = (
                sensor_xy[0] + reach * math.cos(theta),
                sensor_xy[1] + reach * math.sin(theta),
            )
            touched += self.integrate_ray(sensor_xy, end, hit=hit)
        return touched

    # ------------------------------------------------------------- elevation

    def integrate_points(self, points_xyz: np.ndarray) -> int:
        """Fuse a point cloud (N, 3) in world coordinates into the elevation layer."""
        pts = np.asarray(points_xyz, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"expected (N, 3) point cloud, got {pts.shape}")
        finite = pts[np.isfinite(pts).all(axis=1)]
        if finite.size == 0:
            return 0
        c = self.config
        cols = np.floor((finite[:, 0] - c.origin_x_m) / c.cell_size_m).astype(np.int64)
        rows = np.floor((finite[:, 1] - c.origin_y_m) / c.cell_size_m).astype(np.int64)
        keep = (
            (rows >= 0) & (rows < c.height_cells) & (cols >= 0) & (cols < c.width_cells)
        )
        rows, cols, z = rows[keep], cols[keep], finite[keep, 2]
        if rows.size == 0:
            return 0
        flat = rows * c.width_cells + cols
        np.add.at(self.elevation_sum.ravel(), flat, z)
        np.add.at(self.elevation_sq_sum.ravel(), flat, z * z)
        np.add.at(self.observation_count.ravel(), flat, 1)
        return int(rows.size)

    def elevation_map(self) -> np.ndarray:
        """Mean elevation per cell; NaN where never observed."""
        count = self.observation_count.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(count > 0, self.elevation_sum / count, np.nan)
        return mean

    def elevation_variance(self) -> np.ndarray:
        """Per-cell elevation variance — a direct measure of DEM trust."""
        count = self.observation_count.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(count > 0, self.elevation_sum / count, np.nan)
            var = np.where(count > 1, self.elevation_sq_sum / count - mean ** 2, np.nan)
        return np.maximum(var, 0.0)

    # --------------------------------------------------------------- readout

    def probability_map(self) -> np.ndarray:
        """Convert log-odds back to occupancy probability in [0, 1]."""
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

    def occupied_mask(self) -> np.ndarray:
        return self.probability_map() >= self.config.occupied_threshold

    @property
    def explored_fraction(self) -> float:
        """Fraction of cells with any evidence either way."""
        return float((self.log_odds != 0).mean())

    def statistics(self) -> dict[str, Any]:
        prob = self.probability_map()
        return {
            "explored_fraction": round(self.explored_fraction, 4),
            "occupied_cells": int(self.occupied_mask().sum()),
            "mean_occupancy": float(prob.mean()),
            "elevation_coverage": round(
                float((self.observation_count > 0).mean()), 4
            ),
            "ray_updates": self.updates,
            "cell_size_m": self.config.cell_size_m,
            "extent_m": [
                self.config.width_cells * self.config.cell_size_m,
                self.config.height_cells * self.config.cell_size_m,
            ],
        }

    def traced_elevation(self, *, parents: Sequence[Traced[Any]] = ()) -> Traced[np.ndarray]:
        """Export the DEM with provenance, ready for traversability analysis."""
        dem = self.elevation_map()
        coverage = float((self.observation_count > 0).mean())
        return traced(
            dem,
            source=Source.CALCULATED,
            algorithm="OccupancyGrid.elevation_map(running-mean fusion)",
            confidence=max(0.05, min(1.0, coverage)),
            units="m",
            parameters={
                "cell_size_m": self.config.cell_size_m,
                "observations": int(self.observation_count.sum()),
                "coverage": coverage,
            },
            label="elevation_map",
            parents=parents,
        )
