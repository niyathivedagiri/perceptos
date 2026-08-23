"""
perceptos.mars.traversability
=============================

Turns a Digital Elevation Model into a rover cost map.

This is the module that decides whether the rover is allowed to drive
somewhere.  It follows the hazard taxonomy used in real planetary rover
autonomy (MER/MSL/M2020 GESTALT lineage), evaluating three independent
geometric hazards per cell:

* **Slope** — the plane fitted through the local patch.  Beyond the vehicle's
  static stability limit the rover tips or slips irrecoverably.
* **Step** — peak-to-trough height inside a wheel-sized window.  A step taller
  than roughly half the wheel radius cannot be climbed.
* **Roughness** — RMS residual from the fitted plane.  High roughness means
  the terrain is rubbly: drivable, but it costs energy and induces slip.

Each hazard produces a normalised 0–1 severity.  The cost map is their weighted
maximum (not their sum): one lethal hazard is lethal regardless of how benign
the other two are.  Summing would let a very smooth surface "average away" a
cliff — a real failure mode in naive implementations.

Pure NumPy.  No SciPy, no OpenCV, no GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..core.provenance import Source, Traced, traced

__all__ = ["RoverGeometry", "TraversabilityResult", "TraversabilityAnalyser"]


@dataclass(frozen=True, slots=True)
class RoverGeometry:
    """Physical limits of the vehicle.  Defaults approximate a Perseverance-class rover."""

    wheel_radius_m: float = 0.2625
    wheelbase_m: float = 2.7
    ground_clearance_m: float = 0.6
    max_slope_deg: float = 25.0        # sustained drive limit
    critical_slope_deg: float = 30.0   # tip-over risk
    max_step_m: float = 0.35           # ~ wheel radius * 1.3
    max_roughness_m: float = 0.15

    def __post_init__(self) -> None:
        if self.critical_slope_deg <= self.max_slope_deg:
            raise ValueError("critical_slope_deg must exceed max_slope_deg")


@dataclass(slots=True)
class TraversabilityResult:
    """Per-cell hazard analysis of a DEM patch."""

    cost: np.ndarray            # 0 (free) .. 1 (lethal), float32
    hazard_mask: np.ndarray     # bool, True = must not drive
    slope_deg: np.ndarray
    step_m: np.ndarray
    roughness_m: np.ndarray
    dominant_hazard: np.ndarray  # 0 none, 1 slope, 2 step, 3 roughness
    cell_size_m: float
    coverage: float             # fraction of cells with valid elevation

    HAZARD_NAMES = ("none", "slope", "step", "roughness")

    @property
    def traversable_fraction(self) -> float:
        return float(1.0 - self.hazard_mask.mean())

    def statistics(self) -> dict[str, Any]:
        valid = ~np.isnan(self.slope_deg)
        counts = np.bincount(self.dominant_hazard.ravel(), minlength=4)
        return {
            "traversable_fraction": round(self.traversable_fraction, 4),
            "mean_slope_deg": float(np.nanmean(self.slope_deg)) if valid.any() else None,
            "max_slope_deg": float(np.nanmax(self.slope_deg)) if valid.any() else None,
            "max_step_m": float(np.nanmax(self.step_m)) if valid.any() else None,
            "mean_cost": float(self.cost.mean()),
            "coverage": round(self.coverage, 4),
            "hazard_breakdown": {
                name: int(counts[i]) for i, name in enumerate(self.HAZARD_NAMES)
            },
        }


def _sliding_windows(array: np.ndarray, window: int) -> np.ndarray:
    """Edge-replicated sliding windows, shape (H, W, window, window)."""
    pad = window // 2
    padded = np.pad(array, pad, mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, (window, window))


class TraversabilityAnalyser:
    """Converts a DEM into slope/step/roughness hazards and a rover cost map."""

    def __init__(
        self,
        geometry: Optional[RoverGeometry] = None,
        *,
        cell_size_m: float = 0.1,
        window_cells: Optional[int] = None,
        weights: tuple[float, float, float] = (1.0, 1.0, 0.6),
    ) -> None:
        if cell_size_m <= 0:
            raise ValueError("cell_size_m must be positive")
        self.geometry = geometry or RoverGeometry()
        self.cell_size_m = float(cell_size_m)
        # The analysis window should span roughly one wheel diameter: hazards
        # smaller than the wheel are absorbed by the suspension.
        if window_cells is None:
            span = max(3, int(round(2 * self.geometry.wheel_radius_m / self.cell_size_m)))
            window_cells = span if span % 2 == 1 else span + 1
        if window_cells < 3 or window_cells % 2 == 0:
            raise ValueError("window_cells must be an odd integer >= 3")
        self.window_cells = int(window_cells)
        self.weights = tuple(float(w) for w in weights)

    # ------------------------------------------------------------- primitives

    def compute_slope(self, dem: np.ndarray) -> np.ndarray:
        """Surface slope in degrees from the elevation gradient."""
        dz_dy, dz_dx = np.gradient(dem, self.cell_size_m)
        return np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

    def compute_step(self, dem: np.ndarray) -> np.ndarray:
        """Peak-to-trough elevation change inside the wheel-sized window."""
        windows = _sliding_windows(dem, self.window_cells)
        return np.nanmax(windows, axis=(-2, -1)) - np.nanmin(windows, axis=(-2, -1))

    def compute_roughness(self, dem: np.ndarray) -> np.ndarray:
        """RMS residual after removing the local mean — rubble indicator."""
        windows = _sliding_windows(dem, self.window_cells)
        local_mean = np.nanmean(windows, axis=(-2, -1), keepdims=True)
        return np.sqrt(np.nanmean((windows - local_mean) ** 2, axis=(-2, -1)))

    # ------------------------------------------------------------------ public

    def analyse(self, dem: np.ndarray) -> TraversabilityResult:
        """Full hazard analysis of an elevation grid (metres, NaN = no data)."""
        dem = np.asarray(dem, dtype=np.float64)
        if dem.ndim != 2:
            raise ValueError(f"DEM must be 2D, got shape {dem.shape}")
        if min(dem.shape) < self.window_cells:
            raise ValueError(
                f"DEM {dem.shape} smaller than analysis window {self.window_cells}"
            )

        valid = np.isfinite(dem)
        coverage = float(valid.mean())
        # Fill gaps with the global median so gradients stay finite; the cells
        # remain flagged as hazards below, because unknown terrain is not safe.
        filled = np.where(valid, dem, np.nanmedian(dem) if valid.any() else 0.0)

        slope = self.compute_slope(filled)
        step = self.compute_step(filled)
        rough = self.compute_roughness(filled)

        g = self.geometry
        # Normalise each hazard so 1.0 == the vehicle limit.
        slope_sev = np.clip(slope / g.max_slope_deg, 0.0, 2.0)
        step_sev = np.clip(step / g.max_step_m, 0.0, 2.0)
        rough_sev = np.clip(rough / g.max_roughness_m, 0.0, 2.0)

        w_slope, w_step, w_rough = self.weights
        stacked = np.stack(
            [slope_sev * w_slope, step_sev * w_step, rough_sev * w_rough], axis=0
        )
        cost = np.clip(stacked.max(axis=0), 0.0, 1.0).astype(np.float32)

        dominant = (stacked.argmax(axis=0) + 1).astype(np.uint8)
        dominant[stacked.max(axis=0) < 0.2] = 0  # benign terrain has no "dominant" hazard

        hazard = (
            (slope >= g.critical_slope_deg)
            | (step >= g.max_step_m)
            | (rough >= g.max_roughness_m)
            | ~valid  # never drive into terrain we have not observed
        )
        cost[hazard] = 1.0

        return TraversabilityResult(
            cost=cost,
            hazard_mask=hazard,
            slope_deg=slope.astype(np.float32),
            step_m=step.astype(np.float32),
            roughness_m=rough.astype(np.float32),
            dominant_hazard=dominant,
            cell_size_m=self.cell_size_m,
            coverage=coverage,
        )

    def analyse_traced(
        self, dem_traced: Traced[np.ndarray]
    ) -> Traced[TraversabilityResult]:
        """Hazard analysis that carries provenance forward from the DEM."""
        result = self.analyse(dem_traced.value)
        # Confidence is driven by DEM coverage: a sparse stereo reconstruction
        # yields a cost map we should trust less, and the planner must know.
        confidence = float(max(0.05, min(1.0, result.coverage)))
        return dem_traced.derive(
            result,
            algorithm="TraversabilityAnalyser(slope|step|roughness)",
            source=Source.CALCULATED,
            confidence=confidence,
            units="cost",
            parameters={
                "cell_size_m": self.cell_size_m,
                "window_cells": self.window_cells,
                "max_slope_deg": self.geometry.max_slope_deg,
                "max_step_m": self.geometry.max_step_m,
                "max_roughness_m": self.geometry.max_roughness_m,
                "weights": list(self.weights),
            },
            label="traversability_map",
        )
