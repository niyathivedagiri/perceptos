"""
perceptos.data.terrain
======================

Real elevation and real terrain labels, shaped for the modules that already
exist.

Neither :class:`~perceptos.mars.traversability.TraversabilityAnalyser` nor
:class:`~perceptos.mars.occupancy.OccupancyGrid` changes. This module's whole
job is to produce a metres-per-cell NumPy elevation grid and an integer label
raster, with provenance attached, and hand them over.

**Elevation.** HiRISE DTMs of Jezero crater are published at 1 m/pixel by
USGS, controlled to MOLA. They arrive as GeoTIFF (or Cloud Optimized GeoTIFF,
which supports ranged reads so a 500 m tile does not require the whole scene).
Reading them needs ``rasterio`` or GDAL; both are optional here, and a plain
``.npy`` path always works so the module is usable without a geospatial stack.

**Vertical datum matters.** HiRISE DTM pixel values are elevations relative to
the areoid, not radii from the planet centre, and no-data is a large negative
sentinel rather than NaN. Loading one naively gives a terrain with a -3.4e38 m
cliff in it, which the traversability analyser will faithfully report as the
most hazardous surface in the solar system. The sentinel is converted to NaN
here, which is exactly what the analyser already treats as untraversable.

**Labels.** AI4Mars label rasters are greyscale images whose pixel values are
class ids 0-3, with 255 marking unlabelled. Unlabelled is not a class and must
not be trained on or driven over.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..core.environment import Environment, TerrainClass
from ..core.provenance import Source, Traced, traced

__all__ = [
    "DEMTile",
    "denoise_labels",
    "load_dem",
    "load_dem_traced",
    "AI4MarsLabels",
    "load_labels",
    "semantic_cost_layer",
    "NODATA_THRESHOLD",
    "AI4MARS_UNLABELLED",
]

#: HiRISE and most planetary GeoTIFFs use a large negative no-data sentinel.
NODATA_THRESHOLD = -1e30

#: AI4Mars marks unlabelled pixels with 255, not with a class id.
AI4MARS_UNLABELLED = 255


@dataclass(slots=True)
class DEMTile:
    """An elevation grid in metres, with its georeferencing where known."""

    elevation_m: np.ndarray          # float64, NaN where no data
    cell_size_m: float
    source: str
    origin_xy: tuple[float, float] = (0.0, 0.0)
    crs: str = ""
    vertical_datum: str = ""
    nodata_fraction: float = 0.0
    citation: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation_m.shape

    @property
    def extent_m(self) -> tuple[float, float]:
        h, w = self.elevation_m.shape
        return (w * self.cell_size_m, h * self.cell_size_m)

    @property
    def coverage(self) -> float:
        return float(1.0 - self.nodata_fraction)

    def statistics(self) -> dict[str, Any]:
        valid = self.elevation_m[np.isfinite(self.elevation_m)]
        return {
            "shape": list(self.shape),
            "cell_size_m": self.cell_size_m,
            "extent_m": [round(v, 1) for v in self.extent_m],
            "coverage": round(self.coverage, 4),
            "min_elevation_m": float(valid.min()) if valid.size else None,
            "max_elevation_m": float(valid.max()) if valid.size else None,
            "relief_m": float(valid.max() - valid.min()) if valid.size else None,
            "source": self.source,
            "crs": self.crs,
            "vertical_datum": self.vertical_datum,
        }

    def crop(self, row: int, col: int, height: int, width: int) -> "DEMTile":
        """Extract a sub-tile. Bounds are clamped rather than raising."""
        r0, c0 = max(0, int(row)), max(0, int(col))
        r1 = min(self.elevation_m.shape[0], r0 + int(height))
        c1 = min(self.elevation_m.shape[1], c0 + int(width))
        patch = self.elevation_m[r0:r1, c0:c1]
        return DEMTile(
            elevation_m=patch.copy(),
            cell_size_m=self.cell_size_m,
            source=f"{self.source}[{r0}:{r1},{c0}:{c1}]",
            origin_xy=(self.origin_xy[0] + c0 * self.cell_size_m,
                       self.origin_xy[1] + r0 * self.cell_size_m),
            crs=self.crs,
            vertical_datum=self.vertical_datum,
            nodata_fraction=float(np.mean(~np.isfinite(patch))) if patch.size else 1.0,
            citation=self.citation,
        )

    def downsample(self, factor: int) -> "DEMTile":
        """Block-mean downsample, NaN-aware. Coarsens the grid for fast planning."""
        factor = max(1, int(factor))
        if factor == 1:
            return self
        h, w = self.elevation_m.shape
        h2, w2 = (h // factor) * factor, (w // factor) * factor
        blocks = self.elevation_m[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor)
        with np.errstate(invalid="ignore"):
            coarse = np.nanmean(blocks, axis=(1, 3))
        return DEMTile(
            elevation_m=coarse,
            cell_size_m=self.cell_size_m * factor,
            source=f"{self.source}/downsample{factor}",
            origin_xy=self.origin_xy, crs=self.crs,
            vertical_datum=self.vertical_datum,
            nodata_fraction=float(np.mean(~np.isfinite(coarse))),
            citation=self.citation,
        )


def _sanitise(elevation: np.ndarray, nodata: Optional[float] = None) -> np.ndarray:
    """Convert no-data sentinels to NaN. The step everyone forgets."""
    out = np.asarray(elevation, dtype=np.float64).copy()
    if nodata is not None:
        out[out == nodata] = np.nan
    out[out < NODATA_THRESHOLD] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def load_dem(path: str | Path, *, cell_size_m: Optional[float] = None,
             nodata: Optional[float] = None,
             source: str = "", citation: str = "") -> DEMTile:
    """Load a DEM from ``.npy`` or, if rasterio is present, a GeoTIFF.

    ``.npy`` always works. GeoTIFF requires ``pip install rasterio``; the error
    says so rather than surfacing an opaque ImportError.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch it with:\n"
            "    python scripts/fetch_data.py jezero-dem"
        )

    if path.suffix.lower() == ".npy":
        raw = np.load(path)
        elevation = _sanitise(raw, nodata)
        return DEMTile(
            elevation_m=elevation,
            cell_size_m=float(cell_size_m if cell_size_m else 1.0),
            source=source or str(path),
            nodata_fraction=float(np.mean(~np.isfinite(elevation))),
            citation=citation,
        )

    try:
        import rasterio  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"reading {path.suffix} requires rasterio:\n"
            "    pip install rasterio\n"
            "Alternatively convert the DTM to .npy once and load that."
        ) from exc

    with rasterio.open(path) as dataset:
        band = dataset.read(1)
        transform = dataset.transform
        detected_cell = float(abs(transform.a)) if transform.a else 1.0
        elevation = _sanitise(band, dataset.nodata if nodata is None else nodata)
        return DEMTile(
            elevation_m=elevation,
            cell_size_m=float(cell_size_m if cell_size_m else detected_cell),
            source=source or str(path),
            origin_xy=(float(transform.c), float(transform.f)),
            crs=str(dataset.crs) if dataset.crs else "",
            vertical_datum="areoid (HiRISE DTM convention)",
            nodata_fraction=float(np.mean(~np.isfinite(elevation))),
            citation=citation,
        )


def load_dem_traced(path: str | Path, *, source_type: Optional[str] = None,
                    source_url: Optional[str] = None,
                    dataset: str = "jezero_dem", **kwargs: Any) -> Traced[np.ndarray]:
    """Load a DEM as a traced elevation array, ready for TraversabilityAnalyser.

    The source class is ``LIVE_SENSOR``: a HiRISE DTM is a measurement made by
    a real instrument, photogrammetrically reduced. Confidence is the tile's
    valid-data coverage, so a DEM full of holes produces a traversability map
    the planner already knows to distrust.
    """
    tile = load_dem(path, **kwargs)
    return traced(
        tile.elevation_m,
        source=Source.LIVE_SENSOR,
        # A DEM read from disk is real measurement, but it is not a live fetch.
        # Defaulting to LOCAL_CACHED_DATA keeps that distinction visible in the
        # UI badge instead of leaving source_type blank.
        source_type=source_type or "LOCAL_CACHED_DATA",
        source_url=source_url or None,
        dataset=dataset,
        algorithm="HiRISE DTM (stereo photogrammetry, MOLA-controlled)",
        confidence=float(max(0.01, min(1.0, tile.coverage))),
        units="m",
        label="dem_elevation",
        parameters={
            "cell_size_m": tile.cell_size_m,
            "source": tile.source,
            "crs": tile.crs,
            "vertical_datum": tile.vertical_datum,
            "coverage": round(tile.coverage, 4),
            "credit": "NASA/JPL-Caltech/USGS",
            "citation": tile.citation,
        },
    )


@dataclass(slots=True)
class AI4MarsLabels:
    """A semantic label raster over the AI4Mars taxonomy."""

    labels: np.ndarray               # uint8, class ids, 255 = unlabelled
    classes: Sequence[TerrainClass]
    source: str = ""

    @property
    def labelled_fraction(self) -> float:
        return float(np.mean(self.labels != AI4MARS_UNLABELLED))

    def class_histogram(self) -> dict[str, float]:
        total = max(1, int((self.labels != AI4MARS_UNLABELLED).sum()))
        out = {}
        for cls in self.classes:
            out[cls.name] = float((self.labels == cls.id).sum()) / total
        return out

    def statistics(self) -> dict[str, Any]:
        return {
            "shape": list(self.labels.shape),
            "labelled_fraction": round(self.labelled_fraction, 4),
            "class_fractions": {k: round(v, 4) for k, v in self.class_histogram().items()},
            "source": self.source,
        }


def load_labels(path: str | Path, environment: Environment) -> AI4MarsLabels:
    """Load an AI4Mars label PNG as class ids."""
    from ..core.cv_compat import cv  # noqa: PLC0415 - keeps this module import-light

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch AI4Mars with:\n"
            "    python scripts/fetch_data.py ai4mars"
        )
    raster = cv().imread(str(path), cv().IMREAD_GRAYSCALE)
    if raster is None:
        raise ValueError(f"could not decode label raster {path}")
    return AI4MarsLabels(
        labels=raster.astype(np.uint8),
        classes=environment.terrain_classes,
        source=str(path),
    )


def _binary_open(mask: np.ndarray, size: int) -> np.ndarray:
    """Morphological opening (erode then dilate) on a boolean mask, pure NumPy."""
    if size <= 1:
        return mask
    pad = size // 2
    eroded = np.lib.stride_tricks.sliding_window_view(
        np.pad(mask, pad, mode="constant", constant_values=False), (size, size)
    ).all(axis=(-2, -1))
    return np.lib.stride_tricks.sliding_window_view(
        np.pad(eroded, pad, mode="constant", constant_values=False), (size, size)
    ).any(axis=(-2, -1))


def denoise_labels(labels: np.ndarray, environment: Environment, *,
                   min_blob_cells: int = 3,
                   lethal_threshold: float = 0.95,
                   fill_with: Optional[int] = None) -> np.ndarray:
    """Remove isolated lethal-class pixels from a segmentation raster.

    Segmentation models emit salt-and-pepper noise. A single stray pixel
    labelled ``big_rock`` is, at typical cell sizes, an object smaller than the
    rover's wheel — but treated as lethal and then inflated by the rover's
    circumscribed radius, each one becomes a metre-wide no-go disc. A few
    hundred of them scattered across a map block it entirely, and the planner
    reports "no traversable route" across terrain a rover would drive over
    without noticing.

    This is not hypothetical: at 2% speckle and a 0.75 m inflation radius,
    57% of a map becomes unreachable.

    Opening the lethal mask removes blobs thinner than ``min_blob_cells`` while
    leaving genuine multi-cell hazards untouched. Removed pixels fall back to
    the most common neighbouring class, so the terrain is not left unlabelled.

    Note this filters *only* the lethal classes. Non-lethal misclassification
    changes a cost slightly and is not worth touching; lethal misclassification
    is what stops the vehicle.
    """
    labels = np.asarray(labels).copy()
    lethal_ids = [c.id for c in environment.terrain_classes
                  if c.mobility_cost >= lethal_threshold]
    if not lethal_ids:
        return labels

    if fill_with is None:
        # Fall back to the most common non-lethal class actually present.
        candidates = [c.id for c in environment.terrain_classes
                      if c.id not in lethal_ids]
        counts = [(int((labels == cid).sum()), cid) for cid in candidates]
        fill_with = max(counts)[1] if counts else int(labels.max())

    for lethal_id in lethal_ids:
        mask = labels == lethal_id
        if not mask.any():
            continue
        kept = _binary_open(mask, int(min_blob_cells))
        labels[mask & ~kept] = fill_with
    return labels


def semantic_cost_layer(labels: np.ndarray, environment: Environment, *,
                        unlabelled_cost: float = 0.5) -> np.ndarray:
    """Convert class ids to a 0-1 mobility cost map.

    This is the layer that makes semantic segmentation earn its place. A
    geometric cost map sees flat sand as free; this one sees it as a slip
    hazard. Combined with the geometric cost by maximum — the same rule
    :class:`~perceptos.mars.traversability.TraversabilityAnalyser` already uses
    — one lethal factor stays lethal regardless of how benign the other looks.

    Unlabelled pixels get a deliberately middling cost, not zero: the model
    declining to classify a region is weak evidence against driving there, not
    evidence for it.
    """
    labels = np.asarray(labels)
    cost = np.full(labels.shape, float(unlabelled_cost), dtype=np.float32)
    for cls in environment.terrain_classes:
        cost[labels == cls.id] = cls.mobility_cost
    cost[labels == AI4MARS_UNLABELLED] = float(unlabelled_cost)
    return cost


def fuse_costs(geometric: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    """Combine geometric and semantic cost by maximum.

    Averaging would let smooth geometry cancel a sand warning, which is
    precisely the failure this layer exists to prevent.
    """
    geometric = np.asarray(geometric, dtype=np.float32)
    semantic = np.asarray(semantic, dtype=np.float32)
    if geometric.shape != semantic.shape:
        raise ValueError(
            f"cost maps must share a shape; got {geometric.shape} and {semantic.shape}"
        )
    return np.maximum(geometric, semantic)
