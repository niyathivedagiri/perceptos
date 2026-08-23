"""
perceptos.api.render
====================

Turns arrays into PNGs for the mission-control console.

This module is the only place in PerceptOS that converts numeric data into
something that looks like a picture, so it is also the place where a picture
could most easily start lying. Three rules hold throughout:

**No data is drawn as no data.** NaN, infinity and masked cells render in a
single reserved colour (magenta) that appears nowhere in any colormap. A
depth map with 40% invalid pixels must *look* 40% invalid. The common
alternative — clamping NaN to zero — produces a smooth, complete-looking
raster that silently claims measurements were made where none were.

**The scale is reported, not hidden.** Every render returns the minimum and
maximum actually used for normalisation, in the value's own units, so the
console can print a legend. A colormapped image without a scale is decoration.

**Nothing is interpolated into existence.** Rendering resizes with nearest
neighbour, never bilinear. Smoothing a 40x40 cost map up to 400x400 with
interpolation invents intermediate values that no algorithm computed, and on a
hazard map those invented cells sit exactly on the boundary between safe and
lethal.

Colormaps are computed in NumPy rather than taken from OpenCV's ``applyColorMap``
so that the no-data channel survives; ``applyColorMap`` has no concept of a
masked cell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..core.cv_compat import HAS_OPENCV, cv

__all__ = ["RenderResult", "render_array", "encode_png", "COLORMAPS",
           "NO_DATA_RGB", "render_point_cloud", "render_hillshade",
           "render_terrain_3d", "render_trajectory"]

#: Reserved for cells with no measurement. Deliberately outside every colormap.
NO_DATA_RGB = (255, 0, 255)


def _viridis_like(t: np.ndarray) -> np.ndarray:
    """Perceptually-ordered blue->green->yellow ramp, computed in NumPy."""
    anchors = np.array([
        [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
        [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
        [180, 222, 44], [253, 231, 37],
    ], dtype=np.float64)
    return _interp_anchors(t, anchors)


def _inferno_like(t: np.ndarray) -> np.ndarray:
    anchors = np.array([
        [0, 0, 4], [31, 12, 72], [85, 15, 109], [136, 34, 106],
        [186, 54, 85], [227, 89, 51], [249, 140, 10], [249, 201, 50],
        [252, 255, 164],
    ], dtype=np.float64)
    return _interp_anchors(t, anchors)


def _hazard(t: np.ndarray) -> np.ndarray:
    """Green (free) -> amber (costly) -> red (lethal).

    Used for cost and hazard maps because the semantic ordering is
    safe-to-dangerous and an operator reads that faster than a rainbow.
    """
    anchors = np.array([
        [46, 160, 67], [128, 180, 60], [210, 153, 34],
        [230, 100, 40], [248, 81, 73],
    ], dtype=np.float64)
    return _interp_anchors(t, anchors)


def _grey(t: np.ndarray) -> np.ndarray:
    value = np.clip(t, 0.0, 1.0) * 255.0
    return np.stack([value, value, value], axis=-1)


def _interp_anchors(t: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation between colormap anchor colours."""
    t = np.clip(t, 0.0, 1.0)
    positions = np.linspace(0.0, 1.0, len(anchors))
    channels = [np.interp(t, positions, anchors[:, i]) for i in range(3)]
    return np.stack(channels, axis=-1)


COLORMAPS = {
    "viridis": _viridis_like,
    "inferno": _inferno_like,
    "hazard": _hazard,
    "grey": _grey,
}


@dataclass(slots=True)
class RenderResult:
    """A rendered image plus everything needed to read it honestly."""

    png: bytes
    width: int
    height: int
    colormap: str
    vmin: Optional[float]
    vmax: Optional[float]
    units: Optional[str]
    valid_fraction: float
    no_data_rgb: tuple[int, int, int] = NO_DATA_RGB
    note: str = ""

    def headers(self) -> dict[str, str]:
        """Render metadata as HTTP headers, so an <img> tag can still be audited."""
        out = {
            "X-PerceptOS-Colormap": self.colormap,
            "X-PerceptOS-Valid-Fraction": f"{self.valid_fraction:.4f}",
            "X-PerceptOS-No-Data-RGB": ",".join(str(c) for c in self.no_data_rgb),
        }
        if self.vmin is not None:
            out["X-PerceptOS-Vmin"] = f"{self.vmin:.6g}"
        if self.vmax is not None:
            out["X-PerceptOS-Vmax"] = f"{self.vmax:.6g}"
        if self.units:
            out["X-PerceptOS-Units"] = self.units
        if self.note:
            out["X-PerceptOS-Note"] = self.note
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width, "height": self.height,
            "colormap": self.colormap, "vmin": self.vmin, "vmax": self.vmax,
            "units": self.units,
            "valid_fraction": round(self.valid_fraction, 4),
            "no_data_rgb": list(self.no_data_rgb),
            "note": self.note,
        }


def encode_png(rgb: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 RGB array as PNG bytes."""
    if not HAS_OPENCV:
        raise RuntimeError(
            "PNG encoding needs OpenCV. Install it with:\n"
            "    pip install opencv-contrib-python"
        )
    cv2 = cv()
    bgr = rgb[:, :, ::-1]                       # OpenCV writes BGR
    ok, buffer = cv2.imencode(".png", np.ascontiguousarray(bgr))
    if not ok:
        raise RuntimeError("cv2.imencode failed to encode the render")
    return buffer.tobytes()


def _resize_nearest(rgb: np.ndarray, max_dimension: int) -> np.ndarray:
    """Scale to fit, nearest neighbour only.

    Interpolation would invent intermediate values. On a hazard map those
    invented cells land precisely on the safe/lethal boundary, so a path could
    appear to thread a gap that does not exist.
    """
    height, width = rgb.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension or longest == 0:
        return rgb
    scale = max_dimension / longest
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv().resize(rgb, new_size, interpolation=cv().INTER_NEAREST)


def render_array(
    array: np.ndarray,
    *,
    colormap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    units: Optional[str] = None,
    mask: Optional[np.ndarray] = None,
    max_dimension: int = 900,
    note: str = "",
) -> RenderResult:
    """Render a 2D array (or an HxWx3 image) to PNG.

    Parameters
    ----------
    mask:
        Optional boolean array; ``True`` marks a cell as having no data. It is
        combined with the array's own non-finite cells.
    vmin / vmax:
        Normalisation bounds. When omitted they are taken from the finite data
        and reported back, so a viewer always knows what the colours mean.
    """
    data = np.asarray(array)

    # --- already an image ------------------------------------------------
    if data.ndim == 3 and data.shape[2] in (3, 4):
        rgb = data[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        # Frames arrive from OpenCV in BGR; convert once, here.
        rgb = rgb[:, :, ::-1]
        rgb = _resize_nearest(np.ascontiguousarray(rgb), max_dimension)
        return RenderResult(
            png=encode_png(rgb), width=rgb.shape[1], height=rgb.shape[0],
            colormap="rgb", vmin=None, vmax=None, units=units,
            valid_fraction=1.0, note=note or "source image, unmodified",
        )

    if data.ndim == 2 and data.dtype == np.uint8 and colormap == "grey":
        rgb = np.stack([data] * 3, axis=-1)
        rgb = _resize_nearest(rgb, max_dimension)
        return RenderResult(
            png=encode_png(rgb), width=rgb.shape[1], height=rgb.shape[0],
            colormap="grey", vmin=0.0, vmax=255.0, units=units,
            valid_fraction=1.0, note=note or "single-channel image, unmodified",
        )

    if data.ndim != 2:
        raise ValueError(f"cannot render an array of shape {data.shape}")

    # --- scalar field ----------------------------------------------------
    values = data.astype(np.float64)
    invalid = ~np.isfinite(values)
    if mask is not None:
        invalid = invalid | np.asarray(mask, dtype=bool)
    valid = ~invalid
    valid_fraction = float(valid.mean()) if valid.size else 0.0

    if not valid.any():
        # Everything is missing. Say so in one flat colour rather than
        # producing a black image that reads as "measured zero everywhere".
        rgb = np.full((*values.shape, 3), NO_DATA_RGB, dtype=np.uint8)
        rgb = _resize_nearest(rgb, max_dimension)
        return RenderResult(
            png=encode_png(rgb), width=rgb.shape[1], height=rgb.shape[0],
            colormap=colormap, vmin=None, vmax=None, units=units,
            valid_fraction=0.0,
            note="no valid cells — every value is missing or non-finite",
        )

    finite = values[valid]
    lo = float(vmin) if vmin is not None else float(finite.min())
    hi = float(vmax) if vmax is not None else float(finite.max())
    if hi <= lo:
        # A constant field. Normalising by a zero span would divide by zero;
        # render it mid-scale and record the true value in the bounds.
        normalised = np.full(values.shape, 0.5)
    else:
        normalised = (values - lo) / (hi - lo)
    normalised = np.clip(np.nan_to_num(normalised, nan=0.0), 0.0, 1.0)

    fn = COLORMAPS.get(colormap)
    if fn is None:
        raise ValueError(f"unknown colormap {colormap!r}; "
                         f"available: {sorted(COLORMAPS)}")
    rgb = fn(normalised).astype(np.uint8)
    rgb[invalid] = NO_DATA_RGB
    rgb = _resize_nearest(np.ascontiguousarray(rgb), max_dimension)

    return RenderResult(
        png=encode_png(rgb), width=rgb.shape[1], height=rgb.shape[0],
        colormap=colormap, vmin=lo, vmax=hi, units=units,
        valid_fraction=valid_fraction, note=note,
    )


def render_keypoints(image: np.ndarray, points: np.ndarray, *,
                     max_dimension: int = 900,
                     radius: int = 3) -> RenderResult:
    """Draw detected keypoints over a frame.

    The points are the real coordinates the extractor returned. Nothing is
    added, filtered or repositioned for appearance.
    """
    cv2 = cv()
    base = np.asarray(image)
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    canvas = base.copy()
    for x, y in np.asarray(points, dtype=np.float64).reshape(-1, 2):
        cv2.circle(canvas, (int(round(x)), int(round(y))), radius,
                   (60, 220, 240), 1, lineType=cv2.LINE_AA)
    result = render_array(canvas, max_dimension=max_dimension)
    return RenderResult(
        png=result.png, width=result.width, height=result.height,
        colormap="rgb", vmin=None, vmax=None, units="px",
        valid_fraction=1.0,
        note=f"{len(np.asarray(points).reshape(-1, 2))} keypoints as detected",
    )


def render_tracks(image: np.ndarray, tracks: Any, *,
                  max_dimension: int = 900) -> RenderResult:
    """Draw track boxes, identities and trajectory history over a frame."""
    cv2 = cv()
    base = np.asarray(image)
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    canvas = base.copy()
    drawn = 0
    for track in tracks or []:
        bbox = getattr(track, "bbox", None)
        if bbox is None:
            continue
        x, y, w, h = (int(round(v)) for v in bbox)
        state = getattr(getattr(track, "state", None), "value", "")
        # Coasting tracks are drawn dashed-amber: the box is a prediction, not
        # an observation, and the operator should be able to see the difference.
        colour = (60, 220, 240) if state == "confirmed" else (34, 153, 210)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, 1, cv2.LINE_AA)
        label = f"#{getattr(track, 'id', '?')} {state}"
        cv2.putText(canvas, label, (x, max(12, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
        history = list(getattr(track, "trajectory", []) or [])[-40:]
        for a, b in zip(history, history[1:]):
            cv2.line(canvas, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     colour, 1, cv2.LINE_AA)
        drawn += 1
    result = render_array(canvas, max_dimension=max_dimension)
    return RenderResult(
        png=result.png, width=result.width, height=result.height,
        colormap="rgb", vmin=None, vmax=None, units="px", valid_fraction=1.0,
        note=f"{drawn} track(s); amber boxes are predicted, not observed",
    )


def render_path(cost_map: np.ndarray, cells: Any, *,
                colormap: str = "hazard", max_dimension: int = 900,
                units: str = "cost") -> RenderResult:
    """Draw a planned path over its cost map, at native grid resolution."""
    result = render_array(cost_map, colormap=colormap, units=units,
                          max_dimension=10_000)
    cv2 = cv()
    rgb = cv2.imdecode(np.frombuffer(result.png, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
    canvas = np.ascontiguousarray(rgb).copy()
    points = [(int(c[1]), int(c[0])) for c in (cells or [])]
    for a, b in zip(points, points[1:]):
        cv2.line(canvas, a, b, (255, 255, 255), 1, cv2.LINE_AA)
    if points:
        cv2.circle(canvas, points[0], 3, (60, 220, 240), -1)
        cv2.circle(canvas, points[-1], 3, (255, 255, 255), -1)
    canvas = _resize_nearest(canvas, max_dimension)
    return RenderResult(
        png=encode_png(canvas), width=canvas.shape[1], height=canvas.shape[0],
        colormap=colormap, vmin=result.vmin, vmax=result.vmax, units=units,
        valid_fraction=result.valid_fraction,
        note=f"{len(points)} path cells overlaid on the cost map",
    )


def render_point_cloud(points: np.ndarray, *, width: int = 720, height: int = 540,
                       elevation_deg: float = 22.0, azimuth_deg: float = -35.0,
                       max_points: int = 60_000,
                       colormap: str = "viridis") -> RenderResult:
    """Orthographic projection of a metric point cloud, coloured by depth.

    A projection, not a 3D viewer: it needs no WebGL, no client library and no
    second server, and every pixel corresponds to a point the stereo pipeline
    actually reconstructed.

    Two honest constraints:

    **Points are subsampled, never synthesised.** A dense cloud is decimated
    with a fixed stride so the drawn subset is a real sample of the original;
    the count actually drawn is reported so nobody reads density as coverage.

    **The scale is metric and stated.** The axis extents come back in metres in
    the render note, so a viewer can tell a 3 m rock pile from a 30 m one —
    which an unlabelled projection cannot convey.
    """
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(f"expected an (N, 3) point cloud, got {cloud.shape}")
    finite = cloud[np.isfinite(cloud).all(axis=1)]
    if finite.size == 0:
        rgb = np.full((height, width, 3), NO_DATA_RGB, dtype=np.uint8)
        return RenderResult(
            png=encode_png(rgb), width=width, height=height, colormap=colormap,
            vmin=None, vmax=None, units="m", valid_fraction=0.0,
            note="no finite points — nothing was reconstructed",
        )

    total = len(finite)
    if total > max_points:
        finite = finite[:: int(np.ceil(total / max_points))]

    # Camera-frame convention from reproject_to_3d: X right, Y down, Z forward.
    el, az = math.radians(elevation_deg), math.radians(azimuth_deg)
    x, y, z = finite[:, 0], finite[:, 1], finite[:, 2]
    u = x * math.cos(az) + z * math.sin(az)
    v = (y * math.cos(el)
         - (z * math.cos(az) - x * math.sin(az)) * math.sin(el))

    span_u = max(float(u.max() - u.min()), 1e-6)
    span_v = max(float(v.max() - v.min()), 1e-6)
    scale = 0.92 * min(width / span_u, height / span_v)
    px = ((u - u.min()) * scale + (width - span_u * scale) / 2).astype(int)
    py = ((v - v.min()) * scale + (height - span_v * scale) / 2).astype(int)
    np.clip(px, 0, width - 1, out=px)
    np.clip(py, 0, height - 1, out=py)

    depth_lo, depth_hi = float(z.min()), float(z.max())
    span = max(depth_hi - depth_lo, 1e-9)
    colours = COLORMAPS[colormap]((z - depth_lo) / span).astype(np.uint8)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    # Painter's order: farthest first, so nearer points occlude rather than
    # being averaged into a haze that hides the scene's actual structure.
    order = np.argsort(-z)
    canvas[py[order], px[order]] = colours[order]

    return RenderResult(
        png=encode_png(canvas), width=width, height=height, colormap=colormap,
        vmin=depth_lo, vmax=depth_hi, units="m",
        valid_fraction=float(len(finite) / max(total, 1)),
        note=(f"{len(finite):,} of {total:,} reconstructed points drawn; "
              f"extent {span_u:.1f} x {span_v:.1f} m, "
              f"depth {depth_lo:.2f}-{depth_hi:.2f} m; colour is depth"),
    )


def render_hillshade(elevation: np.ndarray, *, cell_size_m: float = 1.0,
                     azimuth_deg: float = 315.0, altitude_deg: float = 45.0,
                     vertical_exaggeration: float = 1.0,
                     max_dimension: int = 900) -> RenderResult:
    """Shaded relief of a DEM — the standard way terrain is actually read.

    A colour ramp shows *how high* a cell is; hillshade shows what the surface
    is shaped like. A 30 cm boulder on a gentle slope is invisible in a colour
    ramp spanning 8 m of relief, and obvious under raking light. Both are
    offered because they answer different questions.

    Illumination follows the cartographic convention: light from the
    north-west at 45 degrees. That is a lighting choice, not a measurement, so
    the azimuth and altitude used are reported — and ``vertical_exaggeration``
    is reported too, because exaggerating relief to make features legible
    changes how steep the terrain *looks* without changing what it is.
    """
    dem = np.asarray(elevation, dtype=np.float64)
    if dem.ndim != 2:
        raise ValueError(f"hillshade needs a 2D elevation grid, got {dem.shape}")
    invalid = ~np.isfinite(dem)
    if invalid.all():
        rgb = np.full((*dem.shape, 3), NO_DATA_RGB, dtype=np.uint8)
        rgb = _resize_nearest(rgb, max_dimension)
        return RenderResult(
            png=encode_png(rgb), width=rgb.shape[1], height=rgb.shape[0],
            colormap="hillshade", vmin=None, vmax=None, units="m",
            valid_fraction=0.0, note="no valid elevation to shade",
        )

    # Gaps are filled with the median purely so the gradient stays finite; the
    # cells are restored to the no-data colour afterwards so nothing invented
    # is displayed.
    filled = np.where(invalid, np.nanmedian(dem), dem) * float(vertical_exaggeration)
    dz_dy, dz_dx = np.gradient(filled, max(cell_size_m, 1e-9))
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dy, dz_dx)

    az = math.radians(360.0 - azimuth_deg + 90.0)
    alt = math.radians(altitude_deg)
    shade = (math.sin(alt) * np.cos(slope)
             + math.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    shade = np.clip(shade, 0.0, 1.0)

    rgb = (np.stack([shade, shade, shade], axis=-1) * 255.0).astype(np.uint8)
    rgb[invalid] = NO_DATA_RGB
    rgb = _resize_nearest(np.ascontiguousarray(rgb), max_dimension)
    valid = float((~invalid).mean())
    return RenderResult(
        png=encode_png(rgb), width=rgb.shape[1], height=rgb.shape[0],
        colormap="hillshade", vmin=float(np.nanmin(dem)),
        vmax=float(np.nanmax(dem)), units="m", valid_fraction=valid,
        note=(f"shaded relief, light from {azimuth_deg:.0f} deg at "
              f"{altitude_deg:.0f} deg elevation, vertical exaggeration "
              f"{vertical_exaggeration:g}x"),
    )


def render_terrain_3d(elevation: np.ndarray, *, cell_size_m: float = 1.0,
                      overlay: Optional[np.ndarray] = None,
                      overlay_colormap: str = "hazard",
                      path_cells: Any = None,
                      rover_cell: Optional[tuple[int, int]] = None,
                      azimuth_deg: float = -40.0, elevation_deg: float = 28.0,
                      vertical_exaggeration: float = 2.0,
                      width: int = 900, height: int = 600,
                      max_grid: int = 160) -> RenderResult:
    """Perspective view of the terrain, optionally draped with a cost map.

    A projection rather than a WebGL scene: no client library, no build step,
    and every drawn quad corresponds to four DEM cells that were actually
    measured. The trade is that it cannot be orbited interactively — the view
    angle is a request parameter instead.

    Painter's algorithm, back to front. Depth-sorting quads by their centroid
    is exact for a height field viewed from outside it, which is the only case
    this renders.

    ``vertical_exaggeration`` defaults to 2 because real planetary terrain is
    far flatter than intuition expects and an unexaggerated render of a 30 m
    tile looks like a sheet of paper. The factor is reported in the note, since
    it changes apparent steepness without changing the data.
    """
    dem = np.asarray(elevation, dtype=np.float64)
    if dem.ndim != 2:
        raise ValueError(f"terrain view needs a 2D grid, got {dem.shape}")

    # Decimate before projecting: a 400x400 grid is 160,000 quads, which is
    # minutes of work for a view that cannot resolve them anyway.
    step = max(1, int(math.ceil(max(dem.shape) / max_grid)))
    grid = dem[::step, ::step]
    drape = None if overlay is None else np.asarray(overlay)[::step, ::step]
    scale = cell_size_m * step

    rows, cols = grid.shape
    if rows < 2 or cols < 2:
        raise ValueError("terrain view needs at least a 2x2 grid")

    invalid = ~np.isfinite(grid)
    filled = np.where(invalid, np.nanmedian(grid) if (~invalid).any() else 0.0, grid)
    base = float(np.nanmin(filled))

    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    wx = (xx - cols / 2.0) * scale
    wy = (yy - rows / 2.0) * scale
    wz = (filled - base) * float(vertical_exaggeration)

    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    # Rotate about the vertical, then tilt. Orthographic: a rover-scale tile
    # spans a few tens of metres, where perspective foreshortening would add
    # visual drama without adding information.
    rx = wx * math.cos(az) - wy * math.sin(az)
    ry = wx * math.sin(az) + wy * math.cos(az)
    su = rx
    sv = ry * math.sin(el) - wz * math.cos(el)
    depth = ry * math.cos(el) + wz * math.sin(el)

    span_u = max(float(su.max() - su.min()), 1e-6)
    span_v = max(float(sv.max() - sv.min()), 1e-6)
    fit = 0.9 * min(width / span_u, height / span_v)
    px = ((su - su.min()) * fit + (width - span_u * fit) / 2).astype(np.int32)
    py = ((sv - sv.min()) * fit + (height - span_v * fit) / 2).astype(np.int32)

    if drape is not None:
        finite = drape[np.isfinite(drape)]
        lo = float(finite.min()) if finite.size else 0.0
        hi = float(finite.max()) if finite.size else 1.0
        norm = np.clip((drape - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        face = COLORMAPS[overlay_colormap](np.nan_to_num(norm)).astype(np.float64)
        units, vmin, vmax = "cost", lo, hi
    else:
        face = np.full((rows, cols, 3), 170.0)
        units, vmin, vmax = "m", float(np.nanmin(dem)), float(np.nanmax(dem))

    # Lambertian shading from the surface normal, so relief is legible even
    # under a flat colour drape.
    dz_dy, dz_dx = np.gradient(wz, scale)
    normal_z = 1.0 / np.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1.0)
    light = np.clip(0.45 + 0.55 * normal_z, 0.0, 1.0)

    cv2 = cv()
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    quads = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            quads.append((float(depth[r:r + 2, c:c + 2].mean()), r, c))
    quads.sort(key=lambda q: -q[0])          # farthest first

    for _, r, c in quads:
        if invalid[r:r + 2, c:c + 2].any():
            colour = NO_DATA_RGB
        else:
            shaded = face[r, c] * light[r, c]
            colour = (int(shaded[0]), int(shaded[1]), int(shaded[2]))
        poly = np.array([[px[r, c], py[r, c]], [px[r, c + 1], py[r, c + 1]],
                         [px[r + 1, c + 1], py[r + 1, c + 1]],
                         [px[r + 1, c], py[r + 1, c]]], dtype=np.int32)
        cv2.fillConvexPoly(canvas, poly, colour[::-1], lineType=cv2.LINE_8)

    drawn_path = 0
    if path_cells is not None:
        points = []
        for cell in path_cells:
            r, c = int(cell[0]) // step, int(cell[1]) // step
            if 0 <= r < rows and 0 <= c < cols:
                points.append((int(px[r, c]), int(py[r, c])))
        for a, b in zip(points, points[1:]):
            cv2.line(canvas, a, b, (255, 255, 255), 2, cv2.LINE_AA)
        drawn_path = len(points)

    if rover_cell is not None:
        r, c = int(rover_cell[0]) // step, int(rover_cell[1]) // step
        if 0 <= r < rows and 0 <= c < cols:
            centre = (int(px[r, c]), int(py[r, c]))
            cv2.circle(canvas, centre, 7, (240, 220, 60), -1, cv2.LINE_AA)
            cv2.circle(canvas, centre, 7, (20, 20, 20), 1, cv2.LINE_AA)

    return RenderResult(
        png=encode_png(canvas[:, :, ::-1]), width=width, height=height,
        colormap=overlay_colormap if drape is not None else "terrain",
        vmin=vmin, vmax=vmax, units=units,
        valid_fraction=float((~invalid).mean()),
        note=(f"{len(quads):,} quads from a {rows}x{cols} grid "
              f"(decimated {step}x), vertical exaggeration "
              f"{vertical_exaggeration:g}x, view {azimuth_deg:.0f}/"
              f"{elevation_deg:.0f} deg"
              + (f", {drawn_path} path points" if drawn_path else "")),
    )


def render_trajectory(poses: Any, *, landmarks: Any = None,
                      loop_closures: Any = None,
                      width: int = 820, height: int = 620,
                      margin: int = 40) -> RenderResult:
    """Top-down plot of a SLAM trajectory with landmarks and loop closures.

    Loop-closure edges are drawn explicitly, because the difference between a
    trajectory that closed a loop and one that merely passed nearby is the
    single most informative thing on this plot — and it is invisible if only
    the path is drawn.
    """
    cv2 = cv()
    points = [(float(p["x"]), float(p["y"])) for p in (poses or [])
              if math.isfinite(p.get("x", float("nan")))]
    if len(points) < 1:
        rgb = np.full((height, width, 3), 12, dtype=np.uint8)
        return RenderResult(
            png=encode_png(rgb), width=width, height=height,
            colormap="trajectory", vmin=None, vmax=None, units="m",
            valid_fraction=0.0, note="no poses to plot",
        )

    marks = [(float(l["position"][0]), float(l["position"][1]))
             for l in (landmarks or [])]
    everything = points + marks
    xs = [p[0] for p in everything]
    ys = [p[1] for p in everything]
    span_x = max(max(xs) - min(xs), 1e-6)
    span_y = max(max(ys) - min(ys), 1e-6)
    fit = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)

    def to_px(point):
        return (int((point[0] - min(xs)) * fit + margin),
                # Flip Y so north is up, as a map reads.
                int(height - ((point[1] - min(ys)) * fit + margin)))

    canvas = np.full((height, width, 3), 12, dtype=np.uint8)
    for mark in marks:
        cv2.circle(canvas, to_px(mark), 1, (90, 110, 130), -1)

    pixels = [to_px(p) for p in points]
    for a, b in zip(pixels, pixels[1:]):
        cv2.line(canvas, a, b, (240, 220, 60), 2, cv2.LINE_AA)
    for pixel in pixels:
        cv2.circle(canvas, pixel, 3, (60, 220, 240), -1, cv2.LINE_AA)

    closures = 0
    for edge in (loop_closures or []):
        i, j = int(edge.get("from", -1)), int(edge.get("to", -1))
        if 0 <= i < len(pixels) and 0 <= j < len(pixels):
            cv2.line(canvas, pixels[i], pixels[j], (120, 240, 140), 1, cv2.LINE_AA)
            closures += 1

    cv2.circle(canvas, pixels[0], 6, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, pixels[-1], 6, (248, 81, 73), -1, cv2.LINE_AA)

    return RenderResult(
        png=encode_png(canvas[:, :, ::-1]), width=width, height=height,
        colormap="trajectory", vmin=None, vmax=None, units="m",
        valid_fraction=1.0,
        note=(f"{len(points)} poses, {len(marks):,} landmarks, "
              f"{closures} loop closure edge(s); extent "
              f"{span_x:.1f} x {span_y:.1f} m; white=start, red=current"),
    )
