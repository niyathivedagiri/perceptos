"""
perceptos.calibration.chessboard
================================

Camera intrinsic calibration, with quality gating that actually rejects.

The usual mistake is to treat ``cv2.calibrateCamera``'s returned RMS as the
quality metric and stop there. RMS is necessary but not sufficient: you can
achieve an RMS of 0.2 px from twenty views all taken from the same angle in
the middle of the frame, and the resulting distortion coefficients will be
badly wrong at the edges, which is exactly where distortion matters.

This module therefore gates on three independent things:

* **RMS reprojection error** — how well the model fits the observations.
* **Corner coverage** — the fraction of the image area the detected corners
  actually span. Low coverage means the edges are extrapolated, not measured.
* **View diversity** — the spread of board orientations. Views that are all
  fronto-parallel leave the focal length and depth poorly separated, which is
  the classic under-constrained calibration.

Every calibration is saved with the metrics that produced it, so a stereo rig
loaded from disk six months later still declares how much it should be
trusted.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..core.cv_compat import CapabilityError, cv
from ..core.provenance import Source, Traced, traced
from ..core.stage import FrameContext, PipelineStage

__all__ = [
    "CalibrationResult",
    "ChessboardCalibrator",
    "UndistortStage",
    "CalibrationError",
]


class CalibrationError(RuntimeError):
    """Calibration could not be computed or failed its quality gate."""


@dataclass(slots=True)
class CalibrationResult:
    """Intrinsics plus the evidence that justifies believing them."""

    camera_matrix: np.ndarray          # 3x3 K
    distortion: np.ndarray             # 1x5 (k1 k2 p1 p2 k3)
    image_size: tuple[int, int]        # (width, height)
    rms_px: float
    per_view_errors: list[float]
    coverage: float                    # fraction of image area spanned by corners
    orientation_spread_deg: float
    n_views: int
    pattern: tuple[int, int]
    square_size_m: float

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])

    @property
    def fov_degrees(self) -> tuple[float, float]:
        w, h = self.image_size
        return (
            float(math.degrees(2 * math.atan(w / (2 * self.fx)))),
            float(math.degrees(2 * math.atan(h / (2 * self.fy)))),
        )

    @property
    def worst_view_error(self) -> float:
        return max(self.per_view_errors) if self.per_view_errors else 0.0

    @property
    def grade(self) -> str:
        """Overall verdict. The weakest of the three criteria decides."""
        if self.rms_px > 2.0 or self.coverage < 0.25 or self.n_views < 5:
            return "reject"
        if self.rms_px > 1.0 or self.coverage < 0.45 or self.orientation_spread_deg < 10:
            return "marginal"
        if self.rms_px > 0.5 or self.coverage < 0.65:
            return "good"
        return "excellent"

    @property
    def confidence(self) -> float:
        """Map the three criteria onto [0, 1] for the provenance record."""
        error_score = 1.0 / (1.0 + self.rms_px)
        coverage_score = min(1.0, self.coverage / 0.7)
        diversity_score = min(1.0, self.orientation_spread_deg / 25.0)
        return float(max(0.0, min(1.0,
            0.5 * error_score + 0.3 * coverage_score + 0.2 * diversity_score)))

    def warnings(self) -> list[str]:
        """Specific, actionable problems — not a generic 'quality is low'."""
        out: list[str] = []
        if self.n_views < 10:
            out.append(f"only {self.n_views} views; 15-20 is the usual target")
        if self.coverage < 0.45:
            out.append(
                f"corners span {self.coverage:.0%} of the frame; move the board "
                "into the corners and edges where distortion is strongest"
            )
        if self.orientation_spread_deg < 10:
            out.append(
                f"board orientations vary by only {self.orientation_spread_deg:.0f} deg; "
                "tilt the board between captures or focal length stays "
                "poorly separated from working distance"
            )
        if self.worst_view_error > 3 * self.rms_px and self.per_view_errors:
            out.append(
                f"one view has {self.worst_view_error:.2f} px error vs "
                f"{self.rms_px:.2f} px overall; consider dropping it"
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion": self.distortion.ravel().tolist(),
            "image_size": list(self.image_size),
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "fov_degrees": list(self.fov_degrees),
            "rms_px": round(self.rms_px, 4),
            "worst_view_error_px": round(self.worst_view_error, 4),
            "per_view_errors": [round(e, 4) for e in self.per_view_errors],
            "coverage": round(self.coverage, 4),
            "orientation_spread_deg": round(self.orientation_spread_deg, 2),
            "n_views": self.n_views,
            "pattern": list(self.pattern),
            "square_size_m": self.square_size_m,
            "grade": self.grade,
            "confidence": round(self.confidence, 3),
            "warnings": self.warnings(),
        }

    # ------------------------------------------------------------- persistence

    def save(self, path: str | Path) -> Path:
        """Write NPZ (arrays) plus a sidecar JSON (human-readable metrics)."""
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            camera_matrix=self.camera_matrix,
            distortion=self.distortion,
            image_size=np.array(self.image_size),
            rms_px=np.array(self.rms_px),
            per_view_errors=np.array(self.per_view_errors),
            coverage=np.array(self.coverage),
            orientation_spread_deg=np.array(self.orientation_spread_deg),
            n_views=np.array(self.n_views),
            pattern=np.array(self.pattern),
            square_size_m=np.array(self.square_size_m),
        )
        path.with_suffix(".json").write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationResult":
        data = np.load(Path(path).with_suffix(".npz"))
        return cls(
            camera_matrix=data["camera_matrix"],
            distortion=data["distortion"],
            image_size=tuple(int(v) for v in data["image_size"]),
            rms_px=float(data["rms_px"]),
            per_view_errors=[float(v) for v in data["per_view_errors"]],
            coverage=float(data["coverage"]),
            orientation_spread_deg=float(data["orientation_spread_deg"]),
            n_views=int(data["n_views"]),
            pattern=tuple(int(v) for v in data["pattern"]),
            square_size_m=float(data["square_size_m"]),
        )

    def traced(self) -> Traced["CalibrationResult"]:
        return traced(
            self,
            source=Source.CALCULATED,
            algorithm="cv2.calibrateCamera(Zhang planar chessboard)",
            confidence=self.confidence,
            units="px",
            label="camera_calibration",
            parameters={"pattern": list(self.pattern), "n_views": self.n_views,
                        "grade": self.grade},
        )


class ChessboardCalibrator:
    """Collect chessboard views, then solve for intrinsics.

    Parameters
    ----------
    pattern:
        Number of *inner* corners, ``(columns, rows)``. A board with 10x7
        squares has a 9x6 inner-corner pattern — the off-by-one that ruins
        more calibrations than any other single mistake.
    square_size_m:
        Physical edge length of one square. Sets the metric scale, so an
        error here scales every downstream distance measurement.
    """

    def __init__(self, *, pattern: tuple[int, int] = (9, 6),
                 square_size_m: float = 0.025,
                 min_views: int = 5,
                 subpixel_window: int = 11) -> None:
        if pattern[0] < 3 or pattern[1] < 3:
            raise ValueError("pattern must be at least 3x3 inner corners")
        if pattern[0] == pattern[1]:
            raise ValueError(
                "a square inner-corner pattern is rotationally ambiguous; "
                "use a board with different column and row counts"
            )
        if square_size_m <= 0:
            raise ValueError("square_size_m must be positive")
        self.pattern = (int(pattern[0]), int(pattern[1]))
        self.square_size_m = float(square_size_m)
        self.min_views = int(min_views)
        self.subpixel_window = int(subpixel_window)
        self.image_points: list[np.ndarray] = []
        self.image_size: Optional[tuple[int, int]] = None
        self._object_grid = self._make_object_grid()

    def _make_object_grid(self) -> np.ndarray:
        cols, rows = self.pattern
        grid = np.zeros((rows * cols, 3), np.float32)
        grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        return grid * self.square_size_m

    # ------------------------------------------------------------- collection

    def add_view(self, gray: np.ndarray) -> bool:
        """Detect the board in one greyscale image. Returns whether it was found."""
        cv2 = cv()
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if self.image_size is None:
            self.image_size = (int(gray.shape[1]), int(gray.shape[0]))
        elif (int(gray.shape[1]), int(gray.shape[0])) != self.image_size:
            raise CalibrationError(
                "all calibration views must share one image size; got "
                f"{(gray.shape[1], gray.shape[0])} after {self.image_size}"
            )

        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                 | cv2.CALIB_CB_FAST_CHECK)
        found, corners = cv2.findChessboardCorners(gray, self.pattern, flags)
        if not found:
            return False
        # Sub-pixel refinement: without it the RMS is dominated by corner
        # quantisation rather than by lens model error.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(
            gray, corners, (self.subpixel_window, self.subpixel_window), (-1, -1), criteria
        )
        self.image_points.append(corners)
        return True

    @property
    def n_views(self) -> int:
        return len(self.image_points)

    # ---------------------------------------------------------------- metrics

    def _coverage(self) -> float:
        """Fraction of image area spanned by the union of all corner sets."""
        if not self.image_points or self.image_size is None:
            return 0.0
        pts = np.vstack([c.reshape(-1, 2) for c in self.image_points])
        width, height = self.image_size
        # Occupancy over a coarse grid — a convex hull would over-credit a
        # sparse ring of views that never covers the middle or the corners.
        bins = 12
        cols = np.clip((pts[:, 0] / width * bins).astype(int), 0, bins - 1)
        rows = np.clip((pts[:, 1] / height * bins).astype(int), 0, bins - 1)
        occupied = np.zeros((bins, bins), dtype=bool)
        occupied[rows, cols] = True
        return float(occupied.mean())

    @staticmethod
    def _orientation_spread(rvecs: list[np.ndarray]) -> float:
        """Standard deviation of board tilt across views, in degrees."""
        if len(rvecs) < 2:
            return 0.0
        angles = [float(np.degrees(np.linalg.norm(r))) for r in rvecs]
        return float(np.std(angles))

    # -------------------------------------------------------------- calibrate

    def calibrate(self, *, strict: bool = False) -> CalibrationResult:
        """Solve for K and the distortion coefficients."""
        if self.n_views < self.min_views:
            raise CalibrationError(
                f"need at least {self.min_views} detected views, have {self.n_views}"
            )
        cv2 = cv()
        object_points = [self._object_grid] * self.n_views
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, self.image_points, self.image_size, None, None
        )

        # Per-view error, recomputed rather than trusted: the aggregate RMS
        # hides a single bad view, and a bad view is worth knowing about.
        per_view: list[float] = []
        for i in range(self.n_views):
            projected, _ = cv2.projectPoints(
                object_points[i], rvecs[i], tvecs[i], K, dist
            )
            residual = self.image_points[i].reshape(-1, 2) - projected.reshape(-1, 2)
            per_view.append(float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))))

        result = CalibrationResult(
            camera_matrix=np.asarray(K, dtype=np.float64),
            distortion=np.asarray(dist, dtype=np.float64),
            image_size=self.image_size,
            rms_px=float(rms),
            per_view_errors=per_view,
            coverage=self._coverage(),
            orientation_spread_deg=self._orientation_spread(rvecs),
            n_views=self.n_views,
            pattern=self.pattern,
            square_size_m=self.square_size_m,
        )
        if strict and result.grade == "reject":
            raise CalibrationError(
                f"calibration rejected (grade={result.grade}): "
                + "; ".join(result.warnings())
            )
        return result

    def reset(self) -> None:
        self.image_points.clear()
        self.image_size = None


class UndistortStage(PipelineStage):
    """Removes lens distortion using a precomputed remap.

    The remap tables are built once and reused. Calling ``cv2.undistort`` per
    frame recomputes them every time, which costs more than the remap itself
    and is a common reason a pipeline mysteriously runs at half speed.
    """

    name = "calibration"
    requires = ("frame",)
    provides = ("undistorted", "gray")
    source = Source.CALCULATED

    def __init__(self, calibration: CalibrationResult, *, alpha: float = 0.0,
                 **params: Any) -> None:
        super().__init__(**params)
        self.calibration = calibration
        self.alpha = float(alpha)
        self._maps: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._new_matrix: Optional[np.ndarray] = None

    def _ensure_maps(self) -> tuple[np.ndarray, np.ndarray]:
        if self._maps is None:
            cv2 = cv()
            cal = self.calibration
            new_matrix, _ = cv2.getOptimalNewCameraMatrix(
                cal.camera_matrix, cal.distortion, cal.image_size, self.alpha
            )
            self._new_matrix = new_matrix
            self._maps = cv2.initUndistortRectifyMap(
                cal.camera_matrix, cal.distortion, None, new_matrix,
                cal.image_size, cv2.CV_16SC2,
            )
        return self._maps

    @property
    def effective_matrix(self) -> np.ndarray:
        """Intrinsics valid for the *undistorted* image, not the raw one."""
        self._ensure_maps()
        return self._new_matrix if self._new_matrix is not None else self.calibration.camera_matrix

    def run(self, ctx: FrameContext) -> str:
        cv2 = cv()
        (frame,) = ctx.require("frame")
        map1, map2 = self._ensure_maps()
        undistorted = cv2.remap(frame.value, map1, map2, cv2.INTER_LINEAR)
        self.emit(ctx, "undistorted", undistorted,
                  algorithm="cv2.remap(initUndistortRectifyMap)",
                  parents=[frame], confidence=self.calibration.confidence, units="px",
                  parameters={"alpha": self.alpha, "grade": self.calibration.grade},
                  artefact=undistorted)
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        self.emit(ctx, "gray", gray, algorithm="cv2.cvtColor(BGR2GRAY)",
                  parents=[frame], units="px")
        return f"undistorted (calibration grade: {self.calibration.grade})"
