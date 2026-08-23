"""
perceptos.stereo.rectify
========================

Epipolar rectification of real Perseverance stereo pairs.

This is the step that was missing, and it is the one that turns
``VALID_STEREO`` into ``RECTIFIED_STEREO`` — the only state in which PerceptOS
will report a metric depth.

The problem
-----------

Block matching (``StereoBM``, ``StereoSGBM``) searches along image scanlines.
That is only correct when the two cameras' epipolar lines *are* the scanlines,
which requires the image planes to be coplanar and row-aligned. Raw Navcam
frames are not: the two cameras are toed in slightly and mounted with their own
small misalignments. Matching raw frames along rows produces a dense, confident
and completely wrong disparity map — which is precisely why
:class:`~perceptos.sensors.dataset_source.StereoDatasetSource` refuses to serve
unrectified pairs to a matcher.

The solution, and why it needs no invented numbers
--------------------------------------------------

JPL publishes the CAHV camera model for the flight cameras. That model contains
everything OpenCV's rectification needs, once decomposed:

.. code-block:: text

    hc = H·A                    horizontal principal point (px)
    vc = V·A                    vertical principal point (px)
    hs = |H - hc·A|             horizontal focal length (px)
    vs = |V - vc·A|             vertical focal length (px)

    Hn = (H - hc·A) / hs        image +x direction, world frame
    Vn = (V - vc·A) / vs        image +y direction, world frame

    R = [Hn; Vn; A]             rotation, world -> camera
    K = [[hs, 0, hc],
         [0, vs, vc],
         [0,  0,  1]]

    C                           camera centre, world frame

The relative pose between the two cameras follows directly, and
``cv2.stereoRectify`` produces the rectifying homographies and the reprojection
matrix ``Q``. Every quantity above is read from the archive. Nothing is
assumed, and if the model is absent this module raises rather than supplying a
default — which is the whole point.

**Distortion is not corrected.** CAHVOR's ``O`` and ``R`` terms describe a
radial model that OpenCV's ``(k1,k2,p1,p2,k3)`` cannot represent directly, and
converting between them approximately would introduce error while *claiming*
correction. :attr:`RectificationResult.distortion_corrected` is therefore
always ``False``, and the residual is reported as an epipolar error in pixels
so the operator can judge whether it matters at their working distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..core.cv_compat import CapabilityError, cv
from ..core.provenance import Source, Traced, traced
from ..data.sources import StereoStatus
from .depth import StereoRig

__all__ = [
    "PinholeModel",
    "RectificationResult",
    "cahv_to_pinhole",
    "rectify_pair",
    "RectificationError",
]


class RectificationError(RuntimeError):
    """The pair cannot be rectified, with the reason stated."""


@dataclass(frozen=True, slots=True)
class PinholeModel:
    """A CAHV model decomposed into the pinhole form OpenCV expects."""

    K: np.ndarray                 # 3x3 intrinsics
    R: np.ndarray                 # 3x3 rotation, world -> camera
    C: np.ndarray                 # 3-vector camera centre, world frame

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def is_right_handed(self) -> bool:
        """A valid rotation has determinant +1; -1 means a mirrored basis."""
        return bool(np.linalg.det(self.R) > 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fx": round(self.fx, 3), "fy": round(self.fy, 3),
            "cx": round(self.cx, 3), "cy": round(self.cy, 3),
            "C": [round(float(v), 6) for v in self.C],
            "R_determinant": round(float(np.linalg.det(self.R)), 6),
            "orthonormal_error": round(self.orthonormality_error(), 8),
        }

    def orthonormality_error(self) -> float:
        """How far ``R`` is from a true rotation matrix.

        The CAHV vectors are published to finite precision and the H/V
        directions are not exactly perpendicular, so the derived basis is only
        approximately orthonormal. Reporting the residual lets a caller judge
        the model rather than assume it is exact.
        """
        return float(np.abs(self.R @ self.R.T - np.eye(3)).max())


def cahv_to_pinhole(camera_model: Any) -> PinholeModel:
    """Decompose a CAHV-family model into ``(K, R, C)``.

    Raises :class:`RectificationError` when the model is absent or incomplete —
    never returns a default.
    """
    if camera_model is None or not getattr(camera_model, "is_usable", False):
        raise RectificationError(
            "no usable CAHV camera model. Rectification needs the published "
            "C, A, H and V vectors; PerceptOS will not assume a focal length "
            "or a baseline. Status stays CALIBRATION_UNAVAILABLE."
        )
    C = np.asarray(camera_model.C, dtype=np.float64)
    A = np.asarray(camera_model.A, dtype=np.float64)
    H = np.asarray(camera_model.H, dtype=np.float64)
    V = np.asarray(camera_model.V, dtype=np.float64)

    norm_a = float(np.linalg.norm(A))
    if norm_a < 1e-9:
        raise RectificationError("camera model axis vector A is degenerate")
    A = A / norm_a                                # A must be unit length

    hc = float(H @ A)
    vc = float(V @ A)
    h_perp = H - hc * A
    v_perp = V - vc * A
    hs = float(np.linalg.norm(h_perp))
    vs = float(np.linalg.norm(v_perp))
    if hs < 1e-9 or vs < 1e-9:
        raise RectificationError(
            f"degenerate camera model: focal lengths ({hs:.3g}, {vs:.3g}) px"
        )

    Hn = h_perp / hs
    Vn = v_perp / vs
    R = np.vstack([Hn, Vn, A])
    K = np.array([[hs, 0.0, hc], [0.0, vs, vc], [0.0, 0.0, 1.0]])
    return PinholeModel(K=K, R=R, C=C)


@dataclass(slots=True)
class RectificationResult:
    """Rectified images plus the geometry and residuals behind them."""

    left: np.ndarray
    right: np.ndarray
    rig: StereoRig
    Q: np.ndarray                       # 4x4 disparity-to-depth reprojection
    baseline_m: float
    status: StereoStatus
    left_model: PinholeModel
    right_model: PinholeModel
    epipolar_error_px: Optional[float] = None
    distortion_corrected: bool = False
    valid_roi: tuple[int, int, int, int] = (0, 0, 0, 0)
    notes: list[str] = field(default_factory=list)

    @property
    def depth_valid(self) -> bool:
        return self.status.depth_valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "stereo_status": self.status.value,
            "depth_valid": self.depth_valid,
            "baseline_m": round(self.baseline_m, 6),
            "focal_length_px": round(self.rig.fx, 3),
            "principal_point": [round(self.rig.cx, 2), round(self.rig.cy, 2)],
            "max_reliable_range_m": round(self.rig.max_reliable_range(), 2),
            "epipolar_error_px": (round(self.epipolar_error_px, 3)
                                  if self.epipolar_error_px is not None else None),
            "distortion_corrected": self.distortion_corrected,
            "valid_roi": list(self.valid_roi),
            "left_model": self.left_model.to_dict(),
            "right_model": self.right_model.to_dict(),
            "image_size": [self.left.shape[1], self.left.shape[0]],
            "notes": list(self.notes),
        }


def _measure_epipolar_error(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    """Median vertical disparity between matched features after rectification.

    On a correctly rectified pair a feature sits on the same row in both
    images, so this should be well under a pixel. It is the honest check on
    whether the published camera model was good enough — reporting
    "rectified" without measuring it would be a claim rather than a result.
    """
    cv2 = cv()
    try:
        detector = (getattr(cv2, "ORB_create", None)
                    or getattr(getattr(cv2, "ORB", None), "create", None))
        if detector is None:
            return None
        orb = detector(nfeatures=600)
        kp1, des1 = orb.detectAndCompute(left, None)
        kp2, des2 = orb.detectAndCompute(right, None)
        if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
            return None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw = matcher.knnMatch(des1, des2, k=2)
        deltas = []
        for candidates in raw:
            if len(candidates) < 2:
                continue
            best, second = candidates
            if best.distance < 0.75 * second.distance:      # Lowe's ratio test
                deltas.append(abs(kp1[best.queryIdx].pt[1] - kp2[best.trainIdx].pt[1]))
        return float(np.median(deltas)) if len(deltas) >= 8 else None
    except Exception:  # noqa: BLE001 - a diagnostic must not break the pipeline
        return None


def rectify_pair(
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_camera: Any,
    right_camera: Any,
    *,
    alpha: float = 0.0,
    measure_residual: bool = True,
) -> RectificationResult:
    """Rectify a real stereo pair using its published CAHV camera models.

    Parameters
    ----------
    alpha:
        ``0`` crops to the region valid in both images (no black borders);
        ``1`` keeps all pixels. Zero is the default because a matcher run over
        black border regions produces spurious matches at the frame edge.
    """
    cv2 = cv()
    if left_image.shape[:2] != right_image.shape[:2]:
        raise RectificationError(
            f"images must share a size; got {left_image.shape[:2]} and "
            f"{right_image.shape[:2]}"
        )

    left_model = cahv_to_pinhole(left_camera)
    right_model = cahv_to_pinhole(right_camera)

    notes: list[str] = []
    for name, model in (("left", left_model), ("right", right_model)):
        residual = model.orthonormality_error()
        if residual > 1e-3:
            notes.append(
                f"{name} camera basis is {residual:.2e} from orthonormal; the "
                "published H and V vectors are not exactly perpendicular"
            )
        if not model.is_right_handed:
            notes.append(f"{name} camera basis is left-handed (det R < 0)")

    # Relative pose: rotation and translation from left camera to right.
    R_rel = right_model.R @ left_model.R.T
    T_rel = right_model.R @ (left_model.C - right_model.C)
    baseline = float(np.linalg.norm(left_model.C - right_model.C))
    if baseline < 1e-6:
        raise RectificationError(
            "the two camera centres coincide; this is not a stereo pair"
        )

    height, width = left_image.shape[:2]
    size = (width, height)
    # Zero distortion coefficients: CAHVOR's O/R terms are a different model
    # and converting them approximately would claim a correction not made.
    zero_dist = np.zeros(5)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        left_model.K, zero_dist, right_model.K, zero_dist,
        size, R_rel, T_rel.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(
        left_model.K, zero_dist, R1, P1, size, cv2.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(
        right_model.K, zero_dist, R2, P2, size, cv2.CV_16SC2)

    left_rect = cv2.remap(left_image, map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_image, map2x, map2y, cv2.INTER_LINEAR)

    # After rectification the effective intrinsics are those in P1, and the
    # baseline is recoverable from P2's translation term: P2[0,3] = -fx * B.
    fx = float(P1[0, 0])
    fy = float(P1[1, 1])
    cx = float(P1[0, 2])
    cy = float(P1[1, 2])
    rectified_baseline = abs(float(P2[0, 3]) / fx) if fx else baseline
    if abs(rectified_baseline - baseline) > 0.05 * max(baseline, 1e-6):
        notes.append(
            f"rectified baseline {rectified_baseline:.4f} m differs from the "
            f"model separation {baseline:.4f} m by more than 5%"
        )

    rig = StereoRig(fx=fx, fy=fy, cx=cx, cy=cy, baseline_m=rectified_baseline)

    error = None
    if measure_residual:
        grey_left = (left_rect if left_rect.ndim == 2
                     else cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY))
        grey_right = (right_rect if right_rect.ndim == 2
                      else cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY))
        error = _measure_epipolar_error(grey_left, grey_right)
        if error is None:
            notes.append("epipolar residual not measurable: too few matches")
        elif error > 2.0:
            notes.append(
                f"epipolar residual {error:.2f} px is large; scanline matching "
                "will be degraded. The published model may not fit this pair."
            )

    # The status is earned, not assigned: a pair only becomes RECTIFIED_STEREO
    # when the residual was measured and is small enough for row-wise matching.
    if error is not None and error <= 2.0:
        status = StereoStatus.RECTIFIED_STEREO
    elif error is None:
        status = StereoStatus.VALID_STEREO
        notes.append("status held at VALID_STEREO: rectification was applied "
                     "but could not be verified, so depth is not offered")
    else:
        status = StereoStatus.VALID_STEREO
        notes.append("status held at VALID_STEREO: measured epipolar error "
                     "exceeds the 2 px threshold for scanline matching")

    return RectificationResult(
        left=left_rect, right=right_rect, rig=rig, Q=np.asarray(Q),
        baseline_m=rectified_baseline, status=status,
        left_model=left_model, right_model=right_model,
        epipolar_error_px=error, distortion_corrected=False,
        valid_roi=tuple(int(v) for v in roi1), notes=notes,
    )


def rectified_traced(result: RectificationResult, *,
                     parents: tuple[Traced[Any], ...] = ()) -> Traced[RectificationResult]:
    """Export a rectification with provenance for the lineage graph."""
    # Confidence falls with epipolar residual: a sub-pixel result is
    # trustworthy, a 2 px one barely supports matching at all.
    if result.epipolar_error_px is None:
        confidence = 0.4
    else:
        confidence = float(max(0.05, min(0.95, 1.0 / (1.0 + result.epipolar_error_px))))
    return traced(
        result,
        source=Source.CALCULATED,
        algorithm="rectify_pair(CAHV decomposition + cv2.stereoRectify)",
        confidence=confidence,
        units="px",
        uncertainty=result.epipolar_error_px,
        label="rectified_stereo_pair",
        parameters={
            "baseline_m": round(result.baseline_m, 6),
            "focal_length_px": round(result.rig.fx, 2),
            "epipolar_error_px": result.epipolar_error_px,
            "distortion_corrected": False,
            "stereo_status": result.status.value,
        },
        parents=parents,
    )
