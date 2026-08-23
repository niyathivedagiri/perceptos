"""
perceptos.stereo.matcher
========================

Block matching and semi-global matching, plus the validation that tells you
whether the disparity map is worth believing.

Two things are handled here that catch people out:

* **Disparity comes back as fixed-point.** ``StereoBM`` and ``StereoSGBM``
  return ``CV_16S`` scaled by 16. Forgetting to divide gives depths that are
  sixteen times too close, and because everything stays self-consistent the
  error is invisible until something is measured against reality.
* **Invalid pixels are a large negative sentinel, not zero.** OpenCV marks
  unmatched pixels with ``(mindisp - 1) * 16``. Treating that as a real
  disparity produces confident nonsense in exactly the regions — occlusions,
  textureless sky, the shadow behind a rock — where a rover most needs to
  know it cannot see.

The validation metrics matter as much as the map: texture coverage,
left-right consistency, and the fraction of pixels at the search-range
boundary (which means the true disparity lies outside the configured range,
so the object is closer than the matcher can represent).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..core.cv_compat import CapabilityError, cv, require, resolve_factory
from ..core.provenance import Source
from ..core.stage import FrameContext, PipelineStage
from .depth import StereoRig, depth_traced

__all__ = ["DisparityResult", "StereoMatcher", "StereoStage", "DepthStage"]

#: OpenCV's fixed-point scale for disparity output.
DISPARITY_SCALE = 16.0


@dataclass(slots=True)
class DisparityResult:
    """A disparity map plus the evidence for trusting it."""

    disparity: np.ndarray        # float32, NaN where invalid
    valid_mask: np.ndarray
    algorithm: str
    num_disparities: int
    min_disparity: int
    at_range_limit: float        # fraction of valid pixels pinned at max disparity
    at_minimum: float = 0.0      # fraction of all pixels unmatched or at infinity

    @property
    def coverage(self) -> float:
        return float(self.valid_mask.mean())

    @property
    def confidence(self) -> float:
        """Coverage, penalised when much of the scene exceeds the search range."""
        range_penalty = 1.0 - min(1.0, self.at_range_limit * 3.0)
        return float(max(0.0, min(1.0, self.coverage * range_penalty)))

    def statistics(self) -> dict[str, Any]:
        valid = self.disparity[self.valid_mask]
        return {
            "algorithm": self.algorithm,
            "coverage": round(self.coverage, 4),
            "confidence": round(self.confidence, 4),
            "at_range_limit": round(self.at_range_limit, 4),
            "at_minimum": round(self.at_minimum, 4),
            "num_disparities": self.num_disparities,
            "min_disparity": self.min_disparity,
            "median_disparity_px": float(np.median(valid)) if valid.size else None,
            "max_disparity_px": float(valid.max()) if valid.size else None,
        }

    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.coverage < 0.3:
            out.append(
                f"only {self.coverage:.0%} of pixels matched; the scene may be "
                "textureless or the pair may not be rectified"
            )
        if self.at_minimum > 0.5:
            out.append(
                f"{self.at_minimum:.0%} of pixels are unmatched or at the minimum "
                "disparity, which carries no range information"
            )
        if self.at_range_limit > 0.05:
            out.append(
                f"{self.at_range_limit:.0%} of matches sit at the search limit; "
                f"objects are closer than num_disparities={self.num_disparities} "
                "can represent - increase it"
            )
        return out


class StereoMatcher:
    """Wraps ``StereoBM`` / ``StereoSGBM`` with validation.

    Parameters
    ----------
    algorithm:
        ``"SGBM"`` (default) or ``"BM"``. SGBM is markedly better on the
        low-texture terrain a rover actually faces; BM is faster and is the
        right choice for a warehouse floor with strong markings.
    num_disparities:
        Search range in pixels. Must be a multiple of 16 — OpenCV silently
        misbehaves otherwise, so it is rounded up here with a recorded note.
    """

    def __init__(self, *, algorithm: str = "SGBM", num_disparities: int = 128,
                 block_size: int = 5, min_disparity: int = 0,
                 uniqueness_ratio: int = 10, speckle_window: int = 100,
                 speckle_range: int = 2, disp12_max_diff: int = 1) -> None:
        self.algorithm = algorithm.upper()
        if self.algorithm not in ("SGBM", "BM"):
            raise ValueError(f"algorithm must be SGBM or BM, got {algorithm!r}")

        rounded = int(np.ceil(max(16, num_disparities) / 16) * 16)
        self.num_disparities = rounded
        self.rounded_from = num_disparities if rounded != num_disparities else None
        self.min_disparity = int(min_disparity)
        self.block_size = int(block_size) | 1        # must be odd
        self._matcher = self._build(uniqueness_ratio, speckle_window, speckle_range,
                                    disp12_max_diff)

    def _build(self, uniqueness: int, speckle_window: int, speckle_range: int,
               disp12: int) -> Any:
        cv2 = cv()
        if self.algorithm == "BM":
            factory = resolve_factory("StereoBM")
            matcher = require("StereoBM", factory)(
                numDisparities=self.num_disparities, blockSize=self.block_size
            )
            matcher.setMinDisparity(self.min_disparity)
            matcher.setUniquenessRatio(uniqueness)
            matcher.setSpeckleWindowSize(speckle_window)
            matcher.setSpeckleRange(speckle_range)
            matcher.setDisp12MaxDiff(disp12)
            return matcher

        factory = resolve_factory("StereoSGBM")
        channels = 1
        # P1/P2 are the smoothness penalties. The 8/32 multipliers are the
        # values from the original SGM paper and OpenCV's own documentation;
        # leaving them at zero disables regularisation, which is the whole
        # point of using SGBM over BM.
        return require("StereoSGBM", factory)(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=8 * channels * self.block_size ** 2,
            P2=32 * channels * self.block_size ** 2,
            disp12MaxDiff=disp12,
            uniquenessRatio=uniqueness,
            speckleWindowSize=speckle_window,
            speckleRange=speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def compute(self, left: np.ndarray, right: np.ndarray) -> DisparityResult:
        """Compute disparity from a *rectified* greyscale stereo pair."""
        cv2 = cv()
        if left.shape != right.shape:
            raise ValueError(
                f"stereo pair must share a shape; got {left.shape} and {right.shape}"
            )
        if left.ndim == 3:
            left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        raw = self._matcher.compute(left, right).astype(np.float32) / DISPARITY_SCALE

        # OpenCV marks unmatched pixels at (minDisparity - 1). Anything at or
        # below that is a sentinel, not a measurement.
        #
        # A match sitting exactly *at* minDisparity is excluded too, and this
        # is the subtle one: it is not a sentinel, but it converts to infinite
        # depth (Z = fB/d with d = 0), so it is unusable as a range. On a
        # textureless surface SGBM returns 0 almost everywhere, and counting
        # those as valid reports 90% coverage for a map containing no distance
        # information at all — a dashboard reading of maximum confidence for a
        # scene the rover genuinely cannot see.
        floor = float(self.min_disparity)
        valid = raw > floor
        disparity = np.where(valid, raw, np.nan).astype(np.float32)

        at_minimum = float((raw <= floor).mean())
        upper = float(self.min_disparity + self.num_disparities) - 1.5
        at_limit = float((raw[valid] >= upper).mean()) if valid.any() else 0.0

        return DisparityResult(
            disparity=disparity,
            valid_mask=valid,
            algorithm=f"cv2.Stereo{self.algorithm}",
            num_disparities=self.num_disparities,
            min_disparity=self.min_disparity,
            at_range_limit=at_limit,
            at_minimum=at_minimum,
        )

    def parameters(self) -> dict[str, Any]:
        out = {
            "algorithm": self.algorithm,
            "num_disparities": self.num_disparities,
            "block_size": self.block_size,
            "min_disparity": self.min_disparity,
        }
        if self.rounded_from is not None:
            out["num_disparities_rounded_from"] = self.rounded_from
        return out


class StereoStage(PipelineStage):
    """Computes disparity from a rectified pair on the context."""

    name = "stereo"
    requires = ("gray", "gray_right")
    provides = ("disparity",)
    source = Source.CALCULATED

    def __init__(self, matcher: Optional[StereoMatcher] = None, **params: Any) -> None:
        super().__init__(**params)
        self.matcher = matcher or StereoMatcher()

    def run(self, ctx: FrameContext) -> str:
        left, right = ctx.require("gray", "gray_right")
        result = self.matcher.compute(left.value, right.value)
        self.emit(ctx, "disparity", result,
                  algorithm=result.algorithm, parents=[left, right],
                  confidence=result.confidence, units="px",
                  parameters=self.matcher.parameters(),
                  artefact=result.disparity)
        warnings = result.warnings()
        message = f"coverage {result.coverage:.1%}"
        return message + (" | " + "; ".join(warnings) if warnings else "")


class DepthStage(PipelineStage):
    """Converts disparity to metric depth with per-pixel uncertainty."""

    name = "depth"
    requires = ("disparity",)
    provides = ("depth",)
    source = Source.CALCULATED

    def __init__(self, rig: Optional[StereoRig] = None, **params: Any) -> None:
        super().__init__(**params)
        self.rig = rig

    def resolve_rig(self, ctx: FrameContext) -> StereoRig:
        """Prefer the rig measured during rectification over a fixed one.

        A rig supplied at construction is a configuration guess; a rig on the
        context was derived from the published camera model of the pair
        actually being processed. When both exist the measured one wins,
        because a stale baseline silently rescales every depth in the frame.
        """
        carried = ctx.get("stereo_rig")
        if carried is not None:
            return carried.value
        if self.rig is None:
            raise RuntimeError(
                "no stereo rig available: neither the context nor this stage "
                "carries one. Depth needs a measured baseline and focal "
                "length; PerceptOS will not assume them."
            )
        return self.rig

    def run(self, ctx: FrameContext) -> str:
        (disparity,) = ctx.require("disparity")
        rig = self.resolve_rig(ctx)
        # Re-wrap the raw disparity array so depth_traced sees the lineage of
        # the matcher output rather than of the DisparityResult container.
        disparity_array = disparity.derive(
            disparity.value.disparity,
            algorithm="DisparityResult.disparity",
            source=Source.CALCULATED,
            units="px",
            label="disparity_map",
        )
        ctx.put("disparity_map", disparity_array, stage=self.name)
        depth = depth_traced(disparity_array, rig)
        ctx.put("depth", depth, stage=self.name, artefact=depth.value.depth_m)
        stats = depth.value.statistics()
        return (f"{stats.get('valid_pixels', 0)} px, "
                f"median {stats.get('median_depth_m', 0):.2f} m, "
                f"reliable to {stats.get('max_reliable_range_m', 0):.1f} m")
