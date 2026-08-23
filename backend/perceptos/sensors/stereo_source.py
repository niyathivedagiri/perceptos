"""
perceptos.sensors.stereo_source
===============================

A stereo frame source that rectifies before serving.

This closes the loop that :mod:`perceptos.sensors.dataset_source` deliberately
left open. That module refuses to hand raw Navcam pairs to a block matcher,
which is correct but leaves the stereo and depth panels with nothing to show.
:class:`RectifiedStereoSource` performs the missing step: it decomposes the
published CAHV camera models, rectifies the pair, verifies the residual, and
only then presents the result as ``RECTIFIED_STEREO``.

The important property is that the verdict is per-pair, not per-source. A
sequence can contain pairs whose camera models are published and pairs whose
are not; each is judged on its own evidence and the ones without calibration
are served as images with depth withheld, rather than dropping them or
inventing geometry for them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..core.cv_compat import cv
from ..core.provenance import Source, Traced, traced
from ..core.stage import FrameContext, PipelineStage
from ..data.sources import DataSourceStatus, StereoStatus
from ..stereo.rectify import (
    RectificationError,
    RectificationResult,
    rectified_traced,
    rectify_pair,
)

__all__ = ["StereoFrame", "RectifiedStereoSource", "RectifiedStereoStage"]


@dataclass(slots=True)
class StereoFrame:
    """One served pair, with its verdict attached."""

    left: np.ndarray
    right: np.ndarray
    status: StereoStatus
    metadata: dict[str, Any]
    rectification: Optional[RectificationResult] = None

    @property
    def depth_valid(self) -> bool:
        return self.rectification is not None and self.rectification.depth_valid


class RectifiedStereoSource:
    """Serves stereo pairs, rectifying each one where its models allow.

    Parameters
    ----------
    pairs:
        ``(left_path, right_path, left_camera_model, right_camera_model,
        metadata)`` tuples. Camera models may be ``None`` — that pair is then
        served as ``CALIBRATION_UNAVAILABLE`` rather than skipped.
    source_kind:
        Provenance class of the underlying imagery. Recorded NASA frames are
        ``LIVE_CAMERA``; rendered pairs are ``SIMULATION``.
    """

    def __init__(self, pairs: Sequence[tuple[Any, Any, Any, Any, dict[str, Any]]],
                 *, source_kind: Source = Source.LIVE_CAMERA,
                 source_status: str = DataSourceStatus.LOCAL_CACHED_DATA.value,
                 loop: bool = False) -> None:
        self.pairs = list(pairs)
        self._source_kind = source_kind
        self.source_status = source_status
        self._validate_labelling()
        self.loop = bool(loop)
        self.index = 0
        self._opened = False
        #: Per-pair outcomes, so the console can report how many rectified.
        self.verdicts: list[dict[str, Any]] = []

    @property
    def source_kind(self) -> Source:
        return self._source_kind

    @source_kind.setter
    def source_kind(self, value: Source) -> None:
        self._source_kind = value
        self._validate_labelling()

    def _validate_labelling(self) -> None:
        """Reject a source whose two labels contradict each other.

        Declaring ``source_kind=SIMULATION`` alongside
        ``source_status=LOCAL_CACHED_DATA`` says the frames are both synthetic
        and real archive data. The provenance layer already refuses to build
        such a value, but it refuses mid-frame, where the failure surfaces as
        a dead acquisition stage several steps from the mistake. Catching it
        at construction points at the line that is actually wrong.

        Synthetic sources are re-labelled rather than rejected outright when
        the status was simply left at its default, since that is a caller
        omission rather than a contradictory claim.
        """
        if self._source_kind is not Source.SIMULATION:
            return
        if self.source_status in (None, "", DataSourceStatus.SIMULATION.value):
            self.source_status = DataSourceStatus.SIMULATION.value
            return
        if self.source_status == DataSourceStatus.LOCAL_CACHED_DATA.value:
            # The default. Treat as an omission and correct it.
            self.source_status = DataSourceStatus.SIMULATION.value
            return
        raise ValueError(
            f"contradictory labelling: source_kind=SIMULATION cannot carry "
            f"source_status={self.source_status!r}. Synthetic frames are not "
            "archive data, and PerceptOS will not present them as such."
        )

    @classmethod
    def from_verified_pairs(cls, verified: Iterable[Any], **kwargs: Any
                            ) -> "RectifiedStereoSource":
        """Build from :class:`~perceptos.data.mars2020_raw.VerifiedStereoPair`.

        Only pairs whose images are already cached locally are included: this
        source reads from disk and does not fetch.
        """
        entries = []
        for pair in verified:
            left, right = pair.left, pair.right
            if left.local_path is None or right.local_path is None:
                continue
            entries.append((
                left.local_path, right.local_path,
                left.camera_model, right.camera_model,
                {"sol": pair.sol, "sclk": pair.sclk,
                 "left_image_id": left.image_id, "right_image_id": right.image_id,
                 "instrument": left.instrument,
                 "date_taken_utc": left.date_taken_utc,
                 "archive_status": pair.status.value,
                 "credit": "NASA/JPL-Caltech"},
            ))
        return cls(entries, **kwargs)

    # ------------------------------------------------------------- lifecycle

    def open(self) -> "RectifiedStereoSource":
        missing = [p for entry in self.pairs for p in entry[:2]
                   if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} stereo image(s) missing, first: {missing[0]}"
            )
        self._opened = True
        self.index = 0
        return self

    def close(self) -> None:
        self._opened = False

    def __len__(self) -> int:
        return len(self.pairs)

    # ------------------------------------------------------------------ read

    def read(self) -> Optional[StereoFrame]:
        """Serve the next pair, rectified if its camera models permit."""
        if not self._opened or not self.pairs:
            return None
        if self.index >= len(self.pairs):
            if not self.loop:
                return None
            self.index = 0

        cv2 = cv()
        left_path, right_path, left_cam, right_cam, meta = self.pairs[self.index]
        self.index += 1
        left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left is None or right is None:
            return None

        try:
            result = rectify_pair(left, right, left_cam, right_cam)
        except RectificationError as exc:
            # No calibration, or a degenerate model. Serve the images anyway —
            # they are still valid for display, features and detection — but
            # withhold depth and say why.
            verdict = {"index": self.index - 1,
                       "status": StereoStatus.CALIBRATION_UNAVAILABLE.value,
                       "reason": str(exc).splitlines()[0]}
            self.verdicts.append(verdict)
            return StereoFrame(
                left=left, right=right,
                status=StereoStatus.CALIBRATION_UNAVAILABLE,
                metadata={**meta, "rectified": False,
                          "rectification_error": str(exc).splitlines()[0],
                          "source_status": self.source_status},
            )

        self.verdicts.append({
            "index": self.index - 1, "status": result.status.value,
            "epipolar_error_px": result.epipolar_error_px,
            "baseline_m": round(result.baseline_m, 5),
        })
        return StereoFrame(
            left=result.left, right=result.right, status=result.status,
            rectification=result,
            metadata={**meta, "rectified": True,
                      "stereo_status": result.status.value,
                      "baseline_m": result.baseline_m,
                      "focal_length_px": result.rig.fx,
                      "epipolar_error_px": result.epipolar_error_px,
                      "distortion_corrected": False,
                      "source_status": self.source_status},
        )

    def describe(self) -> dict[str, Any]:
        rectified = sum(1 for v in self.verdicts
                        if v["status"] == StereoStatus.RECTIFIED_STEREO.value)
        return {
            "type": type(self).__name__,
            "source": self.source_kind.value,
            "source_status": self.source_status,
            "pairs": len(self.pairs),
            "position": self.index,
            "served": len(self.verdicts),
            "rectified": rectified,
            "verdicts": self.verdicts[-10:],
        }


class RectifiedStereoStage(PipelineStage):
    """Seeds the pipeline with a rectified pair and its stereo rig.

    Writes the same context keys the existing stereo modules already declare —
    ``frame``, ``gray``, ``gray_right`` — so
    :class:`~perceptos.stereo.matcher.StereoStage` and
    :class:`~perceptos.stereo.matcher.DepthStage` run downstream unmodified.
    It additionally publishes ``stereo_rig``, which carries the measured
    baseline and focal length so nothing downstream has to guess them.
    """

    name = "acquisition"
    requires = ()
    provides = ("frame", "gray", "gray_right")
    source = Source.LIVE_CAMERA

    def __init__(self, stereo_source: RectifiedStereoSource, **params: Any) -> None:
        super().__init__(**params)
        self.stereo = stereo_source
        self.source = stereo_source.source_kind
        self.last_frame: Optional[StereoFrame] = None

    def run(self, ctx: FrameContext) -> str:
        frame = self.stereo.read()
        if frame is None:
            raise RuntimeError("stereo source exhausted or failed to decode")
        self.last_frame = frame
        cv2 = cv()

        left = traced(
            frame.left, source=self.source,
            algorithm="RectifiedStereoSource.read[left]",
            confidence=1.0 if self.source.is_measured else None,
            units="px", label="left_frame",
            source_type=frame.metadata.get("source_status"),
            dataset=frame.metadata.get("dataset") or "perseverance_jpl",
            parameters=frame.metadata,
        )
        ctx.put("frame", left, stage=self.name, artefact=frame.left)
        self.emit(ctx, "gray", cv2.cvtColor(frame.left, cv2.COLOR_BGR2GRAY),
                  algorithm="cv2.cvtColor(BGR2GRAY)", source=self.source,
                  parents=[left], units="px")
        self.emit(ctx, "gray_right", cv2.cvtColor(frame.right, cv2.COLOR_BGR2GRAY),
                  algorithm="cv2.cvtColor(BGR2GRAY)", source=self.source,
                  parents=[left], units="px", parameters=frame.metadata)

        ctx.metadata["stereo"] = dict(frame.metadata)
        ctx.metadata["stereo_status"] = frame.status.value
        ctx.metadata["stereo_unrectified"] = not frame.depth_valid

        if frame.rectification is not None:
            rectification = rectified_traced(frame.rectification, parents=(left,))
            ctx.put("rectification", rectification, stage=self.name)
            # The rig travels with the frame so DepthStage never needs a
            # constructor-time guess at the baseline.
            ctx.put("stereo_rig",
                    rectification.derive(frame.rectification.rig,
                                         algorithm="RectificationResult.rig",
                                         source=Source.CALCULATED, units="m",
                                         label="stereo_rig"),
                    stage=self.name)

        if frame.status is StereoStatus.CALIBRATION_UNAVAILABLE:
            return ("pair served without calibration — depth withheld: "
                    + str(frame.metadata.get("rectification_error", ""))[:60])
        residual = frame.metadata.get("epipolar_error_px")
        return (f"{frame.status.value}, baseline "
                f"{frame.metadata.get('baseline_m', 0):.4f} m"
                + (f", epipolar residual {residual:.2f} px" if residual is not None
                   else ", residual not measurable"))
