"""
perceptos.sensors.dataset_source
================================

Frame sources backed by recorded datasets.

This is the bridge between the data layer and everything already built. A
:class:`DatasetFrameSource` is a :class:`~perceptos.sensors.camera.FrameSource`
like any other, so every Phase 1 and Phase 2 module runs on real Perseverance
imagery without a single line changing. That is the whole point of the hardware
abstraction: algorithms never learn where their pixels came from.

Provenance is handled honestly. A recorded Navcam frame *is* a real camera
frame, so it carries ``LIVE_CAMERA`` — but the acquisition time is the sol it
was taken on, years ago, not now. The provenance timestamp records when the
value entered this pipeline; the sol, Earth date and spacecraft clock go in the
parameters, so the lineage drawer can show both without conflating them.

:class:`StereoDatasetSource` refuses to hand unrectified pairs to a stereo
matcher. Raw Navcam pairs are not rectified, epipolar lines are not horizontal,
and block matching along scanlines therefore produces a dense, confident and
entirely wrong disparity map. Silently allowing that would be the single worst
credibility failure available to this project.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..core.cv_compat import CapabilityError, cv
from ..core.provenance import Source, Traced, traced
from ..core.stage import FrameContext, PipelineStage
from .camera import FrameSource

__all__ = [
    "DatasetFrameSource",
    "StereoDatasetSource",
    "StereoAcquisitionStage",
    "UnrectifiedPairError",
]


class UnrectifiedPairError(RuntimeError):
    """A stereo pair was requested for matching without rectification."""


@dataclass(slots=True)
class DatasetFrame:
    """One recorded frame plus the metadata that came with it."""

    path: Path
    metadata: dict[str, Any]


class DatasetFrameSource(FrameSource):
    """Replays an ordered set of image files as a camera feed.

    Parameters
    ----------
    paths:
        Image files, in playback order.
    metadata:
        Optional per-frame dicts (sol, earth_date, camera, product_id...),
        aligned with ``paths``. These end up in the lineage record.
    source_kind:
        Provenance class. Defaults to ``LIVE_CAMERA`` because a recorded rover
        frame really was captured by a camera; pass ``SIMULATION`` for rendered
        or synthetic sequences.
    """

    drops_are_measured = True

    def __init__(self, paths: Sequence[str | Path], *,
                 metadata: Optional[Sequence[dict[str, Any]]] = None,
                 source_kind: Source = Source.LIVE_CAMERA,
                 loop: bool = False, fps: float = 1.0,
                 grayscale: bool = False) -> None:
        self.paths = [Path(p) for p in paths]
        if metadata is not None and len(metadata) != len(self.paths):
            raise ValueError(
                f"metadata length {len(metadata)} does not match {len(self.paths)} paths"
            )
        self.metadata = list(metadata) if metadata is not None else [{}] * len(self.paths)
        self.source_kind = source_kind
        self.loop = bool(loop)
        self._fps = float(fps)
        self.grayscale = bool(grayscale)
        self.index = 0
        self._opened = False
        self.decode_failures = 0

    # ----------------------------------------------------------- construction

    @classmethod
    def from_directory(cls, directory: str | Path, *, pattern: str = "*.png",
                       **kwargs: Any) -> "DatasetFrameSource":
        """Build from a folder, using any sidecar ``manifest.json`` for metadata."""
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"{directory} is not a directory")
        paths = sorted(directory.glob(pattern))
        if not paths:
            raise FileNotFoundError(
                f"no files matching {pattern!r} in {directory}. "
                "Run scripts/fetch_data.py to populate it."
            )
        metadata: list[dict[str, Any]] = [{} for _ in paths]
        manifest = directory.parent / "manifest.json"
        if manifest.exists():
            try:
                assets = json.loads(manifest.read_text()).get("assets", {})
                by_name = {Path(a.get("name", "")).name: a.get("metadata", {})
                           for a in assets.values()}
                metadata = [dict(by_name.get(p.name, {})) for p in paths]
            except (json.JSONDecodeError, AttributeError):
                pass       # a broken manifest costs metadata, not the frames
        return cls(paths, metadata=metadata, **kwargs)

    # ------------------------------------------------------------- lifecycle

    def open(self) -> None:
        missing = [p for p in self.paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} frame(s) missing, first: {missing[0]}"
            )
        self._opened = True
        self.index = 0

    def read(self) -> Optional[np.ndarray]:
        if not self._opened:
            return None
        if self.index >= len(self.paths):
            if not self.loop:
                return None
            self.index = 0
        cv2 = cv()
        path = self.paths[self.index]
        self.index += 1
        flag = cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR
        image = cv2.imread(str(path), flag)
        if image is None:
            # A corrupt file in a thousand-frame archive should cost one frame,
            # not the run. Count it so the health panel can show it.
            self.decode_failures += 1
            return None
        return image

    def close(self) -> None:
        self._opened = False

    @property
    def is_open(self) -> bool:
        return self._opened

    @property
    def nominal_fps(self) -> Optional[float]:
        return self._fps

    # --------------------------------------------------------------- metadata

    def current_metadata(self) -> dict[str, Any]:
        """Metadata for the frame most recently returned by :meth:`read`."""
        i = self.index - 1
        return dict(self.metadata[i]) if 0 <= i < len(self.metadata) else {}

    def describe(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "source": self.source_kind.value,
            "frames": len(self.paths),
            "position": self.index,
            "decode_failures": self.decode_failures,
            "directory": str(self.paths[0].parent) if self.paths else None,
        }


class StereoDatasetSource:
    """Serves matched left/right pairs, guarding the rectification contract.

    Parameters
    ----------
    pairs:
        Sequence of ``(left_path, right_path, metadata)`` tuples.
    rectified:
        Whether the pairs are epipolar-rectified. **Defaults to False**, which
        is the truth for raw Navcam data. :meth:`read_for_matching` refuses to
        serve unrectified pairs, so a stereo matcher cannot be handed them by
        accident.
    """

    def __init__(self, pairs: Sequence[tuple[Any, Any, dict[str, Any]]], *,
                 rectified: bool = False, baseline_m: Optional[float] = None,
                 source_kind: Source = Source.LIVE_CAMERA) -> None:
        self.pairs = [(Path(left), Path(right), dict(meta)) for left, right, meta in pairs]
        self.rectified = bool(rectified)
        self.baseline_m = baseline_m
        self.source_kind = source_kind
        self.index = 0
        self._opened = False

    @classmethod
    def from_stereo_pairs(cls, stereo_pairs: Iterable[Any], **kwargs: Any
                          ) -> "StereoDatasetSource":
        """Build from :class:`~perceptos.data.mars2020.StereoPair` objects."""
        entries = []
        for pair in stereo_pairs:
            if pair.left.local_path is None or pair.right.local_path is None:
                continue        # not downloaded yet
            entries.append((
                pair.left.local_path, pair.right.local_path,
                {"sol": pair.sol, "sclk": pair.sclk,
                 "camera_family": pair.camera_family,
                 "left_product_id": pair.left.product.raw if pair.left.product else None,
                 "right_product_id": pair.right.product.raw if pair.right.product else None,
                 "earth_date": pair.left.earth_date,
                 "credit": "NASA/JPL-Caltech"},
            ))
        first = next(iter(stereo_pairs), None)
        kwargs.setdefault("baseline_m", getattr(first, "baseline_m", None))
        kwargs.setdefault("rectified", getattr(first, "rectified", False))
        return cls(entries, **kwargs)

    def open(self) -> "StereoDatasetSource":
        self._opened = True
        self.index = 0
        return self

    def close(self) -> None:
        self._opened = False

    def __len__(self) -> int:
        return len(self.pairs)

    def read(self) -> Optional[tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
        """Return the next ``(left, right, metadata)``, or ``None`` when done.

        Serving works regardless of rectification — you may legitimately want
        raw pairs for feature matching, calibration or display.
        """
        if not self._opened or self.index >= len(self.pairs):
            return None
        cv2 = cv()
        left_path, right_path, meta = self.pairs[self.index]
        self.index += 1
        left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left is None or right is None:
            return None
        return left, right, {**meta, "rectified": self.rectified}

    def read_for_matching(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Return a pair *only* if it is safe to run block matching on it."""
        if not self.rectified:
            raise UnrectifiedPairError(
                "these pairs are not epipolar-rectified, so StereoBM/SGBM would "
                "search along the wrong lines and return a dense, confident and "
                "wrong disparity map.\n"
                "Options:\n"
                "  1. rectify with cv2.stereoRectify (needs the rig geometry), or\n"
                "  2. compute an epipolar rectification from matched features, or\n"
                "  3. use these pairs for feature matching and detection only.\n"
                "Construct with rectified=True once the pairs have been rectified."
            )
        pair = self.read()
        if pair is None:
            raise IndexError("stereo source exhausted")
        return pair

    def describe(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "source": self.source_kind.value,
            "pairs": len(self.pairs),
            "position": self.index,
            "rectified": self.rectified,
            "baseline_m": self.baseline_m,
        }


class StereoAcquisitionStage(PipelineStage):
    """Seeds the pipeline with a rectified stereo pair.

    Writes ``frame``, ``gray`` and ``gray_right``, which is exactly the input
    contract :class:`~perceptos.stereo.matcher.StereoStage` already declares.
    """

    name = "acquisition"
    requires = ()
    provides = ("frame", "gray", "gray_right")
    source = Source.LIVE_CAMERA

    def __init__(self, stereo_source: StereoDatasetSource, **params: Any) -> None:
        super().__init__(**params)
        self.stereo = stereo_source
        self.source = stereo_source.source_kind

    def run(self, ctx: FrameContext) -> str:
        pair = self.stereo.read()
        if pair is None:
            raise RuntimeError("stereo dataset exhausted")
        left, right, meta = pair
        cv2 = cv()

        frame = traced(
            left, source=self.source, algorithm="StereoDatasetSource.read[left]",
            confidence=1.0 if self.source.is_measured else None,
            units="px", label="left_frame", parameters=meta,
        )
        ctx.put("frame", frame, stage=self.name, artefact=left)
        self.emit(ctx, "gray", cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),
                  algorithm="cv2.cvtColor(BGR2GRAY)", source=self.source,
                  parents=[frame], units="px")
        self.emit(ctx, "gray_right", cv2.cvtColor(right, cv2.COLOR_BGR2GRAY),
                  algorithm="cv2.cvtColor(BGR2GRAY)", source=self.source,
                  parents=[frame], units="px", parameters=meta)
        ctx.metadata["stereo"] = meta

        if not self.stereo.rectified:
            # Do not raise: features, detection and display are all valid on
            # raw pairs. Mark the context so the depth stage can refuse.
            ctx.metadata["stereo_unrectified"] = True
            return (f"sol {meta.get('sol', '?')} pair served RAW "
                    "(unrectified - depth disabled)")
        return f"sol {meta.get('sol', '?')} rectified pair"
