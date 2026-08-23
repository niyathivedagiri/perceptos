"""
perceptos.imageproc.filters
===========================

A declarative, inspectable image processing chain.

Rather than a function with fifteen boolean arguments, the chain is a list of
named operations with parameters. That buys three things the UI needs:

* the slider panel is generated from :data:`OPERATIONS` rather than hand-built,
  so adding an operation adds a control automatically;
* each operation is timed individually, so the latency waterfall shows which
  filter is expensive rather than a single opaque "preprocessing" number;
* the chain serialises to JSON, so a mode profile can specify a preprocessing
  recipe and the lineage drawer can show exactly which operations ran.

Ordering is validated: several operations are only meaningful on a single
channel or on a binary image, and applying them out of order silently produces
garbage rather than an error.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np

from ..core.cv_compat import cv
from ..core.provenance import Source
from ..core.stage import FrameContext, PipelineStage

__all__ = ["Operation", "OPERATIONS", "ImageChain", "ImageProcessingStage"]


@dataclass(frozen=True, slots=True)
class Operation:
    """One image operation and everything the UI needs to expose it."""

    name: str
    fn: Callable[..., np.ndarray]
    defaults: Mapping[str, Any] = field(default_factory=dict)
    #: Slider metadata: parameter -> (min, max, step)
    controls: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)
    requires_gray: bool = False
    produces_binary: bool = False
    description: str = ""


def _odd(value: Any, minimum: int = 1) -> int:
    """Coerce a kernel size to an odd integer — OpenCV rejects even kernels."""
    k = max(int(minimum), int(value))
    return k if k % 2 == 1 else k + 1


# --------------------------------------------------------------------------
# Operation implementations
# --------------------------------------------------------------------------

def _gaussian(img: np.ndarray, *, ksize: int = 5, sigma: float = 0.0) -> np.ndarray:
    k = _odd(ksize, 1)
    return cv().GaussianBlur(img, (k, k), float(sigma))


def _median(img: np.ndarray, *, ksize: int = 5) -> np.ndarray:
    return cv().medianBlur(img, _odd(ksize, 3))


def _bilateral(img: np.ndarray, *, diameter: int = 9, sigma_colour: float = 75.0,
               sigma_space: float = 75.0) -> np.ndarray:
    return cv().bilateralFilter(img, int(diameter), float(sigma_colour), float(sigma_space))


def _clahe(img: np.ndarray, *, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    cv2 = cv()
    op = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile), int(tile)))
    if img.ndim == 2:
        return op.apply(img)
    # Colour images are equalised in LAB so only luminance changes; running
    # CLAHE per BGR channel shifts hue and wrecks any colour-based detection.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = op.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _equalize(img: np.ndarray) -> np.ndarray:
    return cv().equalizeHist(img)


def _threshold(img: np.ndarray, *, value: int = 127, invert: bool = False) -> np.ndarray:
    cv2 = cv()
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, out = cv2.threshold(img, int(value), 255, mode)
    return out


def _otsu(img: np.ndarray) -> np.ndarray:
    cv2 = cv()
    _, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def _adaptive(img: np.ndarray, *, block_size: int = 11, offset: float = 2.0,
              gaussian: bool = True) -> np.ndarray:
    cv2 = cv()
    method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if gaussian else cv2.ADAPTIVE_THRESH_MEAN_C
    return cv2.adaptiveThreshold(img, 255, method, cv2.THRESH_BINARY,
                                 _odd(block_size, 3), float(offset))


def _canny(img: np.ndarray, *, low: int = 50, high: int = 150,
           aperture: int = 3) -> np.ndarray:
    return cv().Canny(img, int(low), int(high), apertureSize=_odd(aperture, 3))


def _sobel(img: np.ndarray, *, ksize: int = 3) -> np.ndarray:
    cv2 = cv()
    k = _odd(ksize, 1)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=k)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=k)
    return cv2.convertScaleAbs(cv2.magnitude(gx, gy))


def _laplacian(img: np.ndarray, *, ksize: int = 3) -> np.ndarray:
    cv2 = cv()
    return cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_32F, ksize=_odd(ksize, 1)))


def _morphology(img: np.ndarray, *, operation: str = "open", ksize: int = 3,
                iterations: int = 1) -> np.ndarray:
    cv2 = cv()
    ops = {
        "erode": cv2.MORPH_ERODE, "dilate": cv2.MORPH_DILATE,
        "open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE,
        "gradient": cv2.MORPH_GRADIENT, "tophat": cv2.MORPH_TOPHAT,
        "blackhat": cv2.MORPH_BLACKHAT,
    }
    if operation not in ops:
        raise ValueError(f"unknown morphology {operation!r}; expected one of {sorted(ops)}")
    k = _odd(ksize, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(img, ops[operation], kernel, iterations=int(iterations))


#: Registry the UI reads to build its control panel.
OPERATIONS: dict[str, Operation] = {
    op.name: op for op in [
        Operation("gaussian_blur", _gaussian, {"ksize": 5, "sigma": 0.0},
                  {"ksize": (1, 31, 2), "sigma": (0.0, 10.0, 0.1)},
                  description="Low-pass; suppresses sensor noise before edge detection."),
        Operation("median_blur", _median, {"ksize": 5}, {"ksize": (3, 31, 2)},
                  description="Removes salt-and-pepper noise while preserving edges."),
        Operation("bilateral", _bilateral,
                  {"diameter": 9, "sigma_colour": 75.0, "sigma_space": 75.0},
                  {"diameter": (1, 25, 2), "sigma_colour": (1, 200, 1),
                   "sigma_space": (1, 200, 1)},
                  description="Edge-preserving smoothing; expensive."),
        Operation("clahe", _clahe, {"clip_limit": 2.0, "tile": 8},
                  {"clip_limit": (0.5, 10.0, 0.1), "tile": (2, 16, 1)},
                  description="Local contrast; recovers detail in shadowed terrain."),
        Operation("equalize_hist", _equalize, requires_gray=True,
                  description="Global histogram equalisation."),
        Operation("threshold", _threshold, {"value": 127, "invert": False},
                  {"value": (0, 255, 1)}, requires_gray=True, produces_binary=True,
                  description="Fixed binary threshold."),
        Operation("otsu", _otsu, requires_gray=True, produces_binary=True,
                  description="Automatic threshold from the intensity histogram."),
        Operation("adaptive_threshold", _adaptive,
                  {"block_size": 11, "offset": 2.0, "gaussian": True},
                  {"block_size": (3, 51, 2), "offset": (-20.0, 20.0, 0.5)},
                  requires_gray=True, produces_binary=True,
                  description="Per-region threshold; handles uneven illumination."),
        Operation("canny", _canny, {"low": 50, "high": 150, "aperture": 3},
                  {"low": (0, 255, 1), "high": (0, 255, 1), "aperture": (3, 7, 2)},
                  requires_gray=True, produces_binary=True,
                  description="Hysteresis edge detection."),
        Operation("sobel", _sobel, {"ksize": 3}, {"ksize": (1, 7, 2)}, requires_gray=True,
                  description="First-derivative gradient magnitude."),
        Operation("laplacian", _laplacian, {"ksize": 3}, {"ksize": (1, 7, 2)},
                  requires_gray=True,
                  description="Second-derivative response; also a focus metric."),
        Operation("morphology", _morphology,
                  {"operation": "open", "ksize": 3, "iterations": 1},
                  {"ksize": (1, 21, 2), "iterations": (1, 10, 1)},
                  description="Erode/dilate/open/close on a binary or grey image."),
    ]
}


@dataclass(slots=True)
class ChainStep:
    operation: str
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class ImageChain:
    """An ordered, validated sequence of image operations."""

    def __init__(self, steps: Iterable[Mapping[str, Any] | ChainStep] = ()) -> None:
        self.steps: list[ChainStep] = []
        for step in steps:
            self.add(step) if isinstance(step, ChainStep) else self.add(
                step["operation"], **step.get("params", {})
            )
        self.timings: dict[str, float] = {}

    def add(self, operation: Any, **params: Any) -> "ImageChain":
        if isinstance(operation, ChainStep):
            step = operation
        else:
            if operation not in OPERATIONS:
                raise ValueError(
                    f"unknown operation {operation!r}; available: {sorted(OPERATIONS)}"
                )
            step = ChainStep(operation, dict(params))
        self.steps.append(step)
        return self

    def validate(self, *, input_is_gray: bool) -> list[str]:
        """Report ordering problems that would silently corrupt the output.

        Applying ``canny`` to a colour image, or ``morphology`` before any
        threshold, does not raise in OpenCV — it just produces something that
        looks vaguely plausible and is wrong.
        """
        problems: list[str] = []
        is_gray = bool(input_is_gray)
        is_binary = False
        for i, step in enumerate(self.steps):
            if not step.enabled:
                continue
            op = OPERATIONS[step.operation]
            if op.requires_gray and not is_gray:
                problems.append(
                    f"step {i} ({step.operation}) needs a single-channel image; "
                    "insert a greyscale conversion first"
                )
            if is_binary and step.operation in {
                "clahe", "equalize_hist", "canny", "sobel", "laplacian",
                "gaussian_blur", "bilateral",
            }:
                problems.append(
                    f"step {i} ({step.operation}) runs on an already-binarised "
                    "image; move it before the threshold"
                )
            unknown = set(step.params) - set(op.defaults) - {"operation"}
            if unknown:
                problems.append(
                    f"step {i} ({step.operation}) has unknown parameters {sorted(unknown)}"
                )
            if op.requires_gray or op.produces_binary:
                is_gray = True
            if op.produces_binary:
                is_binary = True
        return problems

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Run the chain, recording per-operation latency."""
        out = image
        self.timings = {}
        for i, step in enumerate(self.steps):
            if not step.enabled:
                continue
            op = OPERATIONS[step.operation]
            params = {**op.defaults, **step.params}
            start = time.perf_counter()
            out = op.fn(out, **params)
            self.timings[f"{i}:{step.operation}"] = (time.perf_counter() - start) * 1000.0
        return out

    @property
    def total_ms(self) -> float:
        return round(sum(self.timings.values()), 3)

    @property
    def slowest(self) -> Optional[str]:
        return max(self.timings, key=self.timings.get) if self.timings else None

    def to_dict(self) -> list[dict[str, Any]]:
        return [
            {"operation": s.operation, "params": dict(s.params), "enabled": s.enabled}
            for s in self.steps
        ]

    @staticmethod
    def control_schema() -> dict[str, Any]:
        """Everything the slider panel needs, generated from the registry."""
        return {
            name: {
                "defaults": dict(op.defaults),
                "controls": {k: list(v) for k, v in op.controls.items()},
                "requires_gray": op.requires_gray,
                "produces_binary": op.produces_binary,
                "description": op.description,
            }
            for name, op in OPERATIONS.items()
        }


class ImageProcessingStage(PipelineStage):
    """Applies an :class:`ImageChain` to the greyscale frame."""

    name = "image_processing"
    requires = ("gray",)
    provides = ("processed",)
    source = Source.CALCULATED

    def __init__(self, chain: Optional[ImageChain] = None, *, strict: bool = True,
                 **params: Any) -> None:
        super().__init__(**params)
        self.chain = chain if chain is not None else ImageChain()
        problems = self.chain.validate(input_is_gray=True)
        if problems and strict:
            raise ValueError("invalid image chain: " + "; ".join(problems))
        self.problems = problems

    def run(self, ctx: FrameContext) -> str:
        (gray,) = ctx.require("gray")
        out = self.chain.apply(gray.value)
        self.emit(ctx, "processed", out,
                  algorithm="ImageChain[" + ",".join(
                      s.operation for s in self.chain.steps if s.enabled) + "]",
                  parents=[gray], units="px",
                  parameters={"steps": self.chain.to_dict(),
                              "timings_ms": {k: round(v, 3)
                                             for k, v in self.chain.timings.items()}},
                  artefact=out)
        slowest = self.chain.slowest
        return f"{len(self.chain.timings)} ops, {self.chain.total_ms:.2f} ms" + (
            f", slowest {slowest}" if slowest else ""
        )
