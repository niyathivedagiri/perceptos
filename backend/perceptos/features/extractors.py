"""
perceptos.features.extractors
=============================

Keypoint detection, description and matching.

The detector menu is built from :func:`~perceptos.core.cv_compat.feature_detectors`
at construction, so requesting AKAZE on a build that lacks it produces a clear
error naming the wheel to install, rather than an ``AttributeError`` deep in a
frame loop.

Two details that are usually wrong in tutorial code:

* **Norm selection follows the descriptor, not the detector.** ORB, AKAZE and
  BRISK produce binary descriptors and must be matched with Hamming distance;
  SIFT and KAZE produce float descriptors and need L2. Matching a binary
  descriptor with L2 "works" — it returns matches, they are just wrong.
* **Lowe's ratio test needs the second-nearest neighbour.** A plain
  nearest-neighbour match always returns something for every keypoint,
  including keypoints with no true correspondence at all. The ratio test is
  what turns matching from a lookup into a decision, and its pass rate is a
  usable confidence signal for everything downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from ..core.cv_compat import CapabilityError, cv, feature_detectors, require
from ..core.provenance import Source
from ..core.stage import FrameContext, PipelineStage

__all__ = ["FeatureSet", "MatchResult", "FeatureExtractor", "FeatureStage"]


@dataclass(slots=True)
class FeatureSet:
    """Keypoints plus descriptors from one image."""

    keypoints: list[Any]
    descriptors: Optional[np.ndarray]
    method: str
    descriptor_kind: str
    image_shape: tuple[int, int]

    def __len__(self) -> int:
        return len(self.keypoints)

    @property
    def points(self) -> np.ndarray:
        """Keypoint coordinates as an (N, 2) float array."""
        if not self.keypoints:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array([kp.pt for kp in self.keypoints], dtype=np.float32)

    @property
    def density(self) -> float:
        """Keypoints per megapixel — comparable across image sizes."""
        h, w = self.image_shape
        megapixels = (h * w) / 1e6
        return float(len(self.keypoints) / megapixels) if megapixels else 0.0

    @property
    def spatial_spread(self) -> float:
        """Fraction of a 8x8 grid containing at least one keypoint.

        Two thousand keypoints all clustered on one textured rock is a much
        weaker basis for pose estimation than two hundred spread across the
        frame, and the raw count cannot tell those apart.
        """
        pts = self.points
        if len(pts) == 0:
            return 0.0
        h, w = self.image_shape
        bins = 8
        cols = np.clip((pts[:, 0] / max(w, 1) * bins).astype(int), 0, bins - 1)
        rows = np.clip((pts[:, 1] / max(h, 1) * bins).astype(int), 0, bins - 1)
        occupied = np.zeros((bins, bins), dtype=bool)
        occupied[rows, cols] = True
        return float(occupied.mean())

    @property
    def quality(self) -> float:
        """Confidence in [0, 1] combining count and spatial spread."""
        count_score = min(1.0, len(self.keypoints) / 300.0)
        return float(max(0.0, min(1.0, 0.5 * count_score + 0.5 * self.spatial_spread)))

    def statistics(self) -> dict[str, Any]:
        responses = [float(kp.response) for kp in self.keypoints] if self.keypoints else []
        return {
            "method": self.method,
            "count": len(self.keypoints),
            "density_per_mpx": round(self.density, 1),
            "spatial_spread": round(self.spatial_spread, 3),
            "quality": round(self.quality, 3),
            "descriptor_kind": self.descriptor_kind,
            "descriptor_shape": list(self.descriptors.shape) if self.descriptors is not None else None,
            "mean_response": round(float(np.mean(responses)), 6) if responses else None,
        }


@dataclass(slots=True)
class MatchResult:
    """Correspondences between two feature sets."""

    matches: list[Any]
    ratio_pass_rate: float
    query_count: int
    train_count: int
    method: str

    def __len__(self) -> int:
        return len(self.matches)

    @property
    def confidence(self) -> float:
        """How much to trust a pose computed from these correspondences."""
        smaller = max(1, min(self.query_count, self.train_count))
        coverage = min(1.0, len(self.matches) / (0.3 * smaller))
        return float(max(0.0, min(1.0, 0.6 * self.ratio_pass_rate + 0.4 * coverage)))

    def point_pairs(self, query: FeatureSet, train: FeatureSet) -> tuple[np.ndarray, np.ndarray]:
        """Matched coordinates as two aligned (N, 2) arrays."""
        if not self.matches:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        src = np.array([query.keypoints[m.queryIdx].pt for m in self.matches], np.float32)
        dst = np.array([train.keypoints[m.trainIdx].pt for m in self.matches], np.float32)
        return src, dst

    def to_dict(self) -> dict[str, Any]:
        distances = [float(m.distance) for m in self.matches]
        return {
            "matches": len(self.matches),
            "ratio_pass_rate": round(self.ratio_pass_rate, 3),
            "confidence": round(self.confidence, 3),
            "mean_distance": round(float(np.mean(distances)), 3) if distances else None,
            "method": self.method,
        }


class FeatureExtractor:
    """Detects and describes keypoints with whichever detector is available.

    Parameters
    ----------
    method:
        ``ORB``, ``SIFT``, ``FAST``, ``AKAZE``, ``BRISK``, ``KAZE`` or
        ``HARRIS``. Use :meth:`available` to list what this build supports.
    max_features:
        Cap on keypoints. Applied by the detector where it supports one, and
        by response-ordered truncation otherwise, so the cap is honoured
        uniformly — FAST and Harris have no built-in limit and will happily
        return fifty thousand corners on a textured image.
    """

    def __init__(self, method: str = "ORB", *, max_features: int = 1000,
                 harris_quality: float = 0.01, harris_min_distance: int = 8) -> None:
        self.method = method.upper()
        self.max_features = int(max_features)
        self.harris_quality = float(harris_quality)
        self.harris_min_distance = int(harris_min_distance)

        registry = feature_detectors()
        if self.method not in registry:
            raise CapabilityError(
                f"unknown feature method {self.method!r}; known: {sorted(registry)}"
            )
        spec = registry[self.method]
        if not spec["available"]:
            raise CapabilityError(
                f"{self.method} is not present in this OpenCV build. "
                f"It ships with {spec['provided_by']}. "
                f"Available here: {sorted(n for n, s in registry.items() if s['available'])}"
            )
        self.descriptor_kind: str = spec["descriptor"]
        self._detector = self._build(spec["factory"])

    @staticmethod
    def available() -> list[str]:
        return sorted(n for n, s in feature_detectors().items() if s["available"])

    def _build(self, factory: Any) -> Any:
        if self.method == "HARRIS":
            return None                      # uses goodFeaturesToTrack directly
        if self.method == "ORB":
            return factory(nfeatures=self.max_features)
        if self.method == "SIFT":
            return factory(nfeatures=self.max_features)
        if self.method == "FAST":
            return factory()                 # no cap; truncated after detection
        return factory()                     # AKAZE / BRISK / KAZE

    # ------------------------------------------------------------------ detect

    def detect(self, gray: np.ndarray) -> FeatureSet:
        """Detect keypoints and compute descriptors where the method has them."""
        cv2 = cv()
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        shape = (int(gray.shape[0]), int(gray.shape[1]))

        if self.method == "HARRIS":
            corners = cv2.goodFeaturesToTrack(
                gray, maxCorners=self.max_features, qualityLevel=self.harris_quality,
                minDistance=self.harris_min_distance, useHarrisDetector=True, k=0.04,
            )
            keypoints = (
                [cv2.KeyPoint(float(x), float(y), 7.0) for x, y in corners.reshape(-1, 2)]
                if corners is not None else []
            )
            return FeatureSet(keypoints, None, self.method, "none", shape)

        if self.method == "FAST":
            keypoints = list(self._detector.detect(gray, None))
            keypoints = self._truncate(keypoints)
            return FeatureSet(keypoints, None, self.method, "none", shape)

        keypoints, descriptors = self._detector.detectAndCompute(gray, None)
        keypoints = list(keypoints or [])
        if descriptors is not None and len(keypoints) > self.max_features:
            order = np.argsort([-kp.response for kp in keypoints])[: self.max_features]
            keypoints = [keypoints[i] for i in order]
            descriptors = descriptors[order]
        return FeatureSet(keypoints, descriptors, self.method, self.descriptor_kind, shape)

    def _truncate(self, keypoints: list[Any]) -> list[Any]:
        if len(keypoints) <= self.max_features:
            return keypoints
        keypoints.sort(key=lambda kp: -kp.response)
        return keypoints[: self.max_features]

    # ------------------------------------------------------------------- match

    def match(self, query: FeatureSet, train: FeatureSet, *,
              ratio: float = 0.75, cross_check: bool = False) -> MatchResult:
        """Match two feature sets using Lowe's ratio test."""
        cv2 = cv()
        if query.descriptors is None or train.descriptors is None:
            raise CapabilityError(
                f"{self.method} produces no descriptors; it is a detector only. "
                "Use ORB/SIFT/AKAZE for matching, or track these keypoints with "
                "optical flow instead."
            )
        if len(query) == 0 or len(train) == 0:
            return MatchResult([], 0.0, len(query), len(train), self.method)

        # Norm follows the descriptor kind, not the detector name.
        norm = cv2.NORM_HAMMING if query.descriptor_kind == "binary" else cv2.NORM_L2
        matcher = cv2.BFMatcher(norm, crossCheck=False)
        k = 2 if len(train) >= 2 else 1
        raw = matcher.knnMatch(query.descriptors, train.descriptors, k=k)

        good: list[Any] = []
        considered = 0
        for candidates in raw:
            if len(candidates) < 2:
                continue
            considered += 1
            best, second = candidates[0], candidates[1]
            if best.distance < ratio * second.distance:
                good.append(best)
        pass_rate = good.__len__() / considered if considered else 0.0

        if cross_check and good:
            reverse = cv2.BFMatcher(norm, crossCheck=False).knnMatch(
                train.descriptors, query.descriptors, k=1
            )
            back = {r[0].queryIdx: r[0].trainIdx for r in reverse if r}
            good = [m for m in good if back.get(m.trainIdx) == m.queryIdx]

        good.sort(key=lambda m: m.distance)
        return MatchResult(good, pass_rate, len(query), len(train), self.method)


class FeatureStage(PipelineStage):
    """Extracts features from the processed (or greyscale) image."""

    name = "features"
    requires = ("gray",)
    provides = ("features",)
    source = Source.CALCULATED

    def __init__(self, extractor: Optional[FeatureExtractor] = None, *,
                 prefer_processed: bool = True, **params: Any) -> None:
        super().__init__(**params)
        self.extractor = extractor or FeatureExtractor("ORB")
        self.prefer_processed = bool(prefer_processed)

    def run(self, ctx: FrameContext) -> str:
        key = "processed" if (self.prefer_processed and ctx.has("processed")) else "gray"
        (image,) = ctx.require(key)
        features = self.extractor.detect(image.value)
        self.emit(ctx, "features", features,
                  algorithm=f"{self.extractor.method}.detectAndCompute",
                  parents=[image], confidence=features.quality, units="keypoints",
                  parameters={"max_features": self.extractor.max_features,
                              "input": key, **features.statistics()})
        return (f"{len(features)} {self.extractor.method} keypoints, "
                f"spread {features.spatial_spread:.2f}")
