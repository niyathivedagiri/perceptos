"""
perceptos.core.cv_compat
========================

One place that knows what this OpenCV build can actually do.

OpenCV's Python API is not stable across builds. Between 4.x and 5.x the
factory convention changed (``cv2.ORB_create`` became ``cv2.ORB.create``),
the legacy tracker namespace moved, and several detectors (AKAZE, BRISK,
KAZE) are absent from some headless and 5.x wheels entirely. SIFT was
patent-encumbered and missing from mainline builds for years.

The naive approach — calling ``cv2.AKAZE_create()`` directly in the feature
module — produces an ``AttributeError`` at runtime on a machine that differs
from the developer's by one wheel. This module resolves every factory once at
import, reports what is genuinely available, and raises a diagnostic error
naming the missing capability and the wheel that provides it.

Nothing here imports OpenCV at module scope in a way that can fail: if OpenCV
is absent entirely, :data:`HAS_OPENCV` is ``False`` and the geometry half of
PerceptOS keeps working.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

__all__ = [
    "HAS_OPENCV",
    "OPENCV_VERSION",
    "CapabilityError",
    "cv",
    "resolve_factory",
    "feature_detectors",
    "tracker_factories",
    "capability_report",
    "require",
]


class CapabilityError(RuntimeError):
    """A requested OpenCV capability is not present in this build."""


try:  # pragma: no cover - environment dependent
    import cv2 as _cv2

    HAS_OPENCV = True
    OPENCV_VERSION = _cv2.__version__
except Exception:  # noqa: BLE001
    _cv2 = None
    HAS_OPENCV = False
    OPENCV_VERSION = None


def cv() -> Any:
    """Return the ``cv2`` module, or raise a diagnostic error."""
    if _cv2 is None:
        raise CapabilityError(
            "OpenCV is not installed. Install it with:\n"
            "    pip install opencv-contrib-python\n"
            "The geometry, planning and estimation modules do not need it."
        )
    return _cv2


def resolve_factory(name: str, namespace: Any = None) -> Optional[Callable[..., Any]]:
    """Find a constructor across the OpenCV 4.x and 5.x naming conventions.

    Tries ``<ns>.<name>_create`` (4.x style) then ``<ns>.<name>.create`` (5.x
    style). Returns ``None`` when neither exists rather than raising, so
    callers can present a menu of what is actually available.
    """
    if _cv2 is None:
        return None
    ns = namespace if namespace is not None else _cv2
    factory = getattr(ns, f"{name}_create", None)
    if callable(factory):
        return factory
    cls = getattr(ns, name, None)
    create = getattr(cls, "create", None) if cls is not None else None
    return create if callable(create) else None


# --------------------------------------------------------------------------
# Feature detectors
# --------------------------------------------------------------------------

#: Detector name -> (OpenCV symbol, descriptor kind, wheel that provides it)
_DETECTOR_SPECS: dict[str, tuple[str, str, str]] = {
    "ORB": ("ORB", "binary", "opencv-python"),
    "SIFT": ("SIFT", "float", "opencv-python >= 4.4"),
    "FAST": ("FastFeatureDetector", "none", "opencv-python"),
    "AKAZE": ("AKAZE", "binary", "opencv-contrib-python (4.x)"),
    "BRISK": ("BRISK", "binary", "opencv-contrib-python (4.x)"),
    "KAZE": ("KAZE", "float", "opencv-contrib-python (4.x)"),
}


def feature_detectors() -> dict[str, dict[str, Any]]:
    """Every feature detector this build supports, with its descriptor kind.

    ``HARRIS`` is always listed: it is implemented from ``cv2.cornerHarris``,
    a core function present in every build, so it is the guaranteed fallback
    when no keypoint detector class is available.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, (symbol, descriptor, wheel) in _DETECTOR_SPECS.items():
        factory = resolve_factory(symbol)
        out[name] = {
            "available": factory is not None,
            "factory": factory,
            "descriptor": descriptor,
            "provided_by": wheel,
        }
    out["HARRIS"] = {
        "available": HAS_OPENCV,
        "factory": None,          # implemented directly from cornerHarris
        "descriptor": "none",
        "provided_by": "opencv-python (core)",
    }
    return out


# --------------------------------------------------------------------------
# Trackers
# --------------------------------------------------------------------------

#: Tracker name -> (symbol, whether it lives in the legacy namespace)
_TRACKER_SPECS: dict[str, tuple[str, bool]] = {
    "CSRT": ("TrackerCSRT", False),
    "KCF": ("TrackerKCF", False),
    "MIL": ("TrackerMIL", False),
    "MOSSE": ("TrackerMOSSE", True),      # legacy-only in every modern build
    "BOOSTING": ("TrackerBoosting", True),
}


def tracker_factories() -> dict[str, Optional[Callable[..., Any]]]:
    """Resolve tracker constructors from both the main and legacy namespaces.

    MOSSE in particular only ever appears under ``cv2.legacy``. Looking for it
    on the top-level module — the obvious thing to do — always fails.
    """
    if _cv2 is None:
        return {name: None for name in _TRACKER_SPECS}
    legacy = getattr(_cv2, "legacy", None)
    out: dict[str, Optional[Callable[..., Any]]] = {}
    for name, (symbol, legacy_only) in _TRACKER_SPECS.items():
        factory = None
        if not legacy_only:
            factory = resolve_factory(symbol)
        if factory is None and legacy is not None:
            factory = resolve_factory(symbol, namespace=legacy)
        out[name] = factory
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def require(capability: str, factory: Any, *, provided_by: str = "") -> Callable[..., Any]:
    """Assert a capability exists, with a message that says how to get it."""
    if factory is None:
        hint = f" Install {provided_by}." if provided_by else ""
        raise CapabilityError(
            f"{capability!r} is not available in OpenCV "
            f"{OPENCV_VERSION or '(not installed)'}.{hint}"
        )
    return factory


def capability_report() -> dict[str, Any]:
    """Full capability snapshot — shown on the system health panel.

    Surfacing this in the UI means a user whose feature extractor menu is
    missing AKAZE can see *why* instead of filing a bug.
    """
    detectors = feature_detectors()
    trackers = tracker_factories()
    cv2_mod = _cv2
    return {
        "opencv_installed": HAS_OPENCV,
        "opencv_version": OPENCV_VERSION,
        "detectors": {
            name: {"available": spec["available"], "descriptor": spec["descriptor"],
                   "provided_by": spec["provided_by"]}
            for name, spec in detectors.items()
        },
        "detectors_available": sorted(n for n, s in detectors.items() if s["available"]),
        "trackers_available": sorted(n for n, f in trackers.items() if f is not None),
        "stereo": {
            "BM": resolve_factory("StereoBM") is not None,
            "SGBM": resolve_factory("StereoSGBM") is not None,
            "wls_filter": hasattr(cv2_mod, "ximgproc") if cv2_mod else False,
        },
        "calibration": bool(cv2_mod and hasattr(cv2_mod, "findChessboardCorners")),
        "optical_flow": bool(cv2_mod and hasattr(cv2_mod, "calcOpticalFlowPyrLK")),
        "pose_recovery": bool(cv2_mod and hasattr(cv2_mod, "findEssentialMat")),
    }
