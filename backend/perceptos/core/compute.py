"""
perceptos.core.compute
======================

Accelerator selection with the fallback order the spec mandates:

    Apple Metal (MPS)  ->  CUDA  ->  CPU

Written so it works even when PyTorch is not installed at all — the geometry
and planning half of PerceptOS is pure NumPy and must stay importable on a
machine with no deep-learning stack.  ``torch`` is imported lazily and every
probe is defensive, because ``torch.backends.mps.is_available()`` raises on
some builds rather than returning ``False``.
"""

from __future__ import annotations

import functools
import os
import platform
from dataclasses import dataclass
from typing import Any, Optional

__all__ = ["DeviceInfo", "select_device", "device_report", "clear_cache"]


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Resolved accelerator, plus why it was chosen."""

    kind: str                 # "mps" | "cuda" | "cpu"
    torch_device: str         # string to hand to torch / ultralytics
    name: str                 # human-readable
    reason: str               # why this one and not a faster one
    torch_available: bool
    fp16_supported: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "torch_device": self.torch_device,
            "name": self.name,
            "reason": self.reason,
            "torch_available": self.torch_available,
            "fp16_supported": self.fp16_supported,
        }


def _try_import_torch() -> Optional[Any]:
    try:
        import torch  # noqa: PLC0415 - lazy on purpose
        return torch
    except Exception:  # noqa: BLE001 - a broken torch install must not be fatal
        return None


def _mps_available(torch: Any) -> bool:
    try:
        backend = getattr(torch.backends, "mps", None)
        return bool(backend and backend.is_available() and backend.is_built())
    except Exception:  # noqa: BLE001
        return False


def _cuda_available(torch: Any) -> bool:
    try:
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:  # noqa: BLE001
        return False


@functools.lru_cache(maxsize=4)
def select_device(prefer: Optional[str] = None) -> DeviceInfo:
    """Pick the best available accelerator.

    Parameters
    ----------
    prefer:
        Force a device (``"mps"``, ``"cuda"``, ``"cpu"``).  Also honoured via
        the ``PERCEPTOS_DEVICE`` environment variable, which wins over the
        argument so an operator can override without editing config.
    """
    forced = os.environ.get("PERCEPTOS_DEVICE", prefer)
    torch = _try_import_torch()

    if torch is None:
        return DeviceInfo(
            kind="cpu",
            torch_device="cpu",
            name=f"CPU ({platform.processor() or platform.machine()})",
            reason="PyTorch not installed; NumPy/OpenCV CPU path active",
            torch_available=False,
            fp16_supported=False,
        )

    if forced:
        forced = forced.lower().strip()
        if forced == "mps" and _mps_available(torch):
            return _mps_info(torch, "forced by PERCEPTOS_DEVICE/prefer")
        if forced == "cuda" and _cuda_available(torch):
            return _cuda_info(torch, "forced by PERCEPTOS_DEVICE/prefer")
        if forced == "cpu":
            return _cpu_info(torch, "forced by PERCEPTOS_DEVICE/prefer")
        # Fall through: requested device unavailable, continue with auto-select.

    if _mps_available(torch):
        return _mps_info(torch, "Apple Metal backend available (preferred)")
    if _cuda_available(torch):
        return _cuda_info(torch, "MPS unavailable; CUDA present")
    return _cpu_info(torch, "no MPS or CUDA backend available")


def _mps_info(torch: Any, reason: str) -> DeviceInfo:
    return DeviceInfo(
        kind="mps",
        torch_device="mps",
        name=f"Apple Metal ({platform.machine()})",
        reason=reason,
        torch_available=True,
        # MPS fp16 is supported but numerically fragile for geometry work;
        # PerceptOS keeps fp32 on Metal and only uses fp16 for detection.
        fp16_supported=True,
    )


def _cuda_info(torch: Any, reason: str) -> DeviceInfo:
    try:
        name = torch.cuda.get_device_name(0)
        major = torch.cuda.get_device_capability(0)[0]
    except Exception:  # noqa: BLE001
        name, major = "CUDA device", 0
    return DeviceInfo(
        kind="cuda",
        torch_device="cuda:0",
        name=name,
        reason=reason,
        torch_available=True,
        fp16_supported=major >= 7,
    )


def _cpu_info(torch: Any, reason: str) -> DeviceInfo:
    return DeviceInfo(
        kind="cpu",
        torch_device="cpu",
        name=f"CPU ({platform.processor() or platform.machine()})",
        reason=reason,
        torch_available=torch is not None,
        fp16_supported=False,
    )


def clear_cache() -> None:
    """Release accelerator memory between missions; safe to call anywhere."""
    torch = _try_import_torch()
    if torch is None:
        return
    try:
        if _cuda_available(torch):
            torch.cuda.empty_cache()
        elif _mps_available(torch):
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def device_report() -> dict[str, Any]:
    """Full compute environment description, surfaced on the health panel."""
    info = select_device()
    torch = _try_import_torch()
    try:
        import cv2  # noqa: PLC0415
        cv_version = cv2.__version__
        cv_threads = cv2.getNumThreads()
    except Exception:  # noqa: BLE001
        cv_version, cv_threads = None, None
    return {
        **info.to_dict(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "torch_version": getattr(torch, "__version__", None),
        "opencv_version": cv_version,
        "opencv_threads": cv_threads,
        "cpu_count": os.cpu_count(),
    }
