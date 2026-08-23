"""
perceptos.detection.yolo
========================

YOLOv11 inference, wrapped so its absence is a degraded mode rather than a
crash.

Ultralytics is a heavy optional dependency: it pulls PyTorch, and on a fresh
Apple Silicon machine that is a multi-gigabyte install. PerceptOS treats it as
optional throughout — the model is imported lazily on first inference, and a
missing package produces a :class:`DetectorUnavailable` with the install
command rather than an ``ImportError`` at module import.

Device selection routes through :mod:`perceptos.core.compute`, so detection
lands on Metal when Metal is present and falls back cleanly otherwise.

Detections are ``ESTIMATED`` values. Each carries the model's own confidence,
which then flows into tracking and, through the lineage graph, into whatever
the mission planner eventually decides.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..core.compute import select_device
from ..core.provenance import Source
from ..core.stage import FrameContext, PipelineStage

__all__ = ["Detection", "DetectorUnavailable", "YoloDetector", "DetectionStage"]


class DetectorUnavailable(RuntimeError):
    """The detection backend or weights are not present."""


@dataclass(slots=True)
class Detection:
    """One detected object."""

    bbox: tuple[float, float, float, float]     # (x, y, w, h) in pixels
    label: str
    confidence: float
    class_id: int = -1
    mask: Optional[np.ndarray] = None

    @property
    def centre(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)

    @property
    def area(self) -> float:
        return float(self.bbox[2] * self.bbox[3])

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [round(v, 1) for v in self.bbox],
            "centre": [round(v, 1) for v in self.centre],
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "has_mask": self.mask is not None,
        }

    def as_track_input(self) -> dict[str, Any]:
        """Shape expected by :class:`~perceptos.tracking.trackers.MultiObjectTracker`."""
        return {"bbox": self.bbox, "label": self.label, "confidence": self.confidence}


class YoloDetector:
    """Ultralytics YOLOv11 wrapper with lazy loading.

    Parameters
    ----------
    weights:
        Path to a ``.pt`` file, or a model name Ultralytics can fetch
        (``yolo11n.pt``). PerceptOS never downloads weights implicitly during a
        pipeline run — a missing local file raises rather than silently
        reaching for the network mid-mission.
    confidence / iou:
        Detection and NMS thresholds.
    classes:
        Optional class-id allowlist. In Mars mode this restricts inference to
        the geology classes so a rock is never reported as a "person".
    """

    def __init__(self, weights: str = "yolo11n.pt", *, confidence: float = 0.35,
                 iou: float = 0.45, image_size: int = 640,
                 classes: Optional[Sequence[int]] = None,
                 allow_download: bool = False) -> None:
        self.weights = str(weights)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.image_size = int(image_size)
        self.classes = list(classes) if classes else None
        self.allow_download = bool(allow_download)
        self._model: Any = None
        self._names: dict[int, str] = {}
        self.device = select_device()
        self.inference_count = 0
        self.total_inference_ms = 0.0

    # ---------------------------------------------------------------- loading

    @staticmethod
    def backend_available() -> bool:
        try:
            import ultralytics  # noqa: F401,PLC0415
            return True
        except Exception:  # noqa: BLE001
            return False

    def load(self) -> Any:
        """Load the model. Idempotent; called automatically on first detect."""
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            raise DetectorUnavailable(
                "Ultralytics is not installed, so YOLO detection is unavailable.\n"
                "    pip install ultralytics\n"
                "Every other PerceptOS module runs without it; the detection "
                "stage will report SKIPPED."
            ) from exc

        path = Path(self.weights)
        if not path.exists() and not self.allow_download:
            raise DetectorUnavailable(
                f"weights {self.weights!r} not found on disk. Place them in "
                "models/, or construct the detector with allow_download=True to "
                "let Ultralytics fetch them. Downloads are off by default so a "
                "mission run never blocks on the network."
            )
        self._model = YOLO(self.weights)
        self._names = dict(getattr(self._model, "names", {}) or {})
        return self._model

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._names)

    # -------------------------------------------------------------- inference

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on one BGR frame."""
        model = self.load()
        start = time.perf_counter()
        results = model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            classes=self.classes,
            device=self.device.torch_device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.inference_count += 1
        self.total_inference_ms += elapsed_ms
        return self._parse(results)

    def _parse(self, results: Iterable[Any]) -> list[Detection]:
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            masks = getattr(result, "masks", None)
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
            clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
            names = dict(getattr(result, "names", self._names) or {})
            for i in range(len(xyxy)):
                x0, y0, x1, y1 = (float(v) for v in xyxy[i])
                class_id = int(clss[i])
                mask = None
                if masks is not None and getattr(masks, "data", None) is not None:
                    data = masks.data
                    array = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
                    if i < len(array):
                        mask = array[i]
                detections.append(Detection(
                    bbox=(x0, y0, x1 - x0, y1 - y0),
                    label=names.get(class_id, str(class_id)),
                    confidence=float(confs[i]),
                    class_id=class_id,
                    mask=mask,
                ))
        return detections

    @property
    def mean_inference_ms(self) -> float:
        return (self.total_inference_ms / self.inference_count) if self.inference_count else 0.0

    def statistics(self) -> dict[str, Any]:
        return {
            "weights": self.weights,
            "device": self.device.torch_device,
            "device_reason": self.device.reason,
            "confidence": self.confidence,
            "iou": self.iou,
            "image_size": self.image_size,
            "loaded": self._model is not None,
            "inferences": self.inference_count,
            "mean_inference_ms": round(self.mean_inference_ms, 2),
        }


class DetectionStage(PipelineStage):
    """Runs object detection, degrading to SKIPPED when the backend is absent.

    A missing detector must not stop a rover from avoiding hazards: geometry
    and planning do not depend on it. The stage therefore disables itself
    permanently on the first :class:`DetectorUnavailable` rather than raising
    once per frame for the rest of the mission.
    """

    name = "detection"
    requires = ("frame",)
    provides = ("detections",)
    source = Source.ESTIMATED

    def __init__(self, detector: Optional[YoloDetector] = None, **params: Any) -> None:
        super().__init__(**params)
        self.detector = detector or YoloDetector()
        self.unavailable_reason: Optional[str] = None

    def skip_reason(self, ctx: FrameContext) -> Optional[str]:
        if self.unavailable_reason is not None:
            return f"detector unavailable: {self.unavailable_reason}"
        return super().skip_reason(ctx)

    def run(self, ctx: FrameContext) -> str:
        (frame,) = ctx.require("frame")
        try:
            detections = self.detector.detect(frame.value)
        except DetectorUnavailable as exc:
            self.unavailable_reason = str(exc).splitlines()[0]
            raise
        mean_conf = (
            float(np.mean([d.confidence for d in detections])) if detections else 0.0
        )
        self.emit(ctx, "detections", [d.as_track_input() for d in detections],
                  algorithm=f"YOLOv11({Path(self.detector.weights).name})",
                  parents=[frame], confidence=mean_conf, units="objects",
                  parameters=self.detector.statistics())
        ctx.metadata["detections_detail"] = [d.to_dict() for d in detections]
        labels = sorted({d.label for d in detections})
        return (f"{len(detections)} objects "
                f"({', '.join(labels) if labels else 'none'}), "
                f"{self.detector.mean_inference_ms:.0f} ms mean")
