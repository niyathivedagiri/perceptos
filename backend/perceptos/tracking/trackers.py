"""
perceptos.tracking.trackers
===========================

Multi-object tracking: stable identities across frames.

The architecture is tracking-by-detection with a Kalman filter per track, and
an OpenCV correlation tracker (CSRT / KCF / MOSSE) as the appearance-based
fallback when detections stop arriving. That combination matters:

* The **Kalman filter** predicts where a target should be, which is what makes
  association robust when two objects cross. Associating on raw detection
  proximity alone swaps their identities at the crossing point every time.
* The **correlation tracker** keeps a target alive through short occlusions
  and detector dropouts using appearance, which the Kalman filter cannot do
  because it only knows motion.
* **Track confirmation** — a track is not reported until it has been seen
  ``min_hits`` times. Without it, every one-frame detector false positive
  becomes a track with an ID, and the mission log fills with phantom objects.

Association uses IoU against the Kalman *prediction*, solved greedily by
descending overlap. Greedy is deliberate: the Hungarian algorithm is optimal
but needs SciPy and the difference is negligible below a few dozen tracks,
which is the regime this runs in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..core.cv_compat import CapabilityError, cv, tracker_factories
from ..core.provenance import Source
from ..core.stage import FrameContext, PipelineStage
from ..estimation.kalman import ConstantAccelerationKF

__all__ = ["BBox", "TrackState", "Track", "MultiObjectTracker", "TrackingStage", "iou"]

BBox = tuple[float, float, float, float]      # (x, y, w, h)


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union of two ``(x, y, w, h)`` boxes."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    intersection = iw * ih
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union > 0 else 0.0


class TrackState(str, Enum):
    TENTATIVE = "tentative"    # seen, not yet confirmed
    CONFIRMED = "confirmed"    # reported to the operator
    COASTING = "coasting"      # predicted only; no recent detection
    LOST = "lost"              # deleted


@dataclass(slots=True)
class Track:
    """One tracked object: identity, motion model, appearance model, history."""

    id: int
    bbox: BBox
    label: str
    kf: ConstantAccelerationKF
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    misses: int = 0
    age: int = 0
    detection_confidence: float = 0.0
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    cv_tracker: Any = None

    @property
    def centre(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)

    @property
    def confidence(self) -> float:
        """Combines motion-model health with the detector's own confidence."""
        return float(0.6 * self.kf.track_confidence + 0.4 * self.detection_confidence)

    def predicted_bbox(self, dt: float) -> BBox:
        """Where the box should be next frame, per the motion model."""
        cx, cy = self.kf.predict_future(dt)[0]
        _, _, w, h = self.bbox
        return (float(cx - w / 2.0), float(cy - h / 2.0), w, h)

    def predict_path(self, horizon_s: float = 1.0, steps: int = 8) -> list[tuple[float, float]]:
        return self.kf.trajectory(horizon_s, steps)

    def to_dict(self) -> dict[str, Any]:
        vx, vy = self.kf.state().velocity
        return {
            "id": self.id,
            "label": self.label,
            "bbox": [round(v, 1) for v in self.bbox],
            "centre": [round(v, 1) for v in self.centre],
            "state": self.state.value,
            "hits": self.hits,
            "misses": self.misses,
            "age": self.age,
            "velocity_px_s": [round(vx, 2), round(vy, 2)],
            "speed_px_s": round(self.kf.state().speed, 2),
            "heading_deg": round(self.kf.state().heading_deg, 1),
            "confidence": round(self.confidence, 3),
            "trajectory": [[round(x, 1), round(y, 1)] for x, y in self.trajectory[-32:]],
        }


class MultiObjectTracker:
    """Tracking-by-detection with Kalman prediction and correlation fallback.

    Parameters
    ----------
    tracker_type:
        OpenCV correlation tracker used for coasting: ``CSRT`` (most accurate),
        ``KCF`` (balanced), ``MOSSE`` (fastest), or ``None`` to rely on the
        Kalman prediction alone.
    iou_threshold:
        Minimum overlap with the *predicted* box for a detection to be
        associated with an existing track.
    max_misses:
        Consecutive frames without a detection before the track is deleted.
    min_hits:
        Detections required before a track is reported as confirmed.
    """

    def __init__(self, *, tracker_type: Optional[str] = "CSRT",
                 iou_threshold: float = 0.3, max_misses: int = 12,
                 min_hits: int = 3, dt: float = 1.0 / 30.0,
                 process_noise_std: float = 20.0,
                 measurement_noise_std: float = 8.0) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_misses = int(max_misses)
        self.min_hits = int(min_hits)
        self.dt = float(dt)
        self.process_noise_std = float(process_noise_std)
        self.measurement_noise_std = float(measurement_noise_std)
        self.tracks: list[Track] = []
        self.frame_count = 0
        self._next_id = 1

        self.tracker_type = tracker_type.upper() if tracker_type else None
        self._tracker_factory = None
        if self.tracker_type:
            factories = tracker_factories()
            factory = factories.get(self.tracker_type)
            if factory is None:
                available = sorted(n for n, f in factories.items() if f is not None)
                raise CapabilityError(
                    f"tracker {self.tracker_type!r} is unavailable in this OpenCV "
                    f"build. Available: {available or 'none'}. MOSSE and BOOSTING "
                    "live under cv2.legacy and need opencv-contrib-python."
                )
            self._tracker_factory = factory

    @staticmethod
    def available_trackers() -> list[str]:
        return sorted(n for n, f in tracker_factories().items() if f is not None)

    # ---------------------------------------------------------------- lifecycle

    def _new_track(self, bbox: BBox, label: str, confidence: float,
                   frame: Optional[np.ndarray]) -> Track:
        cx, cy = bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0
        kf = ConstantAccelerationKF(
            dt=self.dt,
            process_noise_std=self.process_noise_std,
            measurement_noise_std=self.measurement_noise_std,
            initial_position=(cx, cy),
        )
        track = Track(
            id=self._next_id, bbox=bbox, label=label, kf=kf,
            detection_confidence=float(confidence), trajectory=[(cx, cy)],
        )
        self._next_id += 1
        track.cv_tracker = self._make_cv_tracker(frame, bbox)
        return track

    def _make_cv_tracker(self, frame: Optional[np.ndarray], bbox: BBox) -> Any:
        if self._tracker_factory is None or frame is None:
            return None
        try:
            tracker = self._tracker_factory()
            x, y, w, h = (int(round(v)) for v in bbox)
            if w <= 0 or h <= 0:
                return None
            tracker.init(frame, (x, y, w, h))
            return tracker
        except Exception:  # noqa: BLE001 - a tracker that will not init is not fatal
            return None

    # -------------------------------------------------------------- association

    def _associate(self, detections: Sequence[dict[str, Any]]
                   ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Greedy IoU matching against Kalman-predicted boxes."""
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))

        predicted = [t.predicted_bbox(self.dt) for t in self.tracks]
        scores = [
            (iou(predicted[ti], det["bbox"]), ti, di)
            for ti in range(len(self.tracks))
            for di, det in enumerate(detections)
        ]
        scores.sort(reverse=True)

        matched: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for score, ti, di in scores:
            if score < self.iou_threshold:
                break
            if ti in used_tracks or di in used_dets:
                continue
            matched.append((ti, di))
            used_tracks.add(ti)
            used_dets.add(di)

        unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in used_tracks]
        return matched, unmatched_dets, unmatched_tracks

    # ------------------------------------------------------------------- update

    def update(self, frame: Optional[np.ndarray] = None,
               detections: Optional[Sequence[dict[str, Any]]] = None) -> list[Track]:
        """Advance all tracks by one frame.

        ``detections`` is a sequence of ``{"bbox": (x,y,w,h), "label": str,
        "confidence": float}``. Passing ``None`` runs a detection-free frame:
        every track coasts on its correlation tracker and Kalman prediction.
        """
        self.frame_count += 1
        detections = list(detections or [])

        for track in self.tracks:
            track.kf.predict(self.dt)
            track.age += 1

        matched, new_dets, lost_tracks = self._associate(detections)

        for ti, di in matched:
            track = self.tracks[ti]
            det = detections[di]
            bbox = tuple(float(v) for v in det["bbox"])
            cx, cy = bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0
            track.kf.update((cx, cy))
            track.bbox = bbox
            track.hits += 1
            track.misses = 0
            track.detection_confidence = float(det.get("confidence", 1.0))
            track.trajectory.append((cx, cy))
            if track.state in (TrackState.TENTATIVE, TrackState.COASTING) \
                    and track.hits >= self.min_hits:
                track.state = TrackState.CONFIRMED
            # Re-seed appearance from the fresh detection so the correlation
            # tracker does not drift onto the background over a long track.
            track.cv_tracker = self._make_cv_tracker(frame, bbox)

        for ti in lost_tracks:
            track = self.tracks[ti]
            track.misses += 1
            track.kf.mark_missed()
            updated = False
            if track.cv_tracker is not None and frame is not None:
                try:
                    ok, box = track.cv_tracker.update(frame)
                except Exception:  # noqa: BLE001
                    ok, box = False, None
                if ok and box is not None:
                    bbox = tuple(float(v) for v in box)
                    cx, cy = bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0
                    track.bbox = bbox
                    track.kf.update((cx, cy))
                    track.trajectory.append((cx, cy))
                    updated = True
            if not updated:
                # Fall back to pure prediction so the box keeps moving with the
                # motion model rather than freezing in place.
                track.bbox = track.predicted_bbox(0.0)
                track.trajectory.append(track.centre)
            if track.state is TrackState.CONFIRMED:
                track.state = TrackState.COASTING

        for di in new_dets:
            det = detections[di]
            self.tracks.append(self._new_track(
                tuple(float(v) for v in det["bbox"]),
                str(det.get("label", "object")),
                float(det.get("confidence", 1.0)),
                frame,
            ))

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return self.confirmed_tracks()

    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self.tracks
                if t.state in (TrackState.CONFIRMED, TrackState.COASTING)]

    def statistics(self) -> dict[str, Any]:
        by_state: dict[str, int] = {}
        for track in self.tracks:
            by_state[track.state.value] = by_state.get(track.state.value, 0) + 1
        return {
            "active_tracks": len(self.tracks),
            "confirmed": len(self.confirmed_tracks()),
            "by_state": by_state,
            "frames": self.frame_count,
            "next_id": self._next_id,
            "tracker_type": self.tracker_type,
        }


class TrackingStage(PipelineStage):
    """Maintains persistent identities from per-frame detections."""

    name = "tracking"
    requires = ("frame",)
    provides = ("tracks",)
    source = Source.ESTIMATED

    def __init__(self, tracker: Optional[MultiObjectTracker] = None, **params: Any) -> None:
        super().__init__(**params)
        self.tracker = tracker or MultiObjectTracker()

    def run(self, ctx: FrameContext) -> str:
        (frame,) = ctx.require("frame")
        detections_tv = ctx.get("detections")
        detections = detections_tv.value if detections_tv is not None else None
        tracks = self.tracker.update(frame.value, detections)

        parents = [frame] + ([detections_tv] if detections_tv is not None else [])
        mean_confidence = (
            float(np.mean([t.confidence for t in tracks])) if tracks else 0.0
        )
        self.emit(ctx, "tracks", tracks,
                  algorithm=f"MultiObjectTracker(IoU + Kalman + "
                            f"{self.tracker.tracker_type or 'prediction-only'})",
                  parents=parents, confidence=mean_confidence, units="tracks",
                  parameters=self.tracker.statistics())
        coasting = sum(1 for t in tracks if t.state is TrackState.COASTING)
        return (f"{len(tracks)} confirmed"
                + (f", {coasting} coasting" if coasting else "")
                + (" (no detections this frame)" if detections is None else ""))
