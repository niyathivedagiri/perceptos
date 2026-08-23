"""
perceptos.sensors.camera
========================

Frame acquisition with honest health accounting.

Four source types behind one interface: webcam/USB index, video file, IP or
RTSP stream, and a synthetic pattern generator. The synthetic source exists so
the whole pipeline is testable in CI and on a machine with no camera, and it
tags its frames ``SIMULATION`` so those frames can never be mistaken for real
ones downstream.

The health accounting is the part that usually gets faked. Three distinct
numbers matter and they are not interchangeable:

* **FPS** — frames actually delivered per second, measured over a rolling
  window, not the camera's advertised rate.
* **Latency** — time spent inside the blocking ``read()`` call. This is
  acquisition cost only; pipeline processing time is measured separately by
  the stage timing, so a slow detector never gets misattributed to the camera.
* **Dropped frames** — for a *file* source, the difference between the frame
  index the file reports and the number we have consumed; the decoder knows
  exactly. For a *live* source there is no ground truth, so the count is
  derived from elapsed time against the nominal rate and is explicitly marked
  ``ESTIMATED``. Reporting a live drop count as if it were measured is the
  common lie in camera dashboards.
"""

from __future__ import annotations

import abc
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..core.cv_compat import CapabilityError, cv
from ..core.provenance import Source, Traced, traced
from ..core.stage import FrameContext, PipelineStage

__all__ = [
    "FrameSource",
    "VideoCaptureSource",
    "WebcamSource",
    "VideoFileSource",
    "IPCameraSource",
    "SyntheticSource",
    "CameraHealth",
    "CameraManager",
    "AcquisitionStage",
]


@dataclass(slots=True)
class CameraHealth:
    """Rolling acquisition telemetry."""

    fps: float = 0.0
    fps_nominal: Optional[float] = None
    latency_ms: float = 0.0
    latency_p95_ms: float = 0.0
    frames_delivered: int = 0
    frames_dropped: int = 0
    read_failures: int = 0
    uptime_s: float = 0.0
    dropped_is_measured: bool = False
    connected: bool = False
    #: Rolling-window samples behind `fps`. Below `MIN_SAMPLES` the rate is
    #: not yet meaningful and must not drive the grade.
    fps_samples: int = 0

    MIN_SAMPLES: int = 5

    @property
    def drop_rate(self) -> float:
        total = self.frames_delivered + self.frames_dropped
        return self.frames_dropped / total if total else 0.0

    @property
    def grade(self) -> str:
        if not self.connected:
            return "offline"
        if self.read_failures > 0 and self.frames_delivered == 0:
            return "failed"
        # A rate computed from one or two intervals is noise. Grading on it
        # makes every freshly opened camera flash "degraded" for a second,
        # which trains operators to ignore the indicator.
        if self.fps_samples < self.MIN_SAMPLES:
            return "starting"
        if self.drop_rate > 0.20 or (self.fps_nominal and self.fps < 0.5 * self.fps_nominal):
            return "degraded"
        if self.drop_rate > 0.05:
            return "marginal"
        return "nominal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "fps": round(self.fps, 2),
            "fps_nominal": self.fps_nominal,
            "latency_ms": round(self.latency_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "frames_delivered": self.frames_delivered,
            "frames_dropped": self.frames_dropped,
            "drop_rate": round(self.drop_rate, 4),
            "dropped_is_measured": self.dropped_is_measured,
            "read_failures": self.read_failures,
            "uptime_s": round(self.uptime_s, 1),
            "fps_samples": self.fps_samples,
            "connected": self.connected,
            "grade": self.grade,
        }


class FrameSource(abc.ABC):
    """Anything that can yield frames."""

    #: Provenance source class for frames from this input.
    source_kind: Source = Source.LIVE_CAMERA
    #: True when the decoder can report exact drop counts.
    drops_are_measured: bool = False

    @abc.abstractmethod
    def open(self) -> None: ...

    @abc.abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Return the next frame (BGR) or ``None`` if unavailable."""

    @abc.abstractmethod
    def close(self) -> None: ...

    @property
    def nominal_fps(self) -> Optional[float]:
        return None

    @property
    def is_open(self) -> bool:
        return True

    def describe(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "source": self.source_kind.value}


class VideoCaptureSource(FrameSource):
    """Base for anything backed by ``cv2.VideoCapture``."""

    def __init__(self, target: Any, *, api_preference: Optional[int] = None) -> None:
        self.target = target
        self.api_preference = api_preference
        self._cap: Any = None

    def open(self) -> None:
        cv2 = cv()
        if self.api_preference is not None:
            self._cap = cv2.VideoCapture(self.target, self.api_preference)
        else:
            self._cap = cv2.VideoCapture(self.target)
        if not self._cap.isOpened():
            self._cap = None
            raise CapabilityError(f"could not open video source {self.target!r}")

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def nominal_fps(self) -> Optional[float]:
        if self._cap is None:
            return None
        fps = float(self._cap.get(cv().CAP_PROP_FPS))
        # Webcams frequently report 0 or a nonsense value here.
        return fps if 0.1 < fps < 1000 else None

    def property_snapshot(self) -> dict[str, Any]:
        if self._cap is None:
            return {}
        cv2 = cv()
        return {
            "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.nominal_fps,
            "backend": self._cap.getBackendName(),
        }

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "target": str(self.target), **self.property_snapshot()}


class WebcamSource(VideoCaptureSource):
    """USB or built-in camera by device index."""

    def __init__(self, index: int = 0, *, width: Optional[int] = None,
                 height: Optional[int] = None, fps: Optional[float] = None) -> None:
        super().__init__(int(index))
        self._requested = (width, height, fps)

    def open(self) -> None:
        super().open()
        cv2 = cv()
        width, height, fps = self._requested
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        if fps:
            self._cap.set(cv2.CAP_PROP_FPS, float(fps))
        # Minimise buffering so `read()` returns the newest frame, not a stale
        # one from the driver queue. Without this a "live" feed can run
        # seconds behind reality while still reporting a healthy FPS.
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001 - unsupported on some backends
            pass


class VideoFileSource(VideoCaptureSource):
    """Recorded video. The decoder knows the true frame index, so drops are exact."""

    drops_are_measured = True

    def __init__(self, path: str, *, loop: bool = False) -> None:
        super().__init__(str(path))
        self.loop = bool(loop)

    def read(self) -> Optional[np.ndarray]:
        frame = super().read()
        if frame is None and self.loop and self._cap is not None:
            self._cap.set(cv().CAP_PROP_POS_FRAMES, 0)
            frame = super().read()
        return frame

    @property
    def position(self) -> int:
        return int(self._cap.get(cv().CAP_PROP_POS_FRAMES)) if self._cap else 0

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv().CAP_PROP_FRAME_COUNT)) if self._cap else 0


class IPCameraSource(VideoCaptureSource):
    """Network camera (HTTP MJPEG, RTSP). Reconnects on stream loss."""

    def __init__(self, url: str, *, reconnect_attempts: int = 3,
                 reconnect_delay_s: float = 1.0) -> None:
        super().__init__(str(url))
        self.reconnect_attempts = int(reconnect_attempts)
        self.reconnect_delay_s = float(reconnect_delay_s)
        self.reconnects = 0

    def read(self) -> Optional[np.ndarray]:
        frame = super().read()
        if frame is not None:
            return frame
        # A network stream dropping a frame is routine; give it a bounded
        # number of reconnects before declaring the camera offline.
        for _ in range(self.reconnect_attempts):
            time.sleep(self.reconnect_delay_s)
            try:
                self.close()
                self.open()
            except CapabilityError:
                continue
            self.reconnects += 1
            frame = super().read()
            if frame is not None:
                return frame
        return None


class SyntheticSource(FrameSource):
    """Deterministic test pattern. Every frame is tagged SIMULATION.

    Generates a moving target against a textured background so feature
    extraction, tracking and optical flow all have something real to work on
    without a camera attached.
    """

    source_kind = Source.SIMULATION
    drops_are_measured = True

    def __init__(self, *, width: int = 640, height: int = 480, fps: float = 30.0,
                 seed: int = 0, total_frames: Optional[int] = None,
                 pan_px_per_frame: float = 0.0) -> None:
        self.width, self.height = int(width), int(height)
        self._fps = float(fps)
        self.total_frames = total_frames
        self.pan_px_per_frame = float(pan_px_per_frame)
        self.index = 0
        rng = np.random.default_rng(seed)
        # The background canvas is wider than the frame so the camera can pan
        # across it. Without ego-motion there is no parallax, visual odometry
        # correctly reports "stationary", and the module never gets exercised.
        self._pan_span = int(abs(self.pan_px_per_frame) * (total_frames or 60)) + 4
        canvas_w = self.width + self._pan_span
        base = rng.integers(40, 90, (self.height, canvas_w), dtype=np.uint8)
        for _ in range(60 + self._pan_span // 8):
            cx, cy = rng.integers(0, canvas_w), rng.integers(0, self.height)
            size = int(rng.integers(8, 28))
            y0, y1 = max(0, cy - size), min(self.height, cy + size)
            x0, x1 = max(0, cx - size), min(canvas_w, cx + size)
            base[y0:y1, x0:x1] = int(rng.integers(100, 220))
        self._canvas = np.stack([base, base, base], axis=2)
        self._opened = False

    @property
    def pan_offset(self) -> int:
        """Ground-truth camera x-offset for the current frame, in pixels."""
        span = max(1, self._pan_span)
        return int(abs(self.pan_px_per_frame) * self.index) % span

    def open(self) -> None:
        self._opened = True
        self.index = 0

    def read(self) -> Optional[np.ndarray]:
        if not self._opened:
            return None
        if self.total_frames is not None and self.index >= self.total_frames:
            return None
        offset = self.pan_offset
        frame = self._canvas[:, offset:offset + self.width].copy()
        # A bright square on a smooth diagonal path — trackable ground truth.
        t = self.index
        cx = int(80 + (t * 4) % max(1, self.width - 160))
        cy = int(self.height / 2 + 60 * np.sin(t * 0.08))
        half = 24
        frame[max(0, cy - half):cy + half, max(0, cx - half):cx + half] = (60, 220, 240)
        self.index += 1
        return frame

    def target_centre(self, index: Optional[int] = None) -> tuple[int, int]:
        """Ground-truth target position — used to validate the tracker."""
        t = self.index - 1 if index is None else index
        return (int(80 + (t * 4) % max(1, self.width - 160)),
                int(self.height / 2 + 60 * np.sin(t * 0.08)))

    def close(self) -> None:
        self._opened = False

    @property
    def is_open(self) -> bool:
        return self._opened

    @property
    def nominal_fps(self) -> Optional[float]:
        return self._fps

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "width": self.width, "height": self.height,
                "fps": self._fps, "synthetic": True}


class CameraManager:
    """Owns one source and its health telemetry."""

    def __init__(self, source: FrameSource, *, window: int = 60,
                 name: str = "camera") -> None:
        self.source = source
        self.name = name
        self.window = int(window)
        self._intervals: deque[float] = deque(maxlen=self.window)
        self._latencies: deque[float] = deque(maxlen=self.window)
        self._last_read_at: Optional[float] = None
        self._opened_at: Optional[float] = None
        self.frames_delivered = 0
        self.read_failures = 0
        self._expected_index = 0

    # ---------------------------------------------------------------- lifecycle

    def open(self) -> "CameraManager":
        self.source.open()
        self._opened_at = time.perf_counter()
        return self

    def close(self) -> None:
        self.source.close()

    def __enter__(self) -> "CameraManager":
        return self.open()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------- read

    def read(self) -> Optional[Traced[np.ndarray]]:
        """Grab one frame as a traced value, updating telemetry."""
        start = time.perf_counter()
        frame = self.source.read()
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._latencies.append(latency_ms)

        if frame is None:
            self.read_failures += 1
            return None

        if self._last_read_at is not None:
            self._intervals.append(start - self._last_read_at)
        self._last_read_at = start
        self.frames_delivered += 1

        return traced(
            frame,
            source=self.source.source_kind,
            algorithm=f"{type(self.source).__name__}.read",
            confidence=1.0 if self.source.source_kind.is_measured else None,
            latency_ms=latency_ms,
            units="px",
            label=f"{self.name}_frame",
            parameters={"shape": list(frame.shape), "index": self.frames_delivered},
        )

    # ----------------------------------------------------------------- health

    @property
    def fps(self) -> float:
        if not self._intervals:
            return 0.0
        mean_interval = sum(self._intervals) / len(self._intervals)
        return 1.0 / mean_interval if mean_interval > 0 else 0.0

    def dropped_frames(self) -> tuple[int, bool]:
        """Return ``(count, is_measured)``.

        For a decoder-backed source the count is exact. For a live camera it is
        inferred from elapsed time against the nominal rate, and the caller is
        told so it can be badged ESTIMATED rather than presented as fact.
        """
        source = self.source
        if isinstance(source, VideoFileSource) and source.is_open:
            consumed = source.position
            return max(0, consumed - self.frames_delivered), True
        if isinstance(source, SyntheticSource):
            return max(0, source.index - self.frames_delivered), True
        nominal = source.nominal_fps
        if nominal and self._opened_at is not None:
            elapsed = time.perf_counter() - self._opened_at
            expected = int(elapsed * nominal)
            return max(0, expected - self.frames_delivered), False
        return 0, False

    def health(self) -> CameraHealth:
        dropped, measured = self.dropped_frames()
        latencies = sorted(self._latencies)
        p95 = latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0.0
        return CameraHealth(
            fps=self.fps,
            fps_nominal=self.source.nominal_fps,
            latency_ms=(sum(self._latencies) / len(self._latencies)) if self._latencies else 0.0,
            latency_p95_ms=p95,
            frames_delivered=self.frames_delivered,
            frames_dropped=dropped,
            read_failures=self.read_failures,
            uptime_s=(time.perf_counter() - self._opened_at) if self._opened_at else 0.0,
            dropped_is_measured=measured,
            fps_samples=len(self._intervals),
            connected=self.source.is_open,
        )

    def traced_health(self) -> Traced[CameraHealth]:
        health = self.health()
        return traced(
            health,
            # FPS and latency are measured; the drop count may be inferred, so
            # the whole record degrades to ESTIMATED when it is.
            source=Source.LIVE_SENSOR if health.dropped_is_measured else Source.ESTIMATED,
            algorithm="CameraManager.health(rolling window)",
            confidence=1.0 if health.dropped_is_measured else 0.6,
            units="fps, ms",
            label=f"{self.name}_health",
            parameters={"window": self.window, "source": self.source.describe()},
        )


class AcquisitionStage(PipelineStage):
    """Seeds the pipeline with a frame and its greyscale conversion."""

    name = "acquisition"
    requires = ()
    provides = ("frame", "gray")
    source = Source.LIVE_CAMERA

    def __init__(self, manager: CameraManager, **params: Any) -> None:
        super().__init__(**params)
        self.manager = manager
        self.source = manager.source.source_kind

    def run(self, ctx: FrameContext) -> str:
        frame = self.manager.read()
        if frame is None:
            raise RuntimeError(
                f"camera {self.manager.name!r} returned no frame "
                f"(failures: {self.manager.read_failures})"
            )
        ctx.put("frame", frame, stage=self.name, artefact=frame.value)
        cv2 = cv()
        gray = cv2.cvtColor(frame.value, cv2.COLOR_BGR2GRAY)
        self.emit(ctx, "gray", gray, algorithm="cv2.cvtColor(BGR2GRAY)",
                  source=frame.source, parents=[frame], units="px")
        health = self.manager.health()
        ctx.metadata["camera_health"] = health.to_dict()
        return f"{health.fps:.1f} fps, {health.latency_ms:.1f} ms, grade {health.grade}"
