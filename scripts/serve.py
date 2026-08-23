#!/usr/bin/env python3
"""
PerceptOS API server.

    python scripts/serve.py                      # synthetic camera, Mars mode
    python scripts/serve.py --source webcam      # live camera
    python scripts/serve.py --source nasa         # cached NASA Perseverance frames
    python scripts/serve.py --source dataset --path datasets/<dir>
    python scripts/serve.py --environment warehouse --port 8080

Then:

    http://localhost:8000/docs                   interactive API documentation
    http://localhost:8000/api/status
    http://localhost:8000/api/pipeline/diagram   Mermaid, generated from the wiring
    ws://localhost:8000/ws/telemetry             live frame + health push

The pipeline is assembled here and injected into the service, so the API layer
never imports OpenCV. If OpenCV is missing the server still starts and serves
environments, missions, lineage and health — the perception stages are simply
absent, and ``/api/status`` says so rather than the process refusing to boot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from perceptos.api.app import create_app                       # noqa: E402
from perceptos.api.service import PerceptOSService             # noqa: E402
from perceptos.core.cv_compat import HAS_OPENCV                # noqa: E402
from perceptos.core.environment import Environment             # noqa: E402
from perceptos.core.pipeline import Pipeline                   # noqa: E402
from perceptos.db.store import MissionStore                    # noqa: E402
from perceptos.health.monitor import HealthMonitor, PowerModel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _stereo_entries_from_manifest(root: Path, assets: dict) -> list:
    """Pair cached images by spacecraft clock and attach their camera models.

    The manifest records each image's metadata, including the SCLK used for
    stereo pairing. Pairs without a camera model on both eyes are excluded:
    they cannot be rectified, and this source exists to produce depth.
    """
    buckets: dict[tuple, dict[str, dict]] = {}
    for asset in assets.values():
        meta = asset.get("metadata") or {}
        eye, sclk, sol = meta.get("eye"), meta.get("sclk"), meta.get("sol")
        if eye not in ("left", "right") or sclk is None:
            continue
        buckets.setdefault((sol, round(float(sclk))), {})[eye] = {
            "path": root / asset["name"], "meta": meta,
        }

    entries = []
    for (sol, sclk), eyes in sorted(buckets.items()):
        if "left" not in eyes or "right" not in eyes:
            continue
        left, right = eyes["left"], eyes["right"]
        if not (left["path"].exists() and right["path"].exists()):
            continue
        # The manifest stores whether a model existed, but not the model
        # itself; re-querying the archive offline is impossible, so pairs
        # without one are skipped rather than given an assumed geometry.
        if not (left["meta"].get("has_calibration")
                and right["meta"].get("has_calibration")):
            continue
        entries.append((left["path"], right["path"], None, None,
                        {"sol": sol, "sclk": sclk,
                         "credit": "NASA/JPL-Caltech",
                         "dataset": "perseverance_jpl"}))
    return entries


def build_pipeline_factory(source: str, path: str | None):
    """Return a factory that assembles a pipeline for a given environment."""

    def factory(environment: Environment) -> Pipeline:
        if not HAS_OPENCV:
            # Degraded mode: everything except the perception stages still works.
            from perceptos.core.provenance import Source
            from perceptos.core.stage import FrameContext, PipelineStage

            class NullAcquisition(PipelineStage):
                name = "acquisition"
                provides = ("frame",)
                source = Source.SIMULATION

                def run(self, ctx: FrameContext) -> str:
                    import numpy as np
                    self.emit(ctx, "frame", np.zeros((8, 8), np.uint8),
                              algorithm="null_source(OpenCV unavailable)",
                              source=Source.SIMULATION, confidence=1.0)
                    return "OpenCV not installed — perception stages disabled"

            return Pipeline([NullAcquisition()], mode=environment.name, seed_keys=())

        from perceptos.features.extractors import FeatureExtractor, FeatureStage
        from perceptos.imageproc.filters import ImageChain, ImageProcessingStage
        from perceptos.sensors.camera import (
            AcquisitionStage,
            CameraManager,
            SyntheticSource,
            WebcamSource,
        )
        from perceptos.tracking.trackers import MultiObjectTracker, TrackingStage

        if source == "stereo":
            # Cached NASA stereo pairs, rectified from their published CAHV
            # camera models. This is the only source that yields metric depth.
            from perceptos.data.cache import DatasetCache
            from perceptos.data.mars2020_raw import JplRawImageClient
            from perceptos.sensors.stereo_source import (
                RectifiedStereoSource,
                RectifiedStereoStage,
            )
            from perceptos.stereo.matcher import DepthStage, StereoMatcher, StereoStage

            cache = DatasetCache(ROOT / "datasets", offline=True)
            client = JplRawImageClient(cache)
            root = Path(path) if path else client.cache.dataset_dir(client.dataset)
            manifest = root / "manifest.json"
            if not manifest.exists():
                raise SystemExit(
                    f"No cached NASA stereo pairs under {root}.\n"
                    "Fetch some first (no API key needed):\n"
                    "    python scripts/fetch_data.py perseverance --sol 203\n"
                    "PerceptOS will not substitute simulated imagery."
                )
            import json as _json
            assets = _json.loads(manifest.read_text()).get("assets", {})
            entries = _stereo_entries_from_manifest(root, assets)
            if not entries:
                raise SystemExit(
                    f"{len(assets)} cached image(s) under {root}, but no pair "
                    "had a published camera model on both eyes. Depth needs "
                    "one; PerceptOS will not assume a baseline. Try another "
                    "sol, or use --source nasa for imagery without depth."
                )
            stereo = RectifiedStereoSource(entries, loop=True).open()
            print(f"  stereo pairs: {len(entries)} from {root}")
            return Pipeline([
                RectifiedStereoStage(stereo),
                ImageProcessingStage(ImageChain().add("clahe", clip_limit=2.0)),
                FeatureStage(FeatureExtractor(
                    "ORB" if "ORB" in FeatureExtractor.available() else "HARRIS",
                    max_features=600)),
                StereoStage(StereoMatcher(algorithm="SGBM", num_disparities=128,
                                          block_size=5)),
                DepthStage(),
            ], mode=environment.name, seed_keys=())

        if source == "nasa":
            # Cached NASA frames, replayed through the same FrameSource
            # interface as everything else. The pipeline cannot tell the
            # difference, which is the whole point of the abstraction.
            from perceptos.sensors.dataset_source import DatasetFrameSource
            root = Path(path) if path else (ROOT / "datasets" / "perseverance_jpl")
            candidates = sorted(root.glob("sol*")) if root.is_dir() else []
            target = Path(path) if (path and Path(path).is_dir()
                                    and any(Path(path).glob("*.png"))) else (
                candidates[0] if candidates else None)
            if target is None:
                raise SystemExit(
                    f"No cached NASA frames under {root}.\n"
                    "Fetch some first (no API key needed):\n"
                    "    python scripts/fetch_data.py perseverance --sol 203\n"
                    "PerceptOS will not substitute simulated data for a missing "
                    "dataset."
                )
            frame_source = DatasetFrameSource.from_directory(target, loop=True)
            print(f"  NASA frames : {target}")
        elif source == "webcam":
            frame_source = WebcamSource(0)
        elif source == "dataset":
            from perceptos.sensors.dataset_source import DatasetFrameSource
            if not path:
                raise SystemExit("--source dataset requires --path")
            frame_source = DatasetFrameSource.from_directory(path, loop=True)
        else:
            frame_source = SyntheticSource(width=640, height=480,
                                           pan_px_per_frame=4.0)

        camera = CameraManager(frame_source, name="navcam").open()
        detector = "ORB" if "ORB" in FeatureExtractor.available() else "HARRIS"
        trackers = MultiObjectTracker.available_trackers()

        return Pipeline(
            [
                AcquisitionStage(camera),
                ImageProcessingStage(ImageChain().add("clahe", clip_limit=2.0)),
                FeatureStage(FeatureExtractor(detector, max_features=600)),
                TrackingStage(MultiObjectTracker(
                    tracker_type=trackers[0] if trackers else None, min_hits=2,
                )),
            ],
            mode=environment.name,
            seed_keys=(),
        )

    return factory


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the PerceptOS API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--environment", default="mars",
                        choices=["mars", "moon", "earth", "warehouse", "factory"])
    parser.add_argument("--source", default="synthetic",
                        choices=["synthetic", "nasa", "stereo", "webcam", "dataset"],
                        help="nasa = cached Perseverance frames; "
                             "stereo = cached pairs rectified for real depth")
    parser.add_argument("--path", default=None, help="dataset directory")
    parser.add_argument("--db", default=str(ROOT / "missions.db"))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    service = PerceptOSService(
        store=MissionStore(args.db),
        pipeline_factory=build_pipeline_factory(args.source, args.path),
        monitor=HealthMonitor(power=PowerModel(), disk_path=str(ROOT)),
        environment=args.environment,
    )
    app = create_app(service)

    print(f"  environment : {service.environment.display_name}")
    print(f"  frame source: {args.source}")
    print(f"  opencv      : {'available' if HAS_OPENCV else 'MISSING - degraded mode'}")
    print(f"  database    : {args.db}")
    print(f"  console     : http://{args.host}:{args.port}/console")
    print(f"  docs        : http://{args.host}:{args.port}/docs")
    print(f"  status      : http://{args.host}:{args.port}/")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
