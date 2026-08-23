#!/usr/bin/env python3
"""
PerceptOS — full perception pipeline demonstration
==================================================

Runs the complete Phase 2 chain as a single validated pipeline:

    acquisition -> image processing -> features -> stereo -> depth
                -> detection -> tracking -> visual odometry

and then demonstrates the three things the spec asks the platform to prove:

1. **The block diagram is generated from the real wiring**, so it cannot drift
   out of date with the code.
2. **Modes switch without a restart** — the same pipeline object is
   reconfigured from a mode profile mid-run.
3. **Every decision is walkable backwards** through the lineage graph.

The camera is a synthetic source, so every frame is tagged ``SIMULATION`` and
that tag propagates through every derived value. Point ``AcquisitionStage`` at
a real ``WebcamSource`` and the identical pipeline produces ``LIVE_CAMERA``
provenance with no other change.

Run:  python demo_perception_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from perceptos.core.compute import device_report                    # noqa: E402
from perceptos.core.cv_compat import HAS_OPENCV, capability_report  # noqa: E402
from perceptos.core.pipeline import Pipeline                        # noqa: E402


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def main() -> int:
    if not HAS_OPENCV:
        print("OpenCV is not installed. Install it with:")
        print("    pip install opencv-contrib-python")
        return 1

    from perceptos.detection.yolo import DetectionStage, YoloDetector
    from perceptos.features.extractors import FeatureExtractor, FeatureStage
    from perceptos.imageproc.filters import ImageChain, ImageProcessingStage
    from perceptos.mars.odometry import VisualOdometry, VisualOdometryStage
    from perceptos.sensors.camera import (
        AcquisitionStage,
        CameraManager,
        SyntheticSource,
    )
    from perceptos.tracking.trackers import MultiObjectTracker, TrackingStage

    # ---------------------------------------------------------- 0. environment
    banner("0  COMPUTE ENVIRONMENT")
    device = device_report()
    print(f"  device            : {device['name']} ({device['kind']})")
    print(f"  reason            : {device['reason']}")
    print(f"  opencv            : {device['opencv_version']}")
    caps = capability_report()
    print(f"  detectors         : {', '.join(caps['detectors_available'])}")
    print(f"  trackers          : {', '.join(caps['trackers_available']) or 'none'}")
    missing = [n for n, s in caps["detectors"].items() if not s["available"]]
    if missing:
        print(f"  unavailable here  : {', '.join(missing)}")
        print("                      (present in opencv-contrib-python 4.x)")

    # ------------------------------------------------------------- 1. assemble
    banner("1  PIPELINE ASSEMBLY")
    # pan_px_per_frame gives the camera real ego-motion, so visual odometry
    # has genuine parallax to solve rather than a stationary scene.
    source = SyntheticSource(width=640, height=480, fps=30.0, total_frames=60,
                             pan_px_per_frame=6.0)
    camera = CameraManager(source, name="navcam").open()

    detector_choice = "ORB" if "ORB" in caps["detectors_available"] else "HARRIS"
    tracker_choice = next(
        (t for t in ("CSRT", "KCF", "MOSSE") if t in caps["trackers_available"]), None
    )

    K = np.array([[520.0, 0.0, 320.0], [0.0, 520.0, 240.0], [0.0, 0.0, 1.0]])

    pipeline = Pipeline(
        [
            AcquisitionStage(camera),
            ImageProcessingStage(
                ImageChain().add("clahe", clip_limit=2.0).add("gaussian_blur", ksize=3)
            ),
            FeatureStage(FeatureExtractor(detector_choice, max_features=600)),
            DetectionStage(YoloDetector(weights="models/yolo11n.pt")),
            TrackingStage(MultiObjectTracker(tracker_type=tracker_choice, min_hits=2,
                                             dt=1 / 30)),
            VisualOdometryStage(VisualOdometry(K, min_features=120)),
        ],
        name="perception",
        mode="earth_inspection",
        seed_keys=(),
    )
    print(f"  {pipeline!r}")
    print("\n  wiring contract (validated at construction):")
    for entry in pipeline.contract():
        requires = ", ".join(entry["requires"]) or "-"
        provides = ", ".join(entry["provides"]) or "-"
        print(f"    {entry['name']:<17} requires [{requires:<22}] provides [{provides}]")

    # -------------------------------------------------------------- 2. execute
    banner("2  EXECUTION  (20 frames)")
    reports = []
    for _ in range(20):
        ctx, report = pipeline.process({})
        reports.append((ctx, report))

    last_ctx, last_report = reports[-1]
    print("  stage                status      latency    detail")
    print("  " + "-" * 70)
    for stage in last_report.stages:
        detail = (stage.message or stage.error or "").splitlines()[0][:34]
        print(f"  {stage.stage:<20} {stage.status.value:<11} "
              f"{stage.latency_ms:>7.2f}ms  {detail}")
    print("  " + "-" * 70)
    print(f"  total {last_report.total_latency_ms:.2f} ms  ->  "
          f"{last_report.to_dict()['fps_ceiling']} fps ceiling")
    print(f"  bottleneck: {last_report.bottleneck.stage if last_report.bottleneck else 'n/a'}")

    health = camera.health()
    print(f"\n  camera: {health.fps:.1f} fps measured "
          f"(nominal {health.fps_nominal}), {health.latency_ms:.2f} ms read, "
          f"{health.frames_dropped} dropped "
          f"({'measured' if health.dropped_is_measured else 'estimated'}), "
          f"grade {health.grade}")

    features = last_ctx.get("features")
    if features is not None:
        stats = features.value.statistics()
        print(f"  features: {stats['count']} {stats['method']} keypoints, "
              f"spread {stats['spatial_spread']}, quality {stats['quality']}")

    # A detector with no weights on disk is expected here — the point is that
    # the rest of the pipeline keeps running and says so plainly.
    detection = next(s for s in last_report.stages if s.stage == "detection")
    print(f"  detection: {detection.status.value} "
          f"— {(detection.message or '').splitlines()[0][:60]}")

    # --------------------------------------------------------- 3. block diagram
    banner("3  SYSTEM BLOCK DIAGRAM  (generated from the wiring, not hand-drawn)")
    print(pipeline.to_mermaid())

    # ----------------------------------------------------------- 4. mode switch
    banner("4  MODE SWITCH  (no restart)")
    print(f"  before: mode={pipeline.mode}")
    for stage in pipeline.stages:
        print(f"    {stage.name:<20} {'enabled' if stage.enabled else 'disabled'}")

    pipeline.apply_mode({
        "mode": "mars_exploration",
        "stages": {
            "tracking": {"enabled": False},          # static scene, no moving targets
            "detection": {"enabled": False},         # geology model not loaded
            "features": {"params": {"max_features": 2000}},
        },
    })
    print(f"\n  after:  mode={pipeline.mode}")
    for stage in pipeline.stages:
        print(f"    {stage.name:<20} {'enabled' if stage.enabled else 'disabled'}")

    ctx, report = pipeline.process({})
    statuses = {s.stage: s.status.value for s in report.stages}
    print(f"\n  next frame statuses: {statuses}")
    print("  the disabled stages are still reported, not silently absent")

    # ------------------------------------------------------------ 5. lineage
    banner("5  DATA LINEAGE  —  where did this value come from?")
    target = last_ctx.get("features") or last_ctx.get("gray")
    explanation = last_ctx.lineage.explain(target.id)
    print(f"  value                 : {target.label}")
    print(f"  derivation depth      : {explanation['depth']} nodes")
    print(f"  weakest confidence    : {explanation['weakest_confidence']}")
    print(f"  simulation in chain   : {explanation['contains_simulation']}")
    print(f"  total latency         : {explanation['total_latency_ms']} ms")
    print("\n  chain:")
    for i, node in enumerate(explanation["chain"]):
        prov = node["provenance"]
        conf = prov["confidence"]
        conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else " -- "
        latency = prov["latency_ms"]
        latency_s = f"{latency:>7.2f}ms" if latency is not None else "     --  "
        print(f"    {i + 1}. {node['label']:<14} {prov['source']:<11} "
              f"conf {conf_s} {latency_s}  {prov['algorithm'][:40]}")

    camera.close()
    print("\n  Every stage is a PipelineStage over the tested Phase 1 kernel.")
    print("  Nothing above is a placeholder.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
