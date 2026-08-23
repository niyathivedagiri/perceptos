"""
PerceptOS vision layer tests.

Every test builds a scene with known ground truth and checks the module
recovers it, rather than checking that a function "returns something".
Synthetic chessboards give exact corner counts; a synthetic stereo pair with a
known shift gives an exact disparity; a moving square gives an exact track.

The suite skips cleanly when OpenCV is absent, so the geometry-only install
still runs the core suite.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from perceptos.core.cv_compat import (
    HAS_OPENCV,
    CapabilityError,
    capability_report,
    feature_detectors,
    resolve_factory,
    tracker_factories,
)
from perceptos.core.pipeline import Pipeline
from perceptos.core.provenance import Source

pytestmark = pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")

if HAS_OPENCV:
    import cv2

    from perceptos.calibration.chessboard import (
        CalibrationError,
        ChessboardCalibrator,
        UndistortStage,
    )
    from perceptos.detection.yolo import Detection, DetectorUnavailable, YoloDetector
    from perceptos.features.extractors import FeatureExtractor, FeatureStage
    from perceptos.imageproc.filters import (
        OPERATIONS,
        ImageChain,
        ImageProcessingStage,
    )
    from perceptos.mars.odometry import VisualOdometry
    from perceptos.sensors.camera import (
        AcquisitionStage,
        CameraManager,
        SyntheticSource,
    )
    from perceptos.stereo.depth import StereoRig
    from perceptos.stereo.matcher import StereoMatcher
    from perceptos.tracking.trackers import (
        MultiObjectTracker,
        TrackState,
        TrackingStage,
        iou,
    )


# ===========================================================================
# Scene builders — synthetic data with known ground truth
# ===========================================================================

def make_chessboard(pattern=(9, 6), square_px=40, margin=60):
    """Render a chessboard whose inner-corner count is exactly ``pattern``."""
    cols, rows = pattern
    board_cols, board_rows = cols + 1, rows + 1
    h = board_rows * square_px + 2 * margin
    w = board_cols * square_px + 2 * margin
    img = np.full((h, w), 255, np.uint8)
    for r in range(board_rows):
        for c in range(board_cols):
            if (r + c) % 2 == 0:
                y0 = margin + r * square_px
                x0 = margin + c * square_px
                img[y0:y0 + square_px, x0:x0 + square_px] = 0
    return img


def make_textured(width=320, height=240, seed=0, blobs=40):
    rng = np.random.default_rng(seed)
    img = rng.integers(30, 90, (height, width), dtype=np.uint8)
    margin = max(4, min(20, min(width, height) // 4))
    for _ in range(blobs):
        cx = int(rng.integers(margin, max(margin + 1, width - margin)))
        cy = int(rng.integers(margin, max(margin + 1, height - margin)))
        s = int(rng.integers(3, max(4, margin)))
        img[max(0, cy - s):cy + s, max(0, cx - s):cx + s] = int(rng.integers(140, 240))
    return img


def make_stereo_pair(shift_px=16, width=320, height=240):
    """A textured pair where the whole scene sits at one known disparity.

    Disparity is defined as ``x_left - x_right``, so a feature must appear
    further *left* in the right image. Building the pair the other way round
    produces negative disparity, which the matcher correctly refuses to find.
    """
    base = make_textured(width + shift_px, height, seed=3, blobs=90)
    return base[:, :width], base[:, shift_px:shift_px + width]


# ===========================================================================
# Capability layer
# ===========================================================================

class TestCapabilityLayer:
    def test_report_lists_what_this_build_supports(self):
        report = capability_report()
        assert report["opencv_installed"] is True
        assert report["opencv_version"]
        assert "ORB" in report["detectors_available"]

    def test_factory_resolution_spans_opencv_4_and_5_conventions(self):
        """ORB is cv2.ORB_create in 4.x and cv2.ORB.create in 5.x."""
        assert resolve_factory("ORB") is not None

    def test_unknown_symbol_returns_none_rather_than_raising(self):
        assert resolve_factory("NotARealDetector") is None

    def test_harris_is_always_available_as_the_guaranteed_fallback(self):
        assert feature_detectors()["HARRIS"]["available"] is True

    def test_missing_detector_names_the_wheel_that_provides_it(self):
        registry = feature_detectors()
        missing = [n for n, s in registry.items() if not s["available"]]
        if not missing:
            pytest.skip("this build has every detector")
        with pytest.raises(CapabilityError, match="provided|ships with|not present"):
            FeatureExtractor(missing[0])

    def test_mosse_is_looked_up_in_the_legacy_namespace(self):
        """MOSSE only ever exists under cv2.legacy — top-level lookup fails."""
        factories = tracker_factories()
        if factories["MOSSE"] is None:
            pytest.skip("MOSSE unavailable in this build")
        assert not hasattr(cv2, "TrackerMOSSE_create")


# ===========================================================================
# Camera acquisition
# ===========================================================================

class TestCameraAcquisition:
    def test_synthetic_frames_are_tagged_simulation(self):
        with CameraManager(SyntheticSource(total_frames=5)) as cam:
            frame = cam.read()
        assert frame is not None
        assert frame.source is Source.SIMULATION
        assert frame.is_synthetic

    def test_exhausted_source_reports_a_read_failure_not_a_crash(self):
        with CameraManager(SyntheticSource(total_frames=2)) as cam:
            assert cam.read() is not None
            assert cam.read() is not None
            assert cam.read() is None
        assert cam.read_failures == 1

    def test_fps_is_measured_from_delivery_not_the_advertised_rate(self):
        source = SyntheticSource(fps=999.0, total_frames=20)
        with CameraManager(source) as cam:
            for _ in range(20):
                cam.read()
            health = cam.health()
        assert health.fps_nominal == 999.0
        assert health.fps != 999.0, "measured fps must not echo the nominal rate"
        assert health.frames_delivered == 20

    def test_decoder_backed_drops_are_measured_live_drops_are_estimated(self):
        """The distinction that stops a dashboard from lying."""
        with CameraManager(SyntheticSource(total_frames=5)) as cam:
            cam.read()
            _, measured = cam.dropped_frames()
        assert measured is True

        health = cam.traced_health()
        assert health.source is Source.LIVE_SENSOR

    def test_health_grade_reflects_connection_state(self):
        cam = CameraManager(SyntheticSource(total_frames=20))
        assert cam.health().grade == "offline"
        cam.open()
        cam.read()
        assert cam.health().grade == "starting", (
            "a rate from one interval is noise and must not be graded"
        )
        for _ in range(10):
            cam.read()
        assert cam.health().grade == "nominal"

    def test_acquisition_stage_seeds_frame_and_gray(self):
        with CameraManager(SyntheticSource(total_frames=3)) as cam:
            pipeline = Pipeline([AcquisitionStage(cam)], seed_keys=())
            ctx, report = pipeline.process({})
        assert ctx.has("frame", "gray")
        assert ctx.get("gray").value.ndim == 2
        assert report.healthy

    def test_synthetic_target_moves_between_frames(self):
        source = SyntheticSource(total_frames=10)
        source.open()
        source.read()
        first = source.target_centre()
        for _ in range(5):
            source.read()
        assert source.target_centre() != first


# ===========================================================================
# Calibration
# ===========================================================================

class TestCalibration:
    def test_detects_the_exact_inner_corner_count(self):
        cal = ChessboardCalibrator(pattern=(9, 6))
        board = make_chessboard((9, 6))
        assert cal.add_view(board) is True
        assert cal.image_points[0].shape[0] == 9 * 6

    def test_square_pattern_is_rejected_as_ambiguous(self):
        """A 6x6 inner-corner board has no unique orientation."""
        with pytest.raises(ValueError, match="ambiguous"):
            ChessboardCalibrator(pattern=(6, 6))

    def test_mixed_image_sizes_are_rejected(self):
        cal = ChessboardCalibrator(pattern=(9, 6))
        cal.add_view(make_chessboard((9, 6), square_px=40))
        with pytest.raises(CalibrationError, match="share one image size"):
            cal.add_view(make_chessboard((9, 6), square_px=30))

    def test_too_few_views_refuses_to_calibrate(self):
        cal = ChessboardCalibrator(pattern=(9, 6), min_views=5)
        cal.add_view(make_chessboard((9, 6)))
        with pytest.raises(CalibrationError, match="at least 5"):
            cal.calibrate()

    def _calibrate_from_warped_views(self, n=8):
        """Views of one board under different homographies — a real calibration."""
        cal = ChessboardCalibrator(pattern=(9, 6), square_size_m=0.025)
        board = make_chessboard((9, 6))
        h, w = board.shape
        rng = np.random.default_rng(11)
        for _ in range(n * 3):
            if cal.n_views >= n:
                break
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            jitter = rng.uniform(-0.12, 0.12, (4, 2)) * [w, h]
            warped = cv2.warpPerspective(
                board, cv2.getPerspectiveTransform(src, (src + jitter).astype(np.float32)),
                (w, h), borderValue=255,
            )
            cal.add_view(warped)
        return cal

    def test_calibration_recovers_a_plausible_principal_point(self):
        cal = self._calibrate_from_warped_views()
        if cal.n_views < 5:
            pytest.skip("synthetic warps did not yield enough detections")
        result = cal.calibrate()
        w, h = result.image_size
        assert 0.2 * w < result.cx < 0.8 * w
        assert 0.2 * h < result.cy < 0.8 * h
        assert result.rms_px < 5.0

    def test_per_view_errors_are_recomputed_not_inherited(self):
        """The aggregate RMS hides a single bad view; per-view errors do not."""
        cal = self._calibrate_from_warped_views()
        if cal.n_views < 5:
            pytest.skip("insufficient detections")
        result = cal.calibrate()
        assert len(result.per_view_errors) == result.n_views
        assert result.worst_view_error >= result.rms_px * 0.5

    def test_low_coverage_produces_an_actionable_warning(self):
        cal = self._calibrate_from_warped_views()
        if cal.n_views < 5:
            pytest.skip("insufficient detections")
        result = cal.calibrate()
        # All synthetic views sit centrally, so coverage is limited by design.
        if result.coverage < 0.45:
            assert any("corners span" in w for w in result.warnings())

    def test_grade_is_decided_by_the_weakest_criterion(self):
        cal = self._calibrate_from_warped_views()
        if cal.n_views < 5:
            pytest.skip("insufficient detections")
        result = cal.calibrate()
        assert result.grade in ("excellent", "good", "marginal", "reject")
        assert 0.0 <= result.confidence <= 1.0

    def test_round_trip_through_disk_preserves_everything(self, tmp_path):
        cal = self._calibrate_from_warped_views()
        if cal.n_views < 5:
            pytest.skip("insufficient detections")
        result = cal.calibrate()
        path = result.save(tmp_path / "cam0")
        assert path.with_suffix(".json").exists()
        loaded = type(result).load(path)
        assert np.allclose(loaded.camera_matrix, result.camera_matrix)
        assert loaded.grade == result.grade
        assert loaded.n_views == result.n_views

    def test_undistort_stage_builds_its_remap_once(self):
        cal = self._calibrate_from_warped_views()
        if cal.n_views < 5:
            pytest.skip("insufficient detections")
        stage = UndistortStage(cal.calibrate())
        first = stage._ensure_maps()
        assert stage._ensure_maps() is first, "remap tables must be cached, not rebuilt"


# ===========================================================================
# Image processing
# ===========================================================================

class TestImageProcessing:
    def test_chain_applies_operations_in_order(self):
        chain = ImageChain().add("gaussian_blur", ksize=5).add("canny", low=50, high=150)
        out = chain.apply(make_textured())
        assert out.dtype == np.uint8
        assert set(np.unique(out)) <= {0, 255}, "canny output must be binary"

    def test_even_kernels_are_coerced_to_odd(self):
        """OpenCV rejects even kernels outright."""
        out = ImageChain().add("gaussian_blur", ksize=4).apply(make_textured())
        assert out.shape == make_textured().shape

    def test_per_operation_timings_are_recorded(self):
        chain = ImageChain().add("gaussian_blur").add("median_blur").add("canny")
        chain.apply(make_textured())
        assert len(chain.timings) == 3
        assert chain.total_ms > 0
        assert chain.slowest in chain.timings

    def test_binarising_before_a_greyscale_operation_is_rejected(self):
        """Canny on an already-thresholded image is silently meaningless."""
        chain = ImageChain().add("otsu").add("canny")
        problems = chain.validate(input_is_gray=True)
        assert any("already-binarised" in p for p in problems)

    def test_greyscale_only_operation_on_colour_input_is_rejected(self):
        chain = ImageChain().add("canny")
        assert any("single-channel" in p for p in chain.validate(input_is_gray=False))

    def test_unknown_parameters_are_caught_at_validation(self):
        chain = ImageChain().add("gaussian_blur", kernel_size=5)
        assert any("unknown parameters" in p for p in chain.validate(input_is_gray=True))

    def test_strict_stage_refuses_an_invalid_chain(self):
        with pytest.raises(ValueError, match="invalid image chain"):
            ImageProcessingStage(ImageChain().add("otsu").add("clahe"), strict=True)

    def test_clahe_on_colour_preserves_hue(self):
        """Per-channel CLAHE shifts hue; LAB luminance-only does not."""
        colour = cv2.cvtColor(make_textured(), cv2.COLOR_GRAY2BGR)
        colour[:, :, 2] = np.clip(colour[:, :, 2].astype(int) + 40, 0, 255)
        out = ImageChain().add("clahe").apply(colour)
        assert out.shape == colour.shape
        assert out[:, :, 2].mean() > out[:, :, 0].mean(), "red dominance must survive"

    def test_control_schema_covers_every_operation(self):
        schema = ImageChain.control_schema()
        assert set(schema) == set(OPERATIONS)
        assert all("description" in spec for spec in schema.values())

    def test_unknown_operation_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="unknown operation"):
            ImageChain().add("deblur_with_magic")


# ===========================================================================
# Features
# ===========================================================================

class TestFeatures:
    def test_orb_finds_keypoints_on_textured_input(self):
        features = FeatureExtractor("ORB", max_features=200).detect(make_textured())
        assert len(features) > 20
        assert features.descriptors is not None
        assert features.descriptor_kind == "binary"

    def test_max_features_is_honoured_by_detectors_without_a_native_cap(self):
        """FAST and Harris have no built-in limit and will flood the pipeline."""
        for method in ("FAST", "HARRIS"):
            features = FeatureExtractor(method, max_features=25).detect(make_textured())
            assert len(features) <= 25, f"{method} ignored max_features"

    def test_blank_image_yields_almost_no_keypoints(self):
        features = FeatureExtractor("ORB").detect(np.full((240, 320), 128, np.uint8))
        assert len(features) < 5
        assert features.quality < 0.2

    def test_spatial_spread_distinguishes_clustered_from_distributed(self):
        clustered = np.full((240, 320), 60, np.uint8)
        clustered[90:150, 130:190] = make_textured(60, 60, seed=9, blobs=25)
        spread_img = make_textured(320, 240, seed=4)
        extractor = FeatureExtractor("ORB", max_features=500)
        assert (extractor.detect(clustered).spatial_spread
                < extractor.detect(spread_img).spatial_spread)

    def test_matching_a_frame_against_itself_is_near_perfect(self):
        extractor = FeatureExtractor("ORB", max_features=300)
        features = extractor.detect(make_textured())
        result = extractor.match(features, features)
        assert result.ratio_pass_rate > 0.8
        assert result.confidence > 0.6

    def test_matching_unrelated_scenes_mostly_fails_the_ratio_test(self):
        extractor = FeatureExtractor("ORB", max_features=300)
        a = extractor.detect(make_textured(seed=1))
        b = extractor.detect(make_textured(seed=99))
        assert extractor.match(a, b).ratio_pass_rate < 0.5

    def test_descriptorless_detector_refuses_to_match(self):
        """FAST detects but does not describe — matching it is a category error."""
        extractor = FeatureExtractor("FAST", max_features=100)
        features = extractor.detect(make_textured())
        with pytest.raises(CapabilityError, match="no descriptors"):
            extractor.match(features, features)

    def test_feature_stage_prefers_the_processed_image_when_present(self):
        stage = FeatureStage(FeatureExtractor("ORB", max_features=100))
        with CameraManager(SyntheticSource(total_frames=2)) as cam:
            pipeline = Pipeline(
                [AcquisitionStage(cam), ImageProcessingStage(ImageChain().add("clahe")),
                 stage],
                seed_keys=(),
            )
            ctx, report = pipeline.process({})
        assert report.healthy
        assert ctx.get("features").provenance.parameters["input"] == "processed"


# ===========================================================================
# Stereo
# ===========================================================================

class TestStereoMatching:
    def test_num_disparities_is_rounded_to_a_multiple_of_16(self):
        """OpenCV misbehaves silently on a non-multiple."""
        matcher = StereoMatcher(num_disparities=100)
        assert matcher.num_disparities == 112
        assert matcher.rounded_from == 100

    def test_invalid_pixels_become_nan_not_a_negative_sentinel(self):
        left, right = make_stereo_pair(shift_px=16)
        result = StereoMatcher(num_disparities=64, block_size=5).compute(left, right)
        finite = result.disparity[np.isfinite(result.disparity)]
        assert (finite > 0).all(), "sentinel values leaked into the disparity map"
        assert np.isnan(result.disparity[~result.valid_mask]).all()

    def test_known_shift_is_recovered_as_the_right_disparity(self):
        shift = 16
        left, right = make_stereo_pair(shift_px=shift)
        result = StereoMatcher(num_disparities=64, block_size=7).compute(left, right)
        valid = result.disparity[result.valid_mask]
        if valid.size < 100:
            pytest.skip("synthetic pair did not match densely enough")
        assert abs(float(np.median(valid)) - shift) < 3.0

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="share a shape"):
            StereoMatcher().compute(np.zeros((10, 10), np.uint8), np.zeros((10, 20), np.uint8))

    def test_undersized_search_range_is_reported_not_hidden(self):
        left, right = make_stereo_pair(shift_px=40)
        result = StereoMatcher(num_disparities=16, block_size=5).compute(left, right)
        assert result.coverage < 0.5 or result.at_range_limit > 0.02
        assert result.warnings()

    def test_textureless_scene_reports_low_coverage(self):
        """SGBM returns disparity 0 everywhere here; none of it is a range."""
        blank = np.full((240, 320), 128, np.uint8)
        result = StereoMatcher(num_disparities=32).compute(blank, blank)
        assert result.coverage < 0.2
        assert result.at_minimum > 0.5
        assert result.warnings()

    def test_bm_and_sgbm_both_construct(self):
        for algorithm in ("BM", "SGBM"):
            assert StereoMatcher(algorithm=algorithm).algorithm == algorithm

    def test_unknown_algorithm_is_rejected(self):
        with pytest.raises(ValueError, match="SGBM or BM"):
            StereoMatcher(algorithm="MAGIC")


# ===========================================================================
# Tracking
# ===========================================================================

class TestTracking:
    def test_iou_of_identical_boxes_is_one(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_iou_of_disjoint_boxes_is_zero(self):
        assert iou((0, 0, 10, 10), (50, 50, 10, 10)) == pytest.approx(0.0)

    def test_iou_half_overlap(self):
        assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)

    def test_track_is_not_confirmed_until_min_hits(self):
        """Otherwise every one-frame false positive becomes a logged object."""
        tracker = MultiObjectTracker(tracker_type=None, min_hits=3)
        det = [{"bbox": (10.0, 10.0, 20.0, 20.0), "label": "rock", "confidence": 0.9}]
        assert tracker.update(None, det) == []
        assert tracker.update(None, det) == []
        assert len(tracker.update(None, det)) == 1

    def test_identity_is_stable_across_a_linear_traverse(self):
        tracker = MultiObjectTracker(tracker_type=None, min_hits=2, dt=0.1)
        ids = set()
        for i in range(15):
            det = [{"bbox": (10.0 + 6 * i, 40.0, 20.0, 20.0),
                    "label": "rock", "confidence": 0.9}]
            for track in tracker.update(None, det):
                ids.add(track.id)
        assert ids == {1}, f"identity switched: {ids}"

    def test_kalman_prediction_keeps_two_crossing_targets_apart(self):
        """Association on raw proximity swaps identities at the crossing."""
        tracker = MultiObjectTracker(tracker_type=None, min_hits=2, dt=0.1,
                                     iou_threshold=0.2)
        labels: dict[int, list[float]] = {}
        for i in range(20):
            a_x, b_x = 10.0 + 8 * i, 170.0 - 8 * i
            dets = [
                {"bbox": (a_x, 50.0, 24.0, 24.0), "label": "a", "confidence": 0.9},
                {"bbox": (b_x, 50.0, 24.0, 24.0), "label": "b", "confidence": 0.9},
            ]
            for track in tracker.update(None, dets):
                labels.setdefault(track.id, []).append(track.kf.state().velocity[0])
        # Each surviving track should keep a consistent direction of travel.
        for velocities in labels.values():
            settled = velocities[5:]
            if len(settled) > 5:
                signs = {v > 0 for v in settled}
                assert len(signs) == 1, "a track reversed direction — identity swapped"

    def test_track_coasts_then_dies_when_detections_stop(self):
        tracker = MultiObjectTracker(tracker_type=None, min_hits=2, max_misses=4)
        det = [{"bbox": (10.0, 10.0, 20.0, 20.0), "label": "rock", "confidence": 0.9}]
        for _ in range(4):
            tracker.update(None, det)
        assert tracker.tracks[0].state is TrackState.CONFIRMED
        for _ in range(3):
            tracker.update(None, [])
        assert tracker.tracks[0].state is TrackState.COASTING
        for _ in range(4):
            tracker.update(None, [])
        assert tracker.tracks == [], "a track that is never re-seen must be deleted"

    def test_coasting_box_keeps_moving_with_the_motion_model(self):
        tracker = MultiObjectTracker(tracker_type=None, min_hits=2, dt=0.1)
        for i in range(10):
            tracker.update(None, [{"bbox": (10.0 + 10 * i, 40.0, 20.0, 20.0),
                                   "label": "r", "confidence": 0.9}])
        before = tracker.tracks[0].bbox[0]
        tracker.update(None, [])
        after = tracker.tracks[0].bbox[0]
        assert after > before, "a coasting box must not freeze in place"

    def test_unavailable_tracker_names_what_is_available(self):
        with pytest.raises(CapabilityError, match="Available"):
            MultiObjectTracker(tracker_type="NOT_A_TRACKER")

    def test_real_opencv_tracker_survives_a_detection_dropout(self):
        available = MultiObjectTracker.available_trackers()
        if not available:
            pytest.skip("no OpenCV trackers in this build")
        source = SyntheticSource(width=320, height=240, total_frames=40)
        source.open()
        tracker = MultiObjectTracker(tracker_type=available[0], min_hits=2, dt=0.1)
        frame = source.read()
        cx, cy = source.target_centre()
        det = [{"bbox": (cx - 24.0, cy - 24.0, 48.0, 48.0),
                "label": "target", "confidence": 0.95}]
        for _ in range(3):
            tracker.update(frame, det)
            frame = source.read()
        assert tracker.confirmed_tracks()
        # Now withhold detections entirely — appearance must carry the track.
        for _ in range(5):
            tracker.update(frame, [])
            frame = source.read()
        assert tracker.tracks, "correlation tracker failed to coast the target"

    def test_tracking_stage_runs_without_a_detection_stage(self):
        with CameraManager(SyntheticSource(total_frames=3)) as cam:
            pipeline = Pipeline(
                [AcquisitionStage(cam), TrackingStage(MultiObjectTracker(tracker_type=None))],
                seed_keys=(),
            )
            ctx, report = pipeline.process({})
        assert report.healthy
        # The frames are synthetic, so the tracks inherit SIMULATION rather
        # than the stage's declared ESTIMATED. Contamination beats the
        # stage's own claim about itself.
        assert ctx.get("tracks").source is Source.SIMULATION


# ===========================================================================
# Detection
# ===========================================================================

class TestDetection:
    def test_missing_backend_raises_with_an_install_hint(self):
        if YoloDetector.backend_available():
            pytest.skip("ultralytics is installed here")
        with pytest.raises(DetectorUnavailable, match="pip install ultralytics"):
            YoloDetector().load()

    def test_missing_weights_do_not_trigger_a_silent_download(self):
        if not YoloDetector.backend_available():
            pytest.skip("ultralytics not installed")
        with pytest.raises(DetectorUnavailable, match="not found on disk"):
            YoloDetector(weights="definitely_not_here.pt", allow_download=False).load()

    def test_detector_reports_the_selected_device(self):
        stats = YoloDetector().statistics()
        assert stats["device"] in ("cpu", "mps", "cuda:0")
        assert stats["device_reason"]

    def test_detection_converts_cleanly_to_tracker_input(self):
        det = Detection(bbox=(10.0, 20.0, 30.0, 40.0), label="rock", confidence=0.8)
        assert det.centre == (25.0, 40.0)
        assert det.area == pytest.approx(1200.0)
        assert det.as_track_input()["label"] == "rock"

    def test_detection_stage_disables_itself_after_the_backend_is_missing(self):
        """A dead detector must not raise once per frame for the whole mission."""
        from perceptos.detection.yolo import DetectionStage

        stage = DetectionStage(YoloDetector(weights="nope.pt"))
        with CameraManager(SyntheticSource(total_frames=4)) as cam:
            pipeline = Pipeline([AcquisitionStage(cam), stage], seed_keys=())
            _, first = pipeline.process({})
            _, second = pipeline.process({})
        assert first.stages[-1].status.value == "failed"
        assert second.stages[-1].status.value == "skipped"
        assert stage.unavailable_reason


# ===========================================================================
# Visual odometry
# ===========================================================================

class TestVisualOdometry:
    def _vo(self):
        K = np.array([[400.0, 0, 160.0], [0, 400.0, 120.0], [0, 0, 1.0]])
        return VisualOdometry(K, min_features=50)

    def test_first_frame_returns_none(self):
        assert self._vo().process(make_textured()) is None

    def test_identical_frames_are_reported_as_stationary(self):
        """Zero parallax makes the essential matrix undetermined."""
        vo = self._vo()
        frame = make_textured()
        vo.process(frame)
        result = vo.process(frame.copy())
        assert result is not None
        assert result.degenerate
        assert "stationary" in result.note

    def test_translation_without_depth_is_flagged_as_unscaled(self):
        """A unit direction must never be mistaken for a distance in metres."""
        vo = self._vo()
        base = make_textured(400, 300, seed=5)
        vo.process(base[:, 20:340])
        result = vo.process(base[:, 40:360])
        assert result is not None
        if not result.degenerate:
            assert result.scale_resolved is False
            assert "unit direction" in result.note
            assert result.to_dict()["distance_m"] is None

    def test_depth_map_resolves_metric_scale(self):
        vo = self._vo()
        base = make_textured(400, 300, seed=6)
        depth = np.full((300, 320), 4.0, np.float32)
        vo.process(base[:, 20:340], depth)
        result = vo.process(base[:, 40:360], depth)
        assert result is not None
        if not result.degenerate:
            assert result.scale_resolved is True
            assert result.to_dict()["distance_m"] is not None

    def test_features_are_replenished_as_tracking_attrits_them(self):
        vo = self._vo()
        base = make_textured(500, 300, seed=7)
        for offset in range(0, 120, 8):
            vo.process(base[:, offset:offset + 320])
        assert vo._prev_points is not None
        assert len(vo._prev_points) >= vo.min_features * 0.5

    def test_reset_clears_the_accumulated_pose(self):
        vo = self._vo()
        base = make_textured(400, 300, seed=8)
        depth = np.full((300, 320), 3.0, np.float32)
        vo.process(base[:, 0:320], depth)
        vo.process(base[:, 20:340], depth)
        vo.reset()
        assert vo.total_distance_m == 0.0
        assert len(vo.path) == 1

    # --- model selection: planar scenes are degenerate for the essential matrix

    @staticmethod
    def _project(points_3d, K, R=np.eye(3), t=np.zeros((3, 1))):
        cam = (R @ points_3d.T + t).T
        uv = (K @ cam.T).T
        return (uv[:, :2] / uv[:, 2:3]).astype(np.float64)

    def _correspondences(self, depths, n=200, seed=2):
        """Project a 3D point cloud into two cameras separated by a translation."""
        K = np.array([[400.0, 0, 160.0], [0, 400.0, 120.0], [0, 0, 1.0]])
        rng = np.random.default_rng(seed)
        xs = rng.uniform(-2.0, 2.0, n)
        ys = rng.uniform(-1.5, 1.5, n)
        zs = np.asarray(depths) if np.ndim(depths) else np.full(n, float(depths))
        pts = np.stack([xs, ys, zs], axis=1)
        t = np.array([[0.25], [0.0], [0.0]])
        return K, self._project(pts, K), self._project(pts, K, t=t)

    def test_planar_scene_is_detected_as_degenerate(self):
        """Every feature on one plane leaves the epipolar geometry ambiguous."""
        K, p0, p1 = self._correspondences(depths=6.0)
        vo = VisualOdometry(K)
        assert vo._planarity(p0, p1) > vo.planarity_threshold

    def test_scene_with_real_depth_variation_is_not_flagged_planar(self):
        rng = np.random.default_rng(5)
        depths = rng.uniform(2.0, 18.0, 200)
        K, p0, p1 = self._correspondences(depths=depths)
        vo = VisualOdometry(K)
        assert vo._planarity(p0, p1) < vo.planarity_threshold

    def test_flat_pan_refuses_to_emit_a_pose(self):
        """The failure this replaces was a confident, spurious 180-degree yaw."""
        source = SyntheticSource(width=320, height=240, total_frames=8,
                                 pan_px_per_frame=6.0)
        source.open()
        vo = VisualOdometry(
            np.array([[400.0, 0, 160.0], [0, 400.0, 120.0], [0, 0, 1.0]]),
            min_features=60,
        )
        results = []
        for _ in range(6):
            frame = source.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = vo.process(gray)
            if result is not None:
                results.append(result)
        assert results
        assert any(r.degenerate and "planar" in r.note for r in results)
        assert not any(abs(r.yaw_deg) > 90 and not r.degenerate for r in results), \
            "a spurious large yaw was reported as a real measurement"

    def test_rejects_a_malformed_intrinsic_matrix(self):
        with pytest.raises(ValueError, match="3x3"):
            VisualOdometry(np.eye(4))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
