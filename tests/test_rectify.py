"""
PerceptOS stereo rectification tests.

Rectification is verified against *known* geometry, not by inspection. Each
test builds a synthetic pair from an explicit camera model, so the correct
focal length, baseline and epipolar alignment are known in advance and the
decomposition can be checked against them.

The tests that matter most are the refusals:

* :meth:`TestCahvDecomposition.test_missing_model_refuses_rather_than_defaults`
* :meth:`TestRectifiedStatus.test_status_is_earned_by_measurement`

Both would pass trivially if the module were willing to supply a plausible
focal length or to assert success without measuring it.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from perceptos.core.cv_compat import HAS_OPENCV
from perceptos.core.provenance import Source
from perceptos.data.sources import StereoStatus

pytestmark = pytest.mark.skipif(not HAS_OPENCV, reason="rectification needs OpenCV")

if HAS_OPENCV:
    import cv2

    from perceptos.data.mars2020_raw import CameraModel
    from perceptos.stereo.rectify import (
        PinholeModel,
        RectificationError,
        cahv_to_pinhole,
        rectified_traced,
        rectify_pair,
    )

# A textbook Navcam-like rig: 1250 px focal length, 0.42 m baseline along Y,
# both cameras looking down +X. Every expected value below follows from these.
FOCAL = 1250.0
BASELINE = 0.42
WIDTH, HEIGHT = 320, 240


def make_model(cy_offset: float, *, focal: float = FOCAL,
               principal=(160.0, 120.0), toe_in_rad: float = 0.0) -> CameraModel:
    """Build a CAHV model with a known focal length, centre and pointing.

    ``H = cx*A + focal*Hn`` and ``V = cy*A + focal*Vn`` by construction, which
    is exactly the relation :func:`cahv_to_pinhole` inverts.
    """
    cx, cy = principal
    axis = np.array([math.cos(toe_in_rad), math.sin(toe_in_rad), 0.0])
    # Image +x runs along world -Y; image +y runs along world -Z.
    h_dir = np.array([-math.sin(toe_in_rad), math.cos(toe_in_rad), 0.0])
    v_dir = np.array([0.0, 0.0, -1.0])
    return CameraModel(
        model_type="CAHV",
        C=[0.8, cy_offset, -1.9],
        A=list(axis),
        H=list(cx * axis + focal * h_dir),
        V=list(cy * axis + focal * v_dir),
    )


def textured(width=WIDTH, height=HEIGHT, seed=0, shift=0):
    """A textured scene, optionally shifted horizontally to fake disparity."""
    rng = np.random.default_rng(seed)
    canvas = rng.integers(30, 90, (height, width + 64), dtype=np.uint8)
    for _ in range(90):
        x = int(rng.integers(8, width + 48))
        y = int(rng.integers(8, height - 8))
        s = int(rng.integers(3, 11))
        canvas[max(0, y - s):y + s, max(0, x - s):x + s] = int(rng.integers(140, 240))
    return canvas[:, shift:shift + width]


# ===========================================================================
# CAHV decomposition
# ===========================================================================

class TestCahvDecomposition:
    def test_recovers_the_known_focal_length(self):
        model = cahv_to_pinhole(make_model(0.0))
        assert model.fx == pytest.approx(FOCAL, abs=0.5)
        assert model.fy == pytest.approx(FOCAL, abs=0.5)

    def test_recovers_the_known_principal_point(self):
        model = cahv_to_pinhole(make_model(0.0, principal=(170.0, 130.0)))
        assert model.cx == pytest.approx(170.0, abs=0.5)
        assert model.cy == pytest.approx(130.0, abs=0.5)

    def test_recovers_the_camera_centre(self):
        model = cahv_to_pinhole(make_model(-0.21))
        assert model.C.tolist() == pytest.approx([0.8, -0.21, -1.9])

    def test_the_derived_basis_is_a_real_rotation(self):
        model = cahv_to_pinhole(make_model(0.0))
        assert model.orthonormality_error() < 1e-9
        assert abs(abs(np.linalg.det(model.R)) - 1.0) < 1e-9

    def test_orthonormality_residual_is_reported_not_hidden(self):
        """Published H and V are not exactly perpendicular; say by how much."""
        model = cahv_to_pinhole(make_model(0.0))
        assert "orthonormal_error" in model.to_dict()

    def test_missing_model_refuses_rather_than_defaults(self):
        """The central rule: no invented focal length, ever."""
        with pytest.raises(RectificationError, match="will not assume"):
            cahv_to_pinhole(None)

    def test_partial_model_is_refused(self):
        partial = CameraModel(model_type="CAHV", C=[0, 0, 0], A=[1, 0, 0])
        with pytest.raises(RectificationError):
            cahv_to_pinhole(partial)

    def test_degenerate_axis_is_refused(self):
        degenerate = CameraModel(model_type="CAHV", C=[0, 0, 0], A=[0, 0, 0],
                                 H=[1, 0, 0], V=[0, 1, 0])
        with pytest.raises(RectificationError, match="degenerate"):
            cahv_to_pinhole(degenerate)

    def test_collinear_h_and_a_is_refused(self):
        """H parallel to A gives a zero focal length, not a small one."""
        collinear = CameraModel(model_type="CAHV", C=[0, 0, 0], A=[1, 0, 0],
                                H=[5, 0, 0], V=[0, 1, 0])
        with pytest.raises(RectificationError, match="degenerate"):
            cahv_to_pinhole(collinear)


# ===========================================================================
# Rectification geometry
# ===========================================================================

class TestRectification:
    def _pair(self, **kwargs):
        left = textured(seed=4, shift=0)
        right = textured(seed=4, shift=8)          # 8 px of horizontal disparity
        return rectify_pair(left, right,
                            make_model(-BASELINE / 2),
                            make_model(+BASELINE / 2), **kwargs)

    def test_baseline_matches_the_camera_separation(self):
        """Measured from the two C vectors — not assumed."""
        result = self._pair()
        assert result.baseline_m == pytest.approx(BASELINE, rel=0.02)

    def test_focal_length_survives_rectification(self):
        result = self._pair()
        assert result.rig.fx == pytest.approx(FOCAL, rel=0.05)

    def test_output_images_keep_their_size(self):
        result = self._pair()
        assert result.left.shape[:2] == (HEIGHT, WIDTH)
        assert result.right.shape[:2] == (HEIGHT, WIDTH)

    def test_the_reprojection_matrix_is_consistent_with_the_rig(self):
        """Q[3,2] = -1/baseline; if these disagree, depth would be wrong."""
        result = self._pair()
        assert result.Q.shape == (4, 4)
        implied = abs(1.0 / result.Q[3, 2]) if result.Q[3, 2] else None
        assert implied == pytest.approx(result.baseline_m, rel=0.02)

    def test_mismatched_image_sizes_are_refused(self):
        with pytest.raises(RectificationError, match="share a size"):
            rectify_pair(np.zeros((10, 10), np.uint8), np.zeros((20, 20), np.uint8),
                         make_model(-0.2), make_model(0.2))

    def test_coincident_cameras_are_not_a_stereo_pair(self):
        with pytest.raises(RectificationError, match="not a stereo pair"):
            rectify_pair(textured(), textured(),
                         make_model(0.0), make_model(0.0))

    def test_distortion_is_never_claimed_to_be_corrected(self):
        """CAHVOR's O/R terms are not OpenCV's k1..k3; approximating would lie."""
        result = self._pair()
        assert result.distortion_corrected is False
        assert result.to_dict()["distortion_corrected"] is False

    def test_max_reliable_range_is_derived_from_the_real_rig(self):
        result = self._pair()
        expected = 0.05 * result.rig.fx * result.baseline_m / result.rig.disparity_sigma_px
        assert result.rig.max_reliable_range() == pytest.approx(expected, rel=1e-6)


# ===========================================================================
# Status is earned, not assigned
# ===========================================================================

class TestRectifiedStatus:
    def test_status_is_earned_by_measurement(self):
        """Rectification claims nothing it has not verified."""
        left = textured(seed=7, shift=0)
        right = textured(seed=7, shift=6)
        result = rectify_pair(left, right, make_model(-BASELINE / 2),
                              make_model(+BASELINE / 2))
        if result.epipolar_error_px is None:
            assert result.status is StereoStatus.VALID_STEREO
            assert result.depth_valid is False
            assert any("could not be verified" in n for n in result.notes)
        else:
            assert result.epipolar_error_px <= 2.0
            assert result.status is StereoStatus.RECTIFIED_STEREO
            assert result.depth_valid is True

    def test_a_verified_pair_is_the_only_route_to_depth(self):
        for status in StereoStatus:
            expected = status in (StereoStatus.RECTIFIED_STEREO,
                                  StereoStatus.SIMULATION)
            assert status.depth_valid is expected

    def test_skipping_the_measurement_withholds_the_claim(self):
        """Without a residual there is no evidence, so no RECTIFIED_STEREO."""
        result = rectify_pair(textured(seed=2), textured(seed=2, shift=5),
                              make_model(-BASELINE / 2), make_model(+BASELINE / 2),
                              measure_residual=False)
        assert result.status is StereoStatus.VALID_STEREO
        assert result.depth_valid is False
        assert result.epipolar_error_px is None

    def test_a_toed_in_rig_still_rectifies(self):
        """Real Navcams are toed in; rectification exists to remove that."""
        result = rectify_pair(
            textured(seed=9), textured(seed=9, shift=7),
            make_model(-BASELINE / 2, toe_in_rad=0.0),
            make_model(+BASELINE / 2, toe_in_rad=math.radians(1.5)),
        )
        assert result.baseline_m == pytest.approx(BASELINE, rel=0.05)
        assert result.left.shape == result.right.shape


# ===========================================================================
# Provenance
# ===========================================================================

class TestRectificationProvenance:
    def test_traced_result_carries_the_residual_as_uncertainty(self):
        result = rectify_pair(textured(seed=3), textured(seed=3, shift=6),
                              make_model(-BASELINE / 2), make_model(+BASELINE / 2))
        value = rectified_traced(result)
        assert value.source is Source.CALCULATED
        assert 0.0 < value.confidence <= 0.95
        assert value.provenance.uncertainty == result.epipolar_error_px
        assert value.provenance.parameters["distortion_corrected"] is False

    def test_confidence_never_reaches_one(self):
        """The camera model is published data, not a calibration we performed."""
        result = rectify_pair(textured(seed=5), textured(seed=5, shift=4),
                              make_model(-BASELINE / 2), make_model(+BASELINE / 2))
        assert rectified_traced(result).confidence < 1.0

    def test_an_unverified_rectification_has_lower_confidence(self):
        verified = rectified_traced(
            rectify_pair(textured(seed=6), textured(seed=6, shift=5),
                         make_model(-BASELINE / 2), make_model(+BASELINE / 2)))
        unverified = rectified_traced(
            rectify_pair(textured(seed=6), textured(seed=6, shift=5),
                         make_model(-BASELINE / 2), make_model(+BASELINE / 2),
                         measure_residual=False))
        assert unverified.confidence <= verified.confidence


# ===========================================================================
# Full pipeline: rectified pair -> disparity -> metric depth
# ===========================================================================

class TestRectifiedStereoPipeline:
    def _source(self, tmp_path, shift=16, *, calibrated=True):
        from perceptos.sensors.stereo_source import RectifiedStereoSource

        tmp_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tmp_path / "L.png"),
                    cv2.cvtColor(textured(seed=11, shift=0), cv2.COLOR_GRAY2BGR))
        cv2.imwrite(str(tmp_path / "R.png"),
                    cv2.cvtColor(textured(seed=11, shift=shift), cv2.COLOR_GRAY2BGR))
        left_cam = make_model(-BASELINE / 2) if calibrated else None
        right_cam = make_model(+BASELINE / 2) if calibrated else None
        return RectifiedStereoSource(
            [(tmp_path / "L.png", tmp_path / "R.png", left_cam, right_cam,
              {"sol": 203, "credit": "NASA/JPL-Caltech"})],
            source_kind=Source.LIVE_CAMERA, loop=True,
        ).open()

    def _pipeline(self, source):
        from perceptos.core.pipeline import Pipeline
        from perceptos.sensors.stereo_source import RectifiedStereoStage
        from perceptos.stereo.matcher import DepthStage, StereoMatcher, StereoStage

        return Pipeline([
            RectifiedStereoStage(source),
            StereoStage(StereoMatcher(algorithm="SGBM", num_disparities=64,
                                      block_size=7)),
            DepthStage(),
        ], seed_keys=())

    def test_depth_recovers_a_known_range(self, tmp_path):
        """shift = f*B/Z, so the true depth is known before the test runs."""
        shift = 16
        expected = FOCAL * BASELINE / shift
        ctx, report = self._pipeline(self._source(tmp_path, shift)).process({})
        assert report.healthy
        stats = ctx.get("depth").value.statistics()
        assert stats["median_depth_m"] == pytest.approx(expected, rel=0.02)

    def test_a_nearer_scene_reads_nearer(self, tmp_path):
        near, far = 32, 16
        ctx_near, _ = self._pipeline(self._source(tmp_path / "a", near)).process({})
        ctx_far, _ = self._pipeline(self._source(tmp_path / "b", far)).process({})
        assert (ctx_near.get("depth").value.statistics()["median_depth_m"]
                < ctx_far.get("depth").value.statistics()["median_depth_m"])

    def test_the_measured_rig_beats_a_stale_constructor_rig(self, tmp_path):
        """A stale baseline silently rescales every depth in the frame."""
        from perceptos.stereo.depth import StereoRig
        from perceptos.stereo.matcher import DepthStage

        wrong = StereoRig(fx=600.0, fy=600.0, cx=160.0, cy=120.0,
                          baseline_m=0.10)
        stage = DepthStage(wrong)
        ctx, _ = self._pipeline(self._source(tmp_path, 16)).process({})
        assert stage.resolve_rig(ctx).baseline_m == pytest.approx(BASELINE, rel=0.02)

    def test_depth_without_any_rig_refuses_rather_than_assuming(self, tmp_path):
        from perceptos.core.stage import FrameContext
        from perceptos.stereo.matcher import DepthStage

        with pytest.raises(RuntimeError, match="will not assume"):
            DepthStage().resolve_rig(FrameContext(frame_id=1))

    def test_an_uncalibrated_pair_is_served_but_depth_is_withheld(self, tmp_path):
        """Images stay usable for display and features; only depth is refused."""
        source = self._source(tmp_path, 16, calibrated=False)
        frame = source.read()
        assert frame is not None
        assert frame.status is StereoStatus.CALIBRATION_UNAVAILABLE
        assert frame.depth_valid is False
        assert frame.left.shape == frame.right.shape
        assert "rectification_error" in frame.metadata

    def test_the_stage_marks_an_uncalibrated_context_as_unrectified(self, tmp_path):
        from perceptos.core.pipeline import Pipeline
        from perceptos.sensors.stereo_source import RectifiedStereoStage

        source = self._source(tmp_path, 16, calibrated=False)
        ctx, report = Pipeline([RectifiedStereoStage(source)],
                               seed_keys=()).process({})
        assert report.healthy
        assert ctx.metadata["stereo_unrectified"] is True
        assert ctx.get("stereo_rig") is None

    def test_depth_inherits_the_imagery_source_type(self, tmp_path):
        """Cached NASA imagery must not produce a blank badge downstream."""
        source = self._source(tmp_path, 16)
        source.source_status = "LOCAL_CACHED_DATA"
        ctx, _ = self._pipeline(source).process({})
        assert ctx.get("frame").provenance.source_type == "LOCAL_CACHED_DATA"
        assert ctx.get("depth").provenance.source_type == "LOCAL_CACHED_DATA"

    def test_depth_traces_back_to_the_source_imagery(self, tmp_path):
        ctx, _ = self._pipeline(self._source(tmp_path, 16)).process({})
        chain = ctx.lineage.explain(ctx.get("depth").id)
        labels = [n["label"] for n in chain["chain"]]
        assert "left_frame" in labels
        assert "disparity" in labels
        assert "depth_map" in labels
        assert chain["contains_simulation"] is False

    def test_a_simulated_pair_contaminates_its_depth(self, tmp_path):
        source = self._source(tmp_path, 16)
        source.source_kind = Source.SIMULATION
        ctx, _ = self._pipeline(source).process({})
        assert ctx.get("depth").source is Source.SIMULATION
        assert ctx.lineage.explain(ctx.get("depth").id)["contains_simulation"]

    def test_the_source_reports_how_many_pairs_rectified(self, tmp_path):
        source = self._source(tmp_path, 16)
        source.read()
        described = source.describe()
        assert described["served"] == 1
        assert described["rectified"] == 1

    def test_contradictory_labelling_is_rejected_at_construction(self, tmp_path):
        """SIMULATION frames are not archive data and cannot claim to be."""
        from perceptos.sensors.stereo_source import RectifiedStereoSource

        with pytest.raises(ValueError, match="contradictory labelling"):
            RectifiedStereoSource(
                [(tmp_path / "a.png", tmp_path / "b.png", None, None, {})],
                source_kind=Source.SIMULATION,
                source_status="NASA_PUBLIC_NO_KEY",
            )

    def test_a_simulated_source_is_relabelled_when_status_was_defaulted(self, tmp_path):
        from perceptos.sensors.stereo_source import RectifiedStereoSource

        source = RectifiedStereoSource(
            [(tmp_path / "a.png", tmp_path / "b.png", None, None, {})],
            source_kind=Source.SIMULATION,
        )
        assert source.source_status == "SIMULATION"

    def test_missing_image_files_are_caught_at_open(self, tmp_path):
        from perceptos.sensors.stereo_source import RectifiedStereoSource

        source = RectifiedStereoSource([(tmp_path / "nope.png", tmp_path / "no2.png",
                                         make_model(-0.2), make_model(0.2), {})])
        with pytest.raises(FileNotFoundError, match="missing"):
            source.open()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
