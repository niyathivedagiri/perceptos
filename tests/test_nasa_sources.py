"""
PerceptOS NASA source, stereo-verification and console tests.

Every test runs offline against an injected fake transport carrying the JPL
feed's documented schema. Two of these tests exist specifically to prevent a
class of misrepresentation rather than a class of crash:

* :meth:`TestStereoVerification.test_missing_camera_model_refuses_to_guess`
* :meth:`TestProvenanceExtensions.test_simulation_cannot_claim_an_external_source`

Both would pass trivially if the code were willing to fill in a plausible
default. They fail if it ever starts doing so.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from fastapi.testclient import TestClient

from perceptos.api.app import create_app
from perceptos.api.service import PerceptOSService
from perceptos.core.provenance import (
    ProvenanceError,
    Provenance,
    Source,
    traced,
)
from perceptos.data.cache import DatasetCache, FakeTransport
from perceptos.data.mars2020_raw import (
    JPL_RAW_API,
    CameraModel,
    JplRawImageClient,
    RawImage,
)
from perceptos.data.sources import (
    NASA_SERVICES,
    DataSourceStatus,
    LabelProvenance,
    StereoStatus,
    credentials_report,
    resolve_key_status,
)
from perceptos.db.store import MissionStore

BASE = "https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/"

# A CAHVOR model shaped like JPL's published Navcam components. The two camera
# centres are 0.42 m apart in Y, which is the Navcam stereo separation.
LEFT_MODEL = {
    "camera_model_type": "CAHVOR",
    "camera_model_component_list": {
        "C": "(0.8,-0.21,-1.9)",
        "A": "(1.0,0.0,0.0)",
        "H": "(0.0,1250.0,0.0)",
        "V": "(0.0,0.0,1250.0)",
        "O": "(1.0,0.0,0.0)",
        "R": "(0.0,-0.15,0.02)",
    },
}
RIGHT_MODEL = {
    "camera_model_type": "CAHVOR",
    "camera_model_component_list": {
        "C": "(0.8,0.21,-1.9)",
        "A": "(1.0,0.0,0.0)",
        "H": "(0.0,1250.0,0.0)",
        "V": "(0.0,0.0,1250.0)",
    },
}


def feed_image(image_id, instrument, sol=203, sclk="684970679.123",
               camera=None, site=12, drive=44, sample="Full"):
    """One entry shaped like the JPL raw_images feed returns."""
    cam = dict(camera or {})
    cam["instrument"] = instrument
    return {
        "imageid": image_id,
        "sol": sol,
        "camera": cam,
        "attitude": "(0.4156,-0.0041,-0.0095,0.9095)",
        "site": site,
        "drive": drive,
        "sample_type": sample,
        "date_taken_utc": "2021-09-15T18:22:41Z",
        "date_taken_mars": "Sol-00203M15:47:06.123",
        "date_received": "2021-09-16T02:10:00Z",
        "extended": {"sclk": sclk, "dimension": "(1288,968)",
                     "subframeRect": "(1,1,1288,968)", "scaleFactor": "1"},
        "image_files": {
            "full_res": f"{BASE}{sol:05d}/ids/edr/browse/ncam/{image_id}.png",
            "medium": f"{BASE}{sol:05d}/ids/edr/browse/ncam/{image_id}_800.jpg",
        },
        "credit": "NASA/JPL-Caltech",
    }


def make_client(tmp_path, images):
    transport = FakeTransport(
        json_routes={JPL_RAW_API: lambda params: {"images": images,
                                                  "total_results": len(images)}},
        byte_routes={BASE: b"\x89PNG\r\n\x1a\nFAKE"},
    )
    return JplRawImageClient(DatasetCache(tmp_path, transport=transport)), transport


# ===========================================================================
# Data source vocabulary
# ===========================================================================

class TestDataSourceStatus:
    def test_simulation_is_never_classed_as_real(self):
        assert DataSourceStatus.SIMULATION.is_real is False
        assert DataSourceStatus.SIMULATION.is_live is False

    def test_cached_data_is_real_but_not_live(self):
        """The distinction that stops a cached frame being sold as current."""
        cached = DataSourceStatus.LOCAL_CACHED_DATA
        assert cached.is_real is True
        assert cached.is_live is False

    def test_unavailable_is_distinct_from_simulation(self):
        """A missing dataset must not be silently replaced by synthetic data."""
        assert DataSourceStatus.UNAVAILABLE is not DataSourceStatus.SIMULATION
        assert DataSourceStatus.UNAVAILABLE.is_real is False
        assert DataSourceStatus.UNAVAILABLE.severity == "error"

    def test_synthetic_and_missing_states_are_visually_loud(self):
        assert DataSourceStatus.SIMULATION.severity == "warn"
        assert DataSourceStatus.NASA_PUBLIC_NO_KEY.severity == "ok"

    def test_every_status_has_a_badge_and_severity(self):
        for status in DataSourceStatus:
            assert status.badge and status.severity in ("ok", "info", "warn", "error")


class TestNasaServiceRegistry:
    def test_the_primary_perseverance_source_needs_no_key(self):
        """The finding that makes the whole key question mostly moot."""
        assert NASA_SERVICES["jpl_raw_images"].requires_api_key is False
        assert NASA_SERVICES["jpl_raw_images"].env_var is None

    def test_the_fallback_source_does_need_a_key(self):
        service = NASA_SERVICES["mars_photos"]
        assert service.requires_api_key is True
        assert service.env_var == "NASA_API_KEY"
        assert "30 requests/hour" in service.rate_limit
        assert "1,000 requests/hour" in service.rate_limit

    def test_every_service_states_what_happens_without_a_key(self):
        for service in NASA_SERVICES.values():
            assert service.on_missing_key
            assert service.credit
            assert service.licence

    def test_missing_key_resolves_to_demo_not_to_an_error(self, monkeypatch):
        monkeypatch.delenv("NASA_API_KEY", raising=False)
        assert resolve_key_status() is DataSourceStatus.NASA_DEMO_KEY

    def test_a_blank_key_is_treated_as_demo_because_that_is_what_is_sent(self):
        assert resolve_key_status("   ") is DataSourceStatus.NASA_DEMO_KEY

    def test_a_real_key_is_recognised(self, monkeypatch):
        monkeypatch.setenv("NASA_API_KEY", "abc123realkey")
        assert resolve_key_status() is DataSourceStatus.NASA_PERSONAL_KEY

    def test_credentials_report_answers_the_key_question_directly(self):
        report = credentials_report("DEMO_KEY")
        assert report["primary_source_requires_key"] is False
        assert "jpl_raw_images" in report["services_without_key"]
        assert report["services_requiring_key"] == ["mars_photos"]


# ===========================================================================
# Camera models — the basis for honest stereo
# ===========================================================================

class TestCameraModel:
    def test_parses_a_cahvor_component_list(self):
        model = CameraModel.parse(LEFT_MODEL)
        assert model is not None
        assert model.model_type == "CAHVOR"
        assert model.is_usable
        assert model.C == [0.8, -0.21, -1.9]

    def test_focal_length_is_derived_from_h_and_a(self):
        """|H - (H·A)A| — real geometry, not a hardcoded constant."""
        model = CameraModel.parse(LEFT_MODEL)
        assert model.focal_length_px == pytest.approx(1250.0, abs=1.0)

    def test_baseline_is_the_distance_between_camera_centres(self):
        left = CameraModel.parse(LEFT_MODEL)
        right = CameraModel.parse(RIGHT_MODEL)
        assert left.baseline_to(right) == pytest.approx(0.42, abs=1e-6)

    def test_absent_model_returns_none_rather_than_a_default(self):
        assert CameraModel.parse({"instrument": "NAVCAM_LEFT"}) is None

    def test_partial_model_is_reported_unusable_not_completed(self):
        model = CameraModel.parse({"camera_model_type": "CAHV",
                                   "camera_model_component_list": {"C": "(1,2,3)"}})
        assert model is not None
        assert model.is_usable is False
        assert model.focal_length_px is None
        assert model.baseline_to(model) is not None  # C exists, so this is fine

    def test_distortion_is_never_claimed_to_be_applied(self):
        assert CameraModel.parse(LEFT_MODEL).to_dict()["distortion_applied"] is False


# ===========================================================================
# JPL client
# ===========================================================================

class TestJplRawImageClient:
    def test_query_sends_no_api_key(self, tmp_path):
        client, transport = make_client(tmp_path, [feed_image("NLF_0203_A", "NAVCAM_LEFT")])
        client.query(sol=203)
        _, params = transport.calls[0]
        assert "api_key" not in params
        assert params["feed"] == "raw_images"
        assert params["category"] == "mars2020"

    def test_parses_the_documented_schema(self, tmp_path):
        client, _ = make_client(tmp_path, [
            feed_image("NLF_0203_A", "NAVCAM_LEFT", camera=LEFT_MODEL)])
        image = client.query(sol=203)[0]
        assert image.sol == 203
        assert image.sclk == pytest.approx(684970679.123)
        assert image.instrument == "NAVCAM_LEFT"
        assert image.eye == "left"
        assert image.site == 12 and image.drive == 44
        assert image.has_calibration is True
        assert image.date_taken_utc.startswith("2021-09-15")

    def test_missing_fields_degrade_to_warnings_not_exceptions(self, tmp_path):
        """One odd entry must not abort a fetch of a thousand images."""
        bare = {"imageid": "X", "sol": 5,
                "image_files": {"full_res": f"{BASE}00005/x.png"}}
        client, _ = make_client(tmp_path, [bare])
        image = client.query(sol=5)[0]
        assert image.has_calibration is False
        assert any("camera model" in w for w in image.parse_warnings)
        assert any("spacecraft clock" in w for w in image.parse_warnings)

    def test_entry_without_an_image_url_is_dropped(self, tmp_path):
        client, _ = make_client(tmp_path, [{"imageid": "X", "sol": 5, "image_files": {}}])
        assert client.query(sol=5) == []

    def test_download_records_nasa_provenance(self, tmp_path):
        client, _ = make_client(tmp_path, [
            feed_image("NLF_0203_A", "NAVCAM_LEFT", camera=LEFT_MODEL)])
        image = client.query(sol=203)[0]
        asset = client.download(image)
        params = asset.provenance_parameters()
        assert params["credit"] == "NASA/JPL-Caltech"
        assert params["source_service"] == "jpl_raw_images"
        assert params["source_status"] == DataSourceStatus.NASA_PUBLIC_NO_KEY.value
        assert params["sol"] == 203
        assert asset.path.exists()

    def test_a_downloaded_image_reports_itself_as_cached(self, tmp_path):
        client, _ = make_client(tmp_path, [feed_image("NLF_0203_A", "NAVCAM_LEFT")])
        image = client.query(sol=203)[0]
        assert image.to_dict()["source_status"] == "NASA_PUBLIC_NO_KEY"
        client.download(image)
        assert image.to_dict()["source_status"] == "LOCAL_CACHED_DATA"


# ===========================================================================
# Stereo verification — the "do not fake stereo" requirement
# ===========================================================================

class TestStereoVerification:
    def _pair(self, tmp_path, **kwargs):
        left_cam = kwargs.pop("left_camera", LEFT_MODEL)
        right_cam = kwargs.pop("right_camera", RIGHT_MODEL)
        left_sclk = kwargs.pop("left_sclk", "684970679.123")
        right_sclk = kwargs.pop("right_sclk", "684970679.140")
        right_instrument = kwargs.pop("right_instrument", "NAVCAM_RIGHT")
        right_sol = kwargs.pop("right_sol", 203)
        images = [
            feed_image("NLF_0203_A", "NAVCAM_LEFT", sclk=left_sclk, camera=left_cam),
            feed_image("NRF_0203_A", right_instrument, sol=right_sol,
                       sclk=right_sclk, camera=right_cam, **kwargs),
        ]
        client, _ = make_client(tmp_path, images)
        return client, client.query()

    def test_a_good_pair_is_valid_stereo_with_a_measured_baseline(self, tmp_path):
        client, images = self._pair(tmp_path)
        pair = client.find_stereo_pairs(images)[0]
        assert pair.status is StereoStatus.VALID_STEREO
        assert pair.baseline_m == pytest.approx(0.42, abs=1e-6)
        assert pair.focal_length_px == pytest.approx(1250.0, abs=1.0)
        assert any("measured baseline" in e for e in pair.evidence)

    def test_even_a_good_pair_is_not_rectified_so_depth_is_not_valid(self, tmp_path):
        """The archive publishes raw frames. Rectification is a separate step."""
        client, images = self._pair(tmp_path)
        pair = client.find_stereo_pairs(images)[0]
        assert pair.rectified is False
        assert pair.depth_valid is False

    def test_missing_camera_model_refuses_to_guess(self, tmp_path):
        """The central requirement: no invented baseline, ever."""
        client, images = self._pair(tmp_path, left_camera=None, right_camera=None)
        pair = client.find_stereo_pairs(images)[0]
        assert pair.status is StereoStatus.CALIBRATION_UNAVAILABLE
        assert pair.baseline_m is None
        assert pair.focal_length_px is None
        assert any("will not be assumed" in c for c in pair.concerns)

    def test_calibration_unavailable_forbids_depth(self, tmp_path):
        client, images = self._pair(tmp_path, left_camera=None, right_camera=None)
        assert client.find_stereo_pairs(images)[0].depth_valid is False

    def test_non_simultaneous_frames_are_not_a_stereo_pair(self, tmp_path):
        """Two Navcam images an hour apart are two pointings, not a pair."""
        client, images = self._pair(tmp_path, right_sclk="684974279.000")
        pairs = client.find_stereo_pairs(images)
        # Clock rounding puts them in different buckets, so no pair forms at all.
        assert pairs == []

    def test_directly_verified_non_simultaneous_pair_is_rejected(self, tmp_path):
        """A perfect camera model does not rescue two frames an hour apart."""
        client, images = self._pair(tmp_path, right_sclk="684974279.000")
        pair = client.verify_stereo(images[0], images[1])
        assert pair.confidence == 0.0
        assert pair.depth_valid is False
        assert any("separate acquisitions" in c for c in pair.concerns)
        assert any("REJECTED as a stereo pair" in c for c in pair.concerns)

    def test_calibrated_but_non_simultaneous_scores_below_uncalibrated_simultaneous(
        self, tmp_path
    ):
        """Simultaneity is a precondition, not a factor to be traded away."""
        client_a, images_a = self._pair(tmp_path / "a", right_sclk="684974279.000")
        calibrated_apart = client_a.verify_stereo(images_a[0], images_a[1])
        client_b, images_b = self._pair(tmp_path / "b", left_camera=None,
                                        right_camera=None)
        uncalibrated_together = client_b.verify_stereo(images_b[0], images_b[1])
        assert calibrated_apart.confidence < uncalibrated_together.confidence

    def test_mismatched_instruments_are_rejected(self, tmp_path):
        client, images = self._pair(tmp_path, right_instrument="MCZ_RIGHT")
        pair = client.verify_stereo(images[0], images[1])
        assert pair.confidence == 0.0
        assert any("not a designated stereo pair" in c for c in pair.concerns)

    def test_different_sols_are_rejected(self, tmp_path):
        client, images = self._pair(tmp_path, right_sol=204)
        pair = client.verify_stereo(images[0], images[1])
        assert pair.confidence == 0.0
        assert any("different sols" in c for c in pair.concerns)

    def test_rover_movement_between_exposures_is_flagged(self, tmp_path):
        client, images = self._pair(tmp_path, site=13, drive=99)
        pair = client.verify_stereo(images[0], images[1])
        assert any("rover moved" in c for c in pair.concerns)

    def test_confidence_never_reaches_one(self, tmp_path):
        """Unrectified frames and third-party camera models leave real doubt."""
        client, images = self._pair(tmp_path)
        assert client.find_stereo_pairs(images)[0].confidence < 1.0

    def test_every_pair_records_the_evidence_for_its_verdict(self, tmp_path):
        client, images = self._pair(tmp_path)
        payload = client.find_stereo_pairs(images)[0].to_dict()
        for field in ("left_image_id", "right_image_id", "left_timestamp_utc",
                      "temporal_delta_s", "stereo_status", "baseline_m",
                      "stereo_confidence", "evidence", "concerns"):
            assert field in payload

    def test_only_rectified_or_simulated_states_permit_depth(self):
        for status in StereoStatus:
            expected = status in (StereoStatus.RECTIFIED_STEREO, StereoStatus.SIMULATION)
            assert status.depth_valid is expected


# ===========================================================================
# Label provenance
# ===========================================================================

class TestLabelProvenance:
    def test_ground_truth_is_distinct_from_model_prediction(self):
        """Showing AI4Mars masks as segmentation output would misrepresent them."""
        assert LabelProvenance.GROUND_TRUTH_LABEL != LabelProvenance.MODEL_PREDICTION

    def test_every_label_state_has_a_severity(self):
        for provenance in LabelProvenance:
            assert provenance.severity in ("ok", "info", "warn", "error")


# ===========================================================================
# Provenance extensions
# ===========================================================================

class TestProvenanceExtensions:
    def test_new_fields_are_carried_and_serialised(self):
        value = traced(
            1.0, source=Source.LIVE_CAMERA, algorithm="JplRawImageClient",
            confidence=0.9, uncertainty=0.05, units="m",
            source_type="NASA_PUBLIC_NO_KEY", dataset="perseverance_jpl",
            source_url="https://mars.nasa.gov/rss/api/",
        )
        payload = value.provenance.to_dict()
        assert payload["source_type"] == "NASA_PUBLIC_NO_KEY"
        assert payload["dataset"] == "perseverance_jpl"
        assert payload["uncertainty"] == 0.05
        assert payload["source_url"].startswith("https://mars.nasa.gov")

    def test_simulation_cannot_claim_an_external_source(self):
        """A synthetic value must never be labelled as archive-derived."""
        with pytest.raises(ProvenanceError, match="cannot declare source_type"):
            Provenance(source=Source.SIMULATION, algorithm="synthetic",
                       source_type="NASA_PUBLIC_NO_KEY")

    def test_dataset_flows_downstream_without_being_restated(self):
        """A slope map from a Jezero DEM is still Jezero data."""
        dem = traced([1.0], source=Source.LIVE_SENSOR, algorithm="HiRISE DTM",
                     confidence=0.9, dataset="jezero_dem",
                     source_url="https://astrogeology.usgs.gov/x")
        slope = dem.derive([2.0], algorithm="slope")
        traversability = slope.derive([3.0], algorithm="traversability")
        assert traversability.provenance.dataset == "jezero_dem"

    def test_contamination_overrides_a_declared_external_source(self):
        synthetic = traced(1.0, source=Source.SIMULATION, algorithm="synthetic_dem")
        child = synthetic.derive(2.0, algorithm="slope",
                                 source_type="NASA_PUBLIC_NO_KEY")
        assert child.source is Source.SIMULATION
        assert child.provenance.source_type == "SIMULATION"

    def test_negative_uncertainty_is_rejected(self):
        with pytest.raises(ProvenanceError, match="uncertainty cannot be negative"):
            Provenance(source=Source.CALCULATED, algorithm="x", uncertainty=-1.0)

    def test_uncertainty_and_confidence_are_independent(self):
        """A depth can be well-trusted and still have a wide error bar."""
        value = traced(4.0, source=Source.CALCULATED, algorithm="stereo",
                       confidence=0.9, uncertainty=0.35, units="m")
        assert value.provenance.confidence == 0.9
        assert value.provenance.uncertainty == 0.35


# ===========================================================================
# API surface
# ===========================================================================

@pytest.fixture
def client():
    service = PerceptOSService(store=MissionStore(":memory:"))
    with TestClient(create_app(service)) as test_client:
        yield test_client


class TestRootAndConsole:
    def test_root_returns_json_for_a_non_browser_client(self, client):
        body = client.get("/").json()
        assert body["service"] == "PerceptOS"
        assert body["api_status"] == "ok"
        assert body["routes"]["count"] > 0
        assert body["routes"]["documentation"] == "/docs"
        assert "timestamp_utc" in body

    def test_root_returns_html_for_a_browser(self, client):
        response = client.get("/", headers={"accept": "text/html"})
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "PERCEPTOS" in response.text

    def test_root_reports_real_lineage_counts_not_placeholders(self, client):
        assert client.get("/").json()["lineage"]["status"] == "empty"

    def test_docs_and_openapi_still_work(self, client):
        assert client.get("/docs").status_code == 200
        schema = client.get("/openapi.json").json()
        assert schema["info"]["title"] == "PerceptOS"
        assert "/" in schema["paths"]

    def test_console_is_served_from_the_same_origin(self, client):
        response = client.get("/console")
        assert response.status_code == 200
        assert "DATA LINEAGE" in response.text
        assert "/api/lineage/" in response.text

    def test_console_is_excluded_from_the_openapi_schema(self, client):
        assert "/console" not in client.get("/openapi.json").json()["paths"]


class TestDataSourcesEndpoint:
    def test_reports_that_the_primary_source_needs_no_key(self, client):
        body = client.get("/api/datasources").json()
        assert body["primary_source_requires_key"] is False
        assert body["primary_perseverance_source"] == "jpl_raw_images"

    def test_lists_every_service_with_its_key_requirement(self, client):
        services = client.get("/api/datasources").json()["services"]
        assert services["jpl_raw_images"]["requires_api_key"] is False
        assert services["mars_photos"]["requires_api_key"] is True
        assert services["mars_photos"]["env_var"] == "NASA_API_KEY"

    def test_exposes_the_status_vocabulary_for_the_ui(self, client):
        body = client.get("/api/datasources").json()
        assert body["status_vocabulary"]["SIMULATION"]["severity"] == "warn"
        assert body["stereo_vocabulary"]["CALIBRATION_UNAVAILABLE"]["depth_valid"] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
