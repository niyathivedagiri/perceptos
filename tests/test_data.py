"""
PerceptOS data layer tests.

Every test here runs offline. The NASA clients take an injected
:class:`~perceptos.data.cache.HttpTransport`, so the API behaviour is tested
against canned responses matching the real schema, with no network, no API key
and no rate limit.

The stereo pairing tests use real Perseverance product IDs taken from NASA's
published archive, because the pairing rule is the piece most likely to be got
wrong and the one whose failure is least visible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from perceptos.core.cv_compat import HAS_OPENCV
from perceptos.core.environment import (
    AI4MARS_CLASSES,
    ENVIRONMENTS,
    DataSourceSpec,
    Environment,
    TerrainClass,
    get_environment,
    register_environment,
)
from perceptos.core.provenance import Source
from perceptos.data.cache import (
    DatasetCache,
    FakeTransport,
    HttpError,
    LICENCES,
)
from perceptos.data.mars2020 import (
    PERSEVERANCE_CAMERAS,
    Mars2020Client,
    ProductId,
    RoverImage,
)
from perceptos.data.terrain import (
    AI4MARS_UNLABELLED,
    NODATA_THRESHOLD,
    denoise_labels,
    fuse_costs,
    load_dem,
    load_dem_traced,
    semantic_cost_layer,
)
from perceptos.mars.traversability import TraversabilityAnalyser

# Real product IDs from NASA's published Perseverance archive.
NLF_203 = "NLF_0203_0684970679_051ECM_N0071716NCAM02203_10_195J"
NRF_203 = "NRF_0203_0684970679_051ECM_N0071716NCAM02203_10_195J"
NRF_020 = "NRF_0020_0668728815_613ECM_N0030770NCAM00305_04_195J"

BASE = "https://mars.nasa.gov/mars2020-raw-images/pub/ods/surface/sol/"


def photo(product_id: str, camera: str, sol: int, photo_id: int) -> dict:
    """A record shaped exactly like NASA's mars-photos API returns."""
    return {
        "id": photo_id,
        "sol": sol,
        "earth_date": "2021-09-15",
        "camera": {"id": 26, "name": camera,
                   "full_name": PERSEVERANCE_CAMERAS[camera][1], "rover_id": 8},
        "img_src": f"{BASE}{sol:05d}/ids/edr/browse/ncam/{product_id}.png",
        "rover": {"id": 8, "name": "Perseverance", "status": "active"},
    }


# ===========================================================================
# Product ID parsing — the stereo pairing key
# ===========================================================================

class TestProductId:
    def test_parses_a_real_navcam_product_id(self):
        parsed = ProductId.parse(NLF_203)
        assert parsed is not None
        assert parsed.camera_prefix == "NL"
        assert parsed.variant == "F"
        assert parsed.sol == 203
        assert parsed.sclk == 684970679
        assert parsed.eye == "left"

    def test_parses_from_a_full_url(self):
        url = f"{BASE}00203/ids/edr/browse/ncam/{NRF_203}.png"
        parsed = ProductId.parse(url)
        assert parsed is not None and parsed.eye == "right"

    def test_stereo_partners_share_the_spacecraft_clock(self):
        """This is the pairing rule. If it breaks, depth silently goes wrong."""
        left, right = ProductId.parse(NLF_203), ProductId.parse(NRF_203)
        assert left.stereo_key == right.stereo_key
        assert left.eye != right.eye

    def test_different_acquisitions_do_not_share_a_key(self):
        assert ProductId.parse(NLF_203).stereo_key != ProductId.parse(NRF_020).stereo_key

    def test_unrecognised_id_returns_none_rather_than_raising(self):
        """One odd filename must not abort a thousand-image fetch."""
        assert ProductId.parse("some_other_archive_naming.jpg") is None
        assert ProductId.parse("") is None


# ===========================================================================
# Cache and transport
# ===========================================================================

class TestDatasetCache:
    def _cache(self, tmp_path, payload=b"IMAGEBYTES"):
        transport = FakeTransport(byte_routes={"https://": payload})
        return DatasetCache(tmp_path, transport=transport), transport

    def test_download_records_checksum_and_licence(self, tmp_path):
        cache, _ = self._cache(tmp_path)
        asset = cache.fetch("https://example.test/a.png", dataset="d", name="a.png",
                            licence=LICENCES["nasa_pd"])
        assert asset.path.exists()
        assert len(asset.sha256) == 64
        assert asset.licence.credit == "NASA/JPL-Caltech"

    def test_second_fetch_is_a_cache_hit_not_a_download(self, tmp_path):
        cache, transport = self._cache(tmp_path)
        for _ in range(3):
            cache.fetch("https://example.test/a.png", dataset="d", name="a.png",
                        licence=LICENCES["nasa_pd"])
        assert cache.downloads == 1
        assert cache.hits == 2
        assert len(transport.calls) == 1

    def test_corrupted_cache_file_is_re_downloaded(self, tmp_path):
        """A truncated file from an interrupted download must not be trusted."""
        cache, _ = self._cache(tmp_path)
        asset = cache.fetch("https://example.test/a.png", dataset="d", name="a.png",
                            licence=LICENCES["nasa_pd"])
        asset.path.write_bytes(b"TRUNCATED")
        cache.fetch("https://example.test/a.png", dataset="d", name="a.png",
                    licence=LICENCES["nasa_pd"])
        assert cache.downloads == 2
        assert asset.path.read_bytes() == b"IMAGEBYTES"

    def test_checksum_mismatch_is_rejected_loudly(self, tmp_path):
        cache, _ = self._cache(tmp_path)
        with pytest.raises(HttpError, match="checksum mismatch"):
            cache.fetch("https://example.test/a.png", dataset="d", name="a.png",
                        licence=LICENCES["nasa_pd"], expected_sha256="0" * 64)

    def test_offline_mode_refuses_with_a_useful_message(self, tmp_path):
        cache = DatasetCache(tmp_path, transport=FakeTransport(), offline=True)
        with pytest.raises(HttpError, match="offline mode"):
            cache.fetch("https://example.test/a.png", dataset="d", name="a.png",
                        licence=LICENCES["nasa_pd"])

    def test_source_doc_is_written_automatically(self, tmp_path):
        cache, _ = self._cache(tmp_path)
        cache.fetch("https://example.test/a.png", dataset="d", name="a.png",
                    licence=LICENCES["ai4mars"])
        doc = (tmp_path / "d" / "SOURCE.md").read_text()
        assert "Licence" in doc and "Cite as" in doc
        assert "CVPRW 2021" in doc

    def test_corrupt_manifest_does_not_destroy_the_cache(self, tmp_path):
        cache, _ = self._cache(tmp_path)
        cache.fetch("https://example.test/a.png", dataset="d", name="a.png",
                    licence=LICENCES["nasa_pd"])
        cache.manifest_path("d").write_text("{ not json")
        assert cache.load_manifest("d") == {"dataset": "d", "assets": {}}

    def test_every_shipped_licence_declares_commercial_use(self):
        """A portfolio that may go commercial needs this answered up front."""
        for name, licence in LICENCES.items():
            assert licence.commercial_use is not None, f"{name} leaves it unstated"


# ===========================================================================
# NASA Mars 2020 client
# ===========================================================================

class TestMars2020Client:
    def _client(self, tmp_path, photos):
        transport = FakeTransport(
            json_routes={"https://api.nasa.gov": lambda params: {
                "photos": [p for p in photos
                           if params.get("camera") in (None, p["camera"]["name"])]
            }},
            byte_routes={BASE: b"PNGDATA"},
        )
        cache = DatasetCache(tmp_path, transport=transport)
        return Mars2020Client(cache, api_key="TEST_KEY"), transport

    def test_query_parses_the_documented_schema(self, tmp_path):
        client, _ = self._client(tmp_path, [photo(NLF_203, "NAVCAM_LEFT", 203, 1)])
        images = client.query(sol=203, camera="NAVCAM_LEFT")
        assert len(images) == 1
        assert images[0].sol == 203
        assert images[0].earth_date == "2021-09-15"
        assert images[0].product.sclk == 684970679

    def test_query_requires_sol_or_earth_date(self, tmp_path):
        client, _ = self._client(tmp_path, [])
        with pytest.raises(ValueError, match="sol or earth_date"):
            client.query()

    def test_unknown_camera_is_rejected_with_the_valid_list(self, tmp_path):
        client, _ = self._client(tmp_path, [])
        with pytest.raises(ValueError, match="unknown camera"):
            client.query(sol=203, camera="LEFT_EYEBALL")

    def test_stereo_matching_pairs_on_spacecraft_clock(self, tmp_path):
        client, _ = self._client(tmp_path, [])
        images = [
            RoverImage(1, 203, "d", "NAVCAM_LEFT", "L", "x", ProductId.parse(NLF_203)),
            RoverImage(2, 203, "d", "NAVCAM_RIGHT", "R", "y", ProductId.parse(NRF_203)),
        ]
        pairs = client.match_stereo(images)
        assert len(pairs) == 1
        assert pairs[0].sclk == 684970679
        assert pairs[0].camera_family == "navcam"

    def test_unmatched_frames_are_dropped_not_paired_by_proximity(self, tmp_path):
        """A wrongly matched pair is far worse than a missing one."""
        client, _ = self._client(tmp_path, [])
        images = [
            RoverImage(1, 203, "d", "NAVCAM_LEFT", "L", "x", ProductId.parse(NLF_203)),
            RoverImage(2, 20, "d", "NAVCAM_RIGHT", "R", "y", ProductId.parse(NRF_020)),
        ]
        assert client.match_stereo(images) == []

    def test_pairs_carry_a_camera_specific_baseline(self, tmp_path):
        client, _ = self._client(tmp_path, [])
        pairs = client.match_stereo([
            RoverImage(1, 203, "d", "NAVCAM_LEFT", "L", "x", ProductId.parse(NLF_203)),
            RoverImage(2, 203, "d", "NAVCAM_RIGHT", "R", "y", ProductId.parse(NRF_203)),
        ])
        assert pairs[0].baseline_m is not None and pairs[0].baseline_m > 0

    def test_pairs_are_flagged_unrectified_by_default(self, tmp_path):
        """Raw Navcam data is not rectified, and must not claim to be."""
        client, _ = self._client(tmp_path, [])
        pairs = client.match_stereo([
            RoverImage(1, 203, "d", "NAVCAM_LEFT", "L", "x", ProductId.parse(NLF_203)),
            RoverImage(2, 203, "d", "NAVCAM_RIGHT", "R", "y", ProductId.parse(NRF_203)),
        ])
        assert pairs[0].rectified is False

    def test_stereo_pairs_queries_both_eyes(self, tmp_path):
        client, transport = self._client(tmp_path, [
            photo(NLF_203, "NAVCAM_LEFT", 203, 1),
            photo(NRF_203, "NAVCAM_RIGHT", 203, 2),
        ])
        pairs = client.stereo_pairs(203, family="NAVCAM")
        assert len(pairs) == 1
        cameras = {c[1].get("camera") for c in transport.calls if c[1]}
        assert {"NAVCAM_LEFT", "NAVCAM_RIGHT"} <= cameras

    def test_non_stereo_family_is_rejected(self, tmp_path):
        client, _ = self._client(tmp_path, [])
        with pytest.raises(ValueError, match="not a stereo camera family"):
            client.stereo_pairs(203, family="SKYCAM")

    def test_download_stores_sol_and_credit_in_provenance(self, tmp_path):
        client, _ = self._client(tmp_path, [photo(NLF_203, "NAVCAM_LEFT", 203, 1)])
        image = client.query(sol=203, camera="NAVCAM_LEFT")[0]
        asset = client.download(image)
        params = asset.provenance_parameters()
        assert params["sol"] == 203
        assert params["credit"] == "NASA/JPL-Caltech"
        assert params["eye"] == "left"
        assert image.local_path is not None and image.local_path.exists()


# ===========================================================================
# Environments
# ===========================================================================

class TestEnvironments:
    def test_all_five_environments_are_registered(self):
        assert {"mars", "moon", "earth", "warehouse", "factory"} <= set(ENVIRONMENTS)

    def test_unknown_environment_lists_the_valid_names(self):
        with pytest.raises(KeyError, match="available"):
            get_environment("jupiter")

    def test_mars_is_remote_and_earth_is_not(self):
        assert get_environment("mars").is_remote()
        assert get_environment("mars").supervision == "autonomous"
        assert not get_environment("warehouse").is_remote()

    def test_gravity_does_not_change_the_geometric_slope_limit(self):
        """Static tip-over is geometry, not gravity. Getting this wrong is common."""
        mars, moon = get_environment("mars"), get_environment("moon")
        assert mars.gravity_ms2 != moon.gravity_ms2
        # Limits differ because the vehicles differ, not because g differs.
        moon_same_rover = moon.with_rover(mars.rover)
        assert moon_same_rover.rover.critical_slope_deg == mars.rover.critical_slope_deg

    def test_gravity_does_change_energy_cost(self):
        mars, earth = get_environment("mars"), get_environment("earth")
        assert (earth.energy_per_metre_wh(1000) > mars.energy_per_metre_wh(1000))

    def test_sand_is_costly_despite_being_geometrically_flat(self):
        """The whole reason semantic segmentation earns its place."""
        mars = get_environment("mars")
        sand = mars.terrain_class("sand")
        soil = mars.terrain_class("soil")
        assert sand.mobility_cost > soil.mobility_cost
        assert sand.slip_risk > 0.5

    def test_terrain_class_lookup_by_id_and_name_agree(self):
        mars = get_environment("mars")
        assert mars.terrain_class(2) is mars.terrain_class("sand")

    def test_unknown_terrain_class_lists_the_taxonomy(self):
        with pytest.raises(KeyError, match="available"):
            get_environment("mars").terrain_class("lava")

    def test_duplicate_class_ids_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate terrain class"):
            Environment(
                name="bad", display_name="Bad", gravity_ms2=9.8,
                rover=get_environment("earth").rover,
                terrain_classes=(TerrainClass(0, "a"), TerrainClass(0, "b")),
                data=DataSourceSpec(),
            )

    def test_cost_out_of_range_is_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            TerrainClass(0, "impossible", mobility_cost=1.5)

    def test_environments_declare_whether_real_data_exists(self):
        """Stating 'no public source' beats leaving it to be discovered."""
        mars = get_environment("mars").to_dict()
        moon = get_environment("moon").to_dict()
        assert mars["real_data_available"]["imagery"] is True
        assert mars["real_data_available"]["elevation"] is True
        assert moon["real_data_available"]["imagery"] is False
        assert "no large public surface-imagery" in moon["data"]["notes"].lower()

    def test_registering_a_duplicate_requires_overwrite(self):
        env = get_environment("earth")
        with pytest.raises(ValueError, match="already registered"):
            register_environment(env)
        register_environment(env, overwrite=True)   # explicit is fine


# ===========================================================================
# Terrain: real elevation and real labels
# ===========================================================================

class TestTerrainLoading:
    def _write_dem(self, tmp_path, array):
        path = tmp_path / "dem.npy"
        np.save(path, array)
        return path

    def test_nodata_sentinel_becomes_nan(self, tmp_path):
        """A raw -3.4e38 sentinel would read as the deepest canyon in the solar system."""
        dem = np.full((20, 20), 100.0)
        dem[5:8, 5:8] = -3.4028235e38
        tile = load_dem(self._write_dem(tmp_path, dem), cell_size_m=1.0)
        assert np.isnan(tile.elevation_m[6, 6])
        assert tile.coverage < 1.0
        assert np.nanmin(tile.elevation_m) > NODATA_THRESHOLD

    def test_missing_dem_points_at_the_fetcher(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="fetch_data.py"):
            load_dem(tmp_path / "absent.npy")

    def test_traced_dem_is_a_measurement_with_coverage_confidence(self, tmp_path):
        dem = np.full((20, 20), 50.0)
        dem[:5] = np.nan
        tv = load_dem_traced(self._write_dem(tmp_path, dem), cell_size_m=1.0)
        assert tv.source is Source.LIVE_SENSOR
        assert tv.confidence == pytest.approx(0.75, abs=0.02)
        assert tv.provenance.parameters["credit"] == "NASA/JPL-Caltech/USGS"

    def test_real_dem_flows_into_the_existing_analyser_unchanged(self, tmp_path):
        """The point of the whole data layer: no module changes."""
        rows, cols = 60, 60
        ramp = np.tile(np.linspace(0.0, 6.0, cols), (rows, 1))
        ramp[20:24, 20:24] += 0.9                       # a boulder
        ramp[40:45, 10:50] = -3.4028235e38              # a no-data gap
        tv = load_dem_traced(self._write_dem(tmp_path, ramp), cell_size_m=0.25)

        analyser = TraversabilityAnalyser(get_environment("mars").rover,
                                          cell_size_m=0.25, window_cells=3)
        result = analyser.analyse_traced(tv)
        assert result.value.hazard_mask[41:44, 20:40].all(), "no-data must be lethal"
        assert result.source is Source.CALCULATED
        assert result.confidence < 1.0

    def test_downsample_preserves_extent_and_is_nan_aware(self, tmp_path):
        dem = np.full((40, 40), 10.0)
        dem[0, 0] = np.nan
        tile = load_dem(self._write_dem(tmp_path, dem), cell_size_m=0.5)
        coarse = tile.downsample(4)
        assert coarse.shape == (10, 10)
        assert coarse.cell_size_m == pytest.approx(2.0)
        assert coarse.extent_m == pytest.approx(tile.extent_m)
        assert np.isfinite(coarse.elevation_m[0, 0]), "one NaN must not poison a block"

    def test_crop_clamps_rather_than_raising(self, tmp_path):
        tile = load_dem(self._write_dem(tmp_path, np.zeros((30, 30))), cell_size_m=1.0)
        assert tile.crop(25, 25, 100, 100).shape == (5, 5)


class TestSemanticCost:
    def test_flat_sand_costs_more_than_flat_soil(self):
        mars = get_environment("mars")
        labels = np.array([[0, 2], [1, 3]], dtype=np.uint8)   # soil sand bedrock rock
        cost = semantic_cost_layer(labels, mars)
        assert cost[0, 1] > cost[0, 0]
        assert cost[1, 1] == pytest.approx(1.0)               # big rock impassable

    def test_unlabelled_is_middling_not_free(self):
        """A model declining to classify is not evidence that driving is safe."""
        labels = np.full((4, 4), AI4MARS_UNLABELLED, dtype=np.uint8)
        cost = semantic_cost_layer(labels, get_environment("mars"))
        assert 0.3 < float(cost.mean()) < 0.7

    def test_fusion_takes_the_maximum_so_sand_survives_flat_geometry(self):
        geometric = np.zeros((4, 4), np.float32)              # perfectly flat
        semantic = np.full((4, 4), 0.75, np.float32)          # sand
        fused = fuse_costs(geometric, semantic)
        assert float(fused.min()) == pytest.approx(0.75), (
            "averaging would let smooth geometry cancel a sand warning"
        )

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="share a shape"):
            fuse_costs(np.zeros((4, 4)), np.zeros((4, 5)))


class TestLabelDenoising:
    """Segmentation speckle, inflated by the rover radius, blocks whole maps."""

    def _labels(self, seed=3):
        rng = np.random.default_rng(seed)
        labels = np.zeros((80, 80), dtype=np.uint8)      # all soil
        labels[30:38, 40:48] = 3                         # a genuine boulder
        labels[rng.random((80, 80)) < 0.02] = 3          # speckle
        return labels

    def test_isolated_lethal_pixels_are_removed(self):
        env = get_environment("mars")
        labels = self._labels()
        before = int((labels == 3).sum())
        after = int((denoise_labels(labels, env, min_blob_cells=3) == 3).sum())
        assert after < before

    def test_genuine_multi_cell_hazards_survive(self):
        env = get_environment("mars")
        cleaned = denoise_labels(self._labels(), env, min_blob_cells=3)
        core = cleaned[32:36, 42:46]
        assert (core == 3).all(), "a real boulder must not be filtered away"

    def test_removed_pixels_become_the_dominant_non_lethal_class(self):
        env = get_environment("mars")
        cleaned = denoise_labels(self._labels(), env, min_blob_cells=3)
        assert AI4MARS_UNLABELLED not in np.unique(cleaned)

    def test_only_lethal_classes_are_filtered(self):
        """Non-lethal misclassification shifts a cost slightly; it is not worth touching."""
        env = get_environment("mars")
        labels = np.zeros((40, 40), dtype=np.uint8)
        labels[10, 10] = 2                                # an isolated sand pixel
        cleaned = denoise_labels(labels, env, min_blob_cells=3)
        assert cleaned[10, 10] == 2

    def test_denoising_shortens_the_planned_route(self):
        """Each speckle pixel becomes a metre-wide no-go disc after inflation."""
        from perceptos.mars.planner import AStarPlanner

        env = get_environment("mars")
        labels = self._labels(seed=11)
        planner = AStarPlanner(inflation_radius_m=0.75, cost_weight=4.0)

        noisy = planner.plan(semantic_cost_layer(labels, env),
                             (2, 2), (77, 77), cell_size_m=0.25)
        clean = planner.plan(
            semantic_cost_layer(denoise_labels(labels, env, min_blob_cells=3), env),
            (2, 2), (77, 77), cell_size_m=0.25,
        )
        assert clean.length_m < noisy.length_m, (
            "speckle forces the planner into needless detours"
        )

    def test_dense_speckle_blocks_the_map_entirely(self):
        """At higher densities this stops being a detour and becomes a wall."""
        from perceptos.mars.planner import AStarPlanner, PlanningFailure

        env = get_environment("mars")
        rng = np.random.default_rng(11)
        labels = np.zeros((80, 80), dtype=np.uint8)
        labels[rng.random((80, 80)) < 0.05] = 3
        labels[0:6, 0:6] = 0                     # keep start and goal clear
        labels[74:80, 74:80] = 0
        planner = AStarPlanner(inflation_radius_m=0.75, cost_weight=4.0)

        with pytest.raises(PlanningFailure):
            planner.plan(semantic_cost_layer(labels, env), (2, 2), (77, 77),
                         cell_size_m=0.25)

        cleaned = denoise_labels(labels, env, min_blob_cells=3)
        plan = planner.plan(semantic_cost_layer(cleaned, env), (2, 2), (77, 77),
                            cell_size_m=0.25)
        assert plan.length_m > 0, "denoising must restore a traversable route"


# ===========================================================================
# Dataset frame sources
# ===========================================================================

@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
class TestDatasetFrameSource:
    def _images(self, tmp_path, n=3):
        import cv2
        paths = []
        for i in range(n):
            path = tmp_path / f"frame_{i:03d}.png"
            cv2.imwrite(str(path), np.full((32, 32, 3), 10 * (i + 1), np.uint8))
            paths.append(path)
        return paths

    def test_replays_frames_in_order_as_live_camera(self, tmp_path):
        from perceptos.sensors.camera import CameraManager
        from perceptos.sensors.dataset_source import DatasetFrameSource

        source = DatasetFrameSource(self._images(tmp_path))
        with CameraManager(source) as cam:
            frames = [cam.read() for _ in range(3)]
        assert all(f is not None for f in frames)
        assert frames[0].source is Source.LIVE_CAMERA, (
            "a recorded rover frame really was captured by a camera"
        )
        assert frames[0].value.mean() < frames[2].value.mean()

    def test_exhausts_cleanly_then_reports_a_read_failure(self, tmp_path):
        from perceptos.sensors.camera import CameraManager
        from perceptos.sensors.dataset_source import DatasetFrameSource

        with CameraManager(DatasetFrameSource(self._images(tmp_path, 2))) as cam:
            cam.read(); cam.read()
            assert cam.read() is None
        assert cam.read_failures == 1

    def test_missing_file_is_caught_at_open_not_mid_run(self, tmp_path):
        from perceptos.sensors.dataset_source import DatasetFrameSource
        source = DatasetFrameSource([tmp_path / "nope.png"])
        with pytest.raises(FileNotFoundError, match="missing"):
            source.open()

    def test_empty_directory_points_at_the_fetcher(self, tmp_path):
        from perceptos.sensors.dataset_source import DatasetFrameSource
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="fetch_data.py"):
            DatasetFrameSource.from_directory(tmp_path / "empty")

    def test_metadata_length_mismatch_is_rejected(self, tmp_path):
        from perceptos.sensors.dataset_source import DatasetFrameSource
        with pytest.raises(ValueError, match="does not match"):
            DatasetFrameSource(self._images(tmp_path, 3), metadata=[{}])

    def test_unrectified_pairs_are_refused_for_matching(self, tmp_path):
        """The single most dangerous silent failure available to this project."""
        from perceptos.sensors.dataset_source import (
            StereoDatasetSource,
            UnrectifiedPairError,
        )
        paths = self._images(tmp_path, 2)
        source = StereoDatasetSource([(paths[0], paths[1], {"sol": 203})],
                                     rectified=False).open()
        with pytest.raises(UnrectifiedPairError, match="rectif"):
            source.read_for_matching()

    def test_unrectified_pairs_are_still_served_for_other_uses(self, tmp_path):
        from perceptos.sensors.dataset_source import StereoDatasetSource
        paths = self._images(tmp_path, 2)
        source = StereoDatasetSource([(paths[0], paths[1], {"sol": 203})]).open()
        left, right, meta = source.read()
        assert left.shape == right.shape
        assert meta["rectified"] is False

    def test_rectified_pairs_pass_the_guard(self, tmp_path):
        from perceptos.sensors.dataset_source import StereoDatasetSource
        paths = self._images(tmp_path, 2)
        source = StereoDatasetSource([(paths[0], paths[1], {"sol": 203})],
                                     rectified=True).open()
        assert source.read_for_matching() is not None

    def test_stereo_acquisition_stage_marks_unrectified_context(self, tmp_path):
        from perceptos.core.pipeline import Pipeline
        from perceptos.sensors.dataset_source import (
            StereoAcquisitionStage,
            StereoDatasetSource,
        )
        paths = self._images(tmp_path, 2)
        source = StereoDatasetSource([(paths[0], paths[1], {"sol": 203})]).open()
        pipeline = Pipeline([StereoAcquisitionStage(source)], seed_keys=())
        ctx, report = pipeline.process({})
        assert report.healthy
        assert ctx.has("frame", "gray", "gray_right")
        assert ctx.metadata.get("stereo_unrectified") is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
