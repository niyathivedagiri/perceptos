"""
PerceptOS Phase 4 visualisation tests.

These cover the endpoints and rendering the mission-control panels consume.
The most important ones are not about pixels — they are about what the pixels
are allowed to claim:

* :meth:`TestRendering.test_no_data_renders_as_no_data` — a NaN cell must never
  be drawn as a measured value.
* :meth:`TestRendering.test_resizing_never_interpolates` — upscaling a hazard
  map must not invent cells on the safe/lethal boundary.
* :meth:`TestTerrainEndpoint.test_missing_dem_is_unavailable_not_simulation`
* :meth:`TestNavigationEndpoint.test_planning_without_terrain_is_refused`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from fastapi.testclient import TestClient

from perceptos.api.analysis import NavigationService, TerrainService
from perceptos.api.app import create_app
from perceptos.api.service import PerceptOSService
from perceptos.core.cv_compat import HAS_OPENCV
from perceptos.core.pipeline import Pipeline
from perceptos.core.provenance import Source
from perceptos.core.stage import FrameContext, PipelineStage
from perceptos.data.sources import DataSourceStatus
from perceptos.db.store import MissionStore

pytestmark = pytest.mark.skipif(not HAS_OPENCV, reason="rendering needs OpenCV")

if HAS_OPENCV:
    import cv2

    from perceptos.api.render import (
        NO_DATA_RGB,
        encode_png,
        render_array,
        render_keypoints,
    )


def decode(png: bytes) -> np.ndarray:
    """PNG bytes back to an RGB array, for asserting on actual pixels."""
    bgr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    return bgr[:, :, ::-1]


# ===========================================================================
# A tiny pipeline that produces renderable slots without a camera
# ===========================================================================

class SeedStage(PipelineStage):
    name = "acquisition"
    provides = ("frame", "gray")
    source = Source.SIMULATION

    def run(self, ctx: FrameContext) -> str:
        rng = np.random.default_rng(ctx.frame_id)
        frame = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
        self.emit(ctx, "frame", frame, algorithm="synthetic_seed",
                  source=Source.SIMULATION, confidence=1.0, units="px",
                  label="frame", artefact=frame)
        self.emit(ctx, "gray", frame[:, :, 0].copy(), algorithm="channel0",
                  source=Source.SIMULATION, units="px", label="gray")
        return "seeded"


def make_pipeline(environment):
    return Pipeline([SeedStage()], mode=environment.name, seed_keys=())


@pytest.fixture
def service(tmp_path):
    return PerceptOSService(store=MissionStore(":memory:"),
                            pipeline_factory=make_pipeline,
                            datasets_root=str(tmp_path / "datasets"))


@pytest.fixture
def client(service):
    with TestClient(create_app(service)) as test_client:
        yield test_client


# ===========================================================================
# Rendering
# ===========================================================================

class TestRendering:
    def test_no_data_renders_as_no_data(self):
        """Clamping NaN to zero would produce a complete-looking lie."""
        data = np.ones((10, 10), dtype=np.float64)
        data[3:6, 3:6] = np.nan
        result = render_array(data, colormap="viridis", max_dimension=10)
        rgb = decode(result.png)
        assert tuple(rgb[4, 4]) == NO_DATA_RGB
        assert tuple(rgb[0, 0]) != NO_DATA_RGB
        assert result.valid_fraction == pytest.approx(0.91, abs=0.01)

    def test_an_entirely_missing_field_says_so(self):
        result = render_array(np.full((8, 8), np.nan), max_dimension=8)
        assert result.valid_fraction == 0.0
        assert "no valid cells" in result.note
        assert np.all(decode(result.png) == np.array(NO_DATA_RGB))

    def test_the_scale_used_is_reported(self):
        """A colormapped image without a legend is decoration."""
        result = render_array(np.linspace(2.0, 9.0, 64).reshape(8, 8), units="m")
        assert result.vmin == pytest.approx(2.0)
        assert result.vmax == pytest.approx(9.0)
        assert result.headers()["X-PerceptOS-Units"] == "m"

    def test_a_constant_field_does_not_divide_by_zero(self):
        result = render_array(np.full((8, 8), 5.0))
        assert result.vmin == result.vmax == pytest.approx(5.0)
        assert result.valid_fraction == 1.0

    def test_resizing_never_interpolates(self):
        """Interpolation on a hazard map invents cells on the safe/lethal edge."""
        data = np.zeros((4, 4))
        data[2:, :] = 1.0
        result = render_array(data, colormap="hazard", max_dimension=40)
        rgb = decode(result.png)
        unique = {tuple(c) for c in rgb.reshape(-1, 3)}
        assert len(unique) == 2, f"nearest-neighbour must not blend: got {unique}"

    def test_boolean_masks_are_honoured(self):
        data = np.ones((6, 6))
        mask = np.zeros((6, 6), dtype=bool)
        mask[0, 0] = True
        result = render_array(data, mask=mask, max_dimension=6)
        assert tuple(decode(result.png)[0, 0]) == NO_DATA_RGB

    def test_colour_images_pass_through_unchanged_in_size(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        result = render_array(image, max_dimension=900)
        assert (result.width, result.height) == (30, 20)
        assert result.colormap == "rgb"

    def test_unknown_colormap_is_rejected(self):
        with pytest.raises(ValueError, match="unknown colormap"):
            render_array(np.zeros((4, 4)), colormap="rainbow_unicorn")

    def test_three_dimensional_non_image_arrays_are_rejected(self):
        with pytest.raises(ValueError, match="cannot render"):
            render_array(np.zeros((2, 2, 2, 2)))

    def test_keypoint_overlay_reports_the_real_count(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        points = np.array([[5.0, 5.0], [10.0, 20.0], [25.0, 8.0]])
        result = render_keypoints(image, points)
        assert "3 keypoints" in result.note


# ===========================================================================
# Frame endpoints
# ===========================================================================

class TestFrameEndpoints:
    def test_no_frame_yet_is_unavailable_with_a_command_to_run(self, client):
        body = client.get("/api/frame").json()
        assert body["available"] is False
        assert body["status"] == "UNAVAILABLE"
        assert "pipeline/run" in body["reason"]
        assert body["slots"] == []

    def test_rendering_a_slot_before_any_frame_is_404(self, client):
        response = client.get("/api/frame/gray.png")
        assert response.status_code == 404
        assert "UNAVAILABLE" in response.json()["detail"]

    def test_slots_appear_after_a_run(self, client):
        client.post("/api/pipeline/run", json={"frames": 2})
        body = client.get("/api/frame").json()
        assert body["available"] is True
        assert body["frame_id"] == 2
        keys = {s["key"] for s in body["slots"]}
        assert {"frame", "gray"} <= keys

    def test_every_slot_carries_its_provenance(self, client):
        client.post("/api/pipeline/run", json={"frames": 1})
        for slot in client.get("/api/frame").json()["slots"]:
            assert slot["node_id"]
            assert slot["source"] == "simulation"
            assert slot["source_type"] == "SIMULATION"

    def test_rendered_png_carries_audit_headers(self, client):
        client.post("/api/pipeline/run", json={"frames": 1})
        response = client.get("/api/frame/gray.png")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.headers["X-PerceptOS-Source"] == "simulation"
        assert response.headers["X-PerceptOS-Source-Type"] == "SIMULATION"
        assert "X-PerceptOS-Valid-Fraction" in response.headers
        assert response.headers["X-PerceptOS-Node-Id"]

    def test_a_rendered_node_id_resolves_in_the_lineage_api(self, client):
        """The image and the drawer must agree on which value is shown."""
        client.post("/api/pipeline/run", json={"frames": 1})
        node_id = client.get("/api/frame/gray.png").headers["X-PerceptOS-Node-Id"]
        payload = client.get(f"/api/lineage/{node_id}").json()
        assert payload["node"]["label"] == "gray"
        assert payload["depth"] >= 1

    def test_an_unknown_slot_is_404_not_a_blank_image(self, client):
        client.post("/api/pipeline/run", json={"frames": 1})
        response = client.get("/api/frame/lidar.png")
        assert response.status_code == 404
        assert "UNAVAILABLE" in response.json()["detail"]


class TestPointCloudEndpoint:
    def test_no_depth_means_no_point_cloud(self, client):
        """Geometry that was not measured is not reconstructed."""
        client.post("/api/pipeline/run", json={"frames": 1})
        response = client.get("/api/frame/pointcloud.png")
        assert response.status_code == 404
        assert "RECTIFIED_STEREO" in response.json()["detail"]
        assert "not reconstruct geometry it has not measured" in response.json()["detail"]

    def test_rendering_reports_the_real_point_count(self):
        from perceptos.api.render import render_point_cloud

        rng = np.random.default_rng(0)
        cloud = np.stack([rng.uniform(-2, 2, 5000), rng.uniform(-1, 1, 5000),
                          rng.uniform(3, 9, 5000)], axis=1)
        result = render_point_cloud(cloud, width=200, height=160)
        assert "5,000" in result.note
        assert result.vmin == pytest.approx(3.0, abs=0.1)
        assert result.vmax == pytest.approx(9.0, abs=0.1)
        assert result.units == "m"

    def test_subsampling_is_reported_not_hidden(self):
        from perceptos.api.render import render_point_cloud

        rng = np.random.default_rng(1)
        cloud = rng.uniform(1, 5, (40_000, 3))
        result = render_point_cloud(cloud, max_points=5_000, width=200, height=160)
        assert result.valid_fraction < 1.0
        assert "of 40,000" in result.note

    def test_an_empty_cloud_renders_as_no_data(self):
        from perceptos.api.render import NO_DATA_RGB, render_point_cloud

        result = render_point_cloud(np.full((10, 3), np.nan), width=40, height=40)
        assert result.valid_fraction == 0.0
        assert "nothing was reconstructed" in result.note
        assert np.all(decode(result.png) == np.array(NO_DATA_RGB))

    def test_a_malformed_cloud_is_rejected(self):
        from perceptos.api.render import render_point_cloud

        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            render_point_cloud(np.zeros((10, 2)))


class TestStereoEndpoint:
    def test_a_pipeline_without_a_right_eye_reports_unavailable(self, client):
        client.post("/api/pipeline/run", json={"frames": 1})
        body = client.get("/api/stereo").json()
        assert body["available"] is False
        assert body["stereo_status"] == "UNAVAILABLE"
        assert body["depth_valid"] is False

    def test_depth_is_never_valid_without_rectification(self, client):
        client.post("/api/pipeline/run", json={"frames": 1})
        assert client.get("/api/stereo").json()["depth_valid"] is False


# ===========================================================================
# Terrain
# ===========================================================================

class TestTerrainEndpoint:
    def test_missing_dem_is_unavailable_not_simulation(self, client):
        """The central availability rule, enforced at the endpoint."""
        body = client.get("/api/terrain").json()
        assert body["status"] == DataSourceStatus.UNAVAILABLE.value
        assert body["available"] is False
        assert "fetch_data.py jezero-dem" in body["reason"]
        assert body["products"] == []

    def test_terrain_layers_404_when_no_dem_exists(self, client):
        assert client.get("/api/terrain/slope.png").status_code == 404

    def test_simulation_requires_an_explicit_opt_in(self, client):
        assert client.get("/api/terrain").json()["available"] is False
        body = client.get("/api/terrain?allow_simulation=true").json()
        assert body["available"] is True
        assert body["status"] == "SIMULATION"

    def test_simulated_terrain_is_tagged_all_the_way_down(self, client):
        body = client.get("/api/terrain?allow_simulation=true").json()
        assert body["dem_provenance"]["source"] == "simulation"
        assert body["dem_provenance"]["source_type"] == "SIMULATION"

    def test_all_advertised_layers_actually_render(self, client):
        body = client.get("/api/terrain?allow_simulation=true").json()
        for product in body["products"]:
            response = client.get(
                f"/api/terrain/{product['key']}.png?allow_simulation=true")
            assert response.status_code == 200, product["key"]
            assert response.headers["content-type"] == "image/png"
            assert response.headers["X-PerceptOS-Source-Type"] == "SIMULATION"

    def test_ray_occupancy_is_reported_unavailable_rather_than_flat(self, client):
        """A DEM carries no ray evidence; a flat 0.5 map would look computed."""
        body = client.get("/api/terrain?allow_simulation=true").json()
        assert body["occupancy"]["ray_occupancy_status"] == "UNAVAILABLE"
        keys = {p["key"] for p in body["products"]}
        assert "coverage" in keys and "occupancy" not in keys

    def test_terrain_statistics_come_from_the_existing_analyser(self, client):
        stats = client.get("/api/terrain?allow_simulation=true").json()["statistics"]
        for field in ("traversable_fraction", "mean_slope_deg", "max_step_m",
                      "hazard_breakdown"):
            assert field in stats

    def test_a_real_dem_on_disk_is_reported_as_cached_real_data(self, tmp_path):
        """Presence of the file is the mode switch — not a config flag."""
        root = tmp_path / "datasets" / "jezero_dem"
        root.mkdir(parents=True)
        dem = np.tile(np.linspace(0.0, 3.0, 60), (60, 1))
        np.save(root / "tile.npy", dem)

        service = PerceptOSService(store=MissionStore(":memory:"),
                                   pipeline_factory=make_pipeline,
                                   datasets_root=str(tmp_path / "datasets"))
        with TestClient(create_app(service)) as client:
            body = client.get("/api/terrain").json()
            assert body["available"] is True
            assert body["status"] == DataSourceStatus.LOCAL_CACHED_DATA.value
            assert body["dem_provenance"]["source"] == "live_sensor"
            assert client.get("/api/terrain/slope.png").status_code == 200

    def test_a_dem_with_no_data_does_not_break_the_lineage_api(self, tmp_path):
        """NaN is not valid JSON. A real DEM is full of it.

        `float(nan)` serialises to the bare token `NaN`, which json.dumps emits
        and no strict parser accepts — every lineage request touching a real
        survey returned 500 before this was handled. Synthetic arrays are
        always finite, so only real data exposes it.
        """
        root = tmp_path / "datasets" / "jezero_dem"
        root.mkdir(parents=True)
        dem = np.tile(np.linspace(0.0, 3.0, 80), (80, 1))
        dem[20:40, 20:40] = -3.4028235e38          # HiRISE no-data sentinel
        np.save(root / "tile.npy", dem)

        service = PerceptOSService(store=MissionStore(":memory:"),
                                   pipeline_factory=make_pipeline,
                                   datasets_root=str(tmp_path / "datasets"))
        with TestClient(create_app(service)) as client:
            body = client.get("/api/terrain").json()
            assert body["available"] is True
            response = client.get(f"/api/lineage/{body['dem_node_id']}")
            assert response.status_code == 200, "NaN broke JSON serialisation"
            summary = response.json()["node"]["value"]
            assert summary["min"] is not None
            assert np.isfinite(summary["min"]), "sentinel leaked into the summary"

    def test_a_real_dem_reports_cached_not_blank_provenance(self, tmp_path):
        root = tmp_path / "datasets" / "jezero_dem"
        root.mkdir(parents=True)
        np.save(root / "tile.npy", np.tile(np.linspace(0.0, 2.0, 60), (60, 1)))
        service = PerceptOSService(store=MissionStore(":memory:"),
                                   pipeline_factory=make_pipeline,
                                   datasets_root=str(tmp_path / "datasets"))
        with TestClient(create_app(service)) as client:
            prov = client.get("/api/terrain").json()["dem_provenance"]
            assert prov["source_type"] == "LOCAL_CACHED_DATA"
            assert prov["dataset"] == "jezero_dem"

    def test_terrain_nodes_are_traceable_through_the_lineage_api(self, client):
        body = client.get("/api/terrain?allow_simulation=true").json()
        payload = client.get(f"/api/lineage/{body['traversability_node_id']}").json()
        assert payload["origin"] == "terrain"
        assert any("Traversability" in a for a in payload["algorithms"])
        assert payload["contains_simulation"] is True


# ===========================================================================
# Navigation
# ===========================================================================

class TestTerrain3DViews:
    def test_literal_routes_are_not_shadowed_by_the_parameterised_one(self, client):
        """Starlette matches in declaration order; this bit me twice."""
        for path in ("hillshade", "view3d"):
            response = client.get(f"/api/terrain/{path}.png?allow_simulation=true")
            assert response.status_code == 200, path
            assert response.headers["content-type"] == "image/png"

    def test_both_views_404_without_a_dem(self, client):
        assert client.get("/api/terrain/hillshade.png").status_code == 404
        assert client.get("/api/terrain/view3d.png").status_code == 404

    def test_hillshade_reports_its_illumination(self):
        from perceptos.api.render import render_hillshade

        dem = np.tile(np.linspace(0, 5, 40), (40, 1))
        result = render_hillshade(dem, cell_size_m=0.25, azimuth_deg=270.0)
        assert "270 deg" in result.note
        assert result.valid_fraction == 1.0

    def test_hillshade_marks_no_data_cells(self):
        from perceptos.api.render import NO_DATA_RGB, render_hillshade

        dem = np.tile(np.linspace(0, 5, 40), (40, 1))
        dem[10:20, 10:20] = np.nan
        result = render_hillshade(dem, cell_size_m=0.25, max_dimension=40)
        assert tuple(decode(result.png)[15, 15]) == NO_DATA_RGB
        assert result.valid_fraction < 1.0

    def test_flat_and_sloped_terrain_shade_differently(self):
        """If they matched, the hillshade would be conveying nothing."""
        from perceptos.api.render import render_hillshade

        flat = render_hillshade(np.zeros((40, 40)), cell_size_m=0.25)
        ramp = render_hillshade(np.tile(np.linspace(0, 20, 40), (40, 1)),
                                cell_size_m=0.25)
        assert not np.array_equal(decode(flat.png), decode(ramp.png))

    def test_vertical_exaggeration_is_reported_as_a_display_choice(self, client):
        response = client.get("/api/terrain/view3d.png"
                              "?allow_simulation=true&vertical_exaggeration=5")
        assert response.status_code == 200
        assert response.headers["X-PerceptOS-Vertical-Exaggeration"] == "5.0"
        assert "exaggeration 5x" in response.headers["X-PerceptOS-Note"]

    def test_the_3d_view_reports_its_decimation(self, client):
        note = client.get("/api/terrain/view3d.png?allow_simulation=true"
                          ).headers["X-PerceptOS-Note"]
        assert "quads" in note and "decimated" in note

    def test_an_unknown_drape_layer_is_404(self, client):
        response = client.get("/api/terrain/view3d.png"
                              "?allow_simulation=true&drape=magnetism")
        assert response.status_code == 404
        assert "UNAVAILABLE" in response.json()["detail"]

    def test_the_planned_route_appears_in_the_3d_view(self, client):
        client.post("/api/navigation/plan?allow_simulation=true", json={})
        note = client.get("/api/terrain/view3d.png?allow_simulation=true"
                          ).headers["X-PerceptOS-Note"]
        assert "path points" in note

    def test_a_grid_too_small_to_project_is_rejected(self):
        from perceptos.api.render import render_terrain_3d

        with pytest.raises(ValueError, match="at least a 2x2"):
            render_terrain_3d(np.zeros((1, 1)))

    def test_trajectory_plot_marks_loop_closures(self):
        from perceptos.api.render import render_trajectory

        poses = [{"x": float(i), "y": 0.0} for i in range(12)]
        without = render_trajectory(poses)
        with_loop = render_trajectory(poses, loop_closures=[{"from": 0, "to": 11}])
        assert "0 loop closure" in without.note
        assert "1 loop closure" in with_loop.note
        assert not np.array_equal(decode(without.png), decode(with_loop.png))

    def test_an_empty_trajectory_says_so(self):
        from perceptos.api.render import render_trajectory

        result = render_trajectory([])
        assert result.valid_fraction == 0.0
        assert "no poses" in result.note


class TestNavigationEndpoint:
    def test_no_plan_yet_is_unavailable(self, client):
        body = client.get("/api/navigation").json()
        assert body["available"] is False
        assert body["status"] == "UNAVAILABLE"
        assert body["terrain_ready"] is False

    def test_planning_without_terrain_is_refused(self, client):
        response = client.post("/api/navigation/plan", json={})
        assert response.status_code == 409
        assert "elevation model" in response.json()["detail"]

    def test_path_render_404s_without_a_plan(self, client):
        assert client.get("/api/navigation/path.png").status_code == 404

    def test_planning_over_simulated_terrain_succeeds_and_is_tagged(self, client):
        response = client.post("/api/navigation/plan?allow_simulation=true",
                               json={"planner": "theta"})
        assert response.status_code == 200
        body = response.json()
        assert body["length_m"] > 0
        assert body["planner"] == "theta"
        assert body["provenance"]["source"] == "simulation"

    def test_both_planners_are_reachable(self, client):
        for planner in ("astar", "theta"):
            body = client.post("/api/navigation/plan?allow_simulation=true",
                               json={"planner": planner}).json()
            assert body["planner"] == planner
            assert body["length_m"] > 0

    def test_an_unknown_planner_is_rejected_with_the_valid_list(self, client):
        response = client.post("/api/navigation/plan?allow_simulation=true",
                               json={"planner": "teleport"})
        assert response.status_code == 400
        assert "theta" in response.json()["detail"]

    def test_the_planned_path_renders(self, client):
        client.post("/api/navigation/plan?allow_simulation=true", json={})
        response = client.get("/api/navigation/path.png")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_replanning_records_a_real_repair_cost(self, client):
        client.post("/api/navigation/plan?allow_simulation=true", json={})
        response = client.post("/api/navigation/replan",
                               json={"changes": [[40, 40, 1.0], [40, 41, 1.0]]})
        assert response.status_code == 200
        event = response.json()["event"]
        assert "repair_expansions" in event
        assert event["initial_expansions"] > 0
        assert event["cells_changed"] == 2
        assert client.get("/api/navigation").json()["replans"]

    def test_replanning_without_a_plan_is_refused(self, client):
        response = client.post("/api/navigation/replan",
                               json={"changes": [[1, 1, 1.0]]})
        assert response.status_code == 409

    def test_malformed_changes_are_rejected(self, client):
        client.post("/api/navigation/plan?allow_simulation=true", json={})
        assert client.post("/api/navigation/replan",
                           json={"changes": [[1, 1]]}).status_code == 400

    def test_a_plan_traces_back_to_the_dem(self, client):
        """Safe path -> traversability -> elevation, through the lineage API."""
        body = client.post("/api/navigation/plan?allow_simulation=true",
                           json={}).json()
        payload = client.get(f"/api/lineage/{body['plan_node_id']}").json()
        labels = [n["label"] for n in payload["chain"]]
        assert "dem_elevation" in labels
        assert "traversability_map" in labels
        assert "safe_path_plan" in labels
        assert payload["depth"] >= 3


# ===========================================================================
# Services in isolation
# ===========================================================================

class TestTerrainService:
    def test_find_dem_returns_none_when_the_root_is_absent(self, tmp_path):
        assert TerrainService(datasets_root=tmp_path / "nope").find_dem() is None

    def test_large_dems_are_decimated_not_truncated(self, tmp_path):
        """A 4000x4000 tile must not stall the console, nor lose its extent."""
        from perceptos.core.environment import get_environment
        root = tmp_path / "jezero_dem"
        root.mkdir(parents=True)
        np.save(root / "big.npy", np.random.default_rng(0).random((900, 900)))
        service = TerrainService(datasets_root=tmp_path)
        products = service.analyse(get_environment("mars"), max_cells=200)
        assert products.available
        assert max(products.dem.value.shape) <= 200
        assert products.cell_size_m > get_environment("mars").cell_size_m


class TestNavigationService:
    def test_default_goal_uses_the_planners_own_lethality_rule(self, client):
        """hazard_mask and cost >= hazard_threshold are different sets.

        The analyser's mask marks outright limit breaches (~2% of a real tile);
        the planners block on saturated weighted severity (~6%). Choosing a
        goal against the smaller set yields cells the planner then refuses.
        """
        from perceptos.api.analysis import NavigationService

        products = client.app.state.service.analyse_terrain(allow_simulation=True)
        result = products.traversability.value
        strict = (result.cost >= 0.95)
        assert strict.sum() >= result.hazard_mask.sum()

        safe = NavigationService._safe_cells(products, rover_radius_m=0.5)
        assert not safe[strict].any(), "a cell the planner blocks was called safe"

    def test_reachability_respects_the_no_corner_cutting_rule(self, client):
        """A naive 8-connected fill treats a diagonal gap as passable."""
        from perceptos.api.analysis import NavigationService

        # A staircase where the ONLY link between cells is the diagonal.
        safe = np.array([
            [True,  False, False],
            [False, True,  False],
            [False, False, True],
        ])
        reachable = NavigationService._reachable_from(safe, (0, 0))
        assert reachable[0, 0]
        assert not reachable[1, 1], (
            "the rover cannot squeeze through the corner gap between two rocks"
        )

        # Open one orthogonal and the step becomes legal.
        safe[0, 1] = True
        safe[1, 0] = True
        assert NavigationService._reachable_from(safe, (0, 0))[1, 1]

    def test_auto_selected_goal_is_always_plannable(self, client):
        """A default goal in a disconnected region makes every plan fail."""
        response = client.post("/api/navigation/plan?allow_simulation=true",
                               json={})
        assert response.status_code == 200
        assert response.json()["provenance"]["parameters"]["goal_auto_selected"] is True

    def test_an_explicit_goal_is_never_silently_relocated(self, client):
        """A commanded destination is honoured exactly, or refused.

        Quietly moving it to somewhere reachable would be worse than failing:
        the operator would get a route to a place they did not ask for.
        """
        from perceptos.api.analysis import NavigationService

        products = client.app.state.service.analyse_terrain(allow_simulation=True)
        safe = NavigationService._safe_cells(products, rover_radius_m=0.5)
        start = NavigationService._default_start(products, 0.5)
        reachable = NavigationService._reachable_from(safe, start)
        rows, cols = reachable.shape
        target = np.argwhere(reachable)[len(np.argwhere(reachable)) // 2]
        goal = [int(target[0]), int(target[1])]

        body = client.post(
            "/api/navigation/plan?allow_simulation=true",
            json={"start": list(start), "goal": goal},
        ).json()
        assert body["goal_cell"] == goal
        assert body["provenance"]["parameters"]["goal_auto_selected"] is False

    def test_an_unreachable_explicit_goal_is_refused_not_moved(self, client):
        from perceptos.api.analysis import NavigationService

        products = client.app.state.service.analyse_terrain(allow_simulation=True)
        safe = NavigationService._safe_cells(products, rover_radius_m=0.5)
        blocked = np.argwhere(~safe)
        if not len(blocked):
            pytest.skip("this synthetic terrain has no blocked cells")
        goal = [int(blocked[0][0]), int(blocked[0][1])]
        response = client.post("/api/navigation/plan?allow_simulation=true",
                               json={"goal": goal})
        assert response.status_code == 409

    def test_dstar_is_not_built_until_a_replan_is_requested(self, client):
        """Its initialisation is a full backward search; planning must not pay it.

        Building D* Lite eagerly made every plan request ~4x slower for callers
        who never replan.
        """
        from perceptos.api.app import get_service

        client.post("/api/navigation/plan?allow_simulation=true", json={})
        service = get_service(client.app)
        assert service.navigation.dstar is None
        assert service.navigation._dstar_seed is not None

        client.post("/api/navigation/replan", json={"changes": [[40, 40, 1.0]]})
        assert service.navigation.dstar is not None

    def test_planning_without_terrain_raises_with_diagnostics(self):
        from perceptos.api.analysis import TerrainProducts
        from perceptos.mars.planner import PlanningFailure

        empty = TerrainProducts(status=DataSourceStatus.UNAVAILABLE,
                                reason="no dem")
        with pytest.raises(PlanningFailure) as exc:
            NavigationService().plan(empty)
        assert exc.value.diagnostics["terrain_status"] == "UNAVAILABLE"


# ===========================================================================
# Console
# ===========================================================================

class TestConsolePanels:
    def test_console_declares_every_new_panel(self, client):
        html = client.get("/console").text
        for token in ("renderPerception", "renderTerrain", "renderNavigation"):
            assert token in html

    def test_console_calls_the_real_endpoints(self, client):
        html = client.get("/console").text
        for endpoint in ("/api/frame", "/api/stereo", "/api/terrain",
                         "/api/navigation", "/api/navigation/path.png"):
            assert endpoint in html

    def test_console_has_an_explicit_unavailable_renderer(self, client):
        """Panels without data must say UNAVAILABLE, not draw an empty chart."""
        html = client.get("/console").text
        assert "function unavailable(" in html
        assert "UNAVAILABLE" in html

    def test_console_never_hardcodes_a_measurement(self, client):
        """No seeded numbers: every displayed value must come from the API."""
        html = client.get("/console").text
        for forbidden in ("mockData", "sampleFrame", "fakeTelemetry",
                          "placeholderPath", "DEMO_VALUES"):
            assert forbidden not in html


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
