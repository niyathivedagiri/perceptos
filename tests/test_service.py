"""
PerceptOS service layer tests.

Everything runs against an in-memory database and a trivial two-stage pipeline,
so the API is testable without OpenCV, a camera, or a network. That is the
point of injecting the pipeline factory rather than constructing one inside the
app.

The tests that matter most are the persistence ones: a lineage chain written to
SQLite must reconstruct into the same payload the live graph produces, because
that equivalence is what lets the UI drawer answer questions about a mission
recorded months ago.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

# The service layer is optional: the geometry, vision and planning halves of
# PerceptOS run without a web framework. Skip rather than error on import, so a
# minimal `pip install numpy pytest` still gets a clean run instead of a wall of
# collection errors that look like the project is broken.
pytest.importorskip("fastapi", reason="install with: pip install fastapi httpx")
pytest.importorskip("httpx", reason="FastAPI TestClient needs httpx")

from fastapi.testclient import TestClient

from perceptos.api.app import create_app
from perceptos.api.service import PerceptOSService, ServiceError
from perceptos.core.pipeline import Pipeline
from perceptos.core.provenance import Source, traced
from perceptos.core.stage import FrameContext, PipelineStage
from perceptos.db.store import MissionStore
from perceptos.health.monitor import HAS_PSUTIL, HealthMonitor, PowerModel


# ===========================================================================
# A minimal pipeline: no camera, no OpenCV
# ===========================================================================

class SyntheticSeedStage(PipelineStage):
    name = "acquisition"
    provides = ("frame",)
    source = Source.SIMULATION

    def __init__(self, **params):
        super().__init__(**params)
        self.counter = 0

    def run(self, ctx: FrameContext) -> str:
        self.counter += 1
        self.emit(ctx, "frame", np.full((4, 4), self.counter, np.uint8),
                  algorithm="synthetic_seed", source=Source.SIMULATION,
                  confidence=1.0, units="px", label="raw_frame")
        return f"frame {self.counter}"


class AnalyseStage(PipelineStage):
    name = "analysis"
    requires = ("frame",)
    provides = ("decision",)

    def run(self, ctx: FrameContext) -> str:
        (frame,) = ctx.require("frame")
        mean = float(frame.value.mean())
        self.emit(ctx, "decision", {"steer": "left" if mean % 2 else "right"},
                  algorithm="ThresholdPolicy", source=Source.ESTIMATED,
                  confidence=0.77, parents=[frame], label="drive_decision")
        return "decided"


def make_pipeline(environment) -> Pipeline:
    return Pipeline([SyntheticSeedStage(), AnalyseStage()],
                    mode=environment.name, seed_keys=())


@pytest.fixture
def service():
    return PerceptOSService(store=MissionStore(":memory:"),
                            pipeline_factory=make_pipeline)


@pytest.fixture
def client(service):
    with TestClient(create_app(service)) as test_client:
        yield test_client


# ===========================================================================
# Persistence
# ===========================================================================

class TestMissionStore:
    def test_schema_initialises_and_reports_version(self):
        from perceptos.db.store import SCHEMA_VERSION
        store = MissionStore(":memory:")
        assert store.statistics()["schema_version"] == SCHEMA_VERSION
        assert store.statistics()["missions"] == 0

    def test_a_database_from_an_older_schema_is_refused_not_corrupted(self, tmp_path):
        """Writing v2 rows into a v1 file would silently lose the new columns."""
        import sqlite3
        path = tmp_path / "old.db"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE schema_info (version INTEGER NOT NULL, "
                     "created_utc REAL NOT NULL)")
        conn.execute("INSERT INTO schema_info VALUES (1, 0.0)")
        conn.commit()
        conn.close()
        with pytest.raises(RuntimeError, match="schema v1"):
            MissionStore(path)

    def test_foreign_keys_are_actually_enforced(self):
        """SQLite disables them per connection; a schema alone enforces nothing."""
        import sqlite3
        store = MissionStore(":memory:")
        with pytest.raises(sqlite3.IntegrityError):
            with store.transaction() as conn:
                conn.execute(
                    "INSERT INTO frames (mission_id, frame_index, captured_utc, "
                    "total_latency_ms, healthy) VALUES ('ghost', 1, 0, 0, 1)"
                )

    def test_mission_lifecycle(self):
        store = MissionStore(":memory:")
        mission_id = store.start_mission(name="t", environment="mars", mode="mars")
        record = store.get_mission(mission_id)
        assert record.status == "running" and record.ended_utc is None
        store.end_mission(mission_id)
        assert store.get_mission(mission_id).status == "complete"
        assert store.get_mission(mission_id).duration_s is not None

    def test_deleting_a_mission_cascades_to_its_frames(self, service):
        service.start_mission(name="cascade")
        service.run(3)
        mission_id = service.state.mission_id
        assert service.store.statistics()["frames"] == 3
        assert service.store.delete_mission(mission_id) is True
        assert service.store.statistics()["frames"] == 0
        assert service.store.statistics()["lineage_nodes"] == 0

    def test_unknown_mission_returns_none_not_an_exception(self):
        assert MissionStore(":memory:").get_mission("nope") is None

    def test_telemetry_round_trip_preserves_order(self):
        store = MissionStore(":memory:")
        mission_id = store.start_mission(name="t", environment="mars", mode="mars")
        for i in range(5):
            store.record_telemetry(mission_id, "health", {"cpu": i})
            time.sleep(0.001)
        samples = store.get_telemetry(mission_id, "health")
        assert [s["payload"]["cpu"] for s in samples] == [0, 1, 2, 3, 4]

    def test_numpy_values_survive_serialisation(self):
        """Metadata routinely contains numpy scalars that json.dumps rejects."""
        store = MissionStore(":memory:")
        mission_id = store.start_mission(name="t", environment="mars", mode="mars")
        store.record_telemetry(mission_id, "x", {"a": np.float32(1.5),
                                                 "b": np.arange(3)})
        assert store.get_telemetry(mission_id, "x")[0]["payload"]["a"] == 1.5


class TestPersistedLineage:
    def test_persisted_chain_matches_the_live_chain(self, service):
        """The equivalence that lets the drawer answer about old missions."""
        service.start_mission(name="lineage")
        service.process_frame()
        node = service.search_lineage(label="drive_decision")[0]

        live = service.explain(node["id"])
        persisted = service.store.explain(service.state.mission_id, node["id"])

        assert live["origin"] == "live"
        assert live["depth"] == persisted["depth"] == 2
        assert live["algorithms"] == persisted["algorithms"]
        assert live["weakest_confidence"] == persisted["weakest_confidence"]
        assert live["contains_simulation"] == persisted["contains_simulation"] is True

    def test_simulation_flag_survives_the_database(self, service):
        service.start_mission(name="sim")
        service.process_frame()
        node = service.search_lineage(label="drive_decision")[0]
        assert node["provenance"]["is_synthetic"] is True
        payload = service.store.explain(service.state.mission_id, node["id"])
        assert payload["contains_simulation"] is True

    def test_ancestors_are_resolved_in_one_recursive_query(self, service):
        service.start_mission(name="chain")
        service.run(2)
        node = service.search_lineage(label="drive_decision")[0]
        payload = service.store.explain(service.state.mission_id, node["id"])
        roots = payload["roots"]
        assert len(roots) == 1
        assert roots[0]["provenance"]["source"] == "simulation"

    def test_contamination_survives_the_database(self, service):
        """The AnalyseStage declares ESTIMATED, but its input is synthetic.

        The provenance layer overrides the stage's claim about itself, and that
        override must persist: a value stored as ESTIMATED would let a mission
        review months later mistake a simulated run for a real one.
        """
        service.start_mission(name="filter")
        service.run(2)
        assert service.search_lineage(source="estimated") == []
        simulated = service.search_lineage(source="simulation")
        assert len(simulated) == 4                   # 2 frames x 2 nodes
        assert all(n["provenance"]["is_synthetic"] for n in simulated)

    def test_lineage_search_filters_by_label_and_frame(self, service):
        service.start_mission(name="filter")
        service.run(3)
        assert len(service.search_lineage(label="drive_decision")) == 3
        assert len(service.search_lineage(frame_index=2)) == 2

    def test_repeated_frames_do_not_duplicate_shared_ancestors(self, service):
        service.start_mission(name="dedup")
        service.run(4)
        stats = service.store.statistics()
        assert stats["lineage_nodes"] == 8      # 2 per frame, no duplicates
        assert stats["lineage_edges"] == 4


# ===========================================================================
# Health
# ===========================================================================

class TestHealthMonitor:
    def test_system_metrics_are_measured_or_none_never_guessed(self):
        health = HealthMonitor().system()
        if HAS_PSUTIL:
            assert health.cpu_percent is not None
            assert health.memory_total_mb > 0
        else:
            assert health.cpu_percent is None
        assert health.grade in ("nominal", "loaded", "critical", "unknown")

    def test_battery_is_always_labelled_simulation(self):
        """There is no rover battery in a laptop. The tile must say so."""
        power = HealthMonitor().traced_power()
        assert power.source is Source.SIMULATION
        assert "not measured" in power.provenance.parameters["note"]

    def test_system_metrics_are_labelled_as_a_real_measurement(self):
        system = HealthMonitor().traced_system()
        expected = Source.LIVE_SENSOR if HAS_PSUTIL else Source.CALCULATED
        assert system.source is expected

    def test_snapshot_carries_provenance_on_both_halves(self):
        snapshot = HealthMonitor().snapshot()
        assert snapshot["power"]["_provenance"]["source"] == "simulation"
        assert snapshot["system"]["_provenance"]["is_synthetic"] is False

    def test_power_model_discharges_while_driving_and_recovers_at_idle(self):
        model = PowerModel(state_of_charge=0.5)
        for _ in range(20):
            model.step(60.0, driving=True)
        driving_soc = model.state_of_charge
        assert driving_soc < 0.5
        for _ in range(20):
            model.step(60.0, driving=False, compute_load=0.0)
        assert model.state_of_charge > driving_soc

    def test_endurance_is_none_when_the_generator_covers_the_load(self):
        model = PowerModel(generator_w=500.0)
        assert model.endurance_hours(driving=True) is None
        assert model.to_dict(driving=True)["net_positive"] is True

    def test_state_of_charge_is_clamped(self):
        model = PowerModel(state_of_charge=0.02, generator_w=0.0)
        for _ in range(500):
            model.step(60.0, driving=True)
        assert model.state_of_charge == 0.0


# ===========================================================================
# Service runtime
# ===========================================================================

class TestService:
    def test_data_mode_is_inspected_not_declared(self, service):
        service.ensure_pipeline()
        assert service.data_mode == "simulation"

    def test_cannot_start_two_missions_at_once(self, service):
        service.start_mission(name="a")
        with pytest.raises(ServiceError, match="already running"):
            service.start_mission(name="b")

    def test_stopping_without_a_mission_is_an_error(self, service):
        with pytest.raises(ServiceError, match="no mission"):
            service.stop_mission()

    def test_frames_are_only_persisted_while_recording(self, service):
        service.run(3)
        assert service.store.statistics()["frames"] == 0
        service.start_mission(name="rec")
        service.run(2)
        assert service.store.statistics()["frames"] == 2

    def test_environment_switch_rebuilds_the_pipeline(self, service):
        service.ensure_pipeline()
        service.set_environment("warehouse")
        assert service.environment.name == "warehouse"
        assert service.pipeline.mode == "warehouse"

    def test_missing_pipeline_factory_says_so(self):
        bare = PerceptOSService(store=MissionStore(":memory:"))
        with pytest.raises(ServiceError, match="pipeline_factory"):
            bare.process_frame()

    def test_subscribers_receive_frames_and_a_failing_one_is_dropped(self, service):
        received: list[dict] = []
        service.subscribe(received.append)

        def explode(_message):
            raise RuntimeError("browser tab closed")

        service.subscribe(explode)
        service.run(2)
        assert len(received) == 2
        assert len(service._subscribers) == 1, "a failing subscriber must be removed"


# ===========================================================================
# HTTP API
# ===========================================================================

class TestAPI:
    def test_status_reports_environment_and_data_mode(self, client):
        body = client.get("/api/status").json()
        assert body["environment"]["name"] == "mars"
        assert body["service"]["running"] is False

    def test_environments_endpoint_lists_all_five(self, client):
        body = client.get("/api/environments").json()
        names = {e["name"] for e in body["environments"]}
        assert names == {"mars", "moon", "earth", "warehouse", "factory"}
        assert body["active"] == "mars"

    def test_unknown_environment_is_404_with_the_valid_list(self, client):
        response = client.get("/api/environments/jupiter")
        assert response.status_code == 404
        assert "available" in response.json()["detail"]

    def test_switching_environment_takes_effect_immediately(self, client):
        client.get("/api/pipeline")                     # force pipeline construction
        assert client.post("/api/environments/active",
                           json={"environment": "moon"}).status_code == 200
        assert client.get("/api/status").json()["environment"]["name"] == "moon"

    def test_pipeline_diagram_is_mermaid_from_real_wiring(self, client):
        text = client.get("/api/pipeline/diagram").text
        assert text.startswith("graph LR")
        assert "acquisition" in text and "|frame|" in text

    def test_toggling_a_stage_changes_the_contract(self, client):
        client.post("/api/pipeline/stages/analysis", json={"enabled": False})
        stages = {s["name"]: s for s in client.get("/api/pipeline").json()["stages"]}
        assert stages["analysis"]["enabled"] is False

    def test_unknown_stage_is_409_naming_the_real_stages(self, client):
        response = client.post("/api/pipeline/stages/warp_drive", json={"enabled": True})
        assert response.status_code == 409
        assert "acquisition" in response.json()["detail"]

    def test_mission_lifecycle_over_http(self, client):
        created = client.post("/api/missions", json={"name": "api-run"})
        assert created.status_code == 201
        mission_id = created.json()["id"]

        assert client.post("/api/pipeline/run", json={"frames": 3}).status_code == 200
        frames = client.get(f"/api/missions/{mission_id}/frames").json()["frames"]
        assert len(frames) == 3

        stopped = client.post("/api/missions/active/stop", json={"status": "complete"})
        assert stopped.json()["status"] == "complete"
        assert stopped.json()["frames"] == 3

    def test_starting_two_missions_returns_409(self, client):
        client.post("/api/missions", json={"name": "first"})
        conflict = client.post("/api/missions", json={"name": "second"})
        assert conflict.status_code == 409
        assert "already running" in conflict.json()["detail"]

    def test_frame_detail_includes_stage_timings(self, client):
        mission_id = client.post("/api/missions", json={}).json()["id"]
        client.post("/api/pipeline/run", json={"frames": 1})
        detail = client.get(f"/api/missions/{mission_id}/frames/1").json()
        stages = {s["stage"] for s in detail["stages"]}
        assert stages == {"acquisition", "analysis"}
        assert all(s["latency_ms"] >= 0 for s in detail["stages"])

    def test_lineage_endpoint_answers_why(self, client):
        client.post("/api/missions", json={})
        client.post("/api/pipeline/run", json={"frames": 1})
        node = client.get("/api/lineage", params={"label": "drive_decision"}
                          ).json()["nodes"][0]
        payload = client.get(f"/api/lineage/{node['id']}").json()
        assert payload["depth"] == 2
        assert "ThresholdPolicy" in payload["algorithms"]
        assert payload["contains_simulation"] is True
        assert payload["origin"] == "live"

    def test_lineage_for_an_old_mission_comes_from_the_database(self, client):
        mission_id = client.post("/api/missions", json={}).json()["id"]
        client.post("/api/pipeline/run", json={"frames": 1})
        node = client.get("/api/lineage", params={"label": "drive_decision"}
                          ).json()["nodes"][0]
        client.post("/api/missions/active/stop", json={})
        client.post("/api/missions", json={"name": "second"})   # new live lineage

        payload = client.get(f"/api/lineage/{node['id']}",
                             params={"mission_id": mission_id}).json()
        assert payload["origin"] == "persisted"
        assert payload["depth"] == 2

    def test_unknown_lineage_node_is_404_with_a_useful_hint(self, client):
        client.post("/api/missions", json={})
        response = client.get("/api/lineage/deadbeef")
        assert response.status_code == 404
        assert "mission_id" in response.json()["detail"]

    def test_health_separates_measured_from_modelled(self, client):
        body = client.get("/api/health").json()
        assert body["power"]["_provenance"]["source"] == "simulation"
        assert body["system"]["_provenance"]["source"] in ("live_sensor", "calculated")
        assert "pipeline" in body and "store" in body

    def test_health_is_recorded_as_telemetry_during_a_mission(self, client):
        mission_id = client.post("/api/missions", json={}).json()["id"]
        client.get("/api/health")
        client.get("/api/health")
        samples = client.get(f"/api/missions/{mission_id}/telemetry/health"
                             ).json()["samples"]
        assert len(samples) == 2

    def test_deleting_a_mission_returns_204_then_404(self, client):
        mission_id = client.post("/api/missions", json={}).json()["id"]
        client.post("/api/missions/active/stop", json={})
        assert client.delete(f"/api/missions/{mission_id}").status_code == 204
        assert client.get(f"/api/missions/{mission_id}").status_code == 404

    def test_run_request_is_bounded(self, client):
        assert client.post("/api/pipeline/run", json={"frames": 10_000}
                           ).status_code == 422

    def test_openapi_schema_is_generated(self, client):
        schema = client.get("/openapi.json").json()
        assert "/api/lineage/{node_id}" in schema["paths"]
        assert schema["info"]["title"] == "PerceptOS"

    def test_websocket_pushes_a_hello_then_frames(self, client):
        with client.websocket_connect("/ws/telemetry") as socket:
            hello = socket.receive_json()
            assert hello["type"] == "hello"
            client.post("/api/pipeline/run", json={"frames": 1})
            message = socket.receive_json()
            assert message["type"] in ("frame", "health")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
