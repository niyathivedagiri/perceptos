"""
PerceptOS replay and report tests.

Replay is verified by asserting that what comes back is what was *recorded* —
not something recomputed that happens to look similar. Reports are verified on
the credibility surface: a synthetic mission must be labelled as one in every
representation, because a report is the artefact most likely to be read out of
the context that produced it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from fastapi.testclient import TestClient

from perceptos.api.app import create_app
from perceptos.api.replay import ReplayService
from perceptos.api.service import PerceptOSService
from perceptos.core.pipeline import Pipeline
from perceptos.core.provenance import Source
from perceptos.core.stage import FrameContext, PipelineStage
from perceptos.db.store import MissionStore
from perceptos.reports.mission import ReportBuilder


class SeedStage(PipelineStage):
    name = "acquisition"
    provides = ("frame",)
    source = Source.SIMULATION

    def run(self, ctx: FrameContext) -> str:
        self.emit(ctx, "frame", np.full((4, 4), ctx.frame_id, np.uint8),
                  algorithm="synthetic_seed", source=Source.SIMULATION,
                  confidence=1.0, units="px", label="frame")
        return f"frame {ctx.frame_id}"


class AnalyseStage(PipelineStage):
    name = "analysis"
    requires = ("frame",)
    provides = ("decision",)

    def run(self, ctx: FrameContext) -> str:
        (frame,) = ctx.require("frame")
        self.emit(ctx, "decision", {"steer": "left"}, algorithm="ThresholdPolicy",
                  source=Source.ESTIMATED, confidence=0.77, parents=[frame],
                  label="safe_path_plan")
        return "decided"


def make_pipeline(environment):
    return Pipeline([SeedStage(), AnalyseStage()], mode=environment.name,
                    seed_keys=())


@pytest.fixture
def service():
    return PerceptOSService(store=MissionStore(":memory:"),
                            pipeline_factory=make_pipeline)


@pytest.fixture
def client(service):
    with TestClient(create_app(service)) as test_client:
        yield test_client


@pytest.fixture
def recorded(client):
    """A completed mission with five recorded frames."""
    mission_id = client.post("/api/missions", json={"name": "replay-me"}).json()["id"]
    client.post("/api/pipeline/run", json={"frames": 5})
    client.post("/api/missions/active/stop", json={})
    return mission_id


# ===========================================================================
# Replay
# ===========================================================================

class TestReplayControls:
    def test_no_session_is_unavailable_with_the_command_to_open_one(self, client):
        body = client.get("/api/replay").json()
        assert body["open"] is False
        assert body["status"] == "UNAVAILABLE"
        assert "open" in body["reason"]

    def test_opening_an_unknown_mission_is_404(self, client):
        assert client.post("/api/replay/nope/open").status_code == 404

    def test_a_mission_with_no_frames_cannot_be_replayed(self, client):
        """Started but never run: there is genuinely nothing to play back."""
        mission_id = client.post("/api/missions", json={"name": "empty"}).json()["id"]
        client.post("/api/missions/active/stop", json={})
        response = client.post(f"/api/replay/{mission_id}/open")
        assert response.status_code == 409
        assert "recorded no frames" in response.json()["detail"]

    def test_opening_reports_the_full_frame_range(self, client, recorded):
        body = client.post(f"/api/replay/{recorded}/open").json()
        assert body["total_frames"] == 5
        assert body["position"] == 0
        assert body["at_start"] is True

    def test_seek_step_play_and_pause(self, client, recorded):
        client.post(f"/api/replay/{recorded}/open")
        assert client.post("/api/replay/seek", json={"position": 3}).json()["position"] == 3
        assert client.post("/api/replay/step?delta=1").json()["position"] == 4
        assert client.post("/api/replay/step?delta=-2").json()["position"] == 2
        assert client.post("/api/replay/play?speed=4").json()["speed"] == 4.0
        assert client.post("/api/replay/pause").json()["playing"] is False

    def test_seeking_past_the_end_clamps(self, client, recorded):
        client.post(f"/api/replay/{recorded}/open")
        body = client.post("/api/replay/seek", json={"position": 9999}).json()
        assert body["position"] == 4
        assert body["at_end"] is True

    def test_controls_without_a_session_are_refused(self, client):
        assert client.post("/api/replay/step").status_code == 409
        assert client.get("/api/replay/frame").status_code == 409


class TestReplayContent:
    def test_a_replayed_frame_is_the_recorded_one(self, client, recorded):
        """Recomputing would make replay a demo rather than a record."""
        client.post(f"/api/replay/{recorded}/open")
        live = client.get(f"/api/missions/{recorded}/frames/3").json()
        replayed = client.post("/api/replay/seek", json={"position": 2})
        frame = client.get("/api/replay/frame").json()
        assert frame["frame_index"] == 3
        assert frame["total_latency_ms"] == live["total_latency_ms"]
        assert frame["replayed"] is True
        assert "Nothing is recomputed" in frame["note"]

    def test_a_replayed_frame_carries_its_lineage(self, client, recorded):
        client.post(f"/api/replay/{recorded}/open")
        frame = client.get("/api/replay/frame").json()
        assert frame["lineage_nodes"]
        labels = {n["label"] for n in frame["lineage_nodes"]}
        assert "safe_path_plan" in labels

    def test_a_lineage_node_from_replay_resolves_months_later(self, client, recorded):
        client.post(f"/api/replay/{recorded}/open")
        node = client.get("/api/replay/frame").json()["lineage_nodes"][0]
        payload = client.get(f"/api/lineage/{node['id']}",
                             params={"mission_id": recorded}).json()
        assert payload["origin"] == "persisted"
        assert payload["depth"] >= 1

    def test_the_timeline_reports_real_latency_statistics(self, client, recorded):
        client.post(f"/api/replay/{recorded}/open")
        timeline = client.get("/api/replay/timeline").json()
        assert len(timeline["frames"]) == 5
        stats = timeline["statistics"]
        assert stats["count"] == 5
        assert stats["max_latency_ms"] >= stats["mean_latency_ms"]

    def test_the_stage_profile_identifies_the_bottleneck(self, client, recorded):
        client.post(f"/api/replay/{recorded}/open")
        profile = client.get("/api/replay/profile").json()
        assert profile["bottleneck"] in ("acquisition", "analysis")
        assert {s["stage"] for s in profile["stages"]} == {"acquisition", "analysis"}


class TestCounterfactual:
    def test_it_requires_a_recorded_decision(self, client):
        response = client.post("/api/replay/counterfactual", json={})
        assert response.status_code == 409
        assert "Plan a route first" in response.json()["detail"]

    def test_changing_a_parameter_produces_a_diff(self, client):
        client.post("/api/navigation/plan?allow_simulation=true",
                    json={"planner": "astar"})
        body = client.post("/api/replay/counterfactual",
                           json={"planner": "theta"}).json()
        assert "planner" in body["parameter"]
        assert body["verdict"]

    def test_identical_parameters_report_no_change(self, client):
        client.post("/api/navigation/plan?allow_simulation=true",
                    json={"planner": "theta", "cost_weight": 4.0})
        body = client.post("/api/replay/counterfactual",
                           json={"planner": "theta"}).json()
        assert body["parameter"] == "no parameters changed"

    def test_a_failing_alternative_is_a_result_not_an_error(self, client):
        """'This setting would have found no route' is the answer, not a 500."""
        client.post("/api/navigation/plan?allow_simulation=true", json={})
        response = client.post("/api/replay/counterfactual",
                               json={"inflation_radius_m": 9.0})
        assert response.status_code == 200
        body = response.json()
        if not body["alternative"]["planned"]:
            assert "found no route" in body["verdict"]
            assert body["alternative"]["reason"]

    def test_an_unknown_planner_is_rejected(self, client):
        client.post("/api/navigation/plan?allow_simulation=true", json={})
        response = client.post("/api/replay/counterfactual",
                               json={"planner": "teleport"})
        assert response.status_code == 400
        assert "theta" in response.json()["detail"]


# ===========================================================================
# Reports
# ===========================================================================

class TestReports:
    def test_an_unknown_mission_is_404(self, client):
        assert client.get("/api/missions/nope/report").status_code == 404

    def test_json_report_contains_every_section(self, client, recorded):
        payload = client.get(f"/api/missions/{recorded}/report").json()
        for section in ("mission", "credibility", "performance", "stages",
                        "key_values", "limitations"):
            assert section in payload

    def test_a_synthetic_mission_is_labelled_in_every_format(self, client, recorded):
        """A report outlives its context; an unlabelled synthetic figure is worst here."""
        payload = client.get(f"/api/missions/{recorded}/report").json()
        assert payload["credibility"]["contains_simulation"] is True

        markdown = client.get(f"/api/missions/{recorded}/report?format=markdown").text
        assert "SIMULATION" in markdown
        assert "describes no physical event" in markdown

    def test_key_values_reference_their_lineage_node(self, client, recorded):
        payload = client.get(f"/api/missions/{recorded}/report").json()
        assert payload["key_values"]
        node_id = payload["key_values"][0]["id"]
        traced = client.get(f"/api/lineage/{node_id}",
                            params={"mission_id": recorded}).json()
        assert traced["node"]["id"] == node_id

    def test_key_values_are_deduplicated_across_frames(self, client, recorded):
        """500 identical rows would bury the summary."""
        labels = [v["label"] for v in
                  client.get(f"/api/missions/{recorded}/report").json()["key_values"]]
        assert len(labels) == len(set(labels))

    def test_limitations_are_derived_from_what_was_recorded(self, client, recorded):
        limitations = client.get(f"/api/missions/{recorded}/report").json()["limitations"]
        joined = " ".join(limitations)
        assert "simulated data" in joined
        assert "No depth was computed" in joined
        assert "dead reckoning" in joined

    def test_csv_has_one_row_per_stage_per_frame(self, client, recorded):
        text = client.get(f"/api/missions/{recorded}/report?format=csv").text
        rows = [r for r in text.strip().splitlines() if r]
        assert rows[0].startswith("frame_index,latency_ms")
        assert len(rows) == 1 + 5 * 2          # header + 5 frames x 2 stages

    def test_markdown_includes_the_stage_profile(self, client, recorded):
        markdown = client.get(f"/api/missions/{recorded}/report?format=markdown").text
        assert "## Stage profile" in markdown
        assert "acquisition" in markdown
        assert "## Key values and their derivation" in markdown

    def test_an_unknown_format_is_rejected_with_the_valid_list(self, client, recorded):
        response = client.get(f"/api/missions/{recorded}/report?format=pdf")
        assert response.status_code == 400
        assert "json, csv or markdown" in response.json()["detail"]

    def test_performance_percentiles_are_computed_from_real_frames(self, client,
                                                                   recorded):
        performance = client.get(f"/api/missions/{recorded}/report"
                                 ).json()["performance"]
        assert performance["frames"] == 5
        assert performance["max_latency_ms"] >= performance["p95_latency_ms"]
        assert performance["p95_latency_ms"] >= performance["mean_latency_ms"] * 0.5

    def test_a_report_builds_directly_from_the_store(self, service):
        service.start_mission(name="direct")
        service.run(3)
        mission_id = service.state.mission_id
        service.stop_mission()
        report = ReportBuilder(service.store).build(mission_id)
        assert report.payload["performance"]["frames"] == 3
        assert report.to_markdown().startswith("# Mission report")
        assert report.to_json().startswith("{")


# ===========================================================================
# SLAM API
# ===========================================================================

class TestSlamEndpoint:
    def test_a_pipeline_without_slam_says_so(self, client):
        body = client.get("/api/slam").json()
        assert body["available"] is False
        assert body["status"] == "UNAVAILABLE"
        assert "no SLAM stage" in body["reason"]

    def test_optimising_without_a_slam_stage_is_refused(self, client):
        assert client.post("/api/slam/optimise").status_code == 409

    def test_slam_reports_its_map_when_present(self, service):
        from perceptos.core.environment import get_environment
        from perceptos.slam.system import SlamStage

        stage = SlamStage()
        for i in range(6):
            stage.system.process(
                __import__("perceptos.slam.graph", fromlist=["Pose2D"]).Pose2D(
                    float(i), 0.0, 0.0), timestamp=float(i))

        def factory(environment):
            return Pipeline([SeedStage(), stage], mode=environment.name,
                            seed_keys=())

        runtime = PerceptOSService(store=MissionStore(":memory:"),
                                   pipeline_factory=factory)
        runtime.ensure_pipeline()
        with TestClient(create_app(runtime)) as client:
            body = client.get("/api/slam").json()
            assert body["available"] is True
            assert body["statistics"]["keyframes"] > 0
            assert body["landmark_status"] == "UNAVAILABLE"
            assert "unknown scale" in body["landmark_reason"]


class TestConsolePhase5Panels:
    def test_console_declares_the_new_panels(self, client):
        html = client.get("/console").text
        for token in ("renderSlam", "renderReplay", "doCounterfactual"):
            assert token in html

    def test_console_calls_the_real_phase5_endpoints(self, client):
        html = client.get("/console").text
        for endpoint in ("/api/slam", "/api/replay", "/api/replay/counterfactual",
                         "format=markdown"):
            assert endpoint in html

    def test_the_slam_panel_surfaces_the_odometry_residual(self, client):
        """The number that actually reveals a false loop closure."""
        html = client.get("/console").text
        assert "max_odometry_residual" in html
        assert "SUSPECT LOOP CLOSURE" in html

    def test_the_slam_panel_names_dead_reckoning_when_no_loops_close(self, client):
        assert "dead reckoning" in client.get("/console").text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
