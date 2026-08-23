"""
perceptos.api.replay
====================

Mission replay — the digital twin.

Everything replayed here comes from the mission database. Nothing is
regenerated, re-inferred or smoothed: a replayed frame reports exactly the
latency, stage statuses and lineage that were recorded when it ran. That is
what makes replay usable for review rather than for demonstration.

Two capabilities:

**Scrubbing.** Step, seek and play through a recorded mission, inspecting any
frame's stage timings and lineage. The recorded lineage is the same payload the
live drawer shows, so a value can be traced months after the run.

**Counterfactual comparison.** Re-run a recorded *terrain* decision with
different planner parameters and diff the outcome. This is the feature that
turns replay from a playback tool into an engineering one: "would a smaller
inflation radius have found a route?" is answerable, with both the original and
the alternative shown side by side.

The counterfactual is honest about its scope. It replans over the *recorded*
cost map, so it answers what a different planner would have decided given the
same perception. It cannot answer what a different perception pipeline would
have seen, because those frames were never captured — and a replay that
silently re-ran perception on regenerated inputs would be comparing against
something that never happened.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..db.store import MissionStore
from ..mars.planner import AStarPlanner, PathPlan, PlanningFailure
from ..mars.planners import ThetaStarPlanner

__all__ = ["ReplaySession", "ReplayService", "CounterfactualResult"]

PLANNERS = {"astar": AStarPlanner, "theta": ThetaStarPlanner}


@dataclass(slots=True)
class ReplaySession:
    """Playback position within one recorded mission."""

    mission_id: str
    mission: dict[str, Any]
    frame_indices: list[int]
    position: int = 0
    playing: bool = False
    speed: float = 1.0
    opened_utc: float = field(default_factory=time.time)

    @property
    def total(self) -> int:
        return len(self.frame_indices)

    @property
    def current_frame(self) -> Optional[int]:
        if not self.frame_indices:
            return None
        return self.frame_indices[min(self.position, self.total - 1)]

    @property
    def at_end(self) -> bool:
        return self.position >= self.total - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "mission": self.mission,
            "position": self.position,
            "total_frames": self.total,
            "current_frame": self.current_frame,
            "playing": self.playing,
            "speed": self.speed,
            "at_end": self.at_end,
            "at_start": self.position <= 0,
        }


@dataclass(slots=True)
class CounterfactualResult:
    """One "what if" comparison, with both outcomes stated."""

    parameter: str
    baseline: dict[str, Any]
    alternative: dict[str, Any]
    differences: dict[str, Any]
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "baseline": self.baseline,
            "alternative": self.alternative,
            "differences": self.differences,
            "verdict": self.verdict,
        }


class ReplayService:
    """Opens recorded missions and steps through them."""

    def __init__(self, store: MissionStore) -> None:
        self.store = store
        self.session: Optional[ReplaySession] = None

    # ------------------------------------------------------------- lifecycle

    def open(self, mission_id: str) -> ReplaySession:
        record = self.store.get_mission(mission_id)
        if record is None:
            raise KeyError(f"no mission {mission_id!r}")
        frames = self.store.get_frames(mission_id, limit=100_000)
        if not frames:
            raise ValueError(
                f"mission {mission_id!r} recorded no frames. It may have been "
                "started but never run; there is nothing to replay."
            )
        self.session = ReplaySession(
            mission_id=mission_id, mission=record.to_dict(),
            frame_indices=[f["frame_index"] for f in frames],
        )
        return self.session

    def close(self) -> None:
        self.session = None

    def _require(self) -> ReplaySession:
        if self.session is None:
            raise RuntimeError(
                "no replay session open. POST /api/replay/{mission_id}/open first."
            )
        return self.session

    # -------------------------------------------------------------- controls

    def seek(self, position: int) -> ReplaySession:
        session = self._require()
        session.position = max(0, min(int(position), session.total - 1))
        return session

    def step(self, delta: int = 1) -> ReplaySession:
        return self.seek(self._require().position + int(delta))

    def play(self, *, speed: float = 1.0) -> ReplaySession:
        session = self._require()
        session.playing = True
        session.speed = float(speed)
        return session

    def pause(self) -> ReplaySession:
        session = self._require()
        session.playing = False
        return session

    # --------------------------------------------------------------- content

    def frame(self, index: Optional[int] = None) -> dict[str, Any]:
        """The recorded detail for one frame, exactly as it was stored."""
        session = self._require()
        frame_index = index if index is not None else session.current_frame
        if frame_index is None:
            raise ValueError("this mission has no frames")
        detail = self.store.get_frame_detail(session.mission_id, frame_index)
        if detail is None:
            raise KeyError(f"no frame {frame_index} in {session.mission_id!r}")
        nodes = self.store.find_nodes(session.mission_id,
                                      frame_index=frame_index, limit=200)
        return {
            **detail,
            "position": session.position,
            "total_frames": session.total,
            "lineage_nodes": [
                {"id": n["id"], "label": n["label"], "stage": n["stage"],
                 "provenance": n["provenance"]}
                for n in nodes
            ],
            "replayed": True,
            "note": ("All values are read from the mission record. Nothing is "
                     "recomputed, so the timings are the ones actually observed."),
        }

    def timeline(self) -> dict[str, Any]:
        """Per-frame latency and health across the whole mission."""
        session = self._require()
        frames = self.store.get_frames(session.mission_id, limit=100_000)
        latencies = [f["total_latency_ms"] for f in frames]
        unhealthy = [f["frame_index"] for f in frames if not f["healthy"]]
        return {
            "mission_id": session.mission_id,
            "frames": [
                {"frame_index": f["frame_index"],
                 "latency_ms": round(f["total_latency_ms"], 3),
                 "healthy": f["healthy"]}
                for f in frames
            ],
            "statistics": {
                "count": len(frames),
                "mean_latency_ms": (round(float(np.mean(latencies)), 3)
                                    if latencies else None),
                "max_latency_ms": (round(float(np.max(latencies)), 3)
                                   if latencies else None),
                "p95_latency_ms": (round(float(np.percentile(latencies, 95)), 3)
                                   if latencies else None),
                "unhealthy_frames": unhealthy,
            },
        }

    def stage_profile(self) -> dict[str, Any]:
        """Where the time actually went, aggregated across the mission."""
        session = self._require()
        totals: dict[str, list[float]] = {}
        statuses: dict[str, dict[str, int]] = {}
        for index in session.frame_indices:
            detail = self.store.get_frame_detail(session.mission_id, index)
            if detail is None:
                continue
            for stage in detail["stages"]:
                totals.setdefault(stage["stage"], []).append(stage["latency_ms"])
                bucket = statuses.setdefault(stage["stage"], {})
                bucket[stage["status"]] = bucket.get(stage["status"], 0) + 1
        profile = [
            {"stage": name,
             "mean_latency_ms": round(float(np.mean(values)), 3),
             "max_latency_ms": round(float(np.max(values)), 3),
             "share": round(float(np.sum(values)), 3),
             "statuses": statuses.get(name, {})}
            for name, values in totals.items()
        ]
        profile.sort(key=lambda entry: -entry["share"])
        return {"stages": profile,
                "bottleneck": profile[0]["stage"] if profile else None}

    # -------------------------------------------------------- counterfactual

    def counterfactual_plan(self, cost_map: np.ndarray, *,
                            start: tuple[int, int], goal: tuple[int, int],
                            cell_size_m: float,
                            baseline: dict[str, Any],
                            planner: str = "theta",
                            cost_weight: Optional[float] = None,
                            inflation_radius_m: Optional[float] = None
                            ) -> CounterfactualResult:
        """Re-plan the recorded terrain with different parameters and diff.

        Both outcomes are reported, including the case where the alternative
        fails. "This setting would have found no route" is a result, not an
        error to be swallowed.
        """
        key = planner.lower()
        if key not in PLANNERS:
            raise ValueError(f"unknown planner {planner!r}; "
                             f"available: {sorted(PLANNERS)}")
        weight = (cost_weight if cost_weight is not None
                  else baseline.get("cost_weight", 4.0))
        radius = (inflation_radius_m if inflation_radius_m is not None
                  else baseline.get("inflation_radius_m", 0.5))

        engine = PLANNERS[key](cost_weight=float(weight),
                               inflation_radius_m=float(radius))
        changed = []
        if cost_weight is not None and cost_weight != baseline.get("cost_weight"):
            changed.append(f"cost_weight {baseline.get('cost_weight')} -> {cost_weight}")
        if (inflation_radius_m is not None
                and inflation_radius_m != baseline.get("inflation_radius_m")):
            changed.append(f"inflation_radius_m "
                           f"{baseline.get('inflation_radius_m')} -> {inflation_radius_m}")
        if key != baseline.get("planner"):
            changed.append(f"planner {baseline.get('planner')} -> {key}")
        parameter = "; ".join(changed) or "no parameters changed"

        try:
            plan: Optional[PathPlan] = engine.plan(
                cost_map, start, goal, cell_size_m=cell_size_m)
            alternative = plan.to_dict()
            alternative["planned"] = True
        except PlanningFailure as exc:
            alternative = {"planned": False, "reason": str(exc),
                           "diagnostics": exc.diagnostics}
            plan = None

        differences: dict[str, Any] = {}
        if plan is not None and baseline.get("planned", True):
            for field_name in ("length_m", "mean_cell_cost", "max_cell_cost",
                               "turns_deg", "nodes_expanded"):
                before = baseline.get(field_name)
                after = alternative.get(field_name)
                if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                    differences[field_name] = {
                        "baseline": before, "alternative": after,
                        "delta": round(after - before, 4),
                    }

        if plan is None:
            verdict = ("The alternative parameters found no route. The recorded "
                       "decision was the safer one available.")
        elif not differences:
            verdict = "No comparable outcome: the baseline recorded no plan."
        else:
            length = differences.get("length_m", {}).get("delta", 0.0)
            risk = differences.get("max_cell_cost", {}).get("delta", 0.0)
            if abs(length) < 1e-6 and abs(risk) < 1e-6:
                verdict = "Identical route: these parameters changed nothing here."
            elif risk < -1e-6:
                verdict = (f"Safer route (max cell cost {risk:+.3f}) at "
                           f"{length:+.2f} m of length.")
            elif risk > 1e-6:
                verdict = (f"Shorter by {-length:.2f} m but riskier "
                           f"(max cell cost {risk:+.3f}).")
            else:
                verdict = f"Same risk, length {length:+.2f} m."

        return CounterfactualResult(
            parameter=parameter, baseline=baseline, alternative=alternative,
            differences=differences, verdict=verdict,
        )
