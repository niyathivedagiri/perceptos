"""
perceptos.reports.mission
=========================

Mission reports — JSON, CSV and Markdown.

A report that states conclusions without their derivation is a summary, not a
record. Every headline figure here carries the lineage node it came from, so a
reader six months later can ask the running system where a number originated
and get the same answer the operator saw at the time.

Three formats, chosen for what they are actually used for:

* **JSON** — the machine-readable archive. Complete, including full lineage.
* **CSV** — per-frame timings, for someone who wants to plot latency in a
  spreadsheet without writing a parser.
* **Markdown** — the human-readable summary, renderable to PDF by any of the
  usual tools without PerceptOS taking on a PDF dependency it does not need.

**The data-credibility policy applies to reports too.** A mission run on
synthetic inputs is labelled SIMULATION in the report header, in the summary
table and against every affected figure. A report is the artefact most likely
to outlive the context that produced it, so it is the one where an unlabelled
synthetic number would do the most damage.
"""

from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..db.store import MissionStore

__all__ = ["MissionReport", "ReportBuilder"]


@dataclass(slots=True)
class MissionReport:
    """A generated report in every supported representation."""

    mission_id: str
    payload: dict[str, Any]
    generated_utc: float

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.payload, indent=indent, sort_keys=False)

    def to_csv(self) -> str:
        """Per-frame timings and stage statuses."""
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["frame_index", "latency_ms", "healthy",
                         "stage", "stage_status", "stage_latency_ms"])
        for frame in self.payload.get("frames", []):
            stages = frame.get("stages") or [{}]
            for stage in stages:
                writer.writerow([
                    frame.get("frame_index"),
                    round(frame.get("total_latency_ms", 0.0), 3),
                    frame.get("healthy"),
                    stage.get("stage", ""),
                    stage.get("status", ""),
                    round(stage.get("latency_ms", 0.0), 3),
                ])
        return buffer.getvalue()

    def to_markdown(self) -> str:
        p = self.payload
        mission = p["mission"]
        credibility = p["credibility"]
        lines = [
            f"# Mission report — {mission['name']}",
            "",
            f"**Mission ID:** `{self.mission_id}`  ",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(self.generated_utc))}",
            "",
        ]

        # The credibility banner comes first and is unmissable. A report is the
        # artefact most likely to be read out of context.
        if credibility["contains_simulation"]:
            lines += [
                "> ## ⚠ SIMULATION",
                "> This mission ran on synthetic inputs. Every figure below "
                "derives from generated data and describes no physical event.",
                "",
            ]
        else:
            lines += [
                f"> **Data sources:** {', '.join(credibility['source_types']) or 'unrecorded'}",
                "",
            ]

        lines += [
            "## Summary", "",
            "| Field | Value |", "|---|---|",
            f"| Environment | {mission['environment']} |",
            f"| Status | {mission['status']} |",
            f"| Duration | {mission['duration_s']} s |",
            f"| Frames processed | {mission['frames']} |",
            f"| Lineage nodes | {mission['lineage_nodes']} |",
            f"| Data mode | {mission['data_mode']} |",
            "",
            "## Performance", "",
            "| Metric | Value |", "|---|---|",
        ]
        performance = p["performance"]
        for key, label in (("mean_latency_ms", "Mean frame latency"),
                           ("p95_latency_ms", "95th percentile latency"),
                           ("max_latency_ms", "Worst frame latency"),
                           ("unhealthy_frames", "Frames with a failed stage")):
            lines.append(f"| {label} | {performance.get(key)} |")
        lines.append("")

        if p["stages"]:
            lines += ["## Stage profile", "",
                      "| Stage | Mean (ms) | Max (ms) | Statuses |", "|---|---|---|---|"]
            for stage in p["stages"]:
                statuses = ", ".join(f"{k}×{v}" for k, v in stage["statuses"].items())
                lines.append(f"| {stage['stage']} | {stage['mean_latency_ms']} "
                             f"| {stage['max_latency_ms']} | {statuses} |")
            lines.append("")

        lines += ["## Data provenance", "",
                  "| Source class | Nodes |", "|---|---|"]
        for source, count in sorted(credibility["source_counts"].items()):
            lines.append(f"| `{source}` | {count} |")
        lines.append("")

        if p["key_values"]:
            lines += [
                "## Key values and their derivation", "",
                "Every figure below is traceable: query "
                "`GET /api/lineage/{node_id}` for the full chain.", "",
                "| Value | Source | Confidence | Algorithm | Node |",
                "|---|---|---|---|---|",
            ]
            for value in p["key_values"]:
                provenance = value["provenance"]
                confidence = provenance.get("confidence")
                lines.append(
                    f"| {value['label']} | `{provenance.get('source_type') or provenance['source']}` "
                    f"| {round(confidence, 3) if confidence is not None else '—'} "
                    f"| {provenance['algorithm'][:44]} | `{value['id'][:12]}` |"
                )
            lines.append("")

        if p.get("limitations"):
            lines += ["## Limitations", ""]
            lines += [f"- {item}" for item in p["limitations"]]
            lines.append("")

        lines += ["---", "",
                  "Generated by PerceptOS. Values labelled `SIMULATION` are "
                  "synthetic; values labelled `UNAVAILABLE` were not measured "
                  "and have not been substituted."]
        return "\n".join(lines)


class ReportBuilder:
    """Assembles a mission report from the database."""

    #: Values worth surfacing as headline figures if the mission produced them.
    KEY_LABELS = ("safe_path_plan", "traversability_map", "dem_elevation",
                  "depth_map", "fused_pose", "slam_trajectory",
                  "rectified_stereo_pair", "wheel_slip")

    def __init__(self, store: MissionStore) -> None:
        self.store = store

    def build(self, mission_id: str, *, max_frames: int = 5000) -> MissionReport:
        record = self.store.get_mission(mission_id)
        if record is None:
            raise KeyError(f"no mission {mission_id!r}")

        frames = self.store.get_frames(mission_id, limit=max_frames)
        detailed = []
        for frame in frames:
            detail = self.store.get_frame_detail(mission_id, frame["frame_index"])
            if detail is not None:
                detailed.append(detail)

        latencies = [f["total_latency_ms"] for f in frames]
        unhealthy = [f["frame_index"] for f in frames if not f["healthy"]]

        nodes = self.store.find_nodes(mission_id, limit=20_000)
        source_counts: dict[str, int] = {}
        source_types: set[str] = set()
        for node in nodes:
            provenance = node["provenance"]
            source_counts[provenance["source"]] = (
                source_counts.get(provenance["source"], 0) + 1)
            if provenance.get("source_type"):
                source_types.add(provenance["source_type"])

        key_values = [n for n in nodes if n["label"] in self.KEY_LABELS]
        # One representative per label: a 500-frame mission would otherwise
        # produce 500 identical rows and bury the summary.
        seen: set[str] = set()
        unique_keys = []
        for node in key_values:
            if node["label"] in seen:
                continue
            seen.add(node["label"])
            unique_keys.append(node)

        limitations = self._limitations(nodes, source_types)

        payload = {
            "report_version": 1,
            "generated_utc": time.time(),
            "mission": record.to_dict(),
            "credibility": {
                "contains_simulation": source_counts.get("simulation", 0) > 0,
                "source_counts": source_counts,
                "source_types": sorted(source_types),
                "datasets": sorted({n["provenance"]["dataset"] for n in nodes
                                    if n["provenance"].get("dataset")}),
            },
            "performance": {
                "frames": len(frames),
                "mean_latency_ms": (round(float(np.mean(latencies)), 3)
                                    if latencies else None),
                "p95_latency_ms": (round(float(np.percentile(latencies, 95)), 3)
                                   if latencies else None),
                "max_latency_ms": (round(float(np.max(latencies)), 3)
                                   if latencies else None),
                "unhealthy_frames": len(unhealthy),
                "unhealthy_frame_indices": unhealthy[:50],
            },
            "stages": self._stage_profile(detailed),
            "key_values": unique_keys,
            "frames": detailed,
            "limitations": limitations,
        }
        return MissionReport(mission_id=mission_id, payload=payload,
                             generated_utc=payload["generated_utc"])

    @staticmethod
    def _stage_profile(detailed: list[dict[str, Any]]) -> list[dict[str, Any]]:
        totals: dict[str, list[float]] = {}
        statuses: dict[str, dict[str, int]] = {}
        for frame in detailed:
            for stage in frame.get("stages", []):
                totals.setdefault(stage["stage"], []).append(stage["latency_ms"])
                bucket = statuses.setdefault(stage["stage"], {})
                bucket[stage["status"]] = bucket.get(stage["status"], 0) + 1
        profile = [
            {"stage": name,
             "mean_latency_ms": round(float(np.mean(values)), 3),
             "max_latency_ms": round(float(np.max(values)), 3),
             "total_ms": round(float(np.sum(values)), 3),
             "statuses": statuses.get(name, {})}
            for name, values in totals.items()
        ]
        profile.sort(key=lambda entry: -entry["total_ms"])
        return profile

    @staticmethod
    def _limitations(nodes: list[dict[str, Any]],
                     source_types: set[str]) -> list[str]:
        """State what this mission's data does *not* support concluding.

        Generated from what was recorded rather than written by hand, so it
        cannot drift out of step with the run it describes.
        """
        limitations: list[str] = []
        labels = {n["label"] for n in nodes}

        if any(n["provenance"]["source"] == "simulation" for n in nodes):
            limitations.append(
                "This mission contains simulated data. Figures derived from it "
                "describe generated inputs, not a physical event.")
        if "SIMULATION" in source_types and len(source_types) > 1:
            limitations.append(
                "Real and simulated sources are both present. Check each "
                "figure's source label individually; the mission total is not "
                "uniformly one or the other.")
        if "depth_map" not in labels:
            limitations.append(
                "No depth was computed, so no distance in this report is a "
                "stereo measurement.")
        if "slam_trajectory" not in labels:
            limitations.append(
                "No SLAM ran, so any trajectory here is dead reckoning with "
                "unbounded drift.")
        if not any(n["provenance"].get("dataset") for n in nodes):
            limitations.append(
                "No named dataset is recorded against any value in this "
                "mission.")
        return limitations
