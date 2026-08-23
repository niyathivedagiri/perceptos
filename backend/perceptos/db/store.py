"""
perceptos.db.store
==================

SQLite persistence for missions, frames, stage timings and — the part that
matters — the **lineage graph**.

A mission you cannot replay the reasoning of is a mission you cannot review.
Storing frames and detections is easy and mostly useless on its own: six months
later the question is never "what did the rover see?" but "why did it turn
left?". That question is answerable only if the derivation graph survived, so
lineage nodes and edges are first-class tables here rather than a blob of JSON
in a column.

Design notes:

* **WAL journal mode.** The API reads while the pipeline writes. In the default
  rollback journal those block each other, and a telemetry poll can stall a
  frame. WAL lets readers proceed during a write.
* **Foreign keys are switched on explicitly.** SQLite ships with them disabled
  per-connection, so a schema full of ``REFERENCES`` clauses that nobody
  enabled enforces precisely nothing — a silent, extremely common mistake.
* **One connection per thread.** SQLite connections are not thread-safe by
  default and FastAPI serves from a thread pool, so connections are held in
  thread-local storage rather than shared.
* **Artefacts are not stored.** Frames and cost maps go to the artefact store
  on disk; the database keeps the reference. A megapixel image per row turns a
  mission log into tens of gigabytes.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

from ..core.lineage import LineageGraph
from ..core.stage import FrameContext

__all__ = ["MissionStore", "MissionRecord", "SCHEMA_VERSION"]

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (
    version     INTEGER NOT NULL,
    created_utc REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS missions (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    environment   TEXT NOT NULL,
    mode          TEXT NOT NULL,
    started_utc   REAL NOT NULL,
    ended_utc     REAL,
    status        TEXT NOT NULL DEFAULT 'running',
    data_mode     TEXT NOT NULL DEFAULT 'simulation',
    notes         TEXT DEFAULT '',
    config_json   TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS frames (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id     TEXT NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
    frame_index    INTEGER NOT NULL,
    captured_utc   REAL NOT NULL,
    total_latency_ms REAL NOT NULL,
    healthy        INTEGER NOT NULL,
    metadata_json  TEXT DEFAULT '{}',
    UNIQUE (mission_id, frame_index)
);
CREATE INDEX IF NOT EXISTS idx_frames_mission ON frames(mission_id, frame_index);

CREATE TABLE IF NOT EXISTS stage_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id     INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    stage        TEXT NOT NULL,
    status       TEXT NOT NULL,
    latency_ms   REAL NOT NULL,
    message      TEXT DEFAULT '',
    error        TEXT
);
CREATE INDEX IF NOT EXISTS idx_stage_frame ON stage_results(frame_id);

CREATE TABLE IF NOT EXISTS lineage_nodes (
    node_id       TEXT NOT NULL,
    mission_id    TEXT NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
    frame_index   INTEGER,
    label         TEXT NOT NULL,
    stage         TEXT DEFAULT '',
    source        TEXT NOT NULL,
    algorithm     TEXT NOT NULL,
    confidence    REAL,
    latency_ms    REAL,
    units         TEXT,
    timestamp_utc REAL NOT NULL,
    is_synthetic  INTEGER NOT NULL,
    value_json    TEXT DEFAULT 'null',
    params_json   TEXT DEFAULT '{}',
    artefact_key  TEXT,
    source_type   TEXT,
    source_url    TEXT,
    dataset       TEXT,
    uncertainty   REAL,
    PRIMARY KEY (mission_id, node_id)
);
CREATE INDEX IF NOT EXISTS idx_lineage_mission ON lineage_nodes(mission_id, frame_index);
CREATE INDEX IF NOT EXISTS idx_lineage_source  ON lineage_nodes(mission_id, source);

CREATE TABLE IF NOT EXISTS lineage_edges (
    mission_id TEXT NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
    parent_id  TEXT NOT NULL,
    child_id   TEXT NOT NULL,
    PRIMARY KEY (mission_id, parent_id, child_id)
);
CREATE INDEX IF NOT EXISTS idx_edges_child ON lineage_edges(mission_id, child_id);

CREATE TABLE IF NOT EXISTS telemetry (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id  TEXT NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
    recorded_utc REAL NOT NULL,
    channel     TEXT NOT NULL,
    source      TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_telemetry ON telemetry(mission_id, channel, recorded_utc);
"""


@dataclass(slots=True)
class MissionRecord:
    id: str
    name: str
    environment: str
    mode: str
    started_utc: float
    ended_utc: Optional[float]
    status: str
    data_mode: str
    notes: str = ""
    frames: int = 0
    lineage_nodes: int = 0

    @property
    def duration_s(self) -> Optional[float]:
        return (self.ended_utc - self.started_utc) if self.ended_utc else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "environment": self.environment,
            "mode": self.mode, "started_utc": self.started_utc,
            "ended_utc": self.ended_utc, "status": self.status,
            "data_mode": self.data_mode, "notes": self.notes,
            "frames": self.frames, "lineage_nodes": self.lineage_nodes,
            "duration_s": round(self.duration_s, 2) if self.duration_s else None,
        }


class MissionStore:
    """Thread-safe SQLite repository for mission data."""

    def __init__(self, path: str | Path = "missions.db") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # An in-memory database is per-connection, so a thread-local pool would
        # give every thread its own empty database. Share one connection there.
        self._shared: Optional[sqlite3.Connection] = None
        if self.path == ":memory:":
            self._shared = self._new_connection()
            self._lock = threading.Lock()
        else:
            self._lock = threading.Lock()
        self._initialise()

    # ------------------------------------------------------------ connections

    def _new_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False,
                               detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        # SQLite disables foreign keys per connection by default: a schema full
        # of REFERENCES clauses enforces nothing until this is switched on.
        conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    @property
    def _connection(self) -> sqlite3.Connection:
        if self._shared is not None:
            return self._shared
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._new_connection()
            self._local.conn = conn
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connection
        with self._lock:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _initialise(self) -> None:
        with self.transaction() as conn:
            conn.executescript(_SCHEMA)
            row = conn.execute("SELECT version FROM schema_info LIMIT 1").fetchone()
            if row is None:
                conn.execute("INSERT INTO schema_info VALUES (?, ?)",
                             (SCHEMA_VERSION, time.time()))
            elif row["version"] != SCHEMA_VERSION:
                raise RuntimeError(
                    f"database at {self.path} is schema v{row['version']}, "
                    f"this build expects v{SCHEMA_VERSION}. Migrate or start a "
                    "new database rather than writing incompatible rows."
                )

    def close(self) -> None:
        for conn in filter(None, [self._shared, getattr(self._local, "conn", None)]):
            conn.close()
        self._shared = None
        self._local.conn = None

    # --------------------------------------------------------------- missions

    def start_mission(self, *, name: str, environment: str, mode: str,
                      data_mode: str = "simulation", notes: str = "",
                      config: Optional[dict[str, Any]] = None) -> str:
        mission_id = uuid.uuid4().hex[:16]
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO missions (id, name, environment, mode, started_utc, "
                "status, data_mode, notes, config_json) VALUES (?,?,?,?,?,?,?,?,?)",
                (mission_id, name, environment, mode, time.time(), "running",
                 data_mode, notes, json.dumps(config or {})),
            )
        return mission_id

    def end_mission(self, mission_id: str, *, status: str = "complete") -> None:
        with self.transaction() as conn:
            conn.execute("UPDATE missions SET ended_utc = ?, status = ? WHERE id = ?",
                         (time.time(), status, mission_id))

    def get_mission(self, mission_id: str) -> Optional[MissionRecord]:
        conn = self._connection
        row = conn.execute("SELECT * FROM missions WHERE id = ?", (mission_id,)).fetchone()
        if row is None:
            return None
        frames = conn.execute("SELECT COUNT(*) c FROM frames WHERE mission_id = ?",
                              (mission_id,)).fetchone()["c"]
        nodes = conn.execute("SELECT COUNT(*) c FROM lineage_nodes WHERE mission_id = ?",
                             (mission_id,)).fetchone()["c"]
        return self._to_record(row, frames, nodes)

    def list_missions(self, *, limit: int = 50) -> list[MissionRecord]:
        rows = self._connection.execute(
            "SELECT m.*, "
            "(SELECT COUNT(*) FROM frames f WHERE f.mission_id = m.id) AS frame_count, "
            "(SELECT COUNT(*) FROM lineage_nodes l WHERE l.mission_id = m.id) AS node_count "
            "FROM missions m ORDER BY m.started_utc DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [self._to_record(r, r["frame_count"], r["node_count"]) for r in rows]

    @staticmethod
    def _to_record(row: sqlite3.Row, frames: int, nodes: int) -> MissionRecord:
        return MissionRecord(
            id=row["id"], name=row["name"], environment=row["environment"],
            mode=row["mode"], started_utc=row["started_utc"],
            ended_utc=row["ended_utc"], status=row["status"],
            data_mode=row["data_mode"], notes=row["notes"] or "",
            frames=frames, lineage_nodes=nodes,
        )

    def delete_mission(self, mission_id: str) -> bool:
        with self.transaction() as conn:
            cursor = conn.execute("DELETE FROM missions WHERE id = ?", (mission_id,))
            return cursor.rowcount > 0

    # ----------------------------------------------------------------- frames

    def record_frame(self, mission_id: str, ctx: FrameContext,
                     report: Any) -> int:
        """Persist one processed frame: timings, stage results and lineage."""
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT OR REPLACE INTO frames (mission_id, frame_index, captured_utc, "
                "total_latency_ms, healthy, metadata_json) VALUES (?,?,?,?,?,?)",
                (mission_id, ctx.frame_id, ctx.created_at,
                 float(report.total_latency_ms), int(bool(report.healthy)),
                 json.dumps(_safe(ctx.metadata))),
            )
            frame_row_id = int(cursor.lastrowid)
            conn.executemany(
                "INSERT INTO stage_results (frame_id, stage, status, latency_ms, "
                "message, error) VALUES (?,?,?,?,?,?)",
                [(frame_row_id, s.stage, s.status.value, float(s.latency_ms),
                  s.message or "", s.error) for s in report.stages],
            )
            self._write_lineage(conn, mission_id, ctx.lineage, ctx.frame_id)
        return frame_row_id

    def _write_lineage(self, conn: sqlite3.Connection, mission_id: str,
                       lineage: LineageGraph, frame_index: Optional[int]) -> None:
        payload = lineage.to_dict()["nodes"]
        node_rows, edge_rows = [], []
        for node in payload:
            prov = node["provenance"]
            node_rows.append((
                node["id"], mission_id, frame_index, node["label"], node["stage"],
                prov["source"], prov["algorithm"], prov["confidence"],
                prov["latency_ms"], prov["units"], prov["timestamp_utc"],
                int(bool(prov["is_synthetic"])), json.dumps(_safe(node["value"])),
                json.dumps(_safe(prov["parameters"])), node["artefact_key"],
                prov.get("source_type"), prov.get("source_url"),
                prov.get("dataset"), prov.get("uncertainty"),
            ))
            edge_rows.extend((mission_id, parent, node["id"])
                             for parent in node["parents"])
        # INSERT OR IGNORE: lineage graphs are append-only and a frame may share
        # ancestors with an earlier one (a loaded calibration, for instance).
        conn.executemany(
            "INSERT OR IGNORE INTO lineage_nodes VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            node_rows,
        )
        conn.executemany("INSERT OR IGNORE INTO lineage_edges VALUES (?,?,?)", edge_rows)

    def get_frames(self, mission_id: str, *, limit: int = 200,
                   offset: int = 0) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT * FROM frames WHERE mission_id = ? ORDER BY frame_index "
            "LIMIT ? OFFSET ?", (mission_id, int(limit), int(offset)),
        ).fetchall()
        return [
            {"frame_index": r["frame_index"], "captured_utc": r["captured_utc"],
             "total_latency_ms": r["total_latency_ms"], "healthy": bool(r["healthy"]),
             "metadata": json.loads(r["metadata_json"] or "{}")}
            for r in rows
        ]

    def get_frame_detail(self, mission_id: str, frame_index: int
                         ) -> Optional[dict[str, Any]]:
        conn = self._connection
        row = conn.execute(
            "SELECT * FROM frames WHERE mission_id = ? AND frame_index = ?",
            (mission_id, int(frame_index)),
        ).fetchone()
        if row is None:
            return None
        stages = conn.execute(
            "SELECT stage, status, latency_ms, message, error FROM stage_results "
            "WHERE frame_id = ? ORDER BY id", (row["id"],),
        ).fetchall()
        return {
            "frame_index": row["frame_index"],
            "captured_utc": row["captured_utc"],
            "total_latency_ms": row["total_latency_ms"],
            "healthy": bool(row["healthy"]),
            "metadata": json.loads(row["metadata_json"] or "{}"),
            "stages": [dict(s) for s in stages],
        }

    # ---------------------------------------------------------------- lineage

    def explain(self, mission_id: str, node_id: str) -> Optional[dict[str, Any]]:
        """Reconstruct a derivation chain from the database.

        This is the persisted twin of :meth:`LineageGraph.explain` — the same
        payload, answerable months later from a mission file rather than only
        from the live process.
        """
        conn = self._connection
        node = conn.execute(
            "SELECT * FROM lineage_nodes WHERE mission_id = ? AND node_id = ?",
            (mission_id, node_id),
        ).fetchone()
        if node is None:
            return None

        # Recursive CTE walks the ancestor closure in one query rather than
        # issuing one round trip per level.
        chain = conn.execute(
            """
            WITH RECURSIVE ancestry(node_id) AS (
                SELECT ?
                UNION
                SELECT e.parent_id FROM lineage_edges e
                JOIN ancestry a ON e.child_id = a.node_id
                WHERE e.mission_id = ?
            )
            SELECT n.* FROM lineage_nodes n
            JOIN ancestry a ON n.node_id = a.node_id
            WHERE n.mission_id = ?
            ORDER BY n.timestamp_utc
            """,
            (node_id, mission_id, mission_id),
        ).fetchall()

        nodes = [self._node_dict(r) for r in chain]
        latencies = [n["provenance"]["latency_ms"] for n in nodes
                     if n["provenance"]["latency_ms"] is not None]
        confidences = [n["provenance"]["confidence"] for n in nodes
                       if n["provenance"]["confidence"] is not None]
        parents_of = {}
        for row in conn.execute(
            "SELECT parent_id, child_id FROM lineage_edges WHERE mission_id = ?",
            (mission_id,),
        ).fetchall():
            parents_of.setdefault(row["child_id"], []).append(row["parent_id"])
        for entry in nodes:
            entry["parents"] = parents_of.get(entry["id"], [])

        return {
            "mission_id": mission_id,
            "node": self._node_dict(node),
            "chain": nodes,
            "roots": [n for n in nodes if not n["parents"]],
            "algorithms": [n["provenance"]["algorithm"] for n in nodes],
            "depth": len(nodes),
            "total_latency_ms": round(sum(latencies), 3) if latencies else None,
            "weakest_confidence": min(confidences) if confidences else None,
            "contains_simulation": any(n["provenance"]["source"] == "simulation"
                                       for n in nodes),
            "datasets": sorted({n["provenance"]["dataset"] for n in nodes
                                if n["provenance"].get("dataset")}),
            "source_urls": sorted({n["provenance"]["source_url"] for n in nodes
                                   if n["provenance"].get("source_url")}),
            "source_types": sorted({n["provenance"]["source_type"] for n in nodes
                                    if n["provenance"].get("source_type")}),
        }

    def find_nodes(self, mission_id: str, *, label: Optional[str] = None,
                   source: Optional[str] = None, frame_index: Optional[int] = None,
                   limit: int = 100) -> list[dict[str, Any]]:
        clauses = ["mission_id = ?"]
        params: list[Any] = [mission_id]
        if label:
            clauses.append("label LIKE ?")
            params.append(f"%{label}%")
        if source:
            clauses.append("source = ?")
            params.append(source)
        if frame_index is not None:
            clauses.append("frame_index = ?")
            params.append(int(frame_index))
        params.append(int(limit))
        rows = self._connection.execute(
            f"SELECT * FROM lineage_nodes WHERE {' AND '.join(clauses)} "
            "ORDER BY timestamp_utc DESC LIMIT ?", params,
        ).fetchall()
        return [self._node_dict(r) for r in rows]

    @staticmethod
    def _node_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["node_id"],
            "label": row["label"],
            "stage": row["stage"],
            "frame_index": row["frame_index"],
            "parents": [],
            "value": json.loads(row["value_json"] or "null"),
            "artefact_key": row["artefact_key"],
            "provenance": {
                "source": row["source"],
                "algorithm": row["algorithm"],
                "confidence": row["confidence"],
                "latency_ms": row["latency_ms"],
                "units": row["units"],
                "timestamp_utc": row["timestamp_utc"],
                "is_synthetic": bool(row["is_synthetic"]),
                "parameters": json.loads(row["params_json"] or "{}"),
                "source_type": row["source_type"],
                "source_url": row["source_url"],
                "dataset": row["dataset"],
                "uncertainty": row["uncertainty"],
            },
        }

    # -------------------------------------------------------------- telemetry

    def record_telemetry(self, mission_id: str, channel: str, payload: Any, *,
                         source: str = "calculated") -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO telemetry (mission_id, recorded_utc, channel, source, "
                "payload_json) VALUES (?,?,?,?,?)",
                (mission_id, time.time(), channel, source, json.dumps(_safe(payload))),
            )

    def get_telemetry(self, mission_id: str, channel: str, *,
                      limit: int = 500) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT recorded_utc, source, payload_json FROM telemetry "
            "WHERE mission_id = ? AND channel = ? ORDER BY recorded_utc DESC LIMIT ?",
            (mission_id, channel, int(limit)),
        ).fetchall()
        return [
            {"recorded_utc": r["recorded_utc"], "source": r["source"],
             "payload": json.loads(r["payload_json"])}
            for r in reversed(rows)
        ]

    # ------------------------------------------------------------- statistics

    def statistics(self) -> dict[str, Any]:
        conn = self._connection
        def count(table: str) -> int:
            return conn.execute(f"SELECT COUNT(*) c FROM {table}").fetchone()["c"]
        size = Path(self.path).stat().st_size if self.path != ":memory:" \
            and Path(self.path).exists() else 0
        return {
            "path": self.path,
            "schema_version": SCHEMA_VERSION,
            "missions": count("missions"),
            "frames": count("frames"),
            "lineage_nodes": count("lineage_nodes"),
            "lineage_edges": count("lineage_edges"),
            "telemetry_samples": count("telemetry"),
            "size_bytes": size,
        }


def _safe(value: Any) -> Any:
    """Coerce to something ``json.dumps`` will accept, without raising."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    for attr in ("to_dict", "tolist", "item"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return _safe(method())
            except Exception:  # noqa: BLE001
                pass
    return repr(value)[:512]
