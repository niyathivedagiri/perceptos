"""
perceptos.core.lineage
======================

The Data Lineage drawer, server side.

Every :class:`~perceptos.core.provenance.Traced` value produced during a run is
registered here.  Because each value carries its parents' ids, the recorder can
answer the only question that matters in an autonomy review:

    "The rover decided to steer left.  *Why?*"

:meth:`LineageGraph.explain` walks backwards from any node id and returns the
full derivation chain — raw frame, every processing step, every algorithm,
intermediate results, confidences and latencies — which is exactly the payload
the frontend drawer renders.

Artefacts (raw frames, disparity maps, cost maps) are *not* stored in the graph.
They are written to an artefact store and referenced by key, so a long mission
does not exhaust memory.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from .provenance import Source, Traced

__all__ = ["LineageGraph", "LineageNode", "ArtefactStore"]


@dataclass(slots=True)
class LineageNode:
    """A single recorded value in the derivation graph."""

    id: str
    label: str
    stage: str
    parents: tuple[str, ...]
    provenance: dict[str, Any]
    summary: Any
    artefact_key: Optional[str] = None
    children: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "stage": self.stage,
            "parents": list(self.parents),
            "children": list(self.children),
            "provenance": self.provenance,
            "value": self.summary,
            "artefact_key": self.artefact_key,
        }


class ArtefactStore:
    """Bounded LRU cache of heavyweight intermediates (frames, maps, clouds).

    Keeps the last ``capacity`` artefacts so the lineage drawer can show the
    actual raw image behind a detection, without an unbounded memory footprint
    during a multi-hour traverse.
    """

    def __init__(self, capacity: int = 256) -> None:
        self.capacity = int(capacity)
        self._items: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def put(self, key: str, artefact: Any) -> str:
        with self._lock:
            self._items[key] = artefact
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return key

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._items:
                return None
            self._items.move_to_end(key)
            return self._items[key]

    def __len__(self) -> int:
        return len(self._items)


class LineageGraph:
    """Append-only DAG of every traced value in a run."""

    def __init__(self, *, artefact_capacity: int = 256) -> None:
        self._nodes: dict[str, LineageNode] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()
        self.artefacts = ArtefactStore(artefact_capacity)

    # ------------------------------------------------------------------ write

    def record(
        self,
        value: Traced[Any],
        *,
        stage: str = "",
        artefact: Any = None,
        label: Optional[str] = None,
    ) -> str:
        """Register a traced value.  Returns its node id."""
        with self._lock:
            if value.id in self._nodes:
                return value.id
            artefact_key = None
            if artefact is not None:
                artefact_key = self.artefacts.put(f"art::{value.id}", artefact)
            node = LineageNode(
                id=value.id,
                label=label or value.label or value.provenance.algorithm,
                stage=stage,
                parents=value.parents,
                provenance=value.provenance.to_dict(),
                summary=value.to_dict()["value"],
                artefact_key=artefact_key,
            )
            self._nodes[node.id] = node
            self._order.append(node.id)
            for parent_id in node.parents:
                parent = self._nodes.get(parent_id)
                if parent is not None and node.id not in parent.children:
                    parent.children.append(node.id)
            return node.id

    def record_many(self, values: Iterable[Traced[Any]], *, stage: str = "") -> list[str]:
        return [self.record(v, stage=stage) for v in values]

    # ------------------------------------------------------------------- read

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    def node(self, node_id: str) -> Optional[LineageNode]:
        return self._nodes.get(node_id)

    def ancestors(self, node_id: str) -> list[LineageNode]:
        """All upstream nodes, ordered root-first (topologically sane)."""
        seen: set[str] = set()
        chain: list[LineageNode] = []

        def walk(nid: str) -> None:
            if nid in seen:
                return
            seen.add(nid)
            node = self._nodes.get(nid)
            if node is None:
                return
            for parent_id in node.parents:
                walk(parent_id)
            chain.append(node)

        walk(node_id)
        return chain

    def explain(self, node_id: str) -> dict[str, Any]:
        """Full lineage payload for one value — what the drawer renders.

        Includes the derivation chain, the total latency accumulated along it,
        the weakest confidence anywhere upstream (the real trustworthiness of
        the number), and whether any simulated input contaminated the result.
        """
        node = self._nodes.get(node_id)
        if node is None:
            raise KeyError(f"unknown lineage node: {node_id!r}")

        chain = self.ancestors(node_id)
        latencies = [
            n.provenance.get("latency_ms")
            for n in chain
            if n.provenance.get("latency_ms") is not None
        ]
        confidences = [
            n.provenance.get("confidence")
            for n in chain
            if n.provenance.get("confidence") is not None
        ]
        roots = [n for n in chain if not n.parents]

        # The external origins of this value: the datasets and URLs at the
        # roots of the chain. This is what lets the drawer say "this path came
        # from the Jezero DEM" without the caller tracking it separately.
        datasets = sorted({n.provenance.get("dataset") for n in chain
                           if n.provenance.get("dataset")})
        source_urls = sorted({n.provenance.get("source_url") for n in chain
                              if n.provenance.get("source_url")})
        source_types = sorted({n.provenance.get("source_type") for n in chain
                               if n.provenance.get("source_type")})

        return {
            "node": node.to_dict(),
            "chain": [n.to_dict() for n in chain],
            "roots": [n.to_dict() for n in roots],
            "algorithms": [n.provenance["algorithm"] for n in chain],
            "datasets": datasets,
            "source_urls": source_urls,
            "source_types": source_types,
            "depth": len(chain),
            "total_latency_ms": round(sum(latencies), 3) if latencies else None,
            "weakest_confidence": min(confidences) if confidences else None,
            "contains_simulation": any(
                n.provenance.get("source") == Source.SIMULATION.value for n in chain
            ),
            "mermaid": self.to_mermaid(node_id),
        }

    def to_mermaid(self, node_id: Optional[str] = None) -> str:
        """Render the (sub)graph as Mermaid — used for docs and the drawer."""
        chain = self.ancestors(node_id) if node_id else list(self._nodes.values())
        lines = ["graph TD"]
        ids = {n.id for n in chain}
        for n in chain:
            safe = n.label.replace('"', "'")
            conf = n.provenance.get("confidence")
            suffix = f"<br/>conf {conf:.2f}" if isinstance(conf, (int, float)) else ""
            lines.append(f'  {n.id}["{safe}<br/>{n.provenance["source"]}{suffix}"]')
        for n in chain:
            for p in n.parents:
                if p in ids:
                    lines.append(f"  {p} --> {n.id}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [self._nodes[i].to_dict() for i in self._order],
            "count": len(self._nodes),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    def find(self, predicate: Callable[[LineageNode], bool]) -> list[LineageNode]:
        return [self._nodes[i] for i in self._order if predicate(self._nodes[i])]
