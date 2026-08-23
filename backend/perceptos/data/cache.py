"""
perceptos.data.cache
====================

Dataset acquisition primitives: an injectable HTTP transport and a
content-addressed cache that records where every byte came from.

Two design choices carry the weight here.

**The transport is injected, not imported.** Every client in this package takes
a :class:`HttpTransport`. In production that is :class:`UrllibTransport`; in
tests it is a fake that serves canned responses. This means the NASA clients
are fully testable with no network, no API key and no rate limit — which
matters because a test suite that silently depends on a remote service is a
test suite that fails on a train.

**Nothing is cached without provenance.** Every download writes a manifest
entry recording the URL, SHA-256, byte count, retrieval time, licence and
citation, and every dataset directory gets a ``SOURCE.md``. A project whose
entire argument is "every value declares its origin" cannot have a
``datasets/`` folder full of anonymous files.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

__all__ = [
    "HttpTransport",
    "UrllibTransport",
    "FakeTransport",
    "HttpError",
    "RateLimited",
    "DatasetCache",
    "CachedAsset",
    "DatasetLicence",
]


class HttpError(RuntimeError):
    """A request failed. Carries the status code where one exists."""

    def __init__(self, message: str, *, status: Optional[int] = None,
                 url: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.url = url


class RateLimited(HttpError):
    """The remote service asked us to slow down (HTTP 429)."""


class HttpTransport(Protocol):
    """Minimal transport interface. Implement this to swap in a fake."""

    def get_json(self, url: str, params: Mapping[str, Any]) -> dict[str, Any]: ...

    def get_bytes(self, url: str) -> bytes: ...


class UrllibTransport:
    """Standard-library HTTP. No third-party dependency, works everywhere.

    Retries on 429 and 5xx with exponential backoff, because NASA's API is
    free, shared, and will rate-limit a bulk download. Failing the whole fetch
    on the first 429 makes the downloader useless for the job it exists to do.
    """

    def __init__(self, *, timeout_s: float = 30.0, max_retries: int = 4,
                 user_agent: str = "PerceptOS/0.3 (robotics perception platform)",
                 backoff_base_s: float = 1.5) -> None:
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.user_agent = user_agent
        self.backoff_base_s = float(backoff_base_s)
        self.requests_made = 0

    def _open(self, url: str) -> bytes:
        last: Optional[Exception] = None
        for attempt in range(self.max_retries):
            request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
            try:
                self.requests_made += 1
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code == 429:
                    last = RateLimited(f"rate limited by {url}", status=429, url=url)
                elif exc.code < 500:
                    raise HttpError(f"HTTP {exc.code} for {url}", status=exc.code,
                                    url=url) from exc
            except urllib.error.URLError as exc:
                last = HttpError(f"network error for {url}: {exc.reason}", url=url)
            if attempt < self.max_retries - 1:
                time.sleep(self.backoff_base_s ** (attempt + 1))
        raise last if isinstance(last, HttpError) else HttpError(
            f"failed after {self.max_retries} attempts: {url}", url=url
        )

    def get_json(self, url: str, params: Mapping[str, Any]) -> dict[str, Any]:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        full = f"{url}?{query}" if query else url
        return json.loads(self._open(full).decode("utf-8"))

    def get_bytes(self, url: str) -> bytes:
        return self._open(url)


class FakeTransport:
    """Deterministic transport for tests. Serves canned responses by URL prefix."""

    def __init__(self, json_routes: Optional[Mapping[str, Any]] = None,
                 byte_routes: Optional[Mapping[str, bytes]] = None) -> None:
        self.json_routes = dict(json_routes or {})
        self.byte_routes = dict(byte_routes or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_json(self, url: str, params: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append((url, dict(params)))
        for prefix, payload in self.json_routes.items():
            if url.startswith(prefix):
                return payload(params) if callable(payload) else payload
        raise HttpError(f"no fake route for {url}", status=404, url=url)

    def get_bytes(self, url: str) -> bytes:
        self.calls.append((url, {}))
        for prefix, payload in self.byte_routes.items():
            if url.startswith(prefix):
                return payload
        raise HttpError(f"no fake byte route for {url}", status=404, url=url)


@dataclass(frozen=True, slots=True)
class DatasetLicence:
    """Licence and citation for a dataset. Required, not optional."""

    name: str
    url: str = ""
    citation: str = ""
    credit: str = ""
    commercial_use: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "url": self.url, "citation": self.citation,
            "credit": self.credit, "commercial_use": self.commercial_use,
        }


#: Licences for the sources PerceptOS ships fetchers for.
LICENCES: dict[str, DatasetLicence] = {
    "nasa_pd": DatasetLicence(
        name="Public Domain (NASA)",
        url="https://www.nasa.gov/nasa-brand-center/images-and-media/",
        credit="NASA/JPL-Caltech",
        commercial_use=True,
    ),
    "usgs_pd": DatasetLicence(
        name="Public Domain (USGS)",
        url="https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits",
        credit="NASA/JPL-Caltech/USGS",
        commercial_use=True,
    ),
    "ai4mars": DatasetLicence(
        name="Open (cite the paper)",
        url="https://zenodo.org/records/15995036",
        citation=(
            "Swan, R. M., et al. AI4MARS: A Dataset for Terrain-Aware Autonomous "
            "Driving on Mars. CVPRW 2021. doi:10.1109/CVPRW53098.2021.00226"
        ),
        credit="NASA/JPL-Caltech",
        commercial_use=True,
    ),
}


@dataclass(frozen=True, slots=True)
class CachedAsset:
    """One file in the cache, with everything needed to justify trusting it."""

    path: Path
    url: str
    sha256: str
    bytes: int
    retrieved_utc: float
    dataset: str
    licence: DatasetLicence
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "url": self.url,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "retrieved_utc": self.retrieved_utc,
            "dataset": self.dataset,
            "licence": self.licence.to_dict(),
            "metadata": dict(self.metadata),
        }

    def provenance_parameters(self) -> dict[str, Any]:
        """Fields to embed in a Traced value's provenance."""
        return {
            "dataset": self.dataset,
            "source_url": self.url,
            "sha256": self.sha256[:16],
            "credit": self.licence.credit,
            "retrieved_utc": self.retrieved_utc,
            **dict(self.metadata),
        }


class DatasetCache:
    """Content-addressed download cache with a per-dataset manifest.

    Files land at ``<root>/<dataset>/<name>``. The manifest is keyed by URL, so
    a repeat fetch is a no-op and the whole corpus is reproducible from the
    manifest alone — which is the point: the repository ships the recipe, not
    the 18 GB.
    """

    MANIFEST = "manifest.json"

    def __init__(self, root: str | Path = "datasets", *,
                 transport: Optional[HttpTransport] = None,
                 offline: bool = False) -> None:
        self.root = Path(root)
        self.transport = transport if transport is not None else UrllibTransport()
        self.offline = bool(offline)
        self.downloads = 0
        self.hits = 0

    # ------------------------------------------------------------------ paths

    def dataset_dir(self, dataset: str) -> Path:
        path = self.root / dataset
        path.mkdir(parents=True, exist_ok=True)
        return path

    def manifest_path(self, dataset: str) -> Path:
        return self.dataset_dir(dataset) / self.MANIFEST

    def load_manifest(self, dataset: str) -> dict[str, Any]:
        path = self.manifest_path(dataset)
        if not path.exists():
            return {"dataset": dataset, "assets": {}}
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            # A truncated manifest must not destroy the cache; rebuild it.
            return {"dataset": dataset, "assets": {}}

    def _save_manifest(self, dataset: str, manifest: Mapping[str, Any]) -> None:
        self.manifest_path(dataset).write_text(json.dumps(manifest, indent=2, sort_keys=True))

    # -------------------------------------------------------------- retrieval

    def fetch(self, url: str, *, dataset: str, name: str,
              licence: DatasetLicence,
              metadata: Optional[Mapping[str, Any]] = None,
              expected_sha256: Optional[str] = None) -> CachedAsset:
        """Download ``url`` unless it is already cached. Returns the asset."""
        manifest = self.load_manifest(dataset)
        entry = manifest["assets"].get(url)
        target = self.dataset_dir(dataset) / name

        if entry is not None and target.exists():
            # Verify rather than assume: a half-written file from an
            # interrupted download is the classic silent corruption.
            if self._sha256_file(target) == entry["sha256"]:
                self.hits += 1
                return CachedAsset(
                    path=target, url=url, sha256=entry["sha256"], bytes=entry["bytes"],
                    retrieved_utc=entry["retrieved_utc"], dataset=dataset,
                    licence=licence, metadata=entry.get("metadata", {}),
                )

        if self.offline:
            raise HttpError(
                f"offline mode: {name!r} is not cached. Run the fetcher with "
                "network access first, or use datasets/samples/.",
                url=url,
            )

        payload = self.transport.get_bytes(url)
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 and digest != expected_sha256:
            raise HttpError(
                f"checksum mismatch for {url}\n  expected {expected_sha256}\n"
                f"  got      {digest}\nThe remote file changed or the download "
                "was corrupted; the manifest is now untrustworthy.",
                url=url,
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        self.downloads += 1

        asset = CachedAsset(
            path=target, url=url, sha256=digest, bytes=len(payload),
            retrieved_utc=time.time(), dataset=dataset, licence=licence,
            metadata=dict(metadata or {}),
        )
        manifest["assets"][url] = {
            "name": name, "sha256": digest, "bytes": len(payload),
            "retrieved_utc": asset.retrieved_utc, "metadata": dict(metadata or {}),
        }
        manifest["licence"] = licence.to_dict()
        self._save_manifest(dataset, manifest)
        self.write_source_doc(dataset, licence)
        return asset

    def get_json(self, url: str, params: Mapping[str, Any]) -> dict[str, Any]:
        if self.offline:
            raise HttpError(f"offline mode: cannot query {url}", url=url)
        return self.transport.get_json(url, params)

    # --------------------------------------------------------------- reporting

    @staticmethod
    def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(chunk), b""):
                digest.update(block)
        return digest.hexdigest()

    def write_source_doc(self, dataset: str, licence: DatasetLicence) -> Path:
        """Write the SOURCE.md that makes the folder self-describing."""
        manifest = self.load_manifest(dataset)
        assets = manifest.get("assets", {})
        total = sum(a.get("bytes", 0) for a in assets.values())
        lines = [
            f"# Data source — {dataset}",
            "",
            f"- **Licence:** {licence.name}",
        ]
        if licence.url:
            lines.append(f"- **Licence URL:** {licence.url}")
        if licence.credit:
            lines.append(f"- **Credit:** {licence.credit}")
        if licence.citation:
            lines.append(f"- **Cite as:** {licence.citation}")
        lines += [
            f"- **Files:** {len(assets)}",
            f"- **Total size:** {total / 1e6:.1f} MB",
            "",
            "Reproduce with `python scripts/fetch_data.py`. Checksums are in "
            "`manifest.json`; this directory is gitignored by design.",
            "",
        ]
        path = self.dataset_dir(dataset) / "SOURCE.md"
        path.write_text("\n".join(lines))
        return path

    def statistics(self) -> dict[str, Any]:
        datasets = {}
        if self.root.exists():
            for child in sorted(self.root.iterdir()):
                if child.is_dir() and (child / self.MANIFEST).exists():
                    manifest = self.load_manifest(child.name)
                    assets = manifest.get("assets", {})
                    datasets[child.name] = {
                        "files": len(assets),
                        "bytes": sum(a.get("bytes", 0) for a in assets.values()),
                    }
        return {
            "root": str(self.root),
            "offline": self.offline,
            "downloads": self.downloads,
            "cache_hits": self.hits,
            "datasets": datasets,
        }
