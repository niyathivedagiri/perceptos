"""
perceptos.data.mars2020
=======================

Client for NASA's Mars Rover Photos API, specialised for Perseverance.

Endpoint::

    https://api.nasa.gov/mars-photos/api/v1/rovers/perseverance/photos
        ?sol=<n>&camera=<CODE>&api_key=<key>

The response is a ``photos`` array whose entries carry ``id``, ``sol``,
``earth_date``, a ``camera`` object and ``img_src``.

**Stereo pairing is the non-obvious part.** The API returns left and right
frames as unrelated records; nothing in the JSON says which right image
partners which left one. The pairing key is hidden in the filename. A
Perseverance product ID looks like::

    NLF_0203_0684970679_051ECM_N0071716NCAM02203_10_195J
    ^^^ ^^^^ ^^^^^^^^^^
     |    |      +-- spacecraft clock (SCLK) at acquisition
     |    +--------- sol
     +-------------- camera: N=Navcam, L=Left, F=Full frame

Left and right images of one stereo acquisition **share the SCLK value**,
because they are exposed together. Pairing on ``(sol, sclk)`` is therefore
exact. Pairing on list adjacency or on timestamp proximity — the two obvious
approaches — silently mismatches frames whenever the rover takes a mosaic,
and a mismatched stereo pair produces a confident, completely wrong depth map.

This module downloads and describes images. It does **not** rectify them:
Navcam pairs are raw, and rectification needs the rig geometry. See
:mod:`perceptos.sensors.dataset_source` for how unrectified pairs are flagged
so the stereo matcher cannot be handed them by accident.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .cache import LICENCES, CachedAsset, DatasetCache, HttpError

__all__ = [
    "NASA_API_BASE",
    "PERSEVERANCE_CAMERAS",
    "ProductId",
    "RoverImage",
    "StereoPair",
    "Mars2020Client",
]

NASA_API_BASE = "https://api.nasa.gov/mars-photos/api/v1/rovers"

#: API camera codes -> (product-ID prefix, human name, stereo partner code).
PERSEVERANCE_CAMERAS: dict[str, tuple[str, str, Optional[str]]] = {
    "NAVCAM_LEFT": ("NL", "Navigation Camera - Left", "NAVCAM_RIGHT"),
    "NAVCAM_RIGHT": ("NR", "Navigation Camera - Right", "NAVCAM_LEFT"),
    "FRONT_HAZCAM_LEFT_A": ("FL", "Front Hazcam - Left A", "FRONT_HAZCAM_RIGHT_A"),
    "FRONT_HAZCAM_RIGHT_A": ("FR", "Front Hazcam - Right A", "FRONT_HAZCAM_LEFT_A"),
    "REAR_HAZCAM_LEFT": ("RL", "Rear Hazcam - Left", "REAR_HAZCAM_RIGHT"),
    "REAR_HAZCAM_RIGHT": ("RR", "Rear Hazcam - Right", "REAR_HAZCAM_LEFT"),
    "MCZ_LEFT": ("ZL", "Mastcam-Z - Left", "MCZ_RIGHT"),
    "MCZ_RIGHT": ("ZR", "Mastcam-Z - Right", "MCZ_LEFT"),
    "SKYCAM": ("SK", "MEDA Skycam", None),
    "SUPERCAM_RMI": ("LR", "SuperCam Remote Micro-Imager", None),
}

#: Approximate Navcam stereo baseline in metres (Mars 2020 engineering cameras).
#: Used only as a default; a real depth product should come from calibration.
NAVCAM_BASELINE_M = 0.42
HAZCAM_BASELINE_M = 0.25

_PRODUCT_RE = re.compile(
    r"^(?P<cam>[A-Z]{2})(?P<variant>[A-Z])_(?P<sol>\d{4})_(?P<sclk>\d{10})_"
)


@dataclass(frozen=True, slots=True)
class ProductId:
    """Parsed Perseverance product identifier."""

    raw: str
    camera_prefix: str      # NL, NR, FL, ...
    variant: str            # F=full frame, T=thumbnail, S=subframe, ...
    sol: int
    sclk: int               # spacecraft clock — the stereo pairing key

    @property
    def stereo_key(self) -> tuple[int, int]:
        """Left and right frames of one acquisition share this key exactly."""
        return (self.sol, self.sclk)

    @property
    def eye(self) -> Optional[str]:
        if len(self.camera_prefix) != 2:
            return None
        return {"L": "left", "R": "right"}.get(self.camera_prefix[1])

    @classmethod
    def parse(cls, value: str) -> Optional["ProductId"]:
        """Parse a product ID or an image URL. Returns ``None`` if unrecognised.

        Returning None rather than raising is deliberate: NASA's archive spans
        several naming eras and a single odd filename must not abort a
        thousand-image fetch.
        """
        stem = Path(str(value).split("?")[0]).stem
        match = _PRODUCT_RE.match(stem)
        if not match:
            return None
        return cls(
            raw=stem,
            camera_prefix=match.group("cam"),
            variant=match.group("variant"),
            sol=int(match.group("sol")),
            sclk=int(match.group("sclk")),
        )


@dataclass(frozen=True, slots=True)
class RoverImage:
    """One image record from the API, plus its parsed identity."""

    id: int
    sol: int
    earth_date: str
    camera: str
    camera_full_name: str
    url: str
    product: Optional[ProductId] = None
    local_path: Optional[Path] = None

    @property
    def eye(self) -> Optional[str]:
        return self.product.eye if self.product else None

    @property
    def stereo_key(self) -> Optional[tuple[int, int]]:
        return self.product.stereo_key if self.product else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "sol": self.sol, "earth_date": self.earth_date,
            "camera": self.camera, "camera_full_name": self.camera_full_name,
            "url": self.url, "product_id": self.product.raw if self.product else None,
            "sclk": self.product.sclk if self.product else None,
            "eye": self.eye,
            "local_path": str(self.local_path) if self.local_path else None,
            "credit": "NASA/JPL-Caltech",
        }


@dataclass(slots=True)
class StereoPair:
    """A matched left/right acquisition."""

    left: RoverImage
    right: RoverImage
    sol: int
    sclk: int
    rectified: bool = False
    baseline_m: Optional[float] = None

    @property
    def camera_family(self) -> str:
        prefix = self.left.product.camera_prefix[0] if self.left.product else "?"
        return {"N": "navcam", "F": "front_hazcam", "R": "rear_hazcam",
                "Z": "mastcam_z"}.get(prefix, "unknown")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sol": self.sol, "sclk": self.sclk,
            "camera_family": self.camera_family,
            "rectified": self.rectified,
            "baseline_m": self.baseline_m,
            "left": self.left.to_dict(), "right": self.right.to_dict(),
        }


class Mars2020Client:
    """Queries and caches Perseverance imagery.

    Parameters
    ----------
    cache:
        A :class:`~perceptos.data.cache.DatasetCache`. Supply one built with a
        ``FakeTransport`` to test without a network.
    api_key:
        NASA API key. ``DEMO_KEY`` works but is heavily rate-limited (roughly
        30 requests/hour by IP), which is why a real key is worth the sixty
        seconds it takes to get from api.nasa.gov.
    """

    def __init__(self, cache: DatasetCache, *, api_key: str = "DEMO_KEY",
                 rover: str = "perseverance") -> None:
        self.cache = cache
        self.api_key = api_key
        self.rover = rover
        self.dataset = f"{rover}_raw"

    # ------------------------------------------------------------------ query

    def query(self, *, sol: Optional[int] = None, camera: Optional[str] = None,
              earth_date: Optional[str] = None, page: int = 1) -> list[RoverImage]:
        """List images for a sol (or Earth date) and optional camera."""
        if sol is None and earth_date is None:
            raise ValueError("provide either sol or earth_date")
        if camera and camera.upper() not in PERSEVERANCE_CAMERAS:
            raise ValueError(
                f"unknown camera {camera!r}; known: {sorted(PERSEVERANCE_CAMERAS)}"
            )
        payload = self.cache.get_json(
            f"{NASA_API_BASE}/{self.rover}/photos",
            {"sol": sol, "earth_date": earth_date,
             "camera": camera.upper() if camera else None,
             "page": page, "api_key": self.api_key},
        )
        return [self._parse(record) for record in payload.get("photos", [])]

    def latest(self) -> list[RoverImage]:
        payload = self.cache.get_json(
            f"{NASA_API_BASE}/{self.rover}/latest_photos", {"api_key": self.api_key}
        )
        return [self._parse(r) for r in payload.get("latest_photos", [])]

    @staticmethod
    def _parse(record: dict[str, Any]) -> RoverImage:
        camera = record.get("camera", {}) or {}
        url = record.get("img_src", "")
        return RoverImage(
            id=int(record.get("id", -1)),
            sol=int(record.get("sol", -1)),
            earth_date=str(record.get("earth_date", "")),
            camera=str(camera.get("name", "")),
            camera_full_name=str(camera.get("full_name", "")),
            url=url,
            product=ProductId.parse(url),
        )

    # ------------------------------------------------------------- pairing

    @staticmethod
    def match_stereo(images: Iterable[RoverImage]) -> list[StereoPair]:
        """Group left/right frames into stereo pairs by ``(sol, sclk)``.

        Images whose product ID does not parse, or which have no partner, are
        dropped rather than paired by proximity — an unmatched frame is far
        less harmful than a wrongly matched one.
        """
        buckets: dict[tuple[int, int], dict[str, RoverImage]] = {}
        for image in images:
            key = image.stereo_key
            eye = image.eye
            if key is None or eye is None:
                continue
            # Prefer full frames over thumbnails when both exist for one eye.
            existing = buckets.setdefault(key, {}).get(eye)
            if existing is None or (
                image.product and image.product.variant == "F"
                and existing.product and existing.product.variant != "F"
            ):
                buckets[key][eye] = image

        pairs: list[StereoPair] = []
        for (sol, sclk), eyes in sorted(buckets.items()):
            if "left" in eyes and "right" in eyes:
                left = eyes["left"]
                family_prefix = left.product.camera_prefix[0] if left.product else "?"
                baseline = {"N": NAVCAM_BASELINE_M, "F": HAZCAM_BASELINE_M,
                            "R": HAZCAM_BASELINE_M}.get(family_prefix)
                pairs.append(StereoPair(
                    left=left, right=eyes["right"], sol=sol, sclk=sclk,
                    rectified=False, baseline_m=baseline,
                ))
        return pairs

    def stereo_pairs(self, sol: int, *, family: str = "NAVCAM") -> list[StereoPair]:
        """Fetch and match stereo pairs for one sol from a camera family."""
        left_code = f"{family.upper()}_LEFT"
        right_code = PERSEVERANCE_CAMERAS.get(left_code, (None, None, None))[2]
        if right_code is None:
            raise ValueError(
                f"{family!r} is not a stereo camera family; stereo families: "
                f"{sorted({c.rsplit('_', 1)[0] for c, v in PERSEVERANCE_CAMERAS.items() if v[2]})}"
            )
        images = self.query(sol=sol, camera=left_code)
        images += self.query(sol=sol, camera=right_code)
        return self.match_stereo(images)

    # ----------------------------------------------------------- downloading

    def download(self, image: RoverImage) -> CachedAsset:
        """Cache one image locally and return the asset with its provenance."""
        name = f"sol{image.sol:05d}/{(image.product.raw if image.product else str(image.id))}.png"
        asset = self.cache.fetch(
            image.url,
            dataset=self.dataset,
            name=name,
            licence=LICENCES["nasa_pd"],
            metadata={
                "sol": image.sol,
                "earth_date": image.earth_date,
                "camera": image.camera,
                "camera_full_name": image.camera_full_name,
                "product_id": image.product.raw if image.product else None,
                "sclk": image.product.sclk if image.product else None,
                "eye": image.eye,
                "rover": self.rover,
            },
        )
        object.__setattr__(image, "local_path", asset.path)
        return asset

    def download_pairs(self, pairs: Sequence[StereoPair], *,
                       limit: Optional[int] = None) -> list[StereoPair]:
        """Download both eyes of each pair. Partial failures skip that pair."""
        done: list[StereoPair] = []
        for pair in pairs[: limit or len(pairs)]:
            try:
                self.download(pair.left)
                self.download(pair.right)
            except HttpError:
                # One unreachable frame should not abort a bulk fetch, but a
                # half-downloaded pair is useless, so drop the whole pair.
                continue
            done.append(pair)
        return done

    def sol_directory(self, sol: int) -> Path:
        return self.cache.dataset_dir(self.dataset) / f"sol{sol:05d}"
