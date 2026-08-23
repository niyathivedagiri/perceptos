"""
perceptos.data.mars2020_raw
===========================

Client for the **JPL Mars 2020 Raw Image API** — the machine interface behind
NASA's official Perseverance raw-image archive.

::

    https://mars.nasa.gov/rss/api/?feed=raw_images&category=mars2020&feedtype=json

**No API key.** This matters: the commonly-used ``api.nasa.gov/mars-photos``
endpoint needs a key and is rate-limited to 30 requests/hour on ``DEMO_KEY``,
whereas this one is open. It also returns far more per image — spacecraft
clock, UTC and Mars local time, rover attitude, site and drive indices, and,
where JPL publishes it, the **camera model**.

Why the camera model is the whole point
---------------------------------------

The instruction "do not fake stereo" has a concrete technical meaning. To turn
a disparity into a distance you need the focal length and the baseline. Neither
is derivable from a pair of PNGs. The Mars Rover Photos API does not supply
them, so any depth computed from its output would rest on invented numbers.

This API publishes a ``camera_model_component_list`` in the CAHV family
(CAHV, CAHVOR, CAHVORE — the model JPL has used for flight cameras since
Viking). From it:

* **C** is the camera centre in rover coordinates. The distance between the
  left and right C vectors *is* the physical baseline — measured, not assumed.
* **A** is the pointing axis, **H** and **V** the horizontal and vertical
  image plane vectors. ``|H - (H·A)A|`` gives the horizontal focal length in
  pixels.

So when the model is present, PerceptOS derives real geometry. When it is
absent, :class:`~perceptos.data.sources.StereoStatus.CALIBRATION_UNAVAILABLE`
is reported and no depth is offered. There is no third path where a plausible
default gets filled in.

Verification note
-----------------

The response schema below follows JPL's published feed. The development
sandbox for this module has no route to ``mars.nasa.gov``, so the parser was
built against the documented schema and exercised with recorded fixtures rather
than against a live response. Every field access is therefore defensive: a
missing or renamed key degrades that one field to ``None`` and is reported in
:attr:`RawImage.parse_warnings`, rather than raising and losing the fetch.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .cache import LICENCES, CachedAsset, DatasetCache, HttpError
from .sources import DataSourceStatus, StereoStatus

__all__ = [
    "JPL_RAW_API",
    "CameraModel",
    "RawImage",
    "VerifiedStereoPair",
    "JplRawImageClient",
]

JPL_RAW_API = "https://mars.nasa.gov/rss/api/"

#: Instrument codes that form stereo pairs, mapped to their partner.
STEREO_PARTNERS: dict[str, str] = {
    "NAVCAM_LEFT": "NAVCAM_RIGHT",
    "NAVCAM_RIGHT": "NAVCAM_LEFT",
    "FRONT_HAZCAM_LEFT_A": "FRONT_HAZCAM_RIGHT_A",
    "FRONT_HAZCAM_RIGHT_A": "FRONT_HAZCAM_LEFT_A",
    "REAR_HAZCAM_LEFT": "REAR_HAZCAM_RIGHT",
    "REAR_HAZCAM_RIGHT": "REAR_HAZCAM_LEFT",
    "MCZ_LEFT": "MCZ_RIGHT",
    "MCZ_RIGHT": "MCZ_LEFT",
}

_VECTOR_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def _parse_vector(value: Any) -> Optional[list[float]]:
    """Parse ``"(1.0,2.0,3.0)"`` or a list into floats. ``None`` if unparseable."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            return None
    found = _VECTOR_RE.findall(str(value))
    return [float(v) for v in found] if found else None


@dataclass(frozen=True, slots=True)
class CameraModel:
    """A CAHV-family camera model as published by JPL.

    Only the fields PerceptOS can actually use are extracted. ``O``, ``R`` and
    the CAHVORE extras are retained raw so nothing is silently discarded, but
    the distortion terms are not applied — reporting an undistorted model as
    distortion-corrected would be its own small lie.
    """

    model_type: str                       # CAHV | CAHVOR | CAHVORE
    C: Optional[list[float]] = None       # camera centre, rover frame, metres
    A: Optional[list[float]] = None       # pointing axis (unit)
    H: Optional[list[float]] = None       # horizontal image-plane vector
    V: Optional[list[float]] = None       # vertical image-plane vector
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_usable(self) -> bool:
        """True when the four core vectors are present and well-formed."""
        return all(v is not None and len(v) == 3 for v in (self.C, self.A, self.H, self.V))

    @property
    def focal_length_px(self) -> Optional[float]:
        """Horizontal focal length in pixels: ``|H - (H·A)A|``."""
        if not self.is_usable:
            return None
        dot = sum(h * a for h, a in zip(self.H, self.A))
        residual = [h - dot * a for h, a in zip(self.H, self.A)]
        return float(math.sqrt(sum(r * r for r in residual)))

    @property
    def principal_point(self) -> Optional[tuple[float, float]]:
        """``(cx, cy) = (H·A, V·A)`` in pixels."""
        if not self.is_usable:
            return None
        return (float(sum(h * a for h, a in zip(self.H, self.A))),
                float(sum(v * a for v, a in zip(self.V, self.A))))

    def baseline_to(self, other: "CameraModel") -> Optional[float]:
        """Physical distance between two camera centres, in metres.

        This is the measured baseline. It is the number that makes depth from
        this archive legitimate rather than invented.
        """
        if self.C is None or other.C is None:
            return None
        if len(self.C) != 3 or len(other.C) != 3:
            return None
        return float(math.sqrt(sum((a - b) ** 2 for a, b in zip(self.C, other.C))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "C": self.C, "A": self.A, "H": self.H, "V": self.V,
            "usable": self.is_usable,
            "focal_length_px": round(self.focal_length_px, 2)
            if self.focal_length_px else None,
            "principal_point": [round(v, 2) for v in self.principal_point]
            if self.principal_point else None,
            "distortion_applied": False,
            "extra_keys": sorted(self.extra),
        }

    @classmethod
    def parse(cls, camera: dict[str, Any]) -> Optional["CameraModel"]:
        """Build from the feed's ``camera`` object. ``None`` if no model present."""
        model_type = str(camera.get("camera_model_type") or "").strip()
        components = camera.get("camera_model_component_list")
        if not model_type and not components:
            return None

        vectors: dict[str, Optional[list[float]]] = {}
        extra: dict[str, Any] = {}

        if isinstance(components, dict):
            for name, value in components.items():
                key = str(name).strip().upper()
                parsed = _parse_vector(value)
                if key in ("C", "A", "H", "V") and parsed and len(parsed) == 3:
                    vectors[key] = parsed
                else:
                    extra[str(name)] = value
        elif isinstance(components, (list, tuple)):
            # Flat list convention: C, A, H, V in order, three floats each.
            flat: list[float] = []
            for item in components:
                parsed = _parse_vector(item)
                if parsed:
                    flat.extend(parsed)
            for index, key in enumerate(("C", "A", "H", "V")):
                chunk = flat[index * 3:index * 3 + 3]
                if len(chunk) == 3:
                    vectors[key] = chunk
            if len(flat) > 12:
                extra["trailing_components"] = flat[12:]
        elif isinstance(components, str):
            flat = _parse_vector(components) or []
            for index, key in enumerate(("C", "A", "H", "V")):
                chunk = flat[index * 3:index * 3 + 3]
                if len(chunk) == 3:
                    vectors[key] = chunk

        if not vectors:
            return None
        return cls(model_type=model_type or "CAHV", extra=extra,
                   **{k: vectors.get(k) for k in ("C", "A", "H", "V")})


@dataclass(slots=True)
class RawImage:
    """One image from the JPL archive, with everything the feed supplied."""

    image_id: str
    sol: int
    instrument: str
    url_full_res: str
    url_medium: str = ""
    date_taken_utc: str = ""
    date_taken_mars: str = ""            # LMST
    date_received_utc: str = ""
    sclk: Optional[float] = None         # spacecraft clock — the pairing key
    attitude: Optional[list[float]] = None
    site: Optional[int] = None
    drive: Optional[int] = None
    dimension: Optional[list[float]] = None
    sample_type: str = ""                # Full | Thumbnail | Subframe
    camera_model: Optional[CameraModel] = None
    credit: str = "NASA/JPL-Caltech"
    local_path: Optional[Path] = None
    parse_warnings: list[str] = field(default_factory=list)

    @property
    def eye(self) -> Optional[str]:
        upper = self.instrument.upper()
        if "LEFT" in upper:
            return "left"
        if "RIGHT" in upper:
            return "right"
        return None

    @property
    def is_full_frame(self) -> bool:
        return self.sample_type.lower().startswith("full")

    @property
    def has_calibration(self) -> bool:
        return self.camera_model is not None and self.camera_model.is_usable

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "sol": self.sol,
            "instrument": self.instrument,
            "eye": self.eye,
            "url_full_res": self.url_full_res,
            "date_taken_utc": self.date_taken_utc,
            "date_taken_mars": self.date_taken_mars,
            "sclk": self.sclk,
            "attitude": self.attitude,
            "site": self.site,
            "drive": self.drive,
            "dimension": self.dimension,
            "sample_type": self.sample_type,
            "has_calibration": self.has_calibration,
            "camera_model": self.camera_model.to_dict() if self.camera_model else None,
            "credit": self.credit,
            "local_path": str(self.local_path) if self.local_path else None,
            "parse_warnings": list(self.parse_warnings),
            "source_status": (DataSourceStatus.LOCAL_CACHED_DATA.value
                              if self.local_path
                              else DataSourceStatus.NASA_PUBLIC_NO_KEY.value),
        }


@dataclass(slots=True)
class VerifiedStereoPair:
    """A stereo pair with the evidence for, or against, trusting it.

    Every field below is either read from the archive or computed from it.
    Nothing is defaulted. ``baseline_m`` is ``None`` when the camera models
    were absent, and callers must handle that rather than receive a guess.
    """

    left: RawImage
    right: RawImage
    status: StereoStatus
    sol: int
    sclk: Optional[float]
    temporal_delta_s: Optional[float]
    baseline_m: Optional[float]
    focal_length_px: Optional[float]
    rectified: bool
    confidence: float
    evidence: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)

    @property
    def depth_valid(self) -> bool:
        return self.status.depth_valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_image_id": self.left.image_id,
            "right_image_id": self.right.image_id,
            "left_instrument": self.left.instrument,
            "right_instrument": self.right.instrument,
            "left_timestamp_utc": self.left.date_taken_utc,
            "right_timestamp_utc": self.right.date_taken_utc,
            "sol": self.sol,
            "sclk": self.sclk,
            "temporal_delta_s": self.temporal_delta_s,
            "stereo_status": self.status.value,
            "severity": self.status.severity,
            "depth_valid": self.depth_valid,
            "rectified": self.rectified,
            "baseline_m": self.baseline_m,
            "focal_length_px": self.focal_length_px,
            "stereo_confidence": round(self.confidence, 3),
            "evidence": list(self.evidence),
            "concerns": list(self.concerns),
            "credit": "NASA/JPL-Caltech",
        }


class JplRawImageClient:
    """Queries and caches Perseverance imagery from the keyless JPL feed."""

    #: Maximum acquisition-time difference for a pair to be considered
    #: simultaneous. Navcam stereo frames are exposed within milliseconds of
    #: each other; a second apart means two separate pointings.
    MAX_TEMPORAL_DELTA_S = 2.0

    def __init__(self, cache: DatasetCache, *, dataset: str = "perseverance_jpl") -> None:
        self.cache = cache
        self.dataset = dataset
        self.source_status = DataSourceStatus.NASA_PUBLIC_NO_KEY
        self.last_warnings: list[str] = []

    # ------------------------------------------------------------------ query

    def query(self, *, sol: Optional[int] = None, instrument: Optional[str] = None,
              num: int = 50, page: int = 0,
              order: str = "sol desc") -> list[RawImage]:
        """List images. No credentials are sent because none are required."""
        params: dict[str, Any] = {
            "feed": "raw_images",
            "category": "mars2020",
            "feedtype": "json",
            "num": int(num),
            "page": int(page),
            "order": order,
            "extended": "",
        }
        if sol is not None:
            # The feed filters sols with a range condition.
            params["condition_2"] = f"{int(sol)}:sol:gte"
            params["condition_3"] = f"{int(sol)}:sol:lte"
        if instrument:
            params["search"] = instrument.upper()

        payload = self.cache.get_json(JPL_RAW_API, params)
        records = payload.get("images", []) if isinstance(payload, dict) else []
        images = [self._parse(record) for record in records]
        images = [i for i in images if i is not None]
        if instrument:
            images = [i for i in images if i.instrument.upper() == instrument.upper()]
        self.last_warnings = sorted({w for i in images for w in i.parse_warnings})
        return images

    def _parse(self, record: dict[str, Any]) -> Optional[RawImage]:
        """Defensively parse one feed entry. Returns ``None`` if unusable."""
        warnings: list[str] = []
        files = record.get("image_files") or {}
        url = files.get("full_res") or files.get("large") or files.get("medium") or ""
        if not url:
            return None

        camera = record.get("camera") or {}
        instrument = str(camera.get("instrument") or record.get("camera_vector") or "")
        if not instrument:
            instrument = str(record.get("instrument") or "UNKNOWN")
            warnings.append("instrument not in camera object; fell back to record field")

        extended = record.get("extended") or {}
        sclk = None
        raw_sclk = extended.get("sclk") if isinstance(extended, dict) else None
        if raw_sclk not in (None, "", "UNK"):
            try:
                sclk = float(raw_sclk)
            except (TypeError, ValueError):
                warnings.append(f"unparseable sclk {raw_sclk!r}")
        else:
            warnings.append("no spacecraft clock in feed entry")

        model = None
        try:
            model = CameraModel.parse(camera)
        except Exception:  # noqa: BLE001 - a malformed model is not fatal
            warnings.append("camera model present but could not be parsed")
        if model is None:
            warnings.append("no camera model published for this image")

        image_id = str(record.get("imageid") or record.get("image_id")
                       or Path(url).stem)
        try:
            sol = int(record.get("sol", -1))
        except (TypeError, ValueError):
            sol = -1
            warnings.append("unparseable sol")

        return RawImage(
            image_id=image_id,
            sol=sol,
            instrument=instrument.upper(),
            url_full_res=url,
            url_medium=str(files.get("medium") or ""),
            date_taken_utc=str(record.get("date_taken_utc") or ""),
            date_taken_mars=str(record.get("date_taken_mars") or ""),
            date_received_utc=str(record.get("date_received") or ""),
            sclk=sclk,
            attitude=_parse_vector(record.get("attitude")),
            site=_as_int(record.get("site")),
            drive=_as_int(record.get("drive")),
            dimension=_parse_vector(extended.get("dimension")
                                    if isinstance(extended, dict) else None),
            sample_type=str(record.get("sample_type") or ""),
            camera_model=model,
            credit=str(record.get("credit") or "NASA/JPL-Caltech"),
            parse_warnings=warnings,
        )

    # -------------------------------------------------------- stereo pairing

    def verify_stereo(self, left: RawImage, right: RawImage) -> VerifiedStereoPair:
        """Assess a candidate pair and report exactly what can be claimed.

        This never returns "probably fine". Each check either contributes
        evidence or raises a concern, and the resulting
        :class:`~perceptos.data.sources.StereoStatus` is the weakest state
        consistent with the evidence.
        """
        evidence: list[str] = []
        concerns: list[str] = []

        # --- 1. are they a stereo instrument pair at all? -------------------
        expected = STEREO_PARTNERS.get(left.instrument)
        instruments_ok = expected == right.instrument
        if instruments_ok:
            evidence.append(f"{left.instrument} and {right.instrument} are a "
                            "designated stereo pair")
        else:
            concerns.append(f"{left.instrument} and {right.instrument} are not "
                            "a designated stereo pair")

        # --- 2. same acquisition? ------------------------------------------
        delta: Optional[float] = None
        if left.sclk is not None and right.sclk is not None:
            delta = abs(left.sclk - right.sclk)
            if delta <= self.MAX_TEMPORAL_DELTA_S:
                evidence.append(f"spacecraft clocks agree to {delta:.3f} s")
            else:
                concerns.append(f"spacecraft clocks differ by {delta:.1f} s - "
                                "these are separate acquisitions")
        else:
            concerns.append("spacecraft clock missing from one or both images; "
                            "simultaneity cannot be established")

        same_sol = left.sol == right.sol
        if not same_sol:
            concerns.append(f"different sols ({left.sol} vs {right.sol})")

        # --- 3. same rover pose? -------------------------------------------
        if left.site is not None and right.site is not None:
            if (left.site, left.drive) == (right.site, right.drive):
                evidence.append(f"same site/drive ({left.site}/{left.drive}) - "
                                "the rover did not move between exposures")
            else:
                concerns.append("different site/drive indices - the rover moved")

        # --- 4. calibration --------------------------------------------------
        baseline = focal = None
        if left.has_calibration and right.has_calibration:
            baseline = left.camera_model.baseline_to(right.camera_model)
            focal = left.camera_model.focal_length_px
            if baseline and baseline > 0:
                evidence.append(
                    f"camera models published ({left.camera_model.model_type}); "
                    f"measured baseline {baseline:.4f} m, focal {focal:.1f} px"
                )
            else:
                concerns.append("camera models present but the baseline "
                                "computed to zero or nothing")
                baseline = None
        else:
            concerns.append(
                "no usable camera model in the archive for this pair; baseline "
                "and focal length are unknown and will not be assumed"
            )

        # --- 5. verdict ------------------------------------------------------
        simultaneous = delta is not None and delta <= self.MAX_TEMPORAL_DELTA_S
        calibrated = baseline is not None and focal is not None

        # Simultaneity is a hard precondition, not a scoring factor. Two
        # Navcam frames an hour apart show different scenes under different
        # lighting from a possibly different pose; a perfect camera model does
        # not make them a stereo pair. Ranking a calibrated-but-non-simultaneous
        # pair above an uncalibrated simultaneous one would invert the priority.
        if not instruments_ok or not same_sol or not simultaneous:
            status = StereoStatus.UNRECTIFIED_PAIR
            concerns.append(
                "REJECTED as a stereo pair: the two frames are not a single "
                "simultaneous exposure from a designated stereo instrument. "
                "Usable as two independent images only."
            )
        elif calibrated:
            # Real geometry and a real simultaneous acquisition — but the
            # archive publishes raw frames, so epipolar lines are not yet
            # horizontal. Rectification is a separate, later step.
            status = StereoStatus.VALID_STEREO
        else:
            status = StereoStatus.CALIBRATION_UNAVAILABLE

        confidence = self._confidence(instruments_ok, simultaneous, calibrated,
                                      same_sol, left, right)

        return VerifiedStereoPair(
            left=left, right=right, status=status, sol=left.sol,
            sclk=left.sclk, temporal_delta_s=delta, baseline_m=baseline,
            focal_length_px=focal, rectified=False, confidence=confidence,
            evidence=evidence, concerns=concerns,
        )

    @staticmethod
    def _confidence(instruments_ok: bool, simultaneous: bool, calibrated: bool,
                    same_sol: bool, left: RawImage, right: RawImage) -> float:
        """Confidence that this pair is genuinely usable for stereo.

        Deliberately not 1.0 even in the best case: the frames are unrectified
        and the camera models are published values rather than a calibration
        this system performed, so some residual uncertainty is honest.
        """
        # Any failed precondition means zero, not a reduced score: a
        # non-simultaneous pair is not a weak stereo pair, it is not one.
        if not (instruments_ok and same_sol and simultaneous):
            return 0.0
        score = 0.35
        score += 0.35 if calibrated else 0.0
        score += 0.15 if (left.is_full_frame and right.is_full_frame) else 0.0
        if left.dimension and right.dimension and left.dimension == right.dimension:
            score += 0.10
        return round(min(score, 0.95), 3)

    def find_stereo_pairs(self, images: Iterable[RawImage]) -> list[VerifiedStereoPair]:
        """Group images into verified pairs. Unmatched frames are dropped."""
        buckets: dict[tuple[int, Any, str], dict[str, RawImage]] = {}
        for image in images:
            eye = image.eye
            partner = STEREO_PARTNERS.get(image.instrument)
            if eye is None or partner is None:
                continue
            family = image.instrument.replace("_LEFT", "").replace("_RIGHT", "")
            # Round the clock so the two eyes of one exposure land together
            # without merging separate pointings seconds apart.
            clock = round(image.sclk, 0) if image.sclk is not None else None
            key = (image.sol, clock, family)
            existing = buckets.setdefault(key, {}).get(eye)
            if existing is None or (image.is_full_frame and not existing.is_full_frame):
                buckets[key][eye] = image

        pairs = []
        for _, eyes in sorted(buckets.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            if "left" in eyes and "right" in eyes:
                pairs.append(self.verify_stereo(eyes["left"], eyes["right"]))
        return pairs

    # ----------------------------------------------------------- downloading

    def download(self, image: RawImage) -> CachedAsset:
        """Cache one image and return its asset with full provenance."""
        suffix = Path(image.url_full_res.split("?")[0]).suffix or ".png"
        name = f"sol{image.sol:05d}/{image.image_id}{suffix}"
        asset = self.cache.fetch(
            image.url_full_res,
            dataset=self.dataset,
            name=name,
            licence=LICENCES["nasa_pd"],
            metadata={
                "image_id": image.image_id,
                "sol": image.sol,
                "instrument": image.instrument,
                "eye": image.eye,
                "sclk": image.sclk,
                "date_taken_utc": image.date_taken_utc,
                "date_taken_mars": image.date_taken_mars,
                "site": image.site,
                "drive": image.drive,
                "has_calibration": image.has_calibration,
                "camera_model_type": (image.camera_model.model_type
                                      if image.camera_model else None),
                "source_service": "jpl_raw_images",
                "source_status": DataSourceStatus.NASA_PUBLIC_NO_KEY.value,
            },
        )
        image.local_path = asset.path
        return asset

    def download_pair(self, pair: VerifiedStereoPair) -> Optional[VerifiedStereoPair]:
        """Download both eyes. Returns ``None`` if either fetch fails."""
        try:
            self.download(pair.left)
            self.download(pair.right)
        except HttpError:
            return None
        return pair

    def sol_directory(self, sol: int) -> Path:
        return self.cache.dataset_dir(self.dataset) / f"sol{sol:05d}"


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
