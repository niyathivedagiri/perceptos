"""
perceptos.core.environment
==========================

The environment abstraction: what changes between Mars, the Moon, a warehouse
and a factory floor, and — more importantly — what does not.

The design rule is that **an environment is data, not a code path**. There is
no ``if mode == "mars"`` anywhere in the perception pipeline. An
:class:`Environment` bundles the vehicle geometry, the terrain taxonomy, the
data sources and the physical constants; the pipeline reads those and behaves
accordingly. Adding a new planet is a new :class:`Environment`, not a new
branch.

A note on gravity, because it is routinely got wrong: the **geometric
tip-over limit is independent of gravity**. For a rigid vehicle on a slope,
static rollover occurs when the centre of mass passes outside the wheelbase,
which is pure geometry — a rover that tips at 45 degrees on Earth tips at 45
degrees on the Moon. Likewise the friction-limited slip angle is
``tan(theta) = mu``, also gravity-free. Gravity enters through *dynamics*:
energy per metre, braking distance, settling time, sinkage in loose regolith,
and the reduced normal force that makes wheels dig in more readily. So the
slope limits below are set by terrain and vehicle, and the gravity value is
used where it genuinely belongs — the energy model and the dynamics — rather
than being sprinkled over the slope thresholds to look physical.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional

from ..mars.traversability import RoverGeometry

__all__ = [
    "TerrainClass",
    "DataSourceSpec",
    "Environment",
    "ENVIRONMENTS",
    "get_environment",
    "register_environment",
]


@dataclass(frozen=True, slots=True)
class TerrainClass:
    """One semantic terrain label and what it means for driving.

    ``mobility_cost`` is the extra traversal cost this surface carries *at zero
    slope*. It is what makes semantic segmentation worth having: sand is
    geometrically flat and lethal, and a purely geometric cost map cannot see
    that. This is the surface that immobilised Spirit at Troy.
    """

    id: int
    name: str
    mobility_cost: float = 0.0        # 0 = free, 1 = impassable
    slip_risk: float = 0.0            # expected slip ratio on this surface
    science_value: float = 0.0        # relevance for target selection
    description: str = ""

    def __post_init__(self) -> None:
        for field_name in ("mobility_cost", "slip_risk", "science_value"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must lie in [0, 1], got {value}")


#: AI4Mars taxonomy. Label rasters use pixel values 0-3; 255 marks no-label.
AI4MARS_CLASSES: tuple[TerrainClass, ...] = (
    TerrainClass(0, "soil", mobility_cost=0.10, slip_risk=0.15, science_value=0.3,
                 description="Fine-grained regolith; nominal driving surface."),
    TerrainClass(1, "bedrock", mobility_cost=0.25, slip_risk=0.05, science_value=0.9,
                 description="Exposed rock; excellent traction, high science value."),
    TerrainClass(2, "sand", mobility_cost=0.75, slip_risk=0.65, science_value=0.2,
                 description="Loose sand. Geometrically flat but a sinkage hazard - "
                             "the failure mode that ended Spirit's mission."),
    TerrainClass(3, "big_rock", mobility_cost=1.00, slip_risk=0.10, science_value=0.7,
                 description="Rock large enough to strike the belly pan. Impassable."),
)

EARTH_CLASSES: tuple[TerrainClass, ...] = (
    TerrainClass(0, "pavement", mobility_cost=0.05, slip_risk=0.02),
    TerrainClass(1, "grass", mobility_cost=0.20, slip_risk=0.15),
    TerrainClass(2, "gravel", mobility_cost=0.35, slip_risk=0.30),
    TerrainClass(3, "obstacle", mobility_cost=1.00, slip_risk=0.05),
)

INDOOR_CLASSES: tuple[TerrainClass, ...] = (
    TerrainClass(0, "floor", mobility_cost=0.02, slip_risk=0.01),
    TerrainClass(1, "marked_lane", mobility_cost=0.02, slip_risk=0.01),
    TerrainClass(2, "restricted", mobility_cost=0.85, slip_risk=0.01,
                 description="Keep-out zone: human walkway, charging bay, spill."),
    TerrainClass(3, "obstacle", mobility_cost=1.00, slip_risk=0.01),
)


@dataclass(frozen=True, slots=True)
class DataSourceSpec:
    """Where this environment's data comes from, per tier.

    Empty strings mean "no public source; use simulation or your own capture",
    which is stated explicitly rather than left for someone to discover.
    """

    imagery: str = ""
    imagery_url: str = ""
    elevation: str = ""
    elevation_url: str = ""
    labels: str = ""
    labels_url: str = ""
    detection_weights: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "imagery": self.imagery, "imagery_url": self.imagery_url,
            "elevation": self.elevation, "elevation_url": self.elevation_url,
            "labels": self.labels, "labels_url": self.labels_url,
            "detection_weights": self.detection_weights, "notes": self.notes,
        }

    @property
    def has_real_imagery(self) -> bool:
        return bool(self.imagery_url)

    @property
    def has_real_elevation(self) -> bool:
        return bool(self.elevation_url)


@dataclass(frozen=True, slots=True)
class Environment:
    """A complete operating context. Swap this, not the pipeline."""

    name: str
    display_name: str
    gravity_ms2: float
    rover: RoverGeometry
    terrain_classes: tuple[TerrainClass, ...]
    data: DataSourceSpec
    cell_size_m: float = 0.1
    cruise_speed_mps: float = 1.0
    one_way_light_time_s: tuple[float, float] = (0.0, 0.0)
    stages: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if self.gravity_ms2 <= 0:
            raise ValueError("gravity must be positive")
        ids = [c.id for c in self.terrain_classes]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate terrain class ids in {self.name!r}")

    # -------------------------------------------------------------- taxonomy

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in sorted(self.terrain_classes, key=lambda c: c.id))

    def terrain_class(self, key: int | str) -> TerrainClass:
        for cls in self.terrain_classes:
            if cls.id == key or cls.name == key:
                return cls
        raise KeyError(
            f"{key!r} is not a terrain class in {self.name!r}; "
            f"available: {list(self.class_names)}"
        )

    def mobility_costs(self) -> dict[int, float]:
        """Class id -> mobility cost. Feeds the semantic cost layer."""
        return {c.id: c.mobility_cost for c in self.terrain_classes}

    def slip_risks(self) -> dict[int, float]:
        return {c.id: c.slip_risk for c in self.terrain_classes}

    # -------------------------------------------------------------- physics

    def energy_per_metre_wh(self, mass_kg: float, rolling_resistance: float = 0.3) -> float:
        """Rolling-resistance energy cost. This is where gravity legitimately enters."""
        joules_per_m = rolling_resistance * mass_kg * self.gravity_ms2
        return float(joules_per_m / 3600.0)

    def is_remote(self) -> bool:
        """True when light time makes teleoperation impossible."""
        return max(self.one_way_light_time_s) > 1.0

    @property
    def supervision(self) -> str:
        return "autonomous" if self.is_remote() else "supervised"

    # ------------------------------------------------------------ variations

    def with_rover(self, rover: RoverGeometry) -> "Environment":
        return replace(self, rover=rover)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "gravity_ms2": self.gravity_ms2,
            "cell_size_m": self.cell_size_m,
            "cruise_speed_mps": self.cruise_speed_mps,
            "one_way_light_time_s": list(self.one_way_light_time_s),
            "supervision": self.supervision,
            "terrain_classes": [
                {"id": c.id, "name": c.name, "mobility_cost": c.mobility_cost,
                 "slip_risk": c.slip_risk, "science_value": c.science_value,
                 "description": c.description}
                for c in sorted(self.terrain_classes, key=lambda c: c.id)
            ],
            "rover": {
                "max_slope_deg": self.rover.max_slope_deg,
                "critical_slope_deg": self.rover.critical_slope_deg,
                "max_step_m": self.rover.max_step_m,
                "max_roughness_m": self.rover.max_roughness_m,
                "wheel_radius_m": self.rover.wheel_radius_m,
            },
            "data": self.data.to_dict(),
            "real_data_available": {
                "imagery": self.data.has_real_imagery,
                "elevation": self.data.has_real_elevation,
                "labels": bool(self.data.labels_url),
            },
            "description": self.description,
        }


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------

MARS = Environment(
    name="mars",
    display_name="Mars Exploration",
    gravity_ms2=3.721,
    rover=RoverGeometry(
        wheel_radius_m=0.2625, wheelbase_m=2.7, ground_clearance_m=0.6,
        max_slope_deg=25.0, critical_slope_deg=30.0,
        max_step_m=0.35, max_roughness_m=0.15,
    ),
    terrain_classes=AI4MARS_CLASSES,
    cell_size_m=0.25,
    cruise_speed_mps=0.042,          # ~150 m/h, Perseverance-class
    one_way_light_time_s=(240.0, 1320.0),
    data=DataSourceSpec(
        imagery="Perseverance Navcam / Hazcam raw images",
        imagery_url="https://api.nasa.gov/mars-photos/api/v1/rovers/perseverance/photos",
        elevation="HiRISE DTM of Jezero crater, 1 m/pixel",
        elevation_url=(
            "https://astrogeology.usgs.gov/search/map/Mars/Mars2020/"
            "JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40"
        ),
        labels="AI4Mars v0.6 semantic segmentation",
        labels_url="https://zenodo.org/records/15995036",
        detection_weights="models/geology_yolo11n.pt",
        notes="Navcam pairs are raw and require rectification before stereo matching.",
    ),
    description=(
        "Perseverance-class surface autonomy in Jezero crater. Light time makes "
        "teleoperation impossible, so every decision is made locally."
    ),
)

MOON = Environment(
    name="moon",
    display_name="Lunar Exploration",
    gravity_ms2=1.625,
    rover=RoverGeometry(
        wheel_radius_m=0.25, wheelbase_m=2.0, ground_clearance_m=0.4,
        max_slope_deg=20.0, critical_slope_deg=25.0,
        max_step_m=0.30, max_roughness_m=0.12,
    ),
    terrain_classes=AI4MARS_CLASSES,   # regolith taxonomy transfers reasonably
    cell_size_m=0.25,
    cruise_speed_mps=0.10,
    one_way_light_time_s=(1.2, 1.4),
    data=DataSourceSpec(
        imagery="",
        elevation="LOLA global DEM (~118 m/px); LRO NAC DTMs (~2 m/px)",
        elevation_url="https://pds-geosciences.wustl.edu/missions/lro/",
        notes=(
            "No large public surface-imagery corpus comparable to AI4Mars. "
            "Terrain classification transfers from Mars only approximately: "
            "lunar regolith is far more abrasive and the near-total absence of "
            "atmospheric scattering makes shadows genuinely black, which breaks "
            "feature extraction in ways Mars imagery never does."
        ),
    ),
    description="Lunar surface autonomy. Same pipeline, harsher illumination.",
)

EARTH = Environment(
    name="earth",
    display_name="Earth Inspection",
    gravity_ms2=9.807,
    rover=RoverGeometry(
        wheel_radius_m=0.15, wheelbase_m=1.2, ground_clearance_m=0.25,
        max_slope_deg=20.0, critical_slope_deg=28.0,
        max_step_m=0.15, max_roughness_m=0.08,
    ),
    terrain_classes=EARTH_CLASSES,
    cell_size_m=0.1,
    cruise_speed_mps=1.2,
    one_way_light_time_s=(0.0, 0.0),
    data=DataSourceSpec(
        imagery="Local webcam or recorded video",
        detection_weights="models/yolo11n.pt",
        notes="COCO-pretrained detection works out of the box.",
    ),
    description="Outdoor ground robot with a live operator in the loop.",
)

WAREHOUSE = Environment(
    name="warehouse",
    display_name="Warehouse Robotics",
    gravity_ms2=9.807,
    rover=RoverGeometry(
        wheel_radius_m=0.10, wheelbase_m=0.8, ground_clearance_m=0.05,
        max_slope_deg=5.0, critical_slope_deg=8.0,
        max_step_m=0.02, max_roughness_m=0.01,
    ),
    terrain_classes=INDOOR_CLASSES,
    cell_size_m=0.05,
    cruise_speed_mps=1.5,
    one_way_light_time_s=(0.0, 0.0),
    data=DataSourceSpec(
        imagery="Local camera or recorded footage",
        detection_weights="models/yolo11n.pt",
        notes=(
            "Floors are flat by construction, so geometric traversability adds "
            "little. The load moves to tracking: the hazards here are people and "
            "forklifts, which move."
        ),
    ),
    stages={"traversability": {"enabled": False}, "tracking": {"enabled": True}},
    description="Indoor AMR. Flat ground, dynamic obstacles, humans present.",
)

FACTORY = Environment(
    name="factory",
    display_name="Industrial Inspection",
    gravity_ms2=9.807,
    rover=RoverGeometry(
        wheel_radius_m=0.08, wheelbase_m=0.6, ground_clearance_m=0.04,
        max_slope_deg=4.0, critical_slope_deg=6.0,
        max_step_m=0.015, max_roughness_m=0.008,
    ),
    terrain_classes=INDOOR_CLASSES,
    cell_size_m=0.02,
    cruise_speed_mps=0.6,
    one_way_light_time_s=(0.0, 0.0),
    data=DataSourceSpec(
        imagery="Fixed inspection cameras or your own capture",
        notes=(
            "Fine cell size because the interesting defects are millimetric. "
            "Detection matters more than navigation here."
        ),
    ),
    stages={"traversability": {"enabled": False}, "detection": {"enabled": True}},
    description="Static or slow-moving inspection of machinery and structures.",
)


ENVIRONMENTS: dict[str, Environment] = {
    env.name: env for env in (MARS, MOON, EARTH, WAREHOUSE, FACTORY)
}


def get_environment(name: str) -> Environment:
    """Look up an environment by name."""
    key = str(name).lower().strip()
    if key not in ENVIRONMENTS:
        raise KeyError(
            f"unknown environment {name!r}; available: {sorted(ENVIRONMENTS)}"
        )
    return ENVIRONMENTS[key]


def register_environment(environment: Environment, *, overwrite: bool = False) -> None:
    """Add a custom environment. This is the extension point for a new mission."""
    if environment.name in ENVIRONMENTS and not overwrite:
        raise ValueError(
            f"environment {environment.name!r} already registered; "
            "pass overwrite=True to replace it"
        )
    ENVIRONMENTS[environment.name] = environment
