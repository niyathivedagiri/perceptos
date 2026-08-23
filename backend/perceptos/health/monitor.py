"""
perceptos.health.monitor
========================

System and vehicle health, with every number carrying its true source.

The temptation on a health panel is to make it complete: show a battery
percentage, a motor temperature, a wheel current, because a mission-control UI
with blank tiles looks unfinished. That is exactly how a dashboard starts
lying. There is no battery in a laptop running a rover simulator — or rather
there is, and it is the laptop's, which is not what the tile claims to show.

So this module divides telemetry into two sets and labels them differently:

* **Measured** — CPU, memory, disk, process statistics, accelerator device.
  These come from ``psutil`` and the compute layer and are tagged
  ``LIVE_SENSOR``.
* **Modelled** — battery state of charge, motor temperature, and anything else
  a real rover would have that this machine does not. These are computed from
  an explicit power model and tagged ``SIMULATION``, so the UI badges them and
  no operator mistakes a model output for a reading.

``psutil`` is optional. Without it the system metrics report ``None`` rather
than a plausible-looking guess.
"""

from __future__ import annotations

import os
import platform
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.compute import device_report
from ..core.provenance import Source, Traced, traced

__all__ = ["SystemHealth", "PowerModel", "HealthMonitor", "HAS_PSUTIL"]

try:  # pragma: no cover - environment dependent
    import psutil

    HAS_PSUTIL = True
except Exception:  # noqa: BLE001
    psutil = None
    HAS_PSUTIL = False


@dataclass(slots=True)
class SystemHealth:
    """Measured host metrics. ``None`` means unavailable, never a guess."""

    cpu_percent: Optional[float]
    memory_percent: Optional[float]
    memory_used_mb: Optional[float]
    memory_total_mb: Optional[float]
    disk_free_gb: Optional[float]
    disk_percent: Optional[float]
    process_rss_mb: Optional[float]
    process_threads: Optional[int]
    load_average: Optional[tuple[float, float, float]]
    uptime_s: float
    device: dict[str, Any]

    @property
    def grade(self) -> str:
        if self.cpu_percent is None:
            return "unknown"
        if (self.memory_percent or 0) > 92 or (self.disk_percent or 0) > 95:
            return "critical"
        if self.cpu_percent > 90 or (self.memory_percent or 0) > 80:
            return "loaded"
        return "nominal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_total_mb": self.memory_total_mb,
            "disk_free_gb": self.disk_free_gb,
            "disk_percent": self.disk_percent,
            "process_rss_mb": self.process_rss_mb,
            "process_threads": self.process_threads,
            "load_average": list(self.load_average) if self.load_average else None,
            "uptime_s": round(self.uptime_s, 1),
            "device": self.device,
            "grade": self.grade,
            "metrics_available": HAS_PSUTIL,
        }


@dataclass(slots=True)
class PowerModel:
    """An explicit rover power model. Its outputs are SIMULATION, always.

    Parameters follow a Perseverance-class vehicle: an MMRTG supplying a
    roughly constant trickle into a battery that buffers peak loads. The point
    of modelling it rather than showing a fixed number is that the model can be
    interrogated — every value it produces names the assumptions that made it.
    """

    capacity_wh: float = 1080.0          # 2x 43 Ah Li-ion, ~28 V
    generator_w: float = 110.0           # MMRTG electrical output
    idle_draw_w: float = 90.0
    drive_draw_w: float = 200.0
    compute_draw_w: float = 30.0
    state_of_charge: float = 0.85

    def step(self, dt_s: float, *, driving: bool = False,
             compute_load: float = 0.5) -> float:
        """Advance the model. Returns the new state of charge in [0, 1]."""
        draw = self.idle_draw_w + self.compute_draw_w * max(0.0, min(1.0, compute_load))
        if driving:
            draw += self.drive_draw_w
        net_w = self.generator_w - draw
        delta = (net_w * dt_s / 3600.0) / self.capacity_wh
        self.state_of_charge = float(max(0.0, min(1.0, self.state_of_charge + delta)))
        return self.state_of_charge

    def endurance_hours(self, *, driving: bool = False,
                        compute_load: float = 0.5) -> Optional[float]:
        """Hours until depletion at the current draw, or ``None`` if charging."""
        draw = self.idle_draw_w + self.compute_draw_w * compute_load
        if driving:
            draw += self.drive_draw_w
        net_w = draw - self.generator_w
        if net_w <= 0:
            return None                  # generator covers the load indefinitely
        return float(self.state_of_charge * self.capacity_wh / net_w)

    def to_dict(self, *, driving: bool = False,
                compute_load: float = 0.5) -> dict[str, Any]:
        endurance = self.endurance_hours(driving=driving, compute_load=compute_load)
        return {
            "state_of_charge": round(self.state_of_charge, 4),
            "percent": round(self.state_of_charge * 100.0, 1),
            "capacity_wh": self.capacity_wh,
            "remaining_wh": round(self.state_of_charge * self.capacity_wh, 1),
            "generator_w": self.generator_w,
            "endurance_hours": round(endurance, 2) if endurance is not None else None,
            "net_positive": endurance is None,
            "driving": driving,
            "assumptions": (
                "MMRTG-class constant generator into a buffering battery; "
                "draw is idle + compute + drive. No physical battery is read."
            ),
        }


class HealthMonitor:
    """Collects host metrics and rover power state, each with honest provenance."""

    def __init__(self, *, power: Optional[PowerModel] = None,
                 disk_path: str = ".") -> None:
        self.power = power if power is not None else PowerModel()
        self.disk_path = disk_path
        self.started = time.time()
        self._process = psutil.Process(os.getpid()) if HAS_PSUTIL else None
        if self._process is not None:
            # First call to cpu_percent always returns 0.0; prime it so the
            # first reading an operator sees is real rather than a misleading
            # zero on a busy machine.
            self._process.cpu_percent(None)
            psutil.cpu_percent(None)

    # ------------------------------------------------------------- measured

    def system(self) -> SystemHealth:
        cpu = memory_percent = used = total = rss = None
        threads = None
        load: Optional[tuple[float, float, float]] = None
        if HAS_PSUTIL:
            cpu = float(psutil.cpu_percent(None))
            virtual = psutil.virtual_memory()
            memory_percent = float(virtual.percent)
            used = round(virtual.used / 1e6, 1)
            total = round(virtual.total / 1e6, 1)
            if self._process is not None:
                rss = round(self._process.memory_info().rss / 1e6, 1)
                threads = int(self._process.num_threads())
        try:
            usage = shutil.disk_usage(self.disk_path)
            free_gb = round(usage.free / 1e9, 2)
            disk_percent = round(100.0 * usage.used / usage.total, 1)
        except OSError:
            free_gb = disk_percent = None
        if hasattr(os, "getloadavg"):
            try:
                load = tuple(float(v) for v in os.getloadavg())  # type: ignore[assignment]
            except OSError:
                load = None

        return SystemHealth(
            cpu_percent=cpu, memory_percent=memory_percent, memory_used_mb=used,
            memory_total_mb=total, disk_free_gb=free_gb, disk_percent=disk_percent,
            process_rss_mb=rss, process_threads=threads, load_average=load,
            uptime_s=time.time() - self.started, device=device_report(),
        )

    def traced_system(self) -> Traced[SystemHealth]:
        health = self.system()
        return traced(
            health,
            # Genuinely read off the host. Not a rover, but a real measurement.
            source=Source.LIVE_SENSOR if HAS_PSUTIL else Source.CALCULATED,
            algorithm="psutil host metrics" if HAS_PSUTIL else "shutil.disk_usage only",
            confidence=1.0 if HAS_PSUTIL else 0.4,
            units="percent, MB, GB",
            label="system_health",
            parameters={"psutil": HAS_PSUTIL, "platform": platform.platform()},
        )

    # ------------------------------------------------------------- modelled

    def traced_power(self, *, driving: bool = False,
                     compute_load: float = 0.5) -> Traced[dict[str, Any]]:
        """Rover power state. Always SIMULATION — there is no rover battery here."""
        return traced(
            self.power.to_dict(driving=driving, compute_load=compute_load),
            source=Source.SIMULATION,
            algorithm="PowerModel(MMRTG generator + buffering battery)",
            confidence=1.0,
            units="Wh, hours",
            label="rover_power",
            parameters={"capacity_wh": self.power.capacity_wh,
                        "generator_w": self.power.generator_w,
                        "note": "modelled, not measured"},
        )

    # --------------------------------------------------------------- combined

    def snapshot(self, *, driving: bool = False,
                 compute_load: Optional[float] = None) -> dict[str, Any]:
        """Everything the health panel shows, each half labelled with its source."""
        system = self.traced_system()
        load = compute_load
        if load is None:
            cpu = system.value.cpu_percent
            load = (cpu / 100.0) if cpu is not None else 0.5
        power = self.traced_power(driving=driving, compute_load=load)
        return {
            "system": {
                **system.value.to_dict(),
                "_provenance": system.provenance.to_dict(),
            },
            "power": {
                **power.value,
                "_provenance": power.provenance.to_dict(),
            },
            "recorded_utc": time.time(),
        }
