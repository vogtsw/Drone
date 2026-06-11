"""Toy low-altitude digital twin used as the TCEA testbed.

The world keeps two copies of the radio state:

* ``truth``  — the real environment. Measurements are drawn from it (+noise).
* ``twin``   — the digital twin used for predictions. Drift is injected by
  mutating ``truth`` while leaving ``twin`` stale, exactly the failure mode
  the agent must detect, attribute and repair (spec section 2).

Physics: a simplified 3GPP-style model. Received power from a gNB at a 3D
point is ``tx_power + vertical_antenna_gain - path_loss - env_attenuation``.
The serving cell is the strongest one, on both the truth and the twin side.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from tcea.config import TceaConfig


@dataclass
class GnbConfig:
    gnb_id: str
    x: float
    y: float
    height: float = 25.0
    tilt_deg: float = 6.0  # mechanical downtilt, positive = down
    tx_power_dbm: float = 43.0
    max_gain_db: float = 15.0
    vertical_hpbw_deg: float = 20.0


@dataclass
class ZoneState:
    zone_id: str
    x0: float
    y0: float
    size: float
    env_attenuation_db: float
    gnbs: list[GnbConfig]
    bias_correction_db: float = 0.0  # learned calibration offset (twin side only)


@dataclass
class ConfigChange:
    zone_id: str
    gnb_id: str
    field_name: str
    old_value: float
    new_value: float
    window: int


@dataclass
class GeoEvent:
    zone_id: str
    description: str
    window: int


@dataclass
class PairedBatch:
    """One monitoring window of paired prediction/measurement samples."""

    zone_id: str
    window: int
    points: np.ndarray        # (n, 3) -> x, y, height
    predictions: np.ndarray   # twin RSRP, dBm
    measurements: np.ndarray  # truth RSRP + noise, dBm

    @property
    def errors(self) -> np.ndarray:
        return self.predictions - self.measurements

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.errors**2)))

    @property
    def bias(self) -> float:
        return float(np.mean(self.errors))


def _rsrp(gnb: GnbConfig, px: float, py: float, ph: float, env_att: float) -> float:
    dh = max(math.hypot(px - gnb.x, py - gnb.y), 1.0)
    dz = ph - gnb.height
    d3d = math.sqrt(dh * dh + dz * dz)
    # elevation angle of the receiver as seen from the antenna (deg, up = +)
    elev = math.degrees(math.atan2(dz, dh))
    # boresight points down by tilt_deg; off-axis angle drives the gain loss
    off_axis = elev + gnb.tilt_deg
    gain = gnb.max_gain_db - min(
        12.0 * (off_axis / gnb.vertical_hpbw_deg) ** 2, 30.0
    )
    path_loss = 28.0 + 22.0 * math.log10(d3d)
    return gnb.tx_power_dbm + gain - path_loss - env_att


def _zone_rsrp(zone: ZoneState, points: np.ndarray) -> np.ndarray:
    out = np.empty(len(points))
    for i, (px, py, ph) in enumerate(points):
        out[i] = max(
            _rsrp(g, float(px), float(py), float(ph), zone.env_attenuation_db)
            for g in zone.gnbs
        )
    return out


class DigitalTwinWorld:
    """Truth + twin state, drift injection, and the data interfaces of spec 8.1."""

    def __init__(self, n_zones: int = 4, seed: int = 0, config: TceaConfig | None = None):
        self.config = config or TceaConfig()
        self.rng = np.random.default_rng(seed)
        self.truth: dict[str, ZoneState] = {}
        self.twin: dict[str, ZoneState] = {}
        self.config_log: list[ConfigChange] = []
        self.geo_events: list[GeoEvent] = []
        self.noise_sigma: dict[str, float] = {}
        self.gps_ok_fraction: dict[str, float] = {}
        self.frozen_points: dict[str, np.ndarray] = {}

        for z in range(n_zones):
            zone_id = f"Z-{z:04d}"
            x0, y0, size = z * 1500.0, 0.0, 1000.0
            gnbs = [
                GnbConfig(
                    gnb_id=f"gNB-{z}{i}",
                    x=x0 + size * (0.3 + 0.4 * i),
                    y=y0 + size * (0.3 + 0.4 * i),
                )
                for i in range(2)
            ]
            zone = ZoneState(
                zone_id=zone_id, x0=x0, y0=y0, size=size,
                env_attenuation_db=3.0, gnbs=gnbs,
            )
            self.truth[zone_id] = zone
            self.twin[zone_id] = copy.deepcopy(zone)
            self.noise_sigma[zone_id] = 2.0
            self.gps_ok_fraction[zone_id] = 0.98
            self.frozen_points[zone_id] = self._sample_points(
                zone_id, self.config.frozen_points_per_zone
            )

    # ------------------------------------------------------------------ #
    # Sampling / prediction / measurement
    # ------------------------------------------------------------------ #
    def _sample_points(self, zone_id: str, n: int) -> np.ndarray:
        z = self.truth[zone_id]
        xs = self.rng.uniform(z.x0, z.x0 + z.size, n)
        ys = self.rng.uniform(z.y0, z.y0 + z.size, n)
        hs = self.rng.uniform(30.0, 120.0, n)
        return np.column_stack([xs, ys, hs])

    def predict(self, zone_id: str, points: np.ndarray,
                zone_override: ZoneState | None = None) -> np.ndarray:
        zone = zone_override if zone_override is not None else self.twin[zone_id]
        return _zone_rsrp(zone, points) + zone.bias_correction_db

    def measure(self, zone_id: str, points: np.ndarray) -> np.ndarray:
        truth = _zone_rsrp(self.truth[zone_id], points)
        noise = self.rng.normal(0.0, self.noise_sigma[zone_id], len(points))
        return truth + noise

    def sample_window(self, zone_id: str, window: int, n: int | None = None) -> PairedBatch:
        n = n or self.config.samples_per_window
        points = self._sample_points(zone_id, n)
        return PairedBatch(
            zone_id=zone_id,
            window=window,
            points=points,
            predictions=self.predict(zone_id, points),
            measurements=self.measure(zone_id, points),
        )

    # ------------------------------------------------------------------ #
    # Drift injection (benchmark scenario primitives, spec 10.3)
    # ------------------------------------------------------------------ #
    def inject_config_change(self, zone_id: str, gnb_index: int,
                             new_tilt_deg: float, window: int,
                             logged: bool = True) -> None:
        """Reality changes a gNB tilt; the twin keeps the stale value."""
        gnb = self.truth[zone_id].gnbs[gnb_index]
        old = gnb.tilt_deg
        gnb.tilt_deg = new_tilt_deg
        if logged:
            self.config_log.append(ConfigChange(
                zone_id=zone_id, gnb_id=gnb.gnb_id, field_name="tilt_deg",
                old_value=old, new_value=new_tilt_deg, window=window,
            ))

    def inject_environment_change(self, zone_id: str, delta_att_db: float,
                                  window: int, surveyed: bool = True) -> None:
        """Reality gains extra attenuation (new construction / vegetation)."""
        self.truth[zone_id].env_attenuation_db += delta_att_db
        if surveyed:
            self.geo_events.append(GeoEvent(
                zone_id=zone_id,
                description=f"construction detected (~{delta_att_db:.0f} dB extra loss)",
                window=window,
            ))

    def inject_data_quality_issue(self, zone_id: str, noise_sigma: float = 6.0,
                                  gps_ok_fraction: float = 0.55) -> None:
        self.noise_sigma[zone_id] = noise_sigma
        self.gps_ok_fraction[zone_id] = gps_ok_fraction

    # ------------------------------------------------------------------ #
    # Staging for safe repair (executor mutates a copy; publisher commits)
    # ------------------------------------------------------------------ #
    def stage_zone(self, zone_id: str) -> ZoneState:
        return copy.deepcopy(self.twin[zone_id])

    def commit_zone(self, staged: ZoneState) -> None:
        self.twin[staged.zone_id] = staged
