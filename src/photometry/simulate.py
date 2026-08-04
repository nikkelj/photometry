"""End-to-end simulation scenario configuration and driver."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .attitude import PrincipalAxisSpin
from .constellation import WalkerConstellation
from .frames import radec_to_unit
from .measurements import ObservationSet
from .sensing import SensorConfig, simulate_detections
from .shapes import FacetModel, box_wing


@dataclass
class Scenario:
    """Baseline scenario: 10k-sat Walker shell observing a tumbling box-wing."""

    n_planes: int = 100
    sats_per_plane: int = 100
    shell_altitude_km: float = 550.0
    shell_inclination_deg: float = 53.0

    target_altitude_km: float = 620.0
    target_inclination_deg: float = 72.0
    target_raan_deg: float = 40.0
    target_arg_lat_deg: float = 10.0

    pole_ra_deg: float = 200.0
    pole_dec_deg: float = 35.0
    spin_period_s: float = 127.4
    spin_phase_rad: float = 0.7

    sun_ra_deg: float = 130.0
    sun_dec_deg: float = 15.0

    duration_s: float = 2 * 5820.0  # ~2 target orbits
    dt_s: float = 4.0
    seed: int = 12345

    sensors: SensorConfig = field(default_factory=SensorConfig)

    def constellation(self) -> WalkerConstellation:
        return WalkerConstellation(
            self.n_planes, self.sats_per_plane, self.shell_altitude_km,
            self.shell_inclination_deg,
        )

    def target_orbit(self) -> WalkerConstellation:
        orb = WalkerConstellation(1, 1, self.target_altitude_km, self.target_inclination_deg)
        orb._raan = np.array([np.radians(self.target_raan_deg)])
        orb._u0 = np.array([np.radians(self.target_arg_lat_deg)])
        return orb

    def target_shape(self) -> FacetModel:
        return box_wing()

    def target_attitude(self) -> PrincipalAxisSpin:
        return PrincipalAxisSpin(
            self.pole_ra_deg, self.pole_dec_deg, self.spin_period_s, self.spin_phase_rad
        )

    def sun_eci(self) -> np.ndarray:
        return radec_to_unit(self.sun_ra_deg, self.sun_dec_deg)

    def truth_dict(self) -> dict:
        d = asdict(self)
        d.pop("sensors")
        return d


def run(scenario: Scenario | None = None) -> tuple[ObservationSet, Scenario]:
    sc = scenario or Scenario()
    rng = np.random.default_rng(sc.seed)
    t_grid = np.arange(0.0, sc.duration_s, sc.dt_s)
    obs = simulate_detections(
        sc.constellation(),
        sc.target_orbit(),
        sc.target_shape(),
        sc.target_attitude(),
        sc.sun_eci(),
        t_grid,
        sc.sensors,
        rng,
    )
    return obs, sc
