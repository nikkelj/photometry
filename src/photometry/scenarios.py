"""Day-in-the-life fleet study scenarios: satellite type x attitude mode.

Library targets carry their catalog altitude in `real_altitude_km`.
`FleetScenario.orbit()` flies that altitude (same 70° study plane) so
visibility is honest: co-alt and below-shell objects never rise above the
observer's local horizontal, and the +5° ADCS suite cannot see them.

The published 21-scenario numbers use an explicit 620 km / 70° torus
(`study_orbit()` / `orbit(study_torus=True)`) so those targets stay above
the shell by construction. That path is a named study geometry, not a
silent relocation of ISS / Hubble / LINK / DTC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .attitude import FixedInertial, LvlhHold, PrincipalAxisSpin, TorqueFreeTumble
from .constellation import WalkerConstellation
from .frames import minimal_rotation_from_z, radec_to_unit
from .sensing import SensorConfig
from .shapes import LIBRARY

STUDY_ALT_KM = 620.0
STUDY_INC_DEG = 70.0
STUDY_RAAN_DEG = 40.0
STUDY_ARG_LAT_DEG = 10.0
SUN_RA_DEG, SUN_DEC_DEG = 130.0, 15.0

TUMBLE = dict(pole_ra_deg=200.0, pole_dec_deg=35.0, period_s=127.4, phase_rad=0.7)
HUBBLE_TARGET_RA, HUBBLE_TARGET_DEC = 80.0, -20.0

# non-principal-axis tumble truth (LINK anomaly-like): triaxial inertia,
# initial rate tilted off the minor axis so omega nutates
MULTIAXIS = dict(inertia=(1.0, 2.6, 3.2),
                 omega0_body=(0.049, 0.006, 0.014),
                 att_ra_deg=200.0, att_dec_deg=35.0)


def circular_orbit(altitude_km: float,
                   inclination_deg: float = STUDY_INC_DEG,
                   raan_deg: float = STUDY_RAAN_DEG,
                   arg_lat_deg: float = STUDY_ARG_LAT_DEG) -> WalkerConstellation:
    """Single circular target orbit in the study plane at `altitude_km`."""
    orb = WalkerConstellation(1, 1, altitude_km, inclination_deg)
    orb._raan = np.array([np.radians(raan_deg)])
    orb._u0 = np.array([np.radians(arg_lat_deg)])
    return orb


def study_orbit() -> WalkerConstellation:
    """Named 620 km / 70° torus used by published fleet results."""
    return circular_orbit(STUDY_ALT_KM)


def sun_eci() -> np.ndarray:
    return radec_to_unit(SUN_RA_DEG, SUN_DEC_DEG)


def make_attitude(mode: str, orbit: WalkerConstellation, sun: np.ndarray,
                  articulated_model: bool,
                  duration_s: float = 86400.0) -> tuple[object, bool]:
    """Attitude model + whether gimbaled facets actively track the sun."""
    if mode == "multiaxis_tumble":
        r0 = minimal_rotation_from_z(
            radec_to_unit(MULTIAXIS["att_ra_deg"], MULTIAXIS["att_dec_deg"]))
        return TorqueFreeTumble(MULTIAXIS["inertia"], MULTIAXIS["omega0_body"],
                                r0, t_max=duration_s, dt=1.0), False
    if mode == "ops":
        return LvlhHold(orbit), articulated_model
    if mode == "low_drag":
        return LvlhHold(orbit, roll_deg=90.0), False
    if mode == "tumble":
        # propeller tumble about the body long axis: photometrically loud,
        # unlike a flat spin about a panel normal which barely modulates
        return PrincipalAxisSpin(TUMBLE["pole_ra_deg"], TUMBLE["pole_dec_deg"],
                                 TUMBLE["period_s"], TUMBLE["phase_rad"],
                                 body_axis=(1.0, 0.0, 0.0)), False
    if mode == "sun_point":
        return FixedInertial.z_toward(sun), False
    if mode == "science":
        return FixedInertial.pointing(HUBBLE_TARGET_RA, HUBBLE_TARGET_DEC), articulated_model
    if mode == "safe_sun":
        return FixedInertial.z_toward(sun), False
    raise ValueError(f"unknown mode {mode!r}")


@dataclass
class FleetScenario:
    sat: str
    mode: str
    real_altitude_km: float
    duration_s: float = 86400.0
    dt_s: float = 6.0
    seed: int = 777
    sensors: SensorConfig = field(default_factory=SensorConfig)

    @property
    def name(self) -> str:
        return f"{self.sat}__{self.mode}"

    def shape(self):
        return LIBRARY[self.sat]()

    def orbit(self, *, study_torus: bool = False) -> WalkerConstellation:
        """Catalog altitude by default; pass `study_torus=True` for the 620 km path."""
        if study_torus:
            return study_orbit()
        return circular_orbit(self.real_altitude_km)


SCENARIOS: list[FleetScenario] = [
    FleetScenario("starlink_v15", "ops", 550),
    FleetScenario("starlink_v15", "low_drag", 550),
    FleetScenario("starlink_v15", "tumble", 550),
    FleetScenario("starlink_v2mini", "ops", 530),
    FleetScenario("starlink_v2mini", "low_drag", 530),
    FleetScenario("starlink_v2mini", "tumble", 530),
    FleetScenario("starlink_v2mini_dtc", "ops", 360),
    FleetScenario("starlink_v2mini_dtc", "low_drag", 360),
    FleetScenario("starlink_v2mini_dtc", "tumble", 360),
    FleetScenario("bluewalker3", "ops", 500),
    FleetScenario("bluewalker3", "sun_point", 500),
    FleetScenario("bluewalker3", "tumble", 500),
    FleetScenario("hubble", "science", 530),
    FleetScenario("hubble", "safe_sun", 530),
    FleetScenario("hubble", "tumble", 530),
    FleetScenario("iss", "ops", 420),
    FleetScenario("iss", "tumble", 420),
    # Katalyst LINK (Swift reboost): real state as of late 2026-07 is a
    # multi-axis spin after losing 2 of 3 reaction wheels; Swift's orbit
    # (the mission target) is ~500 km / 20.6 deg
    FleetScenario("katalyst_link", "ops", 500),
    FleetScenario("katalyst_link", "sun_point", 500),
    FleetScenario("katalyst_link", "tumble", 500),
    FleetScenario("katalyst_link", "multiaxis_tumble", 500),
]


def by_name(name: str) -> FleetScenario:
    for s in SCENARIOS:
        if s.name == name:
            return s
    raise KeyError(name)
