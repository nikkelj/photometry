"""Shared harness for the study scripts.

Every study script used to carry its own copy of the fleet simulation
setup, the truth-attitude sampler, and the attitude-error metric. They
live here once; a study script is now only its study logic.
"""

from __future__ import annotations

import numpy as np

from . import scenarios as sc
from .attitude import FixedInertial, LvlhHold, PrincipalAxisSpin
from .constellation import WalkerConstellation
from .frames import unit_to_radec
from .measurements import ObservationSet
from .sensing import SensorConfig, simulate_detections
from .shapes import FacetModel

SHELL = dict(n_planes=100, n_per_plane=100, altitude_km=550.0,
             inclination_deg=53.0)


def study_constellation() -> WalkerConstellation:
    return WalkerConstellation(SHELL["n_planes"], SHELL["n_per_plane"],
                               SHELL["altitude_km"], SHELL["inclination_deg"])


def simulate_target(shape: FacetModel, attitude, rng: np.random.Generator,
                    duration_s: float, dt_s: float = 6.0,
                    articulate: bool | None = None,
                    articulate_offset_deg: float = 0.0,
                    sensors: SensorConfig | None = None) -> ObservationSet | None:
    """Full-fleet simulation of one target on the study orbit.

    Returns None when the target is never detected (below the limiting
    magnitude) instead of raising — undetectable is a study outcome.
    """
    orbit, sun = sc.study_orbit(), sc.sun_eci()
    art = shape.articulated if articulate is None else articulate
    t_grid = np.arange(0.0, duration_s, dt_s)
    try:
        return simulate_detections(study_constellation(), orbit, shape, attitude,
                                   sun, t_grid, sensors or SensorConfig(), rng,
                                   articulate=art,
                                   articulate_offset_deg=articulate_offset_deg)
    except RuntimeError:
        return None


def make_truth_attitude(mode: str, orbit, sun: np.ndarray,
                        rng: np.random.Generator,
                        period_range=(40.0, 500.0)):
    """Random truth attitude for a named mode (study sampler convention)."""
    if mode == "ops":
        return LvlhHold(orbit)
    if mode == "low_drag":
        return LvlhHold(orbit, roll_deg=90.0)
    if mode in ("sun_point", "safe_sun"):
        return FixedInertial.z_toward(sun)
    if mode == "science":
        return FixedInertial.pointing(rng.uniform(0, 360), rng.uniform(-60, 60))
    if mode == "tumble":
        pole = rng.normal(size=3)
        ra, dec = unit_to_radec(pole / np.linalg.norm(pole))
        axis = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)][rng.integers(3)]
        return PrincipalAxisSpin(float(ra), float(dec),
                                 float(rng.uniform(*period_range)),
                                 float(rng.uniform(0, 2 * np.pi)), body_axis=axis)
    raise ValueError(mode)


def attitude_error_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """Rotation angle between two body->ECI matrices (degrees)."""
    c = (np.trace(r_a @ r_b.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def window(obs: ObservationSet, t0: float, width_s: float) -> ObservationSet:
    return obs.subset(np.nonzero((obs.t_s >= t0) & (obs.t_s < t0 + width_s))[0])
