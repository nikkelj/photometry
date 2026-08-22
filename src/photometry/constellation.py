"""Walker-delta constellation with LVLH-mounted star trackers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frames import MU_EARTH, R_EARTH, lvlh_basis


@dataclass
class WalkerConstellation:
    """Circular Walker-delta shell.

    n_planes x sats_per_plane satellites, common altitude and inclination,
    RAAN spread over 2*pi, inter-plane phasing factor f (Walker t/p/f).
    """

    n_planes: int = 100
    sats_per_plane: int = 100
    altitude_km: float = 550.0
    inclination_deg: float = 53.0
    phasing: int = 17

    _raan: np.ndarray = field(init=False, repr=False)
    _u0: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        p = np.repeat(np.arange(self.n_planes), self.sats_per_plane)
        s = np.tile(np.arange(self.sats_per_plane), self.n_planes)
        n_total = self.n_planes * self.sats_per_plane
        self._raan = 2 * np.pi * p / self.n_planes
        self._u0 = 2 * np.pi * (s / self.sats_per_plane + self.phasing * p / n_total)

    @property
    def n_sats(self) -> int:
        return self.n_planes * self.sats_per_plane

    @property
    def semi_major_axis_km(self) -> float:
        return R_EARTH + self.altitude_km

    @property
    def mean_motion(self) -> float:
        return float(np.sqrt(MU_EARTH / self.semi_major_axis_km**3))

    def single_states(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """States of a single-sat 'constellation' at an array of times (K,3)."""
        if self.n_sats != 1:
            raise ValueError("single_states requires a one-satellite orbit")
        a, n = self.semi_major_axis_km, self.mean_motion
        t = np.asarray(t, dtype=float)
        u = self._u0[0] + n * t
        cu, su = np.cos(u), np.sin(u)
        cO, sO = np.cos(self._raan[0]), np.sin(self._raan[0])
        ci, si = np.cos(np.radians(self.inclination_deg)), np.sin(np.radians(self.inclination_deg))
        r = a * np.stack([cO * cu - sO * su * ci, sO * cu + cO * su * ci, su * si], axis=-1)
        v = a * n * np.stack(
            [-cO * su - sO * cu * ci, -sO * su + cO * cu * ci, cu * si], axis=-1
        )
        return r, v

    def orbit_normal(self) -> np.ndarray:
        """Unit orbit-normal (angular momentum direction) for a single-sat orbit."""
        r, v = self.single_states(np.array([0.0]))
        from .frames import unit as _unit
        return _unit(np.cross(r[0], v[0]))

    def states(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """ECI positions and velocities (km, km/s) of all sats at time t (s)."""
        a, n = self.semi_major_axis_km, self.mean_motion
        u = self._u0 + n * t
        cu, su = np.cos(u), np.sin(u)
        cO, sO = np.cos(self._raan), np.sin(self._raan)
        ci, si = np.cos(np.radians(self.inclination_deg)), np.sin(np.radians(self.inclination_deg))
        r = a * np.stack([cO * cu - sO * su * ci, sO * cu + cO * su * ci, su * si], axis=-1)
        v = a * n * np.stack(
            [-cO * su - sO * cu * ci, -sO * su + cO * cu * ci, cu * si], axis=-1
        )
        return r, v


def tracker_boresights_lvlh(elevation_deg: float | tuple[float, ...] = 5.0,
                            azimuths_deg: tuple[float, ...] = (0.0, 120.0, 240.0)) -> np.ndarray:
    """Star tracker boresights in the LVLH (along, cross, up) frame, shape (T,3).

    Elevation is measured up from the local horizontal plane; azimuth 0 is the
    along-track direction. A scalar elevation is broadcast across azimuths;
    a sequence must match `azimuths_deg` (or be length 1) so mixed cants
    (e.g. a hosted down-looker next to the +5° ADCS suite) share this helper.
    """
    az = np.asarray(azimuths_deg, dtype=float)
    el = np.asarray(elevation_deg, dtype=float)
    if el.ndim == 0 or el.size == 1:
        el = np.full(az.shape, float(el.reshape(())))
    elif az.size == 1:
        az = np.full(el.shape, float(az.reshape(())))
    elif el.shape != az.shape:
        raise ValueError("elevation_deg and azimuths_deg must be the same length")
    el = np.radians(el)
    az = np.radians(az)
    return np.stack(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
        axis=-1,
    )


def boresights_eci(r: np.ndarray, v: np.ndarray, b_lvlh: np.ndarray) -> np.ndarray:
    """Map LVLH boresights (T,3) to ECI for each satellite state.

    r, v: (N,3). Returns (N,T,3).
    """
    along, cross, up = lvlh_basis(r, v)
    basis = np.stack([along, cross, up], axis=-1)  # (N,3,3) columns are LVLH axes
    return np.einsum("nij,tj->nti", basis, b_lvlh)
