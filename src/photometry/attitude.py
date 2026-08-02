"""Target attitude models.

All models expose `eci_to_body(t, v_eci)` mapping ECI vectors (K,3) at times
t (K,) into the body frame. Convention: v_body = R_bi(t)^T v_eci, implemented
as `v @ R` where R maps body to ECI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frames import (
    lvlh_basis,
    minimal_rotation_between,
    minimal_rotation_from_z,
    radec_to_unit,
    rodrigues,
    rotate_about_axis,
    unit,
)


@dataclass
class PrincipalAxisSpin:
    """Uniform rotation about an inertially fixed spin pole.

    `body_axis` is the body-frame axis aligned with the pole: (0,0,1) is a
    flat spin about the plate normal for a panel sat, (1,0,0) a propeller
    tumble about the long axis.
    """

    pole_ra_deg: float
    pole_dec_deg: float
    period_s: float
    phase_rad: float = 0.0
    body_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    _pole: np.ndarray = field(init=False, repr=False)
    _r0: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._pole = radec_to_unit(self.pole_ra_deg, self.pole_dec_deg)
        self._r0 = minimal_rotation_between(np.asarray(self.body_axis, dtype=float),
                                            self._pole)

    @property
    def pole(self) -> np.ndarray:
        return self._pole

    def angles(self, t: np.ndarray) -> np.ndarray:
        return 2 * np.pi * np.asarray(t, dtype=float) / self.period_s + self.phase_rad

    def eci_to_body(self, t: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
        v_despun = rotate_about_axis(v_eci, self._pole, -self.angles(t))
        return v_despun @ self._r0

    def body_to_eci_matrix(self, t: float) -> np.ndarray:
        return rodrigues(self._pole, float(self.angles(np.asarray(t)))) @ self._r0


@dataclass
class LvlhHold:
    """Attitude locked to the LVLH frame of `orbit` (a single-sat orbit).

    Body axes at zero offsets: +x along-track, +y cross-track, +z zenith.
    Offsets are applied as body-side rotations: yaw about z, then pitch about
    y, then roll about x — e.g. roll_deg=90 is a knife-edge ("low drag")
    orientation with the bus plane containing the velocity vector.
    """

    orbit: object
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    _r_off: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rz = rodrigues(np.array([0.0, 0.0, 1.0]), np.radians(self.yaw_deg))
        ry = rodrigues(np.array([0.0, 1.0, 0.0]), np.radians(self.pitch_deg))
        rx = rodrigues(np.array([1.0, 0.0, 0.0]), np.radians(self.roll_deg))
        self._r_off = rz @ ry @ rx

    def eci_to_body(self, t: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
        r, v = self.orbit.single_states(np.asarray(t, dtype=float))
        along, cross, up = lvlh_basis(r, v)
        v_lvlh = np.stack(
            [np.sum(v_eci * along, axis=-1), np.sum(v_eci * cross, axis=-1),
             np.sum(v_eci * up, axis=-1)], axis=-1)
        return v_lvlh @ self._r_off

    def body_to_eci_matrix(self, t: float) -> np.ndarray:
        r, v = self.orbit.single_states(np.atleast_1d(float(t)))
        along, cross, up = lvlh_basis(r, v)
        m = np.stack([along[0], cross[0], up[0]], axis=-1)
        return m @ self._r_off


@dataclass
class FixedInertial:
    """Inertially fixed attitude given by a body-to-ECI rotation matrix."""

    r_bi: np.ndarray

    @classmethod
    def z_toward(cls, u_eci: np.ndarray) -> "FixedInertial":
        return cls(minimal_rotation_from_z(unit(np.asarray(u_eci, dtype=float))))

    @classmethod
    def pointing(cls, ra_deg: float, dec_deg: float) -> "FixedInertial":
        """Body +x toward (ra, dec) — e.g. a telescope boresight."""
        x_dir = radec_to_unit(ra_deg, dec_deg)
        r_z_to_x = rodrigues(np.array([0.0, 1.0, 0.0]), np.pi / 2)
        return cls(minimal_rotation_from_z(x_dir) @ r_z_to_x)

    def eci_to_body(self, t: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
        return v_eci @ self.r_bi

    def body_to_eci_matrix(self, t: float) -> np.ndarray:
        return self.r_bi


def spin_body_directions(
    pole: np.ndarray, period_s: float, phase_rad: float, t: np.ndarray,
    v_eci: np.ndarray, body_axis: np.ndarray | tuple = (0.0, 0.0, 1.0)
) -> np.ndarray:
    """Functional form of PrincipalAxisSpin.eci_to_body for inversion search loops."""
    pole = unit(np.asarray(pole, dtype=float))
    theta = 2 * np.pi * np.asarray(t, dtype=float) / period_s + phase_rad
    v_despun = rotate_about_axis(v_eci, pole, -theta)
    return v_despun @ minimal_rotation_between(np.asarray(body_axis, dtype=float), pole)
