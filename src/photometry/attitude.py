"""Target attitude models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frames import minimal_rotation_from_z, radec_to_unit, rotate_about_axis, unit


@dataclass
class PrincipalAxisSpin:
    """Uniform rotation about an inertially fixed spin pole.

    The body +z axis is aligned with the pole; the body spins about it at
    2*pi/period. R(t) maps body-frame vectors to ECI.
    """

    pole_ra_deg: float
    pole_dec_deg: float
    period_s: float
    phase_rad: float = 0.0

    _pole: np.ndarray = field(init=False, repr=False)
    _r0: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._pole = radec_to_unit(self.pole_ra_deg, self.pole_dec_deg)
        self._r0 = minimal_rotation_from_z(self._pole)

    @property
    def pole(self) -> np.ndarray:
        return self._pole

    def angles(self, t: np.ndarray) -> np.ndarray:
        return 2 * np.pi * np.asarray(t, dtype=float) / self.period_s + self.phase_rad

    def eci_to_body(self, t: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
        """Map ECI vectors (K,3) at times t (K,) into the body frame."""
        v_despun = rotate_about_axis(v_eci, self._pole, -self.angles(t))
        return v_despun @ self._r0


def spin_body_directions(
    pole: np.ndarray, period_s: float, phase_rad: float, t: np.ndarray, v_eci: np.ndarray
) -> np.ndarray:
    """Functional form of PrincipalAxisSpin.eci_to_body for inversion search loops."""
    pole = unit(np.asarray(pole, dtype=float))
    theta = 2 * np.pi * np.asarray(t, dtype=float) / period_s + phase_rad
    v_despun = rotate_about_axis(v_eci, pole, -theta)
    return v_despun @ minimal_rotation_from_z(pole)
