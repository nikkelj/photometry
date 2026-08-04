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


@dataclass
class TorqueFreeTumble:
    """Torque-free rigid-body rotation for a triaxial inertia tensor.

    Integrates Euler's equations (body frame) and the attitude kinematics
    R_dot = R [omega]x with RK4 onto a table at `dt` resolution, then
    evaluates R(t) with an exact-in-omega sub-step rotation. This is the
    general non-principal-axis tumble: omega nutates around the angular
    momentum vector unless started on a principal axis.

    inertia: principal moments (I1, I2, I3), any scale (ratios matter)
    omega0_body: initial body-frame rate vector (rad/s)
    r0: body-to-ECI attitude at t = t_ref
    """

    inertia: tuple[float, float, float]
    omega0_body: tuple[float, float, float]
    r0: np.ndarray
    t_max: float
    dt: float = 0.5
    t_ref: float = 0.0

    _r_table: np.ndarray = field(init=False, repr=False)
    _w_table: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        i_diag = np.asarray(self.inertia, dtype=float)
        inv_i = 1.0 / i_diag

        def wdot(w):
            return inv_i * (-np.cross(w, i_diag * w))

        n = int(np.ceil(self.t_max / self.dt)) + 2
        r_tab = np.empty((n, 3, 3))
        w_tab = np.empty((n, 3))
        r = np.asarray(self.r0, dtype=float).copy()
        w = np.asarray(self.omega0_body, dtype=float).copy()
        h = self.dt
        for k in range(n):
            r_tab[k], w_tab[k] = r, w
            # RK4 for omega; attitude via exact rotation about mid-step omega
            k1 = wdot(w)
            k2 = wdot(w + 0.5 * h * k1)
            k3 = wdot(w + 0.5 * h * k2)
            k4 = wdot(w + h * k3)
            w_mid = w + (h / 12.0) * (k1 + 4 * k2 + k3)  # ~mid-step rate
            w = w + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            ang = np.linalg.norm(w_mid) * h
            if ang > 0:
                r = r @ rodrigues(w_mid / np.linalg.norm(w_mid), ang)
            if k % 200 == 0:  # re-orthonormalize
                u, _, vt = np.linalg.svd(r)
                r = u @ vt
        self._r_table, self._w_table = r_tab, w_tab

    def _r_at(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tt = np.clip(np.asarray(t, dtype=float) - self.t_ref, 0.0, self.t_max)
        k = np.clip((tt / self.dt).astype(int), 0, len(self._r_table) - 1)
        return self._r_table[k], self._w_table[k], tt - k * self.dt

    def eci_to_body(self, t: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
        r_k, w_k, dt_sub = self._r_at(t)
        v0 = np.einsum("kji,kj->ki", r_k, v_eci)  # R_k^T v
        wn = np.linalg.norm(w_k, axis=-1)
        ok = wn > 0
        axis = np.where(ok[:, None], w_k / np.maximum(wn, 1e-15)[:, None], 0.0)
        theta = -(wn * dt_sub)  # inverse of the sub-step body rotation
        c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
        return (v0 * c + np.cross(axis, v0) * s
                + axis * np.sum(axis * v0, axis=-1, keepdims=True) * (1 - c))

    def body_to_eci_matrix(self, t: float) -> np.ndarray:
        r_k, w_k, dt_sub = self._r_at(np.atleast_1d(float(t)))
        r_k, w_k, dt_sub = r_k[0], w_k[0], float(dt_sub[0])
        wn = float(np.linalg.norm(w_k))
        if wn == 0:
            return r_k
        return r_k @ rodrigues(w_k / wn, wn * dt_sub)


def spin_body_directions(
    pole: np.ndarray, period_s: float, phase_rad: float, t: np.ndarray,
    v_eci: np.ndarray, body_axis: np.ndarray | tuple = (0.0, 0.0, 1.0)
) -> np.ndarray:
    """Functional form of PrincipalAxisSpin.eci_to_body for inversion search loops."""
    pole = unit(np.asarray(pole, dtype=float))
    theta = 2 * np.pi * np.asarray(t, dtype=float) / period_s + phase_rad
    v_despun = rotate_about_axis(v_eci, pole, -theta)
    return v_despun @ minimal_rotation_between(np.asarray(body_axis, dtype=float), pole)
