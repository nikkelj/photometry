"""Vector and frame utilities (ECI, LVLH, rotations)."""

from __future__ import annotations

import numpy as np

MU_EARTH = 398600.4418  # km^3/s^2
R_EARTH = 6378.137  # km


def unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalize along `axis`; safe for zero vectors (returns zeros)."""
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


def radec_to_unit(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def unit_to_radec(u: np.ndarray) -> tuple[float, float]:
    ra = np.degrees(np.arctan2(u[..., 1], u[..., 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(u[..., 2], -1, 1)))
    return ra, dec


def rodrigues(axis: np.ndarray, angle: float | np.ndarray) -> np.ndarray:
    """Rotation matrix (or stack) for rotation of `angle` about unit `axis`.

    axis: (3,), angle: scalar -> (3,3);  angle: (K,) -> (K,3,3).
    """
    a = np.asarray(axis, dtype=float)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    angle = np.asarray(angle, dtype=float)
    c, s = np.cos(angle), np.sin(angle)
    eye = np.eye(3)
    if angle.ndim == 0:
        return eye * c + s * K + (1 - c) * np.outer(a, a)
    return (
        eye[None] * c[:, None, None]
        + K[None] * s[:, None, None]
        + np.outer(a, a)[None] * (1 - c)[:, None, None]
    )


def rotate_about_axis(v: np.ndarray, axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues rotation of vectors v (...,3) about unit axis by angle (...,)."""
    a = np.asarray(axis, dtype=float)
    angle = np.asarray(angle, dtype=float)[..., None]
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(np.broadcast_to(a, v.shape), v) * s + a * (v @ a)[..., None] * (1 - c)


def minimal_rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector a to unit vector b (minimal angle)."""
    a = unit(np.asarray(a, dtype=float))
    b = unit(np.asarray(b, dtype=float))
    c = float(np.dot(a, b))
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        # 180 deg: rotate about any axis perpendicular to a
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = unit(np.cross(a, helper))
        return rodrigues(axis, np.pi)
    axis = unit(np.cross(a, b))
    return rodrigues(axis, np.arccos(np.clip(c, -1, 1)))


def minimal_rotation_from_z(pole: np.ndarray) -> np.ndarray:
    """Rotation matrix taking body +z to inertial `pole` (minimal-angle rotation)."""
    return minimal_rotation_between(np.array([0.0, 0.0, 1.0]), pole)


def lvlh_basis(r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LVLH unit vectors for states r, v of shape (N,3).

    Returns (along, cross, up): up is radial-out, along is the in-plane
    velocity direction, cross completes the right-handed triad (up x along).
    """
    up = unit(r)
    along = unit(v - up * np.sum(v * up, axis=-1, keepdims=True))
    cross = np.cross(up, along)
    return along, cross, up


def fibonacci_sphere(n: int) -> np.ndarray:
    """n roughly-uniform unit vectors on the sphere, shape (n,3)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1
    )


def in_earth_shadow(r: np.ndarray, sun_dir: np.ndarray) -> np.ndarray:
    """Cylindrical Earth-shadow test for positions r (...,3), sun unit vector (3,)."""
    proj = r @ sun_dir
    perp = np.linalg.norm(r - proj[..., None] * sun_dir, axis=-1)
    return (proj < 0) & (perp < R_EARTH)
