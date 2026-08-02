"""Shape recovery: extended Gaussian image (EGI) by non-negative least squares.

For a fixed attitude trajectory the brightness is linear in the per-facet
albedo-areas rho_d*A (Lambert kernel) AND in the specular
coefficient-areas k_s*A (Phong kernel at fixed exponent) over any candidate
normal set, so the joint solve is one convex NNLS problem. A bank of Phong
exponents brackets unknown lobe widths; residual outliers (e.g. lobe shapes
outside the bank) are still sigma-clipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from ..frames import fibonacci_sphere
from ..measurements import ObservationSet


@dataclass
class EGISolution:
    normals: np.ndarray        # (C,3) candidate normals, body frame
    albedo_area: np.ndarray    # (C,) recovered rho_d * A per candidate normal
    specular_area: np.ndarray  # (C,) recovered k_s * A summed over the exponent bank
    inlier_mask: np.ndarray    # (K,) detections kept after outlier clipping
    residual_rms: float
    phong_exponents: tuple[float, ...]


def lambert_design_matrix(normals: np.ndarray, u_sun_body: np.ndarray,
                          u_obs_body: np.ndarray) -> np.ndarray:
    """(K,C) kernel: brightness per unit albedo-area for each candidate normal."""
    mu_s = np.clip(u_sun_body @ normals.T, 0.0, None)
    mu_o = np.clip(u_obs_body @ normals.T, 0.0, None)
    return mu_s * mu_o / np.pi


def phong_design_matrix(normals: np.ndarray, u_sun_body: np.ndarray,
                        u_obs_body: np.ndarray, exponent: float) -> np.ndarray:
    """(K,C) kernel: brightness per unit k_s*A for a Phong lobe of given exponent."""
    mu_s = np.clip(u_sun_body @ normals.T, 0.0, None)  # (K,C)
    mu_o = np.clip(u_obs_body @ normals.T, 0.0, None)
    # reflection of u_sun about each candidate normal, dotted with u_obs:
    # r.u_o = 2*(n.u_s)*(n.u_o) - u_s.u_o
    su = np.sum(u_sun_body * u_obs_body, axis=-1)  # (K,)
    cos_spec = np.clip(2 * mu_s * mu_o - su[:, None], 0.0, None)
    return (exponent + 2) / (2 * np.pi) * cos_spec**exponent * mu_s * mu_o


def solve_egi(
    obs: ObservationSet,
    u_sun_body: np.ndarray,
    u_obs_body: np.ndarray,
    n_candidates: int = 300,
    phong_exponents: tuple[float, ...] = (20.0, 200.0),
    clip_sigma: float = 4.0,
    max_obs: int = 4000,
    seed: int = 0,
) -> EGISolution:
    rng = np.random.default_rng(seed)
    k = len(obs)
    keep = np.arange(k)
    if k > max_obs:
        keep = np.sort(rng.choice(k, max_obs, replace=False))
    b = obs.normalized_brightness()[keep]
    us, uo = u_sun_body[keep], u_obs_body[keep]
    # brightness sigma from magnitude sigma: dB = 0.4 ln10 * B * dm
    sigma_b = 0.4 * np.log(10) * b * obs.mag_sigma[keep]

    normals = fibonacci_sphere(n_candidates)
    g_lam = lambert_design_matrix(normals, us, uo)
    g_spec = [phong_design_matrix(normals, us, uo, e) for e in phong_exponents]
    g = np.hstack([g_lam, *g_spec])

    inliers = np.ones(len(b), dtype=bool)
    x = np.zeros(g.shape[1])
    for _ in range(3):
        gw = g[inliers] / sigma_b[inliers, None]
        bw = b[inliers] / sigma_b[inliers]
        x, _ = nnls(gw, bw)
        resid = (b - g @ x) / sigma_b
        new_inliers = np.abs(resid) < clip_sigma
        if new_inliers.sum() == inliers.sum():
            break
        inliers = new_inliers

    rms = float(np.sqrt(np.mean(((b - g @ x) / sigma_b)[inliers] ** 2)))
    full_mask = np.zeros(k, dtype=bool)
    full_mask[keep[inliers]] = True
    diffuse = x[:n_candidates]
    specular = x[n_candidates:].reshape(len(phong_exponents), n_candidates).sum(axis=0)
    return EGISolution(normals=normals, albedo_area=diffuse, specular_area=specular,
                       inlier_mask=full_mask, residual_rms=rms,
                       phong_exponents=tuple(phong_exponents))


def unique_normal_groups(normals: np.ndarray, tol_deg: float = 1.0) -> list[np.ndarray]:
    """Group facet indices whose normals coincide (photometry cannot split them)."""
    cosc = np.cos(np.radians(tol_deg))
    groups: list[np.ndarray] = []
    assigned = np.zeros(len(normals), dtype=bool)
    for i in range(len(normals)):
        if assigned[i]:
            continue
        sel = normals @ normals[i] > cosc
        sel &= ~assigned
        groups.append(np.nonzero(sel)[0])
        assigned |= sel
    return groups


def match_to_true_facets(
    sol: EGISolution, true_normals: np.ndarray, cone_deg: float = 15.0
) -> tuple[list[np.ndarray], np.ndarray]:
    """Sum recovered albedo-area within a cone of each *unique* true normal.

    Coincident facet normals (e.g. a solar panel coplanar with a bus face)
    are merged first so overlapping cones are not double-counted. Returns
    (groups, matched_albedo_area) with one entry per unique normal group.
    """
    groups = unique_normal_groups(true_normals)
    cosc = np.cos(np.radians(cone_deg))
    out = np.zeros(len(groups))
    for i, g in enumerate(groups):
        n = true_normals[g[0]]
        sel = sol.normals @ n > cosc
        out[i] = sol.albedo_area[sel].sum()
    return groups, out
