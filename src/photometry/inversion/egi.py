"""Shape recovery: extended Gaussian image (EGI) by non-negative least squares.

For a fixed attitude trajectory the Lambertian brightness is linear in the
per-facet albedo-areas rho_d*A over any candidate normal set, so the EGI
solve is a convex NNLS problem. Specular glints violate the Lambert kernel
and are removed by sigma-clipping before the final fit.
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
    inlier_mask: np.ndarray    # (K,) detections kept after glint clipping
    residual_rms: float


def lambert_design_matrix(normals: np.ndarray, u_sun_body: np.ndarray,
                          u_obs_body: np.ndarray) -> np.ndarray:
    """(K,C) kernel: brightness per unit albedo-area for each candidate normal."""
    mu_s = np.clip(u_sun_body @ normals.T, 0.0, None)
    mu_o = np.clip(u_obs_body @ normals.T, 0.0, None)
    return mu_s * mu_o / np.pi


def solve_egi(
    obs: ObservationSet,
    u_sun_body: np.ndarray,
    u_obs_body: np.ndarray,
    n_candidates: int = 300,
    clip_sigma: float = 3.0,
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
    g = lambert_design_matrix(normals, us, uo)

    inliers = np.ones(len(b), dtype=bool)
    x = np.zeros(n_candidates)
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
    return EGISolution(normals=normals, albedo_area=x, inlier_mask=full_mask,
                       residual_rms=rms)


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
