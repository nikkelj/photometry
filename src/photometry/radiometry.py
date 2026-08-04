"""Photometric forward model: facet fluxes and apparent magnitudes."""

from __future__ import annotations

import numpy as np

from .shapes import FacetModel

SUN_APPARENT_MAG = -26.74


def facet_brightness(
    model: FacetModel,
    u_sun_body: np.ndarray,
    u_obs_body: np.ndarray,
    normals: np.ndarray | None = None,
) -> np.ndarray:
    """Per-facet brightness contributions B_f (F,K) for K geometries.

    B_f = A_f * f_r(u_s,u_o;n_f) * (n_f.u_s) * (n_f.u_o) with a
    Lambert + Phong BRDF. Total flux ratio at range d (same length units as d)
    is sum_f B_f / d^2. Pass `normals` (F,K,3) for articulated facets whose
    normals vary per geometry; default uses the model's rest normals.
    """
    if normals is None:
        n = model.normals  # (F,3)
        mu_s = np.clip(n @ u_sun_body.T, 0.0, None)  # (F,K)
        mu_o = np.clip(n @ u_obs_body.T, 0.0, None)
        refl = 2 * mu_s[..., None] * n[:, None, :] - u_sun_body[None, :, :]
    else:
        mu_s = np.clip(np.einsum("fkj,kj->fk", normals, u_sun_body), 0.0, None)
        mu_o = np.clip(np.einsum("fkj,kj->fk", normals, u_obs_body), 0.0, None)
        refl = 2 * mu_s[..., None] * normals - u_sun_body[None, :, :]
    cos_spec = np.clip(np.einsum("fkj,kj->fk", refl, u_obs_body), 0.0, None)
    f_r = (
        model.rho_d[:, None] / np.pi
        + model.k_s[:, None]
        * (model.n_ph[:, None] + 2)
        / (2 * np.pi)
        * cos_spec ** model.n_ph[:, None]
    )
    return model.areas[:, None] * f_r * mu_s * mu_o


def apparent_magnitude(
    model: FacetModel,
    u_sun_body: np.ndarray,
    u_obs_body: np.ndarray,
    range_km: np.ndarray,
    normals: np.ndarray | None = None,
) -> np.ndarray:
    """Apparent visual magnitude for K geometries (areas m^2, range km)."""
    b_total = facet_brightness(model, u_sun_body, u_obs_body, normals).sum(axis=0)
    flux_ratio = b_total / (np.asarray(range_km) * 1000.0) ** 2
    return SUN_APPARENT_MAG - 2.5 * np.log10(np.clip(flux_ratio, 1e-30, None))


def mag_to_normalized_brightness(mag: np.ndarray, range_km: np.ndarray) -> np.ndarray:
    """Invert magnitude to range-normalized brightness B = F/F_sun * d^2 (m^2 units).

    This is the geometry-only quantity the inversion consumes: range is known
    from multi-observer orbit determination, so the 1/d^2 factor is removed
    exactly.
    """
    flux_ratio = 10 ** (-0.4 * (np.asarray(mag) - SUN_APPARENT_MAG))
    return flux_ratio * (np.asarray(range_km) * 1000.0) ** 2
