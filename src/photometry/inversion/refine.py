"""Matched-model refinement: full-resolution attitude + residual EGI.

Once Tier-2 matching has identified a library model + attitude hypothesis,
this promotes the winner to a refined product:

1. Attitude refinement — for the fitted families (spin / fixed-inertial),
   re-optimize pole/period/phase at full data resolution from the coarse
   match solution. Named operational laws (LVLH-hold etc.) have no free
   parameters.
2. Residual EGI — subtract the matched model's predicted brightness and
   solve a *signed* ridge least-squares EGI on the residuals. Deviations
   from the catalog (a missing/extra panel, a bent array, changed albedo)
   appear as localized positive or negative oriented area; a catalog-true
   target leaves only noise. This is the "does reality match the model"
   product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..attitude import PrincipalAxisSpin
from ..frames import fibonacci_sphere, radec_to_unit
from ..measurements import ObservationSet
from ..radiometry import facet_brightness
from ..shapes import FacetModel
from .egi import lambert_design_matrix


@dataclass
class RefinementResult:
    hypothesis: str
    arrays_tracking: bool
    spin_params: tuple | None       # refined (ra, dec, period, phase, ax, ay, az)
    cost_coarse: float
    cost_refined: float
    residual_normals: np.ndarray    # (C,3) candidate normals, body frame
    residual_albedo_area: np.ndarray  # (C,) SIGNED rho*A deviation vs model
    residual_rms_before: float      # weighted rms of (meas - model) brightness
    residual_rms_after: float       # ... after removing the residual-EGI fit


def _mag_cost(shape, attitude, articulate, obs, meas_mag, w, offset_sigma):
    u_s = attitude.eci_to_body(obs.t_s, obs.sun_eci)
    u_o = attitude.eci_to_body(obs.t_s, obs.u_obs_from_target())
    normals = shape.body_normals(u_s, articulate=articulate)
    b = facet_brightness(shape, u_s, u_o, normals).sum(axis=0)
    dm = meas_mag - (-2.5 * np.log10(np.clip(b, 1e-9, None)))
    o = np.sum(w * dm) / (np.sum(w) + 1.0 / offset_sigma**2)
    r = np.sqrt(w) * (dm - o)
    a = 3.0
    rho = np.where(np.abs(r) < a, r**2, 2 * a * np.abs(r) - a**2)
    return float(np.mean(rho) + (o / offset_sigma) ** 2 / len(dm))


def refine_match(
    obs: ObservationSet,
    shape: FacetModel,
    hypothesis: str,
    arrays_tracking: bool,
    attitude,
    spin_params: tuple | None,
    offset_sigma: float = 0.5,
    max_obs: int = 4000,
    n_residual_candidates: int = 300,
    ridge: float = 1e-2,
    seed: int = 0,
) -> RefinementResult:
    rng = np.random.default_rng(seed)
    sub = obs
    if len(obs) > max_obs:
        sub = obs.subset(np.sort(rng.choice(len(obs), max_obs, replace=False)))
    meas_mag = -2.5 * np.log10(np.clip(sub.normalized_brightness(), 1e-6, None))
    w = 1.0 / np.maximum(sub.mag_sigma, 1e-3) ** 2

    cost_coarse = _mag_cost(shape, attitude, arrays_tracking, sub, meas_mag, w,
                            offset_sigma)
    refined_spin = spin_params
    if hypothesis in ("spin_fit", "inertial_fit") and spin_params is not None:
        ra0, dec0, per0, ph0, ax_, ay, az = spin_params
        axis = (ax_, ay, az)

        def objective(x):
            att = PrincipalAxisSpin(x[0], x[1], x[2], x[3], body_axis=axis)
            return _mag_cost(shape, att, arrays_tracking, sub, meas_mag, w,
                             offset_sigma)

        res = minimize(objective, x0=[ra0, dec0, per0, ph0],
                       method="Nelder-Mead",
                       options=dict(maxiter=800, xatol=1e-4, fatol=1e-8))
        refined_spin = (float(res.x[0] % 360), float(res.x[1]),
                        float(res.x[2]), float(res.x[3] % (2 * np.pi)),
                        *axis)
        attitude = PrincipalAxisSpin(*refined_spin[:4], body_axis=axis)
    cost_refined = _mag_cost(shape, attitude, arrays_tracking, sub, meas_mag, w,
                             offset_sigma)

    # --- signed residual EGI on top of the matched model ------------------
    u_s = attitude.eci_to_body(sub.t_s, sub.sun_eci)
    u_o = attitude.eci_to_body(sub.t_s, sub.u_obs_from_target())
    normals = shape.body_normals(u_s, articulate=arrays_tracking)
    b_model = facet_brightness(shape, u_s, u_o, normals).sum(axis=0)
    b_meas = sub.normalized_brightness()
    resid = b_meas - b_model
    sigma_b = 0.4 * np.log(10) * np.clip(b_meas, 1e-9, None) * sub.mag_sigma

    cand = fibonacci_sphere(n_residual_candidates)
    g = lambert_design_matrix(cand, u_s, u_o)
    gw = g / sigma_b[:, None]
    rw = resid / sigma_b
    # signed ridge solve: deviations may be missing OR extra area
    lhs = gw.T @ gw + ridge * np.trace(gw.T @ gw) / len(cand) * np.eye(len(cand))
    x = np.linalg.solve(lhs, gw.T @ rw)

    rms_before = float(np.sqrt(np.mean(rw**2)))
    rms_after = float(np.sqrt(np.mean((rw - gw @ x) ** 2)))
    return RefinementResult(
        hypothesis=hypothesis, arrays_tracking=arrays_tracking,
        spin_params=refined_spin, cost_coarse=cost_coarse,
        cost_refined=cost_refined,
        residual_normals=cand, residual_albedo_area=x,
        residual_rms_before=rms_before, residual_rms_after=rms_after,
    )
