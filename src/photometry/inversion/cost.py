"""Shared photometric cost with Tobit-style censoring.

Calibrated rows contribute a Huber-robust standardized magnitude residual.
Censored rows (saturated streaks) contribute a one-sided term: the target
was measurably *brighter* than the sensor cap, so the model is penalized
only when it predicts fainter than the cap. Dropping saturated events —
instead of keeping them as lower bounds — truncates the bright tail and
biases identification toward smaller objects (the ISS-tumble -> "Hubble"
failure mode).

The photometric offset (albedo scale) is solved in closed form on the
calibrated rows: free when offset_sigma is None, otherwise shrunk by a
zero-mean Gaussian prior so absolute brightness stays informative.
"""

from __future__ import annotations

import numpy as np

from ..measurements import ObservationSet
from ..radiometry import facet_brightness
from ..shapes import FacetModel


def prepare_meas(obs: ObservationSet) -> dict:
    """Precompute measurement arrays reused across many cost evaluations."""
    meas_mag = -2.5 * np.log10(np.clip(obs.normalized_brightness(), 1e-6, None))
    cens = obs.censored.astype(bool)
    w = 1.0 / np.maximum(obs.mag_sigma, 1e-3) ** 2
    return dict(obs=obs, meas_mag=meas_mag, cens=cens, w=w)


def huber_mag_cost(
    shape: FacetModel,
    attitude,
    articulate: bool,
    prep: dict,
    offset_sigma: float | None = None,
    huber_a: float = 3.0,
) -> float:
    obs, meas_mag, cens, w = prep["obs"], prep["meas_mag"], prep["cens"], prep["w"]
    u_sun = attitude.eci_to_body(obs.t_s, obs.sun_eci)
    u_obs = attitude.eci_to_body(obs.t_s, obs.u_obs_from_target())
    normals = shape.body_normals(u_sun, articulate=articulate)
    b = facet_brightness(shape, u_sun, u_obs, normals).sum(axis=0)
    model_mag = -2.5 * np.log10(np.clip(b, 1e-9, None))
    dm = meas_mag - model_mag

    ok = ~cens
    if not ok.any():
        return 1e9
    if offset_sigma is None:
        o = np.sum(w[ok] * dm[ok]) / np.sum(w[ok])
        penalty = 0.0
    else:
        o = np.sum(w[ok] * dm[ok]) / (np.sum(w[ok]) + 1.0 / offset_sigma**2)
        penalty = (o / offset_sigma) ** 2 / max(ok.sum(), 1)

    r = np.sqrt(w[ok]) * (dm[ok] - o)
    rho = np.where(np.abs(r) < huber_a, r**2,
                   2 * huber_a * np.abs(r) - huber_a**2)
    cost = float(np.sum(rho))

    if cens.any():
        # violation only when the model (with offset) is fainter than the cap:
        # meas_mag holds the cap, true magnitude <= cap
        viol = np.clip((model_mag[cens] + o) - meas_mag[cens], 0.0, None)
        rc = np.sqrt(w[cens]) * viol
        rho_c = np.where(rc < huber_a, rc**2, 2 * huber_a * rc - huber_a**2)
        cost += float(np.sum(rho_c))

    n = ok.sum() + cens.sum()
    return cost / n + penalty
