"""Spin-pole and phase estimation given a shape hypothesis.

Global grid search over pole direction x spin phase (with candidate periods,
typically the periodogram peak and its double), followed by local refinement.
The photometric offset (albedo scale) is solved analytically per candidate,
so the search is insensitive to overall albedo-area scaling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..attitude import spin_body_directions
from ..frames import fibonacci_sphere, unit, unit_to_radec
from ..measurements import ObservationSet
from ..radiometry import facet_brightness
from ..shapes import FacetModel


@dataclass
class PoleSolution:
    pole: np.ndarray
    period_s: float
    phase_rad: float
    cost: float
    pole_ra_deg: float
    pole_dec_deg: float
    grid_poles: np.ndarray
    grid_costs: np.ndarray
    body_axis: tuple = (0.0, 0.0, 1.0)


def _model_mags(shape, pole, period, phase, obs: ObservationSet,
                body_axis=(0.0, 0.0, 1.0)) -> np.ndarray:
    u_sun_body = spin_body_directions(pole, period, phase, obs.t_s, obs.sun_eci,
                                      body_axis)
    u_obs_body = spin_body_directions(pole, period, phase, obs.t_s,
                                      obs.u_obs_from_target(), body_axis)
    b = facet_brightness(shape, u_sun_body, u_obs_body).sum(axis=0)
    return -2.5 * np.log10(np.clip(b, 1e-9, None))


def _cost(shape, pole, period, phase, obs, meas_mag, w,
          body_axis=(0.0, 0.0, 1.0), offset_sigma=None, cens=None) -> float:
    model_mag = _model_mags(shape, pole, period, phase, obs, body_axis)
    dm = meas_mag - model_mag
    ok = slice(None) if cens is None else ~cens
    if offset_sigma is None:
        o = np.sum(w[ok] * dm[ok]) / np.sum(w[ok])  # free offset (albedo scale)
        penalty = 0.0
    else:
        # zero-mean Gaussian prior on the offset keeps absolute brightness
        # informative (range is known) while absorbing albedo uncertainty
        o = np.sum(w[ok] * dm[ok]) / (np.sum(w[ok]) + 1.0 / offset_sigma**2)
        penalty = (o / offset_sigma) ** 2 / len(dm)
    # Huber-style soft clip so specular-glint mismatches don't dominate
    a = 3.0
    r = np.sqrt(w[ok]) * (dm[ok] - o)
    rho = np.where(np.abs(r) < a, r**2, 2 * a * np.abs(r) - a**2)
    total = float(np.sum(rho))
    n = len(dm) if cens is None else int(np.sum(~cens))
    if cens is not None and cens.any():
        # censored rows: penalize only when the model predicts fainter than
        # the saturation cap — the saturation timing pattern carries the
        # spin when the bright phases themselves are censored
        viol = np.clip((model_mag[cens] + o) - meas_mag[cens], 0.0, None)
        rc = np.sqrt(w[cens]) * viol
        total += float(np.sum(np.where(rc < a, rc**2, 2 * a * rc - a**2)))
        n += int(cens.sum())
    return total / max(n, 1) + penalty


def grid_search_pole(
    obs: ObservationSet,
    shape: FacetModel,
    candidate_periods: list[float],
    n_poles: int = 400,
    n_phases: int = 16,
    max_obs: int = 2500,
    seed: int = 0,
    body_axes: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    offset_sigma: float | None = None,
    include_censored: bool = False,
) -> PoleSolution:
    """Grid over pole x phase x candidate period x body spin axis, then refine.

    The body spin axis is which principal body axis lies along the pole —
    a flat spin and a propeller tumble of the same object photometrically
    differ enormously, so it is part of the hypothesis space. With
    include_censored, saturated rows enter as one-sided brighter-than-cap
    terms — for heavily saturated targets the saturation timing pattern is
    where the spin information lives.
    """
    rng = np.random.default_rng(seed)
    if not include_censored:
        obs = obs.uncensored()  # saturated rows carry no calibrated magnitude
    if len(obs) > max_obs:
        obs = obs.subset(np.sort(rng.choice(len(obs), max_obs, replace=False)))
    cens = obs.censored.astype(bool) if include_censored else None

    meas_mag = -2.5 * np.log10(np.clip(obs.normalized_brightness(), 1e-6, None))
    w = 1.0 / np.maximum(obs.mag_sigma, 1e-3) ** 2
    # spin states with pole p are equivalent to -p with reversed phase rate;
    # a hemisphere grid (dec >= 0 relative to arbitrary axis) would bias the
    # reported pole, so search the full sphere and report the best mode.
    poles = fibonacci_sphere(n_poles)
    phases = np.linspace(0, 2 * np.pi, n_phases, endpoint=False)

    grid_costs = np.full(len(poles), np.inf)
    best = (np.inf, None)
    for axis in body_axes:
        for period in candidate_periods:
            for i, p in enumerate(poles):
                c_best = np.inf
                for phase in phases:
                    c = _cost(shape, p, period, phase, obs, meas_mag, w, axis,
                              offset_sigma, cens)
                    if c < c_best:
                        c_best = c
                    if c < best[0]:
                        best = (c, (p, period, phase, axis))
                grid_costs[i] = min(grid_costs[i], c_best)

    p0, period0, phase0, axis0 = best[1]
    ra0, dec0 = unit_to_radec(p0)

    def objective(x):
        ra, dec, per, ph = x
        pole = np.array(
            [
                np.cos(np.radians(dec)) * np.cos(np.radians(ra)),
                np.cos(np.radians(dec)) * np.sin(np.radians(ra)),
                np.sin(np.radians(dec)),
            ]
        )
        return _cost(shape, pole, per, ph, obs, meas_mag, w, axis0,
                     offset_sigma, cens)

    res = minimize(
        objective,
        x0=[ra0, dec0, period0, phase0],
        method="Nelder-Mead",
        options=dict(maxiter=600, xatol=1e-3, fatol=1e-6),
    )
    ra, dec, per, ph = res.x
    pole = unit(
        np.array(
            [
                np.cos(np.radians(dec)) * np.cos(np.radians(ra)),
                np.cos(np.radians(dec)) * np.sin(np.radians(ra)),
                np.sin(np.radians(dec)),
            ]
        )
    )
    return PoleSolution(
        pole=pole,
        period_s=float(per),
        phase_rad=float(ph % (2 * np.pi)),
        cost=float(res.fun),
        pole_ra_deg=float(ra % 360),
        pole_dec_deg=float(dec),
        grid_poles=poles,
        grid_costs=grid_costs,
        body_axis=tuple(float(a) for a in axis0),
    )


def pole_error_deg(est_pole: np.ndarray, true_pole: np.ndarray) -> float:
    """Angular error, accounting for the spin-axis sign ambiguity."""
    c = abs(float(np.dot(unit(est_pole), unit(true_pole))))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def ladder_spin_search(
    obs: ObservationSet,
    shape: FacetModel,
    period_range_s: tuple[float, float] = (40.0, 800.0),
    window_s: float = 2000.0,
    n_poles: int = 120,
    n_phases: int = 10,
    max_obs: int = 1500,
    seed: int = 0,
    body_axes: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    offset_sigma: float | None = None,
    include_censored: bool = True,
) -> PoleSolution:
    """Coherent period-ladder spin search for when the periodogram fails.

    Over a long arc the cost basin in period has width ~P^2/arc (sub-second
    for a day), far too narrow for any coarse sweep. So: (1) dense period
    grid on a short window, spaced at the window's own basin width
    P^2/(2*window) so no basin is skipped; (2) grid over pole x phase x
    body axis at each candidate; (3) re-polish (period, phase, pole) over
    progressively longer arcs, each stage's accuracy bracketing the next
    stage's basin. Censored rows are kept by default: the ladder mostly
    serves saturation-dominated targets, whose spin signature lives in
    the saturation timing pattern.
    """
    if not include_censored:
        obs = obs.uncensored()
    t0 = float(obs.t_s.min())
    arc = float(obs.t_s.max() - t0)

    # ladder-density period candidates for the window
    periods = []
    p = period_range_s[0]
    while p <= period_range_s[1]:
        periods.append(p)
        p += p * p / (2.0 * window_s)
    win = obs.subset(np.nonzero(obs.t_s <= t0 + window_s)[0])
    sol = grid_search_pole(win, shape, candidate_periods=periods,
                           n_poles=n_poles, n_phases=n_phases, max_obs=max_obs,
                           seed=seed, body_axes=body_axes,
                           offset_sigma=offset_sigma,
                           include_censored=include_censored)

    # extend coherence: refine on growing arcs
    span = window_s
    while span < arc:
        span = min(4 * span, arc)
        sub = obs.subset(np.nonzero(obs.t_s <= t0 + span)[0])
        rng = np.random.default_rng(seed)
        if len(sub) > max_obs:
            sub = sub.subset(np.sort(rng.choice(len(sub), max_obs, replace=False)))
        meas_mag = -2.5 * np.log10(np.clip(sub.normalized_brightness(), 1e-6, None))
        w = 1.0 / np.maximum(sub.mag_sigma, 1e-3) ** 2
        cens = sub.censored.astype(bool) if include_censored else None
        axis = sol.body_axis

        def objective(x):
            ra, dec, per, ph = x
            pole = np.array([np.cos(np.radians(dec)) * np.cos(np.radians(ra)),
                             np.cos(np.radians(dec)) * np.sin(np.radians(ra)),
                             np.sin(np.radians(dec))])
            return _cost(shape, pole, per, ph, sub, meas_mag, w, axis,
                         offset_sigma, cens)

        res = minimize(objective,
                       x0=[sol.pole_ra_deg, sol.pole_dec_deg, sol.period_s,
                           sol.phase_rad],
                       method="Nelder-Mead",
                       options=dict(maxiter=500, xatol=1e-4, fatol=1e-7))
        ra, dec, per, ph = res.x
        pole = unit(np.array([np.cos(np.radians(dec)) * np.cos(np.radians(ra)),
                              np.cos(np.radians(dec)) * np.sin(np.radians(ra)),
                              np.sin(np.radians(dec))]))
        sol = PoleSolution(pole=pole, period_s=float(per),
                           phase_rad=float(ph % (2 * np.pi)),
                           cost=float(res.fun), pole_ra_deg=float(ra % 360),
                           pole_dec_deg=float(dec),
                           grid_poles=sol.grid_poles, grid_costs=sol.grid_costs,
                           body_axis=axis)
    return sol
