"""Evaluation: learned proposal -> physics polish, against the grid search.

The go/no-go for a learned proposer is not its raw accuracy but whether
the classical fit, started from the net's proposal, lands in the right
basin — faster and at least as often as the brute-force grid. So every
comparison here runs the *same* forward-model cost (`pole_search._cost`)
from two different starting points: the net's millisecond proposal, or
the grid search's exhaustive sweep.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize

from ..attitude import PrincipalAxisSpin
from ..frames import unit, unit_to_radec
from ..inversion.periodogram import best_period, brightness_periodogram
from ..inversion.pole_search import _cost, grid_search_pole, pole_error_deg
from ..measurements import ObservationSet
from .data import AXES, GeometryPool, random_spin, render_window, tokens_from_obs
from .model import SpinNet


def load_model(path, model_kw: dict | None = None) -> SpinNet:
    m = SpinNet(**(model_kw or {}))
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


@dataclass
class Proposal:
    pole: np.ndarray
    period_s: float
    axis_idx: int
    axis_probs: np.ndarray
    t_ms: float
    p_ls: float = 0.0


@torch.no_grad()
def propose(model: SpinNet, obs: ObservationSet, width_s: float,
            n_tokens: int, rng: np.random.Generator,
            n_draws: int = 4) -> Proposal:
    """Net proposal for one window, averaged over token subsamples:
    poles combine axially (principal eigenvector of sum p p^T)."""
    t0 = float(obs.t_s.min())
    tic = time.time()
    poles, logps, probs = [], [], []
    # full Tier-0 statistic (arc-sized grid, all calibrated rows) — the same
    # periodogram the classical baseline gets; training used a row-capped
    # coarse-grid version for speed
    if len(obs.uncensored()) >= 8:
        periods, power = brightness_periodogram(obs, period_range_s=(20.0, 900.0))
        p_ls = best_period(periods, power)
    else:
        p_ls = None
    for _ in range(n_draws):
        tok, mask, p_ls = tokens_from_obs(obs, t0, width_s, n_tokens, rng, p_ls)
        p, lp, ax = model(torch.from_numpy(tok[None]), torch.from_numpy(mask[None]))
        poles.append(p[0].numpy())
        logps.append(float(lp[0]))
        probs.append(torch.softmax(ax[0], -1).numpy())
    P = np.stack(poles)
    w, v = np.linalg.eigh(P.T @ P)
    pole = unit(v[:, -1])
    pr = np.mean(probs, axis=0)
    # period = periodogram period x the net's harmonic correction
    period = p_ls * float(np.exp(np.median(logps)))
    return Proposal(pole, period, int(np.argmax(pr)), pr,
                    (time.time() - tic) * 1e3, float(p_ls))


def _prep(obs: ObservationSet, max_obs: int, rng: np.random.Generator):
    o = obs.uncensored()
    if len(o) > max_obs:
        o = o.subset(np.sort(rng.choice(len(o), max_obs, replace=False)))
    meas = -2.5 * np.log10(np.clip(o.normalized_brightness(), 1e-6, None))
    w = 1.0 / np.maximum(o.mag_sigma, 1e-3) ** 2
    return o, meas, w


def polish(obs: ObservationSet, shape, prop: Proposal,
           offset_sigma: float = 0.5, max_obs: int = 1200, seed: int = 0,
           n_phases: int = 8, harmonics=(1.0, 2.0)) -> dict:
    """Nelder-Mead from the proposal (phase gridded; period harmonics
    {P, 2P} as the classical pipeline does). Same cost as the grid search."""
    rng = np.random.default_rng(seed)
    o, meas, w = _prep(obs, max_obs, rng)
    axis = AXES[prop.axis_idx]
    tic = time.time()
    best = (np.inf, None)
    for h in harmonics:
        per = prop.period_s * h
        for ph in np.linspace(0, 2 * np.pi, n_phases, endpoint=False):
            c = _cost(shape, prop.pole, per, ph, o, meas, w, axis, offset_sigma)
            if c < best[0]:
                best = (c, (per, ph))
    per0, ph0 = best[1]
    ra0, dec0 = unit_to_radec(prop.pole)

    def f(x):
        ra, dec, per, ph = x
        pole = np.array([np.cos(np.radians(dec)) * np.cos(np.radians(ra)),
                         np.cos(np.radians(dec)) * np.sin(np.radians(ra)),
                         np.sin(np.radians(dec))])
        return _cost(shape, pole, per, ph, o, meas, w, axis, offset_sigma)

    res = minimize(f, [ra0, dec0, per0, ph0], method="Nelder-Mead",
                   options=dict(maxiter=600, xatol=1e-3, fatol=1e-6))
    for _ in range(2):
        r2 = minimize(f, res.x, method="Nelder-Mead",
                      options=dict(maxiter=600, xatol=1e-5, fatol=1e-9))
        if r2.fun < res.fun:
            res = r2
    ra, dec, per, ph = res.x
    pole = unit(np.array([np.cos(np.radians(dec)) * np.cos(np.radians(ra)),
                          np.cos(np.radians(dec)) * np.sin(np.radians(ra)),
                          np.sin(np.radians(dec))]))
    return dict(pole=pole, period_s=float(per), phase_rad=float(ph % (2 * np.pi)),
                axis=axis, cost=float(res.fun), t_s=time.time() - tic)


def baseline(obs: ObservationSet, shape, offset_sigma: float = 0.5,
             n_poles: int = 120, n_phases: int = 10, max_obs: int = 1200,
             seed: int = 0) -> dict:
    """The classical path: periodogram -> {P, 2P} -> full pole x phase x
    axis grid -> simplex. Same cost function, exhaustive start."""
    tic = time.time()
    periods, power = brightness_periodogram(obs, period_range_s=(20.0, 900.0))
    p = best_period(periods, power)
    sol = grid_search_pole(obs, shape, candidate_periods=[p, 2 * p],
                           n_poles=n_poles, n_phases=n_phases, max_obs=max_obs,
                           seed=seed, offset_sigma=offset_sigma)
    return dict(pole=sol.pole, period_s=sol.period_s, phase_rad=sol.phase_rad,
                axis=sol.body_axis, cost=sol.cost, period_ls=float(p),
                t_s=time.time() - tic)


def period_err_frac(p_est: float, p_true: float) -> float:
    """Relative error, harmonic-tolerant ({P, 2P, P/2} all count)."""
    return float(min(abs(p_est * h - p_true) / p_true for h in (1.0, 2.0, 0.5)))


def score(sol: dict, truth: PrincipalAxisSpin) -> dict:
    return dict(pole_err_deg=pole_error_deg(sol["pole"], truth.pole),
                period_err_frac=period_err_frac(sol["period_s"], truth.period_s),
                axis_ok=bool(tuple(sol["axis"]) == tuple(truth.body_axis)),
                cost=sol["cost"], t_s=sol["t_s"])


def synthetic_case(pool: GeometryPool, shape, rng: np.random.Generator,
                   width_s: float, min_rows: int = 64):
    """A held-out (ObservationSet window, truth) pair from the geometry pool."""
    for _ in range(50):
        sel = pool.window(rng, width_s)
        att = random_spin(rng)
        keep, mag, cens = render_window(pool, sel, shape, att, rng)
        if len(keep) < min_rows:
            continue
        rows = sel[keep]
        k = len(rows)
        obs = ObservationSet(
            t_s=pool.t_s[rows], obs_id=np.zeros(k, dtype=int),
            tracker_id=np.zeros(k, dtype=int), obs_pos_km=np.zeros((k, 3)),
            los_eci=pool.los_eci[rows], sun_eci=pool.sun_eci[rows],
            range_km=pool.range_km[rows], mag=mag,
            mag_sigma=np.where(cens, 0.3, 0.08), sensor_bias=np.zeros(k),
            censored=cens.astype(int))
        return obs, att
    raise RuntimeError("no detectable synthetic case")


def evaluate_case(model: SpinNet, obs: ObservationSet, shape,
                  truth: PrincipalAxisSpin, rng: np.random.Generator,
                  width_s: float, n_tokens: int, run_baseline: bool = True) -> dict:
    prop = propose(model, obs, width_s, n_tokens, rng)
    raw = dict(pole=prop.pole, period_s=prop.period_s, axis=AXES[prop.axis_idx],
               cost=np.nan, t_s=prop.t_ms / 1e3)
    out = dict(n_rows=len(obs), net_raw=score(raw, truth),
               p_ls=prop.p_ls,
               # Tier-0 solvable: the periodogram landed within 1% of the
               # period or a harmonic — the regime any method can work in
               ls_ok=bool(period_err_frac(prop.p_ls, truth.period_s) < 0.01))
    pol = polish(obs, shape, prop)
    out["net_polished"] = score(pol, truth)
    out["net_polished"]["t_total_s"] = pol["t_s"] + prop.t_ms / 1e3
    if run_baseline:
        out["grid"] = score(baseline(obs, shape), truth)
    return out
