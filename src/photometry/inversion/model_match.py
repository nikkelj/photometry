"""Tier-2 library model matching.

Given an ObservationSet from an unknown target, sweep a library of candidate
shape models x attitude hypotheses x solar-array configurations and score
each with the photometric forward model. The result is an identification:
"best explained as <model> in <attitude mode> with arrays <tracking|frozen>",
with the full ranked table so near-ties are visible.

Absolute brightness matters here: range is known from orbit determination,
so a v1.5-sized and a v2-mini-sized model differ by ~1.7 mag in absolute
terms. Costs therefore use a zero-mean Gaussian prior on the photometric
offset (offset_sigma, mag) rather than a fully free offset — loose enough
to absorb albedo uncertainty, tight enough that size still discriminates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..attitude import FixedInertial, LvlhHold, PrincipalAxisSpin
from ..measurements import ObservationSet
from ..radiometry import facet_brightness
from ..shapes import LIBRARY, FacetModel
from .classify import INERTIAL_PERIOD_S
from .pole_search import grid_search_pole


@dataclass
class MatchResult:
    model: str
    hypothesis: str          # lvlh_ops | lvlh_low_drag | sun_point | spin_fit | inertial_fit
    arrays_tracking: bool
    cost: float
    spin_params: tuple | None = None   # (ra, dec, period, phase, ax, ay, az)

    @property
    def label(self) -> str:
        cfg = "arrays tracking" if self.arrays_tracking else "arrays frozen"
        return f"{self.model} | {self.hypothesis} | {cfg}"


def _huber_offset_cost(shape: FacetModel, attitude, articulate: bool,
                       obs: ObservationSet, meas_mag: np.ndarray,
                       w: np.ndarray, offset_sigma: float) -> float:
    u_sun_body = attitude.eci_to_body(obs.t_s, obs.sun_eci)
    u_obs_body = attitude.eci_to_body(obs.t_s, obs.u_obs_from_target())
    normals = shape.body_normals(u_sun_body, articulate=articulate)
    b = facet_brightness(shape, u_sun_body, u_obs_body, normals).sum(axis=0)
    dm = meas_mag - (-2.5 * np.log10(np.clip(b, 1e-9, None)))
    o = np.sum(w * dm) / (np.sum(w) + 1.0 / offset_sigma**2)
    penalty = (o / offset_sigma) ** 2 / len(dm)
    r = np.sqrt(w) * (dm - o)
    a = 3.0
    rho = np.where(np.abs(r) < a, r**2, 2 * a * np.abs(r) - a**2)
    return float(np.mean(rho) + penalty)


def match_library(
    obs: ObservationSet,
    orbit,
    sun: np.ndarray,
    spin_candidate_periods: list[float],
    candidates: list[str] | None = None,
    offset_sigma: float = 0.5,
    max_obs: int = 1500,
    n_poles: int = 150,
    n_phases: int = 10,
    seed: int = 0,
) -> list[MatchResult]:
    """Score every (model, attitude hypothesis, array config); ranked best-first."""
    names = candidates or [n for n in LIBRARY if n != "rocket_body"]
    rng = np.random.default_rng(seed)
    sub = obs
    if len(obs) > max_obs:
        sub = obs.subset(np.sort(rng.choice(len(obs), max_obs, replace=False)))
    meas_mag = -2.5 * np.log10(np.clip(sub.normalized_brightness(), 1e-6, None))
    w = 1.0 / np.maximum(sub.mag_sigma, 1e-3) ** 2

    named = [
        ("lvlh_ops", LvlhHold(orbit)),
        ("lvlh_low_drag", LvlhHold(orbit, roll_deg=90.0)),
        ("sun_point", FixedInertial.z_toward(sun)),
    ]

    results: list[MatchResult] = []
    for name in names:
        shape = LIBRARY[name]()
        art_options = (True, False) if shape.articulated else (False,)
        for hyp, att in named:
            for art in art_options:
                c = _huber_offset_cost(shape, att, art, sub, meas_mag, w,
                                       offset_sigma)
                results.append(MatchResult(name, hyp, art, c))
        for fam, periods, axes in [
            ("spin_fit", spin_candidate_periods,
             ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
            ("inertial_fit", [INERTIAL_PERIOD_S], ((0.0, 0.0, 1.0),)),
        ]:
            sol = grid_search_pole(sub, shape, candidate_periods=periods,
                                   n_poles=n_poles, n_phases=n_phases,
                                   max_obs=max_obs, seed=seed, body_axes=axes,
                                   offset_sigma=offset_sigma)
            results.append(MatchResult(
                name, fam, False, sol.cost,
                spin_params=(sol.pole_ra_deg, sol.pole_dec_deg, sol.period_s,
                             sol.phase_rad, *sol.body_axis)))

    results.sort(key=lambda r: r.cost)
    return results


def best_per_model(results: list[MatchResult]) -> dict[str, MatchResult]:
    out: dict[str, MatchResult] = {}
    for r in results:
        if r.model not in out:
            out[r.model] = r
    return out
