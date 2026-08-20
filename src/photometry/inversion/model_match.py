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


from .cost import huber_mag_cost, prepare_meas
from .pole_search import ladder_spin_search


def stratified_subsample(obs: ObservationSet, max_obs: int,
                         rng: np.random.Generator,
                         max_censored_frac: float = 0.35) -> ObservationSet:
    """Subsample capping the censored fraction.

    Heavily saturated targets (ISS-class) can be ~80% censored rows; an
    unstratified draw then starves the attitude fit of calibrated
    photometry. Censored rows still enter as brightness lower bounds, just
    capped so they inform size without drowning the light-curve shape.
    """
    if len(obs) <= max_obs:
        return obs
    cal = np.nonzero(obs.censored == 0)[0]
    cen = np.nonzero(obs.censored == 1)[0]
    n_cen = min(len(cen), int(max_censored_frac * max_obs),
                max_obs - min(len(cal), max_obs - int(max_censored_frac * max_obs)))
    n_cal = min(len(cal), max_obs - n_cen)
    pick = np.concatenate([
        rng.choice(cal, n_cal, replace=False) if n_cal < len(cal) else cal,
        rng.choice(cen, n_cen, replace=False) if n_cen < len(cen) else cen,
    ])
    return obs.subset(np.sort(pick))


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
    refine_top_k: int = 3,
    library: dict | None = None,
) -> list[MatchResult]:
    """Score every (model, attitude hypothesis, array config); ranked best-first.

    Two stages: a coarse sweep over the full library, then a finer attitude
    search (denser pole/phase grid, larger sample) for the top-k models —
    a coarse grid can leave a hard-to-fit true model stuck behind a
    mediocre-everywhere impostor.

    `library` swaps in an alternative name->builder dict (e.g. the
    200-entry generated library); default is the curated shapes.LIBRARY.
    """
    lib = library or LIBRARY
    names = candidates or [n for n in lib if n != "rocket_body"]
    rng = np.random.default_rng(seed)
    sub = stratified_subsample(obs, max_obs, rng)
    prep = prepare_meas(sub)

    named = [
        ("lvlh_ops", LvlhHold(orbit)),
        ("lvlh_low_drag", LvlhHold(orbit, roll_deg=90.0)),
        ("sun_point", FixedInertial.z_toward(sun)),
    ]

    def score_model(name: str, sub_, prep_, np_, nph_, mo_,
                    use_ladder=False) -> list[MatchResult]:
        shape = lib[name]()
        out: list[MatchResult] = []
        art_options = (True, False) if shape.articulated else (False,)
        for hyp, att in named:
            for art in art_options:
                c = huber_mag_cost(shape, att, art, prep_, offset_sigma)
                out.append(MatchResult(name, hyp, art, c))
        for fam, periods, axes in [
            ("spin_fit", spin_candidate_periods,
             ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
            ("inertial_fit", [INERTIAL_PERIOD_S], ((0.0, 0.0, 1.0),)),
        ]:
            if fam == "spin_fit" and use_ladder:
                sol = ladder_spin_search(sub_, shape, window_s=3000.0,
                                         max_obs=mo_, seed=seed,
                                         offset_sigma=offset_sigma)
            else:
                sol = grid_search_pole(sub_, shape, candidate_periods=periods,
                                       n_poles=np_, n_phases=nph_,
                                       max_obs=mo_, seed=seed, body_axes=axes,
                                       offset_sigma=offset_sigma)
            # the grid search fits calibrated rows only; re-score with the
            # shared censored cost so families rank on the same footing —
            # and under both array configs: an inertially pointed telescope
            # still sun-tracks its arrays
            att = PrincipalAxisSpin(sol.pole_ra_deg, sol.pole_dec_deg,
                                    sol.period_s, sol.phase_rad,
                                    body_axis=sol.body_axis)
            spin_p = (sol.pole_ra_deg, sol.pole_dec_deg, sol.period_s,
                      sol.phase_rad, *sol.body_axis)
            for art in art_options:
                c = huber_mag_cost(shape, att, art, prep_, offset_sigma)
                out.append(MatchResult(name, fam, art, c, spin_params=spin_p))
        return out

    results: list[MatchResult] = []
    for name in names:
        results.extend(score_model(name, sub, prep, n_poles, n_phases, max_obs))
    results.sort(key=lambda r: r.cost)

    if refine_top_k > 0:
        finalists = list(dict.fromkeys(r.model for r in results))[:refine_top_k]
        sub2 = stratified_subsample(obs, 2 * max_obs, np.random.default_rng(seed + 1))
        prep2 = prepare_meas(sub2)
        # heavy saturation guts the periodogram (bright spin phases are the
        # censored ones), so run the coherent period-ladder search instead
        use_ladder = float(np.mean(obs.censored)) > 0.5
        keep = [r for r in results if r.model not in finalists]
        for name in finalists:
            keep.extend(score_model(name, sub2, prep2, 2 * n_poles,
                                    n_phases + 6, 2 * max_obs,
                                    use_ladder=use_ladder))
        results = sorted(keep, key=lambda r: r.cost)
    return results


def best_per_model(results: list[MatchResult]) -> dict[str, MatchResult]:
    out: dict[str, MatchResult] = {}
    for r in results:
        if r.model not in out:
            out[r.model] = r
    return out
