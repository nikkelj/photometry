"""Attitude-mode classification: score a bank of attitude hypotheses.

Tier-2 of the design: named operational hypotheses (LVLH-hold, knife-edge,
sun-point) cost nothing to evaluate; free-parameter families (uniform spin,
arbitrary fixed inertial attitude) are fitted by grid search. The fixed
inertial family reuses the spin machinery with an effectively infinite
period — pole (2 DOF) + phase (1 DOF) then spans SO(3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..measurements import ObservationSet
from ..shapes import FacetModel
from .cost import huber_mag_cost, prepare_meas
from .pole_search import PoleSolution, grid_search_pole

INERTIAL_PERIOD_S = 1e12


@dataclass
class HypothesisScore:
    name: str
    cost: float
    attitude: object
    articulate: bool
    spin_solution: PoleSolution | None = None


def classify_modes(
    obs: ObservationSet,
    shape: FacetModel,
    named_hypotheses: dict[str, tuple[object, bool]],
    spin_candidate_periods: list[float],
    max_obs: int = 2000,
    seed: int = 0,
) -> tuple[list[HypothesisScore], HypothesisScore]:
    """Score named hypotheses plus fitted spin and fixed-inertial families.

    named_hypotheses: {name: (attitude_model, articulate)}.
    Returns (all scores sorted by cost, best score).
    """
    rng = np.random.default_rng(seed)
    sub = obs
    if len(obs) > max_obs:
        sub = obs.subset(np.sort(rng.choice(len(obs), max_obs, replace=False)))
    prep = prepare_meas(sub)

    scores: list[HypothesisScore] = []
    for name, (att, articulate) in named_hypotheses.items():
        scores.append(HypothesisScore(
            name, huber_mag_cost(shape, att, articulate, prep), att, articulate))

    from ..attitude import PrincipalAxisSpin

    for fam, periods, axes in [
        ("spin_fit", spin_candidate_periods,
         ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        # a fixed attitude has no meaningful body spin axis: pole+phase
        # already spans SO(3), so one axis suffices
        ("inertial_fit", [INERTIAL_PERIOD_S], ((0.0, 0.0, 1.0),)),
    ]:
        sol = grid_search_pole(sub, shape, candidate_periods=periods,
                               n_poles=250, n_phases=12, max_obs=max_obs,
                               seed=seed, body_axes=axes)
        att = PrincipalAxisSpin(sol.pole_ra_deg, sol.pole_dec_deg,
                                sol.period_s, sol.phase_rad,
                                body_axis=sol.body_axis)
        # re-score with the shared (censoring-aware) cost for fair ranking
        scores.append(HypothesisScore(
            fam, huber_mag_cost(shape, att, False, prep), att, False, sol))

    scores.sort(key=lambda s: s.cost)
    return scores, scores[0]
