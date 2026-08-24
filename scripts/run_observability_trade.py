"""Observability trade study: fit quality vs observing-constellation size.

The full-fleet simulations carry every detection's `obs_id`, so a smaller
observing constellation is exactly an observer-ID subsample of the same
run: choosing N of the 10,000 satellite IDs and keeping only their rows
reproduces the data a N-satellite shell (same geometry, same tasking)
would have collected. The identical inversion chain — periodogram → pole
search → EGI → glint/Wahba — runs at every size, so the curves measure
how information density converts into fit quality with zero change to
the algorithms.

Scenario: katalyst_link tumble (principal-axis spin, glinting arrays) on
the 24 h / dt 6 s fleet run. A stable pointed target
(starlink_v15 ops) contributes detection-rate scaling only.

Output: results/observability/summary.json (per size x seed rows).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from photometry import scenarios as sc
from photometry.attitude import PrincipalAxisSpin
from photometry.frames import fibonacci_sphere, radec_to_unit, unit
from photometry.inversion.egi import match_to_true_facets, solve_egi
from photometry.inversion.glint import (attitude_error_deg, correspond_glints,
                                        detect_glints, wahba_waypoints)
from photometry.inversion.periodogram import best_period, brightness_periodogram
from photometry.inversion.pole_search import grid_search_pole, pole_error_deg
from photometry.measurements import ObservationSet
from photometry.shapes import LIBRARY

N_CONSTELLATION = 10_000
SIZES = [1, 3, 10, 30, 100, 300, 1000, 3000, 10_000]
SEEDS = [0, 1, 2]
P_TRUE = sc.TUMBLE["period_s"]
POLE_TRUE = radec_to_unit(sc.TUMBLE["pole_ra_deg"], sc.TUMBLE["pole_dec_deg"])


def subsample_constellation(obs: ObservationSet, n: int,
                            seed: int) -> ObservationSet:
    """Rows seen by a random n-satellite subset of the shell."""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(N_CONSTELLATION, size=n, replace=False)
    mask = np.isin(obs.obs_id, chosen)
    return obs.subset(np.nonzero(mask)[0])


def direction_coverage(obs: ObservationSet, n_cells: int = 300) -> dict:
    """Fraction of direction cells sampled around the target (ECI)."""
    grid = fibonacci_sphere(n_cells)
    u_obs = obs.u_obs_from_target()
    pab = unit(obs.sun_eci + u_obs)
    out = {}
    for tag, u in [("obs", u_obs), ("pab", pab)]:
        hit = np.zeros(n_cells, dtype=bool)
        if len(u):
            hit[np.argmax(u @ grid.T, axis=1)] = True
        out[tag] = float(hit.mean())
    return out


def period_metrics(obs: ObservationSet) -> dict:
    t0 = time.time()
    periods, power = brightness_periodogram(obs, period_range_s=(20.0, 600.0))
    p = best_period(periods, power)
    # the pole search receives {p, 2p}; success = either harmonic lands on
    # the true period within 1%
    errs = [abs(cand - P_TRUE) / P_TRUE for cand in (p, 2 * p, p / 2)]
    return dict(period_peak_s=float(p), period_err_frac=float(min(errs)),
                period_ok=bool(min(errs) < 0.01), t_periodogram=time.time() - t0)


def fit_metrics(obs: ObservationSet, per: dict, shape, att_true) -> dict:
    out = {}
    t0 = time.time()
    p = per["period_peak_s"]
    sol = grid_search_pole(obs, shape, candidate_periods=[p, 2 * p],
                           n_poles=100, n_phases=8, max_obs=1200,
                           offset_sigma=0.5)
    out["pole_err_deg"] = pole_error_deg(sol.pole, POLE_TRUE)
    out["period_fit_err_frac"] = abs(sol.period_s - P_TRUE) / P_TRUE
    out["body_axis"] = list(sol.body_axis)
    out["fit_cost"] = sol.cost
    out["t_pole"] = time.time() - t0

    att_fit = PrincipalAxisSpin(sol.pole_ra_deg, sol.pole_dec_deg,
                                sol.period_s, sol.phase_rad,
                                body_axis=sol.body_axis)

    t0 = time.time()
    u_sun_b = att_fit.eci_to_body(obs.t_s, obs.sun_eci)
    u_obs_b = att_fit.eci_to_body(obs.t_s, obs.u_obs_from_target())
    egi = solve_egi(obs, u_sun_b, u_obs_b, n_candidates=200, max_obs=2500)
    _, matched = match_to_true_facets(egi, shape.normals)
    total = float(egi.albedo_area.sum())
    out["egi_within_15deg"] = float(matched.sum() / total) if total > 0 else 0.0
    out["t_egi"] = time.time() - t0

    t0 = time.time()
    glints = detect_glints(obs, period_s=sol.period_s)
    pairs = correspond_glints(obs, glints, shape, att_fit, articulate=False)
    wps = wahba_waypoints(pairs, att_fit, window_s=600.0)
    errs = [attitude_error_deg(w.r_bi, att_true.body_to_eci_matrix(w.t_s))
            for w in wps]
    out.update(n_glint_candidates=int(len(glints.idx)),
               n_wahba_pairs=int(len(pairs.t_s)), n_waypoints=len(wps),
               waypoint_err_median_deg=float(np.median(errs)) if errs else None,
               t_glint=time.time() - t0)
    return out


def main() -> None:
    out_dir = Path("results/observability")
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_full = ObservationSet.from_npz(
        Path("results/fleet/katalyst_link__tumble/observations.npz"))
    obs_stable = ObservationSet.from_npz(
        Path("results/fleet/starlink_v15__ops/observations.npz"))
    shape = LIBRARY["katalyst_link"]()
    orbit, sun = sc.study_orbit(), sc.sun_eci()
    att_true, _ = sc.make_attitude("tumble", orbit, sun, False)

    rows = []
    for n in SIZES:
        seeds = SEEDS if n < N_CONSTELLATION else [0]
        for seed in seeds:
            sub = subsample_constellation(obs_full, n, seed)
            sub_stable = subsample_constellation(obs_stable, n, seed)
            rec = dict(n_sats=n, seed=seed, n_rows=len(sub),
                       n_rows_stable=len(sub_stable),
                       n_observers=int(len(np.unique(sub.obs_id))) if len(sub) else 0,
                       **direction_coverage(sub))
            median_gap = None
            if len(sub) > 1:
                gaps = np.diff(np.sort(sub.t_s))
                median_gap = float(np.median(gaps))
            rec["median_gap_s"] = median_gap
            if len(sub) >= 50:
                per = period_metrics(sub)
                rec.update(per)
                try:
                    rec.update(fit_metrics(sub, per, shape, att_true))
                except Exception as e:  # noqa: BLE001 — record, keep sweeping
                    rec["fit_error"] = str(e)
            print(f"N={n:6d} seed={seed} rows={rec['n_rows']:6d} "
                  f"cov_obs={rec['obs']:.2f} "
                  f"P_ok={rec.get('period_ok', '-')} "
                  f"pole={rec.get('pole_err_deg', float('nan')):.2f} "
                  f"egi={rec.get('egi_within_15deg', float('nan')):.2f} "
                  f"wp={rec.get('n_waypoints', '-')}",
                  flush=True)
            rows.append(rec)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(dict(scenario="katalyst_link__tumble",
                       stable_scenario="starlink_v15__ops",
                       n_constellation=N_CONSTELLATION, period_true_s=P_TRUE,
                       rows=rows), f, indent=1)
    print("wrote", out_dir / "summary.json")


if __name__ == "__main__":
    main()
