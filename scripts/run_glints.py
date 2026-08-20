"""Validate the glint detector + Wahba waypoint tier (chart 14).

For each scenario: run the layered detector against the simulator's own
truth glint labels (specular-dominated rows), then gate/correspond the
survivors under the *fitted* attitude hypothesis from the fleet study (not
truth), solve windowed Davenport q-method waypoints, and score waypoint
attitude error against truth — alongside the hypothesis's own error, so
the plot shows what the waypoints add.

Scenarios:
  katalyst_link__tumble       principal-axis truth, good hypothesis — the
                              clean case: waypoints should track truth
  starlink_v15__tumble        hypothesis locked a symmetry-twin body axis;
                              shows waypoints inheriting twin ambiguity
  katalyst_link__multiaxis_tumble  torque-free truth gated by the best
                              principal-axis approximation — the case the
                              waypoints exist for (dynamics-fit seeding)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from photometry import scenarios as sc
from photometry.attitude import PrincipalAxisSpin
from photometry.inversion.cost import huber_mag_cost, prepare_meas
from photometry.inversion.glint import (
    attitude_error_deg,
    correspond_glints,
    detect_glints,
    fold_filter_applicable,
    phase_fold_filter,
    truth_glint_labels,
    wahba_waypoints,
)
from photometry.measurements import ObservationSet
from photometry.shapes import LIBRARY

OUT = Path("results/glints")

CASES = [
    # scenario, gate cone (deg), waypoint window (s)
    ("katalyst_link__tumble", 4.0, 900.0),
    ("starlink_v15__tumble", 4.0, 3600.0),
    ("katalyst_link__multiaxis_tumble", 10.0, 1200.0),
]


def fitted_hypothesis(name: str, obs: ObservationSet, shape):
    """PrincipalAxisSpin from the fleet-study fit, phase refit (not stored)."""
    r = json.load(open(Path("results/fleet") / name / "result.json"))
    ra, dec = r["spin_pole_est_radec_deg"]
    period = r["spin_period_est_s"]
    axis = tuple(r["spin_body_axis_est"])
    rng = np.random.default_rng(3)
    sub = obs.uncensored()
    if len(sub) > 1200:
        sub = sub.subset(np.sort(rng.choice(len(sub), 1200, replace=False)))
    prep = prepare_meas(sub)

    def cost(phase):
        att = PrincipalAxisSpin(ra, dec, period, float(phase), body_axis=axis)
        return huber_mag_cost(shape, att, False, prep, offset_sigma=0.5)

    grid = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    ph0 = grid[int(np.argmin([cost(p) for p in grid]))]
    res = minimize_scalar(cost, bracket=(ph0 - 0.1, ph0, ph0 + 0.1))
    return PrincipalAxisSpin(ra, dec, period, float(res.x % (2 * np.pi)),
                             body_axis=axis), period


def run_case(name: str, cone_deg: float, window_s: float) -> dict:
    scen = sc.by_name(name)
    shape = LIBRARY[scen.sat]()
    obs = ObservationSet.from_npz(Path("results/fleet") / name / "observations.npz")
    orbit, sun = sc.study_orbit(), sc.sun_eci()
    truth_att, _ = sc.make_attitude(scen.mode, orbit, sun, False)

    # ground truth for scoring only
    truth = truth_glint_labels(obs, shape, truth_att, articulate=False)
    cens = obs.censored.astype(bool)

    hyp, period = fitted_hypothesis(name, obs, shape)
    g_raw = detect_glints(obs, period_s=period)
    fold_ok = fold_filter_applicable(obs, g_raw)
    g = phase_fold_filter(obs, g_raw, period) if fold_ok else g_raw

    def pr(gs):
        """precision/recall on calibrated rows (censored scored separately)."""
        det = np.zeros(len(obs), dtype=bool)
        det[gs.idx] = True
        det_cal, tr_cal = det & ~cens, truth & ~cens
        tp = int((det_cal & tr_cal).sum())
        prec = tp / max(int(det_cal.sum()), 1)
        rec = tp / max(int(tr_cal.sum()), 1)
        return prec, rec, tp

    p_raw, r_raw, _ = pr(g_raw)
    p_fold, r_fold, _ = pr(g)

    pairs = correspond_glints(obs, g, shape, hyp, articulate=False,
                              cone_deg=cone_deg)
    wps = wahba_waypoints(pairs, hyp, window_s=window_s, min_pairs=3)

    wp_err, hyp_err, wp_t, wp_cond = [], [], [], []
    for wp in wps:
        r_true = truth_att.body_to_eci_matrix(wp.t_s)
        wp_err.append(attitude_error_deg(wp.r_bi, r_true))
        hyp_err.append(attitude_error_deg(hyp.body_to_eci_matrix(wp.t_s), r_true))
        wp_t.append(wp.t_s)
        wp_cond.append(wp.condition)

    # oracle variant: gate + propagate with truth. Separates "how much
    # attitude information do the glints carry" (this) from "how much the
    # bootstrap hypothesis can currently harvest" (above); the gap is the
    # iterate-refine headroom.
    pairs_o = correspond_glints(obs, g, shape, truth_att, articulate=False,
                                cone_deg=4.0)
    wps_o = wahba_waypoints(pairs_o, truth_att, window_s=window_s, min_pairs=3)
    owp_err = [attitude_error_deg(wp.r_bi, truth_att.body_to_eci_matrix(wp.t_s))
               for wp in wps_o]
    owp_t = [wp.t_s for wp in wps_o]

    out = dict(
        scenario=name,
        n_rows=len(obs),
        n_censored=int(cens.sum()),
        n_truth_glints_cal=int((truth & ~cens).sum()),
        detector_raw=dict(n=len(g_raw.idx), precision=p_raw, recall=r_raw),
        fold_filter_applied=fold_ok,
        detector_folded=dict(n=len(g.idx), precision=p_fold, recall=r_fold),
        n_pairs=int(len(pairs.t_s)),
        n_waypoints=len(wps),
        pair_gate_cone_deg=cone_deg,
        waypoint_window_s=window_s,
        waypoint_err_deg=wp_err,
        hypothesis_err_deg=hyp_err,
        waypoint_t_s=wp_t,
        waypoint_condition=wp_cond,
        waypoint_err_median=float(np.median(wp_err)) if wp_err else None,
        hypothesis_err_median=float(np.median(hyp_err)) if hyp_err else None,
        oracle=dict(n_pairs=int(len(pairs_o.t_s)), n_waypoints=len(wps_o),
                    err_deg=owp_err,
                    err_median=float(np.median(owp_err)) if owp_err else None),
    )
    # arrays for the chart
    np.savez_compressed(
        OUT / f"{name}.npz",
        t_s=obs.t_s, mag_excess_idx=g_raw.idx, folded_idx=g.idx,
        truth_mask=truth, censored=cens,
        mag_n=-2.5 * np.log10(np.clip(obs.normalized_brightness(), 1e-9, None)),
        pair_row=pairs.row, pair_facet=pairs.facet,
        wp_t=np.array(wp_t), wp_err=np.array(wp_err),
        hyp_err=np.array(hyp_err), wp_cond=np.array(wp_cond),
        owp_t=np.array(owp_t), owp_err=np.array(owp_err),
        period_s=period,
        hyp_curve_t=np.arange(0.0, 86400.0, 600.0),
        hyp_curve_err=np.array([
            attitude_error_deg(hyp.body_to_eci_matrix(t),
                               truth_att.body_to_eci_matrix(t))
            for t in np.arange(0.0, 86400.0, 600.0)]))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for name, cone, win in CASES:
        print(f"=== {name}")
        r = run_case(name, cone, win)
        results.append(r)
        print(f"  detector: raw P={r['detector_raw']['precision']:.3f} "
              f"R={r['detector_raw']['recall']:.3f} -> folded "
              f"P={r['detector_folded']['precision']:.3f} "
              f"R={r['detector_folded']['recall']:.3f}  "
              f"pairs={r['n_pairs']} waypoints={r['n_waypoints']}")
        if r["waypoint_err_median"] is not None:
            print(f"  waypoint err median {r['waypoint_err_median']:.2f} deg "
                  f"(hypothesis {r['hypothesis_err_median']:.2f} deg)")
        o = r["oracle"]
        print(f"  oracle gate: pairs={o['n_pairs']} waypoints={o['n_waypoints']}"
              + (f" err median {o['err_median']:.2f} deg"
                 if o["err_median"] is not None else ""))
    with open(OUT / "summary.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("wrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
