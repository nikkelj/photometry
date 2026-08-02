"""Run the baseline simulation and inversion, saving results for charting.

Usage: python scripts/run_pipeline.py [outdir]

Outputs (default results/):
  observations.npz / observations.csv  — the measurement set (real-data schema)
  truth.json                            — scenario truth parameters
  inversion.npz                         — periodogram, pole grid, EGI solution
  summary.json                          — headline recovery metrics
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from photometry.attitude import PrincipalAxisSpin, spin_body_directions
from photometry.inversion.egi import match_to_true_facets, solve_egi
from photometry.inversion.periodogram import best_period, brightness_periodogram
from photometry.inversion.pole_search import grid_search_pole, pole_error_deg
from photometry.simulate import Scenario, run


def main(outdir: str = "results") -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    print("Simulating constellation detections...")
    t0 = time.time()
    obs, sc = run(Scenario())
    print(f"  {len(obs)} detections from {len(np.unique(obs.obs_id))} observers "
          f"in {time.time()-t0:.1f}s")
    obs.to_npz(out / "observations.npz")
    obs.to_csv(out / "observations.csv")
    (out / "truth.json").write_text(json.dumps(sc.truth_dict(), indent=2))

    shape = sc.target_shape()
    truth_att = sc.target_attitude()

    print("Periodogram...")
    periods, power = brightness_periodogram(obs)
    p_ls = best_period(periods, power)
    print(f"  Lomb-Scargle peak: {p_ls:.2f} s (truth {sc.spin_period_s:.2f} s)")

    print("Pole grid search + refinement...")
    t0 = time.time()
    # two-fold symmetric shapes fold the light curve at half the spin period,
    # so try the periodogram peak and its double
    sol = grid_search_pole(obs, shape, candidate_periods=[p_ls, 2 * p_ls])
    err_pole = pole_error_deg(sol.pole, truth_att.pole)
    err_period = abs(sol.period_s - sc.spin_period_s)
    print(f"  pole (ra,dec)=({sol.pole_ra_deg:.1f},{sol.pole_dec_deg:.1f}) deg, "
          f"err {err_pole:.2f} deg; period {sol.period_s:.3f} s "
          f"(err {err_period*1000:.0f} ms) in {time.time()-t0:.1f}s")

    print("EGI shape inversion (using estimated attitude)...")
    est_att_pole = sol.pole
    u_sun_body = spin_body_directions(est_att_pole, sol.period_s, sol.phase_rad,
                                      obs.t_s, obs.sun_eci)
    u_obs_body = spin_body_directions(est_att_pole, sol.period_s, sol.phase_rad,
                                      obs.t_s, obs.u_obs_from_target())
    egi = solve_egi(obs, u_sun_body, u_obs_body)
    true_aa = shape.diffuse_albedo_area()
    groups, matched = match_to_true_facets(egi, shape.normals)
    group_true_aa = np.array([true_aa[g].sum() for g in groups])
    group_labels = np.array(
        [" + ".join(shape.labels[i] for i in g) for g in groups]
    )
    total_true = true_aa.sum()
    total_rec = egi.albedo_area.sum()
    capture = matched.sum() / max(total_rec, 1e-9)
    print(f"  recovered total rho*A {total_rec:.2f} m^2 (truth {total_true:.2f}); "
          f"{100*capture:.0f}% of recovered area within 15 deg of true normals; "
          f"glints clipped: {np.sum(~egi.inlier_mask)}")

    # truth body-frame geometry for charts (coverage plots)
    u_obs_body_true = truth_att.eci_to_body(obs.t_s, obs.u_obs_from_target())
    u_sun_body_true = truth_att.eci_to_body(obs.t_s, obs.sun_eci)

    np.savez_compressed(
        out / "inversion.npz",
        periods=periods, power=power, best_period_ls=p_ls,
        grid_poles=sol.grid_poles, grid_costs=sol.grid_costs,
        est_pole=sol.pole, est_period=sol.period_s, est_phase=sol.phase_rad,
        egi_normals=egi.normals, egi_albedo_area=egi.albedo_area,
        egi_inlier_mask=egi.inlier_mask, egi_residual_rms=egi.residual_rms,
        true_normals=shape.normals, true_albedo_area=true_aa,
        group_true_albedo_area=group_true_aa, matched_albedo_area=matched,
        group_labels=group_labels, facet_labels=np.array(shape.labels),
        u_obs_body_true=u_obs_body_true, u_sun_body_true=u_sun_body_true,
    )

    summary = dict(
        n_detections=int(len(obs)),
        n_observers=int(len(np.unique(obs.obs_id))),
        arc_hours=float((obs.t_s.max() - obs.t_s.min()) / 3600),
        median_range_km=float(np.median(obs.range_km)),
        phase_angle_span_deg=[float(obs.phase_angle_deg().min()),
                              float(obs.phase_angle_deg().max())],
        period_ls_s=float(p_ls),
        period_est_s=float(sol.period_s),
        period_true_s=float(sc.spin_period_s),
        period_err_ms=float(err_period * 1000),
        pole_est_radec_deg=[float(sol.pole_ra_deg), float(sol.pole_dec_deg)],
        pole_true_radec_deg=[float(sc.pole_ra_deg), float(sc.pole_dec_deg)],
        pole_err_deg=float(err_pole),
        egi_total_albedo_area_m2=float(total_rec),
        egi_true_total_albedo_area_m2=float(total_true),
        egi_capture_frac_15deg=float(capture),
        egi_residual_rms=float(egi.residual_rms),
        n_glints_clipped=int(np.sum(~egi.inlier_mask)),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote", out / "summary.json")


if __name__ == "__main__":
    main(*sys.argv[1:])
