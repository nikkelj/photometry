"""Day-in-the-life fleet study: simulate + invert each satellite x mode scenario.

Usage:
  python scripts/run_fleet_study.py                    # all scenarios
  python scripts/run_fleet_study.py name1 name2 ...    # a subset

Per scenario writes results/fleet/<name>/{observations.npz,result.json,inversion.npz}.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from photometry import scenarios as sc
from photometry.constellation import WalkerConstellation
from photometry.inversion.classify import classify_modes
from photometry.inversion.egi import solve_egi
from photometry.inversion.periodogram import best_period, brightness_periodogram
from photometry.sensing import simulate_detections


def run_scenario(s: sc.FleetScenario, outroot: Path) -> dict:
    out = outroot / s.name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(s.seed)
    shell = WalkerConstellation()
    orbit = sc.study_orbit()
    sun = sc.sun_eci()
    shape = s.shape()
    att_true, articulate_true = sc.make_attitude(s.mode, orbit, sun,
                                                 shape.articulated)

    t0 = time.time()
    t_grid = np.arange(0.0, s.duration_s, s.dt_s)
    obs = simulate_detections(shell, orbit, shape, att_true, sun, t_grid,
                              s.sensors, rng, articulate=articulate_true)
    t_sim = time.time() - t0
    obs.to_npz(out / "observations.npz")

    periods, power = brightness_periodogram(obs, period_range_s=(20.0, 8000.0),
                                            n_periods=3000)
    p_ls = best_period(periods, power)
    # orbital harmonics dominate the full band for controlled targets; also
    # take the strongest sub-orbital peak as a spin candidate
    band = periods < 2000.0
    p_band = best_period(periods[band], power[band])
    spin_candidates = sorted({round(p, 3) for p in
                              (p_ls, 2 * p_ls, p_band, 2 * p_band)})

    named = {
        "lvlh_ops": (sc.make_attitude("ops", orbit, sun, shape.articulated)[0],
                     shape.articulated),
        "lvlh_low_drag": (sc.make_attitude("low_drag", orbit, sun, False)[0], False),
        "sun_point": (sc.make_attitude("sun_point", orbit, sun, False)[0], False),
    }
    t0 = time.time()
    scores, best = classify_modes(obs, shape, named,
                                  spin_candidate_periods=spin_candidates,
                                  seed=s.seed)
    t_cls = time.time() - t0

    # EGI under the best-hypothesis attitude
    u_sun_body = best.attitude.eci_to_body(obs.t_s, obs.sun_eci)
    u_obs_body = best.attitude.eci_to_body(obs.t_s, obs.u_obs_from_target())
    egi = solve_egi(obs, u_sun_body, u_obs_body)

    # truth-mode expectation for scoring the classifier
    expected = {"ops": "lvlh_ops", "low_drag": "lvlh_low_drag",
                "tumble": "spin_fit", "sun_point": "sun_point",
                "science": "inertial_fit", "safe_sun": "sun_point"}[s.mode]
    # sun_point and inertial_fit describe the same physical attitude family;
    # accept inertial_fit when a fixed sun-pointing truth is recovered
    accept = {expected} | ({"inertial_fit"} if expected == "sun_point" else set())
    mode_correct = best.name in accept

    result = dict(
        scenario=s.name, sat=s.sat, mode_true=s.mode,
        real_altitude_km=s.real_altitude_km,
        study_altitude_km=sc.STUDY_ALT_KM,
        n_detections=int(len(obs)),
        n_observers=int(len(np.unique(obs.obs_id))),
        sim_seconds=round(t_sim, 1), classify_seconds=round(t_cls, 1),
        period_ls_s=float(p_ls),
        hypothesis_costs={x.name: float(x.cost) for x in scores},
        mode_best=best.name, mode_expected=expected,
        mode_correct=bool(mode_correct),
        egi_total_albedo_area_m2=float(egi.albedo_area.sum()),
        egi_total_specular_area_m2=float(egi.specular_area.sum()),
        egi_residual_rms=float(egi.residual_rms),
        true_diffuse_albedo_area_m2=float(shape.diffuse_albedo_area().sum()),
    )
    if best.spin_solution is not None:
        result["spin_period_est_s"] = float(best.spin_solution.period_s)
        result["spin_pole_est_radec_deg"] = [
            float(best.spin_solution.pole_ra_deg),
            float(best.spin_solution.pole_dec_deg)]
        result["spin_body_axis_est"] = list(best.spin_solution.body_axis)

    np.savez_compressed(
        out / "inversion.npz",
        periods=periods, power=power,
        egi_normals=egi.normals, egi_albedo_area=egi.albedo_area,
        egi_specular_area=egi.specular_area,
        best_mode=np.array(best.name),
        best_spin_params=np.array(
            [best.spin_solution.pole_ra_deg, best.spin_solution.pole_dec_deg,
             best.spin_solution.period_s, best.spin_solution.phase_rad,
             *best.spin_solution.body_axis]
            if best.spin_solution is not None else [np.nan] * 7),
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main(*names: str) -> None:
    outroot = Path("results/fleet")
    todo = [sc.by_name(n) for n in names] if names else sc.SCENARIOS
    rows = []
    for s in todo:
        print(f"=== {s.name} ===", flush=True)
        r = run_scenario(s, outroot)
        rows.append(r)
        print(f"  {r['n_detections']} det | best={r['mode_best']} "
              f"(expected {r['mode_expected']}, "
              f"{'OK' if r['mode_correct'] else 'MISS'}) | "
              f"EGI rho*A {r['egi_total_albedo_area_m2']:.1f} "
              f"(truth {r['true_diffuse_albedo_area_m2']:.1f}) | "
              f"sim {r['sim_seconds']}s cls {r['classify_seconds']}s", flush=True)
    if not names:
        (outroot / "fleet_summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
