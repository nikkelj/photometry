"""Train and evaluate the learned spin-state proposer (SpinNet).

Stages (all outputs under results/learn/):
  train   set-transformer on geometry-pool re-rendered examples; held-out
          shapes never seen in training
  eval    (a) 40 held-out synthetic windows: net raw / net+polish / grid
          (b) the fleet-study tumble scenarios (24 h full sims): same
  tf      torque-free seeding on the multi-axis LINK scenario (stretch)

Usage: python scripts/run_spin_net.py [--steps N] [--skip-train] [--tf]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from photometry import scenarios as sc
from photometry.attitude import PrincipalAxisSpin
from photometry.frames import unit_to_radec
from photometry.learn.data import GeometryPool
from photometry.learn.evaluate import (evaluate_case, load_model, polish,
                                       propose, synthetic_case)
from photometry.measurements import ObservationSet
from photometry.registry import unified_library
from photometry.studies import attitude_error_deg

OUT = Path("results/learn")
WIDTH_S = 7200.0
N_TOKENS = 256
TRAIN_SHAPES = ["katalyst_link", "box_wing", "starlink_v2mini", "ru_persona",
                "cn_yaogan_sar", "oneweb", "iceye", "cn_yaogan_eo",
                "iridium_next", "planet_skysat", "ru_lotos_s", "capella"]
HELDOUT_SHAPES = ["starlink_v15", "hubble", "bluewalker3", "jp_igs_radar",
                  "umbra", "starlink_v2mini_dtc"]
POOL_RUNS = ["katalyst_link__tumble", "starlink_v15__ops",
             "starlink_v2mini__tumble", "hubble__tumble"]
FLEET_TUMBLES = ["katalyst_link__tumble", "starlink_v15__tumble",
                 "starlink_v2mini__tumble", "starlink_v2mini_dtc__tumble",
                 "bluewalker3__tumble", "hubble__tumble"]


def fleet_truth() -> PrincipalAxisSpin:
    return PrincipalAxisSpin(sc.TUMBLE["pole_ra_deg"], sc.TUMBLE["pole_dec_deg"],
                             sc.TUMBLE["period_s"], sc.TUMBLE["phase_rad"],
                             body_axis=(1.0, 0.0, 0.0))


def main() -> None:
    global WIDTH_S
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--n-synth", type=int, default=40)
    ap.add_argument("--tf", action="store_true")
    ap.add_argument("--width", type=float, default=WIDTH_S,
                    help="window length (s); the Tier-0 period ceiling rises "
                         "with arc length")
    args = ap.parse_args()
    WIDTH_S = args.width
    OUT.mkdir(parents=True, exist_ok=True)

    lib, _ = unified_library()
    pool = GeometryPool.from_runs(
        [Path("results/fleet") / r / "observations.npz" for r in POOL_RUNS])
    print(f"geometry pool: {len(pool.t_s)} rows from {len(POOL_RUNS)} runs",
          flush=True)

    if not args.skip_train:
        from photometry.learn.train import train
        shapes = [lib[n]() for n in TRAIN_SHAPES]
        train(pool, shapes, OUT, steps=args.steps, batch=args.batch,
              n_tokens=N_TOKENS, width_s=WIDTH_S)
    model = load_model(OUT / "spin_net.pt")

    rng = np.random.default_rng(1234)
    summary: dict = dict(train_shapes=TRAIN_SHAPES, heldout_shapes=HELDOUT_SHAPES,
                         width_s=WIDTH_S, n_tokens=N_TOKENS)

    # (a) held-out synthetic windows
    synth = []
    heldout = [lib[n]() for n in HELDOUT_SHAPES]
    for i in range(args.n_synth):
        shape = heldout[i % len(heldout)]
        obs, truth = synthetic_case(pool, shape, rng, WIDTH_S)
        r = evaluate_case(model, obs, shape, truth, rng, WIDTH_S, N_TOKENS)
        r.update(shape=shape.name, period_true=truth.period_s)
        synth.append(r)
        print(f"synth {i:2d} {shape.name:20s} P={truth.period_s:6.1f}s "
              f"net {r['net_raw']['pole_err_deg']:5.1f}deg/"
              f"{r['net_raw']['period_err_frac']*100:5.1f}% -> polished "
              f"{r['net_polished']['pole_err_deg']:5.2f}deg "
              f"[{r['net_polished']['t_total_s']:.1f}s] | grid "
              f"{r['grid']['pole_err_deg']:5.2f}deg [{r['grid']['t_s']:.0f}s]",
              flush=True)
    summary["synthetic"] = synth

    # (b) fleet-study tumble scenarios (full simulator, 24 h)
    fleet = []
    truth = fleet_truth()
    for name in FLEET_TUMBLES:
        p = Path("results/fleet") / name / "observations.npz"
        if not p.exists():
            continue
        obs = ObservationSet.from_npz(p)
        shape = lib[name.split("__")[0]]()
        # a 2 h window with enough rows, away from the arc start
        for t0 in np.arange(3600.0, 20 * 3600.0, 1800.0):
            sel = np.nonzero((obs.t_s >= t0) & (obs.t_s < t0 + WIDTH_S))[0]
            if len(sel) >= 150:
                break
        win = obs.subset(sel)
        r = evaluate_case(model, win, shape, truth, rng, WIDTH_S, N_TOKENS)
        r.update(scenario=name, t0=float(t0))
        fleet.append(r)
        print(f"fleet {name:28s} net {r['net_raw']['pole_err_deg']:5.1f}deg/"
              f"{r['net_raw']['period_err_frac']*100:5.1f}% -> polished "
              f"{r['net_polished']['pole_err_deg']:5.2f}deg "
              f"[{r['net_polished']['t_total_s']:.1f}s] | grid "
              f"{r['grid']['pole_err_deg']:5.2f}deg [{r['grid']['t_s']:.0f}s]",
              flush=True)
    summary["fleet"] = fleet

    # (c) torque-free seeding (stretch)
    if args.tf:
        from photometry.inversion.torquefree import fit_torque_free
        name = "katalyst_link__multiaxis_tumble"
        obs = ObservationSet.from_npz(Path("results/fleet") / name / "observations.npz")
        shape = lib["katalyst_link"]()
        orbit, sun = sc.study_orbit(), sc.sun_eci()
        att_true, _ = sc.make_attitude("multiaxis_tumble", orbit, sun, False)
        t0 = float(obs.t_s.min())
        win = obs.subset(np.nonzero(obs.t_s < t0 + WIDTH_S)[0])
        prop = propose(model, win, WIDTH_S, N_TOKENS, rng)
        pol = polish(win, shape, prop)
        ra, dec = unit_to_radec(pol["pole"])
        seed_spin = PrincipalAxisSpin(float(ra), float(dec), pol["period_s"],
                                      pol["phase_rad"], body_axis=pol["axis"])
        tic = time.time()
        fit = fit_torque_free(obs, shape, seed_spin,
                              seed_periods=[prop.period_s, 2 * prop.period_s])
        horizon = 3600.0
        att_fit = fit.attitude(t_max=horizon + 60)
        errs = [attitude_error_deg(att_fit.body_to_eci_matrix(float(t)),
                                   att_true.body_to_eci_matrix(float(t)))
                for t in fit.t_ref + np.arange(0.0, horizon, 30.0)]
        summary["torque_free"] = dict(
            net_period=prop.period_s, seed_pole_err=float(
                np.degrees(np.arccos(abs(float(pol["pole"] @ att_true.body_to_eci_matrix(t0)[:, 0]))))),
            cost=fit.cost, cost_uniform_seed=fit.cost_uniform_seed,
            inertia_est=list(fit.inertia), omega0_est=list(fit.omega0_body),
            att_err_median_deg=float(np.median(errs)),
            att_err_mean_deg=float(np.mean(errs)), t_s=time.time() - tic)
        print("torque-free seeded:", {k: (round(v, 3) if isinstance(v, float) else v)
                                      for k, v in summary["torque_free"].items()},
              flush=True)

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("wrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
