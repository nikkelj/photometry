"""Unified-registry identification with SATCAT priors: end-to-end demo.

Three cases against the full 300-model unified registry (catalog
families + generated library + intel annex), 3 h fleet arcs:

  1. ICEYE SAR tumbling, associated identity "ICEYE-X44" — the prior
     names the exact template; SAR twins from the generated library are
     expected inside the photometric equivalence class.
  2. A 3U cubesat in ops, associated identity "LEMUR-2 KAREN-B" — the
     motivating case: the catalog-scale study showed cubesat twins are
     photometrically undecidable (au_cube01 -> nl_cube01 etc.); the
     mapping should settle exactly that draw.
  3. Yaogan-SAR (annex truth) with identity "YAOGAN-33", which maps only
     to the low-confidence `classified_unpublished` placeholder — the
     safety demo: a weak/wrong catalog hint must NOT override decisive
     photometry, and only acts inside the twin margin.

Each case runs the funnel twice-in-one: photometric shortlist seeded
with the prior families, full match on the shortlist, then the ranking
with and without the prior rerank. Output: results/unified_id/summary.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from photometry import scenarios as sc
from photometry.attitude import LvlhHold, PrincipalAxisSpin
from photometry.constellation import WalkerConstellation
from photometry.inversion.model_match import best_per_model, match_library
from photometry.inversion.periodogram import best_period, brightness_periodogram
from photometry.inversion.prefilter import shortlist_library
from photometry.registry import (rerank_with_prior, satcat_prior,
                                 seed_shortlist, unified_library)
from photometry.sensing import SensorConfig, simulate_detections

DURATION_S = 3 * 3600.0
DT_S = 6.0

CASES = [
    dict(truth="iceye", mode="tumble", identity="ICEYE-X44"),
    # NOTE: the honest-scale cubesat_3u template was tried here and is
    # invisible to the fleet (zero detections in 3 h — a real 3U at these
    # ranges sits below the limiting magnitude; the generated-library
    # cubesats that caused twin confusion carry oversized deployables).
    # OneWeb vs the generated box-wing comms is the bright twin case.
    dict(truth="oneweb", mode="ops", identity="ONEWEB-0611"),
    dict(truth="cn_yaogan_sar", mode="ops", identity="YAOGAN-33"),
]


def make_attitude(mode: str, orbit, rng):
    if mode == "ops":
        return LvlhHold(orbit)
    pole = rng.normal(size=3)
    pole /= np.linalg.norm(pole)
    from photometry.frames import unit_to_radec
    ra, dec = unit_to_radec(pole)
    return PrincipalAxisSpin(float(ra), float(dec),
                             float(rng.uniform(60.0, 300.0)),
                             float(rng.uniform(0, 2 * np.pi)),
                             body_axis=(1.0, 0.0, 0.0))


def main() -> None:
    out_dir = Path("results/unified_id")
    out_dir.mkdir(parents=True, exist_ok=True)
    lib, meta = unified_library()
    print(f"unified registry: {len(lib)} models", flush=True)
    orbit, sun = sc.study_orbit(), sc.sun_eci()
    cache: dict = {}
    rows = []
    for i, case in enumerate(CASES):
        rng = np.random.default_rng(300 + i)
        shape = lib[case["truth"]]()
        att = make_attitude(case["mode"], orbit, rng)
        constellation = WalkerConstellation(100, 100, 550.0, 53.0)
        t_grid = np.arange(0.0, DURATION_S, DT_S)
        try:
            obs = simulate_detections(constellation, orbit, shape, att, sun,
                                      t_grid, SensorConfig(), rng,
                                      articulate=shape.articulated)
        except RuntimeError:
            rows.append(dict(case=case, status="undetectable", n_rows=0))
            print(f"{case['truth']:16s} {case['mode']:6s} UNDETECTABLE "
                  "(below limiting magnitude)", flush=True)
            continue

        prior = satcat_prior([case["identity"]])
        t0 = time.time()
        if len(obs.uncensored()) >= 50:
            periods, power = brightness_periodogram(obs, period_range_s=(20.0, 900.0))
            p_ls = best_period(periods, power)
        else:
            p_ls = 120.0
        shortlist, fo = shortlist_library(obs, lib, orbit, sun, k=16,
                                          feature_cache=cache)
        shortlist = seed_shortlist(shortlist, prior, lib)
        t_short = time.time() - t0

        t0 = time.time()
        results = match_library(
            obs, orbit, sun, spin_candidate_periods=[p_ls, 2 * p_ls],
            candidates=shortlist, library=lib, offset_sigma=0.5, max_obs=900,
            n_poles=48, n_phases=8, seed=1, refine_top_k=2)
        t_match = time.time() - t0

        ranked = [(n, float(r.cost)) for n, r in best_per_model(results).items()]
        reranked, info = rerank_with_prior(ranked, prior)

        def rank_of(seq, name):
            names = [n for n, _ in seq]
            return names.index(name) + 1 if name in names else None

        rec = dict(
            case=case, n_rows=len(obs), prior=prior,
            shortlist=shortlist,
            prior_seeded=[n for n in shortlist if prior.get(n, 0.0) > 0],
            photometric=[(n, c) for n, c in ranked[:8]],
            reranked=[(n, c) for n, c in reranked[:8]],
            rank_before=rank_of(ranked, case["truth"]),
            rank_after=rank_of(reranked, case["truth"]),
            twin_class=info["twin_class"], twin_names=info.get("twin_names"),
            prior_applied=info["prior_applied"],
            changed_top1=info["changed_top1"],
            t_shortlist=t_short, t_match=t_match,
        )
        rows.append(rec)
        print(f"{case['truth']:16s} {case['mode']:6s} id='{case['identity']}' "
              f"rank {rec['rank_before']} -> {rec['rank_after']} "
              f"(twin class {rec['twin_class']}, top1 changed: "
              f"{rec['changed_top1']}) [{t_match:.0f}s]", flush=True)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(dict(n_models=len(lib), rows=rows), f, indent=1)
    print("wrote", out_dir / "summary.json")


if __name__ == "__main__":
    main()
