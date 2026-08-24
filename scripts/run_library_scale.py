"""Catalog-scale identification test: 209-model library, sampled targets.

The operational question the curated 9-model study cannot answer: does
identification survive a hypothesis space of hundreds of models? For each
sampled target (mixed families, countries, attitude modes, array-control
modes) the full funnel runs exactly as an operational system would:

  periodogram -> two-channel shortlist (209 -> k) -> full matcher on the
  shortlist -> ranked identification

Reported per target: whether the truth survived the shortlist, its final
rank, the top-1 identification and its family, the cost margin, and
timing. Two targets fly with arrays deliberately off-pointed 25 deg — a
control-state the matcher has no hypothesis for — to measure robustness
to hypothesis-space gaps.
"""

from __future__ import annotations

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from photometry import scenarios as sc
from photometry.attitude import FixedInertial, LvlhHold, PrincipalAxisSpin
from photometry.constellation import WalkerConstellation
from photometry.frames import radec_to_unit, unit_to_radec
from photometry.inversion.model_match import best_per_model, match_library
from photometry.inversion.periodogram import best_period, brightness_periodogram
from photometry.inversion.prefilter import model_features, shortlist_library
from photometry.library200 import full_library
from photometry.sensing import SensorConfig, simulate_detections

OUT = Path("results/library_scale")
N_GENERATED = 20
CURATED = [("starlink_v2mini", "ops", "track"), ("katalyst_link", "tumble", "track"),
           ("hubble", "science", "track"), ("bluewalker3", "tumble", "track")]
DURATION_S = 3 * 3600.0
DT_S = 6.0
SHORTLIST_K = 16

_LIB = None
_META = None
_CACHE = None


def sample_targets(seed: int = 42) -> list[dict]:
    lib, meta = full_library()
    rng = np.random.default_rng(seed)
    gen = [m for m in meta]
    # spread across families: round-robin over shuffled family groups
    by_fam: dict[str, list] = {}
    for m in gen:
        by_fam.setdefault(m["family"], []).append(m)
    for v in by_fam.values():
        rng.shuffle(v)
    fams = sorted(by_fam)
    picks = []
    i = 0
    while len(picks) < N_GENERATED:
        fam = fams[i % len(fams)]
        if by_fam[fam]:
            picks.append(by_fam[fam].pop())
        i += 1
    targets = []
    for j, m in enumerate(picks):
        mode = str(rng.choice(m["attitude_modes"]))
        if m["array_modes"]:
            arr = str(rng.choice(m["array_modes"]))
        else:
            arr = "n/a"
        targets.append(dict(model=m["name"], family=m["family"],
                            country=m["country"], mode=mode, array_mode=arr,
                            seed=1000 + j))
    # force two articulated targets into the off-pointed control state the
    # matcher has no hypothesis for
    forced = 0
    for t in targets:
        if forced < 2 and t["array_mode"] == "track":
            m = next(x for x in meta if x["name"] == t["model"])
            if m["articulated"] and "offset" in m["array_modes"]:
                t["array_mode"] = "offset"
                forced += 1
    for j, (name, mode, arr) in enumerate(CURATED):
        fam = "curated"
        targets.append(dict(model=name, family=fam, country="-", mode=mode,
                            array_mode=arr, seed=2000 + j))
    return targets


def make_truth_attitude(mode: str, orbit, sun, rng: np.random.Generator):
    if mode == "ops":
        return LvlhHold(orbit)
    if mode == "low_drag":
        return LvlhHold(orbit, roll_deg=90.0)
    if mode in ("sun_point", "safe_sun"):
        return FixedInertial.z_toward(sun)
    if mode == "science":
        return FixedInertial.pointing(rng.uniform(0, 360), rng.uniform(-60, 60))
    if mode == "tumble":
        pole = rng.normal(size=3)
        ra, dec = unit_to_radec(pole / np.linalg.norm(pole))
        period = float(rng.uniform(40.0, 500.0))
        axis = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)][rng.integers(3)]
        return PrincipalAxisSpin(float(ra), float(dec), period,
                                 float(rng.uniform(0, 2 * np.pi)),
                                 body_axis=axis)
    raise ValueError(mode)


def run_target(t: dict) -> dict:
    global _LIB, _META, _CACHE
    lib = _LIB
    orbit, sun = sc.study_orbit(), sc.sun_eci()
    rng = np.random.default_rng(t["seed"])
    shape = lib[t["model"]]()
    att = make_truth_attitude(t["mode"], orbit, sun, rng)
    articulate = t["array_mode"] in ("track", "offset")
    offset = 25.0 if t["array_mode"] == "offset" else 0.0

    t0 = time.time()
    constellation = WalkerConstellation(100, 100, 550.0, 53.0)
    t_grid = np.arange(0.0, DURATION_S, DT_S)
    try:
        obs = simulate_detections(constellation, orbit, shape, att, sun,
                                  t_grid, SensorConfig(), rng,
                                  articulate=articulate,
                                  articulate_offset_deg=offset)
    except RuntimeError:
        return dict(t, status="undetectable", n_rows=0)
    t_sim = time.time() - t0

    t0 = time.time()
    if len(obs.uncensored()) >= 50:
        periods, power = brightness_periodogram(obs, period_range_s=(20.0, 900.0))
        p_ls = best_period(periods, power)
    else:
        p_ls = 120.0
    shortlist, fo = shortlist_library(obs, lib, orbit, sun, k=SHORTLIST_K,
                                      feature_cache=_CACHE)
    t_short = time.time() - t0
    in_short = t["model"] in shortlist
    short_rank = shortlist.index(t["model"]) + 1 if in_short else None

    t0 = time.time()
    results = match_library(
        obs, orbit, sun, spin_candidate_periods=[p_ls, 2 * p_ls],
        candidates=shortlist, library=lib, offset_sigma=0.5, max_obs=900,
        n_poles=48, n_phases=8, seed=1, refine_top_k=2)
    t_match = time.time() - t0

    per_model = list(best_per_model(results).items())
    ranked = [n for n, _ in per_model]
    top1_name, top1 = per_model[0]
    rank = ranked.index(t["model"]) + 1 if t["model"] in ranked else None
    true_cost = dict(per_model)[t["model"]].cost if t["model"] in ranked else None
    fam_of = {m["name"]: m["family"] for m in _META}
    return dict(
        t, status="ok", n_rows=len(obs),
        censored_frac=float(np.mean(obs.censored)),
        period_ls_s=float(p_ls),
        shortlist=shortlist, in_shortlist=in_short, shortlist_rank=short_rank,
        top1=top1_name, top1_family=fam_of.get(top1_name, "curated"),
        top1_cost=float(top1.cost), top1_label=top1.label,
        true_rank=rank, true_cost=float(true_cost) if true_cost else None,
        margin=float(dict(per_model)[t["model"]].cost / max(top1.cost, 1e-9))
        if t["model"] in ranked else None,
        t_sim=t_sim, t_shortlist=t_short, t_match=t_match,
    )


def _init():
    global _LIB, _META, _CACHE
    _LIB, _META = full_library()
    _CACHE = {}
    for name, build in _LIB.items():
        _CACHE[name] = model_features(build())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    targets = sample_targets()
    print(f"{len(targets)} targets")
    _init()  # parent builds the feature cache once; fork shares it
    with Pool(3) as pool:
        rows = pool.map(run_target, targets)
    ok = [r for r in rows if r["status"] == "ok"]
    top1_hits = sum(1 for r in ok if r["true_rank"] == 1)
    short_hits = sum(1 for r in ok if r["in_shortlist"])
    fam_hits = sum(1 for r in ok if r["true_rank"] == 1
                   or r["top1_family"] == r["family"])
    summary = dict(
        n_targets=len(rows),
        n_ok=len(ok),
        n_undetectable=sum(1 for r in rows if r["status"] == "undetectable"),
        shortlist_recall=short_hits / max(len(ok), 1),
        top1_accuracy=top1_hits / max(len(ok), 1),
        family_accuracy=fam_hits / max(len(ok), 1),
        library_size=len(_LIB),
        shortlist_k=SHORTLIST_K,
        rows=rows,
    )
    with open(OUT / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['model']:20s} {r['mode']:10s} {r['status']}")
            continue
        print(f"{r['model']:20s} {r['mode']:10s} arr={r['array_mode']:6s} "
              f"short={'Y' if r['in_shortlist'] else 'N'} "
              f"rank={r['true_rank']} top1={r['top1']} "
              f"({r['t_match']:.0f}s)")
    print(f"\nshortlist recall {summary['shortlist_recall']:.2f}  "
          f"top-1 {summary['top1_accuracy']:.2f}  "
          f"family {summary['family_accuracy']:.2f}")
    print("wrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
