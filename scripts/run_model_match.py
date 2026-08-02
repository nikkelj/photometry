"""Library model matching demo: identify model + attitude + array config.

For each requested fleet scenario, pretend the target is unknown: sweep the
whole model library x attitude hypotheses x array configurations against
the scenario's observations and report the ranked identification.

Usage: python scripts/run_model_match.py [scenario ...]
Writes results/model_match/<scenario>.json and a summary chart.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from photometry import scenarios as sc
from photometry.inversion.model_match import best_per_model, match_library
from photometry.inversion.periodogram import best_period, brightness_periodogram
from photometry.measurements import ObservationSet
from photometry.shapes import LIBRARY

SURFACE = "#1a1a19"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
BASELINE = "#383835"
S3_AQUA = "#199e70"
RED = "#e66767"
SEQ = LinearSegmentedColormap.from_list("seq", [
    "#0d366b", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#cde2fb"])

DEFAULT_SCENARIOS = [
    "katalyst_link__tumble", "starlink_v15__tumble", "starlink_v2mini__ops",
    "starlink_v2mini_dtc__ops", "bluewalker3__sun_point", "hubble__science",
    "iss__ops",
]


def run_one(name: str, outdir: Path) -> dict:
    s = sc.by_name(name)
    obs = ObservationSet.from_npz(Path("results/fleet") / name / "observations.npz")
    orbit = sc.study_orbit()
    sun = sc.sun_eci()

    periods, power = brightness_periodogram(obs, period_range_s=(20.0, 8000.0),
                                            n_periods=3000)
    p_ls = best_period(periods, power)
    band = periods < 2000.0
    p_band = best_period(periods[band], power[band])
    spin_candidates = sorted({round(p, 3) for p in
                              (p_ls, 2 * p_ls, p_band, 2 * p_band)})

    t0 = time.time()
    results = match_library(obs, orbit, sun, spin_candidates)
    dt = time.time() - t0

    best = results[0]
    row = dict(
        scenario=name, true_model=s.sat, true_mode=s.mode,
        identified_model=best.model, identified_hypothesis=best.hypothesis,
        identified_arrays_tracking=bool(best.arrays_tracking),
        correct_model=bool(best.model == s.sat),
        match_seconds=round(dt, 1),
        best_cost_per_model={m: r.cost for m, r in best_per_model(results).items()},
        top5=[dict(label=r.label, cost=float(r.cost)) for r in results[:5]],
    )
    if best.spin_params is not None:
        row["identified_spin"] = list(best.spin_params)
    (outdir / f"{name}.json").write_text(json.dumps(row, indent=2))
    print(f"{name}: identified as [{best.label}] "
          f"({'CORRECT' if row['correct_model'] else 'WRONG — truth ' + s.sat}) "
          f"in {dt:.0f}s")
    return row


def chart(rows: list[dict], charts_dir: Path) -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.labelcolor": INK_2, "axes.edgecolor": BASELINE,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    })
    models = [n for n in LIBRARY if n != "rocket_body"]
    cost = np.array([[r["best_cost_per_model"].get(m, np.nan) for m in models]
                     for r in rows])
    rel = cost / np.nanmin(cost, axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 2.4))
    im = ax.imshow(np.log10(rel), cmap=SEQ.reversed(), aspect="auto",
                   vmin=0, vmax=2)
    for i, r in enumerate(rows):
        j_id = models.index(r["identified_model"])
        ok = r["correct_model"]
        ax.scatter([j_id], [i], marker="o", s=110, facecolors="none",
                   edgecolors=S3_AQUA if ok else RED, linewidths=2.2)
        j_true = models.index(r["true_model"])
        if j_true != j_id:
            ax.scatter([j_true], [i], marker="x", s=70, color=RED, linewidths=1.8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['scenario']}\n→ {r['identified_hypothesis']}, "
                        f"arrays {'tracking' if r['identified_arrays_tracking'] else 'frozen'}"
                        for r in rows], fontsize=8)
    n_ok = sum(r["correct_model"] for r in rows)
    ax.set_title(f"Library model identification — {n_ok}/{len(rows)} correct "
                 "(○ identified; ✕ truth when missed)", pad=12)
    cb = fig.colorbar(im, shrink=0.8, pad=0.02)
    cb.set_label("log10 best cost per model / row best", color=INK_2)
    plt.setp(cb.ax.get_yticklabels(), color=MUTED)
    cb.outline.set_edgecolor(BASELINE)
    fig.savefig(charts_dir / "08_model_match.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", charts_dir / "08_model_match.png")


def main(*names: str) -> None:
    outdir = Path("results/model_match")
    outdir.mkdir(parents=True, exist_ok=True)
    todo = list(names) if names else DEFAULT_SCENARIOS
    rows = [run_one(n, outdir) for n in todo]
    # include any previously computed scenarios in the chart
    all_rows = {r["scenario"]: r for r in rows}
    for p in outdir.glob("*.json"):
        r = json.loads(p.read_text())
        all_rows.setdefault(r["scenario"], r)
    chart(sorted(all_rows.values(), key=lambda r: r["scenario"]),
          Path("results/charts"))


if __name__ == "__main__":
    main(*sys.argv[1:])
