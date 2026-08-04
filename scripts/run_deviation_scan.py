"""Fleet-wide catalog-deviation alert scan.

For every scenario with a Tier-2 identification: refine against the
*identified* model and run the deviation assessment. Also includes one
seeded-deviation control — the DTC scenario refined against the plain
v2 mini model (a genuinely missing antenna) — to exercise the detector.

Ground truth for scoring: an alert is CORRECT when the identified model
differs from the true one (misidentification) or the model was seeded
wrong; a non-alert is CORRECT when the identified model is the truth.

Writes results/deviation/<name>.json and results/charts/11_deviation_scan.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photometry import scenarios as sc
from photometry.attitude import FixedInertial, LvlhHold, PrincipalAxisSpin
from photometry.inversion.deviation import assess_deviation
from photometry.inversion.refine import refine_match
from photometry.measurements import ObservationSet
from photometry.shapes import LIBRARY

SURFACE = "#1a1a19"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
BASELINE = "#383835"
S1_BLUE = "#3987e5"
S3_AQUA = "#199e70"
RED = "#e66767"


def attitude_from_match(match: dict, orbit, sun):
    hyp = match["identified_hypothesis"]
    if hyp in ("spin_fit", "inertial_fit"):
        p = match["identified_spin"]
        return hyp, PrincipalAxisSpin(p[0], p[1], p[2], p[3],
                                      body_axis=tuple(p[4:7])), tuple(p)
    att = {"lvlh_ops": LvlhHold(orbit),
           "lvlh_low_drag": LvlhHold(orbit, roll_deg=90.0),
           "sun_point": FixedInertial.z_toward(sun)}[hyp]
    return hyp, att, None


def scan_one(scenario: str, model: str, match: dict, orbit, sun,
             seeded_wrong: bool) -> dict:
    obs = ObservationSet.from_npz(
        Path("results/fleet") / scenario / "observations.npz")
    hyp, att, spin = attitude_from_match(match, orbit, sun)
    shape = LIBRARY[model]()
    arrays_tracking = bool(match["identified_arrays_tracking"]) and shape.articulated
    r = refine_match(obs, shape, hyp, arrays_tracking, att, spin)
    a = assess_deviation(r)
    truth_model = sc.by_name(scenario).sat
    deviation_truth = seeded_wrong or (model != truth_model)
    return dict(
        scenario=scenario, refined_model=model, truth_model=truth_model,
        seeded_wrong=seeded_wrong, hypothesis=hyp,
        alert=a.alert, alert_correct=bool(a.alert == deviation_truth),
        deviation_truth=bool(deviation_truth),
        rms_ratio=a.rms_ratio, fit_quality=a.fit_quality,
        peak_albedo_area=a.peak_albedo_area,
        cluster_albedo_area=a.cluster_albedo_area,
        reason=a.reason,
    )


def main() -> None:
    outdir = Path("results/deviation")
    outdir.mkdir(parents=True, exist_ok=True)
    orbit, sun = sc.study_orbit(), sc.sun_eci()

    jobs = []
    for p in sorted(Path("results/model_match").glob("*.json")):
        match = json.loads(p.read_text())
        jobs.append((match["scenario"], match["identified_model"], match, False))
    # seeded-deviation control: refine DTC data against the plain v2 mini
    dtc = json.loads((Path("results/model_match")
                      / "starlink_v2mini_dtc__ops.json").read_text())
    jobs.append(("starlink_v2mini_dtc__ops", "starlink_v2mini", dtc, True))

    rows = []
    for scenario, model, match, seeded in jobs:
        row = scan_one(scenario, model, match, orbit, sun, seeded)
        rows.append(row)
        tag = "ALERT" if row["alert"] else "clear"
        ok = "ok" if row["alert_correct"] else "WRONG"
        label = scenario + (" [seeded wrong model]" if seeded else "")
        print(f"{label:48s} vs {model:20s} {tag:5s} ({ok}) "
              f"ratio {row['rms_ratio']:.2f} fitq {row['fit_quality']:.2f}")
        (outdir / f"{scenario}{'__seeded' if seeded else ''}.json").write_text(
            json.dumps(row, indent=2))

    n_ok = sum(r["alert_correct"] for r in rows)
    print(f"alert decisions correct: {n_ok}/{len(rows)}")

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.labelcolor": INK_2, "axes.edgecolor": BASELINE,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
        "legend.frameon": False, "legend.labelcolor": INK_2,
    })
    rows_s = sorted(rows, key=lambda r: r["rms_ratio"])
    y = np.arange(len(rows_s))
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(rows_s) + 2.2))
    colors = [RED if r["alert"] else S1_BLUE for r in rows_s]
    ax.barh(y, [r["rms_ratio"] for r in rows_s], height=0.6, color=colors)
    ax.axvline(1.15, color=INK_2, ls="--", lw=1.2)
    ax.text(1.155, len(rows_s) - 0.4, "structure threshold", color=INK_2,
            fontsize=8.5, rotation=0)
    labels = []
    for r in rows_s:
        lab = r["scenario"] + (" [seeded wrong]" if r["seeded_wrong"] else "")
        if r["fit_quality"] > 4.0:
            lab += "  (poor fit)"
        labels.append(lab)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    for yi, r in zip(y, rows_s):
        mark = "✓" if r["alert_correct"] else "✗"
        ax.text(max(r["rms_ratio"], 1.0) + 0.015, yi, mark,
                color=S3_AQUA if r["alert_correct"] else RED,
                va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("residual rms ratio (before / after residual-EGI absorption)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    n_alerts = sum(r["alert"] for r in rows_s)
    ax.set_title(f"Catalog-deviation alert scan — {n_ok}/{len(rows_s)} decisions "
                 f"correct, {n_alerts} alerts (red = alert; poor-fit criterion "
                 "also triggers)", pad=12)
    out = Path("results/charts/11_deviation_scan.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
