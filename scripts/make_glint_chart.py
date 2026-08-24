"""Chart 14 — glint detector + Wahba waypoint validation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#1a1a19"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
BASELINE = "#383835"
S1_BLUE = "#3987e5"
S2_ORANGE = "#d95926"
S3_AQUA = "#1baf7a"
S4_MAGENTA = "#d55181"
S5_YELLOW = "#c98500"

CASES = [
    ("katalyst_link__tumble", "Katalyst LINK — principal-axis tumble\n(good hypothesis)"),
    ("starlink_v15__tumble", "Starlink v1.5 — tumble\n(hypothesis locked a symmetry twin)"),
    ("katalyst_link__multiaxis_tumble", "Katalyst LINK — torque-free tumble\n(coarse principal-axis hypothesis)"),
]


def style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.labelcolor": INK_2, "axes.edgecolor": BASELINE,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.labelcolor": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def phase_excess(d):
    """Recompute the phase-conditioned brightness excess for every row."""
    t, mag_n, cens = d["t_s"], d["mag_n"], d["censored"].astype(bool)
    period = float(d["period_s"])
    pb = ((t / period) % 1.0 * 32).astype(int)
    tb = ((t - t.min()) / 7200.0).astype(int)
    key = pb * (tb.max() + 1) + tb
    exc = np.full(len(t), np.nan)
    for k in np.unique(key):
        s = (key == k) & ~cens
        if s.sum() >= 5:
            exc[key == k] = np.median(mag_n[s]) - mag_n[key == k]
    return exc


def main() -> None:
    style()
    out = Path("results/charts")
    out.mkdir(parents=True, exist_ok=True)
    summary = {r["scenario"]: r for r in
               json.load(open("results/glints/summary.json"))}

    fig = plt.figure(figsize=(13.5, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25])

    # --- top-left: excess separation histogram (katalyst tumble) ----------
    ax = fig.add_subplot(gs[0, 0])
    d = np.load(Path("results/glints") / "katalyst_link__tumble.npz")
    exc = phase_excess(d)
    truth = d["truth_mask"].astype(bool)
    ok = ~np.isnan(exc)
    bins = np.linspace(-2, 6, 60)
    ax.hist(exc[ok & ~truth], bins=bins, color=S1_BLUE, alpha=0.75, log=True,
            label="non-glint rows")
    ax.hist(exc[ok & truth], bins=bins, color=S2_ORANGE, alpha=0.9, log=True,
            label="true glint rows (sim label)")
    ax.axvline(1.5, color=S5_YELLOW, lw=1.6, ls="--", label="detector floor (1.5 mag)")
    ax.set_xlabel("brightness above phase-conditioned fleet median (mag)")
    ax.set_ylabel("rows (log)")
    ax.set_title("Detector statistic — Katalyst LINK tumble\n"
                 "phase-conditioning cancels the spin modulation", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")

    # --- top-middle: detector precision/recall ----------------------------
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(CASES))
    prec = [summary[n]["detector_raw"]["precision"] for n, _ in CASES]
    rec = [summary[n]["detector_raw"]["recall"] for n, _ in CASES]
    ax.bar(x - 0.18, prec, width=0.34, color=S1_BLUE, label="precision")
    ax.bar(x + 0.18, rec, width=0.34, color=S3_AQUA, label="recall")
    for xi, (n, _) in zip(x, CASES):
        ax.annotate(f"{summary[n]['n_pairs']} pairs\npass gate",
                    (xi, max(prec[xi], rec[xi]) + 0.04), ha="center",
                    fontsize=8.5, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(["LINK\ntumble", "v1.5\ntumble", "LINK\ntorque-free"],
                       fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("fraction")
    ax.set_title("Detector performance (calibrated rows)\n"
                 "geometric gate lifts pair purity to ~100%", fontsize=10.5)
    ax.legend(fontsize=9)

    # --- top-right: pair yield, realistic vs oracle gate -------------------
    ax = fig.add_subplot(gs[0, 2])
    real = [summary[n]["n_pairs"] for n, _ in CASES]
    orac = [summary[n]["oracle"]["n_pairs"] for n, _ in CASES]
    ax.bar(x - 0.18, real, width=0.34, color=S1_BLUE, label="pipeline hypothesis gate")
    ax.bar(x + 0.18, orac, width=0.34, color=S4_MAGENTA, label="oracle (truth) gate")
    ax.set_xticks(x)
    ax.set_xticklabels(["LINK\ntumble", "v1.5\ntumble", "LINK\ntorque-free"],
                       fontsize=9)
    ax.set_ylabel("Wahba pairs")
    ax.set_title("Pair yield vs gating hypothesis —\n"
                 "the torque-free gap is the iterate-refine headroom",
                 fontsize=10.5)
    ax.legend(fontsize=9)

    # --- bottom row: waypoint attitude error timelines ---------------------
    for j, (name, title) in enumerate(CASES):
        ax = fig.add_subplot(gs[1, j])
        d = np.load(Path("results/glints") / f"{name}.npz")
        h = d["hyp_curve_err"]
        ax.plot(d["hyp_curve_t"] / 3600, h, color=MUTED, lw=1.4,
                label="light-curve fit hypothesis")
        if len(d["owp_t"]):
            ax.scatter(d["owp_t"] / 3600, d["owp_err"], s=42, marker="o",
                       facecolors="none", edgecolors=S4_MAGENTA, linewidths=1.6,
                       label="waypoints, oracle gate")
        if len(d["wp_t"]):
            ax.scatter(d["wp_t"] / 3600, d["wp_err"], s=34, color=S1_BLUE,
                       zorder=5, label="waypoints, pipeline gate")
        ax.set_yscale("log")
        ax.set_ylim(0.03, 200)
        ax.set_xlabel("time (h)")
        if j == 0:
            ax.set_ylabel("attitude error vs truth (°)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "Glint detection → Wahba's problem: absolute attitude waypoints from specular events\n"
        "each glint pairs a body-frame facet normal with the inertial phase-angle bisector; "
        "Davenport's q-method solves windowed pairs into attitude fixes",
        fontweight="bold", fontsize=12.5)
    path = out / "14_glint_wahba.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
