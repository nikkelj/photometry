"""Chart 17: observability trade — fit quality vs observing-constellation size."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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


def by_size(rows, key, agg=np.median):
    """(sizes, aggregated value, per-seed values) for rows carrying `key`."""
    sizes = sorted({r["n_sats"] for r in rows})
    xs, ys, scatter = [], [], []
    for n in sizes:
        vals = [r[key] for r in rows
                if r["n_sats"] == n and r.get(key) is not None]
        if vals:
            xs.append(n)
            ys.append(agg(vals))
            scatter.extend((n, v) for v in vals)
    return np.array(xs), np.array(ys), scatter


def main() -> None:
    style()
    s = json.load(open("results/observability/summary.json"))
    rows = s["rows"]
    out = Path("results/charts")
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6), constrained_layout=True)

    # (a) data volume
    ax = axes[0, 0]
    for key, color, label in [("n_rows", S1_BLUE, "tumbling servicer"),
                              ("n_rows_stable", S5_YELLOW, "stable flat-sat (ops)")]:
        xs, ys, sc = by_size(rows, key)
        ok = ys > 0
        ax.plot(xs[ok], ys[ok], "o-", color=color, lw=2, ms=5, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("observing satellites"); ax.set_ylabel("detections per day")
    ax.set_title("Data volume scales linearly —\nbut information does not",
                 fontsize=11)
    ax.legend(fontsize=9)

    # (b) directional coverage
    ax = axes[0, 1]
    for key, color, label in [("obs", S1_BLUE, "viewing directions"),
                              ("pab", S2_ORANGE, "phase-angle bisectors")]:
        xs, ys, sc = by_size(rows, key)
        ax.plot(xs, 100 * ys, "o-", color=color, lw=2, ms=5, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("observing satellites")
    ax.set_ylabel("% of direction cells sampled (of 300)")
    ax.set_title("Directional coverage: the tomography\nresource (24 h arc)",
                 fontsize=11)
    ax.legend(fontsize=9)

    # (c) period recovery
    ax = axes[0, 2]
    sizes = sorted({r["n_sats"] for r in rows})
    xs, fr = [], []
    for n in sizes:
        vals = [r.get("period_ok") for r in rows if r["n_sats"] == n
                and "period_ok" in r]
        if vals:
            xs.append(n); fr.append(np.mean(vals))
    ax.plot(xs, np.array(fr) * 100, "o-", color=S3_AQUA, lw=2.2, ms=6)
    ax.set_xscale("log"); ax.set_ylim(-5, 105)
    ax.set_xlabel("observing satellites")
    ax.set_ylabel("% of seeds recovering the period")
    ax.set_title("Spin period (127.4 s): all-or-nothing\nonce sampling density arrives",
                 fontsize=11)

    # (d) pole error
    ax = axes[1, 0]
    xs, ys, sc = by_size(rows, "pole_err_deg")
    if sc:
        px, py = zip(*sc)
        ax.scatter(px, py, s=26, color=S1_BLUE, alpha=0.5, label="per seed")
    ax.plot(xs, ys, "o-", color=S1_BLUE, lw=2.2, ms=6, label="median")
    ax.axhline(5, color=BASELINE, lw=1)
    ax.text(1.2, 5.6, "5° (useful attitude)", color=MUTED, fontsize=8.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("observing satellites"); ax.set_ylabel("spin-pole error (deg)")
    ax.set_title("Attitude: pole error collapses at the\nperiod threshold", fontsize=11)
    ax.text(0.03, 0.05,
            "left-of-knee lows are lucky seeds riding a wrong period —\n"
            "not usable attitude (period unrecovered below N≈1000)",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.legend(fontsize=9)

    # (e) EGI quality
    ax = axes[1, 1]
    xs, ys, sc = by_size(rows, "egi_within_15deg")
    if sc:
        px, py = zip(*sc)
        ax.scatter(px, np.array(py) * 100, s=26, color=S2_ORANGE, alpha=0.5)
    ax.plot(xs, ys * 100, "o-", color=S2_ORANGE, lw=2.2, ms=6)
    ax.set_xscale("log"); ax.set_ylim(0, 105)
    ax.set_xlabel("observing satellites")
    ax.set_ylabel("% recovered area within 15° of truth")
    ax.set_title("Shape (EGI): rides the attitude —\nthen saturates", fontsize=11)

    # (f) Wahba waypoints
    ax = axes[1, 2]
    xs, ys, _ = by_size(rows, "n_wahba_pairs")
    ax.plot(xs, np.maximum(ys, 0.5), "o-", color=S4_MAGENTA, lw=2, ms=5,
            label="Wahba pairs (median)")
    xs2, ys2, _ = by_size(rows, "n_waypoints")
    ax.plot(xs2, np.maximum(ys2, 0.5), "s--", color=S5_YELLOW, lw=1.8, ms=5,
            label="attitude waypoints")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("observing satellites")
    ax.set_ylabel("count per day")
    ax.set_title("Glint geometry: vector-pair yield\n(needs the fleet)", fontsize=11)
    ax.legend(fontsize=9)

    fig.suptitle(
        "Observability trade — the same inversion chain vs observing-constellation "
        "size (observer-ID subsamples of the full-fleet run;\n"
        "katalyst_link tumble, 24 h arc; 3 seeds per size)",
        fontweight="bold", fontsize=13)
    path = out / "17_observability_trade.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
