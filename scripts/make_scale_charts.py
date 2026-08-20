"""Charts 15-16 — catalog-scale identification results (209-model library)."""

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


def main() -> None:
    style()
    out = Path("results/charts")
    out.mkdir(parents=True, exist_ok=True)
    s = json.load(open("results/library_scale/summary.json"))
    ok = [r for r in s["rows"] if r["status"] == "ok"]

    # ---- chart 15: funnel + rank outcomes --------------------------------
    fig = plt.figure(figsize=(13.5, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.25])

    ax = fig.add_subplot(gs[0, 0])
    stages = ["catalog", "feature\ngate", "shortlist\n(k=16)", "top-1"]
    vals = [s["library_size"], 100, s["shortlist_k"], 1]
    ax.bar(stages, vals, color=[MUTED, S1_BLUE, S3_AQUA, S2_ORANGE])
    for i, v in enumerate(vals):
        ax.annotate(str(v), (i, v), ha="center", va="bottom", fontsize=10,
                    color=INK_2)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 400)
    ax.set_ylabel("candidate models")
    ax.set_title("The identification funnel\n(cost per stage: ms → ~1 s → minutes)",
                 fontsize=10.5)

    ax = fig.add_subplot(gs[0, 1])
    ranks = [r["true_rank"] if r["in_shortlist"] and r["true_rank"] else 99
             for r in ok]
    bins = [1, 2, 3, 6, 17, 100]
    labels = ["rank 1", "rank 2", "rank 3-5", "rank 6-16", "cut at\nshortlist"]
    counts = [sum(1 for x in ranks if bins[i] <= x < bins[i + 1])
              for i in range(len(bins) - 1)]
    colors = [S3_AQUA, S1_BLUE, S5_YELLOW, S2_ORANGE, S4_MAGENTA]
    ax.bar(labels, counts, color=colors)
    for i, v in enumerate(counts):
        ax.annotate(str(v), (i, v), ha="center", va="bottom", color=INK_2)
    ax.set_ylabel("targets")
    ax.set_title(
        f"True-model outcome, {len(ok)} targets\n"
        f"top-1 {100 * s['top1_accuracy']:.0f}% | family "
        f"{100 * s['family_accuracy']:.0f}% | shortlist recall "
        f"{100 * s['shortlist_recall']:.0f}%", fontsize=10.5)

    ax = fig.add_subplot(gs[0, 2])
    hit = [r for r in ok if r["true_rank"] == 1]
    near = [r for r in ok if r["true_rank"] and r["true_rank"] > 1]
    cut = [r for r in ok if not r["in_shortlist"]]
    ax.scatter([r["top1_cost"] for r in hit],
               [r["margin"] or 1.0 for r in hit], s=42, color=S3_AQUA,
               label="correct top-1")
    ax.scatter([r["top1_cost"] for r in near],
               [r["margin"] for r in near], s=46, color=S5_YELLOW,
               marker="s", label="wrong top-1, truth ranked")
    ax.scatter([r["top1_cost"] for r in cut],
               [28.0] * len(cut), s=46, color=S4_MAGENTA,
               marker="D", label="truth cut at shortlist (plotted at top)")
    ax.axhline(1.0, color=BASELINE, lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("best (top-1) cost — fit quality")
    ax.set_ylabel("true-model cost / top-1 cost")
    ax.set_title("Self-assessment: margins separate\nclean IDs from trouble",
                 fontsize=10.5)
    ax.legend(fontsize=9)

    fig.suptitle(
        f"Catalog-scale identification — {s['library_size']}-model library, "
        "3 h arcs, mixed families / countries / modes",
        fontweight="bold", fontsize=12.5)
    p = out / "15_library_scale.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)

    # ---- chart 16: per-target table rendered as a matrix -----------------
    fig, ax = plt.subplots(figsize=(13.5, 0.34 * len(ok) + 1.8))
    ax.axis("off")
    cols = ["target", "family", "mode", "arrays", "rows", "short", "rank",
            "identified as", "margin"]
    cells = []
    cell_colors = []
    for r in sorted(ok, key=lambda x: (x["true_rank"] or 99)):
        rank = r["true_rank"] if r["in_shortlist"] else "cut"
        good = rank == 1
        cells.append([
            r["model"], r["family"], r["mode"], r["array_mode"],
            str(r["n_rows"]), "Y" if r["in_shortlist"] else "N",
            str(rank), r["top1"],
            f"{r['margin']:.2f}" if r["margin"] else "—"])
        base = "#1f2a1f" if good else ("#2a1f1f" if rank in (99, "cut") or
                                       (isinstance(rank, int) and rank > 3)
                                       else "#2a281f")
        cell_colors.append([base] * len(cols))
    tab = ax.table(cellText=cells, colLabels=cols, loc="center",
                   cellColours=cell_colors,
                   colColours=[BASELINE] * len(cols))
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.2)
    tab.scale(1.0, 1.25)
    for (i, j), c in tab.get_celld().items():
        c.set_edgecolor(GRID)
        c.set_text_props(color=INK_2 if i else INK)
    undet = [r for r in s["rows"] if r["status"] == "undetectable"]
    note = (f"   +{len(undet)} sampled targets undetectable at study "
            "geometry (too faint for the mag-7.5 trackers): "
            + ", ".join(r["model"] for r in undet)) if undet else ""
    ax.set_title("Per-target identification outcomes — 209-model library"
                 + note, fontsize=10.5, color=INK, pad=14)
    p = out / "16_library_scale_table.png"
    fig.savefig(p, dpi=160, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
