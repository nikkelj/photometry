"""Fleet-study summary charts (dark theme) from results/fleet/*/result.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from photometry.shapes import GIMBAL_FIXED, LIBRARY

SURFACE = "#1a1a19"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
BASELINE = "#383835"
S1_BLUE = "#3987e5"
S2_ORANGE = "#d95926"
S3_AQUA = "#199e70"
SEQ = LinearSegmentedColormap.from_list("seq", [
    "#0d366b", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#cde2fb"])

HYPS = ["lvlh_ops", "lvlh_low_drag", "sun_point", "spin_fit", "inertial_fit"]


def style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.labelcolor": INK_2, "axes.edgecolor": BASELINE,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": False, "font.size": 10, "axes.titlesize": 12,
        "legend.frameon": False, "legend.labelcolor": INK_2,
    })


def load_rows(fleet_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(fleet_dir.iterdir()):
        f = p / "result.json"
        if f.exists():
            rows.append(json.loads(f.read_text()))
    return rows


def chart_modes(rows: list[dict], out: Path) -> None:
    names = [r["scenario"] for r in rows]
    cost = np.array([[r["hypothesis_costs"].get(h, np.nan) for h in HYPS]
                     for r in rows])
    rel = cost / np.nanmin(cost, axis=1, keepdims=True)  # 1 = best fit
    fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(rows) + 2.2))
    im = ax.imshow(np.log10(rel), cmap=SEQ.reversed(), aspect="auto",
                   vmin=0, vmax=2)
    for i, r in enumerate(rows):
        j_best = HYPS.index(r["mode_best"])
        ok = r["mode_correct"]
        ax.scatter([j_best], [i], marker="o", s=90,
                   facecolors="none",
                   edgecolors=S3_AQUA if ok else "#e66767", linewidths=2)
        j_exp = HYPS.index(r["mode_expected"])
        if j_exp != j_best and not ok:
            ax.scatter([j_exp], [i], marker="x", s=60, color="#e66767",
                       linewidths=1.6)
    ax.set_xticks(range(len(HYPS)))
    ax.set_xticklabels(HYPS, fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8.5)
    n_ok = sum(r["mode_correct"] for r in rows)
    ax.set_title(f"Attitude-mode classification — {n_ok}/{len(rows)} correct "
                 "(○ selected; ✕ expected when missed)", pad=12)
    cb = fig.colorbar(im, shrink=0.8, pad=0.02)
    cb.set_label("log10 cost / best cost (0 = best fit)", color=INK_2)
    plt.setp(cb.ax.get_yticklabels(), color=MUTED)
    cb.outline.set_edgecolor(BASELINE)
    fig.savefig(out / "06_fleet_modes.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / "06_fleet_modes.png")


def chart_egi(rows: list[dict], out: Path) -> None:
    names = [r["scenario"] for r in rows]
    rec = np.array([r["egi_total_albedo_area_m2"] for r in rows])
    full = np.array([r["true_diffuse_albedo_area_m2"] for r in rows])
    static = []
    for r in rows:
        shape = LIBRARY[r["sat"]]()
        fixed = shape.gimbal_mode == GIMBAL_FIXED
        for i in range(shape.n_facets):
            if shape.mirror_of[i] >= 0 and not fixed[shape.mirror_of[i]]:
                fixed[i] = False
        static.append(float(shape.diffuse_albedo_area()[fixed].sum()))
    static = np.array(static)

    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(rows) + 2.2))
    ax.barh(y, rec, height=0.55, color=S2_ORANGE, label="recovered diffuse ρ·A")
    ax.scatter(full, y, marker="|", s=260, color=S1_BLUE, linewidths=2.4,
               label="truth ρ·A (all facets)")
    ax.scatter(static, y, marker="|", s=260, color=S3_AQUA, linewidths=2.4,
               label="truth ρ·A (body-fixed facets only)")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("diffuse albedo·area (m², log scale)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_title("EGI recovery vs truth — sun-tracking panels are not "
                 "body-fixed, so ops modes recover the bus", pad=12)
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(out / "07_fleet_egi.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / "07_fleet_egi.png")


def main(fleet_dir: str = "results/fleet", charts_dir: str = "results/charts") -> None:
    style()
    rows = load_rows(Path(fleet_dir))
    if not rows:
        raise SystemExit("no fleet results found")
    out = Path(charts_dir)
    out.mkdir(parents=True, exist_ok=True)
    chart_modes(rows, out)
    chart_egi(rows, out)
    (Path(fleet_dir) / "fleet_summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
