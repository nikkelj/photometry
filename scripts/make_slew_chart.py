"""Chart 18: maneuver-slew detectability — 90/180 deg pre-burn yaw-arounds."""

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
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.labelcolor": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def yaw_truth(t, s):
    t = np.asarray(t)
    t0, sl, h = s["t_start_s"], s["slew_s"], s["hold_s"]
    up = np.clip((t - t0) / sl, 0, 1)
    dn = np.clip((t - t0 - sl - h) / sl, 0, 1)
    sm = lambda x: x * x * (3 - 2 * x)  # noqa: E731
    return sm(up) - sm(dn)


def main() -> None:
    style()
    s = json.load(open("results/slew/summary.json"))
    rows = {r["case"]: r for r in s["rows"]}
    craft = ["starlink_v15", "starlink_v2mini", "ru_persona", "cn_yaogan_sar"]
    controls = {r["model"]: r for r in s["rows"] if r["yaw_deg"] is None}

    fig, axes = plt.subplots(4, 2, figsize=(15.0, 13.0), constrained_layout=True)
    for i, name in enumerate(craft):
        for j, yaw in enumerate([90, 180]):
            ax = axes[i, j]
            r = rows[f"{name}__yaw{yaw}"]
            tc = np.array(r["centers"]) / 60.0
            z = np.array(r["z"])
            # truth maneuver shading
            t0, sl, h = s["t_start_s"], s["slew_s"], s["hold_s"]
            ax.axvspan(t0 / 60, (t0 + 2 * sl + h) / 60, color="#26323a")
            ax.axvspan((t0 + sl) / 60, (t0 + sl + h) / 60, color="#2d3a44")
            ax.plot(tc, z, color=S1_BLUE, lw=1.6, label="nominal-hypothesis z")
            ax.axhline(s["z_thresh"], color=S4_MAGENTA, lw=1.0, ls="--")
            for (e0, e1) in r["events"]:
                ax.plot([e0 / 60, e1 / 60],
                        [ax.get_ylim()[1] * 0.02] * 2, color=S4_MAGENTA, lw=4)
            ax.set_ylabel("residual z", color=S1_BLUE)
            ax2 = ax.twinx()
            ax2.grid(False)
            ax2.spines["right"].set_visible(True)
            ax2.spines["right"].set_color(BASELINE)
            yc = np.array(r["yaw_centers"]) / 60.0
            ytr = np.array(r["yaw_track"])
            contr = np.array(r["yaw_contrast"])
            good = contr > np.median(contr) * 0.3
            ax2.plot(yc, yaw * yaw_truth(np.array(r["yaw_centers"]), s),
                     color=INK_2, lw=1.2, ls=":", label="truth yaw")
            ax2.scatter(yc[good], ytr[good], s=10, color=S5_YELLOW,
                        label="windowed yaw estimate")
            ax2.scatter(yc[~good], ytr[~good], s=8, color=MUTED, alpha=0.4)
            ax2.set_ylim(-200, 200)
            ax2.set_ylabel("yaw (deg)", color=S5_YELLOW)
            ax2.tick_params(axis="y", colors=MUTED)
            fit = r.get("fit")
            zc = controls.get(name)
            bits = [f"z_max {r['z_max']:.1f}"]
            if zc:
                bits.append(f"control z_max {zc['z_max']:.1f}")
            if fit:
                bits.append(f"fit: t0 err {fit['err_t_start_s']:+.0f} s, "
                            f"slew err {fit['err_slew_s']:+.0f} s, "
                            f"hold err {fit['err_hold_s']:+.0f} s, "
                            f"yaw err {fit['err_yaw_deg']:.1f}°")
            elif not r["events"]:
                bits.append("NOT DETECTED")
            ax.set_title(f"{name} — {yaw}° yaw-around   ({'; '.join(bits)})",
                         fontsize=10)
            if i == 3:
                ax.set_xlabel("time (min)")
            if i == 0 and j == 0:
                ax.legend(loc="upper left", fontsize=8)
                ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Maneuver-slew detectability — 6 min yaw-out, 15 min burn hold, 6 min "
        "return (shaded); windowed nominal-hypothesis residual (blue, robust z)\n"
        "and windowed constant-yaw estimate (yellow; grey = low yaw "
        "identifiability); detections in magenta",
        fontweight="bold", fontsize=13)
    out = Path("results/charts/18_slew_detectability.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
