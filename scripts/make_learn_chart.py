"""Chart 19: learned spin-state proposer vs grid search."""

from __future__ import annotations

import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photometry.charts import *  # noqa: F401,F403  (shared palette + style)


def main() -> None:
    style()
    s = json.load(open("results/learn/summary.json"))
    hist = json.load(open("results/learn/train_history.json"))["history"]
    synth, fleet = s["synthetic"], s["fleet"]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), constrained_layout=True)

    # (a) training curve
    ax = axes[0, 0]
    st = [h["step"] for h in hist]
    for key, color in [("pole", S1_BLUE), ("period", S2_ORANGE), ("axis", S3_AQUA)]:
        ax.plot(st, [h[key] for h in hist], color=color, lw=1.8, label=key)
    ax.set_yscale("log")
    ax.set_xlabel("training step"); ax.set_ylabel("loss component")
    ax.set_title("Training (on-the-fly re-rendered geometry pool)", fontsize=11)
    ax.legend(fontsize=9)

    # (b) raw net accuracy on held-out shapes: pole error vs period error
    ax = axes[0, 1]
    pe = np.array([r["net_raw"]["pole_err_deg"] for r in synth])
    pf = np.array([r["net_raw"]["period_err_frac"] for r in synth]) * 100
    ax.scatter(pf, pe, s=30, color=S1_BLUE, alpha=0.8)
    ax.axhline(15, color=BASELINE, lw=1); ax.axvline(5, color=BASELINE, lw=1)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("net period error (%, harmonic-tolerant)")
    ax.set_ylabel("net pole error (deg, axial)")
    ax.set_title(f"Raw proposal, {len(synth)} held-out synthetic windows\n"
                 f"(shapes never trained on) — median "
                 f"{np.median(pe):.1f}° / {np.median(pf):.1f}%", fontsize=11)

    # (c) after physics polish vs grid — pole error paired
    ax = axes[0, 2]
    a = np.array([r["net_polished"]["pole_err_deg"] for r in synth])
    b = np.array([r["grid"]["pole_err_deg"] for r in synth])
    ax.scatter(b, a, s=30, color=S3_AQUA, alpha=0.85)
    lim = max(a.max(), b.max(), 1.0) * 1.2
    ax.plot([0.01, lim], [0.01, lim], color=BASELINE, lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("grid search pole error (deg)")
    ax.set_ylabel("net-seeded polish pole error (deg)")
    ok_a, ok_b = np.mean(a < 2.0) * 100, np.mean(b < 2.0) * 100
    ax.set_title(f"Same cost function, two starting points\n"
                 f"< 2° success: net-seeded {ok_a:.0f}% vs grid {ok_b:.0f}%",
                 fontsize=11)

    # (d) time
    ax = axes[1, 0]
    t_net = [r["net_polished"]["t_total_s"] for r in synth]
    t_grid = [r["grid"]["t_s"] for r in synth]
    ax.boxplot([t_net, t_grid], labels=["net + polish", "grid search"],
               medianprops=dict(color=S5_YELLOW),
               boxprops=dict(color=INK_2), whiskerprops=dict(color=MUTED),
               capprops=dict(color=MUTED), flierprops=dict(markeredgecolor=MUTED))
    ax.set_yscale("log"); ax.set_ylabel("seconds per window")
    ax.set_title(f"Wall time — median {np.median(t_net):.1f} s vs "
                 f"{np.median(t_grid):.0f} s", fontsize=11)

    # (e) fleet scenarios bar
    ax = axes[1, 1]
    names = [r["scenario"].replace("__tumble", "") for r in fleet]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [r["net_polished"]["pole_err_deg"] for r in fleet], 0.4,
           color=S3_AQUA, label="net-seeded polish")
    ax.bar(x + 0.2, [r["grid"]["pole_err_deg"] for r in fleet], 0.4,
           color=S2_ORANGE, label="grid search")
    ax.set_yscale("log"); ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("pole error (deg)")
    ax.set_title("Fleet-study tumblers (full 24 h simulator, 2 h window)",
                 fontsize=11)
    ax.legend(fontsize=9)

    # (f) text: torque-free
    ax = axes[1, 2]
    ax.axis("off")
    tf = s.get("torque_free")
    lines = ["Torque-free seeding (multi-axis LINK)"]
    if tf:
        lines += [f"net period {tf['net_period']:.1f} s",
                  f"seed pole error {tf['seed_pole_err']:.1f}°",
                  f"fit cost {tf['cost']:.2f} (uniform seed {tf['cost_uniform_seed']:.2f})",
                  f"inertia est {np.round(tf['inertia_est'], 2).tolist()}",
                  f"attitude error over 1 h window:",
                  f"   median {tf['att_err_median_deg']:.1f}°, mean {tf['att_err_mean_deg']:.1f}°",
                  f"(documented multi-start baseline: 50.7° / 56.0°)",
                  f"time {tf['t_s']:.0f} s"]
    else:
        lines.append("(not run)")
    ax.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", fontsize=11,
            color=INK_2, family="monospace", transform=ax.transAxes)

    fig.suptitle("Learned spin-state proposer (set transformer, 0.48 M params, CPU) "
                 "— the net proposes, the forward model verifies and polishes",
                 fontweight="bold", fontsize=13)
    out = Path("results/charts/19_spin_net.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
