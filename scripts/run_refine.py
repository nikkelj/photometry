"""Matched-model refinement demo: residual EGI as a catalog-deviation detector.

Three cases:
  A. katalyst_link__tumble refined against its (correct) matched model —
     residual should be noise.
  B. starlink_v2mini_dtc__ops deliberately refined against the PLAIN
     v2 mini model — the missing DTC antenna panel should appear as
     localized positive residual area near the panel normals (body +/-z).
  C. same scenario refined against the correct DTC model — flat again.

Writes results/refine/<case>.json and results/charts/09_residual_egi.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from photometry import scenarios as sc
from photometry.attitude import FixedInertial, LvlhHold, PrincipalAxisSpin
from photometry.inversion.refine import refine_match
from photometry.measurements import ObservationSet
from photometry.shapes import LIBRARY

SURFACE = "#1a1a19"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
BASELINE = "#383835"
DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#3987e5", "#86b6ef", "#383835", "#e8a1a1", "#e66767"])

CASES = [
    dict(key="A_link_tumble_correct", scenario="katalyst_link__tumble",
         model="katalyst_link", title="LINK tumble vs correct model"),
    dict(key="B_dtc_ops_wrong_model", scenario="starlink_v2mini_dtc__ops",
         model="starlink_v2mini", title="DTC ops vs plain v2 mini\n(antenna missing from model)"),
    dict(key="C_dtc_ops_correct", scenario="starlink_v2mini_dtc__ops",
         model="starlink_v2mini_dtc", title="DTC ops vs correct model"),
]


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


def main() -> None:
    outdir = Path("results/refine")
    outdir.mkdir(parents=True, exist_ok=True)
    orbit = sc.study_orbit()
    sun = sc.sun_eci()

    results = []
    for case in CASES:
        obs = ObservationSet.from_npz(
            Path("results/fleet") / case["scenario"] / "observations.npz")
        match = json.loads(
            (Path("results/model_match") / f"{case['scenario']}.json").read_text())
        hyp, att, spin = attitude_from_match(match, orbit, sun)
        shape = LIBRARY[case["model"]]()
        arrays_tracking = bool(match["identified_arrays_tracking"]) and shape.articulated
        r = refine_match(obs, shape, hyp, arrays_tracking, att, spin)
        results.append((case, r))
        row = dict(
            case=case["key"], scenario=case["scenario"], model=case["model"],
            hypothesis=hyp, arrays_tracking=arrays_tracking,
            cost_coarse=r.cost_coarse, cost_refined=r.cost_refined,
            residual_rms_before=r.residual_rms_before,
            residual_rms_after=r.residual_rms_after,
            spin_refined=list(r.spin_params) if r.spin_params else None,
            peak_residual_albedo_area=float(
                r.residual_albedo_area[np.argmax(np.abs(r.residual_albedo_area))]),
        )
        (outdir / f"{case['key']}.json").write_text(json.dumps(row, indent=2))
        print(f"{case['key']}: cost {r.cost_coarse:.2f} -> {r.cost_refined:.2f}, "
              f"residual rms {r.residual_rms_before:.2f} -> {r.residual_rms_after:.2f}, "
              f"peak residual {row['peak_residual_albedo_area']:+.2f} m^2")

    # chart: signed residual EGI maps
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.labelcolor": INK_2, "axes.edgecolor": BASELINE,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    })
    vmax = max(np.abs(r.residual_albedo_area).max() for _, r in results)
    fig = plt.figure(figsize=(14, 4.6), constrained_layout=True)
    for i, (case, r) in enumerate(results):
        ax = fig.add_subplot(1, 3, i + 1, projection="mollweide")
        ax.grid(True, color="#2c2c2a", lw=0.5)
        lon = np.arctan2(r.residual_normals[:, 1], r.residual_normals[:, 0])
        lat = np.arcsin(np.clip(r.residual_normals[:, 2], -1, 1))
        scmap = ax.scatter(lon, lat, c=r.residual_albedo_area, cmap=DIVERGING,
                           norm=TwoSlopeNorm(0, -vmax, vmax), s=22, linewidths=0)
        ax.set_title(f"{case['title']}\nresid rms {r.residual_rms_before:.2f} → "
                     f"{r.residual_rms_after:.2f}", fontsize=10, pad=12)
        ax.tick_params(labelsize=6, colors=MUTED)
        if case["key"].startswith("B"):
            # expected deviation: the DTC panel's +/-z normals (body frame)
            ax.scatter([0, 0], [np.pi / 2 - 1e-3, -np.pi / 2 + 1e-3],
                       facecolors="none", edgecolors=INK, s=200, linewidths=1.4)
            ax.text(0.02, 0.03, "rings: DTC panel normals (±z)", color=INK_2,
                    fontsize=8, transform=ax.transAxes)
    cb = fig.colorbar(scmap, ax=fig.axes, shrink=0.75, pad=0.015)
    cb.set_label("signed residual ρ·A vs matched model (m²)", color=INK_2)
    plt.setp(cb.ax.get_yticklabels(), color=MUTED)
    cb.outline.set_edgecolor(BASELINE)
    fig.suptitle("Residual EGI after matched-model refinement — catalog deviations "
                 "appear as localized signed area", fontweight="bold")
    out = Path("results/charts/09_residual_egi.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
