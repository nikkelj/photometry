"""Torque-free (non-principal-axis) attitude refinement demo.

Fits full rigid-body dynamics to the katalyst_link__multiaxis_tumble
scenario, seeded from the Tier-2 uniform-spin solution, and compares
bus-frame attitude error vs nav truth for both models — over the fit
window and a forward-prediction span. Saves the fit for the movie
renderer and results/charts/10_torquefree.png.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from photometry import scenarios as sc
from photometry.attitude import PrincipalAxisSpin
from photometry.inversion.torquefree import fit_torque_free
from photometry.measurements import ObservationSet

SURFACE = "#1a1a19"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
BASELINE = "#383835"
S1_BLUE = "#3987e5"
S2_ORANGE = "#d95926"

SCENARIO = "katalyst_link__multiaxis_tumble"


def att_err_deg(r_a, r_b):
    c = (np.trace(r_a @ r_b.T) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def main() -> None:
    fleet = Path("results/fleet") / SCENARIO
    obs = ObservationSet.from_npz(fleet / "observations.npz")
    result = json.loads((fleet / "result.json").read_text())
    with np.load(fleet / "inversion.npz") as z:
        ra, dec, per, ph, ax_, ay, az = [float(v) for v in z["best_spin_params"]]
    seed_spin = PrincipalAxisSpin(ra, dec, per, ph, body_axis=(ax_, ay, az))

    s = sc.by_name(SCENARIO)
    orbit, sun = sc.study_orbit(), sc.sun_eci()
    att_true, _ = sc.make_attitude(s.mode, orbit, sun, False,
                                   duration_s=s.duration_s)

    # data-driven |omega| candidates: distinct light-curve peaks + doubles
    # (flat facets modulate at twice the rotation frequency)
    from photometry.inversion.periodogram import brightness_periodogram
    periods, power = brightness_periodogram(obs, period_range_s=(20.0, 600.0))
    peaks = []
    for i in np.argsort(power)[::-1]:
        if all(abs(periods[i] - p) > 3 for p in peaks):
            peaks.append(float(periods[i]))
        if len(peaks) >= 3:
            break
    seed_periods = sorted({round(p, 2) for p in peaks + [2 * p for p in peaks]})
    print(f"Fitting torque-free dynamics (seed periods {seed_periods})...")
    t0 = time.time()
    fit = fit_torque_free(obs, s.shape(), seed_spin, seed_periods=seed_periods)
    print(f"  {fit.n_evals} evals in {time.time()-t0:.0f}s | cost: uniform "
          f"{fit.cost_uniform_seed:.2f} -> torque-free {fit.cost:.2f}")
    print(f"  inertia ratios {np.round(fit.inertia, 3)} (truth {sc.MULTIAXIS['inertia']})")
    print(f"  omega0 {np.round(fit.omega0_body, 4)} rad/s "
          f"(truth {sc.MULTIAXIS['omega0_body']})")

    # attitude error time series: fit window + forward prediction
    horizon = 4 * 3600.0
    att_fit = fit.attitude(t_max=horizon + 60)
    ts = fit.t_ref + np.arange(0.0, horizon, 20.0)
    err_u, err_tf = [], []
    for t in ts:
        r_t = att_true.body_to_eci_matrix(float(t))
        err_u.append(att_err_deg(seed_spin.body_to_eci_matrix(float(t)), r_t))
        err_tf.append(att_err_deg(att_fit.body_to_eci_matrix(float(t)), r_t))
    err_u, err_tf = np.array(err_u), np.array(err_tf)
    t_h = (ts - fit.t_ref) / 3600

    win_end = 3600.0 / 3600
    in_win = t_h <= win_end
    summary = dict(
        scenario=SCENARIO,
        cost_uniform=fit.cost_uniform_seed, cost_torquefree=fit.cost,
        inertia_est=list(fit.inertia),
        inertia_true=list(sc.MULTIAXIS["inertia"]),
        omega0_est=list(fit.omega0_body),
        omega0_true=list(sc.MULTIAXIS["omega0_body"]),
        t_ref=fit.t_ref,
        r0=fit.r0.tolist(),
        err_window_mean_uniform=float(err_u[in_win].mean()),
        err_window_mean_torquefree=float(err_tf[in_win].mean()),
        err_predict_mean_uniform=float(err_u[~in_win].mean()),
        err_predict_mean_torquefree=float(err_tf[~in_win].mean()),
    )
    (fleet / "torquefree.json").write_text(json.dumps(summary, indent=2))
    print(f"  window attitude err: uniform {summary['err_window_mean_uniform']:.1f} deg"
          f" -> torque-free {summary['err_window_mean_torquefree']:.1f} deg")
    print(f"  4h-prediction err:  uniform {summary['err_predict_mean_uniform']:.1f} deg"
          f" -> torque-free {summary['err_predict_mean_torquefree']:.1f} deg")

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.labelcolor": INK_2, "axes.edgecolor": BASELINE,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.labelcolor": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(t_h, err_u, color=S1_BLUE, lw=1.4, label="uniform-spin fit (Tier-2)")
    ax.plot(t_h, err_tf, color=S2_ORANGE, lw=1.4,
            label="torque-free fit (Tier-3)")
    ax.axvspan(0, win_end, color="#2c2c2a", alpha=0.6)
    ax.text(win_end / 2, ax.get_ylim()[1] * 0.94, "fit window", color=MUTED,
            ha="center", fontsize=9)
    ax.set_xlabel("hours past fit epoch")
    ax.set_ylabel("bus-frame attitude error vs nav truth (°)")
    ax.set_title("Non-principal-axis tumble: uniform spin cannot follow the "
                 "nutation; torque-free dynamics can", pad=10)
    ax.legend(loc="upper right")
    out = Path("results/charts/10_torquefree.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
