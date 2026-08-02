"""Render the dark-themed chart set from saved pipeline results.

Usage: python scripts/make_charts.py [results_dir] [charts_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from photometry.measurements import ObservationSet

# Dark-mode chart tokens (validated reference palette)
SURFACE = "#1a1a19"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
BASELINE = "#383835"
S1_BLUE = "#3987e5"
S2_ORANGE = "#d95926"
S3_AQUA = "#199e70"

# single-hue sequential ramp (blue), dark-mode: low recedes to surface, high pops
SEQ_STEPS = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf",
             "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef",
             "#9ec5f4", "#b7d3f6", "#cde2fb"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue_dark", SEQ_STEPS)


def style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": BASELINE,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "figure.titlesize": 15,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
    })


def lonlat(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors -> (lon, lat) radians for mollweide axes, lon in [-pi, pi]."""
    lon = np.arctan2(u[..., 1], u[..., 0])
    lat = np.arcsin(np.clip(u[..., 2], -1, 1))
    return lon, lat


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def chart_summary_tiles(summary: dict, out: Path) -> None:
    tiles = [
        (f"{summary['n_detections']:,}", "photometric detections",
         f"{summary['arc_hours']:.1f} h arc · {summary['n_observers']} observers"),
        (f"{summary['pole_err_deg']:.2f}°", "spin-pole error",
         f"est ({summary['pole_est_radec_deg'][0]:.1f}°, {summary['pole_est_radec_deg'][1]:.1f}°) "
         f"vs truth ({summary['pole_true_radec_deg'][0]:.0f}°, {summary['pole_true_radec_deg'][1]:.0f}°)"),
        (f"{summary['period_err_ms']:.0f} ms", "spin-period error",
         f"est {summary['period_est_s']:.3f} s vs truth {summary['period_true_s']:.1f} s"),
        (f"{100 * summary['egi_capture_frac_15deg']:.0f}%", "EGI area near true normals",
         f"recovered ρA {summary['egi_total_albedo_area_m2']:.1f} m² "
         f"(truth {summary['egi_true_total_albedo_area_m2']:.1f} m²)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.6))
    for ax, (value, label, sub) in zip(axes, tiles):
        ax.set_axis_off()
        ax.text(0.02, 0.98, label.upper(), fontsize=9.5, color=MUTED,
                va="top", transform=ax.transAxes)
        ax.text(0.02, 0.62, value, fontsize=27, color=INK, fontweight="bold",
                va="center", transform=ax.transAxes)
        ax.text(0.02, 0.16, sub, fontsize=8.5, color=INK_2, va="center",
                transform=ax.transAxes, wrap=True)
    fig.suptitle("Constellation photometry inversion — initial results", x=0.02,
                 ha="left", fontweight="bold")
    save(fig, out / "00_summary_tiles.png")


def chart_lightcurve(obs: ObservationSet, truth: dict, out: Path) -> None:
    t_h = obs.t_s / 3600
    m_norm = obs.mag - 5 * np.log10(obs.range_km / 1000.0)
    phase = obs.phase_angle_deg()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6.6), height_ratios=[1.5, 1], constrained_layout=True
    )
    sc = ax1.scatter(t_h, m_norm, c=phase, cmap=SEQ_CMAP, s=5, linewidths=0)
    ax1.invert_yaxis()
    ax1.set_xlabel("time (hours)")
    ax1.set_ylabel("magnitude @ 1000 km")
    ax1.set_title("Aggregated light curve — every point is a different star tracker")
    cb = fig.colorbar(sc, ax=ax1, pad=0.01)
    cb.set_label("solar phase angle (°)", color=INK_2)
    cb.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cb.ax.get_yticklabels(), color=MUTED)
    cb.outline.set_edgecolor(BASELINE)

    # zoom on the densest few spin periods
    period = truth["spin_period_s"]
    nbins = int(obs.t_s.max() // (5 * period)) + 1
    counts, edges = np.histogram(obs.t_s, bins=nbins)
    i0 = np.argmax(counts)
    t_lo, t_hi = edges[i0], edges[i0] + 5 * period
    sel = (obs.t_s >= t_lo) & (obs.t_s < t_hi)
    ax2.scatter(obs.t_s[sel] - t_lo, m_norm[sel], c=phase[sel], cmap=SEQ_CMAP,
                s=14, linewidths=0)
    for k in range(1, 5):
        ax2.axvline(k * period, color=BASELINE, lw=0.8, ls="--")
    ax2.invert_yaxis()
    ax2.set_xlabel(f"seconds past t = {t_lo/3600:.2f} h   (dashed = true spin period)")
    ax2.set_ylabel("magnitude @ 1000 km")
    ax2.set_title("Zoom: five spin periods — deep spikes are solar-panel glints")
    save(fig, out / "01_lightcurve.png")


def chart_coverage(obs: ObservationSet, inv: dict, out: Path) -> None:
    fig = plt.figure(figsize=(12, 4.6), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="mollweide")
    ax1.grid(True, color=GRID, lw=0.5)
    u_obs = inv["u_obs_body_true"]
    lon, lat = lonlat(u_obs)
    t_h = obs.t_s / 3600
    sc = ax1.scatter(lon, lat, c=t_h, cmap=SEQ_CMAP, s=2.5, linewidths=0)
    lon_s, lat_s = lonlat(inv["u_sun_body_true"])
    ax1.scatter(lon_s, lat_s, color=S2_ORANGE, s=4, linewidths=0)
    ax1.text(0.99, 0.02, "orange: sun directions", color=S2_ORANGE, fontsize=9,
             ha="right", transform=ax1.transAxes)
    ax1.set_title("Observer directions in the target body frame", pad=14)
    ax1.tick_params(labelsize=7, colors=MUTED)
    cb = fig.colorbar(sc, ax=ax1, shrink=0.7, pad=0.02)
    cb.set_label("time (h)", color=INK_2)
    plt.setp(cb.ax.get_yticklabels(), color=MUTED)
    cb.outline.set_edgecolor(BASELINE)

    ax2 = fig.add_subplot(1, 2, 2)
    phase = obs.phase_angle_deg()
    bins = np.arange(0, 181, 5)
    ax2.hist(phase, bins=bins, color=S1_BLUE, edgecolor=SURFACE, linewidth=1.2)
    ax2.set_xlabel("solar phase angle (°)")
    ax2.set_ylabel("detections")
    ax2.set_title("Phase-angle diversity of the aggregate")
    save(fig, out / "02_coverage.png")


def chart_periodogram(inv: dict, truth: dict, out: Path) -> None:
    periods, power = inv["periods"], inv["power"]
    sel = periods <= 300
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(periods[sel], power[sel], color=S1_BLUE, lw=1.4)
    p_true = truth["spin_period_s"]
    ax.axvline(p_true, color=MUTED, ls="--", lw=1)
    p_ls = float(inv["best_period_ls"])
    i_pk = np.argmax(power)
    ax.plot([periods[i_pk]], [power[i_pk]], "o", ms=8, color=S2_ORANGE)
    ax.annotate(f"peak {p_ls:.2f} s  (truth {p_true:.1f} s)",
                xy=(periods[i_pk], power[i_pk]), xytext=(12, -2),
                textcoords="offset points", color=INK, fontsize=10)
    ax.set_xlabel("period (s)")
    ax.set_ylabel("Lomb–Scargle power")
    ax.set_title("Spin-period recovery from the multi-observer brightness stream")
    save(fig, out / "03_periodogram.png")


def chart_pole_search(inv: dict, truth: dict, summary: dict, out: Path) -> None:
    poles, costs = inv["grid_poles"], inv["grid_costs"]
    quality = (costs.max() - costs) / (costs.max() - costs.min())
    lon, lat = lonlat(poles)
    fig = plt.figure(figsize=(9, 5.2))
    ax = fig.add_subplot(projection="mollweide")
    ax.grid(True, color=GRID, lw=0.5)
    order = np.argsort(quality)
    sc = ax.scatter(lon[order], lat[order], c=quality[order], cmap=SEQ_CMAP,
                    s=26, linewidths=0)
    for u, marker, color, name in [
        (np.array(inv["est_pole"]), "x", S2_ORANGE, "estimate"),
        (pole_from_radec(truth), "o", INK, "truth"),
    ]:
        lo, la = lonlat(u[None, :])
        if marker == "o":
            ax.scatter(lo, la, facecolors="none", edgecolors=color, s=180, linewidths=1.6)
        else:
            ax.scatter(lo, la, marker=marker, color=color, s=110, linewidths=2.2)
    ax.text(0.5, -0.08, "○ truth pole    × estimate", color=INK_2, fontsize=10,
            ha="center", transform=ax.transAxes)
    ax.text(0.5, -0.15,
            f"pole error {summary['pole_err_deg']:.2f}°   "
            "(spin-axis sign ambiguity makes the antipode an equivalent mode)",
            color=MUTED, fontsize=9, ha="center", transform=ax.transAxes)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_title("Spin-pole grid search over the celestial sphere", pad=14)
    cb = fig.colorbar(sc, shrink=0.65, pad=0.02)
    cb.set_label("fit quality (1 = best)", color=INK_2)
    plt.setp(cb.ax.get_yticklabels(), color=MUTED)
    cb.outline.set_edgecolor(BASELINE)
    save(fig, out / "04_pole_search.png")


def pole_from_radec(truth: dict) -> np.ndarray:
    ra, dec = np.radians(truth["pole_ra_deg"]), np.radians(truth["pole_dec_deg"])
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def chart_egi(inv: dict, out: Path) -> None:
    labels = [str(s) for s in inv["group_labels"]]
    true_aa = inv["group_true_albedo_area"]
    rec_aa = inv["matched_albedo_area"]
    x = np.arange(len(labels))
    w = 0.38
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 4.6), width_ratios=[1.15, 1], constrained_layout=True
    )
    ax1.bar(x - w / 2, true_aa, w, color=S1_BLUE, label="truth ρ·A")
    ax1.bar(x + w / 2, rec_aa, w, color=S2_ORANGE, label="recovered (15° cone)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
    ax1.set_ylabel("diffuse albedo·area (m²)")
    ax1.set_title("EGI recovery per unique facet normal (estimated attitude)")
    ax1.legend()
    ax1.grid(axis="x", visible=False)

    # recovered EGI on the sphere of candidate normals
    ax2.remove()
    ax2 = fig.add_subplot(1, 2, 2, projection="mollweide")
    ax2.grid(True, color=GRID, lw=0.5)
    nrm, aa = inv["egi_normals"], inv["egi_albedo_area"]
    lon, lat = lonlat(nrm)
    nz = aa > 1e-3
    ax2.scatter(lon[~nz], lat[~nz], color=BASELINE, s=3, linewidths=0)
    sc = ax2.scatter(lon[nz], lat[nz], c=aa[nz], cmap=SEQ_CMAP,
                     s=18 + 220 * aa[nz] / max(aa.max(), 1e-9), linewidths=0)
    lon_t, lat_t = lonlat(inv["true_normals"])
    ax2.scatter(lon_t, lat_t, facecolors="none", edgecolors=INK, s=200, linewidths=1.2)
    ax2.text(0.99, 0.02, "rings: true facet normals", color=INK_2, fontsize=9,
             ha="right", transform=ax2.transAxes)
    ax2.tick_params(labelsize=7, colors=MUTED)
    ax2.set_title("Recovered EGI on the body-frame sphere", pad=14)
    cb = fig.colorbar(sc, shrink=0.65, pad=0.02)
    cb.set_label("ρ·A (m²)", color=INK_2)
    plt.setp(cb.ax.get_yticklabels(), color=MUTED)
    cb.outline.set_edgecolor(BASELINE)
    save(fig, out / "05_egi.png")


def main(results_dir: str = "results", charts_dir: str = "results/charts") -> None:
    res = Path(results_dir)
    out = Path(charts_dir)
    out.mkdir(parents=True, exist_ok=True)
    style()

    obs = ObservationSet.from_npz(res / "observations.npz")
    truth = json.loads((res / "truth.json").read_text())
    summary = json.loads((res / "summary.json").read_text())
    with np.load(res / "inversion.npz") as z:
        inv = {k: z[k] for k in z.files}

    chart_summary_tiles(summary, out)
    chart_lightcurve(obs, truth, out)
    chart_coverage(obs, inv, out)
    chart_periodogram(inv, truth, out)
    chart_pole_search(inv, truth, summary, out)
    chart_egi(inv, out)


if __name__ == "__main__":
    main(*sys.argv[1:])
