"""PAB (phase-angle-bisector) photogram demos.

Chart 12 — Fresnel normalization of the PAB photogram:
  raw brightness binned on the body-frame bisector direction vs the same
  data normalized by the deterministic phase envelope
  cos^2(alpha/2) * F_unpol(alpha/2) — plus the glint-amplitude-vs-phase
  scatter showing the envelope sitting on the data.

Chart 13 — beta-season sampling coverage:
  the bisector of sun and observer always lies within 90 deg of the sun,
  so the body-frame PAB coverage at any epoch is confined to the sun-side
  cap; as the season (sun geometry) changes, the reachable set moves.
  Rendered as exposure maps for two sun geometries, with the unsampled
  set masked — the coverage argument for exposure-normalizing any PAB
  product rather than "beta-correcting" its values.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from photometry import scenarios as sc
from photometry.frames import fibonacci_sphere, unit
from photometry.measurements import ObservationSet
from photometry.simulate import Scenario, run

SURFACE = "#1a1a19"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
BASELINE = "#383835"
S1_BLUE = "#3987e5"
S2_ORANGE = "#d95926"
SEQ = LinearSegmentedColormap.from_list("seq", [
    "#0d366b", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#cde2fb"])

N_GLASS = 1.5  # cover-glass refractive index for the Fresnel factor


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


def fresnel_unpolarized(theta_i: np.ndarray, n: float = N_GLASS) -> np.ndarray:
    """Exact unpolarized Fresnel reflectance, air -> dielectric."""
    ci = np.cos(theta_i)
    st = np.sin(theta_i) / n
    ct = np.sqrt(np.clip(1 - st**2, 0.0, 1.0))
    rs = ((ci - n * ct) / (ci + n * ct)) ** 2
    rp = ((n * ci - ct) / (n * ci + ct)) ** 2
    return 0.5 * (rs + rp)


def phase_envelope(alpha_rad: np.ndarray) -> np.ndarray:
    """Deterministic PAB-peak amplitude envelope: foreshortening x Fresnel.

    At the exact glint condition both cosines equal cos(alpha/2) and the
    specular incidence angle is alpha/2. Normalized to 1 at alpha = 0.
    """
    half = alpha_rad / 2
    f0 = fresnel_unpolarized(np.array([0.0]))[0]
    return np.cos(half) ** 2 * fresnel_unpolarized(half) / f0


def lonlat(u):
    return (np.arctan2(u[..., 1], u[..., 0]),
            np.arcsin(np.clip(u[..., 2], -1, 1)))


def bin_max(values, pab, grid, cos_bin):
    """Max of `values` within each grid cap (max highlights glints)."""
    out = np.full(len(grid), np.nan)
    hit = pab @ grid.T > cos_bin  # (K, C)
    for j in range(len(grid)):
        sel = hit[:, j]
        if sel.any():
            out[j] = values[sel].max()
    return out


def chart_pab_normalization(out: Path) -> None:
    name = "katalyst_link__tumble"
    obs = ObservationSet.from_npz(Path("results/fleet") / name / "observations.npz")
    obs = obs.uncensored()
    orbit, sun = sc.study_orbit(), sc.sun_eci()
    att, _ = sc.make_attitude("tumble", orbit, sun, False)

    u_s = att.eci_to_body(obs.t_s, obs.sun_eci)
    u_o = att.eci_to_body(obs.t_s, obs.u_obs_from_target())
    pab = unit(u_s + u_o)
    alpha = np.radians(obs.phase_angle_deg())
    b = obs.normalized_brightness()
    env = phase_envelope(alpha)

    grid = fibonacci_sphere(400)
    cos_bin = np.cos(np.radians(6.0))
    raw = bin_max(b, pab, grid, cos_bin)
    norm = bin_max(b / env, pab, grid, cos_bin)

    fig = plt.figure(figsize=(13.5, 7.6), constrained_layout=True)
    for i, (vals, title) in enumerate([
            (raw, "RAW PAB photogram — max brightness per bisector direction\n"
                  "(same panels, different apparent strength across phase)"),
            (norm, "NORMALIZED — divided by cos²(α/2) · Fresnel(α/2)\n"
                   "(the deterministic phase envelope removed)")]):
        ax = fig.add_subplot(2, 2, i + 1, projection="mollweide")
        ax.grid(True, color=GRID, lw=0.5)
        lo, la = lonlat(grid)
        ok = ~np.isnan(vals)
        v = np.log10(np.clip(vals, 1e-3, None))
        ax.scatter(lo[~ok], la[~ok], color=BASELINE, s=4, linewidths=0)
        p = ax.scatter(lo[ok], la[ok], c=v[ok], cmap=SEQ, s=22, linewidths=0)
        # LINK wing normals are body +/-z
        ax.scatter([0, 0], [np.pi / 2 - 1e-3, -np.pi / 2 + 1e-3],
                   facecolors="none", edgecolors=INK, s=180, linewidths=1.3)
        ax.tick_params(labelsize=6, colors=MUTED)
        ax.set_title(title, fontsize=11, pad=12)
        cb = fig.colorbar(p, ax=ax, shrink=0.65, pad=0.02)
        cb.set_label("log10 brightness (m²)", color=INK_2)
        plt.setp(cb.ax.get_yticklabels(), color=MUTED)
        cb.outline.set_edgecolor(BASELINE)

    # glint amplitude vs phase: rows whose bisector sits ON a wing normal.
    # tight cone: at Phong exponent 800 the lobe half-width is ~3 deg, so a
    # loose cone admits rows decades down the lobe and buries the envelope
    ax = fig.add_subplot(2, 1, 2)
    on_normal = np.abs(pab[:, 2]) > np.cos(np.radians(2.5))
    a_deg = np.degrees(alpha[on_normal])
    b_g = b[on_normal]
    ax.scatter(a_deg, b_g, s=22, color=S1_BLUE, alpha=0.85,
               label="detections with bisector within 2.5° of a wing normal")
    aa = np.linspace(np.radians(max(5.0, a_deg.min() - 5)),
                     np.radians(min(150.0, a_deg.max() + 5)), 200)
    # scale both reference curves to the upper edge of the low-phase data
    low = b_g[a_deg < np.percentile(a_deg, 25)]
    ref = np.percentile(low if len(low) else b_g, 95)
    ref_a = np.radians(np.percentile(a_deg, 12))
    ax.plot(np.degrees(aa), ref / phase_envelope(ref_a) * phase_envelope(aa),
            color=S2_ORANGE, lw=2.2,
            label="cos²(α/2) · Fresnel(α/2) envelope (n = 1.5) — nearly flat: "
                  "Fresnel brightening cancels foreshortening")
    cos_only = np.cos(aa / 2) ** 2 / np.cos(ref_a / 2) ** 2 * ref
    ax.plot(np.degrees(aa), cos_only, color=MUTED, lw=1.6, ls="--",
            label="cos²(α/2) alone — what pure foreshortening would predict")
    ax.set_yscale("log")
    ax.set_xlabel("solar phase angle α (°)")
    ax.set_ylabel("range-normalized brightness (m²)")
    ax.set_title("Glint amplitude vs phase — both factors are deterministic, "
                 "so removing them is normalization, not correction", fontsize=11)
    ax.legend(loc="lower left", fontsize=9.5)
    fig.suptitle("PAB photogram Fresnel normalization — Katalyst LINK tumble, "
                 "true attitude, calibrated rows", fontweight="bold")
    path = out / "12_pab_photogram.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def chart_beta_coverage(out: Path) -> None:
    """PAB exposure maps for two sun geometries (two 'beta seasons')."""
    grid = fibonacci_sphere(400)
    cos_bin = np.cos(np.radians(6.0))
    seasons = [("season A — sun RA 130°, Dec +15°", 130.0, 15.0),
               ("season B — sun RA 250°, Dec −20°", 250.0, -20.0)]
    fig = plt.figure(figsize=(13.5, 4.8), constrained_layout=True)
    counts_all = []
    for i, (label, ra, dec) in enumerate(seasons):
        s = Scenario(duration_s=2 * 5820.0, dt_s=6.0,
                     sun_ra_deg=ra, sun_dec_deg=dec)
        obs, scn = run(s)
        obs = obs.uncensored()
        att = scn.target_attitude()
        u_s = att.eci_to_body(obs.t_s, obs.sun_eci)
        u_o = att.eci_to_body(obs.t_s, obs.u_obs_from_target())
        pab = unit(u_s + u_o)
        counts = (pab @ grid.T > cos_bin).sum(axis=0).astype(float)
        counts_all.append(counts)

        ax = fig.add_subplot(1, 2, i + 1, projection="mollweide")
        ax.grid(True, color=GRID, lw=0.5)
        lo, la = lonlat(grid)
        empty = counts == 0
        ax.scatter(lo[empty], la[empty], color="#4a1c1c", s=8, linewidths=0)
        p = ax.scatter(lo[~empty], la[~empty], c=np.log10(counts[~empty]),
                       cmap=SEQ, s=22, linewidths=0)
        # body-frame sun cone: sun direction at t=0 for reference
        sb = u_s[0]
        slo, sla = lonlat(sb[None, :])
        ax.scatter(slo, sla, marker="*", s=140, color="#c98500")
        ax.tick_params(labelsize=6, colors=MUTED)
        ax.set_title(f"{label}\n({int(empty.sum())}/{len(grid)} directions "
                     "never sampled — dark red)", fontsize=10.5, pad=12)
        cb = fig.colorbar(p, ax=ax, shrink=0.65, pad=0.02)
        cb.set_label("log10 samples per bin", color=INK_2)
        plt.setp(cb.ax.get_yticklabels(), color=MUTED)
        cb.outline.set_edgecolor(BASELINE)

    both_empty = (counts_all[0] == 0) & (counts_all[1] == 0)
    fig.suptitle(
        "PAB exposure maps — the bisector always lies within 90° of the sun, so each "
        "'beta season' can only sample the sun-side cap of the body sphere\n"
        f"(union of two seasons still leaves {int(both_empty.sum())}/{len(grid)} "
        "directions unsampled; a PAB product must mask or exposure-normalize these, "
        "not value-correct them)", fontweight="bold", fontsize=12)
    path = out / "13_pab_coverage.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    style()
    out = Path("results/charts")
    out.mkdir(parents=True, exist_ok=True)
    chart_pab_normalization(out)
    chart_beta_coverage(out)
