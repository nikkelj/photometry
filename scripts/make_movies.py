"""Render per-scenario LVLH-frame validation movies (GIF).

Each frame shows, in the target's LVLH frame (x along-track, y cross-track,
z radial-out):
  - the TRUTH shape (polygons, truth attitude + articulation) — the
    "navigation data" the inversion is judged against
  - the INVERTED product: a convex body Minkowski-reconstructed from the
    recovered EGI (a solid whose face normals/areas match the recovered
    oriented-area distribution), oriented by the *estimated* attitude and
    centered on the truth (photometry carries no position information).
    Closure faces — where the EGI has no recovered area and the body is
    capped by a bounding cage — draw as dim edges. Falls back to exploded
    EGI disks if reconstruction fails.
  - a 3D view plus 2D projections onto the three LVLH planes

Usage: python scripts/make_movies.py [scenario ...]   (default: all found)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from photometry import scenarios as sc
from photometry.attitude import PrincipalAxisSpin
from photometry.frames import in_earth_shadow, lvlh_basis, minimal_rotation_between, unit
from photometry.inversion.minkowski import hull_from_egi
from photometry.shapes import GIMBAL_FIXED

SURFACE = "#1a1a19"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
BASELINE = "#383835"
TRUTH_FILL = "#86b6ef"
TRUTH_EDGE = "#3987e5"
EST_COLOR = "#d95926"
SUN_COLOR = "#c98500"

N_FRAMES = 144
FPS = 12


def style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.labelcolor": INK_2, "axes.edgecolor": BASELINE,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9,
        "axes.titlesize": 10, "legend.frameon": False,
        "legend.labelcolor": INK_2,
    })


def estimated_attitude(result: dict, inv: dict, orbit, sun):
    mode = result["mode_best"]
    if mode in ("spin_fit", "inertial_fit"):
        ra, dec, per, ph, ax_, ay, az = inv["best_spin_params"]
        return PrincipalAxisSpin(float(ra), float(dec), float(per), float(ph),
                                 body_axis=(float(ax_), float(ay), float(az)))
    named = {"lvlh_ops": "ops", "lvlh_low_drag": "low_drag",
             "sun_point": "sun_point"}
    att, _ = sc.make_attitude(named[mode], orbit, sun, True)
    return att


def articulated_polys(shape, u_sun_body_1: np.ndarray) -> list[np.ndarray]:
    """Polygon vertex arrays in the body frame with articulation applied."""
    n_cur = shape.body_normals(u_sun_body_1[None, :], articulate=True)[:, 0, :]
    polys = []
    for verts, fi in shape.polygons:
        v = verts
        if shape.gimbal_mode[fi] != GIMBAL_FIXED:
            r = minimal_rotation_between(shape.normals[fi], n_cur[fi])
            c = verts.mean(axis=0)
            v = (verts - c) @ r.T + c
        polys.append(v)
    return polys


def egi_disks(inv: dict, r_shell: float, top_n: int = 24) -> list[np.ndarray]:
    """Disk polygons (body frame) for the strongest recovered EGI directions."""
    normals = inv["egi_normals"]
    w = inv["egi_albedo_area"] + inv["egi_specular_area"]
    order = np.argsort(w)[::-1]
    order = [i for i in order[:top_n] if w[i] > 0.03 * w.max() and w[i] > 1e-3]
    th = np.linspace(0, 2 * np.pi, 17)
    disks = []
    for i in order:
        n = unit(normals[i])
        area_est = w[i] / 0.35  # nominal albedo -> physical area guess
        rad = min(np.sqrt(area_est / np.pi), 1.2 * r_shell)
        helper = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
        e1 = unit(np.cross(n, helper))
        e2 = np.cross(n, e1)
        c = 0.45 * r_shell * n
        disks.append(c + rad * (np.outer(np.cos(th), e1) + np.outer(np.sin(th), e2)))
    return disks


def to_lvlh(verts_body: np.ndarray, r_bi: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Body-frame vertices -> LVLH components. basis rows = (along, cross, up)."""
    return (verts_body @ r_bi.T) @ basis.T


def attitude_error_deg(r_est: np.ndarray, r_true: np.ndarray) -> float:
    c = (np.trace(r_est @ r_true.T) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def render_movie(name: str, fleet_dir: Path, out_dir: Path) -> Path:
    s = sc.by_name(name)
    result = json.loads((fleet_dir / name / "result.json").read_text())
    with np.load(fleet_dir / name / "inversion.npz") as z:
        inv = {k: z[k] for k in z.files}

    orbit = sc.study_orbit()
    sun = sc.sun_eci()
    shape = s.shape()
    att_true, articulate = sc.make_attitude(s.mode, orbit, sun, shape.articulated)
    att_est = estimated_attitude(result, inv, orbit, sun)

    # frame cadence: resolve the spin for tumblers, one orbit for controlled
    p_orb = 2 * np.pi / orbit.mean_motion
    spin_p = result.get("spin_period_est_s")
    fast = s.mode == "tumble" and spin_p and spin_p < 2000
    frame_dt = (spin_p / 10) if fast else (p_orb / N_FRAMES)

    # start the window where the target is sunlit
    t_scan = np.arange(0, p_orb, 30.0)
    r_scan, _ = orbit.single_states(t_scan)
    sunlit = ~in_earth_shadow(r_scan, sun)
    t0 = float(t_scan[np.argmax(sunlit)])
    times = t0 + frame_dt * np.arange(N_FRAMES)

    r_shell = shape.characteristic_radius()
    lim = 1.15 * r_shell
    # inverted shape: Minkowski-reconstructed convex hull from the EGI —
    # an actual solid, directly comparable to the truth polygons. Falls
    # back to exploded EGI disks if the reconstruction fails.
    try:
        hull = hull_from_egi(inv["egi_normals"], inv["egi_albedo_area"],
                             inv["egi_specular_area"])
        est_body = [f.vertices for f in hull if not f.is_closure]
        est_closure = [f.vertices for f in hull if f.is_closure]
        if not est_body:
            raise RuntimeError("empty hull")
    except Exception as e:
        print(f"  hull reconstruction failed ({e}); using EGI disks")
        est_body = egi_disks(inv, r_shell)
        est_closure = []

    style()
    fig = plt.figure(figsize=(10.5, 7.0))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.55, 1], hspace=0.32, wspace=0.12)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xz = fig.add_subplot(gs[1, 1])
    ax_yz = fig.add_subplot(gs[2, 1])
    planes = [(ax_xy, 0, 1, "along-track", "cross-track"),
              (ax_xz, 0, 2, "along-track", "radial (up)"),
              (ax_yz, 1, 2, "cross-track", "radial (up)")]

    errs = []

    def draw(fi):
        t = float(times[fi])
        r_eci, v_eci = orbit.single_states(np.array([t]))
        along, cross, up = lvlh_basis(r_eci, v_eci)
        basis = np.stack([along[0], cross[0], up[0]])  # rows

        r_true = att_true.body_to_eci_matrix(t)
        r_est = att_est.body_to_eci_matrix(t)
        err = attitude_error_deg(r_est, r_true)
        errs.append(err)

        u_sun_body_true = att_true.eci_to_body(np.array([t]), sun[None, :])[0]
        polys_body = articulated_polys(shape, u_sun_body_true) if articulate \
            else [v for v, _ in shape.polygons]
        polys_lvlh = [to_lvlh(v, r_true, basis) for v in polys_body]
        disks_lvlh = [to_lvlh(d, r_est, basis) for d in est_body]
        closure_lvlh = [to_lvlh(d, r_est, basis) for d in est_closure]
        sun_lvlh = basis @ sun

        ax3.clear()
        ax3.set_facecolor(SURFACE)
        ax3.add_collection3d(Poly3DCollection(
            polys_lvlh, facecolors=TRUTH_FILL, edgecolors=TRUTH_EDGE,
            linewidths=0.7, alpha=0.30))
        ax3.add_collection3d(Poly3DCollection(
            disks_lvlh, facecolors=EST_COLOR, edgecolors=EST_COLOR,
            linewidths=0.5, alpha=0.45))
        if closure_lvlh:
            ax3.add_collection3d(Poly3DCollection(
                closure_lvlh, facecolors="none", edgecolors=EST_COLOR,
                linewidths=0.4, alpha=0.25))
        a = lim
        ax3.quiver(-a, 0, 0, 0.55 * a, 0, 0, color=MUTED, arrow_length_ratio=0.12)
        ax3.text(-a + 0.6 * a, 0, 0.06 * a, "v (along)", color=MUTED, fontsize=8)
        ax3.quiver(0, 0, 0, *(0.9 * a * sun_lvlh), color=SUN_COLOR,
                   arrow_length_ratio=0.10)
        ax3.text(*(1.02 * a * sun_lvlh), "sun", color=SUN_COLOR, fontsize=8)
        ax3.set_xlim(-lim, lim); ax3.set_ylim(-lim, lim); ax3.set_zlim(-lim, lim)
        ax3.set_xlabel("along", color=INK_2, labelpad=-4)
        ax3.set_ylabel("cross", color=INK_2, labelpad=-4)
        ax3.set_zlabel("up", color=INK_2, labelpad=-4)
        ax3.tick_params(labelsize=6, colors=MUTED, pad=-2)
        ax3.xaxis.pane.set_color(SURFACE); ax3.yaxis.pane.set_color(SURFACE)
        ax3.zaxis.pane.set_color(SURFACE)
        ax3.grid(False)
        ax3.set_title(
            f"{s.sat}  —  truth mode: {s.mode}   estimated: {result['mode_best']}"
            f"\nt = {t/3600:.2f} h    bus-frame attitude error {err:5.1f}°"
            " (includes any body-symmetry ambiguity)",
            color=INK, fontsize=11, pad=2)

        for ax, i, j, xl, yl in planes:
            ax.clear()
            ax.set_facecolor(SURFACE)
            for v in polys_lvlh:
                ax.fill(v[:, i], v[:, j], color=TRUTH_FILL, alpha=0.25, lw=0)
                ax.plot(np.append(v[:, i], v[0, i]), np.append(v[:, j], v[0, j]),
                        color=TRUTH_EDGE, lw=0.7)
            for d in disks_lvlh:
                ax.fill(d[:, i], d[:, j], color=EST_COLOR, alpha=0.18, lw=0)
                ax.plot(np.append(d[:, i], d[0, i]), np.append(d[:, j], d[0, j]),
                        color=EST_COLOR, lw=0.9, alpha=0.85)
            for d in closure_lvlh:
                ax.plot(np.append(d[:, i], d[0, i]), np.append(d[:, j], d[0, j]),
                        color=EST_COLOR, lw=0.4, alpha=0.35)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xlabel(xl, fontsize=7.5, labelpad=1)
            ax.set_ylabel(yl, fontsize=7.5, labelpad=1)
            ax.tick_params(labelsize=6)
            ax.grid(color=GRID, lw=0.4)

        if fi == 0:
            fig.legend(handles=[
                mpl.patches.Patch(facecolor=TRUTH_FILL, edgecolor=TRUTH_EDGE,
                                  alpha=0.5, label="truth shape @ truth attitude (nav data)"),
                mpl.patches.Patch(facecolor=EST_COLOR, alpha=0.6,
                                  label="inverted shape (Minkowski hull from EGI) @ estimated attitude"),
            ], loc="lower left", fontsize=8.5, bbox_to_anchor=(0.02, 0.015))

    anim = FuncAnimation(fig, draw, frames=N_FRAMES, interval=1000 / FPS)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.gif"
    anim.save(path, writer=PillowWriter(fps=FPS), dpi=72)
    plt.close(fig)
    mean_err = float(np.mean(errs)) if errs else float("nan")
    print(f"wrote {path}  (mean attitude err over movie: {mean_err:.1f} deg)")
    return path


def main(*names: str) -> None:
    fleet = Path("results/fleet")
    out = Path("results/movies")
    todo = list(names) if names else sorted(
        p.name for p in fleet.iterdir()
        if p.is_dir() and (p / "result.json").exists())
    for n in todo:
        render_movie(n, fleet, out)


if __name__ == "__main__":
    main(*sys.argv[1:])
