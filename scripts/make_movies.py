"""Render per-scenario LVLH-frame validation movies (GIF), split-plot layout.

Top-left 3D  — truth shape (nav data) + the model-free inversion products:
               Minkowski hull from the EGI (solid) and raw EGI disks
               (oriented area, exploded outward; photometry recovers no
               position information).
Top-right 3D — the Tier-2 identification: best-matching library model at
               the identified attitude and array configuration, over faint
               truth. This is "what the pipeline would report for an
               unknown object" vs what is really there.
Bottom       — 2D projections onto the three LVLH planes (truth, hull,
               matched model).

Identification is read from results/model_match/<scenario>.json when
present; otherwise falls back to the scenario's own classifier result
(true shape + best mode).

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
from photometry.attitude import (FixedInertial, LvlhHold, PrincipalAxisSpin,
                                 TorqueFreeTumble)
from photometry.frames import in_earth_shadow, lvlh_basis, minimal_rotation_between, unit
from photometry.inversion.minkowski import hull_from_egi
from photometry.shapes import GIMBAL_FIXED, LIBRARY

SURFACE = "#1a1a19"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
BASELINE = "#383835"
TRUTH_FILL = "#86b6ef"
TRUTH_EDGE = "#3987e5"
EST_COLOR = "#d95926"
DISK_COLOR = "#d55181"
MATCH_FILL = "#199e70"
MATCH_EDGE = "#1baf7a"
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


def load_identification(name: str, result: dict, inv: dict) -> dict:
    """Best-match model + attitude spec, from Tier-2 matching if available."""
    p = Path("results/model_match") / f"{name}.json"
    if p.exists():
        m = json.loads(p.read_text())
        return dict(model=m["identified_model"],
                    hypothesis=m["identified_hypothesis"],
                    arrays_tracking=bool(m["identified_arrays_tracking"]),
                    spin=m.get("identified_spin"),
                    cost=m["top5"][0]["cost"], source="library match")
    spin = None
    if result["mode_best"] in ("spin_fit", "inertial_fit"):
        spin = [float(v) for v in inv["best_spin_params"]]
    return dict(model=result["sat"], hypothesis=result["mode_best"],
                arrays_tracking=result["mode_best"] == "lvlh_ops",
                spin=spin, cost=result["hypothesis_costs"][result["mode_best"]],
                source="classifier (true shape)")


def attitude_from_spec(hypothesis: str, spin, orbit, sun):
    if hypothesis in ("spin_fit", "inertial_fit"):
        return PrincipalAxisSpin(spin[0], spin[1], spin[2], spin[3],
                                 body_axis=tuple(spin[4:7]))
    return {"lvlh_ops": LvlhHold(orbit),
            "lvlh_low_drag": LvlhHold(orbit, roll_deg=90.0),
            "sun_point": FixedInertial.z_toward(sun)}[hypothesis]


def articulated_polys(shape, u_sun_body_1: np.ndarray,
                      articulate: bool) -> list[np.ndarray]:
    """Polygon vertex arrays in the body frame with articulation applied."""
    if not articulate:
        return [v for v, _ in shape.polygons]
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


def _setup_3d(ax, lim):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel("along", color=INK_2, labelpad=-4)
    ax.set_ylabel("cross", color=INK_2, labelpad=-4)
    ax.set_zlabel("up", color=INK_2, labelpad=-4)
    ax.tick_params(labelsize=6, colors=MUTED, pad=-2)
    ax.xaxis.pane.set_color(SURFACE)
    ax.yaxis.pane.set_color(SURFACE)
    ax.zaxis.pane.set_color(SURFACE)
    ax.grid(False)


def render_movie(name: str, fleet_dir: Path, out_dir: Path) -> Path:
    s = sc.by_name(name)
    result = json.loads((fleet_dir / name / "result.json").read_text())
    with np.load(fleet_dir / name / "inversion.npz") as z:
        inv = {k: z[k] for k in z.files}
    ident = load_identification(name, result, inv)

    orbit = sc.study_orbit()
    sun = sc.sun_eci()
    shape = s.shape()
    att_true, articulate = sc.make_attitude(s.mode, orbit, sun, shape.articulated)
    match_shape = LIBRARY[ident["model"]]()
    att_match = attitude_from_spec(ident["hypothesis"], ident["spin"], orbit, sun)
    match_articulate = ident["arrays_tracking"] and match_shape.articulated

    # a Tier-3 torque-free fit, when present, supersedes the uniform spin
    tf_path = fleet_dir / name / "torquefree.json"
    if tf_path.exists():
        tf = json.loads(tf_path.read_text())
        att_match = TorqueFreeTumble(tuple(tf["inertia_est"]),
                                     tuple(tf["omega0_est"]),
                                     np.array(tf["r0"]), t_max=90000.0,
                                     dt=1.0, t_ref=tf["t_ref"])
        ident["hypothesis"] = "torque_free"
        ident["source"] = "Tier-3 torque-free fit"
        ident["cost"] = tf.get("cost_torquefree_polished",
                               tf.get("cost_torquefree", float("nan")))

    # frame cadence: resolve the spin for tumblers, one orbit for controlled
    p_orb = 2 * np.pi / orbit.mean_motion
    spin_p = result.get("spin_period_est_s")
    fast = s.mode in ("tumble", "multiaxis_tumble") and spin_p and spin_p < 2000
    frame_dt = (spin_p / 10) if fast else (p_orb / N_FRAMES)

    # start the window where the target is sunlit
    t_scan = np.arange(0, p_orb, 30.0)
    r_scan, _ = orbit.single_states(t_scan)
    sunlit = ~in_earth_shadow(r_scan, sun)
    t0 = float(t_scan[np.argmax(sunlit)])
    times = t0 + frame_dt * np.arange(N_FRAMES)

    r_shell = shape.characteristic_radius()
    lim = 1.15 * max(r_shell, match_shape.characteristic_radius())
    try:
        hull = hull_from_egi(inv["egi_normals"], inv["egi_albedo_area"],
                             inv["egi_specular_area"])
        est_body = [f.vertices for f in hull if not f.is_closure]
        est_closure = [f.vertices for f in hull if f.is_closure]
    except Exception as e:
        print(f"  hull reconstruction failed ({e}); rendering EGI disks only")
        est_body, est_closure = [], []
    disks_raw = egi_disks(inv, r_shell)

    style()
    fig = plt.figure(figsize=(12.5, 8.0))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.75, 1], hspace=0.28,
                          wspace=0.55)
    ax3l = fig.add_subplot(gs[0, :3], projection="3d")
    ax3r = fig.add_subplot(gs[0, 3:], projection="3d")
    ax_xy = fig.add_subplot(gs[1, 0:2])
    ax_xz = fig.add_subplot(gs[1, 2:4])
    ax_yz = fig.add_subplot(gs[1, 4:6])
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
        r_match = att_match.body_to_eci_matrix(t)
        err = attitude_error_deg(r_match, r_true)
        errs.append(err)

        u_sun_true = att_true.eci_to_body(np.array([t]), sun[None, :])[0]
        u_sun_match = att_match.eci_to_body(np.array([t]), sun[None, :])[0]
        polys_lvlh = [to_lvlh(v, r_true, basis)
                      for v in articulated_polys(shape, u_sun_true, articulate)]
        match_lvlh = [to_lvlh(v, r_match, basis)
                      for v in articulated_polys(match_shape, u_sun_match,
                                                 match_articulate)]
        hull_lvlh = [to_lvlh(d, r_match, basis) for d in est_body]
        closure_lvlh = [to_lvlh(d, r_match, basis) for d in est_closure]
        raw_lvlh = [to_lvlh(d, r_match, basis) for d in disks_raw]
        sun_lvlh = basis @ sun

        # left: truth + model-free inversion
        ax3l.clear()
        _setup_3d(ax3l, lim)
        ax3l.add_collection3d(Poly3DCollection(
            polys_lvlh, facecolors=TRUTH_FILL, edgecolors=TRUTH_EDGE,
            linewidths=0.7, alpha=0.30))
        if hull_lvlh:
            ax3l.add_collection3d(Poly3DCollection(
                hull_lvlh, facecolors=EST_COLOR, edgecolors=EST_COLOR,
                linewidths=0.5, alpha=0.45))
        if closure_lvlh:
            ax3l.add_collection3d(Poly3DCollection(
                closure_lvlh, facecolors="none", edgecolors=EST_COLOR,
                linewidths=0.4, alpha=0.25))
        if raw_lvlh:
            ax3l.add_collection3d(Poly3DCollection(
                raw_lvlh, facecolors="none", edgecolors=DISK_COLOR,
                linewidths=0.8, alpha=0.55))
        ax3l.quiver(0, 0, 0, *(0.9 * lim * sun_lvlh), color=SUN_COLOR,
                    arrow_length_ratio=0.10)
        ax3l.text(*(1.02 * lim * sun_lvlh), "sun", color=SUN_COLOR, fontsize=8)
        ax3l.set_title("model-free inversion: EGI + Minkowski hull",
                       color=INK_2, fontsize=10, pad=0)

        # right: truth (faint) + identified library model at identified attitude
        ax3r.clear()
        _setup_3d(ax3r, lim)
        ax3r.add_collection3d(Poly3DCollection(
            polys_lvlh, facecolors=TRUTH_FILL, edgecolors=TRUTH_EDGE,
            linewidths=0.5, alpha=0.15))
        ax3r.add_collection3d(Poly3DCollection(
            match_lvlh, facecolors=MATCH_FILL, edgecolors=MATCH_EDGE,
            linewidths=0.8, alpha=0.35))
        ax3r.quiver(0, 0, 0, *(0.9 * lim * sun_lvlh), color=SUN_COLOR,
                    arrow_length_ratio=0.10)
        ax3r.set_title(
            f"identified: {ident['model']} | {ident['hypothesis']} | "
            f"arrays {'tracking' if ident['arrays_tracking'] else 'frozen'}",
            color=MATCH_EDGE, fontsize=10, pad=0)

        for ax, i, j, xl, yl in planes:
            ax.clear()
            ax.set_facecolor(SURFACE)
            for v in polys_lvlh:
                ax.fill(v[:, i], v[:, j], color=TRUTH_FILL, alpha=0.22, lw=0)
                ax.plot(np.append(v[:, i], v[0, i]), np.append(v[:, j], v[0, j]),
                        color=TRUTH_EDGE, lw=0.7)
            for d in hull_lvlh:
                ax.plot(np.append(d[:, i], d[0, i]), np.append(d[:, j], d[0, j]),
                        color=EST_COLOR, lw=0.8, alpha=0.75)
            for v in match_lvlh:
                ax.plot(np.append(v[:, i], v[0, i]), np.append(v[:, j], v[0, j]),
                        color=MATCH_EDGE, lw=0.9, alpha=0.85)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xlabel(xl, fontsize=7.5, labelpad=1)
            ax.set_ylabel(yl, fontsize=7.5, labelpad=1)
            ax.tick_params(labelsize=6)
            ax.grid(color=GRID, lw=0.4)

        fig.suptitle(
            f"{s.sat}  —  truth mode: {s.mode}    "
            f"[{ident['source']}, cost {ident['cost']:.2f}]\n"
            f"t = {t/3600:.2f} h    bus-frame attitude error {err:5.1f}°"
            " (includes any body-symmetry ambiguity)",
            color=INK, fontsize=11, y=0.99)

        if fi == 0:
            fig.legend(handles=[
                mpl.patches.Patch(facecolor=TRUTH_FILL, edgecolor=TRUTH_EDGE,
                                  alpha=0.5, label="truth @ truth attitude (nav data)"),
                mpl.patches.Patch(facecolor=EST_COLOR, alpha=0.6,
                                  label="Minkowski hull from EGI @ estimated attitude"),
                mpl.patches.Patch(facecolor="none", edgecolor=DISK_COLOR,
                                  label="raw EGI disks (oriented area, exploded)"),
                mpl.patches.Patch(facecolor=MATCH_FILL, edgecolor=MATCH_EDGE,
                                  alpha=0.6, label="identified library model @ identified attitude"),
            ], loc="lower left", fontsize=8, ncol=2,
                bbox_to_anchor=(0.02, 0.005))

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
