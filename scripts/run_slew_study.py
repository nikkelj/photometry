"""Maneuver-slew detectability study: pre-burn 90/180 deg yaw-arounds.

Satellites commonly yaw 90 or 180 deg over several minutes before an
orbit-adjust burn, hold through the burn, and slew back. This study asks
whether the existing stack detects and fits that profile on its natural
timescale, using three tools it already has:

  1. sliding-window residual detector — score the *nominal* hypothesis
     (LVLH ops, arrays tracking) in short windows; a maneuver shows up as
     a robust-z excursion of the window cost. This is the stored
     change-point idea at a single resolution.
  2. windowed yaw estimator — per window, grid the constant-yaw LVLH
     hypothesis over theta and take the argmin: a model-referenced
     yaw(t) track with an identifiability contrast (max-min cost over
     theta) that says whether yaw is observable *at all* for this craft.
  3. parametric slew fit — fit (t_start, slew_s, hold_s, yaw_hold) of
     `LvlhYawSlew` by Nelder-Mead seeded from the detector, on the full
     arc.

Craft are chosen to span the observability spectrum: 1-axis-array birds
(v1.5, Persona, Yaogan-SAR) change their achievable array pointing when
the gimbal axis yaws away from along-track, while a 2-axis-array
flat-sat (v2 mini) keeps its arrays sun-locked through the slew and its
bus top/bottom normals are yaw-invariant — near-worst case. Control runs
(no maneuver) calibrate false alarms.

Output: results/slew/summary.json + per-scenario window tables.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from photometry import scenarios as sc
from photometry.attitude import LvlhHold, LvlhYawSlew
from photometry.constellation import WalkerConstellation
from photometry.inversion.cost import huber_mag_cost, prepare_meas
from photometry.library200 import full_library
from photometry.measurements import ObservationSet
from photometry.sensing import SensorConfig, simulate_detections

DURATION_S = 3 * 3600.0
DT_S = 6.0
T_START = 5400.0     # slew begins 90 min in
SLEW_S = 360.0       # 6 min out, 6 min back
HOLD_S = 900.0       # 15 min burn hold
WIN_S = 120.0
STRIDE_S = 60.0
Z_THRESH = 5.0

CRAFT = ["starlink_v15", "starlink_v2mini", "ru_persona", "cn_yaogan_sar"]
CASES = [(c, y) for c in CRAFT for y in (90.0, 180.0)]
CONTROLS = ["starlink_v15", "starlink_v2mini"]


def window_costs(obs: ObservationSet, shape, att, articulate: bool,
                 win_s=WIN_S, stride_s=STRIDE_S) -> tuple[np.ndarray, np.ndarray]:
    """(window centers, mean huber cost of `att` per window)."""
    t0, t1 = obs.t_s.min(), obs.t_s.max()
    centers, costs = [], []
    t = t0
    while t + win_s <= t1:
        sel = np.nonzero((obs.t_s >= t) & (obs.t_s < t + win_s))[0]
        if len(sel) >= 8:
            prep = prepare_meas(obs.subset(sel))
            centers.append(t + win_s / 2)
            costs.append(huber_mag_cost(shape, att, articulate, prep,
                                        offset_sigma=0.5))
        t += stride_s
    return np.array(centers), np.array(costs)


def robust_z(costs: np.ndarray) -> np.ndarray:
    med = np.median(costs)
    mad = np.median(np.abs(costs - med)) * 1.4826
    return (costs - med) / max(mad, 1e-9)


def window_yaw_track(obs: ObservationSet, shape, orbit, articulate: bool,
                     theta_grid=None, win_s=WIN_S, stride_s=STRIDE_S):
    """Per-window constant-yaw scan: (centers, yaw_est, contrast)."""
    if theta_grid is None:
        theta_grid = np.arange(-180.0, 180.0, 7.5)
    atts = {th: LvlhHold(orbit, yaw_deg=float(th)) for th in theta_grid}
    t0, t1 = obs.t_s.min(), obs.t_s.max()
    centers, yaw_est, contrast = [], [], []
    t = t0
    while t + win_s <= t1:
        sel = np.nonzero((obs.t_s >= t) & (obs.t_s < t + win_s))[0]
        if len(sel) >= 8:
            prep = prepare_meas(obs.subset(sel))
            c = np.array([huber_mag_cost(shape, atts[th], articulate, prep,
                                         offset_sigma=0.5)
                          for th in theta_grid])
            i = int(np.argmin(c))
            centers.append(t + win_s / 2)
            yaw_est.append(float(theta_grid[i]))
            contrast.append(float(c.max() - c.min()))
        t += stride_s
    return np.array(centers), np.array(yaw_est), np.array(contrast)


def detect(centers: np.ndarray, z: np.ndarray,
           thresh=Z_THRESH, consecutive=2) -> list[tuple[float, float]]:
    """Contiguous windows with z >= thresh (needs `consecutive` in a row)."""
    hot = z >= thresh
    events, start = [], None
    run = 0
    for i, h in enumerate(hot):
        if h:
            run += 1
            if run == consecutive:
                start = centers[i - consecutive + 1]
        else:
            if start is not None:
                events.append((float(start), float(centers[i - 1])))
            start, run = None, 0
    if start is not None:
        events.append((float(start), float(centers[-1])))
    return events


def fit_slew(obs: ObservationSet, shape, orbit, articulate: bool,
             seed_t0: float, seed_yaw: float) -> dict:
    prep = prepare_meas(obs)

    def cost(x):
        t0, slew, hold, yaw = x
        if not (0 < t0 < DURATION_S and 60 < slew < 1800 and
                60 < hold < 3600):
            return 1e6
        att = LvlhYawSlew(orbit, yaw, t0, slew, hold)
        return huber_mag_cost(shape, att, articulate, prep, offset_sigma=0.5)

    best = None
    for slew0 in (240.0, 480.0):
        for hold0 in (600.0, 1200.0):
            res = minimize(cost, [seed_t0, slew0, hold0, seed_yaw],
                           method="Nelder-Mead",
                           options=dict(maxiter=400, xatol=1.0, fatol=1e-6))
            if best is None or res.fun < best.fun:
                best = res
    t0, slew, hold, yaw = best.x
    return dict(t_start_s=float(t0), slew_s=float(slew), hold_s=float(hold),
                yaw_hold_deg=float(yaw), cost=float(best.fun))


def run_case(lib, name: str, yaw: float | None, seed=99) -> dict:
    orbit, sun = sc.study_orbit(), sc.sun_eci()
    shape = lib[name]()
    articulate = shape.articulated
    if yaw is None:
        att_true = LvlhHold(orbit)
        label = f"{name}__control"
    else:
        att_true = LvlhYawSlew(orbit, yaw, T_START, SLEW_S, HOLD_S)
        label = f"{name}__yaw{int(yaw)}"
    rng = np.random.default_rng(seed)
    constellation = WalkerConstellation(100, 100, 550.0, 53.0)
    t_grid = np.arange(0.0, DURATION_S, DT_S)
    t0 = time.time()
    obs = simulate_detections(constellation, orbit, shape, att_true, sun,
                              t_grid, SensorConfig(), rng,
                              articulate=articulate)
    t_sim = time.time() - t0

    nominal = LvlhHold(orbit)
    centers, costs = window_costs(obs, shape, nominal, articulate)
    z = robust_z(costs)
    events = detect(centers, z)
    tc, yaw_tr, contr = window_yaw_track(obs, shape, orbit, articulate)

    out = dict(case=label, model=name, yaw_deg=yaw, n_rows=len(obs),
               t_sim=t_sim,
               centers=centers.tolist(), cost=costs.tolist(), z=z.tolist(),
               events=events,
               yaw_centers=tc.tolist(), yaw_track=yaw_tr.tolist(),
               yaw_contrast=contr.tolist(),
               z_max=float(z.max()) if len(z) else None)
    if yaw is not None and events:
        # seed the parametric fit from the first detection and the track
        in_ev = (tc >= events[0][0]) & (tc <= events[-1][1])
        seed_yaw = float(np.median(yaw_tr[in_ev])) if in_ev.any() else yaw
        fit = fit_slew(obs, shape, orbit, articulate,
                       seed_t0=events[0][0] - WIN_S, seed_yaw=seed_yaw)
        fit["err_t_start_s"] = fit["t_start_s"] - T_START
        fit["err_slew_s"] = fit["slew_s"] - SLEW_S
        fit["err_hold_s"] = fit["hold_s"] - HOLD_S
        # yaw ambiguity: for symmetric buses -yaw and yaw+180 twins exist
        cands = [fit["yaw_hold_deg"], -fit["yaw_hold_deg"],
                 fit["yaw_hold_deg"] + 180, fit["yaw_hold_deg"] - 180]
        fit["err_yaw_deg"] = float(min(abs(c - yaw) for c in cands))
        out["fit"] = fit
    print(f"{label:28s} rows={len(obs):6d} z_max={out['z_max'] and round(out['z_max'],1)} "
          f"events={len(events)} "
          f"fit_err_t0={out.get('fit', {}).get('err_t_start_s')}",
        flush=True)
    return out


def main() -> None:
    out_dir = Path("results/slew")
    out_dir.mkdir(parents=True, exist_ok=True)
    lib, _ = full_library()
    rows = []
    for name, yaw in CASES:
        rows.append(run_case(lib, name, yaw))
    for name in CONTROLS:
        rows.append(run_case(lib, name, None))
    with open(out_dir / "summary.json", "w") as f:
        json.dump(dict(t_start_s=T_START, slew_s=SLEW_S, hold_s=HOLD_S,
                       win_s=WIN_S, stride_s=STRIDE_S, z_thresh=Z_THRESH,
                       duration_s=DURATION_S, dt_s=DT_S, rows=rows), f)
    print("wrote", out_dir / "summary.json")


if __name__ == "__main__":
    main()
