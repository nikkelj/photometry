"""Tier-3 torque-free attitude refinement.

A uniform spin cannot represent non-principal-axis rotation: the body
nutates, so brightness features drift against any constant-rate model.
This fit promotes a uniform-spin seed to full torque-free rigid-body
dynamics — parameters are the inertia ratios (I1 fixed at 1), the initial
body rate vector, and a rotation-vector perturbation of the seed's
initial attitude — optimized on an observation window with the shared
censoring-aware photometric cost.

Each evaluation integrates the quaternion + Euler-rate ODE with solve_ivp
directly at the (sorted) observation times — no fixed-step table — so an
8-parameter Nelder-Mead run stays tractable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from ..attitude import PrincipalAxisSpin, TorqueFreeTumble
from ..frames import rodrigues
from ..measurements import ObservationSet
from ..shapes import FacetModel
from .cost import huber_mag_cost, prepare_meas


@dataclass
class TorqueFreeFit:
    inertia: tuple[float, float, float]
    omega0_body: tuple[float, float, float]
    r0: np.ndarray
    t_ref: float
    cost: float
    cost_uniform_seed: float
    n_evals: int

    def attitude(self, t_max: float, dt: float = 1.0) -> TorqueFreeTumble:
        return TorqueFreeTumble(self.inertia, self.omega0_body, self.r0,
                                t_max=t_max, dt=dt, t_ref=self.t_ref)


def _quat_from_matrix(r: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    if w > 1e-8:
        return np.array([w, (r[2, 1] - r[1, 2]) / (4 * w),
                         (r[0, 2] - r[2, 0]) / (4 * w),
                         (r[1, 0] - r[0, 1]) / (4 * w)])
    # fallback for 180-degree rotations
    d = np.diag(r)
    i = int(np.argmax(d))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(1e-12, 1 + d[i] - d[j] - d[k])) / 2
    q = np.zeros(4)
    q[1 + i] = s
    q[0] = (r[k, j] - r[j, k]) / (4 * s)
    q[1 + j] = (r[j, i] + r[i, j]) / (4 * s)
    q[1 + k] = (r[k, i] + r[i, k]) / (4 * s)
    return q


def _matrices_from_quats(q: np.ndarray) -> np.ndarray:
    """(K,4) unit quaternions (body->ECI) -> (K,3,3) rotation matrices."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)], -1),
    ], axis=1)


def solve_attitude_at(inertia: np.ndarray, omega0: np.ndarray, r0: np.ndarray,
                      t_eval: np.ndarray, rtol: float = 1e-6) -> np.ndarray:
    """Integrate torque-free dynamics; return body->ECI matrices at t_eval (>=0)."""
    inv_i = 1.0 / inertia

    def rhs(_t, y):
        q, w = y[:4], y[4:]
        dq = 0.5 * np.array([
            -q[1] * w[0] - q[2] * w[1] - q[3] * w[2],
            q[0] * w[0] + q[2] * w[2] - q[3] * w[1],
            q[0] * w[1] + q[3] * w[0] - q[1] * w[2],
            q[0] * w[2] + q[1] * w[1] - q[2] * w[0],
        ])
        dw = inv_i * (-np.cross(w, inertia * w))
        return np.concatenate([dq, dw])

    y0 = np.concatenate([_quat_from_matrix(r0), omega0])
    # detections share epochs: integrate at unique times, scatter back
    tu, inv = np.unique(t_eval, return_inverse=True)
    sol = solve_ivp(rhs, (0.0, float(tu.max()) + 1e-9), y0,
                    t_eval=tu, rtol=rtol, atol=1e-10, method="DOP853")
    q = sol.y[:4].T
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return _matrices_from_quats(q)[inv]


class _RowAttitude:
    """Attitude adapter: per-observation-row rotation matrices."""

    def __init__(self, r_rows: np.ndarray):
        self.r_rows = r_rows

    def eci_to_body(self, t: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
        return np.einsum("kji,kj->ki", self.r_rows, v_eci)


def fit_torque_free(
    obs: ObservationSet,
    shape: FacetModel,
    seed_spin: PrincipalAxisSpin,
    seed_periods: list[float] | None = None,
    window_s: float = 3600.0,
    t_ref: float | None = None,
    max_obs: int = 800,
    offset_sigma: float = 0.5,
    n_finalists: int = 4,
    seed: int = 0,
) -> TorqueFreeFit:
    """Multi-start torque-free fit.

    Deeply coupled tumbles (spin and nutation periods comparable) have no
    quasi-uniform window, so a single uniform-spin seed usually sits in the
    wrong basin. Starts are built from data-driven |omega| candidates
    (light-curve periodogram peaks and their doubles) x body axes x two
    initial-attitude variants, each tilted 15 deg off-axis so nutation is
    excitable. Two phases: inertia frozen (6 params), then full 8-param
    refinement of the best starts.
    """
    rng = np.random.default_rng(seed)
    if t_ref is None:
        t_ref = float(obs.t_s.min())

    r_seed = seed_spin.body_to_eci_matrix(t_ref)
    if seed_periods is None:
        seed_periods = [seed_spin.period_s]

    # window ladder: coherence over N rotations makes the omega basin
    # ~1/N of the rate — but the first window must also carry enough rows
    # that aliases lose to the true rate, so it is sized for statistics
    # (~900 s) as well as rotations, then extended and re-refined.
    p_min = min(seed_periods)
    w0_s = max(3.0 * p_min, 900.0)
    windows = sorted({min(window_s, w) for w in
                      (w0_s, 4.0 * w0_s, window_s)})
    preps, t_rels = [], []
    for w_s, cap in zip(windows, (400, 600, max_obs)):
        in_win = (obs.t_s >= t_ref) & (obs.t_s <= t_ref + w_s)
        win = obs.subset(np.nonzero(in_win)[0])
        if len(win) > cap:
            win = win.subset(np.sort(rng.choice(len(win), cap, replace=False)))
        preps.append(prepare_meas(win))
        t_rels.append(win.t_s - t_ref)
    cost_seed = huber_mag_cost(shape, seed_spin, False, preps[-1], offset_sigma)

    evals = [0]

    def unpack(x):
        li2, li3, wx, wy, wz, rx, ry, rz = x
        rvec = np.array([rx, ry, rz])
        ang = np.linalg.norm(rvec)
        dr = rodrigues(rvec / ang, ang) if ang > 1e-12 else np.eye(3)
        inertia = np.array([1.0, np.exp(li2), np.exp(li3)])
        return inertia, np.array([wx, wy, wz]), dr @ r_seed

    def make_objective(stage: int):
        prep, t_rel = preps[stage], t_rels[stage]

        def objective(x):
            evals[0] += 1
            inertia, w0, r0 = unpack(x)
            try:
                r_rows = solve_attitude_at(inertia, w0, r0, t_rel)
            except Exception:
                return 1e9
            return huber_mag_cost(shape, _RowAttitude(r_rows), False, prep,
                                  offset_sigma)
        return objective

    # start bank: |omega| x body axis (tilted 15 deg) x initial attitude
    # x inertia guess — a frozen wrong inertia decoheres the nutation
    # within a few periods and would reject the correct rate starts
    axes = np.eye(3)
    tilt = rodrigues(np.array([0.0, 0.0, 1.0]), np.radians(15.0))
    tilt2 = rodrigues(np.array([1.0, 0.0, 0.0]), np.radians(15.0))
    r0_variants = [np.zeros(3), np.array([0.0, 0.0, np.pi / 2])]
    inertia_variants = [(1.8, 2.4), (2.6, 3.2)]
    starts = []
    for p in seed_periods:
        wm = 2 * np.pi / p
        for i, a in enumerate(axes):
            w0 = wm * ((tilt if i != 2 else tilt2) @ a)
            for rv in r0_variants:
                for i2, i3 in inertia_variants:
                    starts.append(np.array([np.log(i2), np.log(i3), *w0, *rv]))

    # stage 0 phase A on the shortest window, inertia frozen
    obj0 = make_objective(0)
    scored = []
    for x0 in starts:
        frozen = x0[:2].copy()

        def obj_a(x6):
            return obj0(np.concatenate([frozen, x6]))

        res = minimize(obj_a, x0[2:], method="Nelder-Mead",
                       options=dict(maxiter=400, xatol=1e-5, fatol=1e-7))
        scored.append((res.fun, np.concatenate([frozen, res.x])))
    scored.sort(key=lambda s: s[0])

    # stage 0 phase B: full 8 params, short window
    stage_pool = []
    for _, x0 in scored[: 2 * n_finalists]:
        res = minimize(obj0, x0, method="Nelder-Mead",
                       options=dict(maxiter=600, xatol=1e-6, fatol=1e-8))
        stage_pool.append((res.fun, res.x))
    stage_pool.sort(key=lambda s: s[0])

    # extend through the window ladder
    for stage in range(1, len(windows)):
        obj_s = make_objective(stage)
        n_keep = max(n_finalists - stage, 1)
        nxt = []
        for _, x0 in stage_pool[:n_keep + 1]:
            res = minimize(obj_s, x0, method="Nelder-Mead",
                           options=dict(maxiter=800, xatol=1e-6, fatol=1e-8))
            nxt.append((res.fun, res.x))
        nxt.sort(key=lambda s: s[0])
        stage_pool = nxt

    best = stage_pool[0]
    inertia, w0, r0 = unpack(best[1])
    return TorqueFreeFit(
        inertia=tuple(float(v) for v in inertia),
        omega0_body=tuple(float(v) for v in w0),
        r0=r0, t_ref=t_ref,
        cost=float(best[0]), cost_uniform_seed=float(cost_seed),
        n_evals=evals[0])
