"""Glint detection and Wahba-problem attitude waypoints.

Specular glints are the one moment photometry yields a *direction* instead
of a scalar: at a glint, the phase-angle bisector (known in ECI from that
row's sun/observer geometry) coincides with a facet normal (known in the
body frame from the library model). Each confirmed glint is therefore a
matched vector pair — exactly the input to Wahba's problem — and a window
of pairs solved with Davenport's q-method gives an absolute attitude
waypoint, the anchor the scalar light-curve fits lack.

Detection is layered, cheapest and most assumption-free first:
  1. cross-sectional outlier test against the fleet baseline. With a spin
     period in hand (Tier 0 always runs first) the baseline is conditioned
     on rotational phase — brightness is compared against other observers
     at the *same spin phase*, so the diffuse spin modulation cancels and
     only the observer-specific specular spike stands out. Without a
     period (stable pointed targets) plain time bins serve.
  2. censored rows: saturation in this data *is* mostly glinting; every
     censored row is a candidate carrying a brightness lower bound.
  3. phase-fold repeatability: true glints recur at the same rotational
     phase for a slowly-moving observer; cosmic rays and blends don't.
  4. geometric gate + correspondence: under an attitude hypothesis the
     candidate's body-frame bisector must sit within the specular lobe of
     an actual facet normal — passing assigns the Wahba correspondence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frames import unit
from ..measurements import ObservationSet
from ..shapes import FacetModel


@dataclass
class GlintSet:
    """Candidate glint rows with detection metadata."""

    idx: np.ndarray            # (G,) row indices into the source ObservationSet
    excess_mag: np.ndarray     # (G,) brightness above the fleet median (mag);
    #                            censored rows carry a lower bound
    from_censored: np.ndarray  # (G,) bool
    pab_eci: np.ndarray        # (G,3) phase-angle bisector, ECI


def detect_glints(
    obs: ObservationSet,
    period_s: float | None = None,
    n_phase_bins: int = 32,
    chunk_s: float = 7200.0,
    bin_s: float = 120.0,
    k_mad: float = 2.5,
    floor_mag: float = 1.5,
    include_censored: bool = True,
) -> GlintSet:
    """Layer 1+2: cross-sectional MAD outlier test + censored candidates.

    With `period_s`, rows are binned by (rotational phase, coarse time
    chunk): the phase conditioning makes the baseline the fleet's median
    brightness *at that spin phase*, so the diffuse spin modulation — which
    would otherwise inflate the spread by magnitudes — cancels, and the
    time chunking lets the illumination geometry drift. Without a period,
    plain `bin_s` time bins are used (valid for pointed, slowly-modulating
    targets). Within each bin the median of the *calibrated* rows
    estimates the common-mode brightness and the MAD its cross-observer
    spread; calibrated rows brighter than the median by
    max(floor_mag, k_mad * 1.4826 * MAD) are flagged. Censored rows are
    candidates by construction (their excess vs the median is a lower
    bound). Bins with too few calibrated rows for a stable baseline are
    skipped for calibrated flagging but still admit censored rows.
    """
    mag_n = -2.5 * np.log10(np.clip(obs.normalized_brightness(), 1e-9, None))
    cens = obs.censored.astype(bool)
    if period_s is not None:
        pb = ((obs.t_s / period_s) % 1.0 * n_phase_bins).astype(int)
        tb = ((obs.t_s - obs.t_s.min()) / chunk_s).astype(int)
        bins = pb * (tb.max() + 1) + tb
    else:
        bins = np.floor((obs.t_s - obs.t_s.min()) / bin_s).astype(int)

    flag = np.zeros(len(obs), dtype=bool)
    excess = np.zeros(len(obs))
    for b in np.unique(bins):
        in_bin = bins == b
        cal = in_bin & ~cens
        if cal.sum() >= 5:
            med = np.median(mag_n[cal])
            mad = np.median(np.abs(mag_n[cal] - med))
            thr = max(floor_mag, k_mad * 1.4826 * mad)
            hit = cal & (mag_n < med - thr)  # brighter = smaller magnitude
            flag |= hit
            excess[in_bin] = med - mag_n[in_bin]
        elif cal.sum() >= 1:
            excess[in_bin] = np.median(mag_n[cal]) - mag_n[in_bin]
        if include_censored:
            flag |= in_bin & cens
    idx = np.nonzero(flag)[0]
    pab = unit(obs.sun_eci[idx] + obs.u_obs_from_target()[idx])
    return GlintSet(idx=idx, excess_mag=excess[idx],
                    from_censored=cens[idx], pab_eci=pab)


def phase_fold_filter(
    obs: ObservationSet,
    glints: GlintSet,
    period_s: float,
    tol_cycles: float = 0.06,
    min_repeats: int = 2,
) -> GlintSet:
    """Layer 3: keep candidates that recur at a consistent rotational phase.

    Folds each observer's candidate times at the period and keeps those
    with at least `min_repeats - 1` other candidates from the same observer
    within `tol_cycles` (circular). One-off events — cosmic rays, blends,
    single noise excursions — fail; a genuine glint repeats every
    revolution for as long as that observer's bisector geometry lasts.

    Only meaningful when candidates are dense enough that a real glint
    *would* repeat within an observer's arc — at fleet scale (short dt,
    many revolutions per pass) it is the false-positive killer, but in a
    sparse regime it rejects everything. Callers should check
    `fold_filter_applicable` first and skip the layer when it returns
    False (the geometric gate then carries the outlier rejection).
    """
    phase = (obs.t_s[glints.idx] / period_s) % 1.0
    who = obs.obs_id[glints.idx]
    keep = np.zeros(len(glints.idx), dtype=bool)
    for o in np.unique(who):
        sel = np.nonzero(who == o)[0]
        if len(sel) < min_repeats:
            continue
        ph = phase[sel]
        d = np.abs(ph[:, None] - ph[None, :])
        d = np.minimum(d, 1.0 - d)
        near = (d < tol_cycles).sum(axis=1) - 1  # exclude self
        keep[sel] = near >= (min_repeats - 1)
    sub = np.nonzero(keep)[0]
    return GlintSet(idx=glints.idx[sub], excess_mag=glints.excess_mag[sub],
                    from_censored=glints.from_censored[sub],
                    pab_eci=glints.pab_eci[sub])


def fold_filter_applicable(obs: ObservationSet, glints: GlintSet,
                           min_repeats: int = 2) -> bool:
    """Whether candidates are dense enough per observer for phase folding."""
    if len(glints.idx) == 0:
        return False
    who = obs.obs_id[glints.idx]
    _, counts = np.unique(who, return_counts=True)
    return float(np.median(counts)) >= min_repeats


def specular_facets(shape: FacetModel, min_n_ph: float = 80.0) -> np.ndarray:
    """Indices of facets glossy enough to produce a distinct glint lobe."""
    return np.nonzero(shape.n_ph >= min_n_ph)[0]


def lobe_half_width_deg(n_ph: float) -> float:
    """Half-power half-width of a cos^n Phong lobe (degrees)."""
    return float(np.degrees(np.arccos(0.5 ** (1.0 / max(n_ph, 1.0)))))


@dataclass
class WahbaPairs:
    """Matched (body-normal, ECI-bisector) vector pairs with correspondences."""

    t_s: np.ndarray        # (P,)
    b_body: np.ndarray     # (P,3) facet normal at glint time, body frame
    r_eci: np.ndarray      # (P,3) phase-angle bisector, ECI
    facet: np.ndarray      # (P,) facet index in the shape model
    weight: np.ndarray     # (P,)
    row: np.ndarray        # (P,) source row index in the ObservationSet


def correspond_glints(
    obs: ObservationSet,
    glints: GlintSet,
    shape: FacetModel,
    attitude,
    articulate: bool,
    cone_deg: float = 4.0,
    min_n_ph: float = 80.0,
) -> WahbaPairs:
    """Layer 4: geometric gate + facet correspondence under a hypothesis.

    Maps each candidate's bisector into the body frame with the hypothesis
    attitude and accepts it if it lies within the gate cone of a specular
    facet normal (articulated normals evaluated at that row's sun
    geometry). The per-facet gate is max(cone_deg, 1.5x the facet's
    half-power lobe width): a matte-r antenna face glints over a wider
    bisector cone than n=800 cover glass and the gate must admit that.
    The accepted pair is (facet normal in body frame, bisector in ECI) —
    the hypothesis only *gates and assigns*; the Wahba solve then
    re-estimates attitude from the pairs alone, so a coarse hypothesis
    yields a refined absolute waypoint.
    """
    if len(glints.idx) == 0:
        return WahbaPairs(*[np.zeros((0, 3)) if i in (1, 2) else np.zeros(0)
                            for i in range(6)])
    rows = glints.idx
    t = obs.t_s[rows]
    u_sun_b = attitude.eci_to_body(t, obs.sun_eci[rows])
    pab_b = attitude.eci_to_body(t, glints.pab_eci)
    spec = specular_facets(shape, min_n_ph)
    normals = shape.body_normals(u_sun_b, articulate=articulate)  # (F,G,3)
    cos_gate = {int(f): np.cos(np.radians(max(
        cone_deg, 1.5 * lobe_half_width_deg(shape.n_ph[f])))) for f in spec}

    t_out, b_out, r_out, f_out, w_out, row_out = [], [], [], [], [], []
    for j in range(len(rows)):
        best_f, best_margin = -1, 0.0
        for f in spec:
            c = float(normals[f, j] @ pab_b[j])
            margin = c - cos_gate[int(f)]
            if margin > best_margin:
                best_f, best_margin = f, margin
        if best_f < 0:
            continue
        t_out.append(t[j])
        b_out.append(normals[best_f, j])
        r_out.append(glints.pab_eci[j])
        f_out.append(best_f)
        # censored rows: direction is exact even though amplitude is a bound
        w_out.append(1.0)
        row_out.append(rows[j])
    return WahbaPairs(
        t_s=np.array(t_out), b_body=np.array(b_out).reshape(-1, 3),
        r_eci=np.array(r_out).reshape(-1, 3), facet=np.array(f_out, dtype=int),
        weight=np.array(w_out), row=np.array(row_out, dtype=int))


def davenport_q(b_body: np.ndarray, r_eci: np.ndarray,
                weights: np.ndarray) -> np.ndarray:
    """Davenport's q-method: optimal R (body->ECI) with r ~ R b.

    Maximizes sum w_i r_i . R b_i via the largest eigenvector of the 4x4 K
    matrix. Returns the 3x3 rotation.
    """
    B = (weights[:, None, None] * r_eci[:, :, None] * b_body[:, None, :]).sum(axis=0)
    S = B + B.T
    sigma = np.trace(B)
    # sign convention verified against exact synthetic pairs: with
    # B = sum w r b^T, the z vector is [B32-B23, B13-B31, B21-B12]
    z = np.array([B[2, 1] - B[1, 2], B[0, 2] - B[2, 0], B[1, 0] - B[0, 1]])
    K = np.empty((4, 4))
    K[0, 0] = sigma
    K[0, 1:] = z
    K[1:, 0] = z
    K[1:, 1:] = S - sigma * np.eye(3)
    vals, vecs = np.linalg.eigh(K)
    q = vecs[:, np.argmax(vals)]  # (q0, q1, q2, q3), scalar first
    q0, q1, q2, q3 = q
    return np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0**2 - q1**2 - q2**2 + q3**2],
    ])


@dataclass
class Waypoint:
    t_s: float                # epoch (window center)
    r_bi: np.ndarray          # body->ECI attitude at the epoch
    n_pairs: int
    resid_deg: float          # rms angular residual of the pairs
    condition: float          # anisotropy: smallest/largest pair-spread axis


def wahba_waypoints(
    pairs: WahbaPairs,
    attitude,
    window_s: float = 400.0,
    min_pairs: int = 3,
) -> list[Waypoint]:
    """Windowed q-method attitude waypoints.

    The body rotates within a window, so pairs are propagated to the
    window-center epoch with the *hypothesis* dynamics before the solve:
    assuming R(t) ~ R(tc) [R_hyp(tc)^T R_hyp(t)], each body vector maps as
    b~ = R_hyp(tc)^T R_hyp(t) b. The q-method then recovers R(tc) from the
    pairs alone. Windows need pairs whose bisectors are not collinear —
    `condition` reports the spread anisotropy (small = one-directional
    geometry, weakly constrained about that axis; see the beta-coverage
    charts for why bisectors cluster sunward).
    """
    if len(pairs.t_s) == 0:
        return []
    order = np.argsort(pairs.t_s)
    t = pairs.t_s[order]
    out: list[Waypoint] = []
    t0, t_end = t[0], t[-1]
    n_win = max(1, int(np.ceil((t_end - t0) / window_s)))
    for k in range(n_win):
        lo, hi = t0 + k * window_s, t0 + (k + 1) * window_s
        sel = order[(t >= lo) & (t < hi)]
        if len(sel) < min_pairs:
            continue
        tc = float(np.mean(pairs.t_s[sel]))
        r_c = attitude.body_to_eci_matrix(tc)
        b_prop = np.empty((len(sel), 3))
        for i, s in enumerate(sel):
            r_t = attitude.body_to_eci_matrix(float(pairs.t_s[s]))
            b_prop[i] = r_c.T @ (r_t @ pairs.b_body[s])
        r_est = davenport_q(b_prop, pairs.r_eci[sel], pairs.weight[sel])
        pred = (r_est @ b_prop.T).T
        cosr = np.clip(np.sum(pred * pairs.r_eci[sel], axis=-1), -1, 1)
        resid = float(np.sqrt(np.mean(np.degrees(np.arccos(cosr)) ** 2)))
        # geometry conditioning from the singular values of the pair matrix
        svals = np.linalg.svd(pairs.r_eci[sel], compute_uv=False)
        cond = float(svals[-1] / svals[0]) if svals[0] > 0 else 0.0
        out.append(Waypoint(t_s=tc, r_bi=r_est, n_pairs=int(len(sel)),
                            resid_deg=resid, condition=cond))
    return out


def attitude_error_deg(r_est: np.ndarray, r_true: np.ndarray) -> float:
    """Rotation angle between two attitude matrices (degrees)."""
    c = (np.trace(r_est @ r_true.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def truth_glint_labels(obs: ObservationSet, shape: FacetModel, attitude,
                       articulate: bool,
                       min_spec_frac: float = 0.5,
                       min_n_ph: float = 80.0) -> np.ndarray:
    """Ground-truth glint mask from the simulator's own forward model:
    rows where the *narrow-lobe* specular terms (facets with Phong
    exponent >= min_n_ph — the ones that produce distinct bisector-locked
    spikes rather than broad MLI sheen) carry at least `min_spec_frac` of
    the modeled flux. Used only to score the detector, never by it."""
    from ..radiometry import facet_brightness

    u_sun = attitude.eci_to_body(obs.t_s, obs.sun_eci)
    u_obs = attitude.eci_to_body(obs.t_s, obs.u_obs_from_target())
    normals = shape.body_normals(u_sun, articulate=articulate)
    b_full = facet_brightness(shape, u_sun, u_obs, normals).sum(axis=0)
    hi = shape.n_ph >= min_n_ph
    no_hi = FacetModel(
        normals=shape.normals, areas=shape.areas, rho_d=shape.rho_d,
        k_s=np.where(hi, 0.0, shape.k_s), n_ph=shape.n_ph,
        labels=shape.labels, gimbal_mode=shape.gimbal_mode,
        gimbal_axis=shape.gimbal_axis, mirror_of=shape.mirror_of,
        name=shape.name)
    b_rest = facet_brightness(no_hi, u_sun, u_obs, normals).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        spec_frac = np.where(b_full > 0, 1.0 - b_rest / b_full, 0.0)
    return spec_frac >= min_spec_frac
