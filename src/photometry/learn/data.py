"""Training-data generator for amortized spin-state inference.

The expensive part of simulating this sensor is the *geometry* — which
of 30,000 trackers sees the target, when, from where — and that depends
on orbits and the constellation, not on the target's attitude or shape
(except through detection gating, which is re-applied here). So the
generator reuses the (t, sun, line-of-sight, range) rows of existing
full-fleet runs as a geometry pool and re-renders brightness under fresh
random truth with the same forward model the simulator uses. A training
example costs milliseconds instead of minutes, and the label is exact.

Each example is a *set* of detection tokens from one window:

  token = [t_norm, sun_eci(3), u_obs_eci(3), mag_rel, censored]

with mag_rel the range-normalized magnitude minus the window median (so
absolute albedo/size is deliberately hidden — the net must read spin
from modulation and geometry, like the periodogram does). Labels:
spin pole (axial), log period, body spin axis class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..attitude import PrincipalAxisSpin
from ..frames import unit, unit_to_radec
from ..measurements import ObservationSet
from ..radiometry import apparent_magnitude, mag_to_normalized_brightness
from ..shapes import FacetModel

N_FEATURES = 9
AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


@dataclass
class GeometryPool:
    """Pooled detection geometry from full-fleet runs (attitude-agnostic)."""

    t_s: np.ndarray
    sun_eci: np.ndarray
    los_eci: np.ndarray
    range_km: np.ndarray
    source: np.ndarray        # (K,) run index each row came from
    spans: list[tuple[float, float]] = field(default_factory=list)

    @classmethod
    def from_runs(cls, paths: list[Path]) -> "GeometryPool":
        parts, spans = [], []
        for i, p in enumerate(paths):
            o = ObservationSet.from_npz(p)
            parts.append((o.t_s, o.sun_eci, o.los_eci, o.range_km,
                          np.full(len(o), i)))
            spans.append((float(o.t_s.min()), float(o.t_s.max())))
        return cls(*[np.concatenate(x) for x in zip(*parts)], spans=spans)

    def window(self, rng: np.random.Generator, width_s: float,
               min_rows: int = 64) -> np.ndarray:
        """Row indices of one random window from one random run."""
        for _ in range(50):
            i = rng.integers(len(self.spans))
            lo, hi = self.spans[i]
            if hi - lo < width_s:
                continue
            t0 = rng.uniform(lo, hi - width_s)
            sel = np.nonzero((self.source == i) & (self.t_s >= t0)
                             & (self.t_s < t0 + width_s))[0]
            if len(sel) >= min_rows:
                return sel
        raise RuntimeError("no window with enough rows")


@dataclass
class SpinSample:
    tokens: np.ndarray       # (N, 9)
    mask: np.ndarray         # (N,) True = real token
    pole: np.ndarray         # (3,)
    log_period: float
    axis_idx: int
    period_s: float
    phase_rad: float


def random_spin(rng: np.random.Generator,
                period_range=(40.0, 600.0)) -> PrincipalAxisSpin:
    pole = unit(rng.normal(size=3))
    ra, dec = unit_to_radec(pole)
    period = float(np.exp(rng.uniform(*np.log(period_range))))
    axis = AXES[rng.integers(3)]
    return PrincipalAxisSpin(float(ra), float(dec), period,
                             float(rng.uniform(0, 2 * np.pi)), body_axis=axis)


def render_window(pool: GeometryPool, sel: np.ndarray, shape: FacetModel,
                  att, rng: np.random.Generator, limiting_mag=7.5,
                  saturation_mag=-1.0, noise_sigma=0.08,
                  albedo_offset_sigma=0.3):
    """Forward-model brightness for pooled geometry rows under `att`.

    Returns (kept row indices into sel, mag, censored) after detection
    gating; the shape is rendered with arrays frozen (tumbling convention).
    """
    t = pool.t_s[sel]
    sun_b = att.eci_to_body(t, pool.sun_eci[sel])
    obs_b = att.eci_to_body(t, -pool.los_eci[sel])
    mag = apparent_magnitude(shape, sun_b, obs_b, pool.range_km[sel])
    mag = mag + rng.normal(0.0, albedo_offset_sigma)  # unknown albedo scale
    detectable = mag < limiting_mag
    cens = mag <= saturation_mag
    mag = np.where(cens, saturation_mag, mag + rng.normal(0.0, noise_sigma, len(mag)))
    keep = np.nonzero(detectable)[0]
    return keep, mag[keep], cens[keep]


def tokens_from_rows(t: np.ndarray, sun: np.ndarray, u_obs: np.ndarray,
                     mag: np.ndarray, range_km: np.ndarray, cens: np.ndarray,
                     t0: float, width_s: float, n_tokens: int,
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Build a padded (n_tokens, 9) token array + mask from detection rows."""
    if len(t) > n_tokens:
        pick = np.sort(rng.choice(len(t), n_tokens, replace=False))
        t, sun, u_obs, mag, range_km, cens = (a[pick] for a in
                                              (t, sun, u_obs, mag, range_km, cens))
    b = np.clip(mag_to_normalized_brightness(mag, range_km), 1e-9, None)
    m_n = -2.5 * np.log10(b)
    ok = ~cens.astype(bool)
    med = np.median(m_n[ok]) if ok.any() else np.median(m_n)
    feats = np.column_stack([
        (t - t0) / width_s, sun, u_obs, (m_n - med) / 2.0, cens.astype(float)])
    out = np.zeros((n_tokens, N_FEATURES), dtype=np.float32)
    mask = np.zeros(n_tokens, dtype=bool)
    out[:len(feats)] = feats
    mask[:len(feats)] = True
    return out, mask


def sample_example(pool: GeometryPool, shapes: list[FacetModel],
                   rng: np.random.Generator, width_s: float = 7200.0,
                   n_tokens: int = 256, min_rows: int = 48) -> SpinSample:
    for _ in range(50):
        sel = pool.window(rng, width_s)
        shape = shapes[rng.integers(len(shapes))]
        att = random_spin(rng)
        keep, mag, cens = render_window(pool, sel, shape, att, rng)
        if len(keep) < min_rows:
            continue
        rows = sel[keep]
        t0 = float(pool.t_s[sel].min())
        tok, mask = tokens_from_rows(pool.t_s[rows], pool.sun_eci[rows],
                                     -pool.los_eci[rows], mag,
                                     pool.range_km[rows], cens, t0, width_s,
                                     n_tokens, rng)
        return SpinSample(tok, mask, att.pole.astype(np.float32),
                          float(np.log(att.period_s)), AXES.index(att.body_axis),
                          att.period_s, att.phase_rad)
    raise RuntimeError("could not draw a detectable example")


def tokens_from_obs(obs: ObservationSet, t0: float, width_s: float,
                    n_tokens: int, rng: np.random.Generator):
    """Tokens for a real/simulated ObservationSet window (eval path)."""
    sel = np.nonzero((obs.t_s >= t0) & (obs.t_s < t0 + width_s))[0]
    o = obs.subset(sel)
    return tokens_from_rows(o.t_s, o.sun_eci, o.u_obs_from_target(), o.mag,
                            o.range_km, o.censored, t0, width_s, n_tokens, rng)


def sample_batch(pool, shapes, rng, batch: int, **kw):
    ex = [sample_example(pool, shapes, rng, **kw) for _ in range(batch)]
    return (np.stack([e.tokens for e in ex]), np.stack([e.mask for e in ex]),
            np.stack([e.pole for e in ex]),
            np.array([e.log_period for e in ex], dtype=np.float32),
            np.array([e.axis_idx for e in ex]))
