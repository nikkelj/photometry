"""Cheap-feature library prefilter for catalog-scale identification.

Fitting every library model's attitude against every unknown target does
not scale: the expensive stage costs minutes per (target, model). The
operational architecture is therefore funnel-shaped — a feature gate that
prunes a few hundred candidates to a shortlist in milliseconds, then the
full photometric fit on the shortlist only.

Features are chosen to be attitude-free on both sides:

  size:       the median range-normalized brightness over the arc vs the
              model's median Lambertian brightness over random sun/observer
              geometry. Range is known, so this is absolute — the same
              lever that separates v2-mini from DTC in the curated study.
  amplitude:  the spread (p90/p10) of brightness — a flat-plate sat can
              swing magnitudes while a closed box cannot, whatever the
              attitude. Weakly weighted: an unlucky attitude can suppress
              a capable model's amplitude, so this feature only nudges.

Censored (saturated) rows enter the observed size feature at the cap, so
for heavily censored targets the observed size is a lower bound; the size
mismatch is scored one-sided in that regime (models fainter than the
bound are penalized, brighter ones only mildly).
"""

from __future__ import annotations

import numpy as np

from ..frames import fibonacci_sphere, unit
from ..measurements import ObservationSet
from ..radiometry import facet_brightness
from ..shapes import FacetModel


def model_brightness_samples(shape: FacetModel, n: int = 96,
                             seed: int = 5,
                             phase_range_deg: tuple[float, float] = (40.0, 100.0),
                             ) -> np.ndarray:
    """Modeled range-normalized brightness (m^2) over random geometries.

    Random body orientations with sun/observer separated by a phase angle
    drawn from the operating range. Articulated facets are evaluated
    sun-tracking (their usual state; frozen arrays only lower brightness
    toward the diffuse floor, which the amplitude feature absorbs).
    """
    rng = np.random.default_rng(seed)

    def phase_partner(u, alpha_deg):
        """Rotate each direction by the phase angle about a random axis."""
        alpha = np.radians(alpha_deg)
        helper = rng.normal(size=u.shape)
        axis = unit(np.cross(u, helper))
        c, s = np.cos(alpha)[:, None], np.sin(alpha)[:, None]
        return (u * c + np.cross(axis, u) * s
                + axis * np.sum(axis * u, axis=-1, keepdims=True) * (1 - c))

    alpha = rng.uniform(*phase_range_deg, size=n)
    sphere = fibonacci_sphere(n)
    # three viewing regimes matching the attitude-hypothesis families the
    # matcher entertains — a nadir-locked flat sat shows observers a very
    # different face mix than the same sat tumbling:
    #   tumble:    both directions random over the sphere
    #   lvlh ops:  observers concentrated near the body x-y equator — the
    #              5-deg-canted trackers view co-altitude targets nearly
    #              horizontally (the visibility-envelope geometry), so an
    #              LVLH-held target is seen edge-on, not from below
    #   sun-point: sun held near body +z (safe/array-sun modes)
    equator_obs = unit(sphere - 0.75 * sphere[:, 2:3] * np.array([0.0, 0.0, 1.0]))
    sun_near_z = unit(np.array([0.0, 0.0, 1.0]) + 0.5 * sphere)
    regimes = [
        (sphere, phase_partner(sphere, alpha)),
        (phase_partner(equator_obs, alpha), equator_obs),
        (sun_near_z, phase_partner(sun_near_z, alpha)),
    ]
    out = []
    for art in ((True, False) if shape.articulated else (False,)):
        for u_sun, u_obs in regimes:
            normals = shape.body_normals(u_sun, articulate=art)
            out.append(facet_brightness(shape, u_sun, u_obs, normals).sum(axis=0))
    return np.stack(out)


def model_features(shape: FacetModel,
                   min_detectable_m2: float = 0.05) -> dict:
    """Feature *intervals* across the model's array-control states.

    A model doesn't have one apparent size — arrays tracking vs frozen (or
    an unknown attitude favoring/hiding the wing) moves it. The interval
    [lo, hi] over control states absorbs that, and the prefilter scores
    distance-outside-interval rather than distance-to-point.

    min_detectable_m2 emulates the sensor's limiting magnitude at typical
    range (~mag 7.5 at ~1500 km ≈ 0.05 m^2): observed rows only exist when
    the target was bright enough to detect, so the observed median is
    Malmquist-biased bright, and the model statistic must be computed over
    the same truncated population.
    """
    samples = np.clip(model_brightness_samples(shape), 1e-6, None)
    sizes_lo, sizes_hi, amps = [], [], []
    for row in samples:
        det = row[row > min_detectable_m2]
        if len(det) < max(8, 0.2 * len(row)):
            det = row
        med = np.log10(np.median(det))
        sizes_lo.append(med)
        sizes_hi.append(med)
        amps.append(np.log10(np.percentile(det, 90) / np.percentile(det, 10)))
    # tumble-regime brightness quantiles (rows 0, and 3 if articulated:
    # the first regime of each control state) — the attitude-free
    # "photometric fingerprint" used to rank candidates for tumbling
    # targets, where coarse attitude fitting is uninformative
    q_grid = np.arange(5, 96, 10)
    n_states = samples.shape[0] // 3
    tq = []
    for s in range(n_states):
        row = np.clip(samples[3 * s], 1e-6, None)
        det = row[row > min_detectable_m2]
        if len(det) < max(8, 0.2 * len(row)):
            det = row
        tq.append([float(v) for v in np.log10(np.percentile(det, q_grid))])
    return dict(
        log_size_lo=float(min(sizes_lo)), log_size_hi=float(max(sizes_hi)),
        log_amp_lo=float(min(amps)), log_amp_hi=float(max(amps)),
        # per-(control state x viewing regime) medians: each is the size
        # anchor of one concrete hypothesis mode, kept for tie-breaking
        size_anchors=[float(s) for s in sizes_lo],
        tumble_quantiles=tq,
    )


def obs_features(obs: ObservationSet) -> dict:
    b = np.clip(obs.normalized_brightness(), 1e-6, None)
    cens_frac = float(np.mean(obs.censored))
    return dict(
        log_size=float(np.log10(np.median(b))),
        log_amp=float(np.log10(np.percentile(b, 90) / np.percentile(b, 10))),
        censored_frac=cens_frac,
    )


def named_fit_score(
    obs: ObservationSet,
    shape: FacetModel,
    orbit,
    sun: np.ndarray,
    offset_sigma: float = 0.5,
) -> float:
    """Best cost over the named (stable) attitude hypotheses only — a
    handful of forward evaluations, ~30 ms per model at 300 rows. This is
    the shortlist channel for stably-pointed targets; it is deliberately
    NOT asked about tumbles, where a coarse un-refined spin grid ranks by
    noise (measured: truth ranks of 25-100 on tumbling validation cases).
    """
    from ..attitude import FixedInertial, LvlhHold
    from .cost import huber_mag_cost, prepare_meas

    prep = prepare_meas(obs)
    art_opts = (True, False) if shape.articulated else (False,)
    best = np.inf
    named = [LvlhHold(orbit), LvlhHold(orbit, roll_deg=90.0),
             FixedInertial.z_toward(sun)]
    for att in named:
        for art in art_opts:
            best = min(best, huber_mag_cost(shape, att, art, prep, offset_sigma))
    return float(best)


def quantile_distance(obs: ObservationSet, fm: dict) -> float:
    """Attitude-free brightness-distribution distance to a model's tumble
    fingerprint: mean |Δ| of log-brightness deciles vs the model's
    precomputed tumble-regime quantiles (best over control states). Ranks
    tumbling candidates without any attitude search."""
    b = np.clip(obs.uncensored().normalized_brightness(), 1e-6, None)
    if len(b) < 20:
        return np.inf
    q = np.log10(np.percentile(b, np.arange(5, 96, 10)))
    return float(min(np.mean(np.abs(q - np.asarray(mq)))
                     for mq in fm["tumble_quantiles"]))


def shortlist_library(
    obs: ObservationSet,
    library: dict,
    orbit,
    sun: np.ndarray,
    k: int = 15,
    feature_k: int = 100,
    max_obs: int = 300,
    seed: int = 0,
    feature_cache: dict | None = None,
) -> tuple[list[str], dict]:
    """Funnel: feature gate (catalog -> feature_k), then a two-channel
    union pick (feature_k -> k):

      channel A  named-hypothesis photometric fit — catches stably
                 pointed targets with real physics (truth at rank 1 on
                 every stable validation case)
      channel B  tumble-fingerprint quantile distance — catches tumbling
                 targets, where cheap attitude fitting is uninformative;
                 for heavily censored targets (quantiles meaningless) the
                 feature-gate order stands in

    The shortlist alternates A/B picks so either channel alone being right
    puts truth in the top ~2k/2.
    """
    names, fo = prefilter_library(obs, library, k=feature_k,
                                  feature_cache=feature_cache)
    rng = np.random.default_rng(seed)
    sub = obs.uncensored()
    if len(sub) > max_obs:
        sub = sub.subset(np.sort(rng.choice(len(sub), max_obs, replace=False)))
    cache = feature_cache if feature_cache is not None else {}

    a_rank = sorted((named_fit_score(sub, library[n](), orbit, sun), n)
                    for n in names)
    if fo["censored_frac"] > 0.3:
        b_rank = [(float(i), n) for i, n in enumerate(names)]
    else:
        b_rank = sorted((quantile_distance(obs, cache[n]), n) for n in names)

    picked: list[str] = []
    ia = ib = 0
    while len(picked) < min(k, len(names)):
        if len(picked) % 2 == 0:
            while a_rank[ia][1] in picked:
                ia += 1
            picked.append(a_rank[ia][1])
        else:
            while b_rank[ib][1] in picked:
                ib += 1
            picked.append(b_rank[ib][1])
    return picked, fo


def prefilter_library(
    obs: ObservationSet,
    library: dict,
    k: int = 15,
    amp_weight: float = 0.35,
    size_tol: float = 0.10,
    amp_tol: float = 0.40,
    feature_cache: dict | None = None,
) -> tuple[list[str], dict]:
    """Rank library models by feature distance; return the top-k names.

    feature_cache maps name -> model_features(shape) and is filled on
    first use so repeated targets don't rebuild every model.
    """
    fo = obs_features(obs)
    cache = feature_cache if feature_cache is not None else {}
    scores = []
    censored_heavy = fo["censored_frac"] > 0.3
    for name, build in library.items():
        if name not in cache:
            cache[name] = model_features(build())
        fm = cache[name]
        # size: distance outside the model's [lo, hi] control-state band
        below = fo["log_size"] - fm["log_size_hi"]  # obs brighter than model
        above = fm["log_size_lo"] - fo["log_size"]  # obs fainter than model
        if censored_heavy:
            # observed size is a lower bound: model too faint disqualifies,
            # model brighter than the bound is expected
            size_pen = max(below - size_tol, 0.0) + 0.3 * max(above - size_tol, 0.0)
        else:
            size_pen = max(below - size_tol, 0.0, above - size_tol)
        # amplitude: one-sided — a stable attitude suppresses any model's
        # modulation, but a target swinging MORE than the model ever can
        # is disqualifying. Meaningless under heavy censoring (caps
        # compress the swing).
        if censored_heavy:
            amp_pen = 0.0
        else:
            amp_pen = max(fo["log_amp"] - fm["log_amp_hi"] - amp_tol, 0.0)
        # soft tie-break among in-tolerance models: distance to the nearest
        # hypothesis-mode anchor (a tumbling target should compare against
        # the model's tumble median, a nadir-locked one against its
        # edge-view median), weighted low so it never overrides a hard
        # mismatch
        soft = 0.15 * min(abs(a - fo["log_size"]) for a in fm["size_anchors"])
        scores.append((size_pen + amp_weight * amp_pen + soft, name))
    scores.sort()
    return [n for _, n in scores[:k]], fo
