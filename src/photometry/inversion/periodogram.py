"""Spin-period recovery from the aggregated brightness stream."""

from __future__ import annotations

import numpy as np
from scipy.signal import lombscargle

from ..measurements import ObservationSet


def brightness_periodogram(
    obs: ObservationSet,
    period_range_s: tuple[float, float] = (10.0, 600.0),
    n_periods: int = 4000,
    oversample: float = 4.0,
    max_obs: int = 8000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Lomb–Scargle periodogram of the range-normalized log-brightness.

    With observers spread all around the target the sampled brightness is
    a superposition of a slow geometry trend and the fast spin signature;
    the spin period (or half of it, for two-fold symmetric shapes) shows up
    directly. The frequency grid is sized to the arc length (peak width
    ~1/T_arc) so long arcs cannot step over a narrow spin peak; n_periods
    is a floor on the grid size. Returns (periods_s, normalized_power).
    """
    obs = obs.uncensored()
    t, y0 = obs.t_s, obs.normalized_brightness()
    if len(t) > max_obs:
        idx = np.random.default_rng(seed).choice(len(t), max_obs, replace=False)
        t, y0 = t[idx], y0[idx]
    y = -2.5 * np.log10(np.clip(y0, 1e-6, None))
    return _ls_grid(t, y, period_range_s, n_periods, oversample)


def _ls_grid(t: np.ndarray, y_mag: np.ndarray, period_range_s, n_periods=4000,
             oversample=4.0) -> tuple[np.ndarray, np.ndarray]:
    """Array-level Lomb-Scargle on magnitudes (mean removed, arc-sized grid)."""
    y = y_mag - np.mean(y_mag)
    t_arc = max(t.max() - t.min(), 1.0)
    f_lo, f_hi = 1.0 / period_range_s[1], 1.0 / period_range_s[0]
    n = int(np.clip((f_hi - f_lo) * oversample * t_arc, n_periods, 300_000))
    freqs = np.linspace(f_lo, f_hi, n)
    # chunk the frequency axis: scipy's lombscargle broadcasts an
    # (n_freqs, n_samples) intermediate, which is many GB for a day-long arc
    power = np.empty(n)
    step = max(1, 2_000_000 // max(len(t), 1))
    for i in range(0, n, step):
        power[i:i + step] = lombscargle(t, y, 2 * np.pi * freqs[i:i + step],
                                        normalize=True)
    return 1.0 / freqs, power


def _ls_peak(t: np.ndarray, y_mag: np.ndarray, period_range_s,
             max_rows: int = 600, n_periods: int = 800,
             oversample: float = 2.0) -> float:
    """Peak period for raw (t, magnitude) arrays — the Tier-0 statistic
    without an ObservationSet (used by the learned proposer's tokens).

    Row-capped (evenly strided) and grid-coarsened: scipy's Lomb-Scargle
    is O(rows x frequencies) with trig per pair, and at training-data
    rates (~thousands of windows) the full grid dominated the step time
    9:1. A strong spin line is found just as well from 600 rows."""
    if len(t) < 8:
        return float(np.sqrt(period_range_s[0] * period_range_s[1]))
    if len(t) > max_rows:
        pick = np.linspace(0, len(t) - 1, max_rows).astype(int)
        t, y_mag = t[pick], y_mag[pick]
    periods, power = _ls_grid(t, y_mag, period_range_s, n_periods=n_periods,
                              oversample=oversample)
    return best_period(periods, power)


def best_period(periods: np.ndarray, power: np.ndarray) -> float:
    return float(periods[np.argmax(power)])
