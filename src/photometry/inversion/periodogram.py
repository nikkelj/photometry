"""Spin-period recovery from the aggregated brightness stream."""

from __future__ import annotations

import numpy as np
from scipy.signal import lombscargle

from ..measurements import ObservationSet


def brightness_periodogram(
    obs: ObservationSet,
    period_range_s: tuple[float, float] = (10.0, 600.0),
    n_periods: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """Lomb–Scargle periodogram of the range-normalized log-brightness.

    With observers spread all around the target the sampled brightness is
    a superposition of a slow geometry trend and the fast spin signature;
    the spin period (or half of it, for two-fold symmetric shapes) shows up
    directly. Returns (periods_s, normalized_power).
    """
    t = obs.t_s
    y = -2.5 * np.log10(np.clip(obs.normalized_brightness(), 1e-6, None))
    y = y - np.mean(y)
    periods = np.linspace(period_range_s[0], period_range_s[1], n_periods)
    omega = 2 * np.pi / periods
    power = lombscargle(t, y, omega, normalize=True)
    return periods, power


def best_period(periods: np.ndarray, power: np.ndarray) -> float:
    return float(periods[np.argmax(power)])
