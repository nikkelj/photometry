"""Opportunistic detection of a target by the constellation's star trackers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attitude import PrincipalAxisSpin
from .constellation import WalkerConstellation, boresights_eci, tracker_boresights_lvlh
from .frames import in_earth_shadow, lvlh_basis, unit
from .measurements import ObservationSet
from .radiometry import apparent_magnitude
from .shapes import FacetModel


@dataclass
class SensorConfig:
    fov_half_angle_deg: float = 7.0
    tracker_elevation_deg: float = 5.0
    tracker_azimuths_deg: tuple[float, ...] = (0.0, 120.0, 240.0)
    limiting_mag: float = 7.5
    saturation_mag: float = -1.0
    mag_noise_sigma: float = 0.08
    sun_exclusion_deg: float = 40.0
    min_los_elevation_deg: float = 1.0  # keep the line of sight above the Earth limb
    max_range_km: float = 3000.0


def simulate_detections(
    constellation: WalkerConstellation,
    target_orbit: WalkerConstellation,
    target_shape: FacetModel,
    target_attitude: PrincipalAxisSpin,
    sun_eci: np.ndarray,
    t_grid: np.ndarray,
    sensors: SensorConfig,
    rng: np.random.Generator,
) -> ObservationSet:
    """Step the constellation and target through time and collect detections.

    The target orbit is passed as a single-satellite WalkerConstellation for
    convenience (circular orbit, arbitrary plane/anomaly).
    """
    b_lvlh = tracker_boresights_lvlh(sensors.tracker_elevation_deg, sensors.tracker_azimuths_deg)
    cos_fov = np.cos(np.radians(sensors.fov_half_angle_deg))
    cos_sun_excl = np.cos(np.radians(sensors.sun_exclusion_deg))
    sin_min_el = np.sin(np.radians(sensors.min_los_elevation_deg))
    sun = unit(np.asarray(sun_eci, dtype=float))

    rows: list[dict] = []
    for t in t_grid:
        r_tgt, _ = target_orbit.states(float(t))
        r_tgt = r_tgt[0]
        if in_earth_shadow(r_tgt[None, :], sun)[0]:
            continue

        r_sat, v_sat = constellation.states(float(t))
        d_vec = r_tgt[None, :] - r_sat  # (N,3)
        rng_km = np.linalg.norm(d_vec, axis=-1)
        near = (rng_km < sensors.max_range_km) & (rng_km > 1.0)
        if not near.any():
            continue
        idx = np.nonzero(near)[0]
        los = d_vec[idx] / rng_km[idx, None]

        # LOS must clear the Earth limb (elevation above local horizontal)
        _, _, up = lvlh_basis(r_sat[idx], v_sat[idx])
        el_ok = np.sum(los * up, axis=-1) > sin_min_el

        bores = boresights_eci(r_sat[idx], v_sat[idx], b_lvlh)  # (n,T,3)
        in_fov = np.einsum("ntj,nj->nt", bores, los) > cos_fov
        sun_ok = (bores @ sun) < cos_sun_excl
        hits = in_fov & sun_ok & el_ok[:, None]
        if not hits.any():
            continue

        n_i, t_i = np.nonzero(hits)
        sat_idx = idx[n_i]
        k = len(sat_idx)
        u_obs_eci = -los[n_i]  # target -> observer
        att_t = np.full(k, float(t))
        u_sun_body = target_attitude.eci_to_body(att_t, np.broadcast_to(sun, (k, 3)).copy())
        u_obs_body = target_attitude.eci_to_body(att_t, u_obs_eci)
        mag_true = apparent_magnitude(target_shape, u_sun_body, u_obs_body, rng_km[sat_idx])

        detectable = (mag_true < sensors.limiting_mag) & (mag_true > sensors.saturation_mag)
        for j in np.nonzero(detectable)[0]:
            rows.append(
                dict(
                    t_s=float(t),
                    obs_id=int(sat_idx[j]),
                    tracker_id=int(t_i[j]),
                    obs_pos_km=r_sat[sat_idx[j]],
                    los_eci=los[n_i[j]],
                    sun_eci=sun,
                    range_km=float(rng_km[sat_idx[j]]),
                    mag=float(mag_true[j] + rng.normal(0, sensors.mag_noise_sigma)),
                    mag_sigma=sensors.mag_noise_sigma,
                    sensor_bias=0.0,
                )
            )

    if not rows:
        raise RuntimeError("no detections — adjust geometry or sensor limits")

    def col(name):
        return np.array([r[name] for r in rows])

    return ObservationSet(
        t_s=col("t_s"),
        obs_id=col("obs_id").astype(int),
        tracker_id=col("tracker_id").astype(int),
        obs_pos_km=np.stack([r["obs_pos_km"] for r in rows]),
        los_eci=np.stack([r["los_eci"] for r in rows]),
        sun_eci=np.stack([r["sun_eci"] for r in rows]),
        range_km=col("range_km"),
        mag=col("mag"),
        mag_sigma=col("mag_sigma"),
        sensor_bias=col("sensor_bias"),
    )
