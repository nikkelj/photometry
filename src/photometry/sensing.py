"""Opportunistic detection of a target by the constellation's star trackers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constellation import WalkerConstellation, boresights_eci, tracker_boresights_lvlh
from .frames import in_earth_shadow, lvlh_basis, unit
from .measurements import ObservationSet
from .radiometry import apparent_magnitude
from .shapes import FacetModel

# ~200 km airglow limb as seen from the 550 km shell (see layer_limb_elevation_deg).
AIRGLOW_KEEP_OUT_DEG = -18.0


@dataclass
class HostedSSAConfig:
    """Extra SSA heads on a subset of buses. Does not recant the ADCS suite.

    The default is a 4th-head down-looker: −8° LVLH elevation, 12° half-angle,
    along-track, hosted on every `host_stride`-th bus. LOS is clipped at the
    ~200 km airglow limb (~−18° from 550 km), not through nadir. Mixed cants
    (e.g. a +8° search companion) are the same config with extra elevations.
    """

    fov_half_angle_deg: float = 12.0
    elevations_deg: tuple[float, ...] = (-8.0,)
    azimuths_deg: tuple[float, ...] = (0.0,)
    min_los_elevation_deg: float = AIRGLOW_KEEP_OUT_DEG
    host_stride: int = 10

    def __post_init__(self) -> None:
        if self.host_stride < 1:
            raise ValueError("host_stride must be >= 1")
        n_el, n_az = len(self.elevations_deg), len(self.azimuths_deg)
        if n_el != n_az and min(n_el, n_az) != 1:
            raise ValueError("elevations_deg and azimuths_deg must match or broadcast")

    def boresights_lvlh(self) -> np.ndarray:
        return tracker_boresights_lvlh(self.elevations_deg, self.azimuths_deg)


@dataclass
class SensorConfig:
    """ADCS star-tracker suite (lost-in-space): three +5° LVLH heads, 7° FOV.

    Optional `hosted_ssa` adds extra heads on a subset of buses; it does not
    widen or recant these three.
    """

    fov_half_angle_deg: float = 7.0
    tracker_elevation_deg: float = 5.0
    tracker_azimuths_deg: tuple[float, ...] = (0.0, 120.0, 240.0)
    limiting_mag: float = 7.5
    saturation_mag: float = -1.0
    mag_noise_sigma: float = 0.08
    sun_exclusion_deg: float = 40.0
    min_los_elevation_deg: float = 1.0  # keep the line of sight above the local horizontal
    max_range_km: float = 3000.0
    hosted_ssa: HostedSSAConfig | None = None

    def adcs_boresights_lvlh(self) -> np.ndarray:
        return tracker_boresights_lvlh(self.tracker_elevation_deg, self.tracker_azimuths_deg)


def hosted_ssa_config(host_stride: int = 10, **ssa_kwargs) -> SensorConfig:
    """ADCS suite plus a hosted down-looker payload on every Nth bus."""
    return SensorConfig(hosted_ssa=HostedSSAConfig(host_stride=host_stride, **ssa_kwargs))


def heads_in_view(
    r_sat: np.ndarray,
    v_sat: np.ndarray,
    r_tgt: np.ndarray,
    b_lvlh: np.ndarray,
    fov_half_angle_deg: float,
    min_los_elevation_deg: float,
    sun_eci: np.ndarray | None = None,
    sun_exclusion_deg: float = 40.0,
) -> np.ndarray:
    """Boolean (N, T): each head sees `r_tgt` (FOV + LOS elevation + optional sun)."""
    r_sat = np.asarray(r_sat, dtype=float)
    v_sat = np.asarray(v_sat, dtype=float)
    r_tgt = np.asarray(r_tgt, dtype=float)
    if r_sat.ndim == 1:
        r_sat = r_sat[None, :]
        v_sat = v_sat[None, :]
    d = r_tgt - r_sat
    rng = np.linalg.norm(d, axis=-1, keepdims=True)
    los = np.divide(d, rng, out=np.zeros_like(d), where=rng > 0)
    return _hits_from_los(
        r_sat, v_sat, los, b_lvlh, fov_half_angle_deg, min_los_elevation_deg,
        None if sun_eci is None else unit(np.asarray(sun_eci, dtype=float)),
        sun_exclusion_deg,
    )


def _hits_from_los(
    r_sat: np.ndarray,
    v_sat: np.ndarray,
    los: np.ndarray,
    b_lvlh: np.ndarray,
    fov_half_angle_deg: float,
    min_los_elevation_deg: float,
    sun: np.ndarray | None,
    sun_exclusion_deg: float,
) -> np.ndarray:
    _, _, up = lvlh_basis(r_sat, v_sat)
    el_ok = np.sum(los * up, axis=-1) > np.sin(np.radians(min_los_elevation_deg))
    bores = boresights_eci(r_sat, v_sat, b_lvlh)
    in_fov = np.einsum("ntj,nj->nt", bores, los) > np.cos(np.radians(fov_half_angle_deg))
    hits = in_fov & el_ok[:, None]
    if sun is not None:
        hits = hits & ((bores @ sun) < np.cos(np.radians(sun_exclusion_deg)))
    return hits


def simulate_detections(
    constellation: WalkerConstellation,
    target_orbit: WalkerConstellation,
    target_shape: FacetModel,
    target_attitude,
    sun_eci: np.ndarray,
    t_grid: np.ndarray,
    sensors: SensorConfig,
    rng: np.random.Generator,
    articulate: bool = False,
) -> ObservationSet:
    """Step the constellation and target through time and collect detections.

    The target orbit is passed as a single-satellite WalkerConstellation for
    convenience (circular orbit, arbitrary plane/anomaly). `target_attitude`
    is any model exposing eci_to_body(t, v). With `articulate` True, gimbaled
    facets track the sun per the shape's articulation rules.
    """
    adcs_b = sensors.adcs_boresights_lvlh()
    n_adcs = adcs_b.shape[0]
    ssa = sensors.hosted_ssa
    ssa_b = ssa.boresights_lvlh() if ssa is not None and ssa.azimuths_deg else None
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

        adcs_hits = _hits_from_los(
            r_sat[idx], v_sat[idx], los, adcs_b,
            sensors.fov_half_angle_deg, sensors.min_los_elevation_deg,
            sun, sensors.sun_exclusion_deg,
        )
        n_i, t_i = np.nonzero(adcs_hits)

        if ssa_b is not None:
            hosted = np.nonzero((idx % ssa.host_stride) == 0)[0]
            if hosted.size:
                ssa_hits = _hits_from_los(
                    r_sat[idx[hosted]], v_sat[idx[hosted]], los[hosted], ssa_b,
                    ssa.fov_half_angle_deg, ssa.min_los_elevation_deg,
                    sun, sensors.sun_exclusion_deg,
                )
                h_n, h_t = np.nonzero(ssa_hits)
                n_i = np.concatenate([n_i, hosted[h_n]])
                t_i = np.concatenate([t_i, n_adcs + h_t])

        if n_i.size == 0:
            continue

        sat_idx = idx[n_i]
        k = len(sat_idx)
        u_obs_eci = -los[n_i]  # target -> observer
        att_t = np.full(k, float(t))
        u_sun_body = target_attitude.eci_to_body(att_t, np.broadcast_to(sun, (k, 3)).copy())
        u_obs_body = target_attitude.eci_to_body(att_t, u_obs_eci)
        normals = target_shape.body_normals(u_sun_body, articulate=articulate)
        mag_true = apparent_magnitude(target_shape, u_sun_body, u_obs_body,
                                      rng_km[sat_idx], normals=normals)

        detectable = mag_true < sensors.limiting_mag
        saturated = mag_true <= sensors.saturation_mag
        for j in np.nonzero(detectable)[0]:
            # saturated streaks are still detections: the object is known
            # to be brighter than the cap, so record a censored row with
            # mag = cap rather than dropping the event
            cens = bool(saturated[j])
            mag = (sensors.saturation_mag if cens
                   else float(mag_true[j] + rng.normal(0, sensors.mag_noise_sigma)))
            rows.append(
                dict(
                    t_s=float(t),
                    obs_id=int(sat_idx[j]),
                    tracker_id=int(t_i[j]),
                    # copy: row views would pin every per-step state array
                    # in memory for the whole run
                    obs_pos_km=r_sat[sat_idx[j]].copy(),
                    los_eci=los[n_i[j]].copy(),
                    sun_eci=sun,
                    range_km=float(rng_km[sat_idx[j]]),
                    mag=mag,
                    mag_sigma=0.3 if cens else sensors.mag_noise_sigma,
                    sensor_bias=0.0,
                    censored=int(cens),
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
        censored=col("censored").astype(int),
    )
