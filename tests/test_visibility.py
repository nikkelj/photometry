"""ADCS +5° / 7° / min_el=1° cannot see co-alt or below-shell targets."""

import numpy as np
import pytest

from photometry.attitude import PrincipalAxisSpin
from photometry.constellation import WalkerConstellation, boresights_eci, tracker_boresights_lvlh
from photometry.frames import R_EARTH, los_elevation_deg, lvlh_basis, unit
from photometry.scenarios import STUDY_ALT_KM, by_name, study_orbit
from photometry.sensing import SensorConfig, simulate_detections
from photometry.shapes import box_wing

# Catalog altitudes the library records, plus co-alt with the 550 km shell.
CATALOG_ALTS_KM = {
    "co-alt": 550.0,
    "hubble": 530.0,
    "link": 500.0,
    "iss": 420.0,
    "dtc": 360.0,
}


def _equatorial_pair(obs_alt_km: float, tgt_alt_km: float, tgt_true_anomaly_deg: float):
    """One equatorial observer at u=0 and one target at the given anomaly."""
    obs = WalkerConstellation(1, 1, obs_alt_km, 0.0)
    obs._raan = np.array([0.0])
    obs._u0 = np.array([0.0])
    tgt = WalkerConstellation(1, 1, tgt_alt_km, 0.0)
    tgt._raan = np.array([0.0])
    tgt._u0 = np.array([np.radians(tgt_true_anomaly_deg)])
    return obs, tgt


def _adcs_hits(r_sat, v_sat, r_tgt, sensors: SensorConfig) -> np.ndarray:
    """Boolean (N, T): ADCS heads that pass FOV + min LOS elevation."""
    r_sat = np.asarray(r_sat, dtype=float)
    v_sat = np.asarray(v_sat, dtype=float)
    r_tgt = np.asarray(r_tgt, dtype=float)
    if r_sat.ndim == 1:
        r_sat = r_sat[None, :]
        v_sat = v_sat[None, :]
    los = unit(r_tgt - r_sat)
    _, _, up = lvlh_basis(r_sat, v_sat)
    el_ok = np.sum(los * up, axis=-1) > np.sin(np.radians(sensors.min_los_elevation_deg))
    b = tracker_boresights_lvlh(sensors.tracker_elevation_deg, sensors.tracker_azimuths_deg)
    in_fov = np.einsum("ntj,nj->nt", boresights_eci(r_sat, v_sat, b), los) > np.cos(
        np.radians(sensors.fov_half_angle_deg)
    )
    return in_fov & el_ok[:, None]


def test_adcs_suite_is_plus5_seven_deg():
    s = SensorConfig()
    assert s.tracker_elevation_deg == 5.0
    assert s.fov_half_angle_deg == 7.0
    assert s.min_los_elevation_deg == 1.0
    assert s.tracker_azimuths_deg == (0.0, 120.0, 240.0)


def test_coalt_and_below_never_above_local_horizontal():
    rng = np.random.default_rng(0)
    r_obs = (R_EARTH + 550.0) * np.array([1.0, 0.0, 0.0])
    for name, tgt_alt in CATALOG_ALTS_KM.items():
        a = R_EARTH + tgt_alt
        theta = rng.uniform(0.02, 2.8, size=40)
        r_tgt = a * np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=-1)
        el = los_elevation_deg(np.broadcast_to(r_obs, r_tgt.shape), r_tgt)
        assert np.all(el <= 1e-12), name


def test_adcs_plus5_misses_coalt_and_below_shell():
    adcs = SensorConfig()
    # In-plane anomalies that put the LOS near the horizon belt (still el < 0).
    cases = (
        ("co-alt", 550.0, 16.0),
        ("hubble", 530.0, 14.0),
        ("link", 500.0, 12.0),
        ("iss", 420.0, 10.0),
        ("dtc", 360.0, 12.0),
    )
    for name, tgt_alt, anomaly in cases:
        obs, tgt = _equatorial_pair(550.0, tgt_alt, anomaly)
        r_o, v_o = obs.states(0.0)
        r_t, _ = tgt.states(0.0)
        assert los_elevation_deg(r_o, r_t)[0] < 0, name
        assert not _adcs_hits(r_o, v_o, r_t[0], adcs).any(), name


def test_fleet_scenario_orbit_defaults_to_catalog_altitude():
    iss = by_name("iss__ops")
    hubble = by_name("hubble__science")
    link = by_name("katalyst_link__ops")
    dtc = by_name("starlink_v2mini_dtc__ops")
    assert iss.orbit().altitude_km == 420.0
    assert hubble.orbit().altitude_km == 530.0
    assert link.orbit().altitude_km == 500.0
    assert dtc.orbit().altitude_km == 360.0
    assert iss.orbit(study_torus=True).altitude_km == STUDY_ALT_KM
    assert study_orbit().altitude_km == STUDY_ALT_KM


def test_simulate_detections_adcs_misses_below_shell():
    obs, tgt = _equatorial_pair(550.0, 420.0, 10.0)
    att = PrincipalAxisSpin(200.0, 35.0, 127.4, 0.7)
    sun = np.array([0.0, 0.0, 1.0])
    with pytest.raises(RuntimeError, match="no detections"):
        simulate_detections(
            obs, tgt, box_wing(), att, sun, np.array([0.0]), SensorConfig(),
            np.random.default_rng(0),
        )
