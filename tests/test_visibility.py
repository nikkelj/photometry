"""Visibility theorem: ADCS +5° misses co-alt / below-shell; hosted SSA does not."""

import numpy as np
import pytest

from photometry.attitude import PrincipalAxisSpin
from photometry.constellation import WalkerConstellation, tracker_boresights_lvlh
from photometry.frames import R_EARTH, layer_limb_elevation_deg, los_elevation_deg
from photometry.scenarios import STUDY_ALT_KM, by_name, study_orbit
from photometry.sensing import (
    AIRGLOW_KEEP_OUT_DEG,
    SensorConfig,
    heads_in_view,
    hosted_ssa_config,
    simulate_detections,
)
from photometry.shapes import box_wing


def _equatorial_pair(obs_alt_km: float, tgt_alt_km: float, tgt_true_anomaly_deg: float):
    """One equatorial observer at u=0 and one target at the given anomaly."""
    obs = WalkerConstellation(1, 1, obs_alt_km, 0.0)
    obs._raan = np.array([0.0])
    obs._u0 = np.array([0.0])
    tgt = WalkerConstellation(1, 1, tgt_alt_km, 0.0)
    tgt._raan = np.array([0.0])
    tgt._u0 = np.array([np.radians(tgt_true_anomaly_deg)])
    return obs, tgt


def test_coalt_and_below_never_above_local_horizontal():
    rng = np.random.default_rng(0)
    r_obs = (R_EARTH + 550.0) * np.array([1.0, 0.0, 0.0])
    for tgt_alt in (550.0, 530.0, 420.0, 360.0):
        a = R_EARTH + tgt_alt
        theta = rng.uniform(0.02, 2.8, size=40)
        r_tgt = a * np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=-1)
        el = los_elevation_deg(np.broadcast_to(r_obs, r_tgt.shape), r_tgt)
        assert np.all(el <= 1e-12), tgt_alt


def test_airglow_limb_from_550km_shell():
    el = layer_limb_elevation_deg(R_EARTH + 550.0, layer_altitude_km=200.0)
    assert el == pytest.approx(-18.3, abs=0.3)
    assert AIRGLOW_KEEP_OUT_DEG == pytest.approx(el, abs=0.5)


def test_adcs_plus5_misses_coalt_and_below_shell():
    adcs = SensorConfig()
    b = adcs.adcs_boresights_lvlh()
    cases = (
        (550.0, 16.0),  # co-alt, LOS elevation = −θ/2 = −8°
        (420.0, 10.0),  # ISS-like, LOS elevation ≈ −11°
        (360.0, 12.0),  # DTC-like
    )
    for tgt_alt, anomaly in cases:
        obs, tgt = _equatorial_pair(550.0, tgt_alt, anomaly)
        r_o, v_o = obs.states(0.0)
        r_t, _ = tgt.states(0.0)
        assert los_elevation_deg(r_o, r_t)[0] < 0
        hits = heads_in_view(
            r_o, v_o, r_t[0], b, adcs.fov_half_angle_deg, adcs.min_los_elevation_deg,
        )
        assert not hits.any(), (tgt_alt, anomaly)


def test_hosted_ssa_sees_coalt_and_below_shell():
    ssa = hosted_ssa_config(host_stride=1)
    b = ssa.hosted_ssa.boresights_lvlh()
    cases = (
        (550.0, 16.0),
        (420.0, 10.0),
        (360.0, 12.0),
    )
    for tgt_alt, anomaly in cases:
        obs, tgt = _equatorial_pair(550.0, tgt_alt, anomaly)
        r_o, v_o = obs.states(0.0)
        r_t, _ = tgt.states(0.0)
        hits = heads_in_view(
            r_o, v_o, r_t[0], b,
            ssa.hosted_ssa.fov_half_angle_deg, ssa.hosted_ssa.min_los_elevation_deg,
        )
        assert hits.any(), (tgt_alt, anomaly, float(los_elevation_deg(r_o, r_t)[0]))


def test_ssa_is_extra_head_not_adcs_recant():
    adcs = SensorConfig()
    ssa = hosted_ssa_config()
    assert adcs.tracker_elevation_deg == 5.0
    assert adcs.fov_half_angle_deg == 7.0
    assert adcs.tracker_azimuths_deg == (0.0, 120.0, 240.0)
    assert np.allclose(ssa.adcs_boresights_lvlh(), adcs.adcs_boresights_lvlh())
    assert ssa.hosted_ssa is not None
    assert ssa.hosted_ssa.elevations_deg == (-8.0,)
    assert ssa.hosted_ssa.fov_half_angle_deg == 12.0


def test_hosted_ssa_only_on_stride_subset():
    ssa = hosted_ssa_config(host_stride=10)
    # two equatorial buses at the same state; only index 0 is a host
    r = np.array([[R_EARTH + 550.0, 0.0, 0.0], [R_EARTH + 550.0, 0.0, 0.0]])
    v = np.array([[0.0, 7.6, 0.0], [0.0, 7.6, 0.0]])
    # target ~16° ahead, co-alt — in the down-looker FOV
    th = np.radians(16.0)
    a = R_EARTH + 550.0
    r_tgt = a * np.array([np.cos(th), np.sin(th), 0.0])
    b = ssa.hosted_ssa.boresights_lvlh()
    hits = heads_in_view(
        r, v, r_tgt, b,
        ssa.hosted_ssa.fov_half_angle_deg, ssa.hosted_ssa.min_los_elevation_deg,
    )
    hosted = (np.arange(2) % ssa.hosted_ssa.host_stride) == 0
    assert hits[0].any() and hosted[0]
    assert not hosted[1]


def test_mixed_cant_boresights():
    b = tracker_boresights_lvlh((-8.0, 8.0), (0.0, 180.0))
    assert b.shape == (2, 3)
    assert np.allclose(b[:, 2], np.sin(np.radians([-8.0, 8.0])))
    assert np.allclose(np.linalg.norm(b, axis=-1), 1)


def test_fleet_scenario_orbit_defaults_to_catalog_altitude():
    iss = by_name("iss__ops")
    dtc = by_name("starlink_v2mini_dtc__ops")
    assert iss.orbit().altitude_km == 420.0
    assert dtc.orbit().altitude_km == 360.0
    assert iss.orbit(study_torus=True).altitude_km == STUDY_ALT_KM
    assert study_orbit().altitude_km == STUDY_ALT_KM


def test_simulate_detections_adcs_misses_below_shell_ssa_hits():
    obs, tgt = _equatorial_pair(550.0, 420.0, 10.0)
    att = PrincipalAxisSpin(200.0, 35.0, 127.4, 0.7)
    sun = np.array([0.0, 0.0, 1.0])
    t_grid = np.array([0.0])
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="no detections"):
        simulate_detections(obs, tgt, box_wing(), att, sun, t_grid, SensorConfig(), rng)
    ssa_obs = simulate_detections(
        obs, tgt, box_wing(), att, sun, t_grid, hosted_ssa_config(host_stride=1), rng,
    )
    assert len(ssa_obs) >= 1
    assert np.all(ssa_obs.tracker_id >= 3)
