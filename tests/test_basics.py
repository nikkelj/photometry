import numpy as np
import pytest

from photometry.attitude import PrincipalAxisSpin, spin_body_directions
from photometry.constellation import WalkerConstellation, tracker_boresights_lvlh
from photometry.frames import MU_EARTH, lvlh_basis, radec_to_unit, unit
from photometry.measurements import ObservationSet
from photometry.radiometry import apparent_magnitude, facet_brightness, mag_to_normalized_brightness
from photometry.shapes import box_wing
from photometry.simulate import Scenario, run


def test_walker_orbit_radius_and_speed():
    c = WalkerConstellation(4, 5, altitude_km=550.0, inclination_deg=53.0)
    r, v = c.states(1234.0)
    a = c.semi_major_axis_km
    assert np.allclose(np.linalg.norm(r, axis=-1), a, rtol=1e-12)
    assert np.allclose(np.linalg.norm(v, axis=-1), np.sqrt(MU_EARTH / a), rtol=1e-12)


def test_lvlh_basis_orthonormal():
    c = WalkerConstellation(2, 3)
    r, v = c.states(500.0)
    along, cross, up = lvlh_basis(r, v)
    assert np.allclose(np.sum(along * up, axis=-1), 0, atol=1e-12)
    assert np.allclose(np.cross(up, along), cross, atol=1e-12)
    assert np.allclose(np.linalg.norm(cross, axis=-1), 1, atol=1e-12)


def test_tracker_elevation():
    b = tracker_boresights_lvlh(5.0)
    # LVLH z-component is sin(5 deg) for all three trackers
    assert np.allclose(b[:, 2], np.sin(np.radians(5.0)))
    assert np.allclose(np.linalg.norm(b, axis=-1), 1)


def test_spin_pole_is_body_z():
    att = PrincipalAxisSpin(200.0, 35.0, 120.0, 0.3)
    t = np.array([0.0, 37.0, 91.0])
    pole_body = att.eci_to_body(t, np.tile(att.pole, (3, 1)))
    assert np.allclose(pole_body, [0, 0, 1], atol=1e-12)


def test_spin_functional_matches_class():
    att = PrincipalAxisSpin(123.0, -20.0, 200.0, 1.1)
    t = np.linspace(0, 500, 7)
    v = unit(np.random.default_rng(0).normal(size=(7, 3)))
    a = att.eci_to_body(t, v)
    b = spin_body_directions(att.pole, 200.0, 1.1, t, v)
    assert np.allclose(a, b, atol=1e-12)


def test_magnitude_roundtrip():
    shape = box_wing()
    u = unit(np.array([[0.3, 0.4, 0.86], [-0.5, 0.2, 0.84]]))
    rng_km = np.array([800.0, 1500.0])
    mag = apparent_magnitude(shape, u, u, rng_km)
    b = mag_to_normalized_brightness(mag, rng_km)
    b_direct = facet_brightness(shape, u, u).sum(axis=0)
    assert np.allclose(b, b_direct, rtol=1e-10)


def test_backscatter_brightest_at_zero_phase():
    shape = box_wing()
    n = unit(np.array([[0.0, 0.0, 1.0]]))
    off = unit(np.array([[0.0, np.sin(0.8), np.cos(0.8)]]))
    b0 = facet_brightness(shape, n, n).sum()
    b1 = facet_brightness(shape, n, off).sum()
    assert b0 > b1


def test_observation_roundtrip(tmp_path):
    sc = Scenario(duration_s=240.0, dt_s=4.0)
    obs, _ = run(sc)
    assert len(obs) > 10
    p = tmp_path / "obs.npz"
    obs.to_npz(p)
    obs2 = ObservationSet.from_npz(p)
    assert np.allclose(obs.mag, obs2.mag)
    c = tmp_path / "obs.csv"
    obs.to_csv(c)
    obs3 = ObservationSet.from_csv(c)
    assert np.allclose(obs.los_eci, obs3.los_eci)
    assert obs3.obs_id.dtype.kind == "i"


def test_phase_angle_range():
    sc = Scenario(duration_s=240.0, dt_s=4.0)
    obs, _ = run(sc)
    ph = obs.phase_angle_deg()
    assert np.all((ph >= 0) & (ph <= 180))


def test_library_models_well_formed():
    from photometry.shapes import LIBRARY
    for name, factory in LIBRARY.items():
        m = factory()
        assert np.allclose(np.linalg.norm(m.normals, axis=-1), 1), name
        assert np.all(m.areas > 0), name
        assert len(m.labels) == m.n_facets, name
        assert m.polygons, name
        for i in range(m.n_facets):
            if m.mirror_of[i] >= 0:
                assert np.allclose(m.normals[i], -m.normals[m.mirror_of[i]]), name


def test_articulation_tracks_sun():
    from photometry.shapes import GIMBAL_1AXIS, GIMBAL_2AXIS, starlink_v15, starlink_v2mini
    u_sun = unit(np.array([[0.4, -0.5, 0.77], [0.9, 0.1, 0.42]]))
    m2 = starlink_v2mini()
    n = m2.body_normals(u_sun, articulate=True)
    for i in range(m2.n_facets):
        if m2.gimbal_mode[i] == GIMBAL_2AXIS and m2.mirror_of[i] < 0:
            assert np.allclose(n[i], u_sun)  # perfect sun pointing
    m1 = starlink_v15()
    n = m1.body_normals(u_sun, articulate=True)
    for i in range(m1.n_facets):
        if m1.gimbal_mode[i] == GIMBAL_1AXIS and m1.mirror_of[i] < 0:
            g = m1.gimbal_axis[i]
            assert np.allclose(n[i] @ g, 0, atol=1e-12)  # stays in gimbal plane
            assert np.all(np.einsum("kj,kj->k", n[i], u_sun) > 0)


def test_lvlh_hold_attitude():
    from photometry.attitude import LvlhHold
    from photometry.constellation import WalkerConstellation
    orb = WalkerConstellation(1, 1, 620.0, 70.0)
    att = LvlhHold(orb)
    t = np.array([0.0, 1000.0, 4000.0])
    r, v = orb.single_states(t)
    up_body = att.eci_to_body(t, unit(r))
    assert np.allclose(up_body, [0, 0, 1], atol=1e-12)  # body z = zenith
    knife = LvlhHold(orb, roll_deg=90.0)
    up_knife = knife.eci_to_body(t, unit(r))
    assert np.allclose(up_knife, [0, 1, 0], atol=1e-12)  # rolled 90 about x


def test_spin_body_axis():
    att = PrincipalAxisSpin(200.0, 35.0, 120.0, 0.3, body_axis=(1, 0, 0))
    t = np.array([0.0, 40.0, 77.0])
    axis_body = att.eci_to_body(t, np.tile(att.pole, (3, 1)))
    assert np.allclose(axis_body, [1, 0, 0], atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_minkowski_cube_reconstruction():
    from photometry.inversion.minkowski import reconstruct_hull
    normals = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                        [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    faces = reconstruct_hull(normals, np.full(6, 4.0))
    egi = [f for f in faces if not f.is_closure]
    assert len(egi) == 6
    total = sum(f.area for f in egi)
    assert abs(total - 24.0) / 24.0 < 0.15
    v = np.vstack([f.vertices for f in faces])
    ext = v.max(0) - v.min(0)
    assert np.all(np.abs(ext - 2.0) < 0.4)  # ~2 m cube
    assert np.abs(v.mean(0)).max() < 0.2  # recentered near origin
