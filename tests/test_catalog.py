"""Facet catalog: family builders, articulation, thrust, SATCAT coverage."""

import numpy as np
import pytest

from photometry.frames import unit
from photometry.catalog import (
    FAMILIES,
    coverage_report,
    family,
    load_snapshot,
    resolve,
    snapshot_meta,
)
from photometry.shapes import (
    GIMBAL_1AXIS,
    GIMBAL_2AXIS,
    LIBRARY,
    starlink_v2mini,
)

STUDY_LIBRARY = {
    "box_wing", "rocket_body", "starlink_v15", "starlink_v2mini",
    "starlink_v2mini_dtc", "bluewalker3", "hubble", "iss", "katalyst_link",
}


def test_study_library_keys_unchanged():
    assert set(LIBRARY) == STUDY_LIBRARY


def test_every_family_builds():
    for name in FAMILIES:
        m = family(name)
        assert m.n_facets >= 2, name
        assert np.allclose(np.linalg.norm(m.normals, axis=-1), 1), name
        assert np.all(m.areas > 0), name
        assert len(m.labels) == m.n_facets, name
        assert len(m.material_class) == m.n_facets, name
        assert all(m.material_class), name
        assert len(m.optical_provenance) == m.n_facets, name
        assert len(m.ir_provenance) == m.n_facets, name
        assert m.alpha_ir.shape == (m.n_facets,)
        assert m.epsilon_ir.shape == (m.n_facets,)
        assert m.family_id
        assert m.sources, name
        for i in range(m.n_facets):
            if m.mirror_of[i] >= 0:
                assert np.allclose(m.normals[i], -m.normals[m.mirror_of[i]]), name


def test_oneaxis_and_twoaxis_articulation():
    u_sun = unit(np.array([[0.4, -0.5, 0.77], [0.9, 0.1, 0.42]]))
    m2 = family("starlink_v2mini")
    n = m2.body_normals(u_sun, articulate=True)
    saw2 = False
    for i in range(m2.n_facets):
        if m2.gimbal_mode[i] == GIMBAL_2AXIS and m2.mirror_of[i] < 0:
            assert np.allclose(n[i], u_sun)
            saw2 = True
    assert saw2
    m1 = family("oneweb")
    n = m1.body_normals(u_sun, articulate=True)
    saw1 = False
    for i in range(m1.n_facets):
        if m1.gimbal_mode[i] == GIMBAL_1AXIS and m1.mirror_of[i] < 0:
            g = m1.gimbal_axis[i]
            assert np.allclose(n[i] @ g, 0, atol=1e-12)
            assert np.all(np.einsum("kj,kj->k", n[i], u_sun) > 0)
            saw1 = True
    assert saw1
    # BlueWalker-class arrays stay fixed
    bw = family("bluewalker3")
    assert not bw.articulated


def test_thrust_vectors_unit_and_documented():
    for name in FAMILIES:
        m = family(name)
        assert m.thrust_attitude, name
        assert m.thrust_propulsion, name
        assert m.thrust_notes, name
        if len(m.thrust_body) == 0:
            continue
        nrm = np.linalg.norm(m.thrust_body, axis=1)
        assert np.allclose(nrm, 1.0), name
        assert m.thrust_body.shape[1] == 3


def test_reviewer_card_starlink_v2mini():
    m = starlink_v2mini()
    card = m.describe()
    assert "starlink_v2mini" in card
    assert "2-axis" in card
    assert "thrust" in card.lower()
    assert any(c == "MLI" for c in m.material_class)
    assert any(c == "CELLS" for c in m.material_class)
    assert m.thrust_propulsion == "ep"
    dtc = starlink_v2mini(dtc=True)
    assert any("dtc" in lab.lower() for lab in dtc.labels)
    assert dtc.family_id == "starlink_v2mini_dtc"


def test_iss_thrust_is_lvlh_plus_x():
    m = family("iss")
    assert m.thrust_attitude == "lvlh"
    assert np.allclose(m.thrust_body, [[1.0, 0.0, 0.0]])


def test_mapping_examples():
    assert resolve("STARLINK-11072 [DTC]").family_id == "starlink_v2mini_dtc"
    assert resolve("STARLINK-1008", launch_date="2019-11-11").family_id == "starlink_v15"
    assert resolve("STARLINK-30000", launch_date="2024-01-01").family_id == "starlink_v2mini"
    assert resolve("ONEWEB-0560").family_id == "oneweb"
    assert resolve("KUIPER-00008").family_id == "kuiper"
    assert resolve("QIANFAN-1").family_id == "qianfan"
    assert resolve("FLOCK 4Q-16").family_id == "planet_superdove"
    assert resolve("SKYSAT-A").family_id == "planet_skysat"
    assert resolve("ICEYE-X2").family_id == "iceye"
    assert resolve("IRIDIUM 106").family_id == "iridium_next"
    assert resolve("ISS (ZARYA)").family_id == "iss"
    assert resolve("HST").family_id == "hubble"
    assert resolve("LINK", cospar="2026-152A").family_id == "katalyst_link"
    assert resolve("FALCON 9 R/B", object_type="R/B").family_id == "falcon9_s2"
    assert resolve("INTELSAT 902 (IS-902)", period_min=1436.0).family_id == "geo_bus"
    assert resolve("NAVSTAR 82 (USA 343)").family_id == "gnss_meo"
    assert resolve("GSAT0101 (GALILEO-PFM)").family_id == "galileo"
    assert resolve("USA 105").family_id == "classified_unpublished"
    # Starshield internals are not invented
    assert resolve("USA 105").notes.lower().find("starshield") >= 0
    # pass-2: must not dump these to leo_box_wing
    named = {
        "CAPELLA-11 (ACADIA-1)": "capella",
        "UMBRA-07": "umbra",
        "HAWK-A": "hawkeye360",
        "GLOBAL-2": "blacksky",
        "GLOBALSTAR M069": "globalstar2",
        "ORBCOMM FM06": "orbcomm_og2",
        "SENTINEL-1A": "sentinel1",
        "SENTINEL-2A": "sentinel2",
        "LANDSAT 8": "landsat8",
        "CSS (TIANHE)": "css_tianhe",
        "LEGION 1": "maxar_legion",
        "WORLDVIEW-3 (WV-3)": "worldview",
        "SPACEMOBILE-001": "bluebird",
        "SPACEMOBILE-006": "bluebird_block2",
        "TERRA": "terra",
        "NOAA 20 (JPSS-1)": "jpss",
        "SITRO-AIS 5 (KATYS)": "cubesat_6u",
        "INMARSAT 3-F1": "geo_bus",
        "GOES 16": "goes_r",
        "SENTINEL-6A": "sentinel6",
        "JASON-3": "sentinel6",
        "METOP-B": "metop",
        "ASTROCAST-0401": "cubesat_3u",
        "NUSAT-1": "cubesat_6u",
        "TERRASAR-X": "terrasat_x",
        "PAZ": "terrasat_x",
        "COSMO-SKYMED 1": "cosmo_skymed",
        "ALOS-2": "alos2",
        "RADARSAT-2": "radarsat2",
        "GAOFEN-3": "gaofen3",
        "SAOCOM 1A": "saocom",
        "SWARM A": "esa_swarm",
        "ION SCV-011": "ion_scv",
        "GHGSAT-C1": "ghgsat",
        "GRUS-1A": "grus",
        "TOMORROW-S1": "cubesat_6u",
        "FLOCK 4Q-16": "planet_superdove",
        "SNUSAT-2": "cubesat_3u",
    }
    for name, fam in named.items():
        hit = resolve(name)
        assert hit.family_id == fam, (name, hit)
        assert hit.family_id != "leo_box_wing"
    assert resolve("SNUSAT-1").family_id != "cubesat_6u"
    assert resolve("SNUSAT-1").family_id == "cubesat_3u"
    assert resolve("DRAGONFLY").family_id != "cargo_dragon"
    assert resolve("GAOFEN-1").family_id == "leo_box_wing"
    assert resolve("JILIN-1").family_id == "leo_box_wing"
    assert resolve("CSG-1").family_id == "leo_box_wing"
    assert resolve("YAOGAN 1").family_id == "classified_unpublished"
    assert resolve("YZ-1 R/B", object_type="R/B").family_id == "cz_upper"
    assert resolve("ATLAS 5 CENTAUR R/B", object_type="R/B").family_id == "centaur"
    assert resolve("ATLAS AGENA D R/B", object_type="R/B").family_id == "agena"
    assert resolve("FREGAT R/B", object_type="R/B").family_id == "fregat"
    assert resolve("SL-8 R/B", object_type="R/B").family_id == "kosmos_3m"
    assert resolve("SL-12 R/B", object_type="R/B").family_id == "proton_block_d"
    assert resolve("PSLV R/B", object_type="R/B").family_id == "pslv_ps4"
    assert resolve("IUS R/B(1)", object_type="R/B").family_id == "ius"
    assert resolve("SCOUT G-1 R/B", object_type="R/B").family_id == "scout"
    assert resolve("PEGASUS R/B", object_type="R/B").family_id == "pegasus"
    assert resolve("ANIK C1 R/B [PAM-D]", object_type="R/B").family_id == "pam_d"
    assert resolve("TITAN 3C TRANSTAGE R/B", object_type="R/B").family_id == "titan_transtage"
    assert resolve("TITAN 4 R/B", object_type="R/B").family_id == "titan_core"
    assert resolve("ATLAS 2A R/B", object_type="R/B").family_id == "atlas_core"
    assert resolve("SATURN 5 R/B", object_type="R/B").family_id == "saturn_sivb"
    assert resolve("AVUM R/B", object_type="R/B").family_id == "avum"
    assert resolve("FIREFLY ALPHA R/B", object_type="R/B").family_id == "firefly_alpha"
    assert resolve("SL-23 R/B", object_type="R/B").family_id == "dnepr"
    assert resolve("THOR ABLESTAR R/B", object_type="R/B").family_id == "thor_ablestar"
    assert resolve("THOR BURNER 2 R/B", object_type="R/B").family_id == "burner2"
    assert resolve("STARLINK-1008", launch_date="2019-11-11").confidence == "high"


def test_coverage_against_vendored_snapshot():
    meta = snapshot_meta()
    assert meta["snapshot_utc"] == "2026-08-21"
    rows = load_snapshot()
    assert len(rows) == meta["n_rows"]
    report = coverage_report(rows)
    pay = report["active_payloads"]
    assert pay["n"] == meta["n_active_payloads"]
    assert pay["fraction"] >= 0.99
    # Starlink alone is ~65% of the active catalog; named families must beat that.
    assert pay["fraction_named"] > 0.80
    assert report["rocket_bodies"]["fraction"] >= 0.99
    assert report["rocket_bodies"]["fraction_named"] > 0.55
    assert not report["mapped_to_unknown_family"]
    assert "starlink_v2mini" in pay["by_family"]
    assert "oneweb" in pay["by_family"]
    assert "kuiper" in pay["by_family"]
    assert "capella" in pay["by_family"]
    assert "sentinel1" in pay["by_family"]
    assert "goes_r" in pay["by_family"]
    assert "terrasat_x" in pay["by_family"]
    assert "gaofen3" in pay["by_family"]


def test_surfaces_split_on_new_families():
    m = family("capella")
    classes = set(m.material_class)
    assert "MLI" in classes and "CELLS" in classes and "ANTENNA" in classes
    assert all(p == "unknown" for p in m.ir_provenance)
    assert np.all(np.isnan(m.alpha_ir))


def test_unknown_family_raises():
    with pytest.raises(KeyError):
        family("not_a_real_family")
