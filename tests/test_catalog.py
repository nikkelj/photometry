"""Facet catalog: family builders, articulation, thrust, SATCAT coverage."""

import numpy as np
import pytest

from photometry.frames import unit
from photometry.catalog import (
    FAMILIES,
    coverage_report,
    family,
    leftover_inventory,
    load_snapshot,
    resolve,
    snapshot_meta,
)
from photometry.shapes import (
    ATT_LVLH,
    ATT_NADIR,
    CELLS,
    GIMBAL_1AXIS,
    GIMBAL_2AXIS,
    LIBRARY,
    PANEL_BACK,
    _Builder,
    iss as study_iss,
    polygon_area_normal,
    starlink_v15 as study_starlink_v15,
    starlink_v2mini,
)

HIGH_COUNT = (
    "starlink_v15", "starlink_v2mini", "starlink_v2mini_dtc",
    "oneweb", "kuiper", "geo_bus", "iss",
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
        "CYGFM01": "cygnss",
        "STARLING 1": "cubesat_6u",
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
    assert resolve("H-3 R/B", object_type="R/B").family_id == "h3_upper"
    assert resolve("H-2A R/B", object_type="R/B").family_id == "h2_upper"
    assert resolve("IABS R/B", object_type="R/B").family_id == "iabs"
    assert resolve("SL-11 R/B", object_type="R/B").family_id == "tsyklon3"
    assert resolve("STARLINK-1008", launch_date="2019-11-11").confidence == "high"


def test_study_library_geometry_not_overwritten():
    """620 km study meshes stay on shapes.LIBRARY; catalog uses FCC/NASA."""
    study = study_starlink_v15()
    study_arr = [study.areas[i] for i, lab in enumerate(study.labels)
                 if "array" in lab.lower() and study.mirror_of[i] < 0]
    assert study_arr and abs(study_arr[0] - 8.1 * 2.7) < 1e-6
    cat = family("starlink_v15")
    cat_arr = [cat.areas[i] for i, lab in enumerate(cat.labels)
               if "array" in lab.lower() and cat.mirror_of[i] < 0]
    assert cat_arr and abs(cat_arr[0] - 8.1 * 2.8) < 1e-6
    study_i = study_iss()
    study_cells = sum(study_i.areas[i] for i, lab in enumerate(study_i.labels)
                      if "arrays" in lab.lower() and study_i.mirror_of[i] < 0)
    assert abs(study_cells - 2 * 35 * 24) < 1e-6
    cat_i = family("iss")
    cat_cells = sum(cat_i.areas[i] for i, lab in enumerate(cat_i.labels)
                    if "arrays" in lab.lower() and cat_i.mirror_of[i] < 0)
    assert abs(cat_cells - 2 * 35 * 36) < 1e-6


def test_photometry_family_quality():
    ow = family("oneweb")
    ys = [v[1] for verts, _ in ow.polygons for v in verts]
    assert abs(max(ys) - min(ys) - 5.0) < 0.05
    assert any(ow.gimbal_mode[i] == GIMBAL_1AXIS for i in range(ow.n_facets))
    sl = family("starlink_v15")
    assert any(sl.gimbal_mode[i] == GIMBAL_1AXIS for i in range(sl.n_facets))
    v2 = family("starlink_v2mini")
    assert any(v2.gimbal_mode[i] == GIMBAL_2AXIS for i in range(v2.n_facets))
    geo = family("geo_bus")
    assert any(geo.gimbal_mode[i] == GIMBAL_2AXIS for i in range(geo.n_facets))
    assert "WHITE_PAINT" in geo.material_class
    assert "ANTENNA" in geo.material_class
    assert "CELLS" in geo.material_class
    ku = family("kuiper")
    assert ku.dimension_status.get("array_gimbal") == "uncertain"
    iss_card = family("iss").describe()
    assert "2-axis" in iss_card
    assert "radiator" in iss_card.lower() or "deployable" in iss_card


def test_leftover_inventory_lists_unpublished_piles():
    inv = leftover_inventory()
    prefixes = {row["prefix"]: row["n"] for row in inv["leo_prefixes"]}
    assert inv["leo_box_wing_n"] == sum(prefixes.values())
    assert prefixes.get("COSMOS", 0) >= 100
    assert prefixes.get("GEESAT", 0) >= 50
    assert prefixes.get("JILIN", 0) >= 30
    assert "CYGFM" not in prefixes
    names = {row["name"] for row in inv["rocket_body_names"]}
    assert "IABS R/B" not in names
    assert "H-3 R/B" not in names
    assert any("AKM" in n or "PKM" in n for n in names)


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


def test_hinge_1axis_is_rotation_with_cosine_loss():
    b = _Builder("hinge_1")
    b.panel((0, 0, 0), (1, 0, 0), (0, 1, 0), 1.0, 1.0, CELLS, PANEL_BACK,
            "array", gimbal=GIMBAL_1AXIS, gimbal_axis=(1, 0, 0))
    m = b.build()
    sun = unit(np.array([[0.80, 0.40, 0.45], [0.20, -0.60, 0.77]]))
    n = m.body_normals(sun, articulate=True)
    i = 0  # front
    g = np.array([1.0, 0.0, 0.0])
    assert np.allclose(n[i] @ g, 0, atol=1e-12)
    for k, s in enumerate(sun):
        s_perp = s - (s @ g) * g
        s_perp = s_perp / np.linalg.norm(s_perp)
        assert np.allclose(n[i, k], s_perp)
        assert n[i, k] @ s < 0.999   # out-of-plane cosine loss
        assert n[i, k] @ s > 0
    rest = m.body_normals(sun, articulate=False)
    assert np.allclose(rest[i], m.normals[i])


def test_hinge_travel_clamped():
    b = _Builder("clamp")
    b.panel((0, 0, 0), (1, 0, 0), (0, 1, 0), 1.0, 1.0, CELLS, PANEL_BACK,
            "array", gimbal=GIMBAL_1AXIS, gimbal_axis=(1, 0, 0),
            travel=(-0.10, 0.10), travel_status="public")
    m = b.build()
    sun = unit(np.array([[0.0, 1.0, 0.05]]))  # wants ~90° about +x
    th, ph = m.gimbal_angles(sun)
    assert abs(th[0, 0]) <= 0.10 + 1e-12
    assert np.allclose(ph[0], 0)
    n = m.body_normals(sun)[0, 0]
    # not free to snap to s_perp
    g = np.array([1.0, 0.0, 0.0])
    s_perp = sun[0] - (sun[0] @ g) * g
    s_perp = s_perp / np.linalg.norm(s_perp)
    assert np.linalg.norm(n - s_perp) > 0.2


def test_two_axis_shoulder_then_wrist():
    b = _Builder("two")
    b.panel((0, 0, 0), (1, 0, 0), (0, 1, 0), 1.0, 1.0, CELLS, PANEL_BACK,
            "array", gimbal=GIMBAL_2AXIS, gimbal_axis=(1, 0, 0),
            wrist_axis=(0, 1, 0))
    m = b.build()
    sun = unit(np.array([[0.40, -0.50, 0.77], [0.90, 0.10, 0.42]]))
    n = m.body_normals(sun)
    assert np.allclose(n[0], sun)          # unlimited 2-axis reaches the sun
    # wrist locked → same as 1-axis about the shoulder
    b1 = _Builder("locked")
    b1.panel((0, 0, 0), (1, 0, 0), (0, 1, 0), 1.0, 1.0, CELLS, PANEL_BACK,
             "array", gimbal=GIMBAL_2AXIS, gimbal_axis=(1, 0, 0),
             wrist_axis=(0, 1, 0), wrist_travel=(0.0, 0.0))
    locked = b1.build()
    n_lock = locked.body_normals(sun)
    g = np.array([1.0, 0.0, 0.0])
    assert np.allclose(n_lock[0] @ g, 0, atol=1e-12)
    assert np.all(np.einsum("kj,kj->k", n_lock[0], sun) < 0.999)


def test_fixed_arrays_stay_at_rest():
    bw = family("bluewalker3")
    assert not bw.articulated
    sun = unit(np.array([[0.3, 0.4, 0.86]]))
    n = bw.body_normals(sun, articulate=True)
    assert np.allclose(n[:, 0, :], bw.normals)


def test_study_library_articulation_numerically_stable():
    """Unlimited hinge path matches the old sun-track formulas on LIBRARY."""
    sun = unit(np.array([[0.4, -0.5, 0.77], [0.9, 0.1, 0.42]]))
    m1 = study_starlink_v15()
    n = m1.body_normals(sun, articulate=True)
    for i in range(m1.n_facets):
        if m1.gimbal_mode[i] != GIMBAL_1AXIS or m1.mirror_of[i] >= 0:
            continue
        g = m1.gimbal_axis[i]
        s_perp = sun - np.outer(sun @ g, g)
        s_perp = s_perp / np.linalg.norm(s_perp, axis=-1, keepdims=True)
        assert np.allclose(n[i], s_perp)
    m2 = starlink_v2mini()
    n2 = m2.body_normals(sun, articulate=True)
    for i in range(m2.n_facets):
        if m2.gimbal_mode[i] == GIMBAL_2AXIS and m2.mirror_of[i] < 0:
            assert np.allclose(n2[i], sun)


def test_deployable_look_vs_flight_attitude():
    cap = family("capella")
    i = next(j for j, lab in enumerate(cap.labels)
             if "sar" in lab.lower() and cap.mirror_of[j] < 0)
    assert np.allclose(cap.look_body[i], [0, 0, -1])
    assert cap.look_attitude[i] == ATT_NADIR
    assert cap.look_status[i] == "public"
    ice = family("iceye")
    i = next(j for j, lab in enumerate(ice.labels)
             if "sar" in lab.lower() and ice.mirror_of[j] < 0)
    assert ice.look_status[i] == "unknown"
    assert np.allclose(ice.look_body[i], 0)
    geo = family("geo_bus")
    i = next(j for j, lab in enumerate(geo.labels)
             if "dish" in lab.lower() and geo.mirror_of[j] < 0)
    assert np.allclose(geo.look_body[i], [0, 0, -1])
    assert geo.look_status[i] == "typical_class"
    dtc = family("starlink_v2mini_dtc")
    i = next(j for j, lab in enumerate(dtc.labels)
             if "dtc" in lab.lower() and dtc.mirror_of[j] < 0)
    assert np.allclose(dtc.look_body[i], [0, 0, -1])
    v15 = family("starlink_v15")
    i = next(j for j, lab in enumerate(v15.labels)
             if "bus -z" in lab.lower())
    assert np.allclose(v15.look_body[i], [0, 0, -1])
    assert v15.flight_attitude == ATT_LVLH
    um = family("umbra")
    i = next(j for j, lab in enumerate(um.labels)
             if "sar" in lab.lower() and um.mirror_of[j] < 0)
    assert um.look_status[i] == "unknown"


def test_high_count_surfaces_complete():
    for fid in ("starlink_v15", "starlink_v2mini", "starlink_v2mini_dtc",
                "oneweb", "kuiper", "geo_bus", "iss", "iceye", "capella",
                "umbra"):
        m = family(fid)
        assert all(m.material_class), fid
        assert np.all(np.isfinite(m.rho_d)) and np.all(np.isfinite(m.k_s))
        assert np.all(np.isnan(m.alpha_ir))
        assert all(p == "unknown" for p in m.ir_provenance)
        fronts = [i for i, lab in enumerate(m.labels)
                  if "array" in lab.lower() and "front" in lab.lower()
                  and m.mirror_of[i] < 0]
        backs = [i for i, lab in enumerate(m.labels)
                 if "array" in lab.lower() and "back" in lab.lower()]
        if fronts:
            assert all(m.material_class[i] == "CELLS" for i in fronts), fid
        if backs:
            assert all(m.material_class[i] == "PANEL_BACK" for i in backs), fid


def _assert_poly_matches_facet(m, verts, i, *, plane_rtol=1e-6):
    assert 0 <= i < m.n_facets
    area, n = polygon_area_normal(verts)
    assert area > 0
    rel = abs(area - m.areas[i]) / m.areas[i]
    lab = m.labels[i].lower()
    # Cylinder sides/caps: photometry uses the true circular area; the
    # drawable is an inscribed n-gon (chord < arc).
    if "side" in lab or "cap" in lab or len(verts) > 4:
        assert rel < 0.06, (m.name, m.labels[i], rel)
    else:
        assert rel < plane_rtol, (m.name, m.labels[i], rel)
    assert np.allclose(n, m.normals[i], atol=1e-7), (m.name, m.labels[i])


def test_every_existing_polygon_matches_facet():
    """Any stored rest-pose poly must match that facet's area and winding."""
    for fid in FAMILIES:
        m = family(fid)
        seen = set()
        for verts, i in m.polygons:
            _assert_poly_matches_facet(m, verts, i)
            seen.add(i)
        assert len(seen) == len(m.polygons)


def test_high_count_rest_pose_polygons_complete_and_oml():
    """Catalog SATCAT stand-ins already had bus/array/deployable quads.

    This pass locks OML spans to the cited figures and fills mirror-back
    quads on the catalog copies only. Study LIBRARY is not touched.
    """
    spans = {}
    for fid in HIGH_COUNT:
        m = family(fid)
        have = {i for _, i in m.polygons}
        assert have == set(range(m.n_facets)), (fid, m.n_facets - len(have))
        for verts, i in m.polygons:
            _assert_poly_matches_facet(m, verts, i)
        allv = np.vstack([v for v, _ in m.polygons])
        spans[fid] = allv.max(0) - allv.min(0)

    # FCC v1.5: bus 2.8×1.3×0.2, array 2.8×8.1, ~11.2 m end-to-end.
    assert np.allclose(spans["starlink_v15"], [11.2, 2.8, 0.2], atol=1e-6)
    v15 = family("starlink_v15")
    arr = next(v for v, i in v15.polygons if v15.labels[i] == "array front")
    a, _ = polygon_area_normal(arr)
    assert abs(a - 8.1 * 2.8) < 1e-8

    # FCC v2 Mini: two 4.1×12.8 m wings, ~30 m tip-to-tip.
    assert np.allclose(spans["starlink_v2mini"][:2], [30.5, 4.1], atol=1e-6)
    v2 = family("starlink_v2mini")
    fore = next(v for v, i in v2.polygons if v2.labels[i] == "array fore front")
    a, _ = polygon_area_normal(fore)
    assert abs(a - 12.8 * 4.1) < 1e-8
    assert spans["starlink_v2mini_dtc"][2] >= 0.5  # DTC off −z

    # KeepTrack ARROW 5 m tip-to-tip; protoflight Kuiper 10 m; GEO mid 22.8 m.
    assert abs(spans["oneweb"][1] - 5.0) < 1e-6
    assert abs(spans["kuiper"][1] - 10.0) < 1e-6
    assert abs(spans["geo_bus"][1] - 22.8) < 1e-6

    # NASA ISS: 50 m stack, 109 m truss inside a 35×36 m array group.
    iss = family("iss")
    assert abs(spans["iss"][0] - 50.0) < 1e-6
    truss = next(v for v, i in iss.polygons if iss.labels[i] == "truss +x")
    assert abs(truss[:, 1].max() - truss[:, 1].min() - 109.0) < 1e-6
    stbd = next(v for v, i in iss.polygons if iss.labels[i] == "arrays stbd front")
    a, _ = polygon_area_normal(stbd)
    assert abs(a - 35.0 * 36.0) < 1e-8


def test_study_library_polygons_not_churned():
    """Study meshes keep their original drawable lists (no mirror-back fill)."""
    m = study_starlink_v15()
    have = {i for _, i in m.polygons}
    assert len(m.polygons) == 7
    assert any(m.labels[i] == "array back" and i not in have
               for i in range(m.n_facets))
    study = study_iss()
    assert len(study.polygons) == 16
    cat = family("iss")
    assert len(cat.polygons) == cat.n_facets == 20


def test_photometry_ignores_polygons():
    from photometry.radiometry import facet_brightness
    m = family("starlink_v2mini")
    u = unit(np.array([[0.3, 0.4, 0.86]]))
    b0 = facet_brightness(m, u, u)
    m.polygons = []
    b1 = facet_brightness(m, u, u)
    assert np.allclose(b0, b1)


def test_hinge_does_not_rewrite_rest_pose_polygons():
    m = family("starlink_v15")
    before = [np.array(v, copy=True) for v, _ in m.polygons]
    sun = unit(np.array([[0.8, 0.4, 0.45]]))
    m.body_normals(sun, articulate=True)
    after = [v for v, _ in m.polygons]
    for a, b in zip(before, after):
        assert np.allclose(a, b)
