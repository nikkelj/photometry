"""Additional catalog families (pass-2): cited public OML only.

Payloads that were dumping to leo_box_wing, plus lingering upper stages
with published diameters. Materials are split bus / cells / panel-back /
antenna. IR α/ε stay unknown. Hall pointing is not invented.
"""

from __future__ import annotations

from ..shapes import (
    ANTENNA,
    ATT_LVLH,
    ATT_NADIR,
    ATT_STAGE_AXIS,
    ATT_YAW_STEER,
    CELLS,
    DARK,
    GIMBAL_1AXIS,
    GIMBAL_2AXIS,
    GIMBAL_FIXED,
    MLI,
    MLI_SILVER,
    PANEL_BACK,
    PROP_CHEMICAL,
    PROP_UNKNOWN,
    STATUS_PUBLIC,
    STATUS_RANGE,
    STATUS_TYPICAL,
    STATUS_UNCERTAIN,
    STATUS_UNKNOWN,
    WHITE_PAINT,
    FacetModel,
    _Builder,
)
from .families import _two_wing


def _stage(name, length, diameter, *, sources, notes="",
           dimension_status=None, n_side=16) -> FacetModel:
    b = _Builder(name)
    b.prism((0, 0, 0), (0, 0, 1), length, diameter, n_side,
            MLI_SILVER, (MLI, DARK), "stage")
    b.meta(
        family_id=name, sources=sources, notes=notes,
        dimension_status=dimension_status or {},
        thrust_body=[[0.0, 0.0, -1.0]],
        thrust_attitude=ATT_STAGE_AXIS, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Engine along −z of the cylinder; spent-stage attitude "
                     "is typically uncontrolled.",
    )
    return b.build()


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------

def capella() -> FacetModel:
    """Capella Whitney/Acadia: washing-machine bus + 3.5 m mesh SAR."""
    bus = (0.80, 0.70, 0.70)
    b = _Builder("capella")
    b.box((0, 0, 0), bus, MLI, "bus")
    # FCC ODAR: two 500 × 900 mm arrays; panels radial, SAR nadir.
    _two_wing(b, bus, (0.50, 0.90), gimbal=GIMBAL_FIXED, boom=0.10)
    b.panel((0, 0, -0.55), (0, 1, 0), (1, 0, 0), 3.5, 3.5, ANTENNA, MLI,
            "sar dish")
    b.meta(
        family_id="capella",
        sources=("FCC SAT-LOA-20200914-00108: ~110 kg, 3.5 m circular SAR "
                 "reflector; ODAR: two 500×900 mm arrays, 8 m² antenna, "
                 "3 m boom. Nominal attitude: arrays radial, SAR nadir.",
                 "eoPortal Capella X-SAR; Gunter Capella 2–10."),
        notes="Dish is a square photometric stand-in of the 3.5 m mesh. "
              "Arrays GIMBAL_FIXED after deploy (FCC radial/nadir hold).",
        dimension_status={"bus": STATUS_RANGE, "sar": STATUS_PUBLIC,
                          "arrays": STATUS_PUBLIC, "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_NADIR, thrust_propulsion=PROP_UNKNOWN,
        thrust_notes="ODAR lists a thruster mass; body-frame pointing not given.",
        flight_attitude=ATT_NADIR,
    )
    return b.build().set_look(
        "sar", (0, 0, -1), attitude=ATT_NADIR,
        notes="FCC ODAR: arrays radial, SAR nadir.",
        status=STATUS_PUBLIC)


def umbra() -> FacetModel:
    """Umbra SAR: ~65–84 kg, ~10 m² deployable mesh (diameter a range)."""
    bus = (0.70, 0.55, 0.55)
    b = _Builder("umbra")
    b.box((0, 0, 0), bus, MLI, "bus")
    _two_wing(b, bus, (0.50, 0.70), gimbal=GIMBAL_1AXIS, boom=0.10)
    # 10 m² mesh → ~3.6 m equivalent square.
    b.panel((0, 0, -0.45), (0, 1, 0), (1, 0, 0), 3.2, 3.2, ANTENNA, MLI,
            "sar mesh")
    b.meta(
        family_id="umbra",
        sources=("eoPortal Umbra SAR: centre-fed parabolic mesh, deployed "
                 "antenna 10 m².",
                 "SpaceNews 2021-03-24 (Umbra patent): ~4 m class reflector, "
                 "mesh larger than 10 m² folded into a microsat.",
                 "Bus OML is a range; not a company drawing."),
        notes="Mesh size from the public 10 m² figure (range). 1-axis arrays "
              "are a stand-in.",
        dimension_status={"bus": STATUS_RANGE, "sar": STATUS_RANGE,
                          "arrays": STATUS_RANGE, "array_gimbal": STATUS_UNCERTAIN,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_NADIR, thrust_propulsion=PROP_UNKNOWN,
        thrust_notes="Propulsion/pointing not used as a public number.",
        flight_attitude=ATT_NADIR,
    )
    return b.build().set_look(
        "sar", None, attitude=ATT_NADIR,
        notes="Centre-fed mesh; operational look angle unpublished — not invented.",
        status=STATUS_UNKNOWN)


def hawkeye360() -> FacetModel:
    """HawkEye 360 Pathfinder/Cluster: SFL NEMO-15 20×20×44 cm."""
    bus = (0.44, 0.20, 0.20)
    b = _Builder("hawkeye360")
    b.box((0, 0, 0), bus, DARK, "bus",
          mats=[DARK, MLI, MLI, MLI, MLI, ANTENNA])
    _two_wing(b, bus, (0.44, 0.20), gimbal=GIMBAL_FIXED, boom=0.02)
    b.meta(
        family_id="hawkeye360",
        sources=("eoPortal HawkEye 360: NEMO-15, 0.20×0.20×0.44 m, 13.4 kg.",
                 "Later clusters use a 30 kg SFL DEFIANT bus (SFL Cluster 14); "
                 "stand-in is the published NEMO-15 OML, tagged typical_class "
                 "for later clusters."),
        notes="SATCAT HAWK-A/B/C. Body-mounted / small fixed wings. Later "
              "30 kg buses are the same family, not a second mesh.",
        dimension_status={"bus": STATUS_PUBLIC, "arrays": STATUS_TYPICAL,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_UNKNOWN,
        thrust_notes="Formation-flying Δv exists operationally; vector unpublished.",
    )
    return b.build()


def blacksky() -> FacetModel:
    """BlackSky Gen-2 GLOBAL-n: SCOUT bus ~1.0 × 0.5 m, 55 kg."""
    bus = (1.00, 0.50, 0.50)
    b = _Builder("blacksky")
    b.box((0, 0, 0), bus, MLI, "bus",
          mats=[DARK, MLI, MLI, MLI, MLI, DARK])
    _two_wing(b, bus, (0.50, 0.25), gimbal=GIMBAL_FIXED, boom=0.05)
    b.meta(
        family_id="blacksky",
        sources=("KeepTrack / LeoStella SCOUT: GLOBAL-2 1.0 m × 0.5 m, "
                 "1 m span, ~55 kg, SpaceView-24 imager.",
                 "SATCAT names GLOBAL-n (not BLACKSKY)."),
        notes="Pelican is a different Planet vehicle and is not this family.",
        dimension_status={"bus": STATUS_PUBLIC, "arrays": STATUS_RANGE,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_NADIR, thrust_propulsion=PROP_UNKNOWN,
        thrust_notes="Pointing unpublished.",
    )
    return b.build()


def globalstar2() -> FacetModel:
    """Globalstar-2: Thales ELiTeBus-1000, ~700 kg, two wings."""
    bus = (3.1, 1.5, 1.5)
    b = _Builder("globalstar2")
    b.box((0, 0, 0), bus, MLI, "bus",
          mats=[MLI, MLI, MLI, MLI, MLI, ANTENNA])
    # Same bus as Iridium NEXT; 9.4 m span is Iridium-specific — not copied.
    _two_wing(b, bus, (1.5, 2.5), gimbal=GIMBAL_1AXIS)
    b.meta(
        family_id="globalstar2",
        sources=("Gunter Globalstar M073–103: ELiTeBus-1000, ~700 kg, "
                 "2 deployable arrays, 2.4 kW BoL / 1.7 kW EoL.",
                 "ELiTeBus-1000 bus class is the Iridium NEXT platform; "
                 "array span is not published for Globalstar-2 (range)."),
        notes="Bus reused as typical_class of the named ELiTeBus-1000. "
              "Do not treat the 9.4 m Iridium span as a Globalstar number.",
        dimension_status={"bus": STATUS_TYPICAL, "arrays": STATUS_RANGE,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Chemical station-keeping typical of ELiTeBus; vector unpublished.",
    )
    return b.build()


def orbcomm_og2() -> FacetModel:
    """ORBCOMM OG2: Sierra Nevada SN-100A, ~165 kg, 5 m span."""
    bus = (1.00, 0.50, 0.50)
    b = _Builder("orbcomm_og2")
    b.box((0, 0, 0), bus, MLI, "bus",
          mats=[MLI, MLI, MLI, MLI, MLI, ANTENNA])
    _two_wing(b, bus, (0.50, 2.25), gimbal=GIMBAL_1AXIS)
    b.meta(
        family_id="orbcomm_og2",
        sources=("Gunter / KeepTrack OG2: SN-100A, ~165 kg, box+panel+antenna, "
                 "5 m span."),
        notes="Bus 1.0×0.5 m is a range matching the published 5 m span. "
              "OG1 leftovers share the ORBCOMM FM* name.",
        dimension_status={"bus": STATUS_RANGE, "arrays": STATUS_PUBLIC,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_UNKNOWN,
        thrust_notes="Pointing unpublished.",
    )
    return b.build()


def sentinel1() -> FacetModel:
    """Sentinel-1: ESA 21 m class with 12 m C-band SAR + 2×10 m arrays."""
    bus = (3.5, 2.5, 4.0)
    b = _Builder("sentinel1")
    b.box((0, 0, 0), bus, MLI, "bus")
    _two_wing(b, bus, (2.5, 10.0), gimbal=GIMBAL_1AXIS, boom=0.3)
    b.panel((3.5 / 2 + 6.0, 0, -0.4), (0, 1, 0), (1, 0, 0), 0.8, 12.0,
            ANTENNA, MLI, "sar antenna")
    b.meta(
        family_id="sentinel1",
        sources=("ESA Sentinel-1 facts: 21 m long, 2.5 m wide, 4 m high, "
                 "2×10 m solar arrays, 12 m radar antenna."),
        notes="21 m length is bus + deployed SAR. Bus box is the 2.5×4 m "
              "core; SAR is the along-track deployable.",
        dimension_status={"bus": STATUS_PUBLIC, "arrays": STATUS_PUBLIC,
                          "sar": STATUS_PUBLIC, "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Hydrazine orbit-maintenance is public; vector unpublished.",
    )
    return b.build()


def sentinel2() -> FacetModel:
    """Sentinel-2 (and S3 as optical Copernicus class)."""
    bus = (3.4, 1.8, 2.35)
    b = _Builder("sentinel2")
    b.box((0, 0, 0), bus, MLI, "bus",
          mats=[DARK, MLI, MLI, MLI, MLI, WHITE_PAINT])
    _two_wing(b, bus, (2.35, 2.2), gimbal=GIMBAL_1AXIS)
    b.meta(
        family_id="sentinel2",
        sources=("ESA Sentinel-2 facts: 3.4 × 1.8 × 2.35 m, 1140 kg.",
                 "Sentinel-3 is mapped here as an optical Copernicus class "
                 "stand-in (S3 OML not copied as fake S2)."),
        dimension_status={"bus": STATUS_PUBLIC, "arrays": STATUS_RANGE,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_NADIR, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Chemical orbit control public for Copernicus; vector unpublished.",
    )
    return b.build()


def landsat8() -> FacetModel:
    """Landsat 8/9: USGS 3 m × 2.4 m, single 9 × 0.4 m array."""
    bus = (3.0, 2.4, 2.4)
    b = _Builder("landsat8")
    b.box((0, 0, 0), bus, MLI, "bus",
          mats=[DARK, MLI, MLI, MLI, MLI, WHITE_PAINT])
    b.panel((0, 2.4 / 2 + 0.2 + 4.5, 0), (1, 0, 0), (0, 1, 0), 0.4, 9.0,
            CELLS, PANEL_BACK, "array", gimbal=GIMBAL_1AXIS,
            gimbal_axis=(0, 1, 0))
    b.meta(
        family_id="landsat8",
        sources=("USGS Landsat 8: length 3 m, diameter 2.4 m, single "
                 "9 × 0.4 m solar array, ~2071 kg wet."),
        dimension_status={"bus": STATUS_PUBLIC, "arrays": STATUS_PUBLIC,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_NADIR, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Hydrazine; vector unpublished.",
    )
    return b.build()


def css_tianhe() -> FacetModel:
    """CSS Tianhe core module: 16.6 m × 4.2 m (CMSA / public fact sheets)."""
    b = _Builder("css_tianhe")
    b.prism((0, 0, 0), (1, 0, 0), 16.6, 4.2, 12, WHITE_PAINT, (MLI, MLI),
            "tianhe")
    for sign, tag in [(1, "+y"), (-1, "-y")]:
        b.panel((0, sign * 8.0, 0), (1, 0, 0), (0, sign, 0), 3.0, 12.0,
                CELLS, PANEL_BACK, f"arrays {tag}", gimbal=GIMBAL_1AXIS,
                gimbal_axis=(0, 1, 0))
    b.meta(
        family_id="css_tianhe",
        sources=("CMSA / public Tianhe facts: core module 16.6 m × 4.2 m "
                 "diameter. Array area is a range stand-in, not a drawing.",
                 "SATCAT name CSS (TIANHE). Not ISS."),
        notes="Coarse photometric stand-in of the core + wings. Wentian/"
              "Mengtian not modeled separately.",
        dimension_status={"module": STATUS_PUBLIC, "arrays": STATUS_RANGE,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Chinese station reboost exists; body-frame vector not "
                     "used as a public number (unlike ISS +x).",
    )
    return b.build()


def maxar_legion() -> FacetModel:
    """Maxar WorldView Legion: ~3 × 2 × 2 m, ~630 kg dry."""
    bus = (2.0, 2.0, 3.0)
    b = _Builder("maxar_legion")
    b.box((0, 0, 0), bus, MLI, "bus",
          mats=[DARK, MLI, MLI, MLI, MLI, WHITE_PAINT])
    _two_wing(b, bus, (2.0, 3.0), gimbal=GIMBAL_2AXIS)
    b.meta(
        family_id="maxar_legion",
        sources=("L3Harris / Maxar WorldView Legion data sheet: "
                 "~3 m tall × ~2 × 2 m (not including arrays), dry mass "
                 "~630 kg, 518 km."),
        notes="Array span is not on the data sheet (range).",
        dimension_status={"bus": STATUS_PUBLIC, "arrays": STATUS_RANGE,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_NADIR, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Δv typical of agile EO; vector unpublished.",
    )
    return b.build()


def worldview() -> FacetModel:
    """WorldView-1/2/3 class: 5.7 m × 2.5 m, 7.1 m array span."""
    bus = (2.5, 2.5, 5.7)
    b = _Builder("worldview")
    b.box((0, 0, 0), bus, MLI, "bus",
          mats=[DARK, MLI, MLI, MLI, MLI, WHITE_PAINT])
    _two_wing(b, bus, (2.5, 2.3), gimbal=GIMBAL_2AXIS)
    b.meta(
        family_id="worldview",
        sources=("Spaceflight101 / SIC WorldView-3: 5.7 m tall × 2.5 m, "
                 "7.1 m across deployed arrays, ~2800 kg."),
        dimension_status={"bus": STATUS_PUBLIC, "arrays": STATUS_PUBLIC,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_NADIR, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Vector unpublished.",
    )
    return b.build()


def bluebird(block2: bool = False) -> FacetModel:
    """AST SpaceMobile BlueBird. Block 1 = 64 m²; Block 2 = 223 m²."""
    side = 14.9 if block2 else 8.0
    name = "bluebird_block2" if block2 else "bluebird"
    b = _Builder(name)
    b.panel((0, 0, 0), (0, 1, 0), (1, 0, 0), side, side, ANTENNA, CELLS,
            "phased array")
    bus = (2.0, 2.0, 2.0) if block2 else (1.5, 1.5, 1.5)
    b.box((0, 0, 0.9 if not block2 else 1.2), bus, MLI, "bus")
    b.meta(
        family_id=name,
        sources=("AST SpaceMobile: BlueBird 1–5 = 693 ft² (64 m²) array; "
                 "next-gen / BlueBird 6+ = ~2400 ft² (223 m²).",
                 "Gunter BlueBird Block 1: ~1.5 t, 10 m-class 64 m² sheet.",
                 "Fixed array (BlueWalker-class), not a gimbal."),
        notes="SATCAT SPACEMOBILE-00n: n≤5 → Block 1, else Block 2.",
        dimension_status={"array": STATUS_PUBLIC, "bus": STATUS_RANGE,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_UNKNOWN,
        thrust_notes="Propulsion type/pointing not published as a vector.",
    )
    return b.build()


def terra() -> FacetModel:
    """NASA Terra / Aqua EOS flagship class."""
    bus = (6.8, 3.5, 3.5)
    b = _Builder("terra")
    b.box((0, 0, 0), bus, MLI, "bus",
          mats=[MLI, MLI, MLI, MLI, WHITE_PAINT, WHITE_PAINT])
    _two_wing(b, bus, (3.5, 5.0), gimbal=GIMBAL_1AXIS)
    b.meta(
        family_id="terra",
        sources=("NASA Terra facts: ~6.8 m length, ~3.5 m class cross-section. "
                 "Aqua (EOS PM) is the same class and maps here.",
                 "Array span is a range, not a copied Terra drawing."),
        dimension_status={"bus": STATUS_PUBLIC, "arrays": STATUS_RANGE,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_NADIR, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Hydrazine; vector unpublished.",
    )
    return b.build()


def jpss() -> FacetModel:
    """Suomi NPP / JPSS (NOAA 20/21)."""
    bus = (4.2, 2.6, 2.6)
    b = _Builder("jpss")
    b.box((0, 0, 0), bus, MLI, "bus",
          mats=[DARK, MLI, MLI, MLI, MLI, WHITE_PAINT])
    _two_wing(b, bus, (2.6, 4.0), gimbal=GIMBAL_1AXIS)
    b.meta(
        family_id="jpss",
        sources=("NOAA/NASA JPSS / Suomi NPP public facts: ~2–4 m class bus, "
                 "single-wing / two-wing solar. Exact OML treated as a range.",
                 "SATCAT: SUOMI NPP, NOAA 20 (JPSS-1), NOAA 21 (JPSS-2)."),
        dimension_status={"bus": STATUS_RANGE, "arrays": STATUS_RANGE,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_NADIR, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Vector unpublished.",
    )
    return b.build()


def galileo() -> FacetModel:
    """Galileo FOC: distinct from GPS-III-class gnss_meo."""
    bus = (2.7, 1.2, 1.1)
    b = _Builder("galileo")
    b.box((0, 0, 0), bus, MLI, "bus")
    _two_wing(b, bus, (1.2, 5.5), gimbal=GIMBAL_2AXIS)
    b.meta(
        family_id="galileo",
        sources=("ESA Galileo FOC public: ~730 kg class, box ~2.7×1.2×1.1 m, "
                 "two 2-axis wings (span a range around 10–15 m).",
                 "SATCAT GSAT0xxx / GALILEO. GPS/GLONASS/BeiDou stay on "
                 "gnss_meo — dims differ enough to split Galileo."),
        dimension_status={"bus": STATUS_RANGE, "arrays": STATUS_RANGE,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_YAW_STEER, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Chemical; vector unpublished.",
    )
    return b.build()


def cygnus() -> FacetModel:
    """Northrop Grumman Cygnus Enhanced: ~6.3 m × 3.07 m."""
    b = _Builder("cygnus")
    b.prism((0, 0, 0), (1, 0, 0), 6.3, 3.07, 12, MLI, (WHITE_PAINT, MLI),
            "cygnus")
    b.meta(
        family_id="cygnus",
        sources=("Northrop Grumman / NASA Cygnus Enhanced: PCM+SM ~6.3 m, "
                 "3.07 m diameter. Mapped if it appears as an active payload."),
        dimension_status={"stage": STATUS_PUBLIC, "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Service-module Δv; vector not encoded (visiting vehicle).",
    )
    return b.build()


def progress() -> FacetModel:
    """Roscosmos Progress: ~7.2 m × 2.72 m."""
    b = _Builder("progress")
    b.prism((0, 0, 0), (1, 0, 0), 7.2, 2.72, 12, MLI, (WHITE_PAINT, MLI),
            "progress")
    b.meta(
        family_id="progress",
        sources=("Public Progress-MS: ~7.2 m length, 2.72 m diameter.",),
        dimension_status={"stage": STATUS_PUBLIC,
                          "thrust_vector": STATUS_PUBLIC},
        thrust_body=[[1.0, 0.0, 0.0]],
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="ISS reboost along +x when docked (same public convention).",
    )
    return b.build()


def goes_r() -> FacetModel:
    """GOES-R series (GOES 16–19): NESDIS 6.1 × 5.6 × 3.9 m + one 5-panel wing."""
    bus = (6.1, 5.6, 3.9)
    b = _Builder("goes_r")
    b.box((0, 0, 0), bus, MLI, "bus",
          mats=[MLI, MLI, MLI, MLI, WHITE_PAINT, WHITE_PAINT])
    # Five panels, each ~13 × 4.5 ft (3.96 × 1.37 m), one 1-axis wing.
    b.panel((0, 5.6 / 2 + 0.4 + 6.85 / 2, 0), (1, 0, 0), (0, 1, 0),
            3.96, 6.85, CELLS, PANEL_BACK, "array", gimbal=GIMBAL_1AXIS,
            gimbal_axis=(0, 1, 0))
    b.meta(
        family_id="goes_r",
        sources=("NESDIS GOES-R Series spacecraft: 6.1 × 5.6 × 3.9 m "
                 "(20 × 18.4 × 12.8 ft).",
                 "NOAA/NASA GOES-T array test: five panels, each ~13 × 4.5 ft, "
                 "single wing rotating once per day."),
        notes="SATCAT GOES 16–19. Older GOES / EWS-G stay on geo_bus.",
        dimension_status={"bus": STATUS_PUBLIC, "arrays": STATUS_PUBLIC,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_NADIR, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="NSSK exists; a single body-frame vector is not encoded "
                     "(unlike the geo_bus +y class convention).",
    )
    return b.build()


def sentinel6() -> FacetModel:
    """Sentinel-6 / Jason-CS: EUMETSAT 5.13 × 4.17 × 2.34 m in-orbit."""
    bus = (5.13, 2.34, 4.17)
    b = _Builder("sentinel6")
    b.box((0, 0, 0), bus, MLI, "bus",
          mats=[DARK, MLI, MLI, MLI, MLI, WHITE_PAINT])
    _two_wing(b, bus, (2.34, 2.0), gimbal=GIMBAL_1AXIS)
    b.meta(
        family_id="sentinel6",
        sources=("EUMETSAT Sentinel-6 factsheet: 5.13 × 4.17 × 2.34 m "
                 "in-orbit configuration, ~1192 kg.",
                 "Jason-3 maps here as the same altimetry class "
                 "(typical_class), not a copied S6 drawing."),
        dimension_status={"bus": STATUS_PUBLIC, "arrays": STATUS_RANGE,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_NADIR, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Hydrazine; vector unpublished.",
    )
    return b.build()


def metop() -> FacetModel:
    """MetOp-B/C: ESA 6.3 × 2.5 × 2.5 m launch + 8 × (1×5 m) single wing."""
    bus = (6.3, 2.5, 2.5)
    b = _Builder("metop")
    b.box((0, 0, 0), bus, MLI, "bus",
          mats=[DARK, MLI, MLI, MLI, MLI, WHITE_PAINT])
    b.panel((0, 2.5 / 2 + 0.4 + 4.0, 0), (1, 0, 0), (0, 1, 0),
            5.0, 8.0, CELLS, PANEL_BACK, "array", gimbal=GIMBAL_1AXIS,
            gimbal_axis=(0, 1, 0))
    b.meta(
        family_id="metop",
        sources=("ESA MetOp spacecraft specifications: launch 6.3 m × 2.5 × 2.5 m; "
                 "in-orbit 17.6 × 6.6 × 5.0 m; eight 1 × 5 m panels, 4085 kg.",
                 "MetOp-SG is a different vehicle and is not this family."),
        dimension_status={"bus": STATUS_PUBLIC, "arrays": STATUS_PUBLIC,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_NADIR, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Hydrazine; vector unpublished.",
    )
    return b.build()


def cargo_dragon() -> FacetModel:
    """SpaceX Cargo Dragon 2: trunk+capsule ~8.1 m × 4.0 m class."""
    b = _Builder("cargo_dragon")
    b.prism((0, 0, 0), (1, 0, 0), 8.1, 4.0, 12, WHITE_PAINT, (DARK, MLI),
            "dragon")
    b.meta(
        family_id="cargo_dragon",
        sources=("SpaceX Dragon 2 public: 4.0 m diameter, capsule+trunk "
                 "length ~8.1 m class."),
        notes="Not Crew Dragon interiors. Visiting vehicle; rarely a free-flyer "
              "in the active SATCAT.",
        dimension_status={"stage": STATUS_PUBLIC, "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Draco clusters; a single body-frame vector is not used.",
    )
    return b.build()


# ---------------------------------------------------------------------------
# Upper stages
# ---------------------------------------------------------------------------

def centaur() -> FacetModel:
    return _stage(
        "centaur", 12.68, 3.05,
        sources=("ULA / Wikipedia Centaur III (Common Centaur): 3.05 m "
                 "diameter, 12.68 m length. Vulcan Centaur V is 5.4 m and "
                 "is the same name-family with diameter tagged range.",),
        notes="SATCAT ATLAS/VULCAN/TITAN … CENTAUR R/B. Stand-in is CIII.",
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_PUBLIC},
    )


def fregat() -> FacetModel:
    return _stage(
        "fregat", 1.88, 3.44,
        sources=("Wikipedia / Lavochkin Fregat: 3.44 m diameter, 1.875 m "
                 "height (Fregat-MT/SB slightly taller — range).",),
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_RANGE},
    )


def soyuz_block_i() -> FacetModel:
    return _stage(
        "soyuz_block_i", 6.7, 2.66,
        sources=("Soyuz Block I / improved Block I: 2.66 m diameter, 6.7 m "
                 "length (public launcher descriptions).",
                 "Maps SL-3/SL-4/SL-6 (R-7 family) lingering stages."),
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_PUBLIC},
    )


def proton_block_d() -> FacetModel:
    return _stage(
        "proton_block_d", 5.5, 4.10,
        sources=("Proton Block D/DM: 4.1 m diameter class, length ~5.5 m "
                 "(public Block D figures). SATCAT SL-12 R/B.",),
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_RANGE},
    )


def kosmos_3m() -> FacetModel:
    return _stage(
        "kosmos_3m", 6.0, 2.40,
        sources=("Kosmos-3M second stage: 2.4 m diameter (public). Length "
                 "is a range. SATCAT SL-8 R/B.",),
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_RANGE},
    )


def tsyklon3() -> FacetModel:
    return _stage(
        "tsyklon3", 3.0, 2.50,
        sources=("Tsyklon-3 third stage: ~2.5 m diameter class. SATCAT SL-14.",),
        notes="Diameter is typical_class of the Tsyklon stack.",
        dimension_status={"diameter": STATUS_TYPICAL, "length": STATUS_RANGE},
    )


def zenit2() -> FacetModel:
    return _stage(
        "zenit2", 10.4, 3.90,
        sources=("Zenit-2 second stage: 3.9 m diameter, ~10.4 m length "
                 "(public Zenit figures). SATCAT SL-16.",),
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_RANGE},
    )


def pslv_ps4() -> FacetModel:
    return _stage(
        "pslv_ps4", 3.0, 2.80,
        sources=("ISRO PSLV PS4: 2.8 m diameter (public). Length is a range. "
                 "SATCAT PSLV R/B.",),
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_RANGE},
    )


def h2_upper() -> FacetModel:
    return _stage(
        "h2_upper", 9.2, 4.00,
        sources=("JAXA H-IIA second stage: 4.0 m diameter (public). Length "
                 "is a range. SATCAT H-2 / H-2A R/B.",),
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_RANGE},
    )


def block_dm() -> FacetModel:
    return _stage(
        "block_dm", 6.3, 3.70,
        sources=("Block DM (Sea Launch / Proton): ~3.7 m diameter class. "
                 "SATCAT BLOCK DM-SL R/B.",),
        dimension_status={"diameter": STATUS_RANGE, "length": STATUS_RANGE},
    )


def delta_upper() -> FacetModel:
    return _stage(
        "delta_upper", 6.0, 2.40,
        sources=("Delta II/III upper stages are 2.4 m class; Delta IV 4 m. "
                 "Stand-in is 2.4 m typical_class, not a named Delta-K CAD.",),
        notes="SATCAT DELTA n R/B. Diameter tagged typical_class.",
        dimension_status={"diameter": STATUS_TYPICAL, "length": STATUS_RANGE},
    )


def ius() -> FacetModel:
    return _stage(
        "ius", 5.18, 2.80,
        sources=("NASA STS-30 press kit / Chandra IUS fact sheet: 17 ft "
                 "(5.18 m) long, 9.25 ft (2.8 m) diameter.",),
        notes="SATCAT IUS R/B(1)/(2).",
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_PUBLIC},
    )


def agena() -> FacetModel:
    return _stage(
        "agena", 7.09, 1.52,
        sources=("Lockheed Agena-D / public NASA-Gemini figures: 60 in "
                 "(1.52 m) diameter; length 6.3–7.09 m (range).",
                 "SATCAT ATLAS/THOR/TITAN … AGENA R/B."),
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_RANGE},
    )


def scout() -> FacetModel:
    return _stage(
        "scout", 2.90, 0.78,
        sources=("NASA Scout fact sheet / public stage table: Antares "
                 "(stage 3) 0.78 m × 2.90 m; Altair (stage 4) 0.64 m. "
                 "Stand-in is Antares, tagged typical_class — SATCAT does "
                 "not say which stage lingered.",),
        notes="SATCAT SCOUT A / X-4 / G-1 R/B.",
        dimension_status={"diameter": STATUS_TYPICAL, "length": STATUS_TYPICAL},
    )


def pegasus() -> FacetModel:
    return _stage(
        "pegasus", 3.11, 1.27,
        sources=("Northrop Grumman Pegasus XL user's guide: 1.27 m "
                 "principal diameter; stage 2 Orion 50XL ~3.11 m. "
                 "Stage 3 is 0.97 m — same name family, diameter a range.",),
        notes="SATCAT PEGASUS R/B. Wings not modeled (spent stage).",
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_RANGE},
    )


def pam_d() -> FacetModel:
    return _stage(
        "pam_d", 2.04, 1.24,
        sources=("Thiokol STAR 48 / McDonnell Douglas PAM-D: 1.24 m "
                 "diameter, ~2.04 m length (NASA TM / public motor table).",
                 "SATCAT … R/B [PAM-D], ATLAS 5 STAR 48-B R/B."),
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_PUBLIC},
    )


def titan_transtage() -> FacetModel:
    return _stage(
        "titan_transtage", 4.57, 3.05,
        sources=("NASA Titan IIIC / public Transtage: 10 ft (3.05 m) "
                 "diameter, ~14.5–15 ft (4.57 m) length.",
                 "SATCAT TITAN 3C TRANSTAGE. Other Titan cores are not this."),
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_PUBLIC},
    )
