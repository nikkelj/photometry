"""Catalog family templates. Photometric stand-ins from public dimensions.

Each factory returns a FacetModel with bus, arrays (gimbal mode explicit),
optional deployables, thrust vs nominal attitude, and per-facet materials.
Unknown quantities are tagged `uncertain` / `unknown` / `range` — never a
fake precise number presented as measured.
"""

from __future__ import annotations

from ..shapes import (
    ANTENNA,
    ATT_LVLH,
    ATT_NADIR,
    ATT_STAGE_AXIS,
    ATT_UNKNOWN,
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
    PROP_EP,
    PROP_NONE,
    PROP_UNKNOWN,
    STATUS_PUBLIC,
    STATUS_RANGE,
    STATUS_TYPICAL,
    STATUS_UNCERTAIN,
    STATUS_UNKNOWN,
    WHITE_PAINT,
    FacetModel,
    _Builder,
    box_wing,
)


def _two_wing(b: _Builder, bus_xyz, panel_wh, *, gimbal, gimbal_axis=(0, 1, 0),
              boom=0.3) -> None:
    """±y solar wings. panel_wh = (chord along x, span along y)."""
    bx, by, _bz = bus_xyz
    w, h = panel_wh
    for sign, tag in [(1, "+y"), (-1, "-y")]:
        cy = sign * (by / 2 + boom + h / 2)
        b.panel((0, cy, 0), (1, 0, 0), (0, sign, 0), w, h, CELLS, PANEL_BACK,
                f"array {tag}", gimbal=gimbal, gimbal_axis=gimbal_axis)


def starlink_v2() -> FacetModel:
    """Starlink v2 (Starship-class) from the same public FCC table as v2 Mini.

    Not mapped from SATCAT names: Falcon-9 Starlinks are v1.5 / v2 Mini, and
    Starship-dispensed vehicles are not distinguished in the public name
    field. Family exists so a reviewer can load the published 6.4×2.7 m bus
    / 6.4×20.2 m arrays. Thrust pointing unpublished.
    """
    b = _Builder("starlink_v2")
    b.box((0, 0, 0), (6.4, 2.7, 0.3), MLI, "bus",
          mats=[MLI, MLI, MLI, MLI, MLI, ANTENNA])
    for sign, tag in [(1, "fore"), (-1, "aft")]:
        cx = sign * (6.4 / 2 + 0.4 + 20.2 / 2)
        b.panel((cx, 0, 0), (sign, 0, 0), (0, sign, 0), 20.2, 6.4, CELLS,
                PANEL_BACK, f"array {tag}", gimbal=GIMBAL_2AXIS)
    b.meta(
        family_id="starlink_v2",
        sources=("McDowell FCC Gen2 table: V2 bus 6.4×2.7 m, array 6.4×20.2 m "
                 "(https://planet4589.org/astro/starsim/index.html).",
                 "SpaceX Gen2 public PDF. Thrust pointing unpublished."),
        notes="Starship-class v2. Not used as a SATCAT default. No Starshield.",
        dimension_status={"bus": STATUS_PUBLIC, "arrays": STATUS_PUBLIC,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_EP,
        thrust_notes="EP public; body-frame vector not published.",
    )
    return b.build()


def oneweb() -> FacetModel:
    """Airbus Arrow / OneWeb: ~1×1×1.3 m bus, two wings, 5 m tip-to-tip."""
    bus = (1.3, 1.0, 1.0)
    b = _Builder("oneweb")
    b.box((0, 0, 0), bus, MLI, "bus",
          mats=[MLI, MLI, MLI, MLI, MLI, ANTENNA])
    # KeepTrack ARROW span 5 m: 1.0 m bus + 2×0.3 m boom + 2×1.7 m wings.
    # Earlier 2.0 m wings produced 5.6 m tip-to-tip (too long vs the cite).
    # ±y wings, hinge along +y (boom): 1-axis, out-of-plane cosine loss remains.
    _two_wing(b, bus, (1.0, 1.7), gimbal=GIMBAL_1AXIS)
    b.meta(
        family_id="oneweb",
        sources=("eoPortal OneWeb: Airbus Arrow, ~150 kg, box ~1×1×1.3 m, "
                 "two deployable solar panels, xenon electric propulsion.",
                 "KeepTrack ARROW bus: span 5 m; SPT-50 or BHT-350 Hall.",
                 ),
        notes="Ku/Ka user/gateway antennas are the nadir (−z) bus face, not a "
              "separate unfurled dish (size of the phased arrays unpublished). "
              "Arrays are 1-axis (sun-tracking wings on a nadir-hold bus); "
              "2-axis is not in the public Arrow/OneWeb record. Cell area "
              "is a range (chord matches the 1 m bus face; span is the "
              "5 m tip-to-tip cite minus booms).",
        dimension_status={"bus": STATUS_PUBLIC, "arrays": STATUS_RANGE,
                          "array_span": STATUS_PUBLIC,
                          "array_gimbal": STATUS_PUBLIC,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_EP,
        thrust_notes="Searched Busek/SpaceNews (BHT-350 for EOR, SK, CA, "
                     "deorbit) and Gunter (SPT-50/BHT-350). No body-frame "
                     "vs ram/nadir published — vector left empty.",
        flight_attitude=ATT_LVLH,
    )
    return b.build().set_look(
        "bus -z", (0, 0, -1), attitude=ATT_LVLH,
        notes="Ku/Ka user face; Earth-pointing service (nadir in LVLH).",
        status=STATUS_TYPICAL).ensure_mirror_polygons()


def kuiper() -> FacetModel:
    """Amazon Leo / Project Kuiper. Mass is public; production OML is not.

    Photometric stand-in uses published-range midpoints, never a fake
    'exact' bus. Two wings; 1-axis is a stand-in (1- vs 2-axis unpublished).
    """
    # KeepTrack Kuiper-P1 protoflight: ~2 m box, 10 m span (secondary).
    # Production stacking photos are the same 2 m / 10 m class.
    bus = (2.0, 1.6, 0.7)
    panel = (2.0, 3.9)             # 1.6 + 2×0.3 boom + 2×3.9 = 10.0 m span
    b = _Builder("kuiper")
    b.box((0, 0, 0), bus, MLI, "bus",
          mats=[MLI, MLI, MLI, MLI, MLI, ANTENNA])
    _two_wing(b, bus, panel, gimbal=GIMBAL_1AXIS)
    b.meta(
        family_id="kuiper",
        sources=("eoPortal / ULA first-deployment mass: ~537–571 kg each; "
                 "trapezoidal bus + deployable arrays; krypton Hall thruster.",
                 "FCC Kuiper System order (2020) and subsequent public PDFs "
                 "give orbit shells, not outer-mold-line drawings.",
                 "KeepTrack KUIPER-P1 protoflight (secondary): ~2 m box, "
                 "10 m span. Not a production drawing.",
                 "SATCAT names KUIPER-#####."),
        notes="Bus ~2 m and 10 m span are range midpoints (protoflight / "
              "stacking class), not a filed OML. Gimbal 1-axis is a "
              "stand-in: 1- vs 2-axis is not in the public record. Nadir "
              "(−z) is the user-antenna face (phased-array size unpublished). "
              "No invented internals.",
        dimension_status={"bus": STATUS_RANGE, "arrays": STATUS_RANGE,
                          "array_gimbal": STATUS_UNCERTAIN,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_EP,
        thrust_notes="Searched FCC Kuiper PDFs and eoPortal (krypton Hall "
                     "exists). No thrust-axis vs ram/nadir/sun published — "
                     "vector left empty.",
        flight_attitude=ATT_LVLH,
    )
    return b.build().set_look(
        "bus -z", (0, 0, -1), attitude=ATT_LVLH,
        notes="User phased-array face is Earth-pointing in public imagery; "
              "exact body-frame boresight unpublished — −z is the stand-in "
              "nadir face, tagged typical_class.",
        status=STATUS_TYPICAL).ensure_mirror_polygons()


def qianfan() -> FacetModel:
    """Qianfan / Thousand Sails / G60: flat-pack, single 1-axis array."""
    bus = (3.0, 1.4, 0.3)
    b = _Builder("qianfan")
    b.box((0, 0, 0), bus, MLI, "bus",
          mats=[MLI, MLI, MLI, MLI, MLI, ANTENNA])
    # 10 m reported span, single wing off +x.
    cx = 3.0 / 2 + 0.3 + 6.7 / 2
    b.panel((cx, 0, 0), (1, 0, 0), (0, 1, 0), 6.7, 2.0, CELLS, PANEL_BACK,
            "array", gimbal=GIMBAL_1AXIS, gimbal_axis=(1, 0, 0))
    b.meta(
        family_id="qianfan",
        sources=("Mallama et al. arXiv:2505.07194: single articulated array "
                 "rotated to track the Sun; Earth-facing bus photometry.",
                 "Public imagery / China-in-space: ~300 kg flat-pack, krypton "
                 "Hall ~20 mN. KeepTrack: 3.0 m × 0.3 m, 10 m span.",
                 "Gunter/KeepTrack numbers are observer-scale, not a filing."),
        notes="Bus width 1.4 m and array 6.7×2.0 m are range midpoints that "
              "reproduce the public 10 m span and 0.3 m stack thickness.",
        dimension_status={"bus_length": STATUS_PUBLIC, "bus_thickness": STATUS_PUBLIC,
                          "bus_width": STATUS_RANGE, "array": STATUS_RANGE,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_EP,
        thrust_notes="Krypton Hall is reported; pointing vs LVLH is not published.",
    )
    return b.build()


def hulianwang() -> FacetModel:
    """Guowang / Hulianwang Digui. Gunter: outer dimensions not published.

    Secondary databases quote ~3×1 m box, 10 m span, ~800 kg, two panels.
    Encoded as uncertain range, not as a precise vehicle.
    """
    bus = (3.0, 1.0, 1.0)
    b = _Builder("hulianwang")
    b.box((0, 0, 0), bus, MLI, "bus",
          mats=[MLI, MLI, MLI, MLI, MLI, ANTENNA])
    _two_wing(b, bus, (1.0, 4.2), gimbal=GIMBAL_1AXIS)
    b.meta(
        family_id="hulianwang",
        sources=("Gunter HWD Type 1/2: mass, power, platform dimensions "
                 "not published; EP and phased arrays believed present.",
                 "KeepTrack quotes 3×1 m, 10 m span, ~800 kg (secondary; "
                 "treated as a range, not a drawing)."),
        notes="1-axis two-wing stand-in. Gimbal axis and exact OML uncertain.",
        dimension_status={"bus": STATUS_UNCERTAIN, "arrays": STATUS_UNCERTAIN,
                          "array_gimbal": STATUS_UNCERTAIN,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_EP,
        thrust_notes="EP believed present (Gunter); pointing unpublished.",
    )
    return b.build()


def iridium_next() -> FacetModel:
    """Iridium NEXT / Thales ELiTeBus: 9.4 m array span, 1-axis sun tracking."""
    bus = (3.1, 1.5, 1.5)
    b = _Builder("iridium_next")
    b.box((0, 0, 0), bus, MLI, "bus",
          mats=[MLI, MLI, MLI, MLI, MLI, ANTENNA])
    # 9.4 m span − 1.5 m bus = 7.9 m / 2 wings; four-panel wings, ~1.8 m chord.
    _two_wing(b, bus, (1.8, 3.95), gimbal=GIMBAL_1AXIS)
    b.meta(
        family_id="iridium_next",
        sources=("Iridium 2014-09-23: NEXT array 9.4 m span, >2 kW, four "
                 "panels, sun-tracking.",
                 "Thales ELiTeBus-1000 / public NEXT descriptions: hydrazine "
                 "1 N class thrusters (chemical), not Hall.",
                 "KeepTrack: 3.1 m × 1.5 m, 9.4 m span, ~860 kg launch."),
        notes="Main mission antenna is the L-band panel on −z (bus face).",
        dimension_status={"bus": STATUS_RANGE, "arrays": STATUS_PUBLIC,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Searched FCC SAT-MOD-20131227-00148 engineering "
                     "statement (nadir-pointing service; hydrazine 1 N class, "
                     "eight thrusters). No single body-frame Δv vector — "
                     "left empty.",
    )
    return b.build()


def planet_superdove() -> FacetModel:
    """Planet SuperDove / Flock: 3U (10×10×30 cm) with two small fixed wings."""
    bus = (0.30, 0.10, 0.10)
    b = _Builder("planet_superdove")
    b.box((0, 0, 0), bus, DARK, "bus")
    _two_wing(b, bus, (0.30, 0.10), gimbal=GIMBAL_FIXED, boom=0.02)
    b.meta(
        family_id="planet_superdove",
        sources=("Planet / ESA PlanetScope: Dove/SuperDove 3U, 10×10×30 cm, "
                 "~5.8 kg.",
                 "USGS OFR 2021-1030-F SuperDove characterization.",
                 "KeepTrack Flock: box + two panels."),
        notes="Deployed wings are body-fixed (GIMBAL_FIXED). Many Doves use "
              "differential drag rather than a published Δv thruster.",
        dimension_status={"bus": STATUS_PUBLIC, "arrays": STATUS_RANGE,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_NADIR, thrust_propulsion=PROP_NONE,
        thrust_notes="No public operational Δv thruster on SuperDove.",
    )
    return b.build()


def planet_skysat() -> FacetModel:
    """Planet SkySat-C class: 60×60×95 cm bus, two 1-axis wings."""
    bus = (0.95, 0.60, 0.60)
    b = _Builder("planet_skysat")
    b.box((0, 0, 0), bus, MLI, "bus",
          mats=[DARK, MLI, MLI, MLI, MLI, DARK])  # +x aperture, −z radiator-ish
    _two_wing(b, bus, (0.60, 0.80), gimbal=GIMBAL_1AXIS, boom=0.15)
    b.meta(
        family_id="planet_skysat",
        sources=("ESA SkySat / Planet product spec: 60×60×95 cm, ~110 kg, "
                 "180 m/s Δv.",
                 "eoPortal: SkySat-1/2 were 60×60×80 cm with body-mounted "
                 "cells; later SSL-built units have deployable arrays."),
        notes="1-axis is a stand-in for the SSL-built deployable wings. "
              "Telescope looks +x (agile pointing, not LVLH-hold).",
        dimension_status={"bus": STATUS_PUBLIC, "arrays": STATUS_RANGE,
                          "array_gimbal": STATUS_UNCERTAIN,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_NADIR, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="180 m/s Δv is public; thruster pointing is not.",
    )
    return b.build()


def iceye() -> FacetModel:
    """ICEYE X-band SAR: washing-machine bus + 3.2×0.4 m along-track antenna."""
    bus = (0.80, 0.70, 0.70)
    b = _Builder("iceye")
    b.box((0, 0, 0), bus, MLI, "bus")
    _two_wing(b, bus, (0.70, 0.90), gimbal=GIMBAL_1AXIS, boom=0.15)
    # Deployed SAR: along +x, nadir-facing (−z). Fixed after deploy.
    cx = 0.80 / 2 + 0.1 + 3.2 / 2
    b.panel((cx, 0, -0.15), (0, 1, 0), (1, 0, 0), 0.40, 3.2, ANTENNA, MLI,
            "sar antenna")
    b.meta(
        family_id="iceye",
        sources=("ICEYE product guide: X-band AESA 3.2 m along-track × 0.4 m, "
                 "mass ~85–110 kg depending on generation.",
                 "Bus is a public 'washing-machine' class; exact OML range."),
        notes="SAR is GIMBAL_FIXED once deployed. Side-looking ops, not a "
              "gimbaled dish.",
        dimension_status={"bus": STATUS_RANGE, "sar": STATUS_PUBLIC,
                          "arrays": STATUS_RANGE, "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_UNKNOWN,
        thrust_notes="Propulsion type/pointing not used as a public number.",
        flight_attitude=ATT_LVLH,
    )
    return b.build().set_look(
        "sar", None, attitude=ATT_LVLH,
        notes="Side-looking X-band AESA. Off-nadir look angle is not in "
              "the product-guide public numbers — not invented. Rest face "
              "in this stand-in is −z only as a placed panel, not an ops look.",
        status=STATUS_UNKNOWN)


def _cubesat(family_id: str, xyz, name: str) -> FacetModel:
    b = _Builder(name)
    b.box((0, 0, 0), xyz, DARK, "bus")
    _two_wing(b, xyz, (xyz[0], xyz[1]), gimbal=GIMBAL_FIXED, boom=0.02)
    b.meta(
        family_id=family_id,
        sources=("CubeSat Design Specification (Cal Poly): 1U = 10×10×10 cm. "
                 "3U/6U/16U envelopes are the public form factor, not a "
                 "named mission drawing."),
        notes="Wings are a typical deployed-cell stand-in, GIMBAL_FIXED. "
              "Many flight 3U/6U have no propulsion.",
        dimension_status={"bus": STATUS_PUBLIC, "arrays": STATUS_TYPICAL,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_UNKNOWN, thrust_propulsion=PROP_UNKNOWN,
        thrust_notes="Propulsion is mission-specific and usually unpublished.",
    )
    return b.build()


def cubesat_3u() -> FacetModel:
    return _cubesat("cubesat_3u", (0.30, 0.10, 0.10), "cubesat_3u")


def cubesat_6u() -> FacetModel:
    return _cubesat("cubesat_6u", (0.30, 0.20, 0.10), "cubesat_6u")


def cubesat_16u() -> FacetModel:
    return _cubesat("cubesat_16u", (0.40, 0.20, 0.20), "cubesat_16u")


def geo_bus() -> FacetModel:
    """Typical 3-axis GEO comms bus (A2100 / Eurostar / 1300 class).

    Dimensions are class ranges, not one vehicle. 2-axis arrays; nadir dish.
    N/S faces are white radiators (GEO class convention); other bus faces MLI.
    """
    bus = (2.5, 2.0, 3.2)
    b = _Builder("geo_bus")
    # box mats: +x −x +y −y +z −z. ±y = N/S radiators on a nadir-hold bus.
    b.box((0, 0, 0), bus, MLI, "bus",
          mats=[MLI, MLI, WHITE_PAINT, WHITE_PAINT, MLI, MLI])
    # Mid-class 15–30 m span ≈ 22.8 m: 2.0 + 2×0.4 boom + 2×10.0 wings.
    _two_wing(b, bus, (3.2, 10.0), gimbal=GIMBAL_2AXIS, boom=0.4)
    b.panel((0, 0, -3.2 / 2 - 0.4), (0, 1, 0), (1, 0, 0), 2.5, 2.5,
            ANTENNA, MLI, "nadir dish",
            look_body=(0, 0, -1), look_attitude=ATT_NADIR,
            look_notes="3-axis GEO class: unfurlable / Gregorian Earth face.",
            look_status=STATUS_TYPICAL)
    b.meta(
        family_id="geo_bus",
        sources=("Class convention for 3-axis GEO comms: box bus, 2-axis "
                 "solar wings, nadir unfurlable / Gregorian antenna, N/S "
                 "radiator faces. Not a named SSL-1300/A2100 drawing.",
                 "N/S station-keeping is conventionally orbit-normal."),
        notes="Bus 2–4 m, array span 15–30 m (stand-in 22.8 m), dish 2–3 m "
              "are typical ranges. 2-axis arrays are the GEO class default. "
              "Do not treat as a specific GEO.",
        dimension_status={"bus": STATUS_TYPICAL, "arrays": STATUS_TYPICAL,
                          "dish": STATUS_TYPICAL,
                          "thrust_vector": STATUS_TYPICAL},
        thrust_body=[[0.0, 1.0, 0.0]],
        thrust_attitude=ATT_NADIR, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Primary NSSK along +y (orbit normal) is a GEO class "
                     "convention for chemical/EP station-keeping, not a "
                     "measured Hall vector on a named satellite. E/W (±x) "
                     "also exists. Not invented ion pointing.",
        flight_attitude=ATT_NADIR,
    )
    return b.build().ensure_mirror_polygons()


def gnss_meo() -> FacetModel:
    """GPS III / GNSS MEO class: Earth-nadir, 2-axis arrays, yaw-steering."""
    bus = (3.4, 1.8, 1.8)
    b = _Builder("gnss_meo")
    b.box((0, 0, 0), bus, MLI, "bus")
    # Lockheed GPS III fact sheet: 27 m span. KeepTrack quotes 14 m — conflict
    # recorded; stand-in uses the Lockheed public 27 m figure.
    _two_wing(b, bus, (1.8, 12.6), gimbal=GIMBAL_2AXIS, boom=0.4)
    b.meta(
        family_id="gnss_meo",
        sources=("Lockheed GPS III public: ~27 m solar-array span, A2100-class.",
                 "KeepTrack NAVSTAR 79/82: 3.4×1.8 m, 14 m span (conflicts "
                 "with 27 m — span treated as a range, stand-in 27 m).",
                 "GLONASS/BeiDou MEO stay on this class; Galileo FOC is "
                 "the separate `galileo` family (dims differ)."),
        notes="Array span public-number conflict 14 vs 27 m. Yaw-steering "
              "attitude, not LVLH ram.",
        dimension_status={"bus": STATUS_RANGE, "arrays": STATUS_RANGE,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_YAW_STEER, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Chemical RCS/Δv typical of GNSS; body-frame vector omitted.",
    )
    return b.build()


def o3b() -> FacetModel:
    """O3b (original) MEO comms: small GEO-like, 2-axis arrays."""
    bus = (1.8, 1.5, 1.5)
    b = _Builder("o3b")
    b.box((0, 0, 0), bus, MLI, "bus")
    _two_wing(b, bus, (1.5, 3.5), gimbal=GIMBAL_2AXIS)
    b.panel((0, 0, -1.5 / 2 - 0.3), (0, 1, 0), (1, 0, 0), 1.2, 1.2,
            ANTENNA, MLI, "nadir antenna")
    b.meta(
        family_id="o3b",
        sources=("Thales Alenia / SES O3b public: ~700 kg class MEO comms "
                 "with two sun-tracking wings and nadir user antennas. Exact "
                 "OML is a range."),
        notes="O3b mPOWER is a different, larger vehicle not modeled here.",
        dimension_status={"bus": STATUS_RANGE, "arrays": STATUS_RANGE,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_NADIR, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Chemical station-keeping typical; vector unpublished.",
    )
    return b.build()


def starlink_v15_fcc() -> FacetModel:
    """SATCAT Starlink v1.5 from the cited FCC table.

    `shapes.starlink_v15()` is the 620 km study LIBRARY copy and is not
    changed. That study mesh used bus y=1.4 m and array chord 2.7 m; the
    McDowell FCC table is bus 2.8×1.3 m and array 2.8×8.1 m. Catalog
    mapping uses this FCC stand-in.
    """
    b = _Builder("starlink_v15")
    b.box((0, 0, 0), (2.8, 1.3, 0.2), MLI, "bus",
          mats=[MLI, MLI, MLI, MLI, MLI, ANTENNA])
    cx = 2.8 / 2 + 0.3 + 8.1 / 2
    b.panel((cx, 0, 0), (1, 0, 0), (0, 1, 0), 8.1, 2.8, CELLS, PANEL_BACK,
            "array", gimbal=GIMBAL_1AXIS, gimbal_axis=(1, 0, 0))
    b.meta(
        family_id="starlink_v15",
        sources=(
            "Jonathan McDowell, public FCC Gen2 dimensions table: v1.5 bus "
            "2.8×1.3 m, array 2.8×8.1 m, mass 303 kg "
            "(https://planet4589.org/astro/starsim/index.html).",
            "Spaceflight Now 2023-02-26: single ~11 m end-to-end wing on v1.5.",
            "SpaceX public: krypton/argon ion (Hall-class) propulsion; "
            "body-frame thrust pointing is not published — left unknown.",
            "SpaceX Brightness Mitigation Best Practices PDF: VisorSat "
            "shade (no published size/angle), later dielectric film on "
            "the nadir face, knife-edge / terminator-tracking conops. "
            "Mallama et al. arXiv:2309.14152. Not meshed — no cited "
            "visor CAD or published gimbal angles.",
        ),
        notes="FCC table (catalog). Study LIBRARY `shapes.starlink_v15` "
              "keeps the earlier 1.4 / 2.7 m stand-in so 620 km inversion "
              "stays reproducible. v1.0 lumped with v1.5. 1-axis shoulder "
              "gimbal. Nadir (−z) is the antenna panel. Cells vs panel-back "
              "split. Do not treat as Starshield CAD. Visor / film / "
              "knife-edge are public practices, not meshable geometry.",
        dimension_status={"bus": STATUS_PUBLIC, "array": STATUS_PUBLIC,
                          "array_gimbal": STATUS_PUBLIC,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_EP,
        thrust_notes="Searched SpaceX Gen2 PDF, FCC dimension table, "
                     "SpaceNews/Spaceflight Now (krypton Hall exists; "
                     "magnitude only). Everyday Astronaut claims ram-facing "
                     "but is not a primary SpaceX/FCC citation — vector left "
                     "empty.",
        flight_attitude=ATT_LVLH,
    )
    return b.build().set_look(
        "bus -z", (0, 0, -1), attitude=ATT_LVLH,
        notes="User phased-array face (Earth / nadir in LVLH-hold).",
        status=STATUS_PUBLIC).ensure_mirror_polygons()


def iss_nasa() -> FacetModel:
    """SATCAT ISS stand-in using NASA public areas.

    `shapes.iss()` is the 620 km study LIBRARY copy (two 35×24 m array
    groups = 1,680 m² cells) and is not changed. NASA cites ~2,500 m²
    of USOS array area and a ~109 m truss; catalog mapping uses that.
    """
    b = _Builder("iss")
    b.box((0, 0, 0), (50, 6, 6), WHITE_PAINT, "modules")
    b.box((0, 0, 4.5), (5, 109, 3), MLI, "truss")
    for sign, tag in [(1, "stbd"), (-1, "port")]:
        # 4 USOS wings of ~35×12 m per side → one 35×36 m photometric group.
        # 2 × 35 × 36 = 2,520 m² ≈ NASA 27,000 ft² / 2,500 m² class.
        b.panel((0, sign * 40, 4.5), (sign, 0, 0), (0, sign, 0), 35, 36,
                CELLS, PANEL_BACK, f"arrays {tag}", gimbal=GIMBAL_2AXIS,
                gimbal_axis=(1, 0, 0), wrist_axis=(0, 1, 0),
                travel_status=STATUS_TYPICAL)
        b.panel((0, sign * 14, 0), (0, sign, 0), (1, 0, 0), 22, 12,
                WHITE_PAINT, WHITE_PAINT, f"radiators {tag}",
                gimbal=GIMBAL_1AXIS, gimbal_axis=(0, 1, 0),
                travel_status=STATUS_TYPICAL)
    b.meta(
        family_id="iss",
        sources=(
            "NASA ISS solar-array article: 27,000 ft² (2,500 m²) USOS "
            "arrays; 240 ft (73 m) pair wingspan. Each original wing "
            "~112×39 ft (35×12 m); eight wings → two 35×36 m groups.",
            "NASA ISS reference: pressurized stack ~50 m class; integrated "
            "truss ~109 m (358 ft). Arrays on alpha/beta 2-axis; radiators "
            "on 1-axis. iROSA overlays are not modelled separately.",
            "Reboost: Progress / Zvezda along +x (velocity) in LVLH.",
        ),
        notes="Coarse photometric stand-in, not station CAD. Study LIBRARY "
              "`shapes.iss` keeps the earlier 35×24 m groups. Catalog uses "
              "the NASA 2,500 m² class. Radiators are white 1-axis "
              "deployables, not cells.",
        dimension_status={"modules": STATUS_RANGE, "truss": STATUS_PUBLIC,
                          "arrays": STATUS_PUBLIC, "radiators": STATUS_RANGE,
                          "thrust_vector": STATUS_PUBLIC},
        thrust_body=[[1.0, 0.0, 0.0]],
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="ISS reboost is publicly along-track (+x in this frame).",
        flight_attitude=ATT_LVLH,
    )
    return b.build().set_look(
        "radiator", None, attitude=ATT_LVLH,
        notes="PVR/EATCS 1-axis about orbit-normal. Operational thermal "
              "look schedule is unpublished — not invented.",
        status=STATUS_UNKNOWN).ensure_mirror_polygons()


def starlink_v2mini_fcc(dtc: bool = False) -> FacetModel:
    """SATCAT Starlink v2 Mini from the cited FCC table.

    Geometry matches `shapes.starlink_v2mini` (already FCC 4.1×2.7 /
    4.1×12.8). Catalog copy adds explicit shoulder+wrist hinges, the
    nadir user-face look, and the DTC look when tagged. Study LIBRARY
    factory is not changed.
    """
    name = "starlink_v2mini_dtc" if dtc else "starlink_v2mini"
    b = _Builder(name)
    b.box((0, 0, 0), (4.1, 2.7, 0.2), MLI, "bus",
          mats=[MLI, MLI, MLI, MLI, MLI, ANTENNA])
    for sign, tag in [(1, "fore"), (-1, "aft")]:
        cx = sign * (4.1 / 2 + 0.4 + 12.8 / 2)
        b.panel((cx, 0, 0), (sign, 0, 0), (0, sign, 0), 12.8, 4.1, CELLS,
                PANEL_BACK, f"array {tag}", gimbal=GIMBAL_2AXIS,
                gimbal_axis=(1, 0, 0), wrist_axis=(0, 1, 0),
                travel_status=STATUS_UNKNOWN)
    if dtc:
        cx = -(4.1 / 2 + 0.2 + 2.0 / 2)
        b.panel((cx, 0, -0.4), (0, 1, 0), (1, 0, 0), 2.3, 2.0, ANTENNA, MLI,
                "dtc antenna",
                look_body=(0, 0, -1), look_attitude=ATT_LVLH,
                look_notes="Press/observer Earth face; DTC size is a range, "
                           "not a SpaceX drawing.",
                look_status=STATUS_RANGE)
    b.meta(
        family_id=name,
        sources=(
            "Jonathan McDowell / SpaceX FCC Gen2: v2 Mini bus 4.1×2.7 m, "
            "each array 4.1×12.8 m, mass ~800 kg "
            "(https://planet4589.org/astro/starsim/index.html).",
            "Spaceflight Now 2023-02-26: two wings, ~30 m tip-to-tip, "
            "116 m² class surface area; shoulder + wrist (2-axis).",
            "Mallama et al. arXiv:2306.06657 (photometric characterization).",
            "Celestrak SATCAT tags some vehicles [DTC]; DTC panel size is an "
            "observer-scale stand-in, not a SpaceX drawing.",
            "SpaceX public: argon/krypton ion propulsion; thrust pointing "
            "unpublished — left unknown. No Starshield internals.",
            "SpaceX Brightness Mitigation Best Practices / Gen2 PDF: no "
            "visor on Mini; nadir dielectric film + terminator-tracking "
            "array off-point (knife-edge to Earth limb). Mallama et al. "
            "arXiv:2306.06657, 2309.14152. Film is a coating; tracking "
            "angles are unpublished — not meshed as a mitigated state.",
        ),
        notes="Two 2-axis arrays: shoulder +x, wrist +y from rest +z. "
              "Travel unpublished (±π stand-in). DTC is a nadir-facing "
              "deployable only when dtc=True. Study LIBRARY "
              "`shapes.starlink_v2mini` is unchanged. Unmitigated "
              "stand-in: no visor CAD, no invented terminator-track "
              "angles, CELLS/MLI not retuned to Mallama.",
        dimension_status={
            "bus": STATUS_PUBLIC, "arrays": STATUS_PUBLIC,
            "array_gimbal": STATUS_PUBLIC,
            "thrust_vector": STATUS_UNKNOWN,
            **({"dtc_antenna": STATUS_RANGE} if dtc else {}),
        },
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_EP,
        thrust_notes="Searched SpaceX Gen2 PDF, FCC table, SpaceNews "
                     "(argon Hall 170 mN / 4.2 kW). No primary body-frame "
                     "vs ram/nadir citation — vector left empty.",
        flight_attitude=ATT_LVLH,
    )
    return b.build().set_look(
        "bus -z", (0, 0, -1), attitude=ATT_LVLH,
        notes="User phased-array face (Earth / nadir in LVLH-hold).",
        status=STATUS_PUBLIC).ensure_mirror_polygons()


def leo_box_wing() -> FacetModel:
    """Fallback LEO payload: the study generic box-wing, catalog-tagged."""
    m = box_wing()
    m.family_id = "leo_box_wing"
    m.name = "leo_box_wing"
    m.notes = "Fallback for active LEO payloads that do not match a named family."
    m.dimension_status = {"bus": STATUS_TYPICAL, "arrays": STATUS_TYPICAL,
                          "thrust_vector": STATUS_UNKNOWN}
    return m


def classified_unpublished() -> FacetModel:
    """USA / classified payloads. Dimensions are not public — do not invent.

    A clearly-labelled uncertain box-wing so the object still *resolves* to
    a family. Every dimension is `uncertain`. Not Starshield CAD.
    """
    bus = (2.0, 1.5, 1.5)
    b = _Builder("classified_unpublished")
    b.box((0, 0, 0), bus, MLI, "bus")
    _two_wing(b, bus, (1.5, 3.0), gimbal=GIMBAL_1AXIS)
    b.meta(
        family_id="classified_unpublished",
        sources=("No public OML. Template exists only so classified SATCAT "
                 "rows resolve to an explicit uncertain family rather than a "
                 "fake precise mesh. Not Starshield / not a named NRO bus."),
        notes="All geometry is a placeholder range. Do not invert this as truth.",
        dimension_status={"bus": STATUS_UNCERTAIN, "arrays": STATUS_UNCERTAIN,
                          "array_gimbal": STATUS_UNCERTAIN,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_UNKNOWN, thrust_propulsion=PROP_UNKNOWN,
        thrust_notes="Propulsion and pointing unpublished by construction.",
    )
    return b.build()


def falcon9_s2() -> FacetModel:
    """Falcon 9 second stage. Diameter is public; length is a short range."""
    # User's guide: 3.66 m diameter. Length commonly 12.6–13.8 m including
    # engine; stand-in 13.8 m with status=range.
    b = _Builder("falcon9_s2")
    b.prism((0, 0, 0), (0, 0, 1), 13.8, 3.66, 16, MLI_SILVER, (MLI, DARK),
            "stage")
    b.meta(
        family_id="falcon9_s2",
        sources=("SpaceX Falcon User's Guide: stage diameter 3.66 m.",
                 "Public length figures 12.6–13.8 m (engine included); "
                 "stand-in 13.8 m."),
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_RANGE},
        thrust_body=[[0.0, 0.0, -1.0]],
        thrust_attitude=ATT_STAGE_AXIS, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Merlin Vacuum along −z. On-orbit spent stages tumble.",
    )
    return b.build()


def cz_upper() -> FacetModel:
    """Long March (CZ) upper stage. Family of diameters; stand-in 2.9 m class."""
    b = _Builder("cz_upper")
    b.prism((0, 0, 0), (0, 0, 1), 8.0, 2.9, 16, MLI_SILVER, (MLI, DARK),
            "stage")
    b.meta(
        family_id="cz_upper",
        sources=("CZ-2/3/4 upper stages are publicly in the ~2.25–3.35 m "
                 "diameter class; length varies by variant. Stand-in is a "
                 "range midpoint, not a named YZ / YF drawing."),
        dimension_status={"diameter": STATUS_RANGE, "length": STATUS_RANGE},
        thrust_body=[[0.0, 0.0, -1.0]],
        thrust_attitude=ATT_STAGE_AXIS, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Engine along −z; spent-stage attitude uncontrolled.",
    )
    return b.build()


def ariane_upper() -> FacetModel:
    """Ariane 5 ESC-A / EPS class upper stage (5.4 m diameter fairing family)."""
    b = _Builder("ariane_upper")
    b.prism((0, 0, 0), (0, 0, 1), 4.7, 5.4, 16, MLI_SILVER, (MLI, DARK),
            "stage")
    b.meta(
        family_id="ariane_upper",
        sources=("Ariane 5 public: ESC-A ~5.4 m diameter class. Length is a "
                 "range across EPS/ESC-A/Ariane 6; stand-in 4.7 m."),
        notes="Covers ARIANE n R/B name pattern; not a per-variant CAD.",
        dimension_status={"diameter": STATUS_RANGE, "length": STATUS_RANGE},
        thrust_body=[[0.0, 0.0, -1.0]],
        thrust_attitude=ATT_STAGE_AXIS, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Engine along −z.",
    )
    return b.build()


def breeze_m() -> FacetModel:
    """Breeze-M: public ~2.5 m diameter, ~2.6 m height class (tank jettisoned)."""
    b = _Builder("breeze_m")
    b.prism((0, 0, 0), (0, 0, 1), 2.6, 2.5, 16, MLI_SILVER, (MLI, DARK),
            "stage")
    b.meta(
        family_id="breeze_m",
        sources=("Public Breeze-M: ~2.5 m diameter, ~2.6 m high core after "
                 "auxiliary tank jettison (Gunter)."),
        dimension_status={"diameter": STATUS_RANGE, "length": STATUS_RANGE},
        thrust_body=[[0.0, 0.0, -1.0]],
        thrust_attitude=ATT_STAGE_AXIS, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Engine along −z. SATCAT also lists BREEZE-M DEB [TANK].",
    )
    return b.build()


def electron_kick() -> FacetModel:
    """Rocket Lab Electron kick / Curie stage. Small; dimensions a range."""
    b = _Builder("electron_kick")
    b.prism((0, 0, 0), (0, 0, 1), 1.2, 1.2, 12, MLI_SILVER, (MLI, DARK),
            "stage")
    b.meta(
        family_id="electron_kick",
        sources=("Electron first/second stage 1.2 m diameter is public "
                 "(Rocket Lab). Kick-stage length is a range; stand-in 1.2 m."),
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_RANGE},
        thrust_body=[[0.0, 0.0, -1.0]],
        thrust_attitude=ATT_STAGE_AXIS, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Curie / HyperCurie along −z. Attitude typically uncontrolled.",
    )
    return b.build()
