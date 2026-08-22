"""Pass-3 families: leftover SAR / civil buses and lingering stages.

Cited public OML only. Chinese optical stacks without a primary drawing
(Jilin, SuperView, GEESat, CentiSpace, Yunhai, optical Gaofen) stay
leo_box_wing. Yaogan stays classified_unpublished. IR α/ε unknown.
"""

from __future__ import annotations

from ..shapes import (
    ANTENNA,
    ATT_LVLH,
    ATT_NADIR,
    CELLS,
    DARK,
    GIMBAL_1AXIS,
    GIMBAL_FIXED,
    MLI,
    PANEL_BACK,
    PROP_CHEMICAL,
    PROP_NONE,
    PROP_UNKNOWN,
    STATUS_PUBLIC,
    STATUS_RANGE,
    STATUS_TYPICAL,
    STATUS_UNKNOWN,
    WHITE_PAINT,
    FacetModel,
    _Builder,
)
from .families import _two_wing
from .families_more import _stage


def terrasat_x() -> FacetModel:
    """TerraSAR-X / TanDEM-X / PAZ: DLR hex 5 × 2.4 m + 5 × 0.8 m SAR."""
    b = _Builder("terrasat_x")
    b.prism((0, 0, 0), (1, 0, 0), 5.0, 2.4, 6, MLI, (MLI, DARK), "bus")
    b.panel((0, 0, -1.4), (0, 1, 0), (1, 0, 0), 0.80, 5.0, ANTENNA, MLI,
            "sar antenna")
    # Body-mounted 5.25 m² generator: stand-in panel on +z.
    b.panel((0, 0, 1.3), (1, 0, 0), (0, 1, 0), 2.1, 2.5, CELLS, PANEL_BACK,
            "body array")
    b.meta(
        family_id="terrasat_x",
        sources=("DLR TerraSAR-X system configuration: 5 m × 2.4 m hexagonal "
                 "bus, 5 m × 0.80 m radar antenna, 5.25 m² body-mounted "
                 "solar generator, ~1.3 t.",
                 "eoPortal TSX. PAZ is the same Astrium Flexbus / antenna "
                 "class (Hisdesat)."),
        notes="SATCAT TERRASAR-X, PAZ. Prefix TERRA is not used (would "
              "steal TERRA the EOS flagship).",
        dimension_status={"bus": STATUS_PUBLIC, "sar": STATUS_PUBLIC,
                          "arrays": STATUS_PUBLIC, "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Hydrazine; vector unpublished.",
    )
    return b.build()


def cosmo_skymed() -> FacetModel:
    """COSMO-SkyMed 1–4: PRIMA bus + 5.7 × 1.4 m X-band SAR, 18.3 m² arrays."""
    bus = (2.5, 1.8, 1.8)
    b = _Builder("cosmo_skymed")
    b.box((0, 0, 0), bus, MLI, "bus")
    _two_wing(b, bus, (1.8, 5.1), gimbal=GIMBAL_1AXIS)
    b.panel((0, 0, -1.2), (0, 1, 0), (1, 0, 0), 1.4, 5.7, ANTENNA, MLI,
            "sar antenna")
    b.meta(
        family_id="cosmo_skymed",
        sources=("eoPortal COSMO-SkyMed: 5.7 × 1.4 m SAA, two arrays 18.3 m² "
                 "combined, ~1700 kg, PRIMA bus.",
                 "CSG (second generation) is a different vehicle and is not "
                 "this family."),
        notes="Bus box is typical_class of PRIMA; antenna and array area "
              "are public.",
        dimension_status={"bus": STATUS_TYPICAL, "sar": STATUS_PUBLIC,
                          "arrays": STATUS_PUBLIC, "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Chemical; vector unpublished.",
    )
    return b.build()


def alos2() -> FacetModel:
    """ALOS-2 Daichi-2: JAXA deployed 9.9 × 16.5 × 3.7 m, PALSAR-2 2.9 × 9.9 m."""
    bus = (3.7, 3.0, 3.0)
    b = _Builder("alos2")
    b.box((0, 0, 0), bus, MLI, "bus",
          mats=[DARK, MLI, MLI, MLI, MLI, WHITE_PAINT])
    _two_wing(b, bus, (3.0, 6.0), gimbal=GIMBAL_1AXIS)
    b.panel((9.9 / 2, 0, -1.6), (0, 1, 0), (1, 0, 0), 2.9, 9.9, ANTENNA, MLI,
            "palsar-2")
    b.meta(
        family_id="alos2",
        sources=("eoPortal / JAXA ALOS-2: deployed 9.9 × 16.5 × 3.7 m, "
                 "2120 kg; PALSAR-2 antenna 2.9 m × 9.9 m.",
                 "ALOS-4 maps here as typical_class of the same L-band "
                 "class, not a copied ALOS-2 drawing."),
        dimension_status={"bus": STATUS_RANGE, "sar": STATUS_PUBLIC,
                          "arrays": STATUS_RANGE, "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_NADIR, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Hydrazine; vector unpublished.",
    )
    return b.build()


def radarsat2() -> FacetModel:
    """RADARSAT-2: CSA 3.7 × 1.36 m bus, 15 × 1.5 m SAR, 3.73 × 1.8 m wings."""
    bus = (1.36, 1.36, 3.7)
    b = _Builder("radarsat2")
    b.box((0, 0, 0), bus, MLI, "bus")
    _two_wing(b, bus, (1.8, 3.73), gimbal=GIMBAL_1AXIS, boom=0.2)
    b.panel((0, 0, -2.0), (0, 1, 0), (1, 0, 0), 1.5, 15.0, ANTENNA, MLI,
            "sar antenna")
    b.meta(
        family_id="radarsat2",
        sources=("CSA RADARSAT technical comparison: bus 3.7 × 1.36 m, "
                 "SAR 15 × 1.5 m, each array 3.73 × 1.8 m, 2200 kg.",
                 "eoPortal RADARSAT-2 (PRIMA heritage). RCM is a different "
                 "vehicle and is not this family."),
        dimension_status={"bus": STATUS_PUBLIC, "sar": STATUS_PUBLIC,
                          "arrays": STATUS_PUBLIC, "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Hydrazine; vector unpublished.",
    )
    return b.build()


def gaofen3() -> FacetModel:
    """Gaofen-3 civil C-band SAR only: 15 × 1.232 m antenna (CNSA / IEEE)."""
    bus = (2.5, 2.0, 2.5)
    b = _Builder("gaofen3")
    b.box((0, 0, 0), bus, MLI, "bus")
    _two_wing(b, bus, (2.0, 3.0), gimbal=GIMBAL_1AXIS)
    b.panel((0, 0, -1.5), (0, 1, 0), (1, 0, 0), 1.232, 15.0, ANTENNA, MLI,
            "sar antenna")
    b.meta(
        family_id="gaofen3",
        sources=("eoPortal GF-3 / Sun et al. Sensors 2017: deployable C-band "
                 "phased array 15 m (azimuth) × 1.232 m (elevation), "
                 "~2279 kg, ZY1000B / CS-L3000B class.",
                 "Optical Gaofen-1/2/5/6/7/9/11/12 are different buses and "
                 "stay leo_box_wing — no single civil OML."),
        notes="SATCAT GAOFEN-3, GAOFEN-3 02, GAOFEN-3 03 only.",
        dimension_status={"bus": STATUS_TYPICAL, "sar": STATUS_PUBLIC,
                          "arrays": STATUS_RANGE, "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Vector unpublished.",
    )
    return b.build()


def saocom() -> FacetModel:
    """SAOCOM-1: INVAP/CONAE 4.7 m × 1.2 m stowed, 10 × 3.5 m L-band SAR."""
    bus = (1.2, 1.2, 4.7)
    b = _Builder("saocom")
    b.box((0, 0, 0), bus, MLI, "bus")
    _two_wing(b, bus, (1.61, 2.69), gimbal=GIMBAL_1AXIS)
    b.panel((0, 0, -2.6), (0, 1, 0), (1, 0, 0), 3.5, 10.0, ANTENNA, MLI,
            "sar antenna")
    b.meta(
        family_id="saocom",
        sources=("INVAP SAOCOM brochure: 4.7 m × 1.2 m stowed, 3000 kg, "
                 "SAR 10 × 3.5 m, three 1.61 × 2.69 m arrays.",
                 "eoPortal SAOCOM: 10 × 3.5 m (35 m²) active phased array."),
        dimension_status={"bus": STATUS_PUBLIC, "sar": STATUS_PUBLIC,
                          "arrays": STATUS_PUBLIC, "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Vector unpublished.",
    )
    return b.build()


def esa_swarm() -> FacetModel:
    """ESA Swarm A/B/C magnetic-field trio — not Swarm Technologies SPACEBEE."""
    b = _Builder("esa_swarm")
    b.box((0, 0, 0), (4.0, 1.5, 0.85), MLI, "bus",
          mats=[CELLS, MLI, CELLS, MLI, DARK, WHITE_PAINT])
    b.panel((4.5, 0, 0), (0, 1, 0), (1, 0, 0), 0.4, 5.1, DARK, MLI,
            "magnetometer boom")
    b.meta(
        family_id="esa_swarm",
        sources=("ESA Swarm facts: 9.1 m long including ~4–5 m boom, "
                 "1.5 m wide, 0.85 m high, ~468 kg.",
                 "eoPortal Swarm. Not Swarm Technologies SPACEBEE "
                 "(absent from this SATCAT extract)."),
        notes="SATCAT SWARM A/B/C. Body-mounted cells on sun-facing sides.",
        dimension_status={"bus": STATUS_PUBLIC, "boom": STATUS_PUBLIC,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Cold-gas CF4; a single body-frame vector is not used.",
    )
    return b.build()


def ion_scv() -> FacetModel:
    """D-Orbit ION Satellite Carrier: public 60 cm / 64U dispenser, ~160 kg."""
    b = _Builder("ion_scv")
    b.box((0, 0, 0), (0.60, 0.60, 0.60), DARK, "bus")
    b.meta(
        family_id="ion_scv",
        sources=("D-Orbit / Wikipedia ION Satellite Carrier: 60 cm cubic "
                 "64U dispenser, ~160 kg. SATCAT ION SCV-nnn."),
        notes="After dispenser empty the bus is still this envelope. "
              "Not a CubeSat deployer CAD of hosted payloads.",
        dimension_status={"bus": STATUS_PUBLIC, "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="PM200 class; vector unpublished.",
    )
    return b.build()


def ghgsat() -> FacetModel:
    """GHGSat-C: ESA Earth Online 20 × 30 × 40 cm SFL NEMO."""
    bus = (0.40, 0.20, 0.30)
    b = _Builder("ghgsat")
    b.box((0, 0, 0), bus, DARK, "bus",
          mats=[DARK, MLI, MLI, MLI, MLI, DARK])
    _two_wing(b, bus, (0.40, 0.20), gimbal=GIMBAL_FIXED, boom=0.02)
    b.meta(
        family_id="ghgsat",
        sources=("ESA Earth Online GHGSat: C-series 20 × 30 × 40 cm, ~16 kg "
                 "(SFL NEMO). Demonstrator Claire was 20 × 20 × 42 cm.",
                 "Later C9+ 16U buses are the same family, tagged "
                 "typical_class."),
        dimension_status={"bus": STATUS_PUBLIC, "arrays": STATUS_TYPICAL,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_NADIR, thrust_propulsion=PROP_NONE,
        thrust_notes="No propulsion (differential drag). Vector omitted.",
    )
    return b.build()


def grus() -> FacetModel:
    """Axelspace GRUS-1: ~0.6 × 0.6 × 0.8 m, ~100 kg."""
    bus = (0.80, 0.60, 0.60)
    b = _Builder("grus")
    b.box((0, 0, 0), bus, MLI, "bus",
          mats=[DARK, MLI, MLI, MLI, MLI, DARK])
    _two_wing(b, bus, (0.60, 0.40), gimbal=GIMBAL_FIXED, boom=0.05)
    b.meta(
        family_id="grus",
        sources=("Axelspace / CREODIAS GRUS-1: ~100 kg; published envelopes "
                 "0.50 × 0.50 × 0.70 m and 0.60 × 0.60 × 0.80 m (range).",
                 "eoPortal GRUS: GRUS-3 is heavier (~150 kg) and maps here "
                 "as typical_class."),
        dimension_status={"bus": STATUS_RANGE, "arrays": STATUS_RANGE,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_NADIR, thrust_propulsion=PROP_UNKNOWN,
        thrust_notes="Pointing unpublished.",
    )
    return b.build()


def atlas_core() -> FacetModel:
    return _stage(
        "atlas_core", 12.0, 3.05,
        sources=("Public Atlas SLV / Atlas II core: 3.05 m (10 ft) diameter. "
                 "Length is a range across D/E/F/H/2A lingering stages.",
                 "SATCAT ATLAS n R/B without CENTAUR / AGENA / STAR 48."),
        notes="Centaur and Agena are separate families and match first.",
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_RANGE},
    )


def titan_core() -> FacetModel:
    return _stage(
        "titan_core", 10.0, 3.05,
        sources=("Public Titan II/III/IV core: 10 ft (3.05 m) diameter "
                 "(NASA Titan IIIC stage descriptions). Length is a range.",
                 "SATCAT TITAN n R/B that is not TRANSTAGE / AGENA."),
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_RANGE},
    )


def saturn_sivb() -> FacetModel:
    return _stage(
        "saturn_sivb", 17.8, 6.60,
        sources=("NASA Saturn V / S-IVB: 21 ft 8 in (6.60 m) diameter, "
                 "~58 ft (17.8 m) length.",),
        notes="SATCAT SATURN 5 R/B.",
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_PUBLIC},
    )


def avum() -> FacetModel:
    return _stage(
        "avum", 1.74, 1.90,
        sources=("ESA Vega stages: AVUM 1.9 m diameter, 1.74 m height.",),
        notes="SATCAT AVUM R/B.",
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_PUBLIC},
    )


def firefly_alpha() -> FacetModel:
    return _stage(
        "firefly_alpha", 5.37, 1.82,
        sources=("Firefly Alpha PUG / public: 1.82 m stage diameter; "
                 "stage 2 length ~5.37 m.",),
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_PUBLIC},
    )


def dnepr() -> FacetModel:
    return _stage(
        "dnepr", 6.0, 3.00,
        sources=("Dnepr / RS-20 / SS-18 public: 3.0 m diameter class. "
                 "SATCAT DNEPR / SL-23 / SS-18 R/B.",),
        notes="Length is a range of the lingering space stage.",
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_RANGE},
    )


def thor_ablestar() -> FacetModel:
    return _stage(
        "thor_ablestar", 4.90, 1.40,
        sources=("Public Able-Star / Ablestar upper stage: ~1.4 m diameter, "
                 "~4.9 m length (NASA / Thor-Ablestar descriptions).",),
        notes="SATCAT THOR ABLESTAR R/B.",
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_RANGE},
    )


def burner2() -> FacetModel:
    return _stage(
        "burner2", 0.84, 0.66,
        sources=("Public Thor Burner 2 / 2A: ~0.66 m diameter solid kick "
                 "stage (USAF / NASA Burner II).",),
        notes="SATCAT THOR BURNER 2 / ATLAS F BURNER 2 R/B.",
        dimension_status={"diameter": STATUS_PUBLIC, "length": STATUS_RANGE},
    )
