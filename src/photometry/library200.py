"""Procedural 200-entry hypothetical spacecraft library, mixed countries.

Each entry is a *hypothetical* spacecraft drawn from a parametric family
whose dimension/configuration ranges are grounded in public references —
Gunter's Space Page (space.skyrocket.de) family pages, eoPortal mission
descriptions, and press-kit numbers — for representative archetypes:

  flat_sat        Starlink v1.5/v2-mini, Qianfan/G60 (~300 kg flat-pack,
                  single array), Guowang: flat bus + 1-2 large arrays
  box_wing_comm   OneWeb (Arrow: ~1x1x1.3 m, 150 kg, two panels),
                  Iridium NEXT (ELiTeBus: 3.1x2.4x1.5 m, ~9.4 m span),
                  Globalstar-2, O3b-class
  eo_imager       Pleiades/SuperView/KOMPSAT/Cartosat-class prism bus with
                  fixed or 1-axis wings
  sar_sat         ICEYE/Capella/NovaSAR/TerraSAR-class: bus + large fixed
                  planar radar aperture + wings
  cubesat         3U-16U with body cells and deployable panel pairs
                  (Planet/Spire/ICEYE-precursor heritage)
  smallsat        50-500 kg generic micro bus, single wing or body cells
  rocket_body     spent upper stages (Falcon 9 S2, CZ-2/3/4, Fregat,
                  Centaur, H-2A class cylinders)
  tug_servicer    OTV/servicer class (ION, Sherpa, MEV/LINK-like): box bus,
                  two wings, reflective capture/antenna face
  science_bus     astronomy/physics buses, inertially pointed
  station_module  large crewed-station modules (Tiangong/ISS-segment class)
  weather_bus     Metop/FY-3-class: big box, one large 1-axis array

Countries are assigned with rough plausibility weights per family
(mega-constellation flat-sats skew US/CN; SAR includes FI/DE/IT/JP/AR;
cubesats are broadly international, etc.). Every entry records the
attitude-mode and array-control-mode sets it plausibly flies, which the
scale-test scenario sampler draws from.

NOTE on references: ESA DISCOS (discosweb.esoc.esa.int) carries measured
cross-sections/dimensions per catalog object and would be the natural
cross-check for these ranges; it needs an API token and is not reachable
from this build environment, so the ranges here rest on Gunter's/eoPortal
numbers. The generator is deterministic (seeded) so entries are stable
across runs.
"""

from __future__ import annotations

from zlib import crc32

import numpy as np

from .shapes import (
    ANTENNA,
    CELLS,
    DARK,
    GIMBAL_1AXIS,
    GIMBAL_2AXIS,
    MLI,
    MLI_SILVER,
    PANEL_BACK,
    WHITE_PAINT,
    FacetModel,
    _Builder,
)

# family -> (count, [(country, weight), ...])
FAMILY_PLAN = {
    "flat_sat": (28, [("us", 8), ("cn", 10), ("ru", 2), ("in", 2), ("eu", 3),
                      ("kr", 1), ("jp", 1), ("tw", 1)]),
    "box_wing_comm": (30, [("us", 7), ("uk", 4), ("cn", 6), ("ru", 4),
                           ("ca", 2), ("eu", 4), ("jp", 2), ("in", 1)]),
    "eo_imager": (26, [("fr", 3), ("de", 2), ("cn", 6), ("in", 3), ("kr", 2),
                       ("jp", 2), ("il", 2), ("us", 3), ("ae", 1), ("tw", 1),
                       ("eg", 1)]),
    "sar_sat": (22, [("fi", 3), ("us", 4), ("de", 2), ("it", 2), ("jp", 3),
                     ("cn", 4), ("in", 1), ("ar", 1), ("kr", 1), ("pl", 1)]),
    "cubesat": (24, [("us", 6), ("jp", 2), ("dk", 1), ("nl", 1), ("pl", 1),
                     ("za", 1), ("sg", 1), ("br", 2), ("tr", 2), ("vn", 1),
                     ("th", 1), ("in", 2), ("cn", 2), ("au", 1)]),
    "smallsat": (20, [("us", 4), ("cn", 3), ("ru", 2), ("ir", 2), ("kp", 1),
                      ("kz", 1), ("pk", 1), ("id", 1), ("mx", 1), ("sa", 1),
                      ("ng", 1), ("ua", 1), ("ch", 1)]),
    "rocket_body": (16, [("us", 4), ("cn", 5), ("ru", 4), ("eu", 1),
                         ("jp", 1), ("in", 1)]),
    "tug_servicer": (12, [("us", 5), ("jp", 2), ("de", 2), ("cn", 2),
                          ("it", 1)]),
    "science_bus": (12, [("us", 4), ("eu", 3), ("jp", 2), ("cn", 2),
                         ("in", 1)]),
    "station_module": (5, [("cn", 2), ("ru", 1), ("us", 2)]),
    "weather_bus": (5, [("eu", 2), ("cn", 2), ("us", 1)]),
}

MODES = {
    "flat_sat": (["ops", "low_drag", "tumble"], ["track", "frozen", "offset"]),
    "box_wing_comm": (["ops", "sun_point", "tumble"], ["track", "frozen", "offset"]),
    "eo_imager": (["ops", "safe_sun", "tumble"], ["track", "frozen"]),
    "sar_sat": (["ops", "sun_point", "tumble"], ["track", "frozen"]),
    "cubesat": (["ops", "sun_point", "tumble"], ["track"]),
    "smallsat": (["ops", "sun_point", "tumble"], ["track", "frozen"]),
    "rocket_body": (["tumble"], []),
    "tug_servicer": (["ops", "sun_point", "tumble"], ["track", "frozen", "offset"]),
    "science_bus": (["science", "safe_sun", "tumble"], ["track", "frozen"]),
    "station_module": (["ops", "tumble"], ["track", "frozen"]),
    "weather_bus": (["ops", "safe_sun", "tumble"], ["track", "frozen"]),
}


def _flat_sat(rng: np.random.Generator, name: str) -> FacetModel:
    b = _Builder(name)
    L = rng.uniform(2.0, 4.4)
    W = rng.uniform(1.2, 2.8)
    b.box((0, 0, 0), (L, W, rng.uniform(0.15, 0.3)), MLI, "bus",
          mats=[MLI, MLI, MLI, MLI, MLI, ANTENNA])
    n_arr = 1 if rng.random() < 0.55 else 2
    area = rng.uniform(8.0, 55.0) / n_arr
    ar = rng.uniform(2.2, 3.4)  # panel aspect
    w = float(np.sqrt(area * ar))
    h = area / w
    gim = GIMBAL_1AXIS if rng.random() < 0.5 else GIMBAL_2AXIS
    for k in range(n_arr):
        sign = 1 if k == 0 else -1
        cx = sign * (L / 2 + 0.35 + w / 2)
        b.panel((cx, 0, 0), (sign, 0, 0), (0, sign, 0), w, h, CELLS,
                PANEL_BACK, f"array{k}", gimbal=gim, gimbal_axis=(1, 0, 0))
    return b.build()


def _box_wing_comm(rng: np.random.Generator, name: str) -> FacetModel:
    b = _Builder(name)
    s = rng.uniform(0.9, 3.2)
    dims = (s, s * rng.uniform(0.7, 1.1), s * rng.uniform(0.5, 1.4))
    b.box((0, 0, 0), dims, MLI, "bus",
          mats=[MLI, MLI, MLI, MLI, ANTENNA, ANTENNA])
    total = rng.uniform(2.0, 24.0)
    n_wing = 2 if rng.random() < 0.8 else 1
    area = total / n_wing
    h = float(np.sqrt(area / rng.uniform(1.8, 3.2)))
    w = area / h
    for k in range(n_wing):
        sign = 1 if k == 0 else -1
        cy = sign * (dims[1] / 2 + 0.3 + w / 2)
        b.panel((0, cy, 0), (1, 0, 0), (0, sign, 0), h, w, CELLS, PANEL_BACK,
                f"wing{k}", gimbal=GIMBAL_1AXIS, gimbal_axis=(0, 1, 0))
    return b.build()


def _eo_imager(rng: np.random.Generator, name: str) -> FacetModel:
    b = _Builder(name)
    dia = rng.uniform(0.6, 2.2)
    length = rng.uniform(1.2, 4.5)
    mat = MLI if rng.random() < 0.6 else WHITE_PAINT
    # optical axis -z (nadir aperture)
    b.prism((0, 0, 0), (0, 0, 1), length, dia, 6, mat, (MLI, DARK), "tube")
    if rng.random() < 0.6:
        n_w = 2 if rng.random() < 0.5 else 1
        area = rng.uniform(1.5, 8.0) / n_w
        h = float(np.sqrt(area / 2.0))
        w = area / h
        for k in range(n_w):
            sign = 1 if k == 0 else -1
            cy = sign * (dia / 2 + 0.25 + w / 2)
            b.panel((0, cy, 0), (1, 0, 0), (0, sign, 0), h, w, CELLS,
                    PANEL_BACK, f"wing{k}", gimbal=GIMBAL_1AXIS,
                    gimbal_axis=(0, 1, 0))
    else:  # body-fixed cells on the zenith face
        b.panel((0, 0, length / 2 + 0.02), (0, 1, 0), (1, 0, 0),
                dia * 0.9, dia * 0.9, CELLS, PANEL_BACK, "body cells")
    return b.build()


def _sar_sat(rng: np.random.Generator, name: str) -> FacetModel:
    b = _Builder(name)
    s = rng.uniform(0.6, 2.0)
    b.box((0, 0, 0), (s, s * 0.8, s * rng.uniform(0.8, 1.6)), MLI, "bus")
    # fixed planar radar aperture, nadir-canted
    a_area = rng.uniform(2.0, 12.0)
    h = float(np.sqrt(a_area / rng.uniform(2.5, 5.0)))
    w = a_area / h
    b.panel((0, 0, -s * 0.8), (0, 1, 0), (1, 0, 0), w, h, ANTENNA, MLI,
            "sar aperture")  # normal -z
    n_w = 2 if rng.random() < 0.6 else 1
    area = rng.uniform(2.0, 10.0) / n_w
    hh = float(np.sqrt(area / 2.2))
    ww = area / hh
    for k in range(n_w):
        sign = 1 if k == 0 else -1
        cy = sign * (s * 0.4 + 0.25 + ww / 2)
        b.panel((0, cy, 0.4), (1, 0, 0), (0, sign, 0), hh, ww, CELLS,
                PANEL_BACK, f"wing{k}", gimbal=GIMBAL_1AXIS,
                gimbal_axis=(0, 1, 0))
    return b.build()


def _cubesat(rng: np.random.Generator, name: str) -> FacetModel:
    b = _Builder(name)
    units = int(rng.choice([3, 6, 12, 16]))
    if units == 3:
        dims = (0.10, 0.10, 0.34)
    elif units == 6:
        dims = (0.10, 0.226, 0.34)
    elif units == 12:
        dims = (0.226, 0.226, 0.34)
    else:
        dims = (0.226, 0.226, 0.454)
    b.box((0, 0, 0), dims, DARK, "bus",
          mats=[CELLS, CELLS, DARK, DARK, DARK, ANTENNA])
    if rng.random() < 0.7:  # deployed wing pair(s)
        w = dims[2]
        h = dims[1] * rng.choice([1, 2, 3])
        for sign in (1, -1):
            cy = sign * (dims[1] / 2 + 0.02 + h / 2)
            b.panel((0, cy, 0), (0, 0, 1), (0, sign, 0), w, h, CELLS,
                    PANEL_BACK, f"wing{'+' if sign > 0 else '-'}")
    return b.build()


def _smallsat(rng: np.random.Generator, name: str) -> FacetModel:
    b = _Builder(name)
    s = rng.uniform(0.4, 1.3)
    b.box((0, 0, 0), (s, s, s * rng.uniform(0.9, 1.6)), MLI, "bus",
          mats=[MLI, MLI, CELLS, CELLS, MLI, ANTENNA])
    if rng.random() < 0.6:
        area = rng.uniform(0.8, 4.0)
        h = float(np.sqrt(area / 2.0))
        w = area / h
        b.panel((0, s / 2 + 0.2 + w / 2, 0), (1, 0, 0), (0, 1, 0), h, w,
                CELLS, PANEL_BACK, "wing", gimbal=GIMBAL_1AXIS,
                gimbal_axis=(0, 1, 0))
    return b.build()


def _rocket_body(rng: np.random.Generator, name: str) -> FacetModel:
    b = _Builder(name)
    dia = float(rng.choice([1.5, 2.25, 2.9, 3.35, 3.66, 4.0]))
    length = dia * rng.uniform(1.8, 3.6)
    mat = MLI_SILVER if rng.random() < 0.6 else WHITE_PAINT
    b.prism((0, 0, 0), (0, 0, 1), length, dia, 12, mat, (MLI, DARK), "stage")
    return b.build()


def _tug_servicer(rng: np.random.Generator, name: str) -> FacetModel:
    b = _Builder(name)
    s = rng.uniform(0.7, 1.6)
    b.box((0, 0, 0), (s, s, s * rng.uniform(1.2, 1.8)), MLI, "bus",
          mats=[ANTENNA, MLI, MLI, MLI, MLI, DARK])
    area = rng.uniform(2.0, 9.0) / 2
    h = float(np.sqrt(area / 2.4))
    w = area / h
    for sign in (1, -1):
        cy = sign * (s / 2 + 0.3 + w / 2)
        b.panel((0, cy, 0), (1, 0, 0), (0, sign, 0), h, w, CELLS, PANEL_BACK,
                f"wing{'+' if sign > 0 else '-'}", gimbal=GIMBAL_1AXIS,
                gimbal_axis=(0, 1, 0))
    return b.build()


def _science_bus(rng: np.random.Generator, name: str) -> FacetModel:
    b = _Builder(name)
    dia = rng.uniform(1.2, 3.5)
    length = dia * rng.uniform(1.5, 3.0)
    b.prism((0, 0, 0), (1, 0, 0), length, dia, 10, MLI_SILVER, (DARK, MLI),
            "tube")
    n_w = 2 if rng.random() < 0.7 else 1
    area = rng.uniform(4.0, 16.0) / n_w
    h = float(np.sqrt(area / 2.6))
    w = area / h
    for k in range(n_w):
        sign = 1 if k == 0 else -1
        cy = sign * (dia / 2 + 0.4 + w / 2)
        b.panel((0, cy, 0), (1, 0, 0), (0, sign, 0), h, w, CELLS, PANEL_BACK,
                f"array{k}", gimbal=GIMBAL_1AXIS, gimbal_axis=(0, 1, 0))
    return b.build()


def _station_module(rng: np.random.Generator, name: str) -> FacetModel:
    b = _Builder(name)
    dia = rng.uniform(3.0, 4.5)
    length = rng.uniform(5.0, 17.0)
    b.prism((0, 0, 0), (1, 0, 0), length, dia, 10, WHITE_PAINT, (MLI, MLI),
            "module")
    area = rng.uniform(60.0, 400.0) / 2
    h = float(np.sqrt(area / 2.8))
    w = area / h
    for sign in (1, -1):
        cy = sign * (dia / 2 + 1.0 + w / 2)
        b.panel((0, cy, 0), (sign, 0, 0), (0, sign, 0), h, w, CELLS,
                PANEL_BACK, f"arrays{'+' if sign > 0 else '-'}",
                gimbal=GIMBAL_2AXIS)
    return b.build()


def _weather_bus(rng: np.random.Generator, name: str) -> FacetModel:
    b = _Builder(name)
    L = rng.uniform(3.0, 6.2)
    W = rng.uniform(1.4, 2.4)
    b.box((0, 0, 0), (L, W, W), MLI, "bus",
          mats=[MLI, MLI, MLI, MLI, MLI, DARK])
    area = rng.uniform(10.0, 30.0)
    h = float(np.sqrt(area / 3.0))
    w = area / h
    b.panel((0, W / 2 + 0.4 + w / 2, 0), (1, 0, 0), (0, 1, 0), h, w, CELLS,
            PANEL_BACK, "array", gimbal=GIMBAL_1AXIS, gimbal_axis=(0, 1, 0))
    return b.build()


_BUILDERS = {
    "flat_sat": _flat_sat,
    "box_wing_comm": _box_wing_comm,
    "eo_imager": _eo_imager,
    "sar_sat": _sar_sat,
    "cubesat": _cubesat,
    "smallsat": _smallsat,
    "rocket_body": _rocket_body,
    "tug_servicer": _tug_servicer,
    "science_bus": _science_bus,
    "station_module": _station_module,
    "weather_bus": _weather_bus,
}

_SHORT = {
    "flat_sat": "flat", "box_wing_comm": "comm", "eo_imager": "eo",
    "sar_sat": "sar", "cubesat": "cube", "smallsat": "small",
    "rocket_body": "rb", "tug_servicer": "tug", "science_bus": "sci",
    "station_module": "stn", "weather_bus": "wx",
}


def generate_library(seed: int = 11):
    """Deterministic 200-entry library.

    Returns (library, metadata): library maps name -> zero-arg builder
    (same contract as shapes.LIBRARY); metadata is a list of dicts with
    country, family, plausible attitude/array modes, and size summaries.
    """
    library: dict = {}
    metadata: list[dict] = []
    for family, (count, weights) in FAMILY_PLAN.items():
        countries = [c for c, w in weights for _ in range(w)]
        if len(countries) != count:
            raise ValueError(f"{family}: weights sum {len(countries)} != {count}")
        per_country: dict[str, int] = {}
        for i, country in enumerate(countries):
            per_country[country] = per_country.get(country, 0) + 1
            name = f"{country}_{_SHORT[family]}{per_country[country]:02d}"
            # crc32, not hash(): str hash is salted per process and would
            # make the "deterministic" library differ between runs
            entry_seed = crc32(f"{seed}|{family}|{name}".encode()) & 0xFFFFFFFF

            def build(family=family, name=name, entry_seed=entry_seed):
                return _BUILDERS[family](np.random.default_rng(entry_seed), name)

            shape = build()
            att_modes, arr_modes = MODES[family]
            if not shape.articulated:
                arr_modes = []
            metadata.append(dict(
                name=name, family=family, country=country,
                n_facets=shape.n_facets,
                diffuse_albedo_area_m2=float(shape.diffuse_albedo_area().sum()),
                total_area_m2=float(shape.areas.sum()),
                articulated=bool(shape.articulated),
                attitude_modes=att_modes, array_modes=arr_modes,
            ))
            library[name] = build
    return library, metadata


# ---------------------------------------------------------------------------
# Named intelligence-satellite annex: living Yaogan / Kosmos / IGS classes
# ---------------------------------------------------------------------------
# These programs publish no engineering data; the geometries below are
# box-wing photometric stand-ins assembled from open-source estimates
# (Gunter's Space Page class entries, russianspaceweb.com, globalsecurity
# .org, launch-mass figures) — deliberately coarser truth claims than the
# curated commercial models. Key anchors: Persona ~7 t, 1.5 m primary,
# Yantar/Resurs-DK heritage cylinder, cruciform arrays (modeled as one
# wing pair); Bars-M ~4 t cartography; Lotos-S ~6 t ELINT with large
# deployed arrays; Yaogan EO on Phoenix-Eye-2-class bus, Yaogan SAR with
# deployable planar aperture, Yaogan NOSS-style ELINT triplet members on
# CAST2000-class buses (~1100 km / 63.4°); IGS Optical / Radar ~1.2-2 t
# with deployable wings, SAR aperture on the radar birds.


def _intel(name, build_fn, att_modes, arr_modes):
    return dict(name=name, build=build_fn, attitude_modes=att_modes,
                array_modes=arr_modes)


def cn_yaogan_eo() -> FacetModel:
    """Yaogan optical (JB-6 / Phoenix-Eye-2-class): nadir telescope prism +
    two 1-axis wings."""
    b = _Builder("cn_yaogan_eo")
    b.prism((0, 0, 0), (0, 0, 1), 3.2, 1.8, 6, MLI, (MLI, DARK), "tube")
    for sign, tag in [(1, "+y"), (-1, "-y")]:
        cy = sign * (0.9 + 0.3 + 1.6)
        b.panel((0, cy, 0.4), (1, 0, 0), (0, sign, 0), 2.0, 3.2, CELLS,
                PANEL_BACK, f"wing {tag}", gimbal=GIMBAL_1AXIS,
                gimbal_axis=(0, 1, 0))
    return b.build()


def cn_yaogan_sar() -> FacetModel:
    """Yaogan SAR (JB-7-class): box bus, deployable planar SAR aperture
    (nadir), two wings."""
    b = _Builder("cn_yaogan_sar")
    b.box((0, 0, 0), (2.2, 2.0, 2.8), MLI, "bus")
    b.panel((0, 0, -1.6), (0, 1, 0), (1, 0, 0), 6.0, 2.6, ANTENNA, MLI,
            "sar aperture")  # normal -z
    for sign, tag in [(1, "+y"), (-1, "-y")]:
        cy = sign * (1.0 + 0.3 + 2.1)
        b.panel((0, cy, 0.6), (1, 0, 0), (0, sign, 0), 2.4, 4.2, CELLS,
                PANEL_BACK, f"wing {tag}", gimbal=GIMBAL_1AXIS,
                gimbal_axis=(0, 1, 0))
    return b.build()


def cn_yaogan_elint() -> FacetModel:
    """Yaogan NOSS-style ELINT triplet member (CAST2000-class): small box,
    fixed panel pair, antenna-farm nadir face."""
    b = _Builder("cn_yaogan_elint")
    b.box((0, 0, 0), (1.4, 1.4, 1.8), MLI, "bus",
          mats=[MLI, MLI, MLI, MLI, MLI, ANTENNA])
    for sign, tag in [(1, "+y"), (-1, "-y")]:
        cy = sign * (0.7 + 0.15 + 1.1)
        b.panel((0, cy, 0), (0, 0, 1), (0, sign, 0), 1.6, 2.2, CELLS,
                PANEL_BACK, f"panel {tag}")  # fixed
    return b.build()


def ru_persona() -> FacetModel:
    """Persona (14F137) optical recon: Yantar/Resurs-DK-heritage cylinder
    (~7 t, 1.5 m primary), wing pair standing in for the cruciform set."""
    b = _Builder("ru_persona")
    b.prism((0, 0, 0), (0, 0, 1), 6.5, 2.7, 10, MLI_SILVER, (MLI, DARK),
            "hull")  # optical axis nadir (-z aperture)
    for sign, tag in [(1, "+y"), (-1, "-y")]:
        cy = sign * (1.35 + 0.4 + 2.4)
        b.panel((0, cy, 1.0), (1, 0, 0), (0, sign, 0), 2.6, 4.8, CELLS,
                PANEL_BACK, f"wing {tag}", gimbal=GIMBAL_1AXIS,
                gimbal_axis=(0, 1, 0))
    return b.build()


def ru_bars_m() -> FacetModel:
    """Bars-M cartography (~4 t): conic-cylinder stand-in + two wings."""
    b = _Builder("ru_bars_m")
    b.prism((0, 0, 0), (0, 0, 1), 4.0, 2.4, 8, MLI, (MLI, DARK), "hull")
    for sign, tag in [(1, "+y"), (-1, "-y")]:
        cy = sign * (1.2 + 0.3 + 1.9)
        b.panel((0, cy, 0.6), (1, 0, 0), (0, sign, 0), 2.2, 3.8, CELLS,
                PANEL_BACK, f"wing {tag}", gimbal=GIMBAL_1AXIS,
                gimbal_axis=(0, 1, 0))
    return b.build()


def ru_lotos_s() -> FacetModel:
    """Lotos-S ELINT (Liana, ~6 t): long cylinder, large wing pair, nadir
    antenna farm."""
    b = _Builder("ru_lotos_s")
    b.prism((0, 0, 0), (1, 0, 0), 8.0, 2.9, 10, MLI, (MLI, DARK), "hull")
    b.panel((0, 0, -1.6), (0, 1, 0), (1, 0, 0), 5.0, 1.8, ANTENNA, MLI,
            "antenna farm")  # normal -z
    for sign, tag in [(1, "+y"), (-1, "-y")]:
        cy = sign * (1.45 + 0.4 + 3.0)
        b.panel((0, cy, 0.6), (1, 0, 0), (0, sign, 0), 3.0, 6.0, CELLS,
                PANEL_BACK, f"wing {tag}", gimbal=GIMBAL_1AXIS,
                gimbal_axis=(0, 1, 0))
    return b.build()


def jp_igs_optical() -> FacetModel:
    """IGS-Optical (~1.2-2 t, classified; ALOS/NEXTAR-heritage stand-in):
    box bus, nadir aperture, one wing pair."""
    b = _Builder("jp_igs_optical")
    b.box((0, 0, 0), (1.6, 1.6, 2.6), MLI, "bus",
          mats=[MLI, MLI, MLI, MLI, MLI, DARK])
    for sign, tag in [(1, "+y"), (-1, "-y")]:
        cy = sign * (0.8 + 0.25 + 1.7)
        b.panel((0, cy, 0.3), (1, 0, 0), (0, sign, 0), 1.9, 3.4, CELLS,
                PANEL_BACK, f"wing {tag}", gimbal=GIMBAL_1AXIS,
                gimbal_axis=(0, 1, 0))
    return b.build()


def jp_igs_radar() -> FacetModel:
    """IGS-Radar (~1.2 t): box bus, deployable planar SAR aperture, wing
    pair."""
    b = _Builder("jp_igs_radar")
    b.box((0, 0, 0), (1.5, 1.5, 2.2), MLI, "bus")
    b.panel((0, 0, -1.3), (0, 1, 0), (1, 0, 0), 3.6, 2.4, ANTENNA, MLI,
            "sar aperture")  # normal -z
    for sign, tag in [(1, "+y"), (-1, "-y")]:
        cy = sign * (0.75 + 0.25 + 1.6)
        b.panel((0, cy, 0.4), (1, 0, 0), (0, sign, 0), 1.8, 3.2, CELLS,
                PANEL_BACK, f"wing {tag}", gimbal=GIMBAL_1AXIS,
                gimbal_axis=(0, 1, 0))
    return b.build()


INTEL_ANNEX = [
    _intel("cn_yaogan_eo", cn_yaogan_eo,
           ["ops", "safe_sun", "tumble"], ["track", "frozen"]),
    _intel("cn_yaogan_sar", cn_yaogan_sar,
           ["ops", "safe_sun", "tumble"], ["track", "frozen"]),
    _intel("cn_yaogan_elint", cn_yaogan_elint,
           ["ops", "tumble"], []),
    _intel("ru_persona", ru_persona,
           ["ops", "safe_sun", "tumble"], ["track", "frozen"]),
    _intel("ru_bars_m", ru_bars_m,
           ["ops", "safe_sun", "tumble"], ["track", "frozen"]),
    _intel("ru_lotos_s", ru_lotos_s,
           ["ops", "safe_sun", "tumble"], ["track", "frozen"]),
    _intel("jp_igs_optical", jp_igs_optical,
           ["ops", "safe_sun", "tumble"], ["track", "frozen"]),
    _intel("jp_igs_radar", jp_igs_radar,
           ["ops", "safe_sun", "tumble"], ["track", "frozen"]),
]

_INTEL_FAMILY = {
    "cn_yaogan_eo": "eo_imager", "cn_yaogan_sar": "sar_sat",
    "cn_yaogan_elint": "smallsat", "ru_persona": "eo_imager",
    "ru_bars_m": "eo_imager", "ru_lotos_s": "science_bus",
    "jp_igs_optical": "eo_imager", "jp_igs_radar": "sar_sat",
}


def intel_annex():
    """(library, metadata) for the named intelligence-satellite annex."""
    library: dict = {}
    metadata: list[dict] = []
    for e in INTEL_ANNEX:
        shape = e["build"]()
        arr_modes = e["array_modes"] if shape.articulated else []
        metadata.append(dict(
            name=e["name"], family=_INTEL_FAMILY[e["name"]],
            country=e["name"].split("_")[0], n_facets=shape.n_facets,
            diffuse_albedo_area_m2=float(shape.diffuse_albedo_area().sum()),
            total_area_m2=float(shape.areas.sum()),
            articulated=bool(shape.articulated),
            attitude_modes=e["attitude_modes"], array_modes=arr_modes,
            annex="intel",
        ))
        library[e["name"]] = e["build"]
    return library, metadata


def full_library(seed: int = 11):
    """Generated 200 + intel annex + curated hand-built models, one
    namespace."""
    from .shapes import LIBRARY

    lib, meta = generate_library(seed)
    annex_lib, annex_meta = intel_annex()
    lib.update(annex_lib)
    meta.extend(annex_meta)
    for name, fn in LIBRARY.items():
        lib[name] = fn
    return lib, meta
