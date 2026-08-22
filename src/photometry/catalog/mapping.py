"""Catalog identifier / COSPAR / name pattern → family id.

Coverage means the fraction of active SATCAT objects that resolve to a
family template, not a unique mesh per NORAD. Confidence is `high` when
the name/COSPAR is unambiguous, `medium` for a public heuristic (e.g.
Starlink generation by launch date), `low` for class fallbacks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# First public v2 Mini launch: Group 6-1 on 2023-02-27
# (Spaceflight Now / SpaceNews). Group 5 (Dec 2022–Feb 2023) was still
# v1.5 into Gen2 shells. SATCAT does not tag generation; McDowell's
# per-object list was not vendored (planet4589 star page 404 from this
# environment). Confidence is therefore high before that date, medium after.
# Starshield is never inferred from a STARLINK name.
_STARLINK_V2MINI_START = "2023-02-27"


@dataclass(frozen=True)
class MapHit:
    family_id: str | None
    confidence: str          # high | medium | low | none
    rule: str
    notes: str = ""

    @property
    def mapped(self) -> bool:
        return self.family_id is not None


# Exact SATCAT names and COSPAR ids (high confidence).
_EXACT_NAME = {
    "ISS (ZARYA)": "iss",
    "HST": "hubble",
    "BLUEWALKER-3": "bluewalker3",
    "LINK": "katalyst_link",
    "CSS (TIANHE)": "css_tianhe",
    "TERRA": "terra",
    "AQUA": "terra",
    "SUOMI NPP": "jpss",
    "PAZ": "terrasat_x",
}
_EXACT_COSPAR = {
    "1998-067A": "iss",       # Zarya / ISS
    "1990-037B": "hubble",
    "2022-111AL": "bluewalker3",
    "2026-152A": "katalyst_link",
}

# Prefix / contains rules applied in order after Starlink.
# Each tuple: (family, kind, pattern, confidence, note)
# kind: prefix, contains, regex
_NAME_RULES: list[tuple[str, str, str, str, str]] = [
    ("oneweb", "prefix", "ONEWEB", "high", "OneWeb Arrow bus"),
    ("kuiper", "prefix", "KUIPER", "high", "Amazon Leo / Project Kuiper"),
    ("qianfan", "prefix", "QIANFAN", "high", "Qianfan / Thousand Sails / G60"),
    ("hulianwang", "prefix", "HULIANWANG", "medium",
     "Guowang; OML not in a primary filing"),
    ("hulianwang", "prefix", "GUOWANG", "medium", "Guowang test objects"),
    ("iridium_next", "prefix", "IRIDIUM", "high",
     "Iridium NEXT (Block-1 leftovers, if any, share the name)"),
    ("planet_superdove", "prefix", "FLOCK", "high", "Planet Dove / SuperDove"),
    ("planet_skysat", "prefix", "SKYSAT", "high", "Planet SkySat"),
    ("iceye", "prefix", "ICEYE", "high", "ICEYE SAR"),
    ("terrasat_x", "prefix", "TERRASAR", "high", "DLR hex + 5×0.8 m SAR"),
    ("terrasat_x", "prefix", "TANDEM-X", "high", "TerraSAR-X twin"),
    ("cosmo_skymed", "prefix", "COSMO-SKYMED", "high", "first-gen CSK; not CSG"),
    ("alos2", "prefix", "ALOS-", "high", "JAXA ALOS-2/4"),
    ("alos2", "prefix", "ALOS ", "high", ""),
    ("radarsat2", "prefix", "RADARSAT-2", "high", ""),
    ("gaofen3", "prefix", "GAOFEN-3", "high", "civil C-band SAR only"),
    ("saocom", "prefix", "SAOCOM", "high", ""),
    ("esa_swarm", "prefix", "SWARM ", "high", "ESA A/B/C; not SPACEBEE"),
    ("ion_scv", "prefix", "ION SCV", "high", "D-Orbit 60 cm / 64U"),
    ("ion_scv", "prefix", "ION-SCV", "high", ""),
    ("ghgsat", "prefix", "GHGSAT", "high", "SFL NEMO 20×30×40 cm"),
    ("grus", "prefix", "GRUS-", "high", "Axelspace ~0.6×0.6×0.8 m"),
    ("gnss_meo", "prefix", "NAVSTAR", "high", "GPS IIF/III stay on gnss_meo"),
    ("galileo", "contains", "GALILEO", "high", "ESA FOC, distinct from GPS-class"),
    ("gnss_meo", "contains", "GLONASS", "high", ""),
    ("gnss_meo", "prefix", "BEIDOU", "high", "BeiDou MEO; GEO/IGSO remapped by period"),
    ("o3b", "prefix", "O3B", "high", "Original O3b; mPOWER not separated"),
    ("capella", "prefix", "CAPELLA", "high", "FCC 3.5 m SAR"),
    ("umbra", "prefix", "UMBRA", "high", ""),
    ("hawkeye360", "prefix", "HAWK-", "high", "HawkEye 360 clusters"),
    ("hawkeye360", "prefix", "HAWKEYE", "high", ""),
    ("blacksky", "prefix", "GLOBAL-", "high", "BlackSky GLOBAL-n; not GLOBALSTAR"),
    ("globalstar2", "prefix", "GLOBALSTAR", "high", "ELiTeBus-1000"),
    ("orbcomm_og2", "prefix", "ORBCOMM", "high", ""),
    ("sentinel1", "prefix", "SENTINEL-1", "high", "ESA 12 m C-band SAR"),
    ("sentinel2", "prefix", "SENTINEL-2", "high", ""),
    ("sentinel2", "prefix", "SENTINEL-3", "medium", "optical Copernicus class; S2 OML"),
    ("landsat8", "prefix", "LANDSAT", "high", ""),
    ("maxar_legion", "prefix", "LEGION", "high", ""),
    ("worldview", "prefix", "WORLDVIEW", "high", ""),
    ("css_tianhe", "prefix", "CSS", "high", ""),
    ("jpss", "prefix", "NOAA 2", "high", "JPSS NOAA 20/21"),
    ("cygnus", "prefix", "CYGNUS", "high", ""),
    ("progress", "prefix", "PROGRESS", "high", ""),
    ("cargo_dragon", "prefix", "CARGO DRAGON", "high", ""),
    ("cargo_dragon", "prefix", "CREW DRAGON", "high", ""),
    ("cargo_dragon", "prefix", "SPACEX DRAGON", "high", ""),
    ("cargo_dragon", "prefix", "DRAGON CRS", "high", ""),
    ("cubesat_3u", "prefix", "LEMUR", "high", "Spire 3U"),
    ("cubesat_3u", "prefix", "KINEIS", "medium", "3U-class IoT"),
    ("cubesat_3u", "prefix", "TIANQI", "medium", "3U-class IoT"),
    ("cubesat_3u", "prefix", "AEROCUBE", "high", ""),
    ("cubesat_3u", "prefix", "TEVEL", "medium", ""),
    ("cubesat_3u", "prefix", "ASTROCAST", "high", "Astrocast 3U IoT"),
    ("cubesat_3u", "prefix", "FOSSASAT", "high", ""),
    ("cubesat_3u", "prefix", "FOSSASA", "medium", "SATCAT FOSSASA-2E23 typo"),
    ("cubesat_3u", "prefix", "BEESAT", "high", "TU Berlin 1U/3U"),
    ("cubesat_3u", "prefix", "SNUSAT", "high", "SNU 3U; not Satellogic NUSAT"),
    ("cubesat_3u", "prefix", "TIGRISAT", "high", ""),
    ("cubesat_3u", "prefix", "TRISAT", "high", ""),
    ("cubesat_3u", "prefix", "SAUDICOMSAT", "medium", "12 kg-class"),
    ("cubesat_3u", "prefix", "SAUDISAT", "medium", ""),
    ("cubesat_3u", "prefix", "APRIZESAT", "medium", "SpaceQuest ~12 kg"),
    ("cubesat_3u", "contains", "CUBESAT", "high", ""),
    ("cubesat_6u", "prefix", "TOMORROW-", "high", "Tomorrow.io 6U sounder"),
    ("cubesat_6u", "prefix", "WILDFIRE", "medium", "OroraTech 6U/8U class"),
    ("cubesat_6u", "prefix", "FOREST-", "medium", "OroraTech Forest"),
    ("cubesat_6u", "prefix", "TIANYI", "medium", "Changsha 6U-class"),
    ("cubesat_6u", "prefix", "CENTAURI", "medium", "Fleet Space 6U/12U"),
    ("cubesat_16u", "prefix", "IRIDE-MS1-EAGLET", "high",
     "OHB Eaglet II <30×30×57 cm"),
    ("cubesat_6u", "prefix", "SITRO", "medium", "Sitronics AIS 6U-class"),
    ("cubesat_6u", "prefix", "CONNECTA", "medium", ""),
    ("cubesat_6u", "prefix", "NUSAT-", "medium", "Satellogic NewSat / ÑuSat"),
    ("cubesat_6u", "prefix", "NUSAT ", "medium", "Satellogic NewSat / ÑuSat"),
    ("cubesat_6u", "prefix", "BRO-", "medium", "Unseenlabs BRO"),
    ("geo_bus", "prefix", "INTELSAT", "high", ""),
    ("geo_bus", "prefix", "EUTELSAT", "high", ""),
    ("geo_bus", "prefix", "SES-", "high", ""),
    ("geo_bus", "prefix", "ASTRA", "high", ""),
    ("geo_bus", "prefix", "INMARSAT", "high", ""),
    ("geo_bus", "prefix", "ECHOSTAR", "high", ""),
    ("geo_bus", "prefix", "JCSAT", "high", ""),
    ("geo_bus", "prefix", "GALAXY", "medium", "Intelsat Galaxy GEO"),
    ("geo_bus", "prefix", "DIRECTV", "high", ""),
    ("geo_bus", "prefix", "WGS", "high", "WGS GEO"),
    ("geo_bus", "prefix", "TDRS", "high", ""),
    ("geo_bus", "prefix", "ZHONGXING", "high", ""),
    ("geo_bus", "prefix", "TIANLIAN", "high", ""),
    ("geo_bus", "prefix", "GOES", "high", ""),
    ("geo_bus", "prefix", "EWS-G", "high", "ex-GOES"),
]


def _starlink(name: str, launch_date: str | None) -> MapHit:
    u = name.upper()
    if "[DTC]" in u:
        return MapHit("starlink_v2mini_dtc", "high", "starlink_dtc_tag",
                      "SATCAT [DTC] tag; DTC panel size is a range stand-in.")
    if launch_date and launch_date < _STARLINK_V2MINI_START:
        return MapHit("starlink_v15", "high", "starlink_pre_v2mini_date",
                      "Launch before 2023-02-27 (first public v2 Mini, "
                      "Group 6-1). v1.0 lumped with v1.5. Not Starshield.")
    if launch_date and launch_date >= _STARLINK_V2MINI_START:
        return MapHit("starlink_v2mini", "medium", "starlink_post_v2mini_date",
                      "Default v2 Mini after Group 6-1. SATCAT does not tag "
                      "generation; a minority of late v1.5 (Group 5 leftover "
                      "and Vandenberg v1.5) may be mixed in. DTC only if "
                      "[DTC] tagged. Not Starshield.")
    m = re.search(r"STARLINK-(\d+)", u)
    if m and int(m.group(1)) >= 30000:
        return MapHit("starlink_v2mini", "medium", "starlink_serial_30k",
                      "High serial without a launch date; treated as v2 Mini.")
    if m and int(m.group(1)) < 5000:
        return MapHit("starlink_v15", "medium", "starlink_serial_low",
                      "Low serial without a launch date; treated as v1.5.")
    return MapHit("starlink_v2mini", "low", "starlink_default",
                  "STARLINK name, generation not independently tagged.")


def _rocket(name: str) -> MapHit | None:
    u = name.upper()
    if "FALCON 9" in u or u.startswith("FALCON 9") or u.startswith("FALCON HEAVY"):
        return MapHit("falcon9_s2", "high" if "HEAVY" not in u else "medium",
                      "rb_falcon", "Heavy shares the 3.66 m second stage.")
    if u.startswith("CZ-") or u.startswith("CHANG ZHENG") or u.startswith("YZ-") \
            or " YZ-" in u:
        return MapHit("cz_upper", "medium", "rb_cz",
                      "CZ / YZ family; diameter is a range.")
    if u.startswith("SL-24") or u.startswith("BREEZE") or "BRIZ" in u:
        return MapHit("breeze_m", "high", "rb_breeze", "SL-24 is Rockot/Briz-KM.")
    if u.startswith("ARIANE"):
        return MapHit("ariane_upper", "medium", "rb_ariane", "")
    if "ELECTRON" in u:
        return MapHit("electron_kick", "high", "rb_electron", "")
    if "CENTAUR" in u:
        return MapHit("centaur", "high", "rb_centaur", "")
    if "AGENA" in u:
        return MapHit("agena", "high", "rb_agena", "")
    if u.startswith("IUS") or " IUS" in u:
        return MapHit("ius", "high", "rb_ius", "")
    if u.startswith("SCOUT"):
        return MapHit("scout", "medium", "rb_scout",
                      "Lingering Scout stages mix Antares 0.78 m / Altair 0.64 m.")
    if u.startswith("PEGASUS"):
        return MapHit("pegasus", "high", "rb_pegasus", "")
    if "PAM-D" in u or "PAM D" in u or "STAR 48" in u or "STAR-48" in u:
        return MapHit("pam_d", "high", "rb_pam_d", "")
    if "TRANSTAGE" in u or "TRANSTA" in u:
        return MapHit("titan_transtage", "high", "rb_transtage", "")
    if "ABLESTAR" in u:
        return MapHit("thor_ablestar", "high", "rb_ablestar", "")
    if "BURNER" in u:
        return MapHit("burner2", "high", "rb_burner2", "")
    if "THOR DELTA" in u or "THORAD DELTA" in u:
        return MapHit("delta_upper", "medium", "rb_thor_delta",
                      "Thor-Delta lingering stage; Delta-class 2.4 m stand-in.")
    if "ALTAIR" in u:
        return MapHit("scout", "medium", "rb_altair",
                      "Altair is Scout stage 4 (0.64 m); family is typical_class.")
    if u.startswith("TOS") or " TOS" in u:
        return MapHit("ius", "medium", "rb_tos",
                      "Transfer Orbit Stage is IUS-class 2.8 m.")
    if "PAM-S" in u or "PAM S" in u:
        return MapHit("pam_d", "high", "rb_pam_s", "STAR-48 PAM-S.")
    if u.startswith("FREGAT") or " FREGAT" in u:
        return MapHit("fregat", "high", "rb_fregat", "")
    if u.startswith("BLOCK DM"):
        return MapHit("block_dm", "high", "rb_block_dm", "")
    if u.startswith("PSLV") or u.startswith("GSLV"):
        return MapHit("pslv_ps4", "medium" if u.startswith("PSLV") else "low",
                      "rb_isro", "GSLV uses the PSLV-class 2.8–3 m stand-in.")
    if u.startswith("H-2") or u.startswith("H-II") or u.startswith("H2A") \
            or u.startswith("H3 "):
        return MapHit("h2_upper", "high", "rb_h2", "")
    if u.startswith("DELTA"):
        return MapHit("delta_upper", "medium", "rb_delta",
                      "2.4 m typical_class; Delta IV 4 m not split.")
    if u.startswith("SL-12"):
        return MapHit("proton_block_d", "high", "rb_sl12", "")
    if u.startswith("SL-8"):
        return MapHit("kosmos_3m", "high", "rb_sl8", "")
    if u.startswith("SL-14") or u.startswith("SL-19"):
        return MapHit("tsyklon3", "medium", "rb_sl14",
                      "SL-19 is Tsyklon-2; same 2.5 m-class stand-in.")
    if u.startswith("SL-16"):
        return MapHit("zenit2", "high", "rb_sl16", "")
    if u.startswith("SL-23") or u.startswith("DNEPR") or u.startswith("SS-18"):
        return MapHit("dnepr", "high", "rb_dnepr", "")
    if u.startswith("TITAN"):
        return MapHit("titan_core", "medium", "rb_titan_core",
                      "3.05 m Titan core; Transtage/Agena already matched.")
    if u.startswith("ATLAS"):
        return MapHit("atlas_core", "medium", "rb_atlas_core",
                      "3.05 m Atlas core; Centaur/Agena/STAR 48 already matched.")
    if u.startswith("SATURN"):
        return MapHit("saturn_sivb", "high", "rb_sivb", "")
    if u.startswith("AVUM"):
        return MapHit("avum", "high", "rb_avum", "")
    if "FIREFLY" in u:
        return MapHit("firefly_alpha", "high", "rb_firefly", "")
    if u.startswith("SL-3") or u.startswith("SL-4") or u.startswith("SL-6"):
        return MapHit("soyuz_block_i", "medium", "rb_r7",
                      "R-7 family lingering stage; 2.66 m Block I stand-in.")
    return MapHit("rocket_body", "low", "rb_generic",
                  "On-orbit R/B without a more specific public stage family.")


def _period(row_period: str | None) -> float | None:
    if not row_period:
        return None
    try:
        return float(row_period)
    except ValueError:
        return None


def resolve(name: str = "", *, cospar: str = "", object_type: str = "PAY",
            launch_date: str = "", period_min: str | float | None = None,
            ops_status: str = "") -> MapHit:
    """Map a catalog row to a family. Empty family_id means unmapped."""
    name = (name or "").strip()
    cospar = (cospar or "").strip().upper()
    object_type = (object_type or "PAY").upper()
    u = name.upper()

    if cospar in _EXACT_COSPAR:
        return MapHit(_EXACT_COSPAR[cospar], "high", "exact_cospar", "")
    if name in _EXACT_NAME:
        return MapHit(_EXACT_NAME[name], "high", "exact_name", "")

    if object_type in {"R/B", "RB"}:
        return _rocket(name)

    if u.startswith("STARLINK"):
        return _starlink(name, launch_date or None)

    # Galileo uses GSAT0xxx; Indian GEO uses GSAT-N. Distinguish.
    if u.startswith("GSAT0") or (u.startswith("GSAT") and "GALILEO" in u):
        return MapHit("galileo", "high", "galileo_gsat", "")

    if u.startswith("SPACEMOBILE"):
        m = re.search(r"(\d+)", u)
        n = int(m.group(1)) if m else 0
        if 1 <= n <= 5:
            return MapHit("bluebird", "high", "spacemobile_block1",
                          "BlueBird 1–5 = 64 m² array.")
        return MapHit("bluebird_block2", "high", "spacemobile_block2",
                      "BlueBird 6+ = 223 m² array.")

    if u.startswith("NOAA 20") or u.startswith("NOAA 21") or "JPSS" in u:
        return MapHit("jpss", "high", "jpss_noaa", "")

    if re.match(r"GOES[- ]1[6-9]\b", u):
        return MapHit("goes_r", "high", "goes_r_series",
                      "GOES-R series (16–19). Older GOES stay geo_bus.")

    if u.startswith("SENTINEL-6") or u.startswith("JASON"):
        return MapHit("sentinel6", "high" if u.startswith("SENTINEL-6") else "medium",
                      "sentinel6_jason",
                      "Jason-3 is the same altimetry class (typical_class).")

    if u.startswith("METOP-") and "SG" not in u:
        return MapHit("metop", "high", "metop_leos", "")

    p = _period(None if period_min is None else str(period_min))

    for family_id, kind, pat, conf, note in _NAME_RULES:
        matched = (
            (kind == "prefix" and u.startswith(pat))
            or (kind == "contains" and pat in u)
            or (kind == "regex" and re.search(pat, u))
        )
        if not matched:
            continue
        if family_id == "gnss_meo" and p is not None and 1200.0 < p < 1600.0:
            return MapHit("geo_bus", "medium", "beidou_geo",
                          "BeiDou / GNSS name in GEO/IGSO period band.")
        return MapHit(family_id, conf, f"{kind}:{pat}", note)

    if u.startswith("YAM-") or u.startswith("YAM "):
        return MapHit("classified_unpublished", "low", "yam_sda",
                      "York Space / SDA bus OML is not a public drawing.")

    if u.startswith("BANDWAGON") or u.startswith("ISS OBJECT"):
        return MapHit("cubesat_3u", "low", "rideshare_object",
                      "Unnamed rideshare; 3U stand-in, size uncertain.")

    if u.startswith("USA ") or u.startswith("IGS "):
        return MapHit("classified_unpublished", "medium",
                      "usa_classified" if u.startswith("USA ") else "igs_classified",
                      "No public OML; uncertain placeholder, not Starshield CAD.")

    if u.startswith("PRAETORIAN") or u.startswith("SDA_") or " SDA_" in u:
        return MapHit("classified_unpublished", "low", "sda_unpublished",
                      "SDA Transport Layer OML is not a public drawing.")

    if u.startswith("YAOGAN") or u.startswith("SHIYAN") or u.startswith("TJS"):
        return MapHit("classified_unpublished", "low", "cn_unpublished",
                      "No public photometric OML; not a fake SAR/optical CAD.")

    if "6U" in u:
        return MapHit("cubesat_6u", "medium", "name_6u", "")
    if "16U" in u:
        return MapHit("cubesat_16u", "medium", "name_16u", "")
    if "3U" in u or u.startswith("LEMUR") or "DOVE" in u:
        return MapHit("cubesat_3u", "medium", "name_3u", "")

    if u.startswith("TRANSPORTER") or u.startswith("OBJECT ") or u.startswith("OBJECT-"):
        return MapHit("cubesat_3u", "low", "rideshare_object",
                      "Unnamed rideshare; 3U stand-in, size uncertain.")

    p = _period(None if period_min is None else str(period_min))
    if p is not None and 1420.0 < p < 1460.0:
        return MapHit("geo_bus", "medium", "period_geo",
                      "GEO period band; typical 3-axis comms bus.")
    if p is not None and 700.0 < p < 780.0 and (
            "NAV" in u or "GPS" in u or "BEIDOU" in u):
        return MapHit("gnss_meo", "medium", "period_meo_nav", "")

    if object_type == "PAY":
        if p is not None and p < 200.0:
            return MapHit("leo_box_wing", "low", "leo_fallback",
                          "Active LEO payload without a named family.")
        if p is not None and 1420.0 < p < 1460.0:
            return MapHit("geo_bus", "low", "geo_fallback", "")
        return MapHit("leo_box_wing", "low", "pay_fallback",
                      "Active payload; class fallback box-wing.")

    return MapHit(None, "none", "unmapped", "")


def resolve_row(row: dict) -> MapHit:
    """Map a SATCAT-like dict (OBJECT_NAME, OBJECT_ID, …)."""
    return resolve(
        row.get("OBJECT_NAME", ""),
        cospar=row.get("OBJECT_ID", ""),
        object_type=row.get("OBJECT_TYPE", "PAY"),
        launch_date=row.get("LAUNCH_DATE", ""),
        period_min=row.get("PERIOD"),
        ops_status=row.get("OPS_STATUS_CODE", ""),
    )
