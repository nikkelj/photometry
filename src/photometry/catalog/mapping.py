"""Catalog identifier / COSPAR / name pattern → family id.

Coverage means the fraction of active SATCAT objects that resolve to a
family template, not a unique mesh per NORAD. Confidence is `high` when
the name/COSPAR is unambiguous, `medium` for a public heuristic (e.g.
Starlink generation by launch date), `low` for class fallbacks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# First v2 Mini launch (Spaceflight Now / SpaceX): 2023-02-27.
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
    ("gnss_meo", "prefix", "NAVSTAR", "high", "GPS"),
    ("gnss_meo", "contains", "GALILEO", "high", ""),
    ("gnss_meo", "contains", "GLONASS", "high", ""),
    ("gnss_meo", "prefix", "BEIDOU", "high", "BeiDou MEO/GEO/IGSO split later"),
    ("o3b", "prefix", "O3B", "high", "Original O3b; mPOWER not separated"),
    ("cubesat_3u", "prefix", "LEMUR", "high", "Spire 3U"),
    ("cubesat_3u", "prefix", "KINEIS", "medium", "3U-class IoT"),
    ("cubesat_3u", "prefix", "TIANQI", "medium", "3U-class IoT"),
    ("cubesat_3u", "contains", "CUBESAT", "high", ""),
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
]


def _starlink(name: str, launch_date: str | None) -> MapHit:
    u = name.upper()
    if "[DTC]" in u:
        return MapHit("starlink_v2mini_dtc", "high", "starlink_dtc_tag",
                      "SATCAT [DTC] tag; DTC panel size is a range stand-in.")
    if launch_date and launch_date < _STARLINK_V2MINI_START:
        return MapHit("starlink_v15", "medium", "starlink_pre_v2mini_date",
                      "Launch before 2023-02-27 → v1.0/v1.5 lumped as v1.5. "
                      "v1.0 vs v1.5 is not split (public dims nearly share).")
    if launch_date and launch_date >= _STARLINK_V2MINI_START:
        return MapHit("starlink_v2mini", "medium", "starlink_post_v2mini_date",
                      "Default v2 Mini after 2023-02-27. A minority of late "
                      "v1.5 may be mixed in; DTC only if [DTC] tagged.")
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
    if "FALCON 9" in u or u.startswith("FALCON 9"):
        return MapHit("falcon9_s2", "high", "rb_falcon9", "")
    if u.startswith("FALCON HEAVY"):
        return MapHit("falcon9_s2", "medium", "rb_falcon_heavy",
                      "Same 3.66 m second stage as F9.")
    if u.startswith("CZ-") or u.startswith("CHANG ZHENG") or " YZ-" in u:
        return MapHit("cz_upper", "medium", "rb_cz", "CZ/YZ family; diameter is a range.")
    if u.startswith("ARIANE"):
        return MapHit("ariane_upper", "medium", "rb_ariane", "")
    if u.startswith("BREEZE"):
        return MapHit("breeze_m", "high", "rb_breeze", "")
    if "ELECTRON" in u:
        return MapHit("electron_kick", "high", "rb_electron", "")
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
        return MapHit("gnss_meo", "high", "galileo_gsat", "")

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

    if u.startswith("USA "):
        return MapHit("classified_unpublished", "medium", "usa_classified",
                      "No public OML; uncertain placeholder, not Starshield CAD.")

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
