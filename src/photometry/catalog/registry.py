"""Family-id → factory. Includes the study LIBRARY plus catalog templates."""

from __future__ import annotations

from collections.abc import Callable

from ..shapes import LIBRARY, FacetModel
from . import families as _fam

FAMILIES: dict[str, Callable[[], FacetModel]] = {
    **LIBRARY,
    "starlink_v2": _fam.starlink_v2,
    "oneweb": _fam.oneweb,
    "kuiper": _fam.kuiper,
    "qianfan": _fam.qianfan,
    "hulianwang": _fam.hulianwang,
    "iridium_next": _fam.iridium_next,
    "planet_superdove": _fam.planet_superdove,
    "planet_skysat": _fam.planet_skysat,
    "iceye": _fam.iceye,
    "cubesat_3u": _fam.cubesat_3u,
    "cubesat_6u": _fam.cubesat_6u,
    "cubesat_16u": _fam.cubesat_16u,
    "geo_bus": _fam.geo_bus,
    "gnss_meo": _fam.gnss_meo,
    "o3b": _fam.o3b,
    "leo_box_wing": _fam.leo_box_wing,
    "classified_unpublished": _fam.classified_unpublished,
    "falcon9_s2": _fam.falcon9_s2,
    "cz_upper": _fam.cz_upper,
    "ariane_upper": _fam.ariane_upper,
    "breeze_m": _fam.breeze_m,
    "electron_kick": _fam.electron_kick,
}


def family(family_id: str) -> FacetModel:
    """Build the photometric stand-in for `family_id`."""
    try:
        return FAMILIES[family_id]()
    except KeyError as e:
        raise KeyError(f"unknown family {family_id!r}; known: {sorted(FAMILIES)}") from e


def list_families() -> list[str]:
    return list(FAMILIES)
