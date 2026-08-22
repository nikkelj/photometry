"""Family-id → factory. Includes the study LIBRARY plus catalog templates."""

from __future__ import annotations

from collections.abc import Callable

from ..shapes import LIBRARY, FacetModel
from . import families as _fam
from . import families_more as _more
from . import families_pass3 as _p3

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
    "galileo": _more.galileo,
    "o3b": _fam.o3b,
    "leo_box_wing": _fam.leo_box_wing,
    "classified_unpublished": _fam.classified_unpublished,
    "capella": _more.capella,
    "umbra": _more.umbra,
    "hawkeye360": _more.hawkeye360,
    "blacksky": _more.blacksky,
    "globalstar2": _more.globalstar2,
    "orbcomm_og2": _more.orbcomm_og2,
    "sentinel1": _more.sentinel1,
    "sentinel2": _more.sentinel2,
    "landsat8": _more.landsat8,
    "css_tianhe": _more.css_tianhe,
    "maxar_legion": _more.maxar_legion,
    "worldview": _more.worldview,
    "bluebird": _more.bluebird,
    "bluebird_block2": lambda: _more.bluebird(block2=True),
    "terra": _more.terra,
    "jpss": _more.jpss,
    "goes_r": _more.goes_r,
    "sentinel6": _more.sentinel6,
    "metop": _more.metop,
    "cygnus": _more.cygnus,
    "progress": _more.progress,
    "cargo_dragon": _more.cargo_dragon,
    "falcon9_s2": _fam.falcon9_s2,
    "cz_upper": _fam.cz_upper,
    "ariane_upper": _fam.ariane_upper,
    "breeze_m": _fam.breeze_m,
    "electron_kick": _fam.electron_kick,
    "centaur": _more.centaur,
    "fregat": _more.fregat,
    "soyuz_block_i": _more.soyuz_block_i,
    "proton_block_d": _more.proton_block_d,
    "kosmos_3m": _more.kosmos_3m,
    "tsyklon3": _more.tsyklon3,
    "zenit2": _more.zenit2,
    "pslv_ps4": _more.pslv_ps4,
    "h2_upper": _more.h2_upper,
    "block_dm": _more.block_dm,
    "delta_upper": _more.delta_upper,
    "ius": _more.ius,
    "agena": _more.agena,
    "scout": _more.scout,
    "pegasus": _more.pegasus,
    "pam_d": _more.pam_d,
    "titan_transtage": _more.titan_transtage,
    "terrasat_x": _p3.terrasat_x,
    "cosmo_skymed": _p3.cosmo_skymed,
    "alos2": _p3.alos2,
    "radarsat2": _p3.radarsat2,
    "gaofen3": _p3.gaofen3,
    "saocom": _p3.saocom,
    "esa_swarm": _p3.esa_swarm,
    "ion_scv": _p3.ion_scv,
    "ghgsat": _p3.ghgsat,
    "grus": _p3.grus,
    "atlas_core": _p3.atlas_core,
    "titan_core": _p3.titan_core,
    "saturn_sivb": _p3.saturn_sivb,
    "avum": _p3.avum,
    "firefly_alpha": _p3.firefly_alpha,
    "dnepr": _p3.dnepr,
    "thor_ablestar": _p3.thor_ablestar,
    "burner2": _p3.burner2,
}


def family(family_id: str) -> FacetModel:
    """Build the photometric stand-in for `family_id`."""
    try:
        return FAMILIES[family_id]()
    except KeyError as e:
        raise KeyError(f"unknown family {family_id!r}; known: {sorted(FAMILIES)}") from e


def list_families() -> list[str]:
    return list(FAMILIES)
