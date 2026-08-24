"""Unified model registry: catalog families + generated library + annex.

The repo grew two model sources with complementary strengths:

  photometry.catalog   family templates for the *real* active catalog,
                       with a SATCAT name/COSPAR/orbit-class mapping
                       (`resolve`) that says which template an associated
                       object should be
  library200 + annex   the hypothetical mixed-country stress library the
                       catalog-scale identification study runs against,
                       plus the named intelligence-satellite annex

This module merges them into one namespace the identification funnel can
sweep, and — the operationally important part — turns the SATCAT mapping
into a *prior* for identification: association gives a candidate catalog
identity, `resolve` names the family template it should be, and that
prior (a) guarantees the mapped families survive the shortlist funnel
and (b) breaks ties inside the photometric-twin equivalence class, where
the catalog-scale study showed the data alone cannot pick a winner.

The prior deliberately acts ONLY within the twin margin: photometry
overrules a wrong catalog hint whenever the fit is decisive, so a
spoofed or stale association cannot force a bad identification — it can
only settle questions photometry calls a draw.
"""

from __future__ import annotations

import numpy as np

from .catalog import FAMILIES, resolve
from .library200 import generate_library, intel_annex

# catalog families that are spent stages / kick motors: tumble-only truth
_CATALOG_STAGES = {
    "falcon9_s2", "cz_upper", "ariane_upper", "breeze_m", "electron_kick",
    "centaur", "fregat", "soyuz_block_i", "proton_block_d", "kosmos_3m",
    "tsyklon3", "zenit2", "pslv_ps4", "h2_upper", "block_dm", "delta_upper",
    "ius", "agena", "scout", "pegasus", "pam_d", "titan_transtage",
    "atlas_core", "titan_core", "saturn_sivb", "avum", "firefly_alpha",
    "dnepr", "thor_ablestar", "burner2", "h3_upper", "iabs",
    "rocket_body",
}

_PRIOR_WEIGHT = {"high": 1.0, "medium": 0.6, "low": 0.3, "none": 0.0}


def _catalog_modes(fid: str, articulated: bool) -> tuple[list[str], list[str]]:
    if fid in _CATALOG_STAGES:
        return ["tumble"], []
    att = ["ops", "safe_sun", "tumble"]
    return att, (["track", "frozen"] if articulated else [])


def unified_library(seed: int = 11) -> tuple[dict, list[dict]]:
    """(library, metadata) across catalog families, library200, and annex.

    Catalog family ids are registry names (so `resolve()` hits are directly
    addressable); the curated study models ride along inside FAMILIES.
    Generated/annex names never collide with family ids by construction —
    collisions raise rather than silently shadow.
    """
    lib: dict = {}
    meta: list[dict] = []
    for fid, build in FAMILIES.items():
        shape = build()
        att, arr = _catalog_modes(fid, bool(shape.articulated))
        meta.append(dict(
            name=fid, family=fid, country="", source="catalog",
            n_facets=shape.n_facets,
            diffuse_albedo_area_m2=float(shape.diffuse_albedo_area().sum()),
            total_area_m2=float(shape.areas.sum()),
            articulated=bool(shape.articulated),
            attitude_modes=att, array_modes=arr,
        ))
        lib[fid] = build
    for source, (l2, m2) in [("generated", generate_library(seed)),
                             ("intel_annex", intel_annex())]:
        for m in m2:
            if m["name"] in lib:
                raise ValueError(f"registry collision: {m['name']}")
            meta.append(dict(m, source=source))
        lib.update(l2)
    return lib, meta


def satcat_prior(identities: list[str | dict]) -> dict[str, float]:
    """Association candidates -> {registry name: prior weight}.

    Each identity is a SATCAT object name (or a SATCAT-like row dict);
    `catalog.resolve` maps it to a family template with a confidence,
    which becomes a weight (high 1.0 / medium 0.6 / low 0.3). Multiple
    candidate identities union with max weight. Unmapped identities
    contribute nothing — the funnel then runs purely photometric.
    """
    prior: dict[str, float] = {}
    for ident in identities:
        if isinstance(ident, dict):
            from .catalog import resolve_row
            hit = resolve_row(ident)
        else:
            hit = resolve(str(ident))
        if not hit.mapped:
            continue
        w = _PRIOR_WEIGHT.get(hit.confidence, 0.0)
        if w > 0:
            prior[hit.family_id] = max(prior.get(hit.family_id, 0.0), w)
    return prior


def seed_shortlist(shortlist: list[str], prior: dict[str, float],
                   library: dict) -> list[str]:
    """Guarantee prior families survive the funnel: append any mapped
    family present in the registry but missing from the shortlist."""
    out = list(shortlist)
    for name, w in sorted(prior.items(), key=lambda kv: -kv[1]):
        if w > 0 and name in library and name not in out:
            out.append(name)
    return out


def rerank_with_prior(
    ranked: list[tuple[str, float]],
    prior: dict[str, float],
    twin_margin: float = 1.25,
) -> tuple[list[tuple[str, float]], dict]:
    """Reorder the photometric ranking using the catalog prior — but only
    inside the twin equivalence class.

    `ranked` is [(name, cost), ...] best-first (one row per model). The
    equivalence class is every model with cost <= twin_margin * best —
    the band the catalog-scale study measured as photometrically
    undecidable (twins fit within ~1.2x of truth). Within that band,
    order by prior weight (desc), then cost. Outside it, photometry
    stands untouched: a decisive fit beats any catalog hint.

    Returns (reranked, info) where info records the class size and
    whether the prior changed the top-1.
    """
    if not ranked:
        return ranked, dict(twin_class=0, prior_applied=False, changed_top1=False)
    best = ranked[0][1]
    cls = [r for r in ranked if r[1] <= twin_margin * max(best, 1e-9)]
    rest = ranked[len(cls):]
    reord = sorted(cls, key=lambda r: (-prior.get(r[0], 0.0), r[1]))
    changed = bool(reord and ranked and reord[0][0] != ranked[0][0])
    applied = any(prior.get(n, 0.0) > 0 for n, _ in cls)
    return reord + rest, dict(twin_class=len(cls), prior_applied=applied,
                              changed_top1=changed,
                              twin_names=[n for n, _ in cls])


def prior_report(prior: dict[str, float], meta: list[dict]) -> str:
    by_name = {m["name"]: m for m in meta}
    lines = []
    for name, w in sorted(prior.items(), key=lambda kv: -kv[1]):
        m = by_name.get(name)
        tag = f"{m['source']}" if m else "NOT IN REGISTRY"
        lines.append(f"  {name} (w={w:.1f}, {tag})")
    return "\n".join(lines) if lines else "  (no mapped identities)"
