"""Load the vendored Celestrak SATCAT snapshot and report family coverage."""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from importlib import resources
from pathlib import Path

from .mapping import MapHit, resolve_row
from .registry import FAMILIES

_META_NAME = "snapshot_meta.json"


def _data_dir():
    try:
        return resources.files("photometry.catalog").joinpath("data")
    except (ModuleNotFoundError, TypeError):
        return Path(__file__).resolve().parent / "data"


def snapshot_meta() -> dict:
    return json.loads((_data_dir() / _META_NAME).read_text())


def snapshot_path() -> Path:
    return Path(str(_data_dir() / snapshot_meta()["filename"]))


def load_snapshot(path: Path | None = None) -> list[dict]:
    """Active payloads + on-orbit rocket bodies from the vendored snapshot.

    Network is not required. Refresh with scripts/refresh_satcat_snapshot.py.
    """
    p = path or snapshot_path()
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def coverage_report(rows: list[dict] | None = None) -> dict:
    """Fraction of snapshot objects that resolve to a family, with breakdowns."""
    rows = rows if rows is not None else load_snapshot()
    meta = snapshot_meta()
    hits: list[tuple[dict, MapHit]] = [(r, resolve_row(r)) for r in rows]
    payloads = [(r, h) for r, h in hits if r["OBJECT_TYPE"] == "PAY"]
    rbs = [(r, h) for r, h in hits if r["OBJECT_TYPE"] == "R/B"]

    fallback = {"leo_box_wing", "rocket_body", "classified_unpublished"}

    def _summ(pairs):
        n = len(pairs)
        mapped = [h for _, h in pairs if h.mapped]
        named = [h for h in mapped if h.family_id not in fallback]
        by_fam = Counter(h.family_id for h in mapped)
        by_conf = Counter(h.confidence for h in mapped)
        by_rule = Counter(h.rule for h in mapped)
        return {
            "n": n,
            "n_mapped": len(mapped),
            "fraction": (len(mapped) / n) if n else 0.0,
            "n_named": len(named),
            "fraction_named": (len(named) / n) if n else 0.0,
            "by_family": dict(by_fam.most_common()),
            "by_confidence": dict(by_conf),
            "by_rule": dict(by_rule.most_common(20)),
        }

    unmapped = [r["OBJECT_NAME"] for r, h in hits if not h.mapped]
    missing_families = sorted(
        {h.family_id for _, h in hits if h.family_id and h.family_id not in FAMILIES})
    return {
        "snapshot_utc": meta.get("snapshot_utc"),
        "source": meta.get("source"),
        "n_families": len(FAMILIES),
        "family_ids": sorted(FAMILIES),
        "all": _summ(hits),
        "active_payloads": _summ(payloads),
        "rocket_bodies": _summ(rbs),
        "unmapped_examples": unmapped[:20],
        "unmapped_count": len(unmapped),
        "mapped_to_unknown_family": missing_families,
    }
