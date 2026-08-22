"""Load the vendored Celestrak SATCAT snapshot and report family coverage."""

from __future__ import annotations

import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from importlib import resources
from pathlib import Path

from .mapping import MapHit, resolve_row
from .registry import FAMILIES

# First whitespace token, then drop a trailing serial (JILIN-1, CYGFM05).
_PREFIX_SERIAL = re.compile(r"(?<=[A-Z])[-_]?\d.*$")


def name_prefix(name: str) -> str:
    """Reviewer grouping key for leftover SATCAT names.

    `JILIN-1 03` → `JILIN`, `GONETS-M 3` → `GONETS-M`, `IABS R/B` → `IABS`.
    Not a family id — only a prefix table so leftovers stay inspectable.
    """
    tok = (name or "").upper().strip().split()[0] if name else ""
    if not tok:
        return ""
    stripped = _PREFIX_SERIAL.sub("", tok).rstrip("-_")
    return stripped or tok

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


def leftover_inventory(rows: list[dict] | None = None) -> dict:
    """Top SATCAT prefixes still on `leo_box_wing` / generic `rocket_body`.

    Honest leftovers: the table does not invent families. Reviewers can see
    what the 1,857 / 144 fallbacks actually are.
    """
    rows = rows if rows is not None else load_snapshot()
    leo_prefix: Counter[str] = Counter()
    leo_examples: dict[str, list[str]] = defaultdict(list)
    rb_prefix: Counter[str] = Counter()
    rb_names: Counter[str] = Counter()
    rb_examples: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        h = resolve_row(r)
        name = r.get("OBJECT_NAME", "")
        if r.get("OBJECT_TYPE") == "PAY" and h.family_id == "leo_box_wing":
            p = name_prefix(name)
            leo_prefix[p] += 1
            if name not in leo_examples[p] and len(leo_examples[p]) < 5:
                leo_examples[p].append(name)
        elif r.get("OBJECT_TYPE") == "R/B" and h.family_id == "rocket_body":
            p = name_prefix(name)
            rb_prefix[p] += 1
            rb_names[name] += 1
            if name not in rb_examples[p] and len(rb_examples[p]) < 8:
                rb_examples[p].append(name)
    return {
        "leo_box_wing_n": sum(leo_prefix.values()),
        "leo_prefixes": [
            {"prefix": k, "n": v, "examples": leo_examples[k]}
            for k, v in leo_prefix.most_common()
        ],
        "rocket_body_n": sum(rb_prefix.values()),
        "rocket_prefixes": [
            {"prefix": k, "n": v, "examples": rb_examples[k]}
            for k, v in rb_prefix.most_common()
        ],
        "rocket_body_names": [
            {"name": k, "n": v} for k, v in rb_names.most_common()
        ],
    }
