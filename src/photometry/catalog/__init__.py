"""Open-source spacecraft facet catalog.

Family templates (bus + arrays + deployables + thrust vs LVLH + per-facet
materials) mapped onto Celestrak SATCAT names / COSPAR / orbit class.
This is not a parallel shape format: every family builds a `FacetModel`
the existing forward model already consumes. The study `shapes.LIBRARY`
is unchanged so the 620 km inversion scenarios keep the same candidate set.

    from photometry.catalog import family, resolve, coverage_report
    m = family("starlink_v2mini")
    print(m.describe())
    hit = resolve(name="STARLINK-11072 [DTC]")
"""

from .mapping import MapHit, resolve, resolve_row
from .registry import FAMILIES, family, list_families
from .satcat import coverage_report, load_snapshot, snapshot_meta

__all__ = [
    "FAMILIES",
    "MapHit",
    "coverage_report",
    "family",
    "list_families",
    "load_snapshot",
    "resolve",
    "resolve_row",
    "snapshot_meta",
]
