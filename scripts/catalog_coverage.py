#!/usr/bin/env python3
"""Print family coverage of the vendored Celestrak SATCAT snapshot (no network)."""

from __future__ import annotations

import json
import sys

from photometry.catalog import coverage_report


def main() -> int:
    r = coverage_report()
    pay, rb = r["active_payloads"], r["rocket_bodies"]
    print(f"snapshot {r['snapshot_utc']}  source {r['source']}")
    print(f"families ({r['n_families']}): {', '.join(r['family_ids'])}")
    print()
    print(f"active payloads: {pay['n_mapped']}/{pay['n']} mapped "
          f"({100*pay['fraction']:.1f}%), "
          f"named {pay['n_named']}/{pay['n']} "
          f"({100*pay['fraction_named']:.1f}%)")
    print(f"rocket bodies:   {rb['n_mapped']}/{rb['n']} mapped "
          f"({100*rb['fraction']:.1f}%), "
          f"named {rb['n_named']}/{rb['n']} "
          f"({100*rb['fraction_named']:.1f}%)")
    print()
    print("payloads by family:")
    for fam, n in pay["by_family"].items():
        print(f"  {n:6d}  {fam}")
    print()
    print("rocket bodies by family:")
    for fam, n in rb["by_family"].items():
        print(f"  {n:6d}  {fam}")
    print()
    print("payload confidence:", pay["by_confidence"])
    if r["unmapped_count"]:
        print(f"unmapped {r['unmapped_count']}: {r['unmapped_examples']}")
    if r["mapped_to_unknown_family"]:
        print("BUG: mapped to missing family", r["mapped_to_unknown_family"])
        return 1
    if "--json" in sys.argv:
        slim = {k: r[k] for k in
                ("snapshot_utc", "source", "n_families", "family_ids",
                 "unmapped_count", "mapped_to_unknown_family")}
        slim["active_payloads"] = {k: pay[k] for k in
                                   ("n", "n_mapped", "fraction", "n_named",
                                    "fraction_named", "by_family",
                                    "by_confidence")}
        slim["rocket_bodies"] = {k: rb[k] for k in
                                 ("n", "n_mapped", "fraction", "n_named",
                                  "fraction_named", "by_family")}
        print(json.dumps(slim, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
