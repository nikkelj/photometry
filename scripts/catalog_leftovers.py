#!/usr/bin/env python3
"""Top SATCAT prefixes still on leo_box_wing / generic rocket_body.

Does not invent families. Reviewers can see the leftover 1,857 / 144.
No network — uses the vendored snapshot.
"""

from __future__ import annotations

import argparse
import json
import sys

from photometry.catalog import leftover_inventory


def _table(rows, *, n_col="n", key="prefix", limit=None):
    out = []
    for i, row in enumerate(rows):
        if limit is not None and i >= limit:
            break
        ex = row.get("examples") or []
        sample = (", ".join(ex[:3])) if ex else ""
        out.append(f"  {row[n_col]:4d}  {row[key]:28s}  {sample}".rstrip())
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true")
    p.add_argument("--all-prefixes", action="store_true",
                   help="Print every leftover prefix, not the top 40.")
    args = p.parse_args()
    inv = leftover_inventory()
    if args.json:
        print(json.dumps(inv, indent=2))
        return 0
    lim = None if args.all_prefixes else 40
    print(f"leo_box_wing leftovers: {inv['leo_box_wing_n']}")
    print("top prefixes (first SATCAT token, trailing serial stripped):")
    print(_table(inv["leo_prefixes"], limit=lim))
    print()
    print(f"generic rocket_body leftovers: {inv['rocket_body_n']}")
    print("prefixes:")
    print(_table(inv["rocket_prefixes"]))
    print()
    print("exact leftover R/B names:")
    print(_table(inv["rocket_body_names"], key="name"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
