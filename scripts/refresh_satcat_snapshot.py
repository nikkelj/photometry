#!/usr/bin/env python3
"""Refresh the vendored Celestrak SATCAT snapshot (network).

CI tests use the committed gzip; this script is for maintainers.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

URL = "https://celestrak.org/pub/satcat.csv"
KEEP = ["OBJECT_NAME", "OBJECT_ID", "NORAD_CAT_ID", "OBJECT_TYPE",
        "OPS_STATUS_CODE", "OWNER", "LAUNCH_DATE", "PERIOD",
        "INCLINATION", "APOGEE", "PERIGEE"]
ACTIVE = set("+PBSX")
OUT_DIR = Path(__file__).resolve().parents[1] / "src/photometry/catalog/data"


def main() -> int:
    req = urllib.request.Request(URL, headers={"User-Agent": "photometry-catalog/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        last_mod = resp.headers.get("Last-Modified", "")
        raw = resp.read()
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8", errors="replace"))))
    subset = []
    for r in rows:
        if r.get("ORBIT_TYPE") != "ORB" or r.get("DECAY_DATE"):
            continue
        if r.get("OBJECT_TYPE") == "PAY" and r.get("OPS_STATUS_CODE") in ACTIVE:
            subset.append(r)
        elif r.get("OBJECT_TYPE") == "R/B":
            subset.append(r)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    gz_name = f"satcat_active_{day}.csv.gz"
    gz_path = OUT_DIR / gz_name
    with gzip.open(gz_path, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, KEEP, extrasaction="ignore")
        w.writeheader()
        for r in subset:
            w.writerow({k: r.get(k, "") for k in KEEP})
    meta = {
        "source": URL,
        "source_last_modified_utc": last_mod,
        "snapshot_utc": day,
        "selection": "Earth-orbit PAY with OPS_STATUS_CODE in +PBSX plus on-orbit R/B",
        "n_rows": len(subset),
        "n_active_payloads": sum(1 for r in subset if r["OBJECT_TYPE"] == "PAY"),
        "n_rocket_bodies": sum(1 for r in subset if r["OBJECT_TYPE"] == "R/B"),
        "filename": gz_name,
    }
    (OUT_DIR / "snapshot_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", gz_path, "rows", len(subset))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
