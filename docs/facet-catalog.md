# Spacecraft facet catalog

Open-source **family templates** for the active public catalog, mapped onto
Celestrak SATCAT names / COSPAR / orbit class. This is not a parallel CAD
format and not 10,000 unique meshes: every family builds a `FacetModel` the
existing forward model already consumes (`rho_d`, `k_s`, `n_ph`, gimbals).

The 620 km study-orbit library (`photometry.shapes.LIBRARY`) is **unchanged**
so fleet identification numbers stay reproducible. New families live in
`photometry.catalog.FAMILIES`, which includes the study set plus the rest.

```python
from photometry.catalog import family, resolve, coverage_report

m = family("starlink_v2mini")
print(m.describe())          # bus, arrays + gimbal, deployables, thrust, materials
hit = resolve("STARLINK-11072 [DTC]")
print(hit.family_id)         # starlink_v2mini_dtc
print(coverage_report()["active_payloads"]["fraction_named"])
```

```bash
python scripts/catalog_coverage.py          # uses the vendored snapshot (no network)
python -m pytest tests/test_catalog.py tests/test_basics.py
```

## Schema (what a family must carry)

1. **Bus** — box or cylinder facets.
2. **Solar arrays** with an explicit gimbal: `GIMBAL_FIXED` / `GIMBAL_1AXIS` /
   `GIMBAL_2AXIS`. BlueWalker-class sheets stay fixed.
3. **Deployables** when they are photometrically large and public (DTC panel,
   ICEYE 3.2×0.4 m SAR, GEO nadir dish, ISS radiators).
4. **Thrust vs nominal flight attitude.** Body-frame unit vector(s) when
   public (ISS reboost = `+x` in LVLH). Empty array + `thrust_notes` when
   pointing is unpublished — **Hall thruster pointing is never guessed**.
   Attitude labels: `lvlh` (`+x` ram, `+y` orbit-normal, `+z` zenith),
   `nadir`, `yaw_steering`, `stage_axis`, `unknown`.
5. **Per-facet surface properties.** Material class + Lambert–Phong
   `(rho_d, k_s, n_ph)` on every facet. IR α/ε are `NaN` / provenance
   `unknown` unless a spacecraft-specific public number exists (none of the
   current families have one). Handbook class figures are listed in
   `catalog/materials.py` and are **not** copied onto facets as fake
   precision. Optical BRDFs are assumed photometric stand-ins.

Dimension provenance is `public` | `range` | `uncertain` | `typical_class` |
`unknown`. Range means the numeric stand-in is a published-range midpoint.

## Families

| id | What | Arrays | Thrust |
|---|---|---|---|
| `starlink_v15` | FCC v1.5 (v1.0 lumped) | 1-axis | EP public, vector unknown |
| `starlink_v2mini` / `_dtc` | FCC v2 Mini; DTC if SATCAT `[DTC]` | 2-axis | EP public, vector unknown |
| `starlink_v2` | FCC Starship-class v2 | 2-axis | not mapped from SATCAT names |
| `oneweb` | Airbus Arrow 1×1×1.3 m, 5 m span | 1-axis | xenon Hall, vector unknown |
| `kuiper` | Amazon Leo; mass public, OML a range | 1-axis stand-in | Kr Hall, vector unknown |
| `qianfan` | G60 / Thousand Sails flat-pack | 1-axis (Mallama) | Kr Hall reported, vector unknown |
| `hulianwang` | Guowang; Gunter: OML unpublished | 1-axis stand-in | EP believed, vector unknown |
| `iridium_next` | 9.4 m span, sun-tracking | 1-axis | hydrazine, vector unknown |
| `planet_superdove` | 3U Flock | fixed | none public |
| `planet_skysat` | 60×60×95 cm | 1-axis stand-in | 180 m/s public, vector unknown |
| `iceye` | bus range + 3.2×0.4 m SAR | 1-axis + fixed SAR | unknown |
| `cubesat_3u` / `6u` / `16u` | CDS envelopes | fixed | unknown |
| `geo_bus` | typical 3-axis GEO | 2-axis + nadir dish | NSSK +y class convention |
| `gnss_meo` | GPS III / GNSS class | 2-axis | chemical, vector unknown |
| `o3b` | original O3b MEO | 2-axis | chemical, vector unknown |
| `iss` / `hubble` / `bluewalker3` / `katalyst_link` | existing study models | as before | ISS +x public; others unknown/none |
| `falcon9_s2` / `cz_upper` / `ariane_upper` / `breeze_m` / `electron_kick` / `rocket_body` | stages | n/a | −z along cylinder |
| `leo_box_wing` | LEO payload fallback | fixed | unknown |
| `classified_unpublished` | USA / unpublished recon | uncertain placeholder | unknown; **not Starshield CAD** |

Starshield / classified USA objects map to `classified_unpublished` with
every dimension tagged uncertain. SpaceX internals beyond the public FCC
table are not invented.

## Mapping (name / COSPAR → family)

`catalog/mapping.py` is ordered: exact COSPAR/name → Starlink generation
heuristic → named prefixes → rocket-body patterns → GEO period → class
fallbacks.

Starlink generation:

- SATCAT `[DTC]` → `starlink_v2mini_dtc` (high)
- launch date before 2023-02-27 (first public v2 Mini launch) → `starlink_v15`
  (medium; v1.0 lumped with v1.5)
- later launches → `starlink_v2mini` (medium; a minority of late v1.5 may be
  mixed in because SATCAT does not tag generation)

## Coverage snapshot

Vendored Celestrak SATCAT extract (no network in CI):

- file: `src/photometry/catalog/data/satcat_active_2026-08-21.csv.gz`
- source: https://celestrak.org/pub/satcat.csv (Last-Modified 2026-08-21)
- selection: Earth-orbit payloads with ops status `+ P B S X` (Celestrak
  “active”) plus on-orbit rocket bodies

Refresh (optional, hits the network):

```bash
python scripts/refresh_satcat_snapshot.py
```

`fraction` is “resolves to some family” (including fallbacks).
`fraction_named` excludes `leo_box_wing`, `rocket_body`, and
`classified_unpublished`.

## What is unknown (on purpose)

- Starlink / Kuiper / OneWeb / Qianfan **Hall pointing** vs LVLH
- Amazon Leo outer-mold-line (range only)
- Guowang OML (Gunter: not published)
- Starshield and NRO buses
- Spacecraft-specific IR α/ε
- Array 1- vs 2-axis when the public record is silent (called out as a
  stand-in on Kuiper, Hulianwang, SkySat)

Estimator, sensing FOV, and ADCS tracker geometry are not part of this
catalog.
