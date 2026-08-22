# Spacecraft facet catalog

Family templates for the public Celestrak SATCAT, each a `FacetModel` the
existing forward model already consumes. Not a parallel CAD format, not
10,000 unique meshes, not a coverage chase.

```python
from photometry.catalog import family, resolve, coverage_report

m = family("starlink_v2mini")
print(m.describe())                    # bus, gimbals, looks, thrust, materials
hit = resolve("STARLINK-11072 [DTC]")  # → starlink_v2mini_dtc
print(coverage_report()["active_payloads"]["fraction_named"])
```

```bash
python scripts/catalog_coverage.py     # vendored snapshot, no network
python scripts/catalog_leftovers.py    # leftover prefixes
python -m pytest tests/test_catalog.py tests/test_basics.py
```

## Schema

Every family is a `FacetModel` (`shapes.py`) with:

1. **Bus** — box or cylinder facets.
2. **Solar arrays** on the same `body_normals` path as the study library:
   `GIMBAL_FIXED`, `GIMBAL_1AXIS` (Rodrigues about `gimbal_axis`; cosine
   loss remains), `GIMBAL_2AXIS` (shoulder then wrist). Travel defaults to
   ±π when unpublished so study LIBRARY sun-track stays numerically the same.
3. **Deployables** when they are photometrically large and public, plus a
   documented **look** vs `lvlh` / `nadir` / `yaw_steering`. Unpublished
   looks stay `unknown`.
4. **Thrust vs flight attitude.** Body-frame unit vector when a *primary*
   citation exists (ISS reboost = `+x` in LVLH). Empty `(0,3)` + notes
   otherwise. **Hall / ion pointing is never guessed.** LVLH here is `+x`
   ram, `+y` orbit-normal, `+z` zenith.
5. **Per-facet materials.** Optical Lambert–Phong `(rho_d, k_s, n_ph)` on
   every facet (assumed stand-ins). IR α/ε are `NaN` / `unknown` unless a
   spacecraft-specific public number exists (none do). Handbook class
   figures in `catalog/materials.py` are **not** copied onto facets.

Dimension provenance: `public` | `range` | `uncertain` | `typical_class` |
`unknown`. Range = published-range midpoint.

92 family ids live in `catalog/registry.py`. Builders are in
`families.py` / `families_more.py` / `families_pass3.py`. This doc does
not duplicate that list.

## Study LIBRARY vs catalog stand-ins

`photometry.shapes.LIBRARY` (9 study models) is **unchanged** so the 620 km
inversion stays reproducible. Catalog SATCAT mapping overrides four ids
with higher-fidelity stand-ins; `family("…")` returns the catalog copy,
`shapes.starlink_v15()` / `shapes.iss()` still return the study mesh.

| id | study LIBRARY | catalog SATCAT stand-in |
|---|---|---|
| `starlink_v15` | bus y=1.4 m, array chord 2.7 m | FCC 2.8×1.3 m bus, 2.8×8.1 m 1-axis array |
| `starlink_v2mini` / `_dtc` | already FCC 4.1×2.7 / 4.1×12.8 | same OML; catalog adds 2-axis hinges + nadir look |
| `iss` | two 35×24 m array groups | NASA ~2,500 m² arrays, 109 m truss |

Catalog high-count families also carry rest-pose `polygons` (viz only).
Photometry uses areas + normals. Study LIBRARY drawable lists are not
filled the same way (v1.5 study stays 7 polys; catalog v1.5 is 8).

## Coverage (85% / 94% stages)

Vendored extract, no network in CI:

- `src/photometry/catalog/data/satcat_active_2026-08-21.csv.gz`
- Celestrak `satcat.csv`, Last-Modified 2026-08-21
- Earth-orbit payloads with ops status `+ P B S X` plus on-orbit R/B

`fraction` includes fallbacks. `fraction_named` excludes
`leo_box_wing`, `rocket_body`, and `classified_unpublished`.

| | n | named | named % | leftover |
|---|---:|---:|---:|---|
| active payloads | 16,870 | 14,334 | **85.0%** | `leo_box_wing` 1,846 |
| rocket bodies | 2,422 | 2,290 | **94.5%** | `rocket_body` 132 |

Earlier cuts on the same snapshot: 82.3% / 25.5% → 84.0% / 90.0% →
84.9% / 94.1%. **92** families. Payload confidence: low 2,482 / medium
8,967 / high 5,421. `classified_unpublished` 690.

Optional refresh (hits the network): `python scripts/refresh_satcat_snapshot.py`.

## Leftover prefixes

`scripts/catalog_leftovers.py` (`leftover_inventory()`). Prefix = first
whitespace token with a trailing serial stripped. **No family is invented
from a prefix.**

Top `leo_box_wing` prefixes (1,846 total):

| n | prefix | examples |
|---:|---|---|
| 111 | COSMOS | COSMOS 1989 (ETALON 1), COSMOS 2385 |
| 63 | GEESAT | GEESAT-1 01 |
| 48 | JILIN | JILIN-1, JILIN-1 03 |
| 33 | CENTISPACE | CENTISPACE-1 S1 |
| 31 | RASSVET | RASSVET-3 1 |
| 30 | GAOFEN | GAOFEN-1, GAOFEN-2 (optical; GF-3 is named) |
| 23 | GONETS-M | GONETS-M 3 |
| 20 | TIANMU | TIANMU-1 03 |
| 18 | YUNHAI | YUNHAI-1 01 |
| 17 | SHIJIAN | SHIJIAN-6 01A |
| 15 | IRIDE-MS | IRIDE-MS2-HEO-1 |
| 14 | TIANHUI | TIANHUI 2-01A |
| 13 | SUPERVIEW | SUPERVIEW NEO-1 01 |
| 12 | SJ / HJS / SCS | mixed Chinese leftovers |
| 11 | QPS-SAR / AETHER / CHUANGXIN | unpublished or mixed OML |
| 9 | PELICAN / FORMOSAT / CHECKMATE | Planet Pelican unpublished; FORMOSAT-5 ≠ 7 |

Also left on purpose (no primary public OML): optical Gaofen except GF-3,
SuperView Neo, GEESat, CentiSpace, Yunhai, Haiyang, Tianhui, Tianmu,
CSG (not first-gen CSK), Pelican, Rassvet, Gonets-M, IRIDE HEO/FM,
COSMOS mixed buses, QPS-SAR, DiskSat, FORMOSAT mix, STRIX. Yaogan stays
`classified_unpublished`.

Generic `rocket_body` leftovers (**132**):

| n | names | why still generic |
|---:|---|---|
| ~40 | mixed `* AKM` / `* PKM` | sizes differ — not lumped |
| 8 | MINOTAUR / MINOTAUR 1 / 4 | Minotaur I ≠ IV diameter |
| 7 | OV1-* | 1960s, mixed |
| 6 / 6 | Japanese N-1/N-2, M-3S/M-4S | not H-IIA 4 m |
| 5 | DIAMANT | mixed B / B-P4 |
| 4 | TAURUS R/B | Taurus / Minotaur-C mix |
| 3 | KSLV-II, LIJIAN-1 | no primary stage drawing used here |
| 2 | H-1, EPSILON, M-V, VANGUARD, VOLGA, … | H-3 is `h3_upper`; H-1 is not |
| 1 | FALCON 1, NEW GLENN, SLS, LVM3, … | one-offs |

Cited stage maps already in: IABS 2.9×0.68 m, H-3 5.27×12 m, SL-11 →
`tsyklon3`. Starling → existing `cubesat_6u`. `CYGFM` → `cygnss`.

## What is unknown (on purpose)

- **Hall / Δv pointing** vs LVLH for Starlink, Kuiper, OneWeb, Iridium
  NEXT. Primary FCC / ITU / papers say the thruster exists; they do not
  give a body-frame axis. Everyday Astronaut “ram-facing” is not a
  primary citation. Vectors stay empty.
- **IR α/ε** — spacecraft-specific numbers. Handbook class figures are
  not copied onto facets.
- **Chinese optical OML** — Jilin mixed buses, optical Gaofen, SuperView,
  GEESat, CentiSpace, Yunhai, and the leftover table above.
- Amazon Leo OML (range only); Guowang OML (Gunter: unpublished);
  Starshield / NRO (never inferred from a STARLINK name).
- ICEYE / Umbra SAR look; ISS radiator thermal schedule.
- Array 1- vs 2-axis when the public record is silent (called out as a
  stand-in).

Estimator, sensing FOV, and ADCS tracker geometry are not part of this
catalog.

## Photometry sanity (not a new renderer)

Catalog families are already `FacetModel`s. Hook:

```python
from photometry.catalog import family
from photometry.radiometry import apparent_magnitude

m = family("starlink_v15")
n = m.body_normals(u_sun, articulate=True)
mag = apparent_magnitude(m, u_sun, u_obs, range_km, normals=n)
```

`tests/test_catalog.py::test_starlink_catalog_magnitude_order_vs_mallama`
locks an **order-of-magnitude** check at two public geometries (not a fit):

- LVLH-hold, observer nadir (satellite at zenith), sun in the body xz
  plane, arrays articulated. Phase 72° and 40°, range 550 km.
- 72° / 550 km is Mallama’s *characteristic magnitude* geometry
  (arXiv:2210.17268): overhead at end of astronomical twilight.
  Published Gen 1: Original 4.7, VisorSat 6.2, Post-VisorSat 5.5.
- v2 Mini unmitigated class: early-orbit 1000-km mean 5.08
  (arXiv:2306.06657); 550↔1000 km is 1.30 mag. On-station mitigation
  (7.87 at 1000 km) is **not** the comparison — this catalog does not
  model brightness-mitigation attitude or visors.

The Lambert–Phong CELLS/MLI presets and FCC areas are a bit bright vs
those papers (about 1–2 mag). That is expected and is **not** retuned to
match. A 90° zenith + ram-sun geometry is edge-on for these flats and
correctly goes dark; Mallama’s 90° phase-function minimum is a sky
average, not that one pose.

## Mapping pitfalls

Ordered: exact COSPAR/name → Starlink generation → prefixes → R/B
patterns → GEO period → class fallbacks.

- `[DTC]` → `starlink_v2mini_dtc`; launch &lt; 2023-02-27 → `starlink_v15`
  (high; v1.0 lumped); later → `starlink_v2mini` (medium). **Never
  Starshield.**
- `GLOBAL-` = BlackSky; `GLOBALSTAR` = Globalstar.
- Exact `TERRA`/`AQUA` → `terra`; `TERRASAR`/`PAZ` → `terrasat_x`.
- `NUSAT-` = Satellogic; `SNUSAT` is not. Cargo Dragon prefixes only —
  not `DRAGONFLY`. `GAOFEN-3` only. `YZ-` for CZ upper. `CYGFM` →
  `cygnss`. `STARLING` → `cubesat_6u`. ESA Swarm A/B/C ≠ SPACEBEE.
- `H-3` → `h3_upper` (5.27 m); `H-2`/`H-II`/`H2A` → `h2_upper` (4.0 m);
  H-1 stays generic.

## Articulation / look

Same `body_normals` articulator as the study path. Documented looks:

| family | aperture | look | vs | status |
|---|---|---|---|---|
| Starlink v1.5 / v2 Mini | bus −z user array | [0,0,−1] | lvlh | public |
| Starlink DTC | DTC panel | [0,0,−1] | lvlh | range |
| OneWeb / Kuiper | bus −z | [0,0,−1] | lvlh | typical_class |
| GEO bus | nadir dish | [0,0,−1] | nadir | typical_class |
| Capella | SAR mesh | [0,0,−1] | nadir | public (FCC ODAR) |
| ICEYE / Umbra | SAR | — | — | **unknown** |
| ISS radiators | PVR/EATCS | — | lvlh | **unknown** |
