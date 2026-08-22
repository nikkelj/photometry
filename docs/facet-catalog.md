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
| `capella` | FCC 3.5 m mesh SAR | fixed arrays + dish | unknown |
| `umbra` | 10 m² mesh (eoPortal / patent) | 1-axis stand-in + mesh | unknown |
| `hawkeye360` | SFL NEMO-15 20×20×44 cm | fixed | unknown |
| `blacksky` | SCOUT 1.0×0.5 m (`GLOBAL-n`) | fixed | unknown |
| `globalstar2` | ELiTeBus-1000 typical_class | 1-axis; span range | chemical, vector unknown |
| `orbcomm_og2` | SN-100A, 5 m span | 1-axis | unknown |
| `sentinel1` | ESA 21 m / 12 m C-band SAR | 1-axis + SAR | chemical, vector unknown |
| `sentinel2` | ESA 3.4×1.8×2.35 m (S3 class map) | 1-axis | chemical, vector unknown |
| `sentinel6` | EUMETSAT 5.13×4.17×2.34 m | 1-axis | chemical, vector unknown |
| `landsat8` | USGS 3×2.4 m, 9×0.4 m array | 1-axis | chemical, vector unknown |
| `terra` | Terra / Aqua EOS class | 1-axis | chemical, vector unknown |
| `jpss` | Suomi NPP / NOAA 20/21 | 1-axis | chemical, vector unknown |
| `goes_r` | NESDIS 6.1×5.6×3.9 m, 5-panel wing | 1-axis | chemical, vector unknown |
| `metop` | ESA 6.3×2.5×2.5 m + 8×(1×5 m) | 1-axis | chemical, vector unknown |
| `maxar_legion` | ~3×2×2 m data sheet | 2-axis stand-in | unknown |
| `worldview` | 5.7×2.5 m, 7.1 m span | 2-axis | unknown |
| `bluebird` / `_block2` | AST 64 m² / 223 m² | fixed sheet | unknown |
| `css_tianhe` | 16.6×4.2 m core; **not ISS** | 1-axis range | unknown |
| `galileo` | ESA FOC, distinct from GPS-class | 2-axis | chemical, vector unknown |
| `cubesat_3u` / `6u` / `16u` | CDS envelopes | fixed | unknown |
| `geo_bus` | typical 3-axis GEO | 2-axis + nadir dish | NSSK +y class convention |
| `gnss_meo` | GPS IIF/III, GLONASS, BeiDou MEO | 2-axis | chemical, vector unknown |
| `o3b` | original O3b MEO | 2-axis | chemical, vector unknown |
| `cygnus` / `progress` / `cargo_dragon` | visiting vehicles | n/a | Progress +x when docked |
| `iss` / `hubble` / `bluewalker3` / `katalyst_link` | existing study models | as before | ISS +x public; others unknown/none |
| `falcon9_s2` / `cz_upper` / `ariane_upper` / `breeze_m` / `electron_kick` | stages | n/a | −z along cylinder |
| `centaur` / `fregat` / `soyuz_block_i` / `proton_block_d` / `kosmos_3m` / `tsyklon3` / `zenit2` / `pslv_ps4` / `h2_upper` / `block_dm` / `delta_upper` | lingering stages | n/a | −z |
| `ius` / `agena` / `scout` / `pegasus` / `pam_d` / `titan_transtage` | lingering stages | n/a | −z |
| `rocket_body` | generic last-resort stage | n/a | −z |
| `leo_box_wing` | LEO payload fallback | fixed | unknown |
| `classified_unpublished` | USA / IGS / unpublished recon | uncertain placeholder | unknown; **not Starshield CAD** |

Starshield / classified USA objects map to `classified_unpublished` with
every dimension tagged uncertain. SpaceX internals beyond the public FCC
table are not invented.

Not added (no public OML, or not in this SATCAT extract): Swarm Technologies
SPACEBEE (absent from the 2026-08-21 extract; ESA SWARM A/B/C are a different
vehicle and stay `leo_box_wing`), Pelican, JILIN / GAOFEN / GEESAT / CENTISPACE,
TERRASAR-X (1 object; `TERRA` is an exact-name map so a `TERRA` prefix is
unsafe), Agena-era Titan/Thor cores that are not Transtage/Agena/IUS.

## Mapping (name / COSPAR → family)

`catalog/mapping.py` is ordered: exact COSPAR/name → Starlink generation
heuristic → named prefixes → rocket-body patterns → GEO period → class
fallbacks.

Starlink generation (SATCAT does **not** tag v1.5 vs v2 Mini; McDowell's
per-object list was not vendored — planet4589 star page 404 from this
environment):

- SATCAT `[DTC]` → `starlink_v2mini_dtc` (high)
- launch date before 2023-02-27 (first public v2 Mini, Group 6-1 /
  Spaceflight Now + SpaceNews) → `starlink_v15` (**high**; v1.0 lumped
  with v1.5)
- later launches → `starlink_v2mini` (medium; a minority of late v1.5 /
  Group 5 leftover may be mixed in)
- serial fallbacks only if the launch date is missing
- **Starshield is never inferred** from a STARLINK name

GNSS: Galileo FOC (`GSAT0*` / GALILEO) is `galileo`. GPS IIF/III
(NAVSTAR), GLONASS, and BeiDou MEO stay on `gnss_meo` (published dims do
not justify three more meshes). BeiDou GEO/IGSO period remaps to `geo_bus`.

Inmarsat stays on `geo_bus` (generic 3-axis GEO). GOES 16–19 are `goes_r`;
older GOES / EWS-G stay `geo_bus`.

`GLOBAL-n` is BlackSky; `GLOBALSTAR` is Globalstar (order matters).
`NUSAT-` is Satellogic; `SNUSAT` is not. `DRAGON` visiting-vehicle prefixes
only — `DRAGONFLY` is not Cargo Dragon.

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

2026-08-21 snapshot after this pass (`scripts/catalog_coverage.py`):

| | n | mapped | named | named % | leftover fallback |
|---|---:|---:|---:|---:|---|
| active payloads | 16,870 | 16,870 | 14,170 | **84.0%** | `leo_box_wing` 2,024 |
| rocket bodies | 2,422 | 2,422 | 2,181 | **90.0%** | `rocket_body` 241 |

First-pass PR #6 (same snapshot) was 82.3% named payloads / 25.5% named
rocket bodies (`leo_box_wing` 2,393, generic `rocket_body` 1,804).

Payload confidence this pass: low 2,626 / medium 8,918 / high 5,326.

Largest leftover payload piles (no public OML, left as `leo_box_wing`):
COSMOS, JILIN, RASSVET, GONETS-M, CENTISPACE, IRIDE, TIANMU, ION SCV,
Chinese optical/weather stacks, ESA SWARM A/B/C (not Swarm SPACEBEE).

Largest leftover rocket-body piles (no named stage family): Titan/Thor/Atlas
cores that are not Transtage/Agena/IUS/Centaur, plus AKM / IABS / Saturn /
Diamant / Minotaur.

## What is unknown (on purpose)

- Starlink / Kuiper / OneWeb / Iridium NEXT **Hall / Δv pointing** vs LVLH.
  Searched SpaceX Gen2 PDF, FCC dimension table, SpaceNews/Spaceflight Now
  (Hall exists; magnitudes only). Everyday Astronaut “ram-facing” is not a
  primary citation — vector left empty. OneWeb: Busek BHT-350 for EOR/SK/CA/
  deorbit, no axis. Kuiper: FCC/eoPortal Kr Hall, no axis. Iridium NEXT:
  FCC SAT-MOD-20131227-00148 (nadir service, eight 1 N hydrazine thrusters),
  no single body-frame Δv vector.
- Amazon Leo outer-mold-line (range only)
- Guowang OML (Gunter: not published)
- Starshield and NRO buses
- Spacecraft-specific IR α/ε
- Array 1- vs 2-axis when the public record is silent (called out as a
  stand-in on Kuiper, Hulianwang, SkySat, Umbra)

Estimator, sensing FOV, and ADCS tracker geometry are not part of this
catalog.
