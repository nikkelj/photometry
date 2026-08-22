# Spacecraft facet catalog

Open-source **family templates** for the active public catalog, mapped onto
Celestrak SATCAT names / COSPAR / orbit class. This is not a parallel CAD
format and not 10,000 unique meshes: every family builds a `FacetModel` the
existing forward model already consumes (`rho_d`, `k_s`, `n_ph`, gimbals).

The 620 km study-orbit library (`photometry.shapes.LIBRARY`) is **unchanged**
so fleet identification numbers stay reproducible. Catalog SATCAT mapping
uses higher-fidelity stand-ins for Starlink v1.5 / v2 Mini / ISS (FCC /
NASA areas, explicit hinges, documented looks); the study factories
`shapes.starlink_v15()` / `shapes.starlink_v2mini()` / `shapes.iss()` keep
the study meshes. Articulation is the same `body_normals` path (extended
hinge, not a second articulator). High-count catalog stand-ins carry
rest-pose `polygons` (bus / array / deployable quads, including
mirror-backs) for movie/truth viz only — photometry still uses
areas + normals. Study LIBRARY drawable lists are not filled.

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
python scripts/catalog_leftovers.py         # leftover leo_box_wing / rocket_body prefixes
python -m pytest tests/test_catalog.py tests/test_basics.py
```

## Schema (what a family must carry)

1. **Bus** — box or cylinder facets.
2. **Solar arrays** with a real hinge, not a free sun-chasing normal:
   `GIMBAL_FIXED` (rest pose), `GIMBAL_1AXIS` (Rodrigues about `gimbal_axis`
   from rest; out-of-plane cosine loss remains; travel clamped when public),
   `GIMBAL_2AXIS` (shoulder then wrist). BlueWalker-class sheets stay fixed.
   Travel defaults to ±π when unpublished so study LIBRARY sun-track cases
   stay numerically the same.
3. **Deployables** when they are photometrically large and public (DTC panel,
   ICEYE 3.2×0.4 m SAR, GEO nadir dish, ISS radiators), with a documented
   **look** vector vs the nominal flight attitude (`lvlh` / `nadir` /
   `yaw_steering`). Unpublished looks stay `unknown` — not invented.
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
| `starlink_v15` | FCC v1.5 2.8×1.3 m bus, 2.8×8.1 m 1-axis array (catalog; study LIBRARY keeps 1.4 / 2.7) | 1-axis | EP public, vector unknown |
| `starlink_v2mini` / `_dtc` | FCC v2 Mini 4.1×2.7 m, two 4.1×12.8 m 2-axis; DTC 2.0×2.3 m range | 2-axis | EP public, vector unknown |
| `starlink_v2` | FCC Starship-class v2 | 2-axis | not mapped from SATCAT names |
| `oneweb` | Airbus Arrow 1×1×1.3 m, **5.0 m** tip-to-tip (was 5.6) | 1-axis | xenon Hall, vector unknown |
| `kuiper` | Amazon Leo; ~2 m / 10 m span range (protoflight class) | 1-axis stand-in | Kr Hall, vector unknown |
| `qianfan` | G60 / Thousand Sails flat-pack | 1-axis (Mallama) | Kr Hall reported, vector unknown |
| `hulianwang` | Guowang; Gunter: OML unpublished | 1-axis stand-in | EP believed, vector unknown |
| `iridium_next` | 9.4 m span, sun-tracking | 1-axis | hydrazine, vector unknown |
| `planet_superdove` | 3U Flock | fixed | none public |
| `planet_skysat` | 60×60×95 cm | 1-axis stand-in | 180 m/s public, vector unknown |
| `iceye` | bus range + 3.2×0.4 m SAR | 1-axis + fixed SAR | unknown |
| `terrasat_x` | DLR 5×2.4 m hex + 5×0.8 m SAR (PAZ too) | body-mounted 5.25 m² | chemical, vector unknown |
| `cosmo_skymed` | first-gen CSK 5.7×1.4 m SAR (not CSG) | 1-axis, 18.3 m² | chemical, vector unknown |
| `alos2` | JAXA 9.9×16.5×3.7 m, PALSAR-2 2.9×9.9 m | 1-axis | chemical, vector unknown |
| `radarsat2` | CSA 3.7×1.36 m, 15×1.5 m SAR | 1-axis 3.73×1.8 m | chemical, vector unknown |
| `gaofen3` | civil C-band 15×1.232 m only | 1-axis stand-in | unknown |
| `saocom` | INVAP 4.7×1.2 m, 10×3.5 m L-band | 1-axis | chemical, vector unknown |
| `esa_swarm` | ESA 9.1×1.5×0.85 m + boom | body-mounted | cold-gas, vector unused |
| `ion_scv` | D-Orbit 60 cm / 64U | n/a | chemical, vector unknown |
| `ghgsat` | SFL NEMO 20×30×40 cm | fixed | none |
| `cygnss` | NASA CYGNSS 51×64×28 cm, 1.67 m span (`CYGFM`) | fixed | none |
| `grus` | Axelspace ~0.6×0.6×0.8 m | fixed | unknown |
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
| `geo_bus` | typical 3-axis GEO; N/S white radiators; ~22.8 m span | 2-axis + nadir dish | NSSK +y class convention |
| `gnss_meo` | GPS IIF/III, GLONASS, BeiDou MEO | 2-axis | chemical, vector unknown |
| `o3b` | original O3b MEO | 2-axis | chemical, vector unknown |
| `cygnus` / `progress` / `cargo_dragon` | visiting vehicles | n/a | Progress +x when docked |
| `iss` | NASA 2,500 m² / 109 m truss (catalog; study LIBRARY keeps 35×24) | 2-axis arrays + 1-axis radiators | ISS +x public |
| `hubble` / `bluewalker3` / `katalyst_link` | existing study models | as before | unknown/none |
| `falcon9_s2` / `cz_upper` / `ariane_upper` / `breeze_m` / `electron_kick` | stages | n/a | −z along cylinder |
| `centaur` / `fregat` / `soyuz_block_i` / `proton_block_d` / `kosmos_3m` / `tsyklon3` / `zenit2` / `pslv_ps4` / `h2_upper` / `block_dm` / `delta_upper` | lingering stages | n/a | −z |
| `ius` / `agena` / `scout` / `pegasus` / `pam_d` / `titan_transtage` | lingering stages | n/a | −z |
| `atlas_core` / `titan_core` / `saturn_sivb` / `avum` / `firefly_alpha` / `dnepr` / `thor_ablestar` / `burner2` / `h3_upper` / `iabs` | lingering stages | n/a | −z |
| `rocket_body` | generic last-resort stage | n/a | −z |
| `leo_box_wing` | LEO payload fallback | fixed | unknown |
| `classified_unpublished` | USA / IGS / unpublished recon | uncertain placeholder | unknown; **not Starshield CAD** |

Starshield / classified USA objects map to `classified_unpublished` with
every dimension tagged uncertain. SpaceX internals beyond the public FCC
table are not invented.

Not added (no primary public OML — left in `leo_box_wing` on purpose):
Jilin-1 (mixed 40–420 kg buses; eoPortal has mass, not one OML),
optical Gaofen-1/2/5/6/7/9/11/12 (only GF-3 SAR is cited), SuperView Neo
(mass only), GEESat / CentiSpace / Yunhai / Haiyang / Tianhui / Tianmu
(KeepTrack-only or unpublished), COSMO-SkyMed second gen (`CSG-*`),
Pelican, Rassvet, Gonets-M, IRIDE HEO/FM (not Eaglet II), COSMOS, QPS-SAR,
DiskSat, FORMOSAT mix, STRIX. Yaogan stays `classified_unpublished`.
Swarm Technologies SPACEBEE is absent from this extract; ESA `SWARM A/B/C`
is the `esa_swarm` family. `TERRA` remains an exact-name EOS map so
TerraSAR-X is matched on `TERRASAR`, not `TERRA`.

Western leftovers that *did* have a public OML: NASA CYGNSS (`CYGFM`,
new family), NASA Starling → existing `cubesat_6u`. Planet Flock /
SkySat, Spire Lemur, Satellogic ÑuSat, and GHGSat were already mapped;
the leftover table has none of those prefixes.

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
`NUSAT-` is Satellogic; `SNUSAT` is the SNU 3U (`cubesat_3u`). `DRAGON`
visiting-vehicle prefixes only — `DRAGONFLY` is not Cargo Dragon.
`FLOCK` is still SuperDove. `YAM-*` is York/SDA unpublished.

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
| active payloads | 16,870 | 16,870 | 14,334 | **85.0%** | `leo_box_wing` 1,846 |
| rocket bodies | 2,422 | 2,422 | 2,290 | **94.5%** | `rocket_body` 132 |

Earlier cuts on the same snapshot: first-pass 82.3% / 25.5%; then 84.0% /
90.0%; previous hone 84.9% / 94.1% (`leo_box_wing` 1,857, generic
`rocket_body` 144). **92** families.

Payload confidence this pass: low 2,482 / medium 8,967 / high 5,421.

## Leftover inventory

`scripts/catalog_leftovers.py` (also `leftover_inventory()`) lists every
SATCAT prefix still on the fallbacks. Prefix = first whitespace token with
a trailing serial stripped. **No family is invented from a prefix.**

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

Generic `rocket_body` leftovers (**132**; exact names via the script):

| n | names | why still generic |
|---:|---|---|
| ~40 | mixed `* AKM` / `* PKM` (Fengyun-2, GOES 1–7, Meteosat, Himawari, Leasat, Lageos, …) | sizes differ — not lumped |
| 8 | MINOTAUR / MINOTAUR 1 / MINOTAUR 4 | Minotaur I ≠ IV diameter |
| 7 | OV1-* | 1960s OV1 stages, mixed |
| 6 / 6 | Japanese N-1/N-2, M-3S/M-4S | not H-IIA 4 m |
| 5 | DIAMANT | public-ish but mixed B / B-P4 |
| 4 | TAURUS R/B | Taurus / Minotaur-C mix |
| 3 | KSLV-II, LIJIAN-1 | no primary stage drawing used here |
| 2 | H-1, EPSILON, M-V, VANGUARD, VOLGA, … | H-3 is now `h3_upper`; H-1 is not |
| 1 | FALCON 1, NEW GLENN, SLS, LVM3, … | one-offs; dims exist but not worth a family |

Mapped this pass with public stage dims: `IABS R/B` (9; 2.9×0.68 m disc),
`H-3 R/B` (2; JAXA 5.27×12 m), `SL-11 R/B` (Tsyklon-2 → existing
`tsyklon3`).

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
  stand-in on Kuiper, Hulianwang, SkySat, Umbra). OneWeb 1-axis and
  Starlink v1.5 1-axis / v2 Mini 2-axis match the public record. GEO is
  2-axis class convention.

Estimator, sensing FOV, and ADCS tracker geometry are not part of this
catalog.

## Articulation / look (this fidelity pass)

`body_normals` rotates each rest normal about the stored hinge (1-axis)
or shoulder then wrist (2-axis). It is the same articulator the study
path already called — not a second implementation. Travel is ±π unless a
public stop exists (none of the high-count families publish a degree
limit; ISS alpha/beta are treated as continuous `typical_class`).

Documented aperture looks (body frame vs flight attitude):

| family | aperture | look | vs | status |
|---|---|---|---|---|
| Starlink v1.5 / v2 Mini | bus −z user array | [0,0,−1] | lvlh | public |
| Starlink DTC | DTC panel | [0,0,−1] | lvlh | range |
| OneWeb / Kuiper | bus −z | [0,0,−1] | lvlh | typical_class |
| GEO bus | nadir dish | [0,0,−1] | nadir | typical_class |
| Capella | SAR mesh | [0,0,−1] | nadir | public (FCC ODAR) |
| ICEYE / Umbra | SAR | — | — | **unknown** (side-look / unpublished) |
| ISS radiators | PVR/EATCS | — | lvlh | **unknown** (thermal schedule) |

Hall / ion thrust vectors are still empty. No primary ram/nadir/aft
citation was found for Starlink, OneWeb, or Kuiper.
