# photometry

Estimating the attitude and shape of resident space objects (RSOs) from
visual-magnitude measurements aggregated across a large LEO
constellation's star trackers. See
[docs/constellation-photometric-attitude-shape-estimation.md](docs/constellation-photometric-attitude-shape-estimation.md)
for the founding design document; this README is the results write-up. A
dark-themed briefing deck of the same material is at
[docs/photometry_briefing.pptx](docs/photometry_briefing.pptx), and
[docs/real-data-integration.md](docs/real-data-integration.md) is the
guide for wiring the experiment to real data sources.

## The idea in one paragraph

10,000 satellites each carry three star trackers canted ~5° above the
local horizontal to avoid Earth albedo. Any other object above the shell
crosses those fields of view constantly, so the fleet opportunistically
measures thousands of time-tagged brightnesses of the same target per
day — from hundreds of directions at once. A single ground telescope
sees one light curve; the constellation samples the target's reflected
light *field*, which turns the classically ill-posed light-curve
inversion into a well-conditioned tomography problem. From magnitudes
alone we recover spin state, convex shape, catalog identity, solar-array
configuration, and deviations from catalog.

## Quickstart

```bash
pip install -e .
python scripts/run_pipeline.py results          # baseline sim + inversion
python scripts/run_fleet_study.py               # 21-scenario day-in-the-life
python scripts/run_model_match.py               # library identification
python scripts/run_refine.py                    # residual-EGI demo
python scripts/run_deviation_scan.py            # fleet deviation alerts
python scripts/run_torquefree.py                # Tier-3 multi-axis fit
python scripts/make_charts.py; python scripts/make_fleet_charts.py
python scripts/make_movies.py                   # all validation movies
python scripts/catalog_coverage.py              # facet-catalog vs SATCAT snapshot
python -m pytest tests/
```

The spacecraft **facet catalog** (family templates for the active public
catalog, SATCAT name/COSPAR mapping, thrust vs LVLH, per-facet materials)
is documented in
[docs/facet-catalog.md](docs/facet-catalog.md). Study-orbit identification
still uses the original `shapes.LIBRARY`; new families live in
`photometry.catalog`.

---

## Concepts: what photometry can and cannot know

**The measurement.** Every detection is one number: the apparent visual
magnitude of an unresolved dot, plus the time, the observer's position,
and the line of sight. Range is known from multi-observer orbit
determination, so the 1/d² factor is removed exactly and each detection
becomes a sample of the target's *range-normalized brightness* at one
(sun direction, observer direction) pair.

**The EGI — Extended Gaussian Image.** The shape representation that
photometry actually constrains. Take every patch of the target's
surface, note which direction its outward normal points and how much
reflecting area it has, and pile all of that onto a sphere of
directions: the EGI is the function "reflecting area per unit direction"
on that sphere. Two facts make it central here:

1. *It is all photometry can see.* An unresolved brightness measurement
   is a weighted integral of oriented area — every facet's light lands in
   the same pixel, so **position on the body is unrecoverable**. A solar
   panel mounted left of the bus and the same panel mounted right of the
   bus produce identical light curves. What survives is how much area
   faces each way.
2. *It is enough to get a body back.* Minkowski's theorem: a valid EGI
   corresponds to exactly one convex body (up to translation). So we
   invert brightness → EGI (a convex least-squares problem), then
   reconstruct EGI → convex solid (`inversion/minkowski.py`).

**Contrast with photogrammetry.** A photogram is a resolved image;
photogrammetry triangulates *positions* of surface points across many
images to build a model with spatial structure. Photometry is the
unresolved limit — no pixels on the target, ever — so instead of point
positions we get the direction-space dual: oriented area. The movies
render both the raw EGI (magenta disks, positions deliberately
"exploded" outward because they have none) and the Minkowski solid it
implies (orange), against the truth geometry (blue).

**Attitude enters as rotation of the EGI.** The body-frame EGI is
constant; the attitude history rotates it relative to the sun and
observers. That factorization — static shape × time-varying rotation —
is what every estimator below exploits.

---

## The modeling progression

The project alternated between enriching the *truth* (making the
simulated world harder) and strengthening the *algorithms* until they
caught up. Reading this table top-to-bottom is the development history.

### Truth side — what the simulator generates

| Stage | Model added | Why it matters |
|---|---|---|
| T1 | Walker shell (100×100 sats, 550 km/53°), three trackers per sat at +5° LVLH elevation, 7° half-angle FOV, sun exclusion, Earth-limb clearance, eclipse | The opportunistic sensor. Also produced the first architectural finding: up-canted trackers only see objects **above** the shell |
| T2 | Facet radiometry: Lambert + Phong per facet, material classes (MLI, cells, white paint, antenna) | Brightness is BRDF physics, not albedo-times-area; specular glints carry the sharpest attitude information |
| T3 | Generic box-wing target, principal-axis spin | The controlled baseline every algorithm was first proven on |
| T4 | Seven-satellite library from open-source dimensions: Starlink v1.5 / v2 mini / v2 mini DTC, BlueWalker 3, Hubble, ISS, Katalyst LINK | Real geometry diversity: plates, tubes, giant slabs, bus+wings |
| T5 | Attitude modes: LVLH-hold ops, knife-edge low-drag, sun-point, inertial science pointing, propeller tumble | Day-in-the-life truth; each mode has a distinct photometric signature |
| T6 | Solar-array articulation: per-facet gimbals (fixed / 1-axis / 2-axis) tracking the sun in the forward model | Arrays are not body-fixed — a fact the inversion must and does confront |
| T7 | Sensor saturation as *censoring*: saturated streaks recorded as brighter-than-cap lower bounds | ISS-class targets are ~80% censored; dropping those rows biases everything toward small objects |
| T8 | Torque-free rigid-body tumble (triaxial inertia, nutating ω, energy conserved to 3e-8) | LINK's actual anomaly class: multi-axis spin, no principal-axis shortcut |

### Algorithm side — the inversion stack

| Stage | Algorithm | Problem it solved | Evidence |
|---|---|---|---|
| A1 | Lomb–Scargle periodogram on the aggregated brightness stream (arc-length-sized frequency grid) | Spin period from multi-observer data | Baseline period sub-ms |
| A2 | Spin-pole grid search (Fibonacci sphere × phase × body spin axis) + simplex refinement | Full spin state; flat-spin vs propeller distinction | Baseline pole to **0.1°**; LINK tumble pole 0.07°, correct body axis |
| A3 | EGI by non-negative least squares (Lambert kernel), then joint Lambert+Phong basis with dead-column pruning and scale-aware ridge | Shape from brightness; specular leakage bias +25% → −9% | 99% of recovered area within 15° of true normals |
| A4 | Tier-2 mode classifier: named operational laws + fitted spin/inertial families | "Is it controlled, safe-moded, or tumbling?" | 20/21 scenarios correct (the miss is Hubble's axisymmetric tumble — physically unobservable rotation) |
| A5 | Minkowski reconstruction, fixed-point then **variational** (support-functional L-BFGS-B, analytic gradient dV/dh = face areas) | EGI → actual convex solid; slab proportions fixed | v1.5 slab reconstructs 7.7×3.7×1.0 m vs 8.1×2.7 truth, areas matched exactly |
| A6 | Library identification: model × attitude family × array config, 0.5-mag offset prior so absolute brightness carries size | "Which catalog object is this, in what state?" | **20/21 correct** incl. DTC vs plain v2 mini (cost 1.09 vs 2.64) |
| A7 | Matched-model refinement: symmetry-group multi-start (axis swaps, flips, antipodes) + signed residual EGI | Catalog-deviation detection; ambiguity twins resolved | Seeded missing-antenna case: +0.55 m² at the right normal, rms 1.57→1.02 |
| A8 | Deviation alerting (structure ratio + fit-quality criteria) | Automated "reality vs catalog" flag | **20/21 alert decisions correct** fleet-wide |
| A9 | Censoring-aware costs (one-sided Tobit terms), stratified sampling, coherent period ladder | Saturation-dominated targets | ISS size information restored (truth-attitude cost 0.18 vs impostors 8.8+); global search still open |
| A10 | Tier-3 torque-free fit: window-laddered multi-start over inertia + rate + attitude via solve_ivp quaternion path | Non-principal-axis tumbles | Attitude error **halved** vs uniform spin (97.8°→50.7° window, 97.6°→56.0° 4 h prediction) |

---

## Results

### Baseline: tumbling box-wing, 3.2 h arc

5,703 detections from 588 distinct observers, phase angles 14–142°.
Spin period exact to sub-ms, pole to 0.1°, EGI with 99% of recovered
albedo-area within 15° of true facet normals (the anti-sunward facet is
unilluminated all arc and correctly unobservable).

![summary](results/charts/00_summary_tiles.png)
![light curve](results/charts/01_lightcurve.png)
![coverage](results/charts/02_coverage.png)
![periodogram](results/charts/03_periodogram.png)
![pole search](results/charts/04_pole_search.png)
![egi](results/charts/05_egi.png)

### Day-in-the-life fleet study (21 scenarios × 24 h)

![fleet modes](results/charts/06_fleet_modes.png)
![fleet EGI](results/charts/07_fleet_egi.png)

Sun-tracking arrays are not body-fixed, so ops-mode EGIs recover exactly
the bus + fixed antennas; tumble modes recover the full body. ISS is
saturation-limited yet classifies correctly from a few thousand
unsaturated long-range detections per day.

### Library model identification

**How to read the grid:** the attitude-hypothesis bank is
spacecraft-agnostic — LVLH-hold, knife-edge, sun-point, a *fitted*
uniform spin, and a *fitted* fixed-inertial attitude, with both array
configurations wherever the candidate model has gimbals. For every
(scenario, candidate model) cell we fit that entire bank **for that
model** and plot the best cost, normalized per row. So no cell is a null
test: the fitted families always produce a best-effort attitude, which
is why impostor models show finite-but-poor costs rather than N/A —
and semantically odd pairings (a knife-edge ISS) are simply hypotheses
that lose on cost and never win a row. The row label shows the winning
hypothesis and array configuration for the identified model.

![model match](results/charts/08_model_match.png)

The one miss (tumbling ISS → "Hubble") fits nothing well — best cost 29
where genuine matches score ~1, so a fielded system reports "no
confident match"; the deviation alert flags it independently. Root
cause and partial fix are in the censoring section below.

### Matched-model refinement and residual EGI

![residual EGI](results/charts/09_residual_egi.png)

The signed residual EGI is the "does reality match the catalog"
product: matching the DTC scenario against the plain v2 mini recovers
+0.55 m² of residual area within 25° of the missing antenna's nadir
normal, while correct-model controls stay at the noise floor.

### Catalog-deviation alert scan

![deviation scan](results/charts/11_deviation_scan.png)

20/21 decisions correct (alert = residual-structure ratio > 1.15 or
refined cost > 8). The remaining flag — Hubble's axisymmetric tumble —
is arguably a true positive: the identified inertial attitude genuinely
cannot explain the frozen arrays sweeping with the real rotation.

### Torque-free (multi-axis) tumble

![torque-free](results/charts/10_torquefree.png)

### Validation animations

Each movie shows, in the target's LVLH frame: **left** — truth geometry
(blue) with the model-free inversion products (orange Minkowski solid,
magenta raw-EGI disks); **right** — the identified library model at the
identified attitude and array articulation (green) over faint truth;
**bottom** — projections onto the three LVLH planes. Attitude error vs
nav truth is annotated per frame. All 21 are in
[results/movies/](results/movies/); four representative ones:

*Katalyst LINK, principal-axis tumble — identified at the noise floor,
0.1° attitude error, plate solid riding the array plane:*

![LINK tumble](results/movies/katalyst_link__tumble.gif)

*Starlink v2 mini DTC, nadir ops — arrays articulate toward the sun in
the truth render; identification correct including "arrays tracking":*

![DTC ops](results/movies/starlink_v2mini_dtc__ops.gif)

*ISS tumble — the one misidentification, visibly wrong (tiny Hubble tube
inside the real station) and flagged by its outlier cost:*

![ISS tumble](results/movies/iss__tumble.gif)

*Katalyst LINK, multi-axis (torque-free) tumble — Tier-3 fit in the
right panel, error halved vs uniform spin:*

![LINK multiaxis](results/movies/katalyst_link__multiaxis_tumble.gif)

---

## Standing findings

- **Altitude visibility envelope**: +5°-canted trackers only see objects
  above the shell; ISS (420 km), Hubble (~530 km), LINK/Swift (~500 km)
  are invisible to a 550 km shell at their real altitudes. The study
  flies all targets at a common 620 km orbit.
- **Censoring is the tall pole for big objects**: the bright spin phases
  are exactly the saturated ones, so calibrated-only photometry both
  shrinks apparent size and hides the spin. One-sided censored terms
  restore the size information; the spin lives in the *saturation timing
  pattern* and needs the harmonic-aware ladder (open).
- **Discrete ambiguities are structural**: 90° body-axis swaps for
  plates, 180° flips for tubes, pole antipodes. They must be searched
  explicitly (refinement does) and reported as posterior modes.
- **Articulation is information**: "arrays tracking vs frozen" is
  directly identifiable, and an inertially pointed telescope still
  sun-tracks its arrays — omitting that config produced a false alarm
  until modeled.

## Where to head next

1. **Global search for torque-free states**: CMA-ES or
   differentiable-dynamics gradients; Nelder-Mead reaches 50° but not
   the basin floor (truth verifies at cost 1.04 through the same
   machinery).
2. **Harmonic-aware censored period ladder** to close the last
   misidentification (ISS-class tumblers).
3. **Per-sensor photometric bias estimation** — the untouched piece of
   the design doc's calibration story (30k sensors, cooperative
   constellation targets as truth).
4. **Non-convex shape cues**: shadowing signatures at high phase angles;
   the EGI bounds only the convex hull.
5. **Real tracklets**: reduce to `ObservationSet`
   (`src/photometry/measurements.py`) and the identical stack runs — that
   schema is the sim/real seam by construction. The full wiring guide —
   sources by role, the reduction pipeline, the calibration loop, code
   adapters, and a worked loader skeleton — is
   [docs/real-data-integration.md](docs/real-data-integration.md).
