# Stack bloat audit — what to cut, what to keep, and why

Measured at the point the learned proposer (`photometry.learn`) was
added. The stack grew by accretion — every study left a script, every
fix left a module — and it is time to say which parts are load-bearing.

## Where the lines are

| Area | Lines | Files | Verdict |
|---|---|---|---|
| `src/photometry` core (frames, radiometry, attitude, shapes, sensing, simulate, scenarios, measurements, library200, registry, studies, charts) | ~2,700 | 12 | keep; `library200` is the fat one (see cut 3) |
| `src/photometry/inversion` | ~2,100 | 13 | keep the spine; two modules are dead-weight candidates (cuts 4–5) |
| `src/photometry/catalog` | ~2,600 | 8 | keep; it is data, not logic (three `families*.py` are template tables) |
| `src/photometry/learn` | ~480 | 5 | new; optional dependency, torch-free core preserved |
| `scripts/` | ~4,000 → ~3,750 after this pass | 25 | **the bloat lives here** (cuts 1–2) |
| `tests/` | ~925 | 2 | keep |
| docs (README + 4 docs) | ~1,200 | 5 | README needs splitting (cut 7) |
| tracked `results/` | **112 MB** | — | movies 67 MB + fleet npz 38 MB should leave git (cut 6) |

## Cuts made in this pass (safe, test-covered)

1. **One palette, one `style()`** — `photometry/charts.py`. Eight chart
   scripts each carried an identical 22–40 line palette/style block
   (with one drifted color). Replaced by a single import: −206 lines,
   and the theme now has one source of truth.
2. **One study harness** — `photometry/studies.py`. The fleet-simulation
   setup, the truth-attitude sampler, the attitude-error metric, and the
   window helper were copy-pasted across `run_library_scale`,
   `run_unified_id`, `run_slew_study`, `run_torquefree`, and the new
   `run_spin_net`. Now imported. A study script is only its study logic.

## Cuts recommended (need a decision, not just a diff)

3. **Shrink `library200` from 200 to ~60 generated entries.** Its job is
   a *stress test* for the funnel (twin classes, hypothesis-space gaps),
   not coverage — `photometry.catalog` now provides real coverage with
   SATCAT mapping. Twin behavior appears at any density; 200 entries
   just make every registry-wide operation (feature cache, shortlist)
   3× slower. Keep the generator, cut `FAMILY_PLAN` counts. (~0 lines,
   ~3× runtime.)
4. **Delete the Minkowski fixed-point fallback**
   (`minkowski._reconstruct_hull_fixed_point`, ~90 lines). The
   variational solver replaced it; the fallback is reachable only on
   variational failure, which no current test or study exercises. If it
   ever fires, we would rather see the failure than a silently worse
   hull.
5. **Fold `classify.py` into `model_match.py`.** Its only library use is
   the `INERTIAL_PERIOD_S` constant; `classify_modes` survives only for
   `run_fleet_study`. One mode-scoring path, not two.
6. **Take `results/movies` (67 MB) and `results/fleet/*.npz` (38 MB) out
   of git** — git-lfs or a release asset. Keep charts (5 MB) and JSON
   summaries; the README embeds four GIFs — keep exactly those four or
   link them. The npz files are regenerable in ~1 h from
   `run_fleet_study.py`.
7. **Split the README** (535 lines): overview + quickstart + headline
   findings stay (~150 lines); the progression tables and per-study
   write-ups move to `docs/results.md`.
8. **One Huber cost.** `pole_search._cost` and `cost.huber_mag_cost` are
   two implementations of the same Tobit-censored Huber objective (the
   former with a fast spin-only attitude path). Two costs can drift;
   unify on `huber_mag_cost` with a fast-path attitude and delete the
   other (~60 lines, and one fewer place for a censoring bug to hide).

## What the learned proposer lets us cut *if it works*

The set-transformer proposal + physics polish replaces the brute-force
search, not the physics. If the go/no-go metric holds (net-seeded polish
converges at least as often as the exhaustive grid, in seconds instead
of a minute), then:

- `grid_search_pole`'s exhaustive Fibonacci × phase × axis sweep becomes
  an optional fallback (`--exhaustive`), not the default path.
- `ladder_spin_search` (the coherent period ladder for censored
  tumblers, which locked a wrong harmonic on ISS anyway) is deletable —
  the net's period head plus a harmonic-tolerant polish covers it.
- The torque-free multi-start (`fit_torque_free`'s start bank: seed
  periods × 3 axes × 2 attitudes × 2 inertias, two phases) collapses to
  a single seeded fit if the seed lands in-basin.
- The shortlist funnel's channel B (tumble fingerprints) could be
  replaced by the net's mode/spin posterior as a ranking feature.

That is ~400 lines of search machinery and the two documented open
problems, contingent on the measured result in `results/learn/`.

## What must not be cut

- `radiometry.facet_brightness` and `cost.huber_mag_cost`: the forward
  model and its likelihood are the certificate everything else is
  checked against — including the net.
- `periodogram`: a sufficient statistic; cheap, exact, interpretable.
- `egi` + `minkowski` (variational): convex solve with a uniqueness
  theorem behind it.
- `registry` + `catalog.resolve`: the operational identification
  contract (twin-margin prior semantics).
- `deviation`: the self-assessment layer that caught the stack's own
  misidentifications at 20/21.
