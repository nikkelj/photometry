# Wiring the experiment to real data

Everything downstream of measurement reduction is already source-agnostic:
the entire inversion stack (periodogram, spin search, EGI, Minkowski,
library matching, refinement, deviation alerting) consumes only
`photometry.measurements.ObservationSet`. Nothing in `inversion/` knows
whether a row came from the simulator or from a fleet. Wiring up real data
is therefore three jobs: **(1)** reduce raw sensor data to `ObservationSet`
rows, **(2)** supply real orbits where the attitude hypotheses need them,
and **(3)** close the calibration loop. This document is the map.

## 1. The seam: `ObservationSet`, field by field

One row per calibrated detection of one associated target.

| Field | What feeds it from real data | Notes |
|---|---|---|
| `t_s` | Detection mid-exposure time, seconds from your chosen epoch | GPS-disciplined tracker clocks are µs-class; apply light-time correction (range/c, 3–10 ms in LEO) if you care about spin phase below ~0.05° |
| `obs_id`, `tracker_id` | Observing satellite + tracker head identifiers | Keys for per-sensor bias estimation — keep them stable |
| `obs_pos_km` | Observer ECI position from the operator's own ephemeris/GPS | Meters-level; this is the easy one |
| `los_eci` | Unit line of sight observer → target | The star tracker's core competence: its attitude solution converts focal-plane centroids to inertial directions at arcsecond level "for free" |
| `sun_eci` | Sun unit vector at `t_s` from a solar ephemeris (astropy/SPICE DE440) | Per-row, so the fixed-sun sim simplification disappears automatically |
| `range_km` | From orbit determination of the target (multi-observer triangulation, or association against a catalog ephemeris) | Removes 1/d² exactly — the quantity that makes absolute size observable |
| `mag` | Calibrated apparent magnitude from **streak** photometry | Integrate flux along the trail, not the peak: LEO–LEO relative motion smears the target across pixels within one exposure |
| `mag_sigma` | Per-detection photometric uncertainty from the calibration solution | Drives every weight in the stack; do not hardcode |
| `sensor_bias` | Residual per-sensor zero point from the calibration loop (§4) | Zero until the loop runs; the schema carries it so biases never silently vanish |
| `censored` | 1 for saturated streaks, with `mag` set to the saturation limit | Keep these rows. Dropping them biases identification toward small objects; the saturation *timing pattern* carries spin for bright targets |

`ObservationSet.from_csv` / `from_npz` already exist; a loader is just code
that fills this table and calls the constructor. Missing columns
(`censored`, `sensor_bias`) default sanely for partial sources.

## 2. Data sources, by role

**Tracker transient streams (the premise of the experiment).** Star
trackers already detect non-stellar objects every frame — they are the
"unmatched centroids" the star identification step rejects. Wiring this up
for real means an operator (or tracker vendor: Sodern, Jena-Optronik,
Terma, Rocket Lab/Sinclair ST-series) exposing that reject stream as a
downlinked tracklet product: time, focal-plane centroid + attitude (⇒
`los_eci`), instrumental magnitude, saturation flag. The onboard cost is
small because the detection already happens; what's new is keeping instead
of discarding it.

**Ground photometry as a bridge while fleet data is negotiated.** The
identical stack runs on single-site light curves — with honestly degraded
geometry (one observer direction instead of hundreds; expect the classical
ambiguities to return). Real sources: the MMT-9 (Mini-MegaTORTORA)
public satellite light-curve database, commercial SDA photometric networks
(e.g. ExoAnalytic, Slingshot), and the amateur SatObs/PPAS flash-period
records for coarse validation of spin periods. This is the cheapest way to
confront the BRDF models and calibration pipeline with real physics before
any fleet integration.

**Orbits and ephemerides.** Space-Track TLEs propagated with SGP4 are
adequate for association and for the `range_km` of slow work; operator
ephemerides (SpaceX publishes Starlink ephemerides publicly) are better
where available; ILRS laser-ranged orbits give cm-truth for calibration
targets. Site the Sun from DE440 via astropy or SPICE, and realize ECI as
GCRF with IERS earth-orientation data.

**Photometric reference.** Gaia DR3 G-band is the practical all-sky
reference: every tracker frame contains dozens of Gaia stars, so each
frame carries its own zero point. A one-time color transformation from the
tracker's (typically very broad) passband to G, then to V if you want the
catalogs' convention, becomes part of the calibration state.

**Truth for validation.** Three tiers: (i) the constellation's own
satellites — attitudes are telemetered, shapes are CAD-known, and there are
thousands of them in view constantly: this is the cooperative truth set the
whole calibration story leans on; (ii) other operators' cooperative data
(published attitude modes, maneuver logs); (iii) defunct objects with
independently measured spin states from SLR (e.g. Envisat, Topex) as blind
tests for the tumble pipeline.

## 3. The reduction pipeline (raw → ObservationSet)

```
tracker frame ──► streak detection (the star-ID rejects)
              ──► astrometry: centroid + tracker attitude → los_eci
              ──► photometry: streak-integrated flux, per-frame Gaia zero
                  point, color transform → mag, mag_sigma; saturated → censored=1
              ──► association: correlate LOS across observers / against catalog
                  ephemerides → target identity + orbit → range_km
              ──► light-time + annual aberration corrections
              ──► ObservationSet rows (one target per set)
```

Two reduction realities the sim glosses over, in priority order:

1. **Streak photometry is the hard part.** Trail length varies with
   relative velocity and exposure; flux calibration must integrate the
   trail against a trailed PSF, and the faint end is trail-length-dependent.
   Budget most of the reduction effort here.
2. **Association is a fleet-scale problem.** With 30k sensors the
   unmatched-centroid stream is huge; associating it against a catalog
   (and discovering the uncatalogued residue) is its own subsystem. For
   first light, cheat: task the pipeline on *known* targets by predicting
   their transits through each tracker's FOV from catalog ephemerides and
   harvesting only those windows.

## 4. The calibration loop

Run the stack on cooperative constellation targets (truth attitude + CAD
shape known), hold the truth fixed, and solve for the nuisance state
instead: per-sensor zero-point biases (`sensor_bias`), color terms, and
BRDF material parameters. This inverts the normal use of the pipeline —
same forward model, different unknowns. Cadence: continuous, since every
tracker sees sibling satellites daily. This is roadmap item 3 in the
README and the precondition for trusting absolute-brightness (size)
discrimination on real data.

## 5. Code adapters needed on first contact with real data

Small, contained changes — listed so nobody discovers them mid-integration:

- **`EphemerisOrbit` adapter.** The LVLH-hold attitude hypotheses call
  `orbit.single_states(t)`; today that's a circular two-body
  `WalkerConstellation`. Wrap the real target ephemeris (SGP4 or operator
  states, interpolated) in an object exposing the same
  `single_states(t) -> (r_eci, v_eci)` and every hypothesis works
  unchanged.
- **Per-row sun vectors** are already in the schema; delete nothing.
- **Earthshine term.** Real targets are illuminated by Earth albedo too
  (up to a few tenths of a magnitude at these limb-scanning geometries).
  Add a coarse Earth-disk irradiance term to `radiometry.facet_brightness`
  or down-weight geometries where it is strong; the design doc sketches
  the model.
- **Passband.** The Phong/Lambert coefficients in `shapes.py` are
  V-band-ish placeholders; refit them per material class in the tracker's
  actual passband during the calibration loop.
- **Robustness knobs.** Real outliers (cosmic rays, close-approach
  blending, glint saturation partial rows) are wilder than simulated
  noise; the Huber costs and sigma-clipping already exist — expect to
  retune `huber_a`, `clip_sigma`, and `mag_sigma` floors on real
  residual histograms rather than trusting sim defaults.

## 6. A worked loader skeleton

```python
"""Tracklet CSV -> ObservationSet.

Expected columns (one row per calibrated detection):
  utc_iso, sat_id, tracker, obs_x_km, obs_y_km, obs_z_km,
  los_x, los_y, los_z, range_km, mag, mag_sigma, saturated
"""
import numpy as np
from astropy.time import Time
from astropy.coordinates import get_sun, GCRS
from photometry.measurements import ObservationSet

def load_tracklets(path, epoch_iso):
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None,
                         encoding="utf-8")
    t = Time([r["utc_iso"] for r in rows])
    epoch = Time(epoch_iso)
    sun = get_sun(t).transform_to(GCRS(obstime=t)).cartesian
    sun_u = (sun / sun.norm()).get_xyz().value.T          # (K,3) unit
    los = np.stack([rows["los_x"], rows["los_y"], rows["los_z"]], -1)
    los /= np.linalg.norm(los, axis=-1, keepdims=True)
    return ObservationSet(
        t_s=(t - epoch).sec,
        obs_id=rows["sat_id"].astype(int),
        tracker_id=rows["tracker"].astype(int),
        obs_pos_km=np.stack([rows["obs_x_km"], rows["obs_y_km"],
                             rows["obs_z_km"]], -1),
        los_eci=los,
        sun_eci=sun_u,
        range_km=rows["range_km"].astype(float),
        mag=rows["mag"].astype(float),
        mag_sigma=rows["mag_sigma"].astype(float),
        sensor_bias=np.zeros(len(t)),
        censored=rows["saturated"].astype(int),
    )
```

From there, the entire experiment is the same three calls used throughout
this repo: `brightness_periodogram(obs)`, `match_library(obs, orbit, sun,
...)`, `refine_match(...)` — and the same charts and movies render from
the results.
