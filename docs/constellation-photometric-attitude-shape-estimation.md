# Estimating RSO Attitude and Shape from Constellation-Aggregated Star Tracker Photometry

## 1. Problem statement

A constellation of ~10,000 satellites each carries three star trackers, canted ~5° above
the local horizontal in the LVLH frame to keep the Earth limb and albedo out of the field
of view. As a side effect, every tracker stares through a shell of near-tangent geometry
in which other resident space objects (RSOs) routinely transit the FOV. Each opportunistic
detection yields:

- a time-tagged unit vector (angles) to the target in the inertial frame, and
- an instrumental visual magnitude.

The goal: fuse the *aggregate* of visual-magnitude measurements — thousands of
simultaneous and sequential samples from widely separated viewing geometries — to estimate
the target's **attitude** (orientation history) and **shape** (geometry + reflectance).

This is the classical light-curve inversion problem, but with a structural advantage no
single ground site has: instead of one observer sampling the target's reflectance along a
single slowly-varying line of sight, the constellation samples the target's scattered
light field at **many (sun, observer) direction pairs at the same instant**. That converts
an ill-posed temporal inversion into a much better-conditioned tomographic one.

## 2. Why the geometry is so favorable

For a LEO target inside or near the constellation shell:

- At any epoch, tens to hundreds of trackers can have the target in or near their FOV,
  spanning solar phase angles from a few degrees to >120°.
- Simultaneous multi-observer samples separate **attitude** from **shape** effects: a
  single observer sees `B(t) = f(shape, attitude(t), geometry(t))` and cannot easily
  distinguish a shape feature rotating through the glint condition from a geometry change.
  Two hundred observers at one instant sample `f` over a 2-sphere of observer directions
  *at a fixed attitude* — a direct cut through the object's bidirectional reflectance.
- Multi-observer triangulation of the angle measurements gives instantaneous position and
  therefore **range to ~meters**. Range-normalizing the magnitudes removes the 1/d² term
  exactly, leaving only the orientation/shape signal.
- Periodicity analysis is clean: with observers all around the object, synodic/sidereal
  period ambiguities that plague single-site light curves disappear almost immediately.
- The constellation's *own* satellites are truth-attitude calibration targets: their
  telemetered attitudes and known CAD/BRDF models let you calibrate the entire pipeline
  end-to-end on thousands of cooperative targets before pointing it at unknowns.

## 3. Measurement model (forward model)

### 3.1 Radiometry

For observer *k* at time *t*, with target–observer range `d_k`, sun unit vector `u_s` and
observer unit vector `u_k` **expressed in the target body frame** (this is where attitude
enters — the rotation `R(t)` maps inertial directions into the body frame):

```
F_k(t) = (F_sun / d_k²) · Σ_i  A_i · f_r,i(u_s, u_k; n_i) · max(0, n_i·u_s) · max(0, n_i·u_k) · V_i(u_s, u_k)
```

- `A_i`, `n_i`: area and outward normal of facet *i* of the shape model
- `f_r,i`: facet BRDF — diffuse (Lambert, albedo `ρ_d,i`) + specular lobe
  (Cook–Torrance or Ashikhmin–Shirley with parameters per material class: MLI, solar-cell
  cover glass, white paint, bare aluminum)
- `V_i`: mutual visibility/shadowing term (1 for convex bodies; ray-traced for nonconvex)
- Apparent magnitude: `m_k = m_sun − 2.5 log10(F_k / F_sun)`, `m_sun ≈ −26.74`

Secondary illumination by **Earthshine** (the target is lit by Earth albedo even though
the trackers avoid looking at Earth) contributes up to a few tenths of a magnitude and
must be modeled (a coarse Earth-disk irradiance model suffices) or down-weighted at
geometries where it is strong.

### 3.2 Sensor model

Star trackers are not photometers. Convert instrumental to calibrated magnitude by
solving, per sensor, against the background star field (Gaia G / Hipparcos Hp) in every
frame: zero point, color term, PSF-dependent aperture correction, stray-light gradient,
and smear correction for streaked detections (relative rates in LEO–LEO encounters are
km/s; the target is a streak whose integrated flux, not peak, carries the photometry).
Model per-sensor residual bias `b_k` as a nuisance parameter in the inversion — with 30k
sensors, biases average down but must not be allowed to alias into shape.

Realistic per-detection precision: σ ≈ 0.05–0.15 mag after calibration; limiting
magnitude ~6.5–8 depending on tracker class and streak rate; saturation handling for
close/bright passes.

### 3.3 Measurement equation

After range normalization each detection reduces to a tuple

```
z_j = ( t_j, u_s(t_j), u_k(t_j), m_j, σ_j, sensor_id )
→   m_j = M( shape θ_S, attitude R(t_j), BRDF θ_B, u_s, u_k ) + b_k + ν_j
```

The inversion estimates `θ_S` (facet areas/normals), `θ_B` (material parameters),
`R(t)` (attitude trajectory), and dynamics parameters, given thousands of `z_j`.

## 4. Estimation architecture

Layered inference, cheapest hypotheses first:

### Tier 0 — Association, orbit, and normalization
Multi-observer angle tracklets → orbit determination (trivially strong with this many
observers) → per-detection range, sun/observer geometry, eclipse flags. Reject detections
in penumbra; eclipse-entry timing is itself a free orbit/shape-independent cross-check.

### Tier 1 — Dynamics-mode classification
Before estimating attitude, classify the attitude *regime* from the aggregated brightness
field:

- **Frequency analysis** (multi-observer Lomb–Scargle / phase dispersion minimization) →
  inertially spinning, flat-spin, slow tumble, or stabilized.
- A 3-axis stabilized satellite shows brightness that is a smooth function of geometry
  only (reproducible across observers at equal geometry); a tumbler shows periodicity;
  an uncontrolled derelict often shows the characteristic period + precession sidebands.

This selects the dynamics model used as a constraint below: torque-free rigid body
(Euler equations, unknown inertia ratios), gravity-gradient, or controlled
(nadir/sun-pointing hypothesis with small unknown offsets).

### Tier 2 — Known-shape hypothesis (model matching)
Most bright RSOs are known bus types (rocket bodies, common buses, other constellations'
satellites). Maintain a shape/BRDF library. For each candidate model, estimate the
attitude trajectory (Section 5) and score with the measurement likelihood. This is a
multiple-model (MMAE) layer: it usually terminates the problem, and it produces the
attitude product for the dominant use case (e.g., "is that satellite's array pointed at
the sun or is it tumbling?").

### Tier 3 — Unknown shape: joint shape + attitude inversion
Only for objects with no good library match. Described in Section 6.

## 5. Attitude estimation given a shape

State: `x = [ q (quaternion), ω, inertia ratios / control-mode params, facet albedo scale factors ]`.

- **Batch MAP / nonlinear least squares** over an arc, with the attitude dynamics as the
  process constraint (attitude spline or direct multiple shooting on Euler's equations).
  Brightness → attitude is highly nonlinear and multimodal, so:
- **Global search first**: coarse grid / Fourier-informed initialization over pole
  direction × phase × spin period, then local refinement. With multi-observer data the
  likelihood surface is far less multimodal than single-site, but symmetric shapes retain
  discrete ambiguities (e.g., 180° flips for a box) — carry them as explicit hypotheses.
- **Sequential tracking after lock**: multiplicative UKF or Rao–Blackwellized particle
  filter (particles over attitude/rate; conditionally linear albedo-area parameters
  marginalized in closed form) to maintain custody in real time.

### Glints are the killer measurement
Specular returns from solar arrays and flat radiators are enormous (many magnitudes) and
geometrically sharp: a glint seen by observer *k* means some facet normal satisfies
`n ≈ normalize(u_s + u_k)` — the bisector — at that instant. With thousands of observers,
the constellation *scans* the bisector direction continuously, so a rotating panel sweeps
glints across a predictable sequence of observers. Treat glint events as a separate,
high-weight measurement type: each one is effectively a direct 2-DOF attitude measurement
of one facet. A handful of glints can lock the attitude solution that magnitudes alone
would leave ambiguous.

## 6. Shape estimation

### 6.1 Exploit the bilinear structure (Kaasalainen–Torppa, upgraded)
For a **fixed attitude trajectory**, the brightness model is **linear in the products
`ρ_i·A_i`** (albedo-areas) over any fixed set of candidate normals. So alternate:

1. **Shape step (convex):** discretize the normal sphere (~500–2000 directions), solve
   non-negative least squares for the albedo-area vector `g` — the **Extended Gaussian
   Image (EGI)** — with smoothness regularization. This is a convex problem, fast and
   global.
2. **Attitude step:** with `g` fixed, refine `R(t)` and dynamics parameters by NLS
   (Section 5).
3. Iterate (block coordinate descent / EM-like), with the sensor biases `b_k` and BRDF
   parameters updated in an outer loop.
4. **Minkowski reconstruction:** recover the convex polyhedron whose Gaussian image
   matches `g` (Little's algorithm / modern convex-optimization formulations).

The simultaneous multi-angle sampling is what makes step 1 well-posed: at one instant the
constellation measures many linear functionals of `g` with *different* known coefficient
vectors — a genuine tomographic system, rather than the rank-deficient single-observer
case.

### 6.2 Fundamental limits and how the constellation pushes them
- Photometry of a convex body determines the convex shape up to the **albedo-area
  scaling** (a big dark facet ≡ small bright facet) and a global scale. Absolute range is
  known here, so total `ρA` is absolute; breaking ρ-vs-A per facet still needs BRDF
  diversity — which the wide phase-angle spread supplies (diffuse and specular components
  separate cleanly when the same facet is seen at many phase angles).
- **Nonconvexity** is invisible at low phase angle but imprints shadowing signatures at
  high phase angles (>60–90°) — exactly where a limb-scanning constellation observes a
  lot. Model as a convex core + concavity indicator via the visibility term `V_i`, or go
  fully nonconvex with the differentiable renderer below.
- Discrete mirror ambiguities survive for symmetric bodies; report multimodal posteriors.

### 6.3 Modern route: differentiable rendering + learned initialization
Implement the forward model (Section 3) in a differentiable renderer (mesh or neural SDF
shape representation, facet BRDFs, attitude as a quaternion spline constrained by the
dynamics model). Then:

- **Joint gradient-based MAP** over shape, BRDF, attitude, biases — with the coordinate-
  descent solution of 6.1 as the initializer.
- **Amortized inversion**: train a set-transformer that ingests the unordered set of
  `(u_s, u_k, m)` tuples (in a candidate body frame) and outputs shape-class + coarse
  attitude/spin posteriors. Train entirely on the simulation pipeline (Section 8). Use it
  for Tier-1/2 triage and to seed the physics-based optimizer; never as the final answer.

## 7. Uncertainty, observability, and products

- Run **Fisher information / CRLB studies** vs. number of observers, phase-angle spread,
  arc length, and photometric σ, to set requirements: e.g., how many detections for
  1° attitude, for N-facet shape resolution. Expect attitude observability to saturate
  quickly for glinty objects and to be arc-length-limited for near-Lambertian ones.
- Deliver **posterior distributions**, not point estimates: multi-hypothesis attitude
  modes with weights; shape as EGI with covariance + reconstructed hull; dynamics mode
  probabilities. Downstream users (conjunction assessment, intent assessment, drag/SRP
  modeling) consume the full posterior.
- Attitude products feed back into **orbit products**: a solved attitude+shape gives
  attitude-dependent drag and SRP cross-sections, materially improving the constellation's
  own conjunction screening.

## 8. Validation and calibration plan

1. **Simulation pipeline**: propagate the constellation + target truth (orbit, attitude
   dynamics, CAD shape, measured material BRDFs), ray-trace radiometrically correct
   magnitudes with streak smear, Earthshine, and sensor noise; run the full inversion;
   score attitude/shape recovery. This also generates the ML training corpus.
2. **Cooperative truth targets**: constellation satellites observing each other. Truth
   attitude (telemetry) + truth shape (CAD) on thousands of targets, continuously — an
   unprecedented calibration set. Solve for sensor photometric systematics against it.
3. **Known non-cooperative calibrators**: rocket bodies and geodetic satellites with
   independently known spin states (SLR spin solutions) as blind tests.

## 9. System architecture sketch

```
onboard: detection → streak photometry → tracklet (t, angles, inst. mag, QC) → downlink
ground:  photometric calibration (per sensor, per frame, vs. Gaia)
         → association & OD (multi-observer)  → range-normalized brightness stream per RSO
         → Tier 1 dynamics classifier (period/mode)
         → Tier 2 library model matching  → attitude filter per RSO   → attitude product
         → Tier 3 joint inversion (EGI ↔ attitude BCD → Minkowski → diff-render refine)
         → posterior store (attitude modes, shape, BRDF, biases) + feedback to OD/conjunction
```

## 10. Principal risks

| Risk | Mitigation |
|---|---|
| Star-tracker photometric systematics (stray light, streak smear) alias into shape | Per-sensor bias states; cooperative-target calibration; robust loss functions |
| Attitude–shape degeneracy for near-Lambertian, symmetric objects | Report multimodal posteriors; exploit glints and high-phase shadowing; longer arcs |
| Earthshine modeling error at limb-scanning geometry | Coarse Earth irradiance model + geometry-dependent down-weighting |
| Multimodal likelihood traps local optimizers | Global initialization (grid/Fourier/learned), multi-hypothesis MMAE |
| Data volume (30k sensors) | Onboard detection + tracklet compression; only tracklets downlinked |
