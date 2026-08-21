// Dark-themed briefing deck for the constellation photometry project.
// Build: npm install pptxgenjs && node scripts/make_deck.js  (run from repo root)
const pptxgen = require("pptxgenjs");

const REPO = "/home/user/photometry";
const CHARTS = REPO + "/results/charts";
const MOVIES = REPO + "/results/movies";
const SCRATCH = REPO + "/results/deck_stills";

// palette (matches the chart set)
const BG = "0D0D0D";        // page
const SURFACE = "1A1A19";   // cards
const INK = "FFFFFF";
const INK2 = "C3C2B7";
const MUTED = "898781";
const BLUE = "3987E5";
const ORANGE = "D95926";
const AQUA = "1BAF7A";
const MAGENTA = "D55181";
const YELLOW = "C98500";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: "Calibri", bodyFontFace: "Calibri" };

const W = 13.33, H = 7.5;

function base(slide, kicker, title) {
  slide.background = { color: BG };
  if (kicker) slide.addText(kicker.toUpperCase(), {
    x: 0.55, y: 0.32, w: 9.0, h: 0.3, fontSize: 12, color: MUTED,
    charSpacing: 3, bold: true, margin: 0 });
  if (title) slide.addText(title, {
    x: 0.55, y: 0.58, w: 12.2, h: 0.75, fontSize: 30, color: INK, bold: true,
    margin: 0 });
}

function chip(slide, x, y, w, h, opts = {}) {
  slide.addShape("roundRect", { x, y, w, h, rectRadius: 0.08,
    fill: { color: opts.fill || SURFACE },
    line: { color: opts.line || "2C2C2A", width: 1 } });
}

function stat(slide, x, y, w, value, label, color) {
  chip(slide, x, y, w, 1.5);
  slide.addText(value, { x: x + 0.18, y: y + 0.12, w: w - 0.36, h: 0.75,
    fontSize: 34, bold: true, color: color || INK, margin: 0 });
  slide.addText(label, { x: x + 0.18, y: y + 0.88, w: w - 0.36, h: 0.55,
    fontSize: 11.5, color: INK2, margin: 0 });
}

function img(slide, path, x, y, w, h) {
  slide.addImage({ path, x, y, w, h, sizing: { type: "contain", w, h } });
}

// ---------------------------------------------------------------- 1. title
let s = pres.addSlide();
s.background = { color: BG };
s.addText("CONSTELLATION PHOTOMETRY", { x: 0.8, y: 1.15, w: 11.7, h: 0.4,
  fontSize: 15, color: YELLOW, charSpacing: 5, bold: true, margin: 0 });
s.addText("Satellite Attitude & Shape from 30,000 Star Trackers",
  { x: 0.8, y: 1.55, w: 11.7, h: 1.7, fontSize: 44, bold: true, color: INK, margin: 0 });
s.addText(
  "Turning a 10,000-satellite fleet's star trackers into an opportunistic photometric " +
  "sensor network — and inverting the aggregated visual magnitudes into spin state, " +
  "convex shape, catalog identity, array configuration, and catalog deviations.",
  { x: 0.8, y: 3.15, w: 10.8, h: 1.0, fontSize: 16, color: INK2, margin: 0 });
stat(s, 0.8, 4.6, 2.85, "<1 ms / 0.1°", "baseline spin period / pole recovery", BLUE);
stat(s, 3.85, 4.6, 2.85, "20/21", "attitude-mode classifications correct", AQUA);
stat(s, 6.9, 4.6, 2.85, "20/21", "library identifications correct", ORANGE);
stat(s, 9.95, 4.6, 2.85, "+0.55 m²", "seeded missing-antenna deviation detected", MAGENTA);
s.addText("nikkelj/photometry  ·  PRs #1–#3  ·  simulation + inversion stack + catalog-scale funnel, 18 passing tests",
  { x: 0.8, y: 6.75, w: 11.7, h: 0.35, fontSize: 12, color: MUTED, margin: 0 });

// ------------------------------------------------- 2. the idea / geometry
s = pres.addSlide();
base(s, "concept", "Why a constellation beats a telescope");
s.addText([
  { text: "One ground telescope sees one light curve — a single slowly-moving line of sight. ", options: {} },
  { text: "The classical light-curve inversion problem is ill-posed for exactly that reason.\n\n", options: { color: MUTED } },
  { text: "10,000 satellites × 3 star trackers, canted +5° above the local horizontal, ", options: { bold: true, color: BLUE } },
  { text: "opportunistically detect any object above the shell thousands of times per day, from hundreds of directions at once — a tomographic sampling of the target's reflected light field.\n\n", options: {} },
  { text: "Each detection: time, line of sight, apparent magnitude. Range is known from multi-observer orbit determination, so every sample becomes range-normalized brightness at one (sun, observer) direction pair.", options: {} },
], { x: 0.55, y: 1.55, w: 5.4, h: 4.4, fontSize: 15, color: INK2, margin: 0, valign: "top" });
img(s, CHARTS + "/02_coverage.png", 6.2, 1.75, 6.6, 2.65);
s.addText("Observer directions in the target body frame over one day (left) and the phase-angle diversity of the aggregate (right) — coverage no single site can produce.",
  { x: 6.2, y: 4.5, w: 6.6, h: 0.6, fontSize: 11, color: MUTED, margin: 0 });
stat(s, 6.2, 5.35, 3.2, "5,703", "detections, 588 observers — one 3.2 h baseline arc", BLUE);
stat(s, 9.6, 5.35, 3.2, "14–142°", "solar phase angle span of the same arc", YELLOW);

// ------------------------------------------------------------- 3. the EGI
s = pres.addSlide();
base(s, "concept", "The EGI — what photometry can and cannot know");
s.addText([
  { text: "Extended Gaussian Image: ", options: { bold: true, color: ORANGE } },
  { text: "for every surface patch, take its outward normal direction and its reflecting area, and pile the area onto a sphere of directions. The EGI is “reflecting area per unit direction.”\n\n", options: {} },
  { text: "It is ALL photometry can see. ", options: { bold: true } },
  { text: "An unresolved dot's brightness is a weighted integral of oriented area — position on the body is unrecoverable. A panel mounted left of the bus and the same panel mounted right produce identical light curves.\n\n", options: {} },
  { text: "It is ENOUGH to get a body back. ", options: { bold: true } },
  { text: "Minkowski's theorem: a valid EGI corresponds to exactly one convex body (up to translation). We invert brightness → EGI (convex least squares), then EGI → solid (variational support-functional solve).\n\n", options: {} },
  { text: "vs photogrammetry: ", options: { bold: true, color: AQUA } },
  { text: "photogrammetry triangulates point positions from resolved images. Photometry is the unresolved limit — no pixels on target, ever — so we get the direction-space dual of a surface model: oriented area, no positions.", options: {} },
], { x: 0.55, y: 1.55, w: 6.0, h: 5.3, fontSize: 14.5, color: INK2, margin: 0, valign: "top" });
img(s, CHARTS + "/05_egi.png", 6.8, 1.7, 6.0, 2.2);
s.addText("Recovered EGI vs truth for the baseline box-wing: 99% of recovered area within 15° of true facet normals.",
  { x: 6.8, y: 3.95, w: 6.0, h: 0.45, fontSize: 11, color: MUTED, margin: 0 });
img(s, SCRATCH + "/deck_katalyst_link__tumble.png", 6.8, 4.5, 6.0, 2.55);
s.addText("Raw EGI disks (magenta, positions deliberately exploded — they have none), the Minkowski solid (orange), truth (blue).",
  { x: 6.8, y: 7.0, w: 6.0, h: 0.4, fontSize: 11, color: MUTED, margin: 0 });

// ---------------------------------------------- 4. truth-side progression
s = pres.addSlide();
base(s, "modeling progression", "Truth side — making the simulated world real");
const truthRows = [
  ["T1", "Sensor fleet", "Walker shell 100×100 @ 550 km/53°, three +5°-canted trackers per sat, FOV / sun exclusion / Earth-limb / eclipse. Finding: up-canted trackers only see objects ABOVE the shell.", BLUE],
  ["T2", "Radiometry", "Facet BRDFs — Lambert + Phong per material (MLI, solar cells, white paint, antenna). Glints carry the sharpest attitude information.", BLUE],
  ["T3–T4", "Targets", "Generic box-wing baseline, then a 7-satellite library from open-source dimensions: Starlink v1.5 / v2 mini / v2 mini DTC, BlueWalker 3, Hubble, ISS, Katalyst LINK.", AQUA],
  ["T5–T6", "Attitude & articulation", "LVLH-hold ops, knife-edge low-drag, sun-point, inertial pointing, propeller tumble — plus per-facet solar-array gimbals (1-axis / 2-axis) tracking the sun.", AQUA],
  ["T7", "Saturation as censoring", "Saturated streaks recorded as brighter-than-cap lower bounds, not dropped. ISS-class targets are ~80% censored.", ORANGE],
  ["T8", "Torque-free tumble", "Full Euler dynamics, triaxial inertia, nutating ω (energy conserved to 3e-8) — LINK's actual anomaly class.", ORANGE],
];
let ty = 1.5;
for (const [tag, name, desc, color] of truthRows) {
  chip(s, 0.55, ty, 12.25, 0.86);
  s.addText(tag, { x: 0.75, y: ty + 0.08, w: 0.85, h: 0.7, fontSize: 15, bold: true, color, margin: 0, valign: "middle" });
  s.addText(name, { x: 1.7, y: ty + 0.08, w: 2.15, h: 0.7, fontSize: 13.5, bold: true, color: INK, margin: 0, valign: "middle" });
  s.addText(desc, { x: 3.95, y: ty + 0.06, w: 8.6, h: 0.76, fontSize: 11.5, color: INK2, margin: 0, valign: "middle" });
  ty += 0.96;
}

// ------------------------------------------- 5. algorithm-side progression
s = pres.addSlide();
base(s, "modeling progression", "Algorithm side — the inversion stack");
const algoRows = [
  ["A1–A2", "Spin state", "Arc-sized Lomb–Scargle + pole grid (sphere × phase × body spin axis) + refinement", "period sub-ms, pole 0.1°", BLUE],
  ["A3", "Shape (EGI)", "Joint Lambert + specular NNLS, dead-column pruning, scale-aware ridge", "99% of area on true normals", BLUE],
  ["A4", "Mode classifier", "Named operational laws + fitted spin / inertial families", "20/21 scenarios", AQUA],
  ["A5", "Minkowski solid", "Variational support-functional solve (L-BFGS-B, analytic gradient)", "slab proportions exact", AQUA],
  ["A6", "Library ID", "Model × attitude family × array config, 0.5-mag offset prior for size", "20/21 — DTC vs v2 mini separable", ORANGE],
  ["A7–A8", "Deviation alerts", "Symmetry-group refinement + signed residual EGI + alert criteria", "20/21 decisions fleet-wide", ORANGE],
  ["A9", "Censoring-aware costs", "One-sided Tobit terms, stratified sampling, coherent period ladder", "ISS size info restored", MAGENTA],
  ["A10", "Torque-free fit", "Window-laddered multi-start over inertia + rate + attitude", "attitude error halved", MAGENTA],
];
let ay = 1.5;
for (const [tag, name, desc, res, color] of algoRows) {
  chip(s, 0.55, ay, 12.25, 0.62);
  s.addText(tag, { x: 0.75, y: ay + 0.02, w: 0.9, h: 0.58, fontSize: 13, bold: true, color, margin: 0, valign: "middle" });
  s.addText(name, { x: 1.75, y: ay + 0.02, w: 2.2, h: 0.58, fontSize: 12.5, bold: true, color: INK, margin: 0, valign: "middle" });
  s.addText(desc, { x: 4.05, y: ay + 0.02, w: 5.6, h: 0.58, fontSize: 10.5, color: INK2, margin: 0, valign: "middle" });
  s.addText(res, { x: 9.8, y: ay + 0.02, w: 2.9, h: 0.58, fontSize: 10.5, bold: true, color, margin: 0, valign: "middle" });
  ay += 0.72;
}

// --------------------------------------------------------- 6. baseline
s = pres.addSlide();
base(s, "results — baseline", "Tumbling box-wing, one 3.2-hour arc");
img(s, CHARTS + "/00_summary_tiles.png", 0.55, 1.55, 12.25, 2.05);
img(s, CHARTS + "/01_lightcurve.png", 0.55, 3.75, 6.4, 3.4);
img(s, CHARTS + "/03_periodogram.png", 7.15, 3.75, 5.65, 2.5);
s.addText("Every point in the light curve is a different star tracker. The multi-observer periodogram peak is razor sharp — the alias structure that plagues single-site light curves is gone.",
  { x: 7.15, y: 6.35, w: 5.65, h: 0.85, fontSize: 12, color: MUTED, margin: 0 });

// --------------------------------------------------------- 7. pole + egi
s = pres.addSlide();
base(s, "results — baseline", "Spin pole and shape from magnitudes alone");
img(s, CHARTS + "/04_pole_search.png", 0.55, 1.6, 6.1, 3.5);
img(s, CHARTS + "/05_egi.png", 0.55, 5.15, 8.4, 2.1);
s.addText([
  { text: "Pole search: ", options: { bold: true, color: BLUE } },
  { text: "global grid over the celestial sphere; the antipodal twin is the spin-axis sign ambiguity, carried explicitly.\n\n", options: {} },
  { text: "EGI recovery: ", options: { bold: true, color: ORANGE } },
  { text: "the anti-sunward facet is never illuminated during the arc and is correctly reported unobservable — the algorithm knows what it cannot know.", options: {} },
], { x: 7.0, y: 1.7, w: 5.8, h: 3.3, fontSize: 14.5, color: INK2, margin: 0, valign: "top" });

// ------------------------------------------------------ 8. fleet study
s = pres.addSlide();
base(s, "results — fleet study", "21 scenarios × 24 hours: mode classification");
img(s, CHARTS + "/06_fleet_modes.png", 0.55, 1.5, 6.05, 5.7);
s.addText([
  { text: "Seven satellites × their operational modes, each simulated for a day and classified against a bank of attitude hypotheses.\n\n", options: {} },
  { text: "20/21 correct. ", options: { bold: true, color: AQUA } },
  { text: "The single miss is physics, not software: Hubble tumbling about its own cylinder axis is photometrically near-static — an axisymmetric body's rotation about its symmetry axis is unobservable.\n\n", options: {} },
  { text: "Sun-tracking arrays are not body-fixed: ops-mode shape recovery correctly returns the bus + fixed antennas only, with the array's area appearing at the sun direction in the body frame.", options: {} },
], { x: 7.0, y: 1.7, w: 5.8, h: 4.6, fontSize: 14.5, color: INK2, margin: 0, valign: "top" });

// --------------------------------------------------- 9. library ID grid
s = pres.addSlide();
base(s, "results — identification", "Which catalog object is this, in what state?");
img(s, CHARTS + "/08_model_match.png", 0.55, 1.5, 5.6, 5.75);
s.addText([
  { text: "How to read the grid: ", options: { bold: true, color: ORANGE } },
  { text: "the attitude bank is spacecraft-agnostic — LVLH-hold, knife-edge, sun-point, fitted spin, fitted inertial, × array configs where the model has gimbals. Every cell = the best cost of that ENTIRE bank fitted for that candidate model, normalized per row.\n\n", options: {} },
  { text: "No null cells: ", options: { bold: true } },
  { text: "the fitted families always produce a best-effort attitude, so impostors get finite-but-poor costs, and odd pairings (a knife-edge ISS) simply lose and never win a row.\n\n", options: {} },
  { text: "20/21 correct, ", options: { bold: true, color: AQUA } },
  { text: "including v2 mini DTC vs plain v2 mini (cost 1.09 vs 2.64 — a half-bus-length antenna is photometrically detectable) and arrays-tracking vs frozen. The one miss (tumbling ISS → “Hubble”) fits nothing well — best cost ~29 where real matches score ~1 — so a fielded system reports “no confident match”.", options: {} },
], { x: 6.55, y: 1.7, w: 6.25, h: 5.5, fontSize: 13.5, color: INK2, margin: 0, valign: "top" });

// ------------------------------------------- 10. residual EGI + deviation
s = pres.addSlide();
base(s, "results — deviation detection", "Does reality match the catalog?");
img(s, CHARTS + "/09_residual_egi.png", 0.55, 1.5, 12.25, 3.15);
img(s, CHARTS + "/11_deviation_scan.png", 0.55, 4.8, 7.3, 2.5);
s.addText([
  { text: "Signed residual EGI ", options: { bold: true, color: MAGENTA } },
  { text: "on top of the matched model: a missing antenna shows up as +0.55 m² of localized area at the right normal (middle panel); correct models leave noise.\n\n", options: {} },
  { text: "Fleet alert scan: 20/21 ", options: { bold: true, color: AQUA } },
  { text: "decisions correct. The one flag is arguably a true positive — the identified attitude genuinely cannot explain Hubble's frozen arrays sweeping with the real rotation.", options: {} },
], { x: 8.1, y: 5.0, w: 4.7, h: 2.2, fontSize: 12.5, color: INK2, margin: 0, valign: "top" });

// ------------------------------------------------- 11. hard cases
s = pres.addSlide();
base(s, "results — hard cases", "Saturation censoring and multi-axis tumbles");
img(s, CHARTS + "/10_torquefree.png", 0.55, 1.55, 7.1, 3.1);
img(s, SCRATCH + "/deck_iss__tumble.png", 0.55, 4.85, 4.3, 2.35);
s.addText("The one misidentification, visibly wrong: a tiny Hubble tube inside the real ISS — flagged by its outlier cost.",
  { x: 0.55, y: 7.2, w: 4.3, h: 0.3, fontSize: 10, color: MUTED, margin: 0 });
s.addText([
  { text: "Censoring: ", options: { bold: true, color: ORANGE } },
  { text: "an ISS saturates the trackers on its bright spin phases — exactly the samples that carry the spin. One-sided “brighter-than-cap” terms restore size information (truth-attitude ISS fits at 0.18 vs 8.8+ for impostors); the global attitude search under 80% censoring is the open piece.\n\n", options: {} },
  { text: "Multi-axis tumble: ", options: { bold: true, color: MAGENTA } },
  { text: "torque-free dynamics halve the attitude error vs the best uniform-spin fit (97.8° → 50.7° in-window; 56.0° over a 4 h prediction), recovering the dominant rate component to 4%. Full convergence needs a stronger global optimizer — the machinery is verified (truth scores 1.04 through the exact fit path).", options: {} },
], { x: 8.0, y: 1.75, w: 4.8, h: 5.2, fontSize: 13, color: INK2, margin: 0, valign: "top" });

// ------------------------------------- 11b. glint geometry / Wahba
s = pres.addSlide();
base(s, "results — glint geometry", "Wahba's problem: attitude waypoints from specular events");
img(s, CHARTS + "/14_glint_wahba.png", 0.55, 1.5, 8.1, 5.6);
s.addText([
  { text: "A glint is the one moment photometry yields a direction: ", options: { bold: true, color: BLUE } },
  { text: "the phase-angle bisector (known in ECI) coincides with a facet normal (known in the body frame). Each confirmed glint is a matched vector pair — the input to Wahba's problem — and Davenport's q-method turns a window of pairs into an absolute attitude fix.\n\n", options: {} },
  { text: "Detector: ", options: { bold: true } },
  { text: "brightness vs the fleet median at the same rotational phase (cancels the spin modulation); geometric gating assigns the facet and lifts pair purity to ~100%.\n\n", options: {} },
  { text: "Waypoints at 2.9° ", options: { bold: true, color: AQUA } },
  { text: "(LINK tumble). Correct correspondences resolve the v1.5 symmetry twin scalar photometry cannot (5.5° vs 86°).\n\n", options: {} },
  { text: "25 waypoints @ 3.4° ", options: { bold: true, color: MAGENTA } },
  { text: "live in the torque-free arc under oracle gating vs 7 pairs under the current coarse hypothesis — the iterate-refine headroom that will seed the Tier-3 dynamics fit.", options: {} },
], { x: 8.85, y: 1.7, w: 4.0, h: 5.5, fontSize: 12, color: INK2, margin: 0, valign: "top" });

// ------------------------------------- 11c. catalog scale
s = pres.addSlide();
base(s, "results — catalog scale", "209 models: identity becomes class + configuration");
img(s, CHARTS + "/15_library_scale.png", 0.55, 1.5, 12.25, 3.35);
s.addText([
  { text: "24 sampled targets (11 families, 38 countries, mixed modes, two with off-pointed arrays), 3 h arcs, full funnel: feature gate (ms) → two-channel shortlist (~2 s) → matcher (~3 min).\n\n", options: {} },
  { text: "Exact identity 29% · family/class 71%. ", options: { bold: true, color: ORANGE } },
  { text: "10 of 17 misses picked a same-family photometric twin fitting within 1.2× of truth — the serial number is not in the data once the catalog holds look-alikes; class, size, and configuration are.", options: {} },
], { x: 0.55, y: 5.0, w: 6.3, h: 2.3, fontSize: 12.5, color: INK2, margin: 0, valign: "top" });
s.addText([
  { text: "Self-assessment survives scale: ", options: { bold: true, color: AQUA } },
  { text: "every correct ID fits at cost ≤ 1.0; 13/17 wrong ones flag themselves (cost 1.05–25 or shortlist cut). The confident-but-wrong residue is exactly the twin set.\n\n", options: {} },
  { text: "Operational product: ", options: { bold: true, color: YELLOW } },
  { text: "the ranked shortlist with margins — “one of these three 3U cubesats, ops attitude, arrays tracking” — plus orbit-catalog correlation as the free tie-breaker.", options: {} },
], { x: 7.15, y: 5.0, w: 5.65, h: 2.3, fontSize: 12.5, color: INK2, margin: 0, valign: "top" });

// ------------------------------------- 11d. observability trade
s = pres.addSlide();
base(s, "results — observability trade", "How many observers before the tomography turns on?");
img(s, CHARTS + "/17_observability_trade.png", 0.55, 1.5, 8.35, 5.7);
s.addText([
  { text: "A smaller constellation is an observer-ID subsample of the full-fleet run — same shell, same chain, N of 10,000 IDs kept.\n\n", options: {} },
  { text: "The answer is a phase transition at N ≈ 300–1,000: ", options: { bold: true, color: ORANGE } },
  { text: "period recovery 0/3 seeds at 300, 3/3 at 1,000 — and the instant it locks, pole error collapses 85° → 0.1° and EGI jumps to 55–89%, both already at full-fleet quality.\n\n", options: {} },
  { text: "Binding resource: sampling density, not geometry. ", options: { bold: true, color: BLUE } },
  { text: "Directional coverage saturates (~50%, sun-capped at 56%) before period success climbs. And there is no single-observer regime: 1 satellite ≈ 0–3 detections/day.\n\n", options: {} },
  { text: "What still scales at 10k: ", options: { bold: true, color: MAGENTA } },
  { text: "Wahba vector pairs (0 → 4 → 12 → 37/day) — the glint channel earns the full fleet after every scalar metric saturates.", options: {} },
], { x: 9.1, y: 1.7, w: 3.75, h: 5.5, fontSize: 11.5, color: INK2, margin: 0, valign: "top" });

// ------------------------------------- 11e. maneuver slews
s = pres.addSlide();
base(s, "results — maneuver slews", "Pre-burn 90/180° yaw-arounds: detect always, characterize by symmetry");
img(s, CHARTS + "/18_slew_detectability.png", 0.55, 1.5, 8.0, 5.7);
s.addText([
  { text: "6 min yaw-out, 15 min burn hold, 6 min return — 4 craft × {90°, 180°} + controls.\n\n", options: {} },
  { text: "Detection: 8/8 at robust-z 281–1204, latency ≤ 60 s, controls ≤ 4.1, zero false alarms. ", options: { bold: true, color: AQUA } },
  { text: "Even the 2-axis-array flat-sat is loud: panel backs and bus edges swing even when the cells stay sun-locked.\n\n", options: {} },
  { text: "90° holds: fully characterized ", options: { bold: true, color: BLUE } },
  { text: "— hold yaw to 0.1–0.2°, slew start ±0.5–2 min; durations softer (shallow smoothstep tails).\n\n", options: {} },
  { text: "180° holds are invisible ", options: { bold: true, color: ORANGE } },
  { text: "on mirror-symmetric buses — residual returns to baseline mid-burn; the two transients carry everything. Both seen → burn bracketed to ~1 min (Persona: −7 s); fit interpolating a flat basin can slide (v2 mini: +18 min).\n\n", options: {} },
  { text: "Product ordering: ", options: { bold: true, color: YELLOW } },
  { text: "windowed z + yaw track is the robust detector; the parametric slew fit is the refinement.", options: {} },
], { x: 8.75, y: 1.65, w: 4.1, h: 5.6, fontSize: 11, color: INK2, margin: 0, valign: "top" });

// ------------------------------------------------- 12. movies
s = pres.addSlide();
base(s, "results — validation movies", "Watching the inversion against nav truth");
s.addText("Left: truth (blue) + model-free inversion — Minkowski solid (orange), raw EGI disks (magenta). Right: identified library model at identified attitude/articulation (green). Bottom: LVLH-plane projections. 21 animations in results/movies/ — the GIF below animates in slideshow mode.",
  { x: 0.55, y: 1.45, w: 12.25, h: 0.75, fontSize: 13, color: INK2, margin: 0 });
s.addImage({ path: MOVIES + "/katalyst_link__tumble.gif", x: 0.55, y: 2.3, w: 7.6, h: 4.85, sizing: { type: "contain", w: 7.6, h: 4.85 } });
img(s, SCRATCH + "/deck_starlink_v2mini_dtc__ops.png", 8.35, 2.3, 4.45, 2.35);
s.addText("DTC ops: truth arrays articulate toward the sun; identification correct incl. “arrays tracking”",
  { x: 8.35, y: 4.7, w: 4.45, h: 0.35, fontSize: 10, color: MUTED, margin: 0 });
img(s, SCRATCH + "/deck_katalyst_link__multiaxis_tumble.png", 8.35, 5.1, 4.45, 2.0);
s.addText("LINK multi-axis tumble with the Tier-3 torque-free fit in the right panel",
  { x: 8.35, y: 7.14, w: 4.45, h: 0.3, fontSize: 10, color: MUTED, margin: 0 });

// ------------------------------------------------- 13. real data wiring
s = pres.addSlide();
base(s, "path to operations", "Wiring the experiment to real data");
s.addText([
  { text: "The seam is already built: ", options: { bold: true, color: AQUA } },
  { text: "every algorithm consumes only the ObservationSet schema — one row per calibrated detection (time, observer, line of sight, sun, range, magnitude, sigma, bias, censored flag). Real integration = filling that table.", options: {} },
], { x: 0.55, y: 1.5, w: 12.25, h: 0.75, fontSize: 14.5, color: INK2, margin: 0 });
const wire = [
  ["Tracker transient streams", "Star trackers already detect non-stellar objects every frame — the star-ID rejects. The product to downlink: time, centroid + attitude → inertial LOS, streak-integrated magnitude, saturation flag.", BLUE],
  ["Bridge: ground photometry", "MMT-9 public light curves, commercial SDA networks, SatObs flash periods — degraded single-site geometry, but real BRDFs and noise to confront the calibration pipeline early.", BLUE],
  ["Orbits & references", "Space-Track / operator ephemerides for association and range; Gaia DR3 G-band as the per-frame photometric zero point; DE440 sun; ILRS truth orbits for calibrators.", AQUA],
  ["Calibration loop", "Run the stack on the constellation's own satellites (telemetered attitude, CAD shape) with truth held fixed — solve per-sensor biases, color terms, and BRDF materials instead. Continuous, free truth.", ORANGE],
  ["Hard parts, named", "Streak photometry (trailed PSF flux) and fleet-scale association. First light: predict known targets' transits through each FOV and harvest only those windows.", MAGENTA],
  ["Code adapters", "EphemerisOrbit wrapper for real orbits behind the LVLH hypotheses; Earthshine term in radiometry; passband refit of material coefficients; robustness knobs retuned on real residuals.", YELLOW],
];
let wy = 2.32;
for (const [name, desc, color] of wire) {
  chip(s, 0.55, wy, 12.25, 0.7);
  s.addText(name, { x: 0.75, y: wy + 0.03, w: 2.7, h: 0.64, fontSize: 12.5, bold: true, color, margin: 0, valign: "middle" });
  s.addText(desc, { x: 3.55, y: wy + 0.03, w: 9.05, h: 0.64, fontSize: 10.5, color: INK2, margin: 0, valign: "middle" });
  wy += 0.78;
}
s.addText("Full guide: docs/real-data-integration.md — field-by-field schema mapping, reduction pipeline, worked loader skeleton.",
  { x: 0.55, y: wy + 0.05, w: 12.25, h: 0.3, fontSize: 11, color: MUTED, margin: 0 });

// ------------------------------------------------- 14. findings + roadmap
s = pres.addSlide();
base(s, "synthesis", "Standing findings and where to head next");
const findings = [
  ["Altitude envelope", "+5°-canted trackers only see objects above the shell — ISS, Hubble, LINK are invisible to a 550 km shell at their real altitudes.", BLUE],
  ["Censoring is the tall pole", "Bright spin phases are the saturated ones: calibrated-only data shrinks objects and hides spin. The saturation timing pattern is itself signal.", ORANGE],
  ["Ambiguities are structural", "90° plate swaps, 180° tube flips, pole antipodes — search them explicitly, report them as posterior modes.", MAGENTA],
  ["Articulation is information", "Arrays tracking vs frozen is directly identifiable; a pointed telescope still sun-tracks its arrays.", AQUA],
];
let fy = 1.55;
for (const [name, desc, color] of findings) {
  chip(s, 0.55, fy, 6.0, 1.28);
  s.addText(name, { x: 0.75, y: fy + 0.1, w: 5.6, h: 0.35, fontSize: 14, bold: true, color, margin: 0 });
  s.addText(desc, { x: 0.75, y: fy + 0.46, w: 5.6, h: 0.75, fontSize: 11, color: INK2, margin: 0 });
  fy += 1.42;
}
const roadmap = [
  "Glint-seeded torque-free fitting: close the iterate-refine loop (7 → 134 pairs measured headroom), fit dynamics through Wahba waypoints",
  "Close the funnel gaps: third shortlist channel for inertial pointers, arc-aware fingerprints, off-pointed array hypotheses, orbit-catalog tie-breaking among twins",
  "Maneuver-window change-point scanning: freeze the match, score in time bins, recurse on the bins that fail",
  "Harmonic-aware censored period ladder — closes the last curated misidentification (ISS-class tumblers)",
  "Per-sensor photometric bias estimation: 30k sensors, cooperative constellation targets as truth",
  "Real tracklets: reduce to the ObservationSet schema and the identical stack runs",
];
chip(s, 6.85, 1.55, 5.95, 5.55);
s.addText("ROADMAP", { x: 7.1, y: 1.75, w: 5.4, h: 0.3, fontSize: 12, bold: true, color: YELLOW, charSpacing: 3, margin: 0 });
s.addText(roadmap.map((t, i) => ({
  text: t, options: { bullet: true, breakLine: i < roadmap.length - 1, paraSpaceAfter: 10 } })),
  { x: 7.1, y: 2.15, w: 5.45, h: 4.8, fontSize: 13.5, color: INK2, margin: 0, valign: "top" });

pres.writeFile({ fileName: REPO + "/docs/photometry_briefing.pptx" })
  .then(() => console.log("deck written"));
