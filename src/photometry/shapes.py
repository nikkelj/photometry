"""Facet-based target shape models with per-facet reflectance, articulation,
and drawable geometry.

Library models are box-wing approximations assembled from open-source
dimensions (public filings, press kits, observer photography). They are
photometric stand-ins, not engineering CAD: facet areas, normals, and
material classes are what the forward model consumes.

Body-frame convention for all library models (matches LVLH-hold attitude):
    +x along-track (velocity), +y cross-track (orbit normal), +z zenith.

Articulation: each facet carries a gimbal mode —
  GIMBAL_FIXED   normal fixed in the body frame
  GIMBAL_1AXIS   normal rotates about `gimbal_axis` to chase the sun
                 (cosine losses out-of-plane remain)
  GIMBAL_2AXIS   normal points at the sun exactly (shoulder + wrist)
A facet with mirror_of >= 0 is the back side of another facet and always
carries the opposite of its partner's current normal.

Catalog extensions (optional; unused by the 620 km study forward model):
  per-facet material class + IR α/ε (NaN = unknown), body-frame thrust
  unit vectors vs the nominal flight attitude they are defined against,
  and source / dimension-status metadata. See `photometry.catalog`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frames import unit

GIMBAL_FIXED = 0
GIMBAL_1AXIS = 1
GIMBAL_2AXIS = 2

# material presets: (diffuse albedo, specular coefficient, Phong exponent)
# Optical numbers are assumed photometric stand-ins (not spacecraft-measured)
# unless a family notes otherwise. IR α/ε are unknown at the spacecraft
# level; handbook class values live in photometry.catalog.materials.
MLI = (0.30, 0.10, 30.0)
MLI_SILVER = (0.35, 0.35, 60.0)
CELLS = (0.06, 0.55, 800.0)
PANEL_BACK = (0.35, 0.02, 10.0)
WHITE_PAINT = (0.85, 0.03, 15.0)
ANTENNA = (0.20, 0.30, 100.0)
DARK = (0.10, 0.02, 10.0)

PRESET_CLASS = {
    MLI: "MLI",
    MLI_SILVER: "MLI_SILVER",
    CELLS: "CELLS",
    PANEL_BACK: "PANEL_BACK",
    WHITE_PAINT: "WHITE_PAINT",
    ANTENNA: "ANTENNA",
    DARK: "DARK",
}

# Nominal flight attitudes a thrust vector may be defined against.
ATT_LVLH = "lvlh"                 # +x ram, +y orbit-normal, +z zenith
ATT_NADIR = "nadir"               # Earth-pointing; body +z or -z as noted
ATT_SUN_TRACK = "sun_track"
ATT_YAW_STEER = "yaw_steering"    # GNSS-class
ATT_STAGE_AXIS = "stage_axis"     # cylinder +z; on-orbit attitude uncontrolled
ATT_UNKNOWN = "unknown"

PROP_EP = "ep"
PROP_CHEMICAL = "chemical"
PROP_NONE = "none"
PROP_UNKNOWN = "unknown"

# Dimension / pointing provenance (never a fake precise number).
STATUS_PUBLIC = "public"
STATUS_RANGE = "range"          # numeric stand-in is a published range midpoint
STATUS_UNCERTAIN = "uncertain"
STATUS_UNKNOWN = "unknown"
STATUS_TYPICAL = "typical_class"  # class convention, not this spacecraft


@dataclass
class FacetModel:
    """Facet model in the target body frame.

    normals are the *rest* normals; `body_normals` evaluates articulation.
    polygons: drawable (vertices (V,3), facet_index) pairs for rendering —
    the movie/truth-visualization layer only, never the photometry.
    """

    normals: np.ndarray
    areas: np.ndarray
    rho_d: np.ndarray
    k_s: np.ndarray
    n_ph: np.ndarray
    labels: list[str]
    gimbal_mode: np.ndarray = None
    gimbal_axis: np.ndarray = None
    mirror_of: np.ndarray = None
    polygons: list = field(default_factory=list)
    name: str = "unnamed"
    # per-facet surface identity (optical BRDF is still rho_d/k_s/n_ph)
    material_class: list[str] = field(default_factory=list)
    alpha_ir: np.ndarray = None          # NaN = unknown
    epsilon_ir: np.ndarray = None        # NaN = unknown
    optical_provenance: list[str] = field(default_factory=list)
    ir_provenance: list[str] = field(default_factory=list)
    # body-frame thrust; empty (0,3) means pointing unknown / none
    thrust_body: np.ndarray = None
    thrust_attitude: str = ATT_UNKNOWN
    thrust_propulsion: str = PROP_UNKNOWN
    thrust_notes: str = ""
    family_id: str = ""
    sources: tuple[str, ...] = ()
    notes: str = ""
    dimension_status: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        f = len(self.areas)
        if self.gimbal_mode is None:
            self.gimbal_mode = np.zeros(f, dtype=int)
        if self.gimbal_axis is None:
            self.gimbal_axis = np.tile([1.0, 0.0, 0.0], (f, 1))
        if self.mirror_of is None:
            self.mirror_of = np.full(f, -1, dtype=int)
        if self.alpha_ir is None:
            self.alpha_ir = np.full(f, np.nan)
        if self.epsilon_ir is None:
            self.epsilon_ir = np.full(f, np.nan)
        if self.thrust_body is None:
            self.thrust_body = np.zeros((0, 3))
        if len(self.material_class) != f:
            self.material_class = list(self.material_class) + ["unspecified"] * (
                f - len(self.material_class))
        if len(self.optical_provenance) != f:
            self.optical_provenance = list(self.optical_provenance) + [
                "assumed"] * (f - len(self.optical_provenance))
        if len(self.ir_provenance) != f:
            self.ir_provenance = list(self.ir_provenance) + ["unknown"] * (
                f - len(self.ir_provenance))
        if not self.family_id:
            self.family_id = self.name

    @property
    def n_facets(self) -> int:
        return len(self.areas)

    @property
    def articulated(self) -> bool:
        return bool(np.any(self.gimbal_mode != GIMBAL_FIXED))

    def diffuse_albedo_area(self) -> np.ndarray:
        return self.rho_d * self.areas

    def body_normals(self, u_sun_body: np.ndarray,
                     articulate: bool = True) -> np.ndarray:
        """Facet normals (F,K,3) for K sun directions in the body frame."""
        k = len(u_sun_body)
        n = np.repeat(self.normals[:, None, :], k, axis=1)
        if not (articulate and self.articulated):
            return n
        for i in range(self.n_facets):
            if self.mirror_of[i] >= 0:
                continue
            if self.gimbal_mode[i] == GIMBAL_2AXIS:
                n[i] = u_sun_body
            elif self.gimbal_mode[i] == GIMBAL_1AXIS:
                g = self.gimbal_axis[i]
                s_perp = u_sun_body - np.outer(u_sun_body @ g, g)
                nrm = np.linalg.norm(s_perp, axis=-1, keepdims=True)
                ok = nrm[:, 0] > 1e-9
                n[i, ok] = s_perp[ok] / nrm[ok]
        for i in range(self.n_facets):
            if self.mirror_of[i] >= 0:
                n[i] = -n[self.mirror_of[i]]
        return n

    def characteristic_radius(self) -> float:
        """Rough size scale (m) from drawable geometry, for rendering."""
        if not self.polygons:
            return float(np.sqrt(self.areas.max()))
        return float(max(np.abs(v).max() for v, _ in self.polygons))

    def gimbal_name(self, i: int) -> str:
        return {GIMBAL_FIXED: "fixed", GIMBAL_1AXIS: "1-axis",
                GIMBAL_2AXIS: "2-axis"}.get(int(self.gimbal_mode[i]), "?")

    def describe(self) -> str:
        """Catalog card: bus / arrays / deployables / thrust / materials."""
        lines = [f"{self.name}  (family {self.family_id or self.name})",
                 f"  facets: {self.n_facets}"]
        by_role = {"bus": [], "array": [], "deployable": [], "other": []}
        for i, lab in enumerate(self.labels):
            low = lab.lower()
            if any(k in low for k in ("array", "panel", "cells", "wing",
                                      "radiator")):
                role = "array"
            elif any(k in low for k in ("antenna", "dtc", "dish", "sar",
                                        "boom", "capture")):
                role = "deployable"
            elif any(k in low for k in ("bus", "tube", "module", "truss",
                                        "stage")):
                role = "bus"
            else:
                role = "other"
            by_role[role].append(i)
        for role, idxs in by_role.items():
            if not idxs:
                continue
            fronts = [i for i in idxs if self.mirror_of[i] < 0]
            gmodes = sorted({self.gimbal_name(i) for i in fronts})
            mats = sorted({self.material_class[i] for i in idxs})
            area = float(sum(self.areas[i] for i in fronts))
            lines.append(
                f"  {role}: {len(fronts)} front facets, {area:.2f} m^2, "
                f"gimbal {','.join(gmodes)}, materials {','.join(mats)}")
        if len(self.thrust_body) == 0:
            lines.append(
                f"  thrust: pointing {STATUS_UNKNOWN}; "
                f"propulsion={self.thrust_propulsion}; "
                f"attitude={self.thrust_attitude}")
        else:
            vecs = " ".join(
                "[" + ",".join(f"{x:.3f}" for x in v) + "]"
                for v in self.thrust_body)
            lines.append(
                f"  thrust body-frame {vecs} vs {self.thrust_attitude}; "
                f"propulsion={self.thrust_propulsion}")
        if self.thrust_notes:
            lines.append(f"  thrust notes: {self.thrust_notes}")
        n_ir = int(np.sum(~np.isnan(self.alpha_ir)))
        lines.append(
            f"  surfaces: all {self.n_facets} facets have material class + "
            f"Lambert-Phong (rho_d, k_s, n_ph); IR α/ε known on {n_ir}/"
            f"{self.n_facets} (else unknown)")
        if self.dimension_status:
            bits = ", ".join(f"{k}={v}" for k, v in self.dimension_status.items())
            lines.append(f"  dimension status: {bits}")
        if self.sources:
            lines.append("  sources:")
            for s in self.sources:
                lines.append(f"    - {s}")
        if self.notes:
            lines.append(f"  notes: {self.notes}")
        return "\n".join(lines)


class _Builder:
    def __init__(self, name: str):
        self.name = name
        self.normals, self.areas = [], []
        self.rho_d, self.k_s, self.n_ph = [], [], []
        self.labels = []
        self.gmode, self.gaxis, self.mirror = [], [], []
        self.polygons = []
        self.material_class, self.optical_prov, self.ir_prov = [], [], []
        self.alpha_ir, self.epsilon_ir = [], []
        self.family_id = name
        self.sources: tuple[str, ...] = ()
        self.notes = ""
        self.dimension_status: dict = {}
        self.thrust_body = np.zeros((0, 3))
        self.thrust_attitude = ATT_UNKNOWN
        self.thrust_propulsion = PROP_UNKNOWN
        self.thrust_notes = ""

    def meta(self, *, family_id: str | None = None, sources=(), notes: str = "",
             dimension_status: dict | None = None, thrust_body=None,
             thrust_attitude: str = ATT_UNKNOWN,
             thrust_propulsion: str = PROP_UNKNOWN,
             thrust_notes: str = "") -> "_Builder":
        if family_id is not None:
            self.family_id = family_id
        self.sources = tuple(sources)
        self.notes = notes
        self.dimension_status = dict(dimension_status or {})
        if thrust_body is None:
            self.thrust_body = np.zeros((0, 3))
        else:
            tb = np.asarray(thrust_body, dtype=float).reshape(-1, 3)
            nrm = np.linalg.norm(tb, axis=1, keepdims=True)
            self.thrust_body = np.divide(tb, nrm, out=np.zeros_like(tb),
                                         where=nrm > 0)
        self.thrust_attitude = thrust_attitude
        self.thrust_propulsion = thrust_propulsion
        self.thrust_notes = thrust_notes
        return self

    def facet(self, normal, area, mat, label, gimbal=GIMBAL_FIXED,
              gimbal_axis=(1, 0, 0), mirror_of=-1, polygon=None,
              material_class: str | None = None,
              optical_provenance: str = "assumed",
              alpha_ir: float | None = None, epsilon_ir: float | None = None,
              ir_provenance: str = "unknown") -> int:
        i = len(self.areas)
        self.normals.append(unit(np.asarray(normal, dtype=float)))
        self.areas.append(float(area))
        tup = tuple(mat) if not isinstance(mat, tuple) else mat
        self.rho_d.append(float(tup[0]))
        self.k_s.append(float(tup[1]))
        self.n_ph.append(float(tup[2]))
        self.labels.append(label)
        self.gmode.append(gimbal)
        self.gaxis.append(unit(np.asarray(gimbal_axis, dtype=float)))
        self.mirror.append(mirror_of)
        cls = material_class or PRESET_CLASS.get(tup, PRESET_CLASS.get(mat, "custom"))
        self.material_class.append(cls)
        self.optical_prov.append(optical_provenance)
        self.alpha_ir.append(np.nan if alpha_ir is None else float(alpha_ir))
        self.epsilon_ir.append(np.nan if epsilon_ir is None else float(epsilon_ir))
        self.ir_prov.append(ir_provenance)
        if polygon is not None:
            self.polygons.append((np.asarray(polygon, dtype=float), i))
        return i

    def box(self, center, dims, mat, label, mats=None) -> None:
        """Axis-aligned box; mats optionally overrides per +x,-x,+y,-y,+z,-z."""
        c = np.asarray(center, dtype=float)
        d = np.asarray(dims, dtype=float) / 2
        axes = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
        names = ["x", "y", "z"]
        k = 0
        for ax, u_ax, v_ax in axes:
            for sign in (1, -1):
                n = np.zeros(3)
                n[ax] = sign
                area = dims[u_ax] * dims[v_ax]
                e_u, e_v = np.eye(3)[u_ax] * d[u_ax], np.eye(3)[v_ax] * d[v_ax]
                base = c + n * d[ax]
                poly = [base + e_u + e_v, base - e_u + e_v, base - e_u - e_v,
                        base + e_u - e_v]
                m = mats[k] if mats else mat
                sgn = "+" if sign > 0 else "-"
                self.facet(n, area, m, f"{label} {sgn}{names[ax]}", polygon=poly)
                k += 1

    def panel(self, center, u_dir, v_dir, w, h, front_mat, back_mat,
              label, gimbal=GIMBAL_FIXED, gimbal_axis=(1, 0, 0)) -> None:
        """Two-sided flat panel; front normal = u_dir x v_dir."""
        c = np.asarray(center, dtype=float)
        u = unit(np.asarray(u_dir, dtype=float)) * (w / 2)
        v = unit(np.asarray(v_dir, dtype=float)) * (h / 2)
        n = unit(np.cross(u, v))
        poly = [c + u + v, c - u + v, c - u - v, c + u - v]
        i_front = self.facet(n, w * h, front_mat, f"{label} front", gimbal,
                             gimbal_axis, polygon=poly)
        self.facet(-n, w * h, back_mat, f"{label} back", gimbal, gimbal_axis,
                   mirror_of=i_front)

    def prism(self, center, axis_dir, length, diameter, n_side, side_mat,
              cap_mats, label) -> None:
        """Cylinder approximated by n_side rectangular side facets + caps."""
        c = np.asarray(center, dtype=float)
        a = unit(np.asarray(axis_dir, dtype=float))
        # two directions perpendicular to the axis
        tmp = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = unit(np.cross(a, tmp))
        e2 = np.cross(a, e1)
        r = diameter / 2
        side_area = np.pi * diameter * length / n_side
        half = a * (length / 2)
        for i in range(n_side):
            th0, th1 = 2 * np.pi * i / n_side, 2 * np.pi * (i + 1) / n_side
            n = np.cos((th0 + th1) / 2) * e1 + np.sin((th0 + th1) / 2) * e2
            p0 = c + r * (np.cos(th0) * e1 + np.sin(th0) * e2)
            p1 = c + r * (np.cos(th1) * e1 + np.sin(th1) * e2)
            poly = [p0 + half, p1 + half, p1 - half, p0 - half]
            self.facet(n, side_area, side_mat, f"{label} side{i}", polygon=poly)
        cap_area = np.pi * r**2
        for sign, mat, tag in [(1, cap_mats[0], "+cap"), (-1, cap_mats[1], "-cap")]:
            th = np.linspace(0, 2 * np.pi, 13)
            poly = [c + sign * half + r * (np.cos(t) * e1 + np.sin(t) * e2)
                    for t in th[:-1]]
            self.facet(sign * a, cap_area, mat, f"{label} {tag}", polygon=poly)

    def build(self) -> FacetModel:
        return FacetModel(
            normals=np.array(self.normals),
            areas=np.array(self.areas),
            rho_d=np.array(self.rho_d),
            k_s=np.array(self.k_s),
            n_ph=np.array(self.n_ph),
            labels=self.labels,
            gimbal_mode=np.array(self.gmode, dtype=int),
            gimbal_axis=np.array(self.gaxis),
            mirror_of=np.array(self.mirror, dtype=int),
            polygons=self.polygons,
            name=self.name,
            material_class=list(self.material_class),
            alpha_ir=np.array(self.alpha_ir, dtype=float),
            epsilon_ir=np.array(self.epsilon_ir, dtype=float),
            optical_provenance=list(self.optical_prov),
            ir_provenance=list(self.ir_prov),
            thrust_body=np.array(self.thrust_body, dtype=float).reshape(-1, 3),
            thrust_attitude=self.thrust_attitude,
            thrust_propulsion=self.thrust_propulsion,
            thrust_notes=self.thrust_notes,
            family_id=self.family_id,
            sources=self.sources,
            notes=self.notes,
            dimension_status=dict(self.dimension_status),
        )


# ---------------------------------------------------------------------------
# Generic test article (round-1 baseline)
# ---------------------------------------------------------------------------

def box_wing(
    box_dims_m: tuple[float, float, float] = (1.2, 1.0, 2.0),
    panel_area_m2: float = 4.0,
) -> FacetModel:
    """Generic defunct box-wing: 6-face bus + fixed 2-sided solar panel (+x)."""
    b = _Builder("box_wing")
    b.box((0, 0, 0), box_dims_m, MLI, "bus")
    side = float(np.sqrt(panel_area_m2))
    cx = box_dims_m[0] / 2 + 0.3 + side / 2
    b.panel((cx, 0, 0), (0, 0, -1), (0, 1, 0), side, side, CELLS, PANEL_BACK,
            "panel")
    b.meta(
        family_id="box_wing",
        sources=("Generic test article used in the round-1 baseline; not a "
                 "flight vehicle."),
        notes="Defunct box-wing stand-in for inversion tests.",
        dimension_status={"bus": STATUS_TYPICAL, "arrays": STATUS_TYPICAL},
        thrust_attitude=ATT_UNKNOWN, thrust_propulsion=PROP_UNKNOWN,
        thrust_notes="No public vehicle; thrust unknown by construction.",
    )
    return b.build()


def rocket_body(length_m: float = 8.0, diameter_m: float = 2.4,
                n_side_facets: int = 16) -> FacetModel:
    """Cylindrical upper stage."""
    b = _Builder("rocket_body")
    b.prism((0, 0, 0), (0, 0, 1), length_m, diameter_m, n_side_facets,
            MLI_SILVER, (MLI, DARK), "stage")
    b.meta(
        family_id="rocket_body",
        sources=("Generic cylindrical upper-stage stand-in; specific stages "
                 "are in photometry.catalog (falcon9_s2, cz_upper, …)."),
        dimension_status={"stage": STATUS_TYPICAL},
        thrust_body=[[0.0, 0.0, -1.0]],
        thrust_attitude=ATT_STAGE_AXIS,
        thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="Engine along −z of the cylinder; spent-stage attitude "
                     "is typically uncontrolled.",
    )
    return b.build()


# ---------------------------------------------------------------------------
# Satellite library (open-source approximate dimensions)
# ---------------------------------------------------------------------------

def starlink_v15() -> FacetModel:
    """Starlink v1.5: flat bus ~2.8 x 1.4 m, one ~8.1 x 2.7 m solar array on a
    single (shoulder) gimbal — 1-axis sun tracking about the boom axis."""
    b = _Builder("starlink_v15")
    b.box((0, 0, 0), (2.8, 1.4, 0.2), MLI, "bus",
          mats=[MLI, MLI, MLI, MLI, MLI, ANTENNA])  # -z nadir face is antennas
    cx = 2.8 / 2 + 0.3 + 8.1 / 2
    b.panel((cx, 0, 0), (1, 0, 0), (0, 1, 0), 8.1, 2.7, CELLS, PANEL_BACK,
            "array", gimbal=GIMBAL_1AXIS, gimbal_axis=(1, 0, 0))
    b.meta(
        family_id="starlink_v15",
        sources=(
            "Jonathan McDowell, public FCC Gen2 dimensions table: v1.5 bus "
            "2.8×1.3 m, array 2.8×8.1 m, mass 303 kg "
            "(https://planet4589.org/astro/starsim/index.html).",
            "Spaceflight Now 2023-02-26: single ~11 m end-to-end wing on v1.5.",
            "SpaceX public: krypton/argon ion (Hall-class) propulsion; "
            "body-frame thrust pointing is not published — left unknown.",
        ),
        notes="v1.0 lumped with v1.5 (bus width publicly quoted 1.3–1.4 m). "
              "Nadir face is the antenna panel. Do not treat as Starshield CAD.",
        dimension_status={"bus": STATUS_PUBLIC, "array": STATUS_PUBLIC,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_EP,
        thrust_notes="Searched SpaceX Gen2 PDF, FCC dimension table, "
                     "SpaceNews/Spaceflight Now (krypton Hall exists; "
                     "magnitude only). Everyday Astronaut claims ram-facing "
                     "but is not a primary SpaceX/FCC citation — vector left "
                     "empty.",
    )
    return b.build()


def starlink_v2mini(dtc: bool = False) -> FacetModel:
    """Starlink v2 mini: bus ~4.1 x 2.7 m, two ~12.8 x 4.1 m arrays fore/aft on
    shoulder + wrist gimbals — 2-axis (full) sun tracking. The DTC variant adds
    a ~2.0 x 2.3 m direct-to-cell antenna panel deployed off the -x end of the
    bus (about half a bus length), nadir-facing."""
    b = _Builder("starlink_v2mini_dtc" if dtc else "starlink_v2mini")
    b.box((0, 0, 0), (4.1, 2.7, 0.2), MLI, "bus",
          mats=[MLI, MLI, MLI, MLI, MLI, ANTENNA])
    for sign, tag in [(1, "fore"), (-1, "aft")]:
        cx = sign * (4.1 / 2 + 0.4 + 12.8 / 2)
        b.panel((cx, 0, 0), (sign, 0, 0), (0, sign, 0), 12.8, 4.1, CELLS,
                PANEL_BACK, f"array {tag}", gimbal=GIMBAL_2AXIS)
    if dtc:
        cx = -(4.1 / 2 + 0.2 + 2.0 / 2)
        b.panel((cx, 0, -0.4), (0, 1, 0), (1, 0, 0), 2.3, 2.0, ANTENNA, MLI,
                "dtc antenna")  # front normal -z (nadir)
    b.meta(
        family_id="starlink_v2mini_dtc" if dtc else "starlink_v2mini",
        sources=(
            "Jonathan McDowell / SpaceX FCC Gen2: v2 Mini bus 4.1×2.7 m, "
            "each array 4.1×12.8 m, mass ~800 kg "
            "(https://planet4589.org/astro/starsim/index.html).",
            "Spaceflight Now 2023-02-26: two wings, ~30 m tip-to-tip, "
            "116 m² class surface area.",
            "Mallama et al. arXiv:2306.06657 (photometric characterization).",
            "Celestrak SATCAT tags some vehicles [DTC]; DTC panel size is an "
            "observer-scale stand-in, not a SpaceX drawing.",
            "SpaceX public: argon/krypton ion propulsion; thrust pointing "
            "unpublished — left unknown. No Starshield internals.",
        ),
        notes="Two 2-axis arrays. DTC is a nadir-facing deployable only when "
              "dtc=True. Dimensions are the public FCC table, not internals.",
        dimension_status={
            "bus": STATUS_PUBLIC, "arrays": STATUS_PUBLIC,
            "thrust_vector": STATUS_UNKNOWN,
            **({"dtc_antenna": STATUS_RANGE} if dtc else {}),
        },
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_EP,
        thrust_notes="Searched SpaceX Gen2 PDF, FCC table, SpaceNews "
                     "(argon Hall 170 mN / 4.2 kW). No primary body-frame "
                     "vs ram/nadir citation — vector left empty.",
    )
    return b.build()


def bluewalker3() -> FacetModel:
    """AST BlueWalker 3: ~64 m^2 (8 x 8 m) fixed phased array with the small
    bus centered behind it; the array does not articulate. Nadir face is the
    antenna aperture, zenith face carries the solar cells."""
    b = _Builder("bluewalker3")
    b.panel((0, 0, 0), (0, 1, 0), (1, 0, 0), 8.0, 8.0, ANTENNA, CELLS,
            "phased array")  # front normal -z (nadir antenna), back +z cells
    b.box((0, 0, 0.9), (1.5, 1.5, 1.5), MLI, "bus")
    b.meta(
        family_id="bluewalker3",
        sources=(
            "AST SpaceMobile public: BlueWalker 3 ~64 m² (8×8 m) unfurled "
            "phased array (company/press kit, 2022–2023).",
            "Mallama & Cole, brightness of BlueWalker 3.",
        ),
        notes="Array is GIMBAL_FIXED (BlueWalker-class). Bus size is a coarse "
              "stand-in behind the sheet. Thrust pointing unpublished.",
        dimension_status={"array": STATUS_PUBLIC, "bus": STATUS_RANGE,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_UNKNOWN,
        thrust_notes="Propulsion type and body-frame pointing not published.",
    )
    return b.build()


def hubble() -> FacetModel:
    """Hubble: 13.2 m x 4.2 m MLI-silver cylinder (optical axis +x), two
    ~7.1 x 2.6 m rigid arrays on 1-axis boom gimbals (+/-y booms)."""
    b = _Builder("hubble")
    b.prism((0, 0, 0), (1, 0, 0), 13.2, 4.2, 12, MLI_SILVER, (DARK, MLI),
            "tube")
    for sign, tag in [(1, "+y"), (-1, "-y")]:
        cy = sign * (2.1 + 0.4 + 7.1 / 2)
        b.panel((0, cy, 0), (1, 0, 0), (0, sign, 0), 2.6, 7.1, CELLS,
                PANEL_BACK, f"array {tag}", gimbal=GIMBAL_1AXIS,
                gimbal_axis=(0, 1, 0))
    b.meta(
        family_id="hubble",
        sources=(
            "NASA HST fact sheets: OTA tube 13.2 m × 4.2 m diameter.",
            "STIS/NICMOS-era rigid solar arrays ~7.1×2.6 m on ±y booms "
            "(NASA drawings / public photography).",
        ),
        notes="No orbit-raising propulsion in the current mission; RCS only. "
              "Optical axis is +x (not LVLH ram when in science pointing).",
        dimension_status={"tube": STATUS_PUBLIC, "arrays": STATUS_PUBLIC,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_UNKNOWN, thrust_propulsion=PROP_NONE,
        thrust_notes="No operational Δv thruster; RCS only. Vector omitted.",
    )
    return b.build()


def katalyst_link() -> FacetModel:
    """Katalyst Space Technologies LINK (Swift reboost mission, launched
    2026-07-03): ~425 kg servicer, bus ~"large mini-fridge" (~0.9 x 0.9 x
    1.5 m), two solar array wings spanning ~6 m tip-to-tip (~4 kW, so
    ~8 m^2 of cells), robotic capture mechanism on the +x (approach) face.

    As of late July 2026 the spacecraft has two of three reaction wheels
    failed and is in a multi-axis spin — the 'tumble' scenario approximates
    that state with a principal-axis spin about the array boom axis.
    """
    b = _Builder("katalyst_link")
    b.box((0, 0, 0), (0.9, 0.9, 1.5), MLI, "bus",
          mats=[ANTENNA, MLI, MLI, MLI, MLI, DARK])  # +x capture face
    for sign, tag in [(1, "+y"), (-1, "-y")]:
        cy = sign * (0.45 + 0.35 + 2.2 / 2)
        b.panel((0, cy, 0), (1, 0, 0), (0, sign, 0), 1.8, 2.2, CELLS,
                PANEL_BACK, f"array {tag}", gimbal=GIMBAL_1AXIS,
                gimbal_axis=(0, 1, 0))
    b.meta(
        family_id="katalyst_link",
        sources=(
            "Katalyst Space Technologies LINK / Swift reboost public notes "
            "(launch 2026-07-03): ~425 kg servicer, bus described as a "
            "large mini-fridge, ~4 kW / ~6 m tip-to-tip arrays.",
            "SATCAT name LINK, COSPAR 2026-152A.",
        ),
        notes="Capture mechanism on +x. Array area from public kW-class "
              "power, not a drawing. Thrust vector not published.",
        dimension_status={"bus": STATUS_RANGE, "arrays": STATUS_RANGE,
                          "thrust_vector": STATUS_UNKNOWN},
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_UNKNOWN,
        thrust_notes="Servicer propulsion exists operationally; body-frame "
                     "vector is not in the public record.",
    )
    return b.build()


def iss() -> FacetModel:
    """ISS, coarse: module stack (~50 m, along x), main truss (~100 m, along
    y), two solar array groups (~840 m^2 each) on 2-axis alpha/beta gimbals,
    two white radiator groups on 1-axis gimbals."""
    b = _Builder("iss")
    b.box((0, 0, 0), (50, 6, 6), WHITE_PAINT, "modules")
    b.box((0, 0, 4.5), (5, 100, 3), MLI, "truss")
    for sign, tag in [(1, "stbd"), (-1, "port")]:
        b.panel((0, sign * 40, 4.5), (sign, 0, 0), (0, sign, 0), 35, 24,
                CELLS, PANEL_BACK, f"arrays {tag}", gimbal=GIMBAL_2AXIS)
        b.panel((0, sign * 14, 0), (0, sign, 0), (1, 0, 0), 22, 12,
                WHITE_PAINT, WHITE_PAINT, f"radiators {tag}",
                gimbal=GIMBAL_1AXIS, gimbal_axis=(0, 1, 0))
    b.meta(
        family_id="iss",
        sources=(
            "NASA ISS reference: module stack ~50 m, truss ~100 m, main "
            "solar arrays ~2500 m² class (here two 35×24 m groups), "
            "radiators on 1-axis gimbals, arrays on alpha/beta 2-axis.",
            "Reboost: Progress / Zvezda along +x (velocity) in LVLH.",
        ),
        notes="Coarse photometric stand-in, not station CAD.",
        dimension_status={"modules": STATUS_RANGE, "truss": STATUS_RANGE,
                          "arrays": STATUS_RANGE, "radiators": STATUS_RANGE,
                          "thrust_vector": STATUS_PUBLIC},
        thrust_body=[[1.0, 0.0, 0.0]],
        thrust_attitude=ATT_LVLH, thrust_propulsion=PROP_CHEMICAL,
        thrust_notes="ISS reboost is publicly along-track (+x in this frame).",
    )
    return b.build()


LIBRARY = {
    "box_wing": box_wing,
    "rocket_body": rocket_body,
    "starlink_v15": starlink_v15,
    "starlink_v2mini": starlink_v2mini,
    "starlink_v2mini_dtc": lambda: starlink_v2mini(dtc=True),
    "bluewalker3": bluewalker3,
    "hubble": hubble,
    "iss": iss,
    "katalyst_link": katalyst_link,
}
