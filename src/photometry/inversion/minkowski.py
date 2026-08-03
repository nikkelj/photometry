"""Minkowski reconstruction: convex polyhedron from an EGI.

Photometry recovers the extended Gaussian image — oriented area with no
position information. Minkowski's theorem guarantees a unique convex body
(up to translation) whose face normals/areas match a valid EGI. We solve
the inverse problem iteratively (Little-style): parameterize the body by
support distances h_i of one plane per EGI direction, build the polytope,
measure its face areas, and drive them toward the targets using the fact
that moving a plane outward grows its face.

Sparse or hemispheric EGIs (e.g. a two-sided panel) would give an
unbounded intersection, so a coarse spherical "cage" of extra planes at
fixed radius closes the body; cage faces that survive in the final hull
are closure surfaces, flagged so renderers can draw them dimmer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import HalfspaceIntersection

from ..frames import fibonacci_sphere, unit


@dataclass
class HullFace:
    vertices: np.ndarray   # (V,3) ordered polygon in the body frame
    normal: np.ndarray
    area: float
    is_closure: bool       # True for cage faces (no EGI area target)


def _face_polygons(normals: np.ndarray, h: np.ndarray) -> list[np.ndarray | None]:
    """Vertex polygons of the polytope {x : x.n_i <= h_i}, one per plane."""
    halfspaces = np.hstack([normals, -h[:, None]])
    hs = HalfspaceIntersection(halfspaces, np.zeros(3))
    pts = hs.intersections
    scale = max(np.abs(pts).max(), 1e-9)
    polys: list[np.ndarray | None] = []
    for i, n in enumerate(normals):
        on = np.abs(pts @ n - h[i]) < 1e-7 * scale + 1e-12
        p = pts[on]
        if len(p) < 3:
            polys.append(None)
            continue
        c = p.mean(axis=0)
        helper = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
        e1 = unit(np.cross(n, helper))
        e2 = np.cross(n, e1)
        ang = np.arctan2((p - c) @ e2, (p - c) @ e1)
        polys.append(p[np.argsort(ang)])
    return polys


def _poly_area(p: np.ndarray, n: np.ndarray) -> float:
    v = p - p.mean(axis=0)
    cross = np.cross(v, np.roll(v, -1, axis=0)).sum(axis=0)
    return float(abs(cross @ n) / 2)


def reconstruct_hull_variational(
    normals: np.ndarray,
    areas: np.ndarray,
    n_cage: int = 42,
    cage_area_frac: float = 0.02,
    max_iter: int = 400,
) -> list[HullFace]:
    """Variational Minkowski solve: minimize the support functional.

    Minimizing G(h) = (a . h) / V(h)^(1/3) over support distances h > 0
    drives the polytope's face areas proportional to the targets a — the
    classical variational formulation of Minkowski's problem, so slab-like
    EGIs converge to the right proportions where the fixed-point iteration
    stalls (a plate face's area is controlled by the *other* planes, which
    the per-face update cannot reach; the joint gradient can).

    Gradient is analytic: dV/dh_i = A_i(h) (face area), V = (h . A)/3.
    Cage directions carry a small area budget so one-hemisphere EGIs stay
    bounded. A final uniform scale matches total EGI face area to target.
    """
    from scipy.optimize import minimize

    normals = unit(np.asarray(normals, dtype=float))
    areas = np.asarray(areas, dtype=float)
    f = len(normals)
    cage = fibonacci_sphere(n_cage)
    all_n = np.vstack([normals, cage])
    a = np.concatenate([areas,
                        np.full(n_cage, cage_area_frac * areas.sum() / n_cage)])
    side = np.sqrt(areas.max())
    h_min = 0.02 * side

    def face_areas(h: np.ndarray) -> np.ndarray:
        polys = _face_polygons(all_n, h)
        return np.array([
            _poly_area(p, n) if p is not None else 0.0
            for p, n in zip(polys, all_n)])

    def fg(h: np.ndarray):
        area_now = face_areas(h)
        v = float(h @ area_now) / 3.0
        if v <= 0:
            return 1e9, np.zeros_like(h)
        s = float(a @ h)
        g = s / v ** (1 / 3)
        grad = a / v ** (1 / 3) - (s / (3 * v ** (4 / 3))) * area_now
        return g, grad

    h0 = np.full(len(all_n), 0.5 * side)
    res = minimize(fg, h0, jac=True, method="L-BFGS-B",
                   bounds=[(h_min, None)] * len(all_n),
                   options=dict(maxiter=max_iter, ftol=1e-12))
    h = res.x
    # uniform rescale: areas scale as h^2
    achieved = face_areas(h)[:f].sum()
    if achieved > 0:
        h = h * np.sqrt(areas.sum() / achieved)

    polys = _face_polygons(all_n, h)
    faces = []
    for i, p in enumerate(polys):
        if p is None:
            continue
        area_i = _poly_area(p, all_n[i])
        if area_i < 1e-6 * side**2:
            continue
        faces.append(HullFace(vertices=p, normal=all_n[i], area=area_i,
                              is_closure=i >= f))
    return _recentered(faces)


def _recentered(faces: list[HullFace]) -> list[HullFace]:
    if faces:
        centroid = (sum(f.area * f.vertices.mean(axis=0) for f in faces)
                    / sum(f.area for f in faces))
        for face in faces:
            face.vertices = face.vertices - centroid
    return faces


def reconstruct_hull(
    normals: np.ndarray,
    areas: np.ndarray,
    n_iter: int = 200,
    gamma: float = 0.25,
    n_cage: int = 122,
    method: str = "variational",
) -> list[HullFace]:
    """Convex body matching (normals, target face areas) as well as possible.

    normals: (F,3) EGI directions with significant recovered area
    areas:   (F,) target physical face areas (m^2)
    method "variational" (default) falls back to the fixed-point iteration
    on failure; "fixed_point" forces the legacy solver.
    """
    if method == "variational":
        try:
            faces = reconstruct_hull_variational(normals, areas)
            if any(not f.is_closure for f in faces):
                return faces
        except Exception:
            pass
    return _reconstruct_hull_fixed_point(normals, areas, n_iter, gamma, n_cage)


def _reconstruct_hull_fixed_point(
    normals: np.ndarray,
    areas: np.ndarray,
    n_iter: int = 200,
    gamma: float = 0.25,
    n_cage: int = 122,
) -> list[HullFace]:
    """Legacy per-face fixed-point iteration (kept as fallback)."""
    normals = unit(np.asarray(normals, dtype=float))
    areas = np.asarray(areas, dtype=float)
    side = np.sqrt(areas.max())
    cage_r = 1.6 * side
    cage = fibonacci_sphere(n_cage)

    all_n = np.vstack([normals, cage])
    f = len(normals)
    h = np.concatenate([np.full(f, 0.3 * side), np.full(n_cage, cage_r)])
    # floor keeps weak oblique faces from slicing deep into the body core
    # (an unbalanced EGI would otherwise carve wedges out of dominant slabs);
    # hard cap bounds the runaway when a degenerate EGI (e.g. one lit
    # hemisphere) makes the area target unreachable
    h_floor, h_cap = 0.08 * side, 0.99 * cage_r
    h_max = 3.0 * cage_r

    for _ in range(n_iter):
        try:
            polys = _face_polygons(all_n, h)
        except Exception:
            h[:f] = np.minimum(h[:f] * 1.05, h_cap)
            continue
        a_act = np.array([
            _poly_area(polys[i], all_n[i]) if polys[i] is not None else 0.0
            for i in range(f)])
        for i in range(f):
            if a_act[i] <= 0:
                h[i] *= 0.92  # plane fell outside the body: pull it inward
            else:
                # face too small -> push the plane outward (cone growth)
                h[i] *= (areas[i] / a_act[i]) ** gamma
        # prism-like faces (a slab's broad sides) are sized by the *other*
        # planes, which per-face updates cannot reach — a global rescale
        # (cage included) drives total area toward the target
        total = a_act.sum()
        if total > 0:
            # grow-only: shrink would let floor-pinned weak faces (whose
            # areas overshoot) drag the whole body into collapse
            s = float((areas.sum() / total) ** (gamma / 2))
            if s > 1:
                h = np.minimum(h * min(s, 1.1), h_max)
                h_cap = min(max(h_cap, h[f:].max()), h_max)
        h[:f] = np.clip(h[:f], h_floor, h_cap)

    polys = _face_polygons(all_n, h)
    faces = []
    for i, p in enumerate(polys):
        if p is None:
            continue
        a = _poly_area(p, all_n[i])
        if a < 1e-6 * side**2:
            continue
        faces.append(HullFace(vertices=p, normal=all_n[i], area=a,
                              is_closure=i >= f))
    # the EGI carries no position information and the support-vector solve
    # is translation-arbitrary — recenter on the area-weighted centroid so
    # renders compare against an origin-centered truth
    if faces:
        centroid = (sum(f.area * f.vertices.mean(axis=0) for f in faces)
                    / sum(f.area for f in faces))
        for face in faces:
            face.vertices = face.vertices - centroid
    return faces


def hull_from_egi(
    egi_normals: np.ndarray,
    albedo_area: np.ndarray,
    specular_area: np.ndarray,
    nominal_albedo: float = 0.35,
    top_n: int = 24,
    min_frac: float = 0.03,
) -> list[HullFace]:
    """Select significant EGI directions and reconstruct the convex body.

    Physical face areas are estimated as recovered (diffuse + specular)
    coefficient-area divided by a nominal albedo — the absolute scale is
    only as good as that guess; the geometry is what the EGI constrains.
    """
    w = albedo_area + specular_area
    order = np.argsort(w)[::-1]
    keep = [i for i in order[:top_n] if w[i] > min_frac * w.max() and w[i] > 1e-3]
    return reconstruct_hull(egi_normals[keep], w[keep] / nominal_albedo)
