"""Facet-based target shape models with per-facet reflectance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import unit


@dataclass
class FacetModel:
    """Convex facet model in the target body frame.

    normals: (F,3) outward unit normals
    areas:   (F,) m^2
    rho_d:   (F,) Lambertian (diffuse) albedo
    k_s:     (F,) specular reflectance coefficient
    n_ph:    (F,) Phong specular exponent (lobe sharpness)
    labels:  facet names for reporting
    """

    normals: np.ndarray
    areas: np.ndarray
    rho_d: np.ndarray
    k_s: np.ndarray
    n_ph: np.ndarray
    labels: list[str]

    @property
    def n_facets(self) -> int:
        return len(self.areas)

    def diffuse_albedo_area(self) -> np.ndarray:
        """Per-facet rho_d * A — the quantity an EGI inversion recovers."""
        return self.rho_d * self.areas


def box_wing(
    box_dims_m: tuple[float, float, float] = (1.2, 1.0, 2.0),
    panel_area_m2: float = 4.0,
    panel_normal: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> FacetModel:
    """Defunct box-wing satellite: 6-face bus (MLI/paint) + 2-sided solar panel.

    The panel front (cells, toward +panel_normal) is dark but strongly
    specular; the back is painted and diffuse. Bus faces are moderately
    diffuse with a weak specular component (MLI glint).
    """
    lx, ly, lz = box_dims_m
    face_axes = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float
    )
    face_areas = np.array([ly * lz, ly * lz, lx * lz, lx * lz, lx * ly, lx * ly])

    pn = unit(np.array(panel_normal, dtype=float))
    normals = np.vstack([face_axes, pn, -pn])
    areas = np.concatenate([face_areas, [panel_area_m2, panel_area_m2]])
    rho_d = np.concatenate([np.full(6, 0.30), [0.06, 0.35]])
    k_s = np.concatenate([np.full(6, 0.05), [0.55, 0.02]])
    n_ph = np.concatenate([np.full(6, 20.0), [800.0, 10.0]])
    labels = ["+X bus", "-X bus", "+Y bus", "-Y bus", "+Z bus", "-Z bus",
              "panel front", "panel back"]
    return FacetModel(normals, areas, rho_d, k_s, n_ph, labels)


def rocket_body(
    length_m: float = 8.0, diameter_m: float = 2.4, n_side_facets: int = 16
) -> FacetModel:
    """Cylindrical upper stage approximated by side facets + end caps.

    Provided for library/model-matching scaffolding; not used by the
    baseline simulation.
    """
    ang = 2 * np.pi * np.arange(n_side_facets) / n_side_facets
    side_normals = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=-1)
    side_area = np.pi * diameter_m * length_m / n_side_facets
    normals = np.vstack([side_normals, [[0, 0, 1]], [[0, 0, -1]]])
    cap = np.pi * (diameter_m / 2) ** 2
    areas = np.concatenate([np.full(n_side_facets, side_area), [cap, cap]])
    f = normals.shape[0]
    return FacetModel(
        normals,
        areas,
        rho_d=np.full(f, 0.35),
        k_s=np.full(f, 0.15),
        n_ph=np.full(f, 60.0),
        labels=[f"side{i}" for i in range(n_side_facets)] + ["+Z cap", "-Z cap"],
    )
