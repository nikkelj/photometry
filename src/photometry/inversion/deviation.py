"""Automated catalog-deviation alerting.

Wraps matched-model refinement into a decision: does the target's
photometry deviate from its identified catalog model? The residual EGI
absorbs any brightness structure the model cannot explain, so the ratio
of residual rms before/after that absorption measures unexplained
structure; localized same-sign residual area says *where* on the normal
sphere it lives.

An alert therefore fires for either a physical deviation (missing/extra
panel, bent array, changed coating) or a misidentification — both mean
"the catalog model does not explain this object", which is exactly what
an operator wants flagged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frames import unit_to_radec
from .refine import RefinementResult


@dataclass
class DeviationAssessment:
    alert: bool
    rms_ratio: float           # residual rms before / after residual-EGI absorption
    fit_quality: float         # refined cost: ~1 = at noise floor
    peak_direction_radec: tuple[float, float]
    peak_albedo_area: float    # signed rho*A at the strongest deviation
    cluster_albedo_area: float  # same-sign area within the cluster cone
    reason: str


def assess_deviation(
    r: RefinementResult,
    rms_ratio_threshold: float = 1.15,
    fit_quality_threshold: float = 4.0,
    cluster_cone_deg: float = 25.0,
) -> DeviationAssessment:
    ratio = r.residual_rms_before / max(r.residual_rms_after, 1e-9)
    i_pk = int(np.argmax(np.abs(r.residual_albedo_area)))
    n_pk = r.residual_normals[i_pk]
    x_pk = float(r.residual_albedo_area[i_pk])
    cone = r.residual_normals @ n_pk > np.cos(np.radians(cluster_cone_deg))
    same_sign = np.sign(r.residual_albedo_area) == np.sign(x_pk)
    cluster = float(r.residual_albedo_area[cone & same_sign].sum())

    structural = ratio > rms_ratio_threshold
    bad_fit = r.cost_refined > fit_quality_threshold
    alert = structural or bad_fit
    if structural and bad_fit:
        reason = "unexplained structure and poor overall fit"
    elif structural:
        reason = "localized residual structure vs catalog model"
    elif bad_fit:
        reason = "poor fit to catalog model (possible misidentification)"
    else:
        reason = "consistent with catalog model"
    ra, dec = unit_to_radec(n_pk)
    return DeviationAssessment(
        alert=bool(alert), rms_ratio=float(ratio),
        fit_quality=float(r.cost_refined),
        peak_direction_radec=(float(ra), float(dec)),
        peak_albedo_area=x_pk, cluster_albedo_area=cluster, reason=reason)
