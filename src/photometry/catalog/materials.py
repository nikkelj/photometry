"""Material classes used by catalog families.

Optical (rho_d, k_s, n_ph) are the same Lambert+Phong presets the forward
model already uses. They are *assumed photometric stand-ins*, not
spacecraft-measured BRDFs, except where a family docstring says otherwise.

IR solar absorptance α and IR emittance ε are **unknown at the spacecraft
level** unless a public number exists for that vehicle. Handbook values
below are material-class literature, recorded here so a reviewer can see
what we refused to copy onto facets as fake precision. Facet `alpha_ir` /
`epsilon_ir` stay NaN with `ir_provenance="unknown"` unless a family
explicitly attaches a published spacecraft number (none currently do).
"""

from ..shapes import (
    ANTENNA,
    CELLS,
    DARK,
    MLI,
    MLI_SILVER,
    PANEL_BACK,
    WHITE_PAINT,
)

# Handbook / NASA coating class figures (not copied onto facets).
IR_HANDBOOK = {
    "WHITE_PAINT": dict(alpha=0.17, epsilon=0.92, note="Z-93 / AZ-93 class"),
    "CELLS": dict(alpha=0.92, epsilon=0.85, note="typical GaAs coverglass"),
    "MLI": dict(alpha=None, epsilon=None,
                note="outer layer varies (Kapton, silver Teflon, OSR); unknown"),
    "MLI_SILVER": dict(alpha=None, epsilon=None, note="unknown"),
    "PANEL_BACK": dict(alpha=None, epsilon=None, note="unknown"),
    "ANTENNA": dict(alpha=None, epsilon=None, note="unknown"),
    "DARK": dict(alpha=None, epsilon=None, note="unknown"),
}

OPTICAL_PRESETS = {
    "MLI": MLI,
    "MLI_SILVER": MLI_SILVER,
    "CELLS": CELLS,
    "PANEL_BACK": PANEL_BACK,
    "WHITE_PAINT": WHITE_PAINT,
    "ANTENNA": ANTENNA,
    "DARK": DARK,
}

OPTICAL_PROVENANCE = "assumed"  # every preset until a family overrides
