"""Shared dark-theme palette and matplotlib style for every chart script.

Every make_*.py script used to carry its own copy of this block. One
palette, one style(): scripts do `from photometry.charts import *`.
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "SURFACE", "PAGE", "INK", "INK_2", "MUTED", "GRID", "BASELINE",
    "S1_BLUE", "S2_ORANGE", "S3_AQUA", "S4_MAGENTA", "S5_YELLOW",
    "SEQ_STEPS", "SEQ_CMAP", "SEQ", "style",
]

SURFACE = "#1a1a19"
PAGE = "#0d0d0d"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
BASELINE = "#383835"
S1_BLUE = "#3987e5"
S2_ORANGE = "#d95926"
S3_AQUA = "#1baf7a"
S4_MAGENTA = "#d55181"
S5_YELLOW = "#c98500"
SEQ_STEPS = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf",
             "#2a78d6", "#3987e5", "#5598e7", "#86b6ef", "#cde2fb"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue_dark", SEQ_STEPS)
SEQ = SEQ_CMAP


def style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.labelcolor": INK_2, "axes.edgecolor": BASELINE,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.labelcolor": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
    })
