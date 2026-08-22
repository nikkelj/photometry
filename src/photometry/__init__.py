"""Constellation-aggregated star tracker photometry.

Simulation of opportunistic RSO detections by a large LEO constellation's
star trackers, and inversion of the aggregated visual-magnitude stream into
target spin state and shape (extended Gaussian image).

The interface between data and inversion is `measurements.ObservationSet`;
real tracklet data can be loaded into the same schema and run through the
identical inversion code paths.
"""

from . import frames, constellation, shapes, catalog, attitude, radiometry, sensing, measurements, simulate

__all__ = [
    "frames",
    "constellation",
    "shapes",
    "catalog",
    "attitude",
    "radiometry",
    "sensing",
    "measurements",
    "simulate",
]
