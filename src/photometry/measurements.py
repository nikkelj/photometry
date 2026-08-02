"""Measurement schema shared by simulation and (future) real data.

`ObservationSet` is the interface the inversion code consumes. Real star
tracker tracklets should be reduced to this same schema — one row per
calibrated detection — after photometric calibration and orbit determination
(which supplies range). Nothing downstream knows whether the rows came from
the simulator or from the fleet.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields

import numpy as np

from .radiometry import mag_to_normalized_brightness

_VECTOR_FIELDS = {"obs_pos_km", "los_eci", "sun_eci"}


@dataclass
class ObservationSet:
    """K calibrated photometric detections of a single associated target.

    t_s:         (K,) time since epoch, seconds
    obs_id:      (K,) observing satellite index
    tracker_id:  (K,) star tracker index on that satellite (0..2)
    obs_pos_km:  (K,3) observer ECI position
    los_eci:     (K,3) unit line-of-sight observer -> target (the "angles")
    sun_eci:     (K,3) unit sun direction (ECI)
    range_km:    (K,) observer -> target range (from multi-observer OD)
    mag:         (K,) calibrated apparent visual magnitude
    mag_sigma:   (K,) 1-sigma magnitude uncertainty
    sensor_bias: (K,) per-sensor residual zero-point bias estimate (mag);
                 zero in simulation, populated by calibration on real data
    """

    t_s: np.ndarray
    obs_id: np.ndarray
    tracker_id: np.ndarray
    obs_pos_km: np.ndarray
    los_eci: np.ndarray
    sun_eci: np.ndarray
    range_km: np.ndarray
    mag: np.ndarray
    mag_sigma: np.ndarray
    sensor_bias: np.ndarray

    def __len__(self) -> int:
        return len(self.t_s)

    # --- derived quantities -------------------------------------------------

    def u_obs_from_target(self) -> np.ndarray:
        """Unit vector target -> observer (ECI), the direction light leaves toward."""
        return -self.los_eci

    def phase_angle_deg(self) -> np.ndarray:
        """Sun - target - observer angle."""
        cosp = np.sum(self.sun_eci * self.u_obs_from_target(), axis=-1)
        return np.degrees(np.arccos(np.clip(cosp, -1, 1)))

    def normalized_brightness(self) -> np.ndarray:
        """Range-normalized brightness (m^2), bias-corrected."""
        return mag_to_normalized_brightness(self.mag - self.sensor_bias, self.range_km)

    def subset(self, idx: np.ndarray) -> "ObservationSet":
        return ObservationSet(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    # --- I/O ----------------------------------------------------------------

    def to_npz(self, path: str) -> None:
        np.savez_compressed(path, **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def from_npz(cls, path: str) -> "ObservationSet":
        with np.load(path) as z:
            return cls(**{f.name: z[f.name] for f in fields(cls)})

    def to_csv(self, path: str) -> None:
        cols: list[tuple[str, np.ndarray]] = []
        for f in fields(self):
            arr = getattr(self, f.name)
            if f.name in _VECTOR_FIELDS:
                for j, suffix in enumerate("xyz"):
                    cols.append((f"{f.name}_{suffix}", arr[:, j]))
            else:
                cols.append((f.name, arr))
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([name for name, _ in cols])
            for i in range(len(self)):
                w.writerow([arr[i] for _, arr in cols])

    @classmethod
    def from_csv(cls, path: str) -> "ObservationSet":
        data = np.genfromtxt(path, delimiter=",", names=True)
        kwargs = {}
        for f in fields(cls):
            if f.name in _VECTOR_FIELDS:
                kwargs[f.name] = np.stack(
                    [data[f"{f.name}_{s}"] for s in "xyz"], axis=-1
                )
            else:
                arr = data[f.name]
                if f.name in ("obs_id", "tracker_id"):
                    arr = arr.astype(int)
                kwargs[f.name] = arr
        return cls(**kwargs)
