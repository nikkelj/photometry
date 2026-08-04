"""Recompute the EGI for completed fleet scenarios (e.g. after solver changes).

Reuses each scenario's saved observations and best-hypothesis attitude —
no re-simulation, no re-classification.

Usage: python scripts/refresh_egi.py [scenario ...]   (default: all found)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from photometry import scenarios as sc
from photometry.attitude import PrincipalAxisSpin
from photometry.inversion.egi import solve_egi
from photometry.measurements import ObservationSet


def refresh(name: str, fleet: Path) -> None:
    d = fleet / name
    result = json.loads((d / "result.json").read_text())
    obs = ObservationSet.from_npz(d / "observations.npz")
    with np.load(d / "inversion.npz") as z:
        inv = {k: z[k] for k in z.files}

    orbit = sc.study_orbit()
    sun = sc.sun_eci()
    mode = result["mode_best"]
    if mode in ("spin_fit", "inertial_fit"):
        ra, dec, per, ph, ax_, ay, az = inv["best_spin_params"]
        att = PrincipalAxisSpin(float(ra), float(dec), float(per), float(ph),
                                body_axis=(float(ax_), float(ay), float(az)))
    else:
        named = {"lvlh_ops": "ops", "lvlh_low_drag": "low_drag",
                 "sun_point": "sun_point"}
        att, _ = sc.make_attitude(named[mode], orbit, sun, True)

    u_sun_body = att.eci_to_body(obs.t_s, obs.sun_eci)
    u_obs_body = att.eci_to_body(obs.t_s, obs.u_obs_from_target())
    egi = solve_egi(obs, u_sun_body, u_obs_body)

    inv["egi_normals"] = egi.normals
    inv["egi_albedo_area"] = egi.albedo_area
    inv["egi_specular_area"] = egi.specular_area
    np.savez_compressed(d / "inversion.npz", **inv)
    result["egi_total_albedo_area_m2"] = float(egi.albedo_area.sum())
    result["egi_total_specular_area_m2"] = float(egi.specular_area.sum())
    result["egi_residual_rms"] = float(egi.residual_rms)
    (d / "result.json").write_text(json.dumps(result, indent=2))
    print(f"{name}: diffuse {egi.albedo_area.sum():.1f} m^2, "
          f"specular {egi.specular_area.sum():.1f} m^2, "
          f"rms {egi.residual_rms:.2f}")


def main(*names: str) -> None:
    fleet = Path("results/fleet")
    todo = list(names) if names else sorted(
        p.name for p in fleet.iterdir()
        if p.is_dir() and (p / "result.json").exists())
    for n in todo:
        refresh(n, fleet)


if __name__ == "__main__":
    main(*sys.argv[1:])
