#!/usr/bin/env python3
"""Time selected uncached group origins on prepared global fields."""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import xarray as xr

from lossett.calc.compute_group_increments import _solid_angles
from lossett.calc.group_increments import group_increment_shell, latlon_shell_fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-files", nargs="+", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=False)
    with xr.open_zarr(args.baseline) as baseline:
        if baseline.attrs["kernel"] != "heat":
            raise ValueError("this benchmark expects a heat-kernel baseline")
        radius = baseline.attrs["radius_m"]
        heat_time = baseline.attrs["kernel_parameter"]
        pressure = baseline.attrs["coordinate_value"] / 100
    report = {"radius_m": radius, "heat_time": heat_time, "pressure_hPa": pressure, "grids": []}
    for path in args.velocity_files:
        with xr.open_dataset(path, decode_timedelta=False) as source:
            ds = source.sel(pressure=pressure).isel(time=0).load()
        lat, lon = np.deg2rad(ds.latitude.values), np.deg2rad(ds.longitude.values)
        start = time.perf_counter()
        directions, velocity, grad, gradw = latlon_shell_fields(
            lat, lon, ds.u.values, ds.v.values, ds.w.values, radius=radius,
        )
        weights = _solid_angles(lat, lon)
        field_seconds = time.perf_counter() - start
        n = len(lat) * len(lon)
        target = (directions.reshape(n, 3), velocity.reshape(n, 3), grad.reshape(n, 3, 3),
                  gradw.reshape(n, 3), weights.reshape(n))
        samples = []
        for i in (0, len(lat) // 4, len(lat) // 2, 3 * len(lat) // 4, len(lat) - 1):
            start = time.perf_counter()
            result = group_increment_shell(
                directions[i, 0], velocity[i, 0], *target, radius=radius, heat_time=heat_time,
            )
            samples.append({"latitude_degrees": float(ds.latitude.values[i]),
                            "seconds": time.perf_counter() - start,
                            "kernel_mass_error": float(abs(result["kernel_mass"] - 1))})
        report["grids"].append({
            "velocity_file": str(Path(path).resolve()), "shape": [len(lat), len(lon)],
            "field_setup_seconds": field_seconds, "origins": samples,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        })
        (args.output_dir / "timings.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["grids"][-1]), flush=True)


if __name__ == "__main__":
    main()
