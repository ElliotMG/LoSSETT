#!/usr/bin/env python3
"""Time a small synthetic Zarr workflow and compare nested grids."""

import argparse
import json
import resource
import tempfile
import time
from pathlib import Path

import numpy as np
import xarray as xr

from lossett.calc.compute_group_increments import main as workflow


def input_dataset(nlat, nlon):
    lat = np.linspace(-90., 90., nlat)
    lon = np.arange(nlon) * 360. / nlon
    phi, lam = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    normal = np.stack((np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)), axis=-1)
    east = np.stack((-np.sin(lam), np.cos(lam), np.zeros_like(lam)), axis=-1)
    north = np.cross(normal, east)
    heights = np.array([0., 0.01, 0.02])
    velocities = np.stack([
        np.array([0.3 + z, -0.5, 0.4]) + np.cross([0.2, -0.1, 0.6], normal)
        + 0.3 * normal[..., :1] * normal
        for z in heights
    ])
    dims = ("time", "height", "latitude", "longitude")
    return xr.Dataset(
        {
            "u": (dims, np.sum(velocities * east, axis=-1)[None]),
            "v": (dims, np.sum(velocities * north, axis=-1)[None]),
            "w": (dims, np.sum(velocities * normal, axis=-1)[None]),
        },
        coords={"time": [0], "height": heights, "latitude": lat, "longitude": lon},
    )


def run_case(root, label, nlat, nlon):
    source = root / f"{label}.nc"
    start = time.perf_counter()
    input_dataset(nlat, nlon).to_netcdf(source)
    input_write = time.perf_counter() - start
    stores = [root / f"{label}-{i}.zarr" for i in range(3)]
    shell_times = []
    for i, store in enumerate(stores):
        start = time.perf_counter()
        workflow([
            "shell", "--velocity-file", str(source), "--output", str(store), "--time-index", "0",
            "--level-index", str(i), "--level-dim", "height", "--coordinate-type", "height",
            "--coordinate-unit", "m", "--radius", str(1 + i * 0.01), "--heat-time", "0.4",
        ])
        shell_times.append(time.perf_counter() - start)
    target = root / f"{label}-vertical.zarr"
    start = time.perf_counter()
    workflow(["vertical", "--shells", *map(str, stores), "--output", str(target), "--fluid", "boussinesq"])
    vertical_time = time.perf_counter() - start
    with xr.open_zarr(target) as ds:
        result = ds.load()
    with xr.open_zarr(stores[1]) as ds:
        mass_error = float(np.max(np.abs(ds.kernel_mass.values - 1)))
        structure = ds.structure.values.copy()
    return {
        "grid": [nlat, nlon], "input_write_seconds": input_write,
        "shell_seconds": shell_times, "vertical_seconds": vertical_time,
        "max_kernel_mass_error": mass_error,
    }, result, structure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="existing directory for temporary inputs and outputs")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    with tempfile.TemporaryDirectory(prefix="lossett-benchmark-", dir=args.workspace) as directory:
        root = Path(directory)
        warmup, _, _ = run_case(root, "warmup", 9, 16)
        runs = []
        for index in range(args.repeats):
            report, coarse, coarse_structure = run_case(root, f"coarse-{index}", 9, 16)
            runs.append(report)
        fine_report, fine, fine_structure = run_case(root, "fine", 17, 32)
        errors = {}
        for name in ("horizontal", "vertical", "total"):
            a = coarse[name].values
            b = fine[name].values[:, ::2, ::2]
            errors[name] = float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
        errors["structure"] = float(
            np.max(np.abs(coarse_structure - fine_structure[::2, ::2])) / np.max(np.abs(fine_structure))
        )
        print(json.dumps({
            "warmup": warmup, "runs": runs, "fine": fine_report,
            "nested_grid_max_error_over_fine_max": errors,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }, sort_keys=True))


if __name__ == "__main__":
    main()
