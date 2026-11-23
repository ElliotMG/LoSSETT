#!/usr/bin/env python3
"""Small synthetic reference benchmarks; timings are not equivalent workloads."""

import argparse
import contextlib
import io
import json
import resource
import tempfile
import time
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.polynomial.legendre import leggauss

def group_case(nlat, nlon):
    from lossett.calc.group_increments import group_increment_grid, latlon_shell_fields

    mu, lat_weights = leggauss(nlat)
    lat = np.arcsin(mu)
    lon = np.arange(nlon) * 2 * np.pi / nlon
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    u = np.cos(lat2) + 0.2 * np.sin(lon2)
    v = 0.3 * np.cos(lon2)
    w = 0.1 * np.sin(lat2)
    points, velocity, grad, gradw = latlon_shell_fields(lat, lon, u, v, w, radius=1.0)
    weights = np.broadcast_to(lat_weights[:, None] * 2 * np.pi / nlon, lat2.shape)

    def run():
        return group_increment_grid(
            points, velocity, grad, gradw, weights, radius=1.0, heat_time=0.2
        )

    return run


def legacy_input(size, nlevels=1):
    x, y = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    shape = (1, size, size, nlevels)
    def field(values):
        return np.broadcast_to(values[None, :, :, None], shape).copy()
    return xr.Dataset(
        {
            "u": (("time", "x", "y", "pressure"), field(np.sin(2 * np.pi * x / size))),
            "v": (("time", "x", "y", "pressure"), field(np.cos(2 * np.pi * y / size))),
            "w": (("time", "x", "y", "pressure"), field(0.1 * np.sin(2 * np.pi * (x + y) / size))),
        },
        coords={"time": [0], "x": np.arange(size), "y": np.arange(size),
                "pressure": 1000 - 100 * np.arange(nlevels)},
    )


def legacy_control():
    return {
        "max_r": 4.0, "max_r_units": "m", "angle_precision": 1e-10,
        "x_coord_name": "x", "x_coord_units": "m", "x_coord_boundary": "periodic",
        "y_coord_name": "y", "y_coord_units": "m", "y_coord_boundary": "periodic",
    }


def legacy_case(size):
    from lossett.calc.calc_inter_scale_transfers import calc_inter_scale_energy_transfer_kinetic

    ds = legacy_input(size)
    control = legacy_control()

    def run():
        with contextlib.redirect_stdout(io.StringIO()):
            return calc_inter_scale_energy_transfer_kinetic(ds, control, length_scales=np.array([2.0]))

    return run


def legacy_workflow(size, repeats, workspace):
    from lossett.calc.calc_inter_scale_transfers import calc_inter_scale_energy_transfer_kinetic

    with tempfile.TemporaryDirectory(prefix="lossett-legacy-", dir=workspace) as directory:
        root = Path(directory)
        input_path = root / "velocity.nc"
        start = time.perf_counter()
        legacy_input(size, 3).to_netcdf(input_path)
        input_write = time.perf_counter() - start
        runs = []
        for i in range(repeats + 1):
            start = time.perf_counter()
            with xr.open_dataset(input_path) as ds:
                ds.load()
                read_seconds = time.perf_counter() - start
                start = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    result = calc_inter_scale_energy_transfer_kinetic(
                        ds, legacy_control(), length_scales=np.array([2.0])
                    )
                calc_seconds = time.perf_counter() - start
                start = time.perf_counter()
                output = root / f"result-{i}.zarr"
                result.to_dataset().to_zarr(output, mode="w-", zarr_version=2)
                write_seconds = time.perf_counter() - start
            start = time.perf_counter()
            with xr.open_zarr(output) as saved:
                saved.load()
                if not np.all(np.isfinite(saved.Dl_u.values)):
                    raise ValueError("legacy benchmark produced non-finite output")
            runs.append({
                "input_read_seconds": read_seconds, "calculation_seconds": calc_seconds,
                "output_write_seconds": write_seconds, "output_read_seconds": time.perf_counter() - start,
            })
        return {
            "case": "legacy-workflow", "grid": [size, size], "levels": 3,
            "input_write_seconds": input_write, "warmup": runs[0], "runs": runs[1:],
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("group", "legacy", "legacy-workflow"))
    parser.add_argument("--nlat", type=int, default=8)
    parser.add_argument("--nlon", type=int, default=16)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--workspace", type=Path, help="existing directory for temporary workflow data")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.case == "legacy-workflow":
        if args.workspace is None:
            parser.error("legacy-workflow requires --workspace")
        print(json.dumps(legacy_workflow(args.size, args.repeats, args.workspace), sort_keys=True))
        return
    run = group_case(args.nlat, args.nlon) if args.case == "group" else legacy_case(args.size)
    start = time.perf_counter()
    result = run()
    warmup = time.perf_counter() - start
    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = run()
        times.append(time.perf_counter() - start)
    report = {
        "case": args.case, "warmup_seconds": warmup, "seconds": times,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    if args.case == "group":
        report.update(
            grid=[args.nlat, args.nlon], heat_time=0.2,
            max_kernel_mass_error=float(np.max(np.abs(result["kernel_mass"] - 1))),
        )
    else:
        report.update(grid=[args.size, args.size], max_r=4.0, length_scale=2.0)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
