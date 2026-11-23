#!/usr/bin/env python3
"""Compare previous spherical and cached SO(3) calculations on one shell."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

from lossett.calc.compute_delta_u_cubed_spherical import (
    load_geometry,
    load_geometry_chunk,
    pack_active_geometry,
    process_origin_longitudes,
)
from lossett.calc.compute_group_increments import shell_dataset
from lossett.calc.compute_inter_scale_transfers_spherical import compute_inter_scale_kinetic_energy_transfer
from lossett.calc.compute_spherical_geometry import GRID_DEFS, RADIUS_EARTH, build_regular_latlon_grid


def legacy_shell(field, geometry, distances, edges, chunks, *, max_radius, weighted, full_sphere=False):
    """Run the existing spherical increment loop and scale integration in memory."""
    increment_start = time.perf_counter()
    geometry_seconds = 0.
    batches = []
    for chunk in chunks:
        start = time.perf_counter()
        geom, active = load_geometry_chunk(geometry, chunk, edges, max_R=None if full_sphere else max_radius)
        packed = None if active is None else pack_active_geometry(geom, active, use_angular_weights=weighted)
        geometry_seconds += time.perf_counter() - start
        columns = [
            process_origin_longitudes(
                np.array([longitude]), field.u, field.v, geom, active, distances,
                active_geom=packed, dtype=np.float64, use_angular_weights=weighted, w=field.w,
            )
            for longitude in field.longitude.values
        ]
        batches.append(xr.concat(columns, dim="origin_longitude"))
    increments = xr.concat(batches, dim="origin_latitude").rename({"great_circle_distance": "r"})
    increment_seconds = time.perf_counter() - increment_start
    start = time.perf_counter()
    result = compute_inter_scale_kinetic_energy_transfer(
        increments.delta_u_cubed_angular_integral_longitudinal,
        np.array([max_radius / 2]),
        du3_vert=increments.delta_u_cubed_angular_integral_vertical,
    ).load()
    return result, {
        "increment_seconds_including_geometry_read": increment_seconds,
        "geometry_read_and_selection_seconds": geometry_seconds,
        "scale_integration_seconds": time.perf_counter() - start,
    }


def validity(dataset):
    report = {}
    for name in dataset.data_vars:
        value = dataset[name]
        missing = ~np.isfinite(value)
        other_dims = [dim for dim in value.dims if dim != "latitude"]
        report[name] = {
            "nonfinite_count": int(missing.sum().item()),
            "total_count": value.size,
            "affected_latitudes_degrees": value.latitude.where(missing.any(dim=other_dims), drop=True).values.tolist(),
        }
    return report


def check_reference(actual, path, names):
    with xr.open_zarr(path) as source:
        reference = source.load()
    errors = {}
    for name in names:
        value, expected = xr.align(actual[name], reference[name], join="exact")
        value = value.transpose(*expected.dims)
        a, b = value.values, expected.values
        np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b))
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True)
        finite = np.isfinite(b)
        if not np.any(finite):
            raise ValueError(f"{name} has no finite reference values")
        errors[name] = float(np.max(np.abs(a[finite] - b[finite])))
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-file", required=True)
    parser.add_argument("--geometry-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid", choices=GRID_DEFS, default="5p0deg")
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--pressure", type=float, default=200.)
    parser.add_argument("--physical-scale-km", type=float, default=4000.)
    parser.add_argument("--radius", type=float, required=True)
    kernel = parser.add_mutually_exclusive_group(required=True)
    kernel.add_argument("--heat-time", type=float)
    kernel.add_argument("--mollifier-scale", type=float)
    parser.add_argument("--compact-support", action="store_true")
    parser.add_argument("--pair-backend", choices=("numpy", "numba"), default="numpy")
    parser.add_argument("--legacy-coverage", choices=("cap", "global", "both"), default="cap")
    parser.add_argument("--origin-lat-chunksize", type=int, default=4)
    parser.add_argument("--use-angular-weights", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=3000,
                        help="safety limit for this all-origin reference benchmark")
    parser.add_argument("--legacy-reference", help="previous spherical integrated Zarr result to verify")
    parser.add_argument("--group-reference", help="uncached SO(3) shell Zarr result to verify")
    parser.add_argument("--cartesian-src", help="original Cartesian checkout's src directory")
    args = parser.parse_args(argv)
    if (args.repeats < 1 or args.max_points < 1 or args.origin_lat_chunksize < 1
            or not np.isfinite(args.physical_scale_km) or not 0 < args.physical_scale_km <= 15000):
        parser.error("repeats, max-points and chunksize must be positive; scale must be in (0, 15000] km")
    args.output_dir.mkdir(exist_ok=False)
    start = time.perf_counter()
    with xr.open_dataset(args.velocity_file, decode_timedelta=False) as source:
        if args.cartesian_src and (args.time_index != 0 or args.pressure != 200. or source.sizes["time"] != 1):
            raise ValueError("the Cartesian benchmark requires one input time, time index 0 and 200 hPa")
        ds = source.isel(time=[args.time_index]).sel(pressure=[args.pressure]).load()
    input_seconds = time.perf_counter() - start
    field = ds.isel(time=0, pressure=0)
    lon, lat = build_regular_latlon_grid(*GRID_DEFS[args.grid])
    if not np.array_equal(field.latitude.values, lat) or not np.array_equal(field.longitude.values, lon):
        raise ValueError("input coordinates must exactly match the specified legacy grid")
    if len(lat) * len(lon) > args.max_points:
        raise ValueError("grid exceeds --max-points; explicitly raise the limit after checking resources")
    start = time.perf_counter()
    geometry, distances, edges, chunks = load_geometry(
        args.geometry_dir, args.grid, args.origin_lat_chunksize, len(lat), len(lon),
    )
    geometry_open_seconds = time.perf_counter() - start
    report = {
        "velocity_file": str(Path(args.velocity_file).resolve()), "grid": args.grid,
        "shape": [len(lat), len(lon)], "pressure_hPa": args.pressure,
        "time_index": args.time_index, "physical_scale_km": args.physical_scale_km,
        "radius_m": args.radius, "heat_time": args.heat_time, "mollifier_scale": args.mollifier_scale,
        "compact_support": args.compact_support, "pair_backend": args.pair_backend,
        "legacy_coverage": args.legacy_coverage,
        "legacy_angular_weights": args.use_angular_weights,
        "group_atmosphere": "deep", "legacy_radius_m": RADIUS_EARTH,
        "legacy_geometry_dir": str(Path(args.geometry_dir).resolve()),
        "legacy_geometry_dtype": str(geometry.sine_initial_bearing.dtype),
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "input_read_seconds": input_seconds, "legacy_geometry_archive_open_seconds": geometry_open_seconds,
        "timing_scope": "in-memory shell computation; legacy geometry chunk reads included; no output I/O",
        "workload_note": "different kernels, diagnostics and output quantities; not an equivalent-work speedup",
        "runs": [],
    }
    old = old_global = group = None
    try:
        for iteration in range(args.repeats + 1):
            global_timing = None
            old, old_timing = legacy_shell(
                field, geometry, distances, edges, chunks,
                max_radius=args.physical_scale_km * 1000, weighted=args.use_angular_weights,
                full_sphere=args.legacy_coverage == "global",
            )
            if args.legacy_coverage == "both":
                old_global, global_timing = legacy_shell(
                    field, geometry, distances, edges, chunks,
                    max_radius=args.physical_scale_km * 1000, weighted=args.use_angular_weights,
                    full_sphere=True,
                )
            start = time.perf_counter()
            group = shell_dataset(
                ds, time_index=0, level_index=0, level_dim="pressure", coordinate_type="pressure",
                coordinate_unit="hPa", radius=args.radius, heat_time=args.heat_time,
                mollifier_scale=args.mollifier_scale, cache_longitude_moments=True,
                compact_support=args.compact_support, pair_backend=args.pair_backend,
            )
            group_seconds = time.perf_counter() - start
            legacy_total = old_timing["increment_seconds_including_geometry_read"] + old_timing["scale_integration_seconds"]
            run = {"warmup": iteration == 0, "legacy": old_timing,
                   "legacy_total_seconds": legacy_total, "group_shell_seconds": group_seconds,
                   "group_over_legacy_ratio": group_seconds / legacy_total}
            if global_timing is not None:
                run["legacy_global"] = global_timing
                run["legacy_global_total_seconds"] = (
                    global_timing["increment_seconds_including_geometry_read"]
                    + global_timing["scale_integration_seconds"]
                )
            report["runs"].append(run)
            (args.output_dir / "timings.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(run), flush=True)
    finally:
        geometry.close()
    if old is None or group is None or (args.legacy_coverage == "both" and old_global is None):
        raise RuntimeError("benchmark did not produce the requested outputs")
    report["legacy_validity"] = validity(old)
    if old_global is not None:
        report["legacy_global_validity"] = validity(old_global)
    report["group_validity"] = validity(group)
    if args.legacy_reference:
        report["legacy_reference_max_absolute_errors"] = check_reference(
            old, args.legacy_reference, ("DL_u_longitudinal", "DL_u_vert_transverse_times_dz"),
        )
    if args.group_reference:
        report["group_reference_max_absolute_errors"] = check_reference(
            group, args.group_reference,
            ("horizontal", "structure", "vertical_flux", "mixed_moment", "kernel_mass", "direction", "velocity"),
        )
    report["output_write_seconds"] = {}
    outputs = [("legacy-transfer", old), ("group-shell", group)]
    if old_global is not None:
        outputs.append(("legacy-global-transfer", old_global))
    for name, result in outputs:
        start = time.perf_counter()
        result.to_zarr(args.output_dir / f"{name}.zarr", mode="w-", zarr_version=2)
        report["output_write_seconds"][name] = time.perf_counter() - start
    if old_global is not None:
        report["legacy_global_reference_max_absolute_errors"] = check_reference(
            old_global, args.output_dir / "legacy-transfer.zarr",
            ("DL_u_longitudinal", "DL_u_vert_transverse_times_dz"),
        )
    if args.cartesian_src:
        environment = dict(os.environ, PYTHONPATH=str(Path(args.cartesian_src).resolve()))
        command = [
            sys.executable, str(Path(__file__).with_name("benchmark_real_cartesian.py")),
            "--velocity-file", args.velocity_file, "--output-dir", str(args.output_dir / "cartesian"),
            "--physical-scale-km", str(args.physical_scale_km),
        ]
        with (args.output_dir / "cartesian-process.log").open("w") as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        report["cartesian"] = json.loads((args.output_dir / "cartesian" / "timings.json").read_text())
    (args.output_dir / "timings.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(entry["nonfinite_count"] for entry in report["legacy_validity"].values()):
        print(f"WARNING: legacy output contains non-finite values: {report['legacy_validity']}", flush=True)
    if any(entry["nonfinite_count"] for entry in report["group_validity"].values()):
        raise ValueError("group output contains non-finite values")


if __name__ == "__main__":
    main()
