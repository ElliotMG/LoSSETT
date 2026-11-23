#!/usr/bin/env python3
"""Profile a bounded real-field group pilot and time the old Cartesian path."""

import argparse
import cProfile
import json
import os
import pstats
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.polynomial.legendre import leggauss

from lossett.calc.compute_group_increments import _solid_angles, shell_dataset, vertical_dataset
from lossett.calc.group_increments import group_increment_shell, hydrostatic_radius, latlon_shell_fields


def heat_time_for_increment_support(length_m, radius):
    """Match G's degree-one response to a spherical standard mollifier."""
    angle = length_m / radius
    if not 0 < angle < np.pi:
        raise ValueError("physical support must be between zero and half a great circle")
    nodes, weights = leggauss(256)
    mu0 = np.cos(angle)
    mu = mu0 + (1 - mu0) * (nodes + 1) / 2
    profile = np.exp(-1 / (1 - (np.arccos(mu) / angle)**2))
    response = float(np.dot(weights * profile, mu) / np.dot(weights, profile))
    if not 0 < response < 1:
        raise ValueError("mollifier degree-one response cannot be matched by a finite heat kernel")
    return -np.log(response) / 4, response


def standard_atmosphere(pressure):
    """US Standard Atmosphere below 20 km, with constant reference gravity."""
    p = np.asarray(pressure, dtype=float)
    gravity, gas, surface_temp, lapse = 9.80665, 287.05287, 288.15, 0.0065
    base_temp = surface_temp - lapse * 11000
    base_pressure = 101325 * (base_temp / surface_temp)**(gravity / (gas * lapse))
    top_pressure = base_pressure * np.exp(-gravity * 9000 / (gas * base_temp))
    if not np.all(np.isfinite(p)) or np.any((p < top_pressure) | (p > 101325)):
        raise ValueError("standard-atmosphere pressure must correspond to 0--20 km")
    temperature = np.where(p >= base_pressure, surface_temp * (p / 101325)**(gas * lapse / gravity), base_temp)
    height = np.where(
        p >= base_pressure, (surface_temp - temperature) / lapse,
        11000 + gas * base_temp / gravity * np.log(base_pressure / p),
    )
    return p / (gas * temperature), height, gravity


def _timed_process(command, log_path, env):
    start = time.perf_counter()
    with log_path.open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    return {"seconds": time.perf_counter() - start, "returncode": result.returncode, "log": str(log_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-file", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cartesian-src", required=True)
    parser.add_argument("--physical-scale-km", type=float, default=4000.)
    parser.add_argument("--max-group-seconds", type=float, default=1500.)
    args = parser.parse_args()
    if args.max_group_seconds <= 0 or not np.isfinite(args.max_group_seconds):
        parser.error("--max-group-seconds must be positive and finite")
    root = args.output_dir
    root.mkdir(exist_ok=False)
    overall_start = time.perf_counter()
    radius_earth = 6371000.
    heat_time, response = heat_time_for_increment_support(args.physical_scale_km * 1000, radius_earth)
    print(f"Matched heat time {heat_time:.10g}; mollifier degree-one response {response:.10g}", flush=True)
    report = {
        "velocity_file": str(Path(args.velocity_file).resolve()), "grid": "5p0deg",
        "physical_scale_km": args.physical_scale_km, "heat_time": float(heat_time),
        "mollifier_degree_one_response": response, "time_index": 0,
        "group_fluid": "anelastic", "group_atmosphere": "deep",
    }
    cartesian_env = dict(os.environ)
    cartesian_env["PYTHONPATH"] = str(Path(args.cartesian_src).resolve())
    command = [
        sys.executable, str(Path(__file__).with_name("benchmark_real_cartesian.py")),
        "--velocity-file", args.velocity_file, "--output-dir", str(root / "cartesian"),
        "--physical-scale-km", str(args.physical_scale_km),
    ]
    report["cartesian_process"] = _timed_process(command, root / "cartesian-process.log", cartesian_env)
    print("Cartesian process:", report["cartesian_process"], flush=True)
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    start = time.perf_counter()
    with xr.open_dataset(args.velocity_file, decode_timedelta=False) as source:
        ds = source.sel(pressure=[150., 200., 250.]).isel(time=[0]).load()
    report["group_input_read_seconds"] = time.perf_counter() - start
    pressure = ds.pressure.values * 100
    rho, height, gravity = standard_atmosphere(pressure)
    radii = hydrostatic_radius(pressure, rho, gravity, radius_earth + height[0])
    report.update(
        pressure_pa=pressure.tolist(), reference_density_kg_m3=rho.tolist(),
        reference_radius_m=radii.tolist(), gravity_m_s2=gravity,
        radius_approximation="US Standard Atmosphere geopotential height; hydrostatic trapezoidal integration",
    )
    lat, lon = np.deg2rad(ds.latitude.values), np.deg2rad(ds.longitude.values)
    middle = ds.sel(pressure=200.).isel(time=0)
    start = time.perf_counter()
    fields = latlon_shell_fields(lat, lon, middle.u.values, middle.v.values, middle.w.values, radius=radii[1])
    weights = _solid_angles(lat, lon)
    report["group_gradient_geometry_seconds"] = time.perf_counter() - start
    directions, velocity, grad, gradw = fields
    n = directions.shape[0] * directions.shape[1]
    target = (directions.reshape(n, 3), velocity.reshape(n, 3),
              grad.reshape(n, 3, 3), gradw.reshape(n, 3), weights.reshape(n))
    origins = [(0, 0), (len(lat) // 4, 0), (len(lat) // 2, 0),
               (3 * len(lat) // 4, 0), (len(lat) - 1, 0)]

    def run_origin(i, j):
        return group_increment_shell(
            directions[i, j], velocity[i, j], *target, radius=radii[1], heat_time=heat_time,
        )

    run_origin(*origins[2])
    times, mass_errors = [], []
    for origin in origins:
        start = time.perf_counter()
        result = run_origin(*origin)
        times.append(time.perf_counter() - start)
        mass_errors.append(abs(result["kernel_mass"] - 1))
    profile = cProfile.Profile()
    profile.runcall(run_origin, *origins[2])
    profile.dump_stats(root / "group-origin.prof")
    with (root / "group-origin-profile.txt").open("w") as stream:
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(30)
    estimate = 4 * n * max(times)
    report.update(
        origin_sample_seconds=times, origin_sample_kernel_mass_errors=mass_errors,
        estimated_four_grid_seconds=estimate, max_group_seconds=args.max_group_seconds,
    )
    print(f"Sampled origin times {times}; estimated four grid runs {estimate:.1f} s", flush=True)
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if max(mass_errors) > 0.01:
        raise ValueError("sampled origins under-resolve the group kernel; see summary.json")
    if estimate > args.max_group_seconds:
        raise RuntimeError("full group pilot exceeds its time budget; sampled timings are saved")
    shells = []
    shell_times = []
    write_times = []
    for i, level in enumerate(pressure):
        start = time.perf_counter()
        shell = shell_dataset(
            ds, time_index=0, level_index=i, level_dim="pressure", coordinate_type="pressure",
            coordinate_unit="hPa", radius=radii[i], heat_time=heat_time,
        )
        shell.attrs.update(
            velocity_source=str(Path(args.velocity_file).resolve()),
            hydrostatic_pressure_pa=pressure.tolist(), hydrostatic_density=rho.tolist(),
            hydrostatic_initial_radius=float(radii[0]), hydrostatic_gravity=gravity,
        )
        shell_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        shell.to_zarr(root / f"shell-{int(level / 100)}hPa.zarr", mode="w-", zarr_version=2)
        write_times.append(time.perf_counter() - start)
        shells.append(shell)
        print(f"Group {level / 100:g} hPa: {shell_times[-1]:.3f} s", flush=True)
        report["group_shell_seconds"] = shell_times
        report["group_shell_write_seconds"] = write_times
        (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    start = time.perf_counter()
    repeat = shell_dataset(
        ds, time_index=0, level_index=1, level_dim="pressure", coordinate_type="pressure",
        coordinate_unit="hPa", radius=radii[1], heat_time=heat_time,
    )
    report["group_repeat_200_seconds"] = time.perf_counter() - start
    for key in ("horizontal", "structure", "vertical_flux", "mixed_moment"):
        np.testing.assert_allclose(repeat[key].values, shells[1][key].values, rtol=0, atol=0)
    start = time.perf_counter()
    reconstructed = vertical_dataset(shells, fluid="anelastic", reference_density=rho, gravity=gravity)
    report["group_vertical_reconstruction_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    reconstructed.to_zarr(root / "group-transfer.zarr", mode="w-", zarr_version=2)
    report["group_transfer_write_seconds"] = time.perf_counter() - start
    report["group_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["overall_seconds"] = time.perf_counter() - overall_start
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["cartesian_process"]["returncode"] != 0:
        raise RuntimeError("Cartesian benchmark failed; inspect cartesian-process.log")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
