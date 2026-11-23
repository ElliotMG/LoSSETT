#!/usr/bin/env python3
"""Time the prior spherical increment workflow on a prepared 5-degree field."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

from lossett.calc.compute_inter_scale_transfers_spherical import compute_inter_scale_kinetic_energy_transfer


def run(command, log_path):
    start = time.perf_counter()
    with log_path.open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=os.environ.copy())
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-file", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-scale-km", type=float, default=4000.)
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(exist_ok=False)
    geometry = root / "geometry"
    geometry.mkdir()
    geometry_seconds = run(
        [sys.executable, "-m", "lossett.calc.compute_spherical_geometry",
         "--grid", "5p0deg", "--save-path", str(geometry), "--save-trig-fns",
         "--chunk-origin-lat", "4"],
        root / "geometry.log",
    )
    elapsed = []
    for i in range(3):
        destination = root / f"increments-{i}"
        destination.mkdir()
        elapsed.append(run(
            [sys.executable, "-m", "lossett.calc.compute_delta_u_cubed_spherical",
             "--velocity-file", args.velocity_file, "--grid", "5p0deg", "--time-index", "0",
             "--pressure", "200", "--geom-path", str(geometry), "--save-path", str(destination),
             "--origin-lat-chunksize", "4", "--max-R", str(args.physical_scale_km),
             "--calc-dtype", "float64", "--save-dtype", "float64", "--include-w"],
            root / f"increments-{i}.log",
        ))
        print(f"Spherical increment run {i}: {elapsed[-1]:.3f} s", flush=True)
    stores = list((root / "increments-2").glob("*.zarr"))
    if len(stores) != 1:
        raise ValueError(f"expected one spherical increment store, found {len(stores)}")
    start = time.perf_counter()
    with xr.open_zarr(stores[0]) as source:
        ds = source.rename({"great_circle_distance": "r"}).load()
    read_seconds = time.perf_counter() - start
    start = time.perf_counter()
    result = compute_inter_scale_kinetic_energy_transfer(
        ds.delta_u_cubed_angular_integral_longitudinal,
        np.array([args.physical_scale_km * 500]),
        du3_vert=ds.delta_u_cubed_angular_integral_vertical,
    )
    result.load()
    integration_seconds = time.perf_counter() - start
    validity = {}
    for name, value in result.data_vars.items():
        missing = ~np.isfinite(value)
        other_dims = [dim for dim in value.dims if dim != "latitude"]
        missing_latitudes = value.latitude.where(missing.any(dim=other_dims), drop=True).values
        validity[name] = {
            "nonfinite_count": int(missing.sum().item()),
            "total_count": value.size,
            "affected_latitudes_degrees": missing_latitudes.tolist(),
        }
    if any(item["nonfinite_count"] for item in validity.values()):
        print(f"WARNING: non-finite integrated transfer values: {validity}", flush=True)
    start = time.perf_counter()
    result.to_zarr(root / "transfer.zarr", mode="w-", zarr_version=2)
    report = {
        "geometry_cli_seconds": geometry_seconds, "increments_warmup_cli_seconds": elapsed[0],
        "increments_cli_seconds": elapsed[1:], "increment_store_read_seconds": read_seconds,
        "scale_integration_seconds": integration_seconds, "transfer_write_seconds": time.perf_counter() - start,
        "physical_scale_km": args.physical_scale_km, "maximum_distance_km": args.physical_scale_km,
        "pressure_hPa": 200, "angular_weights": False, "includes_w": True,
        "vertical_output": "flux intermediate, not the complete vertical transfer",
        "transfer_validity": validity,
    }
    (root / "timings.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
