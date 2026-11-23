#!/usr/bin/env python3
"""Time the original Cartesian calculation on a prepared real field."""

import argparse
import contextlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import xarray as xr

from lossett.calc.calc_inter_scale_transfers import calc_inter_scale_energy_transfer_kinetic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-file", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-scale-km", type=float, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=False)
    ell = args.physical_scale_km * 500
    control = {
        "max_r": 2 * ell, "max_r_units": "m", "angle_precision": 1e-10,
        "x_coord_name": "longitude", "x_coord_units": "deg", "x_coord_boundary": "periodic",
        "y_coord_name": "latitude", "y_coord_units": "deg", "y_coord_boundary": np.nan,
    }
    start = time.perf_counter()
    with xr.open_dataset(args.velocity_file, decode_timedelta=False) as source:
        ds = source.sel(pressure=[200.]).load()
    read_seconds = time.perf_counter() - start
    samples = []
    with (args.output_dir / "calculation.log").open("w") as log:
        for i in range(3):
            start = time.perf_counter()
            with contextlib.redirect_stdout(log):
                result = calc_inter_scale_energy_transfer_kinetic(ds, control, length_scales=np.array([ell]))
            samples.append(time.perf_counter() - start)
            if result.sizes.get("length_scale") != 1 or not np.all(np.isfinite(result.values)):
                raise ValueError("Cartesian calculation did not produce one finite requested scale")
    start = time.perf_counter()
    result.to_dataset().to_zarr(args.output_dir / "transfer.zarr", mode="w-", zarr_version=2)
    report = {
        "input_read_seconds": read_seconds, "warmup_seconds": samples[0], "seconds": samples[1:],
        "output_write_seconds": time.perf_counter() - start,
        "pressure_hPa": 200, "length_parameter_m": ell, "physical_support_m": 2 * ell,
        "longitude_boundary": "periodic", "latitude_boundary": "missing",
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (args.output_dir / "timings.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
