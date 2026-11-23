#!/usr/bin/env python3
"""Compare cached and reference group shells on a prepared real field."""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import xarray as xr

from lossett.calc.compute_group_increments import shell_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-file", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=False)
    with xr.open_dataset(args.velocity_file, decode_timedelta=False) as source:
        field = source.load()
    with xr.open_zarr(args.baseline) as source:
        baseline = source.load()
    if baseline.attrs["kernel"] != "heat":
        raise ValueError("this benchmark expects a heat-kernel baseline")
    levels = np.flatnonzero(field.pressure.values * 100 == baseline.attrs["coordinate_value"])
    if len(levels) != 1:
        raise ValueError("baseline pressure must identify exactly one input level")
    kwargs = dict(
        time_index=0, level_index=int(levels[0]), level_dim="pressure",
        coordinate_type="pressure", coordinate_unit="hPa", radius=baseline.attrs["radius_m"],
        heat_time=baseline.attrs["kernel_parameter"], atmosphere=baseline.attrs["atmosphere"],
        quadrature_tolerance=baseline.attrs["quadrature_tolerance"],
    )
    report = {"velocity_file": str(Path(args.velocity_file).resolve()),
              "baseline": str(Path(args.baseline).resolve()), "runs": []}
    for cached in (False, True, False, True):
        start = time.perf_counter()
        result = shell_dataset(field, cache_longitude_moments=cached, **kwargs)
        seconds = time.perf_counter() - start
        errors = {}
        for name in ("horizontal", "structure", "vertical_flux", "mixed_moment", "kernel_mass",
                     "velocity", "direction"):
            actual = result[name].values
            expected = baseline[name].values
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
            errors[name] = float(np.max(np.abs(actual - expected)))
        report["runs"].append({"cached": cached, "seconds": seconds, "max_absolute_errors": errors})
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.output_dir / "timings.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["runs"][-1]), flush=True)


if __name__ == "__main__":
    main()
