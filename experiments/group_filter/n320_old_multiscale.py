"""Time ten-scale postprocessing of the saved old spherical n320 increments."""

import json
import os
import resource
import socket
import time
from pathlib import Path

import numpy as np
import xarray as xr

from lossett.calc.compute_inter_scale_transfers_spherical import compute_inter_scale_kinetic_energy_transfer


ROOT = Path("/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/n320_old_spherical_20261124")
DESTINATION = ROOT / "multiscale_500to4000km"
NAMES = ("DL_u_longitudinal", "DL_u_vert_transverse_times_dz")


def main():
    DESTINATION.mkdir(exist_ok=False)
    stores = list((ROOT / "increments").glob("*.zarr"))
    if len(stores) != 1:
        raise ValueError(f"expected exactly one increment store, found {len(stores)}")
    start = time.perf_counter()
    with xr.open_zarr(stores[0]) as source:
        increments = source.rename({"great_circle_distance": "r"}).load()
    read_seconds = time.perf_counter() - start
    supports_km = np.geomspace(500.0, 4000.0, 10)
    start = time.perf_counter()
    result = compute_inter_scale_kinetic_energy_transfer(
        increments.delta_u_cubed_angular_integral_longitudinal,
        supports_km * 500.0,
        du3_vert=increments.delta_u_cubed_angular_integral_vertical,
    ).load()
    integration_seconds = time.perf_counter() - start
    np.testing.assert_allclose(result.L.values, supports_km * 1000.0, rtol=0, atol=1e-8)
    with xr.open_zarr(ROOT / "legacy-transfer.zarr") as source:
        reference = source.load()
    errors = {}
    validity = {}
    for name in NAMES:
        actual = result[name].sel(L=4_000_000.0)
        expected = reference[name].sel(L=4_000_000.0)
        np.testing.assert_allclose(actual.values, expected.values, rtol=1e-12, atol=1e-12, equal_nan=True)
        errors[name] = float(np.nanmax(np.abs(actual.values - expected.values)))
        validity[name] = [
            {"support_km": float(support), "finite": int(np.isfinite(result[name].isel(L=i)).sum().item())}
            for i, support in enumerate(supports_km)
        ]
    start = time.perf_counter()
    result.to_zarr(DESTINATION / "transfer.zarr", mode="w-", zarr_version=2)
    write_seconds = time.perf_counter() - start
    report = {
        "host": socket.gethostname(), "increment_store": str(stores[0]),
        "support_km": supports_km.tolist(), "increment_store_read_seconds": read_seconds,
        "ten_scale_integration_seconds": integration_seconds, "output_write_seconds": write_seconds,
        "reference_4000km_max_absolute_errors": errors, "validity": validity,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "io_cache": "uncontrolled", "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "scope": "ten-scale integration of precomputed increments; excludes increment calculation and geometry",
    }
    with (DESTINATION / "results.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
