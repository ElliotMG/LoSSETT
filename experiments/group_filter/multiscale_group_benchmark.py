"""Compare experimental multi-scale accumulators on selected real n320 origins."""

import json
import os
import resource
import socket
import time
from pathlib import Path

import numpy as np
import xarray as xr

from lossett.calc._group_numba import accumulate_group_pairs
from lossett.calc.compute_group_increments import _solid_angles
from lossett.calc.group_increments import latlon_shell_fields
from lossett.filtering.group_kernels import mollifier_fiber_moments
from multiscale_group_prototype import binned_coefficients, exact_multiscale, integrate_bins


INPUT = Path("/gws/ssde/j25b/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/"
             "DYAMOND_SUMMER/benchmark_antialiased_20160801T00/filtered_n320.nc")
OUTPUT = Path("/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/group_multiscale_n320_shared_tables_20261124")
RADIUS = 6382758.772639869
NAMES = ("horizontal", "structure", "vertical_flux", "mixed_x", "mixed_y", "mixed_z", "mass")


def main():
    OUTPUT.mkdir(exist_ok=False)
    with xr.open_dataset(INPUT, decode_timedelta=False) as ds:
        field = ds.isel(time=0).sel(pressure=200.).load()
    lat = np.deg2rad(field.latitude.values)
    lon = np.deg2rad(field.longitude.values)
    weights = _solid_angles(lat, lon).ravel()
    direction, velocity, grad, gradw = latlon_shell_fields(
        lat, lon, *(field[name].values for name in ("u", "v", "w")), radius=RADIUS,
    )
    direction = direction.reshape(-1, 3)
    velocity = velocity.reshape(-1, 3)
    grad = grad.reshape(-1, 3, 3)
    gradw = gradw.reshape(-1, 3)
    supports_km = np.geomspace(1500., 4000., 10)
    scales = supports_km * 1000 / (4*RADIUS)
    support_angle = 4000e3 / RADIUS
    origins = [(100, 131), (240, 321), (380, 529)]

    start = time.perf_counter()
    for scale in scales:
        mollifier_fiber_moments(np.array([1.]), scale)
    spectral_setup = time.perf_counter() - start

    x, u = direction[0], velocity[0]
    ix = np.array([0], dtype=np.int64)
    moments = np.ones((1, 1, 3))
    exact_multiscale(x, u, direction, velocity, grad, gradw, weights, moments, 1/RADIUS, ix)
    binned_coefficients(x, u, direction, velocity, grad, gradw, weights, 1/RADIUS, ix, support_angle, 2)
    integrate_bins(np.ones((3, 14)), np.ones((1, 3, 3)))
    accumulate_group_pairs(x, u, direction, velocity, grad, gradw, weights,
                           moments[0, :, 0], moments[0, :, 1], moments[0, :, 2], 1/RADIUS, ix)

    node_tables = {}
    node_setup_seconds = {}
    for count in (64, 256, 1024, 4096):
        start = time.perf_counter()
        node_mu = np.cos(np.linspace(0, support_angle, count+1))
        node_tables[count] = np.ascontiguousarray(np.stack([
            np.asarray(mollifier_fiber_moments(node_mu, scale)).T for scale in scales
        ]))
        node_setup_seconds[count] = time.perf_counter() - start

    records = []
    for row, col in origins:
        index = row * len(lon) + col
        x, u = direction[index], velocity[index]
        start = time.perf_counter()
        mu_all = np.clip(direction @ x, -1, 1)
        active = np.flatnonzero(mu_all >= np.cos(support_angle))
        mu = mu_all[active]
        select_seconds = time.perf_counter() - start
        start = time.perf_counter()
        moments = np.zeros((len(scales), len(active), 3))
        for i, scale in enumerate(scales):
            valid = mu >= np.cos(4*scale)
            moments[i, valid, :] = np.asarray(mollifier_fiber_moments(mu[valid], scale)).T
        pair_moment_seconds = time.perf_counter() - start
        start = time.perf_counter()
        reference = np.array([
            accumulate_group_pairs(x, u, direction, velocity, grad, gradw, weights,
                                   *(np.ascontiguousarray(moments[s, :, c]) for c in range(3)),
                                   1/RADIUS, active)
            for s in range(len(scales))
        ])
        repeated_seconds = time.perf_counter() - start
        start = time.perf_counter()
        exact = exact_multiscale(x, u, direction, velocity, grad, gradw, weights,
                                 moments, 1/RADIUS, active)
        exact_seconds = time.perf_counter() - start
        np.testing.assert_allclose(exact, reference, rtol=2e-10, atol=1e-11)
        record = {
            "row": row, "col": col, "latitude_degrees": float(np.rad2deg(lat[row])),
            "active_pairs": len(active), "selection_seconds": select_seconds,
            "exact_pair_moment_seconds": pair_moment_seconds,
            "repeated_single_scale_accumulation_seconds": repeated_seconds,
            "exact_multiscale_accumulation_seconds": exact_seconds,
            "exact_max_absolute_by_channel": dict(zip(NAMES, np.max(np.abs(exact-reference), axis=0).tolist())),
            "reference": reference.tolist(), "bins": [],
        }
        for count in (64, 256, 1024, 4096):
            start = time.perf_counter()
            summary = binned_coefficients(x, u, direction, velocity, grad, gradw, weights,
                                          1/RADIUS, active, support_angle, count)
            bin_seconds = time.perf_counter() - start
            start = time.perf_counter()
            result = integrate_bins(summary, node_tables[count])
            integration_seconds = time.perf_counter() - start
            record["bins"].append({
                "count": count, "bin_width_km": 4000/count,
                "binning_seconds": bin_seconds,
                "integration_seconds": integration_seconds, "summary_bytes": summary.nbytes,
                "max_absolute_by_channel": dict(zip(NAMES, np.max(np.abs(result-exact), axis=0).tolist())),
                "values": result.tolist(),
            })
        records.append(record)
        print(f"Finished origin ({row}, {col})", flush=True)
    with (OUTPUT / "results.json").open("x") as stream:
        json.dump({
            "job_id": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(),
            "input": str(INPUT), "grid_shape": [len(lat), len(lon)], "radius_m": RADIUS,
            "support_G_km": supports_km.tolist(), "spectral_coefficient_setup_seconds": spectral_setup,
            "shared_node_table_setup_seconds": node_setup_seconds,
            "records": records, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "caveats": "three selected origins only; JIT warm; one repetition; no full-ring reuse or I/O",
        }, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
