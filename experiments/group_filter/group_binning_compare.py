"""Compare endpoint-distance and coherent group-angle bins at real origins."""

import argparse
import json
import os
import resource
import socket
import time
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.polynomial.legendre import leggauss

from group_angle_binning import (
    accumulate_group_angle_bins, accumulate_segmented_group_bins, central_mollifier_nodes, integrate_group_angle_bins,
)
from lossett.calc._group_numba import accumulate_group_pairs
from lossett.calc.compute_group_increments import _solid_angles
from lossett.calc.group_increments import latlon_shell_fields
from lossett.filtering.group_kernels import mollifier_fiber_moments
from multiscale_group_prototype import binned_coefficients, exact_multiscale, integrate_bins


ROOT = Path("/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry")
DATA = Path("/gws/ssde/j25b/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/"
            "DYAMOND_SUMMER/benchmark_antialiased_20160801T00")
RADIUS = 6382758.772639869
NAMES = ("horizontal", "structure", "vertical_flux", "mixed_x", "mixed_y", "mixed_z", "mass")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=("n320", "n640"), required=True)
    args = parser.parse_args()
    out = ROOT / f"group_angle_binning_segmented_{args.grid}_20261124"
    out.mkdir(exist_ok=False)
    with xr.open_dataset(DATA / f"filtered_{args.grid}.nc", decode_timedelta=False) as ds:
        field = ds.isel(time=0).sel(pressure=200.).load()
    lat, lon = np.deg2rad(field.latitude.values), np.deg2rad(field.longitude.values)
    weights = _solid_angles(lat, lon).ravel()
    direction, velocity, grad, gradw = latlon_shell_fields(
        lat, lon, *(field[name].values for name in ("u", "v", "w")), radius=RADIUS,
    )
    direction, velocity = direction.reshape(-1, 3), velocity.reshape(-1, 3)
    grad, gradw = grad.reshape(-1, 3, 3), gradw.reshape(-1, 3)
    supports_km = np.array([2000., 4000.])
    scales = supports_km*1000/(4*RADIUS)
    support = 4000e3/RADIUS
    counts = (128, 512, 2048)
    fiber_points = 512
    cospsi = np.cos(2*np.pi*np.arange(fiber_points)/fiber_points)
    gauss_nodes, gauss_weights = leggauss(8)
    tables = {}
    for count in counts:
        start = time.perf_counter()
        mu = np.cos(np.linspace(0, support, count+1))
        endpoint = np.ascontiguousarray(np.stack([
            np.asarray(mollifier_fiber_moments(mu, scale)).T for scale in scales
        ]))
        endpoint_seconds = time.perf_counter()-start
        group, corrections, masses, group_seconds = central_mollifier_nodes(support, count, scales)
        tables[count] = (endpoint, group, {
            "endpoint_table_seconds": endpoint_seconds, "group_table_seconds": group_seconds,
            "group_negative_roundoff_clipped": corrections, "group_unnormalized_masses": masses,
        })
    x, u = direction[0], velocity[0]
    one = np.array([0], dtype=np.int64)
    accumulate_group_angle_bins(x, u, direction, velocity, grad, gradw, weights,
                                1/RADIUS, one, support, 2, cospsi[:2])
    accumulate_segmented_group_bins(x, u, direction, velocity, grad, gradw, weights,
                                     1/RADIUS, one, support, 2, gauss_nodes, gauss_weights)
    integrate_group_angle_bins(np.ones((3, 14)), np.ones((1, 3)))
    binned_coefficients(x, u, direction, velocity, grad, gradw, weights, 1/RADIUS, one, support, 2)
    integrate_bins(np.ones((3, 14)), np.ones((1, 3, 3)))
    exact_multiscale(x, u, direction, velocity, grad, gradw, weights, np.ones((1, 1, 3)), 1/RADIUS, one)
    records = []
    for row, col in (((len(lat)-1)//4, len(lon)//3), ((len(lat)-1)//2, len(lon)//2)):
        index = row*len(lon)+col
        x, u = direction[index], velocity[index]
        start = time.perf_counter()
        mu_all = np.clip(direction@x, -1, 1)
        active = np.flatnonzero(mu_all >= np.cos(support))
        mu = mu_all[active]
        select_seconds = time.perf_counter()-start
        start = time.perf_counter()
        exact_moments = np.zeros((len(scales), len(active), 3))
        for s, scale in enumerate(scales):
            mask = mu >= np.cos(4*scale)
            exact_moments[s, mask] = np.asarray(mollifier_fiber_moments(mu[mask], scale)).T
        moment_seconds = time.perf_counter()-start
        start = time.perf_counter()
        exact = exact_multiscale(x, u, direction, velocity, grad, gradw, weights,
                                 exact_moments, 1/RADIUS, active)
        exact_seconds = time.perf_counter()-start
        independent = np.array([
            accumulate_group_pairs(x, u, direction, velocity, grad, gradw, weights,
                                   *(np.ascontiguousarray(exact_moments[s, :, c]) for c in range(3)),
                                   1/RADIUS, active)
            for s in range(len(scales))
        ])
        np.testing.assert_allclose(exact, independent, rtol=2e-10, atol=1e-11)
        entry = {
            "row": row, "col": col, "latitude_degrees": float(np.rad2deg(lat[row])),
            "active_pairs": len(active), "selection_seconds": select_seconds,
            "exact_pair_moment_seconds": moment_seconds, "exact_accumulation_seconds": exact_seconds,
            "reference": exact.tolist(), "results": [],
        }
        for count in counts:
            endpoint_table, group_table, _ = tables[count]
            start = time.perf_counter()
            radial = binned_coefficients(x, u, direction, velocity, grad, gradw, weights,
                                          1/RADIUS, active, support, count)
            radial_bin_seconds = time.perf_counter()-start
            start = time.perf_counter()
            radial_result = integrate_bins(radial, endpoint_table)
            radial_integral_seconds = time.perf_counter()-start
            start = time.perf_counter()
            central = accumulate_group_angle_bins(x, u, direction, velocity, grad, gradw, weights,
                                                  1/RADIUS, active, support, count, cospsi)
            group_bin_seconds = time.perf_counter()-start
            start = time.perf_counter()
            central_result = integrate_group_angle_bins(central, group_table)
            group_integral_seconds = time.perf_counter()-start
            peaks = np.max(np.abs(exact), axis=0)
            eps = np.finfo(float).tiny
            result = {
                "bins": count, "fiber_points": fiber_points,
                "endpoint_bin_seconds": radial_bin_seconds, "endpoint_integrate_seconds": radial_integral_seconds,
                "group_bin_seconds": group_bin_seconds, "group_integrate_seconds": group_integral_seconds,
                "endpoint_max_abs_errors": dict(zip(NAMES, np.max(np.abs(radial_result-exact), axis=0).tolist())),
                "group_max_abs_errors": dict(zip(NAMES, np.max(np.abs(central_result-exact), axis=0).tolist())),
                "endpoint_peak_normalized_errors": dict(zip(NAMES, (np.max(np.abs(radial_result-exact), axis=0)
                                                                     / np.maximum(peaks, eps)).tolist())),
                "group_peak_normalized_errors": dict(zip(NAMES, (np.max(np.abs(central_result-exact), axis=0)
                                                                  / np.maximum(peaks, eps)).tolist())),
                "endpoint": radial_result.tolist(), "group": central_result.tolist(),
            }
            if count <= 512:
                start = time.perf_counter()
                segmented = accumulate_segmented_group_bins(
                    x, u, direction, velocity, grad, gradw, weights, 1/RADIUS,
                    active, support, count, gauss_nodes, gauss_weights,
                )
                segmented_seconds = time.perf_counter()-start
                start = time.perf_counter()
                segmented_result = integrate_group_angle_bins(segmented, group_table)
                result.update({
                    "segmented_group_bin_seconds": segmented_seconds,
                    "segmented_group_integrate_seconds": time.perf_counter()-start,
                    "segmented_group_order": len(gauss_nodes),
                    "segmented_group_peak_normalized_errors": dict(zip(
                        NAMES, (np.max(np.abs(segmented_result-exact), axis=0)/np.maximum(peaks, eps)).tolist()
                    )),
                    "segmented_group": segmented_result.tolist(),
                })
            entry["results"].append(result)
        records.append(entry)
        print(f"Finished ({row}, {col})", flush=True)
    with (out / "results.json").open("x") as stream:
        json.dump({"host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
                   "grid_shape": [len(lat), len(lon)], "support_G_km": supports_km.tolist(),
                   "table_setup": {count: data[2] for count, data in tables.items()},
                   "records": records, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "caveats": "two origins, two scales, one repetition; JIT warm; fixed fiber quadrature; "
                              "continuous group interpolation normalized; exact reference pair moments not ring-reused"},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
