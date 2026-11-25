"""Share or tabulate exact group moments across origins for the fused multi-scale Numba path.

Three exact-moment strategies are compared, all feeding the same fused accumulator arithmetic:
  per-origin  : evaluate g,h,k at every active pair for every origin (the earlier pilot).
  ring-shared : evaluate once per origin latitude on (target row, longitude lag) and reuse
                for every origin longitude of that ring (exact, by longitude periodicity).
  chebyshev   : scale-dependent piecewise-Chebyshev table of g,h,k in separation angle,
                evaluated inside the accumulator; independent of origin latitude.
"""

import argparse
import json
import math
import os
import resource
import socket
import time
from pathlib import Path

import numpy as np
import xarray as xr
from numba import njit

from lossett.calc._group_numba import accumulate_group_pairs
from lossett.calc.compute_group_increments import _solid_angles
from lossett.calc.group_increments import latlon_shell_fields
from lossett.filtering.group_kernels import mollifier_fiber_moments
from multiscale_group_prototype import _apply, _coefficients, exact_multiscale

ROOT = Path("/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry")
DATA = Path("/gws/ssde/j25b/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/"
            "DYAMOND_SUMMER/benchmark_antialiased_20160801T00")
RADIUS = 6382758.772639869
NAMES = ("horizontal", "structure", "vertical_flux", "mixed_x", "mixed_y", "mixed_z", "mass")


@njit(cache=True)
def exact_ring_table(x, u, directions, velocity, grad, gradw, weights, table,
                     curvature, indices, nlon, col):
    """Fused accumulation using moments stored by (target row, longitude lag)."""
    out = np.zeros((table.shape[0], 7))
    for j in range(len(indices)):
        target = indices[j]
        row, longitude = target // nlon, target % nlon
        lag = (longitude - col) % nlon
        c = _coefficients(x, u, directions[target], velocity[target],
                          grad[target], gradw[target], curvature)
        for scale in range(table.shape[0]):
            _apply(out, scale, c, table[scale, row, lag, 0], table[scale, row, lag, 1],
                   table[scale, row, lag, 2], weights[target])
    return out


@njit(cache=True)
def exact_chebyshev(x, u, directions, velocity, grad, gradw, weights, curvature, indices,
                    support_angles, coefficients):
    """Fused accumulation, evaluating tabulated moments (scale, segment, degree+1, 3)."""
    nscale, nseg, ncoef = coefficients.shape[0], coefficients.shape[1], coefficients.shape[2]
    out = np.zeros((nscale, 7))
    for j in range(len(indices)):
        target = indices[j]
        y = directions[target]
        mu = min(1., max(-1., x[0]*y[0] + x[1]*y[1] + x[2]*y[2]))
        distance = math.acos(mu)
        c = _coefficients(x, u, y, velocity[target], grad[target], gradw[target], curvature)
        for scale in range(nscale):
            if distance >= support_angles[scale]:
                continue
            position = distance/support_angles[scale]*nseg
            seg = min(int(position), nseg-1)
            t = 2*(position-seg)-1
            m0 = m1 = m2 = 0.
            b0 = b1 = b2 = 0.
            d0 = d1 = d2 = 0.
            for k in range(ncoef-1, 0, -1):
                n0 = 2*t*b0 - d0 + coefficients[scale, seg, k, 0]
                n1 = 2*t*b1 - d1 + coefficients[scale, seg, k, 1]
                n2 = 2*t*b2 - d2 + coefficients[scale, seg, k, 2]
                d0, d1, d2 = b0, b1, b2
                b0, b1, b2 = n0, n1, n2
            m0 = t*b0 - d0 + coefficients[scale, seg, 0, 0]
            m1 = t*b1 - d1 + coefficients[scale, seg, 0, 1]
            m2 = t*b2 - d2 + coefficients[scale, seg, 0, 2]
            _apply(out, scale, c, m0, m1, m2, weights[target])
    return out


def chebyshev_tables(scales, nseg, degree):
    """Piecewise Chebyshev (first kind) coefficients of g,h,k versus angle."""
    nodes = np.cos(np.pi*(np.arange(degree+1)+.5)/(degree+1))
    coefficients = np.zeros((len(scales), nseg, degree+1, 3))
    for s, scale in enumerate(scales):
        support = 4*scale
        for seg in range(nseg):
            angles = support*(seg + (nodes[::-1]+1)/2)/nseg
            values = np.asarray(mollifier_fiber_moments(np.cos(angles), scale)).T
            coefficients[s, seg] = np.polynomial.chebyshev.chebfit(nodes[::-1], values, degree)
    return coefficients


def exact_values(mu, scale, mask_support):
    out = np.zeros((len(mu), 3))
    mask = mu >= np.cos(mask_support)
    out[mask] = np.asarray(mollifier_fiber_moments(mu[mask], scale)).T
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=("n320", "n640"), required=True)
    parser.add_argument("--row-fractions", type=float, nargs="+", default=(0.25, 0.5))
    parser.add_argument("--supports-min-km", type=float, default=1500.)
    parser.add_argument("--supports-max-km", type=float, default=4000.)
    parser.add_argument("--nscales", type=int, default=10)
    args = parser.parse_args()
    out = ROOT / f"moment_sharing_{args.grid}_20261124"
    out.mkdir(exist_ok=False)
    with xr.open_dataset(DATA / f"filtered_{args.grid}.nc", decode_timedelta=False) as ds:
        field = ds.isel(time=0).sel(pressure=200.).load()
    lat, lon = np.deg2rad(field.latitude.values), np.deg2rad(field.longitude.values)
    nlat, nlon = len(lat), len(lon)
    weights = _solid_angles(lat, lon).ravel()
    direction, velocity, grad, gradw = latlon_shell_fields(
        lat, lon, *(field[name].values for name in ("u", "v", "w")), radius=RADIUS)
    direction, velocity = direction.reshape(-1, 3), velocity.reshape(-1, 3)
    grad, gradw = grad.reshape(-1, 3, 3), gradw.reshape(-1, 3)
    supports_km = np.geomspace(args.supports_min_km, args.supports_max_km, args.nscales)
    scales = supports_km*1000/(4*RADIUS)
    support_angles = 4*scales
    max_support = float(support_angles.max())
    curvature = 1/RADIUS

    start = time.perf_counter()
    table_settings = ((32, 8), (64, 8), (64, 12), (128, 10))
    cheb = {}
    for nseg, degree in table_settings:
        start = time.perf_counter()
        cheb[(nseg, degree)] = (chebyshev_tables(scales, nseg, degree), time.perf_counter()-start)
    # Pointwise moment accuracy of each tabulation against independent exact evaluations.
    rng = np.random.default_rng(12345)
    pointwise = {}
    sample_mu = np.cos(rng.uniform(0, max_support, 20000))
    for key, (coefficients, _) in cheb.items():
        worst = 0.
        for s, scale in enumerate(scales):
            exact = exact_values(sample_mu, scale, support_angles[s])
            angle = np.arccos(sample_mu)
            inside = angle < support_angles[s]
            position = angle[inside]/support_angles[s]*key[0]
            seg = np.minimum(position.astype(int), key[0]-1)
            t = 2*(position-seg)-1
            approx = np.stack([
                np.array([np.polynomial.chebyshev.chebval(tt, coefficients[s, sg, :, c])
                          for tt, sg in zip(t, seg)]) for c in range(3)], axis=1)
            diff = np.max(np.abs(approx-exact[inside]))
            worst = max(worst, float(diff/np.max(np.abs(exact))))
        pointwise[f"{key[0]}x{key[1]}"] = worst

    # Warm the compiled kernels.
    x0, u0 = direction[0], velocity[0]
    one = np.array([0], dtype=np.int64)
    first_coeff = next(iter(cheb.values()))[0]
    exact_chebyshev(x0, u0, direction, velocity, grad, gradw, weights, curvature, one,
                    support_angles, first_coeff)
    exact_ring_table(x0, u0, direction, velocity, grad, gradw, weights,
                     np.ones((1, nlat, nlon, 3)), curvature, one, nlon, 0)
    exact_multiscale(x0, u0, direction, velocity, grad, gradw, weights,
                     np.ones((1, 1, 3)), curvature, one)
    accumulate_group_pairs(x0, u0, direction, velocity, grad, gradw, weights,
                           np.ones(1), np.ones(1), np.ones(1), curvature, one)

    records = []
    for fraction in args.row_fractions:
        row = int((nlat-1)*fraction)
        columns = (0, nlon//3, nlon//2+7)
        # Ring table: origin at column 0, moments by (target row, longitude lag).
        x0 = direction[row*nlon]
        mu_ring = np.clip(direction@x0, -1, 1)
        active0 = np.flatnonzero(mu_ring >= np.cos(max_support))
        start = time.perf_counter()
        table = np.zeros((len(scales), nlat, nlon, 3))
        mu_sel = mu_ring[active0]
        for s, scale in enumerate(scales):
            mask = mu_sel >= np.cos(support_angles[s])
            values = np.asarray(mollifier_fiber_moments(mu_sel[mask], scale)).T
            flat = table[s].reshape(-1, 3)
            flat[active0[mask]] = values
        ring_table_seconds = time.perf_counter()-start
        per_column = []
        for col in columns:
            index = row*nlon+col
            x, u = direction[index], velocity[index]
            start = time.perf_counter()
            mu_all = np.clip(direction@x, -1, 1)
            active = np.flatnonzero(mu_all >= np.cos(max_support))
            selection_seconds = time.perf_counter()-start
            mu = mu_all[active]
            start = time.perf_counter()
            moments = np.zeros((len(scales), len(active), 3))
            for s, scale in enumerate(scales):
                mask = mu >= np.cos(support_angles[s])
                moments[s, mask] = np.asarray(mollifier_fiber_moments(mu[mask], scale)).T
            per_origin_moment_seconds = time.perf_counter()-start
            reference_times = []
            for _ in range(3):
                start = time.perf_counter()
                reference = exact_multiscale(x, u, direction, velocity, grad, gradw, weights,
                                             moments, curvature, active)
                reference_times.append(time.perf_counter()-start)
            peaks = np.maximum(np.max(np.abs(reference), axis=0), np.finfo(float).tiny)
            shared_times = []
            for _ in range(3):
                start = time.perf_counter()
                shared = exact_ring_table(x, u, direction, velocity, grad, gradw, weights,
                                          table, curvature, active, nlon, col)
                shared_times.append(time.perf_counter()-start)
            entry = {
                "col": col, "active_pairs": len(active), "selection_seconds": selection_seconds,
                "per_origin_moment_seconds": per_origin_moment_seconds,
                "fused_exact_accumulate_seconds": min(reference_times),
                "ring_shared_accumulate_seconds": min(shared_times),
                "ring_shared_max_abs_difference": float(np.max(np.abs(shared-reference))),
                "chebyshev": {},
            }
            for key, (coefficients, _) in cheb.items():
                times = []
                for _ in range(3):
                    start = time.perf_counter()
                    approx = exact_chebyshev(x, u, direction, velocity, grad, gradw, weights,
                                             curvature, active, support_angles, coefficients)
                    times.append(time.perf_counter()-start)
                entry["chebyshev"][f"{key[0]}x{key[1]}"] = {
                    "accumulate_seconds": min(times),
                    "max_peak_normalized_error": float(np.max(np.abs(approx-reference)/peaks)),
                }
            per_column.append(entry)
        records.append({
            "row": row, "latitude_degrees": float(np.rad2deg(lat[row])),
            "ring_table_seconds": ring_table_seconds,
            "ring_table_bytes": table.nbytes, "ring_origins": nlon,
            "columns": per_column,
        })
        print(f"Finished row {row}", flush=True)
        del table
    with (out/"results.json").open("x") as stream:
        json.dump({
            "host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
            "grid": args.grid, "support_km": supports_km.tolist(),
            "chebyshev_build_seconds": {f"{k[0]}x{k[1]}": v[1] for k, v in cheb.items()},
            "chebyshev_pointwise_peak_normalized_error": pointwise,
            "records": records, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "caveats": "best-of-three warm timings on a shared node; pair selection, field "
                       "preparation and I/O excluded except where stated; sampled errors only",
        }, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
