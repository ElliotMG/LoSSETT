"""Fair comparison of exact (ring-shared) and distance-binned coherent group-moment paths.

All paths use the same multiscale accumulator arithmetic and the same physical-space (quadrature) moment
tables, so they work at any G support. For one ring of origins at a fixed latitude:

  A  exact, per origin  : evaluate the 1D piecewise-Chebyshev moment tables inside the fused accumulator.
  B  exact, ring-shared : build moments once per ring on (target row, longitude lag), then accumulate with lookups.
  C  distance-binned    : O(1)-per-pair binning in distance, expansion through a scale-independent fiber table,
                          then integration against the sampled central kernels.

One-off costs (1D tables, kernel nodes, fiber table) are reported separately from per-ring and per-origin costs.
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
from numpy.polynomial.legendre import leggauss

from coherent_fiber_reduction import expand_distance_bins, fiber_hat_table
from group_angle_binning import integrate_group_angle_bins
from lossett.calc.compute_group_increments import _solid_angles
from lossett.calc.group_increments import latlon_shell_fields
from lossett.filtering.group_kernels import _central_kernel_series, _quadrature_moment_table
from moment_sharing_benchmark import exact_chebyshev, exact_ring_table
from multiscale_group_prototype import binned_coefficients

ROOT = Path("/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry")
DATA = Path("/gws/ssde/j25b/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/"
            "DYAMOND_SUMMER/benchmark_antialiased_20160801T00")
RADIUS = 6382758.772639869
NAMES = ("horizontal", "structure", "vertical_flux", "mixed_x", "mixed_y", "mixed_z", "mass")
BUILDER = dict(n_abel=400, n_central=400, n_abel_quad=160)


def moment_tables(scales, n_fiber=128, n_panels=64, order=12):
    """Piecewise-Chebyshev (g, h, k) tables, shape (scales, panels, order+1, 3), and their support angles."""
    tables = [_quadrature_moment_table(float(s), BUILDER["n_abel"], BUILDER["n_central"], BUILDER["n_abel_quad"],
                                       n_fiber, n_panels, order) for s in scales]
    return np.stack([t[1] for t in tables]), np.array([t[0] for t in tables])


def evaluate_moment_table(mu, coefficients, support):
    """Evaluate one scale's table at cosines mu, returning (3, len(mu)); zero outside the support."""
    mu = np.asarray(mu, float)
    distance = np.arccos(np.clip(mu, -1, 1))
    out = np.zeros((3, mu.size))
    inside = distance < support
    panels = coefficients.shape[0]
    position = distance[inside]/support*panels
    panel = np.minimum(position.astype(int), panels-1)
    local = 2*(position-panel)-1
    for index in np.unique(panel):
        mask = panel == index
        for c in range(3):
            out[c, np.flatnonzero(inside)[mask]] = np.polynomial.chebyshev.chebval(local[mask], coefficients[index][:, c])
    return out


@njit(cache=True)
def build_ring_table(x, directions, indices, supports, coefficients, table):
    """Fill table (scale, row, lag, 3) for an origin x at column 0 of its ring."""
    nlon = table.shape[2]
    npanel, ncoef = coefficients.shape[1], coefficients.shape[2]
    for j in range(len(indices)):
        target = indices[j]
        y = directions[target]
        mu = min(1., max(-1., x[0]*y[0]+x[1]*y[1]+x[2]*y[2]))
        distance = math.acos(mu)
        row, lag = target//nlon, target % nlon
        for scale in range(table.shape[0]):
            if distance >= supports[scale]:
                continue
            position = distance/supports[scale]*npanel
            panel = min(int(position), npanel-1)
            t = 2*(position-panel)-1
            for c in range(3):
                b0 = b1 = 0.
                for k in range(ncoef-1, 0, -1):
                    b0, b1 = 2*t*b0-b1+coefficients[scale, panel, k, c], b0
                table[scale, row, lag, c] = t*b0-b1+coefficients[scale, panel, 0, c]


def quadrature_kernel_nodes(support, bins, scales):
    """Nonnegative central G values at uniform rotation-angle nodes, with unit Haar mass."""
    angles = np.linspace(0, support, bins+1)
    nodes, gauss_weights = leggauss(8)
    fraction = (nodes+1)/2
    quadrature_angles = angles[:-1, None]+np.diff(angles)[:, None]*fraction
    haar = 2/np.pi*np.sin(quadrature_angles/2)**2
    out = np.zeros((len(scales), bins+1))
    for i, scale in enumerate(scales):
        series = _central_kernel_series(float(scale), BUILDER["n_abel"], BUILDER["n_central"], BUILDER["n_abel_quad"])
        inside = angles < 4*scale
        values = np.polynomial.chebyshev.chebval(np.clip(angles[inside]/(2*scale)-1, -1, 1), series)
        values = values*np.sign(values[0])  # the series carries an arbitrary overall sign
        if np.min(values) < -1e-10*np.max(values):
            raise ValueError("central kernel negativity exceeds tolerance")
        out[i, inside] = np.maximum(values, 0)
        interpolated = out[i, :-1, None]*(1-fraction)+out[i, 1:, None]*fraction
        out[i] /= np.sum(interpolated*haar*gauss_weights*np.diff(angles)[:, None]/2)
    return out


class DistanceTablePipeline:
    """Distance-binned coherent path: bin, expand through the fiber table, integrate."""

    def __init__(self, support, scales, rotation_bins, distance_bins):
        self.support, self.distance_bins = float(support), distance_bins
        start = time.perf_counter()
        self.kernel = quadrature_kernel_nodes(support, rotation_bins, scales)
        self.kernel_seconds = time.perf_counter()-start
        start = time.perf_counter()
        thetas = np.linspace(0, support, distance_bins+1)
        self.table = fiber_hat_table(thetas, support, rotation_bins, *leggauss(8))
        self.table_seconds = time.perf_counter()-start

    def evaluate(self, x, u, directions, velocity, grad, gradw, weights, curvature, indices):
        summaries = binned_coefficients(x, u, directions, velocity, grad, gradw, weights, curvature, indices,
                                        self.support, self.distance_bins)
        return integrate_group_angle_bins(expand_distance_bins(summaries, self.table), self.kernel)


def best_of(function, repeats=3):
    best, result = float("inf"), None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter()-start)
    return best, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=("n320", "n640"), required=True)
    parser.add_argument("--supports-km", type=float, nargs=2, default=(200., 1000.), help="G radius range")
    parser.add_argument("--nscales", type=int, default=10)
    parser.add_argument("--row-fractions", type=float, nargs="+", default=(0.1, 0.25, 0.5))
    parser.add_argument("--columns", type=int, default=16, help="sampled origins per ring")
    parser.add_argument("--rotation-bins", type=int, default=1024)
    parser.add_argument("--distance-bins", type=int, default=8192)
    parser.add_argument("--tag", default="20261125")
    args = parser.parse_args()
    out = ROOT/f"ring_fair_comparison_{args.grid}_{args.tag}"
    out.mkdir(exist_ok=False)
    with xr.open_dataset(DATA/f"filtered_{args.grid}.nc", decode_timedelta=False) as ds:
        field = ds.isel(time=0).sel(pressure=200.).load()
    lat, lon = np.deg2rad(field.latitude.values), np.deg2rad(field.longitude.values)
    nlat, nlon = len(lat), len(lon)
    weights = _solid_angles(lat, lon).ravel()
    direction, velocity, grad, gradw = latlon_shell_fields(
        lat, lon, *(field[name].values for name in ("u", "v", "w")), radius=RADIUS)
    direction, velocity = direction.reshape(-1, 3), velocity.reshape(-1, 3)
    grad, gradw = grad.reshape(-1, 3, 3), gradw.reshape(-1, 3)
    curvature = 1/RADIUS
    supports_km = np.geomspace(*args.supports_km, args.nscales)
    scales = supports_km*1000/(4*RADIUS)

    start = time.perf_counter()
    coefficients, supports = moment_tables(scales)
    table_build_seconds = time.perf_counter()-start
    artifact = out/"moment_tables_1d.npz"
    np.savez(artifact, coefficients=coefficients, supports=supports, supports_km=supports_km)
    max_support = float(supports.max())
    pipeline = DistanceTablePipeline(max_support, scales, args.rotation_bins, args.distance_bins)
    spacing = max(np.pi/(nlat-1), 2*np.pi/nlon)*RADIUS/1000
    one = np.array([0], dtype=np.int64)
    exact_chebyshev(direction[0], velocity[0], direction, velocity, grad, gradw, weights, curvature, one,
                    supports, coefficients)
    exact_ring_table(direction[0], velocity[0], direction, velocity, grad, gradw, weights,
                     np.zeros((len(scales), nlat, nlon, 3)), curvature, one, nlon, 0)
    build_ring_table(direction[0], direction, one, supports, coefficients, np.zeros((len(scales), nlat, nlon, 3)))
    pipeline.evaluate(direction[0], velocity[0], direction, velocity, grad, gradw, weights, curvature, one)

    records = []
    for fraction in args.row_fractions:
        row = int((nlat-1)*fraction)
        x_ring = direction[row*nlon]
        active_ring = np.flatnonzero(np.clip(direction@x_ring, -1, 1) >= np.cos(max_support))
        table = np.zeros((len(scales), nlat, nlon, 3))
        build_seconds, _ = best_of(lambda: build_ring_table(x_ring, direction, active_ring, supports,
                                                            coefficients, table), 3)
        columns = np.linspace(0, nlon, args.columns, endpoint=False).astype(int)
        per_origin = []
        for col in columns:
            index = row*nlon+col
            x, u = direction[index], velocity[index]
            selection_seconds, active = best_of(lambda: np.flatnonzero(np.clip(direction@x, -1, 1) >= np.cos(max_support)))
            a_seconds, reference = best_of(lambda: exact_chebyshev(
                x, u, direction, velocity, grad, gradw, weights, curvature, active, supports, coefficients))
            b_seconds, shared = best_of(lambda: exact_ring_table(
                x, u, direction, velocity, grad, gradw, weights, table, curvature, active, nlon, col))
            c_seconds, binned = best_of(lambda: pipeline.evaluate(
                x, u, direction, velocity, grad, gradw, weights, curvature, active))
            peaks = np.maximum(np.max(np.abs(reference), axis=0), np.finfo(float).tiny)
            per_origin.append({
                "col": int(col), "active_pairs": len(active), "selection_seconds": selection_seconds,
                "A_exact_per_origin_seconds": a_seconds, "B_ring_lookup_accumulate_seconds": b_seconds,
                "C_distance_table_seconds": c_seconds,
                "B_max_abs_difference_from_A": float(np.max(np.abs(shared-reference))),
                "C_peak_normalized_error_by_scale_and_output": (np.abs(binned-reference)/peaks).tolist(),
            })
        mean = lambda key: float(np.mean([p[key] for p in per_origin]))
        records.append({
            "row": row, "latitude_degrees": float(np.rad2deg(lat[row])), "ring_origins": nlon,
            "ring_active_pairs": len(active_ring), "ring_table_build_seconds": build_seconds,
            "ring_table_dense_bytes": table.nbytes,
            "mean_selection_seconds": mean("selection_seconds"),
            "mean_A_seconds": mean("A_exact_per_origin_seconds"),
            "mean_B_lookup_seconds": mean("B_ring_lookup_accumulate_seconds"),
            "mean_C_seconds": mean("C_distance_table_seconds"),
            "per_ring_total_seconds": {
                "A": nlon*(mean("selection_seconds")+mean("A_exact_per_origin_seconds")),
                "B": build_seconds+nlon*(mean("selection_seconds")+mean("B_ring_lookup_accumulate_seconds")),
                "C": nlon*(mean("selection_seconds")+mean("C_distance_table_seconds"))},
            "origins": per_origin})
        print(f"Finished row {row}", flush=True)
        del table
    with (out/"results.json").open("x") as stream:
        json.dump({"host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"), "grid": args.grid,
                   "largest_spacing_km": spacing, "G_radius_km": supports_km.tolist(),
                   "diameter_in_largest_spacings": (2*supports_km/spacing).tolist(),
                   "one_off": {"moment_tables_1d_build_seconds": table_build_seconds,
                               "moment_tables_1d_bytes": coefficients.nbytes,
                               "artifact_file_bytes": artifact.stat().st_size,
                               "kernel_nodes_seconds": pipeline.kernel_seconds,
                               "fiber_table_seconds": pipeline.table_seconds,
                               "fiber_table_bytes": pipeline.table.nbytes,
                               "rotation_bins": args.rotation_bins, "distance_bins": args.distance_bins},
                   "records": records, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "caveats": "best-of-three warm timings on a shared node; sampled origins per ring; "
                              "pair selection included in per-ring totals; I/O of the field and output excluded; "
                              "cell-area weights, real field, one time/level"},
                  stream, indent=1)
        stream.write("\n")


if __name__ == "__main__":
    main()
