"""Test reusing the existing float32 spherical geometry for group increments."""
import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np
import xarray as xr

from disk_group_numba import accumulate
from lossett.calc._group_numba import accumulate_group_pairs
from lossett.calc.compute_group_increments import _solid_angles
from lossett.calc.group_increments import _transport, latlon_shell_fields
from lossett.filtering.group_kernels import mollifier_fiber_moments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-file", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=6382758.772639869)
    parser.add_argument("--scale", type=float, default=0.1566720654220191)
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with xr.open_dataset(args.velocity_file, decode_timedelta=False) as source:
        field = source.isel(time=0).sel(pressure=200.).load()
    lat, lon = np.deg2rad(field.latitude.values), np.deg2rad(field.longitude.values)
    nlat, nlon = len(lat), len(lon)
    rows = np.unique(np.linspace(0, nlat - 1, args.rows, dtype=int))
    weights = _solid_angles(lat, lon).ravel()
    d, u, grad, gradw = latlon_shell_fields(
        lat, lon, *(field[name].values for name in ("u", "v", "w")), radius=args.radius,
    )
    d, u, grad, gradw = d.reshape(-1, 3), u.reshape(-1, 3), grad.reshape(-1, 3, 3), gradw.reshape(-1, 3)
    scale_cos = np.cos(min(4 * args.scale, np.pi))
    names = ("horizontal", "structure", "vertical_flux", "mixed_x", "mixed_y", "mixed_z", "mass")
    report = {"host": socket.gethostname(), "scope": "sampled latitude rings, all longitudes; in-memory fields",
              "geometry": args.geometry, "origin_rows": rows.tolist(), "runs": []}

    def run(row, active, moments, matrices=None):
        result = np.empty((nlon, 7))
        moment_columns = tuple(np.ascontiguousarray(moments[:, j]) for j in range(3))
        for col in range(nlon):
            i = row * nlon + col
            indices = (active // nlon) * nlon + (active % nlon + col) % nlon
            if matrices is None:
                result[col] = accumulate_group_pairs(d[i], u[i], d, u, grad, gradw, weights,
                                                     *moment_columns, 1 / args.radius, indices)
            else:
                result[col] = accumulate(d[i], u[i], d, u, grad, gradw, weights, moments, matrices,
                                         indices, 1 / args.radius, 2 * np.pi * col / nlon)
        return result

    # Compile or load both signatures outside measured runs.
    i = int(rows[0]) * nlon
    active = np.array([i, i])
    moments = np.ones((2, 3))
    moment_columns = tuple(np.ascontiguousarray(moments[:, j]) for j in range(3))
    accumulate_group_pairs(d[i], u[i], d, u, grad, gradw, weights, *moment_columns, 1 / args.radius, active)
    accumulate(d[i], u[i], d, u, grad, gradw, weights, moments, np.repeat(np.eye(3)[None], 2, axis=0),
               active, 1 / args.radius, 0.)
    reference = {}
    for iteration in range(2):
        setup = compute = 0.
        for row in rows:
            start = time.perf_counter()
            mu = np.clip(d @ d[row * nlon], -1, 1)
            active = np.flatnonzero(mu >= scale_cos - 64 * np.finfo(float).eps)
            moments = np.stack(mollifier_fiber_moments(mu[active], args.scale), axis=1)
            setup += time.perf_counter() - start
            start = time.perf_counter()
            value = run(row, active, moments)
            compute += time.perf_counter() - start
            if iteration == 0:
                reference[int(row)] = (active, value)
            else:
                np.testing.assert_array_equal(value, reference[int(row)][1])
        report["runs"].append({"method": "recompute", "setup_seconds": setup, "compute_seconds": compute,
                               "total_seconds": setup + compute})
        print(json.dumps(report["runs"][-1]), flush=True)

    variables = ("great_circle_distance", "sine_initial_bearing", "cosine_initial_bearing",
                 "sine_final_bearing", "cosine_final_bearing")
    with xr.open_zarr(args.geometry) as geometry:
        np.testing.assert_allclose(np.deg2rad(geometry.latitude.values), lat, rtol=0, atol=1e-14)
        np.testing.assert_allclose(np.deg2rad(geometry.origin_latitude.values), lat, rtol=0, atol=1e-14)
        delta = np.deg2rad(geometry.longitude.values)
        zero = int(np.argmin(np.abs(delta)))
        np.testing.assert_allclose((delta[(np.arange(nlon) + zero) % nlon] - 2 * np.pi * np.arange(nlon) / nlon
                                    + np.pi) % (2 * np.pi) - np.pi, 0, atol=1e-12)
        column_order = (np.arange(nlon) + zero) % nlon
        original_radius = geometry.attrs["sphere_radius_m"]
        report["archived_radius_m"] = original_radius
        reference_max = {
            name: max(float(np.max(np.abs(value[:, j]))) for _, value in reference.values())
            for j, name in enumerate(names)
        }
        report["reference_max_absolute"] = reference_max
        for iteration in range(2):
            read = setup = compute = 0.
            fallback_count = mismatched_targets = 0
            worst = dict.fromkeys(names, 0.)
            for row in rows:
                start = time.perf_counter()
                old = geometry[list(variables)].isel(origin_latitude=int(row)).load()
                read += time.perf_counter() - start
                start = time.perf_counter()
                distance = old["great_circle_distance"].values[:, column_order].ravel()
                mu = np.cos(distance.astype(np.float64) / original_radius)
                active = np.flatnonzero(mu >= scale_cos - 64 * np.finfo(float).eps)
                mismatched_targets += len(np.setxor1d(active, reference[int(row)][0]))
                moments = np.stack(mollifier_fiber_moments(mu[active], args.scale), axis=1)
                trig = [old[name].values[:, column_order].ravel()[active].astype(np.float64)
                        for name in variables[1:]]
                si, ci, sf, cf = trig
                x, y = d[row * nlon], d[active]
                lonx = lon[0]
                laty, lony = lat[active // nlon], lon[active % nlon]
                ex = np.array([-np.sin(lonx), np.cos(lonx), 0.])
                nx = np.array([-np.sin(lat[row]) * np.cos(lonx), -np.sin(lat[row]) * np.sin(lonx), np.cos(lat[row])])
                ey = np.stack((-np.sin(lony), np.cos(lony), np.zeros_like(lony)), axis=-1)
                ny = np.stack((-np.sin(laty) * np.cos(lony), -np.sin(laty) * np.sin(lony), np.cos(laty)), axis=-1)
                tx = si[:, None] * ex + ci[:, None] * nx
                ty = sf[:, None] * ey + cf[:, None] * ny
                matrices = (y[:, :, None] * x + ty[:, :, None] * tx[:, None, :]
                            + np.cross(y, ty)[:, :, None] * np.cross(x, tx)[:, None, :])
                invalid = ~np.isfinite(matrices).all(axis=(1, 2))
                fallback_count += int(invalid.sum())
                if np.any(invalid):
                    matrices[invalid] = np.stack(
                        [_transport(np.eye(3)[a], x, y[invalid]) for a in range(3)], axis=-1,
                    )
                setup += time.perf_counter() - start
                start = time.perf_counter()
                value = run(row, active, moments, matrices)
                compute += time.perf_counter() - start
                if not np.all(np.isfinite(value)):
                    raise ValueError(f"non-finite archived-geometry result at row {row}")
                ref = reference[int(row)][1]
                for j, name in enumerate(names):
                    worst[name] = max(worst[name], float(np.max(np.abs(value[:, j] - ref[:, j]))))
            result = {"method": "existing_float32_geometry", "read_cache": "warm/unknown",
                      "read_seconds": read, "conversion_and_moments_seconds": setup,
                      "compute_seconds": compute, "total_seconds": read + setup + compute,
                      "fallback_matrix_count": fallback_count, "changed_support_target_count": mismatched_targets,
                      "max_absolute_errors": worst,
                      "max_errors_over_global_reference_max": {
                          name: worst[name] / reference_max[name] if reference_max[name] else None for name in names
                      }}
            report["runs"].append(result)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
