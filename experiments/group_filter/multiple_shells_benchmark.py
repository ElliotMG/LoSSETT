"""Verify archived pair and FFT kernels on distinct real pressure shells."""
import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np
import xarray as xr

from disk_group_fft import evaluate, prepare_fields
from disk_group_numba import accumulate
from lossett.calc._group_numba import accumulate_group_pairs
from lossett.calc.compute_group_increments import _solid_angles
from lossett.calc.group_increments import latlon_shell_fields
from lossett.filtering.group_kernels import mollifier_fiber_moments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-file", required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads((args.archive_dir / "results.json").read_text())
    radius, scale = config["radius_m"], config["mollifier_scale"]
    rows = config["origin_rows"]
    with xr.open_dataset(args.velocity_file, decode_timedelta=False) as source:
        dataset = source.isel(time=0).sel(pressure=[150., 200., 250.]).load()
    lat, lon = np.deg2rad(dataset.latitude.values), np.deg2rad(dataset.longitude.values)
    nlat, nlon = len(lat), len(lon)
    if [nlat, nlon] != config["grid_shape"]:
        raise ValueError("archive and velocity grids do not match")
    weights = _solid_angles(lat, lon)
    report = {"host": socket.gethostname(), "scope": "five latitude rings, all longitudes",
              "radius_m": radius, "mollifier_scale": scale,
              "note": "The same test radius and angular kernel are used for each distinct pressure shell.",
              "runs": []}
    for level in (150., 200., 250.):
        field = dataset.sel(pressure=level)
        start = time.perf_counter()
        directions, velocity, grad, gradw = latlon_shell_fields(
            lat, lon, *(field[name].values for name in ("u", "v", "w")), radius=radius,
        )
        field_setup = time.perf_counter() - start
        d, u, gr, gw = directions.reshape(-1, 3), velocity.reshape(-1, 3), grad.reshape(-1, 3, 3), gradw.reshape(-1, 3)
        wt = weights.ravel()
        if level == 150.:
            start = time.perf_counter()
            dummy_indices = np.array([0, 1])
            dummy_moments = np.ones((2, 3))
            columns = tuple(np.ascontiguousarray(dummy_moments[:, j]) for j in range(3))
            accumulate_group_pairs(d[0], u[0], d, u, gr, gw, wt, *columns, 1 / radius, dummy_indices)
            accumulate(d[0], u[0], d, u, gr, gw, wt, dummy_moments,
                       np.repeat(np.eye(3)[None], 2, axis=0), dummy_indices, 1 / radius, 0.)
            report["numba_compile_or_cache_load_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        fft_fields = prepare_fields(lat, lon, velocity, grad, gradw, weights, radius)
        fft_setup = time.perf_counter() - start
        read_pair = compute_pair = read_fft = compute_fft = 0.
        pair_error = fft_error = 0.
        for row in rows:
            mu = np.clip(d @ d[row * nlon], -1, 1)
            active = np.flatnonzero(mu >= np.cos(min(4 * scale, np.pi)) - 64 * np.finfo(float).eps)
            moments = mollifier_fiber_moments(mu[active], scale)
            reference = np.empty((nlon, 7))
            for col in range(nlon):
                i = row * nlon + col
                indices = (active // nlon) * nlon + (active % nlon + col) % nlon
                reference[col] = accumulate_group_pairs(d[i], u[i], d, u, gr, gw, wt,
                                                       *moments, 1 / radius, indices)
            start = time.perf_counter()
            with np.load(args.archive_dir / "pair_archive" / f"{row:05d}.npz") as archive:
                cached_active, cached_moments, matrices = archive["active"], archive["moments"], archive["matrices"]
            read_pair += time.perf_counter() - start
            np.testing.assert_array_equal(cached_active, active)
            start = time.perf_counter()
            actual = np.empty_like(reference)
            for col in range(nlon):
                i = row * nlon + col
                indices = (active // nlon) * nlon + (active % nlon + col) % nlon
                actual[col] = accumulate(d[i], u[i], d, u, gr, gw, wt, cached_moments, matrices,
                                         indices, 1 / radius, 2 * np.pi * col / nlon)
            compute_pair += time.perf_counter() - start
            np.testing.assert_allclose(actual, reference, rtol=2e-9, atol=2e-9)
            pair_error = max(pair_error, float(np.max(np.abs(actual - reference))))
            start = time.perf_counter()
            with np.load(args.archive_dir / "fft_archive" / f"{row:05d}.npz") as archive:
                target_rows, kernels = archive["target_rows"], archive["kernels"]
            read_fft += time.perf_counter() - start
            start = time.perf_counter()
            fft_value = evaluate(fft_fields, row, target_rows, kernels)
            compute_fft += time.perf_counter() - start
            np.testing.assert_allclose(fft_value, reference, rtol=2e-9, atol=2e-9)
            fft_error = max(fft_error, float(np.max(np.abs(fft_value - reference))))
        result = {"pressure_hPa": level, "field_setup_seconds": field_setup,
                  "fft_field_prepare_seconds": fft_setup, "pair_archive_read_seconds": read_pair,
                  "pair_compute_seconds": compute_pair, "fft_archive_read_seconds": read_fft,
                  "fft_compute_seconds": compute_fft, "pair_max_absolute_error": pair_error,
                  "fft_max_absolute_error": fft_error, "read_cache": "warm/unknown"}
        report["runs"].append(result)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
