"""Benchmark chunked disk-precomputed group pairs and longitude FFTs."""
import argparse
import json
import os
import resource
import socket
import time
from pathlib import Path

import numpy as np
import xarray as xr

from disk_group_fft import build_kernels, evaluate, prepare_fields
from disk_group_numba import accumulate
from lossett.calc._group_numba import accumulate_group_pairs
from lossett.calc.compute_group_increments import _solid_angles
from lossett.calc.group_increments import _transport, latlon_shell_fields
from lossett.filtering.group_kernels import mollifier_fiber_moments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-file", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--radius", type=float, default=6382758.772639869)
    parser.add_argument("--scale", type=float, default=0.1566720654220191)
    parser.add_argument("--exclude-pole-origins", action="store_true")
    parser.add_argument("--repair-pole-vectors", action="store_true")
    parser.add_argument(
        "--fft-kernel-dtype", choices=("complex128", "complex64"), default="complex128",
        help="Storage precision of FFT kernel archives; field calculations remain float64/complex128.",
    )
    parser.add_argument(
        "--fft-max-peak-relative-error", type=float,
        help="Required for complex64: per-channel maximum error divided by the reference peak.",
    )
    args = parser.parse_args()
    if args.rows < 1 or args.repeats < 1:
        parser.error("rows and repeats must be positive")
    if args.fft_kernel_dtype == "complex64" and args.fft_max_peak_relative_error is None:
        parser.error("--fft-max-peak-relative-error is required with complex64 storage")
    if args.fft_max_peak_relative_error is not None and (
        not np.isfinite(args.fft_max_peak_relative_error) or args.fft_max_peak_relative_error <= 0
    ):
        parser.error("--fft-max-peak-relative-error must be positive and finite")
    args.output_dir.mkdir(exist_ok=False)
    pair_dir, fft_dir = args.output_dir / "pair_archive", args.output_dir / "fft_archive"
    pair_dir.mkdir()
    fft_dir.mkdir()
    start = time.perf_counter()
    with xr.open_dataset(args.velocity_file, decode_timedelta=False) as source:
        field = source.isel(time=0).sel(pressure=200.).load()
    input_read = time.perf_counter() - start
    lat = np.deg2rad(np.asarray(field.latitude.values, dtype=np.float64))
    lon = np.deg2rad(np.asarray(field.longitude.values, dtype=np.float64))
    nlat, nlon = len(lat), len(lon)
    if args.exclude_pole_origins:
        if not (np.isclose(abs(lat[0]), np.pi / 2) and np.isclose(abs(lat[-1]), np.pi / 2)):
            parser.error("--exclude-pole-origins requires a grid with pole endpoints")
        rows = np.unique(np.linspace(1, nlat - 2, args.rows, dtype=int))
    else:
        rows = np.unique(np.linspace(0, nlat - 1, args.rows, dtype=int))
    velocity_fields = [
        np.array(field[name].values, dtype=np.float64, copy=True) for name in ("u", "v", "w")
    ]
    polar_repairs = []
    if args.repair_pole_vectors:
        east = np.stack((-np.sin(lon), np.cos(lon), np.zeros(nlon)), axis=-1)
        for row in (0, nlat - 1):
            if not np.isclose(abs(lat[row]), np.pi / 2):
                continue
            north = np.stack((
                -np.sin(lat[row]) * np.cos(lon),
                -np.sin(lat[row]) * np.sin(lon),
                np.full(nlon, np.cos(lat[row])),
            ), axis=-1)
            radial = np.stack((
                np.cos(lat[row]) * np.cos(lon),
                np.cos(lat[row]) * np.sin(lon),
                np.full(nlon, np.sin(lat[row])),
            ), axis=-1)
            cartesian = (
                velocity_fields[0][row, :, None] * east
                + velocity_fields[1][row, :, None] * north
                + velocity_fields[2][row, :, None] * radial
            )
            pole_velocity = cartesian.mean(axis=0)
            deviation = np.linalg.norm(cartesian - pole_velocity, axis=1)
            original = [component[row].copy() for component in velocity_fields]
            velocity_fields[0][row] = east @ pole_velocity
            velocity_fields[1][row] = north @ pole_velocity
            velocity_fields[2][row] = radial @ pole_velocity
            polar_repairs.append({
                "row": row,
                "max_cartesian_deviation_m_s": float(deviation.max()),
                "relative_max_cartesian_deviation": float(
                    deviation.max() / max(np.linalg.norm(pole_velocity), np.finfo(float).tiny)
                ),
                "max_component_change_m_s": float(max(
                    np.max(np.abs(component[row] - previous))
                    for component, previous in zip(velocity_fields, original)
                )),
            })
    start = time.perf_counter()
    weights = _solid_angles(lat, lon)
    directions, velocity, grad, gradw = latlon_shell_fields(
        lat, lon, *velocity_fields, radius=args.radius,
    )
    geometry_setup = time.perf_counter() - start
    d = directions.reshape(-1, 3)
    u = velocity.reshape(-1, 3)
    grad_flat, gw_flat, wt = grad.reshape(-1, 3, 3), gradw.reshape(-1, 3), weights.ravel()
    cosine = np.cos(min(4 * args.scale, np.pi))
    report = {
        "host": socket.gethostname(), "velocity_file": str(Path(args.velocity_file).resolve()),
        "grid_shape": [nlat, nlon], "origin_rows": rows.tolist(),
        "origin_latitudes_degrees": np.rad2deg(lat[rows]).tolist(),
        "number_of_origins": len(rows) * nlon, "radius_m": args.radius, "mollifier_scale": args.scale,
        "scope": "selected latitude rings, all longitudes; reports input, field, archive, compute and output stages",
        "origin_row_policy": "exclude poles" if args.exclude_pole_origins else "include grid endpoints",
        "polar_vector_repairs": polar_repairs,
        "input_read_seconds": input_read,
        "field_geometry_setup_seconds": geometry_setup,
        "selected_ring_count": int(len(rows)),
        "selected_output_shape": [len(rows), nlon, 7],
        "fft_kernel_storage_dtype": args.fft_kernel_dtype,
        "fft_max_peak_relative_error": args.fft_max_peak_relative_error,
        "results": {},
    }

    def save():
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    def load_archive(path, cold_advice):
        with path.open("rb") as stream:
            if cold_advice and hasattr(os, "posix_fadvise"):
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            with np.load(stream) as archive:
                return {name: archive[name] for name in archive.files}

    def minimal(row):
        x = d[row * nlon]
        mu = np.clip(d @ x, -1, 1)
        active = np.flatnonzero(mu >= cosine - 64 * np.finfo(float).eps)
        moments = np.stack(mollifier_fiber_moments(mu[active], args.scale), axis=1)
        return active, moments

    def rotation(row, active):
        x = d[row * nlon]
        return np.stack([_transport(np.eye(3)[a], x, d[active]) for a in range(3)], axis=-1)

    def run_pair(row, active, moments, matrices=None):
        result = np.empty((nlon, 7))
        moment_columns = tuple(np.ascontiguousarray(moments[:, j]) for j in range(3))
        for col in range(nlon):
            i = row * nlon + col
            indices = (active // nlon) * nlon + (active % nlon + col) % nlon
            if matrices is None:
                v = accumulate_group_pairs(d[i], u[i], d, u, grad_flat, gw_flat, wt,
                                           *moment_columns, 1 / args.radius, indices)
            else:
                v = accumulate(d[i], u[i], d, u, grad_flat, gw_flat, wt, moments, matrices,
                               indices, 1 / args.radius, 2 * np.pi * col / nlon)
            result[col] = v
        return result

    active, moments = minimal(int(rows[0]))
    matrix = rotation(int(rows[0]), active)
    start = time.perf_counter()
    indices = active[:2]
    moment_columns = tuple(np.ascontiguousarray(moments[:2, j]) for j in range(3))
    accumulate_group_pairs(d[0], u[0], d, u, grad_flat, gw_flat, wt,
                           *moment_columns, 1 / args.radius, indices)
    accumulate(d[0], u[0], d, u, grad_flat, gw_flat, wt, moments[:2], matrix[:2],
               indices, 1 / args.radius, 0.)
    report["numba_compile_or_cache_load_seconds"] = time.perf_counter() - start

    baseline = {}
    pair_runs = []
    for iteration in range(args.repeats):
        prep = compute = 0.
        for row in rows:
            start = time.perf_counter()
            active, moments = minimal(row)
            prep += time.perf_counter() - start
            start = time.perf_counter()
            value = run_pair(row, active, moments)
            compute += time.perf_counter() - start
            if iteration == 0:
                baseline[int(row)] = value
            else:
                np.testing.assert_array_equal(value, baseline[int(row)])
            if np.max(np.abs(value[:, 6] - 1)) > 0.01:
                raise ValueError(f"kernel-mass tolerance exceeded at row {row}")
        pair_runs.append({"moment_setup_seconds": prep, "pair_compute_seconds": compute, "total_seconds": prep + compute})
    report["results"]["numba_recompute"] = pair_runs
    print("numba recompute", pair_runs, flush=True)
    save()

    archive_setup = 0.
    archive_write = 0.
    for row in rows:
        start = time.perf_counter()
        active, moments = minimal(row)
        matrices = rotation(row, active)
        archive_setup += time.perf_counter() - start
        start = time.perf_counter()
        with (pair_dir / f"{row:05d}.npz").open("xb") as stream:
            np.savez(stream, active=active, moments=moments, matrices=matrices)
            stream.flush()
            os.fsync(stream.fileno())
        archive_write += time.perf_counter() - start
    pair_size = sum(path.stat().st_size for path in pair_dir.iterdir())
    report["results"]["pair_archive_creation"] = {
        "geometry_and_moments_seconds": archive_setup, "write_seconds": archive_write, "bytes": pair_size,
    }
    save()
    print("pair archive", report["results"]["pair_archive_creation"], flush=True)

    names = ("horizontal", "structure", "vertical_flux", "mixed_x", "mixed_y", "mixed_z", "mass")
    def errors(value, ref):
        np.testing.assert_allclose(value, ref, rtol=2e-9, atol=2e-9)
        return {name: float(np.max(np.abs(value[:, j] - ref[:, j]))) for j, name in enumerate(names)}

    disk_runs = []
    for iteration in range(args.repeats):
        read = compute = 0.
        output_write = 0.
        worst = dict.fromkeys(names, 0.)
        output = None
        if iteration == 0:
            output = np.lib.format.open_memmap(
                args.output_dir / "numba_output.npy", mode="w+", dtype=np.float64,
                shape=(len(rows), nlon, 7),
            )
        for row_index, row in enumerate(rows):
            start = time.perf_counter()
            archive = load_archive(pair_dir / f"{row:05d}.npz", iteration == 0)
            active, moments, matrices = archive["active"], archive["moments"], archive["matrices"]
            read += time.perf_counter() - start
            start = time.perf_counter()
            value = run_pair(row, active, moments, matrices)
            compute += time.perf_counter() - start
            if output is not None:
                start = time.perf_counter()
                output[row_index] = value
                output_write += time.perf_counter() - start
            for name, value_error in errors(value, baseline[int(row)]).items():
                worst[name] = max(worst[name], value_error)
        if output is not None:
            start = time.perf_counter()
            output.flush()
            output_write += time.perf_counter() - start
            del output
        disk_runs.append({"read_cache": "cold-advised, not guaranteed" if iteration == 0 else "warm/unknown",
                          "read_seconds": read, "pair_compute_seconds": compute, "total_seconds": read + compute,
                          "output_write_seconds": output_write if iteration == 0 else None,
                          "output_bytes": (args.output_dir / "numba_output.npy").stat().st_size if iteration == 0 else None,
                          "max_absolute_errors": worst})
    report["results"]["numba_disk"] = disk_runs
    save()
    print("numba disk", disk_runs, flush=True)

    start = time.perf_counter()
    fields = prepare_fields(lat, lon, velocity, grad, gradw, weights, args.radius)
    field_setup = time.perf_counter() - start
    report["results"]["fft_field_prepare_bytes"] = int(
        sum(value.nbytes for value in fields["spectra"].values())
        + sum(fields[name].nbytes for name in ("radial", "w", "v", "speed2", "rotation"))
    )
    fft_runs = []
    for _ in range(args.repeats):
        setup = compute = 0.
        worst = dict.fromkeys(names, 0.)
        for row in rows:
            start = time.perf_counter()
            target_rows, kernels = build_kernels(lat, nlon, row, args.scale)
            setup += time.perf_counter() - start
            start = time.perf_counter()
            value = evaluate(fields, row, target_rows, kernels)
            compute += time.perf_counter() - start
            for name, value_error in errors(value, baseline[int(row)]).items():
                worst[name] = max(worst[name], value_error)
        fft_runs.append({"kernel_setup_seconds": setup, "fft_compute_seconds": compute,
                         "total_seconds": setup + compute, "max_absolute_errors": worst})
    report["results"]["fft_field_prepare_seconds"] = field_setup
    report["results"]["fft_recompute"] = fft_runs
    save()
    print("fft recompute", fft_runs, flush=True)

    fft_setup = fft_cast = fft_write = fft_verify = 0.
    for row in rows:
        start = time.perf_counter()
        target_rows, kernels = build_kernels(lat, nlon, row, args.scale)
        fft_setup += time.perf_counter() - start
        start = time.perf_counter()
        stored_kernels = kernels.astype(args.fft_kernel_dtype, copy=False)
        if not np.isfinite(stored_kernels).all():
            raise ValueError(f"Non-finite stored FFT kernel values at row {row}")
        fft_cast += time.perf_counter() - start
        path = fft_dir / f"{row:05d}.npz"
        start = time.perf_counter()
        with path.open("xb") as stream:
            np.savez(stream, target_rows=target_rows, kernels=stored_kernels)
            stream.flush()
            os.fsync(stream.fileno())
        fft_write += time.perf_counter() - start
        start = time.perf_counter()
        with np.load(path) as stored:
            if stored["kernels"].dtype != np.dtype(args.fft_kernel_dtype):
                raise ValueError(f"Unexpected stored FFT kernel dtype at row {row}")
            np.testing.assert_array_equal(stored["target_rows"], target_rows)
            np.testing.assert_array_equal(stored["kernels"], stored_kernels)
        fft_verify += time.perf_counter() - start
    report["results"]["fft_archive_creation"] = {
        "kernel_setup_seconds": fft_setup, "cast_and_check_seconds": fft_cast,
        "write_seconds": fft_write, "readback_verification_seconds": fft_verify,
        "bytes": sum(path.stat().st_size for path in fft_dir.iterdir()), "channels": kernels.shape[0],
        "kernel_dtype": args.fft_kernel_dtype,
    }
    save()
    print("fft archive", report["results"]["fft_archive_creation"], flush=True)

    disk_fft_runs = []
    for iteration in range(args.repeats):
        read = compute = 0.
        output_write = 0.
        worst = dict.fromkeys(names, 0.)
        reference_peak = dict.fromkeys(names, 0.)
        output = None
        if iteration == 0:
            output = np.lib.format.open_memmap(
                args.output_dir / "fft_output.npy", mode="w+", dtype=np.float64,
                shape=(len(rows), nlon, 7),
            )
        for row_index, row in enumerate(rows):
            start = time.perf_counter()
            archive = load_archive(fft_dir / f"{row:05d}.npz", iteration == 0)
            target_rows, kernels = archive["target_rows"], archive["kernels"]
            if kernels.dtype != np.dtype(args.fft_kernel_dtype):
                raise ValueError(f"Unexpected FFT kernel dtype at row {row}")
            read += time.perf_counter() - start
            start = time.perf_counter()
            value = evaluate(fields, row, target_rows, kernels)
            compute += time.perf_counter() - start
            if output is not None:
                start = time.perf_counter()
                output[row_index] = value
                output_write += time.perf_counter() - start
            reference = baseline[int(row)]
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite FFT output at row {row}")
            if args.fft_kernel_dtype == "complex128":
                errors(value, reference)
            for j, name in enumerate(names):
                worst[name] = max(worst[name], float(np.max(np.abs(value[:, j] - reference[:, j]))))
                reference_peak[name] = max(reference_peak[name], float(np.max(np.abs(reference[:, j]))))
        if output is not None:
            start = time.perf_counter()
            output.flush()
            output_write += time.perf_counter() - start
            del output
        disk_fft_runs.append({"read_cache": "cold-advised, not guaranteed" if iteration == 0 else "warm/unknown",
                              "read_seconds": read, "fft_compute_seconds": compute, "total_seconds": read + compute,
                              "output_write_seconds": output_write if iteration == 0 else None,
                              "output_bytes": (args.output_dir / "fft_output.npy").stat().st_size if iteration == 0 else None,
                              "max_absolute_errors": worst, "reference_peak_absolute": reference_peak})
        report["results"]["fft_disk"] = disk_fft_runs
        save()
        if args.fft_max_peak_relative_error is not None:
            for name in names:
                if worst[name] > args.fft_max_peak_relative_error * reference_peak[name]:
                    raise ValueError(
                        f"FFT {name} peak-normalized error {worst[name]} exceeds "
                        f"{args.fft_max_peak_relative_error} times reference peak {reference_peak[name]}"
                    )
    print("fft disk", disk_fft_runs, flush=True)


if __name__ == "__main__":
    main()
