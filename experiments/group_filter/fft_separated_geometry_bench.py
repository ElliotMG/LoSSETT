"""Compare scale-independent FFT geometry reuse with the existing kernel builder."""

import argparse
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import xarray as xr

from disk_group_fft import PAIRS, build_kernels, evaluate, prepare_fields
from lossett.calc.compute_group_increments import _solid_angles
from lossett.calc.group_increments import _transport, latlon_shell_fields
from lossett.filtering.group_kernels import mollifier_fiber_moments


NAMES = ("horizontal", "structure", "vertical_flux", "mixed_x", "mixed_y", "mixed_z", "mass")


def build_geometry(lat, nlon, row, max_scale):
    rows = np.flatnonzero(np.abs(lat - lat[row]) <= min(4 * max_scale, np.pi) + 1e-14)
    target_lat = lat[rows]
    delta = 2 * np.pi * np.arange(nlon) / nlon
    c, s = np.cos(delta), np.sin(delta)
    y = np.stack((
        np.cos(target_lat)[:, None] * c,
        np.cos(target_lat)[:, None] * s,
        np.broadcast_to(np.sin(target_lat)[:, None], (len(rows), nlon)),
    ), axis=-1)
    x = np.array([np.cos(lat[row]), 0.0, np.sin(lat[row])])
    mu = np.clip(np.sum(y * x, axis=-1), -1, 1)
    matrix = np.empty((len(rows), nlon, 3, 3))
    for a in range(3):
        transported = _transport(np.eye(3)[a], x, y)
        matrix[..., 0, a] = c * transported[..., 0] + s * transported[..., 1]
        matrix[..., 1, a] = -s * transported[..., 0] + c * transported[..., 1]
        matrix[..., 2, a] = transported[..., 2]
    return rows, mu, matrix


def kernels_from_geometry(lat, row, scale, rows, mu, matrix):
    selected = np.abs(lat[rows] - lat[row]) <= min(4 * scale, np.pi) + 1e-14
    target_rows = rows[selected]
    mu = mu[selected]
    matrix = matrix[selected]
    g, h, k = mollifier_fiber_moments(mu, scale)
    kernels = [g, k]
    kernels.extend(h * matrix[..., b, a] for a in range(3) for b in range(3))
    for a, aa in PAIRS:
        for b, bb in PAIRS:
            value = matrix[..., b, a] * matrix[..., bb, aa]
            if b != bb:
                value = value + matrix[..., bb, a] * matrix[..., b, aa]
            kernels.append(k * value)
    return target_rows, np.fft.rfft(np.stack(kernels), axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--scales", type=float, nargs="+", required=True)
    parser.add_argument("--radius", type=float, default=6382758.772639869)
    args = parser.parse_args()
    if args.rows < 1 or any(not np.isfinite(s) or not 0 < s < np.pi / 2 for s in args.scales):
        parser.error("rows must be positive and scales must be between 0 and pi/2")
    args.output_dir.mkdir(exist_ok=False)
    with xr.open_dataset(args.velocity_file, decode_timedelta=False) as source:
        field = source.isel(time=0).sel(pressure=200.0).load()
    lat = np.deg2rad(np.asarray(field.latitude.values, dtype=np.float64))
    lon = np.deg2rad(np.asarray(field.longitude.values, dtype=np.float64))
    weights = _solid_angles(lat, lon)
    _, velocity, grad, gradw = latlon_shell_fields(
        lat, lon, *(field[name].values for name in ("u", "v", "w")), radius=args.radius,
    )
    fields = prepare_fields(lat, lon, velocity, grad, gradw, weights, args.radius)
    rows = np.unique(np.linspace(0, len(lat) - 1, args.rows, dtype=int))
    coefficient_setup = {}
    for scale in args.scales:
        start = time.perf_counter()
        mollifier_fiber_moments(np.array([1.0]), scale)
        coefficient_setup[str(scale)] = time.perf_counter() - start
    records = []
    for row in rows:
        start = time.perf_counter()
        target_rows, mu, matrix = build_geometry(lat, len(lon), row, max(args.scales))
        geometry_seconds = time.perf_counter() - start
        path = args.output_dir / f"geometry_{row:05d}.npz"
        start = time.perf_counter()
        with path.open("xb") as stream:
            np.savez(stream, target_rows=target_rows, mu=mu, matrix=matrix)
            stream.flush()
            os.fsync(stream.fileno())
        write_seconds = time.perf_counter() - start
        start = time.perf_counter()
        with np.load(path, allow_pickle=False) as archive:
            restored = tuple(archive[name] for name in ("target_rows", "mu", "matrix"))
        read_seconds = time.perf_counter() - start
        for original, loaded in zip((target_rows, mu, matrix), restored):
            np.testing.assert_array_equal(original, loaded)
        record = {
            "row": int(row), "latitude_degrees": float(np.rad2deg(lat[row])),
            "geometry_seconds": geometry_seconds, "write_seconds": write_seconds,
            "read_seconds": read_seconds, "geometry_bytes": path.stat().st_size, "scales": [],
        }
        for scale in args.scales:
            start = time.perf_counter()
            direct_rows, direct_kernels = build_kernels(lat, len(lon), row, scale)
            direct_seconds = time.perf_counter() - start
            start = time.perf_counter()
            split_rows, split_kernels = kernels_from_geometry(lat, row, scale, *restored)
            split_seconds = time.perf_counter() - start
            np.testing.assert_array_equal(split_rows, direct_rows)
            np.testing.assert_array_equal(split_kernels, direct_kernels)
            expected = evaluate(fields, row, direct_rows, direct_kernels)
            actual = evaluate(fields, row, split_rows, split_kernels)
            np.testing.assert_array_equal(actual, expected)
            record["scales"].append({
                "angular_scale": scale, "support_km": 4 * scale * args.radius / 1000,
                "direct_build_seconds": direct_seconds, "split_build_seconds": split_seconds,
                "fft_kernel_bytes": int(split_kernels.nbytes + split_rows.nbytes),
                "max_absolute_errors": {
                    name: float(np.max(np.abs(actual[:, index] - expected[:, index])))
                    for index, name in enumerate(NAMES)
                },
            })
        records.append(record)
        print(json.dumps(record), flush=True)
    with (args.output_dir / "results.json").open("x") as stream:
        json.dump({
            "host": socket.gethostname(), "velocity_file": str(args.velocity_file.resolve()),
            "grid_shape": [len(lat), len(lon)], "radius_m": args.radius,
            "scope": "sampled rings; one repetition; coefficient tables prewarmed; I/O cache uncontrolled",
            "coefficient_setup_seconds": coefficient_setup,
            "records": records,
        }, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
