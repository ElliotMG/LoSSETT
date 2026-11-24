"""Benchmark stored scalar endpoint H applied twice against stored matched G."""

import argparse
import json
import os
import resource
import socket
import time
from pathlib import Path

import numpy as np
import xarray as xr

from lossett.calc.compute_group_increments import _solid_angles
from lossett.filtering.group_kernels import mollifier_fiber_moments, mollifier_les_fiber_moments


INPUT = Path("/gws/ssde/j25b/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/"
             "DYAMOND_SUMMER/benchmark_antialiased_20160801T00/filtered_2p0deg.nc")
OUTPUT = Path("/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/h_twice_scalar_2deg_20261124")
RADIUS = 6382758.772639869
SCALE = 1e6 / RADIUS
NAMES = ("constant", "radial_velocity", "speed_squared")


def build(lat, nlon, row, moment_function, support):
    rows = np.flatnonzero(np.abs(lat - lat[row]) <= support + 1e-14)
    delta = 2*np.pi*np.arange(nlon)/nlon
    mu = np.clip(
        np.sin(lat[row])*np.sin(lat[rows, None])
        + np.cos(lat[row])*np.cos(lat[rows, None])*np.cos(delta),
        -1, 1,
    )
    g, _, _ = moment_function(mu, SCALE)
    return rows, np.fft.rfft(g, axis=-1)


def apply_archive(path, array, weights):
    start = time.perf_counter()
    spectra = np.fft.rfft(array*weights[None, :, :], axis=2)
    prepare = time.perf_counter() - start
    result = np.empty_like(array)
    read = compute = 0.
    for row in range(array.shape[1]):
        start = time.perf_counter()
        with np.load(path / f"{row:04d}.npz", allow_pickle=False) as archive:
            target_rows = archive["target_rows"]
            kernel = archive["kernel"]
        read += time.perf_counter() - start
        start = time.perf_counter()
        result[:, row, :] = np.fft.irfft(
            np.sum(np.conj(kernel)[None, :, :] * spectra[:, target_rows, :], axis=1),
            n=array.shape[2], axis=-1,
        )
        compute += time.perf_counter() - start
    return result, {"field_transform_seconds": prepare, "archive_read_seconds": read,
                    "convolution_seconds": compute}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-file", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=False)
    with xr.open_dataset(args.velocity_file, decode_timedelta=False) as source:
        field = source.isel(time=0).sel(pressure=200.).load()
    lat = np.deg2rad(field.latitude.values)
    lon = np.deg2rad(field.longitude.values)
    u, v, w = (np.asarray(field[name].values, dtype=np.float64) for name in ("u", "v", "w"))
    inputs = np.stack((np.ones_like(w), w, u*u + v*v + w*w))
    if not np.isfinite(inputs).all():
        raise ValueError("input fields must be finite")
    weights = _solid_angles(lat, lon)
    records = {}
    for name, function, support in (
        ("H", mollifier_les_fiber_moments, 2*SCALE),
        ("G", mollifier_fiber_moments, 4*SCALE),
    ):
        folder = args.output_dir / name
        folder.mkdir()
        build_time = write_time = 0.
        for row in range(len(lat)):
            start = time.perf_counter()
            rows, kernel = build(lat, len(lon), row, function, support)
            build_time += time.perf_counter() - start
            start = time.perf_counter()
            with (folder / f"{row:04d}.npz").open("xb") as stream:
                np.savez(stream, target_rows=rows, kernel=kernel)
            write_time += time.perf_counter() - start
        records[name] = {
            "kernel_build_seconds": build_time, "kernel_write_seconds": write_time,
            "archive_bytes": sum(file.stat().st_size for file in folder.iterdir()),
        }
        print(f"Built {name} scalar archive", flush=True)
    direct, records["G"]["pass"] = apply_archive(args.output_dir / "G", inputs, weights)
    intermediate, records["H"]["first_pass"] = apply_archive(args.output_dir / "H", inputs, weights)
    twice, records["H"]["second_pass"] = apply_archive(args.output_dir / "H", intermediate, weights)
    if not np.isfinite(direct).all() or not np.isfinite(twice).all():
        raise ValueError("nonfinite filtered field")
    comparison = {}
    for i, name in enumerate(NAMES):
        diff = twice[i] - direct[i]
        peak = float(np.max(np.abs(direct[i])))
        comparison[name] = {
            "max_absolute_error": float(np.max(np.abs(diff))),
            "rms_absolute_error": float(np.sqrt(np.mean(diff**2))),
            "reference_peak": peak,
            "peak_normalized_max_error": float(np.max(np.abs(diff))/peak) if peak else None,
            "relative_l2_error": float(np.linalg.norm(diff)/np.linalg.norm(direct[i])),
        }
    np.save(args.output_dir / "direct_G.npy", direct)
    np.save(args.output_dir / "twice_H.npy", twice)
    report = {
        "job_id": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(),
        "input": str(args.velocity_file), "grid_shape": [len(lat), len(lon)], "radius_m": RADIUS,
        "G_support_km": 4*SCALE*RADIUS/1000, "H_support_km": 2*SCALE*RADIUS/1000,
        "fields": NAMES, "precision": "complex128 kernel and spectra; float64 field",
        "records": records, "comparison": comparison,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scope": "scalar-only full-grid pilot; one repetition; filesystem cache uncontrolled",
    }
    with (args.output_dir / "results.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
