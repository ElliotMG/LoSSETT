"""Compare scalar G and H twice at selected real-field latitude rings."""

import argparse
import hashlib
import json
import os
import resource
import socket
import time
from pathlib import Path

import numpy as np
import xarray as xr

from lossett.calc.compute_group_increments import _solid_angles
from lossett.filtering.group_kernels import mollifier_les_fiber_moments


ROOT = Path("/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry")
DATA = Path("/gws/ssde/j25b/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/"
            "DYAMOND_SUMMER/benchmark_antialiased_20160801T00")
RADIUS = 6382758.772639869
SCALE = 1e6 / RADIUS
NAMES = ("constant", "radial_velocity", "speed_squared")


def h_scalar(mu, scale, central_value):
    angle = np.arccos(np.clip(mu, -1, 1))
    ratio = angle / (2*scale)
    inside = ratio < 1
    result = np.zeros_like(mu)
    result[inside] = central_value * np.exp(1 - 1/(1-ratio[inside]**2))
    return result


def make_h(lat, lon, row, central_value):
    rows = np.flatnonzero(np.abs(lat - lat[row]) <= 2*SCALE + 1e-14)
    delta = 2*np.pi*np.arange(len(lon))/len(lon)
    mu = np.clip(
        np.sin(lat[row])*np.sin(lat[rows, None])
        + np.cos(lat[row])*np.cos(lat[rows, None])*np.cos(delta), -1, 1,
    )
    return rows, np.fft.rfft(h_scalar(mu, SCALE, central_value), axis=-1)


def save(path, target_rows, kernel):
    with path.open("xb") as stream:
        np.savez(stream, target_rows=target_rows, kernel=kernel)
        stream.flush()
        os.fsync(stream.fileno())


def apply(path, fields, weights, rows):
    start = time.perf_counter()
    spectra = np.fft.rfft(fields * weights[None, :, :], axis=2)
    transform = time.perf_counter() - start
    values = np.empty((len(fields), len(rows), fields.shape[-1]))
    read = compute = 0.
    for pos, row in enumerate(rows):
        start = time.perf_counter()
        with np.load(path / f"{row:05d}.npz", allow_pickle=False) as archive:
            target, kernel = archive["target_rows"], archive["kernel"]
        read += time.perf_counter() - start
        start = time.perf_counter()
        values[:, pos] = np.fft.irfft(
            np.sum(np.conj(kernel)[None, :, :] * spectra[:, target, :], axis=1),
            n=fields.shape[-1], axis=-1,
        )
        compute += time.perf_counter() - start
    return values, {"field_fft_seconds": transform, "archive_read_seconds": read, "compute_seconds": compute}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=("n320", "n640"), required=True)
    args = parser.parse_args()
    source_path = DATA / f"filtered_{args.grid}.nc"
    output = ROOT / f"h_twice_scalar_rings_{args.grid}_20261124"
    output.mkdir(exist_ok=False)
    if args.grid == "n320":
        source_root = ROOT / "full_pipeline_bench_v1/n320"
        manifest = json.loads((source_root / "results.json").read_text())
        archive_root = source_root / "fft_archive"
        grid_shape, scale = manifest["grid_shape"], manifest["mollifier_scale"]
    else:
        source_root = ROOT / "n640_fft_array_benchmark_20261124"
        manifest = json.loads((source_root / "summary.json").read_text())["identity"]
        archive_root = source_root / "kernels"
        grid_shape, scale = manifest["grid_shape"], manifest["angular_scale"]
    if scale != SCALE:
        raise ValueError(f"G archive scale mismatch: {scale} != {SCALE}")
    with xr.open_dataset(source_path, decode_timedelta=False) as source:
        field = source.isel(time=0).sel(pressure=200.).load()
    lat = np.deg2rad(field.latitude.values)
    lon = np.deg2rad(field.longitude.values)
    if [len(lat), len(lon)] != grid_shape:
        raise ValueError("G archive grid shape mismatch")
    if args.grid == "n640" and hashlib.sha256(lat.tobytes()).hexdigest() != manifest["latitude_sha256"]:
        raise ValueError("G archive latitude mismatch")
    weights = _solid_angles(lat, lon)
    u, v, w = (np.asarray(field[name].values, dtype=np.float64) for name in ("u", "v", "w"))
    inputs = np.stack((np.ones_like(w), w, u*u + v*v + w*w))
    if not np.isfinite(inputs).all():
        raise ValueError("input fields must be finite")
    origins = np.array([0, (len(lat)-1)//4, (len(lat)-1)//2])
    required = np.flatnonzero(np.any(np.abs(lat[:, None] - lat[origins]) <= 2*SCALE + 1e-14, axis=1))
    h_dir, g_dir = output / "H", output / "G"
    h_dir.mkdir()
    g_dir.mkdir()
    start = time.perf_counter()
    center = float(mollifier_les_fiber_moments(np.array([1.]), SCALE)[0][0])
    norm_seconds = time.perf_counter() - start
    sample_mu = np.r_[np.linspace(np.cos(2*SCALE), 1., 64), -1.]
    expected = mollifier_les_fiber_moments(sample_mu, SCALE)[0]
    np.testing.assert_allclose(h_scalar(sample_mu, SCALE, center), expected, rtol=2e-13, atol=2e-13)
    h_build = h_write = 0.
    for row in required:
        start = time.perf_counter()
        target, kernel = make_h(lat, lon, row, center)
        h_build += time.perf_counter() - start
        start = time.perf_counter()
        save(h_dir / f"{row:05d}.npz", target, kernel)
        h_write += time.perf_counter() - start
    g_extract = g_write = 0.
    for row in origins:
        start = time.perf_counter()
        with np.load(archive_root / f"{row:05d}.npz", allow_pickle=False) as archive:
            target = archive["target_rows"]
            kernel = archive["kernels"][0]
        expected_rows = np.flatnonzero(np.abs(lat - lat[row]) <= 4*SCALE + 1e-14)
        np.testing.assert_array_equal(target, expected_rows)
        g_extract += time.perf_counter() - start
        start = time.perf_counter()
        save(g_dir / f"{row:05d}.npz", target, kernel)
        g_write += time.perf_counter() - start
    direct, g_timing = apply(g_dir, inputs, weights, origins)
    first, first_timing = apply(h_dir, inputs, weights, required)
    intermediate = np.zeros_like(inputs)
    intermediate[:, required] = first
    twice, second_timing = apply(h_dir, intermediate, weights, origins)
    if not np.isfinite(direct).all() or not np.isfinite(twice).all():
        raise ValueError("nonfinite outputs")
    errors = {}
    for j, name in enumerate(NAMES):
        diff = twice[j]-direct[j]
        peak = float(np.max(np.abs(direct[j])))
        errors[name] = {
            "max_absolute": float(np.max(np.abs(diff))),
            "peak": peak, "max_over_peak": float(np.max(np.abs(diff))/peak),
            "relative_l2": float(np.linalg.norm(diff)/np.linalg.norm(direct[j])),
            "max_absolute_by_ring": np.max(np.abs(diff), axis=1).tolist(),
        }
    archive_full_payload = {}
    for name, support in (("H", 2*SCALE), ("G", 4*SCALE)):
        bands = sum(np.count_nonzero(np.abs(lat - origin) <= support + 1e-14) for origin in lat)
        archive_full_payload[name] = int(bands * ((len(lon)//2+1)*16 + 8))
    np.save(output / "direct_G.npy", direct)
    np.save(output / "twice_H.npy", twice)
    report = {
        "job_id": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(),
        "input": str(source_path), "G_source": str(archive_root), "grid_shape": grid_shape,
        "origin_rows": origins.tolist(), "origin_latitudes_degrees": np.rad2deg(lat[origins]).tolist(),
        "H_required_intermediate_rows": len(required), "radius_m": RADIUS, "H_support_km": 2000.,
        "G_support_km": 4000., "fields": NAMES, "precision": "complex128 spectra and float64 fields",
        "H_norm_seconds": norm_seconds, "H_build_seconds": h_build, "H_write_fsync_seconds": h_write,
        "H_subset_cache_bytes": sum(file.stat().st_size for file in h_dir.iterdir()),
        "G_extract_from_full_increment_archives_seconds": g_extract,
        "G_write_scalar_subset_fsync_seconds": g_write,
        "G_subset_cache_bytes": sum(file.stat().st_size for file in g_dir.iterdir()),
        "full_grid_scalar_payload_estimate_bytes": archive_full_payload,
        "G_pass": g_timing, "H_first_pass": first_timing, "H_second_pass": second_timing,
        "errors": errors, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "caveats": "scalar-only; sparse target rings, all required intermediate H rings; "
                   "G build excluded; one repetition; filesystem caching uncontrolled",
    }
    with (output / "results.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
