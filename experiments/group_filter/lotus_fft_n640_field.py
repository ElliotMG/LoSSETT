"""Time a full n640 field using the experimental FFT kernel archive."""
import hashlib
import inspect
import json
import os
import resource
import socket
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

from disk_group_fft import build_kernels, evaluate, prepare_fields
from lotus_fft_pilot import INPUT, SCALE
from lossett.calc.compute_group_increments import _solid_angles
from lossett.calc.group_increments import _transport, latlon_shell_fields
from lossett.filtering.group_kernels import mollifier_fiber_moments


ROOT = Path("/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry")
ARCHIVE = ROOT / "n640_fft_array_benchmark_20261124"
RADIUS = 6382758.772639869
NAMES = ("horizontal", "structure", "vertical_flux", "mixed_x", "mixed_y", "mixed_z", "mass")


def source_hash(function):
    source = inspect.getsourcefile(function)
    if source is None:
        raise ValueError(f"No source file for {function}")
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def main():
    output = Path(sys.argv[1])
    reference_path = Path(sys.argv[2])
    output.mkdir(exist_ok=False)
    summary = json.loads((ARCHIVE / "summary.json").read_text())
    identity = summary["identity"]
    expected_hashes = {
        "fft": source_hash(build_kernels),
        "transport": source_hash(_transport),
        "moments": source_hash(mollifier_fiber_moments),
    }
    if summary["rows"] != 961 or identity["sources_sha256"] != expected_hashes:
        raise ValueError("FFT archive is incomplete or was built with different source code")
    if identity["angular_scale"] != SCALE or identity["numpy_version"] != np.__version__:
        raise ValueError("FFT archive scale or NumPy version does not match")
    start = time.perf_counter()
    with xr.open_dataset(INPUT, decode_timedelta=False) as source:
        field = source.isel(time=0).sel(pressure=200.).load()
    input_seconds = time.perf_counter() - start
    lat = np.deg2rad(np.asarray(field.latitude.values, dtype=np.float64))
    lon = np.deg2rad(np.asarray(field.longitude.values, dtype=np.float64))
    if [len(lat), len(lon)] != identity["grid_shape"]:
        raise ValueError("FFT archive grid shape does not match")
    if hashlib.sha256(lat.tobytes()).hexdigest() != identity["latitude_sha256"]:
        raise ValueError("FFT archive latitudes do not match")
    spacing = np.diff(lon)
    if not np.allclose(spacing, 2 * np.pi / len(lon), rtol=0, atol=1e-12):
        raise ValueError("FFT archive requires a uniformly spaced periodic longitude grid")
    values = [np.array(field[name].values, dtype=np.float64, copy=True) for name in ("u", "v", "w")]
    start = time.perf_counter()
    weights = _solid_angles(lat, lon)
    _, velocity, grad, gradw = latlon_shell_fields(lat, lon, *values, radius=RADIUS)
    geometry_seconds = time.perf_counter() - start
    start = time.perf_counter()
    fields = prepare_fields(lat, lon, velocity, grad, gradw, weights, RADIUS)
    prepare_seconds = time.perf_counter() - start
    reference = np.load(reference_path)
    sample_rows = np.unique(np.linspace(0, len(lat) - 1, 5, dtype=int))
    if reference.shape != (len(sample_rows), len(lon), len(NAMES)):
        raise ValueError("Unexpected Numba reference output shape")
    samples = {int(row): index for index, row in enumerate(sample_rows)}
    worst = dict.fromkeys(NAMES, 0.0)
    results = np.lib.format.open_memmap(
        output / "fft_output.npy", mode="w+", dtype=np.float64,
        shape=(len(lat), len(lon), len(NAMES)),
    )
    read_seconds = compute_seconds = output_seconds = 0.0
    mass_error = 0.0
    for row in range(len(lat)):
        start = time.perf_counter()
        with (ARCHIVE / "kernels" / f"{row:05d}.npz").open("rb") as stream:
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            with np.load(stream) as archive:
                target_rows, kernels = archive["target_rows"], archive["kernels"]
        read_seconds += time.perf_counter() - start
        start = time.perf_counter()
        result = evaluate(fields, row, target_rows, kernels)
        compute_seconds += time.perf_counter() - start
        if not np.isfinite(result).all():
            raise ValueError(f"Non-finite FFT output at row {row}")
        mass_error = max(mass_error, float(np.max(np.abs(result[:, 6] - 1))))
        if row in samples:
            expected = reference[samples[row]]
            np.testing.assert_allclose(result, expected, rtol=2e-9, atol=2e-9)
            for j, name in enumerate(NAMES):
                worst[name] = max(worst[name], float(np.max(np.abs(result[:, j] - expected[:, j]))))
        start = time.perf_counter()
        results[row] = result
        output_seconds += time.perf_counter() - start
        if row % 50 == 0:
            print(f"row {row}: cumulative read={read_seconds:.2f}s compute={compute_seconds:.2f}s", flush=True)
    if mass_error > 0.01:
        raise ValueError(f"FFT kernel-mass error {mass_error} exceeds 0.01")
    start = time.perf_counter()
    results.flush()
    output_seconds += time.perf_counter() - start
    del results
    report = {
        "host": socket.gethostname(),
        "input": str(INPUT),
        "grid_shape": [len(lat), len(lon)],
        "radius_m": RADIUS,
        "angular_scale": SCALE,
        "archive": str(ARCHIVE),
        "archive_bytes": summary["archive_bytes"],
        "read_cache": "cold-advised, not guaranteed",
        "input_read_seconds": input_seconds,
        "geometry_setup_seconds": geometry_seconds,
        "fft_field_prepare_seconds": prepare_seconds,
        "archive_read_seconds": read_seconds,
        "compute_seconds": compute_seconds,
        "output_write_seconds": output_seconds,
        "sample_rows": sample_rows.tolist(),
        "numba_reference": str(reference_path),
        "max_absolute_errors_vs_numba": worst,
        "max_kernel_mass_error": mass_error,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    with (output / "results.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
