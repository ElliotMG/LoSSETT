"""Build and validate one n640 FFT kernel ring in a LOTUS array task."""
import hashlib
import json
import os
import resource
import socket
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

from disk_group_fft import build_kernels


INPUT = Path(
    "/gws/ssde/j25b/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/"
    "DYAMOND_SUMMER/benchmark_antialiased_20160801T00/filtered_n640.nc"
)
REFERENCE = Path(
    "/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/"
    "full_pipeline_bench_v1/n640/fft_archive"
)
SCALE = 0.1566720654220191


def main():
    output = Path(sys.argv[1])
    task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    with xr.open_dataset(INPUT, decode_timedelta=False) as source:
        lat = np.deg2rad(np.asarray(source.latitude.values, dtype=np.float64))
        nlon = source.sizes["longitude"]
    rows = np.unique(np.linspace(0, len(lat) - 1, 5, dtype=int))
    row = int(rows[task])

    start = time.perf_counter()
    target_rows, kernels = build_kernels(lat, nlon, row, SCALE)
    build_seconds = time.perf_counter() - start
    if not np.isfinite(kernels).all():
        raise ValueError(f"Non-finite FFT kernels for row {row}")
    reference_path = REFERENCE / f"{row:05d}.npz"
    with np.load(reference_path) as reference:
        np.testing.assert_array_equal(target_rows, reference["target_rows"])
        expected = reference["kernels"]
        np.testing.assert_allclose(kernels, expected, rtol=1e-12, atol=1e-11)
        max_abs_error = float(np.max(np.abs(kernels - expected)))

    archive_path = output / f"{row:05d}.npz"
    start = time.perf_counter()
    with archive_path.open("xb") as stream:
        np.savez(stream, target_rows=target_rows, kernels=kernels)
        stream.flush()
        os.fsync(stream.fileno())
    write_seconds = time.perf_counter() - start
    with np.load(archive_path) as written:
        np.testing.assert_array_equal(target_rows, written["target_rows"])
        np.testing.assert_array_equal(kernels, written["kernels"])
    report = {
        "array_job_id": os.environ["SLURM_ARRAY_JOB_ID"],
        "array_task_id": task,
        "host": socket.gethostname(),
        "row": row,
        "latitude_degrees": float(np.rad2deg(lat[row])),
        "grid_shape": [len(lat), nlon],
        "angular_scale": SCALE,
        "latitude_sha256": hashlib.sha256(lat.tobytes()).hexdigest(),
        "source_sha256": hashlib.sha256(
            Path(build_kernels.__code__.co_filename).read_bytes()
        ).hexdigest(),
        "numpy_version": np.__version__,
        "kernel_shape": list(kernels.shape),
        "kernel_dtype": str(kernels.dtype),
        "reference_archive": str(reference_path),
        "max_abs_difference_from_serial": max_abs_error,
        "build_seconds": build_seconds,
        "write_seconds": write_seconds,
        "archive_bytes": archive_path.stat().st_size,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    report_path = output / f"{row:05d}.json"
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
