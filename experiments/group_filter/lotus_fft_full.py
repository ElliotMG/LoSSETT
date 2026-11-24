"""Build a disjoint chunk of experimental n640 FFT kernel archives."""
import hashlib
import inspect
import json
import os
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

from disk_group_fft import build_kernels
from lotus_fft_pilot import INPUT, REFERENCE, SCALE
from lossett.calc.group_increments import _transport
from lossett.filtering.group_kernels import mollifier_fiber_moments


CHUNK_SIZE = 20
REPO = Path("/home/users/dship/python/LoSSETT_spherical")


def source_hash(function):
    source = inspect.getsourcefile(function)
    if source is None:
        raise ValueError(f"No source file for {function}")
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def main():
    root = Path(sys.argv[1])
    resume = sys.argv[2:] == ["--resume"]
    if sys.argv[2:] and not resume:
        raise ValueError("Only --resume is accepted after the output directory")
    task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    with xr.open_dataset(INPUT, decode_timedelta=False) as source:
        lat = np.deg2rad(np.asarray(source.latitude.values, dtype=np.float64))
        nlon = source.sizes["longitude"]
    if (len(lat), nlon) != (961, 1280):
        raise ValueError(f"Unexpected n640 grid shape: {(len(lat), nlon)}")
    start_row = task * CHUNK_SIZE
    end_row = min(start_row + CHUNK_SIZE, len(lat))
    if start_row >= len(lat):
        raise ValueError(f"Array task {task} is outside the grid")
    commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"], text=True
    ).strip()
    if git_status:
        raise ValueError("LoSSETT checkout is not clean; refusing ambiguous provenance")
    identity = {
        "input_coordinate_source": str(INPUT),
        "grid_shape": [len(lat), nlon],
        "latitude_sha256": hashlib.sha256(lat.tobytes()).hexdigest(),
        "longitude_layout": "uniform periodic, 2*pi/nlon",
        "angular_scale": SCALE,
        "filter": "mollifier_fiber_moments; g, h and k",
        "channel_layout": "g, k, 9 h*transport, 36 k*transport-products",
        "fft_convention": "numpy rfft, longitude axis, default normalization",
        "transport_convention": "_transport in group_increments.py",
        "kernel_dtype": "complex128",
        "numpy_version": np.__version__,
        "lossett_commit": commit,
        "sources_sha256": {
            "fft": source_hash(build_kernels),
            "transport": source_hash(_transport),
            "moments": source_hash(mollifier_fiber_moments),
        },
    }
    results = []
    for row in range(start_row, end_row):
        start = time.perf_counter()
        target_rows, kernels = build_kernels(lat, nlon, row, SCALE)
        build_seconds = time.perf_counter() - start
        if kernels.shape != (47, len(target_rows), nlon // 2 + 1):
            raise ValueError(f"Unexpected kernel shape at row {row}: {kernels.shape}")
        if kernels.dtype != np.complex128 or not np.isfinite(kernels).all():
            raise ValueError(f"Invalid kernel values or dtype at row {row}")
        reference_path = REFERENCE / f"{row:05d}.npz"
        if reference_path.exists():
            with np.load(reference_path) as reference:
                np.testing.assert_array_equal(target_rows, reference["target_rows"])
                np.testing.assert_allclose(
                    kernels, reference["kernels"], rtol=1e-12, atol=1e-11
                )
        path = root / "kernels" / f"{row:05d}.npz"
        temporary = root / "kernels" / f".{row:05d}.{os.getpid()}.partial"
        reused_existing = path.exists()
        if reused_existing:
            if not resume:
                raise FileExistsError(f"Refusing to overwrite {path}")
            write_seconds = 0.0
            start = time.perf_counter()
            with np.load(path) as archive:
                np.testing.assert_array_equal(target_rows, archive["target_rows"])
                np.testing.assert_allclose(
                    kernels, archive["kernels"], rtol=1e-12, atol=1e-11
                )
            verify_seconds = time.perf_counter() - start
        else:
            start = time.perf_counter()
            with temporary.open("xb") as stream:
                np.savez(stream, target_rows=target_rows, kernels=kernels)
                stream.flush()
                os.fsync(stream.fileno())
            write_seconds = time.perf_counter() - start
            start = time.perf_counter()
            with np.load(temporary) as archive:
                np.testing.assert_array_equal(target_rows, archive["target_rows"])
                np.testing.assert_array_equal(kernels, archive["kernels"])
            verify_seconds = time.perf_counter() - start
            os.link(temporary, path)
            temporary.unlink()
        results.append({
            "row": row,
            "target_count": len(target_rows),
            "bytes": path.stat().st_size,
            "build_seconds": build_seconds,
            "write_seconds": write_seconds,
            "readback_seconds": verify_seconds,
            "checked_against_serial_reference": reference_path.exists(),
            "reused_existing": reused_existing,
        })
        print(f"row {row}: build {build_seconds:.3f}s, write {write_seconds:.3f}s, "
              f"readback {verify_seconds:.3f}s", flush=True)
    report = {
        "array_job_id": os.environ["SLURM_ARRAY_JOB_ID"],
        "array_task_id": task,
        "host": socket.gethostname(),
        "identity": identity,
        "rows": results,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    report_path = root / "manifests" / f"{task:05d}.json"
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
