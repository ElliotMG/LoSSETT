"""Compare saved FFT kernel precisions on synthetic full-resolution rings."""

import hashlib
import json
import os
import resource
import socket
import time
from pathlib import Path

import numpy as np

from disk_group_fft import build_kernels, evaluate, prepare_fields
from lossett.calc.compute_group_increments import _solid_angles


ROOT = Path("/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/n2560_fft_precision_synthetic_20261124")
RADIUS = 6382758.772639869
SUPPORTS_KM = (100., 500., 2000.)
SEED = 20261124
NAMES = ("horizontal", "structure", "vertical_flux", "mixed_x", "mixed_y", "mixed_z", "mass")


def main():
    task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    lat = np.linspace(-np.pi/2, np.pi/2, 3841)
    lon = np.arange(5120) * (2*np.pi/5120)
    selected = np.random.default_rng(SEED).choice(np.arange(1, len(lat)-1), size=3, replace=False)
    row = int(selected[task // len(SUPPORTS_KM)])
    support = SUPPORTS_KM[task % len(SUPPORTS_KM)]
    scale = support * 1000 / (4*RADIUS)
    output = ROOT / f"row{row:04d}_support{int(support):04d}km"
    output.mkdir(parents=True, exist_ok=False)

    start = time.perf_counter()
    target_rows, kernels = build_kernels(lat, len(lon), row, scale)
    kernel_seconds = time.perf_counter() - start
    if not np.isfinite(kernels).all():
        raise ValueError("Non-finite FFT kernels")
    output_row = int(np.flatnonzero(target_rows == row)[0])
    start = time.perf_counter()
    angular_weights = _solid_angles(lat, lon)[target_rows]
    target_lat = lat[target_rows]
    clat = np.cos(target_lat)[:, None]
    slat = np.sin(target_lat)[:, None]
    clon, slon = np.cos(lon)[None, :], np.sin(lon)[None, :]
    direction = np.stack(np.broadcast_arrays(clat*clon, clat*slon, slat), axis=-1)
    a = np.array([10., -4., 3.])
    omega = np.array([10., 5., 20.])
    b = np.array([0.6, -0.3, 0.4])
    normal_a = np.einsum("...i,i->...", direction, a)
    w = np.einsum("...i,i->...", direction, b)
    velocity = np.cross(omega, direction) + a - normal_a[..., None]*direction + w[..., None]*direction
    projection = np.eye(3) - direction[..., :, None]*direction[..., None, :]
    cross = np.array([[0., -omega[2], omega[1]], [omega[2], 0., -omega[0]],
                      [-omega[1], omega[0], 0.]])
    grad = (np.einsum("...ij,jk,...kl->...il", projection, cross, projection)
            - normal_a[..., None, None]*projection) / RADIUS
    gradw = np.einsum("...ij,j->...i", projection, b) / RADIUS
    del projection, direction
    fields = prepare_fields(target_lat, lon, velocity, grad, gradw, angular_weights, RADIUS)
    del velocity, grad, gradw, angular_weights
    field_seconds = time.perf_counter() - start

    start = time.perf_counter()
    expected = evaluate(fields, output_row, np.arange(len(target_rows)), kernels)
    reference_compute_seconds = time.perf_counter() - start
    if not np.isfinite(expected).all():
        raise ValueError("Non-finite complex128 result")
    c64 = kernels.astype(np.complex64)
    if not np.isfinite(c64).all():
        raise ValueError("Non-finite complex64 kernel")
    paths = {}
    writes = {}
    for label, array in (("complex128", kernels), ("complex64", c64)):
        path = output / f"kernels_{label}.npz"
        start = time.perf_counter()
        with path.open("xb") as stream:
            np.savez(stream, target_rows=target_rows, kernels=array)
            stream.flush()
            os.fsync(stream.fileno())
        writes[label] = time.perf_counter() - start
        paths[label] = path
    del kernels
    start = time.perf_counter()
    with np.load(paths["complex64"], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["target_rows"], target_rows)
        restored = archive["kernels"]
    read_seconds = time.perf_counter() - start
    np.testing.assert_array_equal(restored, c64)
    del c64
    start = time.perf_counter()
    actual = evaluate(fields, output_row, np.arange(len(target_rows)), restored)
    actual_compute_seconds = time.perf_counter() - start
    if not np.isfinite(actual).all():
        raise ValueError("Non-finite complex64 result")
    errors = {}
    for j, name in enumerate(NAMES):
        diff = actual[:, j] - expected[:, j]
        peak = float(np.max(np.abs(expected[:, j])))
        errors[name] = {
            "max_absolute": float(np.max(np.abs(diff))),
            "rms_absolute": float(np.sqrt(np.mean(diff**2))),
            "reference_peak": peak,
            "max_error_over_reference_peak": float(np.max(np.abs(diff))/peak) if peak else None,
        }
    np.save(output / "outputs_complex128.npy", expected)
    np.save(output / "outputs_complex64.npy", actual)
    report = {
        "job_id": os.environ.get("SLURM_JOB_ID"), "task": task, "host": socket.gethostname(),
        "seed": SEED, "selected_rows": selected.tolist(), "row": row,
        "latitude_degrees": float(np.rad2deg(lat[row])), "support_G_km": support,
        "angular_scale": scale, "radius_m": RADIUS, "grid_shape": [len(lat), len(lon)],
        "latitude_sha256": hashlib.sha256(lat.tobytes()).hexdigest(),
        "field": "analytic smooth rotational plus divergent tangent velocity and radial l=1 field",
        "kernel_shape": list(restored.shape), "kernel_build_seconds": kernel_seconds,
        "field_prepare_seconds": field_seconds, "complex128_evaluate_seconds": reference_compute_seconds,
        "complex64_read_seconds": read_seconds, "complex64_evaluate_seconds": actual_compute_seconds,
        "archive_write_seconds": writes, "archive_bytes": {k: v.stat().st_size for k, v in paths.items()},
        "max_mass_error": float(np.max(np.abs(expected[:, 6] - 1))),
        "errors": errors, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "caveats": "synthetic field; exact analytic gradients; uncontrolled I/O cache; one run",
    }
    with (output / "results.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
