"""Verify pilot-array kernels using independently computed Numba outputs."""
import json
import resource
import socket
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

from disk_group_fft import evaluate, prepare_fields
from lotus_fft_pilot import INPUT, SCALE
from lossett.calc.compute_group_increments import _solid_angles
from lossett.calc.group_increments import latlon_shell_fields


def main():
    root = Path(sys.argv[1])
    with xr.open_dataset(INPUT, decode_timedelta=False) as source:
        field = source.isel(time=0).sel(pressure=200.).load()
    lat = np.deg2rad(np.asarray(field.latitude.values, dtype=np.float64))
    lon = np.deg2rad(np.asarray(field.longitude.values, dtype=np.float64))
    rows = np.unique(np.linspace(0, len(lat) - 1, 5, dtype=int))
    expected_names = {f"{int(row):05d}" for row in rows}
    for extension in ("npz", "json"):
        actual_names = {path.stem for path in (root / "kernels").glob(f"*.{extension}")}
        if actual_names != expected_names:
            raise ValueError(f"Incomplete or unexpected {extension} files: {actual_names}")
    reports = [
        json.loads((root / "kernels" / f"{int(row):05d}.json").read_text())
        for row in rows
    ]
    for row, report in zip(rows, reports):
        if report["row"] != row or report["angular_scale"] != SCALE:
            raise ValueError(f"Incorrect metadata for row {row}")
    for key in ("grid_shape", "latitude_sha256", "source_sha256", "numpy_version"):
        if any(report[key] != reports[0][key] for report in reports):
            raise ValueError(f"Inconsistent pilot metadata: {key}")

    radius = 6382758.772639869
    velocity_fields = [
        np.array(field[name].values, dtype=np.float64, copy=True)
        for name in ("u", "v", "w")
    ]
    start = time.perf_counter()
    weights = _solid_angles(lat, lon)
    _, velocity, grad, gradw = latlon_shell_fields(
        lat, lon, *velocity_fields, radius=radius
    )
    fields = prepare_fields(lat, lon, velocity, grad, gradw, weights, radius)
    preparation_seconds = time.perf_counter() - start
    numba = np.load(root / "compare" / "numba_output.npy")
    serial_fft = np.load(root / "compare" / "fft_output.npy")
    expected_shape = (len(rows), len(lon), 7)
    if numba.shape != expected_shape or serial_fft.shape != expected_shape:
        raise ValueError("Unexpected comparison output shape")
    names = ("horizontal", "structure", "vertical_flux", "mixed_x", "mixed_y", "mixed_z", "mass")
    errors = {name: 0.0 for name in names}
    serial_errors = {name: 0.0 for name in names}
    start = time.perf_counter()
    for i, row in enumerate(rows):
        with np.load(root / "kernels" / f"{int(row):05d}.npz") as archive:
            result = evaluate(fields, int(row), archive["target_rows"], archive["kernels"])
        np.testing.assert_allclose(result, numba[i], rtol=2e-9, atol=2e-9)
        np.testing.assert_allclose(result, serial_fft[i], rtol=2e-9, atol=2e-9)
        for j, name in enumerate(names):
            errors[name] = max(errors[name], float(np.max(np.abs(result[:, j] - numba[i, :, j]))))
            serial_errors[name] = max(
                serial_errors[name], float(np.max(np.abs(result[:, j] - serial_fft[i, :, j])))
            )
    report = {
        "host": socket.gethostname(),
        "rows": rows.tolist(),
        "comparison_tolerance": {"rtol": 2e-9, "atol": 2e-9},
        "max_absolute_errors_vs_numba": errors,
        "max_absolute_errors_vs_serial_fft": serial_errors,
        "field_preparation_seconds": preparation_seconds,
        "archive_read_and_evaluation_seconds": time.perf_counter() - start,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    with (root / "pilot_verification.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
