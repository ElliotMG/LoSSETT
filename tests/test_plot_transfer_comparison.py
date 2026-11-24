import json
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

from tools.plot_transfer_comparison import load_numpy, load_zarr


def test_sampled_numpy_uses_recorded_rows_and_checks_source(tmp_path):
    latitude = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])
    longitude = np.array([-180.0, -90.0, 0.0, 90.0])
    coordinates = tmp_path / "input.nc"
    values = np.zeros((2, 4, 7))
    values[0, :, 0] = 1
    values[1, :, 0] = 2
    path = tmp_path / "numba_output.npy"
    np.save(path, values)
    report = {"grid_shape": [5, 4], "origin_rows": [1, 3], "velocity_file": str(coordinates)}
    (tmp_path / "results.json").write_text(json.dumps(report))

    rows, horizontal = load_numpy(path, latitude, longitude, coordinates)
    np.testing.assert_array_equal(rows, [1, 3])
    np.testing.assert_array_equal(horizontal[:, 0], [1, 2])
    with pytest.raises(ValueError, match="input field does not match"):
        load_numpy(path, latitude, longitude, tmp_path / "other.nc")

    report["origin_rows"] = [3, 1]
    (tmp_path / "results.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="invalid or unordered origin_rows"):
        load_numpy(path, latitude, longitude, coordinates)


def test_zarr_requires_single_level_and_matching_coordinates(tmp_path):
    latitude = np.array([-45.0, 0.0, 45.0])
    longitude = np.array([-180.0, -90.0, 0.0, 90.0])
    path = tmp_path / "transfer.zarr"
    dataset = xr.Dataset(
        {"Dl_u": (("length_scale", "longitude", "latitude"), np.ones((1, 4, 3)))},
        coords={"length_scale": [2e6], "longitude": longitude, "latitude": latitude},
    )
    dataset.to_zarr(path, zarr_format=2)
    rows, values = load_zarr(path, "Dl_u", latitude, longitude)
    np.testing.assert_array_equal(rows, [0, 1, 2])
    assert values.shape == (3, 4)
    with pytest.raises(ValueError, match="latitude does not match"):
        load_zarr(path, "Dl_u", latitude + 1, longitude)


def test_plot_command_accepts_partial_comparison(tmp_path):
    latitude = np.array([-90.0, 0.0, 90.0])
    longitude = np.array([-180.0, -90.0, 0.0, 90.0])
    coordinates = tmp_path / "input.nc"
    xr.Dataset(coords={"latitude": latitude, "longitude": longitude}).to_netcdf(coordinates)
    values = np.zeros((1, 4, 7))
    values[0, :, 0] = [1, 2, 3, 4]
    path = tmp_path / "numba_output.npy"
    np.save(path, values)
    (tmp_path / "results.json").write_text(json.dumps({
        "grid_shape": [3, 4], "origin_rows": [1], "velocity_file": str(coordinates),
    }))
    output = tmp_path / "comparison.png"
    script = "tools/plot_transfer_comparison.py"
    command = [sys.executable, script, "--coordinates", str(coordinates),
               "--numba", str(path), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "longitude section at 0 degrees" in result.stdout
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "output already exists" in failed.stderr
