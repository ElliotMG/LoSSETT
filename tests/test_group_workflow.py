import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

from lossett.calc.compute_group_increments import main, shell_dataset, vertical_dataset
from lossett.calc.group_increments import hydrostatic_radius, reconstruct_group_vertical


def test_shell_and_vertical_zarr_workflow(tmp_path):
    lat = np.linspace(-90., 90., 9)
    lon = np.arange(16) * 360. / 16
    phi, lam = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    normal = np.stack((np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)), axis=-1)
    east = np.stack((-np.sin(lam), np.cos(lam), np.zeros_like(lam)), axis=-1)
    north = np.cross(normal, east)
    pressure = np.array([1000., 900., 800., 700., 600.])
    density = np.array([1., 1.2, 1.4, 1.8, 2.])
    velocities = np.stack([
        np.array([1., -0.5, 0.2 + k * 0.1]) + np.cross([0.1, 0.2, 0.4], normal)
        for k in range(5)
    ])
    ds = xr.Dataset(
        {
            "u": (("time", "pressure", "latitude", "longitude"), np.sum(velocities * east, axis=-1)[None]),
            "v": (("time", "pressure", "latitude", "longitude"), np.sum(velocities * north, axis=-1)[None]),
            "w": (("time", "pressure", "latitude", "longitude"), np.sum(velocities * normal, axis=-1)[None]),
        },
        coords={"time": [0], "pressure": pressure, "latitude": lat, "longitude": lon},
    )
    source = tmp_path / "velocity.nc"
    ds.to_netcdf(source)
    stores = [tmp_path / f"shell-{i}.zarr" for i in range(3)]
    for level, store in zip((0, 2, 4), stores):
        args = [
            "shell", "--velocity-file", str(source), "--output", str(store), "--time-index", "0",
            "--level-index", str(level), "--coordinate-type", "pressure", "--coordinate-unit", "hPa",
            "--initial-radius", "1000000", "--density", *map(str, density), "--gravity", "10",
            "--heat-time", "0.4",
        ]
        if level == 0:
            args.append("--cache-longitude-moments")
            subprocess.run(
                [sys.executable, "-m", "lossett.calc.compute_group_increments", *args],
                check=True, capture_output=True, text=True,
            )
        else:
            main(args)
    output = tmp_path / "vertical.zarr"
    main([
        "vertical", "--shells", *map(str, stores), "--output", str(output),
        "--fluid", "anelastic", "--density", "1", "1.4", "2", "--gravity", "10",
    ])
    with xr.open_zarr(output) as result:
        assert result.total.dims == ("level", "latitude", "longitude")
        np.testing.assert_array_equal(result.level.values, pressure[[0, 2, 4]] * 100)
        np.testing.assert_allclose(
            result.radius.values, hydrostatic_radius(pressure * 100, density, 10, 1000000)[[0, 2, 4]]
        )
        np.testing.assert_allclose(result.total, result.horizontal + result.vertical)
        assert result.attrs["fluid"] == "anelastic"
        assert result.attrs["kernel"] == "heat"
        assert result.total.attrs["units"] == "m2 s-3"
        shells = [xr.open_zarr(store).load() for store in stores]
        i, j = 0, 3
        expected = reconstruct_group_vertical(
            result.level.values, result.radius.values, shells[0].direction.values[i, j],
            np.stack([s.velocity.values[i, j] for s in shells]),
            np.array([s.structure.values[i, j] for s in shells]),
            np.array([s.vertical_flux.values[i, j] for s in shells]),
            np.stack([s.mixed_moment.values[i, j] for s in shells]),
            coordinate_type="pressure", fluid="anelastic", reference_density=[1, 1.4, 2], gravity=10,
        )
        np.testing.assert_allclose(result.vertical.values[:, i, j], expected)
        with pytest.raises(ValueError, match="hydrostatic profile"):
            vertical_dataset(shells, fluid="anelastic", reference_density=[1, 2, 2], gravity=10)
        shells[1].attrs["kernel_parameter"] = 0.5
        with pytest.raises(ValueError, match="metadata"):
            vertical_dataset(shells, fluid="anelastic", reference_density=[1, 1.4, 2], gravity=10)
    with pytest.raises(FileExistsError):
        main(["vertical", "--shells", *map(str, stores), "--output", str(output), "--fluid", "boussinesq"])


def test_height_boussinesq_shallow_mollifier(tmp_path):
    lat, lon = np.linspace(-90, 90, 5), np.arange(8) * 45.
    phi, lam = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    normal = np.stack((np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)), axis=-1)
    east = np.stack((-np.sin(lam), np.cos(lam), np.zeros_like(lam)), axis=-1)
    north = np.cross(normal, east)
    velocity = np.array([0.3, -0.5, 0.4])
    dims = ("time", "height", "latitude", "longitude")
    ds = xr.Dataset(
        {name: (dims, np.broadcast_to(np.sum(vector * velocity, axis=-1), (1, 3, 5, 8)).copy())
         for name, vector in (("u", east), ("v", north), ("w", normal))},
        coords={"time": [0], "height": [0., 1., 2.], "latitude": lat, "longitude": lon},
    )
    kwargs = dict(
        time_index=0, level_dim="height", coordinate_type="height", coordinate_unit="m",
        mollifier_scale=1.4, atmosphere="shallow",
    )
    shells = [shell_dataset(ds, level_index=i, radius=100. + i, **kwargs) for i in range(3)]
    result = vertical_dataset(shells, fluid="boussinesq")
    assert result.attrs["atmosphere"] == "shallow"
    assert result.attrs["kernel"] == "mollifier"
    assert result.attrs["fluid"] == "boussinesq"
    assert np.isfinite(result.total.values).all()
    compact_kwargs = {**kwargs, "mollifier_scale": 0.5, "quadrature_tolerance": 0.2}
    reference = shell_dataset(ds, level_index=0, radius=100., **compact_kwargs)
    source, output = tmp_path / "input.nc", tmp_path / "compact.zarr"
    ds.to_netcdf(source)
    main([
        "shell", "--velocity-file", str(source), "--output", str(output), "--time-index", "0",
        "--level-index", "0", "--level-dim", "height", "--coordinate-type", "height",
        "--coordinate-unit", "m", "--radius", "100", "--mollifier-scale", "0.5",
        "--atmosphere", "shallow", "--quadrature-tolerance", "0.2", "--compact-support",
        "--cache-longitude-moments",
    ])
    with xr.open_zarr(output) as compact:
        for name in reference.data_vars:
            np.testing.assert_allclose(compact[name], reference[name], rtol=2e-11, atol=2e-11)
    with pytest.raises(ValueError, match="w must have"):
        shell_dataset(ds.drop_vars("w"), level_index=0, radius=100., **kwargs)
    ds.w.attrs["units"] = "Pa s-1"
    with pytest.raises(ValueError, match="w units must describe velocity"):
        shell_dataset(ds, level_index=0, radius=100., **kwargs)
    ds.w.attrs["units"] = "m s-1"
    ds.height.attrs["units"] = "hPa"
    with pytest.raises(ValueError, match="coordinate unit conflicts"):
        shell_dataset(ds, level_index=0, radius=100., **kwargs)


def test_shell_cli_numba_backend(tmp_path):
    pytest.importorskip("numba")
    lat, lon = np.linspace(-90, 90, 5), np.arange(8) * 45.
    dims = ("time", "height", "latitude", "longitude")
    shape = (1, 1, len(lat), len(lon))
    ds = xr.Dataset(
        {name: (dims, np.zeros(shape)) for name in ("u", "v", "w")},
        coords={"time": [0], "height": [0.], "latitude": lat, "longitude": lon},
    )
    source, output = tmp_path / "input.nc", tmp_path / "numba.zarr"
    ds.to_netcdf(source)
    main([
        "shell", "--velocity-file", str(source), "--output", str(output), "--time-index", "0",
        "--level-index", "0", "--level-dim", "height", "--coordinate-type", "height",
        "--coordinate-unit", "m", "--radius", "100", "--mollifier-scale", "1.4",
        "--pair-backend", "numba", "--cache-longitude-moments", "--compact-support",
    ])
    with xr.open_zarr(output) as actual:
        assert actual.attrs["pair_backend"] == "numba"
        expected = shell_dataset(
            ds, time_index=0, level_index=0, level_dim="height", coordinate_type="height",
            coordinate_unit="m", radius=100., mollifier_scale=1.4,
        )
        for name in expected.data_vars:
            np.testing.assert_allclose(actual[name], expected[name], rtol=2e-11, atol=2e-11)
