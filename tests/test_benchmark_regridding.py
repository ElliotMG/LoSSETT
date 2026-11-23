import numpy as np
import pytest
import xarray as xr

pytest.importorskip("ducc0")

from tools.preprocess_benchmark_fields import filter_shell, main
from lossett.calc.group_increments import latlon_shell_fields


def test_spectral_regridding_removes_aliased_mode_and_preserves_vector():
    lat, lon = np.linspace(-90, 90, 33), np.arange(64) * 360 / 64
    phi, lam = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    normal = np.stack((np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)), axis=-1)
    east = np.stack((-np.sin(lam), np.cos(lam), np.zeros_like(lam)), axis=-1)
    north = np.cross(normal, east)
    velocity = np.cross(np.array([0.3, -0.5, 0.7]), normal)
    u = np.sum(velocity * east, axis=-1)
    v = np.sum(velocity * north, axis=-1)
    degree11 = np.polynomial.legendre.legval(np.sin(phi), [0.] * 11 + [1.])
    w = 0.25 + 0.1 * degree11 + 0.8 * np.cos(phi)**20 * np.cos(20 * lam)
    u[[0, -1]] += 0.1 * np.cos(4 * lam[[0, -1]])
    w[[0, -1]] += 0.2 * np.cos(3 * lam[[0, -1]])
    target_lat, target_lon = lat[::2], lon[::2]
    outputs, limits, adjustments = filter_shell(
        u, v, w, lat, lon, {"coarse": (target_lat, target_lon)}
    )
    u_out, v_out, w_out = outputs["coarse"]
    np.testing.assert_allclose(u_out, np.sum(velocity * east, axis=-1)[::2, ::2], atol=2e-13)
    np.testing.assert_allclose(v_out, np.sum(velocity * north, axis=-1)[::2, ::2], atol=2e-13)
    np.testing.assert_allclose(w_out, 0.25 + 0.075 * degree11[::2, ::2], atol=2e-13)
    assert limits["coarse"][1] < 20
    assert min(adjustments) > 0.19
    latlon_shell_fields(np.deg2rad(target_lat), np.deg2rad(target_lon), u_out, v_out, w_out, radius=1.)
    direct = xr.DataArray(w, dims=("latitude", "longitude"), coords={"latitude": lat, "longitude": lon})
    sampled = direct.interp(latitude=target_lat, longitude=target_lon).values[len(target_lat) // 2]
    aliased_amplitude = 2 * np.dot(sampled - 0.25, np.cos(12 * np.deg2rad(target_lon))) / len(target_lon)
    np.testing.assert_allclose(aliased_amplitude, 0.8, atol=1e-13)


def test_filtered_netcdf_workflow(tmp_path):
    lat, lon = np.linspace(-90, 90, 73), np.arange(144) * 2.5
    shape = (1, 3, lat.size, lon.size)
    ds = xr.Dataset(
        {name: (("time", "pressure", "latitude", "longitude"), np.zeros(shape), {"units": "m s-1"})
         for name in ("u", "v", "w")},
        coords={"time": [0], "pressure": [150., 200., 250.], "latitude": lat, "longitude": lon},
    )
    ds.pressure.attrs["units"] = "hPa"
    source = tmp_path / "input.nc"
    ds.to_netcdf(source)
    output = tmp_path / "filtered"
    main([
        "--velocity-file", str(source), "--output-dir", str(output), "--pressure", "150", "200", "250",
        "--grid", "5p0deg",
    ])
    with xr.open_dataset(output / "filtered_5p0deg.nc") as result:
        assert result.u.shape == (1, 3, 37, 72)
        assert result.attrs["filter_stop_degree"] < 36
        assert result.attrs["pole_treatment"].startswith("longitude mean")
        np.testing.assert_array_equal(result.pressure, [150., 200., 250.])
        np.testing.assert_array_equal(result.w, 0.)
    with pytest.raises(FileExistsError):
        main(["--velocity-file", str(source), "--output-dir", str(output), "--pressure", "200", "--grid", "5p0deg"])
