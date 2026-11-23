import numpy as np
import pytest
import xarray as xr

from lossett.calc.compute_delta_u_cubed_spherical import (
    pack_active_geometry,
    prepare_distance_selection,
    process_origin_longitudes,
)
from lossett.calc.compute_inter_scale_transfers_spherical import compute_inter_scale_kinetic_energy_transfer
from lossett.calc.spherical_geometry import RADIUS_EARTH, compute_geometry
from tools.benchmark_real_fields import heat_time_for_increment_support, standard_atmosphere
from tools.benchmark_spherical_comparison import legacy_shell


def test_heat_calibration_matches_spherical_mollifier():
    time, response = heat_time_for_increment_support(4e6, 6.371e6)
    np.testing.assert_allclose(response, 0.949586432212, atol=1e-12)
    np.testing.assert_allclose(np.exp(-4 * time), response, atol=1e-14)
    with pytest.raises(ValueError):
        heat_time_for_increment_support(np.pi * 6.371e6, 6.371e6)


def test_standard_atmosphere_at_surface_and_tropopause():
    density, height, gravity = standard_atmosphere([101325., 22632.040095])
    np.testing.assert_allclose(density, [1.225, 0.363918], rtol=2e-5)
    np.testing.assert_allclose(height, [0, 11000], atol=0.01)
    assert gravity == 9.80665
    with pytest.raises(ValueError, match="0--20 km"):
        standard_atmosphere([1000.])


@pytest.mark.parametrize("weighted", [False, True])
def test_spherical_benchmark_assembles_and_integrates_existing_increment_path(weighted):
    lat = np.arange(-60., 61., 30.)
    lon = np.arange(0., 360., 45.)
    edges = np.linspace(0, 1.001 * np.pi * RADIUS_EARTH, 7)
    distances = (edges[:-1] + edges[1:]) / 2
    geometry = compute_geometry(lat, lat, lon, edges, dtype=np.float64, trig_fns=True)
    phi, lam = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    field = xr.Dataset(
        {"u": (("latitude", "longitude"), np.cos(phi) + np.sin(lam)),
         "v": (("latitude", "longitude"), np.sin(phi) * np.cos(lam)),
         "w": (("latitude", "longitude"), np.cos(phi) * np.sin(lam))},
        coords={"latitude": lat, "longitude": lon},
    )
    max_radius = 8e6
    actual, timing = legacy_shell(
        field, geometry, distances, edges, [(0, 2), (2, 5)], max_radius=max_radius, weighted=weighted,
    )
    geom = geometry.isel(origin_latitude=slice(2, 3))
    active = prepare_distance_selection(geom.great_circle_distance_bin.values, edges, max_R=max_radius)
    packed = pack_active_geometry(geom, active, use_angular_weights=weighted)
    increment = process_origin_longitudes(
        lon[1:2], field.u, field.v, geom, active, distances, active_geom=packed,
        use_angular_weights=weighted, w=field.w,
    ).rename({"great_circle_distance": "r"})
    expected = compute_inter_scale_kinetic_energy_transfer(
        increment.delta_u_cubed_angular_integral_longitudinal, np.array([max_radius / 2]),
        du3_vert=increment.delta_u_cubed_angular_integral_vertical,
    )
    xr.testing.assert_allclose(actual.sel(latitude=[lat[2]], longitude=lon[1:2]), expected)
    assert timing["increment_seconds_including_geometry_read"] >= timing["geometry_read_and_selection_seconds"]


def test_spherical_benchmark_full_sphere_preserves_in_support_result():
    lat = np.arange(-60., 61., 30.)
    lon = np.arange(0., 360., 45.)
    edges = np.linspace(0, 1.001 * np.pi * RADIUS_EARTH, 7)
    distances = (edges[:-1] + edges[1:]) / 2
    geometry = compute_geometry(lat, lat, lon, edges, dtype=np.float64, trig_fns=True)
    phi, lam = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    field = xr.Dataset(
        {"u": (("latitude", "longitude"), np.cos(phi) + np.sin(lam)),
         "v": (("latitude", "longitude"), np.sin(phi) * np.cos(lam)),
         "w": (("latitude", "longitude"), np.cos(phi) * np.sin(lam))},
        coords={"latitude": lat, "longitude": lon},
    )
    args = (field, geometry, distances, edges, [(0, 2), (2, 5)])
    capped, _ = legacy_shell(*args, max_radius=8e6, weighted=False)
    global_result, _ = legacy_shell(*args, max_radius=8e6, weighted=False, full_sphere=True)
    xr.testing.assert_allclose(global_result, capped)
