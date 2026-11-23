import numpy as np
import pytest
import xarray as xr

from lossett.calc.compute_delta_u_cubed_spherical import (
    pack_active_geometry,
    process_origin_longitudes,
)
from lossett.calc.spherical_geometry import RADIUS_EARTH, compute_geometry


@pytest.mark.parametrize("origin_longitude", [-90.0, 0.0, 90.0])
@pytest.mark.parametrize("with_vertical", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_full_sphere_increment_pipeline(origin_longitude, with_vertical, weighted):
    latitudes = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])
    longitudes = np.arange(-180.0, 180.0, 45.0)
    origin_latitudes = np.array([-30.0, 30.0])
    edges = np.linspace(0.0, 1.001 * np.pi * RADIUS_EARTH, 7)
    distances = 0.5 * (edges[:-1] + edges[1:])
    geometry = compute_geometry(
        origin_latitudes,
        latitudes,
        longitudes,
        edges,
        dtype=np.float64,
        trig_fns=True,
    )

    rng = np.random.default_rng(42)
    coords = {"latitude": latitudes, "longitude": longitudes}
    u, v, w = [
        xr.DataArray(
            rng.normal(size=(len(latitudes), len(longitudes))),
            dims=("latitude", "longitude"),
            coords=coords,
        )
        for _ in range(3)
    ]
    if not with_vertical:
        w = None

    actual = process_origin_longitudes(
        np.array([origin_longitude]),
        u,
        v,
        geometry,
        active_indices=None,
        distances=distances,
        use_angular_weights=weighted,
        w=w,
    )

    active_indices = [
        np.where(np.ones((len(latitudes), len(longitudes)), dtype=bool))
        for _ in origin_latitudes
    ]
    active_geometry = pack_active_geometry(
        geometry, active_indices, use_angular_weights=weighted
    )
    subset = process_origin_longitudes(
        np.array([origin_longitude]),
        u,
        v,
        geometry,
        active_indices=active_indices,
        distances=distances,
        active_geom=active_geometry,
        use_angular_weights=weighted,
        w=w,
    )
    xr.testing.assert_allclose(actual, subset, rtol=1e-12, atol=1e-12)
    unpacked_subset = process_origin_longitudes(
        np.array([origin_longitude]),
        u,
        v,
        geometry,
        active_indices=active_indices,
        distances=distances,
        use_angular_weights=weighted,
        w=w,
    )
    xr.testing.assert_allclose(actual, unpacked_subset, rtol=1e-12, atol=1e-12)

    shift = int(origin_longitude / 45.0)
    target_columns = (np.arange(len(longitudes)) + shift) % len(longitudes)
    for i, origin_latitude in enumerate(origin_latitudes):
        u0 = u.sel(latitude=origin_latitude, longitude=origin_longitude).item()
        v0 = v.sel(latitude=origin_latitude, longitude=origin_longitude).item()
        sin_initial = geometry.sine_initial_bearing.values[i]
        cos_initial = geometry.cosine_initial_bearing.values[i]
        sin_final = geometry.sine_final_bearing.values[i]
        cos_final = geometry.cosine_final_bearing.values[i]
        target_u = u.values[:, target_columns]
        target_v = v.values[:, target_columns]
        du_t = target_u * sin_final + target_v * cos_final - u0 * sin_initial - v0 * cos_initial
        du_n = target_u * cos_final - target_v * sin_final - u0 * cos_initial + v0 * sin_initial
        du_sq = du_t**2 + du_n**2
        if with_vertical:
            w0 = w.sel(latitude=origin_latitude, longitude=origin_longitude).item()
            dw = w.values[:, target_columns] - w0
            du_sq += dw**2

        bins = geometry.great_circle_distance_bin.values[i].ravel()
        weights = geometry.angular_weight.values[i].ravel() if weighted else None
        counts = np.bincount(bins, minlength=len(distances))

        def integrate(values):
            values = values.ravel()
            valid = np.isfinite(values)
            sums = np.bincount(
                bins[valid],
                weights=values[valid] if weights is None else values[valid] * weights[valid],
                minlength=len(distances),
            )
            if weights is None:
                return np.divide(
                    2 * np.pi * sums,
                    counts,
                    out=np.full(len(distances), np.nan),
                    where=counts > 0,
                )
            return sums

        np.testing.assert_allclose(
            actual.delta_u_cubed_angular_integral_longitudinal.values[0, i],
            integrate(du_t * du_sq),
            rtol=1e-12,
            atol=1e-12,
        )
        if with_vertical:
            np.testing.assert_allclose(
                actual.delta_u_cubed_angular_integral_vertical.values[0, i],
                integrate(dw * du_sq),
                rtol=1e-12,
                atol=1e-12,
            )
        else:
            assert "delta_u_cubed_angular_integral_vertical" not in actual
