import numpy as np
import xarray as xr

from lossett.calc.field_increments import (
    compute_delta_u_cubed,
    compute_du3_angular_integral_global,
    compute_du3_angular_integral_subset,
)

TWOPI = 2*np.pi

ATOL_EXACT = 1e-12
RTOL_EXACT = 1e-6

ATOL_APPROX = 1e-5
RTOL_APPROX = 1e-5

ATOL_VAPPROX = 1e-3
ATOL_VVAPPROX = 1e-2

# Helpers

def synthetic_geometry():

    rng = np.random.default_rng(0)

    n_origin_lat = 4
    n_lat = 4
    n_lon = 8
    nbins = 3

    u = xr.DataArray(
        rng.standard_normal((n_lat, n_lon)),
        dims=("latitude", "longitude"),
    )

    v = xr.DataArray(
        rng.standard_normal((n_lat, n_lon)),
        dims=("latitude", "longitude"),
    )

    u0 = xr.DataArray(
        rng.standard_normal(n_origin_lat),
        dims=("origin_latitude",),
    )

    v0 = xr.DataArray(
        rng.standard_normal(n_origin_lat),
        dims=("origin_latitude",),
    )

    weights = rng.random(
        (n_origin_lat, n_lat, n_lon)
    )

    geom_chunk = xr.Dataset(
        {
            "sine_initial_bearing":
                (("origin_latitude","latitude","longitude"),
                 rng.standard_normal((n_origin_lat,n_lat,n_lon))),
            "cosine_initial_bearing":
                (("origin_latitude","latitude","longitude"),
                 rng.standard_normal((n_origin_lat,n_lat,n_lon))),
            "sine_final_bearing":
                (("origin_latitude","latitude","longitude"),
                 rng.standard_normal((n_origin_lat,n_lat,n_lon))),
            "cosine_final_bearing":
                (("origin_latitude","latitude","longitude"),
                 rng.standard_normal((n_origin_lat,n_lat,n_lon))),
            "great_circle_distance_bin":
                (
                    ("origin_latitude","latitude","longitude"),
                    rng.integers(
                        0,
                        nbins,
                        size=(n_origin_lat,n_lat,n_lon),
                        dtype=np.uint32,
                    ),
                ),
            "angular_weight":
                (
                    ("origin_latitude","latitude","longitude"),
                    weights,
                ),
        }
    )

    active_indices = [
        np.where(
            np.ones((n_lat, n_lon), dtype=bool)
        )
        for _ in range(n_origin_lat)
    ]
    return u, v, u0, v0, geom_chunk, active_indices, nbins

# Tests

# compute_delta_u_cubed

def test_constant_velocity_field():

    u = np.full((4, 8), 10.0)
    v = np.full((4, 8), -3.0)

    du3 = compute_delta_u_cubed(
        u=u,
        v=v,
        u0=10.0,
        v0=-3.0,
        sin_init=np.ones((4,8)),
        cos_init=np.zeros((4,8)),
        sin_final=np.ones((4,8)),
        cos_final=np.zeros((4,8))
    )

    exact = 0.0

    np.testing.assert_allclose(
        du3,
        exact,
        atol=ATOL_EXACT,
    )

# compute_du3_angular_integral_subset

def test_global_equals_subset_uniform():

    u, v, u0, v0, geom_chunk, active_indices, nbins = synthetic_geometry()

    result_global = compute_du3_angular_integral_global(
        u, v, u0, v0, geom_chunk, nbins
    )

    result_subset = compute_du3_angular_integral_subset(
        u, v, u0, v0,
        geom_chunk,
        active_indices,
        nbins,
    )

    np.testing.assert_allclose(
        result_global,
        result_subset,
        rtol=RTOL_EXACT,
        atol=ATOL_EXACT,
    )

def test_global_equals_subset_weighted():

    u, v, u0, v0, geom_chunk, active_indices, nbins = synthetic_geometry()

    result_global = compute_du3_angular_integral_global(
        u,
        v,
        u0,
        v0,
        geom_chunk,
        nbins,
        use_angular_weights=True,
    )

    result_subset = compute_du3_angular_integral_subset(
        u,
        v,
        u0,
        v0,
        geom_chunk,
        active_indices,
        nbins,
        use_angular_weights=True,
    )

    np.testing.assert_allclose(
        result_global,
        result_subset,
        rtol=RTOL_EXACT,
        atol=ATOL_EXACT,
    )

