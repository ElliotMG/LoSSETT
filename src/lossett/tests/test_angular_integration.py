import numpy as np
import xarray as xr


from lossett.calc.compute_inter_scale_transfers_spherical import (
    # low level
    bin_integrate,
    angular_integral_by_distance_bin,
    # higher level (should really be separated at a later date)
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


I0 = np.i0(1) # I0 = modified Bessel function


# Helpers


def voronoi_widths_periodic(chi):
    """
    Compute Voronoi cells widths given angular cell centres chi.
    Correct handling of periodicity.
    """
    n = len(chi)


    widths = np.empty(n)


    for i in range(n):


        if i == 0:
            left_mid = 0.5 * (
                chi[-1] + chi[0] - 2*np.pi
            )
        else:
            left_mid = 0.5 * (
                chi[i-1] + chi[i]
            )


        if i == n-1:
            right_mid = 0.5 * (
                chi[i] + chi[0] + 2*np.pi
            )
        else:
            right_mid = 0.5 * (
                chi[i] + chi[i+1]
            )


        widths[i] = right_mid - left_mid


    return widths


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

# voronoi_widths_periodic

def test_voronoi_widths_sum_to_twopi():

    rng = np.random.default_rng(0)

    chi = np.sort(
        rng.random(100) * TWOPI
    )

    weights = voronoi_widths_periodic(chi)

    np.testing.assert_allclose(
        weights.sum(),
        TWOPI,
        atol=ATOL_EXACT,
    )

# bin_integrate

def test_bin_integrate_unweighted():
    values = np.array([1, 2, 3, 4])
    bins = np.array([0, 0, 1, 1])

    result = bin_integrate(
        values,
        bins,
        nbins=2,
    )

    np.testing.assert_allclose(
        result,
        [3, 7],
    )

def test_bin_integrate_weighted():
    values = np.array([1, 2, 3, 4])
    weights = np.array([10, 10, 100, 100])
    bins = np.array([0, 0, 1, 1])


    result = bin_integrate(
        values,
        bins,
        nbins=2,
        weights=weights,
    )


    np.testing.assert_allclose(
        result,
        [30, 700],
    )

# angular_integral_by_distance_bin

def test_nan_values_do_not_contaminate_weighted_integral():

    values = np.array([
        1.0,
        1.0,
        np.nan,
        1.0,
    ])

    weights = np.array([
        1.0,
        1.0,
        1.0,
        1.0,
    ])

    result = angular_integral_by_distance_bin(
        values,
        np.zeros(4, dtype=np.uint32),
        nbins=1,
        weights=weights,
    )

    assert np.isfinite(result).all()

def test_uniform_constant_function():
    n = 16

    result = angular_integral_by_distance_bin(
        np.ones(n),
        np.zeros(n, dtype=np.uint32),
        nbins=1,
    )

    np.testing.assert_allclose(
        result,
        [TWOPI],
    )

def test_weighted_constant_function():
    n = 16

    weights = np.full(
        n,
        TWOPI / n,
    )

    result = angular_integral_by_distance_bin(
        np.ones(n),
        np.zeros(n, dtype=np.uint32),
        nbins=1,
        weights=weights,
    )

    np.testing.assert_allclose(
        result,
        [TWOPI],
    )

def test_uniform_with_missing_points():
    values = np.array([1, 1, 1, 1, np.nan, np.nan])

    bins = np.zeros(6, dtype=np.uint32)

    result = angular_integral_by_distance_bin(
        values,
        bins,
        nbins=1,
    )

    expected = TWOPI * 4/6

    np.testing.assert_allclose(
        result,
        [expected],
    )

def test_weighted_with_missing_points():

    values = np.array([1, 1, 1, 1, np.nan, np.nan])

    weights = np.ones_like(values) * TWOPI / len(values)

    bins = np.zeros(6, dtype=np.uint32)

    result = angular_integral_by_distance_bin(
        values,
        bins,
        nbins=1,
        weights=weights,
    )

    expected = TWOPI * 4/6

    np.testing.assert_allclose(
        result,
        [expected],
    )

def test_weighted_quadrature_irregular_constant():
    """
    Test weighted angular quadrature on an irregular angular mesh.

    Uses

        f(chi) = 1

    whose exact integral is

        ∫ f dchi = 2π.

    This results should be exact to machine precision.
    """

    # Irregular angular locations
    n = 100 # number of points
    rng = np.random.default_rng(0)

    chi = np.sort(
        rng.random(n) * TWOPI
    )

    # Voronoi-cell widths on the circle
    weights = voronoi_widths_periodic(chi)

    result = angular_integral_by_distance_bin(
        np.ones_like(chi),
        np.zeros(n, dtype=np.uint32),
        nbins=1,
        weights=weights,
    )

    np.testing.assert_allclose(
        result,
        [TWOPI],
        atol=ATOL_EXACT,
    )

def test_weighted_quadrature_irregular_cosine():
    """
    Test weighted angular quadrature on an irregular angular mesh.

    Uses

        f(chi) = 1 + cos(chi)

    whose exact integral is

        ∫ f dchi = 2π.

    This result should be approximate.
    """

    tol = ATOL_VAPPROX # has to be lower as cosine does not integrate exactly
    # on an irregular mesh with Voronoi quadrature

    # Irregular angular locations
    n = 100 # number of points
    rng = np.random.default_rng(0)

    chi = np.sort(
        rng.random(n) * TWOPI
    )

    # Voronoi-cell widths on the circle
    weights = voronoi_widths_periodic(chi)

    # Check that the quadrature weights cover the circle
    np.testing.assert_allclose(
        weights.sum(),
        TWOPI,
        atol=tol,
    )

    f = 1.0 + np.cos(chi)

    result = angular_integral_by_distance_bin(
        integrand=f,
        bins=np.zeros(n, dtype=np.uint32),
        nbins=1,
        weights=weights,
    )

    exact = TWOPI

    np.testing.assert_allclose(
        result,
        [exact],
        atol=tol,
    )

def test_weighted_equals_unweighted_on_uniform_mesh():
    """
    This result should be exact.
    """

    n = 100

    chi = np.linspace(
        0.0,
        TWOPI,
        n,
        endpoint=False,
    )

    f = (
        1.0
        + np.cos(chi)
        + 0.5*np.sin(3*chi)
    )

    bins = np.zeros(n, dtype=np.uint32)

    uniform_weights = np.full(
        n,
        TWOPI / n,
    )

    weighted = angular_integral_by_distance_bin(
        f,
        bins,
        nbins=1,
        weights=uniform_weights,
    )

    unweighted = angular_integral_by_distance_bin(
        f,
        bins,
        nbins=1,
    )

    np.testing.assert_allclose(
        weighted,
        unweighted,
        atol=ATOL_EXACT,
    )

def test_weighted_beats_unweighted_on_irregular_mesh():

    # Irregular angular locations
    n = 100 # number of points
    rng = np.random.default_rng(0)

    chi = np.sort(
        rng.random(n) * TWOPI
    )

    weights = voronoi_widths_periodic(chi)

    f = 1.0 + np.cos(chi)

    weighted = angular_integral_by_distance_bin(
        f,
        np.zeros(n, dtype=np.uint32),
        nbins=1,
        weights=weights,
    )[0]

    unweighted = angular_integral_by_distance_bin(
        f,
        np.zeros(n, dtype=np.uint32),
        nbins=1,
    )[0]

    exact = TWOPI

    assert abs(weighted - exact) < abs(unweighted - exact)

def test_uniform_quadrature_integrates_cosine_exactly():

    exact = 0.0
    
    for n in (8, 16, 32, 64, 128):

        chi = np.linspace(
            0.0,
            TWOPI,
            n,
            endpoint=False,
        )

        f = np.cos(chi)

        result = angular_integral_by_distance_bin(
            f,
            np.zeros(n, dtype=np.uint32),
            nbins=1,
        )

        np.testing.assert_allclose(
            result,
            [exact],
            atol=ATOL_EXACT
        )

def test_weighted_quadrature_integrates_cosine_exactly():

    exact = 0.0

    n = 360
        
    rng = np.random.default_rng(n)

    chi = np.sort(
        rng.random(n) * TWOPI
    )

    weights = voronoi_widths_periodic(chi)

    f = np.cos(chi)

    result = angular_integral_by_distance_bin(
        f,
        np.zeros(len(chi), dtype=np.uint32),
        nbins=1,
        weights=weights,
    )

    np.testing.assert_allclose(
        result,
        [exact],
        atol=ATOL_VAPPROX,
    )

def test_uniform_quadrature_converges():

    exact = TWOPI * I0

    errors = []

    for n in (8, 16, 32, 64, 128):

        chi = np.linspace(
            0.0,
            TWOPI,
            n,
            endpoint=False,
        )

        f = np.exp(np.cos(chi)) # must be something that isn't exactly integrated on the uniform mesh

        result = angular_integral_by_distance_bin(
            f,
            np.zeros(n, dtype=np.uint32),
            nbins=1,
        )

        assert np.isfinite(result).all()

        errors.append(np.abs(result[0] - exact))

    assert errors[-1] < 0.1*errors[0]

def test_weighted_quadrature_converges():

    exact = TWOPI * I0

    errors = []

    for n in (8, 16, 32, 64, 128):
        
        rng = np.random.default_rng(n)

        chi = np.sort(
            rng.random(n) * TWOPI
        )

        weights = voronoi_widths_periodic(chi)

        f = np.exp(np.cos(chi)) # keep the same as uniform mesh

        result = angular_integral_by_distance_bin(
            f,
            np.zeros(n, dtype=np.uint32),
            nbins=1,
            weights=weights,
        )

        assert np.isfinite(result).all()

        errors.append(np.abs(result[0] - exact))

    assert errors[-1] < 0.1*errors[0]

def test_uniform_quadrature_is_linear():

    n = 100

    chi = np.linspace(
        0.0,
        TWOPI,
        n,
        endpoint=False,
    )

    f = np.cos(chi)
    g = np.sin(chi)

    a = 2.5
    b = -1.7
    bins = np.zeros(n, dtype=np.uint32)

    lhs = angular_integral_by_distance_bin(
        a*f + b*g,
        bins,
        nbins=1,
    )

    rhs = (
        a * angular_integral_by_distance_bin(
            f,
            bins,
            nbins=1,
        )
        +
        b * angular_integral_by_distance_bin(
            g,
            bins,
            nbins=1,
        )
    )

    np.testing.assert_allclose(
        lhs,
        rhs,
        atol=ATOL_EXACT,
    )

def test_weighted_quadrature_is_linear():

    rng = np.random.default_rng(0)

    chi = np.sort(
        rng.random(100) * TWOPI
    )

    weights = voronoi_widths_periodic(chi)

    f = np.cos(chi)
    g = np.sin(chi)

    a = 2.5
    b = -1.7

    bins = np.zeros(len(chi), dtype=np.uint32)

    lhs = angular_integral_by_distance_bin(
        a*f + b*g,
        bins,
        nbins=1,
        weights=weights,
    )

    rhs = (
        a * angular_integral_by_distance_bin(
            f,
            bins,
            nbins=1,
            weights=weights,
        )
        +
        b * angular_integral_by_distance_bin(
            g,
            bins,
            nbins=1,
            weights=weights,
        )
    )

    np.testing.assert_allclose(
        lhs,
        rhs,
        atol=ATOL_EXACT,
    )

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
