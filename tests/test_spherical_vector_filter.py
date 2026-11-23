import numpy as np
import pytest
from numpy.polynomial import Legendre

from lossett.calc.field_increments import compute_delta_u_cubed
from lossett.filtering.spherical_vector_filter import (
    commuting_vector_kernel,
    filter_vector_field_tangent_normal,
    geodesic_vector_increment,
)


# Unit sphere, with a smooth, nonconstant isotropic kernel.
MU, MU_WEIGHTS = np.polynomial.legendre.leggauss(32)
AZIMUTH = 2 * np.pi * (np.arange(64) + 0.5) / 64
KERNEL_MU = np.exp(3 * (MU - 1))
NORMALIZATION = np.sum(MU_WEIGHTS * KERNEL_MU)
KERNEL_GRID = np.unique(np.concatenate((np.linspace(-1, 1, 8193), MU)))
KERNEL_PROFILE = np.exp(3 * (KERNEL_GRID - 1))


def tangent_basis(points):
    east = np.stack((-points[..., 1], points[..., 0], np.zeros_like(points[..., 0])), axis=-1)
    east = east / np.linalg.norm(east, axis=-1, keepdims=True)
    north = np.cross(points, east)
    return east, north


def harmonic_value(points, degree, axis):
    return Legendre.basis(degree)(points @ axis)


def harmonic_gradient(points, degree, axis):
    cosine = points @ axis
    derivative = Legendre.basis(degree).deriv()(cosine)
    return derivative[..., None] * (axis - cosine[..., None] * points)


def field_value(points, modes):
    result = np.zeros_like(points)
    for degree, axis, gradient_coeff, rotated_coeff in modes:
        gradient = harmonic_gradient(points, degree, axis)
        result += gradient_coeff * gradient
        result += rotated_coeff * np.cross(points, gradient)
    return result


def filter_field(origins, modes, commuting=False):
    """Filter using Gauss-Legendre latitude and uniform azimuth about each origin."""
    origins = np.atleast_2d(origins)
    east, north = tangent_basis(origins)
    radial = (
        np.sin(AZIMUTH)[None, :, None] * east[:, None, :]
        + np.cos(AZIMUTH)[None, :, None] * north[:, None, :]
    )
    normal = (
        np.cos(AZIMUTH)[None, :, None] * east[:, None, :]
        - np.sin(AZIMUTH)[None, :, None] * north[:, None, :]
    )
    sin_r = np.sqrt(1 - MU**2)
    targets = (
        MU[None, :, None, None] * origins[:, None, None, :]
        + sin_r[None, :, None, None] * radial[:, None, :, :]
    )
    final_tangent = (
        -sin_r[None, :, None, None] * origins[:, None, None, :]
        + MU[None, :, None, None] * radial[:, None, :, :]
    )
    target_east, target_north = tangent_basis(targets)
    vectors = field_value(targets, modes)
    eastward = np.sum(vectors * target_east, axis=-1)
    northward = np.sum(vectors * target_north, axis=-1)
    sin_initial = np.broadcast_to(np.sin(AZIMUTH), eastward.shape)
    cos_initial = np.broadcast_to(np.cos(AZIMUTH), eastward.shape)
    sin_final = np.sum(final_tangent * target_east, axis=-1)
    cos_final = np.sum(final_tangent * target_north, axis=-1)
    # Validate that our constructed normal is the transported initial normal.
    np.testing.assert_allclose(
        np.cross(final_tangent, targets),
        np.broadcast_to(normal[:, None, :, :], final_tangent.shape),
        atol=2e-14,
    )
    weights = np.broadcast_to(MU_WEIGHTS[None, :, None], eastward.shape)
    scalar_kernel = np.broadcast_to(KERNEL_MU[None, :, None], eastward.shape)
    shape = (len(origins), -1)
    if commuting:
        kernel = KERNEL_PROFILE
        kwargs = {
            "mu": np.broadcast_to(MU[None, :, None], eastward.shape).reshape(shape),
            "mu_grid": KERNEL_GRID,
        }
    else:
        kernel = scalar_kernel.reshape(shape)
        kwargs = {"mode": "parallel_transport"}
    filtered_east, filtered_north = filter_vector_field_tangent_normal(
        eastward.reshape(shape),
        northward.reshape(shape),
        sin_initial.reshape(shape),
        cos_initial.reshape(shape),
        sin_final.reshape(shape),
        cos_final.reshape(shape),
        kernel,
        weights.reshape(shape),
        **kwargs,
    )
    return filtered_east[:, None] * east + filtered_north[:, None] * north


def scalar_multiplier(degree):
    return np.sum(MU_WEIGHTS * KERNEL_MU * Legendre.basis(degree)(MU)) / NORMALIZATION


def vector_multiplier(degree):
    """Analytic response of parallel-transport averaging of a vector harmonic."""
    polynomial = Legendre.basis(degree)
    eigenvalue = degree * (degree + 1)
    response = polynomial(MU) + (1 - MU) * polynomial.deriv()(MU) / eigenvalue
    return np.sum(MU_WEIGHTS * KERNEL_MU * response) / NORMALIZATION


def unit_vector(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


ORIGINS = np.array([
    unit_vector([1, 0.3, 0.4]),
    unit_vector([-0.2, 1, -0.5]),
    unit_vector([0.3, -0.6, 1]),
])
AXIS = unit_vector([0.4, -0.2, 1])


@pytest.mark.parametrize("degree", [1, 2, 3, 5])
@pytest.mark.parametrize("rotated", [False, True])
def test_filter_harmonic_gradient_modes(degree, rotated):
    modes = [(degree, AXIS, float(not rotated), float(rotated))]
    actual = filter_field(ORIGINS, modes)
    expected = vector_multiplier(degree) * field_value(ORIGINS, modes)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="The parallel-transport scalar-kernel filter has different scalar and vector harmonic multipliers",
)
def test_filter_commutes_with_scalar_gradient(degree):
    modes = [(degree, AXIS, 1.0, 0.0)]
    filtered_gradient = filter_field(ORIGINS, modes)
    gradient_of_filtered_scalar = scalar_multiplier(degree) * field_value(ORIGINS, modes)
    np.testing.assert_allclose(
        filtered_gradient, gradient_of_filtered_scalar, rtol=1e-8, atol=1e-10
    )


def test_degree_one_scalar_and_vector_responses_differ():
    scalar = scalar_multiplier(1)
    vector = vector_multiplier(1)
    np.testing.assert_allclose(vector, (1 + scalar) / 2, atol=1e-14)
    assert abs(vector - scalar) > 0.1


def test_commuting_kernel_linear_and_constant_inputs():
    mu = np.array([-1.0, -0.8, -0.2, 0.3, 1.0])
    np.testing.assert_allclose(commuting_vector_kernel(mu, 1 + mu), (1 + mu) / 2)
    np.testing.assert_array_equal(commuting_vector_kernel(mu, np.ones_like(mu)), 0)


def test_commuting_kernel_converges_to_exponential_reference():
    errors = []
    for size in (65, 257):
        mu = np.linspace(-1, 1, size)
        kernel = np.exp(3 * (mu - 1))
        exact = np.zeros_like(mu)
        exact[1:] = (
            kernel[1:]
            - (kernel[1:] - np.exp(-6)) / (3 * (1 + mu[1:]))
        )
        errors.append(np.max(np.abs(commuting_vector_kernel(mu, kernel) - exact)))
    assert errors[1] < errors[0] / 10
    assert errors[1] < 2e-5


@pytest.mark.parametrize(
    ("mu", "kernel"),
    [
        ([-1, 0, 0, 1], [1, 2, 3, 4]),
        ([-0.9, 0, 1], [1, 2, 3]),
        ([-1, 0, 1], [1, float("nan"), 3]),
        ([-1, 1], [1]),
    ],
)
def test_commuting_kernel_rejects_invalid_input(mu, kernel):
    with pytest.raises(ValueError):
        commuting_vector_kernel(mu, kernel)


def test_default_filter_constructs_vector_kernel_and_uses_scalar_normalization():
    east, north = filter_vector_field_tangent_normal(
        eastward=[1.0, 3.0],
        northward=[0.0, 0.0],
        sin_initial=[0.0, 0.0],
        cos_initial=[1.0, 1.0],
        sin_final=[0.0, 0.0],
        cos_final=[1.0, 1.0],
        kernel=[0.0, 2.0],
        area_weights=[1.0, 1.0],
        mu=[0.0, 1.0],
        mu_grid=[-1.0, 1.0],
    )
    np.testing.assert_allclose([east, north], [7 / 6, 0.0])


def test_default_filter_requires_kernel_geometry():
    with pytest.raises(ValueError, match="requires mu and mu_grid"):
        filter_vector_field_tangent_normal(
            [1.0], [0.0], [0.0], [1.0], [0.0], [1.0], [1.0], [1.0]
        )


def test_legacy_filter_remains_explicitly_available():
    east, north = filter_vector_field_tangent_normal(
        [1.0, 3.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0],
        [0.0, 0.0], [1.0, 1.0], [1.0, 3.0], [1.0, 1.0],
        mode="parallel_transport",
    )
    np.testing.assert_allclose([east, north], [2.5, 0.0])


def test_velocity_increment_matches_structure_function_integrand():
    initial = np.array([0.3, 1.1, -0.5])
    final = np.array([-0.2, 0.9, 0.7])
    u0, v0 = 0.7, -0.4
    u = np.array([1.2, -0.1, 0.8])
    v = np.array([-0.9, 0.3, 2.0])
    sin_initial, cos_initial = np.sin(initial), np.cos(initial)
    sin_final, cos_final = np.sin(final), np.cos(final)
    delta_east, delta_north = geodesic_vector_increment(
        u0, v0, u, v, sin_initial, cos_initial, sin_final, cos_final
    )
    longitudinal = delta_east * sin_initial + delta_north * cos_initial
    expected = longitudinal * (delta_east**2 + delta_north**2)
    actual, _ = compute_delta_u_cubed(
        u, v, u0, v0, sin_initial, cos_initial, sin_final, cos_final
    )
    np.testing.assert_allclose(actual, expected, atol=1e-14)


@pytest.mark.parametrize("degree", [1, 2, 3, 5])
@pytest.mark.parametrize("rotated", [False, True])
def test_commuting_filter_matches_scalar_harmonic_response(degree, rotated):
    modes = [(degree, AXIS, float(not rotated), float(rotated))]
    actual = filter_field(ORIGINS, modes, commuting=True)
    expected = scalar_multiplier(degree) * field_value(ORIGINS, modes)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-8)


def test_commuting_filter_for_mixed_divergence_and_curl_field():
    modes = [
        (1, unit_vector([1, 2, 3]), 1.3, -0.7),
        (2, unit_vector([-2, 1, 1]), -0.4, 0.8),
        (4, unit_vector([1, -2, 0.3]), 0.6, 0.5),
    ]
    actual = filter_field(ORIGINS, modes, commuting=True)
    expected = sum(
        scalar_multiplier(degree) * field_value(ORIGINS, [(degree, axis, a, b)])
        for degree, axis, a, b in modes
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize("commuting", [False, True])
def test_vector_laplacian_commutes_for_mixed_harmonics(commuting):
    modes = [
        (1, unit_vector([1, 2, 3]), 1.3, -0.7),
        (2, unit_vector([-2, 1, 1]), -0.4, 0.8),
        (4, unit_vector([1, -2, 0.3]), 0.6, 0.5),
    ]
    laplacian_modes = [
        (degree, axis, -degree * (degree + 1) * a, -degree * (degree + 1) * b)
        for degree, axis, a, b in modes
    ]
    filtered_laplacian = filter_field(ORIGINS, laplacian_modes, commuting=commuting)
    multiplier = scalar_multiplier if commuting else vector_multiplier
    laplacian_of_filtered = sum(
        -degree * (degree + 1)
        * multiplier(degree)
        * field_value(ORIGINS, [(degree, axis, a, b)])
        for degree, axis, a, b in modes
    )
    np.testing.assert_allclose(
        filtered_laplacian,
        laplacian_of_filtered,
        rtol=2e-6 if commuting else 2e-11,
        atol=3e-7 if commuting else 3e-11,
    )


@pytest.mark.parametrize("commuting", [False, True])
def test_rotation_covariance_for_mixed_velocity_field(commuting):
    axis = unit_vector([1, -1, 2])
    angle = 0.73
    cross = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    rotation = (
        np.eye(3) * np.cos(angle)
        + (1 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )
    modes = [
        (1, unit_vector([1, 2, 3]), 1.3, -0.7),
        (2, unit_vector([-2, 1, 1]), -0.4, 0.8),
        (4, unit_vector([1, -2, 0.3]), 0.6, 0.5),
    ]
    rotated_modes = [(l, rotation @ a, g, c) for l, a, g, c in modes]
    original = filter_field(ORIGINS, modes, commuting=commuting)
    rotated = filter_field(ORIGINS @ rotation.T, rotated_modes, commuting=commuting)
    np.testing.assert_allclose(rotated, original @ rotation.T, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("commuting", [False, True])
def test_global_divergence_and_curl_of_filtered_mixed_field(commuting):
    modes = [
        (1, unit_vector([1, 2, 3]), 1.3, -0.7),
        (2, unit_vector([-2, 1, 1]), -0.4, 0.8),
        (4, unit_vector([1, -2, 0.3]), 0.6, 0.5),
    ]
    mu, weights = np.polynomial.legendre.leggauss(12)
    longitude = np.arange(24) * (2 * np.pi / 24)
    cos_latitude = np.sqrt(1 - mu**2)
    points = np.stack(np.broadcast_arrays(
        cos_latitude[:, None] * np.cos(longitude),
        cos_latitude[:, None] * np.sin(longitude),
        mu[:, None],
    ), axis=-1)
    multiplier = scalar_multiplier if commuting else vector_multiplier
    expected = sum(
        multiplier(degree) * field_value(points, [(degree, axis, a, b)])
        for degree, axis, a, b in modes
    )
    actual = filter_field(points.reshape(-1, 3), modes, commuting=commuting).reshape(points.shape)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=2e-6 if commuting else 2e-10,
        atol=2e-8 if commuting else 2e-10,
    )

    expected_divergence = np.zeros(points.shape[:-1])
    expected_curl = np.zeros(points.shape[:-1])
    for degree, axis, gradient_coeff, rotated_coeff in modes:
        factor = -degree * (degree + 1) * multiplier(degree)
        mode = factor * harmonic_value(points, degree, axis)
        expected_divergence += gradient_coeff * mode
        expected_curl += rotated_coeff * mode

    latitude = np.broadcast_to(np.arcsin(mu)[:, None], points.shape[:-1])
    longitude_grid = np.broadcast_to(longitude, points.shape[:-1])
    step = 2e-5

    def components(latitudes, longitudes):
        locations = np.stack(
            (
                np.cos(latitudes) * np.cos(longitudes),
                np.cos(latitudes) * np.sin(longitudes),
                np.sin(latitudes),
            ),
            axis=-1,
        )
        flat = locations.reshape(-1, 3)
        filtered = np.concatenate([
            filter_field(flat[start:start + 24], modes, commuting=commuting)
            for start in range(0, len(flat), 24)
        ]).reshape(locations.shape)
        east, north = tangent_basis(locations)
        return np.sum(filtered * east, axis=-1), np.sum(filtered * north, axis=-1)

    east_plus, north_plus = components(latitude, longitude_grid + step)
    east_minus, north_minus = components(latitude, longitude_grid - step)
    lat_east_plus, lat_north_plus = components(latitude + step, longitude_grid)
    lat_east_minus, lat_north_minus = components(latitude - step, longitude_grid)
    cos_plus = np.cos(latitude + step)
    cos_minus = np.cos(latitude - step)
    divergence = (
        (east_plus - east_minus)
        + (lat_north_plus * cos_plus - lat_north_minus * cos_minus)
    ) / (2 * step * np.cos(latitude))
    curl = (
        (north_plus - north_minus)
        - (lat_east_plus * cos_plus - lat_east_minus * cos_minus)
    ) / (2 * step * np.cos(latitude))
    rtol = 2e-6 if commuting else 2e-7
    np.testing.assert_allclose(divergence, expected_divergence, rtol=rtol, atol=2e-7)
    np.testing.assert_allclose(curl, expected_curl, rtol=rtol, atol=2e-7)
    assert np.max(np.abs(divergence)) > 1
    assert np.max(np.abs(curl)) > 1
    area_weights = weights[:, None] * (2 * np.pi / len(longitude))
    np.testing.assert_allclose(np.sum(divergence * area_weights), 0.0, atol=2e-7)
    np.testing.assert_allclose(np.sum(curl * area_weights), 0.0, atol=2e-7)
