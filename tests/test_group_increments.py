import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

import lossett.calc.group_increments as group_increments
from lossett.calc.group_increments import (
    group_increment_grid,
    group_increment_shell,
    hydrostatic_radius,
    latlon_shell_fields,
    reconstruct_group_vertical,
)
from lossett.filtering.group_kernels import (
    _heat_weight, _mollifier_spectral_coefficients, heat_fiber_moments, mollifier_fiber_moments,
)


@pytest.mark.parametrize("time", [0.002, 0.1, 0.5])
def test_heat_moments_normalization_and_matched_scalar_multipliers(time):
    mu, weights = leggauss(400 if time < 0.01 else 100)
    g, h, k = heat_fiber_moments(mu, 2 * time)
    np.testing.assert_allclose(2 * np.pi * weights @ g, 1, atol=2e-11)
    for degree in (1, 2, 4):
        coeff = np.zeros(degree + 1)
        coeff[-1] = 1
        response = 2 * np.pi * weights @ (g * np.polynomial.legendre.legval(mu, coeff))
        np.testing.assert_allclose(response, np.exp(-2 * time * degree * (degree + 1)), atol=2e-11)
    assert np.all(np.abs(h) <= g + 1e-12)
    assert np.all(np.abs(k) <= g + 1e-12)


def test_mollifier_matched_group_multipliers():
    mu, weights = leggauss(128)
    scale = 0.6
    g, h, k = mollifier_fiber_moments(mu, scale)
    lower = np.cos(2 * scale)
    x = lower + (1 - lower) * (mu + 1) / 2
    profile = np.exp(-1 / (1 - (np.arccos(x) / (2 * scale))**2))
    for degree in (0, 1, 2, 4):
        coeff = np.zeros(degree + 1)
        coeff[-1] = 1
        h_response = weights @ (profile * np.polynomial.legendre.legval(x, coeff)) / (weights @ profile)
        g_response = 2 * np.pi * weights @ (g * np.polynomial.legendre.legval(mu, coeff))
        np.testing.assert_allclose(g_response, h_response**2, atol=1e-10)
    assert np.all(g >= -1e-12)
    assert np.all(np.abs(h) <= g + 1e-12)
    assert np.all(np.abs(k) <= g + 1e-12)


def test_mollifier_has_exact_compact_support_at_pilot_scale():
    scale = 1e6 / 6382758.772639869
    angles = np.array([0., 4 * scale - 0.01, 4 * scale + 0.01, np.pi])
    moments = mollifier_fiber_moments(np.cos(angles), scale)
    for moment in moments:
        np.testing.assert_array_equal(moment[2:], 0.)
    assert moments[0][0] > 0
    nodes, weights = leggauss(256)
    g, _, _ = mollifier_fiber_moments(nodes, scale)
    lower = np.cos(2 * scale)
    x = lower + (1 - lower) * (nodes + 1) / 2
    profile = np.exp(-1 / (1 - (np.arccos(x) / (2 * scale))**2))
    for degree in (0, 1, 4):
        coeff = np.zeros(degree + 1)
        coeff[-1] = 1
        h_response = weights @ (profile * np.polynomial.legendre.legval(x, coeff)) / (weights @ profile)
        g_response = 2 * np.pi * weights @ (g * np.polynomial.legendre.legval(nodes, coeff))
        np.testing.assert_allclose(g_response, h_response**2, atol=1e-10)


def test_mollifier_reuses_spectral_coefficients(monkeypatch):
    original = np.polynomial.legendre.leggauss
    counts = []

    def count_quadrature(n):
        counts.append(n)
        return original(n)

    monkeypatch.setattr(np.polynomial.legendre, "leggauss", count_quadrature)
    _mollifier_spectral_coefficients.cache_clear()
    try:
        mu = np.array([1., 0.9, 0.7, -0.5])
        first = mollifier_fiber_moments(mu, 0.5)
        initial_calls = len(counts)
        second = mollifier_fiber_moments(mu[::-1], 0.5)
        assert initial_calls > 0
        assert len(counts) == initial_calls
        for expected, actual in zip(first, second):
            np.testing.assert_array_equal(actual[::-1], expected)
    finally:
        _mollifier_spectral_coefficients.cache_clear()


def test_latlon_gradients_for_solid_rotation():
    lat = np.linspace(-1.2, 1.2, 129)
    lon = np.arange(128) * 2 * np.pi / 128
    u = np.broadcast_to(np.cos(lat)[:, None], (len(lat), len(lon)))
    zero = np.zeros_like(u)
    points, velocity, grad, gradw = latlon_shell_fields(lat, lon, u, zero, zero, radius=2.)
    expected = np.cross(np.array([0., 0., 1.]), points)
    np.testing.assert_allclose(velocity, expected, atol=1e-14)
    strain_twice = grad + np.swapaxes(grad, -1, -2)
    np.testing.assert_allclose(strain_twice[1:-1], 0, atol=3e-5)
    np.testing.assert_allclose(strain_twice, 0, atol=6e-5)
    np.testing.assert_array_equal(gradw, 0)


@pytest.mark.parametrize("nlat", [33, 65])
def test_polar_gradients_for_smooth_cartesian_velocity(nlat):
    lat = np.linspace(-np.pi / 2, np.pi / 2, nlat)
    lon = np.arange(64) * 2 * np.pi / 64
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    normal = np.stack((np.cos(lat2) * np.cos(lon2), np.cos(lat2) * np.sin(lon2), np.sin(lat2)), axis=-1)
    east = np.stack((-np.sin(lon2), np.cos(lon2), np.zeros_like(lon2)), axis=-1)
    north = np.cross(normal, east)
    fixed = np.array([0.3, -0.5, 0.4])
    omega = np.array([-0.2, 0.1, 0.6])
    velocity = fixed + np.cross(omega, normal)
    u = np.sum(velocity * east, axis=-1)
    v = np.sum(velocity * north, axis=-1)
    w = np.sum(velocity * normal, axis=-1)
    points, actual_velocity, grad, gradw = latlon_shell_fields(lat, lon, u, v, w, radius=2.)
    np.testing.assert_allclose(actual_velocity, velocity, atol=1e-14)
    for i in (0, -1):
        p = np.eye(3) - np.outer(points[i, 0], points[i, 0])
        cross = np.cross(omega, np.eye(3)).T
        expected_grad = p @ cross @ p / 2 - np.dot(fixed, points[i, 0]) * p / 2
        expected_gradw = p @ fixed / 2
        np.testing.assert_allclose(grad[i], np.broadcast_to(expected_grad, grad[i].shape), atol=1e-14)
        np.testing.assert_allclose(gradw[i], np.broadcast_to(expected_gradw, gradw[i].shape), atol=1e-14)


def test_polar_fields_reject_inconsistent_vectors():
    lat = np.linspace(-np.pi / 2, np.pi / 2, 9)
    lon = np.arange(16) * 2 * np.pi / 16
    u = np.ones((lat.size, lon.size))
    with pytest.raises(ValueError, match="same Cartesian vector"):
        latlon_shell_fields(lat, lon, u, np.zeros_like(u), np.zeros_like(u), radius=1.)


def test_polar_gradient_converges_on_nonlinear_field():
    errors = []
    for nlat in (33, 65):
        lat = np.linspace(-np.pi / 2, np.pi / 2, nlat)
        lon = np.arange(64) * 2 * np.pi / 64
        lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
        n = np.stack((np.cos(lat2) * np.cos(lon2), np.cos(lat2) * np.sin(lon2), np.sin(lat2)), axis=-1)
        east = np.stack((-np.sin(lon2), np.cos(lon2), np.zeros_like(lon2)), axis=-1)
        north = np.cross(n, east)
        velocity = np.zeros_like(n)
        velocity[..., 0] = n[..., 0] * n[..., 2]
        u = np.sum(velocity * east, axis=-1)
        v = np.sum(velocity * north, axis=-1)
        w = np.sum(velocity * n, axis=-1)
        _, _, grad, _ = latlon_shell_fields(lat, lon, u, v, w, radius=1.)
        errors.append(abs(grad[-1, 0, 0, 0] - 1))
    assert 3.9 < errors[0] / errors[1] < 4.1


def test_hydrostatic_radius_constant_density():
    pressure = np.array([100., 90., 70.])
    np.testing.assert_allclose(hydrostatic_radius(pressure, [2., 2., 2.], 5., 10.), [10., 11., 13.])


def test_grid_rejects_underresolved_filter():
    points = np.concatenate((np.eye(3), -np.eye(3)))
    with pytest.raises(ValueError, match="under-resolves"):
        group_increment_grid(
            points, np.zeros((6, 3)), np.zeros((6, 3, 3)), np.zeros((6, 3)),
            np.full(6, 4 * np.pi / 6), radius=1., heat_time=0.001,
        )


@pytest.mark.parametrize("kernel", [{"heat_time": 0.2}, {"mollifier_scale": 0.6}])
def test_cached_longitude_moments_match_reference(kernel):
    lat = np.linspace(-np.pi / 2, np.pi / 2, 3)
    lon = 0.37 + np.arange(4) * np.pi / 2
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    points = np.stack((np.cos(lat2) * np.cos(lon2), np.cos(lat2) * np.sin(lon2), np.sin(lat2)), axis=-1)
    velocity = np.stack((points[..., 1] + 0.2, points[..., 2]**2, points[..., 0] - 0.3), axis=-1)
    projector = np.eye(3) - points[..., :, None] * points[..., None, :]
    grad = np.einsum("...ij,jk,...kl->...il", projector, np.diag([0.3, -0.5, 0.2]), projector)
    gradw = np.einsum("...ij,j->...i", projector, [0.2, -0.1, 0.4])
    weights = np.broadcast_to(np.array([1., 2., 1.])[:, None] * np.pi / 4, lat2.shape)
    args = (points, velocity, grad, gradw, weights)
    reference = group_increment_grid(*args, radius=2., quadrature_tolerance=1., **kernel)
    cached = group_increment_grid(*args, radius=2., quadrature_tolerance=1., cache_longitude_moments=True, **kernel)
    for key in reference:
        np.testing.assert_allclose(cached[key], reference[key], rtol=2e-12, atol=2e-12)
    for caching in (False, True):
        with pytest.raises(ValueError, match="under-resolves"):
            group_increment_grid(*args, radius=2., quadrature_tolerance=1e-8,
                                 cache_longitude_moments=caching, **kernel)


def test_cached_longitude_moments_reject_irregular_grid():
    lat, lon = np.meshgrid(np.linspace(-1, 1, 3), np.arange(4) * np.pi / 2, indexing="ij")
    points = np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1)
    points[1, 1] = points[1, 0]
    with pytest.raises(ValueError, match="uniform global longitude"):
        group_increment_grid(
            points, np.zeros_like(points), np.zeros(points.shape + (3,)),
            np.zeros_like(points), np.full((3, 4), np.pi / 3),
            radius=1., heat_time=0.2, cache_longitude_moments=True,
        )


@pytest.mark.parametrize("scale", [np.pi / 8, 0.5, 1.0])
@pytest.mark.parametrize("cache", [False, True])
def test_compact_support_matches_full_grid(scale, cache, monkeypatch):
    lat, lon = np.meshgrid(np.linspace(-np.pi / 2, np.pi / 2, 5), 0.37 + np.arange(8) * np.pi / 4, indexing="ij")
    points = np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1)
    velocity = np.stack((points[..., 1] + 0.2, points[..., 2]**2, points[..., 0] - 0.3), axis=-1)
    projector = np.eye(3) - points[..., :, None] * points[..., None, :]
    grad = np.einsum("...ij,jk,...kl->...il", projector, np.diag([0.3, -0.5, 0.2]), projector)
    gradw = np.einsum("...ij,j->...i", projector, [0.2, -0.1, 0.4])
    weights = np.full(lat.shape, 4 * np.pi / lat.size)
    args = (points, velocity, grad, gradw, weights)
    options = dict(radius=2., quadrature_tolerance=5., mollifier_scale=scale, cache_longitude_moments=cache)
    reference = group_increment_grid(*args, **options)
    sizes = []
    original = group_increments._group_increment_shell

    def record_targets(*args, **kwargs):
        sizes.append(len(args[2]))
        return original(*args, **kwargs)

    monkeypatch.setattr(group_increments, "_group_increment_shell", record_targets)
    subset = group_increment_grid(*args, compact_support=True, **options)
    assert len(sizes) == lat.size
    if 4 * scale < np.pi:
        assert max(sizes) < lat.size
    else:
        assert min(sizes) == lat.size
    for key in reference:
        np.testing.assert_allclose(subset[key], reference[key], rtol=2e-11, atol=2e-11)


def test_compact_support_validates_whole_grid_and_rejects_heat():
    lat, lon = np.meshgrid(np.linspace(-np.pi / 2, np.pi / 2, 5), np.arange(8) * np.pi / 4, indexing="ij")
    points = np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1)
    velocity = np.zeros_like(points)
    grad = np.zeros(points.shape + (3,))
    weights = np.full(lat.shape, 4 * np.pi / lat.size)
    args = (points, velocity, grad, velocity.copy(), weights)
    with pytest.raises(ValueError, match="matched mollifier"):
        group_increment_grid(*args, radius=1., heat_time=0.2, compact_support=True)
    velocity[-1, 0] = np.nan
    with pytest.raises(ValueError, match="inputs must be finite"):
        group_increment_grid(*args, radius=1., mollifier_scale=0.5, compact_support=True)


def test_compact_support_reports_empty_selection():
    lat, lon = np.meshgrid(np.linspace(-np.pi / 2, np.pi / 2, 3), np.arange(4) * np.pi / 2, indexing="ij")
    points = (1 - 4e-11) * np.stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1,
    )
    velocity = np.zeros_like(points)
    weights = np.full(lat.shape, 4 * np.pi / lat.size)
    with pytest.raises(ValueError, match="under-resolves.*no targets in support"):
        group_increment_grid(
            points, velocity, np.zeros(points.shape + (3,)), velocity, weights,
            radius=1., mollifier_scale=1e-8, compact_support=True, cache_longitude_moments=True,
        )


def test_shell_against_explicit_fiber_quadrature():
    x = np.array([0., 0., 1.])
    y = np.array([[0.6, 0, 0.8], [-0.8, 0, 0.6]])
    u = np.array([0.3, -0.5, 0.4])
    uy = np.array([[0.1, 0.7, -0.3], [0.5, -0.4, 0.9]])
    grad = np.array([np.diag([0., 1.2, 0.]), np.diag([0., -0.4, 0.])])
    gradw = np.array([[0., 0.2, 0.], [0., -0.6, 0.]])
    weights = np.array([2 * np.pi, 2 * np.pi])
    radius, time = 3., 0.2
    actual = group_increment_shell(x, u, y, uy, grad, gradw, weights, radius=radius, heat_time=time)
    psi = 2 * np.pi * np.arange(4096) / 4096
    c, s = np.cos(psi), np.sin(psi)
    wx = u @ x
    expected = dict(horizontal=0., structure=0., vertical_flux=0., mixed_moment=np.zeros(3))
    for j in range(len(y)):
        e1 = np.array([y[j, 2], 0., -y[j, 0]])
        e2 = np.array([0., 1., 0.])
        wy = uy[j] @ y[j]
        vy = uy[j] - wy * y[j]
        p = u[0] * e1 + u[1] * e2
        rp = (c[:, None] * p + s[:, None] * np.cross(y[j], p))
        d = vy - rp
        dw = wy - wx
        q = np.sum(d * d, axis=1) + dw**2
        cos_rotation = (1 + y[j, 2]) * (1 + c) / 2 - 1
        weight = _heat_weight(cos_rotation, 2 * time) / (4 * np.pi)
        projection = np.eye(3) - np.outer(y[j], y[j])
        strain = (grad[j] + grad[j].T) / 2 + wy * projection / radius
        b = gradw[j] - vy / radius
        horizontal = (np.trace(grad[j]) + 2 * wx / radius) * q
        horizontal += 2 * np.einsum("ni,ij,nj->n", d, strain, d) + 2 * dw * (d @ b)
        d_unrotated = c[:, None] * d - s[:, None] * np.cross(y[j], d)
        back = np.stack((d_unrotated @ e1, d_unrotated @ e2, np.full(len(psi), dw)), axis=1)
        expected["horizontal"] -= weights[j] * np.mean(weight * horizontal) / 4
        expected["structure"] += weights[j] * np.mean(weight * q)
        expected["vertical_flux"] += weights[j] * np.mean(weight * dw * q)
        expected["mixed_moment"] += weights[j] * np.mean((weight * dw)[:, None] * back, axis=0)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-12, atol=2e-12)


def test_vertical_height_pressure_and_options():
    height = np.linspace(0, 4, 81)
    radius = 100 + height
    rho = np.exp(-height / 10)
    gravity = 2.
    # p'(z)=-rho*g exactly; derivatives converge under refinement.
    pressure = 20 * np.exp(-height / 10)
    velocity = np.stack([height, height**2, 1 + height], axis=1)
    structure = 2 + height
    flux = height**2
    moment = np.stack([height, 2 * height, height**2], axis=1)
    args = (radius, [0, 0, 1], velocity, structure, flux, moment)
    deep = reconstruct_group_vertical(height, *args, fluid="anelastic", reference_density=rho)
    shallow = reconstruct_group_vertical(height, *args, fluid="anelastic", reference_density=rho, atmosphere="shallow")
    np.testing.assert_allclose(deep - shallow, -flux / (2 * radius))
    boussinesq = reconstruct_group_vertical(height, *args)
    np.testing.assert_allclose(deep - boussinesq, (flux + velocity[:, 2] * structure) / 40, atol=2e-14)
    from_pressure = reconstruct_group_vertical(pressure, *args, coordinate_type="pressure", fluid="anelastic", reference_density=rho, gravity=gravity)
    np.testing.assert_allclose(from_pressure[1:-1], deep[1:-1], rtol=0.0002, atol=0.0002)
    with pytest.raises(ValueError, match="reference_density"):
        reconstruct_group_vertical(pressure, *args, coordinate_type="pressure", gravity=gravity)


def test_constant_cartesian_flow_horizontal_and_vertical_cancel():
    mu, lat_weights = leggauss(8)
    phi = np.arange(16) * 2 * np.pi / 16
    z, az = np.meshgrid(mu, phi, indexing="ij")
    points = np.stack([np.sqrt(1-z*z)*np.cos(az), np.sqrt(1-z*z)*np.sin(az), z], axis=-1)
    weights = np.broadcast_to(lat_weights[:, None] * 2 * np.pi / 16, z.shape)
    u = np.broadcast_to([0.3, -0.5, 0.4], points.shape)
    radius = 3.
    w = np.sum(u * points, axis=-1)
    v = u - w[..., None] * points
    projector = np.eye(3) - points[..., :, None] * points[..., None, :]
    grad = -w[..., None, None] * projector / radius
    gradw = v / radius
    result = group_increment_grid(points, u, grad, gradw, weights, radius=radius, heat_time=0.2)
    np.testing.assert_allclose(result["horizontal"], result["vertical_flux"] / (2 * radius), atol=1e-14)
    origin = (3, 5)
    levels = np.array([2.9, 3., 3.1])
    vertical = reconstruct_group_vertical(
        levels, levels, points[origin], np.tile(u[origin], (3, 1)),
        np.full(3, result["structure"][origin]),
        np.full(3, result["vertical_flux"][origin]),
        np.tile(result["mixed_moment"][origin], (3, 1)),
    )
    np.testing.assert_allclose(result["horizontal"][origin] + vertical[1], 0, atol=1e-14)


def test_anelastic_group_separation_identity():
    mu, wm = leggauss(18)
    phi = 2 * np.pi * np.arange(36) / 36
    z, az = np.meshgrid(mu, phi, indexing="ij")
    y = np.stack([np.sqrt(1-z*z)*np.cos(az), np.sqrt(1-z*z)*np.sin(az), z], axis=-1).reshape(-1, 3)
    weights = np.broadcast_to(wm[:, None] * 2 * np.pi / len(phi), z.shape).ravel()

    def field(position):
        x, yy, z = np.moveaxis(position, -1, 0)
        mass = np.stack([x + yy**2, -yy + z**2, np.zeros_like(x)], axis=-1)
        rho = 1 + np.sum(position**2, axis=-1)
        j = np.zeros(position.shape[:-1] + (3, 3))
        j[..., 0, 0] = 1
        j[..., 0, 1] = 2 * yy
        j[..., 1, 1] = -1
        j[..., 1, 2] = 2 * z
        jacobian = j / rho[..., None, None] - mass[..., :, None] * (2 * position)[..., None, :] / rho[..., None, None]**2
        return mass / rho[..., None], jacobian

    def evaluate(x):
        radius = np.linalg.norm(x)
        origin = x / radius
        ux, jx = field(x)
        uy, jy = field(radius * y)
        wy = np.sum(uy * y, axis=1)
        vy = uy - wy[:, None] * y
        projection = np.eye(3) - y[:, :, None] * y[:, None, :]
        grad = np.einsum("nik,nkl,nlj->nij", projection, jy, projection) - wy[:, None, None] * projection / radius
        gradw = np.einsum("nji,nj->ni", jy, y)
        gradw -= np.sum(gradw * y, axis=1)[:, None] * y
        gradw += vy / radius
        result = group_increment_shell(origin, ux, y, uy, grad, gradw, weights, radius=radius, heat_time=0.2)
        return result, ux, jx, uy, jy

    x = np.array([0.6, -0.4, 0.8])
    radius = np.linalg.norm(x)
    origin = x / radius
    eps = 2e-5
    shells = [evaluate((radius + shift) * origin) for shift in (-eps, 0, eps)]
    results = [item[0] for item in shells]
    ux, jx, uy, jy = shells[1][1:]
    vertical = reconstruct_group_vertical(
        radius + np.array([-eps, 0, eps]), radius + np.array([-eps, 0, eps]), origin,
        np.array([item[1] for item in shells]),
        np.array([r["structure"] for r in results]),
        np.array([r["vertical_flux"] for r in results]),
        np.array([r["mixed_moment"] for r in results]),
        fluid="anelastic", reference_density=1 + (radius + np.array([-eps, 0, eps]))**2,
    )[1]
    grad_s = np.array([
        (evaluate(x + eps * axis)[0]["structure"] - evaluate(x - eps * axis)[0]["structure"]) / (2 * eps)
        for axis in np.eye(3)
    ])
    nx = jx @ ux
    ny = np.einsum("nij,nj->ni", jy, uy)
    cross = np.cross(origin, y)
    cosine = y @ origin

    def minimum_rotation(v):
        first = np.cross(cross, v)
        return v + first + np.cross(cross, first) / (1 + cosine)[:, None]

    ru, rn = minimum_rotation(ux), minimum_rotation(nx)
    psi = 2 * np.pi * np.arange(128) / 128
    c, s = np.cos(psi), np.sin(psi)

    def fiber_rotate(v):
        return (c[None, :, None] * v[:, None, :]
                + s[None, :, None] * np.cross(y, v)[:, None, :]
                + (1-c)[None, :, None] * np.sum(y * v, axis=1)[:, None, None] * y[:, None, :])

    du = uy[:, None, :] - fiber_rotate(ru)
    dn = ny[:, None, :] - fiber_rotate(rn)
    cos_angle = (1 + cosine[:, None]) * (1 + c[None, :]) / 2 - 1
    W = _heat_weight(cos_angle, 0.4) / (4 * np.pi)
    X = -0.5 * np.dot(weights, np.mean(W * np.sum(du * dn, axis=2), axis=1))
    np.testing.assert_allclose(results[1]["horizontal"] + vertical, X + ux @ grad_s / 4, atol=3e-8)
