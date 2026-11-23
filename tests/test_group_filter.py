import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss, legval

from lossett.calc.group_filter import group_filter_shell
from lossett.calc.group_increments import group_increment_shell, reconstruct_group_vertical
from lossett.filtering.group_kernels import _heat_weight, mollifier_les_fiber_moments


def spherical_quadrature(nlat=16):
    mu, weights = leggauss(nlat)
    phi = 2 * np.pi * np.arange(2 * nlat) / (2 * nlat)
    z, az = np.meshgrid(mu, phi, indexing="ij")
    points = np.stack((np.sqrt(1-z*z)*np.cos(az), np.sqrt(1-z*z)*np.sin(az), z), axis=-1)
    area = np.broadcast_to(weights[:, None] * 2 * np.pi / len(phi), z.shape)
    return points.reshape(-1, 3), area.ravel()


def test_tensor_filter_matches_explicit_fiber_rotations():
    x = np.array([0., 0., 1.])
    y = np.array([[0.6, 0., 0.8], [-0.8, 0., 0.6]])
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(2, 3))
    tensors = rng.normal(size=(2, 3, 3))
    weights = np.array([2 * np.pi, 2 * np.pi])
    time = 0.3
    result = group_filter_shell(x, y, vectors, tensors, weights, heat_time=time, quadrature_tolerance=100)
    psi = 2 * np.pi * np.arange(2048) / 2048
    c, s = np.cos(psi), np.sin(psi)
    spin = np.zeros((len(psi), 3, 3))
    spin[:, 0, 0] = c
    spin[:, 0, 1] = s
    spin[:, 1, 0] = -s
    spin[:, 1, 1] = c
    spin[:, 2, 2] = 1
    expected_v = np.zeros(3)
    expected_t = np.zeros((3, 3))
    for i in range(len(y)):
        e1 = np.array([y[i, 2], 0., -y[i, 0]])
        frame = np.stack((e1, [0., 1., 0.], y[i]), axis=1)
        rotation = np.einsum("nij,jk->nik", spin, frame.T)
        W = _heat_weight((1 + y[i, 2]) * (1 + c) / 2 - 1, time) / (4 * np.pi)
        expected_v += weights[i] * np.mean(W[:, None] * np.einsum("nij,j->ni", rotation, vectors[i]), axis=0)
        expected_t += weights[i] * np.mean(
            W[:, None, None] * np.einsum("nik,kl,njl->nij", rotation, tensors[i], rotation), axis=0
        )
    np.testing.assert_allclose(result["vector"], expected_v, atol=1e-12)
    np.testing.assert_allclose(result["tensor"], expected_t, atol=1e-12)


@pytest.mark.parametrize("kernel", ["heat", "mollifier"])
def test_group_filter_full_vector_and_tensor_harmonics(kernel):
    points, weights = spherical_quadrature(80 if kernel == "mollifier" else 30)
    x = np.array([0.3, -0.4, np.sqrt(0.75)])
    vector = np.array([0.7, -0.2, 0.4])
    tensor = np.outer(vector, vector)
    if kernel == "heat":
        args = {"heat_time": 0.35}
        m1, m2 = np.exp(-0.35 * 2), np.exp(-0.35 * 6)
    else:
        scale = 0.6
        args = {"mollifier_scale": scale}
        mu, w = leggauss(300)
        lower = np.cos(2 * scale)
        m = lower + (1 - lower) * (mu + 1) / 2
        profile = np.exp(-1 / (1 - (np.arccos(m) / (2 * scale))**2))
        m1, m2 = [np.dot(w * profile, legval(m, [0] * n + [1])) / np.dot(w, profile) for n in (1, 2)]
    result = group_filter_shell(
        x, points, np.broadcast_to(vector, points.shape),
        np.broadcast_to(tensor, (len(points), 3, 3)), weights, **args
    )
    iso = np.trace(tensor) * np.eye(3) / 3
    np.testing.assert_allclose(result["vector"], m1 * vector, atol=3e-7)
    np.testing.assert_allclose(result["tensor"], iso + m2 * (tensor - iso), atol=3e-7)


def test_mollifier_les_moments_against_abel_fiber_integral():
    scale = 0.6
    lower = np.cos(2 * scale)
    roots, quadrature = leggauss(200)
    mu_nodes = lower + (1 - lower) * (roots + 1) / 2
    theta = np.arccos(mu_nodes)
    profile = np.exp(-1 / (1 - (theta / (2 * scale))**2))
    normalization = np.pi * (1 - lower) * (quadrature @ profile)
    psi = 2 * np.pi * np.arange(2048) / 2048
    c = np.cos(psi)
    for mu in (0.25, 0.85):
        z = (1 + mu) * (1 + c) / 4
        upper = np.sqrt(np.maximum(z - (1 + lower) / 2, 0))
        u = upper[:, None] * (roots[None, :] + 1) / 2
        x = z[:, None] - u**2
        angle = np.arccos(np.clip(2 * x - 1, -1, 1))
        r = angle / (2 * scale)
        derivative = np.zeros_like(angle)
        valid = (r < 1) & (angle > 0)
        derivative[valid] = (
            np.exp(-1 / (1 - r[valid]**2)) / normalization
            * angle[valid] / (scale**2 * np.sin(angle[valid]) * (1 - r[valid]**2)**2)
        )
        weight = 8 * np.pi * np.sqrt(z) * upper / 2 * (derivative @ quadrature)
        expected = np.array([np.mean(weight * factor) / (4 * np.pi)
                             for factor in (np.ones_like(c), c, 2*c*c - 1)])
        actual = np.array(mollifier_les_fiber_moments(mu, scale))
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize(
    ("scale", "degree", "expected"),
    [(0.1, 29, -0.0117080023184), (0.6, 5, -0.0495582411597)],
)
def test_mollifier_cannot_be_a_real_matched_kernel_at_this_scale(scale, degree, expected):
    mu, weights = leggauss(400)
    lower = np.cos(2 * scale)
    x = lower + (1 - lower) * (mu + 1) / 2
    profile = np.exp(-1 / (1 - (np.arccos(x) / (2 * scale))**2))
    response = weights @ (profile * legval(x, [0] * degree + [1])) / (weights @ profile)
    np.testing.assert_allclose(response, expected, atol=1e-10)
    assert response < 0


def test_matched_les_shell_identity_with_anelastic_flow():
    points, weights = spherical_quadrature(8)
    time = 0.4
    radius = 1.3
    step = 1e-4

    def field(position):
        x, y, z = np.moveaxis(position, -1, 0)
        r2 = np.sum(position**2, axis=-1)
        p = x + x*y + x*z + y*z + 0.3*z*z
        dp = np.stack((1+y+z, x+z, x+y+0.6*z), axis=-1)
        hess = np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.6]])
        mass = np.stack((2*y*p + r2*dp[..., 1], -2*x*p - r2*dp[..., 0], np.zeros_like(x)), axis=-1)
        dm = np.zeros(position.shape[:-1] + (3, 3))
        dm[..., 0, :] = (2*np.eye(3)[1]*p[..., None] + 2*y[..., None]*dp
                         + 2*position*dp[..., 1, None] + r2[..., None]*hess[1])
        dm[..., 1, :] = -(2*np.eye(3)[0]*p[..., None] + 2*x[..., None]*dp
                          + 2*position*dp[..., 0, None] + r2[..., None]*hess[0])
        rho = 1 + r2
        return mass / rho[..., None], dm / rho[..., None, None] - mass[..., :, None] * (2*position)[..., None, :] / rho[..., None, None]**2

    def samples(r):
        velocity, jac = field(r * points)
        w = np.sum(velocity * points, axis=1)
        v = velocity - w[:, None] * points
        projection = np.eye(3) - points[:, :, None] * points[:, None, :]
        grad = np.einsum("nik,nkl,nlj->nij", projection, jac, projection) - w[:, None, None] * projection / r
        gradw = np.einsum("nji,nj->ni", jac, points)
        gradw -= np.sum(gradw * points, axis=1)[:, None] * points
        gradw += v / r
        return velocity, grad, gradw

    def filtered(x, velocity):
        result = group_filter_shell(
            x / np.linalg.norm(x), points, velocity,
            velocity[:, :, None] * velocity[:, None, :], weights,
            heat_time=time, quadrature_tolerance=1e-4,
        )
        U = result["vector"]
        return U, result["tensor"] - np.outer(U, U)

    levels = radius + np.array([-step, 0., step])
    outputs = []
    for r in levels:
        velocity, grad, gradw = samples(r)
        shell = []
        for origin, u in zip(points, velocity):
            inc = group_increment_shell(origin, u, points, velocity, grad, gradw, weights, radius=r, heat_time=time)
            U, stress = filtered(r * origin, velocity)
            wr = u @ origin
            flux = ((U @ U / 2) * (U @ origin) + (U @ stress) @ origin
                    - (u @ u / 2) * wr + wr * inc["structure"] / 4)
            shell.append((inc, flux, U, stress))
        outputs.append((velocity, shell))

    differences = []
    local_index = 23
    for i, origin in enumerate(points):
        incs = [o[1][i][0] for o in outputs]
        vertical = reconstruct_group_vertical(
            levels, levels, origin, np.array([o[0][i] for o in outputs]),
            np.array([inc["structure"] for inc in incs]),
            np.array([inc["vertical_flux"] for inc in incs]),
            np.array([inc["mixed_moment"] for inc in incs]),
            fluid="anelastic", reference_density=1 + levels**2,
        )[1]
        gradient = np.empty((3, 3))
        x = radius * origin
        for j, axis in enumerate(np.eye(3)):
            xp, xm = x + step * axis, x - step * axis
            up = filtered(xp, samples(np.linalg.norm(xp))[0])[0]
            um = filtered(xm, samples(np.linalg.norm(xm))[0])[0]
            gradient[:, j] = (up - um) / (2 * step)
        stress = outputs[1][1][i][3]
        les = -np.sum(stress * gradient)
        differences.append((incs[1]["horizontal"] + vertical, les))
        if i == local_index:
            local_left = differences[-1][0] - les
    means = weights @ np.array(differences) / (4 * np.pi)
    mean_flux = np.array([weights @ np.array([sample[1] for sample in o[1]]) / (4 * np.pi) for o in outputs])
    rho = 1 + levels**2
    rhs = ((rho[2]*levels[2]**2*mean_flux[2] - rho[0]*levels[0]**2*mean_flux[0])
           / (2*step*rho[1]*radius**2))
    assert abs(means[1]) > 1e-4
    assert abs(rhs) > 1e-4
    np.testing.assert_allclose(means[0] - means[1], rhs, rtol=2e-7, atol=1e-7)

    def filter_vector(x, vector, filter_time):
        return group_filter_shell(
            x / np.linalg.norm(x), points, vector, np.zeros((len(points), 3, 3)),
            weights, heat_time=filter_time, quadrature_tolerance=1e-4,
        )["vector"]

    def base_flux(x):
        r = np.linalg.norm(x)
        normal = x / r
        velocity, grad, gradw = samples(r)
        u, _ = field(x)
        U, stress = filtered(x, velocity)
        inc = group_increment_shell(normal, u, points, velocity, grad, gradw, weights, radius=r, heat_time=time)
        energy_velocity = np.sum(velocity**2, axis=1)[:, None] * velocity / 2
        filtered_energy_velocity = filter_vector(x, energy_velocity, 2 * time)
        A = (U @ U / 2) * U + stress @ U
        return (1 + r*r) * (
            A - (filtered_energy_velocity + (u @ u / 2) * u) / 2 + u * inc["structure"] / 4
        )

    x = radius * points[local_index]
    divergence = sum(
        (base_flux(x + step * axis)[j] - base_flux(x - step * axis)[j]) / (2 * step)
        for j, axis in enumerate(np.eye(3))
    ) / (1 + radius**2)
    u, jac = field(x)
    n = jac @ u
    target_u, target_jac = field(radius * points)
    target_n = np.einsum("nij,nj->ni", target_jac, target_u)
    U = outputs[1][1][local_index][2]
    filtered_n = filter_vector(x, target_n, time)
    group_n = filter_vector(x, target_n, 2 * time)
    group_u = filter_vector(x, target_u, 2 * time)
    commutator_divergence = (u @ group_n + group_u @ n - 2 * U @ filtered_n) / 2
    local_right = divergence + commutator_divergence
    assert abs(local_left) > 1e-3
    np.testing.assert_allclose(local_left, local_right, rtol=2e-7, atol=1e-7)
