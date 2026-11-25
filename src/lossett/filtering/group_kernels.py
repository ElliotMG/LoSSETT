"""Central SO(3) kernels and their endpoint fiber moments."""

from functools import lru_cache

import numpy as np


def _heat_weight(cos_angle, time):
    """Heat-kernel density relative to normalized Haar measure."""
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    if time < 0.1:
        # The omitted image terms are below double precision in this regime.
        numerator = np.zeros_like(angle)
        for image in (-1, 0, 1):
            shift = angle - 2 * np.pi * image
            numerator += (-1.0) ** image * shift * np.exp(-shift**2 / (4 * time))
        denom = np.sin(angle / 2)
        ratio = np.empty_like(angle)
        nonzero = angle > 0
        ratio[nonzero] = numerator[nonzero] / denom[nonzero]
        # Limit as the angle tends to zero.
        image_limit = 0.0
        for image in (-1, 0, 1):
            shift = -2 * np.pi * image
            image_limit += (-1.0) ** image * np.exp(
                -shift**2 / (4 * time)
            ) * (1 - shift**2 / (2 * time))
        ratio[~nonzero] = 2 * image_limit
        return np.sqrt(np.pi) * np.exp(time / 4) * ratio / (2 * time**1.5)

    nmax = int(np.ceil(np.sqrt(-np.log(1e-16) / time))) + 4
    degrees = np.arange(nmax + 1)
    coefficients = (2 * degrees + 1) * np.exp(-time * degrees * (degrees + 1))
    # chi_n(phi) = 1 + 2 sum_{j=1}^n cos(j phi).
    cosine_coefficients = np.cumsum(coefficients[::-1])[::-1]
    return np.polynomial.chebyshev.chebval(
        cos_angle, np.r_[cosine_coefficients[0], 2 * cosine_coefficients[1:]]
    )


def heat_fiber_moments(mu, time, *, tolerance=1e-10, max_angles=32768):
    """Return g, h, k per unit solid angle for the SO(3) heat kernel.

    Its degree-n multiplier is exp(-time*n*(n+1)).  A filter with time
    ``2*time`` is the exact group convolution of two filters with ``time``.
    Fiber quadrature is refined until all three moments converge.
    """
    mu = np.asarray(mu, dtype=float)
    if not np.all(np.isfinite(mu)) or np.any((mu < -1) | (mu > 1)):
        raise ValueError("mu must be finite and between -1 and 1")
    if not np.isfinite(time) or time <= 0:
        raise ValueError("time must be positive and finite")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    if max_angles < 32:
        raise ValueError("max_angles must be at least 32")

    original_shape = mu.shape
    flat = mu.ravel()
    result = np.empty((3, flat.size))
    # Limit temporary allocations independently of the size of the input grid.
    for start in range(0, flat.size, 128):
        x = flat[start:start + 128]
        previous = None
        count = 32
        while count <= max_angles:
            psi = 2 * np.pi * np.arange(count) / count
            c = np.cos(psi)
            cosine = (1 + x[:, None]) * (1 + c[None, :]) / 2 - 1
            weight = _heat_weight(cosine, time)
            current = np.stack([
                np.mean(weight, axis=1),
                np.mean(weight * c, axis=1),
                np.mean(weight * (2 * c**2 - 1), axis=1),
            ]) / (4 * np.pi)
            if previous is not None:
                scale = max(float(np.max(np.abs(current))), 1.0)
                if np.max(np.abs(current - previous)) <= tolerance * scale:
                    result[:, start:start + len(x)] = current
                    break
            previous = current
            count *= 2
        else:
            raise ValueError("SO(3) fiber quadrature did not converge; increase max_angles")
    return tuple(component.reshape(original_shape) for component in result)


def mollifier_les_fiber_moments(mu, angular_scale, *, tolerance=1e-9, max_nodes=1024):
    """Return the moments of H with the standard-mollifier scalar endpoint kernel."""
    mu = np.asarray(mu, dtype=float)
    if not np.all(np.isfinite(mu)) or np.any((mu < -1) | (mu > 1)):
        raise ValueError("mu must be finite and between -1 and 1")
    if not np.isfinite(angular_scale) or not 0 < angular_scale < np.pi / 2:
        raise ValueError("angular_scale must be between 0 and pi/2")
    if not np.isfinite(tolerance) or tolerance <= 0 or max_nodes < 64:
        raise ValueError("tolerance must be positive and max_nodes at least 64")
    lower = np.cos(2 * angular_scale)
    flat = mu.ravel()
    previous = None
    count = 32
    while count <= max_nodes:
        nodes, weights = np.polynomial.legendre.leggauss(count)

        def profile(x):
            theta = np.arccos(np.clip(x, -1, 1))
            inside = theta < 2 * angular_scale
            value = np.zeros_like(theta)
            ratio = theta[inside] / (2 * angular_scale)
            value[inside] = np.exp(-1 / (1 - ratio**2))
            return value

        full_x = lower + (1 - lower) * (nodes + 1) / 2
        normalization = np.pi * (1 - lower) * (weights @ profile(full_x))
        scalar = profile(flat) / normalization
        width = np.maximum(flat - lower, 0)
        x = lower + width[:, None] * (nodes[None, :] + 1) / 2
        values = profile(x) / normalization
        integral = width / 2 * (values @ weights)
        integral_h = width / 2 * ((values * (1 + 2*x - flat[:, None])) @ weights)
        denominator = 1 + flat
        h = scalar.copy()
        k = scalar.copy()
        valid = denominator > 0
        h[valid] -= integral[valid] / denominator[valid]
        k[valid] = h[valid] - 3 * integral_h[valid] / denominator[valid]**2
        current = np.stack((scalar, h, k))
        if previous is not None:
            scale = max(float(np.max(np.abs(current))), 1.0)
            if np.max(np.abs(current - previous)) <= tolerance * scale:
                return tuple(component.reshape(mu.shape) for component in current)
        previous = current
        count *= 2
    raise ValueError("mollifier LES moments did not converge; increase max_nodes")


_PANEL_NODES = 32


@lru_cache(maxsize=32)
def _mollifier_spectral_coefficients(angular_scale, degree):
    """Squared scalar multipliers of the standard mollifier, weighted by 2l+1.

    The multipliers are integrals of the mollifier against P_l. They are built
    with composite Gauss-Legendre panels in the colatitude and the three-term
    Legendre recurrence, so memory is O(degree) rather than O(degree**2).
    """
    nodes, node_weights = np.polynomial.legendre.leggauss(_PANEL_NODES)
    panels = max(1, -(-(2 * degree + 2) // _PANEL_NODES))
    width = 2 * angular_scale / panels
    starts = width * np.arange(panels)
    theta = (starts[:, None] + width * (nodes[None, :] + 1) / 2).ravel()
    x = np.cos(theta)
    profile = np.exp(-1 / (1 - (theta / (2 * angular_scale))**2))
    weighted = np.tile(node_weights * width / 2, panels) * np.sin(theta) * profile
    multiplier = np.empty(degree + 1)
    previous = np.ones_like(x)
    multiplier[0] = weighted @ previous
    if degree >= 1:
        current = x.copy()
        multiplier[1] = weighted @ current
        for order in range(1, degree):
            following = ((2 * order + 1) * x * current - order * previous) / (order + 1)
            previous, current = current, following
            multiplier[order + 1] = weighted @ current
    multiplier /= multiplier[0]
    return (2 * np.arange(degree + 1) + 1) * multiplier**2


def _jacobi_series(mu, coefficients, beta, start):
    p0 = np.ones_like(mu)
    result = coefficients[start] * p0
    if start + 1 == len(coefficients):
        return result
    p1 = (-beta + (beta + 2) * mu) / 2
    result += coefficients[start + 1] * p1
    for n in range(2, len(coefficients) - start):
        a = (2 * n + beta - 1) * ((2 * n + beta) * (2 * n + beta - 2) * mu - beta**2)
        b = 2 * (n - 1) * (n + beta - 1) * (2 * n + beta)
        p = (a * p1 - b * p0) / (2 * n * (n + beta) * (2 * n + beta - 2))
        result += coefficients[start + n] * p
        p0, p1 = p1, p
    return result


def mollifier_fiber_moments(mu, angular_scale, *, tolerance=1e-9, max_degree=16384):
    """Matched group moments for a scalar-endpoint standard mollifier H.

    The mollifier is exp(-1/(1-(theta/(2*angular_scale))**2)) inside
    theta < 2*angular_scale, and zero outside. Its matched group filter G
    has the squared scalar degree multipliers of H. Angular scale is radians.
    Successive spectral and integration refinements must agree.
    """
    mu = np.asarray(mu, dtype=float)
    if not np.all(np.isfinite(mu)) or np.any((mu < -1) | (mu > 1)):
        raise ValueError("mu must be finite and between -1 and 1")
    if not np.isfinite(angular_scale) or not 0 < angular_scale < np.pi / 2:
        raise ValueError("angular_scale must be between 0 and pi/2")
    if not np.isfinite(tolerance) or tolerance <= 0 or max_degree < 64:
        raise ValueError("tolerance must be positive and max_degree at least 64")
    original_shape = mu.shape
    flat = mu.ravel()
    previous = None
    degree = 32
    while degree <= max_degree:
        coeff = _mollifier_spectral_coefficients(angular_scale, degree)
        normalized = coeff / (4 * np.pi)
        half = (1 + flat) / 2
        # Jacobi factors are the spin-one and spin-two rotation matrix elements.
        current = np.stack((
            np.polynomial.legendre.legval(flat, normalized),
            half * _jacobi_series(flat, normalized, 2, 1),
            half**2 * _jacobi_series(flat, normalized, 4, 2),
        ))
        if previous is not None:
            scale = max(float(np.max(np.abs(current))), 1.0)
            tail_indicator = np.sum(coeff[-degree // 4:] * (2 * np.arange(degree + 1)[-degree // 4:] + 1)) / (4 * np.pi)
            if np.max(np.abs(current - previous)) <= tolerance * scale and tail_indicator <= tolerance * scale:
                if 4 * angular_scale < np.pi:
                    current[:, flat < np.cos(4 * angular_scale)] = 0
                return tuple(component.reshape(original_shape) for component in current)
        previous = current
        degree *= 2
    raise ValueError(
        "mollifier group moments did not converge; increase max_degree or use "
        "mollifier_fiber_moments_quadrature"
    )


def _mollifier_profile(theta, angular_scale):
    ratio = (theta / (2 * angular_scale))**2
    out = np.zeros_like(theta)
    inside = ratio < 1
    out[inside] = np.exp(-1 / (1 - ratio[inside]))
    return out


@lru_cache(maxsize=16)
def _central_kernel_series(angular_scale, n_abel, n_central, n_abel_quad):
    """Chebyshev series in rotation angle of the matched central kernel G(phi).

    The scalar multipliers of the mollifier are half-integer cosine transforms of
    its Abel transform W, so the squared multipliers give
    sin(phi/2) G(phi) proportional to the convolution of W with W'. Everything is
    evaluated in physical space, with no degree truncation. The overall constant
    is left to the caller, which normalizes the scalar moment g.
    """
    cheb = np.polynomial.chebyshev
    s = angular_scale
    x_nodes, x_weights = np.polynomial.legendre.leggauss(n_abel_quad)
    v, v_weights = (x_nodes + 1) / 2, x_weights / 2
    first = np.cos(np.pi * (np.arange(n_abel) + .5) / n_abel)[::-1]
    t = s * (first + 1)
    separation = (2 * s - t)[:, None] * v[None, :]**2
    theta = t[:, None] + separation
    # cos(t)-cos(theta) written as a product to avoid cancellation at small angles
    denominator = np.maximum(np.sqrt(2 * np.sin((theta + t[:, None]) / 2) * np.sin(separation / 2)), 1e-300)
    jacobian = 2 * (2 * s - t)[:, None] * v[None, :]
    abel = np.sum(v_weights * jacobian * _mollifier_profile(theta, s) * np.sin(theta) / denominator, axis=1)
    abel_series = cheb.chebfit(first, abel, n_abel - 1)
    derivative_series = cheb.chebder(abel_series) / s

    def abel_even(x):
        return cheb.chebval(np.clip(np.abs(x) / s - 1, -1, 1), abel_series) * (np.abs(x) < 2 * s)

    def abel_slope(x):
        return np.sign(x) * cheb.chebval(np.clip(np.abs(x) / s - 1, -1, 1), derivative_series) * (np.abs(x) < 2 * s)

    second = np.cos(np.pi * (np.arange(n_central) + .5) / n_central)[::-1]
    phi = 2 * s * (second + 1)
    convolution = np.zeros(n_central)
    quad_nodes, quad_weights = np.polynomial.legendre.leggauss(max(300, n_central // 2))
    for index, angle in enumerate(phi):
        lower, upper = max(angle - 2 * s, -2 * s), min(angle + 2 * s, 2 * s)
        a = lower + (upper - lower) * (quad_nodes + 1) / 2
        convolution[index] = np.sum(quad_weights * (upper - lower) / 2 * abel_even(a) * abel_slope(angle - a))
    return cheb.chebfit(second, convolution / np.sin(phi / 2), n_central - 1)


def _central_fiber_moments(mu, angular_scale, series, n_fiber):
    """Unnormalized (g, h, k) fiber integrals of the central kernel at each mu."""
    cheb = np.polynomial.chebyshev
    support = 4 * angular_scale
    nodes, weights = np.polynomial.legendre.leggauss(n_fiber)
    out = np.zeros((3, len(mu)))
    for index, value in enumerate(mu):
        argument = 2 * (1 + np.cos(support)) / (1 + value) - 1
        if argument >= 1:
            continue
        psi_max = np.arccos(max(argument, -1))
        psi = psi_max * (nodes + 1) / 2
        phi = np.arccos(np.clip((1 + value) * (1 + np.cos(psi)) / 2 - 1, -1, 1))
        kernel = cheb.chebval(np.clip(phi / (2 * angular_scale) - 1, -1, 1), series) * psi_max * weights
        out[:, index] = (kernel.sum(), (kernel * np.cos(psi)).sum(), (kernel * np.cos(2 * psi)).sum())
    return out


@lru_cache(maxsize=16)
def _quadrature_moment_table(angular_scale, n_abel, n_central, n_abel_quad, n_fiber, n_panels, order):
    series = _central_kernel_series(angular_scale, n_abel, n_central, n_abel_quad)
    support = 4 * angular_scale
    nodes, node_weights = np.polynomial.legendre.leggauss(400)
    distance = support * (nodes + 1) / 2
    scalar = _central_fiber_moments(np.cos(distance), angular_scale, series, n_fiber)[0]
    constant = 1 / (2 * np.pi * np.sum(scalar * np.sin(distance) * support / 2 * node_weights))
    cheb = np.cos(np.pi * (np.arange(order + 1) + .5) / (order + 1))[::-1]
    coefficients = np.zeros((n_panels, order + 1, 3))
    for panel in range(n_panels):
        angles = support * (panel + (cheb + 1) / 2) / n_panels
        values = constant * _central_fiber_moments(np.cos(angles), angular_scale, series, n_fiber).T
        coefficients[panel] = np.polynomial.chebyshev.chebfit(cheb, values, order)
    return support, coefficients


def mollifier_fiber_moments_quadrature(mu, angular_scale, *, n_abel=400, n_central=400,
                                       n_abel_quad=160, n_fiber=128, n_panels=64, order=12):
    """Matched group moments of the standard mollifier by physical-space quadrature.

    Same quantities as mollifier_fiber_moments, but the central kernel is built
    from the Abel transform of the mollifier and fiber-integrated directly, so
    there is no degree limit and the cost is nearly independent of the support.
    The moments are tabulated on piecewise Chebyshev panels in the separation
    angle. Requires 4*angular_scale < pi.
    """
    mu = np.asarray(mu, dtype=float)
    if not np.all(np.isfinite(mu)) or np.any((mu < -1) | (mu > 1)):
        raise ValueError("mu must be finite and between -1 and 1")
    if not np.isfinite(angular_scale) or not 0 < angular_scale < np.pi / 4:
        raise ValueError("angular_scale must be between 0 and pi/4")
    support, coefficients = _quadrature_moment_table(
        float(angular_scale), n_abel, n_central, n_abel_quad, n_fiber, n_panels, order)
    flat = mu.ravel()
    distance = np.arccos(flat)
    out = np.zeros((3, flat.size))
    inside = distance <= support
    position = distance[inside] / support * n_panels
    panel = np.minimum(position.astype(int), n_panels - 1)
    local = 2 * (position - panel) - 1
    values = np.zeros((3, position.size))
    for index in np.unique(panel):
        mask = panel == index
        for component in range(3):
            values[component, mask] = np.polynomial.chebyshev.chebval(
                local[mask], coefficients[index][:, component])
    out[:, inside] = values
    return tuple(component.reshape(mu.shape) for component in out)
