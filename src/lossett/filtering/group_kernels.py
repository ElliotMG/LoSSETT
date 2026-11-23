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


@lru_cache(maxsize=32)
def _mollifier_spectral_coefficients(angular_scale, degree):
    nodes, weights = np.polynomial.legendre.leggauss(2 * degree + 2)
    lower = np.cos(2 * angular_scale)
    x = lower + (1 - lower) * (nodes + 1) / 2
    theta = np.arccos(np.clip(x, -1, 1))
    profile = np.exp(-1 / (1 - (theta / (2 * angular_scale))**2))
    weighted = weights * profile
    basis = np.polynomial.legendre.legvander(x, degree)
    multiplier = (weighted @ basis) / np.sum(weighted)
    return (2 * np.arange(degree + 1) + 1) * multiplier**2


def mollifier_fiber_moments(mu, angular_scale, *, tolerance=1e-9, max_degree=2048):
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
        summed = np.cumsum(coeff[::-1])[::-1]
        cheb_coeff = np.r_[summed[0], 2 * summed[1:]]
        psi = 2 * np.pi * np.arange(degree + 3) / (degree + 3)
        cosine = np.cos(psi)
        current = np.empty((3, flat.size))
        for start in range(0, flat.size, 128):
            x = flat[start:start + 128]
            angle_cosine = (1 + x[:, None]) * (1 + cosine[None, :]) / 2 - 1
            weight = np.polynomial.chebyshev.chebval(angle_cosine, cheb_coeff)
            current[:, start:start + len(x)] = np.stack([
                np.mean(weight, axis=1),
                np.mean(weight * cosine, axis=1),
                np.mean(weight * (2 * cosine**2 - 1), axis=1),
            ]) / (4 * np.pi)
        if previous is not None:
            scale = max(float(np.max(np.abs(current))), 1.0)
            tail_indicator = np.sum(coeff[-degree // 4:] * (2 * np.arange(degree + 1)[-degree // 4:] + 1)) / (4 * np.pi)
            if np.max(np.abs(current - previous)) <= tolerance * scale and tail_indicator <= tolerance * scale:
                if 4 * angular_scale < np.pi:
                    current[:, flat < np.cos(4 * angular_scale)] = 0
                return tuple(component.reshape(original_shape) for component in current)
        previous = current
        degree *= 2
    raise ValueError("mollifier group moments did not converge; increase max_degree")
