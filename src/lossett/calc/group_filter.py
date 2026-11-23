"""Coherent vector and rank-two filtering with a central SO(3) weight."""

import numpy as np

from lossett.calc.group_increments import _transport
from lossett.filtering.group_kernels import heat_fiber_moments, mollifier_les_fiber_moments


def group_filter_shell(
    origin_direction, target_directions, vectors, tensors, solid_angle_weights,
    *, heat_time=None, mollifier_scale=None, fiber_tolerance=1e-9,
    quadrature_tolerance=1e-2,
):
    """Filter Cartesian vectors and rank-two tensors at one shell origin.

    All target tensor indices are pulled back by the same rotation. The
    result uses the LES filter H: choose its heat time or scalar mollifier
    angular scale, matching the corresponding group_increment_shell option.
    Weights integrate solid angle over the full sphere, without renormalizing
    individual samples or the paired kernels.
    """
    x = np.asarray(origin_direction, dtype=float)
    y = np.asarray(target_directions, dtype=float)
    v = np.asarray(vectors, dtype=float)
    tensor = np.asarray(tensors, dtype=float)
    weights = np.asarray(solid_angle_weights, dtype=float)
    if x.shape != (3,) or y.ndim != 2 or y.shape[1] != 3:
        raise ValueError("origin direction must have shape (3,) and targets (N, 3)")
    n = len(y)
    if v.shape != (n, 3) or tensor.shape != (n, 3, 3) or weights.shape != (n,):
        raise ValueError("vector, tensor and weight shapes must match target directions")
    if not all(np.all(np.isfinite(a)) for a in (x, y, v, tensor, weights)):
        raise ValueError("inputs must be finite")
    if not np.isclose(np.linalg.norm(x), 1, rtol=0, atol=1e-10) or not np.allclose(
        np.linalg.norm(y, axis=1), 1, rtol=0, atol=1e-10
    ):
        raise ValueError("directions must be unit vectors")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 4 * np.pi, rtol=1e-6):
        raise ValueError("solid angle weights must be nonnegative and sum to 4*pi")
    if not np.isfinite(quadrature_tolerance) or quadrature_tolerance <= 0:
        raise ValueError("quadrature_tolerance must be positive and finite")
    if (heat_time is None) == (mollifier_scale is None):
        raise ValueError("specify exactly one of heat_time and mollifier_scale")
    mu = np.clip(y @ x, -1, 1)
    if heat_time is not None:
        g, h, k = heat_fiber_moments(mu, heat_time, tolerance=fiber_tolerance)
    else:
        g, h, k = mollifier_les_fiber_moments(mu, mollifier_scale, tolerance=fiber_tolerance)
    mass = weights @ g
    if abs(mass - 1) > quadrature_tolerance:
        raise ValueError(f"endpoint quadrature under-resolves the kernel: kernel mass is {mass:.8g}")

    target_x = np.broadcast_to(x, y.shape)
    rotation = np.stack([
        _transport(np.broadcast_to(axis, y.shape), y, target_x) for axis in np.eye(3)
    ], axis=-1)
    transported_v = np.einsum("nij,nj->ni", rotation, v)
    radial_v = transported_v @ x
    horizontal_v = transported_v - radial_v[:, None] * x
    vector_samples = h[:, None] * horizontal_v + (g * radial_v)[:, None] * x

    transported_t = np.einsum("nik,nkl,njl->nij", rotation, tensor, rotation)
    projection = np.eye(3) - np.outer(x, x)
    horizontal = np.einsum("ij,njk,kl->nil", projection, transported_t, projection)
    radial = np.einsum("i,nij,j->n", x, transported_t, x)[:, None, None] * np.outer(x, x)
    mixed = transported_t - horizontal - radial
    trace = np.trace(horizontal, axis1=1, axis2=2)
    isotropic = trace[:, None, None] * projection / 2
    antisymmetric = (horizontal - np.swapaxes(horizontal, 1, 2)) / 2
    traceless = (horizontal + np.swapaxes(horizontal, 1, 2)) / 2 - isotropic
    tensor_samples = (
        g[:, None, None] * (isotropic + antisymmetric + radial)
        + h[:, None, None] * mixed + k[:, None, None] * traceless
    )
    return {
        "vector": np.einsum("n,ni->i", weights, vector_samples),
        "tensor": np.einsum("n,nij->ij", weights, tensor_samples),
        "kernel_mass": mass,
    }
