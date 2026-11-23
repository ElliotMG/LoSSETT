import numpy as np


def commuting_vector_kernel(mu, scalar_kernel):
    """Construct the parallel-transport kernel paired with a scalar kernel.

    ``mu`` is a strictly increasing one-dimensional grid from -1 to 1,
    where mu = cos(geodesic_distance / sphere_radius). ``scalar_kernel``
    contains the scalar kernel evaluated on this grid. The returned samples
    approximate, by piecewise-linear quadrature,

        h(mu) = g(mu) - integral_{-1}^mu g(t) dt / (1 + mu).

    The value at mu = -1 is the continuous limit, zero. For gradient,
    divergence and curl commutation, integrate h times the transported vector
    using the *scalar* kernel's normalization, not h's normalization. The
    continuous identities are only approximated on a finite sample grid.

    ``filter_vector_field_tangent_normal`` constructs and interpolates these
    samples automatically by default.
    """
    mu = np.asarray(mu, dtype=float)
    scalar_kernel = np.asarray(scalar_kernel, dtype=float)
    if mu.ndim != 1 or scalar_kernel.shape != mu.shape or mu.size < 2:
        raise ValueError("mu and scalar_kernel must be one-dimensional arrays of equal length")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(scalar_kernel)):
        raise ValueError("mu and scalar_kernel must contain only finite values")
    if mu[0] != -1.0 or mu[-1] != 1.0 or np.any(np.diff(mu) <= 0):
        raise ValueError("mu must increase strictly from -1 to 1")

    # This cumulative form avoids subtracting nearly equal kernel integrals.
    increments = np.diff(scalar_kernel) * (1 + (mu[:-1] + mu[1:]) / 2)
    result = np.zeros_like(mu)
    result[1:] = np.cumsum(increments) / (1 + mu[1:])
    if not np.all(np.isfinite(result)):
        raise ValueError("vector kernel calculation produced non-finite values")
    return result


def parallel_transport_vector_components(
    eastward,
    northward,
    sin_initial,
    cos_initial,
    sin_final,
    cos_final,
):
    """Transport endpoint vector components into the origin tangent frame."""
    tangent = eastward * sin_final + northward * cos_final
    normal = eastward * cos_final - northward * sin_final

    return (
        tangent * sin_initial + normal * cos_initial,
        tangent * cos_initial - normal * sin_initial,
    )


def geodesic_vector_increment(
    eastward_origin,
    northward_origin,
    eastward_target,
    northward_target,
    sin_initial,
    cos_initial,
    sin_final,
    cos_final,
):
    """Return the endpoint-minus-origin vector increment in the origin frame."""
    transported_east, transported_north = parallel_transport_vector_components(
        eastward_target,
        northward_target,
        sin_initial,
        cos_initial,
        sin_final,
        cos_final,
    )

    return (
        transported_east - eastward_origin,
        transported_north - northward_origin,
    )


def filter_vector_field_tangent_normal(
    eastward,
    northward,
    sin_initial,
    cos_initial,
    sin_final,
    cos_final,
    kernel,
    area_weights,
    sample_axis=-1,
    *,
    mu=None,
    mu_grid=None,
    mode="commuting",
):
    """Filter vector samples after transporting each into the origin frame.

    By default ``kernel`` is a one-dimensional scalar-kernel profile on
    ``mu_grid``, which must increase from -1 to 1. ``mu`` contains
    cos(distance / sphere_radius) at the vector sample points. The commuting
    vector kernel and scalar normalization are derived from that profile.

    ``mode="parallel_transport"`` retains the former same-kernel filter:
    ``kernel`` must then broadcast with the vector samples. ``area_weights``
    always broadcasts with the vector samples. Missing samples and finite
    quadrature can break the continuous commutation identities.
    """
    if mode == "commuting":
        if mu is None or mu_grid is None:
            raise ValueError("commuting mode requires mu and mu_grid")
        mu_grid = np.asarray(mu_grid, dtype=float)
        kernel = np.asarray(kernel, dtype=float)
        vector_kernel = commuting_vector_kernel(mu_grid, kernel)
        mu = np.asarray(mu, dtype=float)
        if not np.all(np.isfinite(mu)) or np.any((mu < -1) | (mu > 1)):
            raise ValueError("mu must contain finite values in [-1, 1]")
        normalization_kernel = np.interp(mu, mu_grid, kernel)
        kernel = np.interp(mu, mu_grid, vector_kernel)
    elif mode == "parallel_transport":
        if mu is not None or mu_grid is not None:
            raise ValueError("mu and mu_grid are not used in parallel_transport mode")
        normalization_kernel = kernel
    else:
        raise ValueError(f"Unknown vector filter mode: {mode}")

    arrays = np.broadcast_arrays(
        np.asarray(eastward),
        np.asarray(northward),
        np.asarray(sin_initial),
        np.asarray(cos_initial),
        np.asarray(sin_final),
        np.asarray(cos_final),
        np.asarray(kernel),
        np.asarray(area_weights),
        np.asarray(normalization_kernel),
    )
    (
        eastward,
        northward,
        sin_initial,
        cos_initial,
        sin_final,
        cos_final,
        kernel,
        area_weights,
        normalization_kernel,
    ) = arrays
    weights = kernel * area_weights
    normalization_weights = normalization_kernel * area_weights
    valid = (
        np.isfinite(eastward)
        & np.isfinite(northward)
        & np.isfinite(sin_initial)
        & np.isfinite(cos_initial)
        & np.isfinite(sin_final)
        & np.isfinite(cos_final)
        & np.isfinite(weights)
        & np.isfinite(normalization_weights)
    )
    weights = np.where(valid, weights, 0.0)
    weight_sum = np.sum(
        np.where(valid, normalization_weights, 0.0),
        axis=sample_axis,
        keepdims=True,
    )
    if np.any(~np.isfinite(weight_sum)) or np.any(weight_sum <= 0):
        raise ValueError("Each vector filter must have positive finite total weight")

    transported_east, transported_north = parallel_transport_vector_components(
        eastward,
        northward,
        sin_initial,
        cos_initial,
        sin_final,
        cos_final,
    )
    filtered_east = np.sum(
        np.where(valid, transported_east, 0.0) * weights,
        axis=sample_axis,
    ) / np.squeeze(weight_sum, axis=sample_axis)
    filtered_north = np.sum(
        np.where(valid, transported_north, 0.0) * weights,
        axis=sample_axis,
    ) / np.squeeze(weight_sum, axis=sample_axis)
    return filtered_east, filtered_north
