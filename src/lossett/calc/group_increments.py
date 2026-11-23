"""Coherent SO(3) increment diagnostic on one shell and radial reconstruction."""

import numpy as np

from lossett.filtering.group_kernels import heat_fiber_moments, mollifier_fiber_moments


def latlon_shell_fields(latitude, longitude, eastward, northward, radial, *, radius):
    """Construct Cartesian fields and intrinsic gradients on a global lat-lon grid.

    Latitude and longitude are in radians. Longitude must be uniform and
    periodic without a duplicate endpoint. At a pole, Cartesian velocities
    must represent the same physical vector at every longitude. Polar
    gradients use the first Fourier mode on the nearest latitude ring.
    """
    lat = np.asarray(latitude, dtype=float)
    lon = np.asarray(longitude, dtype=float)
    u = np.asarray(eastward, dtype=float)
    v = np.asarray(northward, dtype=float)
    w = np.asarray(radial, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1 or lat.size < 3 or lon.size < 3:
        raise ValueError("latitude and longitude must be one-dimensional with at least three points")
    if u.shape != (lat.size, lon.size) or v.shape != u.shape or w.shape != u.shape:
        raise ValueError("velocity fields must have shape (latitude, longitude)")
    if not all(np.all(np.isfinite(a)) for a in (lat, lon, u, v, w)):
        raise ValueError("coordinates and velocities must be finite")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    if np.any(np.abs(lat) > np.pi / 2) or not (np.all(np.diff(lat) > 0) or np.all(np.diff(lat) < 0)):
        raise ValueError("latitude must be monotonic and within the poles")
    step = 2 * np.pi / lon.size
    if not np.allclose(np.diff(lon), step, rtol=1e-8, atol=1e-12):
        raise ValueError("longitude must be uniform, increasing and global without a duplicate endpoint")
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    clat, slat = np.cos(lat2), np.sin(lat2)
    clon, slon = np.cos(lon2), np.sin(lon2)
    normal = np.stack((clat * clon, clat * slon, slat), axis=-1)
    east = np.stack((-slon, clon, np.zeros_like(slon)), axis=-1)
    north = np.stack((-slat * clon, -slat * slon, clat), axis=-1)
    velocity = u[..., None] * east + v[..., None] * north + w[..., None] * normal

    def longitude_derivative(a):
        return (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1)) / (2 * step)

    poles = np.abs(lat) >= np.pi / 2 - 1e-14
    if np.any(poles) and lon.size < 4:
        raise ValueError("polar differentiation requires at least four longitudes")
    denominator = radius * np.where(poles[:, None], 1, clat)
    ue = (longitude_derivative(u) - v * slat) / denominator
    ve = (longitude_derivative(v) + u * slat) / denominator
    un = np.gradient(u, lat, axis=0, edge_order=2) / radius
    vn = np.gradient(v, lat, axis=0, edge_order=2) / radius
    we = longitude_derivative(w) / denominator
    wn = np.gradient(w, lat, axis=0, edge_order=2) / radius
    grad = (
        (ue[..., None] * east + ve[..., None] * north)[..., :, None] * east[..., None, :]
        + (un[..., None] * east + vn[..., None] * north)[..., :, None] * north[..., None, :]
    )
    gradw = we[..., None] * east + wn[..., None] * north
    for i in np.flatnonzero(poles):
        if i not in (0, len(lat) - 1):
            raise ValueError("poles must be latitude endpoints")
        pole_velocity = velocity[i].mean(axis=0)
        if np.max(np.abs(velocity[i] - pole_velocity)) > 1e-10 + 1e-7 * np.max(np.abs(velocity[i])):
            raise ValueError("velocity at each pole must be the same Cartesian vector at all longitudes")
        near = 1 if i == 0 else i - 1
        distance = radius * np.sin(abs(lat[near] - lat[i]))
        harmonics = np.stack((np.cos(lon), np.sin(lon)), axis=0)
        d_velocity = 2 * harmonics @ velocity[near] / (lon.size * distance)
        d_w = 2 * harmonics @ w[near] / (lon.size * distance)
        projector = np.diag([1., 1., 0.])
        derivative = np.zeros((3, 3))
        derivative[:, :2] = d_velocity.T
        polar_w = np.dot(pole_velocity, normal[i, 0])
        grad[i] = projector @ derivative @ projector - polar_w * projector / radius
        gradw[i] = np.array([d_w[0], d_w[1], 0.])
    return normal, velocity, grad, gradw


def hydrostatic_radius(pressure, reference_density, gravity, initial_radius):
    """Integrate dp/da=-rho*g along an ordered reference pressure profile."""
    pressure = np.asarray(pressure, dtype=float)
    rho = np.asarray(reference_density, dtype=float)
    if pressure.ndim != 1 or pressure.size < 2 or rho.shape != pressure.shape:
        raise ValueError("pressure and density must be matching one-dimensional profiles")
    if not np.all(np.isfinite(pressure)) or np.any(pressure <= 0) or not np.all(np.isfinite(rho)) or np.any(rho <= 0):
        raise ValueError("pressure and density must be positive and finite")
    if not (np.all(np.diff(pressure) > 0) or np.all(np.diff(pressure) < 0)):
        raise ValueError("pressure must be strictly monotonic")
    if not np.isfinite(gravity) or gravity <= 0 or not np.isfinite(initial_radius) or initial_radius <= 0:
        raise ValueError("gravity and initial_radius must be positive and finite")
    inv = 1 / (rho * gravity)
    result = initial_radius + np.r_[0., np.cumsum(-np.diff(pressure) * (inv[:-1] + inv[1:]) / 2)]
    if np.any(result <= 0):
        raise ValueError("hydrostatic reconstruction produced a nonpositive radius")
    return result


def _transport(vector, source, target):
    """Apply the minimum rotation from source to target, including antipodes."""
    cosine = np.clip(np.sum(source * target, axis=-1), -1, 1)
    cross = np.cross(source, target)
    sine = np.linalg.norm(cross, axis=-1)
    axis = cross / np.where(sine > 1e-14, sine, 1)[..., None]
    result = (
        cosine[..., None] * vector
        + sine[..., None] * np.cross(axis, vector)
        + (1 - cosine)[..., None] * np.sum(axis * vector, axis=-1, keepdims=True) * axis
    )
    opposite = (sine <= 1e-14) & (cosine < 0)
    if np.any(opposite):
        axes = np.eye(3)[np.argmin(np.abs(source), axis=-1)]
        axis = np.cross(source, axes)
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
        rotated = 2 * np.sum(axis * vector, axis=-1, keepdims=True) * axis - vector
        result = np.where(opposite[..., None], rotated, result)
    return result


def group_increment_shell(
    origin_direction, origin_velocity, target_directions, target_velocity,
    horizontal_velocity_gradient, horizontal_w_gradient, solid_angle_weights,
    *, radius, heat_time=None, mollifier_scale=None, atmosphere="deep", fiber_tolerance=1e-9,
):
    """Compute one origin's matched-group moments and horizontal transfer.

    Cartesian vectors are expressed in a common Earth-centred frame.
    ``horizontal_velocity_gradient`` is the intrinsic covariant derivative
    of horizontal velocity, embedded as a 3x3 tensor at each target;
    ``horizontal_w_gradient`` is the intrinsic horizontal gradient of w.
    Targets and their weights must cover the whole sphere; weights integrate
    solid angle, not physical surface area. The increment kernel has heat
    time 2*heat_time; the matched LES filter has heat_time. Alternatively,
    mollifier_scale selects a matched group filter whose LES scalar endpoint
    kernel is the standard mollifier with that angular scale in radians.
    """
    return _group_increment_shell(
        origin_direction, origin_velocity, target_directions, target_velocity,
        horizontal_velocity_gradient, horizontal_w_gradient, solid_angle_weights,
        radius=radius, heat_time=heat_time, mollifier_scale=mollifier_scale,
        atmosphere=atmosphere, fiber_tolerance=fiber_tolerance,
    )


def _validate_group_targets(y, uy, grad, gradw, weights):
    n = y.shape[0]
    if uy.shape != (n, 3) or grad.shape != (n, 3, 3) or gradw.shape != (n, 3) or weights.shape != (n,):
        raise ValueError("target values, gradients and weights must match target directions")
    if not all(np.all(np.isfinite(a)) for a in (y, uy, grad, gradw, weights)):
        raise ValueError("inputs must be finite")
    if not np.allclose(np.linalg.norm(y, axis=1), 1, rtol=0, atol=1e-10):
        raise ValueError("directions must be unit vectors")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 4 * np.pi, rtol=1e-6):
        raise ValueError("solid angle weights must be nonnegative and sum to 4*pi")
    tensor_normal = np.einsum("nij,nj->ni", grad, y)
    normal_tensor = np.einsum("ni,nij->nj", y, grad)
    gradient_normal = np.sum(gradw * y, axis=1)
    if (np.max(np.abs(tensor_normal)) > 1e-12 + 1e-9 * np.max(np.abs(grad))
            or np.max(np.abs(normal_tensor)) > 1e-12 + 1e-9 * np.max(np.abs(grad))
            or np.max(np.abs(gradient_normal)) > 1e-12 + 1e-9 * np.max(np.abs(gradw))):
        raise ValueError("horizontal gradients must be tangential at each target")


def _group_increment_shell(
    origin_direction, origin_velocity, target_directions, target_velocity,
    horizontal_velocity_gradient, horizontal_w_gradient, solid_angle_weights,
    *, radius, heat_time, mollifier_scale, atmosphere, fiber_tolerance, fiber_moments=None,
    targets_validated=False,
):
    x = np.asarray(origin_direction, dtype=float)
    u = np.asarray(origin_velocity, dtype=float)
    y = np.asarray(target_directions, dtype=float)
    uy = np.asarray(target_velocity, dtype=float)
    grad = np.asarray(horizontal_velocity_gradient, dtype=float)
    gradw = np.asarray(horizontal_w_gradient, dtype=float)
    weights = np.asarray(solid_angle_weights, dtype=float)
    if x.shape != (3,) or u.shape != (3,) or y.ndim != 2 or y.shape[1] != 3:
        raise ValueError("origin vectors must have shape (3,) and target directions (N, 3)")
    if not all(np.all(np.isfinite(a)) for a in (x, u)):
        raise ValueError("inputs must be finite")
    if not np.isclose(np.linalg.norm(x), 1, rtol=0, atol=1e-10):
        raise ValueError("directions must be unit vectors")
    if not targets_validated:
        _validate_group_targets(y, uy, grad, gradw, weights)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    if atmosphere not in ("deep", "shallow"):
        raise ValueError("atmosphere must be 'deep' or 'shallow'")
    wx = np.dot(x, u)
    wy = np.sum(y * uy, axis=1)
    vx = u - wx * x
    vy = uy - wy[:, None] * y
    p = _transport(vx, x, y)
    transported_vy = _transport(vy, y, np.broadcast_to(x, y.shape))
    dw = wy - wx
    if (heat_time is None) == (mollifier_scale is None):
        raise ValueError("specify exactly one of heat_time and mollifier_scale")
    if fiber_moments is not None:
        g, h, k = fiber_moments
    else:
        mu = np.clip(y @ x, -1, 1)
        if heat_time is not None:
            g, h, k = heat_fiber_moments(mu, 2 * heat_time, tolerance=fiber_tolerance)
        else:
            g, h, k = mollifier_fiber_moments(mu, mollifier_scale, tolerance=fiber_tolerance)

    vv = np.sum(vy * vy, axis=1)
    pp = np.dot(vx, vx)
    vp = np.sum(vy * p, axis=1)
    q = g * (vv + pp + dw**2) - 2 * h * vp
    identity_h = np.eye(3) - y[:, :, None] * y[:, None, :]
    curvature = 1 / radius if atmosphere == "deep" else 0.0
    strain = (grad + np.swapaxes(grad, -1, -2)) / 2 + curvature * wy[:, None, None] * identity_h
    b = gradw - curvature * vy
    A = (
        g[:, None, None] * vy[:, :, None] * vy[:, None, :]
        - h[:, None, None] * (vy[:, :, None] * p[:, None, :] + p[:, :, None] * vy[:, None, :])
        + k[:, None, None] * p[:, :, None] * p[:, None, :]
        + ((g - k) * pp / 2)[:, None, None] * identity_h
    )
    B = dw[:, None] * (g[:, None] * vy - h[:, None] * p)
    divergence = np.trace(grad, axis1=-2, axis2=-1)
    integrand = (divergence + 2 * curvature * wx) * q
    integrand += 2 * np.einsum("nij,nij->n", A, strain) + 2 * np.sum(B * b, axis=1)
    m = dw[:, None] * (h[:, None] * transported_vy - g[:, None] * vx + (g * dw)[:, None] * x)
    return {
        "horizontal": -0.25 * np.dot(weights, integrand),
        "structure": np.dot(weights, q),
        "vertical_flux": np.dot(weights, dw * q),
        "mixed_moment": np.einsum("n,ni->i", weights, m),
        "kernel_mass": np.dot(weights, g),
    }


def group_increment_grid(
    directions, velocity, horizontal_velocity_gradient, horizontal_w_gradient,
    solid_angle_weights, *, radius, heat_time=None, mollifier_scale=None,
    atmosphere="deep", fiber_tolerance=1e-9, quadrature_tolerance=1e-2,
    cache_longitude_moments=False, compact_support=False, pair_backend="numpy",
):
    """Apply the reference single-shell implementation at all grid points.

    All grid dimensions precede the vector or tensor component dimensions.
    No missing-data masking or local renormalization is performed. The optional cache
    requires a latitude-by-longitude grid invariant under uniform rotations.
    ``compact_support`` restricts matched-mollifier targets to their exact
    support and requires the same regular-longitude grid.
    ``pair_backend="numba"`` uses compiled pairwise accumulation.
    """
    directions = np.asarray(directions, dtype=float)
    if directions.ndim < 2 or directions.shape[-1] != 3:
        raise ValueError("directions must have shape (*grid, 3)")
    grid_shape = directions.shape[:-1]
    n = int(np.prod(grid_shape))
    velocity = np.asarray(velocity, dtype=float)
    grad = np.asarray(horizontal_velocity_gradient, dtype=float)
    gradw = np.asarray(horizontal_w_gradient, dtype=float)
    weights = np.asarray(solid_angle_weights, dtype=float)
    if not np.isfinite(quadrature_tolerance) or quadrature_tolerance <= 0:
        raise ValueError("quadrature_tolerance must be positive and finite")
    if velocity.shape != grid_shape + (3,) or grad.shape != grid_shape + (3, 3) or gradw.shape != grid_shape + (3,) or weights.shape != grid_shape:
        raise ValueError("grid input shapes do not match")
    if pair_backend not in ("numpy", "numba"):
        raise ValueError("pair_backend must be 'numpy' or 'numba'")
    accumulate_group_pairs = None
    if pair_backend == "numba":
        try:
            from lossett.calc._group_numba import accumulate_group_pairs
        except ModuleNotFoundError as exc:
            if exc.name != "numba":
                raise
            raise ModuleNotFoundError("the numba pair backend requires the acceleration extra") from exc
    if compact_support and (mollifier_scale is None or heat_time is not None):
        raise ValueError("compact support requires the matched mollifier kernel")
    nlon = grid_shape[1] if len(grid_shape) == 2 else 0
    if cache_longitude_moments or compact_support:
        if len(grid_shape) != 2 or grid_shape[1] < 3 or not np.all(np.isfinite(directions)):
            raise ValueError("moment caching requires a finite two-dimensional regular-longitude grid")
        angle = 2 * np.pi * np.arange(nlon) / nlon
        x0, y0 = directions[:, 0, 0, None], directions[:, 0, 1, None]
        expected = np.stack(
            (x0 * np.cos(angle) - y0 * np.sin(angle),
             x0 * np.sin(angle) + y0 * np.cos(angle),
             np.broadcast_to(directions[:, 0, 2, None], grid_shape)), axis=-1,
        )
        if not np.allclose(directions, expected, rtol=0, atol=16 * np.finfo(float).eps):
            raise ValueError("moment caching requires a uniform global longitude rotation grid")
        if (heat_time is None) == (mollifier_scale is None):
            raise ValueError("specify exactly one of heat_time and mollifier_scale")
    directions = directions.reshape(n, 3)
    velocity = velocity.reshape(n, 3)
    grad = grad.reshape(n, 3, 3)
    gradw = gradw.reshape(n, 3)
    weights = weights.reshape(n)
    support_cosine = -1.
    if compact_support or pair_backend == "numba":
        _validate_group_targets(directions, velocity, grad, gradw, weights)
    if pair_backend == "numba":
        if (heat_time is None) == (mollifier_scale is None):
            raise ValueError("specify exactly one of heat_time and mollifier_scale")
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be positive and finite")
        if atmosphere not in ("deep", "shallow"):
            raise ValueError("atmosphere must be 'deep' or 'shallow'")
    if compact_support:
        if mollifier_scale is None:
            raise ValueError("compact support requires the matched mollifier kernel")
        if not np.isfinite(mollifier_scale) or not 0 < mollifier_scale < np.pi / 2:
            raise ValueError("angular_scale must be between 0 and pi/2")
        support_cosine = np.cos(min(4 * mollifier_scale, np.pi))
    output = {
        "horizontal": np.empty(n),
        "structure": np.empty(n),
        "vertical_flux": np.empty(n),
        "mixed_moment": np.empty((n, 3)),
        "kernel_mass": np.empty(n),
    }
    active = np.empty(0, dtype=np.intp)
    cached = None
    for i in range(n):
        moments = None
        indices = None
        if compact_support:
            origin_lon = i % nlon
            if origin_lon == 0:
                mu = np.clip(directions @ directions[i], -1, 1)
                active = np.flatnonzero(mu >= support_cosine - 64 * np.finfo(float).eps)
                if active.size == 0:
                    raise ValueError(f"endpoint quadrature under-resolves the kernel at origin {i}: no targets in support")
                if cache_longitude_moments:
                    cached = mollifier_fiber_moments(mu[active], mollifier_scale, tolerance=fiber_tolerance)
            indices = (active // nlon) * nlon + (active % nlon + origin_lon) % nlon
            if cache_longitude_moments:
                if cached is None:
                    raise RuntimeError("compact fiber moments were not initialized")
                moments = cached
        elif cache_longitude_moments:
            origin_lon = i % nlon
            if origin_lon == 0:
                mu = np.clip(directions @ directions[i], -1, 1)
                if heat_time is not None:
                    cached = heat_fiber_moments(mu, 2 * heat_time, tolerance=fiber_tolerance)
                else:
                    cached = mollifier_fiber_moments(mu, mollifier_scale, tolerance=fiber_tolerance)
                cached = tuple(value.reshape(grid_shape) for value in cached)
            if cached is None:
                raise RuntimeError("fiber moments were not initialized")
            moments = tuple(np.roll(value, origin_lon, axis=1).ravel() for value in cached)
        if pair_backend == "numba":
            if accumulate_group_pairs is None:
                raise RuntimeError("the numba pair backend was not initialized")
            if moments is None:
                targets = directions if indices is None else directions[indices]
                mu = np.clip(targets @ directions[i], -1, 1)
                if heat_time is not None:
                    moments = heat_fiber_moments(mu, 2 * heat_time, tolerance=fiber_tolerance)
                else:
                    moments = mollifier_fiber_moments(mu, mollifier_scale, tolerance=fiber_tolerance)
            raw = accumulate_group_pairs(
                directions[i], velocity[i], directions, velocity, grad, gradw, weights,
                *moments, 1 / radius if atmosphere == "deep" else 0., indices,
            )
            result = {
                "horizontal": raw[0], "structure": raw[1], "vertical_flux": raw[2],
                "mixed_moment": np.array(raw[3:6]), "kernel_mass": raw[6],
            }
        else:
            y = directions if indices is None else directions[indices]
            uy = velocity if indices is None else velocity[indices]
            target_grad = grad if indices is None else grad[indices]
            target_gradw = gradw if indices is None else gradw[indices]
            target_weights = weights if indices is None else weights[indices]
            result = _group_increment_shell(
                directions[i], velocity[i], y, uy, target_grad, target_gradw, target_weights,
                radius=radius, heat_time=heat_time, mollifier_scale=mollifier_scale, atmosphere=atmosphere,
                fiber_tolerance=fiber_tolerance, fiber_moments=moments, targets_validated=compact_support,
            )
        if abs(result["kernel_mass"] - 1) > quadrature_tolerance:
            raise ValueError(
                f"endpoint quadrature under-resolves the kernel at origin {i}: "
                f"kernel mass is {result['kernel_mass']:.8g}"
            )
        for key in output:
            output[key][i] = result[key]
    return {
        key: value.reshape(grid_shape + ((3,) if key == "mixed_moment" else ()))
        for key, value in output.items()
    }


def reconstruct_group_vertical(
    coordinate, radius, radial_direction, velocity, structure, vertical_flux, mixed_moment,
    *, coordinate_type="height", fluid="boussinesq", atmosphere="deep",
    reference_density=None, gravity=None,
):
    """Reconstruct vertical transfer along axis 0; pressure is in pascals.

    Velocity and mixed moment use Cartesian vector components on a fixed
    radial ray. For pressure, hydrostatic conversion uses the supplied
    reference density and positive gravitational acceleration.
    """
    coord = np.asarray(coordinate, dtype=float)
    radius = np.asarray(radius, dtype=float)
    direction = np.asarray(radial_direction, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    structure = np.asarray(structure, dtype=float)
    flux = np.asarray(vertical_flux, dtype=float)
    moment = np.asarray(mixed_moment, dtype=float)
    if coord.ndim != 1 or coord.size < 3 or radius.shape != coord.shape:
        raise ValueError("coordinate and radius must be one-dimensional with at least three levels")
    if not (np.all(np.diff(coord) > 0) or np.all(np.diff(coord) < 0)):
        raise ValueError("coordinate must be strictly monotonic")
    if direction.shape != (3,) or not np.all(np.isfinite(direction)) or not np.isclose(np.linalg.norm(direction), 1, rtol=0, atol=1e-10):
        raise ValueError("radial_direction must be a finite unit vector")
    if velocity.shape != (coord.size, 3) or moment.shape != velocity.shape or structure.shape != coord.shape or flux.shape != coord.shape:
        raise ValueError("level arrays have inconsistent shapes")
    if not all(np.all(np.isfinite(a)) for a in (coord, radius, velocity, structure, flux, moment)) or np.any(radius <= 0):
        raise ValueError("all inputs must be finite and radii positive")
    if coordinate_type not in ("height", "pressure") or fluid not in ("boussinesq", "anelastic") or atmosphere not in ("deep", "shallow"):
        raise ValueError("invalid coordinate, fluid or atmosphere option")
    rho = None if reference_density is None else np.asarray(reference_density, dtype=float)
    if rho is not None and (rho.shape != coord.shape or not np.all(np.isfinite(rho)) or np.any(rho <= 0)):
        raise ValueError("reference_density must be a positive finite profile")
    if (fluid == "anelastic" or coordinate_type == "pressure") and rho is None:
        raise ValueError("reference_density is required for anelastic or pressure calculations")
    if coordinate_type == "height" and not np.allclose(np.diff(radius), np.diff(coord), rtol=1e-9, atol=1e-8):
        raise ValueError("height coordinate must increase with radius by the same amount")
    if coordinate_type == "pressure":
        if np.any(coord <= 0):
            raise ValueError("pressure must be positive")
        if rho is None:
            raise ValueError("reference_density is required for pressure calculations")
        if gravity is None or not np.isfinite(gravity) or gravity <= 0:
            raise ValueError("pressure coordinates require positive finite gravity")
        if np.any(np.diff(radius) * np.diff(coord) >= 0):
            raise ValueError("pressure and radius must vary in opposite directions")
        factor = -rho * gravity
    else:
        factor = np.ones_like(coord)
    def derivative(a):
        shape = (-1,) + (1,) * (a.ndim - 1)
        return np.gradient(a, coord, axis=0, edge_order=2) * factor.reshape(shape)

    w = velocity @ direction
    curvature = 2 / radius if atmosphere == "deep" else np.zeros_like(radius)
    if fluid == "anelastic":
        if rho is None:
            raise ValueError("reference_density is required for anelastic calculations")
        density_gradient = derivative(np.log(rho))
    else:
        density_gradient = np.zeros_like(coord)
    return -0.25 * (
        derivative(flux) + derivative(w) * structure
        + 2 * np.sum(moment * derivative(velocity), axis=1)
        + curvature * flux + density_gradient * (flux + w * structure)
    )
