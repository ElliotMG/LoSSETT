"""Experimental exact discrete longitude correlations for two group moments."""
import time

import numpy as np

from lossett.calc.group_increments import _transport, group_increment_grid
from lossett.filtering.group_kernels import heat_fiber_moments, mollifier_fiber_moments


def correlate(kernel, field):
    return np.fft.irfft(
        np.sum(np.conj(np.fft.rfft(kernel, axis=1)) * np.fft.rfft(field, axis=1), axis=0),
        n=field.shape[1],
    )


def fft_group(latitude, longitude, velocity, grad, gradw, weights, *, radius,
              heat_time=None, mollifier_scale=None, atmosphere="deep"):
    nlat, nlon = len(latitude), len(longitude)
    coslon, sinlon = np.cos(longitude), np.sin(longitude)
    local = velocity.copy()
    local[..., 0] = coslon * velocity[..., 0] + sinlon * velocity[..., 1]
    local[..., 1] = -sinlon * velocity[..., 0] + coslon * velocity[..., 1]
    rotation = np.zeros((nlon, 3, 3))
    rotation[:, 0, 0] = rotation[:, 1, 1] = coslon
    rotation[:, 1, 0] = sinlon
    rotation[:, 0, 1] = -sinlon
    rotation[:, 2, 2] = 1
    grad = np.einsum("lpi,ylpq,lqj->ylij", rotation, grad, rotation)
    gradw = np.einsum("lpi,ylp->yli", rotation, gradw)
    radial = np.stack((np.cos(latitude), np.zeros(nlat), np.sin(latitude)), axis=1)
    w = np.sum(local * radial[:, None, :], axis=-1)
    v = local - w[..., None] * radial[:, None, :]
    speed2 = np.sum(local**2, axis=-1)
    curvature = 1 / radius if atmosphere == "deep" else 0.
    projector = np.eye(3) - radial[:, :, None] * radial[:, None, :]
    strain = (grad + np.swapaxes(grad, -1, -2)) / 2 + curvature * w[..., None, None] * projector[:, None]
    bfield = gradw - curvature * v
    ev = np.einsum("...ij,...j->...i", strain, v)
    vev = np.sum(v * ev, axis=-1)
    vb = np.sum(v * bfield, axis=-1)
    div = np.trace(grad, axis1=-2, axis2=-1)
    projected_trace = np.einsum("yij,ylij->yl", projector, strain)
    delta = 2 * np.pi * np.arange(nlon) / nlon
    c, s = np.cos(delta), np.sin(delta)
    y = np.stack((np.cos(latitude)[:, None] * c,
                  np.cos(latitude)[:, None] * s,
                  np.broadcast_to(np.sin(latitude)[:, None], (nlat, nlon))), axis=-1)
    structure = np.empty((nlat, nlon))
    vertical = np.empty_like(structure)
    horizontal = np.empty_like(structure)
    mixed = np.empty((nlat, nlon, 3))
    mass = np.empty_like(structure)
    for i, x in enumerate(radial):
        mu = np.clip(np.sum(y * x, axis=-1), -1, 1)
        if heat_time is not None:
            g, h, k = heat_fiber_moments(mu, 2 * heat_time)
        else:
            g, h, k = mollifier_fiber_moments(mu, mollifier_scale)
        matrix = np.empty((nlat, nlon, 3, 3))
        for a in range(3):
            transported = _transport(np.eye(3)[a], x, y)
            matrix[..., 0, a] = c * transported[..., 0] + s * transported[..., 1]
            matrix[..., 1, a] = -s * transported[..., 0] + c * transported[..., 1]
            matrix[..., 2, a] = transported[..., 2]
        wx, vx, ux2 = w[i], v[i], speed2[i]
        c0 = correlate(g, weights)
        c2 = correlate(g, weights * speed2)
        cw = correlate(g, weights * w)
        cw2 = correlate(g, weights * w**2)
        cw_speed2 = correlate(g, weights * w * speed2)
        d = np.zeros((nlon, 3))
        dw = np.zeros((nlon, 3))
        div_d = np.zeros((nlon, 3))
        strain_d = np.zeros((nlon, 3))
        b_d = np.zeros((nlon, 3))
        b_w_d = np.zeros((nlon, 3))
        quadratic = np.zeros((nlon, 3, 3))
        for a in range(3):
            for b in range(3):
                hk = h * matrix[..., b, a]
                d[:, a] += correlate(hk, weights * v[..., b])
                dw[:, a] += correlate(hk, weights * w * v[..., b])
                div_d[:, a] += correlate(hk, weights * div * v[..., b])
                strain_d[:, a] += correlate(hk, weights * ev[..., b])
                b_d[:, a] += correlate(hk, weights * bfield[..., b])
                b_w_d[:, a] += correlate(hk, weights * w * bfield[..., b])
                for aa in range(3):
                    for bb in range(3):
                        quadratic[:, a, aa] += correlate(
                            k * matrix[..., b, a] * matrix[..., bb, aa],
                            weights * strain[..., b, bb],
                        )
        dot = np.sum(vx * d, axis=1)
        dotw = np.sum(vx * dw, axis=1)
        structure[i] = c2 + ux2 * c0 - 2 * wx * cw - 2 * dot
        vertical[i] = (cw_speed2 - wx * c2 + (ux2 + 2 * wx**2) * cw
                       - wx * ux2 * c0 - 2 * wx * cw2 - 2 * dotw + 2 * wx * dot)
        pp = np.sum(vx**2, axis=1)
        div_q = (correlate(g, weights * div * speed2)
                 + ux2 * correlate(g, weights * div)
                 - 2 * wx * correlate(g, weights * div * w)
                 - 2 * np.sum(vx * div_d, axis=1))
        ae = (correlate(g, weights * vev) - 2 * np.sum(vx * strain_d, axis=1)
              + np.einsum("la,lab,lb->l", vx, quadratic, vx)
              + pp / 2 * correlate(g - k, weights * projected_trace))
        bb = (correlate(g, weights * w * vb) - wx * correlate(g, weights * vb)
              - np.sum(vx * b_w_d, axis=1) + wx * np.sum(vx * b_d, axis=1))
        horizontal[i] = -0.25 * (div_q + 2 * curvature * wx * structure[i] + 2 * ae + 2 * bb)
        mlocal = (dw - wx[:, None] * d - vx * (cw - wx * c0)[:, None]
                  + x * (cw2 - 2 * wx * cw + wx**2 * c0)[:, None])
        mixed[i] = np.einsum("lij,lj->li", rotation, mlocal)
        mass[i] = c0
    return {"horizontal": horizontal, "structure": structure, "vertical_flux": vertical,
            "mixed_moment": mixed, "kernel_mass": mass}


if __name__ == "__main__":
    from lossett.calc.compute_group_increments import _solid_angles

    lat = np.linspace(-np.pi / 2, np.pi / 2, 7)
    lon = 0.37 + np.arange(12) * (2 * np.pi / 12)
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    points = np.stack((np.cos(lat2) * np.cos(lon2), np.cos(lat2) * np.sin(lon2), np.sin(lat2)), axis=-1)
    rng = np.random.default_rng(42)
    velocity = rng.normal(size=points.shape)
    weights = _solid_angles(lat, lon)
    projector = np.eye(3) - points[..., :, None] * points[..., None, :]
    grad = np.einsum("...ij,...jk,...kl->...il", projector, rng.normal(size=projector.shape), projector)
    gradw = np.einsum("...ij,...j->...i", projector, rng.normal(size=points.shape))
    for kernel in ({"heat_time": 0.3}, {"mollifier_scale": 0.5}):
      for atmosphere in ("deep", "shallow"):
        start = time.perf_counter()
        result = fft_group(lat, lon, velocity, grad, gradw, weights, radius=2., atmosphere=atmosphere, **kernel)
        elapsed = time.perf_counter() - start
        expected = group_increment_grid(points, velocity, grad, gradw, weights, radius=2.,
                                        atmosphere=atmosphere, quadrature_tolerance=5., **kernel)
        for name, actual in result.items():
            error = np.max(np.abs(actual - expected[name]))
            print(kernel, atmosphere, name, "max error", error, "fft seconds", elapsed, flush=True)
            np.testing.assert_allclose(actual, expected[name], rtol=2e-10, atol=2e-10)
