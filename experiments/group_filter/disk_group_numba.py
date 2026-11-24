"""Experimental pairwise backend using per-latitude disk-backed rotations."""
import numpy as np
from numba import njit


@njit
def accumulate(x, u, directions, velocity, grad, gradw, weights, moments, matrices, indices, curvature, angle):
    wx = x[0] * u[0] + x[1] * u[1] + x[2] * u[2]
    vx = u - wx * x
    pp = vx[0] * vx[0] + vx[1] * vx[1] + vx[2] * vx[2]
    c, s = np.cos(angle), np.sin(angle)
    vx0, vx1, vx2 = c * vx[0] + s * vx[1], -s * vx[0] + c * vx[1], vx[2]
    horizontal = structure = flux = mass = 0.
    m0 = m1 = m2 = 0.
    p = np.empty(3)
    back = np.empty(3)
    vy = np.empty(3)
    for j in range(len(indices)):
        target = indices[j]
        y, uy = directions[target], velocity[target]
        wy = y[0] * uy[0] + y[1] * uy[1] + y[2] * uy[2]
        for a in range(3):
            vy[a] = uy[a] - wy * y[a]
        vy0, vy1, vy2 = c * vy[0] + s * vy[1], -s * vy[0] + c * vy[1], vy[2]
        rotation = matrices[j]
        p0 = rotation[0, 0] * vx0 + rotation[0, 1] * vx1 + rotation[0, 2] * vx2
        p1 = rotation[1, 0] * vx0 + rotation[1, 1] * vx1 + rotation[1, 2] * vx2
        p2 = rotation[2, 0] * vx0 + rotation[2, 1] * vx1 + rotation[2, 2] * vx2
        b0 = rotation[0, 0] * vy0 + rotation[1, 0] * vy1 + rotation[2, 0] * vy2
        b1 = rotation[0, 1] * vy0 + rotation[1, 1] * vy1 + rotation[2, 1] * vy2
        b2 = rotation[0, 2] * vy0 + rotation[1, 2] * vy1 + rotation[2, 2] * vy2
        p[0], p[1], p[2] = c * p0 - s * p1, s * p0 + c * p1, p2
        back[0], back[1], back[2] = c * b0 - s * b1, s * b0 + c * b1, b2
        dw = wy - wx
        g, h, k = moments[j]
        vv = vy[0] * vy[0] + vy[1] * vy[1] + vy[2] * vy[2]
        vp = vy[0] * p[0] + vy[1] * p[1] + vy[2] * p[2]
        q = g * (vv + pp + dw * dw) - 2 * h * vp
        div = grad[target, 0, 0] + grad[target, 1, 1] + grad[target, 2, 2]
        integrand = (div + 2 * curvature * wx) * q
        for a in range(3):
            integrand += 2 * dw * (g * vy[a] - h * p[a]) * (gradw[target, a] - curvature * vy[a])
            for b in range(3):
                proj = (1. if a == b else 0.) - y[a] * y[b]
                strain = (grad[target, a, b] + grad[target, b, a]) / 2 + curvature * wy * proj
                tensor = (g * vy[a] * vy[b] - h * (vy[a] * p[b] + p[a] * vy[b])
                          + k * p[a] * p[b] + (g - k) * pp * proj / 2)
                integrand += 2 * tensor * strain
        weight = weights[target]
        horizontal += weight * integrand
        structure += weight * q
        flux += weight * dw * q
        m0 += weight * dw * (h * back[0] - g * vx[0] + g * dw * x[0])
        m1 += weight * dw * (h * back[1] - g * vx[1] + g * dw * x[1])
        m2 += weight * dw * (h * back[2] - g * vx[2] + g * dw * x[2])
        mass += weight * g
    return np.array([-0.25 * horizontal, structure, flux, m0, m1, m2, mass])
