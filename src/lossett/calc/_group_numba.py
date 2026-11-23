"""Fused pairwise accumulation for the coherent group shell diagnostic."""

import math

from numba import njit


@njit(inline="always")
def _transport(v0, v1, v2, s0, s1, s2, t0, t1, t2):
    cosine = min(1., max(-1., s0 * t0 + s1 * t1 + s2 * t2))
    a0 = s1 * t2 - s2 * t1
    a1 = s2 * t0 - s0 * t2
    a2 = s0 * t1 - s1 * t0
    sine = math.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
    if sine <= 1e-14:
        if cosine >= 0:
            return cosine * v0, cosine * v1, cosine * v2
        if abs(s0) <= abs(s1) and abs(s0) <= abs(s2):
            a0, a1, a2 = 0., s2, -s1
        elif abs(s1) <= abs(s2):
            a0, a1, a2 = -s2, 0., s0
        else:
            a0, a1, a2 = s1, -s0, 0.
        norm = math.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
        a0, a1, a2 = a0 / norm, a1 / norm, a2 / norm
        dot = 2 * (a0 * v0 + a1 * v1 + a2 * v2)
        return dot * a0 - v0, dot * a1 - v1, dot * a2 - v2
    a0, a1, a2 = a0 / sine, a1 / sine, a2 / sine
    dot = (1 - cosine) * (a0 * v0 + a1 * v1 + a2 * v2)
    return (
        cosine * v0 + sine * (a1 * v2 - a2 * v1) + dot * a0,
        cosine * v1 + sine * (a2 * v0 - a0 * v2) + dot * a1,
        cosine * v2 + sine * (a0 * v1 - a1 * v0) + dot * a2,
    )


@njit(cache=True)
def accumulate_group_pairs(x, u, directions, velocity, grad, gradw, weights, g, h, k, curvature, indices):
    wx = x[0] * u[0] + x[1] * u[1] + x[2] * u[2]
    vx = (u[0] - wx * x[0], u[1] - wx * x[1], u[2] - wx * x[2])
    pp = vx[0] * vx[0] + vx[1] * vx[1] + vx[2] * vx[2]
    horizontal = structure = vertical_flux = mass = 0.
    m0 = m1 = m2 = 0.
    for j in range(len(g)):
        target = j if indices is None else indices[j]
        y = directions[target]
        uy = velocity[target]
        wy = y[0] * uy[0] + y[1] * uy[1] + y[2] * uy[2]
        vy = (uy[0] - wy * y[0], uy[1] - wy * y[1], uy[2] - wy * y[2])
        p = _transport(vx[0], vx[1], vx[2], x[0], x[1], x[2], y[0], y[1], y[2])
        back = _transport(vy[0], vy[1], vy[2], y[0], y[1], y[2], x[0], x[1], x[2])
        dw = wy - wx
        vv = vy[0] * vy[0] + vy[1] * vy[1] + vy[2] * vy[2]
        vp = vy[0] * p[0] + vy[1] * p[1] + vy[2] * p[2]
        gj, hj, kj = g[j], h[j], k[j]
        q = gj * (vv + pp + dw * dw) - 2 * hj * vp
        divergence = grad[target, 0, 0] + grad[target, 1, 1] + grad[target, 2, 2]
        integrand = (divergence + 2 * curvature * wx) * q
        for a in range(3):
            b = gradw[target, a] - curvature * vy[a]
            integrand += 2 * dw * (gj * vy[a] - hj * p[a]) * b
            for bidx in range(3):
                projection = (1. if a == bidx else 0.) - y[a] * y[bidx]
                strain = (grad[target, a, bidx] + grad[target, bidx, a]) / 2
                strain += curvature * wy * projection
                tensor = (
                    gj * vy[a] * vy[bidx]
                    - hj * (vy[a] * p[bidx] + p[a] * vy[bidx])
                    + kj * p[a] * p[bidx]
                    + (gj - kj) * pp * projection / 2
                )
                integrand += 2 * tensor * strain
        weight = weights[target]
        horizontal += weight * integrand
        structure += weight * q
        vertical_flux += weight * dw * q
        m0 += weight * dw * (hj * back[0] - gj * vx[0] + gj * dw * x[0])
        m1 += weight * dw * (hj * back[1] - gj * vx[1] + gj * dw * x[1])
        m2 += weight * dw * (hj * back[2] - gj * vx[2] + gj * dw * x[2])
        mass += weight * gj
    return -0.25 * horizontal, structure, vertical_flux, m0, m1, m2, mass
