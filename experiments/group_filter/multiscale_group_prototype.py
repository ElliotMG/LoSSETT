"""Experimental multi-scale group-increment accumulators for selected origins.

The inputs are the existing accumulator's arrays with per-pair moments
provided separately. Binning uses linear interpolation in angular distance.
"""

import math

import numpy as np
from numba import njit

from lossett.calc._group_numba import _transport


@njit(inline="always")
def _coefficients(x, u, y, uy, grad, gradw, curvature):
    wx = x[0] * u[0] + x[1] * u[1] + x[2] * u[2]
    vx = (u[0] - wx*x[0], u[1] - wx*x[1], u[2] - wx*x[2])
    pp = vx[0]**2 + vx[1]**2 + vx[2]**2
    wy = y[0]*uy[0] + y[1]*uy[1] + y[2]*uy[2]
    vy = (uy[0] - wy*y[0], uy[1] - wy*y[1], uy[2] - wy*y[2])
    p = _transport(vx[0], vx[1], vx[2], x[0], x[1], x[2], y[0], y[1], y[2])
    back = _transport(vy[0], vy[1], vy[2], y[0], y[1], y[2], x[0], x[1], x[2])
    dw = wy - wx
    qg = vy[0]**2 + vy[1]**2 + vy[2]**2 + pp + dw**2
    qh = -2 * (vy[0]*p[0] + vy[1]*p[1] + vy[2]*p[2])
    divergence = grad[0, 0] + grad[1, 1] + grad[2, 2]
    ag = (divergence + 2*curvature*wx) * qg
    ah = (divergence + 2*curvature*wx) * qh
    ak = 0.
    for a in range(3):
        b = gradw[a] - curvature*vy[a]
        ag += 2*dw*vy[a]*b
        ah -= 2*dw*p[a]*b
        for bidx in range(3):
            projection = (1. if a == bidx else 0.) - y[a]*y[bidx]
            strain = (grad[a, bidx] + grad[bidx, a])/2 + curvature*wy*projection
            ag += 2*(vy[a]*vy[bidx] + pp*projection/2)*strain
            ah -= 2*(vy[a]*p[bidx] + p[a]*vy[bidx])*strain
            ak += 2*(p[a]*p[bidx] - pp*projection/2)*strain
    return (
        -0.25*ag, -0.25*ah, -0.25*ak,
        qg, qh, dw*qg, dw*qh,
        dw*(-vx[0] + dw*x[0]), dw*back[0],
        dw*(-vx[1] + dw*x[1]), dw*back[1],
        dw*(-vx[2] + dw*x[2]), dw*back[2],
        1.,
    )


@njit(inline="always")
def _apply(out, scale, c, g, h, k, weight):
    out[scale, 0] += weight*(c[0]*g + c[1]*h + c[2]*k)
    out[scale, 1] += weight*(c[3]*g + c[4]*h)
    out[scale, 2] += weight*(c[5]*g + c[6]*h)
    out[scale, 3] += weight*(c[7]*g + c[8]*h)
    out[scale, 4] += weight*(c[9]*g + c[10]*h)
    out[scale, 5] += weight*(c[11]*g + c[12]*h)
    out[scale, 6] += weight*c[13]*g


@njit(cache=True)
def exact_multiscale(x, u, directions, velocity, grad, gradw, weights,
                     moments, curvature, indices):
    """Accumulate precomputed exact moments, shape (scales, pairs, 3)."""
    out = np.zeros((moments.shape[0], 7))
    for j in range(len(indices)):
        target = indices[j]
        c = _coefficients(x, u, directions[target], velocity[target],
                          grad[target], gradw[target], curvature)
        for scale in range(moments.shape[0]):
            _apply(out, scale, c, moments[scale, j, 0], moments[scale, j, 1],
                   moments[scale, j, 2], weights[target])
    return out


@njit(cache=True)
def binned_coefficients(x, u, directions, velocity, grad, gradw, weights,
                        curvature, indices, support_angle, bins):
    """Bin weighted pair coefficients for piecewise-linear moment interpolation."""
    summaries = np.zeros((bins + 1, 14))
    for j in range(len(indices)):
        target = indices[j]
        y = directions[target]
        cosine = min(1., max(-1., x[0]*y[0] + x[1]*y[1] + x[2]*y[2]))
        distance = math.acos(cosine)
        if distance > support_angle:
            continue
        position = distance / support_angle * bins
        left = min(int(position), bins - 1)
        right_weight = position - left
        c = _coefficients(x, u, y, velocity[target], grad[target], gradw[target], curvature)
        for channel in range(14):
            value = weights[target] * c[channel]
            summaries[left, channel] += value*(1 - right_weight)
            summaries[left + 1, channel] += value*right_weight
    return summaries


@njit(cache=True)
def integrate_bins(summaries, moments):
    """Apply moments at bin nodes, shape (scales, bins + 1, 3)."""
    out = np.zeros((moments.shape[0], 7))
    for node in range(len(summaries)):
        c = summaries[node]
        for scale in range(moments.shape[0]):
            _apply(out, scale, c, moments[scale, node, 0], moments[scale, node, 1],
                   moments[scale, node, 2], 1.)
    return out
