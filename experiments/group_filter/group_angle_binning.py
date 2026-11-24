"""Experimental group-angle binning of coherent increment coefficients."""

import math
import time

import numpy as np
from numba import njit

from lossett.filtering.group_kernels import _mollifier_spectral_coefficients
from multiscale_group_prototype import _coefficients


def central_mollifier_nodes(support_angle, bins, scales, degree=2048):
    """Sample nonnegative central G weights on common rotation-angle nodes."""
    angles = np.linspace(0, support_angle, bins+1)
    weights = np.zeros((len(scales), bins+1))
    corrections = []
    masses = []
    nodes, gauss_weights = np.polynomial.legendre.leggauss(8)
    fraction = (nodes+1)/2
    quadrature_angles = angles[:-1, None] + np.diff(angles)[:, None]*fraction
    haar = 2/np.pi * np.sin(quadrature_angles/2)**2
    start = time.perf_counter()
    for i, scale in enumerate(scales):
        coeff = _mollifier_spectral_coefficients(float(scale), degree)
        summed = np.cumsum(coeff[::-1])[::-1]
        cheb = np.r_[summed[0], 2*summed[1:]]
        inside = angles < 4*scale
        values = np.polynomial.chebyshev.chebval(np.cos(angles[inside]), cheb)
        floor = float(np.min(values))
        peak = float(np.max(values))
        if floor < -1e-10*peak:
            raise ValueError(f"group-kernel spectral negativity exceeds tolerance: {floor}/{peak}")
        corrections.append(max(0., -floor))
        weights[i, inside] = np.maximum(values, 0)
        interpolated = (weights[i, :-1, None]*(1-fraction)
                        + weights[i, 1:, None]*fraction)
        mass = float(np.sum(interpolated*haar*gauss_weights*np.diff(angles)[:, None]/2))
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError(f"group-kernel normalization is invalid: {mass}")
        masses.append(mass)
        weights[i] /= mass
    return weights, corrections, masses, time.perf_counter()-start


@njit(cache=True)
def accumulate_group_angle_bins(x, u, directions, velocity, grad, gradw, weights,
                                curvature, indices, support_angle, bins, cospsi):
    """Bin pair/fiber contributions against linear hats of rotation angle."""
    summaries = np.zeros((bins+1, 14))
    factor = 1/(4*math.pi*len(cospsi))
    for target in indices:
        y = directions[target]
        mu = min(1., max(-1., x[0]*y[0]+x[1]*y[1]+x[2]*y[2]))
        c = _coefficients(x, u, y, velocity[target], grad[target], gradw[target], curvature)
        for psi in cospsi:
            cosphi = min(1., max(-1., (1+mu)*(1+psi)/2-1))
            angle = math.acos(cosphi)
            if angle > support_angle:
                continue
            position = angle/support_angle*bins
            left = min(int(position), bins-1)
            fraction = position-left
            spin2 = 2*psi*psi-1
            for channel in range(14):
                spin = psi if channel in (1, 4, 6, 8, 10, 12) else spin2 if channel == 2 else 1.
                value = factor*weights[target]*c[channel]*spin
                summaries[left, channel] += value*(1-fraction)
                summaries[left+1, channel] += value*fraction
    return summaries


@njit(cache=True)
def integrate_group_angle_bins(summaries, kernel_nodes):
    """Apply each sampled central kernel to the shared fiber-bin summaries."""
    out = np.zeros((kernel_nodes.shape[0], 7))
    for scale in range(len(kernel_nodes)):
        for node in range(len(summaries)):
            c, w = summaries[node], kernel_nodes[scale, node]
            out[scale, 0] += w*(c[0]+c[1]+c[2])
            out[scale, 1] += w*(c[3]+c[4])
            out[scale, 2] += w*(c[5]+c[6])
            out[scale, 3] += w*(c[7]+c[8])
            out[scale, 4] += w*(c[9]+c[10])
            out[scale, 5] += w*(c[11]+c[12])
            out[scale, 6] += w*c[13]
    return out


@njit(cache=True)
def accumulate_segmented_group_bins(x, u, directions, velocity, grad, gradw, weights,
                                    curvature, indices, support_angle, bins, nodes, quadrature):
    """Bin coherent pair contributions with Gauss integration on each fiber interval."""
    summaries = np.zeros((bins+1, 14))
    angle_step = support_angle/bins
    for target in indices:
        y = directions[target]
        mu = min(1., max(-1., x[0]*y[0]+x[1]*y[1]+x[2]*y[2]))
        theta = math.acos(mu)
        if theta > support_angle:
            continue
        c = _coefficients(x, u, y, velocity[target], grad[target], gradw[target], curvature)
        first = min(int(theta/angle_step), bins-1)
        for b in range(first, bins):
            low = max(b*angle_step, theta)
            high = (b+1)*angle_step
            c0 = min(1., max(-1., 2*(1+math.cos(low))/(1+mu)-1))
            c1 = min(1., max(-1., 2*(1+math.cos(high))/(1+mu)-1))
            psi0, psi1 = math.acos(c0), math.acos(c1)
            width = psi1-psi0
            for q in range(len(nodes)):
                psi = psi0 + width*(nodes[q]+1)/2
                spin1 = math.cos(psi)
                cosphi = min(1., max(-1., (1+mu)*(1+spin1)/2-1))
                phi = math.acos(cosphi)
                fraction = min(1., max(0., phi/angle_step-b))
                factor = weights[target]*quadrature[q]*width/(8*math.pi**2)
                spin2 = 2*spin1*spin1-1
                for channel in range(14):
                    spin = spin1 if channel in (1, 4, 6, 8, 10, 12) else spin2 if channel == 2 else 1.
                    value = factor*c[channel]*spin
                    summaries[b, channel] += value*(1-fraction)
                    summaries[b+1, channel] += value*fraction
    return summaries


def group_angle_moments(mu, support_angle, kernel_nodes, fiber_points):
    """Endpoint moments of one interpolated central weight, by fiber quadrature."""
    mu = np.asarray(mu)
    psi = 2*np.pi*np.arange(fiber_points)/fiber_points
    cosine = np.cos(psi)
    result = np.empty((len(mu), 3))
    rotation_nodes = np.linspace(0, support_angle, len(kernel_nodes))
    for start in range(0, len(mu), 256):
        chunk = mu[start:start+256]
        phi = np.arccos(np.clip((1+chunk[:, None])*(1+cosine[None, :])/2-1, -1, 1))
        value = np.interp(phi, rotation_nodes, kernel_nodes, right=0.)
        result[start:start+len(chunk), 0] = np.mean(value, axis=1)/(4*np.pi)
        result[start:start+len(chunk), 1] = np.mean(value*cosine, axis=1)/(4*np.pi)
        result[start:start+len(chunk), 2] = np.mean(value*(2*cosine*cosine-1), axis=1)/(4*np.pi)
    return result


def segmented_group_angle_moments(mu, support_angle, kernel_nodes, order):
    """Integrate each piecewise-linear group-angle interval separately."""
    mu = np.asarray(mu)
    if np.any(mu <= -1) or np.any(mu > 1):
        raise ValueError("segmented fiber quadrature requires -1 < mu <= 1")
    nodes, gauss_weights = np.polynomial.legendre.leggauss(order)
    angles = np.linspace(0, support_angle, len(kernel_nodes))
    angle_step = support_angle/(len(kernel_nodes)-1)
    result = np.empty((len(mu), 3))
    for start in range(0, len(mu), 32):
        chunk = mu[start:start+32]
        theta = np.arccos(chunk)
        edge_angles = np.maximum(angles[None, :], theta[:, None])
        cospsi = 2*(1+np.cos(edge_angles))/(1+chunk[:, None])-1
        edge_psi = np.arccos(np.clip(cospsi, -1, 1))
        widths = edge_psi[:, 1:] - edge_psi[:, :-1]
        psi = edge_psi[:, :-1, None] + widths[..., None]*(nodes+1)/2
        cospsi_quad = np.cos(psi)
        phi = np.arccos(np.clip((1+chunk[:, None, None])*(1+cospsi_quad)/2-1, -1, 1))
        fraction = np.clip((phi-angles[None, :-1, None])/angle_step, 0, 1)
        values = (kernel_nodes[None, :-1, None]*(1-fraction)
                  + kernel_nodes[None, 1:, None]*fraction)
        weights = widths[..., None]*gauss_weights[None, None, :]/(8*np.pi**2)
        for spin in range(3):
            factor = 1 if spin == 0 else cospsi_quad if spin == 1 else 2*cospsi_quad**2-1
            result[start:start+len(chunk), spin] = np.sum(weights*values*factor, axis=(1, 2))
    return result
