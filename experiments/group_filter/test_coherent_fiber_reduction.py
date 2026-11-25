"""Tests for the coherent-binning fiber-work reductions (run with PYTHONPATH=src:experiments/group_filter).

Convention shared with group_angle_binning.accumulate_segmented_group_bins: the fiber integral is taken over
the full circle (twice the integral over psi in [0, pi]) with weight 1/(8 pi^2), for the spin factors
1, cos(psi), cos(2 psi).
"""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

import coherent_fiber_reduction as cfr
from group_angle_binning import central_mollifier_nodes

SUPPORT = 0.6
GAUSS = leggauss(8)


def spin_factors(psi):
    return np.stack((np.ones_like(psi), np.cos(psi), np.cos(2*psi)))


def numeric_fiber(mu, node_function, bins):
    """Reference fiber integral of hat_k(psi)*spin(psi) by dense midpoint quadrature."""
    count = 400000
    psi = (np.arange(count)+.5)*np.pi/count
    hats = node_function(mu, psi)
    return np.einsum("sn,kn->sk", spin_factors(psi), hats)*(2*np.pi/count)/(8*np.pi**2)


def phi_hats(bins):
    def function(mu, psi):
        phi = np.arccos(np.clip((1+mu)*(1+np.cos(psi))/2-1, -1, 1))
        nodes = np.linspace(0, SUPPORT, bins+1)
        return np.stack([np.interp(phi, nodes, np.eye(bins+1)[k], right=0.) for k in range(bins+1)])
    return function


def z_hats(znodes):
    def function(mu, psi):
        z = (1+mu)/2*(1+np.cos(psi))/2
        ascending = znodes[::-1]
        return np.stack([np.interp(z, ascending, np.eye(len(znodes))[::-1][k], left=0., right=0.)
                         for k in range(len(znodes))])
    return function


def test_sin_power_integrals_match_quadrature():
    u = 0.9
    nodes, weights = leggauss(60)
    grid = u*(nodes+1)/2
    expected = [np.sum(weights*np.sin(grid)**(2*n))*u/2 for n in range(4)]
    assert np.allclose(cfr._sin_power_integrals(u), expected, rtol=0, atol=1e-13)


@pytest.mark.parametrize("mu", [0.999, 0.9, 0.85, 0.7])
def test_z_hat_fibers_match_quadrature(mu):
    bins = 12
    angles = np.linspace(0, SUPPORT, bins+1)
    znodes = np.cos(angles/2)**2
    out = np.zeros((3, bins+1))
    cfr.z_hat_fibers(mu, znodes, out)
    expected = numeric_fiber(mu, z_hats(znodes), bins)
    assert np.allclose(out[:, :-1], expected[:, :-1], rtol=0, atol=1e-9)
    assert np.allclose(out, expected, rtol=0, atol=2e-7)   # reference is limited by the hat jump at the support edge


@pytest.mark.parametrize("mu", [0.999, 0.9, 0.85])
def test_distance_table_nodes_match_quadrature(mu):
    bins = 12
    thetas = np.array([np.arccos(mu)])
    table = cfr.fiber_hat_table(np.repeat(thetas, 2), SUPPORT, bins, *GAUSS)
    expected = numeric_fiber(mu, phi_hats(bins), bins)
    assert np.allclose(table[:, 0, :-1], expected[:, :-1], rtol=0, atol=1e-9)
    assert np.allclose(table[:, 0, :], expected, rtol=0, atol=2e-7)


def test_distance_table_interpolation_converges():
    bins, mu = 24, np.cos(0.3137)
    exact = numeric_fiber(mu, phi_hats(bins), bins)
    errors = []
    for distance_bins in (96, 384, 1536):
        thetas = np.linspace(0, SUPPORT, distance_bins+1)
        table = cfr.fiber_hat_table(thetas, SUPPORT, bins, *GAUSS)
        j = int(np.arccos(mu)/(SUPPORT/distance_bins))
        fraction = np.arccos(mu)/(SUPPORT/distance_bins)-j
        interpolated = (1-fraction)*table[:, j, :]+fraction*table[:, j+1, :]
        errors.append(np.max(np.abs(interpolated-exact)))
    assert errors[1] < errors[0]/2 and errors[2] < errors[1]/2


def test_z_kernel_has_unit_haar_mass_and_is_nonnegative():
    scales = np.array([0.157])
    znodes, kernel, masses = cfr.z_kernel_nodes(1.0, 64, scales)
    assert np.all(kernel >= 0)
    assert np.allclose(masses, masses[0])
    assert np.all(kernel[0][np.arccos(np.sqrt(znodes))*2 > 4*scales[0]] == 0)
    nodes, weights = leggauss(40)
    total = 0.
    for k in range(len(znodes)-1):
        lo, hi = znodes[k+1], znodes[k]
        z = lo+(hi-lo)*(nodes+1)/2
        f = (z-lo)/(hi-lo)
        value = kernel[0][k+1]*(1-f)+kernel[0][k]*f
        total += np.sum(value*2/np.pi*np.sqrt((1-z)/z)*weights)*(hi-lo)/2
    assert total == pytest.approx(1., abs=1e-7)


def test_phi_kernel_unit_haar_mass_convention():
    """Both kernel families must share the Haar-probability normalization, so their results are comparable."""
    bins = 256
    kernel = central_mollifier_nodes(1.0, bins, np.array([0.157]))[0][0]
    angles = np.linspace(0, 1.0, bins+1)
    mass = np.sum(0.5*(kernel[1:]*np.sin(angles[1:]/2)**2+kernel[:-1]*np.sin(angles[:-1]/2)**2)
                  * np.diff(angles))*2/np.pi
    assert mass == pytest.approx(1., abs=1e-4)
