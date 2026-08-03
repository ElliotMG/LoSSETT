import numpy as np

from lossett.calc.compute_spherical_geometry import compute_angular_weights

# Helper functions

def make_ring_test_geometry(chi):

    chi = np.asarray(chi)

    distance_bin = np.zeros(
        (1, len(chi)),
        dtype=np.uint8,
    )

    sin_chi = np.sin(chi)[None, :]
    cos_chi = np.cos(chi)[None, :]

    return (
        distance_bin,
        sin_chi,
        cos_chi,
    )

def angular_integral(f, weights):
    return np.sum(
        f * weights
    )

# Tests

def test_constant_integral():
    """
    Analytically, the integral of 1 over an angular interval of 2*pi
    is equal to 2*pi.
    """
    n = 128

    chi = np.linspace(
        0,
        2*np.pi,
        n,
        endpoint=False,
    )

    distance_bin, sin_chi, cos_chi = make_ring_test_geometry(chi)

    weights = compute_angular_weights(
        distance_bin,
        sin_chi,
        cos_chi,
        nbins=1,
    )

    result = angular_integral(
        np.ones(n),
        weights,
    )

    assert np.isclose(
        result,
        2*np.pi,
        atol=1e-6,
    )

def test_sine_integral():
    """
    Analytically, the integral of sin(x) over an angular interval of 2*pi
    is zero.
    """
    n = 128

    chi = np.linspace(
        0,
        2*np.pi,
        n,
        endpoint=False,
    )

    distance_bin, sin_chi, cos_chi = make_ring_test_geometry(chi)

    weights = compute_angular_weights(
        distance_bin,
        sin_chi,
        cos_chi,
        nbins=1,
    )

    result = np.sum(
        np.sin(chi)*weights
    )

    assert abs(result) < 1e-6

def test_cosine_integral():
    """
    Analytically, the integral of cos(x) over an angular interval of 2*pi
    is zero.
    """
    n = 128

    chi = np.linspace(
        0,
        2*np.pi,
        n,
        endpoint=False,
    )

    distance_bin, sin_chi, cos_chi = make_ring_test_geometry(chi)

    weights = compute_angular_weights(
        distance_bin,
        sin_chi,
        cos_chi,
        nbins=1,
    )

    result = np.sum(
        np.cos(chi)*weights
    )

    assert abs(result) < 1e-6

def test_fourier_mode():
    """
    Analytically, the integral of cos(5*x) over an angular interval of 2*pi
    is zero. This high Fourier mode tests angular anisotropy.
    """

    n = 128

    chi = np.linspace(
        0,
        2*np.pi,
        n,
        endpoint=False,
    )

    distance_bin, sin_chi, cos_chi = make_ring_test_geometry(chi)

    weights = compute_angular_weights(
        distance_bin,
        sin_chi,
        cos_chi,
        nbins=1,
    )

    result = np.sum(
        np.cos(5*chi)*weights
    )

    assert abs(result) < 1e-6

