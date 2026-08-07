import numpy as np

from lossett.calc.spherical_geometry import compute_angular_weights

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

# Tests

def test_weights_sum_to_two_pi():

    n = 64

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

    assert np.isclose(
        weights.sum(),
        2*np.pi,
        atol=1e-6,
    )

def test_uniform_ring():

    n = 64

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

    expected = 2*np.pi / n

    assert np.allclose(
        weights,
        expected,
        atol=1e-6,
    )

def test_duplicate_bearings():

    chi = np.array([
        0,
        0,
        np.pi/2,
        np.pi,
        3*np.pi/2,
    ])

    distance_bin, sin_chi, cos_chi = make_ring_test_geometry(chi)

    weights = compute_angular_weights(
        distance_bin,
        sin_chi,
        cos_chi,
        nbins=1,
    )

    assert np.isclose(
        weights.sum(),
        2*np.pi,
        atol=1e-6,
    )
