"""Tests for the fair ring comparison (run with PYTHONPATH=src:experiments/group_filter)."""

import numpy as np
import pytest

import ring_fair_comparison as rfc
from group_angle_binning import central_mollifier_nodes
from lossett.calc.compute_group_increments import _solid_angles
from lossett.calc.group_increments import latlon_shell_fields
from lossett.filtering.group_kernels import mollifier_fiber_moments_quadrature
from moment_sharing_benchmark import exact_chebyshev, exact_ring_table

RADIUS = 6382758.772639869


@pytest.fixture(scope="module")
def small_field():
    nlat, nlon = 37, 48
    lat = np.linspace(-np.pi/2, np.pi/2, nlat)
    lon = 2*np.pi*np.arange(nlon)/nlon
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    u = 10*np.cos(la)**2*np.sin(2*lo)
    v = 5*np.sin(3*lo)*np.cos(la)**2
    w = 0.1*np.cos(lo)*np.cos(la)**2
    direction, velocity, grad, gradw = latlon_shell_fields(lat, lon, u, v, w, radius=RADIUS)
    return {"nlat": nlat, "nlon": nlon, "lat": lat, "lon": lon,
            "weights": _solid_angles(lat, lon).ravel(), "direction": direction.reshape(-1, 3),
            "velocity": velocity.reshape(-1, 3), "grad": grad.reshape(-1, 3, 3), "gradw": gradw.reshape(-1, 3)}


SCALES = np.array([1500e3, 2500e3])/(4*RADIUS)


def test_moment_tables_reproduce_quadrature_builder():
    coefficients, supports = rfc.moment_tables(SCALES)
    assert coefficients.shape[:2] == (2, 64) and coefficients.shape[3] == 3
    mu = np.cos(np.linspace(0, supports[0]*.999, 17))
    expected = np.asarray(mollifier_fiber_moments_quadrature(mu, SCALES[0]))
    actual = rfc.evaluate_moment_table(mu, coefficients[0], supports[0])
    assert np.allclose(actual, expected, rtol=0, atol=1e-12*np.max(np.abs(expected)))


def test_ring_table_accumulation_matches_direct_chebyshev(small_field):
    f = small_field
    coefficients, supports = rfc.moment_tables(SCALES)
    row, col = 10, 7
    index = row*f["nlon"]+col
    x, u = f["direction"][index], f["velocity"][index]
    mu = np.clip(f["direction"]@x, -1, 1)
    active = np.flatnonzero(mu >= np.cos(supports.max()))
    table = np.zeros((len(SCALES), f["nlat"], f["nlon"], 3))
    rfc.build_ring_table(f["direction"][row*f["nlon"]], f["direction"], active_for_origin(f, row, supports),
                         supports, coefficients, table)
    shared = exact_ring_table(x, u, f["direction"], f["velocity"], f["grad"], f["gradw"], f["weights"], table,
                              1/RADIUS, active, f["nlon"], col)
    direct = exact_chebyshev(x, u, f["direction"], f["velocity"], f["grad"], f["gradw"], f["weights"], 1/RADIUS,
                             active, supports, coefficients)
    assert np.max(np.abs(shared-direct)) <= 1e-12*np.max(np.abs(direct))


def active_for_origin(f, row, supports):
    x = f["direction"][row*f["nlon"]]
    return np.flatnonzero(np.clip(f["direction"]@x, -1, 1) >= np.cos(supports.max()))


def test_quadrature_kernel_nodes_agree_with_spectral_kernel():
    bins, support = 256, 4*SCALES[1]
    quadrature = rfc.quadrature_kernel_nodes(support, bins, SCALES[1:])
    spectral = central_mollifier_nodes(support, bins, SCALES[1:])[0]
    assert np.all(quadrature >= 0)
    assert np.max(np.abs(quadrature-spectral)) <= 1e-6*np.max(spectral)


def test_distance_table_pipeline_approximates_exact(small_field):
    f = small_field
    coefficients, supports = rfc.moment_tables(SCALES)
    row, col = 18, 11
    index = row*f["nlon"]+col
    x, u = f["direction"][index], f["velocity"][index]
    active = np.flatnonzero(np.clip(f["direction"]@x, -1, 1) >= np.cos(supports.max()))
    direct = exact_chebyshev(x, u, f["direction"], f["velocity"], f["grad"], f["gradw"], f["weights"], 1/RADIUS,
                             active, supports, coefficients)
    pipeline = rfc.DistanceTablePipeline(supports.max(), SCALES, rotation_bins=256, distance_bins=1024)
    result = pipeline.evaluate(x, u, f["direction"], f["velocity"], f["grad"], f["gradw"], f["weights"],
                               1/RADIUS, active)
    peaks = np.max(np.abs(direct), axis=0)
    assert np.max(np.abs(result-direct)/peaks) < 5e-3
