"""Compare latitude quadrature for exact scalar harmonic filter responses."""

import json
import os
import resource
import socket
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from lossett.calc.compute_group_increments import _solid_angles
from lossett.filtering.group_kernels import mollifier_fiber_moments, mollifier_les_fiber_moments
from h_twice_rings_benchmark import RADIUS, SCALE, h_scalar


OUTPUT = Path("/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/h_twice_quadrature_harmonics_cc_20261124")
NAMES = ("constant", "z_degree1", "x_degree1", "zonal_degree2")
ORIGINS = np.deg2rad(np.array([0., 45., 80.]))


def clenshaw_curtis_weights(intervals):
    theta = np.pi*np.arange(intervals+1)/intervals
    result = np.empty(intervals+1)
    if intervals % 2:
        result[0] = result[-1] = 1/intervals**2
        indices = np.arange(1, (intervals-1)//2+1)
    else:
        result[0] = result[-1] = 1/(intervals**2-1)
        indices = np.arange(1, intervals//2)
    interior = np.ones(intervals-1)
    for k in indices:
        interior -= 2*np.cos(2*k*theta[1:-1])/(4*k*k-1)
    if intervals % 2 == 0:
        interior -= np.cos(intervals*theta[1:-1])/(intervals**2-1)
    result[1:-1] = 2*interior/intervals
    np.testing.assert_allclose(np.sum(result), 2., rtol=0, atol=1e-14)
    np.testing.assert_allclose(result @ np.cos(theta)**2, 2/3, rtol=0, atol=1e-14)
    if np.any(result <= 0):
        raise ValueError("Clenshaw-Curtis latitude weights must be positive")
    return result


def direction_moments(target_lat, lat, nlon, radius, kernel):
    rows = np.flatnonzero(np.abs(lat-target_lat) <= radius+1e-14)
    delta = 2*np.pi*np.arange(nlon)/nlon
    mu = np.clip(np.sin(target_lat)*np.sin(lat[rows, None])
                 + np.cos(target_lat)*np.cos(lat[rows, None])*np.cos(delta), -1, 1)
    return rows, np.fft.rfft(kernel(mu), axis=-1)


def apply_one(target_lat, lat, nlon, spectra, radius, kernel):
    rows, response = direction_moments(target_lat, lat, nlon, radius, kernel)
    return np.fft.irfft(np.sum(np.conj(response)[None, :, :]*spectra[:, rows, :], axis=1),
                         n=nlon, axis=-1)


def main():
    OUTPUT.mkdir(exist_ok=False)
    center = float(mollifier_les_fiber_moments(np.array([1.]), SCALE)[0][0])
    nodes, weights = leggauss(256)
    lower = np.cos(2*SCALE)
    mu = lower + (1-lower)*(nodes+1)/2
    shape = np.exp(-1/(1-(np.arccos(mu)/(2*SCALE))**2))
    normalization = weights @ shape
    multipliers = np.array([1., (weights*shape) @ mu/normalization,
                            (weights*shape) @ ((3*mu**2-1)/2)/normalization])
    np.testing.assert_array_less(np.abs(multipliers[1:]), 1.)
    results = []
    for n in (160, 320):
        nlat, nlon = 3*n//2+1, 2*n
        lon = 2*np.pi*np.arange(nlon)/nlon
        for kind in ("cell_midpoint", "clenshaw_curtis", "gauss_legendre"):
            if kind == "cell_midpoint":
                lat = np.linspace(-np.pi/2, np.pi/2, nlat)
                area = _solid_angles(lat, lon)
            elif kind == "clenshaw_curtis":
                lat = np.linspace(-np.pi/2, np.pi/2, nlat)
                area = np.broadcast_to(clenshaw_curtis_weights(nlat-1)[:, None]*2*np.pi/nlon, (nlat, nlon))
            else:
                z, gauss_weights = leggauss(nlat)
                lat = np.arcsin(z)
                area = np.broadcast_to(gauss_weights[:, None]*2*np.pi/nlon, (nlat, nlon))
            z = np.broadcast_to(np.sin(lat)[:, None], (nlat, nlon))
            xx = np.cos(lat)[:, None]*np.cos(lon)[None, :]
            inputs = np.stack((np.ones_like(z), z, xx, (3*z*z-1)/2))
            kernels_h = lambda mu: h_scalar(mu, SCALE, center)
            kernels_g = lambda mu: mollifier_fiber_moments(mu, SCALE)[0]
            start = time.perf_counter()
            spectra = np.fft.rfft(inputs*area[None, :, :], axis=2)
            first = np.empty_like(inputs)
            for i, latitude in enumerate(lat):
                first[:, i] = apply_one(latitude, lat, nlon, spectra, 2*SCALE, kernels_h)
            first_spectra = np.fft.rfft(first*area[None, :, :], axis=2)
            twice = np.stack([apply_one(x, lat, nlon, first_spectra, 2*SCALE, kernels_h)
                              for x in ORIGINS], axis=1)
            h_time = time.perf_counter() - start
            start = time.perf_counter()
            direct = np.stack([apply_one(x, lat, nlon, spectra, 4*SCALE, kernels_g)
                               for x in ORIGINS], axis=1)
            g_time = time.perf_counter() - start
            z0 = np.broadcast_to(np.sin(ORIGINS)[:, None], (len(ORIGINS), nlon))
            x0 = np.cos(ORIGINS)[:, None]*np.cos(lon)[None, :]
            analytic = np.stack((np.ones_like(z0), multipliers[1]**2*z0,
                                 multipliers[1]**2*x0,
                                 multipliers[2]**2*(3*z0*z0-1)/2))
            data = {}
            for i, name in enumerate(NAMES):
                denominator = np.max(np.abs(analytic[i]))
                data[name] = {
                    "H_twice_max_error_over_exact_peak": float(np.max(np.abs(twice[i]-analytic[i]))/denominator),
                    "G_direct_max_error_over_exact_peak": float(np.max(np.abs(direct[i]-analytic[i]))/denominator),
                    "H_twice_minus_G_max_over_exact_peak": float(np.max(np.abs(twice[i]-direct[i]))/denominator),
                }
            results.append({"n": n, "grid_shape": [nlat, nlon], "quadrature": kind,
                            "H_twice_seconds": h_time, "G_three_rings_seconds": g_time, "errors": data})
            print(json.dumps(results[-1]), flush=True)
    with (OUTPUT / "results.json").open("x") as stream:
        json.dump({"host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
                   "G_support_km": 4000., "H_support_km": 2000.,
                   "origin_latitudes_degrees": np.rad2deg(ORIGINS).tolist(),
                   "analytic_H_multipliers": multipliers.tolist(),
                   "caveats": "synthetic smooth scalar harmonics; H full shell; G three rings; "
                              "trapezoidal longitude in both schemes; no field regridding",
                   "results": results, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
