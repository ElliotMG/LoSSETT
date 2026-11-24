"""Probe scalar H quadrature when filter support approaches the grid spacing."""

import json
import os
import socket
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from h_twice_quadrature_study import clenshaw_curtis_weights
from h_twice_rings_benchmark import RADIUS, h_scalar
from lossett.calc.compute_group_increments import _solid_angles
from lossett.filtering.group_kernels import mollifier_les_fiber_moments


ROOT = Path("/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/quadrature_narrow_harmonics_20261124")


def main():
    ROOT.mkdir(exist_ok=False)
    roots, root_weights = leggauss(256)
    origins = np.deg2rad([0., 45., 80., 90.])
    results = []
    for grid in (320, 640):
        nlat, nlon = 3*grid//2+1, 2*grid
        lat = np.linspace(-np.pi/2, np.pi/2, nlat)
        lon = 2*np.pi*np.arange(nlon)/nlon
        z = np.broadcast_to(np.sin(lat)[:, None], (nlat, nlon))
        xx = np.cos(lat)[:, None]*np.cos(lon)[None, :]
        inputs = np.stack((np.ones_like(z), z, xx, (3*z*z-1)/2))
        areas = {
            "cell_midpoint": _solid_angles(lat, lon),
            "clenshaw_curtis": np.broadcast_to(
                clenshaw_curtis_weights(nlat-1)[:, None]*2*np.pi/nlon, (nlat, nlon)),
        }
        for support_km in (25., 50., 100., 200., 500., 2000.):
            scale = support_km*1000/(2*RADIUS)
            center = float(mollifier_les_fiber_moments(np.array([1.]), scale)[0][0])
            lower = np.cos(2*scale)
            q = lower+(1-lower)*(roots+1)/2
            shape = np.exp(-1/(1-(np.arccos(q)/(2*scale))**2))
            weighted = root_weights*shape
            m1 = float(weighted@q/weighted.sum())
            m2 = float(weighted@((3*q*q-1)/2)/weighted.sum())
            expected_z = np.broadcast_to(np.sin(origins)[:, None], (len(origins), nlon))
            exact = np.stack((
                np.ones_like(expected_z), m1*expected_z,
                m1*np.cos(origins)[:, None]*np.cos(lon)[None, :],
                m2*(3*expected_z**2-1)/2,
            ))
            for name, weights in areas.items():
                spectra = np.fft.rfft(inputs*weights[None, :, :], axis=2)
                result = np.empty_like(exact)
                for j, origin in enumerate(origins):
                    rows = np.flatnonzero(np.abs(lat-origin) <= 2*scale + 1e-14)
                    mu = np.clip(np.sin(origin)*np.sin(lat[rows, None])
                                 + np.cos(origin)*np.cos(lat[rows, None])*np.cos(lon), -1, 1)
                    response = np.fft.rfft(h_scalar(mu, scale, center), axis=-1)
                    result[:, j] = np.fft.irfft(
                        np.sum(np.conj(response)[None, :, :]*spectra[:, rows, :], axis=1),
                        n=nlon, axis=-1,
                    )
                maximum = np.max(np.abs(result-exact), axis=(1, 2))
                peaks = np.max(np.abs(exact), axis=(1, 2))
                results.append({
                    "grid": grid, "H_support_km": support_km, "quadrature": name,
                    "max_error_over_exact_peak": dict(zip(
                        ("constant", "z_degree1", "x_degree1", "zonal_degree2"), (maximum/peaks).tolist())),
                    "constant_error_by_latitude": np.max(np.abs(result[0]-1), axis=1).tolist(),
                })
            print(f"Finished n{grid} H support {support_km:g} km", flush=True)
    with (ROOT / "results.json").open("x") as stream:
        json.dump({"host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
                   "origin_latitudes_degrees": np.rad2deg(origins).tolist(),
                   "results": results, "caveats": "synthetic smooth scalar harmonics, one H pass; "
                                                  "not a G transfer or vector/tensor commutator test"},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
