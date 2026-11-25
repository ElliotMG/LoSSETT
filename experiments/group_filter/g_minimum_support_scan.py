"""Scalar G and H accuracy versus kernel support relative to the grid spacing.

For a normalized zonal scalar kernel K, the filter of a degree-l harmonic is lambda_l * Y_l, so the
grid error of sum_j w_j K(x0.x_j) Y_l(x_j) against lambda_l Y_l(x0) measures quadrature and
sampling error for that kernel and grid alone. The matched G = H*H scalar kernel is computed by
direct quadrature of the spherical convolution (no spectral truncation), so very small supports
are reachable. Vector and tensor moments are NOT tested here.
"""

import argparse
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.polynomial import legendre as L
from scipy.interpolate import CubicSpline

from h_twice_quadrature_study import clenshaw_curtis_weights
from lossett.calc.compute_group_increments import _solid_angles
from lossett.filtering.group_kernels import mollifier_fiber_moments

ROOT = Path("/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry")
RADIUS = 6382758.772639869
DEGREES = (0, 1, 2, 4, 8)
AXIS = np.array([.3, .5, .81])
AXIS /= np.linalg.norm(AXIS)


def h_profile(theta, s):
    ratio = (theta/(2*s))**2
    out = np.zeros_like(theta)
    inside = ratio < 1
    out[inside] = np.exp(-1/(1-ratio[inside]))
    return out


def make_kernels(s, ntheta=192, nphi=384, nd=1601):
    """Return callables H(d) and G(d), both normalized over the sphere."""
    tn, tw = leggauss(ntheta)
    theta = s*(tn+1)
    tweight = s*tw*np.sin(theta)
    norm = 2*np.pi*np.sum(tweight*h_profile(theta, s))
    pn, pw = leggauss(nphi)
    phi = np.pi*(pn+1)/2
    pweight = np.pi/2*pw
    d_nodes = 4*s*(1-np.cos(np.pi*np.arange(nd)/(nd-1)))/2   # clustered, includes 0 and 4s
    values = np.zeros(nd)
    for i, d in enumerate(d_nodes):
        cos_theta2 = (np.cos(d)*np.cos(theta)[:, None]
                      + np.sin(d)*np.sin(theta)[:, None]*np.cos(phi)[None, :])
        theta2 = np.arccos(np.clip(cos_theta2, -1, 1))
        inner = 2*np.sum(pweight*h_profile(theta2.ravel(), s).reshape(theta2.shape), axis=1)
        values[i] = np.sum(tweight*h_profile(theta, s)*inner)/norm**2
    values[-1] = 0.
    spline = CubicSpline(d_nodes, values)

    def h_kernel(d):
        return h_profile(np.asarray(d, float), s)/norm

    def g_kernel(d):
        d = np.asarray(d, float)
        out = np.zeros_like(d)
        inside = d < 4*s
        out[inside] = spline(d[inside])
        return out

    return h_kernel, g_kernel


def multipliers(kernel, support):
    nodes, weights = leggauss(1200)
    d = support*(nodes+1)/2
    kd = kernel(d)*np.sin(d)*support/2*weights
    cos_d = np.cos(d)
    return {l: float(2*np.pi*np.sum(kd*L.legval(cos_d, [0]*l+[1]))) for l in DEGREES}


def harmonic(l, x):
    return L.legval(x@AXIS, [0]*l+[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", type=int, nargs="+", default=(320, 640, 1280, 2560))
    parser.add_argument("--radii-in-spacings", type=float, nargs="+",
                        default=(1., 1.5, 2., 3., 4., 5., 6., 8., 10., 15.))
    args = parser.parse_args()
    out = ROOT/"g_minimum_support_scan_20261124"
    out.mkdir(exist_ok=False)
    results = []
    validation = {}
    for km in (4000., 1000.):
        s = km*1000/(4*RADIUS)
        _, g_default = make_kernels(s)
        _, g_fine = make_kernels(s, ntheta=288, nphi=576, nd=2401)
        d = np.linspace(0, 4*s, 41)[:-1]
        entry = {"quadrature_refinement_max_difference": float(np.max(np.abs(g_default(d)-g_fine(d)))
                                                                  / np.max(g_fine(d)))}
        try:
            spectral = np.asarray(mollifier_fiber_moments(np.cos(d), s))[0]
            entry["spectral_max_relative_difference"] = float(np.max(np.abs(g_default(d)-spectral))
                                                              / np.max(spectral))
        except ValueError as error:
            entry["spectral_failed"] = str(error)
        validation[f"G_radius_{km:g}_km"] = entry
        print("validation", km, entry, flush=True)
    for grid in args.grids:
        nlat, nlon = 3*grid//2+1, 2*grid
        lat = np.linspace(-np.pi/2, np.pi/2, nlat)
        lon = 2*np.pi*np.arange(nlon)/nlon
        dlat, dlon = np.pi/(nlat-1), 2*np.pi/nlon
        spacing = max(dlat, dlon*1.)*RADIUS            # largest spacing (equator) in metres
        cell = _solid_angles(lat, lon)
        cc = np.broadcast_to(clenshaw_curtis_weights(nlat-1)[:, None]*2*np.pi/nlon, (nlat, nlon))
        for ratio in args.radii_in_spacings:
            g_radius_m = ratio*spacing
            s = g_radius_m/(4*RADIUS)
            if 4*s > 1.0:
                continue
            start = time.perf_counter()
            h_kernel, g_kernel = make_kernels(s)
            build_seconds = time.perf_counter()-start
            kernels = {"G": (g_kernel, 4*s), "H": (h_kernel, 2*s)}
            lam = {name: multipliers(k, support) for name, (k, support) in kernels.items()}
            for row in sorted({nlat//2, int(nlat*0.5+nlat*30/180), int(nlat*0.5+nlat*60/180),
                               int(nlat*0.5+nlat*85/180), nlat-1}):
                for phase_lat in (0., .5):
                    for phase_lon in (0., .5):
                        phi0 = lat[row]+phase_lat*dlat
                        if phi0 > np.pi/2:
                            continue
                        lam0 = phase_lon*dlon
                        x0 = np.array([np.cos(phi0)*np.cos(lam0), np.cos(phi0)*np.sin(lam0), np.sin(phi0)])
                        for name, (kernel, support) in kernels.items():
                            rows = np.flatnonzero(np.abs(lat-phi0) <= support+1e-12)
                            xyz = np.stack((np.cos(lat[rows])[:, None]*np.cos(lon)[None, :],
                                            np.cos(lat[rows])[:, None]*np.sin(lon)[None, :],
                                            np.broadcast_to(np.sin(lat[rows])[:, None], (len(rows), nlon))), axis=-1)
                            mu = np.clip(xyz@x0, -1, 1)
                            d = np.arccos(mu)
                            kvalues = kernel(d)
                            for quad_name, weights in (("cell", cell), ("clenshaw_curtis", cc)):
                                w = weights[rows]*kvalues
                                total = float(w.sum())
                                for l in DEGREES:
                                    exact = lam[name][l]*harmonic(l, x0[None, :])[0]
                                    raw = float(np.sum(w*harmonic(l, xyz)))
                                    results.append({
                                        "grid": grid, "kernel": name, "quadrature": quad_name,
                                        "support_radius_km": support*RADIUS/1000,
                                        "diameter_in_largest_spacings": 2*support*RADIUS/spacing,
                                        "G_radius_in_largest_spacings": ratio,
                                        "row": row, "latitude_degrees": float(np.rad2deg(phi0)),
                                        "origin_phase_lat_lon": [phase_lat, phase_lon],
                                        "degree": l, "kernel_sum": total,
                                        "abs_error": raw-exact,
                                        "abs_error_renormalized": raw/total-exact if total > 0 else None,
                                        "exact_multiplier": lam[name][l],
                                    })
            print(f"Finished n{grid} G radius {ratio:g} spacings (build {build_seconds:.1f}s)", flush=True)
    with (out/"results.json").open("x") as stream:
        json.dump({"host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
                   "validation": validation, "results": results,
                   "caveats": "scalar kernels only; smooth harmonic test fields; origins on or between "
                              "grid nodes; errors are absolute for unit-amplitude fields"},
                  stream, indent=1)
        stream.write("\n")


if __name__ == "__main__":
    main()
