"""Vector and tensor G-moment accuracy versus support in grid spacings, plus renormalization and commutation.

Extends g_minimum_support_scan.py (scalar only). For the matched group kernel G the filtered fields are

    scalar  S(x0) = sum_j w_j g(mu_j) f(y_j)
    vector  V(x0) = sum_j w_j h(mu_j) Q_j v(y_j)          (spin 1, cos(psi) moment)
    tensor  T(x0) = sum_j w_j k(mu_j) Q_j T(y_j) Q_j^T    (spin 2, cos(2 psi) moment)

where Q_j parallel-transports tangent vectors from y_j to x0 along the great circle. The test fields
are the degree-l harmonic P_l(x.A), its surface gradient and its trace-free surface Hessian. The
reference for each channel is the same continuous integral done in polar coordinates about x0
(Gauss in distance, periodic trapezoid in azimuth), so only grid quadrature/sampling error remains.
The moments g, h, k come from mollifier_fiber_moments_quadrature, so supports below the spectral
builder's reach are included.

Commutation is measured directly: the discrete vector filter of grad f minus the surface gradient
of the discrete scalar filter of f (fourth-order centred difference over origins). It is reported
for the raw weights and for weights divided by sum_j w_j g(mu_j) at each origin, which answers
whether per-origin renormalization preserves commutation.
"""

import argparse
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
from numpy.polynomial import legendre as L
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import CubicSpline

from h_twice_quadrature_study import clenshaw_curtis_weights
from lossett.calc.compute_group_increments import _solid_angles
from lossett.filtering.group_kernels import mollifier_fiber_moments_quadrature

ROOT = Path("/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry")
RADIUS = 6382758.772639869
A = np.array([.3, .5, .81])
A /= np.linalg.norm(A)
EYE = np.eye(3)


def fields(l, x):
    """Scalar, surface gradient and trace-free surface Hessian of P_l(x.A) at unit vectors x (..., 3)."""
    c = [0]*l+[1]
    t = x@A
    d1, d2 = L.legval(t, L.legder(c)), L.legval(t, L.legder(c, 2))
    tangent_A = A-t[..., None]*x
    projector = EYE-x[..., :, None]*x[..., None, :]
    grad = d1[..., None]*tangent_A
    hess = (d2[..., None, None]*tangent_A[..., :, None]*tangent_A[..., None, :]
            - (d1*t)[..., None, None]*projector)
    trace = d2*(1-t**2)-2*d1*t
    return L.legval(t, c), grad, hess-0.5*trace[..., None, None]*projector


def transport(y, x0):
    """Rotations Q_j taking y_j to x0 along the great circle (identity where y = x0)."""
    mu = np.clip(y@x0, -1, 1)
    s = np.sqrt(np.maximum(1-mu**2, 0))
    safe = np.where(s > 1e-12, s, 1.)
    e = (x0[None, :]-mu[:, None]*y)/safe[:, None]
    e_new = (mu[:, None]*x0[None, :]-y)/safe[:, None]
    n = np.cross(y, e)
    q = e_new[:, :, None]*e[:, None, :]+n[:, :, None]*n[:, None, :]
    degenerate = s <= 1e-12
    q[degenerate] = EYE-x0[:, None]*x0[None, :]
    return q, mu


def moment_kernels(scale, nodes=2001):
    d = 4*scale*(1-np.cos(np.pi*np.arange(nodes)/(nodes-1)))/2
    values = np.asarray(mollifier_fiber_moments_quadrature(np.cos(d), scale))
    values[:, -1] = 0.
    return [CubicSpline(d, values[i]) for i in range(3)], 4*scale


def evaluate(x0, y, w, splines, support, degree):
    """Raw weighted sums (scalar, vector, tensor) of the degree-l fields."""
    q, mu = transport(y, x0)
    d = np.arccos(mu)
    inside = d < support
    g, h, k = (np.where(inside, s(np.minimum(d, support)), 0.) for s in splines)
    scalar, grad, hess = fields(degree, y)
    vec = np.einsum("nij,nj->ni", q, grad)
    ten = np.einsum("nij,njk,nlk->nil", q, hess, q)
    return (np.sum(w*g*scalar), np.einsum("n,ni->i", w*h, vec), np.einsum("n,nij->ij", w*k, ten),
            float(np.sum(w*g)))


def reference(x0, splines, support, degree, ntheta=300, nphi=256):
    a = np.cross(x0, [0., 0., 1.]) if abs(x0[2]) < .99 else np.cross(x0, [1., 0., 0.])
    a /= np.linalg.norm(a)
    b = np.cross(x0, a)
    nodes, weights = leggauss(ntheta)
    theta = support*(nodes+1)/2
    phi = 2*np.pi*np.arange(nphi)/nphi
    y = (np.cos(theta)[:, None, None]*x0+np.sin(theta)[:, None, None]
         * (np.cos(phi)[None, :, None]*a+np.sin(phi)[None, :, None]*b)).reshape(-1, 3)
    w = np.repeat(weights*support/2*np.sin(theta)*2*np.pi/nphi, nphi)
    return evaluate(x0, y, w, splines, support, degree)


def tangent_frame(x0):
    a = np.cross(x0, A)
    a = a/np.linalg.norm(a) if np.linalg.norm(a) > 1e-8 else np.cross(x0, [1., 0., 0.])
    a /= np.linalg.norm(a)
    return a, np.cross(x0, a)


def scalar_origin_gradient(sample_scalar, x0, step):
    """Fourth-order centred surface gradient of origin-dependent scalar sample_scalar(x)."""
    a, b = tangent_frame(x0)
    out = np.zeros(3)
    for direction in (a, b):
        f = [sample_scalar(x0*np.cos(m*step)+direction*np.sin(m*step)) for m in (-2, -1, 1, 2)]
        out += direction*(8*f[2]-8*f[1]-f[3]+f[0])/(12*step)
    return out


def peaks(degree):
    t = np.linspace(-1, 1, 4001)
    B = np.cross(A, [1., 0., 0.])
    B /= np.linalg.norm(B)
    x = t[:, None]*A+np.sqrt(1-t**2)[:, None]*B
    _, grad, hess = fields(degree, x)
    return 1., max(float(np.max(np.linalg.norm(grad, axis=-1))), 1e-300), \
        max(float(np.max(np.linalg.norm(hess, axis=(-2, -1)))), 1e-300)


def multiplier(spline, support, degree):
    nodes, weights = leggauss(1200)
    d = support*(nodes+1)/2
    return float(2*np.pi*np.sum(spline(d)*np.sin(d)*support/2*weights*L.legval(np.cos(d), [0]*degree+[1])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", type=int, nargs="+", default=(320, 640, 1280, 2560))
    parser.add_argument("--radii-in-spacings", type=float, nargs="+",
                        default=(1., 1.5, 2., 3., 4., 5., 6., 8., 10., 15.))
    parser.add_argument("--tag", default="20261125")
    args = parser.parse_args()
    out = ROOT/f"g_vector_tensor_support_scan_{args.tag}"
    out.mkdir(exist_ok=False)
    scalar_degrees, vector_degrees, tensor_degrees = (0, 1, 2, 4, 8), (1, 2, 4, 8), (2, 4, 8)
    results = []
    for grid in args.grids:
        nlat, nlon = 3*grid//2+1, 2*grid
        lat = np.linspace(-np.pi/2, np.pi/2, nlat)
        lon = 2*np.pi*np.arange(nlon)/nlon
        dlat, dlon = np.pi/(nlat-1), 2*np.pi/nlon
        spacing = max(dlat, dlon)*RADIUS
        cell = _solid_angles(lat, lon)
        cc = np.broadcast_to(clenshaw_curtis_weights(nlat-1)[:, None]*2*np.pi/nlon, (nlat, nlon))
        for ratio in args.radii_in_spacings:
            support_m = ratio*spacing
            scale = support_m/(4*RADIUS)
            if not 0 < scale < np.pi/4:
                continue
            start = time.perf_counter()
            splines, support = moment_kernels(scale)
            build_seconds = time.perf_counter()-start
            for row in sorted({nlat//2, int(nlat*0.5+nlat*30/180), int(nlat*0.5+nlat*60/180),
                               int(nlat*0.5+nlat*85/180), nlat-1}):
                for phase_lat in (0., .5):
                    for phase_lon in (0., .5):
                        phi0 = lat[row]+phase_lat*dlat
                        if phi0 > np.pi/2:
                            continue
                        lam0 = phase_lon*dlon
                        x0 = np.array([np.cos(phi0)*np.cos(lam0), np.cos(phi0)*np.sin(lam0), np.sin(phi0)])
                        for degree in sorted(set(scalar_degrees)):
                            step = 0.02/max(degree, 1)
                            margin = support+4*step+1e-9
                            rows = np.flatnonzero(np.abs(lat-phi0) <= margin)
                            if len(rows) == 0:
                                continue
                            xyz = np.stack((np.cos(lat[rows])[:, None]*np.cos(lon)[None, :],
                                            np.cos(lat[rows])[:, None]*np.sin(lon)[None, :],
                                            np.broadcast_to(np.sin(lat[rows])[:, None], (len(rows), nlon))),
                                           axis=-1).reshape(-1, 3)
                            ref_s, ref_v, ref_t, _ = reference(x0, splines, support, degree)
                            peak_s, peak_v, peak_t = peaks(degree)
                            lam = multiplier(splines[0], support, degree)
                            for quad_name, weights in (("cell", cell), ("clenshaw_curtis", cc)):
                                w = weights[rows].reshape(-1)
                                s, v, t, z = evaluate(x0, xyz, w, splines, support, degree)
                                record = {
                                    "grid": grid, "quadrature": quad_name, "degree": degree,
                                    "G_radius_in_largest_spacings": ratio,
                                    "diameter_in_largest_spacings": 2*ratio,
                                    "support_radius_km": support*RADIUS/1000,
                                    "latitude_degrees": float(np.rad2deg(phi0)), "row": row,
                                    "origin_phase_lat_lon": [phase_lat, phase_lon], "kernel_sum": z,
                                    "scalar_error": abs(s-ref_s)/peak_s,
                                    "scalar_error_renormalized": abs(s/z-ref_s)/peak_s}
                                if degree in vector_degrees:
                                    record["vector_error"] = float(np.linalg.norm(v-ref_v))/peak_v
                                    record["vector_error_renormalized"] = float(np.linalg.norm(v/z-ref_v))/peak_v

                                    def raw_scalar(x):
                                        return evaluate(x, xyz, w, splines, support, degree)[0]

                                    def renorm_scalar(x):
                                        value = evaluate(x, xyz, w, splines, support, degree)
                                        return value[0]/value[3]
                                    grad_raw = scalar_origin_gradient(raw_scalar, x0, step)
                                    grad_renorm = scalar_origin_gradient(renorm_scalar, x0, step)
                                    record["commutator_raw"] = float(np.linalg.norm(v-grad_raw))/peak_v
                                    record["commutator_renormalized"] = float(np.linalg.norm(v/z-grad_renorm))/peak_v
                                    record["continuum_vector_minus_scalar_multiplier"] = float(
                                        np.linalg.norm(ref_v-lam*fields(degree, x0)[1]))/peak_v
                                if degree in tensor_degrees:
                                    record["tensor_error"] = float(np.linalg.norm(t-ref_t))/peak_t
                                    record["tensor_error_renormalized"] = float(np.linalg.norm(t/z-ref_t))/peak_t
                                    record["continuum_tensor_minus_scalar_multiplier"] = float(
                                        np.linalg.norm(ref_t-lam*fields(degree, x0)[2]))/peak_t
                                results.append(record)
            print(f"Finished n{grid} G radius {ratio:g} spacings (build {build_seconds:.1f}s)", flush=True)
    with (out/"results.json").open("x") as stream:
        json.dump({"host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"), "results": results,
                   "caveats": "harmonic test fields; tensor normalised by the sampled peak Frobenius norm; "
                              "commutator uses a fourth-order origin-difference of the discrete scalar filter"},
                  stream, indent=1)
        stream.write("\n")


if __name__ == "__main__":
    main()
