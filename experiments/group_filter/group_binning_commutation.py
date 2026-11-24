"""Commutation pilot for exact and two approximate group-moment filters."""

import json
import os
import resource
import socket
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from group_angle_binning import central_mollifier_nodes, group_angle_moments, segmented_group_angle_moments
from lossett.calc.group_increments import _transport
from lossett.filtering.group_kernels import mollifier_fiber_moments


ROOT = Path("/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry")
OUTPUT = ROOT / "group_binning_commutation_refined_20261124"
RADIUS = 6382758.772639869
SCALE = 1e6/RADIUS
SUPPORT = 4*SCALE


def grid(n):
    z, w = leggauss(n)
    phi = 2*np.pi*np.arange(2*n)/(2*n)
    z2, az = np.meshgrid(z, phi, indexing="ij")
    points = np.stack((np.sqrt(1-z2*z2)*np.cos(az),
                       np.sqrt(1-z2*z2)*np.sin(az), z2), axis=-1).reshape(-1, 3)
    weights = np.broadcast_to(w[:, None]*2*np.pi/(2*n), (n, 2*n)).ravel()
    return points, weights


def main():
    OUTPUT.mkdir(exist_ok=False)
    longitude, latitude = np.deg2rad(25.), np.deg2rad(35.)
    x = np.array([np.cos(latitude)*np.cos(longitude), np.cos(latitude)*np.sin(longitude),
                  np.sin(latitude)])
    e1 = np.array([-np.sin(longitude), np.cos(longitude), 0.])
    e2 = np.cross(x, e1)
    a = np.array([.45, -.50, .74])
    a /= np.linalg.norm(a)
    settings = {}
    for bins in (128, 512):
        mu = np.cos(np.linspace(0, SUPPORT, bins+1))
        endpoint = np.asarray(mollifier_fiber_moments(mu, SCALE)).T
        group, clip, masses, elapsed = central_mollifier_nodes(SUPPORT, bins, [SCALE])
        settings[bins] = (endpoint, group[0], {"group_clipping": clip, "group_mass_before_normalization": masses,
                                                "group_table_seconds": elapsed})
    results = []
    for n in (48, 96, 192):
        points, weights = grid(n)
        scalar = points@a
        gradient = a[None, :] - scalar[:, None]*points
        rotational = np.cross(points, gradient)
        for mode, bins, fiber in (
            ("reference", 0, 0), ("endpoint", 128, 0), ("endpoint", 512, 0),
            ("group_angle", 128, 512), ("group_angle", 512, 512),
            ("group_angle", 512, 2048),
            ("group_segmented", 128, 8), ("group_segmented", 512, 8),
            ("group_segmented", 512, 16),
        ):
            if n == 192 and not (
                mode == "reference" or (mode == "endpoint" and bins == 512)
                or (mode == "group_segmented" and bins == 512 and fiber == 8)
            ):
                continue
            start = time.perf_counter()

            def evaluate(origin):
                mu_all = np.clip(points@origin, -1, 1)
                inside = mu_all >= np.cos(SUPPORT)
                mu, y, w = mu_all[inside], points[inside], weights[inside]
                if mode == "reference":
                    moments = np.asarray(mollifier_fiber_moments(mu, SCALE)).T
                elif mode == "endpoint":
                    distance = np.arccos(mu)
                    nodes = np.linspace(0, SUPPORT, bins+1)
                    moments = np.column_stack([
                        np.interp(distance, nodes, settings[bins][0][:, j]) for j in range(3)
                    ])
                elif mode == "group_angle":
                    moments = group_angle_moments(mu, SUPPORT, settings[bins][1], fiber)
                else:
                    moments = segmented_group_angle_moments(mu, SUPPORT, settings[bins][1], fiber)
                grad_transported = _transport(gradient[inside], y, np.broadcast_to(origin, y.shape))
                rot_transported = _transport(rotational[inside], y, np.broadcast_to(origin, y.shape))
                scalar_out = np.dot(w*moments[:, 0], scalar[inside])
                grad_out = np.einsum("n,ni->i", w*moments[:, 1], grad_transported)
                rot_out = np.einsum("n,ni->i", w*moments[:, 1], rot_transported)
                return scalar_out, grad_out, rot_out

            center = evaluate(x)
            for step in (2e-4, 1e-4):
                displaced = []
                for e in (e1, e2):
                    plus = evaluate(np.cos(step)*x+np.sin(step)*e)
                    minus = evaluate(np.cos(step)*x-np.sin(step)*e)
                    displaced.append((plus, minus))
                grad_filtered = sum(
                    (pair[0][0]-pair[1][0])/(2*step)*e
                    for pair, e in zip(displaced, (e1, e2))
                )
                div_filtered = sum(
                    np.dot(pair[0][1]-pair[1][1], e)/(2*step)
                    for pair, e in zip(displaced, (e1, e2))
                )
                curl_filtered = (
                    np.dot(displaced[0][0][2]-displaced[0][1][2], e2)
                    - np.dot(displaced[1][0][2]-displaced[1][1][2], e1)
                )/(2*step)
                target_div_curl = -2*center[0]
                record = {
                    "n_gauss": n, "nlon": 2*n, "method": mode, "bins": bins,
                    "fiber_points_or_segment_order": fiber, "difference_step_radians": step,
                    "gradient_absolute_commutator": float(np.linalg.norm(center[1]-grad_filtered)),
                    "gradient_reference_norm": float(np.linalg.norm(center[1])),
                    "divergence_absolute_commutator": float(abs(div_filtered-target_div_curl)),
                    "curl_absolute_commutator": float(abs(curl_filtered-target_div_curl)),
                    "divergence_curl_reference": float(abs(target_div_curl)),
                }
                results.append(record)
            print(f"Finished quadrature {n}, {mode}, bins={bins}, fiber={fiber}; "
                  f"{time.perf_counter()-start:.2f}s", flush=True)
    with (OUTPUT / "results.json").open("x") as stream:
        json.dump({"host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
                   "field": "degree-one scalar, its gradient, and its 90-degree rotation",
                   "G_support_km": 4000., "origin": x.tolist(), "group_tables": {k: v[2] for k, v in settings.items()},
                   "results": results, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "caveats": "continuous filtered outputs approximated by endpoint quadrature and finite "
                              "differences; group-angle method also uses discrete fiber quadrature; "
                              "not a discrete-grid derivative commutator or nonlinear transfer test"},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
