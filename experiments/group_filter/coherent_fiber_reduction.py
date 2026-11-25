"""Reduce the per-pair fiber work of coherent group-angle binning.

Mechanism 1 bins pairs linearly in angular distance (as endpoint binning does) and then
expands the per-distance summaries into rotation-angle bins with a precomputed table of
fiber integrals of the hat functions. The table depends on the support, the bin counts and
the quadrature but not on the scale, the origin or the field.

Mechanism 2 interpolates the central weight piecewise-linearly in z = cos(phi/2)**2 instead
of phi. With a = (1+mu)/2 and c = (1+cos psi)/2 one has z = a*c, so the fiber integrals of
the hat functions times (1, cos psi, cos 2 psi) are closed-form in arcsin(sqrt(c)). This is a
different (but still nonnegative, single central) kernel family: its error against the exact
moments is reported, not assumed.
"""

import argparse
import json
import math
import os
import resource
import socket
import time
from pathlib import Path

import numpy as np
import xarray as xr
from numba import njit
from numpy.polynomial.legendre import leggauss

from group_angle_binning import accumulate_segmented_group_bins, central_mollifier_nodes, integrate_group_angle_bins
from lossett.calc.compute_group_increments import _solid_angles
from lossett.calc.group_increments import latlon_shell_fields
from lossett.filtering.group_kernels import _mollifier_spectral_coefficients, mollifier_fiber_moments
from multiscale_group_prototype import _coefficients, binned_coefficients, exact_multiscale

ROOT = Path("/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry")
DATA = Path("/gws/ssde/j25b/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/"
            "DYAMOND_SUMMER/benchmark_antialiased_20160801T00")
RADIUS = 6382758.772639869
NAMES = ("horizontal", "structure", "vertical_flux", "mixed_x", "mixed_y", "mixed_z", "mass")
SPIN0 = [0, 3, 5, 7, 9, 11, 13]
SPIN1 = [1, 4, 6, 8, 10, 12]
SPIN2 = [2]


@njit(cache=True)
def fiber_hat_table(thetas, support_angle, bins, nodes, quadrature):
    """Fiber integrals of rotation-angle hats for a unit pair at each distance, shape (3, M, bins+1)."""
    table = np.zeros((3, len(thetas), bins+1))
    angle_step = support_angle/bins
    for j in range(len(thetas)):
        theta = thetas[j]
        mu = math.cos(theta)
        first = min(int(theta/angle_step), bins-1)
        for b in range(first, bins):
            low = max(b*angle_step, theta)
            high = (b+1)*angle_step
            c0 = min(1., max(-1., 2*(1+math.cos(low))/(1+mu)-1))
            c1 = min(1., max(-1., 2*(1+math.cos(high))/(1+mu)-1))
            psi0, psi1 = math.acos(c0), math.acos(c1)
            width = psi1-psi0
            for q in range(len(nodes)):
                psi = psi0+width*(nodes[q]+1)/2
                spin1 = math.cos(psi)
                phi = math.acos(min(1., max(-1., (1+mu)*(1+spin1)/2-1)))
                fraction = min(1., max(0., phi/angle_step-b))
                factor = quadrature[q]*width/(8*math.pi**2)
                spin2 = 2*spin1*spin1-1
                for spin, value in ((0, 1.), (1, spin1), (2, spin2)):
                    table[spin, j, b] += factor*value*(1-fraction)
                    table[spin, j, b+1] += factor*value*fraction
    return table


def expand_distance_bins(summaries, table):
    """Map distance-bin summaries (M, 14) to rotation-angle summaries (bins+1, 14) via the fiber table."""
    out = np.empty((table.shape[2], 14))
    for channels, spin in ((SPIN0, 0), (SPIN1, 1), (SPIN2, 2)):
        out[:, channels] = table[spin].T @ summaries[:, channels]
    return out


@njit(inline="always")
def _sin_power_integrals(u):
    s2, s4, s6 = math.sin(2*u), math.sin(4*u), math.sin(6*u)
    return (u, u/2-s2/4, 3*u/8-s2/4+s4/32, 5*u/16-15*s2/64+3*s4/64-s6/192)


@njit(cache=True)
def z_hat_fibers(mu, znodes, out):
    """Full-circle fiber integrals of z-hats times (1, cos psi, cos 2 psi), divided by 8 pi^2.

    z = a*c with a = (1+mu)/2 and c = cos(psi/2)**2; out has shape (3, len(znodes)).
    """
    out[:] = 0.
    bins = len(znodes)-1
    a = (1+mu)/2
    scale = 4/(8*math.pi**2)
    for k in range(bins):
        z_upper, z_lower = znodes[k], znodes[k+1]
        if z_lower >= a:
            continue
        delta = z_upper-z_lower
        f_high = _sin_power_integrals(math.atan2(math.sqrt(min(z_upper, a)), math.sqrt(max(a-z_upper, 0.))))
        f_low = _sin_power_integrals(math.atan2(math.sqrt(z_lower), math.sqrt(a-z_lower)))
        m0 = f_high[0]-f_low[0]
        m1 = f_high[1]-f_low[1]
        m2 = f_high[2]-f_low[2]
        m3 = f_high[3]-f_low[3]
        out[0, k] += scale*(a*m1-z_lower*m0)/delta
        out[0, k+1] += scale*(z_upper*m0-a*m1)/delta
        out[1, k] += scale*(2*a*m2-(2*z_lower+a)*m1+z_lower*m0)/delta
        out[1, k+1] += scale*(-2*a*m2+(2*z_upper+a)*m1-z_upper*m0)/delta
        out[2, k] += scale*(8*a*m3-(8*z_lower+8*a)*m2+(8*z_lower+a)*m1-z_lower*m0)/delta
        out[2, k+1] += scale*(-8*a*m3+(8*z_upper+8*a)*m2-(8*z_upper+a)*m1+z_upper*m0)/delta


@njit(cache=True)
def accumulate_z_analytic(x, u, directions, velocity, grad, gradw, weights, curvature, indices,
                          support_angle, znodes):
    """Bin pairs against hats linear in z=cos(phi/2)**2, with closed-form fiber integrals."""
    summaries = np.zeros((len(znodes), 14))
    fibers = np.zeros((3, len(znodes)))
    for target in indices:
        y = directions[target]
        mu = min(1., max(-1., x[0]*y[0]+x[1]*y[1]+x[2]*y[2]))
        if math.acos(mu) > support_angle:
            continue
        c = _coefficients(x, u, y, velocity[target], grad[target], gradw[target], curvature)
        z_hat_fibers(mu, znodes, fibers)
        for channel in range(14):
            spin = 1 if (channel == 1 or channel == 4 or channel == 6 or channel == 8
                         or channel == 10 or channel == 12) else 2 if channel == 2 else 0
            factor = weights[target]*c[channel]
            for k in range(len(znodes)):
                summaries[k, channel] += factor*fibers[spin, k]
    return summaries


def z_kernel_nodes(support_angle, bins, scales, degree=2048):
    """Nonnegative central G values at uniform phi nodes, normalized for hats linear in z."""
    angles = np.linspace(0, support_angle, bins+1)
    znodes = np.cos(angles/2)**2
    nodes, gauss_weights = leggauss(16)
    weights = np.zeros((len(scales), bins+1))
    masses = []
    for i, scale in enumerate(scales):
        coeff = _mollifier_spectral_coefficients(float(scale), degree)
        summed = np.cumsum(coeff[::-1])[::-1]
        cheb = np.r_[summed[0], 2*summed[1:]]
        inside = angles < 4*scale
        values = np.polynomial.chebyshev.chebval(np.cos(angles[inside]), cheb)
        peak = float(np.max(values))
        if float(np.min(values)) < -1e-10*peak:
            raise ValueError("group-kernel spectral negativity exceeds tolerance")
        weights[i, inside] = np.maximum(values, 0)
        mass = 0.
        for k in range(bins):
            z_low, z_high = znodes[k+1], znodes[k]
            z = z_low+(z_high-z_low)*(nodes+1)/2
            fraction = (z-z_low)/(z_high-z_low)
            interpolated = weights[i, k+1]*(1-fraction)+weights[i, k]*fraction
            mass += np.sum(interpolated*2/np.pi*np.sqrt((1-z)/z)*gauss_weights*(z_high-z_low)/2)
        weights[i] /= mass
        masses.append(float(mass))
    return znodes, weights, masses


def best_of(function, repeats=3):
    best, result = float("inf"), None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter()-start)
    return best, result


def peak_errors(result, reference):
    peaks = np.maximum(np.max(np.abs(reference), axis=0), np.finfo(float).tiny)
    return dict(zip(NAMES, (np.max(np.abs(result-reference), axis=0)/peaks).tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=("n320", "n640"), required=True)
    parser.add_argument("--tag", default="20261125")
    args = parser.parse_args()
    out = ROOT/f"coherent_fiber_reduction_{args.grid}_{args.tag}"
    out.mkdir(exist_ok=False)
    with xr.open_dataset(DATA/f"filtered_{args.grid}.nc", decode_timedelta=False) as ds:
        field = ds.isel(time=0).sel(pressure=200.).load()
    lat, lon = np.deg2rad(field.latitude.values), np.deg2rad(field.longitude.values)
    weights = _solid_angles(lat, lon).ravel()
    direction, velocity, grad, gradw = latlon_shell_fields(
        lat, lon, *(field[name].values for name in ("u", "v", "w")), radius=RADIUS)
    direction, velocity = direction.reshape(-1, 3), velocity.reshape(-1, 3)
    grad, gradw = grad.reshape(-1, 3, 3), gradw.reshape(-1, 3)
    supports_km = np.array([2000., 4000.])
    scales = supports_km*1000/(4*RADIUS)
    support = 4000e3/RADIUS
    gauss_nodes, gauss_weights = leggauss(8)
    configurations = [(256, 2048), (512, 2048), (512, 4096)]   # (rotation bins, distance bins)
    setup = {}
    phi_kernels, z_kernels, hat_tables = {}, {}, {}
    for bins in sorted({b for b, _ in configurations}):
        start = time.perf_counter()
        phi_kernels[bins] = central_mollifier_nodes(support, bins, scales)[0]
        phi_seconds = time.perf_counter()-start
        start = time.perf_counter()
        z_kernels[bins] = z_kernel_nodes(support, bins, scales)
        setup[f"kernels_{bins}"] = {"phi_seconds": phi_seconds, "z_seconds": time.perf_counter()-start,
                                    "z_masses": z_kernels[bins][2]}
    for bins, distance_bins in configurations:
        thetas = np.linspace(0, support, distance_bins+1)
        fiber_hat_table(thetas[:2], support, 2, gauss_nodes, gauss_weights)
        start = time.perf_counter()
        hat_tables[(bins, distance_bins)] = fiber_hat_table(thetas, support, bins, gauss_nodes, gauss_weights)
        setup[f"hat_table_bins{bins}_distance{distance_bins}"] = {
            "seconds": time.perf_counter()-start, "bytes": hat_tables[(bins, distance_bins)].nbytes}
    x, u = direction[0], velocity[0]
    one = np.array([0], dtype=np.int64)
    accumulate_z_analytic(x, u, direction, velocity, grad, gradw, weights, 1/RADIUS, one, support,
                          z_kernels[512][0][:3])
    accumulate_segmented_group_bins(x, u, direction, velocity, grad, gradw, weights, 1/RADIUS, one, support,
                                    2, gauss_nodes, gauss_weights)
    binned_coefficients(x, u, direction, velocity, grad, gradw, weights, 1/RADIUS, one, support, 2)
    exact_multiscale(x, u, direction, velocity, grad, gradw, weights, np.ones((1, 1, 3)), 1/RADIUS, one)
    records = []
    for row, col in (((len(lat)-1)//4, len(lon)//3), ((len(lat)-1)//2, len(lon)//2)):
        index = row*len(lon)+col
        x, u = direction[index], velocity[index]
        mu_all = np.clip(direction@x, -1, 1)
        active = np.flatnonzero(mu_all >= np.cos(support))
        mu = mu_all[active]
        exact_moments = np.zeros((len(scales), len(active), 3))
        for s, scale in enumerate(scales):
            mask = mu >= np.cos(4*scale)
            exact_moments[s, mask] = np.asarray(mollifier_fiber_moments(mu[mask], scale)).T
        _, exact = best_of(lambda: exact_multiscale(x, u, direction, velocity, grad, gradw, weights,
                                                    exact_moments, 1/RADIUS, active), 1)
        exact_seconds, _ = best_of(lambda: exact_multiscale(x, u, direction, velocity, grad, gradw, weights,
                                                            exact_moments, 1/RADIUS, active))
        entry = {"row": row, "col": col, "latitude_degrees": float(np.rad2deg(lat[row])),
                 "active_pairs": len(active), "exact_accumulate_seconds": exact_seconds, "results": []}
        for bins, distance_bins in configurations:
            kernel_phi, (znodes, kernel_z, _) = phi_kernels[bins], z_kernels[bins]
            table = hat_tables[(bins, distance_bins)]
            bin_seconds, distance_summaries = best_of(lambda: binned_coefficients(
                x, u, direction, velocity, grad, gradw, weights, 1/RADIUS, active, support, distance_bins))
            expand_seconds, expanded = best_of(lambda: expand_distance_bins(distance_summaries, table))
            integrate_seconds, mu_binned = best_of(lambda: integrate_group_angle_bins(expanded, kernel_phi))
            z_seconds, z_summaries = best_of(lambda: accumulate_z_analytic(
                x, u, direction, velocity, grad, gradw, weights, 1/RADIUS, active, support, znodes))
            z_integrate_seconds, z_result = best_of(lambda: integrate_group_angle_bins(z_summaries, kernel_z))
            result = {
                "bins": bins, "distance_bins": distance_bins,
                "mu_binned_bin_seconds": bin_seconds, "mu_binned_expand_seconds": expand_seconds,
                "mu_binned_integrate_seconds": integrate_seconds,
                "mu_binned_peak_normalized_errors": peak_errors(mu_binned, exact),
                "z_analytic_bin_seconds": z_seconds, "z_analytic_integrate_seconds": z_integrate_seconds,
                "z_analytic_peak_normalized_errors": peak_errors(z_result, exact),
            }
            if bins == 512 and distance_bins == 2048:
                seg_seconds, segmented = best_of(lambda: accumulate_segmented_group_bins(
                    x, u, direction, velocity, grad, gradw, weights, 1/RADIUS, active, support, bins,
                    gauss_nodes, gauss_weights), 1)
                segmented_result = integrate_group_angle_bins(segmented, kernel_phi)
                result.update({
                    "segmented_bin_seconds": seg_seconds,
                    "segmented_peak_normalized_errors": peak_errors(segmented_result, exact),
                    "mu_binned_vs_segmented": peak_errors(mu_binned, segmented_result),
                })
            entry["results"].append(result)
        records.append(entry)
        print(f"Finished ({row}, {col})", flush=True)
    with (out/"results.json").open("x") as stream:
        json.dump({"host": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
                   "grid_shape": [len(lat), len(lon)], "support_G_km": supports_km.tolist(),
                   "setup": setup, "records": records,
                   "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "caveats": "two origins, two scales, best-of-three warm timings on a shared node; "
                              "exact reference uses the spectral builder at all active pairs"},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
