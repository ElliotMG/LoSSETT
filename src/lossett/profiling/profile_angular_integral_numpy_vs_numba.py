#!/usr/bin/env python3

"""
Benchmark LoSSETT angular integration kernels on a real geometry archive
and real velocity field.

Benchmarks:

    angular_integral_by_distance_bin
    angular_integral_unweighted_numba
    angular_integral_weighted_numba
    bin_integrate

using a representative du^3 field generated from the LoSSETT spherical
geometry pipeline.
"""

import numpy as np

from lossett.calc.compute_delta_u_cubed_spherical import (
    load_geometry,
    load_geometry_chunk,
    load_velocity_field,
    GRID_DEFS,
)

from lossett.calc.field_increments import (
    compute_delta_u_cubed_numexpr,
)

from lossett.calc.angular_integration import (
    angular_integral_by_distance_bin,
    angular_integral_unweighted_numba,
    angular_integral_weighted_numba,
    bin_integrate,
)

from lossett.profiling import benchmark


###############################################################################
# USER SETTINGS
###############################################################################

grid = "n640"
date = "20160801"

geom_path = (
    "/work/scratch-pw5/dship/upscale/LoSSETT/"
    "spherical_geometry/"
)

chunk_origin = 16
nbins_fac = 2

repeats = 100


###############################################################################
# LOAD GRID
###############################################################################

print("\nLoading grid...")

lon_step, lat_step = GRID_DEFS[grid]

lons = np.arange(-180.0, 180.0, lon_step)

lats = np.arange(
    -90.0,
    90.0 + lat_step / 2,
    lat_step,
)

###############################################################################
# LOAD GEOMETRY
###############################################################################

print("Loading geometry...")

(
    ds_geom,
    distances,
    distance_edges,
    origin_lat_chunk_bounds,
    nbins,
) = load_geometry(
    geom_path,
    grid,
    chunk_origin,
    nlat=len(lats),
    nlon=len(lons),
    nbins_fac=nbins_fac,
)

olat_chunk = origin_lat_chunk_bounds[0]

geom_chunk, active_indices = load_geometry_chunk(
    ds_geom,
    olat_chunk,
    distance_edges,
    max_R=None,
)

###############################################################################
# LOAD VELOCITY FIELD
###############################################################################

print("Loading velocity field...")

u, v, _ = load_velocity_field(
    date=date,
    interp_lats=lats,
    interp_lons=lons,
)

###############################################################################
# REPRODUCE process_origin_longitude()
###############################################################################

olon = float(lons[0])

lon_step = (
    u.longitude.values[1]
    - u.longitude.values[0]
)

lon_shift = int(round(olon / lon_step))

u0 = u.sel(
    latitude=geom_chunk.origin_latitude,
    longitude=olon,
    method="nearest",
)

v0 = v.sel(
    latitude=geom_chunk.origin_latitude,
    longitude=olon,
    method="nearest",
)

u_roll = (
    u.roll(
        longitude=-lon_shift,
        roll_coords=False,
    )
    .load()
)

v_roll = (
    v.roll(
        longitude=-lon_shift,
        roll_coords=False,
    )
    .load()
)

###############################################################################
# BUILD A REPRESENTATIVE du^3 FIELD
###############################################################################

print("Computing du^3...")

du_cubed = compute_delta_u_cubed_numexpr(
    u_roll.values,
    v_roll.values,
    u0.values[:, None, None],
    v0.values[:, None, None],
    geom_chunk.sine_initial_bearing.values,
    geom_chunk.cosine_initial_bearing.values,
    geom_chunk.sine_final_bearing.values,
    geom_chunk.cosine_final_bearing.values,
)

###############################################################################
# SELECT ONE ORIGIN LATITUDE
###############################################################################

i = 0

integrand = du_cubed[i].ravel()

bins = (
    geom_chunk
    .great_circle_distance_bin
    .isel(origin_latitude=i)
    .values
    .ravel()
)

weights = (
    geom_chunk
    .angular_weight
    .isel(origin_latitude=i)
    .values
    .ravel()
)

valid = np.isfinite(integrand)

integrand_valid = integrand[valid]
bins_valid = bins[valid]
weights_valid = weights[valid]

print("\nArray info")
print("----------")
print(f"integrand shape = {integrand.shape}")
print(f"nbins           = {nbins}")
print(f"valid fraction  = {valid.mean():.6f}")

###############################################################################
# WARM-UP NUMBA
###############################################################################

print("\nCompiling Numba kernels...")

angular_integral_unweighted_numba(
    integrand,
    bins,
    nbins,
)

angular_integral_weighted_numba(
    integrand,
    bins,
    nbins,
    weights,
)

###############################################################################
# CORRECTNESS CHECK
###############################################################################

print("\nCorrectness checks")

ref_unweighted = angular_integral_by_distance_bin(
    integrand,
    bins,
    nbins,
)

test_unweighted = angular_integral_unweighted_numba(
    integrand,
    bins,
    nbins,
)

print(
    "max abs diff (unweighted):",
    np.nanmax(
        np.abs(
            ref_unweighted
            - test_unweighted
        )
    )
)

ref_weighted = angular_integral_by_distance_bin(
    integrand,
    bins,
    nbins,
    weights=weights,
)

test_weighted = angular_integral_weighted_numba(
    integrand,
    bins,
    nbins,
    weights,
)

print(
    "max abs diff (weighted):",
    np.nanmax(
        np.abs(
            ref_weighted
            - test_weighted
        )
    )
)

###############################################################################
# BENCHMARKS
###############################################################################

print("\nRunning benchmarks...")


#
# Full reference implementation
#

t_ref_unweighted, _ = benchmark(
    angular_integral_by_distance_bin,
    integrand,
    bins,
    nbins,
    repeats=repeats,
)

t_ref_weighted, _ = benchmark(
    angular_integral_by_distance_bin,
    integrand,
    bins,
    nbins,
    weights,
    repeats=repeats,
)


#
# Numba implementations
#

t_numba_unweighted, _ = benchmark(
    angular_integral_unweighted_numba,
    integrand,
    bins,
    nbins,
    repeats=repeats,
)

t_numba_weighted, _ = benchmark(
    angular_integral_weighted_numba,
    integrand,
    bins,
    nbins,
    weights,
    repeats=repeats,
)


#
# Raw bin_integrate benchmarks
#

t_bin_integrate_unweighted, _ = benchmark(
    bin_integrate,
    integrand_valid,
    bins_valid,
    nbins,
    repeats=repeats,
)

t_bin_integrate_weighted, _ = benchmark(
    bin_integrate,
    integrand_valid,
    bins_valid,
    nbins,
    weights_valid,
    repeats=repeats,
)

###############################################################################
# RESULTS
###############################################################################

print("\nBenchmark results")
print("-----------------")

print(
    f"angular_integral_by_distance_bin          "
    f"{t_ref_unweighted:.6e} s"
)

print(
    f"angular_integral_by_distance_bin (w)      "
    f"{t_ref_weighted:.6e} s"
)

print(
    f"angular_integral_unweighted_numba         "
    f"{t_numba_unweighted:.6e} s"
)

print(
    f"angular_integral_weighted_numba           "
    f"{t_numba_weighted:.6e} s"
)

print(
    f"bin_integrate (unweighted inputs)         "
    f"{t_bin_integrate_unweighted:.6e} s"
)

print(
    f"bin_integrate (weighted inputs)           "
    f"{t_bin_integrate_weighted:.6e} s"
)

print("\nSpeedups relative to angular_integral_by_distance_bin")
print("-----------------------------------------------------")

print(
    f"Numba unweighted : "
    f"{t_ref_unweighted / t_numba_unweighted:.2f}x"
)

print(
    f"Numba weighted   : "
    f"{t_ref_weighted / t_numba_weighted:.2f}x"
)

print("\nSpeedups relative to bin_integrate")
print("-----------------------------------------------------")

print(
    f"Numba unweighted : "
    f"{t_bin_integrate_unweighted / t_numba_unweighted:.2f}x"
)

print(
    f"Numba weighted   : "
    f"{t_bin_integrate_weighted / t_numba_weighted:.2f}x"
)
