#!/usr/bin/env python3

import numpy as np

from lossett.calc.compute_delta_u_cubed_spherical import (
    load_geometry,
    load_velocity_field,
    load_geometry_chunk,
    GRID_DEFS,
)

from lossett.calc.field_increments import (
    compute_delta_u_cubed,
    compute_delta_u_cubed_numexpr,
)

from lossett.profiling import benchmark

import numexpr as ne
EXPR_DUT = ne.NumExpr(
    "u*sin_final + v*cos_final "
    "- u0*sin_init - v0*cos_init"
)

EXPR_DUN = ne.NumExpr(
    "u*cos_final - v*sin_final "
    "- u0*cos_init + v0*sin_init"
)

EXPR_DU3 = ne.NumExpr(
    "du_t*(du_t*du_t + du_n*du_n)"
)
print(EXPR_DUT.input_names)
print(EXPR_DUN.input_names)
print(EXPR_DU3.input_names)

#
# Configuration
#

grid = "n640"

n_repeats = 100

geom_path = (
    "/work/scratch-pw5/dship/upscale/LoSSETT/"
    "spherical_geometry/"
)

chunk_origin = 16
nbins_fac = 2
date = "20160801"

#
# Build analysis grid
#

lon_step, lat_step = GRID_DEFS[grid]

lons = np.arange(-180., 180., lon_step)
lats = np.arange(-90., 90. + lat_step/2, lat_step)

#
# Load geometry
#

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

#
# Load first chunk exactly as production does
#

olat_chunk = origin_lat_chunk_bounds[0]

geom_chunk, active_indices = load_geometry_chunk(
    ds_geom,
    olat_chunk,
    distance_edges,
    max_R=None,
)

#
# Load winds
#

u, v, _ = load_velocity_field(
    date=date,
    interp_lats=lats,
    interp_lons=lons,
)

#
# Choose one origin longitude
#

olon = float(lons[0])

#
# Reproduce process_origin_longitude()
#

lon_step = (
    u.longitude.values[1]
    - u.longitude.values[0]
)

lon_shift = int(
    round(olon / lon_step)
)

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

#
# Convert everything to NumPy
#

u_np = u_roll.values
v_np = v_roll.values

u0_np = u0.values[:, None, None]
v0_np = v0.values[:, None, None]

sin_init_np = (
    geom_chunk
    .sine_initial_bearing
    .values
)

cos_init_np = (
    geom_chunk
    .cosine_initial_bearing
    .values
)

sin_final_np = (
    geom_chunk
    .sine_final_bearing
    .values
)

cos_final_np = (
    geom_chunk
    .cosine_final_bearing
    .values
)

print("\nArray shapes")
print("------------")
print("u               ", u_np.shape)
print("u0              ", u0_np.shape)
print("sin_init        ", sin_init_np.shape)
print("sin_final       ", sin_final_np.shape)
print(u_np.dtype)
print(sin_init_np.dtype)

#
# Verify correctness
#

du3_numpy = compute_delta_u_cubed(
    u_np,
    v_np,
    u0_np,
    v0_np,
    sin_init_np,
    cos_init_np,
    sin_final_np,
    cos_final_np,
)

du3_numexpr = compute_delta_u_cubed_numexpr(
    u_np,
    v_np,
    u0_np,
    v0_np,
    sin_init_np,
    cos_init_np,
    sin_final_np,
    cos_final_np,
)

print("\nValidation")
print("----------")

print(
    "max abs diff =",
    np.nanmax(
        np.abs(
            du3_numpy
            - du3_numexpr
        )
    )
)

diff = du3_numpy - du3_numexpr

print("numpy dtype   :", du3_numpy.dtype)
print("numexpr dtype :", du3_numexpr.dtype)

print("min numpy :", np.nanmin(du3_numpy))
print("max numpy :", np.nanmax(du3_numpy))

print("min numexpr:", np.nanmin(du3_numexpr))
print("max numexpr:", np.nanmax(du3_numexpr))

idx = np.unravel_index(
    np.nanargmax(np.abs(diff)),
    diff.shape
)

print("worst index:", idx)

print("numpy value :", du3_numpy[idx])
print("numexpr value:", du3_numexpr[idx])
print("difference  :", diff[idx])

#
# Benchmark
#

for nthreads in [1,2,4,8,16]:
    print(f"\nnthreads = {nthreads}")
    numpy_time, numpy_init = benchmark(
        compute_delta_u_cubed,
        u_np,
        v_np,
        u0_np,
        v0_np,
        sin_init_np,
        cos_init_np,
        sin_final_np,
        cos_final_np,
        repeats=n_repeats,
    )

    numexpr_time, numexpr_init = benchmark(
        compute_delta_u_cubed_numexpr,
        u_np,
        v_np,
        u0_np,
        v0_np,
        sin_init_np,
        cos_init_np,
        sin_final_np,
        cos_final_np,
        repeats=n_repeats,
    )

    print("\nBenchmark")
    print("---------")
    
    print(
        f"numpy   : {numpy_time:.6e} s"
    )
    
    print(
        f"numexpr : {numexpr_time:.6e} s"
    )
    
    print(
        f"speedup : "
        f"{numpy_time/numexpr_time:.2f}x"
    )
