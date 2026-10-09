#!/usr/bin/env python3

import time
import numpy as np
import xarray as xr

from lossett.calc.compute_delta_u_cubed_spherical import (
    load_geometry,
    load_geometry_chunk,
    pack_active_geometry,
)

from lossett.calc.field_increments import (
    select_active_points,
)

###############################################################################

GRID = "n1280"
ORIGIN_LAT_CHUNKSIZE = 16
MAX_R_KM = 2000.0

GEOM_PATH = (
    "/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/"
)

VEL_FILE = (
    "/gws/ssde/j25b/kscale/USERS/dship/"
    "LoSSETT_in/preprocessed_kscale_data/"
    "DYAMOND_SUMMER/n1280_regrid/embedded/"
    "channel_n2560_RAL3p2.n1280_GAL9.uvw_embedded_20160801T00.nc"
)

PRESSURE = 200
TIME_INDEX = 0

NREPEAT = 3


def mean_time(func, *args, **kwargs):

    times = []

    for _ in range(NREPEAT):

        t0 = time.perf_counter()

        func(*args, **kwargs)

        times.append(
            time.perf_counter() - t0
        )

    return np.mean(times)


###############################################################################
# LOAD GEOMETRY
###############################################################################

print("Loading geometry ...")

(
    ds_geom,
    distances,
    distance_edges,
    origin_lat_chunk_bounds,
) = load_geometry(
    GEOM_PATH,
    GRID,
    ORIGIN_LAT_CHUNKSIZE,
    nlat=1921,
    nlon=2560,
)

olat_chunk = origin_lat_chunk_bounds[0]

print(
    f"Using latitude chunk {olat_chunk}"
)

###############################################################################
# LOAD GEOMETRY CHUNK
###############################################################################

max_R = MAX_R_KM * 1000.0

print("Loading geometry chunk ...")

geom_chunk, active_indices = load_geometry_chunk(
    ds_geom,
    olat_chunk,
    distance_edges,
    max_R=max_R,
)

###############################################################################
# PACK GEOMETRY
###############################################################################

print()
print("Benchmarking pack_active_geometry ...")

t_pack = mean_time(
    pack_active_geometry,
    geom_chunk,
    active_indices,
    False,          # use_angular_weights
    "spherical",    # method
)

print(
    f"pack_active_geometry = {t_pack:.3f} s"
)

###############################################################################
# REPORT ACTIVE POINT COUNTS
###############################################################################

nactive = []

for ilat, ilon in active_indices:
    nactive.append(len(ilat))

nactive = np.asarray(nactive)

print()
print(
    f"mean active points = {nactive.mean():,.0f}"
)
print(
    f"min active points  = {nactive.min():,}"
)
print(
    f"max active points  = {nactive.max():,}"
)

###############################################################################
# COST OF OLD GEOMETRY EXTRACTION ONLY
###############################################################################

print()
print("Benchmarking old geometry extraction ...")

sin_init = geom_chunk.sine_initial_bearing.values
cos_init = geom_chunk.cosine_initial_bearing.values

sin_final = geom_chunk.sine_final_bearing.values
cos_final = geom_chunk.cosine_final_bearing.values

bins = geom_chunk.great_circle_distance_bin.values


def benchmark_old_geometry():

    for i, (ilat, ilon) in enumerate(active_indices):

        _ = sin_init[i, ilat, ilon]
        _ = cos_init[i, ilat, ilon]

        _ = sin_final[i, ilat, ilon]
        _ = cos_final[i, ilat, ilon]

        _ = bins[i, ilat, ilon]


t_old = mean_time(
    benchmark_old_geometry
)

print(
    f"old geometry gathers = {t_old:.3f} s"
)

###############################################################################
# PACKED ACCESS COST
###############################################################################

packed = pack_active_geometry(
    geom_chunk,
    active_indices,
)

def benchmark_packed_access():

    for geom_i in packed:

        _ = geom_i["sin_init"]
        _ = geom_i["cos_init"]

        _ = geom_i["sin_final"]
        _ = geom_i["cos_final"]

        _ = geom_i["bins"]


t_access = mean_time(
    benchmark_packed_access
)

print(
    f"packed access = {t_access:.6f} s"
)

###############################################################################
# SUMMARY
###############################################################################

print()
print("SUMMARY")
print("-------")

print(
    f"pack geometry      : {t_pack:.3f} s"
)
print(
    f"old gathers        : {t_old:.3f} s"
)
print(
    f"packed reuse cost  : {t_access:.6f} s"
)

if t_pack > 0:
    print(
        f"reuse ratio = {t_old / t_pack:.2f}x"
    )
