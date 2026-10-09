#!/usr/bin/env python3
import time
import numpy as np
import xarray as xr

from lossett.calc.compute_spherical_geometry import (
    build_geometry_filename,
)

from lossett.calc.compute_delta_u_cubed_spherical import (
    load_geometry_chunk,
    pack_active_geometry,
)

# ============================================================
# SETTINGS
# ============================================================

#GRID = "n640"
GRID = "n1280"

GEOM_PATH = (
    "/work/scratch-pw5/dship/upscale/"
    "LoSSETT/spherical_geometry/"
)

if GRID == "n640":
    VELOCITY_FILE = (
        "/work/scratch-pw5/dship/upscale/"
        "LoSSETT/spherical_geometry/"
        "preprocessed_uvw/"
        "glm.n1280_GAL9.uvw_20160801T00_n640.nc"
    )
elif GRID == "n1280":
    VELOCITY_FILE = (
        "/gws/ssde/j25b/kscale/USERS/dship/LoSSETT_in/"
        "preprocessed_kscale_data/DYAMOND_SUMMER/"
        "n1280_regrid/embedded/"
        "channel_n2560_RAL3p2.n1280_GAL9.uvw_embedded_20160801T00.nc"
    )
else:
    import sys
    print(f"GRID = {GRID} not recognised; exiting.")
    sys.exit(1)
    
ORIGIN_LAT_CHUNKSIZE = 16

# Polar test
OLAT_CHUNK = (0, 16)
LAT_INDEX = 0

# Example shift
LON_SHIFT = 500

NREPEAT = 200

MAX_R = 2000e3

# ============================================================
# Helper
# ============================================================

def benchmark(name, func, repeat=NREPEAT):

    func()

    t0 = time.perf_counter()

    for _ in range(repeat):
        func()

    dt = time.perf_counter() - t0

    print(
        f"{name:<35s}"
        f"{dt:10.6f} s    "
        f"{1e3*dt/repeat:8.3f} ms/call"
    )

    return dt


# ============================================================
# Load velocity field
# ============================================================

print("\nLoading velocity field")

ds = xr.open_dataset(
    VELOCITY_FILE,
    decode_timedelta=False,
)

u = ds.u.isel(
    time=0,
    pressure=0,
)

uvals = u.values

nlat, nlon = uvals.shape

print(f"grid shape = {uvals.shape}")

# ============================================================
# Load geometry
# ============================================================

geom_fpath = build_geometry_filename(
    GEOM_PATH,
    GRID,
    trig_fns=True,
    nlat=nlat,
    nbins=nlon // 2,
    chunk_origin=ORIGIN_LAT_CHUNKSIZE,
)

ds_geom = xr.open_zarr(geom_fpath)

distance_edges = np.array(
    ds_geom.great_circle_distance_bin.attrs[
        "distance_bin_edges"
    ]
)

geom_chunk, active_indices = load_geometry_chunk(
    ds_geom,
    OLAT_CHUNK,
    distance_edges,
    max_R=MAX_R,
)

active_geom = pack_active_geometry(
    geom_chunk,
    active_indices,
)

geom_i = active_geom[LAT_INDEX]

ilat = geom_i["ilat"]
ilon = geom_i["ilon"]

print(f"\nn_active = {len(ilat):,}")
print(f"lon_shift = {LON_SHIFT}")

# ============================================================
# Precomputed single shift
# ============================================================

idx_precomputed = (
    ilon + LON_SHIFT
) % nlon

# ============================================================
# Precompute all shifted indices
# ============================================================

print("\nBuilding all shifted index arrays")

t0 = time.perf_counter()

all_shifted = [
    (ilon + shift) % nlon
    for shift in range(nlon)
]

precompute_time = (
    time.perf_counter() - t0
)

print(
    f"precompute all shifts: "
    f"{precompute_time:.3f} s"
)

# ============================================================
# Lookup table
# ============================================================

print("\nBuilding lookup table")

t0 = time.perf_counter()

lookup = np.arange(
    2 * nlon,
    dtype=np.int32,
)

lookup[nlon:] -= nlon

lookup_build_time = (
    time.perf_counter() - t0
)

print(
    f"lookup table build: "
    f"{lookup_build_time:.6f} s"
)

# ============================================================
# Cyclic padding
# ============================================================

print("\nBuilding cyclic padded field")

t0 = time.perf_counter()

u_pad = np.concatenate(
    [uvals, uvals],
    axis=1,
)

pad_build_time = (
    time.perf_counter() - t0
)

print(
    f"cyclic pad build: "
    f"{pad_build_time:.6f} s"
)

# ============================================================
# Benchmark functions
# ============================================================

def modulo_only():

    return (
        ilon + LON_SHIFT
    ) % nlon


def gather_only():

    return uvals[
        ilat,
        idx_precomputed,
    ]


def current_method():

    idx = (
        ilon + LON_SHIFT
    ) % nlon

    return uvals[
        ilat,
        idx,
    ]


def roll_only():

    return np.roll(
        uvals,
        -LON_SHIFT,
        axis=1,
    )


def roll_plus_gather():

    u_roll = np.roll(
        uvals,
        -LON_SHIFT,
        axis=1,
    )

    return u_roll[
        ilat,
        ilon,
    ]


def precomputed_idx():

    return uvals[
        ilat,
        idx_precomputed,
    ]


def precomputed_lookup():

    idx = all_shifted[
        LON_SHIFT
    ]

    return uvals[
        ilat,
        idx,
    ]


def lookup_only():

    return lookup[
        ilon + LON_SHIFT
    ]


def lookup_plus_gather():

    idx = lookup[
        ilon + LON_SHIFT
    ]

    return uvals[
        ilat,
        idx,
    ]

def cyclic_indices_only():

    return ilon + LON_SHIFT


def cyclic_pad_gather():

    idx = ilon + LON_SHIFT

    return u_pad[
        ilat,
        idx,
    ]


# ============================================================
# Run benchmarks
# ============================================================

print("\nBenchmarks\n")

benchmark(
    "Modulo only",
    modulo_only,
)

benchmark(
    "Gather only",
    gather_only,
)

benchmark(
    "CURRENT: modulo + gather",
    current_method,
)

benchmark(
    "ROLL ONLY",
    roll_only,
)

benchmark(
    "ROLL + gather",
    roll_plus_gather,
)

benchmark(
    "PRECOMPUTED idx",
    precomputed_idx,
)

benchmark(
    "PRECOMPUTED lookup+gather",
    precomputed_lookup,
)

benchmark(
    "LOOKUP only",
    lookup_only,
)

benchmark(
    "LOOKUP + gather",
    lookup_plus_gather,
)

benchmark(
    "CYCLIC indices only",
    cyclic_indices_only,
)

benchmark(
    "CYCLIC PAD + gather",
    cyclic_pad_gather,
)

# ============================================================
# Correctness checks
# ============================================================

print("\nCorrectness checks")

current = current_method()

rolled = np.roll(
    uvals,
    -LON_SHIFT,
    axis=1,
)[ilat, ilon]

precomputed = precomputed_idx()

precomputed_lookup_result = precomputed_lookup()

lookup_result = lookup_plus_gather()

print(
    "current vs roll:",
    np.max(
        np.abs(
            current - rolled
        )
    )
)

print(
    "current vs precomputed_idx:",
    np.max(
        np.abs(
            current - precomputed
        )
    )
)

print(
    "current vs precomputed_lookup:",
    np.max(
        np.abs(
            current - precomputed_lookup_result
        )
    )
)

print(
    "current vs lookup:",
    np.max(
        np.abs(
            current - lookup_result
        )
    )
)

print("\nDone.")
