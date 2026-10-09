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

GRID = "n640"

GEOM_PATH = (
    "/work/scratch-pw5/dship/upscale/"
    "LoSSETT/spherical_geometry/"
)

VELOCITY_FILE = (
    "/work/scratch-pw5/dship/upscale/"
    "LoSSETT/spherical_geometry/"
    "preprocessed_uvw/"
    "glm.n1280_GAL9.uvw_20160801T00_n640.nc"
)

ORIGIN_LAT_CHUNKSIZE = 16

OLAT_CHUNK = (0, 16)
LAT_INDEX = 0

LON_SHIFT = 500

MAX_R = 2000e3

NREPEAT = 1000

# ============================================================
# helper
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
# load velocity field
# ============================================================

print("\nLoading velocity field")

ds = xr.open_dataset(
    VELOCITY_FILE,
    decode_timedelta=False,
)

u = ds.u.isel(time=0, pressure=0)
v = ds.v.isel(time=0, pressure=0)
w = ds.w.isel(time=0, pressure=0)

uvals = u.values
vvals = v.values
wvals = w.values

nlat, nlon = uvals.shape

print(f"grid shape = {uvals.shape}")

# ============================================================
# geometry
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
print(f"ilat dtype = {ilat.dtype}")
print(f"ilon dtype = {ilon.dtype}")

# ============================================================
# cyclic padding
# ============================================================

print("\nBuilding cyclic padded fields")

u_pad = np.concatenate(
    [uvals, uvals],
    axis=1,
)

v_pad = np.concatenate(
    [vvals, vvals],
    axis=1,
)

w_pad = np.concatenate(
    [wvals, wvals],
    axis=1,
)

uvw_pad = np.stack(
    [u_pad, v_pad, w_pad],
    axis=0,
)

idx = ilon + LON_SHIFT

# ============================================================
# reduced precision indexing
# ============================================================

ilat32 = ilat.astype(np.int32)
idx32  = idx.astype(np.int32)

# ============================================================
# benchmark functions
# ============================================================

def separate_gather():

    u_sel = u_pad[ilat, idx]
    v_sel = v_pad[ilat, idx]
    w_sel = w_pad[ilat, idx]

    return u_sel, v_sel, w_sel


def separate_gather_int32():

    u_sel = u_pad[ilat32, idx32]
    v_sel = v_pad[ilat32, idx32]
    w_sel = w_pad[ilat32, idx32]

    return u_sel, v_sel, w_sel


def combined_gather():

    uvw_sel = uvw_pad[:, ilat, idx]

    return uvw_sel


def combined_gather_unpack():
    uvw_sel = uvw_pad[:, ilat, idx]

    u_sel = uvw_sel[0]
    v_sel = uvw_sel[1]
    w_sel = uvw_sel[2]

    return u_sel, v_sel, w_sel

# ============================================================
# run
# ============================================================

print("\nBenchmarks\n")

benchmark(
    "separate uvw gathers",
    separate_gather,
)

benchmark(
    "separate uvw gathers (int32)",
    separate_gather_int32,
)

benchmark(
    "Combined uvw gather",
    combined_gather,
)

benchmark(
    "Combined gather + unpack",
    combined_gather_unpack,
)

# ============================================================
# correctness
# ============================================================

print("\nCorrectness check\n")

u1, v1, w1 = separate_gather()
uvw2 = combined_gather()

print(
    "u max diff:",
    np.max(np.abs(u1 - uvw2[0]))
)

print(
    "v max diff:",
    np.max(np.abs(v1 - uvw2[1]))
)

print(
    "w max diff:",
    np.max(np.abs(w1 - uvw2[2]))
)

print("\nDone.")
