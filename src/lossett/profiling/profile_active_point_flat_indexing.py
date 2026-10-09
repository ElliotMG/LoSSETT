#!/usr/bin/env python3

import time
import numpy as np

from lossett.calc.compute_spherical_geometry import (
    GRID_DEFS,
    build_regular_latlon_grid,
)

from lossett.calc.compute_delta_u_cubed_spherical import (
    load_geometry,
    load_velocity_field,
    load_geometry_chunk,
    pack_active_geometry,
)

# =============================================================================
# USER SETTINGS
# =============================================================================

VELOCITY_FILE = (
    "/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/"
    "preprocessed_uvw/glm.n1280_GAL9.uvw_20160801T00_n640.nc"
)
GEOM_PATH = "/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/"

GRID = "n640"

PRESSURE = 200
TIME_INDEX = 0

MAX_R = 2000e3

ORIGIN_CHUNK = 16

NBINS_FAC = 2

NREPEAT = 100

# =============================================================================


def benchmark_current(
    active_geom,
    u_pad,
    v_pad,
    lon_shift,
):
    """
    Current production-style gather.
    """

    t0 = time.perf_counter()

    checksum = 0.0

    for geom_i in active_geom:

        ilat = geom_i["ilat"]
        ilon = geom_i["ilon"]

        ilon_shifted = ilon + lon_shift

        u_sel = u_pad[ilat, ilon_shifted]
        v_sel = v_pad[ilat, ilon_shifted]

        checksum += (
            u_sel.sum()
            + v_sel.sum()
        )

    return time.perf_counter() - t0, checksum


def benchmark_flat(
    active_geom,
    u_flat,
    v_flat,
    nlon_pad,
    lon_shift,
):
    """
    Flat-index gather.
    """

    t0 = time.perf_counter()

    checksum = 0.0

    for geom_i in active_geom:

        ilat = geom_i["ilat"]
        ilon = geom_i["ilon"]

        idx = (
            ilat * nlon_pad
            + ilon
            + lon_shift
        )

        u_sel = u_flat[idx]
        v_sel = v_flat[idx]

        checksum += (
            u_sel.sum()
            + v_sel.sum()
        )

    return time.perf_counter() - t0, checksum


def main():

    print("Loading geometry...")

    lon_step, lat_step = GRID_DEFS[GRID]

    lons, lats = build_regular_latlon_grid(
        lon_step,
        lat_step,
    )

    ds_geom, distances, distance_edges, chunk_bounds = (
        load_geometry(
            GEOM_PATH,
            GRID,
            ORIGIN_CHUNK,
            nlat=len(lats),
            nlon=len(lons),
            nbins_fac=NBINS_FAC,
        )
    )

    print("Loading velocity field...")

    ds_u = load_velocity_field(
        VELOCITY_FILE,
        PRESSURE,
        TIME_INDEX,
    )

    u = ds_u.u.values
    v = ds_u.v.values

    print(
        f"grid shape = {u.shape}"
    )

    print("Creating cyclic padding...")

    u_pad = np.concatenate(
        [u, u],
        axis=1,
    )

    v_pad = np.concatenate(
        [v, v],
        axis=1,
    )

    nlon_pad = u_pad.shape[1]

    print(
        f"padded shape = {u_pad.shape}"
    )

    print("Loading geometry chunk...")

    olat_chunk = chunk_bounds[
        len(chunk_bounds) // 2
    ]

    geom_chunk, active_indices = (
        load_geometry_chunk(
            ds_geom,
            olat_chunk,
            distance_edges,
            max_R=MAX_R,
        )
    )

    active_geom = pack_active_geometry(
        geom_chunk,
        active_indices,
        method="spherical",
    )

    total_active = sum(
        g["ilat"].size
        for g in active_geom
    )

    print(
        f"n_active = {total_active:,}"
    )

    lon_shift = len(lons) // 3

    print(
        f"lon_shift = {lon_shift}"
    )

    print("Creating flat arrays...")

    u_flat = u_pad.ravel()
    v_flat = v_pad.ravel()

    print("\nWarmup...")

    benchmark_current(
        active_geom,
        u_pad,
        v_pad,
        lon_shift,
    )

    benchmark_flat(
        active_geom,
        u_flat,
        v_flat,
        nlon_pad,
        lon_shift,
    )

    print("\nBenchmarking...")

    current_times = []

    for _ in range(NREPEAT):

        t, checksum_current = benchmark_current(
            active_geom,
            u_pad,
            v_pad,
            lon_shift,
        )

        current_times.append(t)

    flat_times = []

    for _ in range(NREPEAT):

        t, checksum_flat = benchmark_flat(
            active_geom,
            u_flat,
            v_flat,
            nlon_pad,
            lon_shift,
        )

        flat_times.append(t)

    current_mean = np.mean(current_times)
    flat_mean = np.mean(flat_times)

    print("\nCorrectness check")
    print("-----------------")

    print(
        f"current checksum = "
        f"{checksum_current:.12e}"
    )

    print(
        f"flat checksum    = "
        f"{checksum_flat:.12e}"
    )

    print(
        f"abs diff         = "
        f"{abs(checksum_current-checksum_flat):.12e}"
    )

    print("\nResults")
    print("-------")

    print(
        f"Current gather : "
        f"{current_mean:.6f} s"
    )

    print(
        f"Flat gather    : "
        f"{flat_mean:.6f} s"
    )

    print(
        f"Speed-up       : "
        f"{current_mean/flat_mean:.3f}x"
    )


if __name__ == "__main__":
    main()
