#!/usr/bin/env python3

import time
import numpy as np

from lossett.calc.compute_delta_u_cubed_spherical import (
    load_geometry,
    load_velocity_field,
    load_geometry_chunk,
    pack_active_geometry,
)

from lossett.calc.field_increments import (
    select_active_points_packed,
)

# ------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------

VELOCITY_FILE = "/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/preprocessed_uvw/glm.n1280_GAL9.uvw_20160801T00_n640.nc"
GEOM_PATH = "/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/"

GRID = "n640"

PRESSURE = 200
TIME_INDEX = 0

MAX_R = 2000e3          # metres
ORIGIN_CHUNK = 16

NBINS_FAC = 2

NREPEAT = 50

INCLUDE_W = False

# ------------------------------------------------------------------

def sort_active_geom_memory_order(active_geom, nlon):

    sorted_geom = []

    for geom_i in active_geom:

        ilat = geom_i["ilat"]
        ilon = geom_i["ilon"]

        flat = ilat * nlon + ilon

        order = np.argsort(flat)

        geom_i_sorted = {
            k: (
                v[order]
                if isinstance(v, np.ndarray)
                and v.shape == ilat.shape
                else v
            )
            for k, v in geom_i.items()
        }

        sorted_geom.append(geom_i_sorted)

    return sorted_geom


def benchmark_gather(
    active_geom,
    u,
    v,
    u0,
    v0,
    lon_shift,
):

    t0 = time.perf_counter()

    total_points = 0

    for i in range(len(active_geom)):

        (
            u_sel,
            v_sel,
            u0_sel,
            v0_sel,
            sin_init_sel,
            cos_init_sel,
            sin_final_sel,
            cos_final_sel,
            bins_sel,
            weights_sel,
            distance_sel,
            w_sel,
            w0_sel,
        ) = select_active_points_packed(
            i,
            u,
            v,
            u0,
            v0,
            active_geom,
            lon_shift,
            w=None,
            w0=None,
            use_cyclic_padding=False,
        )

        total_points += u_sel.size

    return time.perf_counter() - t0, total_points


def main():

    print("Loading geometry...")

    from lossett.calc.compute_spherical_geometry import (
        GRID_DEFS,
        build_regular_latlon_grid,
    )

    lon_step, lat_step = GRID_DEFS[GRID]
    lons, lats = build_regular_latlon_grid(
        lon_step,
        lat_step,
    )

    ds_geom, distances, distance_edges, chunk_bounds = load_geometry(
        GEOM_PATH,
        GRID,
        ORIGIN_CHUNK,
        nlat=len(lats),
        nlon=len(lons),
        nbins_fac=NBINS_FAC,
    )

    print("Loading velocity field...")

    ds_u = load_velocity_field(
        VELOCITY_FILE,
        PRESSURE,
        TIME_INDEX,
    )

    u = ds_u.u
    v = ds_u.v

    olat_chunk = chunk_bounds[len(chunk_bounds)//2]

    geom_chunk, active_indices = load_geometry_chunk(
        ds_geom,
        olat_chunk,
        distance_edges,
        max_R=MAX_R,
    )

    active_geom = pack_active_geometry(
        geom_chunk,
        active_indices,
        method="spherical",
    )

    nlon = u.shape[1]

    print(f"nlon       = {nlon}")
    print(f"chunk size = {len(active_geom)}")

    total_active = sum(
        g["ilat"].size
        for g in active_geom
    )

    print(f"n_active   = {total_active:,}")

    active_geom_sorted = sort_active_geom_memory_order(
        active_geom,
        nlon,
    )

    lon_shift = nlon // 3

    u0 = u.sel(
        latitude=geom_chunk.origin_latitude,
        longitude=0,
        method="nearest",
    )

    v0 = v.sel(
        latitude=geom_chunk.origin_latitude,
        longitude=0,
        method="nearest",
    )

    print("\nWarmup...")

    benchmark_gather(
        active_geom,
        u,
        v,
        u0,
        v0,
        lon_shift,
    )

    benchmark_gather(
        active_geom_sorted,
        u,
        v,
        u0,
        v0,
        lon_shift,
    )

    print("\nBenchmarking...")

    t_original = []

    for _ in range(NREPEAT):

        t, _ = benchmark_gather(
            active_geom,
            u,
            v,
            u0,
            v0,
            lon_shift,
        )

        t_original.append(t)

    t_sorted = []

    for _ in range(NREPEAT):

        t, _ = benchmark_gather(
            active_geom_sorted,
            u,
            v,
            u0,
            v0,
            lon_shift,
        )

        t_sorted.append(t)

    t_original = np.mean(t_original)
    t_sorted = np.mean(t_sorted)

    print("\nResults")
    print("-------")

    print(f"Original ordering : {t_original:.6f} s")
    print(f"Memory ordering   : {t_sorted:.6f} s")

    print(
        f"Speed-up          : "
        f"{t_original/t_sorted:.3f}x"
    )


if __name__ == "__main__":
    main()
