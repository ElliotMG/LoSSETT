#!/usr/bin/env python3
import os
import xarray as xr
import numpy as np
import dask.array as da
import time
import argparse
from numcodecs import Blosc

from lossett.calc.spherical_geometry import (
    build_regular_latlon_grid,
    build_distance_bins,
    compute_geometry,
    RADIUS_EARTH,
    GEOM_DIMS
)

GRID_DEFS = {
    "5p0deg": (5.0, 5.0),
    "2p5deg": (2.5, 2.5),
    "1p0deg": (1.0, 1.0),
    "0p5deg": (0.5, 0.5),
    "n160": (1.125, 0.75),
    "n320": (0.5625, 0.375),
    "n640": (0.28125, 0.1875),
    "n1280": (0.140625, 0.09375),
    "n2560": (0.0703125, 0.046875),
}
COMPRESSOR = Blosc(
    cname="zstd",
    clevel=3,
    shuffle=Blosc.BITSHUFFLE,
)
TRIG_VARS = [
    "sine_initial_bearing",
    "cosine_initial_bearing",
    "sine_final_bearing",
    "cosine_final_bearing",
]
BEARING_VARS = [
    "initial_bearing",
    "final_bearing",
]
DTYPES = {
    "float32": np.float32,
    "float64": np.float64,
}

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--grid",
        default="n320",
    )

    parser.add_argument(
        "--save-path",
        required=True,
    )

    parser.add_argument(
        "--chunk-origin-lat",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--dtype",
        choices=DTYPES.keys(),
        default="float32",
    )

    parser.add_argument(
        "--save-trig-fns",
        action="store_true",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()

def build_geometry_filename(
    save_path,
    grid,
    trig_fns,
    nlat,
    nbins,
    chunk_origin,
):
    if trig_fns:
        trig_str = "_trig_fns"
    else:
        trig_str = ""
    
    geom_fpath = os.path.join(
        save_path,
        f"distances_and_bearings{trig_str}_{grid}_n_lat0_{nlat}_"\
        f"n_dist_bins_{nbins}_chunksize_{chunk_origin}.zarr"
    )
    return geom_fpath

def make_geom_var(shape,dtype,chunks):
    return (
        GEOM_DIMS,
        da.empty(
            shape,
            dtype=dtype,
            chunks=chunks,
        ),
    )

def copy_dataset_attrs(source, target):
    # Dataset-level attributes
    target.attrs.update(source.attrs)

    # Variable-level attributes
    for var in target.data_vars:
        if var in source:
            target[var].attrs.update(
                source[var].attrs
            )

def create_geometry_template(
        origin_latitudes,
        target_latitudes,
        delta_longitudes,
        chunk_origin,
        chunk_lat,
        chunk_lon,
        distance_edges,
        radius=RADIUS_EARTH,
        dtype=np.float32,
        bin_dtype=np.uint8,
        trig_fns=False
):
    shape = (
        len(origin_latitudes),
        len(target_latitudes),
        len(delta_longitudes),
    )
    chunks = (
        chunk_origin,
        chunk_lat,
        chunk_lon,
    )
    coords = {
        "origin_latitude": xr.Variable(
            "origin_latitude",
            origin_latitudes,
            attrs={"units": "degrees_north"},
        ),
        "latitude": xr.Variable(
            "latitude",
            target_latitudes,
            attrs={"units": "degrees_north"},
        ),
        "longitude": xr.Variable(
            "longitude",
            delta_longitudes,
            attrs={"units": "degrees_east"},
        ),
    }

    variables = {
        "great_circle_distance":
        make_geom_var(shape, dtype, chunks),
        "great_circle_distance_bin":
        make_geom_var(shape, bin_dtype, chunks),
    }

    if trig_fns:
        variables.update({
            "sine_initial_bearing":
            make_geom_var(shape, dtype, chunks),
            "cosine_initial_bearing":
            make_geom_var(shape, dtype, chunks),
            "sine_final_bearing":
            make_geom_var(shape, dtype, chunks),
            "cosine_final_bearing":
            make_geom_var(shape, dtype, chunks),
            "angular_weight":
            make_geom_var(shape, dtype, chunks),
        })
    else:
        variables.update({
            "initial_bearing":
            make_geom_var(shape, dtype, chunks),
            "final_bearing":
            make_geom_var(shape, dtype, chunks),
        })

    template = xr.Dataset(
        variables,
        coords=coords,
    )

    # Copy metadata from a minimal geometry dataset.
    metadata_ds = compute_geometry(
        origin_latitudes[:1],
        target_latitudes[:1],
        delta_longitudes[:1],
        distance_edges,
        dtype=dtype,
        bin_dtype=bin_dtype,
        trig_fns=trig_fns,
    )

    copy_dataset_attrs(
        metadata_ds, # source
        template     # target
    )

    return template

def build_encoding(trig_fns):
    geom_vars = (
        ["great_circle_distance"]
        + (
            TRIG_VARS
            if trig_fns
            else BEARING_VARS
        )
    )

    encoding = {
        v: {"compressor": COMPRESSOR}
        for v in geom_vars
    }
    return encoding

def initialize_geometry_store(
    geom_fpath,
    origin_lats,
    target_lats,
    delta_lons,
    chunk_origin,
    chunk_lat,
    chunk_lon,
    dtype,
    bin_dtype,
    trig_fns,
    distance_edges,
    radius=RADIUS_EARTH
):
    """
    Create an empty geometry Zarr store ready for
    incremental writes.
    """

    template = create_geometry_template(
        origin_lats,
        target_lats,
        delta_lons,
        chunk_origin,
        chunk_lat,
        chunk_lon,
        distance_edges,
        dtype=dtype,
        bin_dtype=bin_dtype,
        trig_fns=trig_fns,
        radius=radius
    )

    encoding = build_encoding(trig_fns)

    template.to_zarr(
        geom_fpath,
        mode="w",
        compute=False,
        zarr_format=2,
        encoding=encoding,
    )

def iter_chunks(n, chunk_size):
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        yield start, stop

def write_geometry_chunk(
    geom_fpath,
    i0,
    i1,
    origin_lats,
    target_lats,
    delta_lons,
    distance_edges,
    compute_kwargs
):
    """
    Compute and write one origin-latitude chunk of geometry.
    """

    origin_lat_chunk = origin_lats[i0:i1]

    print(f"{i0} -> {i1}")

    ds_chunk = compute_geometry(
        origin_lat_chunk,
        target_lats,
        delta_lons,
        distance_edges,
        **compute_kwargs
    )

    # remove coordinates before regional write
    ds_chunk = ds_chunk.drop_vars(
        ["latitude", "longitude"],
        errors="ignore",
    )

    ds_chunk.to_zarr(
        geom_fpath,
        region={
            "origin_latitude": slice(i0, i1)
        }
    )

    del ds_chunk

if __name__ == "__main__":
    # parse command-line arguments (TO-DO: ADD YAML CONFIG FILE SUPPORT)
    args = parse_args()

    grid = args.grid
    save_path = args.save_path
    chunk_origin = args.chunk_origin_lat
    save_trig_fns = args.save_trig_fns
    force = args.force
    dtype = DTYPES[args.dtype]

    print(
        "\nCOMPUTING SPHERICAL GEOMETRY\n"
        "\nSetup:\n"
        "----------------------------------------\n"
        f"grid = {grid}\n"
        f"save_path = {save_path}\n"
        f"origin_lat chunksize = {chunk_origin}\n"
        f"save_trig_fns = {save_trig_fns}\n"
        f"dtype = {dtype}\n"
        f"force = {force}\n"
        "----------------------------------------\n"
    )

    ### CONSTRUCT GRID AND DISTANCE BINS
    # construct grid
    lon_step, lat_step = GRID_DEFS[grid]
    lons, lats = build_regular_latlon_grid(lon_step, lat_step)

    origin_lats = lats
    target_lats = lats
    delta_lons = lons

    # construct distance bins
    max_r = np.pi * RADIUS_EARTH # half a great circle
    nbins = len(lons) // 4 # this gives a sampling of 2dx over a half-great-circle
    distance_edges, _ = build_distance_bins(nbins, max_r=np.pi*RADIUS_EARTH)

    ### SETUP ZARR STORE
    # define Zarr store for geometry dataset
    chunk_lat = len(lats)
    chunk_lon = len(lons)
    if nbins < 256:
        bin_dtype = np.uint8
    else:
        bin_dtype = np.uint16
        # np.uint16 is fine for <~60000 bins, so dx >~300 m
        
    geom_fpath = build_geometry_filename(
        save_path,
        grid,
        save_trig_fns,
        len(origin_lats),
        nbins,
        chunk_origin,
    )

    if not os.path.exists(geom_fpath) or force:
        compute_kwargs = {
            "dtype": dtype,
            "bin_dtype": bin_dtype,
            "trig_fns": save_trig_fns,
            "radius": RADIUS_EARTH
        }

        initialize_geometry_store(
            geom_fpath,
            origin_lats,
            target_lats,
            delta_lons,
            chunk_origin,
            chunk_lat,
            chunk_lon,
            dtype,
            bin_dtype,
            save_trig_fns,
            distance_edges,
            radius=RADIUS_EARTH
        )
        
        for i0, i1 in iter_chunks(len(origin_lats), chunk_origin,):
            t0 = time.perf_counter()
            write_geometry_chunk(
                geom_fpath,
                i0,
                i1,
                origin_lats,
                target_lats,
                delta_lons,
                distance_edges,
                compute_kwargs
            )
            print(
                f"Write geometry chunk {i0} -> {i1}: "
                f"{time.perf_counter()-t0:.6f}s"
            )
        #endfor
    #endif
