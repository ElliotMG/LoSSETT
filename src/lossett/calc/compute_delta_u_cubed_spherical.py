#!/usr/bin/env python3
import sys
import os
import numpy as np
import xarray as xr
import dask.array as da
import time
from datetime import datetime, UTC
import argparse
import logging
from importlib.metadata import version
from pathlib import Path

# for profiling
import cProfile
import pstats

from lossett.calc.compute_spherical_geometry import (
    build_geometry_filename, build_regular_latlon_grid,
    GRID_DEFS, RADIUS_EARTH, DTYPES
)
from lossett.calc.field_increments import (
    compute_du3_angular_integral_global,
    compute_du3_angular_integral_subset,
)
from lossett.profiling import Profiler, profile_block

# Module-scope variables
LOSSETT_VN = version("lossett")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--velocity-file",
        required=True,
        help="Preprocessed velocity file; must contain variables called u and v"
    )

    parser.add_argument(
        "--grid",
        required=True,
        choices=GRID_DEFS.keys(),
        help="Grid definition (must be regular lat-lon)"
    )

    parser.add_argument(
        "--time-index",
        type=int,
        required=True,
        help="Time index"
    )

    parser.add_argument(
        "--pressure",
        type=int,
        required=True,
        help="Pressure level in hPa."
    )

    parser.add_argument(
        "--geom-path",
        required=True,
        help="Geometry archive directory"
    )

    parser.add_argument(
        "--save-path",
        required=True,
        help="Output directory"
    )

    parser.add_argument(
        "--outname-root",
        default=None,
        help=(
            "Root name for output file. "
            "Defaults to the velocity file stem."
        )
    )

    parser.add_argument(
        "--origin-lat-chunksize",
        type=int,
        default=4,
        help="Origin latitude chunk size"
    )

    parser.add_argument(
        "--max-R",
        type=float,
        default=None,
        help="Maximum great circle distance in kilometres"
    )

    parser.add_argument(
        "--save-dtype",
        choices=DTYPES.keys(),
        default="float32",
    )

    parser.add_argument(
        "--calc-dtype",
        choices=DTYPES.keys(),
        default="float64",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    parser.add_argument(
        "--profile",
        action="store_true",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    parser.add_argument(
        "--use-angular-weights",
        action="store_true",
        help=(
            "Use precomputed angular quadrature weights instead "
            "of uniform weighting."
        )
    )

    parser.add_argument(
        "--geometry-approx",
        default="spherical",
        choices=["spherical", "tangent_plane", "tangent_quadratic"],
        help=(
            "Approximation to be used in spherical geometry calculation. 'spherical' uses full spherical "
            "geometry but is slower to execute."
        )
    )

    parser.add_argument(
        "--nbins-fac",
        type=int,
        default=2,
        help=(
            "number of distance bins =  number of longitude points / nbins_fac,"
            "so nbins_fac = 2 gives a spacing of delta x over half a great circle"
        ) # should add a check to enforce that nbins must be <= # longitude points in half a great circle
    )

    parser.add_argument(
        "--include-w",
        action="store_true",
        help=(
            "Include vertical velocity in delta_u_cubed calculation?"
        )
    )

    parser.add_argument(
        "--write-buffer-mb",
        type=float,
        default=1000.0,
        help=(
            "Target amount of output data (in MB) to buffer in memory "
            "before writing to disk."
        )
    )

    return parser.parse_args()

def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(message)s"
        ),
    )

def load_geometry(geom_path, grid, chunk_origin, nlat, nlon, nbins_fac=4):
    geom_fpath = build_geometry_filename(
        geom_path,
        grid,
        trig_fns=True,
        nlat=nlat,
        nbins=nlon // nbins_fac,
        chunk_origin=chunk_origin,
    )
    
    ds_geom = xr.open_zarr(geom_fpath)
    origin_chunks = ds_geom.great_circle_distance.chunksizes["origin_latitude"]
    origin_lat_chunk_bounds = get_chunk_bounds(origin_chunks)
    distance_edges = np.array(
        ds_geom.great_circle_distance_bin.attrs["distance_bin_edges"]
    )
    distances = ( distance_edges[1:] + distance_edges[:-1] ) / 2

    return (
        ds_geom, distances, distance_edges, origin_lat_chunk_bounds,
    )

def load_velocity_field(
    velocity_file,
    pressure,
    time_index,
) -> xr.Dataset:

    ds = xr.open_dataset(
        velocity_file,
        decode_timedelta=False,
    )

    drop_vars = [
        v
        for v in [
            "forecast_reference_time",
            "forecast_period",
        ]
        if v in ds
    ]

    if drop_vars:
        ds = ds.drop_vars(drop_vars)

    ds = ds.isel(time=time_index)

    ptol = 0.5 # hPa; suitable for pressure values in the troposphere and stratosphere
    ds = ds.sel(pressure=pressure, method="nearest", tolerance=ptol)

    return ds

def create_du_cubed_template(
        pressure, time,
        distances, origin_latitudes, origin_longitudes,
        chunk_dist, chunk_lat, chunk_lon,
        dtype=np.float32,
        include_vertical=False
):
    variables = {
        "delta_u_cubed_angular_integral_longitudinal": (
            ("great_circle_distance", "origin_latitude", "origin_longitude"),
            #  may be better to have a different dimension ordering, but this is CF-compliant
            da.empty(
                (
                    len(distances),
                    len(origin_latitudes),
                    len(origin_longitudes),
                ),
                dtype=dtype,
                chunks=(
                    chunk_dist,
                    chunk_lat,
                    chunk_lon,
                ),
            ),
        ),
    }
    if include_vertical:
        variables["delta_u_cubed_angular_integral_vertical"] = (
            ("great_circle_distance", "origin_latitude", "origin_longitude"),
            #  may be better to have a different dimension ordering, but this is CF-compliant
            da.empty(
                (
                    len(distances),
                    len(origin_latitudes),
                    len(origin_longitudes),
                ),
                dtype=dtype,
                chunks=(
                    chunk_dist,
                    chunk_lat,
                    chunk_lon,
                ),
            ),
        )
    # should possibly modify to include time, pressure
    template = xr.Dataset(
        variables,
        coords={
            "great_circle_distance": distances,
            "origin_latitude": origin_latitudes,
            "origin_longitude": origin_longitudes,
            "pressure": pressure,
            "time": time,
        },
    )

    # add attributes
    # global
    template.attrs.update(
        {
            "title": "Angular integral of delta_u cubed",
            "project": "LoSSETT",
            "EARTH_RADIUS": RADIUS_EARTH,
        }
    )
    
    # coordinates
    template["great_circle_distance"].attrs = {
        "units": "m",
        "long_name": "Great-circle distance",
    }
    template["origin_latitude"].attrs = {
        "units": "degrees_north",
    }
    template["origin_longitude"].attrs = {
        "units": "degrees_east",
    }
    template["pressure"].attrs = {
        "units": "hPa",
        "long_name": "Pressure level",
    }
    template["time"].attrs = {
        "long_name": "Time (UTC)"
    }

    # data variable(s)
    template["delta_u_cubed_angular_integral_longitudinal"].attrs.update(
        {
            "units": "m3 s-3",
            "long_name": (
                "Angular integral of delta_u cubed (longitudinal)"
            ),
            "description": (
                "Locally angle-integrated longitudinal part of third-order "
                "velocity structure function. Velocity increments computed "
                "on pressure surfaces along great-circle displacements."
            )
        }
    )
    if include_vertical:
        template["delta_u_cubed_angular_integral_vertical"].attrs.update(
            {
                "units": "m3 s-3",
                "long_name": (
                    "Angular integral of delta_u cubed (transverse, vertical)"
                ),
                "description": "Locally angle-integrated vertical transverse part of third-order velocity structure function."
                "Velocity increments computed on pressure surfaces along great-circle displacements."
            }
        )
    return template

def get_chunk_bounds(chunksizes):
    chunk_bounds = []
    start = 0
    for chunk_size in chunksizes:
        stop = start + chunk_size
        chunk_bounds.append((start, stop))
        start = stop
    #endfor
    return chunk_bounds

def prepare_distance_selection(
    distance_bin,
    distance_edges,
    max_R=None,
):
    """
    Returns either:
        None              -> use full field
        active_indices    -> use indexed subset
    """

    if max_R is None:
        return None

    max_bin = np.searchsorted(
        distance_edges,
        max_R,
        side="right",
    ) - 1
    max_bin = min(
        max_bin,
        len(distance_edges) - 2
    )

    if max_bin < 0:
        # edge case handling
        return [
            (
                np.array([], dtype=int),
                np.array([], dtype=int),
            )
            for _ in range(distance_bin.shape[0])
        ]
    else:
        active_indices = []

        for i in range(distance_bin.shape[0]):
            active_indices.append(
                np.where(
                    distance_bin[i] <= max_bin
                )
            )

        return active_indices

def load_geometry_chunk(ds_geom, olat_chunk, distance_edges, max_R=None, profile=False):
    
    # read geometry chunk
    if profile:
        t0 = time.perf_counter()
    geom_chunk = ds_geom.isel(origin_latitude = slice(*olat_chunk)).load()
    if profile:
        logger.debug(
            f"Load geometry: "
            f"{time.perf_counter()-t0:.6f}s"
        )
    
    # extract indices where R <= max_R
    if profile:
        t0 = time.perf_counter()
    active_indices = prepare_distance_selection(
        geom_chunk.great_circle_distance_bin.values,
        distance_edges,
        max_R=max_R,
    )
    if profile:
        logger.debug(
            f"Compute active indices "
            f"{time.perf_counter()-t0:.6f}s"
        )
    return geom_chunk, active_indices

def process_origin_longitude(
    olon,
    u,
    v,
    geom_chunk,
    active_indices,
    distances,
    dtype=np.float64,
    use_angular_weights=False,
    profiler=None,
    method="spherical",
    w=None,
) -> xr.Dataset:
    """
    Compute azimuthally integrated delta-u^3 for a single
    origin longitude.

    Parameters
    ----------
    olon : float
        Origin longitude.

    u, v : xr.DataArray
        Velocity components on the analysis grid.

    geom_chunk : xr.Dataset
        Geometry information for the current chunk
        of origin latitudes.

    active_indices : list or None
        Optional spherical-cap selection.

    distances: np.ndarray
        Centres of great circle distance bins

    dtype:
        dtype for computing integral # THIS SHOULD PROBABLY BE REMOVED

    Returns
    -------
    xr.Dataset containing:
    
    delta_u_cubed_angular_integral_longitudinal
        Longitudinal contribution to (delta u)^3, locally azimuthally-averaged

    delta_u_cubed_angular_integral_vertical
        Vertical transverse contribution to (delta u)^3, locally azimuthally-averaged

    Dimensions: (origin_longitude, origin_latitude,
         great_circle_distance)
    """

    nbins = len(distances)
    lon_step = u.longitude.values[1] - u.longitude.values[0]
    lon_shift = int(round(olon / lon_step))

    # select u0, v0, init & final bearings
    u0 = u.sel(
        latitude=geom_chunk.origin_latitude,
        longitude=olon,
        method="nearest"
    )
    v0 = v.sel(
        latitude=geom_chunk.origin_latitude,
        longitude=olon,
        method="nearest"
    )
                
    # roll wind fields (need to check if actually faster than rolling geometry)
    if profiler:
        t0_roll = time.perf_counter()
    u_roll = u.roll(longitude=-lon_shift, roll_coords=False).load()
    v_roll = v.roll(longitude=-lon_shift, roll_coords=False).load()
    if w is not None:
        w0 = w.sel(
            latitude=geom_chunk.origin_latitude,
            longitude=olon,
            method="nearest"
        )
        w_roll = w.roll(longitude=-lon_shift, roll_coords=False).load()
    else:
        w0=None
        w_roll=None
    
    if profiler:
        profiler.add(
            "field roll",
            time.perf_counter() - t0_roll
        )

    # angular integration
    if profiler:
        t0_ang_int = time.perf_counter()
    if active_indices is None:
        if method != "spherical":
            print(
                f"Error: Cannot compute full sphere with method = {method}; "
                f"must use method = spherical.")
            sys.exit(1)
        # compute over the full sphere
        du_cubed_ang_int_long, du_cubed_ang_int_vert = \
            compute_du3_angular_integral_global(
                u_roll,
                v_roll,
                u0,
                v0,
                geom_chunk,
                nbins,
                dtype=dtype,
                use_angular_weights=use_angular_weights,
                w=w_roll,
                w0=w0,
                profiler=profiler,
            )
    else:
        # compute within a spherical cap of radius max_R
        du_cubed_ang_int_long, du_cubed_ang_int_vert = \
            compute_du3_angular_integral_subset(
                u_roll,
                v_roll,
                u0,
                v0,
                geom_chunk,
                active_indices,
                nbins, # should get nbins from geom_chunk
                dtype=dtype,
                use_angular_weights=use_angular_weights,
                method=method,
                w=w_roll,
                w0=w0,
                profiler=profiler,
            )
    #endif
    
    if profiler:
        profiler.add(
            "angular integration",
            time.perf_counter() - t0_ang_int
        )

    ds = xr.Dataset(
        data_vars = {
            "delta_u_cubed_angular_integral_longitudinal": (
                ("origin_latitude","great_circle_distance"),
                du_cubed_ang_int_long,
            ),
        },
        coords={
            "origin_latitude": geom_chunk.origin_latitude,
            "great_circle_distance": distances,
        },
    )
    if du_cubed_ang_int_vert is not None:
        ds["delta_u_cubed_angular_integral_vertical"] = (
            ("origin_latitude","great_circle_distance"),
            du_cubed_ang_int_vert,
        )
    
    return ds.expand_dims(
        origin_longitude=[olon]
    )

def write_origin_latitude_batch(
    batch,
    batch_start,
    batch_end,
    fpath,
    save_dtype,
):
    ds_write = xr.concat(
        batch,
        dim="origin_latitude"
    )

    ds_write = ds_write.astype(save_dtype)

    ds_write.to_zarr(
        fpath,
        region={
            "origin_latitude": slice(
                batch_start,
                batch_end,
            )
        }
    )

if __name__ == "__main__":
    # user input (TO DO: ADD OPTION TO READ YAML CONFIG FILE)
    args = parse_args()

    grid = args.grid
    chunk_origin = args.origin_lat_chunksize
    max_R = args.max_R
    if max_R is not None:
        # convert to m
        max_R = max_R * 1000. # should do some sensibility check here
        if max_R > 15e6: # disallow subsetting if max_R > 3/8 of a great circle circumference
            max_R = None
    include_vertical = args.include_w
    geom_path = args.geom_path
    save_path = args.save_path
    save_dtype = DTYPES[args.save_dtype]
    calc_dtype = DTYPES[args.calc_dtype]
    force = args.force
    profile = args.profile
    profiler = Profiler() if profile else None
    use_angular_weights = args.use_angular_weights
    nbins_fac = args.nbins_fac
    method = args.geometry_approx
    setup_logging(args.log_level)
    write_buffer_mb = args.write_buffer_mb
    write_buffer_bytes = int(write_buffer_mb * 1024**2)
    pressure = args.pressure
    time_index = args.time_index
    velocity_file = args.velocity_file
    outname_root = args.outname_root

    if outname_root is None:
        outname_root = Path(velocity_file).stem

    logger.info(
        "\n\n"
        "################################################################################\n"
        f"### LoSSETT version: {LOSSETT_VN} #######################################################\n"

        "### Function: compute_du_cubed_ang_int_spherical ###############################\n"    
        "################################################################################\n"
        "\n### CALCULATION INFO.\n"
        f"max_R = {(max_R if max_R is not None else RADIUS_EARTH * np.pi)/1e3:.6g} km\n"
        f"grid = {grid}\n" # to be deprecated -- user should just supply a uvw file
        f"source_file = {velocity_file}\n"
        f"geometry archive directory = {geom_path}\n"
        f"output directory = {save_path}\n"
        f"outfile name root = {outname_root}\n"
        f"origin_latitude chunksize = {chunk_origin}\n"
        f"dtype (calculation) = {calc_dtype}\n"
        f"dtype (output) = {save_dtype}\n"
        f"force = {force}\n"
        f"include_vertical = {include_vertical}\n"
        f"pressure = {pressure} hPa\n"
        f"time index = {time_index}"
    )

    # construct regular lat-lon grid
    lon_step, lat_step = GRID_DEFS[grid]
    lons, lats = build_regular_latlon_grid(lon_step, lat_step)
    origin_lons = lons
    chunk_lat = len(lats)
    chunk_lon = len(lons)

    # load geometry from Zarr store
    ds_geom, distances, distance_edges, origin_lat_chunk_bounds = load_geometry(
        geom_path,
        grid,
        chunk_origin,
        nlat=len(lats),
        nlon=len(lons),
        nbins_fac=nbins_fac
    )

    # load velocity field
    ds_u = load_velocity_field(
        velocity_file,
        pressure,
        time_index,
    )

    # ensure grid consistency
    # latitudes must match geometry exactly
    if not np.allclose(
            ds_u.latitude.values,
            lats,
    ):
        raise ValueError(
            f"Velocity file latitude coordinates do not match grid {grid}."
        )

    # longitudes must have same number of points with same uniform spacing as geometry
    if len(ds_u.longitude) != len(lons):
        raise ValueError(
            f"Velocity file longitude coordinate length {len(ds_u.longitude)}, which "
            f"does not match {grid} grid specification (length = {len(lons)}."
        )
    dlon = np.diff(ds_u.longitude.values)
    if not np.allclose(
            dlon,
            dlon[0],
    ):
        raise ValueError(
            "Velocity file longitude coordinate is not uniformly spaced."
        )

    if not np.isclose(
            dlon[0],
            lon_step,
    ):
        raise ValueError(
            f"Velocity file longitude spacing ({dlon[0]}) does not match expected "
            f"spacing ({lon_step})."
        )

    # extract velocity components
    u = ds_u.u
    v = ds_u.v
    if include_vertical:
        if "w" not in ds_u:
            logger.error(
                "--include-w specified but variable 'w' "
                "not present in velocity file."
            )
            sys.exit(1)
        w = ds_u.w
    else:
        w = None

    # extract pressure and time values
    pressure_value = ds_u.pressure.values
    time_value = ds_u.time.values

    # create Zarr store for azimuthally-integrated delta u cubed
    if max_R is None:
        maxR_str = "_maxR_global"
    else:
        maxR_str = f"_maxR_{int(max_R/1e3):05d}"
    #endif
    if method != "spherical":
        method_str = f"_{method}"
    else:
        method_str = ""
    time_str = np.datetime_as_string(
        time_value,
        unit="h"
    ).replace("-", "").replace(":", "")

    du3_fpath = os.path.join(
        save_path,
        f"{outname_root}"
        f"_{grid}"
        f"_delta_u_cubed"
        f"_t{time_str}"
        f"_p{int(pressure):04d}hPa"
        f"{maxR_str}"
        f"{method_str}.zarr"
    )
    logger.info(
        f"\nOutput file = {du3_fpath}"
    )
    chunk_dist = -1

    if not os.path.exists(du3_fpath) or force:
            
        du3_template = create_du_cubed_template(
            pressure_value, time_value,
            distances, lats, lons,
            chunk_dist, chunk_lat, chunk_origin,
            dtype=save_dtype,
            include_vertical=include_vertical
        )
        du3_template.attrs.update(
            {
                "source_file": velocity_file,
                "source_attributes": repr(u.attrs),
                "lossett_version": LOSSETT_VN,
                "run_command": " ".join(sys.argv),
                "arguments": repr(vars(args)),
                "history": f"{datetime.now(UTC).isoformat()}: "
                "Created by compute_delta_u_cubed_spherical.py"
            }
        )
        du3_template.to_zarr(
            du3_fpath,
            mode="w",
            compute=False,
            zarr_format=2,
        )

        # initialise buffer for caching output before writing
        output_batch = []
        batch_nbytes = 0
        batch_start = None
        batch_end = None

        logger.info("\nEntering latitude loop")
        t0_global = time.perf_counter()

        for olat_chunk in origin_lat_chunk_bounds:
            if profiler:
                t0_lat_chunk = time.perf_counter()
                
            lat_start = (
                ds_geom
                .origin_latitude
                .isel(origin_latitude=olat_chunk[0])
                .values
            )
            lat_end = (
                ds_geom
                .origin_latitude
                .isel(origin_latitude=olat_chunk[1]-1)
                .values
            )
            logger.info(f"\n\nOrigin  latitudes {lat_start} -- {lat_end}")
            
            if profiler:
                t0_geom = time.perf_counter()

            geom_chunk, active_indices = load_geometry_chunk(
                ds_geom, olat_chunk, distance_edges, max_R=max_R
            )
            if profiler:
                profiler.add(
                    "geometry load",
                    time.perf_counter() - t0_geom,
                )

            du_cubed_ang_int =[]
            for ilon, olon in enumerate(origin_lons):
                #if profile:
                #    if ilon == 1:
                #        # allow Numba to compile on the first longitude
                #        prof = cProfile.Profile()
                #        prof.enable()
                logger.debug(f"\nOrigin longitude = {olon}")
                du_cubed_ang_int.append(
                    process_origin_longitude(
                        olon,
                        u,
                        v,
                        geom_chunk,
                        active_indices,
                        distances,
                        dtype=calc_dtype,
                        use_angular_weights=use_angular_weights,
                        method=method,
                        w=w,
                        profiler=profiler,
                    )
                )
                #if profile:
                #    if ilon == 1:
                #        prof.disable()
                #        stats = pstats.Stats(prof)
                #        stats.sort_stats("cumtime")
                #        stats.print_stats(50)
                #        sys.exit(1)
            #endfor
        
            if profiler:
                profiler.add(
                    "origin lon loop",
                    time.perf_counter() - t0_lat_chunk,
                )

            du_cubed_ang_int = xr.concat(
                du_cubed_ang_int,
                dim=xr.DataArray(
                    origin_lons,
                    dims="origin_longitude",
                    name="origin_longitude"
                )
            )
            save_vars = {
                "delta_u_cubed_angular_integral_longitudinal": (
                    du_cubed_ang_int["delta_u_cubed_angular_integral_longitudinal"].dims,
                    du_cubed_ang_int[
                        "delta_u_cubed_angular_integral_longitudinal"
                    ].data
                )
            }

            if include_vertical:
                save_vars["delta_u_cubed_angular_integral_vertical"] = (
                    du_cubed_ang_int["delta_u_cubed_angular_integral_vertical"].dims,
                    du_cubed_ang_int[
                        "delta_u_cubed_angular_integral_vertical"
                    ].data
                )
            
            # create Dataset to write (just save data variables to avoid write errors)
            ds_chunk = xr.Dataset(
                save_vars
            ).reset_coords(drop=True)

            # memory buffer handling
            if batch_start is None:
                batch_start = olat_chunk[0]
            batch_end = olat_chunk[1]
            output_batch.append(ds_chunk)
            batch_nbytes += ds_chunk.to_array().nbytes
            logger.info(
                f"\nBuffered {batch_nbytes/1024**2:.1f} MiB"
            )

            if batch_nbytes >= write_buffer_bytes:
                if profiler:
                    t0_write = time.perf_counter()
                logger.info(
                    "\nSaving output batch for origin_latitude indices "
                    f"{batch_start} to {batch_end}"
                )
                write_origin_latitude_batch(
                    output_batch,
                    batch_start,
                    batch_end,
                    du3_fpath,
                    save_dtype,
                )

                # reset batch variables
                output_batch.clear()
                batch_nbytes = 0
                batch_start = None
                batch_end = None
                if profiler:
                    profiler.add(
                        "batch write",
                        time.perf_counter() - t0_write,
                    )
            #endif
            
            if profiler:
                profiler.add(
                    "total",
                    time.perf_counter() - t0_lat_chunk
                )
                # print profiling
                profiler.report(
                    logger,
                    total_key="total",
                )
                profiler.reset()
            
        #endfor
        # Write any remaining data
        if output_batch:
            logger.info(
                "\n\nSaving output batch for origin_latitude indices "
                f"{batch_start} to {batch_end}"
            )
            write_origin_latitude_batch(
                output_batch,
                batch_start,
                batch_end,
                du3_fpath,
                save_dtype,
            )
        logger.info(
            "\n\nTotal time in computation: "
            f"{time.perf_counter()-t0_global:.6f}"
        )

    logger.info("\n\nEND.")
