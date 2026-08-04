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

from lossett.calc.compute_spherical_geometry import (
    build_geometry_filename, build_regular_latlon_grid,
    GRID_DEFS, RADIUS_EARTH, DTYPES
)
from lossett.calc.field_increments import (
    compute_du3_angular_integral_global,
    compute_du3_angular_integral_subset,
)
from lossett.profiling import (
    profile_block
)

# Module-scope variables
LOSSETT_VN = version("lossett")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--grid",
        required=True,
        choices=GRID_DEFS.keys(),
        help="Grid definition (must be regular lat-lon)"
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

def load_geometry(geom_path, grid, chunk_origin, nlat, nlon):
    geom_fpath = build_geometry_filename(
        geom_path,
        grid,
        trig_fns=True,
        nlat=nlat,
        nbins=nlon // 4,
        chunk_origin=chunk_origin,
    )
    
    ds_geom = xr.open_zarr(geom_fpath)
    origin_chunks = ds_geom.great_circle_distance.chunksizes["origin_latitude"]
    origin_lat_chunk_bounds = get_chunk_bounds(origin_chunks)
    distance_edges = np.array(ds_geom.attrs["distance_bin_edges_m"])
    distances = ( distance_edges[1:] + distance_edges[:-1] ) / 2
    nbins = len(distances)

    return (
        ds_geom, distances, distance_edges, origin_lat_chunk_bounds, nbins
    )

def load_velocity_field(date="20160801", interp_lats=None, interp_lons=None, return_fpath=True):
    # getting the file path should be another function
    # this function should take fpath as an argument & just do the loading & tidying
    u_fpath = "/gws/ssde/j25b/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/"\
        f"DYAMOND_SUMMER/glm.n1280_GAL9.uvw_{date}T00.nc"
    ds_u = xr.open_dataset(
        u_fpath,
        decode_timedelta=False # since we're immediately dropping the timedelta variables
    ).drop_vars(["forecast_reference_time","forecast_period"])
    lon_attrs = ds_u.longitude.attrs
    ds_u.coords["longitude"] = (ds_u.coords["longitude"] + 180) % 360 - 180
    ds_u = ds_u.sortby(ds_u.longitude)
    ds_u.longitude.attrs = lon_attrs
    #times = ds_u.time
    #pressures = ds_u.pressure
    ds_u = ds_u.isel(time=0).sel(pressure=200)

    u = ds_u.u
    v = ds_u.v

    if interp_lats is not None or interp_lons is not None:
        # interpolate to coarser grid for testing
        u = u.interp(latitude=interp_lats, longitude=interp_lons)
        v = v.interp(latitude=interp_lats, longitude=interp_lons)

    if return_fpath:
        return u, v, u_fpath
    else:
        return u, v

def create_du_cubed_template(
        distances, origin_latitudes, origin_longitudes,
        chunk_dist, chunk_lat, chunk_lon,
        #distances, times,  pressures,  origin_latitudes, origin_longitudes,
        #chunk_dist, chunk_time, chunk_pressure, chunk_lat, chunk_lon,
        dtype=np.float32
):
    # should possibly modify to include time, pressure
    template = xr.Dataset(
        {
            "delta_u_cubed_angular_integral": (
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
        },
        coords={
            "great_circle_distance": distances,
            "origin_latitude": origin_latitudes,
            "origin_longitude": origin_longitudes,
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

    # data variable
    template["delta_u_cubed_angular_integral"].attrs.update(
        {
            "units": "m3 s-3",
            "long_name": (
                "Angular integral of delta_u cubed"
            ),
            "description": "Locally angle-integrated third-order longitudinal velocity structure function."
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
    #i_olon,
    u,
    v,
    geom_chunk,
    active_indices,
    distances,
    dtype=np.float64,
    use_angular_weights=False,
    profile=False
) -> xr.DataArray:
    """
    Compute azimuthally integrated delta-u^3 for a single
    origin longitude.

    Parameters
    ----------
    olon : float
        Origin longitude.

    i_olon : int
        Index of origin longitude

    u, v : xr.DataArray
        Velocity components on the analysis grid.

    geom_chunk : xr.Dataset
        Geometry information for the current chunk
        of origin latitudes.

    active_indices : list or None
        Optional spherical-cap selection.

    nbins: int
        Number of great-circle distance bins

    distances: np.ndarray
        Centres of great circle distance bins

    dtype:
        dtype for computing integral # THIS SHOULD PROBABLY BE REMOVED

    Returns
    -------
    xr.DataArray
        (origin_longitude, origin_latitude,
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
    if profile:
        t0 = time.perf_counter()
    u_roll = u.roll(longitude=-lon_shift, roll_coords=False).load()
    v_roll = v.roll(longitude=-lon_shift, roll_coords=False).load()
    if profile:
        logger.debug(
            f"Roll: "
            f"{time.perf_counter()-t0:.6f}s"
        )
    
    if active_indices is None:
        # compute over the full sphere
        du_cubed_ang_int = compute_du3_angular_integral_global(
            u_roll,
            v_roll,
            u0,
            v0,
            geom_chunk,
            nbins,
            dtype=dtype,
            use_angular_weights=use_angular_weights,
        )
    else:
        # compute within a spherical cap of radius max_R
        du_cubed_ang_int = compute_du3_angular_integral_subset(
            u_roll,
            v_roll,
            u0,
            v0,
            geom_chunk,
            active_indices,
            nbins, # should get nbins from geom_chunk
            dtype=dtype,
            use_angular_weights=use_angular_weights,
        )
    #endif
    return xr.DataArray(
        du_cubed_ang_int,
        dims=(
            "origin_latitude",
            "great_circle_distance",
        ),
        coords={
            "origin_latitude": geom_chunk.origin_latitude,
            "great_circle_distance": distances,
        },
        name="delta_u_cubed_angular_integral",
    ).expand_dims(
        origin_longitude=[olon]
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
    geom_path = args.geom_path
    save_path = args.save_path
    save_dtype = DTYPES[args.save_dtype]
    calc_dtype = DTYPES[args.calc_dtype]
    force = args.force
    profile = args.profile
    use_angular_weights = args.use_angular_weights
    setup_logging(args.log_level)
    date = "20160801"

    logger.info(
        "\n\n"
        "################################################################################\n"
        f"### LoSSETT version: {LOSSETT_VN} #######################################################\n"

        "### Function: compute_du_cubed_ang_int_spherical ###############################\n"    
        "################################################################################\n"
        "\n### CALCULATION INFO.\n"
        f"max_R = {(max_R if max_R is not None else RADIUS_EARTH * np.pi)/1e3:.6g} km\n"
        f"grid = {grid}\n" # to be deprecated -- user should just supply a uvw file
        f"source_file = TO_BE_IMPLEMENTED\n"
        f"geometry archive directory = {geom_path}\n"
        f"output directory = {save_path}\n"
        f"origin_latitude chunksize = {chunk_origin}\n"
        f"dtype (calculation) = {calc_dtype}\n"
        f"dtype (output) = {save_dtype}\n"
        f"force = {force}\n"
    )

    # construct regular lat-lon grid
    lon_step, lat_step = GRID_DEFS[grid]
    lons, lats = build_regular_latlon_grid(lon_step, lat_step)
    origin_lons = lons
    chunk_lat = len(lats)
    chunk_lon = len(lons)

    # load geometry from Zarr store
    ds_geom, distances, distance_edges, origin_lat_chunk_bounds, nbins = load_geometry(
        geom_path, grid, chunk_origin, nlat=len(lats), nlon=len(lons)
    )

    # load velocity field
    u, v, u_fpath = load_velocity_field(date=date, interp_lons=lons, interp_lats=lats, return_fpath=True)

    # create Zarr store for azimuthally-integrated delta u cubed
    if max_R is None:
        maxR_str = ""
    else:
        maxR_str = f"_maxR_{int(max_R/1e3):05d}"
    #endif
    du3_fpath = os.path.join(
        save_path,
        f"glm.n1280_GAL9_DS_{date}T00_inter_scale_transfer_of_kinetic_energy_p0200hPa_{grid}{maxR_str}.zarr"
    )
    chunk_dist = -1
    chunk_time = 1
    chunk_pressure = 1

    if not os.path.exists(du3_fpath) or force:
        du3_template = create_du_cubed_template(
            distances, lats, lons,
            chunk_dist, chunk_lat, chunk_origin,
            dtype=save_dtype
        )
        du3_template.attrs.update(
            {
                "source_file": os.path.basename(u_fpath),
                "source_attributes": repr(u.attrs),
                "lossett_version": LOSSETT_VN,
                "run_command": " ".join(sys.argv),
                "arguments": repr(vars(args)),
                "history": f"{datetime.now(UTC).isoformat()}: Created by compute_inter_scale_transfers_spherical.py"
            }
        )
        du3_template.to_zarr(
            du3_fpath,
            mode="w",
            compute=False,
            zarr_format=2,
        )

        for olat_chunk in origin_lat_chunk_bounds:
            lat_start = ds_geom.origin_latitude.isel(origin_latitude=olat_chunk[0]).values
            lat_end = ds_geom.origin_latitude.isel(origin_latitude=olat_chunk[1]-1).values
            logger.info(f"\n\nOrigin  latitudes {lat_start} -- {lat_end}")
            geom_chunk, active_indices = load_geometry_chunk(
                ds_geom, olat_chunk, distance_edges, max_R=max_R
            )

            du_cubed_ang_int =[]
            for olon in origin_lons:
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
                        profile=profile,
                    )
                )                
            #endfor
            ds_out = xr.Dataset(
                {
                    "delta_u_cubed_angular_integral": xr.concat(
                        du_cubed_ang_int,
                        dim=xr.DataArray(
                            origin_lons,
                            dims="origin_longitude",
                            name="origin_longitude"
                        )
                    )
                }
            )

            da = ds_out.delta_u_cubed_angular_integral.astype(save_dtype)
            
            # save latitude chunk (just save the data variable to avoid write errors)
            ds_write = xr.Dataset(
                {
                    "delta_u_cubed_angular_integral": (da.dims, da.data)
                }
            ).reset_coords(drop=True)
            
            logger.info("\nSaving chunk")
            ds_write.to_zarr(
                du3_fpath,
                region = {
                    "origin_latitude": slice(*olat_chunk)
                }
            )
        #endfor

    logger.info("\n\nEND.")
