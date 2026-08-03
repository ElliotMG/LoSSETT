#!/usr/bin/env python3
import os
import numpy as np
import xarray as xr
import dask.array as da
import time
import argparse
import logging
from importlib.metadata import version

from lossett.calc.compute_spherical_geometry import build_geometry_filename, build_regular_latlon_grid, GRID_DEFS, RADIUS_EARTH, DTYPES

LOSSETT_VN = version("lossett")

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

def load_velocity_field(date="20160801", interp_lats=None, interp_lons=None):
    ds_n1280 = xr.open_dataset(
        "/gws/ssde/j25b/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/"\
        f"DYAMOND_SUMMER/glm.n1280_GAL9.uvw_{date}T00.nc",
        decode_timedelta=False # since we're immediately dropping the timedelta variables
    ).drop_vars(["forecast_reference_time","forecast_period"])
    lon_attrs = ds_n1280.longitude.attrs
    ds_n1280.coords["longitude"] = (ds_n1280.coords["longitude"] + 180) % 360 - 180
    ds_n1280 = ds_n1280.sortby(ds_n1280.longitude)
    times = ds_n1280.time
    pressures = ds_n1280.pressure
    ds_n1280 = ds_n1280.isel(time=0).sel(pressure=200)

    u_n1280 = ds_n1280.u
    v_n1280 = ds_n1280.v

    if interp_lats is not None or interp_lons is not None:
        # interpolate to coarser grid for testing
        u = u_n1280.interp(latitude=interp_lats, longitude=interp_lons)
        v = v_n1280.interp(latitude=interp_lats, longitude=interp_lons)
    else:
        u = u_n1280
        v = v_n1280
    
    return u,v

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
            "delta_u_cubed_angular_average": (
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

def compute_delta_u_cubed(u, v, u0, v0, sin_init, cos_init, sin_final, cos_final, w=None):
    # velocity increment tangent to geodesic
    du_t = (
        u*sin_final
        + v*cos_final
        - u0*sin_init
        - v0*cos_init
    )
    # velocity increment normal to geodesic
    du_n = (
        u*cos_final
        -v*sin_final
        -u0*cos_init
        +v0*sin_init
    )
    # compute delta u cubed dot rhat
    du_cubed = du_t * (du_t*du_t + du_n*du_n)

    del du_t
    del du_n

    ### TO DO: ADD VERTICAL VELOCITY CAPABILITY
    
    return du_cubed

def bin_average(values, bins, nbins, weights=None):

    if weights is None:
        weights = np.ones_like(values)
        
    sum_bin = np.bincount(
        bins,
        weights=values,
        minlength=nbins,
    )

    count_bin = np.bincount(
        bins,
        minlength=nbins,
    )
    
    if len(count_bin) != nbins:
        # remove overflow bins (SHOULD PROBABLY CHECK FOR BOTH OVERFLOW AND UNDERFLOW)
        sum_bin = sum_bin[:nbins]
        count_bin = count_bin[:nbins]

    return np.divide(
        sum_bin,
        count_bin,
        out=np.full(nbins, np.nan),
        where=count_bin > 0,
    )

def compute_du3_angular_average_global(
        u,
        v,
        u0,
        v0,
        geom_chunk,
        nbins,
        dtype=np.float32,
        weights=None,
        profile=False
):
    """
    Inputs
    ----------
    u, v
        Rolled wind fields
        (latitude, longitude)

    u0, v0
        Origin winds
        (origin_latitude,)

    geom_chunk
        Geometry chunk containing:
            distance_bin
            sin_init
            cos_init
            sin_final
            cos_final

    nbins
        Number of bins

    Returns
    -------
    means
        (origin_latitude, nbins)
    """
    chunk_len = geom_chunk.sizes["origin_latitude"]
                
    # compute du_cubed
    if profile:
        t0 = time.perf_counter()
    du_cubed = compute_delta_u_cubed(
        u, v, u0, v0,
        geom_chunk.sine_initial_bearing, geom_chunk.cosine_initial_bearing,
        geom_chunk.sine_final_bearing, geom_chunk.cosine_final_bearing,
        w=None
    ).load()
    if profile:
        logger.debug(
            f"du_cubed: "
            f"{time.perf_counter()-t0:.6f}s"
        )
    
    # compute angular average (this should be a function)
    if profile:
        t0 = time.perf_counter()
    means = np.empty((chunk_len, nbins), dtype=dtype)
    for i in range(chunk_len):
        vals = du_cubed.isel(origin_latitude=i).values.ravel()
        bins = geom_chunk.great_circle_distance_bin.isel(origin_latitude=i).values.ravel()
        means[i] = 2*np.pi*bin_average(vals, bins, nbins, weights=weights)
    #endfor
    if profile:
        logger.debug(
            f"angular average (np.bincount): "
            f"{time.perf_counter()-t0:.6f}s"
        )
        
    # clean up
    del du_cubed
            
    return means

def compute_du3_angular_average_subset(
        u,
        v,
        u0,
        v0,
        geom_chunk,
        active_indices,
        nbins,
        dtype=np.float64,
        weights=None,
        profile=False
):
    """
    Inputs
    ----------
    u, v
        Rolled wind fields
        (latitude, longitude)

    u0, v0
        Origin winds
        (origin_latitude,)

    geom_chunk
        Geometry chunk containing:
            distance_bin
            sin_init
            cos_init
            sin_final
            cos_final

    active_indices
        List of (ilat, ilon) tuples,
        one per origin latitude.

    Returns
    -------
    means
        (origin_latitude, nbins)
    """

    if profile:
        t0 = time.perf_counter()
    
    nchunk = len(active_indices)

    means = np.empty(
        (nchunk, nbins),
        dtype=dtype,
    )

    for i, (ilat, ilon) in enumerate(active_indices):

        # gather only active points
        u_sel = u.values[ilat, ilon]
        v_sel = v.values[ilat, ilon]

        u0_sel = u0.values[i]
        v0_sel = v0.values[i]

        sin_init_sel = geom_chunk.sine_initial_bearing.values[
            i, ilat, ilon
        ]
        cos_init_sel = geom_chunk.cosine_initial_bearing.values[
            i, ilat, ilon
        ]

        sin_final_sel = geom_chunk.sine_final_bearing.values[
            i, ilat, ilon
        ]
        cos_final_sel = geom_chunk.cosine_final_bearing.values[
            i, ilat, ilon
        ]

        bins = geom_chunk.great_circle_distance_bin.values[
            i, ilat, ilon
        ]

        # compute du^3
        du3 = compute_delta_u_cubed(
            u_sel, v_sel,
            u0_sel, v0_sel,
            sin_init_sel, cos_init_sel,
            sin_final_sel, cos_final_sel,
            w=None
        )

        # accumulate by bin
        means[i] = 2*np.pi*bin_average(du3, bins, nbins, weights=weights)
        #means[i] = bin_average(du3, bins, nbins, weights=weights)

    #endfor
    if profile:
        logger.debug(
            f"du_cubed AND angular average: "
            f"{time.perf_counter()-t0:.6f}s"
        )
    
    return means

def process_origin_longitude(
    olon,
    #i_olon,
    u,
    v,
    geom_chunk,
    active_indices,
    distances,
    dtype=np.float64,
    weights=None,
    profile=False
) -> xr.DataArray:
    """
    Compute azimuthally averaged delta-u^3 for a single
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
        dtype for computing means # THIS SHOULD PROBABLY BE REMOVED

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
        du_cubed_ang_av = compute_du3_angular_average_global(
            u_roll,
            v_roll,
            u0,
            v0,
            geom_chunk,
            nbins,
            dtype=dtype,
            weights=weights
        )
    else:
        # compute within a spherical cap of radius max_R
        du_cubed_ang_av = compute_du3_angular_average_subset(
            u_roll,
            v_roll,
            u0,
            v0,
            geom_chunk,
            active_indices,
            nbins, # should get nbins from geom_chunk
            dtype=dtype,
            weights=weights
        )
    #endif
    return xr.DataArray(
        du_cubed_ang_av,
        dims=(
            "origin_latitude",
            "great_circle_distance",
        ),
        coords={
            "origin_latitude": geom_chunk.origin_latitude,
            "great_circle_distance": distances,
        },
        name="delta_u_cubed_angular_average",
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
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    date = "20160801"

    logger.info(
        "\n\n"
        "################################################################################\n"
        f"### LoSSETT version: {LOSSETT_VN} #######################################################\n"

        "### Function: compute_du_cubed_ang_av_spherical ################################\n"    
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
    u, v = load_velocity_field(date=date, interp_lons=lons, interp_lats=lats)

    # create Zarr store for azimuthally-averaged delta u cubed
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
        du3_template.to_zarr(
            du3_fpath,
            mode="w",
            compute=False,
            zarr_format=2,
        )

        # generate cos(lat) weights for area weighting
        lat_weights_2d = np.cos(
            np.deg2rad(u.latitude.values)
        )[:, None]
        lat_weights = np.broadcast_to(
            lat_weights_2d,
            u.shape
        ).ravel()

        for olat_chunk in origin_lat_chunk_bounds:
            lat_start = ds_geom.origin_latitude.isel(origin_latitude=olat_chunk[0]).values
            lat_end = ds_geom.origin_latitude.isel(origin_latitude=olat_chunk[1]-1).values
            logger.info(f"\n\nOrigin  latitudes {lat_start} -- {lat_end}")
            geom_chunk, active_indices = load_geometry_chunk(
                ds_geom, olat_chunk, distance_edges, max_R=max_R
            )

            du_cubed_ang_av =[]
            #for i_olon, olon in enumerate(origin_lons):
            for olon in origin_lons:
                logger.debug(f"\nOrigin longitude = {olon}")
                du_cubed_ang_av.append(
                    process_origin_longitude(
                        olon,
                        #i_olon,
                        u,
                        v,
                        geom_chunk,
                        active_indices,
                        distances,
                        dtype=calc_dtype,
                        weights=lat_weights
                    )
                )                
            #endfor
            ds_out = xr.Dataset(
                {
                    "delta_u_cubed_angular_average": xr.concat(
                        du_cubed_ang_av,
                        dim=xr.DataArray(
                            origin_lons,
                            dims="origin_longitude",
                            name="origin_longitude"
                        )
                    )
                }
            )

            da = ds_out.delta_u_cubed_angular_average.astype(save_dtype)
            
            # save latitude chunk (just save the data variable to avoid write errors)
            ds_write = xr.Dataset(
                {
                    "delta_u_cubed_angular_average": (da.dims, da.data)
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
