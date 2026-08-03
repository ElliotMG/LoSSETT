#!/usr/bin/env python3
import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy as cpy
import dask.array as da
import time
from compute_spherical_geometry import build_geometry_filename, build_regular_latlon_grid, GRID_DEFS, RADIUS_EARTH

RADIUS_EARTH = 6371000 # Earth radius in metres

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
    return template;

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
    
    return du_cubed;

def get_chunk_bounds(chunksizes):
    chunk_bounds = []
    start = 0
    for chunk_size in chunksizes:
        stop = start + chunk_size
        chunk_bounds.append((start, stop))
        start = stop
    #endfor
    return chunk_bounds;

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

    max_bin = (
        np.searchsorted(
            distance_edges,
            max_R,
            side="right",
        ) - 1
    )

    active_indices = []

    for i in range(distance_bin.shape[0]):
        active_indices.append(
            np.where(
                distance_bin[i] <= max_bin
            )
        )

    return active_indices;

def reduce_using_indices(
        _u,
        _v,
        u0,
        v0,
        geom_chunk,
        active_indices,
        nbins,
        dtype=np.float32
):
    """
    Parameters
    ----------
    _u, _v
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

    nchunk = len(active_indices)

    means = np.empty(
        (nchunk, nbins),
        dtype=dtype,
    )

    for i, (ilat, ilon) in enumerate(active_indices):

        # gather only active points

        u_sel = _u.values[ilat, ilon]
        v_sel = _v.values[ilat, ilon]

        sin_i = geom_chunk.sine_initial_bearing.values[
            i, ilat, ilon
        ]
        cos_i = geom_chunk.cosine_initial_bearing.values[
            i, ilat, ilon
        ]

        sin_f = geom_chunk.sine_final_bearing.values[
            i, ilat, ilon
        ]
        cos_f = geom_chunk.cosine_final_bearing.values[
            i, ilat, ilon
        ]

        bins = geom_chunk.great_circle_distance_bin.values[
            i, ilat, ilon
        ]

        u0_i = u0.values[i]
        v0_i = v0.values[i]

        # compute du_t

        du_t = (
            u_sel * sin_f
            + v_sel * cos_f
            - u0_i * sin_i
            - v0_i * cos_i
        )

        # compute du_n

        du_n = (
            u_sel * cos_f
            - v_sel * sin_f
            - u0_i * cos_i
            + v0_i * sin_i
        )

        # compute du³

        du3 = du_t * (
            du_t * du_t
            + du_n * du_n
        )

        # accumulate by bin

        sum_bin = np.bincount(
            bins,
            weights=du3,
            minlength=nbins,
        )

        count_bin = np.bincount(
            bins,
            minlength=nbins,
        )
        if len(count_bin) == nbins:
            means[i] = sum_bin / count_bin
        else:
            means[i] = sum_bin[:-1] / count_bin[:-1] # removing overflow bin
        #endif
    #endfor
    
    return means;

if __name__ == "__main__":
    # user input (take from command line or YAML)
    grid = "1p0deg"
    #grid = "n320"
    chunk_origin = 4

    date = "20160801"
    #date = "20160815"
    
    geom_path = "/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/"
    save_path = "/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/"

    # user-defined maximum great-circle distance
    max_R = None # whole globe
    #max_R = 5000*1e6
    #max_R = 2000*1e6
    #max_R = 1000*1e6

    # open geometry from Zarr store
    lon_step, lat_step = GRID_DEFS[grid]
    lons, lats = build_regular_latlon_grid(lon_step, lat_step)
    origin_lons = lons
    chunk_lat = len(lats)
    chunk_lon = len(lons)
    
    geom_fpath = build_geometry_filename(
        save_path,
        grid,
        trig_fns=True,
        nlat=len(lats),
        nbins=len(lons) // 4,
        #nbins=len(lons) // 2,
        #nbins=len(lons), # THIS SHOULD BE len(lons) // 2 BUT CURRENT GEOMEETRY ARCHIVES OMITTED THE FACTOR OF 2
        chunk_origin=chunk_origin,
    )
    
    ds_geom = xr.open_zarr(geom_fpath)
    origin_chunks = ds_geom.great_circle_distance.chunksizes["origin_latitude"]
    origin_lat_chunk_bounds = get_chunk_bounds(origin_chunks)
    distance_edges = np.array(ds_geom.attrs["distance_bin_edges_m"])
    distances = ( distance_edges[1:] + distance_edges[:-1] ) / 2
    nbins = len(distances)

    # load velocity field
    ds_n1280 = xr.open_dataset(
        "/gws/ssde/j25b/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/"\
        f"DYAMOND_SUMMER/glm.n1280_GAL9.uvw_{date}T00.nc"
    ).drop_vars(["forecast_reference_time","forecast_period"])
    lon_attrs = ds_n1280.longitude.attrs
    ds_n1280.coords["longitude"] = (ds_n1280.coords["longitude"] + 180) % 360 - 180
    ds_n1280 = ds_n1280.sortby(ds_n1280.longitude)
    times = ds_n1280.time
    pressures = ds_n1280.pressure
    ds_n1280 = ds_n1280.isel(time=0).sel(pressure=200)

    u_n1280 = ds_n1280.u
    v_n1280 = ds_n1280.v

    # interpolate to coarser grid for testing
    u = u_n1280.interp(latitude=lats, longitude=lons)
    v = v_n1280.interp(latitude=lats, longitude=lons)

    # create Zarr store for azimuthally-averaged delta u cubed
    if max_R is None:
        maxR_str = ""
    else:
        maxR_str = f"_maxR_{int(max_R/1e3):05d}"
    #endif
    du3_fpath = os.path.join(
        save_path,
        f"glm.n1280_GAL9_DS_{date}T00_inter_scale_transfer_of_kinetic_energy_p0200hPa_{grid}{maxR_str}_OLD_WORKING.zarr"
    )
    chunk_dist = -1
    chunk_time = 1
    chunk_pressure = 1
    du3_template = create_du_cubed_template(
        distances, lats, lons,
        chunk_dist, chunk_lat, chunk_origin,
        #distances, times,  pressures, lats, lons,
        #chunk_dist, chunk_time, chunk_pressure, chunk_lat, chunk_origin,
        dtype=np.float32
    )
    du3_template.to_zarr(
        du3_fpath,
        mode="w",
        compute=False,
        zarr_format=2,
    )

    # This should be a function: compute_du_cubed_angular_average

    for olat_chunk in origin_lat_chunk_bounds:
        lat_start = ds_geom.origin_latitude.isel(origin_latitude=olat_chunk[0]).values
        lat_end = ds_geom.origin_latitude.isel(origin_latitude=olat_chunk[1]-1).values
        chunk_len = olat_chunk[1] - olat_chunk[0]
        print(f"\n\nOrigin  latitudes {lat_start} -- {lat_end}")
        # for timing
        t0 = time.perf_counter()
        # read geometry chunk
        geom_chunk = ds_geom.isel(origin_latitude = slice(*olat_chunk)).load()
        print(
            f"Load geometry: "
            f"{time.perf_counter()-t0:.6f}s"
        )

        # only compute du_cubed for R <= max_R
        t0 = time.perf_counter()
        active_indices = prepare_distance_selection(
            geom_chunk.great_circle_distance_bin.values,
            distance_edges,
            max_R=max_R,
        )
        print(
            f"Compute active indices "
            f"{time.perf_counter()-t0:.6f}s"
        )

        du_cubed_ang_av =[]
        for olon in origin_lons:
            print(f"\nOrigin longitude = {olon}")
            # select u0, v0, init & final bearings
            u0 = u.sel(
                latitude=geom_chunk.origin_latitude,
                longitude=olon,
                method="nearest"
            )
            v0 = v.sel(
                latitude=geom_chunk.origin_latitude,
                longitude=olon, method="nearest"
            )

            # roll wind fields (need to check if actually faster than rolling geometry)
            t0 = time.perf_counter()
            lon_shift = int(round(olon / lon_step))
            _u = u.roll(longitude=-lon_shift, roll_coords=False).load()
            _v = v.roll(longitude=-lon_shift, roll_coords=False).load()
            print(
                f"Roll: "
                f"{time.perf_counter()-t0:.6f}s"
            )

            if active_indices is None:
                #
                # compute over the full sphere
                #
                
                # compute du_cubed
                t0 = time.perf_counter()
                du_cubed = compute_delta_u_cubed(
                    _u, _v, u0, v0,
                    geom_chunk.sine_initial_bearing, geom_chunk.cosine_initial_bearing,
                    geom_chunk.sine_final_bearing, geom_chunk.cosine_final_bearing,
                    w=None
                ).load()
                print(
                    f"du_cubed: "
                    f"{time.perf_counter()-t0:.6f}s"
                )

                # compute angular average (this should be  a function)
                t0 = time.perf_counter()
                means = np.empty((chunk_len, nbins))
                for i in range(chunk_len):
                    vals = du_cubed.isel(origin_latitude=i).values.ravel()
                    bins = geom_chunk.great_circle_distance_bin.isel(origin_latitude=i).values.ravel()
                    sum_bin = np.bincount(bins, weights=vals, minlength=nbins)
                    count_bin = np.bincount(bins, minlength=nbins)
                    if len(count_bin) == nbins:
                        means[i] = sum_bin / count_bin
                    else:
                        means[i] = sum_bin[:-1] / count_bin[:-1] # removing overflow bin
                #endfor
                print(
                    f"angular average (np.bincount): "
                    f"{time.perf_counter()-t0:.6f}s"
                )

                # clean up
                del du_cubed
            else:
                #
                # compute within a spherical cap of radius max_R
                #

                t0 = time.perf_counter()
                means = reduce_using_indices(
                    _u,
                    _v,
                    u0,
                    v0,
                    geom_chunk,
                    active_indices,
                    nbins # should get nbins from geom_chunk
                )
                print(
                    f"du_cubed AND angular average (reduce_using_indices): "
                    f"{time.perf_counter()-t0:.6f}s"
                )
            #endif
            da_means = xr.DataArray(
                means,
                dims=(
                    "origin_latitude",
                    "great_circle_distance",
                ),
                coords={
                    "origin_latitude": geom_chunk.origin_latitude,
                    "great_circle_distance": distances,
                },
                name="delta_u_cubed_angular_average",
            )
            du_cubed_ang_av.append(
                da_means.expand_dims(
                    origin_longitude=[olon]
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

        da = ds_out.delta_u_cubed_angular_average

        ds_write = xr.Dataset(
            {
                "delta_u_cubed_angular_average": (da.dims, da.data)
            }
        ).reset_coords(drop=True)
        # save latitude chunk (just save the data variable to avoid write errors)
        print("\nSaving chunk")
        ds_write.to_zarr(
            du3_fpath,
            region = {
                "origin_latitude": slice(*olat_chunk)
            }
        )
    #endfor

    print("\n\n\nEND.")
