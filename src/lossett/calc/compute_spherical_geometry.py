#!/usr/bin/env python3
import os
import xarray as xr
import numpy as np
import dask.array as da
import time
import argparse
from numcodecs import Blosc

RADIUS_EARTH = 6371000 # Earth radius in metres
GEOM_DIMS = ("origin_latitude","latitude","longitude")
GRID_DEFS = {
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

def build_regular_latlon_grid(lon_step, lat_step):
    lons = np.arange(-180., 180., lon_step)
    lats = np.arange(-90., 90., lat_step)
    lats = np.append(lats, lats[-1] + lat_step)
    return lons, lats

def build_distance_bins(nbins, max_r=np.pi*RADIUS_EARTH):
    delta_r = max_r / nbins
    edges = np.arange(
        0.,
        max_r + delta_r / 2.,
        delta_r,
    )
    centres = (
        edges[:-1]
        + edges[1:]
    ) / 2
    return edges, centres

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

def create_geometry_template(
        origin_latitudes, target_latitudes, delta_longitudes,
        chunk_origin, chunk_lat, chunk_lon,
        dtype=np.float32, bin_dtype=np.uint8, trig_fns=False
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

    return xr.Dataset(
        variables,
        coords=coords,
    )

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
        dtype=dtype,
        bin_dtype=bin_dtype,
        trig_fns=trig_fns,
    )

    template.attrs.update(
        {
            "earth_radius_m": RADIUS_EARTH,
            "distance_bin_edges_m": distance_edges.tolist(),
            "distance_bin_max_m": float(distance_edges[-1]),
            "n_distance_bins": len(distance_edges) - 1,
        }
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

def compute_angular_weights(
    distance_bin,
    sin_chi,
    cos_chi,
    nbins,
    dtype=np.float32,
    tol=1e-6
):
    """
    Compute quadrature weights Δχ for each point.

    Parameters
    ----------
    distance_bin : ndarray, shape (nlat, nlon)
        Integer distance bin indices.

    sin_chi, cos_chi : ndarray, shape (nlat, nlon)
        Sine and cosine of bearing angle χ.

    nbins : int
        Number of distance bins.

    Returns
    -------
    angular_weight : ndarray, shape (nlat, nlon)

        Angular-sector width associated with each point.
        For every bin:

            angular_weight[distance_bin == ibin].sum()

        should be approximately 2π.
    """

    angular_weight = np.zeros(
        distance_bin.shape,
        dtype=dtype,
    )

    for ibin in range(nbins):
        # exclude infinite / undefined bearings
        valid = (
            np.isfinite(sin_chi)
            & np.isfinite(cos_chi)
        )
        mask = (
            (distance_bin == ibin)
            & valid
        )

        if not np.any(mask):
            continue

        ilat, ilon = np.where(mask)
        sin_chi_bin = sin_chi[ilat, ilon]
        cos_chi_bin = cos_chi[ilat, ilon]
        assert len(sin_chi_bin) == len(cos_chi_bin)
        npts = len(sin_chi_bin)

        group_cos = np.round(
            cos_chi_bin / tol
        ).astype(np.int32)

        group_sin = np.round(
            sin_chi_bin / tol
        ).astype(np.int32)

        groups = np.column_stack(
            [group_cos, group_sin]
        )

        unique_groups, inverse = np.unique(
            groups,
            axis=0,
            return_inverse=True,
        )
        n_unique_pts = len(unique_groups)

        count_chi = np.bincount(inverse).astype(dtype)

        sin_mean = (
            np.bincount(
                inverse,
                weights=sin_chi_bin
            )
            / count_chi
        )

        cos_mean = (
            np.bincount(
                inverse,
                weights=cos_chi_bin
            )
            / count_chi
        )

        chi_unique = np.mod(
            np.arctan2(
                sin_mean,
                cos_mean,
            ),
            2*np.pi,
        )
        assert np.all(
            np.isfinite(chi_unique)
        )

        #
        # uniform weighting for very small bins
        #
        if n_unique_pts <= 4:
            angular_weight[ilat, ilon] = (
                2 * np.pi / npts
            )
            continue

        #
        # sort around the circle
        #
        order = np.argsort(chi_unique)
        chi_sorted = chi_unique[order]

        #
        # Voronoi cell width in χ
        #
        dchi = np.diff(
            np.concatenate(
                [chi_sorted, [chi_sorted[0] + 2*np.pi]]
            )
        )
        weights_sorted = 0.5 * (
            dchi
            + np.roll(dchi, 1)
        )

        #
        # restore original ordering
        #
        weights = np.empty_like(
            weights_sorted
        )
        weights[order] = weights_sorted
        weights_per_point = (
            weights[inverse] / count_chi[inverse]
        )

        # Consistency checks
        assert np.isclose(
            weights_per_point.sum(),
            2*np.pi,
            atol=tol,
        )
        assert np.isfinite(
            weights_per_point
        ).all()
        assert np.all(weights_per_point > 0)
        
        angular_weight[
            ilat,
            ilon,
        ] = weights_per_point
        
        assert np.isclose(
            angular_weight[mask].sum(),
            2*np.pi,
            atol=tol,
        )

    return angular_weight

def compute_geometry(
        origin_latitudes,
        target_latitudes,
        delta_longitudes,
        distance_edges,
        radius=RADIUS_EARTH,
        dtype=np.float32,
        bin_dtype=np.uint8,
        origin_lat_chunksize=16,
        trig_fns=False,
        dim_order=("origin_latitude", "latitude", "longitude")
):
    """
    Calculates:
      1. the great-circle distance between two points on a spherical surface
         using the Haversine formula;
      2. integer bins of great-circle distance with edges given by distance_edges;
      3. the clockwise angle (bearing) between the great circle path and a line of 
         constant longitude (meridian) passing through the initial point;
      4. the clockwise angle (bearing) between the great-circle path and a line of
         constant longitude (meridian) passing through the final point.
    Optionally calculates:
      5. sine and cosine of both the initial and final bearings.

    Inputs:
    origin_latitudes : np.ndarray, degrees
    target_latitudes : np.ndarray, degrees
    delta_longitudes : np.ndarray, degrees
    distance_edges   : np.ndarray, m

    Returns:
    ds_geom         : xarray Dataset, dimensions (origin_latitude, latitude, longitude)
                      containing the following variables:
    great_circle_distance : great circle distance between (lat1,lon1) and (lat2,lon2).
    great_circle_distance_bin : integer bins of great_circle_distance, with edges given
                      by distance_edges.

    If trig_fns == False:
    initial_bearing : clockwise angle at point (lat1,lon1) between North and the 
                      great circle path connecting (lat1,lon1) to (lat2,lon2). Can return
                      angle in either radians (default) or degrees.
    final_bearing   : clockwise angle at point (lat2,lon2) between North and the 
                      great circle path connecting (lat1,lon1) to (lat2,lon2). Can return
                      angle in either radians (default) or degrees.

    If trig_fns == True:
    sine_initial_bearing   : sine of initial_bearing
    cosine_initial_bearing : cosine of initial bearing
    sine_final_bearing     : sine of final bearing
    cosine_final_bearing   : cosine of final bearing
    """

    lat0 = xr.DataArray(
        np.deg2rad(origin_latitudes),
        dims="origin_latitude",
        coords={"origin_latitude": origin_latitudes},
    )

    lat = xr.DataArray(
        np.deg2rad(target_latitudes),
        dims="latitude",
        coords={"latitude": target_latitudes},
    )

    dlon = xr.DataArray(
        np.deg2rad(delta_longitudes),
        dims="longitude",
        coords={"longitude": delta_longitudes},
    )

    # trig functions for re-use
    sin_lat0 = np.sin(lat0)
    cos_lat0 = np.cos(lat0)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)

    sin_dlon = np.sin(dlon)
    cos_dlon = np.cos(dlon)

    #
    # Haversine distance
    #
    a = (
        np.sin((lat - lat0) / 2.0) ** 2
        + cos_lat0
        * cos_lat
        * np.sin(dlon / 2.0) ** 2
    )

    # clip to avoid division error
    a = a.clip(min=0.0, max=1.0)

    c = 2.0 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1.0 - a),
    )
    assert np.isfinite(c).all()

    distance = (radius * c).astype(dtype)
    distance = distance.transpose(*dim_order)
    distance.name = "great_circle_distance"

    #
    # Distance bin
    #
    distance_bin = xr.DataArray(
        (
            np.digitize(distance, distance_edges) - 1
        ).astype(bin_dtype),
        dims = distance.dims,
        coords = distance.coords,
        name = "great_circle_distance_bin"
    )

    if trig_fns:
        #
        # Initial bearing
        #
        yi = sin_dlon * cos_lat
        xi = (
            cos_lat0 * sin_lat
            - sin_lat0 * cos_lat * cos_dlon
        )
        norm = np.sqrt(xi*xi + yi*yi)
        # sine(initial_bearing)
        sin_init = xr.where(
            norm > 0,
            yi / norm,
            np.nan,
        )
        sin_init = sin_init.transpose(*dim_order)
        sin_init.name = "sine_initial_bearing"
        
        # cosine(initial_bearing)
        cos_init = xr.where(
            norm > 0,
            xi / norm,
            np.nan,
        )
        cos_init = cos_init.transpose(*dim_order)
        cos_init.name = "cosine_initial_bearing"

        del xi
        del yi

        #
        # Angular weights
        #
        angular_weight = xr.zeros_like(
            distance,
            dtype=dtype
        )
        for i in range(len(origin_latitudes)):
            angular_weight.values[i] = (
                compute_angular_weights(
                    distance_bin.isel(origin_latitude=i).values,
                    sin_init.isel(origin_latitude=i).values,
                    cos_init.isel(origin_latitude=i).values,
                    nbins=len(distance_edges)-1,
                    dtype=dtype
                )
            )
        angular_weight.name = "angular_weight"
        angular_weight = angular_weight.transpose(*dim_order)
        
        #
        # Final bearing
        #
        yf = - sin_dlon * cos_lat0
        xf = (
            cos_lat * sin_lat0
            - sin_lat * cos_lat0 * cos_dlon
        )
        norm = np.sqrt(xf*xf + yf*yf)
        # sine(final_bearing)
        sin_final = xr.where(
        norm > 0,
        -yf / norm,
        np.nan,
        )
        sin_final = sin_final.transpose(*dim_order)
        sin_final.name = "sine_final_bearing"
        
        # cosine(final_bearing)
        cos_final = xr.where(
            norm > 0,
            -xf / norm,
            np.nan,
        )
        cos_final = cos_final.transpose(*dim_order)
        cos_final.name = "cosine_final_bearing"

        del xf
        del yf

        ds_geom = xr.merge([distance, distance_bin, sin_init, cos_init, sin_final, cos_final, angular_weight])
    else:
        #
        # Initial bearing
        #
        yi = sin_dlon * cos_lat
        xi = (
            cos_lat0 * sin_lat
            - sin_lat0 * cos_lat * cos_dlon
        )
        initial_bearing = np.arctan2(yi, xi).astype(dtype)
        initial_bearing = initial_bearing.transpose(*dim_order)
        initial_bearing.name = "initial_bearing"

        del xi
        del yi
        
        #
        # Final bearing
        #
        yf = - sin_dlon * cos_lat0
        xf = (
            cos_lat * sin_lat0
            - sin_lat * cos_lat0 * cos_dlon
        )

        final_bearing = np.arctan2(-yf, -xf).astype(dtype)
        fin_bearing = final_bearing.transpose(*dim_order)
        final_bearing.name = "final_bearing"

        del xf
        del yf
        
        ds_geom = xr.merge([distance, distance_bin, initial_bearing, final_bearing])

    return ds_geom

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
            "origin_lat_chunksize": chunk_origin,
            "trig_fns": save_trig_fns,
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
