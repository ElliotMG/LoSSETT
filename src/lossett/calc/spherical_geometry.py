import xarray as xr
import numpy as np

from lossett.calc.angular_integration import(
    voronoi_widths_periodic
)

RADIUS_EARTH = 6371000 # Earth radius in metres

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

def compute_great_circle_distance(
    lat0,
    lat,
    cos_lat0,
    cos_lat,
    dlon,
    radius=RADIUS_EARTH,
):
    # TO DO: rewrite so that the formula only uses sin(lat), sin(lat0)
    # sin(dlon), cos(dlon) & take these as input instead of lat0, lat, dlon
    #  since they're already computed in compute_geometry
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
    
    return radius * c

def compute_distance_bins(distance, distance_edges):
    return (
        np.digitize(distance, distance_edges) - 1
    )

def compute_initial_bearing_trig(
    sin_dlon, cos_dlon,
    sin_lat0, cos_lat0,
    sin_lat, cos_lat
):

    # this bit could be raw numpy
    yi = sin_dlon * cos_lat
    xi = (
        cos_lat0 * sin_lat
        - sin_lat0 * cos_lat * cos_dlon
    )
    norm = np.sqrt(xi*xi + yi*yi)
    # remove unsafe values
    norm = xr.where(norm > 0, norm, np.nan)
    
    # sine(initial_bearing)
    sin_init = yi / norm
        
    # cosine(initial_bearing)
    cos_init = xi / norm

    # cleaning up
    del xi
    del yi
    del norm
    
    return sin_init, cos_init

def compute_final_bearing_trig(
    sin_dlon, cos_dlon,
    sin_lat0, cos_lat0,
    sin_lat, cos_lat
):
    yf = - sin_dlon * cos_lat0
    xf = (
        cos_lat * sin_lat0
        - sin_lat * cos_lat0 * cos_dlon
    )
    norm = np.sqrt(xf*xf + yf*yf)
    # remove unsafe values
    norm = xr.where(norm > 0, norm, np.nan)

    # sine(final_bearing)
    sin_final = -yf / norm
        
    # cosine(initial_bearing)
    cos_final = -xf / norm

    # cleaning up
    del xf
    del yf
    del norm

    return sin_final, cos_final

def compute_initial_bearing(
    sin_dlon, cos_dlon,
    sin_lat0, cos_lat0,
    sin_lat, cos_lat
):
    yi = sin_dlon * cos_lat
    xi = (
        cos_lat0 * sin_lat
        - sin_lat0 * cos_lat * cos_dlon
    )
    initial_bearing = np.arctan2(yi, xi)

    # cleaning up
    del xi
    del yi
    
    return initial_bearing

def compute_final_bearing(
    sin_dlon, cos_dlon,
    sin_lat0, cos_lat0,
    sin_lat, cos_lat
):
    yf = - sin_dlon * cos_lat0
    xf = (
        cos_lat * sin_lat0
        - sin_lat * cos_lat0 * cos_dlon
    )
    final_bearing = np.arctan2(-yf, -xf)

    # cleaning up
    del xf
    del yf
    
    return final_bearing

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
        weights_sorted = voronoi_widths_periodic(chi_sorted)

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

    # Great circle distance
    distance = compute_great_circle_distance(
        lat0,
        lat,
        cos_lat0,
        cos_lat,
        dlon,
    ).astype(dtype)
    distance = distance.transpose(*dim_order)
    distance.name = "great_circle_distance"

    # Great circle distance bin
    distance_bin = compute_distance_bins(distance, distance_edges)
    distance_bin = xr.DataArray(
        distance_bin.astype(bin_dtype),
        dims = distance.dims,
        coords = distance.coords,
        name = "great_circle_distance_bin"
    )

    if trig_fns:
        # Sine, cosine initial bearing
        sin_init, cos_init = compute_initial_bearing_trig(
            sin_dlon, cos_dlon,
            sin_lat0, cos_lat0,
            sin_lat, cos_lat
        )
        sin_init = sin_init.transpose(*dim_order)
        sin_init.name = "sine_initial_bearing"
        cos_init = cos_init.transpose(*dim_order)
        cos_init.name = "cosine_initial_bearing"

        # Angular weights
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
        angular_weight = angular_weight.transpose(*dim_order)
        angular_weight.name = "angular_weight"
        
        # Sine, cosine final bearing
        sin_final, cos_final = compute_final_bearing_trig(
            sin_dlon, cos_dlon,
            sin_lat0, cos_lat0,
            sin_lat, cos_lat
        )
        sin_final = sin_final.transpose(*dim_order)
        sin_final.name = "sine_final_bearing"
        cos_final = cos_final.transpose(*dim_order)
        cos_final.name = "cosine_final_bearing"

        ds_geom = xr.merge([
            distance, distance_bin, sin_init, cos_init,
            sin_final, cos_final, angular_weight
        ])
    else:
        #
        # Initial bearing
        #
        initial_bearing = compute_initial_bearing(
            sin_dlon, cos_dlon,
            sin_lat0, cos_lat0,
            sin_lat, cos_lat
        ).astype(dtype)
        initial_bearing = initial_bearing.transpose(*dim_order)
        initial_bearing.name = "initial_bearing"
        
        #
        # Final bearing
        #
        final_bearing = compute_final_bearing(
            sin_dlon, cos_dlon,
            sin_lat0, cos_lat0,
            sin_lat, cos_lat
        ).astype(dtype)
        final_bearing = final_bearing.transpose(*dim_order)
        final_bearing.name = "final_bearing"
        
        ds_geom = xr.merge([distance, distance_bin, initial_bearing, final_bearing])

    return ds_geom
