import xarray as xr
import numpy as np

from lossett.calc.angular_integration import(
    voronoi_widths_periodic
)

RADIUS_EARTH = 6371000 # Earth radius in metres
GEOM_DIMS = ("origin_latitude","latitude","longitude")

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
    """
    Returns zero-based distance-bin indices.
    Bin i corresponds to

        distance_edges[i] <= r < distance_edges[i+1]
    """

    return (
        np.digitize(distance, distance_edges) - 1
    )

def compute_initial_bearing_trig(
    sin_dlon, cos_dlon,
    sin_lat0, cos_lat0,
    sin_lat, cos_lat
):

    yi = sin_dlon * cos_lat
    xi = (
        cos_lat0 * sin_lat
        - sin_lat0 * cos_lat * cos_dlon
    )
    norm = np.sqrt(xi*xi + yi*yi)
    
    # sine(initial_bearing)
    sin_init = np.divide(
        yi,
        norm,
        out=np.full_like(norm, np.nan),
        where=norm > 0,
    )
        
    # cosine(initial_bearing)
    cos_init = np.divide(
        xi,
        norm,
        out=np.full_like(norm, np.nan),
        where=norm > 0,
    )

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
    
    # sine(final_bearing)
    sin_final = np.divide(
        -yf,
        norm,
        out=np.full_like(norm, np.nan),
        where=norm > 0,
    )
        
    # cosine(final_bearing)
    cos_final = np.divide(
        -xf,
        norm,
        out=np.full_like(norm, np.nan),
        where=norm > 0,
    )

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
        trig_fns=False,
        dims=GEOM_DIMS
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

    coords = {
        "origin_latitude": origin_latitudes,
        "latitude": target_latitudes,
        "longitude": delta_longitudes
    }

    lat0 = np.deg2rad(origin_latitudes)[:, None, None]

    lat = np.deg2rad(target_latitudes)[None, :, None]

    dlon = np.deg2rad(delta_longitudes)[None, None, :]

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
    )
    distance = xr.DataArray(
        distance.astype(dtype),
        dims=dims,
        coords=coords,
        name="great_circle_distance",
        attrs={"units": "m"}
    )

    # Great circle distance bin
    distance_bin = compute_distance_bins(distance.values, distance_edges)
    distance_bin = xr.DataArray(
        distance_bin.astype(bin_dtype),
        dims=dims,
        coords=coords,
        name="great_circle_distance_bin",
        attrs={
            "units": "",
        }
    )
    distance_bin.attrs.update(
        {
            "distance_bin_edges": distance_edges.tolist(),
            "distance_bin_edge_units": "m",
            "distance_bin_max": float(distance_edges[-1]),
            "n_distance_bins": len(distance_edges) - 1
        }
    )

    if trig_fns:
        # Sine, cosine initial bearing
        sin_init, cos_init = compute_initial_bearing_trig(
            sin_dlon, cos_dlon,
            sin_lat0, cos_lat0,
            sin_lat, cos_lat
        )
        sin_init = xr.DataArray(
            sin_init.astype(dtype),
            dims=dims,
            coords=coords,
            name="sine_initial_bearing",
            attrs={"units": ""}
        )
        cos_init = xr.DataArray(
            cos_init.astype(dtype),
            dims=dims,
            coords=coords,
            name="cosine_initial_bearing",
            attrs={"units": ""}
        )

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
        angular_weight.name = "angular_weight"
        angular_weight.attrs.update({"units":"rad"})
        
        # Sine, cosine final bearing
        sin_final, cos_final = compute_final_bearing_trig(
            sin_dlon, cos_dlon,
            sin_lat0, cos_lat0,
            sin_lat, cos_lat
        )
        sin_final = xr.DataArray(
            sin_final.astype(dtype),
            dims=dims,
            coords=coords,
            name="sine_final_bearing",
            attrs={"units": ""}
        )
        cos_final = xr.DataArray(
            cos_final.astype(dtype),
            dims=dims,
            coords=coords,
            name="cosine_final_bearing",
            attrs={"units": ""}
        )

        ds_geom = xr.merge([
            distance, distance_bin, sin_init, cos_init,
            sin_final, cos_final, angular_weight
        ])
    else:
        # Initial bearing
        initial_bearing = compute_initial_bearing(
            sin_dlon, cos_dlon,
            sin_lat0, cos_lat0,
            sin_lat, cos_lat
        )
        initial_bearing = xr.DataArray(
            initial_bearing.astype(dtype),
            dims=dims,
            coords=coords,
            name="initial_bearing",
            attrs={"units": "rad"}
        )
        
        # Final bearing
        final_bearing = compute_final_bearing(
            sin_dlon, cos_dlon,
            sin_lat0, cos_lat0,
            sin_lat, cos_lat
        )
        final_bearing = xr.DataArray(
            final_bearing.astype(dtype),
            dims=dims,
            coords=coords,
            name="final_bearing",
            attrs={"units": "rad"}
        )
        
        ds_geom = xr.merge([distance, distance_bin, initial_bearing, final_bearing])

    ds_geom.attrs.update(
        {
            "sphere_radius_m": radius,
            "dtype": repr(dtype),
            "bin_dtype": repr(bin_dtype),
            "trig_fns": trig_fns,
        }
    )

    return ds_geom
