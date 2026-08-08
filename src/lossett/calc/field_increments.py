import numpy as np
import xarray as xr
import time
import logging
import numexpr as ne

from lossett.calc.angular_integration import (
    angular_integral_by_distance_bin,
)

logger = logging.getLogger(__name__)

EXPR_DUT = ne.NumExpr(
    "u*sin_final + v*cos_final "
    "- u0*sin_init - v0*cos_init"
)

EXPR_DUN = ne.NumExpr(
    "u*cos_final - v*sin_final "
    "- u0*cos_init + v0*sin_init"
)

EXPR_DU3 = ne.NumExpr(
    "du_t*(du_t*du_t + du_n*du_n)"
)

def compute_delta_u_cubed(
    u, v, u0, v0,
    sin_init, cos_init,
    sin_final, cos_final,
    w=None
):
    # velocity increment tangent to geodesic ("longitudinal" part)
    du_t = (
        u*sin_final
        + v*cos_final
        - u0*sin_init
        - v0*cos_init
    )
    # velocity increment normal to geodesic ("transverse" part)
    du_n = (
        u*cos_final
        - v*sin_final
        - u0*cos_init
        + v0*sin_init
    )
    # compute delta u cubed dot rhat
    du_cubed = du_t * (du_t*du_t + du_n*du_n)

    del du_t
    del du_n

    ### TO DO: ADD VERTICAL VELOCITY CAPABILITY
    
    return du_cubed

def compute_delta_u_cubed_numexpr(
    u, v, u0, v0,
    sin_init, cos_init,
    sin_final, cos_final,
    w=None
):
    # velocity increment tangent to geodesic ("longitudinal" part)
    # EXPR_DUT REQUIRES INPUT IN THIS ORDER:
    # 'cos_final', 'cos_init', 'sin_final', 'sin_init', 'u', 'u0', 'v', 'v0'
    du_t = EXPR_DUT(
        cos_final, cos_init,
        sin_final, sin_init,
        u, u0, v, v0
    )
    
    # velocity increment normal to geodesic ("transverse" part)
    # EXPR_DUn REQUIRES INPUT IN THIS ORDER:
    # 'cos_final', 'cos_init', 'sin_final', 'sin_init', 'u', 'u0', 'v', 'v0'
    du_n = EXPR_DUN(
        cos_final, cos_init,
        sin_final, sin_init,
        u, u0, v, v0
    )

    ### TO DO: ADD VERTICAL VELOCITY CAPABILITY
    
    # EXPR_DUN REQUIRES INPUT IN THIS ORDER:
    # 'du_n', 'du_t'
    return EXPR_DU3(
        du_n, du_t
    )

def compute_delta_u_cubed_tangent_plane(
    u,
    v,
    u0,
    v0,
    sin_init,
    cos_init,
):
    """
    Assume the local geodesic frame does not rotate, i.e.
    \delta u_t = (u_f - u_i) sin_i + (v_f - v_i) cos_i
    \delta u_n = (u_f - u_i) cos_i - (v_f - v_i) sin_i
    """
    
    du = u - u0
    dv = v - v0

    # O(1) tangent-plane terms
    du_t = (
        du * sin_init
        + dv * cos_init
    )

    du_n = (
        du * cos_init
        - dv * sin_init
    )

    return du_t * (
        du_t*du_t
        + du_n*du_n
    )

def compute_delta_u_cubed_tangent_quadratic(
    u,
    v,
    u0,
    v0,
    sin_init,
    cos_init,
    delta_alpha,
):
    """
    Include the leading-order curvature correction,
        \delta \alpha = (r / R) * sin_i * tan(lat_i)
    meaning
        \delta u_t = (u_f - u_i) sin_i + (v_f - v_i) cos_i
                     + delta alpha * ( cos_i u_f - sin_i v_f )
        \delta u_n = (u_f - u_i) cos_i - (v_f - v_i) sin_i
                     - delta alpha * ( sin_i u_f + cos_i v_f )
    """
    
    du = u - u0
    dv = v - v0

    # O(1) tangent-plane terms
    du_t = (
        du * sin_init
        + dv * cos_init
    )

    du_n = (
        du * cos_init
        - dv * sin_init
    )

    # O(r/R_sphere) correction
    du_t += (
        delta_alpha
        * (
            u * cos_init
            - v * sin_init
        )
    )

    du_n -= (
        delta_alpha
        * (
            u * sin_init
            + v * cos_init
        )
    )

    return du_t * (
        du_t*du_t
        + du_n*du_n
    )

### TO DO:
### FUNCTIONS BELOW THIS LINE TO BE MOVED TO "structure_functions.py" OR SIMILAR.
### ONLY FUNCTIONS THAT CALCULATE FIELD INCREMENTS TO BE INCLUDED IN
### "field_increments.py".

def select_active_points(
    i, ilat, ilon,
    u, v, u0, v0,
    geom_chunk,
    use_angular_weights=False,
    method = "spherical"    
):
    # TO DO: introduce a dataclass and return just sel = dataclass
    # (should be more robust)
    
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

    bins_sel = geom_chunk.great_circle_distance_bin.values[
        i, ilat, ilon
    ]

    if use_angular_weights:
        weights_sel = geom_chunk.angular_weight.values[
            i,
            ilat,
            ilon,
        ]
    else:
        weights_sel = None

    if method == "tangent_quadratic":
        distance_sel = geom_chunk.great_circle_distance.values[
            i,
            ilat,
            ilon
        ]
    else:
        distance_sel = None

    return (
        u_sel, v_sel, u0_sel, v0_sel,
        sin_init_sel, cos_init_sel,
        sin_final_sel, cos_final_sel,
        bins_sel, weights_sel, distance_sel
    )

def compute_du3_angular_integral_global(
        u,
        v,
        u0,
        v0,
        geom_chunk,
        nbins,
        dtype=np.float64,
        use_angular_weights=False,
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
    integrals
        (origin_latitude, nbins)
    """
    chunk_len = geom_chunk.sizes["origin_latitude"]
                
    # compute du_cubed
    #du_cubed = compute_delta_u_cubed(
    #    u, v, u0, v0,
    #    geom_chunk.sine_initial_bearing, geom_chunk.cosine_initial_bearing,
    #    geom_chunk.sine_final_bearing, geom_chunk.cosine_final_bearing,
    #    w=None
    #).load()
    #du_cubed = compute_delta_u_cubed(
    du_cubed = compute_delta_u_cubed_numexpr(
        u.values,
        v.values,
        u0.values[:,None,None],
        v0.values[:,None,None],
        geom_chunk.sine_initial_bearing.values,
        geom_chunk.cosine_initial_bearing.values,
        geom_chunk.sine_final_bearing.values,
        geom_chunk.cosine_final_bearing.values,
        w=None
    )

    # pre-compute geometry lookups
    bins_cache = [
        arr.ravel()
        for arr in geom_chunk.great_circle_distance_bin.values
    ]

    if use_angular_weights:
        weights_cache = [
            arr.ravel()
            for arr in geom_chunk.angular_weight.values
        ]
    
    # compute angular integral (this should be a function)
    integrals = np.empty((chunk_len, nbins), dtype=dtype)
    for i in range(chunk_len):
        #du3 = du_cubed.isel(origin_latitude=i).values.ravel()
        du3 = du_cubed[i,:,:].ravel()
        #bins = geom_chunk.great_circle_distance_bin.isel(origin_latitude=i).values.ravel()
        if use_angular_weights:
            #weights = geom_chunk.angular_weight.isel(origin_latitude=i).values.ravel()
            integrals[i] = angular_integral_by_distance_bin(
                du3,
                #bins,
                bins_cache[i],
                nbins,
                weights=weights_cache[i],
            )
        else:
            #weights = None
            integrals[i] = angular_integral_by_distance_bin(
                du3,
                #bins,
                bins_cache[i],
                nbins,
            )
        
    # clean up
    del du_cubed
            
    return integrals

def compute_du3_angular_integral_subset(
        u,
        v,
        u0,
        v0,
        geom_chunk,
        active_indices,
        nbins,
        dtype=np.float64,
        use_angular_weights=False,
        method="spherical"
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
    integrals
        (origin_latitude, nbins)
    """
    
    nchunk = len(active_indices)

    integrals = np.empty(
        (nchunk, nbins),
        dtype=dtype,
    )

    for i, (ilat, ilon) in enumerate(active_indices):

        # gather only active points
        (
            u_sel, v_sel, u0_sel, v0_sel,
            sin_init_sel, cos_init_sel,
            sin_final_sel, cos_final_sel,
            bins_sel, weights_sel, distance_sel
        ) = select_active_points(
            i, ilat, ilon, u, v, u0, v0, geom_chunk,
            use_angular_weights=use_angular_weights,
            method=method
        )

        # compute du^3
        if method == "spherical":
            du3 = compute_delta_u_cubed(
                u_sel, v_sel,
                u0_sel, v0_sel,
                sin_init_sel, cos_init_sel,
                sin_final_sel, cos_final_sel,
                w=None
            )
        elif method == "tangent_plane":
            du3 = compute_delta_u_cubed_tangent_plane(
                u_sel,
                v_sel,
                u0_sel,
                v0_sel,
                sin_init_sel,
                cos_init_sel,
            )
        elif method == "tangent_quadratic":
            sphere_radius = geom_chunk.attrs["sphere_radius_m"]
            lat0 = np.deg2rad(u0.latitude.values[i])
            delta_alpha = (
                distance_sel
                * sin_init_sel
                * np.tan(lat0)
                / sphere_radius
            )
            du3 = compute_delta_u_cubed_tangent_quadratic(
                u_sel,
                v_sel,
                u0_sel,
                v0_sel,
                sin_init_sel,
                cos_init_sel,
                delta_alpha,
            )

        # compute angular integral
        integrals[i] = angular_integral_by_distance_bin(
            du3,
            bins_sel,
            nbins,
            weights=weights_sel,
        )
        
    return integrals


