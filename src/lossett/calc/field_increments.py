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

EXPR_DUSQ = ne.NumExpr(
    "du_t*du_t + du_n*du_n + dw*dw"
)

EXPR_DU3_LONG = ne.NumExpr(
    "du_t * du_sq"
)

EXPR_DU3_VERT = ne.NumExpr(
    "dw * du_sq"
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
    
    return du_cubed

def compute_delta_u_cubed_NEW(
    u, v, u0, v0,
    sin_init, cos_init,
    sin_final, cos_final,
    w=None, w0=None,
):
    if (w is None) != (w0 is None):
        raise ValueError("Must provide both w and w0")
    
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
    if w is not None:
        # vertical velocity increment (also normal to geodesic)
        dw = w - w0
        # compute du^2
        du_sq = (du_t*du_t + du_n*du_n + dw*dw)
        # compute delta u cubed dot rhat (longitudinal)
        du_cubed_long = du_t * du_sq
        # compute delta u cubed (transverse, vertical part)
        du_cubed_vert = dw * du_sq
        del du_t
        del du_n
        del dw
        del du_sq
        return du_cubed_long, du_cubed_vert
    else:
        # compute delta u cubed dot rhat (longitudinal)
        du_cubed = du_t * (du_t*du_t + du_n*du_n)
        del du_t
        del du_n
        return du_cubed, None

def compute_delta_u_cubed_numexpr_NEW(
    u, v, u0, v0,
    sin_init, cos_init,
    sin_final, cos_final,
    w=None, w0=None,
):
    # velocity increment tangent to geodesic ("longitudinal" part)
    # EXPR_DUT REQUIRES INPUT IN THIS ORDER:
    # 'cos_final', 'cos_init', 'sin_final', 'sin_init', 'u', 'u0', 'v', 'v0'
    du_t = EXPR_DUT(
        cos_final, cos_init,
        sin_final, sin_init,
        u, u0, v, v0
    )
    
    # velocity increment normal to geodesic (horizontal "transverse" part)
    # EXPR_DUN REQUIRES INPUT IN THIS ORDER:
    # 'cos_final', 'cos_init', 'sin_final', 'sin_init', 'u', 'u0', 'v', 'v0'
    du_n = EXPR_DUN(
        cos_final, cos_init,
        sin_final, sin_init,
        u, u0, v, v0
    )

    if w is not None:
        # vertical velocity increment (vertical "transverse" part)
        dw = w - w0

        # (delta u)^2
        du_sq = EXPR_DUSQ(
            du_n, du_t, dw
        )

        # longitudinal third-order velocity increment
        # EXPR_DU3_LONG REQUIRES INPUT IN THIS ORDER:
        # 'du_n', 'du_t'
        du_cubed_long = EXPR_DU3_LONG(
            du_sq, du_t
        )

        # vertical transverse third-order velocity increment
        # EXPR_DU3_VERT REQUIRES INPUT IN THIS ORDER:
        # 'du_sq', 'dw'
        du_cubed_vert = EXPR_DU3_VERT(
            du_sq, dw
        )
        return du_cubed_long, du_cubed_vert
    else:
        # longitudinal third-order velocity increment
        # EXPR_DU3 REQUIRES INPUT IN THIS ORDER:
        # 'du_n', 'du_t'
        du_cubed = EXPR_DU3(
            du_n, du_t
        )
        return du_cubed, None

def compute_delta_u_cubed_numexpr(
    u, v, u0, v0,
    sin_init, cos_init,
    sin_final, cos_final,
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
    # EXPR_DUN REQUIRES INPUT IN THIS ORDER:
    # 'cos_final', 'cos_init', 'sin_final', 'sin_init', 'u', 'u0', 'v', 'v0'
    du_n = EXPR_DUN(
        cos_final, cos_init,
        sin_final, sin_init,
        u, u0, v, v0
    )
    
    # EXPR_DU3 REQUIRES INPUT IN THIS ORDER:
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
        w=None,
        w0=None,
        profiler=None,
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
    if (w is None) != (w0 is None):
        raise ValueError("Must provide both w and w0")
    
    chunk_len = geom_chunk.sizes["origin_latitude"]

    if profiler:
        t0_du3 = time.perf_counter()
    # compute du_cubed
    #du_cubed_long, du_cubed_vert = compute_delta_u_cubed_NEW(
    du_cubed_long, du_cubed_vert = compute_delta_u_cubed_numexpr_NEW(
    #du_cubed = compute_delta_u_cubed_numexpr(
        u.values,
        v.values,
        u0.values[:,None,None],
        v0.values[:,None,None],
        geom_chunk.sine_initial_bearing.values,
        geom_chunk.cosine_initial_bearing.values,
        geom_chunk.sine_final_bearing.values,
        geom_chunk.cosine_final_bearing.values,
        w=w.values if w is not None else None,
        w0=w0.values[:,None,None] if w is not None else None,
    )
    if profiler:
        profiler.add(
            "angular integral: du^3",
            time.perf_counter() - t0_du3
        )

    # pre-compute geometry lookups
    if profiler:
        t0_cache = time.perf_counter()
    bins_cache = [
        arr.ravel()
        for arr in geom_chunk.great_circle_distance_bin.values
    ]

    if use_angular_weights:
        weights_cache = [
            arr.ravel()
            for arr in geom_chunk.angular_weight.values
        ]
    if profiler:
        profiler.add(
            "angular integral: geom cache",
            time.perf_counter() - t0_cache
        )
    
    # compute angular integral (this should be a function)
    if profiler:
        t0_integral = time.perf_counter()
    integrals_long = np.empty((chunk_len, nbins), dtype=dtype)
    if w is not None:
        integrals_vert = np.empty((chunk_len, nbins), dtype=dtype)
    else:
        integrals_vert = None
    #endif
    
    for i in range(chunk_len):
        weights = weights_cache[i] if use_angular_weights else None
        
        du3_long = du_cubed_long[i,:,:].ravel()
        
        integrals_long[i] = angular_integral_by_distance_bin(
            du3_long,
            bins_cache[i],
            nbins,
            weights=weights,
        )
        if w is not None:
            du3_vert = du_cubed_vert[i,:,:].ravel()
            integrals_vert[i] = angular_integral_by_distance_bin(
                du3_vert,
                bins_cache[i],
                nbins,
                weights=weights,
            )
        #endif
    #endfor
    if profiler:
        profiler.add(
            "angular integral: integral by distance bin",
            time.perf_counter() - t0_integral
        )
        
    # clean up
    del du_cubed_long
    del du_cubed_vert
            
    return integrals_long, integrals_vert

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
        method="spherical",
        w=None,
        w0=None,
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
    if (w is None) != (w0 is None):
        raise ValueError("Must provide both w and w0")
    
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
            #du3 = compute_delta_u_cubed(
            #    u_sel, v_sel,
            #    u0_sel, v0_sel,
            #    sin_init_sel, cos_init_sel,
            #    sin_final_sel, cos_final_sel,
            #    w=None
            #)
            du3 = compute_delta_u_cubed_numexpr(
                u_sel,
                v_sel,
                u0_sel,
                v0_sel,
                sin_init_sel,
                cos_init_sel,
                sin_final_sel,
                cos_final_sel,
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


