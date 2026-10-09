import numpy as np
import xarray as xr
import time
import logging
import numexpr as ne
from numba import njit

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

EXPR_DUSQ_NO_W = ne.NumExpr(
    "du_t*du_t + du_n*du_n"
)

EXPR_DU3_LONG = ne.NumExpr(
    "du_t * du_sq"
)

EXPR_DU3_VERT = ne.NumExpr(
    "dw * du_sq"
)

DU_T = (
    "u*sin_final + v*cos_final "
    "- u0*sin_init - v0*cos_init"
)

DU_N = (
    "u*cos_final - v*sin_final "
    "- u0*cos_init + v0*sin_init"
)

DW = "(w - w0)"

DU_SQ = (
    f"({DU_T})*({DU_T})"
    f"+"
    f"({DU_N})*({DU_N})"
    f"+"
    f"({DW})*({DW})"
)

DU_SQ_NO_W = (
    f"({DU_T})*({DU_T})"
    f"+"
    f"({DU_N})*({DU_N})"
)

EXPR_DU3_LONG_FUSED = ne.NumExpr(
    f"({DU_T})*({DU_SQ})"
)

EXPR_DU3_VERT_FUSED = ne.NumExpr(
    f"({DW})*({DU_SQ})"
)

EXPR_DU3_LONG_FUSED_NO_W = ne.NumExpr(
    f"({DU_T})*({DU_SQ_NO_W})"
)

def compute_delta_u_cubed(
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

def compute_delta_u_cubed_hybrid(
    u, v, u0, v0,
    sin_init, cos_init,
    sin_final, cos_final,
    w=None, w0=None,
):
    if (w is None) != (w0 is None):
        raise ValueError("Must provide both w and w0")
    
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
        du_cubed_long = (
            du_t * du_sq
        )

        # vertical transverse third-order velocity increment
        du_cubed_vert = (
            dw * du_sq
        )
    
        return du_cubed_long, du_cubed_vert
    else:
        # (delta u)^2
        du_sq = EXPR_DUSQ_NO_W(
            du_n, du_t
        )

        # longitudinal third-order velocity increment
        du_cubed_long = (
            du_t * du_sq
        )
    
        return du_cubed_long, None

@njit(cache=True)
def compute_delta_u_cubed_numba(
    u,
    v,
    u0,
    v0,
    sin_init,
    cos_init,
    sin_final,
    cos_final,
    w,
    w0,
):
    n = u.size

    du3_long = np.empty(n, dtype=np.float64)
    du3_vert = np.empty(n, dtype=np.float64)

    for i in range(n):

        du_t = (
            u[i] * sin_final[i]
            + v[i] * cos_final[i]
            - u0 * sin_init[i]
            - v0 * cos_init[i]
        )

        du_n = (
            u[i] * cos_final[i]
            - v[i] * sin_final[i]
            - u0 * cos_init[i]
            + v0 * sin_init[i]
        )

        dw = w[i] - w0

        du_sq = (
            du_t * du_t
            + du_n * du_n
            + dw * dw
        )

        du3_long[i] = du_t * du_sq
        du3_vert[i] = dw * du_sq

    return du3_long, du3_vert

@njit(cache=True)
def compute_delta_u_cubed_numba_no_w(
    u,
    v,
    u0,
    v0,
    sin_init,
    cos_init,
    sin_final,
    cos_final,
):
    n = u.size

    du3_long = np.empty(n, dtype=np.float64)

    for i in range(n):

        du_t = (
            u[i] * sin_final[i]
            + v[i] * cos_final[i]
            - u0 * sin_init[i]
            - v0 * cos_init[i]
        )

        du_n = (
            u[i] * cos_final[i]
            - v[i] * sin_final[i]
            - u0 * cos_init[i]
            + v0 * sin_init[i]
        )

        du3_long[i] = (
            du_t
            * (du_t * du_t + du_n * du_n)
        )

    return du3_long

def compute_delta_u_cubed_numexpr(
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
        # 'du_sq', 'du_t'
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

def compute_delta_u_cubed_numexpr_FUSED(
    u, v, u0, v0,
    sin_init, cos_init,
    sin_final, cos_final,
    w=None, w0=None,
):
    if w is not None:
        INPUT_ORDER = (
            "cos_final",
            "cos_init",
            "sin_final",
            "sin_init",
            "u",
            "u0",
            "v",
            "v0",
            "w",
            "w0",
        )

        assert EXPR_DU3_LONG_FUSED.input_names == INPUT_ORDER

        du_cubed_long = EXPR_DU3_LONG_FUSED(
            cos_final,
            cos_init,
            sin_final,
            sin_init,
            u,
            u0,
            v,
            v0,
            w,
            w0
        )
        du_cubed_vert = EXPR_DU3_VERT_FUSED(
            cos_final,
            cos_init,
            sin_final,
            sin_init,
            u,
            u0,
            v,
            v0,
            w,
            w0
        )
        return du_cubed_long, du_cubed_vert
    else:
        INPUT_ORDER = (
            "cos_final",
            "cos_init",
            "sin_final",
            "sin_init",
            "u",
            "u0",
            "v",
            "v0",
        )

        assert EXPR_DU3_LONG_FUSED.input_names == INPUT_ORDER

        du_cubed_long = EXPR_DU3_LONG_FUSED_NO_W(
            cos_final,
            cos_init,
            sin_final,
            sin_init,
            u,
            u0,
            v,
            v0,
        )
        return du_cubed_long, None

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

def select_active_points_packed(
    i,
    u,
    v,
    u0,
    v0,
    active_geom,
    lon_shift,
    w=None,
    w0=None,
    profiler=None,
    use_cyclic_padding=False,
    u_pad=None,
    v_pad=None,
    w_pad=None,
):
    """
    Select active points using pre-packed geometry.

    Parameters
    ----------
    i : int
        Origin-latitude index within the current chunk.

    u, v : xr.DataArray
        Rolled horizontal velocity fields.

    u0, v0 : xr.DataArray
        Origin velocities.

    active_geom : list[dict]
        Output from pack_active_geometry().

    w, w0 : optional
        Vertical velocity and origin vertical velocity.

    Returns
    -------
    Same tuple as select_active_points().
    """

    if (w is None) != (w0 is None):
        raise ValueError("Must provide both w and w0")

    # setup
    if profiler:
        t0_select_geom = time.perf_counter()
    geom_i = active_geom[i]

    ilat = geom_i["ilat"]
    ilon = geom_i["ilon"]
    if profiler:
        profiler.add(
            "angular integral: select geom",
            time.perf_counter() - t0_select_geom
        )

    if lon_shift == 0:   # avoid printing for every longitude
        logger.info(
            "origin_lat=%d  n_active=%d",
            i,
            ilat.size,
        )

    if profiler:
        t0_modulo = time.perf_counter()

    if use_cyclic_padding:
        ilon_shifted = ilon + lon_shift
    else:
        nlon = u.shape[1]
        ilon_shifted = (ilon + lon_shift) % nlon

    if profiler:
        profiler.add(
            "angular integral: select (longitude shift)",
            time.perf_counter() - t0_modulo
        )

    if profiler:
        t0_uvw = time.perf_counter()

    if use_cyclic_padding:
        u_sel = u_pad[ilat, ilon_shifted]
        v_sel = v_pad[ilat, ilon_shifted]
    else:
        u_sel = u.values[ilat, ilon_shifted]
        v_sel = v.values[ilat, ilon_shifted]

    u0_sel = u0.values[i]
    v0_sel = v0.values[i]

    if w is not None:
        if use_cyclic_padding:
            w_sel = w_pad[ilat, ilon_shifted]
        else:
            w_sel = w.values[ilat, ilon_shifted]

        w0_sel = w0.values[i]

    else:
        w_sel = None
        w0_sel = None
    
    if profiler:
        profiler.add(
            "angular integral: select u,v,w",
            time.perf_counter() - t0_uvw
        )

    return (
        u_sel,
        v_sel,
        u0_sel,
        v0_sel,
        geom_i["sin_init"],
        geom_i["cos_init"],
        geom_i["sin_final"],
        geom_i["cos_final"],
        geom_i["bins"],
        geom_i["weights"],
        geom_i["distance"],
        w_sel,
        w0_sel,
    )

def select_active_points(
    i, ilat, ilon,
    u, v, u0, v0,
    geom_chunk,
    w = None,
    w0 = None,
    use_angular_weights=False,
    method = "spherical",
    profiler = None,
):
    # TO DO: introduce a dataclass and return just sel = dataclass
    # (should be more robust)
    if (w is None) != (w0 is None):
        raise ValueError("Must provide both w and w0")

    if profiler:
        t0_uv = time.perf_counter()
    u_sel = u.values[ilat, ilon]
    v_sel = v.values[ilat, ilon]

    u0_sel = u0.values[i]
    v0_sel = v0.values[i]
    if profiler:
        profiler.add(
            "angular integral: select u,v",
            time.perf_counter() - t0_uv
        )

    if profiler:
        t0_trig_init = time.perf_counter()
    sin_init_sel = geom_chunk.sine_initial_bearing.values[
        i, ilat, ilon
    ]
    cos_init_sel = geom_chunk.cosine_initial_bearing.values[
        i, ilat, ilon
    ]
    if profiler:
        profiler.add(
            "angular integral: select trig. init.",
            time.perf_counter() - t0_trig_init
        )

    if profiler:
        t0_trig_final = time.perf_counter()
    sin_final_sel = geom_chunk.sine_final_bearing.values[
        i, ilat, ilon
    ]
    cos_final_sel = geom_chunk.cosine_final_bearing.values[
        i, ilat, ilon
    ]
    if profiler:
        profiler.add(
            "angular integral: select trig. final",
            time.perf_counter() - t0_trig_final
        )

    if profiler:
        t0_bins = time.perf_counter()
    bins_sel = geom_chunk.great_circle_distance_bin.values[
        i, ilat, ilon
    ]
    if profiler:
        profiler.add(
            "angular integral: select bins",
            time.perf_counter() - t0_bins
        )

    if profiler:
        t0_other = time.perf_counter()
    if w is not None:
        w_sel = w.values[ilat, ilon]
        w0_sel = w0.values[i]
    else:
        w_sel = None
        w0_sel = None

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
    
    if profiler:
        profiler.add(
            "angular integral: select other",
            time.perf_counter() - t0_other
        )

    return (
        u_sel, v_sel, u0_sel, v0_sel,
        sin_init_sel, cos_init_sel,
        sin_final_sel, cos_final_sel,
        bins_sel, weights_sel, distance_sel,
        w_sel, w0_sel,
    )

def compute_du3_angular_integral_global(
        u,
        v,
        u0,
        v0,
        geom_chunk,
        nbins,
        w=None,
        w0=None,
        dtype=np.float64,
        use_angular_weights=False,
        profiler=None,
        backend="auto",
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

    -------
    TO DO:
     - update doc string!!
     - add 'backend' functionality to switch between Numba and hybrid NumExpr
       depending on grid size (or user preference)
    """
    if (w is None) != (w0 is None):
        raise ValueError("Must provide both w and w0")
    
    chunk_len = geom_chunk.sizes["origin_latitude"]

    if profiler:
        t0_du3 = time.perf_counter()
    # compute du_cubed
    du_cubed_long, du_cubed_vert = compute_delta_u_cubed_hybrid(
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
        if profiler:
            t0_read = time.perf_counter()
        weights = weights_cache[i] if use_angular_weights else None
        
        du3_long = du_cubed_long[i,:,:].ravel()
        if profiler:
            profiler.add(
                "angular integral: integral by distance bin (read)",
                time.perf_counter() - t0_read
            )

        if profiler:
            t0_integrate = time.perf_counter()
        integrals_long[i] = angular_integral_by_distance_bin(
            du3_long,
            bins_cache[i],
            nbins,
            weights=weights,
            profiler=profiler,
        )
        if profiler:
            profiler.add(
                "angular integral: integral by distance bin (integrate)",
                time.perf_counter() - t0_integrate
            )
        if w is not None:
            if profiler:
                t0_read = time.perf_counter()
            du3_vert = du_cubed_vert[i,:,:].ravel()
            if profiler:
                profiler.add(
                    "angular integral: integral by distance bin (read)",
                    time.perf_counter() - t0_read
                )

            if profiler:
                t0_integrate = time.perf_counter()
            integrals_vert[i] = angular_integral_by_distance_bin(
                du3_vert,
                bins_cache[i],
                nbins,
                weights=weights,
                profiler=profiler,
            )
            if profiler:
                profiler.add(
                    "angular integral: integral by distance bin (integrate)",
                    time.perf_counter() - t0_integrate
                )
        #endif
    #endfor
    if profiler:
        profiler.add(
            "angular integral: integral by distance bin total",
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
        lon_shift,
        w=None,
        w0=None,
        active_geom=None,
        dtype=np.float64,
        use_angular_weights=False,
        profiler=None,
        method="spherical",
        backend="auto",
        use_cyclic_padding=False,
        u_pad=None,
        v_pad=None,
        w_pad=None,
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

    -------
    TO DO:
     - update doc string!!
     - add 'backend' functionality to switch between Numba and hybrid NumExpr
       depending on grid size (or user preference)
    """
    if (w is None) != (w0 is None):
        raise ValueError("Must provide both w and w0")

    if w is not None and method != "spherical":
        raise NotImplementedError(
            "Vertical velocity currently only supported for "
            "method = 'spherical'"
        )
    
    nchunk = len(active_indices)

    # allocate output arrays
    if profiler:
        t0_alloc = time.perf_counter()
    integrals_long = np.empty(
        (nchunk, nbins),
        dtype=dtype,
    )
    
    if w is not None:
        integrals_vert = np.empty(
            (nchunk, nbins),
            dtype=dtype,
        )
    else:
        integrals_vert = None

    if profiler:
        profiler.add(
            "angular integral: alloc. output arrays",
            time.perf_counter() - t0_alloc
        )

    for i, (ilat, ilon) in enumerate(active_indices):

        # gather only active points
        if profiler:
            t0_gather = time.perf_counter()
        if active_geom is None:
            (
                u_sel, v_sel, u0_sel, v0_sel,
                sin_init_sel, cos_init_sel,
                sin_final_sel, cos_final_sel,
                bins_sel, weights_sel, distance_sel,
                w_sel, w0_sel
            ) = select_active_points(
                i, ilat, ilon, u, v, u0, v0, geom_chunk,
                w=w,
                w0=w0,
                use_angular_weights=use_angular_weights,
                method=method,
                profiler=profiler,
            )
        else:
            (
                u_sel, v_sel, u0_sel, v0_sel,
                sin_init_sel, cos_init_sel,
                sin_final_sel, cos_final_sel,
                bins_sel, weights_sel, distance_sel,
                w_sel, w0_sel
            ) = select_active_points_packed(
                i, u, v, u0, v0,
                active_geom, lon_shift,
                w=w,
                w0=w0,
                profiler=profiler,
                use_cyclic_padding=use_cyclic_padding,
                u_pad=u_pad,
                v_pad=v_pad,
                w_pad=w_pad,
            )
        if profiler:
            profiler.add(
                "angular integral: gather active points",
                time.perf_counter() - t0_gather
            )

        # compute du^3
        if profiler:
            t0_du3 = time.perf_counter()
        if method == "spherical":
            #du3_long, du3_vert = compute_delta_u_cubed_hybrid(
            du3_long, du3_vert = compute_delta_u_cubed_numba(
                u_sel,
                v_sel,
                u0_sel,
                v0_sel,
                sin_init_sel,
                cos_init_sel,
                sin_final_sel,
                cos_final_sel,
                w_sel,
                w0_sel,
                #w=w_sel,
                #w0=w0_sel,
            )
        elif method == "tangent_plane":
            du3_long = compute_delta_u_cubed_tangent_plane(
                u_sel,
                v_sel,
                u0_sel,
                v0_sel,
                sin_init_sel,
                cos_init_sel,
            )
            du3_vert = None
        elif method == "tangent_quadratic":
            sphere_radius = geom_chunk.attrs["sphere_radius_m"]
            lat0 = np.deg2rad(u0.latitude.values[i])
            delta_alpha = (
                distance_sel
                * sin_init_sel
                * np.tan(lat0)
                / sphere_radius
            )
            du3_long = compute_delta_u_cubed_tangent_quadratic(
                u_sel,
                v_sel,
                u0_sel,
                v0_sel,
                sin_init_sel,
                cos_init_sel,
                delta_alpha,
            )
            du3_vert = None
        #endif
        if profiler:
            profiler.add(
                "angular integral: du^3",
                time.perf_counter() - t0_du3
            )

        # compute angular integral
        if profiler:
            t0_integral = time.perf_counter()
            
        integrals_long[i] = angular_integral_by_distance_bin(
            du3_long,
            bins_sel,
            nbins,
            weights=weights_sel,
            profiler=profiler,
        )

        if w is not None:
            integrals_vert[i] = angular_integral_by_distance_bin(
                du3_vert,
                bins_sel,
                nbins,
                weights=weights_sel,
                profiler=profiler,
            )

        if profiler:
            profiler.add(
                "angular integral: integral by distance bin total",
                time.perf_counter() - t0_integral
            )
        
        # clean up
        del du3_long
        del du3_vert
        
    return integrals_long, integrals_vert


