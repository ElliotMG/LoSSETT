import numpy as np
import xarray as xr
import time
import logging

from lossett.calc.angular_integration import (
    angular_integral_by_distance_bin,
)

logger = logging.getLogger(__name__)

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

### TO DO:
### FUNCTIONS BELOW THIS LINE TO BE MOVED TO "structure_functions.py" OR SIMILAR.
### ONLY FUNCTIONS THAT CALCULATE FIELD INCREMENTS TO BE INCLUDED IN
### "field_increments.py".

def select_active_points(
    i, ilat, ilon,
    u, v, u0, v0,
    geom_chunk,
    use_angular_weights=False
):
    
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

    return (
        u_sel, v_sel, u0_sel, v0_sel,
        sin_init_sel, cos_init_sel,
        sin_final_sel, cos_final_sel,
        bins_sel, weights_sel
    )

def compute_du3_angular_integral_global(
        u,
        v,
        u0,
        v0,
        geom_chunk,
        nbins,
        dtype=np.float64,
        profile=False,
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
    
    # compute angular integral (this should be a function)
    if profile:
        t0 = time.perf_counter()
    integrals = np.empty((chunk_len, nbins), dtype=dtype)
    for i in range(chunk_len):
        du3 = du_cubed.isel(origin_latitude=i).values.ravel()
        bins = geom_chunk.great_circle_distance_bin.isel(origin_latitude=i).values.ravel()
        if use_angular_weights:
            weights = geom_chunk.angular_weight.isel(origin_latitude=i).values.ravel()
        else:
            weights = None
        integrals[i] = angular_integral_by_distance_bin(
            du3,
            bins,
            nbins,
            weights=weights,
        )
    #endfor
    if profile:
        logger.debug(
            f"angular integral (np.bincount): "
            f"{time.perf_counter()-t0:.6f}s"
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
        profile=False,
        use_angular_weights=False
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

    if profile:
        t0 = time.perf_counter()
    
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
            bins_sel, weights_sel
        ) = select_active_points(
            i, ilat, ilon, u, v, u0, v0, geom_chunk,
            use_angular_weights=use_angular_weights
        )

        # compute du^3
        du3 = compute_delta_u_cubed(
            u_sel, v_sel,
            u0_sel, v0_sel,
            sin_init_sel, cos_init_sel,
            sin_final_sel, cos_final_sel,
            w=None
        )

        # compute angular integral
        integrals[i] = angular_integral_by_distance_bin(
            du3,
            bins_sel,
            nbins,
            weights=weights_sel,
        )

    #endfor
    if profile:
        logger.debug(
            f"du_cubed AND angular integral: "
            f"{time.perf_counter()-t0:.6f}s"
        )
    
    return integrals


