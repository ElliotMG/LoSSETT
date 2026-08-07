import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
# local imports
from lossett.filtering.kernels import filter_kernel

RADIUS_EARTH = 6.371e6

def get_integration_kernels(
        r,
        length_scales,
        kernel_type="standard_mollifier",
        normalization="sphere",
        sphere_radius=RADIUS_EARTH,
        return_deriv=True
):
    """
    Required inputs:
     - r: sampling points in r-space
     - length_scales: array of length scales
    Optional inputs:
     - kernel_type: name of kernel. The only currently-supported option is standard_mollifier
     - normalization: geometry to assume when computing normalization. Can be 2D Eculidean, 
           3D Euclidean, or 2D spherical.
     - sphere_radius: only relevant for normalization="spherical", so should probably modify...
     - return_deriv: Boolean controlling whether only the kernel, or both kernel and its derivative, 
           are returned.
    """
    # TO-DO: check for units consistency between r, length_scales and sphere_radius!

    # TO-DO: check that max length_scale < max r / 2

    # TO-DO: check that max length_scale < \pi R / 2 if normalization == "sphere"

    G = []
    dG_dr = []
    for length_scale in length_scales:
        # compute normalized dG/dr
        kernel, deriv = filter_kernel(
            length_scale,
            r,
            return_derivative=True,
            normalization="sphere",
            sphere_radius=sphere_radius
        )
        kernel = xr.DataArray(
            kernel,
            coords = {"r": r,"length_scale": length_scale},
            dims = "r",
            name = "G",
            attrs = {"long_name": "filter kernel"}
        )
        deriv = xr.DataArray(
            deriv,
            coords = {"r": r,"length_scale": length_scale},
            dims = "r",
            name = "dG_dr",
            attrs = {"long_name": "r-derivative of filter kernel"}
        )
        G.append(kernel)
        dG_dr.append(deriv)
    
    # concatenate into a single xr.DataArray
    G = xr.concat(G,dim="length_scale")
    dG_dr = xr.concat(dG_dr,dim="length_scale")

    if return_deriv:
        ds = xr.merge([G, dG_dr])
    else:
        ds = xr.merge([G])
    ds.attrs.update({
        "kernel_properties": {
            "name": "standard_mollifier",
            "ratio_rmax_to_ell": 2.0,
            "ratio_L_to_ell": 2.0,
        }
    })

    return ds

