import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
# local imports
from .kernels import filter_kernel

radius_earth = 6.371e6

def get_integration_kernels(
        r,
        length_scales,
        normalization="sphere",
        sphere_radius=radius_earth,
        return_deriv=True
):
    """
    Required inputs:
     - r: sampling points in r-space
     - length_scales: array of length scales
    Optional inputs:
     - 
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
            sphere_radius=radius_earth
        )
        kernel = xr.DataArray(
            kernel,
            coords = {"r": r,"length_scale": length_scale},
            dims = "r"
        )
        deriv = xr.DataArray(
            deriv,
            coords = {"r": r,"length_scale": length_scale},
            dims = "r"
        )
        G.append(kernel)
        dG_dr.append(deriv)
    
    # concatenate into a single xr.DataArray
    G = xr.concat(G,dim="length_scale")
    dG_dr = xr.concat(dG_dr,dim="length_scale")
    G = G.rename("filter_kernel")
    dG_dr = dG_dr.rename("r-derivative_of_filter_kernel")

    if return_deriv:
        return G, dG_dr;
    else:
        return G;

if __name__ == "__main__":
    delta_r = 0.5
    deg2m = 110000
    radius_earth = 6.371e6
    r = np.arange(0,180,delta_r) #spacing in degrees
    r *= deg2m # approx. conversion to m
    l_max = 2e6 # 2000km
    length_scales = np.arange(0, l_max, delta_r*deg2m)[1:]
    print(length_scales)

    G, dG_dr = get_integration_kernels(
        r,
        length_scales,
        normalization="sphere",
        sphere_radius=radius_earth,
        return_deriv=True
    )

    print("\n\n\n", (2*np.pi*radius_earth*G*np.sin(G.r/radius_earth)).integrate("r"))

