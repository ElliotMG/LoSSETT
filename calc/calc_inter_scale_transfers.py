#!/usr/bin/env python3
import os
import sys
import numpy as np
import xarray as xr
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import cartopy as cpy
# local imports
from ..filtering.get_integration_kernels import get_integration_kernels

radius_earth = 6.371e6 # radius of Earth in m
deg_to_m = 110000.0 # conversion of latitudinal degrees to m

def calc_inter_scale_energy_transfer_kinetic(
        ds_u_3D,
        control_dict
):
    # extract control params
    max_r = control_dict["max_r"]
    max_r_units = control_dict["max_r_units"]
    precision = control_dict["angle_precision"]
    x_coord_name = control_dict["x_coord_name"]
    x_coord_units =control_dict["x_coord_units"]
    x_coord_boundary = control_dict["x_coord_boundary"]
    y_coord_name = control_dict["y_coord_name"]
    y_coord_units = control_dict["y_coord_units"]
    y_coord_boundary = control_dict["y_coord_boundary"]
    
    # setup x-y coords, bounds, grid spacings
    x = ds_u_3D[x_coord_name]
    y = ds_u_3D[y_coord_name]
    if x_coord_units != y_coord_units:
        print("\nError! x and y coord units must be the same.")
        sys.exit(1)
    elif x_coord_units == "deg":
        _x = x
        _y = y
        x = _x*deg_to_m
        y = _y*deg_to_m
        if max_r_units == "deg":
            max_r_deg = max_r
            max_r = max_r_deg*deg_to_m
        # add new x, y as coords to ds_u_3D
        ds_u_3D = ds_u_3D.assign_coords({x_coord_name:x.values,y_coord_name:y.values})
        ds_u_3D[x_coord_name].attrs["units"] = "m"
        ds_u_3D[y_coord_name].attrs["units"] = "m"
    
    x_bounds = np.array([x[0].values,x[-1].values])
    y_bounds = np.array([y[0].values,y[-1].values])
    delta_x = np.max(np.diff(x))
    delta_y = np.max(np.diff(y))

    # calculate scale increments
    scale_incs = calc_scale_increments(x,y,max_r,verbose=False)
    r = scale_incs.r

    # compute delta u cubed integrated over angles for all |r|
    print(f"\n\n\nCalculating angular integral for r={r[0].values/1000:.4g} km to r={r[-1].values/1000:.4g}")
    delta_u_cubed = calc_increment_integrand(
        ds_u_3D, scale_incs, calc_delta_u_cubed, delta_x, delta_y,
        xdim=x_coord_name, ydim=y_coord_name, xbounds=x_bounds, ybounds=y_bounds,
        x_bound_field=x_coord_boundary, y_bound_field=y_coord_boundary, precision=precision,
        verbose=True
    )
    delta_u_cubed = delta_u_cubed.transpose("r","time",x_coord_name,y_coord_name,"pressure")
    if x_coord_units == "deg":
        delta_u_cubed = delta_u_cubed.assign_coords(
            {x_coord_name:_x.values,y_coord_name:_y.values}
        )
        delta_u_cubed[x_coord_name].attrs = _x.attrs
        delta_u_cubed[y_coord_name].attrs = _y.attrs
    # add option to save integrand

    # calculate scale-space integral given integrand, length scales, geometry specification
    integrand = delta_u_cubed
    r = integrand.r
    # specify length scales -- should probably be an if statement here to allow the user
    # to specify scales if desired.
    length_scales = r.values[1:len(r)//2]
    # calculare DR
    DR_indicator = calc_scale_space_integral(
        integrand, name="DR_indicator", length_scales=length_scales, weighting="2D"
    ) # should add options for kernel specification
    
    return DR_indicator;

def calc_inter_scale_energy_transfer_thermo():
    return 0;

def calc_scale_increments(
        xcoord, ycoord, max_r, delta_r=None, geometry="Euclidean",
        verbose=False
):
    """
    TO-DO.
    """
    L_x = xcoord[-1] - xcoord[0]
    L_y = ycoord[-1] - ycoord[0]
    delta_x = np.max(np.diff(xcoord))
    delta_y = np.max(np.diff(ycoord))
    delta_r = max(delta_x, delta_y)
    if max_r > min(L_x,L_y)/2:
        max_r = min(L_x,L_y)/2
    if verbose:
        print("L_x = ", L_x)
        print("L_y = ", L_y)
        print("delta_x = ", delta_x)
        print("delta_y = ", delta_y)
        print("delta_r = ", delta_r)
        print("max_r = ", max_r)

    # construct uniform 2D r-space grid
    r_x = np.arange(xcoord[0],xcoord[0]+delta_x*len(xcoord),delta_x)
    r_y = np.arange(ycoord[0],ycoord[0]+delta_y*len(ycoord),delta_y)

    grid = xr.DataArray(
        coords={
            "r_x": r_x,
            "r_y": r_y
        },
        dims = ["r_x", "r_y"]
    )

    origin_x_index = len(xcoord)//2
    origin_y_index = len(ycoord)//2
    r_0 = grid.isel(r_x=origin_x_index, r_y=origin_y_index).rename("origin")
    r_0.data = 0.0
    r_0 = r_0.expand_dims("r_x").expand_dims("r_y")

    distance = np.sqrt( (grid.r_x - r_0.r_x.data)**2 + (grid.r_y - r_0.r_y.data)**2 ).rename("distance")
    angle = np.arctan2( grid.r_y-r_0.r_y.data, grid.r_x-r_0.r_x.data ).T.rename("angle")
    r = np.arange(0,max_r,delta_r)

    if verbose:
        print(r_0)
        print(distance)
        print(angle)
        print(r)

    # construct masks for each r
    mask = []
    for R in r: 
        _mask = xr.where((distance >= R-delta_r/2) & (distance < R+delta_r/2), 1, 0)
        _mask = _mask.assign_coords({"r":R})
        mask.append(_mask)
    mask = xr.concat(mask,dim="r").rename("r_mask")
    ds_mask = xr.merge([mask,distance,angle,r_0])
    
    return ds_mask;

def calc_increment_integrand(
        field, scale_incs, function, delta_x, delta_y,
        xdim, ydim, xbounds, ybounds, x_bound_field=np.nan,
        y_bound_field=np.nan, precision=1e-10, conv_fac=1.0,
        verbose=False
):
    import psutil
    pid = os.getpid()
    python_process = psutil.Process(pid)

    memory_use = python_process.memory_info()[0]/(10**9) # RAM usage in GB
    print(f"\nCurrent memory usage: {memory_use:5g} GB")

    # get origin indices
    r_0 = scale_incs.origin
    dims = r_0.dims
    xdim_loc = dims.index("r_x")
    ydim_loc = dims.index("r_y")
    origin_x_index = np.argwhere(np.isfinite(r_0.values)).squeeze()[xdim_loc]
    origin_y_index = np.argwhere(np.isfinite(r_0.values)).squeeze()[ydim_loc]
    r_integrand = []
    for R in scale_incs.r:
        # get indices
        print(f"\n\n\nr = {R:.5g}")
        mask = scale_incs.r_mask.sel(r=R)
        angle = scale_incs.angle.where(mask==1)
        angle_coord = np.unique(angle)
        angle_coord = angle_coord[np.isfinite(angle_coord)]
        phi_integrand = []
        # can this loop be vectorized somehow?
        for phi in angle_coord:
            loc = xr.where((angle < phi+precision) & (angle >= phi-precision),1,0)
            loc = np.argwhere(loc.values).squeeze()
            # possible to rewrite if,else statement as np.where() call?
            if len(np.shape(loc)) != 1:
                # handle degenerate angles at 0, \pi
                for ip,pt in enumerate(loc):
                    n_x = pt[0] - origin_x_index
                    n_y = pt[1] - origin_y_index
                    field_shifted = roll_with_boundary_handling(
                        field, n_x, n_y, delta_x/conv_fac, delta_y/conv_fac,
                        xdim, ydim, xbounds/conv_fac, ybounds/conv_fac,
                        x_bound_field=x_bound_field, y_bound_field=y_bound_field
                    )
                    _phi_integrand += function(
                        field, field_shifted, n_x*delta_x, n_y*delta_y,
                        delta_r_z=None, dims="2D"
                    )
                _phi_integrand /= len(loc)
            else:
                n_x = loc[0] - origin_x_index
                n_y = loc[1] - origin_y_index
                field_shifted = roll_with_boundary_handling(
                    field, n_x, n_y, delta_x/conv_fac, delta_y/conv_fac,
                    xdim, ydim, xbounds/conv_fac, ybounds/conv_fac,
                    x_bound_field=x_bound_field, y_bound_field=y_bound_field
                )
                _phi_integrand = function(
                    field, field_shifted, n_x*delta_x, n_y*delta_y,
                    delta_r_z=None, dims="2D"
                )
            _phi_integrand = _phi_integrand.assign_coords({"angle":phi})
            _phi_integrand = _phi_integrand.assign_coords({"r":R})
            phi_integrand.append(_phi_integrand)

        phi_integrand = xr.concat(phi_integrand,"angle")
        phi_integral = phi_integrand.integrate("angle").compute()
        r_integrand.append(phi_integral)
        if verbose:
            print("\n", phi_integrand)
            print("\n", phi_integral)

        memory_use = python_process.memory_info()[0]/(10**9) # RAM usage in GB
        print(f"\nCurrent memory usage: {memory_use:5g} GB")
        
    r_integrand = xr.concat(r_integrand,"r")

    memory_use = python_process.memory_info()[0]/(10**9) # RAM usage in GB
    print(f"\n\n\nCurrent memory usage (after calculating r integrand): {memory_use:5g} GB")

    return r_integrand;

def calc_scale_space_integral(integrand, name, length_scales=None, weighting="2D"):
    r = integrand.r
    if length_scales is None:
        length_scales = r.values[1:len(r)//2]

    G, dG_dr = get_integration_kernels(
        r,
        length_scales,
        normalization=weighting,
        return_deriv=True
    )

    # integrate only over the support of dG_dr
    # NOTE: there must be a way to vectorise this?
    print("\nCalculating scale-space integral.")
    integral = []
    for il, ell in enumerate(dG_dr.length_scale):
        print("\n\ell = ", ell)
        integral.append(
            (
                dG_dr.sel(length_scale=ell)*r*integrand
            ).sel(r=slice(0,2*ell)).integrate("r").rename(name)
        )
    integral = xr.concat(integral, "length_scale")
    integral *= (1./4.)
    
    return integral;

def roll_with_boundary_handling(
        data, n_x, n_y, delta_x, delta_y,
        xdim, ydim, xbounds, ybounds,
        x_bound_field=np.nan, y_bound_field=np.nan
):
    # should really calculate perpendicular distance to boundary for each point!
    if (x_bound_field == "periodic") & (y_bound_field == "periodic"):
        rolled = data.roll({xdim:-n_x, ydim:-n_y}, roll_coords=False)
    elif x_bound_field == "periodic":
        rolled = xr.where(
            (np.abs(data[ydim] - ybounds[0])>=np.abs(n_y*delta_y)) & (np.abs(data[ydim] - ybounds[1])>=np.abs(n_y*delta_y)),
            data.roll({xdim:-n_x, ydim:-n_y}, roll_coords=False),
            y_bound_field
        )
    elif y_bound_field == "periodic":
        rolled = xr.where(
            (np.abs(data[xdim] - xbounds[0])>=np.abs(n_x*delta_x)) & (np.abs(data[xdim] - xbounds[1])>=np.abs(n_x*delta_x)),
            data.roll({xdim:-n_x, ydim:-n_y}, roll_coords=False),
            x_bound_field
        )
    else:
        rolled_x = xr.where(
            (np.abs(data[xdim] - xbounds[0])>=np.abs(n_x*delta_x)) & (np.abs(data[xdim] - xbounds[1])>=np.abs(n_x*delta_x)),
            data.roll({xdim:-n_x, ydim:-n_y}, roll_coords=False),
            x_bound_field
        )
        rolled_y = xr.where(
            (np.abs(data[ydim] - ybounds[0])>=np.abs(n_y*delta_y)) & (np.abs(data[ydim] - ybounds[1])>=np.abs(n_y*delta_y)),
            data.roll({xdim:-n_x, ydim:-n_y}, roll_coords=False),
            y_bound_field
        )
        rolled = 0.5 * (rolled_x + rolled_y)
    return rolled;

def calc_delta_u_cubed(ds, ds_shifted, delta_r_x, delta_r_y, delta_r_z=None, dims="2D"):
    ds_increment = ds_shifted - ds
    delta_u = ds_increment["u"]
    delta_v = ds_increment["v"]
    delta_w = ds_increment["w"]

    delta_u_squared = delta_u**2 + delta_v**2 + delta_w**2

    if dims == "2D":
        r_mag = np.sqrt(delta_r_x**2 + delta_r_y**2)
        delta_u_dot_r = (delta_u*delta_r_x + delta_v*delta_r_y) / r_mag
    elif dims == "3D":
        r_mag = np.sqrt(delta_r_x**2 + delta_r_y**2 + delta_r_z**2)
        delta_u_dot_r = (delta_u*delta_r_x + delta_v*delta_r_y + delta_w*delta_r_z) / r_mag

    delta_u_cubed = (delta_u_dot_r * delta_u_squared).rename("delta_u_cubed")
    
    return delta_u_cubed;

def calc_delta_u_delta_scalar_squared(ds, ds_shifted, varname, delta_r_x, delta_r_y, delta_r_z=None, dims="2D"):
    ds_increment = ds_shifted - ds
    delta_u = ds_increment["u"]
    delta_v = ds_increment["v"]
    delta_w = ds_increment["w"]
    delta_scalar_squared = ds_increment[varname]**2

    if dims == "2D":
        r_mag = np.sqrt(delta_r_x**2 + delta_r_y**2)
        delta_u_dot_r = (delta_u*delta_r_x + delta_v*delta_r_y) / r_mag
    elif dims == "3D":
        r_mag = np.sqrt(delta_r_x**2 + delta_r_y**2 + delta_r_z**2)
        delta_u_dot_r = (delta_u*delta_r_x + delta_v*delta_r_y + delta_w*delta_r_z) / r_mag

    delta_u_delta_scalar_squared = (delta_u_dot_r * delta_scalar_squared).rename(f"delta_u_delta_{varname}_squared")
    
    return delta_u_delta_scalar_squared;

def calc_field_increment(
        ds, ds_shifted, delta_r_x, delta_r_y, delta_r_z=None, dims="2D"
):
    ds_increment = ds_shifted - ds
    # should really go through and rename all fields if it's a dataset
    return ds_increment;

def plot_scale_increments(scale_incs, Rsel=[10,20,40], show=False):
    distance = ds_mask.distance
    angle = ds_mask.angle
    mask = scale_incs.mask
    r_x = ds_mask.r_x
    r_y = ds_mask.r_y
    
    # plot
    fig,axes = plt.subplots(
        nrows=2, ncols=3, sharex=True, sharey=True,
        figsize=(18,10)
    )
    vmax_r = distance.max()
    ticks_phi = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    ticklabels_phi = ["$-\pi$", "$-\pi / 2$", "0", "$\pi / 2$", "$\pi$"]
    # plot r
    ax=axes[0,0]
    ax.set_title("$r$")
    pc_r = ax.pcolormesh(r_x,r_y,distance.T,cmap="viridis",vmin=0,vmax=vmax_r)
    plt.colorbar(pc_r)
    # plot phi
    ax=axes[0,1]
    ax.set_title("$\phi$")
    pc_phi = ax.pcolormesh(r_x,r_y,angle.T,cmap="twilight_shifted")
    cb_phi = plt.colorbar(pc_phi, ticks=ticks_phi)
    cb_phi.ax.set_yticklabels(ticklabels_phi)
    # plot mask
    ax=axes[0,2]
    ax.set_title("mask summed over $r$")
    pc_mask = ax.pcolormesh(r_x,r_y,mask.sum("r").T,cmap="viridis")
    plt.colorbar(pc_mask)

    # plot r, phi for various R
    for ir, R in enumerate(Rsel):
        R_circ = mask.sel(r=R,method="nearest").r
        # plot r (select r = R)
        ax=axes[1,0]
        pc_r = ax.pcolormesh(
            r_x,r_y,distance.where(mask.sel(r=R, method="nearest") == 1).T,
            cmap="viridis",vmin=0,vmax=vmax_r
        )
        circle=plt.Circle((r_0.r_x,r_0.r_y), R_circ, color='w', fill=False)
        ax.add_patch(circle)
        if ir == 0:
            ax.set_title(f"$r$ for $r \in [{Rsel[0]:.2g},{Rsel[1]:.2g},{Rsel[2]:.2g}]$")
            cb_r = plt.colorbar(pc_r)
        cb_r.ax.axhline(R_circ,color="w")
        # plot phi (select r = R)
        ax=axes[1,1]
        pc_phi = ax.pcolormesh(
            r_x,r_y,angle.where(mask.sel(r=R, method="nearest") == 1).T,
            cmap="twilight_shifted"
        )
        circle=plt.Circle((r_0.r_x,r_0.r_y), R_circ, color='w', fill=False)
        ax.add_patch(circle)
        if ir == 0:
            ax.set_title(f"$\phi$ for $r \in [{Rsel[0]:.2g},{Rsel[1]:.2g},{Rsel[2]:.2g}]$")
            cb_phi = plt.colorbar(pc_phi, ticks=ticks_phi)
            cb_phi.ax.set_yticklabels(ticklabels_phi)
    # aspect ratio
    for row in axes:
        for ax in row:
            ax.set_aspect("equal")
    # save and show
    plt.savefig(
        f"scale_increments_mask_Lx_{L_x:.3g}_Ly_{L_y:.3g}_Rmax_{max_r:.3g}_dx_{delta_x:.3g}_dy_{delta_y:.3g}.png"
    )
    if show:
        plt.show()
    plt.close()
    return 0;
