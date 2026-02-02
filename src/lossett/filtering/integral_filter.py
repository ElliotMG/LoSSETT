#!/usr/bin/env python3
import os
import sys
import numpy as np
import xarray as xr
from importlib.metadata import version
# local imports
from lossett.calc.calc_inter_scale_transfers import calc_scale_increments, calc_scale_space_integral, calc_increment_integrand

radius_earth = 6.371e6 # radius of Earth in m
deg_to_m = 110000.0 # conversion of latitudinal degrees to m
LOSSETT_VN = version("lossett")

def dummy_function(field, field_shifted, delta_r_x, delta_r_y, **kwargs):
    return field_shifted;

def filter_field(
        field, control_dict,
        length_scales=None, name=None
):
    """
    Note that 2D filtering is currently hard-coded.
    """
    if name is None:
        name = field.name+"_filtered"
    input_attrs = field.attrs
    print("Input data attributes:", repr(input_attrs))
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
    x = field[x_coord_name]
    y = field[y_coord_name]
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
        field = field.assign_coords({x_coord_name:x.values,y_coord_name:y.values})
        field[x_coord_name].attrs["units"] = "m"
        field[y_coord_name].attrs["units"] = "m"
    
    x_bounds = np.array([x[0].values,x[-1].values])
    y_bounds = np.array([y[0].values,y[-1].values])
    delta_x = np.max(np.diff(x))
    delta_y = np.max(np.diff(y))

    # calculate scale increments
    scale_incs = calc_scale_increments(x,y,max_r,verbose=False)
    scale_incs.r.attrs["units"] = "m"
    r = scale_incs.r
    
    # assign and/or check length scales for filtering
    min_ell = r.values[1]
    max_ell = r.values[len(r)//2]
    if length_scales is None:
        length_scales = r.values[1:len(r)//2]
    # ensure ascending
    print("\n\n\n")
    print(f"min_ell = {min_ell}")
    print(f"max_ell = {min_ell}")
    length_scales = np.sort(length_scales)
    print("length_scales = ", length_scales)
    length_scales = length_scales[
        (length_scales <= max_ell)&(length_scales >= min_ell)
    ]
    print("clipped length_scales = ", length_scales)
    sys.exit(1)
    if len(length_scales) == 0:
        print(f"\nError! Invalid length_scales specified. \ell must be <= {max_ell:.5g} m")
        sys.exit(1)
    #endif
    
    # shifted field
    field_shifted = calc_increment_integrand(
        field, scale_incs, dummy_function, delta_x, delta_y,
        xdim=x_coord_name, ydim=y_coord_name, xbounds=x_bounds, ybounds=y_bounds,
        x_bound_field=x_coord_boundary, y_bound_field=y_coord_boundary,
        precision=precision, verbose=False
    )
    if x_coord_units == "deg":
        field_shifted = field_shifted.assign_coords(
            {x_coord_name:_x.values,y_coord_name:_y.values}
        )
        field_shifted[x_coord_name].attrs = _x.attrs
        field_shifted[y_coord_name].attrs = _y.attrs
    # filtered field
    field_filtered = calc_scale_space_integral(
        field_shifted, length_scales=length_scales, geometry="2D",
        name=name, kernel_gradient=False
    )
    #field_filtered = field_filtered.transpose(
    #    "r","time",x_coord_name,y_coord_name,"pressure"
    #)

    field_filtered = field_filtered.assign_attrs(
        {
            "units": input_attrs["units"],
            "description": f"Spatially filtered (a.k.a. coarse-grained) {field.name}",
            "LoSSETT_version": LOSSETT_VN,
            "input_data_attributes": repr(input_attrs)
        }
    ) # should add also kernel_attrs dict (for kernel type, dimensionality, length scale-to-resolution conversion)
    # and integration_attrs dict (for integral approximations e.g. Cartesian vs. spherical, uniform grid etc.)
    return field_filtered;



