import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from lossett.calc.calc_inter_scale_transfers import calc_scale_increments, roll_with_boundary_handling, calc_delta_u_cubed

radius_earth = 6.371e6 # radius of Earth in m
deg_to_m = 110000.0 # conversion of latitudinal degrees to m

def compute_function_for_given_angle(
        field, loc, function, delta_x, delta_y, xdim, ydim, xbounds, ybounds,
        x_bound_field=np.nan, y_bound_field=np.nan, conv_fac=1.0
):
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
            func += function(
                field, field_shifted, n_x*delta_x, n_y*delta_y,
                delta_r_z=None, dims="2D"
            )
        func /= len(loc)
    else:
        n_x = loc[0] - origin_x_index
        n_y = loc[1] - origin_y_index
        field_shifted = roll_with_boundary_handling(
            field, n_x, n_y, delta_x/conv_fac, delta_y/conv_fac,
            xdim, ydim, xbounds/conv_fac, ybounds/conv_fac,
            x_bound_field=x_bound_field, y_bound_field=y_bound_field
        )
        func = function(
            field, field_shifted, n_x*delta_x, n_y*delta_y,
            delta_r_z=None, dims="2D"
        )
    return func;

def compute_angular_integrand_old(
        field, angle, function, delta_x, delta_y, xdim, ydim, xbounds, ybounds,
        x_bound_field=np.nan, y_bound_field=np.nan, conv_fac=1.0
):
    precision = 1e-10
    # define angle coord
    angle_coord = np.unique(angle)
    angle_coord = angle_coord[np.isfinite(angle_coord)]
    # initialise integrand
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
        phi_integrand.append(_phi_integrand)

    phi_integrand = xr.concat(phi_integrand,"angle")
    return phi_integrand;

def compute_angular_integrand_old_with_angle_func(
        field, angle, function, delta_x, delta_y, xdim, ydim, xbounds, ybounds,
        x_bound_field=np.nan, y_bound_field=np.nan, conv_fac=1.0
):
    precision = 1e-10
    # define angle coord
    angle_coord = np.unique(angle)
    angle_coord = angle_coord[np.isfinite(angle_coord)]
    # initialise integrand
    phi_integrand = []
    # can this loop be vectorized somehow?
    for phi in angle_coord:
        loc = xr.where((angle < phi+precision) & (angle >= phi-precision),1,0)
        loc = np.argwhere(loc.values).squeeze()
        _phi_integrand = compute_function_for_given_angle(
            field, loc, function, delta_x, delta_y, xdim, ydim, xbounds, ybounds,
            x_bound_field=x_bound_field, y_bound_field=y_bound_field, conv_fac=1.0
        )
        _phi_integrand = _phi_integrand.assign_coords({"angle":phi})
        phi_integrand.append(_phi_integrand)

    phi_integrand = xr.concat(phi_integrand,"angle")
    return phi_integrand;
    
def compute_angular_integrand_vectorised(
        field, angle, function, delta_x, delta_y, xdim, ydim, xbounds, ybounds,
        x_bound_field=np.nan, y_bound_field=np.nan, conv_fac=1.0
):
    return phi_integrand;

if __name__ == "__main__":
    DATA_DIR = "/gws/nopw/j04/kscale/USERS/emg/data/DYAMOND_Summer/"
    OUT_DIR = "/gws/nopw/j04/kscale/USERS/dship/LoSSETT_out/"
    PLOT_DIR = "/home/users/dship/python/upscale/plots/"

    simid = "CTC5RAL"

    ds_u_3D_RAL = xr.open_mfdataset(
        [
            os.path.join(DATA_DIR,fname) for fname in [
                f"u_DS_3D_{simid}.nc",
                f"v_DS_3D_{simid}.nc",
                f"w_DS_3D_{simid}.nc"
            ]
        ],
        mask_and_scale = True
    )
    ds_u_3D_RAL_t0_p200 = ds_u_3D_RAL.rename(
        {
            "upward_air_velocity":"w",
            "x_wind":"u",
            "y_wind":"v"
        }
    ).isel(time=0).sel(pressure=200)
    print(ds_u_3D_RAL_t0_p200)

    test_data = ds_u_3D_RAL_t0_p200

    lon = test_data.longitude
    lon_bounds = np.array([lon[0].values,lon[-1].values])
    lat = test_data.latitude
    lat_bounds = np.array([lat[0].values,lat[-1].values])

    lon_bound_field = "periodic"
    lat_bound_field = np.nan
    
    delta_lon = np.max(np.diff(lon))
    delta_lat = np.max(np.diff(lat))
    
    lon_m = lon*deg_to_m
    lon_m_bounds = np.array([lon_m[0].values,lon_m[-1].values])
    lat_m = lat*deg_to_m
    lat_m_bounds = np.array([lat_m[0].values,lat_m[-1].values])
    # should really add lon_m, lat_m to ds as coords -- would avoid needing to use
    # conv_fac later
    
    delta_lon_m = np.max(np.diff(lon_m))
    delta_lat_m = np.max(np.diff(lat_m))

    max_r_deg = 40.0
    max_r_m = max_r_deg * deg_to_m

    prec = 1e-10

    scale_incs_deg = CalcScaleIncrements(lon,lat,max_r_deg,verbose=False)
    scale_incs_m = CalcScaleIncrements(lon_m,lat_m,max_r_m,verbose=False)

    # specify length scales
    r = scale_incs_m.r
    length_scales = r.values[1:len(r)//2]

    # get origin
    r_0 = scale_incs_m.origin
    dims = r_0.dims
    xdim_loc = dims.index("r_x")
    ydim_loc = dims.index("r_y")
    origin_x_index = np.argwhere(np.isfinite(r_0.values)).squeeze()[xdim_loc]
    origin_y_index = np.argwhere(np.isfinite(r_0.values)).squeeze()[ydim_loc]

    # specify test distance, R
    R_deg = 3.0 # degrees
    mask_deg = scale_incs_deg.r_mask.sel(r=R_deg,method="nearest")
    angle_deg = scale_incs_deg.angle.where(mask_deg==1)
    R_m = R_deg*deg_to_m
    mask_m = scale_incs_m.r_mask.sel(r=R_m,method="nearest")
    angle_m = scale_incs_m.angle.where(mask_m==1)

    # old is identical to old_with_angle_func
    phi_integrand_deg = compute_angular_integrand_old(
        test_data, angle_deg, calc_delta_u_cubed, delta_lon, delta_lat,
        "longitude", "latitude", lon_bounds, lat_bounds,
        x_bound_field=lon_bound_field, y_bound_field=lat_bound_field
    )
    phi_integrand_deg = phi_integrand_deg.assign_coords({"r":R_deg})
    print(phi_integrand_deg)

    plt.figure()
    phi_integrand_deg.sel(latitude=0,method="nearest").plot(cmap="viridis")
    plt.show()
    
    phi_integrand_deg = compute_angular_integrand_old_with_angle_func(
        test_data, angle_deg, calc_delta_u_cubed, delta_lon, delta_lat,
        "longitude", "latitude", lon_bounds, lat_bounds,
        x_bound_field=lon_bound_field, y_bound_field=lat_bound_field
    )
    phi_integrand_deg = phi_integrand_deg.assign_coords({"r":R_deg})
    print(phi_integrand_deg)

    plt.figure()
    phi_integrand_deg.sel(latitude=0,method="nearest").plot(cmap="viridis")
    plt.show()

    sys.exit(1)
    print("\n\n\n",test_data)
    test_data = test_data.assign_coords(
        {
            "longitude_m": ("longitude", lon_m.data),
            "latitude_m": ("latitude", lat_m.data)
        }
    ).swap_dims(
        {
            "longitude": "longitude_m",
            "latitude": "latitude_m"
        }
    )
    print("\n\n\n",test_data)
    
    phi_integrand_m = compute_angular_integrand_old_with_angle_func(
        test_data, angle_m, calc_delta_u_cubed, delta_lon_m, delta_lat_m,
        "longitude_m", "latitude_m", lon_m_bounds, lat_m_bounds,
        x_bound_field=lon_bound_field, y_bound_field=lat_bound_field
    )
    phi_integrand_m = phi_integrand_m.assign_coords({"r":R_m})
    print(phi_integrand_m)

    plt.figure()
    phi_integrand_m.sel(latitude_m=0,method="nearest").plot(cmap="viridis")
    plt.show()

    #compute_angular_integrand_vectorised()
