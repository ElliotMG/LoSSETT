#!/usr/bin/env python3
import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy as cpy
# local imports
from lossett.calc.calc_inter_scale_transfers import calc_scale_increments, calc_scale_space_integral, calc_increment_integrand

radius_earth = 6.371e6 # radius of Earth in m
deg_to_m = 110000.0 # conversion of latitudinal degrees to m

def dummy_function(field, field_shifted, delta_r_x, delta_r_y, **kwargs):
    return field_shifted;

def filter_field(
        field, scale_incs, delta_x, delta_y,
        xdim, ydim, xbounds, ybounds,
        length_scales=None, name=None,
        x_bound_field=None, y_bound_field=None,
        conv_fac=1.0
):
    """
    Note that 2D filtering is currently hard-coded.
    """
    if name is None:
        name = field.name+"_filtered"
    # shifted field
    field_shifted = calc_increment_integrand(
        field, scale_incs, dummy_function, delta_x, delta_y,
        xdim, ydim, xbounds, ybounds,
        x_bound_field=x_bound_field, y_bound_field=y_bound_field,
        conv_fac=conv_fac
    )
    # filtered field
    field_filtered = calc_scale_space_integral(
        field_shifted, length_scales=length_scales, geometry="2D",
        name=name, kernel_gradient=False
    )
    return field_filtered;

if __name__ == "__main__":
    DATA_DIR = "/gws/nopw/j04/kscale/USERS/emg/data/DYAMOND_Summer/"
    OUT_DIR = "/gws/nopw/j04/kscale/USERS/dship/LoSSETT_out/"
    PLOT_DIR = "/home/users/dship/python/upscale/plots/"

    simid = "CTC5RAL"
    simid_long = "RAL3_n2560"
    #simid = "CTC5GAL"
    #simid_long = "GAL9_n2560"

    #mode = "precip"
    mode = "velocity"

    if mode == "velocity":
        vars = ["u","v","w"]
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
        ds_u_3D_RAL_p200 = ds_u_3D_RAL.rename(
            {
                "upward_air_velocity":"w",
                "x_wind":"u",
                "y_wind":"v"
            }
        ).isel(time=-1).sel(pressure=200).isel(latitude=slice(1,-1))
        test_data = ds_u_3D_RAL_p200

    elif mode == "precip":
        vars = ["precipitation_rate"]
        test_data = xr.open_dataset(
            os.path.join(
                DATA_DIR, "precip", "channel_"+simid_long,
                f"{simid_long}_DMn1280GAL9_precip_all.nc"
            ),
            mask_and_scale = True
        )
        test_data = test_data.isel(time=-1).isel(latitude=slice(1,-1))

    elif mode == "both":
        print("Not yet implemented.")
        sys.exit(1)
        
    else:
        print("Not yet implemented.")
        sys.exit(1)

    print(test_data)
    time = test_data.time

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

    max_r_deg = 10.0
    max_r_m = max_r_deg * deg_to_m

    prec = 1e-10

    scale_incs_deg = calc_scale_increments(lon,lat,max_r_deg,verbose=False)
    scale_incs_m = calc_scale_increments(lon_m,lat_m,max_r_m,verbose=False)

    # specify length scales
    r = scale_incs_m.r
    r.attrs["units"] = "m"
    length_scales = r.values[1:len(r)//2]

    # calc filtered fields
    for var in vars:
        field = test_data[var]
        field_filtered = filter_field(
            field, scale_incs_m, delta_lon_m, delta_lat_m,
            "longitude", "latitude", lon_m_bounds, lat_m_bounds,
            length_scales=length_scales, name=var+"_filtered",
            x_bound_field=lon_bound_field, y_bound_field=lat_bound_field,
            conv_fac=deg_to_m
        )
        print(field_filtered)
        # plot
        if var == "u":
            vmax = 60
            vmin = -vmax
            cmap="RdBu_r"
        elif var == "v":
            vmax = 40
            vmin = -vmax
            cmap="RdBu_r"
        elif var == "w":
            vmax = 0.08
            vmin = -vmax
            cmap="RdBu_r"
        elif var == "precipitation_rate":
            vmax = 10.0
            vmin = 0
            cmap="viridis"
        fig, axes = plt.subplots(
            nrows=2,ncols=2,
            subplot_kw={"projection":cpy.crs.PlateCarree()},
            figsize=(20,8)
        )
        ax = axes[0,0]
        pcol = ax.pcolormesh(
            field.longitude, field.latitude, field,
            vmin=vmin, vmax=vmax, cmap=cmap
        )
        ax.set_title(f"{field.name}")
        ax = axes[0,1]
        ell = field_filtered.length_scale.sel(length_scale=110*1000,method="nearest")
        ax.pcolormesh(
            field_filtered.longitude, field_filtered.latitude,
            field_filtered.sel(length_scale=ell),
            vmin=vmin, vmax=vmax, cmap=cmap
        )
        ax.set_title(f"{field_filtered.name}, L = {2*ell/1000:.3g}km")
        ax = axes[1,0]
        ell = field_filtered.length_scale.sel(length_scale=220*1000,method="nearest")
        ax.pcolormesh(
            field_filtered.longitude, field_filtered.latitude,
            field_filtered.sel(length_scale=ell),
            vmin=vmin, vmax=vmax, cmap=cmap
        )
        ax.set_title(f"{field_filtered.name}, L = {2*ell/1000:.3g}km")
        ax = axes[1,1]
        ell = field_filtered.length_scale.sel(length_scale=440*1000,method="nearest")
        ax.pcolormesh(
            field_filtered.longitude, field_filtered.latitude,
            field_filtered.sel(length_scale=ell),
            vmin=vmin, vmax=vmax, cmap=cmap
        )
        ax.set_title(f"{field_filtered.name}, L = {2*ell/1000:.3g}km")
        # tidy up
        plt.suptitle(f"CTC {simid_long}, {time.data}")
        for ax in axes.flatten():
            ax.coastlines()
        # colourbar
        plt.tight_layout()
        lower_left = axes[-1,0].get_position()
        lower_right = axes[-1,-1].get_position()
        tot_width = lower_right.x0 + lower_right.width - lower_left.x0
        width = 0.8*tot_width
        left = lower_left.x0 + (tot_width - width)/2
        bottom = lower_left.y0 - 0.05
        height = 0.03
        cax = fig.add_axes([left, bottom, width, height])
        cbar = fig.colorbar(
            pcol, cax=cax, orientation="horizontal",
            extend="both", label=f"[{field.units}]"
        )
        plt.savefig(
            os.path.join(PLOT_DIR,f"filtering_test_{var}_{simid}_{time.data}.png")
        )
        plt.show()

    print("\n\n\nEND.")



