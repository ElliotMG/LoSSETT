#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import cartopy as cpy

#from ..preprocessing.preprocess_kscale import load_u_dc009
from ..calc.calc_inter_scale_transfers import calc_inter_scale_energy_transfer_kinetic

if __name__ == "__main__":
    # should take all of these from command line or an options file
    # simulation specification
    month = int(sys.argv[1])
    day = int(sys.argv[2])
    simid = "u-dc009"
    tsteps_per_day = 4
    lon_bound_field = np.nan
    lat_bound_field = np.nan

    # output directory
    OUT_DIR = f"/gws/nopw/j04/kscale/USERS/dship/LoSSETT_out/{simid}"
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # calculation specification
    max_r_deg = 20.0 # should be command line option!
    tsteps = 4
    tchunks = 4
    pchunks = 7
    prec = 1e-10

    control_dict = {
        "max_r": max_r_deg,
        "max_r_units": "deg",
        "angle_precision": prec,
        "x_coord_name": "longitude",
        "x_coord_units": "deg",
        "x_coord_boundary": lon_bound_field,
        "y_coord_name": "latitude",
        "y_coord_units": "deg",
        "y_coord_boundary": lat_bound_field
    }

    # open data
    #ds_u_3D = load_kscale(simid,period,"0p5deg")
    # u,v files: every 2 days; 3-hourly; 7 pressure levels (no 300hPa)
    # w files: every 10 days; 6-hourly; 8 pressure levels
    DATA_DIR = "/work/scratch-pw2/bakera/n1280_data_for_upscale/u-dc009/"
    start = dt.datetime(1980,month,day)
    ds_u_3D = xr.open_mfdataset(
        [
            os.path.join(
                DATA_DIR,
                "x_wind",
                f"u-dc009_ap7_x_wind_1980{month:02d}_NAtl.nc"
            ),
            os.path.join(
                DATA_DIR,
                "y_wind",
                f"u-dc009_ap7_y_wind_1980{month:02d}_NAtl.nc"
            ),
            os.path.join(
                DATA_DIR,
                "upward_air_velocity",
                f"u-dc009_apc_upward_air_velocity_1980{month:02d}_NAtl.nc"
            )
        ],
        drop_variables=["leadtime","latitude_longitude"],
        mask_and_scale=True,
        join="inner" # means only take matching temporal data; need to check whether values
        # are instantaneous or time-average as this would need to change if the latter
    ).rename(
        {
            "upward_air_velocity":"w",
            "x_wind":"u",
            "y_wind":"v"
        }
    )
    print(ds_u_3D)
    ds_u_3D = ds_u_3D.transpose("time","longitude","latitude","pressure")
    print(ds_u_3D)

    # subset single day
    # rewrite this to work properly with datetimes
    ds_u_3D = ds_u_3D.isel(time = slice((day-1)*tsteps_per_day,day*tsteps_per_day))

    # get start date + time
    date_str = f"{start.year:04d}-{start.month:02d}-{start.day:02d}"

    # subset time; chunk time
    ds_u_3D = ds_u_3D.isel(time=slice(0,tsteps)).chunk(chunks={"time":tchunks,"pressure":pchunks})

    # calculate kinetic DR indicator
    print(f"\n\n\nCalculating {simid} DR indicator for {date_str}")
    print("\nInput data:\n",ds_u_3D)
    DR_indicator = calc_inter_scale_energy_transfer_kinetic(
        ds_u_3D, control_dict
    )

    # save to NetCDF
    n_l = len(DR_indicator.length_scale)
    fpath = os.path.join(OUT_DIR, f"inter_scale_energy_transfer_kinetic_{simid}_Nl_{n_l}_{date_str}.nc")
    print(f"\n{DR_indicator.name}:\n",DR_indicator)
    print(f"\nSaving {DR_indicator.name} to NetCDF at location {fpath}.")
    DR_indicator.to_netcdf(fpath)

    print("\n\n\nEND.\n")
