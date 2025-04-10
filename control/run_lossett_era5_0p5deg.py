#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import cartopy as cpy

from ..preprocessing import preprocess_era5
from ..calc.calc_inter_scale_transfers import calc_inter_scale_energy_transfer_kinetic

if __name__ == "__main__":
    # should take all of these from command line or an options file
    # simulation specification
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    day = int(sys.argv[3])
    tsteps_per_day = 8
    sampling = f"{int(24/tsteps_per_day)}h"
    print(sampling)
    lon_bound_field = "periodic"
    lat_bound_field = np.nan # not really sure how to deal with the poles?

    # output directory
    OUT_DIR = f"/gws/nopw/j04/kscale/USERS/dship/LoSSETT_out/ERA5"
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # calculation specification
    max_r_deg = 40.0 # should be command line option!
    tsteps = 8
    tchunks = 2
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
    var_names, yearmonths, data_dir = preprocess_era5.setup_vars_yearmonth(
        year, month, sampling="3h", return_dates=False
    )
    ds_u_3D = preprocess_era5.load_era5(var_names, yearmonths, data_dir)

    # subset single day
    ds_u_3D = ds_u_3D.isel(time = slice((day-1)*tsteps_per_day,day*tsteps_per_day))
    print(ds_u_3D)

    # create date string
    date_str = f"{year:04d}-{month:02d}-{day:02d}"

    print(f"\n\n\nCalculating ERA5 DR indicator for {date_str}")

    # subset time; chunk time
    ds_u_3D = ds_u_3D.isel(time=slice(0,tsteps)).chunk(chunks={"time":tchunks})
    print("\nInput data:\n",ds_u_3D)

    # calculate kinetic DR indicator
    DR_indicator = calc_inter_scale_energy_transfer_kinetic(
        ds_u_3D, control_dict
    )

    # save to NetCDF
    n_l = len(DR_indicator.length_scale)
    fpath = os.path.join(OUT_DIR, f"inter_scale_energy_transfer_kinetic_ERA5_0p5deg_Nl_{n_l}_{date_str}.nc")
    print(f"\n{DR_indicator.name}:\n",DR_indicator)
    print(f"\nSaving {DR_indicator.name} to NetCDF at location {fpath}.")
    DR_indicator.to_netcdf(fpath)

    print("\n\n\nEND.\n")
