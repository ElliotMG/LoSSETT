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

from lossett_control.preprocessing import preprocess_era5
from lossett.calc.calc_inter_scale_transfers import calc_inter_scale_energy_transfer_kinetic

if __name__ == "__main__":
    # should take all of these from command line or an options file
    # simulation specification
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    day = int(sys.argv[3])
    tsteps_per_day = 8
    sampling = f"{int(24/tsteps_per_day)}h"
    lon_bound_field = "periodic"
    lat_bound_field = np.nan # not really sure how to deal with the poles?

    # calculation specification
    max_r_deg = float(sys.argv[4])
    #max_r_deg = 33.0
    tsteps = 8
    tchunks = 8
    pchunks = 12
    prec = 1e-10

    # output directory
    #OUT_DIR_ROOT = "/gws/nopw/j04/kscale/USERS/dship/LoSSETT_out/"
    OUT_DIR_ROOT = "/work/scratch-pw4/dship/LoSSETT/output/"
    OUT_DIR = os.path.join(OUT_DIR_ROOT, "ERA5")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

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
        year, month, sampling="3h", return_dates=False, moist=True
    )
    ds_u_3D = preprocess_era5.load_era5(var_names, yearmonths, data_dir, drop_non_vel=False)

    # subset single day
    ds_u_3D = ds_u_3D.isel(time = slice((day-1)*tsteps_per_day,day*tsteps_per_day))
    print(ds_u_3D)
    sys.exit(1)

    # create date string
    date_str = f"{year:04d}-{month:02d}-{day:02d}"

    print(f"\n\n\nCalculating ERA5 DR indicator for {date_str}")

    # subset time; chunk time
    ds_u_3D = ds_u_3D.isel(time=slice(0,tsteps)).chunk(chunks={"time":tchunks,"pressure":pchunks})
    print("\nInput data:\n",ds_u_3D)
    
    # specify length scales (10 length scales per decade unless 2dx > spacing between consecutive \ell)
    length_scales = np.array(
        [55,110,165,220,275,330,400,500,640,800,1000,1250,1600,2000,2500,3200,4000,5000]
    )
    length_scales = 1000.0 * length_scales # convert to m; ensure float

    # calculate kinetic DR indicator
    Dl_u = calc_inter_scale_energy_transfer_kinetic(
        ds_u_3D, control_dict,
        length_scales=length_scales
    )
    
    # ensure correct dimension ordering
    Dl_u = Dl_u.transpose("length_scale","time","pressure","latitude","longitude")

    # save to NetCDF
    n_l = len(Dl_u.length_scale)
    L_min = Dl_u.length_scale[0].values/1000
    L_max = Dl_u.length_scale[-1].values/1000
    fpath = os.path.join(
        OUT_DIR,
        "ERA5_inter_scale_transfer_of_kinetic_energy_0p5deg_"\
        f"Lmin_{L_min:05.0f}_Lmax_{L_max:05.0f}_{date_str}.nc"
    )
    print(f"\n{Dl_u.name}:\n",Dl_u)
    print(f"\nSaving {Dl_u.name} to NetCDF at location {fpath}.")
    Dl_u.to_netcdf(fpath)

    print("\n\n\nEND.\n")
