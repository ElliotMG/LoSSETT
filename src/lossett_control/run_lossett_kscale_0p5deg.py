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

from lossett_control.preprocessing.preprocess_kscale import load_kscale
from lossett.calc.calc_inter_scale_transfers import calc_inter_scale_energy_transfer_kinetic

if __name__ == "__main__":
    # should take all of these from command line or an options file
    # simulation specification
    period = sys.argv[1]
    simid = sys.argv[2]
    tsteps_per_day = 8
    lon_bound_field = "periodic"
    lat_bound_field = np.nan

    # output directory
    OUT_DIR = f"/gws/nopw/j04/kscale/USERS/dship/LoSSETT_out/{period}"
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # day of simulation
    day = int(sys.argv[3])

    # calculation specification
    max_r_deg = 5.0 # should be command line option!
    tsteps = 8
    tchunks = 8
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
    ds_u_3D = load_kscale(simid,period,"0p5deg")

    # subset single day
    ds_u_3D = ds_u_3D.isel(time = slice((day-1)*tsteps_per_day,day*tsteps_per_day))

    # get start date + time
    date = pd.Timestamp(ds_u_3D.time[0].values).to_pydatetime()
    date_str = f"{date.year:04d}-{date.month:02d}-{date.day:02d}"

    print(f"\n\n\nCalculating {simid} DR indicator for {date_str}")

    # subset time; chunk time
    ds_u_3D = ds_u_3D.isel(time=slice(0,tsteps)).chunk(chunks={"time":tchunks})
    print("\nInput data:\n",ds_u_3D)

    # calculate kinetic DR indicator
    DR_indicator = calc_inter_scale_energy_transfer_kinetic(
        ds_u_3D, control_dict
    )

    # save to NetCDF
    n_l = len(DR_indicator.length_scale)
    fpath = os.path.join(OUT_DIR, f"inter_scale_energy_transfer_kinetic_{period}_{simid}_Nl_{n_l}_{date_str}.nc")
    print(f"\n{DR_indicator.name}:\n",DR_indicator)
    print(f"\nSaving {DR_indicator.name} to NetCDF at location {fpath}.")
    DR_indicator.to_netcdf(fpath)

    print("\n\n\nEND.\n")
