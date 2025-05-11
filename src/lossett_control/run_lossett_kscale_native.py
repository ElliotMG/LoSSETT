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
import mo_pack

from ..preprocessing.preprocess_kscale import load_kscale_native
from ..calc.calc_inter_scale_transfers import calc_inter_scale_energy_transfer_kinetic

if __name__ == "__main__":
    # should take all of these from command line or an options file
    # simulation specification
    period = sys.argv[1]
    dri_mod_id = sys.argv[2]
    nest_mod_id = sys.argv[3]
    tsteps_per_day = 8
    lon_bound_field = "periodic"
    lat_bound_field = np.nan

    if nest_mod_id in ["None","none","glm"]:
        nest_mod_id = "glm"

    # output directory
    OUT_DIR = f"/gws/nopw/j04/kscale/USERS/dship/LoSSETT_out/{period}/{dri_mod_id}/{nest_mod_id}"
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # day & hour of simulation
    date = sys.argv[4]
    hour = int(sys.argv[5])
    datetime = dt.datetime.strptime(date, "%Y-%m-%d").replace(hour=hour)
    dt_str = f"{datetime.year:04d}{datetime.month:02d}{datetime.day:02d}T{(datetime.hour%12)*12:02d}"

    # calculation specification
    max_r_deg = 0.4 # should be command line option!
    tsteps = 1
    tchunks = 1
    pchunks = 1
    prec = 1e-10
    chunk_latlon = False
    subset_lat = True

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
    ds_u_3D = load_kscale_native(
        period,datetime,driving_model=dri_mod_id,nested_model=nest_mod_id
    )

    ## subset single day
    #ds_u_3D = ds_u_3D.isel(time = slice((day-1)*tsteps_per_day,day*tsteps_per_day))
    
    # subset single time and pressure level
    plev=200
    tstep=0
    ds_u_3D = ds_u_3D.isel(time=tstep).sel(pressure=plev,method="nearest")
    ds_u_3D = ds_u_3D.expand_dims(dim=["time","pressure"])
    print(ds_u_3D)

    if nest_mod_id == "glm":
        print(f"\n\n\nCalculating {period} global {dri_mod_id} DR indicator for {dt_str}")
    else:
        print(f"\n\n\nCalculating {period} {nest_mod_id} (driven by {dri_mod_id}) DR indicator for {dt_str}")

    # subset time; chunk time
    ds_u_3D = ds_u_3D.isel(time=slice(0,tsteps)).chunk(chunks={"time":tchunks,"pressure":pchunks})

    # chunk lat & lon (TEST!)
    if chunk_latlon:
        latchunks = 2560
        lonchunks = 2560
        ds_u_3D = ds_u_3D.chunk(chunks={"longitude":lonchunks,"latitude":latchunks})
    subset_str=""
    if subset_lat:
        latmin = -50
        latmax = 50
        ds_u_3D = ds_u_3D.sel(latitude=slice(latmin,latmax))
        subset_str = "_50S-50N"

    print("\nInput data:\n",ds_u_3D)
    
    # calculate kinetic DR indicator
    DR_indicator = calc_inter_scale_energy_transfer_kinetic(
        ds_u_3D, control_dict
    )

    # save to NetCDF
    n_l = len(DR_indicator.length_scale)
    fpath = os.path.join(
        OUT_DIR,
        f"{nest_mod_id}.{dri_mod_id}_inter_scale_energy_transfer_kinetic_Nl_{n_l}_{dt_str}{subset_str}_p{plev:04d}_tstep{tstep}.nc"
    )
    print(f"\n{DR_indicator.name}:\n",DR_indicator)
    print(f"\nSaving {DR_indicator.name} to NetCDF at location {fpath}.")
    DR_indicator.to_netcdf(fpath)

    print("\n\n\nEND.\n")
