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
import datetime as dt

from ..preprocessing import preprocess_era5
from ..preprocessing.preprocess_era5 import rename_dims
from ..calc.calc_inter_scale_transfers import calc_inter_scale_energy_transfer_kinetic

def load_era5_2deg():#var_names, yearmonths, data_dir):
    # Load files
    data_dir = "/gws/nopw/j04/kscale/USERS/dship/ERA5/3hourly/"
    var_names = ["u_component_of_wind","v_component_of_wind","vertical_velocity","temperature"]
    yearmonths = ["201608"]
    sampling = "3h"
    start_date = dt.date(2016,8,1)
    ds = xr.merge(
        [
            xr.open_dataset(
                os.path.join(
                    data_dir, f"era5_{var_name}_{yearmonth}_{sampling}_2deg.nc"
                ),
                mask_and_scale=True
            )for var_name in var_names for yearmonth in yearmonths
        ]
    ) # open_mfdataset wasn't working for some reason? Was throwing
    # "ValueError: Could not find any dimension coordinates to use to order the datasets for concatenation"
    # but it shouldn't have been trying to concatenate anything?

    # Tidy up dataset (sort out coordinate names, drop unneeded variables; chunk)
    ds = ds.drop_vars("valid_time_bnds")
    ds = rename_dims(ds)
    ds = ds.chunk(
        chunks={
            "time":8,
            "longitude":180,
            "latitude":90,
            "pressure":12
        }
    )
    
    # Fix time coordinates (DOUBLE CHECK THIS IS CORRECT!)
    fix_time_coord = False
    if fix_time_coord:
        start = dt.datetime(start_date.year,start_date.month,start_date.day,3)
        ds = ds.assign_coords(
            {
                "time": np.array(
                    [start + i*dt.timedelta(hours=3) for i in range(len(ds.time))]
                )
            }
        )

    # Check lat,lon direction and conventions (e.g. is lon [-180,180] or [0,360]?)
    lon_attrs = ds.longitude.attrs
    ds.coords["longitude"] = (ds.coords["longitude"] + 180) % 360 - 180
    ds = ds.sortby(ds.longitude)
    ds.longitude.attrs = lon_attrs
    ds = ds.sortby(ds.latitude) # has no effect if lat already correctly oriented

    # Calculate density (should probably include moisture! At least optionally)
    gas_const_dry_air = 287.05
    g = 9.81
    p = xr.DataArray(
        dims = {"pressure":ds.pressure.data},
        data = ds.pressure.data,
        attrs = {'units': 'hPa'}
    )
    rho = ((p*100) / (gas_const_dry_air * ds.t))
    # Calculate w from omega for ERA5 (should probably save, then add check to see if it already exists)
    ds = ds.rename({"w":"omega"})
    ds = ds.assign(w=np.divide(-ds.omega,(rho*g)))
    ds.w.attrs["units"] = "m s-1" # should add other attribs from omega
    ds = ds.drop_vars(["omega","t"])

    # 9. Transpose
    ds = ds.transpose("time","pressure","latitude","longitude")
    return ds;

if __name__ == "__main__":
    # should take all of these from command line or an options file
    # simulation specification
    year = 2016
    month = 8
    day = 1
    tsteps_per_day = 8
    sampling = f"{int(24/tsteps_per_day)}h"
    print(f"\nTemporal sampling rate = {sampling}")
    lon_bound_field = "periodic"
    lat_bound_field = np.nan # not really sure how to deal with the poles?

    # output directory
    OUT_DIR = "/gws/nopw/j04/kscale/USERS/dship/LoSSETT_out/ERA5"
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # calculation specification
    max_r_deg = 10.0 # should be command line option!
    tsteps = 2
    tchunks = 1
    pchunks = 12
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
    ds_u_3D = load_era5_2deg()
    print("\n\n\nOriginal dataset:\n", ds_u_3D)

    ## subset single day
    #ds_u_3D = ds_u_3D.isel(time = slice((day-1)*tsteps_per_day,day*tsteps_per_day))
    
    ## subset single time
    #tstep=0
    #ds_u_3D = ds_u_3D.isel(time=tstep)
    #ds_u_3D = ds_u_3D.expand_dims(dim="time")

    ## subset single pressure
    plev=200
    ds_u_3d = ds_u_3D.sel(pressure=plev,method="nearest")
    #ds_u_3D = ds_u_3D.expand_dims(dim="pressure")

    # create date string
    date_str = f"{year:04d}-{month:02d}-{day:02d}"

    print(f"\n\n\nCalculating ERA5 DR indicator for {date_str}")

    # subset time; chunk time
    ds_u_3D = ds_u_3D.isel(time=slice(0,tsteps)).chunk(chunks={"time":tchunks})#,"pressure":pchunks})
    print("\nInput data:\n",ds_u_3D)

    # calculate kinetic DR indicator
    DR_indicator = calc_inter_scale_energy_transfer_kinetic(
        ds_u_3D, control_dict
    )
    print(f"\n{DR_indicator.name}:\n",DR_indicator)

    # save to NetCDF
    n_l = len(DR_indicator.length_scale)
    fpath = os.path.join(OUT_DIR, f"inter_scale_energy_transfer_kinetic_ERA5_2deg_Nl_{n_l:02d}_{date_str}.nc")
    print(f"\nSaving {DR_indicator.name} to NetCDF at location {fpath}.")
    DR_indicator.to_netcdf(fpath)

    print("\n\n\nEND.\n")
