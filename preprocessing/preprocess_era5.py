#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt

def setup_vars_DS(return_dates=False):
    # 1. data directory
    data_dir = "/gws/nopw/j04/kscale/USERS/dship/ERA5/3hourly/"

    # 2. variable names
    u_name = "u_component_of_wind"
    v_name = "v_component_of_wind"
    w_name = "vertical_velocity"
    t_name = "temperature"
    q_name = "specific_humidity"
    var_names = [u_name, v_name, w_name, t_name, q_name]

    # 3. define date range
    start_date = dt.date(2016,8,1) # take as command line argument, or from options file
    ndays = 40
    dates = [start_date + dt.timedelta(i) for i in range(ndays)]
    # check whether start_date and end_date are in different months!
    yearmonths = np.unique([date.strftime("%Y%m") for date in dates])
    
    if return_dates:
        return var_names, yearmonths, data_dir, dates;
    else:
        return var_names, yearmonths, data_dir;

def setup_vars_DW(return_dates=False):
    # 1. data directory
    data_dir = "/gws/nopw/j04/kscale/USERS/dship/ERA5/3hourly/"

    # 2. variable names
    u_name = "u_component_of_wind"
    v_name = "v_component_of_wind"
    w_name = "vertical_velocity"
    t_name = "temperature"
    q_name = "specific_humidity"
    var_names = [u_name, v_name, w_name, t_name, q_name]

    # 3. define date range
    start_date = dt.date(2020,1,1) # take as command line argument, or from options file
    ndays = 40
    dates = [start_date + dt.timedelta(i) for i in range(ndays)]
    # check whether start_date and end_date are in different months!
    yearmonths = np.unique([date.strftime("%Y%m") for date in dates])
    
    if return_dates:
        return var_names, yearmonths, data_dir, dates;
    else:
        return var_names, yearmonths, data_dir;

def setup_vars_yearmonth(year,month,return_dates=False,sampling="3h"):
    # 1. data directory
    data_dir = f"/gws/nopw/j04/kscale/USERS/dship/ERA5/{sampling}ourly/"

    # 2. variable names
    u_name = "u_component_of_wind"
    v_name = "v_component_of_wind"
    w_name = "vertical_velocity"
    t_name = "temperature"
    q_name = "specific_humidity"
    var_names = [u_name, v_name, w_name, t_name, q_name]

    # 3. define date range
    start_date = dt.date(year,month,1)
    ndays = 28 # obviously this is wrong...
    dates = [start_date + dt.timedelta(i) for i in range(ndays)]
    yearmonths = [f"{year:04d}{month:02d}"]
    
    if return_dates:
        return var_names, yearmonths, data_dir, dates;
    else:
        return var_names, yearmonths, data_dir;

def load_era5(var_names, yearmonths, data_dir, sampling="3h"):
    # Load files
    ds = xr.open_mfdataset(
        [
            os.path.join(
                data_dir, f"era5_{var_name}_{yearmonth}_{sampling}_0p5deg.nc"
            ) for var_name in var_names for yearmonth in yearmonths
        ],
        mask_and_scale=True
    )

    # Tidy up dataset (sort out coordinate names, drop unneeded variables; chunk)
    ds = ds.drop_vars("valid_time_bnds")
    ds = rename_dims(ds)
    ds = ds.chunk(
        chunks={
            "time":8,
            "longitude":720,
            "latitude":360,
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

    # Calculate density (should probably include moisture! At least optionally)
    gas_const_dry_air = 287.05
    g = 9.81
    p = xr.DataArray(
        dims = {"pressure":ds.pressure.data},
        data = ds.pressure.data,
        attrs = {'units': 'hPa'}
    )
    rho = ((p*100) / (gas_const_dry_air * ds.t))
    # Calculate w from omega for ERA5 (should probably save, then add check tos ee if it already exists)
    ds = ds.rename({"w":"omega"})
    ds = ds.assign(w=np.divide(-ds.omega,(rho*g)))
    ds.w.attrs["units"] = "m s-1" # should add other attribs from omega
    ds = ds.drop_vars(["omega","t","q"])

    # 9. Transpose
    ds = ds.transpose("time","pressure","latitude","longitude")
    return ds;

# function to sort out dims
def rename_dims(ds):    
    try:
        ds = ds.rename({'lon':'longitude'})
    except:
        pass
    try:
        ds = ds.rename({'lat':'latitude'})
    except:
        pass
    try:
        ds = ds.rename({'pressure_level':'pressure'})
    except:
        pass
    try:
        ds = ds.rename({'valid_time':'time'})
    except:
        pass
        
    return ds;

if __name__ == "__main__":
    # Get required file metadata
    #var_names, yearmonths, data_dir = setup_vars_DS(return_dates=False)
    var_names, yearmonths, data_dir = setup_vars_DW(return_dates=False)
    var_names, yearmonths, data_dir = setup_vars_yearmonth(
        2005, 1, sampling="3h", return_dates=False
    )

    print(var_names)

    # Load ERA5
    ds_era5 = load_era5(var_names, yearmonths, data_dir)
    print(ds_era5)
