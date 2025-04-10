#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt

def setup_vars_DS(return_dates=False,grid="0p5deg"):
    # This currently uses the already-processed DS data in Elliot's directory;
    # would be better to use the data in $KSCALE/DATA instead
    
    # 1. data directory
    data_dir = "/gws/nopw/j04/kscale/USERS/emg/data/DYAMOND_Summer/"

    # 2. variable names
    u_name = "u"
    v_name = "v"
    w_name = "w"
    var_names = [u_name, v_name, w_name]

    # 3. define date range
    start_date = dt.date(2016,8,1) # take as command line argument, or from options file
    ndays = 40
    dates = [start_date + dt.timedelta(i) for i in range(ndays)]
    
    if return_dates:
        return var_names, data_dir, dates;
    else:
        return var_names, data_dir;

def setup_vars_DW(return_dates=False,grid="0p5deg"):

    return 0;

def load_kscale(simid, period, grid):
    tsteps_per_day = 8
    if period == "DS":
        var_names, data_dir, dates = setup_vars_DS(return_dates=True,grid=grid)
        
    if period == "DS" and grid == "0p5deg":
        # This currently uses the already-processed DS data in Elliot's directory;
        # would be better to use the data in $KSCALE/DATA instead
        ds_u_3D = xr.open_mfdataset(
            [
                os.path.join(data_dir,f"{var_name}_DS_3D_{simid}.nc")\
                for var_name in var_names
            ],
            mask_and_scale = True
        )
        # subset dates
        start = dates[0]
        start_datetime = dt.datetime(start.year,start.month,start.day,3)
        end = dates[-1] + dt.timedelta(1)
        end_datetime = dt.datetime(end.year,end.month,end.day,0)
        ds_u_3D = ds_u_3D.sel(time=slice(start_datetime,end_datetime))

    # rename variables
    ds_u_3D = ds_u_3D.rename(
        {
            "x_wind":"u",
            "y_wind":"v",
            "upward_air_velocity":"w"
        }
    )
    
    return ds_u_3D;

if __name__ == "__main__":
    ds = load_kscale("CTC5RAL","DS","0p5deg")
    print(ds)
