#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
import iris

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

def load_kscale_native(period,datetime,driving_model,nested_model=None):
    DATA_DIR_ROOT = "/gws/nopw/j04/kscale/"
    dt_str = f"{datetime.year:04d}{datetime.month:02d}{datetime.day:02d}T{(datetime.hour%12)*12:02d}"
    
    # DYAMOND 3
    if period == "DYAMOND3":
        DATA_DIR = os.path.join(DATA_DIR_ROOT,"DYAMOND3_data")
        if driving_model == "n2560RAL3":
            DATA_DIR = os.path.join(DATA_DIR,"5km-RAL3")
            dri_mod_str = "n2560_RAL3p3"
        elif driving_model == "n1280GAL9":
            DATA_DIR = os.path.join(DATA_DIR,"10km-GAL9-nest")
            dri_mod_str = "n1280_GAL9_nest"
        elif driving_model == "n1280CoMA9":
            DATA_DIR = os.path.join(DATA_DIR,"10km-CoMA9")
            dri_mod_str = "n1280_CoMA9"
        if nested_model is None or nested_model == "glm":
            DATA_DIR = os.path.join(DATA_DIR,"glm","field.pp","apverc.pp")
            nest_mod_str = "glm"

        fpath = os.path.join(DATA_DIR,f"{nest_mod_str}.{dri_mod_str}.apverc_{dt_str}.pp")

    # DYAMOND SUMMER
    elif period == "DYAMOND_SUMMER":
        DATA_DIR = os.path.join(DATA_DIR_ROOT, "DATA","outdir_20160801T0000Z")
        if driving_model == "n1280RAL3":
            DATA_DIR = os.path.join(DATA_DIR,"DMn1280RAL3")
        elif driving_model == "n1280GAL9":
            DATA_DIR = os.path.join(DATA_DIR,"DMn1280GAL9")

    # DYAMOND WINTER
    elif period == "DYAMOND_WINTER":
        DATA_DIR = os.path.join(DATA_DIR_ROOT, "DATA","outdir_20200120T0000Z")

    # LOAD u,v,w from PP file using Iris
    if not os.path.exists(fpath):
        print(f"ERROR: data does not exist at {fpath}")
        sys.exit(1)
    else:
        print(f"Loading velocity data from {fpath}")
        data_iris = iris.load(fpath)
        
    print(data_iris)
    uvw = []
    # convert to Xarray
    for name in ["x_wind","y_wind","upward_air_velocity"]:
        vel_cpt = xr.DataArray.from_iris(data_iris.extract_cube(iris.Constraint(name=name)))
        print(vel_cpt)
        uvw.append(vel_cpt)
    ds = xr.merge(uvw).rename(
        {
            "x_wind": "u",
            "y_wind": "v",
            "upward_air_velocity": "w"
        }
    )
    return ds;

if __name__ == "__main__":
    period="DYAMOND3"
    datetime = dt.datetime(2020,9,15,0)
    driving_model = "n2560RAL3"
    ds = load_kscale_native(period,datetime,driving_model,nested_model=None)
    ds_u_t0_p200 = ds.isel(time=0).sel(pressure=200,method="nearest")
    print(ds_u_t0_p200)
    sys.exit(1)
    ds = load_kscale("CTC5RAL","DS","0p5deg")
    print(ds)
