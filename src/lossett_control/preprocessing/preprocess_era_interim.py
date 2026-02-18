#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt

def setup_vars_yearmonth(year,month,return_dates=False,sampling="3h",moist=False):
    # 1. data directory
    data_dir = f"/gws/nopw/j04/kscale/USERS/dship/ERA5/{sampling}ourly/"

    # 2. variable names
    u_name = "u_component_of_wind"
    v_name = "v_component_of_wind"
    w_name = "vertical_velocity"
    t_name = "temperature"
    q_name = "specific_humidity"
    if moist:
        var_names = [u_name, v_name, w_name, t_name, q_name]
    else:
        var_names = [u_name, v_name, w_name, t_name]

    # 3. define date range
    start_date = dt.date(year,month,1)
    ndays = 28 # obviously this is wrong...
    dates = [start_date + dt.timedelta(i) for i in range(ndays)]
    yearmonths = [f"{year:04d}{month:02d}"]
    
    if return_dates:
        return var_names, yearmonths, data_dir, dates;
    else:
        return var_names, yearmonths, data_dir;

def get_fpaths_era_interim(dates, hours=[0,6,12,18]):
    DATA_DIR = "/badc/ecmwf-era-interim/data/gg/ap/"
    fpaths = [
        os.path.join(
            DATA_DIR,
            f"{date.year:04d}",
            f"{date.month:02d}",
            f"{date.day:02d}",
            f"ggap{date.year:04d}{date.month:02d}{date.day:02d}{hour:02d}00.nc"
        ) for hour in hours for date in dates
    ]
    return fpaths;

def load_era_interim(
        dates,
        hours=[0,6,12,18],
        var_names=["u","v","w"],
        plevs=[925,850,700,600,500,400,300,250,200,150,100]
):
    fpaths = get_fpaths_era_interim(dates, hours=hours)
    # Load files
    ds = xr.open_mfdataset(
        fpaths,
        mask_and_scale=True
    )

    # Tidy up dataset
    ds = rename_dims(ds)
    ds = ds.rename(
        {
            "U": "u",
            "V": "v",
            "W": "omega",
            "Q": "q"
        }
    )

    # Check lat,lon direction and conventions (e.g. is lon [-180,180] or [0,360]?)
    lon_attrs = ds.longitude.attrs
    ds.coords["longitude"] = (ds.coords["longitude"] + 180) % 360 - 180
    ds = ds.sortby(ds.longitude)
    ds.longitude.attrs = lon_attrs
    ds = ds.sortby(ds.latitude) # has no effect if lat already correctly oriented

    # Subset pressure levels
    ds = ds.sel(pressure=plevs, method="nearest")

    if "w" in var_names:
        # Calculate density (should probably include moisture! At least optionally)
        gas_const_dry_air = 287.05
        g = 9.81
        p = xr.DataArray(
            dims = {"pressure":ds.pressure.data},
            data = ds.pressure.data,
            attrs = {'units': 'hPa'}
        )
        rho = ((p*100) / (gas_const_dry_air * ds.T))
        # Calculate w from omega (should probably save, then add check to see if it already exists)
        ds = ds.assign(w=np.divide(-ds.omega,(rho*g)))
        ds.w.attrs["units"] = "m s-1" # should add other attribs from omega
        ds = ds.assign(rho=rho)
        ds.rho.attrs = \
            {
                "units" : "kg m-3",
                "description": "dry air density computed via ideal gas law"
            }
    #endif

    # Transpose
    ds = ds.transpose("time","pressure","latitude","longitude")

    # Extract required variables
    ds = ds[var_names]
    
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
        ds = ds.rename({'p':'pressure'})
    except:
        pass
    try:
        ds = ds.rename({'valid_time':'time'})
    except:
        pass
    try:
        ds = ds.rename({'t':'time'})
    except:
        pass
        
    return ds;

if __name__ == "__main__":
    startdate = dt.datetime(2005,1,1)
    enddate = dt.datetime(2005,1,2)
    ndays = 1+(enddate - startdate).days
    print(ndays)
    dates = [startdate + dt.timedelta(days=iday) for iday in range(ndays)]
    print(dates)

    ds = load_era_interim(dates)
    print(ds)
