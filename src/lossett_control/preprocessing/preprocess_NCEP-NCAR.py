#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

def setup_vars_NCEP_NCAR(moist=False):
    # 1. data directory
    data_dir = f"/home/users/dship/NCEP-NCAR_Reanalysis_1/"

    # 2. variable names
    u_name = "uwnd"
    v_name = "vwnd"
    omega_name = "omega"
    t_name = "air"
    q_name = "shum"
    if moist:
        var_names = [u_name, v_name, omega_name, t_name, q_name]
        new_var_names = ["u", "v", "omega", "T", "q"]
    else:
        var_names = [u_name, v_name, omega_name, t_name]
        new_var_names = ["u", "v", "omega", "T"]
        
    return var_names, new_var_names, data_dir;

def load_NCEP_NCAR(var_names, new_var_names, year, data_dir, moist=False):
    
    # Load files
    ds = xr.open_mfdataset(
        [
            os.path.join(
                data_dir, f"{var_name}.{year}.nc"
            ) for var_name in var_names
        ],
        mask_and_scale=True
    )

    # rename coords and variables for consistency with other datasets
    ds = ds.rename(
        {
            "lon": "longitude",
            "lat": "latitude",
            "level": "pressure"
        }
    )
    for iv, var_name in enumerate(var_names):
        ds = ds.rename({var_name:new_var_names[iv]})

    # Convert longitude from [0,360] to [-180,180]
    lon_attrs = ds.longitude.attrs
    ds.coords["longitude"] = (ds.coords["longitude"] + 180) % 360 - 180
    ds = ds.sortby(ds.longitude)
    ds.longitude.attrs = lon_attrs
    # convert latitude from [90,-90] to [-90,90]
    ds = ds.sortby(ds.latitude) # has no effect if lat already correctly oriented

    # re-chunk to give a sensible number of chunks
    ds = ds.chunk(chunks={
        "time": 40, # 4 per day = 10 day chunks
        "pressure": 17
    })

    if moist:
        # q has no values above p = 300 hPa; fill with zeros
        q_attrs = ds.q.attrs
        ds = ds.rename({"q":"q_nan"})
        ds = ds.assign(q=ds.q_nan.where(ds.pressure >= 300, other=0.0))
        ds.q.attrs = q_attrs
        ds = ds.drop_vars("q_nan")
    #endif
    # omega has no values above p = 100 hPa; fill with zeros
    omega_attrs = ds.omega.attrs
    ds = ds.rename({"omega":"omega_nan"})
    ds = ds.assign(omega=ds.omega_nan.where(ds.pressure >= 100, other=0.0))
    ds.omega.attrs = omega_attrs
    ds = ds.drop_vars("omega_nan")
    
    # Calculate density
    gas_const_dry_air = 287.05
    g = 9.81
    p = xr.DataArray(
        dims = {"pressure":ds.pressure.data},
        data = ds.pressure.data,
        attrs = {'units': 'hPa'},
        name = "pressure"
    )
    if moist:
        Tv = ds.T * (1 + 0.608*ds.q)
        rho = ((p*100) / (gas_const_dry_air * Tv))
    else:
        rho = ((p*100) / (gas_const_dry_air * ds.T))
    rho.name = "rho"
    rho.attrs["units"] = "kg m-3"
    if moist:
        rho.attrs["description"] = "Air density calculated from pressure and virtual temperature "\
            "(specific humidity only) via the ideal gas law."
    else:
        rho.attrs["description"] = "Air density calculated from pressure and dry air temperature "\
            "via the ideal gas law."
    ds["rho"] = rho
        
    # Calculate w from omega (should probably save, then add check to see if it already exists)
    ds["w"] = np.divide(-ds.omega,(rho*g))
    ds.w.attrs["units"] = "m s-1" # should add other attribs from omega?
    ds.w.attrs["description"] = "Vertical velocity calculated from pressure vertical velocity (omega) "\
        "and air density (rho) via hydrostatic balance."
    ds = ds.drop_vars(["omega"])

    print("\n\n\n", ds)

    # Transpose
    ds = ds.transpose("time","pressure","latitude","longitude")
    return ds;

if __name__ == "__main__":
    year = int(sys.argv[1]) # currently can only be 2005 or 2016
    if year not in [2005, 2016]:
        print("Input error! Year must be 2005 or 2016.")
        sys.exit(1)

    moist = False
    
    # Get required file metadata
    var_names, new_var_names, data_dir = setup_vars_NCEP_NCAR(moist=moist)

    # Load NCEP-NCAR for specified year
    ds = load_NCEP_NCAR(var_names, new_var_names, year, data_dir, moist=moist)
    print("\n\n\nNCEP-NCAR data:\n", ds)

    # save processed data
    if moist:
        fpath = os.path.join(data_dir, f"NCEP_NCAR_uvwTqrho_moist_{year}.nc")
    else:
        fpath = os.path.join(data_dir, f"NCEP_NCAR_uvwTrho_{year}.nc")
    print(f"\n\nSaving processed data to {fpath}")
    ds.to_netcdf(fpath)
    
    print("\n\n\nEND.")
    
