#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
import iris

from lossett_control.preprocessing.preprocess_kscale import \
    parse_period_id, parse_dri_mod_id, parse_nest_mod_id, \
    embed_inner_grid_in_global

if __name__ == "__main__":
    _period=sys.argv[1]
    _dri_mod_id = sys.argv[2]
    _nest_mod_id = sys.argv[3]
    outer_grid = sys.argv[4]
    year = int(sys.argv[5])
    month = int(sys.argv[6])
    day = int(sys.argv[7])
    hour = int(sys.argv[8])
    plevs = [100,150,200,250,300,400,500,600,700,850,925,1000]
    #plevs = [200]
    
    # parse period, driving model, nested model
    period = parse_period_id(_period)
    dri_mod_id, dri_mod_str = parse_dri_mod_id(period,_dri_mod_id)
    nest_mod_id, nest_mod_str = parse_nest_mod_id(period,dri_mod_id,_nest_mod_id)
    grid_id = outer_grid.lower()
    
    if grid_id != "n1280":
        print(f"Error! Outer grid {grid_id} not yet implemented. Exiting.")
        sys.exit(1)
    #endif

    # set up input and output directories
    DATA_DIR = \
            f"/gws/nopw/j04/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/{period}"
    SAVE_DIR = os.path.join(DATA_DIR, f"{grid_id}_regrid", "embedded")
    Path(SAVE_DIR).mkdir(parents=True,exist_ok=True)

    # get date, time
    datetime = dt.datetime(year,month,day,hour)
    dt_str = f"{datetime.year:04d}{datetime.month:02d}{datetime.day:02d}T{(datetime.hour//12)*12:02d}"
    
    print("\n\n\nPreprocessing details:")
    print(
        "\n\nPreprocessing details:\n"\
        f"period \t\t= {period}\n"\
        f"driving_model \t= {dri_mod_str} (ID: {dri_mod_id})\n"\
        f"nested_model \t= {nest_mod_str} (ID: {nest_mod_id})\n"\
        f"datetime \t= {dt_str}\n"\
        f"save dir.\t= {SAVE_DIR}\n"\
    )

    # load datasets
    fpath_inner = os.path.join(
        DATA_DIR,
        f"{grid_id}_regrid",
        f"{nest_mod_str}.{dri_mod_str}.uvw_{dt_str}_{grid_id}.nc"
    )
    fpath_outer = os.path.join(
        DATA_DIR,
        "time_interpolated",
        f"glm.{dri_mod_str}.uvw_t_interp_{dt_str}.nc"
    )
    ds_inner = xr.open_dataset(
        fpath_inner,
        mask_and_scale=True,
        drop_variables="leadtime"
    ).sel(pressure=plevs,method="nearest")
    ds_outer = xr.open_dataset(
        fpath_outer,
        mask_and_scale=True,
        drop_variables="leadtime"
    ).sel(pressure=plevs,method="nearest")
    
    print("\n\nInner:\n",ds_inner)
    print("\n\nOuter:\n",ds_outer)

    # perform embedding
    ds_embedded = embed_inner_grid_in_global(
        ds_outer, ds_inner,
        inner_type="channel",
        boundary_method="interp"
    )

    # ensure correct dimension ordering
    ds_embedded = ds_embedded.transpose("time","pressure","latitude","longitude")

    #import matplotlib.pyplot as plt
    #plt.figure(figsize=(10,8))
    #ds_embedded.u.isel(time=0).plot()
    #plt.figure(figsize=(10,8))
    #ds_embedded.v.isel(time=0).plot()
    #plt.figure(figsize=(10,8))
    #ds_embedded.w.isel(time=0).plot(vmin=-0.1,vmax=0.1,cmap="RdBu_r")
    #plt.show()
    #sys.exit(1)
    
    # save
    fpath = os.path.join(SAVE_DIR, f"{nest_mod_str}.{dri_mod_str}.uvw_embedded_{dt_str}.nc")
    print(f"\n\n\nSaving embedded dataset to {fpath}")
    ds_embedded.to_netcdf(fpath)

    print("\n\n\nEND.")
