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
    load_kscale_native, interp_time_driving_model

if __name__ == "__main__":
    _period=sys.argv[1]
    _dri_mod_id = sys.argv[2]
    startyear = int(sys.argv[3])
    startmonth = int(sys.argv[4])
    startday = int(sys.argv[5])
    endyear = int(sys.argv[6])
    endmonth = int(sys.argv[7])
    endday = int(sys.argv[8])

    DATA_DIR_ROOT = "/gws/nopw/j04/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/"
    
    # parse period, driving model
    period = parse_period_id(_period)
    dri_mod_id, dri_mod_str = parse_dri_mod_id(period,_dri_mod_id)

    # set up output directory
    SAVE_DIR = os.path.join(DATA_DIR_ROOT, period, "time_interpolated")
    Path(SAVE_DIR).mkdir(parents=True,exist_ok=True)

    # set up time range
    start = dt.datetime(startyear,startmonth,startday)
    end = dt.datetime(endyear,endmonth,endday)
    diff = end - start
    ndays = diff.days + 1
    dates = [start + dt.timedelta(iday) for iday in range(ndays)]
    
    print(
        "\n\nPreprocessing details:\n\n"\
        f"period \t\t= {period}\n"\
        f"driving_model \t= {dri_mod_str} (ID: {dri_mod_id})\n"\
        f"start \t\t= {start.year:04d}{start.month:02d}{start.day:02d}\n"\
        f"end \t\t= {end.year:04d}{end.month:02d}{end.day:02d}\n"\
        f"ndays \t\t= {ndays}\n"\
        f"save dir.\t= {SAVE_DIR}"
    )

    ds = xr.open_mfdataset(
        [
            os.path.join(
                DATA_DIR_ROOT,
                period,
                f"glm.{dri_mod_str}.uvw_"\
                f"{date.year:04d}{date.month:02d}{date.day:02d}T{hour:02d}.nc"
            ) for hour in [0,12] for date in dates
        ],
        mask_and_scale = True,
        drop_variables = ["forecast_reference_time", "forecast_period"]
    )

    print(ds)
    print(ds.time)

    #ds_interp = interp_time_driving_model(
    #    ds,
    #    time_offset=np.timedelta64(1, "h"),
    #    lag=True,
    #    method="linear",
    #    retain_original_times=True,
    #    keep_initial_time=True
    #)
    #print(ds_interp)
    
    for id, date in enumerate(dates):
        print(f"\n\n\nDate = {date.year:04d}-{date.month:02d}-{date.day:02d}")
        for hour in [0,12]:
            _ds = ds.sel(
                time=slice(
                    date + dt.timedelta(seconds=hour*3600),
                    date + dt.timedelta(seconds=hour*3600)\
                    + dt.timedelta(seconds=12*3600)
                )
            )
            
            if id == 0 and hour == 0:
                keep_initial_time = True
            else:
                keep_initial_time = False
            #endif

            # perform time interpolation
            ds_interp = interp_time_driving_model(
                _ds,
                time_offset=np.timedelta64(1, "h"),
                lag=True,
                method="linear",
                retain_original_times=True,
                keep_initial_time=keep_initial_time
            )

            # save
            dt_str = f"{date.year:04d}{date.month:02d}{date.day:02d}T{hour:02d}"
            fpath = os.path.join(SAVE_DIR, f"glm.{dri_mod_str}.uvw_t_interp_{dt_str}.nc")
            print(f"\nSaving time-interpolated u,v,w for T{hour:02d} to {fpath}")
            ds_interp.to_netcdf(fpath)
        #endfor
    #endfor

    print("\n\n\nEND.")
