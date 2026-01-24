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
    load_kscale_native

if __name__ == "__main__":
    _period=sys.argv[1]
    _dri_mod_id = sys.argv[2]
    _nest_mod_id = sys.argv[3]
    year = int(sys.argv[4])
    month = int(sys.argv[5])
    day = int(sys.argv[6])
    hour = int(sys.argv[7])
    save_nc = True
    force = True
    plevs = [100,150,200,250,300,400,500,600,700,850,925,1000]
    #plevs = [200]
    
    # parse period, driving model, nested model
    period = parse_period_id(_period)
    dri_mod_id, dri_mod_str = parse_dri_mod_id(period,_dri_mod_id)
    nest_mod_id, nest_mod_str = parse_nest_mod_id(period,dri_mod_id,_nest_mod_id)

    datetime = dt.datetime(year,month,day,hour)
    dt_str = f"{datetime.year:04d}{datetime.month:02d}{datetime.day:02d}T{(datetime.hour//12)*12:02d}"
    
    print("\n\n\nPreprocessing details:")
    print(
        "\n\nPreprocessing details:\n"\
        f"period \t\t= {period}\n"\
        f"driving_model \t= {dri_mod_str} (ID: {dri_mod_id})\n"\
        f"nested_model \t= {nest_mod_str} (ID: {nest_mod_id})\n"\
        f"datetime \t= {dt_str}\n"
    )

    ds = load_kscale_native(
        period,
        datetime,
        driving_model=dri_mod_id,
        nested_model=nest_mod_id,
        save_nc=save_nc,
        force=force
    ).sel(pressure=plevs,method="nearest")

    # ensure correct dimension ordering
    ds = ds.transpose("time","pressure","latitude","longitude")

    print("\n\n\nEND.")
