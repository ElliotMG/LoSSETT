#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import datetime
import pandas as pd
import cartopy as cpy

from lossett_control.preprocessing import preprocess_era_interim
from lossett.calc.calc_inter_scale_transfers import calc_inter_scale_energy_transfer_kinetic

if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    day = int(sys.argv[3])
    tsteps_per_day = 4
    sampling = f"{int(24/tsteps_per_day)}h"
    lon_bound_field = "periodic"
    lat_bound_field = np.nan

    max_r_deg = float(sys.argv[4])
    tsteps = 6
    tchunks = 6
    pchunks = 11
    prec = 1e-10

    OUT_DIR_ROOT = "/work/scratch-pw4/emg/LoSSETT/output/"
    OUT_DIR = os.path.join(OUT_DIR_ROOT, "ERA_Interim")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

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

    # open the data
    var_names, yearmonths, data_dir = preprocess_era_interim.setup_vars_yearmonth(
        year, month, sampling="6h", return_dates=False, moist=False
    )
    ds_u_3D = preprocess_era_interim.load_era_interim(var_names, yearmonths, data_dir)
    print(ds_u_3D)