#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import datetime as dt

from plot_DR_indicator import plot_DR_indicator_lon_lat

if __name__ == "__main__":
    DATA_DIR = "/work/scratch-pw2/dship/upscale/LoSSETT_out/"
    PLOT_DIR = "/home/users/dship/python/upscale/plots/"
    startdate = "1980-09-02"
    start = dt.datetime.strptime(startdate, "%Y-%m-%d")

    # plotting switches
    show = True

    # simulation specification
    simid = "u-dc009"
    tsteps_per_day = 4

    # calculation specification
    n_scales = 70
    tsteps = 1

    # load data
    DR_indicator = xr.open_dataset(
        os.path.join(DATA_DIR,f"DR_test_u-dc009_Nl_{n_scales}_{startdate}_t0-{tsteps-1}.nc")
    )["DR_indicator"]

    print(DR_indicator.length_scale)

    # define length scales for plotting
    chosen_scales = [30000,100000,200000,500000,750000,1000000]
    scale_subset_str = f"_{len(chosen_scales)}_scales"

    # plotting options
    nrows = 2
    ncols = 3

    # create lon-lat maps save directory
    SAVE_DIR = os.path.join(PLOT_DIR, simid, "lon_lat_maps")
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    p_levs = DR_indicator.pressure

    for p in p_levs:
        print(f"\n\n\nPressure = {p} hPa")
        
        _SAVE_DIR = os.path.join(SAVE_DIR, f"{int(p):04d}hPa")
        Path(_SAVE_DIR).mkdir(parents=True, exist_ok=True)

        for tstep in range(0,len(DR_indicator.time)):
            time_str = f"time_t{tstep}"
            mean_str = f"tstep {tstep}"

            print(f"Plotting {mean_str}")
                
            # define output filename & plot title
            fname = os.path.join(
                _SAVE_DIR,
                f"DR_indicator_{simid}_Nl_{n_scales}{scale_subset_str}_p{int(p):04d}_{startdate}_{time_str}.png"
            )
            title = f"DR indicator for {simid} {startdate} {mean_str} at {int(p)} hPa"

            # plot DR indicator
            plot_DR_indicator_lon_lat(
                DR_indicator.sel(pressure=p,method="nearest").isel(time=tstep).T,
                chosen_scales, title, fname, nrows, ncols,
                x_coord="longitude", y_coord="latitude", show=show
            )
