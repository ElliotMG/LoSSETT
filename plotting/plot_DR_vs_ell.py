#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy as cpy
import datetime as dt

if __name__ == "__main__":
    DATA_DIR = "/gws/nopw/j04/kscale/USERS/dship/LoSSETT_out/"
    PLOT_DIR = "/home/users/dship/python/upscale/plots/"
    
    # simulation specification
    simid = sys.argv[1] #"CTC5RAL"

    # DR indicator specification
    n_scales = 32
    tsteps = 8
    startdate = "2016-08-10"

    startdate = dt.datetime(2016,8,1)

    # load time-mean DR indicator (compute & save if it doesn't exist)
    fpath_mean = os.path.join(
        DATA_DIR,
        f"DR_test_{simid}_Nl_{n_scales}_time-mean.nc"
    )
    if not os.path.exists(fpath_mean):
        # load DR indicator
        DR = xr.open_mfdataset(
            [
                os.path.join(
                    DATA_DIR,
                    f"DR_test_{simid}_Nl_{n_scales}_"\
                    f"{date.year:04d}-{date.month:02d}-{date.day:02d}_t0-{tsteps-1}.nc"
                ) for date in [startdate + dt.timedelta(i) for i in range(40)]
            ]
        )["DR_indicator"]

        # re-chunk
        DR = DR.chunk(
            chunks={"pressure":1,"length_scale":8}
        )
        print(DR)

        # take mean
        DR_mean = DR.mean(dim="time").compute()
        print(DR_mean)

        # save mean
        DR_mean.to_netcdf(fpath_mean)

        DR_mean.close()
        #endif

    DR_tmean = xr.open_dataset(fpath_mean)["DR_indicator"]

    print(DR_tmean)

    # subset 15S-15N & take horizontal mean
    DR_tmean_hmean = DR_tmean.sel(latitude=slice(-15,15)).mean(dim=["latitude","longitude"]).compute()

    cmap = mpl.colormaps["inferno"]
    colours = cmap(np.linspace(0,1,len(DR_tmean_hmean.pressure)))

    fig, axes = plt.subplots(1,1,figsize=(10,8))
    for il, lev in enumerate(DR_tmean_hmean.pressure):
        plt.plot(
            2*DR_tmean_hmean.length_scale/1000.0,
            DR_tmean_hmean.sel(pressure=lev),
            label = f"{lev.values}",
            color = colours[il],
        )
    plt.xlabel(r"$2\ell$ [km]")
    plt.ylabel(r"$D_\ell(\mathbf{u})$ [m$^2$ s$^{-3}$]")
    plt.ylim([-1.5e-4,5.5e-4])
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(os.path.join(PLOT_DIR,f"Dl_vs_ell_DS_{simid}_test.png"))
    #plt.show()
    plt.close()


    ells = np.array([110,220,440,880])
    ells = 1000.0*ells
    vmag = 1e-4
    for ell in ells:
        DR_tmean_latmean = DR_tmean.sel(latitude=slice(-15,15)).mean(dim="latitude")
        fig, axes = plt.subplots(1,1,figsize=(15,8))
        pc = plt.pcolormesh(
            DR_tmean_latmean.longitude,
            DR_tmean_latmean.pressure,
            DR_tmean_latmean.sel(length_scale=ell, method="nearest"),
            cmap="RdBu_r",
            vmin=-vmag,
            vmax=vmag
        )
        plt.colorbar(pc, extend="both", label=r"$D_\ell(\mathbf{u})$ [m$^2$ s$^{-3}$]")
        plt.xlabel(r"longitude [deg. E]")
        plt.ylabel(r"$p$ [hPa]")
        plt.gca().invert_yaxis()
        plt.grid()
        plt.savefig(os.path.join(PLOT_DIR,f"Dl_{simid}_lon_pressure_xsection_15SN_latmean_time-mean_ell{ell/1000:.4g}_test.png"))
        #plt.show()
        plt.close()

        DR_tmean_lonmean = DR_tmean.sel(latitude=slice(-24,24)).mean(dim="longitude")
        fig, axes = plt.subplots(1,1,figsize=(15,8))
        pc = plt.pcolormesh(
            DR_tmean_lonmean.latitude,
            DR_tmean_lonmean.pressure,
            DR_tmean_lonmean.sel(length_scale=ell, method="nearest").T,
            cmap="RdBu_r",
            vmin=-vmag,
            vmax=vmag
        )
        plt.colorbar(pc, extend="both", label=r"$D_\ell(\mathbf{u})$ [m$^2$ s$^{-3}$]")
        plt.xlabel(r"latitude [deg. N]")
        plt.ylabel(r"$p$ [hPa]")
        plt.gca().invert_yaxis()
        plt.grid()
        plt.savefig(os.path.join(PLOT_DIR,f"Dl_{simid}_lat_pressure_xsection_24SN_lonmean_time-mean_ell{ell/1000:.4g}_test.png"))
        plt.show()
        plt.close()

    # write new script to plot time-mean DR at each level, and time mean DR cross section, at various length scales
    
    
    
