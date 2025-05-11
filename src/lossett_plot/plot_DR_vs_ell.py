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

def plot_histogram_vs_coord_contour(
        var, coord, bins, var_name=None, var_units=None, coord_name=None,coord_units=None,
        title=None, hist_norm=None, x_logscale=False, y_logscale=False, invert_yaxis=False
):
    # setup
    if var_name is None:
        var_name = var.name
    if var_units is None:
        var_units = var.attrs["units"]
    if coord_name is None:
        coord_name = coord.name
    if coord_units is None:
        coord_units = coord.attrs["units"]

    var = var.sel({coord.name:coord},method="nearest")
    dims=list(var.dims)
    mean_dims = dims
    mean_dims.remove(coord_name)
        
    hist = []
    for coord_value in coord:
        _hist, _bins = np.histogram(
            var.sel({coord_name:coord_value}).data,
            bins=bins
        )
        hist.append(_hist)
        
    hist = np.array(hist)
    bins_reduced = (bins[1:] + bins[:-1]) / 2

    fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(12,8))
    pc = plt.pcolormesh(bins_reduced, coord, hist, norm=hist_norm)
    plt.plot(var.mean(dim=mean_dims), coord, color="C1")
    plt.xlabel(f"{var_name} [{var_units}]")
    plt.ylabel(f"{coord_name} [{coord_units}]")
    plt.colorbar(pc, label="frequency")
    if title is not None:
        plt.title(title)
    if x_logscale:
        plt.xscale("log")
    if y_logscale:
        plt.yscale("log")
    if invert_yaxis:
        plt.gca().invert_yaxis()
    
    return fig; # should really return Axes object?

def plot_histogram_vs_coord_line(
        var, coord, bins, var_name=None, var_units=None, coord_name=None, coord_units=None,
        title=None, colours=None # add normalization option
):
    # setup
    if var_name is None:
        var_name = var.name
    if var_units is None:
        var_units = var.attrs["units"]
    if coord_name is None:
        coord_name = coord.name
    if coord_units is None:
        coord_units = coord.attrs["units"]

    var = var.sel({coord.name:coord},method="nearest")
    dims=list(var.dims)
    mean_dims = dims
    mean_dims.remove(coord_name)
        
    hist = []
    for coord_value in coord:
        _hist, _bins = np.histogram(
            var.sel({coord_name:coord_value}).data,
            bins=bins
        )
        hist.append(_hist)

    bins_reduced = (bins[1:] + bins[:-1]) / 2
    hist=xr.DataArray(
        data=np.array(hist),
        dims=[coord_name,"bins"],
        coords={coord_name:coord.values,"bins":bins_reduced}
    )
    tot = hist.sum(dim="bins")

    hist = hist/tot

    fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(12,8))
    for ic, coord_value in enumerate(coord):
        plt.plot(
            hist.bins, hist.sel({coord_name:coord_value},method="nearest"), color=colours[ic],
            label=f"{coord_value:.4g} {coord_units}", marker=".",
            linestyle=""
        )
    plt.xlabel(f"{var_name} [{var_units}]")
    plt.ylabel(f"frequency density") # chould really divide by total number to give PDF
    plt.legend(loc="best")
    plt.yscale("log")
    plt.grid()
    if title is not None:
        plt.title(title)
    
    return fig; # should really return Axes object?

if __name__ == "__main__":
    DATA_DIR = "/gws/nopw/j04/kscale/USERS/dship/LoSSETT_out/"
    PLOT_DIR = "/home/users/dship/python/upscale/plots/"
    
    # simulation specification
    simid = sys.argv[1]

    # DR indicator specification
    n_scales = 32
    tsteps = 8

    # date
    startdate = dt.datetime(2016,8,1)

    # load time-mean DR indicator (compute & save if it doesn't exist)
    if simid == "CTC5GAL" or simid == "CTC5RAL":
        fpath_mean = os.path.join(
            DATA_DIR,
            f"DR_test_{simid}_Nl_{n_scales}_DyS_time-mean.nc"
        )
    elif simid == "ERA5":
        fpath_mean = os.path.join(
            DATA_DIR,
            "ERA5",
            f"inter_scale_energy_transfer_kinetic_ERA5_0p5deg_Nl_31_DyS_time-mean.nc"
        )
    else:
        print(f"\nERROR: loading for simid = {simid} not implemented yet!")
        sys.exit(1)

    if not os.path.exists(fpath_mean):
        print(f"\n\n\nComputing time-mean DR indicator for {simid}")
        # load DR indicator
        if simid == "CTC5GAL" or simid == "CTC5RAL":
            DR = xr.open_mfdataset(
                [
                    os.path.join(
                        DATA_DIR,
                        f"DR_test_{simid}_Nl_{n_scales}_"\
                        f"{date.year:04d}-{date.month:02d}-{date.day:02d}_t0-{tsteps-1}.nc"
                    ) for date in [startdate + dt.timedelta(i) for i in range(40)]
                ]
            )["DR_indicator"]
        elif simid == "ERA5":
            DR = xr.open_mfdataset(
                [
                    os.path.join(
                        DATA_DIR,
                        "ERA5",
                        "inter_scale_energy_transfer_kinetic_ERA5_0p5deg_Nl_31_"\
                        f"{date.year:04d}-{date.month:02d}-{date.day:02d}.nc"
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
    print(f"\n\n\nTime-mean DR indicator for {simid}:\n",DR_tmean)

    # plotting switches & strings
    show = False
    cmap = mpl.colormaps["plasma"]
    colours_p = cmap(np.linspace(0,1,len(DR_tmean.pressure)))

    DR_tmean = DR_tmean.assign_coords({"length_scale":DR_tmean.length_scale.values/1000})
    DR_tmean.length_scale.attrs["units"] = "km"
    ells = np.array([110,220,440,880])
    ells = DR_tmean.sel(length_scale=ells).length_scale
    coord_name="pressure"
    coord = DR_tmean[coord_name]
    bins=np.linspace(-1e-3,1e-3,101)
    hist_norm = mpl.colors.LogNorm()

    for ell in ells:
        var = DR_tmean.sel(latitude=slice(-15,15)).sel(length_scale=ell,method="nearest")
        fig = plot_histogram_vs_coord_line(
            var, var.pressure, bins,
            var_name=r"$\mathcal{D}_\ell$", var_units=r"m$^\text{2}$ s$^\text{-3}$", coord_units="hPa",
            title=f"DYAMOND Summer time-mean {simid} (mean 15S-15N), "+r"$\ell = "+f"{ell:.4g}$ km",
            colours=colours_p
        )
        plt.savefig(os.path.join(PLOT_DIR,f"{simid}_DS_histogram_Dl_vs_pressure_ell{ell:.4g}km_line.png"))
        plt.show()
        
        fig = plot_histogram_vs_coord_contour(
            var, var.pressure, bins,
            var_name=r"$\mathcal{D}_\ell$", var_units=r"m$^\text{2}$ s$^\text{-3}$", coord_units="hPa",
            title=f"DYAMOND Summer time-mean {simid} (mean 15S-15N), "+r"$\ell = "+f"{ell:.4g}$ km",
            hist_norm=hist_norm, x_logscale=False, y_logscale=True, invert_yaxis=True
        )
        plt.savefig(os.path.join(PLOT_DIR,f"{simid}_DS_histogram_Dl_vs_pressure_ell{ell:.4g}km_contour.png"))
        plt.show()

    

    for p in DR_tmean.pressure:
        var = DR_tmean.sel(latitude=slice(-15,15)).sel(pressure=p,method="nearest")
        colours_l = cmap(np.linspace(0,1,len(ells)))
        fig = plot_histogram_vs_coord_line(
            var, ells, bins,
            var_name=r"$\mathcal{D}_\ell$", var_units=r"m$^\text{2}$ s$^\text{-3}$", coord_units="km",
            title=f"DYAMOND Summer time-mean {simid} (mean 15S-15N), $p = {p:.4g}$ hPa", colours=colours_l
        )
        plt.savefig(os.path.join(PLOT_DIR,f"{simid}_DS_histogram_Dl_vs_length_scale_p{p:.4g}hPa_line.png"))
        plt.show()

    
    sys.exit(1)

    # subset 15S-15N & take horizontal mean
    DR_tmean_hmean = DR_tmean.sel(latitude=slice(-15,15)).mean(dim=["latitude","longitude"]).compute()

    # plot time-mean Dl vs l
    print("\n\n\nPlotting time-mean D_\ell vs \ell")

    fig, axes = plt.subplots(1,1,figsize=(10,8))
    for il, lev in enumerate(DR_tmean_hmean.pressure):
        plt.plot(
            2*DR_tmean_hmean.length_scale,
            DR_tmean_hmean.sel(pressure=lev),
            label = f"{lev.values}",
            color = colours[il],
        )
    plt.xlabel(r"$2\ell$ [km]")
    plt.ylabel(r"$D_\ell(\mathbf{u})$ [m$^2$ s$^{-3}$]")
    plt.ylim([-1.5e-4,5.5e-4])
    plt.legend(loc="best")
    plt.xscale("log")
    plt.grid()
    plt.title(f"DYAMOND Summer time-mean {simid} (mean 15S-15N)")
    plt.savefig(os.path.join(PLOT_DIR,f"Dl_vs_ell_DS_{simid}_test.png"))
    if show:
        plt.show()
    plt.close()


    vmag = 1e-4
    

    # set up normalization & contour levels
    log_lvls = np.logspace(-5,-3.5,9,base=10)
    cont_lvls = np.concatenate((-log_lvls[::-1],np.array([0]),log_lvls))
    norm = mpl.colors.SymLogNorm(linthresh=1e-5, linscale=0.1, vmin=-5e-4, vmax=5e-4, base=10)
    log_cticks=[-5e-4,-1e-4,-1e-5,0,1e-5,1e-4,5e-4]
    log_cticklabels=[
        r"$-5 \times 10^{-4}$", r"$-1 \times 10^{-4}$", r"$-1 \times 10^{-5}$", "$0$",
        r"$1 \times 10^{-5}$", r"$1 \times 10^{-4}$", r"$5 \times 10^{-4}$"]
    print("\n\n\nPlotting latitude-pressure and longitude-pressure cross sections")
    for ell in ells:
        print(f"\n\ell = {ell:.4g} km")
        DR_tmean_latmean = DR_tmean.sel(latitude=slice(-15,15)).mean(dim="latitude")
        fig, axes = plt.subplots(1,1,figsize=(15,8))
        pc = plt.pcolormesh(
            DR_tmean_latmean.longitude,
            DR_tmean_latmean.pressure,
            DR_tmean_latmean.sel(length_scale=ell, method="nearest"),
            cmap="RdBu_r",
            norm=norm,
            shading="auto"
        )
        cb = plt.colorbar(
            pc, extend="both", ticks=log_cticks,
            label=r"$D_\ell(\mathbf{u})$ [m$^2$ s$^{-3}$]",
            orientation="vertical"
        )
        cb.ax.set_yticklabels(log_cticklabels)
        plt.xlabel(r"longitude [deg. E]")
        plt.ylabel(r"$p$ [hPa]")
        plt.gca().invert_yaxis()
        plt.grid()
        plt.title(f"DYAMOND Summer time-mean {simid} (mean 15S-15N), $\ell = {ell:.4g}$ km")
        plt.savefig(os.path.join(PLOT_DIR,f"Dl_{simid}_lon_pressure_xsection_15SN_latmean_time-mean_ell{ell:.4g}_test.png"))
        if show:
            plt.show()
        plt.close()

        DR_tmean_lonmean = DR_tmean.sel(latitude=slice(-24,24)).mean(dim="longitude")
        fig, axes = plt.subplots(1,1,figsize=(15,8))
        pc = plt.pcolormesh(
            DR_tmean_lonmean.latitude,
            DR_tmean_lonmean.pressure,
            DR_tmean_lonmean.sel(length_scale=ell, method="nearest").T,
            cmap="RdBu_r",
            norm=norm,
            shading="auto"
        )
        plt.colorbar(pc, extend="both", label=r"$D_\ell(\mathbf{u})$ [m$^2$ s$^{-3}$]")
        plt.xlabel(r"latitude [deg. N]")
        plt.ylabel(r"$p$ [hPa]")
        plt.gca().invert_yaxis()
        plt.grid()
        plt.title(f"DYAMOND Summer time-mean {simid} (mean 0-360E), $\ell = {ell:.4g}$ km")
        plt.savefig(os.path.join(PLOT_DIR,f"Dl_{simid}_lat_pressure_xsection_24SN_lonmean_time-mean_ell{ell:.4g}_test.png"))
        if show:
            plt.show()
        plt.close()

    # write new script to plot time-mean DR at each level, and time mean DR cross section, at various length scales
    
    
    
