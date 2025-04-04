#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy as cpy

def plot_DR_indicator_lon_lat(
        DR_indicator, chosen_scales, title, fname, nrows, ncols,
        x_coord="longitude", y_coord="latitude", show=False
):
    x = DR_indicator[x_coord]
    y = DR_indicator[y_coord]
    mag=5e-3
    fig,_axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(20,10),#(ncols*5,nrows*2), # should set from expected aspect ratio of subplots
        subplot_kw={"projection":cpy.crs.PlateCarree()}
    )
    axes = _axes.T.flatten()
    for il, ell in enumerate(chosen_scales):
        ax = axes[il]
        DR_plot = DR_indicator.sel(
            length_scale=ell, method="nearest"
        )
        mean = DR_plot.mean([x_coord,y_coord])
        if mean >= 0.0:
            colour = "C3"
        elif mean < 0.0:
            colour = "C0"
        
        pc = ax.pcolormesh(
            x, y, DR_plot,
            cmap="RdBu_r", vmin=-mag, vmax=mag
        )
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        if il >= nrows:
            gl.left_labels = False
        if (il+1)%nrows != 0:
            gl.bottom_labels = False
        ax.set_title(f"$2\ell = {2*ell/1000:.4g}$ km")
        ax.text(
            0.8, 1.05, f"{mean:.3g}",
            color=colour, transform=ax.transAxes
        )
    # colourbar
    xmin = axes[0].get_position().xmin
    xmax = axes[-1].get_position().xmax
    axes_width = xmax-xmin
    cbar_width = 0.5*axes_width
    plt.subplots_adjust(bottom=0.15)
    cax = fig.add_axes([xmin+(axes_width-cbar_width)/2.0,0.08,cbar_width,0.025])
    plt.colorbar(pc,cax=cax,orientation="horizontal",extend="both",label=r"$D_\ell(\mathbf{u})$ [m$^2$ s$^{-3}$]")
    plt.suptitle(title)
    plt.savefig(fname)
    if show:
        plt.show()
    return 0;

def plot_DR_indicator_lon_pressure(
        DR_indicator, chosen_scales, title, fname, nrows, ncols,
        x_coord="longitude", z_coord="pressure", show=False,
        z_coord_pressure=True
):
    x = DR_indicator[x_coord]
    z = DR_indicator[z_coord]
    mag=2.5e-4
    fig,_axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(20,10)
    )
    axes = _axes.T.flatten()
    for il, ell in enumerate(chosen_scales):
        ax = axes[il]
        DR_plot = DR_indicator.sel(
            length_scale=ell, method="nearest"
        )        
        pc = ax.pcolormesh(
            x, z, DR_plot,
            cmap="RdBu_r", vmin=-mag, vmax=mag
        )
        ax.grid()
        if il >= nrows:
            ax.set_yticklabels([])
        if (il+1)%nrows != 0:
            ax.set_xticklabels([])
        ax.set_title(f"$2\ell = {2*ell/1000:.4g}$ km")
        if z_coord_pressure:
            ax.invert_yaxis()
            ax.set_yscale("log")
        
    # colourbar
    xmin = axes[0].get_position().xmin
    xmax = axes[-1].get_position().xmax
    axes_width = xmax-xmin
    cbar_width = 0.5*axes_width
    plt.subplots_adjust(bottom=0.15)
    cax = fig.add_axes([xmin+(axes_width-cbar_width)/2.0,0.08,cbar_width,0.025])
    plt.colorbar(pc,cax=cax,orientation="horizontal",extend="both",label=r"$D_\ell(\mathbf{u})$ [m$^2$ s$^{-3}$]")
    plt.suptitle(title)
    plt.savefig(fname)
    if show:
        plt.show()
    return 0;

if __name__ == "__main__":
    DATA_DIR = "/gws/nopw/j04/kscale/USERS/dship/LoSSETT_out/"
    PLOT_DIR = "/home/users/dship/python/upscale/plots/"

    # plotting switches
    plot_lon_pressure_xsections = False#True
    plot_lon_lat_maps = True
    subset_scales=True
    show=False
    
    # simulation specification
    simid = sys.argv[1] #"CTC5RAL"

    # DR indicator specification
    n_scales = 32
    tsteps = 8
    startdate = sys.argv[2] #"2016-08-01"

    # load DR indicator
    DR_indicator = xr.open_dataset(
        os.path.join(DATA_DIR, f"DR_test_{simid}_Nl_{n_scales}_{startdate}_t0-{tsteps-1}.nc")
    )["DR_indicator"]
    
    # get start datetime
    start = pd.Timestamp(DR_indicator.time[0].values).to_pydatetime()

    # define length scales for plotting
    chosen_scales = DR_indicator.length_scale.values[:20]
    print(chosen_scales)
    scale_subset_str = ""

    # plotting options
    nrows = 5
    ncols = 4
    time_mean = False
    if subset_scales:
        scale_subset_indices = [1,3,7,15]
        nrows=2
        ncols=2
        chosen_scales = chosen_scales[scale_subset_indices]
        scale_subset_str += f"_{len(scale_subset_indices)}_scales"
        print(chosen_scales)

    if plot_lon_pressure_xsections:
        # create lon-pressure cross-sections save directory
        SAVE_DIR = os.path.join(PLOT_DIR, simid, "lon_pressure_xsections")
        Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

        # subset IO + MC
        lon_min=40
        lon_max=180
        lat_min=-5
        lat_max=5
        subset_str=f"lon_{lon_min}-{lon_max}E_lat_{lat_min}-{lat_max}N"
        subset_name="IO_plus_MC_subset"

        _DR_indicator = DR_indicator.sel(
            longitude=slice(lon_min,lon_max),latitude=slice(lat_min,lat_max)
        ).mean(dim="latitude")

        _SAVE_DIR = os.path.join(SAVE_DIR, subset_name)
        Path(_SAVE_DIR).mkdir(parents=True, exist_ok=True)

        if time_mean:
            _DR_indicator = _DR_indicator.mean("time")
            time_str = f"time_mean_t0-{tsteps-1}"
            mean_str = f"time mean first {tsteps}*3hours"
                
            # define output filename & plot title
            fname = os.path.join(
                _SAVE_DIR,
                f"DR_indicator_{simid}_Nl_{n_scales}{scale_subset_str}_{subset_str}_{startdate}_{time_str}.png"
            )
            title = f"{simid} DR indicator longitude-pressure cross-section averaged {lat_min}-{lat_max}N for {startdate} {mean_str}"
        
            plot_DR_indicator_lon_pressure(
                _DR_indicator,
                chosen_scales, title, fname, nrows, ncols,
                x_coord="longitude", z_coord="pressure", show=show,
                z_coord_pressure=True
            )
        else:
            for tstep in range(0,len(DR_indicator.time)):
                time_str = f"time_t{tstep}"
                mean_str = f"tstep {tstep}"

                print(f"Plotting {mean_str}")
                
                # define output filename & plot title
                fname = os.path.join(
                    _SAVE_DIR,
                    f"DR_indicator_{simid}_Nl_{n_scales}{scale_subset_str}_{subset_str}_{startdate}_{time_str}.png"
                )
                title = f"{simid} DR indicator longitude-pressure cross-section averaged {lat_min}-{lat_max}N for {startdate} {mean_str}"
        
                plot_DR_indicator_lon_pressure(
                    _DR_indicator.isel(time=tstep),
                    chosen_scales, title, fname, nrows, ncols,
                    x_coord="longitude", z_coord="pressure", show=show,
                    z_coord_pressure=True
                )
            #endfor
        #endif
    #endif

    if plot_lon_lat_maps:
        # create lon-lat maps save directory
        SAVE_DIR = os.path.join(PLOT_DIR, simid, "lon_lat_maps")
        Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
        
        # subset Maritime Continent (MC)
        #_DR_indicator = DR_indicator.sel(longitude=slice(90,160),latitude=slice(-15,15))
        #subset_str = "MC_subset"
        
        # subset Indian Ocean (IO)
        _DR_indicator = DR_indicator.sel(longitude=slice(40,110),latitude=slice(-15,15))
        subset_str = "IO_subset"
        
        # subsetting/mean operations
        p_levs = [200,850]#[100,200,300,400,500,600,700,850]

        print("\n\n\nPlotting lon-lat maps")
    
        for p in p_levs:
            print(f"\n\n\nPressure = {p} hPa")
            
            _SAVE_DIR = os.path.join(SAVE_DIR, f"{p:04d}hPa", subset_str)
            Path(_SAVE_DIR).mkdir(parents=True, exist_ok=True)
            
            if time_mean:
                _DR_indicator = _DR_indicator.sel(pressure=p,method="nearest").mean("time")
                time_str = f"time_mean_t0-{tsteps-1}"
                mean_str = f"time mean first {tsteps}*3hours"
                
                # define output filename & plot title
                fname = os.path.join(
                    _SAVE_DIR,
                    f"DR_indicator_{simid}_Nl_{n_scales}{scale_subset_str}_{startdate}_{time_str}_{subset_str}.png"
                )
                title = f"DR indicator for {simid} {startdate} {mean_str} at {p} hPa"
                
                # plot DR indicator
                plot_DR_indicator_lon_lat(
                    _DR_indicator, chosen_scales, title, fname, nrows, ncols,
                    x_coord="longitude", y_coord="latitude", show=show
                )
            else:
                for tstep in range(0,len(DR_indicator.time)):
                    time_str = f"time_t{tstep}"
                    mean_str = f"tstep {tstep}"

                    print(f"Plotting {mean_str}")
                
                    # define output filename & plot title
                    fname = os.path.join(
                        _SAVE_DIR,
                        f"DR_indicator_{simid}_Nl_{n_scales}{scale_subset_str}_p{p:04d}_{startdate}_{time_str}_{subset_str}.png"
                    )
                    title = f"DR indicator for {simid} {startdate} {mean_str} at {p} hPa"

                    # plot DR indicator
                    plot_DR_indicator_lon_lat(
                        _DR_indicator.sel(pressure=p,method="nearest").isel(time=tstep),
                        chosen_scales, title, fname, nrows, ncols,
                        x_coord="longitude", y_coord="latitude", show=show
                    )
                #endfor
            #endif
        #endfor
    #endif

    print("\n\n\nEND.")
