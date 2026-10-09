#!/usr/bin/env python3
import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy as cpy
import os

from lossett.filtering.get_integration_kernels import get_integration_kernels
from lossett.filtering.integration import integrate_over_scales

KE_TRANSFER_NORMALIZATION = 1./4.

fpaths = {
    "2p5deg": "/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/"\
    "glm.n1280_GAL9_DS_DATET00_inter_scale_transfer_of_kinetic_energy_"\
    "p0200hPa_2p5deg.zarr",
    "1p0deg": "/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/"\
    "glm.n1280_GAL9_DS_DATET00_inter_scale_transfer_of_kinetic_energy_"\
    "p0200hPa_1p0deg.zarr",
    "n320": "/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/"\
    f"glm.n1280_GAL9_DS_DATET00_inter_scale_transfer_of_kinetic_energy_"\
    "p0200hPa_n320.zarr",
    "n320_maxR_5000": "/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/"\
    f"glm.n1280_GAL9_DS_DATET00_inter_scale_transfer_of_kinetic_energy_"\
    "p0200hPa_n320_maxR_05000.zarr",
    "n640_maxR_5000": "/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/"\
    f"glm.n1280_GAL9_DS_DATET00_inter_scale_transfer_of_kinetic_energy_"\
    "p0200hPa_n640_maxR_05000.zarr",
    "n640_maxR_5000": "/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/"\
    f"glm.n1280_GAL9_DS_DATET00_inter_scale_transfer_of_kinetic_energy_"\
    "p0200hPa_n640_maxR_02000.zarr",
    "n1280_maxR_2000": "/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/"\
    f"glm.n1280_GAL9_DS_DATET00_inter_scale_transfer_of_kinetic_energy_"\
    "p0200hPa_n1280_maxR_02000.zarr"
}

length_scale_sets = {
    "2p5deg": np.array(
        [250,500,750,1000,1250,1600,2000,2500,3200,4000,5000,6400,8000,10000],
        dtype=np.float32
    ),
    "1p0deg": np.array(
        [110,220,330,440,550,660,800,1000,1250,1600,2000,2500,3200,4000,5000,6400,8000,10000],
        dtype=np.float32
    ),
    "0p5deg": np.array(
        [55,110,220,330,440,550,660,800,1000,1250,1600,2000,2500,3200,4000,5000,6400,8000,10000],
        dtype=np.float32
    ),
    "0p25deg": np.array(
        [28,55,110,220,330,440,550,660,800,1000,1250,1600,2000,2500,3200,4000,5000,6400,8000,10000],
        dtype=np.float32
    ),
    "n320": np.array(
        [64,128,200,250,320,400,500,640,800,1000,1250,1600,2000,2500,3200,4000,5000,6400,8000,10000],
        dtype=np.float32
    ),
    "n320_maxR_5000": np.array(
        [64,128,200,250,320,400,500,640,800,1000,1250,1600,2000,2500,],
        dtype=np.float32
    ),
    "n640": np.array(
        [32,64,100,125,160,200,250,320,400,500,640,800,1000,1250,1600,2000,2500,3200,4000,5000,6400,8000,10000],
        dtype=np.float32
    ),
    "n640_maxR_5000": np.array(
        [32,64,100,125,160,200,250,320,400,500,640,800,1000,1250,1600,2000,2500],
        dtype=np.float32
    ),
    "n1280_maxR_2000": np.array(
        [32,48,64,80,100,125,160,200,250,320,400,500,640,800,1000],
        dtype=np.float32
    ),
    "n1280_maxR_2000": np.array(
        [16,32,48,64,80,100,125,160,200,250,320,400,500,640,800,1000],
        dtype=np.float32
    ),
    "n2560_maxR_500": np.array(
        [8,16,24,32,40,48,64,80,100,125,160,200,250],
        dtype=np.float32
    )
}

def compute_inter_scale_kinetic_energy_transfer(
    du3_long,
    length_scales,
    du3_vert=None,
    ratio_rmax_to_ell=None,
    ratio_L_to_ell=None,
    norm_factor=KE_TRANSFER_NORMALIZATION
):
    """
    ratio_rmax_to_ell: this is the support of the kernel as a function of ell
    ratio_L_to_ell: this is the ratio between the physical length scale (i.e. effective resolution)
    of the kernel, and its internal length scale parameter.
    """

    r = du3_long.r

    ds_kernel = get_integration_kernels(
        r.values,
        length_scales,
        kernel_type="standard_mollifier",
        normalization="spherical",
        return_deriv=True,
    )

    kernel_props = ds_kernel.attrs["kernel_properties"]

    dG_dr = ds_kernel.dG_dr

    if ratio_rmax_to_ell is None:
        ratio_rmax_to_ell = kernel_props["ratio_rmax_to_ell"]
    if ratio_L_to_ell is None:
        ratio_L_to_ell = kernel_props["ratio_L_to_ell"]

    DL_u_long = (
        integrate_over_scales(
            du3_long,
            dG_dr * dG_dr.r,
            ratio_rmax_to_ell=ratio_rmax_to_ell,
            scale_dim="length_scale",
            radial_dim="r",
        )
        * norm_factor
    ).rename("DL_u_longitudinal")

    DL_u_long = DL_u_long.assign_coords(
        {
            "L":
            DL_u_long.length_scale
            * ratio_L_to_ell
        }
    )

    DL_u_long.attrs.update(
        {
            "kernel_properties": kernel_props,
            "ratio_physical_to_kernel_length_scale": ratio_L_to_ell,
            "units": "m2 s-3",
            "long_name": "longitudinal part of D_L(u)",
        }
    )

    DL_u = DL_u_long.to_dataset()

    if du3_vert is not None:
        G = ds_kernel.G
        
        DL_u_vert = - (
            integrate_over_scales(
                du3_vert,
                G * G.r,
                ratio_rmax_to_ell=ratio_rmax_to_ell,
                scale_dim="length_scale",
                radial_dim="r",
            )
            * norm_factor
        ).rename("DL_u_vert_transverse_times_dz")
        
        DL_u_vert = DL_u_long.assign_coords(
            {
                "L":
                DL_u_vert.length_scale
                * ratio_L_to_ell
            }
        )
        
        DL_u_vert.attrs.update(
            {
                "kernel_properties": kernel_props,
                "ratio_physical_to_kernel_length_scale": ratio_L_to_ell,
                "units": "m3 s-3",
                "long_name": "\int G_L dw |du|^2 dr",
                "description": "\int G_L dw |du|^2 dr [for computing vertical "\
                "transverse part of D_L(u)]",
            }
        )

        DL_u["DL_u_vert_transverse_times_dz"] = DL_u_vert

    DL_u = DL_u.swap_dims(
        {"length_scale": "L"}
    )

    DL_u = DL_u.rename(
        {
            "origin_longitude": "longitude",
            "origin_latitude": "latitude"
        }
    )
    return DL_u

def rho_US_SA(p):
    # p must be in hPa!

    # constants
    p0 = 1013.25 # hPa
    rho0 = 1.225 # kg m-3
    T0 = 288.15 # K
    T_strat = 216.65 # K
    p_strat = 226.32 # hPa
    R = 287.053 # J kg-1 K-1

    rho = []
    
    for plev in p:
        if plev >= p_strat:
            _rho = rho0 * (plev / p0)**(0.8097)
        else:
            _rho = (
                plev * 100 #  convert to Pa
                /
                (R * T_strat)
            )
        rho.append(_rho)

    print(rho)
    rho = np.array(rho)
    print(rho)

    rho = xr.DataArray(
        rho,
        coords={"pressure": p},
        dims="pressure",
        attrs={
            "name": "rho",
            "long_name": "density",
            "units": "kg m-3",
            "description": "1976 US Standard Atmosphere density"
        }
    )

    rho.pressure.attrs.update({"units": "hPa"})
    return rho

if __name__ == "__main__":
    date = "20160801"
    hour = 0
    time_str = f"{date}T{hour:02d}"
    #grid="2p5deg"
    #grid="1p0deg"
    #grid="0p5deg"
    #grid="0p25deg"
    #grid = "n320"
    #grid = "n320_maxR_5000"
    #grid = "n640_maxR_5000"
    grid = "n640"#_maxR_2000"
    #grid = "n1280_maxR_2000"

    save_path = "/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/"
    outname_root = "glm.n1280_GAL9"
    method_str = ""
    #maxR_str = "_maxR_global"
    maxR_str = "_maxR_02000"

    #p = np.array([100,150,200,250,300,400,500,600,700,850,925])
    p = np.array([200])#,850])
    g = 9.81 # m s-2
    rho = rho_US_SA(p)

    ds_du3 = xr.open_mfdataset(
        [
            os.path.join(
                save_path,
                f"{outname_root}"
                f"_{grid}"
                f"_delta_u_cubed"
                f"_t{time_str}"
                f"_p{int(pressure):04d}hPa"
                f"{maxR_str}"
                f"{method_str}.zarr"
            ) for pressure in p
        ],
        combine="nested",
        concat_dim="pressure",
        engine="zarr"
    ).rename({"great_circle_distance":"r"})

    r = ds_du3.r
    print("\n\n\n",r,"\n\n\n")
    length_scales = length_scale_sets[grid]*1000.

    DL_u = compute_inter_scale_kinetic_energy_transfer(
        ds_du3.delta_u_cubed_angular_integral_longitudinal,
        length_scales,
        #du3_vert=ds_du3.delta_u_cubed_angular_integral_vertical,
        ratio_rmax_to_ell=None,
        ratio_L_to_ell=None,
    )
    
    print("\n\n\n",DL_u,"\n\n\n")
    """
    DL_u_vert = (
        rho * g * DL_u.DL_u_vert_transverse_times_dz.chunk(
            chunks={"pressure":len(p)}
        ).differentiate("pressure")
        *
        1 / 100 # convert pressure from hPa to Pa
    ).rename("DL_u_vert_transverse")
    DL_u_vert.attrs = DL_u.DL_u_vert_transverse_times_dz.attrs
    DL_u_vert.attrs.update(
        {
            "units": "m2 s-3",
            "long_name": "transverse vertical part of D_L(u)",
        }
    )
    DL_u["DL_u_vert_transverse"] = DL_u_vert.chunk(chunks={"pressure":len(p)})
    """

    DL_u_zonal_mean = DL_u.mean("longitude")

    # PLOTS
    mag=5e-4
    cmap="RdBu_r"
    projection = cpy.crs.Robinson()

    

    # plot hor & vert parts for L = 500, L = 1000 km
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(20,10),
        subplot_kw={"projection":projection}
    )
    ax = axes[0,0]
    """
    DL_u.DL_u_vert_transverse.sel(
        L=500*1e3,
        method="nearest"
    ).sel(
        pressure=200,
        method="nearest",
    ).plot.pcolormesh(
        ax=ax,
        vmin=-mag,
        vmax=mag,
        cmap=cmap,
        transform=cpy.crs.PlateCarree(),
    )
    """
    ax = axes[0,1]
    DL_u.DL_u_longitudinal.sel(
        L=500*1e3,
        method="nearest"
    ).sel(
        pressure=200,
        method="nearest",
    ).plot.pcolormesh(
        ax=ax,
        vmin=-mag,
        vmax=mag,
        cmap=cmap,
        transform=cpy.crs.PlateCarree(),
    )
    ax = axes[1,0]
    """
    DL_u.DL_u_vert_transverse.sel(
        L=1000*1e3,
        method="nearest"
    ).sel(
        pressure=200,
        method="nearest",
    ).plot.pcolormesh(
        ax=ax,
        vmin=-mag,
        vmax=mag,
        cmap=cmap,
        transform=cpy.crs.PlateCarree(),
    )
    """
    ax = axes[1,1]
    DL_u.DL_u_longitudinal.sel(
        L=1000*1e3,
        method="nearest"
    ).sel(
        pressure=200,
        method="nearest",
    ).plot.pcolormesh(
        ax=ax,
        vmin=-mag,
        vmax=mag,
        cmap=cmap,
        transform=cpy.crs.PlateCarree(),
    )
    for ax in axes.flatten():
        ax.coastlines()
        ax.grid()

    plt.show()
    sys.exit(1)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(20,10),
        sharex=True,
        sharey=True,
    )
    ax=axes[0]
    DL_u_zonal_mean.DL_u_longitudinal.sel(
        L=5000*1e3,
        method="nearest"
    ).plot.pcolormesh(
        ax=ax,
        vmin=-mag,
        vmax=mag,
        cmap=cmap,
    )
    ax=axes[1]
    DL_u_zonal_mean.DL_u_vert_transverse.sel(
        L=5000*1e3,
        method="nearest"
    ).plot.pcolormesh(
        ax=ax,
        vmin=-1e-3*mag,
        vmax=1e-3*mag,
        cmap=cmap,
    )
    for ax in axes:
        ax.grid()
        ax.set_yscale("log")
        ax.yaxis.set_inverted(True)
    plt.show()
    
    sys.exit(1)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(20,10),
        subplot_kw={"projection":projection}
    )
    ax = axes[0,0]
    DL_u.isel(L=0).plot.pcolormesh(
        ax=ax,
        vmin=-mag,
        vmax=mag,
        cmap=cmap,
        transform=cpy.crs.PlateCarree(),
    )
    ax = axes[0,1]
    DL_u.sel(L=1000*1e3, method="nearest").plot.pcolormesh(
        ax=ax,
        vmin=-mag,
        vmax=mag,
        cmap=cmap,
        transform=cpy.crs.PlateCarree(),
    )
    ax = axes[1,0]
    DL_u.sel(L=5000*1e3, method="nearest").plot.pcolormesh(
        ax=ax,
        vmin=-mag,
        vmax=mag,
        cmap=cmap,
        transform=cpy.crs.PlateCarree(),
    )
    ax = axes[1,1]
    DL_u.isel(L=-1).plot.pcolormesh(
        ax=ax,
        vmin=-mag,
        vmax=mag,
        cmap=cmap,
        transform=cpy.crs.PlateCarree(),
    )
    for ax in axes.flatten():
        ax.coastlines()
        ax.grid()

    # 40S - 40N
    slice_lat = slice(-40,40)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,10), subplot_kw={"projection":projection})
    ax = axes[0,0]
    DL_u.isel(L=0).sel(latitude=slice_lat).plot.pcolormesh(
        ax=ax,
        vmin=-mag,
        vmax=mag,
        cmap=cmap,
        transform=cpy.crs.PlateCarree(),
    )
    ax = axes[0,1]
    DL_u.sel(L=1000*1e3, method="nearest").sel(latitude=slice_lat).plot.pcolormesh(
        ax=ax,
        vmin=-mag,
        vmax=mag,
        cmap=cmap,
        transform=cpy.crs.PlateCarree(),
    )
    ax = axes[1,0]
    DL_u.sel(L=5000*1e3, method="nearest").sel(latitude=slice_lat).plot.pcolormesh(
        ax=ax,
        vmin=-mag,
        vmax=mag,
        cmap=cmap,
        transform=cpy.crs.PlateCarree(),
    )
    ax = axes[1,1]
    DL_u.isel(L=-1).sel(latitude=slice_lat).plot.pcolormesh(
        ax=ax,
        vmin=-mag,
        vmax=mag,
        cmap=cmap,
        transform=cpy.crs.PlateCarree(),
    )
    for ax in axes.flatten():
        ax.coastlines()
        ax.grid()
    plt.show()
