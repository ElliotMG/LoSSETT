#!/usr/bin/env python3
import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy as cpy

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
        [16,32,48,64,80,100,125,160,200,250,320,400,500,640,800,1000],
        dtype=np.float32
    ),
    "n2560_maxR_500": np.array(
        [8,16,24,32,40,48,64,80,100,125,160,200,250],
        dtype=np.float32
    )
}

def compute_inter_scale_kinetic_energy_transfer(
    du3,
    length_scales,
    ratio_rmax_to_ell=None,
    ratio_L_to_ell=None,
    norm_factor=KE_TRANSFER_NORMALIZATION
):
    """
    ratio_rmax_to_ell: this is the support of the kernel as a function of ell
    ratio_L_to_ell: this is the ratio between the physical length scale (i.e. effective resolution)
    of the kernel, and its internal length scale parameter.
    """

    r = du3.r

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

    DL_u = (
        integrate_over_scales(
            du3,
            dG_dr * dG_dr.r,
            ratio_rmax_to_ell=ratio_rmax_to_ell,
            scale_dim="length_scale",
            radial_dim="r",
        )
        * norm_factor
    )

    DL_u = DL_u.assign_coords(
        {
            "L":
            DL_u.length_scale
            * ratio_L_to_ell
        }
    )

    DL_u.attrs.update(
        {
            "kernel_properties": kernel_props,
            "ratio_physical_to_kernel_length_scale": ratio_L_to_ell,
            "units": "m2 s-3"
        }
    )

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

if __name__ == "__main__":
    date = "20160801"
    #grid="2p5deg"
    grid="1p0deg"
    #grid = "n320_maxR_5000"
    #grid = "n640_maxR_5000"
    #grid = "n1280_maxR_2000"
    fpath = fpaths[grid].replace("DATE", date)
    ds = xr.open_zarr(fpath)

    du3 = ds.delta_u_cubed_angular_integral.rename({"great_circle_distance":"r"})

    r = du3.r
    print("\n\n\n",r,"\n\n\n")
    length_scales = length_scale_sets[grid]*1000.

    DL_u = compute_inter_scale_kinetic_energy_transfer(
        du3,
        length_scales,
        ratio_rmax_to_ell=None,
        ratio_L_to_ell=None,
    )
    
    print("\n\n\n",DL_u,"\n\n\n")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,10), subplot_kw={"projection":cpy.crs.Robinson()})
    ax = axes[0,0]
    DL_u.isel(L=0).plot.pcolormesh(ax=ax, transform=cpy.crs.PlateCarree())
    ax = axes[0,1]
    DL_u.sel(L=1000*1e3, method="nearest").plot.pcolormesh(ax=ax, transform=cpy.crs.PlateCarree())
    ax = axes[1,0]
    DL_u.sel(L=5000*1e3, method="nearest").plot.pcolormesh(ax=ax, transform=cpy.crs.PlateCarree())
    ax = axes[1,1]
    DL_u.isel(L=-1).plot.pcolormesh(ax=ax, transform=cpy.crs.PlateCarree())

    for ax in axes.flatten():
        ax.coastlines()
        ax.grid()
    plt.show()
