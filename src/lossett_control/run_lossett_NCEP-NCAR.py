#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import mo_pack
import numpy as np
import xarray as xr
import datetime as dt
import calendar

from lossett.calc.calc_inter_scale_transfers import calc_inter_scale_energy_transfer_kinetic

if __name__ == "__main__":
    DATA_DIR = "/home/users/dship/NCEP-NCAR_Reanalysis_1/"
    # should take all of these from command line or an options file
    # simulation specification
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    max_r_deg = float(sys.argv[3])
    OUT_DIR = DATA_DIR #sys.argv[4]

    moist = False

    if moist:
        fpath = os.path.join(DATA_DIR, f"NCEP_NCAR_uvwTqrho_moist_{year}.nc")
    else:
        fpath = os.path.join(DATA_DIR, f"NCEP_NCAR_uvwTrho_{year}.nc")
    
    tsteps_per_file = 4 #required
    lon_bound_field = "periodic"
    lat_bound_field = np.nan

    # start and end dates
    ndays = calendar.monthrange(year, month)[1]
    start = dt.datetime(year,month,1,hour=0,minute=0)
    end = dt.datetime(year,month,ndays,hour=23,minute=59)

    # calculation specification
    chunk_latlon = False
    subset_lat = False # should be optional argument!
    latmin = -60
    latmax = 60
    tsteps = 4
    tchunks = 40
    pchunks = 17
    prec = 1e-10

    # output directory
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    print(
        "\nCalculation specifications:\n"\
        f"out_dir \t= {OUT_DIR}\n"\
        f"max_r_deg \t= {max_r_deg:.1f}\n"\
        f"tchunks \t= {tchunks}\n"\
        f"pchunks \t= {pchunks}\n"\
        f"subset_lat \t= {subset_lat}\n"\
    )

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

    # open data
    if moist:
        ds_u_3D = xr.open_dataset(
            fpath,
            mask_and_scale=True,
            drop_variables={"T","q","rho"}
        )
    else:
        ds_u_3D = xr.open_dataset(
            fpath,
            mask_and_scale=True,
            drop_variables={"T","rho"}
        )

    # subset time to month
    ds_u_3D = ds_u_3D.sel(time=slice(start,end))
    # chunk time
    ds_u_3D = ds_u_3D.chunk(chunks={"time":tchunks})

    # subset lat, lon
    subset_str=""
    if subset_lat:
        ds_u_3D = ds_u_3D.sel(latitude=slice(latmin,latmax))
        # modify to check sign of latmin, latmax to correctly infer South/North
        subset_str = f"_{np.abs(latmin):02d}S-{latmax:02d}N"

    print("\nInput data:\n",ds_u_3D)
    
    # calculate kinetic DR indicator
    Dl_u = calc_inter_scale_energy_transfer_kinetic(
        ds_u_3D, control_dict
    )

    # save to NetCDF
    n_l = len(Dl_u.length_scale)
    L_min = Dl_u.length_scale[0].values/1000
    L_max = Dl_u.length_scale[-1].values/1000
    if moist:
        fpath = os.path.join(
            OUT_DIR,
            f"NCEP-NCAR_inter_scale_transfer_of_kinetic_energy_"\
            f"Lmin_{L_min:05.0f}_Lmax_{L_max:05.0f}_{year:04d}{month:02d}{subset_str}"\
            "_moist_density.nc"
        )
    else:
        fpath = os.path.join(
            OUT_DIR,
            f"NCEP-NCAR_inter_scale_transfer_of_kinetic_energy_"\
            f"Lmin_{L_min:05.0f}_Lmax_{L_max:05.0f}_{year:04d}{month:02d}{subset_str}.nc"
        )
    print(f"\n{Dl_u.name}:\n",Dl_u)
    print(f"\nSaving {Dl_u.name} to NetCDF at location {fpath}.")
    Dl_u.to_netcdf(fpath)

    print("\n\n\nEND.\n")
