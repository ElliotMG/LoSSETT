#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import mo_pack
import numpy as np
import xarray as xr
import datetime as dt

from lossett_control.preprocessing.preprocess_kscale import load_kscale_0p5deg
from lossett.calc.calc_inter_scale_transfers import calc_inter_scale_energy_transfer_kinetic

if __name__ == "__main__":
    # should take all of these from command line or an options file
    # simulation specification
    period = sys.argv[1]
    dri_mod_id = sys.argv[2]
    nest_mod_id = sys.argv[3]
    tsteps_per_day = 8
    lon_bound_field = "periodic"
    lat_bound_field = np.nan

    if dri_mod_id == "n1280GAL9":
        dri_mod_str = "n1280_GAL9"
    elif dri_mod_id == "n2560GAL9":
        dri_mod_str = "n2560_GAL9"
    elif dri_mod_id == "n1280RAL3":
        dri_mod_str = "n1280_RAL3"
    elif dri_mod_id == "n2560RAL3":
        dri_mod_str = "n2560_RAL3"

    if nest_mod_id in ["None","none","glm"]:
        nest_mod_id = "glm"
        nest_mod_str = "glm"

    # day & hour of simulation
    year = int(sys.argv[4])
    month = int(sys.argv[5])
    day = int(sys.argv[6])
    datetime = dt.datetime(year,month,day)
    dt_str = f"{datetime.year:04d}{datetime.month:02d}{datetime.day:02d}"

    # calculation specification
    chunk_latlon = False
    subset_lat = False
    max_r_deg = float(sys.argv[7])
    tsteps = 8
    tchunks = 8
    pchunks = 1
    prec = 1e-10

    # output directory
    OUT_DIR = sys.argv[8]
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # optional arguments
    try:
        tstep = int(sys.argv[9])
    except (ValueError, IndexError):
        tstep = None
        single_t = False
    else:
        single_t = True
        tchunks = 1

    try:
        plev = int(sys.argv[10])
    except (ValueError, IndexError):
        plev = None
        single_p = False
    else:
        single_p = True
        pchunks = 1

    print(
        "\n\nInput data specifications:\n"\
        f"period \t\t= {period}\n"\
        f"driving_model \t= {dri_mod_id}\n"\
        f"nested_model \t= {nest_mod_id}\n"\
        f"datetime \t= {dt_str}\n"\
    )
    print(
        "\nCalculation specifications:\n"\
        f"out_dir \t= {OUT_DIR}\n"\
        f"single_t \t= {single_t}\n"\
        f"single_p \t= {single_p}\n"\
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

    # load data
    ds_u_3D = load_kscale_0p5deg(
        period,datetime,driving_model=dri_mod_id,nested_model=nest_mod_id
    )

    t_str = ""
    if single_t:
        # subset single time
        ds_u_3D = ds_u_3D.isel(time=tstep)
        ds_u_3D = ds_u_3D.expand_dims(dim="time")
        t_str=f"_tstep{tstep}"
    else:
        # subset time; chunk time
        if tsteps_per_day != tsteps:
            t_str = f"_tstep0-{tsteps-1}"
        ds_u_3D = ds_u_3D.isel(time=slice(0,tsteps)).chunk(chunks={"time":tchunks})

    # subset single pressure level
    if single_p:
        ds_u_3D = ds_u_3D.sel(pressure=plev,method="nearest")
        ds_u_3D = ds_u_3D.expand_dims(dim="pressure")
        p_str = f"_p{plev:04d}"
    else:
        ds_u_3D = ds_u_3D.chunk(chunks={"pressure":pchunks})
        p_str = ""

    # subset lat & lon
    subset_str=""
    if subset_lat:
        latmin = -50
        latmax = 50
        ds_u_3D = ds_u_3D.sel(latitude=slice(latmin,latmax))
        subset_str = "_50S-50N"

    # chunk lat & lon (TEST!)
    if chunk_latlon:
        latchunks = 2560
        lonchunks = 2560
        ds_u_3D = ds_u_3D.chunk(chunks={"longitude":lonchunks,"latitude":latchunks})

    if nest_mod_id == "glm":
        print(f"\n\n\nCalculating {period} global {dri_mod_id} Dl_u for {dt_str}")
    else:
        print(f"\n\n\nCalculating {period} {nest_mod_id} (driven by {dri_mod_id}) Dl_u for {dt_str}")

    print("\nInput data:\n",ds_u_3D)
    
    # calculate kinetic DR indicator
    Dl_u = calc_inter_scale_energy_transfer_kinetic(
        ds_u_3D, control_dict
    )

    # save to NetCDF
    n_l = len(Dl_u.length_scale)
    L_min = Dl_u.length_scale[0].values/1000
    L_max = Dl_u.length_scale[-1].values/1000
    fpath = os.path.join(
        OUT_DIR,
        f"{nest_mod_id}.{dri_mod_id}_0p5deg_inter_scale_transfer_of_kinetic_energy_"\
        f"Lmin_{L_min:05.0f}_Lmax_{L_max:05.0f}_{dt_str}{subset_str}{p_str}{t_str}.nc"
    )
    print(f"\n{Dl_u.name}:\n",Dl_u)
    print(f"\nSaving {Dl_u.name} to NetCDF at location {fpath}.")
    Dl_u.to_netcdf(fpath)

    print("\n\n\nEND.\n")
