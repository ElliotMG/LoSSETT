#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import mo_pack
import numpy as np
import xarray as xr
import datetime as dt

from lossett_control.preprocessing.preprocess_kscale import load_kscale_native, check_longitude, \
    parse_period_id, parse_dri_mod_id, parse_nest_mod_id
from lossett.calc.calc_inter_scale_transfers import calc_inter_scale_energy_transfer_kinetic

if __name__ == "__main__":
    # should take all of these from command line or an options file
    # simulation specification
    _period = sys.argv[1]
    _dri_mod_id = sys.argv[2]
    _nest_mod_id = sys.argv[3]
    #tsteps_per_day = 8
    tsteps_per_file = 4 #required
    lon_bound_field = "periodic"
    lat_bound_field = np.nan

    # parse period, driving model, nested model
    period = parse_period_id(_period)
    dri_mod_id, dri_mod_str = parse_dri_mod_id(period,_dri_mod_id)
    nest_mod_id, nest_mod_str = parse_nest_mod_id(period,dri_mod_id,_nest_mod_id)

    # day & hour of simulation
    year = int(sys.argv[4])
    month = int(sys.argv[5])
    day = int(sys.argv[6])
    hour = int(sys.argv[7]) # optional
    datetime = dt.datetime(year,month,day,hour)
    dt_str = f"{datetime.year:04d}{datetime.month:02d}{datetime.day:02d}T{(datetime.hour//12)*12:02d}"

    # calculation specification
    chunk_latlon = False
    subset_lat = False # should be optional argument!
    latmin = -50
    latmax = 50
    max_r_deg = float(sys.argv[8]) # required
    tsteps = 4
    tchunks = 1
    pchunks = 1
    prec = 1e-10

    # output directory
    OUT_DIR_ROOT = sys.argv[9] # required
    OUT_DIR = os.path.join(OUT_DIR_ROOT, period)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # optional arguments
    try:
        tstep = int(sys.argv[10]) # optional
    except (IndexError, ValueError):
        tstep = None
        single_t = False
    else:
        single_t = True

    try:
        plev = int(sys.argv[11]) # optional
    except (IndexError, ValueError):
        plev = None
        single_p = False
    else:
        single_p = True

    try:
        load_nc = sys.argv[12] # optional
    except:
        load_nc = False
    else:
        if load_nc == "true" or load_nc == "True":
            load_nc = True
        else:
            load_nc = False    

    print(
        "\n\nInput data specifications:\n"\
        f"period \t\t= {period}\n"\
        f"driving_model \t= {dri_mod_str} (ID: {dri_mod_id})\n"\
        f"nested_model \t= {nest_mod_str} (ID: {nest_mod_id})\n"\
        f"datetime \t= {dt_str}\n"\
    )
    print(
        "\nCalculation specifications:\n"\
        f"out_dir \t= {OUT_DIR}\n"\
        f"load_nc \t= {load_nc}\n"\
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

    # open data
    if load_nc:
        DATA_DIR = "/work/scratch-pw2/dship/LoSSETT/preprocessed_kscale_data"
        if dri_mod_id == "n2560RAL3":
            dri_mod_str = "n2560_RAL3p3"
        fpath = os.path.join(DATA_DIR,f"{nest_mod_str}.{dri_mod_str}.uvw_{dt_str}.nc")
        print(f"\nLoading via tmp NetCDF from {fpath}")
        ds_u_3D = xr.open_dataset(fpath)
    else:
        ds_u_3D = load_kscale_native(
            period,datetime,driving_model=dri_mod_id,nested_model=nest_mod_id
        )
    
    if single_t:
        # subset single time
        ds_u_3D = ds_u_3D.isel(time=tstep)
        ds_u_3D = ds_u_3D.expand_dims(dim="time")
        t_str=f"_tstep{tstep}"
    else:
        t_str = ""
        if tsteps != tsteps_per_file:
            # subset time
            t_str = f"_tstep0-{tsteps-1}"
            ds_u_3D = ds_u_3D.isel(time=slice(0,tsteps))
        # chunk time
        ds_u_3D = ds_u_3D.chunk(chunks={"time":tchunks})

    # subset single pressure level
    if single_p:
        ds_u_3D = ds_u_3D.sel(pressure=plev,method="nearest")
        ds_u_3D = ds_u_3D.expand_dims(dim="pressure")
        p_str = f"_p{plev:04d}"
    else:
        p_str = ""
        plevs = [50,200,500,700,850]#[200,850]
        ds_u_3D = ds_u_3D.sel(pressure=plevs,method="nearest").chunk(
            chunks={"pressure":pchunks}
        )
    ds_u_3D.pressure.attrs["units"] = "hPa"

    # chunk lat & lon (TEST!)
    if chunk_latlon:
        latchunks = 2560
        lonchunks = 2560
        ds_u_3D = ds_u_3D.chunk(chunks={"longitude":lonchunks,"latitude":latchunks})
    subset_str=""
    if subset_lat:
        ds_u_3D = ds_u_3D.sel(latitude=slice(latmin,latmax))
        # modify to check sign of latmin, latmax to correctly infer South/North
        subset_str = f"_{latmin:02d}S-{latmax:02d}N"

    # check longitude is [-180,180]
    ds_u_3D = check_longitude(ds_u_3D)

    if nest_mod_id == "glm":
        print(f"\n\n\nCalculating {period} global {dri_mod_str} DR indicator for {dt_str}")
    else:
        print(f"\n\n\nCalculating {period} {nest_mod_str} (driven by {dri_mod_str}) DR indicator for {dt_str}")

    print("\nInput data:\n",ds_u_3D)
    
    # specify length scales (10 length scales per decade unless 2dx > spacing between consecutive \ell)
    length_scales = np.array(
        [8,16,24,32,40,48,64,80,100,125,160,200,250,320,400]
    )
    length_scales = 1000.0 * length_scales # convert to m; ensure float
    
    # calculate kinetic DR indicator
    Dl_u = calc_inter_scale_energy_transfer_kinetic(
        ds_u_3D, control_dict, length_scales=length_scales
    )
    
    # ensure correct dimension ordering
    Dl_u = Dl_u.transpose("length_scale","time","pressure","latitude","longitude")

    # save to NetCDF
    n_l = len(Dl_u.length_scale)
    L_min = Dl_u.length_scale[0].values/1000
    L_max = Dl_u.length_scale[-1].values/1000
    fpath = os.path.join(
        OUT_DIR,
        f"{nest_mod_str}.{dri_mod_str}_inter_scale_transfer_of_kinetic_energy_"\
        f"Lmin_{L_min:05.0f}_Lmax_{L_max:05.0f}_{dt_str}{subset_str}{p_str}{t_str}.nc"
    )
    print(f"\n{Dl_u.name}:\n",Dl_u)
    print(f"\nSaving {Dl_u.name} to NetCDF at location {fpath}.")
    Dl_u.to_netcdf(fpath)

    print("\n\n\nEND.\n")
