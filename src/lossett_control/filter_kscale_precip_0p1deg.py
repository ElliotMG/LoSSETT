#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import datetime as dt

# local imports
from lossett_control.preprocessing.preprocess_kscale import load_kscale_0p5deg, \
    parse_period_id, parse_dri_mod_id, parse_nest_mod_id
from lossett.filtering.integral_filter import filter_field

if __name__ == "__main__":
    #OUT_DIR = "/gws/nopw/j04/kscale/USERS/dship/LoSSETT_out/"
    OUT_DIR = "/work/scratch-pw2/dship/LoSSETT/output/kscale/"

    # should take all of these from command line or an options file
    # simulation specification
    varname = "precipitation_rate"
    period = parse_period_id(sys.argv[1])
    dri_mod_id, dri_mod_str = parse_dri_mod_id(period,sys.argv[2])
    nest_mod_id, nest_mod_str = parse_nest_mod_id(period,dri_mod_id,sys.argv[3])
    tsteps = 24
    lon_bound_field = "periodic"
    lat_bound_field = np.nan

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
    tsteps_per_day = 24
    tchunks = 8
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
        f"max_r_deg \t= {max_r_deg:.1f}\n"\
        f"tchunks \t= {tchunks}\n"\
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

    if period == "DYAMOND_SUMMER":
        period_dirstr = "DYAMOND_Summer"
        startdate = dt.datetime(2016,8,1)
    elif period == "DYAMOND_WINTER":
        period_dirstr = "DYAMOND_Winter"
        startdate = dt.datetime(2020,1,20)
    elif period == "DYAMOND3":
        period_dirstr = period
        startdate = dt.datetime(2020,1,20)

    if dri_mod_id == "n1280gal9":
        dri_mod_fstr = "DMn1280GAL9"
    elif dri_mod_id == "n1280ral3":
        dri_mod_fstr = "DMn1280RAL3"

    if nest_mod_id == "channeln2560ral3":
        nest_mod_fstr = "RAL3_n2560"
        nest_mod_dirstr = "channel_RAL3_n2560"
    elif nest_mod_id == "channeln2560gal9":
        nest_mod_fstr = "GAL9_n2560"
        nest_mod_dirstr = "channel_GAL9_n2560"
    elif nest_mod_id == "glm":
        if dri_mod_id == "n1280gal9":
            nest_mod_fstr = "GAL9_n1280"
            nest_mod_dirstr = "global_GAL9_n1280"
        elif dri_mod_id == "n1280ral3":
            nest_mod_fstr = "RAL3_n1280"
            nest_mod_dirstr = "DMn1280RAL3"
        #endif
    #endif
            
    # load data
    field = xr.open_dataset(
        os.path.join(
            "/gws/nopw/j04/kscale/USERS/emg/data/",
            period_dirstr, "precip", nest_mod_dirstr,
            f"{nest_mod_fstr}_{dri_mod_fstr}_precip_all.nc"
        ),
        mask_and_scale = True
    )["precipitation_rate"]
    field = field.isel(latitude=slice(1,-1))
    # subset single day
    ndays = (datetime - startdate).days
    field = field.isel(time=slice(ndays*tsteps_per_day,(ndays+1)*tsteps_per_day))

    t_str = ""
    if single_t:
        # subset single time
        field = field.isel(time=tstep)
        field = field.expand_dims(dim="time")
        t_str=f"_tstep{tstep}"
    else:
        # subset time; chunk time
        if tsteps_per_day != tsteps:
            t_str = f"_tstep0-{tsteps-1}"
        field = field.isel(time=slice(0,tsteps)).chunk(chunks={"time":tchunks})

    # subset lat & lon
    subset_str=""
    if subset_lat:
        latmin = -50
        latmax = 50
        field = field.sel(latitude=slice(latmin,latmax))
        subset_str = "_50S-50N"

    if nest_mod_id == "glm":
        print(f"\n\n\nCalculating {period} global {dri_mod_id} filtered {varname} for {dt_str}")
    else:
        print(f"\n\n\nCalculating {period} {nest_mod_str} (driven by {dri_mod_str}) filtered {varname} for {dt_str}")

    print("\nInput data:\n",field)

    # filter field
    length_scales = 1000.0*np.array(
        [11,22,33,44,55,66,77,88,99,110,
         140,175,220,275,350,440,550,700,880,1100]
    )
    print("\n\n\nLength scales:\n",length_scales)
    field_filtered = filter_field(
        field, control_dict,
        length_scales=length_scales, name=varname+"_filtered"
    )
    print("\n\n\nFiltered field:\n",field_filtered)

    # save to NetCDF
    n_l = len(field_filtered.length_scale)
    L_min = field_filtered.length_scale[0].values/1000
    L_max = field_filtered.length_scale[-1].values/1000
    fpath = os.path.join(
        OUT_DIR,
        f"{nest_mod_str}.{dri_mod_str}_0p5deg_{varname}_filtered_"\
        f"Lmin_{L_min:05.0f}_Lmax_{L_max:05.0f}_{dt_str}{subset_str}{t_str}.nc"
    )
    print(f"\nFiltered {varname}:\n",field_filtered)
    print(f"\nSaving {field_filtered.name} to NetCDF at location {fpath}.")
    field_filtered.to_netcdf(fpath)

    print("\n\n\nEND.\n")
