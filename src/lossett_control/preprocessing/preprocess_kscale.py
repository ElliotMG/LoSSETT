#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
import iris

def setup_vars_DS(return_dates=False,grid="0p5deg"):
    # This currently uses the already-processed DS data in Elliot's directory;
    # would be better to use the data in $KSCALE/DATA instead
    
    # 1. data directory
    data_dir = "/gws/nopw/j04/kscale/USERS/emg/data/DYAMOND_Summer/"

    # 2. variable names
    u_name = "u"
    v_name = "v"
    w_name = "w"
    var_names = [u_name, v_name, w_name]

    # 3. define date range
    start_date = dt.date(2016,8,1) # take as command line argument, or from options file
    ndays = 40
    dates = [start_date + dt.timedelta(i) for i in range(ndays)]
    
    if return_dates:
        return var_names, data_dir, dates;
    else:
        return var_names, data_dir;

def setup_vars_DW(return_dates=False,grid="0p5deg"):

    return 0;

def load_kscale(simid, period, grid):
    tsteps_per_day = 8
    if period == "DS":
        var_names, data_dir, dates = setup_vars_DS(return_dates=True,grid=grid)
        
    if period == "DS" and grid == "0p5deg":
        # This currently uses the already-processed DS data in Elliot's directory;
        # would be better to use the data in $KSCALE/DATA instead
        ds_u_3D = xr.open_mfdataset(
            [
                os.path.join(data_dir,f"{var_name}_DS_3D_{simid}.nc")\
                for var_name in var_names
            ],
            mask_and_scale = True
        )
        # subset dates
        start = dates[0]
        start_datetime = dt.datetime(start.year,start.month,start.day,3)
        end = dates[-1] + dt.timedelta(1)
        end_datetime = dt.datetime(end.year,end.month,end.day,0)
        ds_u_3D = ds_u_3D.sel(time=slice(start_datetime,end_datetime))

    # rename variables
    ds_u_3D = ds_u_3D.rename(
        {
            "x_wind":"u",
            "y_wind":"v",
            "upward_air_velocity":"w"
        }
    )
    
    return ds_u_3D;

def load_kscale_0p5deg(
        period,
        datetime,
        driving_model,
        nested_model=None,
        plevs=[100,150,200,250,300,400,500,600,700,850,925,1000]
):
    DATA_DIR_ROOT = "/gws/nopw/j04/kscale"
    dt_str = f"{datetime.year:04d}{datetime.month:02d}{datetime.day:02d}"

    # should add a check that dates are in correct bounds!

    # DYAMOND SUMMER
    if period == "DYAMOND_SUMMER":
        DATA_DIR = os.path.join(DATA_DIR_ROOT,"DATA","outdir_20160801T0000Z")
        t0_str = "20160801T0000Z"
        
        # specify driving model
        if driving_model == "n1280RAL3":
            DATA_DIR = os.path.join(DATA_DIR,"DMn1280RAL3")
            dri_mod_str = "n1280_RAL3p2"
        elif driving_model == "n1280GAL9":
            DATA_DIR = os.path.join(DATA_DIR,"DMn1280GAL9")
            dri_mod_str = "n1280_GAL9"
        #endif

        # specify nested model
        if nested_model is None or nested_model == "glm":
            DATA_DIR = os.path.join(DATA_DIR,f"global_{dri_mod_str}")
            nest_mod_str = "glm"
            domain_str = "global"
        elif nested_model == "channel_n2560_RAL3":
            DATA_DIR = os.path.join(DATA_DIR,"channel_n2560_RAL3p2")
            domain_str = "channel"
        elif nested_model == "channel_n2560_GAL9":
            DATA_DIR = os.path.join(DATA_DIR,"channel_n2560_GAL9")
            domain_str = "channel"
        else:
            print(f"Nested model {nested_model} not yet supported.")
            sys.exit(1)
        #endif
    #endif

    # DYAMOND WINTER
    elif period == "DYAMOND_WINTER":
        DATA_DIR = os.path.join(DATA_DIR_ROOT,"DATA","outdir_20200120T0000Z")
        t0_str = "20200120T0000Z"
        
        # specify driving model
        if driving_model == "n1280RAL3":
            DATA_DIR = os.path.join(DATA_DIR,"DMn1280RAL3")
            dri_mod_str = "n1280_RAL3p2"
        elif driving_model == "n1280GAL9":
            DATA_DIR = os.path.join(DATA_DIR,"DMn1280GAL9")
            dri_mod_str = "n1280_GAL9"
        #endif

        # specify nested model
        if nested_model is None or nested_model == "glm":
            DATA_DIR = os.path.join(DATA_DIR,f"global_{dri_mod_str}")
            nest_mod_str = "glm"
            domain_str = "global"
        elif nested_model == "channel_n2560_RAL3":
            DATA_DIR = os.path.join(DATA_DIR,"channel_n2560_RAL3p2")
            domain_str = "channel"
        elif nested_model == "channel_n2560_GAL9":
            DATA_DIR = os.path.join(DATA_DIR,"channel_n2560_GAL9")
            domain_str = "channel"
        else:
            print(f"Nested model {nested_model} not yet supported.")
            sys.exit(1)
        #endif
    #endif

    # DYAMOND 3
    elif period == "DYAMOND3":
        print("DYAMOND3 data coarsened to 0.5deg is not yet available.")
        sys.exit(1)
    #endif

    ds_u_3D = []
    for plev in plevs:
        ds = xr.open_dataset(
            os.path.join(
                DATA_DIR,
                f"profile_{plev}",
                f"{dt_str}_{t0_str}_{domain_str}_profile_3hourly_{plev}_05deg.nc"
            ),
            drop_variables=["forecast_reference_time","forecast_period"],
            mask_and_scale=True
        ).assign_coords({"pressure":plev}).rename({"x_wind":"u","y_wind":"v","upward_air_velocity":"w"})
        ds_u_3D.append(ds[["u","v","w"]])
    ds_u_3D = xr.concat(ds_u_3D,dim="pressure")
    
    return ds_u_3D;

def load_kscale_native(
        period,
        datetime,
        driving_model,
        nested_model=None,
        return_iris=False,
        save_nc=False
):
    DATA_DIR_ROOT = "/gws/nopw/j04/kscale/"
    dt_str = f"{datetime.year:04d}{datetime.month:02d}{datetime.day:02d}T{(datetime.hour//12)*12:02d}"
    
    # DYAMOND 3
    if period == "DYAMOND3": # change to allow also Dy3, D3
        DATA_DIR = os.path.join(DATA_DIR_ROOT,"DYAMOND3_data")

        # specify driving model
        if driving_model == "n2560RAL3":
            DATA_DIR = os.path.join(DATA_DIR,"5km-RAL3")
            dri_mod_str = "n2560_RAL3p3"
        elif driving_model == "n1280GAL9":
            DATA_DIR = os.path.join(DATA_DIR,"10km-GAL9-nest")
            dri_mod_str = "n1280_GAL9_nest"
        elif driving_model == "n1280CoMA9":
            DATA_DIR = os.path.join(DATA_DIR,"10km-CoMA9")
            dri_mod_str = "n1280_CoMA9"

        # specify nested model
        if nested_model is None or nested_model == "glm":
            DATA_DIR = os.path.join(DATA_DIR,"glm","field.pp","apverc.pp")
            nest_mod_str = "glm"

        fpath = os.path.join(DATA_DIR,f"{nest_mod_str}.{dri_mod_str}.apverc_{dt_str}.pp")

    #endif

    # DyS and DyW native res. data not yet on GWS DATA, so read from Elliot's GWS USER
    # root file path = /gws/nopw/j04/kscale/USERS/emg/data/native_res_deterministic/{period}/{model_id}/
    
    # DYAMOND SUMMER
    elif period == "DYAMOND_SUMMER": # change to allow also DyS, DS, DYAMOND1, Dy1, D1
        # DyS native res. data not yet on GWS, so read from Elliot's scratch
        #DATA_DIR = os.path.join(DATA_DIR_ROOT, "DATA","outdir_20160801T0000Z")
        DATA_DIR = "/gws/nopw/j04/kscale/USERS/emg/data/native_res_deterministic/DS"

        start_date = dt.datetime(2016,8,1,0)
        delta = datetime - start_date
        hrs_since_start = int(delta.total_seconds()/3600)
        hr_str = f"{hrs_since_start:03d}"

        # specify driving model
        if driving_model != "n1280GAL9":
            print(f"Error! Period {period} has no driving model named {driving_model}.")
            sys.exit(1)
        
        dri_mod_str = "n1280_GAL9"

        # specify nested model
        if nested_model is None or nested_model == "glm":
            DATA_DIR = os.path.join(DATA_DIR,"global_n1280_GAL9")
            nest_mod_str = "glm"
        elif nested_model == "CTCn2560GAL9":
            DATA_DIR = os.path.join(DATA_DIR,"CTC_N2560_GAL9")
            nest_mod_str = "CTC_n2560_GAL9"
        elif nested_model == "CTCn2560RAL3p2":
            DATA_DIR = os.path.join(DATA_DIR,"CTC_N2560_RAL3p2")
            nest_mod_str = "CTC_n2560_RAL3p2"
        elif nested_model == "CTCkm4p4RAL3p2":
            DATA_DIR = os.path.join(DATA_DIR,"CTC_N2560_GAL3p2")
            nest_mod_str = "CTC_km4p4_RAL3p2"
        else:
            print(f"Error! Period {period} has no nested model named {nested_model}.")
            sys.exit(1)

        fpath = os.path.join(DATA_DIR,f"20160801T0000Z_{nest_mod_str}_pverc{hr_str}.pp")

    #endif
            

    # DYAMOND WINTER
    elif period == "DYAMOND_WINTER": # change to allow also DyW, DW, DYAMOND2, Dy2, D2
        DATA_DIR = os.path.join(DATA_DIR_ROOT, "DATA","outdir_20200120T0000Z")

    # LOAD u,v,w from PP file using Iris
    if not os.path.exists(fpath):
        print(f"ERROR: data does not exist at {fpath}")
        sys.exit(1)
    else:
        print(f"Loading velocity data from {fpath}")
        data_iris = iris.load(fpath)

    # extract u,v,w
    names = ["x_wind","y_wind","upward_air_velocity"]
    name_cons = [iris.Constraint(name=name) for name in names]
    u = data_iris.extract_cube(name_cons[0])
    v = data_iris.extract_cube(name_cons[1])
    w = data_iris.extract_cube(name_cons[2])
    # u,v,w are on B-grid (u,v at cell vertices, w at cell centres)
    # thus linearly interpolate w to cell vertices (done lazily)
    w = w.regrid(u[0,0,:,:],iris.analysis.Linear())
    u.rename("u")
    v.rename("v")
    w.rename("w")
    data_iris = iris.cube.CubeList([u,v,w])
    
    # convert to xarray Dataset
    uvw = [xr.DataArray.from_iris(vel_cpt) for vel_cpt in [u,v,w]]
    ds = xr.merge(uvw)

    # save NetCDF to scratch
    if save_nc:
        from pathlib import Path
        #SAVE_DIR = "/work/scratch-pw2/dship/LoSSETT/preprocessed_kscale_data"
        #SAVE_DIR = "/gws/nopw/j04/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data"
        SAVE_DIR = "/work/scratch-nopw2/dship/LoSSETT/preprocessed_kscale_data"
        Path(SAVE_DIR).mkdir(parents=True,exist_ok=True)
        fpath = os.path.join(SAVE_DIR,f"{nest_mod_str}.{dri_mod_str}.uvw_{dt_str}.nc")
        if not os.path.exists(fpath):
            print(f"\n\n\nSaving velocity data to NetCDF at {fpath}.")
            ds.to_netcdf(fpath) # available engines: netcdf4, h5netcdf, scipy
    
    if return_iris:
        return ds, data_iris;
    else:
        return ds;

def global_regrid(field, target_grid):
    return 0;

def nest_in_global_grid():
    return 0;

if __name__ == "__main__":
    period=sys.argv[1]
    driving_model = sys.argv[2]
    nested_model = sys.argv[3]
    grid = sys.argv[4]
    year = int(sys.argv[5])
    month = int(sys.argv[6])
    day = int(sys.argv[7])
    hour = int(sys.argv[8])
    save_nc = False
    datetime = dt.datetime(year,month,day,hour)
    print("\n\n\nPreprocessing details:")
    print(
        f"\nPeriod: {period}, driving model: {driving_model}, nested_model = {nested_model}, "\
        f"grid = {grid}, date = {year:04d}-{month:02d}-{day:02d}, hour = {hour:02d}"
    )
    if nested_model == "none":
        nested_model=None
    if grid == "native":
        ds = load_kscale_native(
            period,
            datetime,
            driving_model,
            nested_model=nested_model,
            save_nc=save_nc
        )
    elif grid == "0p5deg":
        ds = load_kscale_0p5deg(
            period,
            datetime,
            driving_model,
            nested_model=None,
            plevs=[100,150,200,250,300,400,500,600,700,850,925,1000]
        )
    print(ds)
    print("\n\n\nEND.")
