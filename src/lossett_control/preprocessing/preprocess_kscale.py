#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
import iris

DyS = ["dyamondsummer","dyamonds","dyamond1","dys","dy1","ds","d1"]

DyW = ["dyamondwinter","dyamondw","dyamond2","dyw","dy2","dw","d2"]

Dy3 = ["dyamond3","dy3","d3"]

dri_mod_str_dict = {
    "DYAMOND_SUMMER": {
        "n1280ral3": "n1280_RAL3p2",
        "n1280gal9": "n1280_GAL9"
    },
    "DYAMOND_WINTER": {
        "n1280ral3": "n1280_RAL3p2",
        "n1280gal9": "n1280_GAL9"
    },
    "DYAMOND3": {
        "n2560ral3": "n2560_RAL3p2",
        "n1280gal9": "n1280_GAL9",
        "n1280coma9": "n1280_CoMA9"
    }
}

nest_mod_str_dict = {
    "DYAMOND_SUMMER": {
        "n1280ral3": {
            "glm": "glm",
            "channeln2560ral3": "channel_n2560_RAL3p2",
            "channeln2560gal9": "channel_n2560_GAL9",
            "channelkm4p4ral3": "channel_km4p4_RAL3p2",
            "lamafricakm2p2ral3": "lam_africa_km2p2_RAL3p2",
            "lamindiakm2p2ral3": "lam_india_km2p2_RAL3p2",
            "lamsamericakm2p2ral3": "lam_samerica_km2p2_RAL3p2",
            "lamseakm2p2ral3": "lam_sea_km2p2_RAL3p2"
        },
        "n1280gal9": {
            "glm": "glm",
            "channeln2560ral3": "channel_n2560_RAL3p2",
            "channeln2560gal9": "channel_n2560_GAL9",
            "channelkm4p4ral3": "channel_km4p4_RAL3p2",
            "lamafricakm2p2ral3": "lam_africa_km2p2_RAL3p2",
            "lamafricakm4p4ral3": "lam_africa_km4p4_RAL3p2",
            "lamindiakm2p2ral3": "lam_india_km2p2_RAL3p2",
            "lamsamericakm2p2ral3": "lam_samerica_km2p2_RAL3p2",
            "lamsamericakm4p4ral3": "lam_samerica_km4p4_RAL3p2",
            "lamseakm2p2ral3": "lam_sea_km2p2_RAL3p2",
            "lamseakm4p4ral3": "lam_sea_km4p4_RAL3p2"
        },
    },
    "DYAMOND_WINTER": {
        "n1280ral3": {
            "glm": "glm",
            "channeln2560ral3": "channel_n2560_RAL3p2",
            "channeln2560gal9": "channel_n2560_GAL9",
            "channelkm4p4ral3": "channel_km4p4_RAL3p2",
            "lamafricakm2p2ral3": "lam_africa_km2p2_RAL3p2",
            "lamindiakm2p2ral3": "lam_india_km2p2_RAL3p2",
            "lamsamericakm2p2ral3": "lam_samerica_km2p2_RAL3p2",
            "lamseakm2p2ral3": "lam_sea_km2p2_RAL3p2"
        },
        "n1280gal9": {
            "glm": "glm",
            "channeln2560ral3": "channel_n2560_RAL3p2",
            "channeln2560gal9": "channel_n2560_GAL9",
            "channelkm4p4ral3": "channel_km4p4_RAL3p2",
            "lamafricakm2p2ral3": "lam_africa_km2p2_RAL3p2",
            "lamafricakm4p4ral3": "lam_africa_km4p4_RAL3p2",
            "lamindiakm2p2ral3": "lam_india_km2p2_RAL3p2",
            "lamsamericakm2p2ral3": "lam_samerica_km2p2_RAL3p2",
            "lamsamericakm4p4ral3": "lam_samerica_km4p4_RAL3p2",
            "lamseakm2p2ral3": "lam_sea_km2p2_RAL3p2",
            "lamseakm4p4ral3": "lam_sea_km4p4_RAL3p2"
        },
    },
    "DYAMOND3": {
        "n2560ral3": {
            "glm": "glm"
        },
        "n1280gal9": {
            "glm": "glm",
            "channelkm4p4ral3": "channel_km4p4_RAL3p3",
            "channelkm4p4coma9": "channel_km4p4_CoMA9",
            "lamafricakm4p4ral3": "lam_africa_km4p4_RAL3p3",
            "lamafricakm4p4coma9": "lam_africa_km4p4_CoMA9",
            "lamsamericakm4p4ral3": "lam_samerica_km4p4_RAL3p2",
            "lamsamericakm4p4coma9": "lam_samerica_km4p4_CoMA9",
            "lamseakm4p4ral3": "lam_sea_km4p4_RAL3p2",
            "lamseakm4p4coma9": "lam_sea_km4p4_CoMA9"
        },
        "n1280coma9": {
            "glm": "glm"
        },
    }
}

def parse_period_id(_period):
    # parse period
    period = _period.lower().replace("_","")

    if period in DyS:
        period = "DYAMOND_SUMMER"
    elif period in DyW:
        period = "DYAMOND_WINTER"
    elif period in Dy3:
        period = "DYAMOND3"
    else:
        print(f"Error: No period matching ID {_period}.")
        sys.exit(1)
    return period;

def parse_dri_mod_id(period,_dri_mod_id):
    # parse driving model ID
    dri_mod_id = _dri_mod_id.lower().replace("_","").replace("p2","").replace("p3","")

    # get driving model string
    try:
        dri_mod_str = dri_mod_str_dict[period][dri_mod_id]
    except:
        print(f"Error: No global model matching ID {_dri_mod_id} for period {period}.")
        sys.exit(1)
    
    return dri_mod_id, dri_mod_str;

def parse_nest_mod_id(period,dri_mod_id,_nest_mod_id):
    # parse nested model ID
    nest_mod_id = _nest_mod_id.lower().replace("_","").replace("p2","").replace("p3","")
    if nest_mod_id in ["none","glm"]:
        nest_mod_id = "glm"
    elif nest_mod_id.startswith("ctc"):
        nest_mod_id = "channel"+nest_mod_id.removeprefix("ctc")

    # get nested model string
    try:
        nest_mod_str = nest_mod_str_dict[period][dri_mod_id][nest_mod_id]
    except:
        print(f"Error: No nested model matching ID {_nest_mod_id} for period {period} driven by {dri_mod_id}.")
        sys.exit(1)
        
    return nest_mod_id, nest_mod_str;

def check_longitude(ds, out="180"):
    # Check lon direction and conventions (e.g. is lon [-180,180] or [0,360]?)
    if (ds.longitude[-1].values > 180) and (out == "180"):
        lon_attrs = ds.coords["longitude"].attrs
        ds.coords["longitude"] = (ds.coords["longitude"] + 180) % 360 - 180
        ds.longitude.attrs = lon_attrs
    elif (ds.longitude[0].values >= 0) and (out == "360"):
        print("Error! out == 360 not yet implemented!")
        sys.exit(1)
    ds = ds.sortby(ds.longitude) # has no effect if lon already correctly oriented
    return ds;

def embed_inner_grid_in_global(outer, inner, inner_type="channel", boundary_method="interp"):
    thresh=1e10
    outer = check_longitude(outer,out="180")
    inner = check_longitude(inner,out="180")
    lat_boundary_pad = 5 # number of cells to remove at latitudinal boundaries
    inner = inner.isel(latitude=slice(lat_boundary_pad,-lat_boundary_pad))
    if boundary_method == "interp":
        embedded = inner.combine_first(outer)
        embedded = embedded.chunk(
            chunks={"latitude":len(embedded.latitude),
                    "longitude":len(embedded.longitude)}
        ).compute()

        # hard-coded linear interpolation assuming uniform lat-lon grid
        # could possibly actually code up the MetUM open boundary condition?
        delta_lat = np.abs(inner.latitude[-1] - inner.latitude[-2]).values
        lat_boundary_north_1 = inner.latitude[-2]
        lat_boundary_north_2 = embedded.latitude.sel(
            latitude = inner.latitude[-1] + delta_lat, method="nearest"
        )
        lat_boundary_south_1 = inner.latitude[1]
        lat_boundary_south_2 = embedded.latitude.sel(
            latitude = inner.latitude[0] - delta_lat, method="nearest"
        )
        # interpolate northern boundary
        embedded.loc[{"latitude": inner.latitude[-1]}] = \
            0.5 * (
                embedded.loc[{"latitude": lat_boundary_north_1}]\
                + embedded.loc[{"latitude": lat_boundary_north_2}]
                )
        # interpolate southern boundary
        embedded.loc[{"latitude": inner.latitude[0]}] = \
            0.5 * (
                embedded.loc[{"latitude": lat_boundary_south_1}]\
                + embedded.loc[{"latitude": lat_boundary_south_2}]
                )
        
        if inner_type == "lam":
            print("\nError! inner_type = 'lam' not yet implemented. Exiting.")
            sys.exit(1)
    elif boundary_method == "replace":
        # check that magnitude of boundary values is sensible
        if inner.isel(latitude=0) >= 1e10 or inner.isel(latitude=-1) >= 1e10:
            # remove values closest to latudinal boundaries
            inner = inner.isel(latitude=slice(1,-1))
        #endif
        if inner.isel(longitude=0) >= 1e10 or inner.isel(longitude=-1) >= 1e10:
            # remove values closest to longitudinal boundaries
            inner = inner.isel(longitude=slice(1,-1))
        #endif
        embedded = inner.combine_first(outer)
    #endif
    
    return embedded;

def interp_time_driving_model(
        ds,
        time_offset=np.timedelta64(1, "h"),
        lag=True,
        method="linear",
        retain_original_times=True,
        keep_initial_time=True
        # should also implement a keep_final_time for lag=False
):
    # should check time type here
    times = ds.time[1:]
    if lag:
        offset_times = times - time_offset
    else:
        offset_times = times + time_offset
    #endif
    
    ds_time_interp = ds.interp(time=offset_times, method=method)
    if retain_original_times:
        ds_time_interp = ds_time_interp.assign_coords({"time": times})
    #endif
    if keep_initial_time:
        ds_time_interp = xr.concat([ds.isel(time=0),ds_time_interp],dim="time")
    # ensure correct dimension ordering
    ds_time_interp = ds_time_interp.transpose("time","pressure","latitude","longitude")
    # should change attributes to note that time interpolation has been performed
    return ds_time_interp;

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
    # parse period, driving_model, nested_model here
    period = parse_period_id(period)
    dri_mod_id, dri_mod_str = parse_dri_mod_id(period,driving_model)
    nest_mod_id, nest_mod_str = parse_nest_mod_id(period,dri_mod_id,nested_model)

    # DYAMOND SUMMER
    if period == "DYAMOND_SUMMER":
        DATA_DIR = os.path.join(DATA_DIR_ROOT,"DATA","outdir_20160801T0000Z")
        t0_str = "20160801T0000Z"
        
        # specify driving model
        if dri_mod_id == "n1280ral3":
            DATA_DIR = os.path.join(DATA_DIR,"DMn1280RAL3")
        elif dri_mod_id == "n1280gal9":
            DATA_DIR = os.path.join(DATA_DIR,"DMn1280GAL9")
        else:
            print(f"\nDriving model must be one of n1280ral3, n1280gal9, not{dri_mod_id}.")
            sys.exit(1)
        #endif

        # specify nested model
        if nest_mod_id == "glm":
            DATA_DIR = os.path.join(DATA_DIR,f"global_{dri_mod_str}")
            domain_str = "global"
        elif nest_mod_id == "channeln2560ral3":
            DATA_DIR = os.path.join(DATA_DIR,"channel_n2560_RAL3p2")
            domain_str = "channel"
        elif nest_mod_id == "channeln2560gal9":
            DATA_DIR = os.path.join(DATA_DIR,"channel_n2560_GAL9")
            domain_str = "channel"
        else:
            print(f"Nested model {nest_mod_str} not yet supported.")
            sys.exit(1)
        #endif
    #endif

    # DYAMOND WINTER
    elif period == "DYAMOND_WINTER":
        DATA_DIR = os.path.join(DATA_DIR_ROOT,"DATA","outdir_20200120T0000Z")
        t0_str = "20200120T0000Z"
        
        # specify driving model
        if dri_mod_id == "n1280ral3":
            DATA_DIR = os.path.join(DATA_DIR,"DMn1280RAL3")
        elif dri_mod_id == "n1280gal9":
            DATA_DIR = os.path.join(DATA_DIR,"DMn1280GAL9")
        #endif

        # specify nested model
        if nest_mod_id == "glm":
            DATA_DIR = os.path.join(DATA_DIR,f"global_{dri_mod_str}")
            domain_str = "global"
        elif nest_mod_id == "channeln2560ral3":
            DATA_DIR = os.path.join(DATA_DIR,"channel_n2560_RAL3p2")
            domain_str = "channel"
        elif nest_mod_id == "channeln2560gal9":
            DATA_DIR = os.path.join(DATA_DIR,"channel_n2560_GAL9")
            domain_str = "channel"
        else:
            print(f"Nested model {nest_mod_str} not yet supported.")
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
        ).assign_coords({"pressure":np.float32(plev)}).rename(
            {"x_wind":"u","y_wind":"v","upward_air_velocity":"w"}
        )
        ds_u_3D.append(ds[["u","v","w"]])
    ds_u_3D = xr.concat(ds_u_3D,dim="pressure")
    # strip nonsense values at boundaries
    ds_u_3D = ds_u_3D.isel(latitude=slice(1,-1))
    
    return ds_u_3D;

def load_kscale_native(
        period,
        datetime,
        driving_model,
        nested_model=None,
        return_iris=False,
        save_nc=False,
        force=False
):
    DATA_DIR_ROOT = "/gws/nopw/j04/kscale/"
    dt_str = f"{datetime.year:04d}{datetime.month:02d}{datetime.day:02d}T{(datetime.hour//12)*12:02d}"
    
    # should add a check that dates are in correct bounds!
    # parse period, driving_model, nested_model here
    period = parse_period_id(period)
    dri_mod_id, dri_mod_str = parse_dri_mod_id(period,driving_model)
    nest_mod_id, nest_mod_str = parse_nest_mod_id(period,dri_mod_id,nested_model)
    
    # DYAMOND 3
    if period == "DYAMOND3":
        DATA_DIR = os.path.join(DATA_DIR_ROOT,"DYAMOND3_data")

        # specify driving model
        if driving_model == "n2560ral3":
            DATA_DIR = os.path.join(DATA_DIR,"5km-RAL3")
            dri_mod_str = "n2560_RAL3p3"
        elif driving_model == "n1280gal9":
            DATA_DIR = os.path.join(DATA_DIR,"10km-GAL9-nest")
            dri_mod_str = "n1280_GAL9_nest"
        elif driving_model == "n1280coma9":
            DATA_DIR = os.path.join(DATA_DIR,"10km-CoMA9")
            dri_mod_str = "n1280_CoMA9"

        # specify nested model
        if nested_model is None or nested_model == "glm":
            DATA_DIR = os.path.join(DATA_DIR,"glm","field.pp","apverc.pp")
            nest_mod_str = "glm"
        elif driving_model != "n1280gal9":
            print(f"Error! Driving model {dri_mod_str} has no nested models.")
            sys.exit(1)
        elif nested_model == "channelkm4p4ral3":
            DATA_DIR = os.path.join(DATA_DIR,"CTC_km4p4_RAL3P3","field.pp","apverc.pp")
            nest_mod_str = "CTC_km4p4_RAL3P3"
        elif nested_model == "africakm4p4ral3":
            DATA_DIR = os.path.join(DATA_DIR,"Africa_km4p4_RAL3P3","field.pp","apverc.pp")
            nest_mod_str = "Africa_km4p4_RAL3P3"
        elif nested_model == "samerkm4p4ral3":
            DATA_DIR = os.path.join(DATA_DIR,"SAmer_km4p4_RAL3P3","field.pp","apverc.pp")
            nest_mod_str = "SAmer_km4p4_RAL3P3"
        elif nested_model == "seakm4p4ral3":
            DATA_DIR = os.path.join(DATA_DIR,"SEA_km4p4_RAL3P3","field.pp","apverc.pp")
            nest_mod_str = "SEA_km4p4_RAL3P3"
        elif nested_model == "channelkm4p4coma9":
            DATA_DIR = os.path.join(DATA_DIR,"CTC_km4p4_CoMA9_TBv1","field.pp","apverc.pp")
            nest_mod_str = "CTC_km4p4_CoMA9_TBv1"
        elif nested_model == "africakm4p4coma9":
            DATA_DIR = os.path.join(DATA_DIR,"Africa_km4p4_CoMA9_TBv1","field.pp","apverc.pp")
            nest_mod_str = "Africa_km4p4_CoMA9_TBv1"
        elif nested_model == "seakm4p4coma9":
            DATA_DIR = os.path.join(DATA_DIR,"SEA_km4p4_CoMA9_TBv1","field.pp","apverc.pp")
            nest_mod_str = "SEA_km4p4_CoMA9_TBv1"
        elif nested_model == "samerkm4p4coma9":
            DATA_DIR = os.path.join(DATA_DIR,"SAmer_km4p4_CoMA9_TBv1","field.pp","apverc.pp")
            nest_mod_str = "SAmer_km4p4_CoMA9_TBv1"
        else:
            print(f"Nested model {nested_model} not yet supported (or does not exist).")
            sys.exit(1)

        fpath = os.path.join(DATA_DIR,f"{nest_mod_str}.{dri_mod_str}.apverc_{dt_str}.pp")

    #endif

    # DyS and DyW native res. data not yet on GWS DATA, so read from Elliot's GWS USER
    # root file path = /gws/nopw/j04/kscale/USERS/emg/data/native_res_deterministic/{period}/{model_id}/
    
    # DYAMOND SUMMER
    elif period == "DYAMOND_SUMMER":
        # DyS native res. data not yet on GWS, so read from Elliot's scratch
        #DATA_DIR = os.path.join(DATA_DIR_ROOT, "DATA","outdir_20160801T0000Z")
        DATA_DIR = "/gws/nopw/j04/kscale/USERS/emg/data/native_res_deterministic/DS"

        start_date = dt.datetime(2016,8,1,0)
        delta = datetime - start_date
        hrs_since_start = int(delta.total_seconds()/3600)
        hr_str = f"{hrs_since_start:03d}"

        # specify driving model
        if driving_model != "n1280gal9":
            print(f"Error! Period {period} has no driving model named {driving_model}.")
            sys.exit(1)
        
        dri_mod_str = "n1280_GAL9"

        # specify nested model
        if nested_model is None or nested_model == "glm":
            DATA_DIR = os.path.join(DATA_DIR,"global_n1280_GAL9")
            nest_mod_str = "glm"
        elif nested_model == "channeln2560gal9":
            DATA_DIR = os.path.join(DATA_DIR,"CTC_N2560_GAL9")
            nest_mod_str = "CTC_n2560_GAL9"
        elif nested_model == "channeln2560ral3":
            DATA_DIR = os.path.join(DATA_DIR,"CTC_N2560_RAL3p2")
            nest_mod_str = "CTC_n2560_RAL3p2"
        elif nested_model == "channelkm4p4ral3":
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

    # add units to pressure coord
    ds.pressure.attrs["units"] = "hPa"

    # save NetCDF to scratch
    if save_nc:
        from pathlib import Path
        #SAVE_DIR = "/work/scratch-pw4/dship/LoSSETT/preprocessed_kscale_data"
        SAVE_DIR = f"/gws/nopw/j04/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/{period}"
        Path(SAVE_DIR).mkdir(parents=True,exist_ok=True)
        fpath = os.path.join(SAVE_DIR,f"{nest_mod_str}.{dri_mod_str}.uvw_{dt_str}.nc")
        if not os.path.exists(fpath) or force:
            print(f"\n\n\nSaving velocity data to NetCDF at {fpath}.")
            ds.to_netcdf(fpath) # available engines: netcdf4, h5netcdf, scipy
    
    if return_iris:
        return ds, data_iris;
    else:
        return ds;

if __name__ == "__main__":
    _period=sys.argv[1]
    _dri_mod_id = sys.argv[2]
    _nest_mod_id = sys.argv[3]
    grid = sys.argv[4]
    year = int(sys.argv[5])
    month = int(sys.argv[6])
    day = int(sys.argv[7])
    hour = int(sys.argv[8])
    save_nc = False
    #plevs = [100,150,200,250,300,400,500,600,700,850,925,1000]
    plevs = [200]
    
    # parse period, driving model, nested model
    period = parse_period_id(_period)
    dri_mod_id, dri_mod_str = parse_dri_mod_id(period,_dri_mod_id)
    nest_mod_id, nest_mod_str = parse_nest_mod_id(period,dri_mod_id,_nest_mod_id)

    datetime = dt.datetime(year,month,day,hour)
    dt_str = f"{datetime.year:04d}{datetime.month:02d}{datetime.day:02d}T{(datetime.hour//12)*12:02d}"
    
    print("\n\n\nPreprocessing details:")
    print(
        "\n\nPreprocessing details:\n"\
        f"period \t\t= {period}\n"\
        f"driving_model \t= {dri_mod_str} (ID: {dri_mod_id})\n"\
        f"nested_model \t= {nest_mod_str} (ID: {nest_mod_id})\n"\
        f"datetime \t= {dt_str}\n"\
        f"grid \t\t= {grid}\n"\
    )
        
    if grid == "native":
        ds = load_kscale_native(
            period,
            datetime,
            driving_model=dri_mod_id,
            nested_model=nest_mod_id,
            save_nc=save_nc
        ).sel(pressure=plevs,method="nearest")
    elif grid == "n1280":
        DATA_DIR = \
            "/gws/nopw/j04/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/DYAMOND_SUMMER/n1280_regrid"
        fpath = os.path.join(DATA_DIR,f"{nest_mod_str}.{dri_mod_str}.uvw_{dt_str}_n1280.nc")
        ds_inner = xr.open_dataset(
            fpath,mask_and_scale=True,drop_variables="leadtime"
        ).sel(pressure=plevs,method="nearest")
        ds_outer = load_kscale_native(
            period,
            datetime,
            driving_model=dri_mod_id,
            nested_model="glm"
        ).sel(pressure=plevs,method="nearest")
    elif grid == "0p5deg":
        ds_inner = load_kscale_0p5deg(
            period,
            datetime,
            driving_model=dri_mod_id,
            nested_model=nest_mod_id,
            plevs=plevs
        )
        ds_outer = load_kscale_0p5deg(
            period,
            datetime,
            driving_model=dri_mod_id,
            nested_model="glm",
            plevs=plevs
        )
    #endif
    
    print("\n\nInner:\n",ds_inner)
    print("\n\nOuter:\n",ds_outer)

    # time-interpolate driving field
    ds_outer_t_interp = interp_time_driving_model(ds_outer, time_offset=np.timedelta64(1, "h"))
    print("\n\nOuter (time-interpolated to 1H lag):\n",ds_outer_t_interp)
    
    import matplotlib.pyplot as plt
    tstep = 1
    
    #plt.figure()
    #ds_outer.u.isel(time=tstep).plot()
    #plt.figure()
    #ds_outer_t_interp.u.isel(time=tstep).plot()
    #plt.show()
    
    ds_embed_interp = embed_inner_grid_in_global(
        ds_outer,
        ds_inner,
        boundary_method="interp"
    )
    ds_embed_interp_time_interp = embed_inner_grid_in_global(
        ds_outer_t_interp,
        ds_inner,
        boundary_method="interp"
    )
    ds_embed_replace = embed_inner_grid_in_global(
        ds_outer,
        ds_inner,
        boundary_method="replace"
    )

    # ADD PLOTS ZOOMED IN TO REGIONS!

    # plot
    # boundary values interpolated
    plt.figure()
    ds_embed_interp.u.isel(time=tstep).plot()
    plt.title("embedded, boundary interpolated")
    plt.figure()
    (ds_embed_interp-ds_outer).u.isel(time=tstep).plot()
    plt.title("embedded, boundary interpolated (diff from driving)")
    plt.figure()
    ds_embed_interp_time_interp.u.isel(time=tstep).plot()
    plt.title("embedded, boundary interpolated, time interpolated")
    plt.figure()
    (ds_embed_interp_time_interp-ds_outer).u.isel(time=tstep).plot()
    plt.title("embedded, boundary interpolated, time interpolated (diff from driving)")
    # boundary values replaced
    #plt.figure()
    #ds_embed_replace.u.isel(time=tstep).plot()
    #plt.title("embedded, boundary replaced")
    #plt.figure()
    #(ds_embed_replace-ds_outer).u.isel(time=tstep).plot()
    #plt.title("embedded, boundary replaced (diff from driving)")
    
    plt.show()
    
    print("\n\n\nEND.")
