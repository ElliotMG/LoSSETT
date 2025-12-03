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
        "n2560ral3": "n1280_RAL3p2",
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
        print(f"Error: No nested model matching ID {_nest_mod_id} for period {period} driven by {dri_mod_str}.")
        sys.exit(1)
        
    return nest_mod_id, nest_mod_str;

def embed_inner_grid_in_global(outer, inner, type="channel", method="interp"):
    thresh=1e10
    if method == "interp":
        _embedded = inner.combine_first(outer)
        # replace daft values with nan
        _embedded = _embedded.where(_embedded < thresh, np.nan)
        _embedded = _embedded.chunk(
            chunks={"latitude":len(_embedded.latitude),
                    "longitude":len(_embedded.longitude)}
        ).compute()
        embedded = _embedded.interpolate_na(dim="latitude",method="cubic")
        if type == "lam":
            embedded = _embedded.interpolate_na(dim="longitude",method="cubic")
        if "time" in embedded.dims:
            embedded = embedded.interpolate_na(dim="time", method="cubic")
    elif method == "replace":
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
        save_nc=False
):
    DATA_DIR_ROOT = "/gws/nopw/j04/kscale/"
    dt_str = f"{datetime.year:04d}{datetime.month:02d}{datetime.day:02d}T{(datetime.hour//12)*12:02d}"
    
    # should add a check that dates are in correct bounds!
    # should parse period, driving_model, nested_model here
    
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
        elif driving_model != "n1280GAL9":
            print(f"Error! Driving model {dri_mod_str} has no nested models.")
            sys.exit(1)
        elif nested_model == "CTC_km4p4_RAL3":
            DATA_DIR = os.path.join(DATA_DIR,"CTC_km4p4_RAL3P3","field.pp","apverc.pp")
            nest_mod_str = "CTC_km4p4_RAL3P3"
        elif nested_model == "Africa_km4p4_RAL3":
            DATA_DIR = os.path.join(DATA_DIR,"Africa_km4p4_RAL3P3","field.pp","apverc.pp")
            nest_mod_str = "Africa_km4p4_RAL3P3"
        elif nested_model == "SAmer_km4p4_RAL3":
            DATA_DIR = os.path.join(DATA_DIR,"SAmer_km4p4_RAL3P3","field.pp","apverc.pp")
            nest_mod_str = "SAmer_km4p4_RAL3P3"
        elif nested_model == "SEA_km4p4_RAL3":
            DATA_DIR = os.path.join(DATA_DIR,"SEA_km4p4_RAL3P3","field.pp","apverc.pp")
            nest_mod_str = "SEA_km4p4_RAL3P3"
        elif nested_model == "CTC_km4p4_CoMA9":
            DATA_DIR = os.path.join(DATA_DIR,"CTC_km4p4_CoMA9_TBv1","field.pp","apverc.pp")
            nest_mod_str = "CTC_km4p4_CoMA9_TBv1"
        elif nested_model == "Africa_km4p4_CoMA9":
            DATA_DIR = os.path.join(DATA_DIR,"Africa_km4p4_CoMA9_TBv1","field.pp","apverc.pp")
            nest_mod_str = "Africa_km4p4_CoMA9_TBv1"
        elif nested_model == "SEA_km4p4_CoMA9":
            DATA_DIR = os.path.join(DATA_DIR,"SEA_km4p4_CoMA9_TBv1","field.pp","apverc.pp")
            nest_mod_str = "SEA_km4p4_CoMA9_TBv1"
        elif nested_model == "SAmer_km4p4_CoMA9":
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
        elif nested_model == "CTC_n2560_GAL9":
            DATA_DIR = os.path.join(DATA_DIR,"CTC_N2560_GAL9")
            nest_mod_str = "CTC_n2560_GAL9"
        elif nested_model == "CTC_n2560_RAL3":
            DATA_DIR = os.path.join(DATA_DIR,"CTC_N2560_RAL3p2")
            nest_mod_str = "CTC_n2560_RAL3p2"
        elif nested_model == "CTC_km4p4_RAL3":
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
        #SAVE_DIR = "/work/scratch-pw2/dship/LoSSETT/preprocessed_kscale_data"
        SAVE_DIR = f"/gws/nopw/j04/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/{period}"
        #SAVE_DIR = f"/work/scratch-nopw2/dship/LoSSETT/preprocessed_kscale_data/{period}"
        Path(SAVE_DIR).mkdir(parents=True,exist_ok=True)
        fpath = os.path.join(SAVE_DIR,f"{nest_mod_str}.{dri_mod_str}.uvw_{dt_str}.nc")
        if not os.path.exists(fpath):
            print(f"\n\n\nSaving velocity data to NetCDF at {fpath}.")
            ds.to_netcdf(fpath) # available engines: netcdf4, h5netcdf, scipy
    
    if return_iris:
        return ds, data_iris;
    else:
        return ds;

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
    #plevs = [100,150,200,250,300,400,500,600,700,850,925,1000]
    plevs = [200]
    print("\n\n\nPreprocessing details:")
    print(
        f"\nPeriod: {period}, driving model: {driving_model}, nested_model = {nested_model}, "\
        f"grid = {grid}, date = {year:04d}-{month:02d}-{day:02d}, hour = {hour:02d}"
    )
    
    if nested_model in ["None","none","glm","global"]:
        nested_model = "glm"
        
    if grid == "native":
        ds = load_kscale_native(
            period,
            datetime,
            driving_model,
            nested_model=nested_model,
            save_nc=save_nc
        )
    elif grid == "0p5deg":
        ds_inner = load_kscale_0p5deg(
            period,
            datetime,
            driving_model,
            nested_model=nested_model,
            plevs=plevs
        )
        ds_outer = load_kscale_0p5deg(
            period,
            datetime,
            driving_model,
            nested_model="glm",
            plevs=plevs
        )
    # sys.exit(0)
    
    print("\n\nInner:\n",ds_inner)
    print("\n\nOuter:\n",ds_outer)
    ds_embed_interp = embed_inner_grid_in_global(
        ds_outer,
        ds_inner,
        method="interp"
    )
    ds_embed_replace = embed_inner_grid_in_global(
        ds_outer,
        ds_inner,
        method="replace"
    )

    import matplotlib.pyplot as plt
    plt.figure()
    ds_embed_interp.u.isel(time=0).plot()
    plt.figure()
    (ds_embed_interp-ds_outer).u.isel(time=0).plot()
    plt.figure()
    ds_embed_replace.u.isel(time=0).plot()
    plt.figure()
    (ds_embed_replace-ds_outer).u.isel(time=0).plot()
    plt.show()
    
    print("\n\n\nEND.")
