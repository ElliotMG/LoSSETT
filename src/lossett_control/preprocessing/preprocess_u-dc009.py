#!/usr/bin/env python3
import os
import sys
import xarray as xr
import numpy as np
import datetime as dt
import pandas as pd

import CalcScaleIncrements as csi

radius_earth = 6.371e6 # radius of Earth in m
deg_to_m = 110000.0 # conversion of latitudinal degrees to m

if __name__ == "__main__":
    DATA_DIR = "/work/scratch-pw2/bakera/n1280_data_for_upscale/u-dc009/"
    OUT_DIR = "/work/scratch-pw2/dship/upscale/LoSSETT_out/"
    startdate = "1980-09-01"
    start = dt.datetime.strptime(startdate, "%Y-%m-%d")
    print(startdate, start)

    # simulation specification
    simid = "u-dc009"
    tsteps_per_day = 4
    lon_bound_field = np.nan
    lat_bound_field = np.nan

    # day of simulation
    day = 2

    # calculation specification
    max_r_deg = 20.0
    tsteps = 4
    tchunks = 4
    prec = 1e-10
    
    # open data
    # u,v files: every 2 days; 3-hourly; 7 pressure levels (no 300hPa)
    # w files: every 10 days; 6-hourly; 8 pressure levels
    ds_u_3D = xr.open_mfdataset(
        [
            os.path.join(
                DATA_DIR,
                "x_wind",
                f"u-dc009_x_wind_dc009a.p7{dt.datetime.strftime(start,'%Y%m%d')}.nc"
            ),
            os.path.join(
                DATA_DIR,
                "y_wind",
                f"u-dc009_y_wind_dc009a.p7{dt.datetime.strftime(start,'%Y%m%d')}.nc"
            ),
            os.path.join(
                DATA_DIR,
                "upward_air_velocity",
                f"u-dc009_upward_air_velocity_dc009a.pc{dt.datetime.strftime(start,'%Y%m%d')}.nc"
            )
        ],
        mask_and_scale=True,
        join="inner"
    )
    print(ds_u_3D)
    
    # subset single day
    ds_u_3D = ds_u_3D.isel(time = slice((day-1)*tsteps_per_day,day*tsteps_per_day))
    print(ds_u_3D)

    # get start date + time
    #start = pd.Timestamp(ds_u_3D.time.values[0]).to_pydatetime()
    start = ds_u_3D.time.values[0]
    startdate = f"{start.year:04d}-{start.month:02d}-{start.day:02d}"

    print(f"\n\n\nCalculating DR indicator for {startdate}")

    # subset time; chunk time
    ds_u_3D = ds_u_3D.rename(
        {
            "upward_air_velocity":"w",
            "x_wind":"u",
            "y_wind":"v"
        }
    ).isel(time=slice(0,tsteps)).chunk(chunks={"time":tchunks})
    print("\nInput data:\n",ds_u_3D)

    # setup x-y coords, bounds, grid spacings
    lon = ds_u_3D.longitude
    lon_bounds = np.array([lon[0].values,lon[-1].values])
    lat = ds_u_3D.latitude
    lat_bounds = np.array([lat[0].values,lat[-1].values])
    
    delta_lon = np.max(np.diff(lon))
    delta_lat = np.max(np.diff(lat))
    
    lon_m = lon*deg_to_m
    lon_m_bounds = np.array([lon_m[0].values,lon_m[-1].values])
    lat_m = lat*deg_to_m
    lat_m_bounds = np.array([lat_m[0].values,lat_m[-1].values])
    
    delta_lon_m = np.max(np.diff(lon_m))
    delta_lat_m = np.max(np.diff(lat_m))

    # add lon_m, lat_m as coords to ds_u_3D
    ds_u_3D = ds_u_3D.assign_coords({"longitude":lon_m.values,"latitude":lat_m.values})

    print(ds_u_3D)

    scale_incs = csi.calc_scale_increments(lon_m,lat_m,max_r_deg*deg_to_m,verbose=True)
    r = scale_incs.r

    # compute delta u cubed integrated over angles for all |r|
    print(f"\n\n\nCalculating angular integral for r={r[0].values/1000:.4g} km to r={r[-1].values/1000:.4g}")
    delta_u_cubed = csi.calc_increment_integrand(
        ds_u_3D, scale_incs, csi.calc_delta_u_cubed, delta_lon_m, delta_lat_m,
        xdim="longitude", ydim="latitude", xbounds=lon_m_bounds, ybounds=lat_m_bounds,
        x_bound_field=lon_bound_field, y_bound_field=lat_bound_field, precision=prec,
        verbose=True
    )

    # calculate scale-space integral given integrand, length scales, geometry specification
    integrand = delta_u_cubed.assign_coords(
        {"longitude":lon.values,"latitude":lat.values}
    ).transpose("r","time","longitude","latitude","pressure")
    print(integrand)
    r = integrand.r
    # specify length scales -- should probably be an if statement here to allow the user
    # to specify scales if desired.
    length_scales = r.values[1:len(r)//2]
    print(length_scales)

    fpath = os.path.join(OUT_DIR, f"DR_test_{simid}_Nl_{len(length_scales)}_{startdate}_t0-{tsteps-1}.nc")
    DR_indicator = csi.calc_scale_space_integral(
        integrand, fpath, name="DR_indicator", length_scales=length_scales, weighting="2D"
    ) # should add options for kernel specification
    sys.exit(0)
