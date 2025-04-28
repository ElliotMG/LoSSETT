#!/usr/bin/env python3
import os
import numpy as np
import xarray as xr
import sys
import datetime as dt

# function to sort out dims
def rename_dims(ds):    
    try:
        ds = ds.rename({'lon':'longitude'})
    except:
        pass
    try:
        ds = ds.rename({'lat':'latitude'})
    except:
        pass
    try:
        ds = ds.rename({'pressure_level':'level'})
    except:
        pass
    try:
        ds = ds.rename({'valid_time':'time'})
    except:
        pass
        
    return ds

year=2016
month=8

data_dir = '/gws/nopw/j04/kscale/USERS/dship/ERA5/3hourly/'
ds_u    = xr.open_dataset(os.path.join(data_dir, f'era5_u_DS_3h_0p5deg.nc'))
ds_v = xr.open_dataset(os.path.join(data_dir, f'era5_v_DS_3h_0p5deg.nc'))
ds_omega = xr.open_dataset(os.path.join(data_dir, f'era5_vertical_velocity_DS_3h_0p5deg.nc'))
ds_T = xr.open_dataset(os.path.join(data_dir, f'era5_T_DS_3h_0p5deg.nc'))
# ds_q = xr.open_dataset(os.path.join(data_dir,f'era5_specific_humidity_{year:04d}{month:02d}_3h_0p5deg.nc'))

ds_u = rename_dims(ds_u)
ds_v = rename_dims(ds_v)
ds_omega = rename_dims(ds_omega)
ds_T = rename_dims(ds_T)
# ds_q = rename_dims(ds_q)

lon     = ds_u.variables['longitude'][:]
lat     = ds_u.variables['latitude'][:]
lev     = ds_u.variables['level'][:]
time    = ds_u.variables['time'][:]
# print(time)
# sys.exit()
p = xr.DataArray(
    dims = {'level':lev.data},
    data = lev.data,
    attrs = {'units': 'hPa'}
)

# fix time coordinates
fix_time_coord = True
if fix_time_coord:
    start = dt.datetime(year,month,1,3)
    times = np.array([start + i*dt.timedelta(hours=3) for i in range(len(time))])
    
    ds_u = ds_u.assign_coords({"time": times})
    ds_v = ds_v.assign_coords({"time": times})
    ds_omega = ds_omega.assign_coords({"time": times})

start_date = dt.datetime(2016, 8, 1)
end_date = dt.datetime(2016, 8, 2)
diro = '/gws/nopw/j04/kscale/USERS/emg/data/LoSSETT_out/ERA5/'
current_date = start_date
while current_date <= end_date:
    date = current_date.strftime('%Y-%m-%d')
    fname_str = f'era5_{date}'
    # Read u, v, and omega data
    u = ds_u['u']
    u = u.sel(time=date)
    v = ds_v['v']
    v = v.sel(time=date)
    omega = ds_omega['w']  # ERA5 'w' is in Pa s**-1 !!!
    omega = omega.sel(time=date)
    # q = ds_q['q']
    # q = q.sel(time=date)
    temp = ds_T['t']
    temp = temp.sel(time=date)
    # Calculate density
    rgas = 287.05
    g    = 9.81
    rho = ((p*100) / (rgas * temp))
    rho = rho.transpose('time','level','latitude','longitude')
    
    current_date += dt.timedelta(days=1)


# Calculate w from omega for ERA5
w = np.divide(-omega,(rho*g))
print('u,v,w calculated from ERA5 data')
# sys.exit(0)
#==============================================================================

###############################################################################
### Running the LoSSETT functions after having read in and prepped the data ###
###############################################################################

## Select length scale lmax
Nlmax = 20

# Dimensions
nt = len(time)
nz = len(lev)
ny = len(lat)
nx = len(lon)

# Horizontal (dR) and vertical (dZ) grid step in m.
dR = abs((lon[0] - lon[1]) * 110000)
dZ = 400  # Suggests interp to 400m grid spacing required
# Horizontal size of the domain
lbox = abs((lon[-1] - lon[0]) * 110000)

# Call CalcPartitionIncrement
sys.path.append('/home/users/emg97/emgScripts/LoSSETT')
from CalcPartitionIncrement import CalcPartitionIncrement
dR, Nlmax, nphiinc, llx, lly, philsmooth, Nls = CalcPartitionIncrement(dR,Nlmax)
#Call CalcDRDir_2D
from CalcDRDir_2D import CalcDRDir_2D
CalcDRDir_2D(dR, Nlmax, u, v, w, nphiinc, llx, lly, philsmooth, Nls, fname_str, diro, verbose=True)

# print(f'Processing complete for day {day}')