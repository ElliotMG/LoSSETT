import os
import numpy as np
from netCDF4 import Dataset
import xarray as xr
import sys
%xmode Plain

data_dir = '/gws/nopw/j04/kscale/USERS/dship/ERA5/'

ds_u = xr.open_dataset(os.path.join(data_dir, 'era5_u_component_of_wind_200501_12hourly_0p75deg.nc'))
lon = ds_u.variables['lon'][:]
lat = ds_u.variables['lat'][:]
lev = ds_u.variables['level'][:]
time = ds_u.variables['time'][:]
print('lon lat lev time loaded')

# Read u, v, and omega data
u = ds_u['u']
print('u loaded')

ds_v = xr.open_dataset(os.path.join(data_dir, 'era5_v_component_of_wind_200501_12hourly_0p75deg.nc'))
v = ds_v['v']
print('v loaded')

ds_omega = xr.open_dataset(os.path.join(data_dir, 'era5_vertical_velocity_200501_12hourly_0p75deg.nc'))
omega = ds_omega['w']  # ERA5 'w' is in Pa s**-1 !!!
print('omega loaded')
# omega = omega.chunk({"time":2})

# Convert omega to w
w = omega * -9.81 * 0.5  # rho = 0.5 for now, rho to be loaded in properly from file or calculated from temp and pressure
w.attrs["units"] = "m s**-1"

## Select length scale lmax
Nlmax = 10

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
# dR

%load_ext autoreload
%autoreload 2

# Call CalcPartitionIncrement
from CalcPartitionIncrement import CalcPartitionIncrement
dR, Nlmax, nphiinc, llx, lly = CalcPartitionIncrement(dR,Nlmax)

from CalcDRDir_2D import CalcDRDir_2D
CalcDRDir_2D(dR, Nlmax, u, v, w, nphiinc, llx, lly)