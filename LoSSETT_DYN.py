import os
import numpy as np
from netCDF4 import Dataset
import xarray as xr
from scipy.ndimage import shift
from scipy.ndimage import pad



data_dir = '/storage/silver/diamet/po918217/era5/'

ds_u = xr.open_dataset(os.path.join(data_dir, 'era5_u_component_of_wind_2005_12hourly.nc'))
lon = nc.variables['longitude'][:]
lat = nc.variables['latitude'][:]
lev = nc.variables['level'][:]
print('lon lat lev loaded')

# Read u, v, and omega data
u = ds_u['u'].values
print('u loaded')

ds_v = xr.open_dataset(os.path.join(data_dir, 'era5_v_component_of_wind_2005_12hourly.nc'))
v = ds_v['v'].values
print('v loaded')

ds_omega = xr.open_dataset(os.path.join(data_dir, 'era5_vertical_velocity_2005_12hourly.nc'))
omega = ds_omega['w'].values  # ERA5 'w' is in Pa s**-1 !!!
print('omega loaded')

# Convert omega to w
w = omega * -9.81 * 0.5  # rho = 0.5 for now, rho to be loaded in properly from file or calculated from temp and pressure

Nlmax = 1

n1, n2, n3, n4 = u.shape

# Horizontal (dR) and vertical (dZ) grid step in m.
dR = abs((lon[0] - lon[1]) * 110000)
dZ = 400  # Suggests interp to 400m grid spacing required

# Horizontal size of the domain
lbox = abs((lon[-1] - lon[0]) * 110000)

# Call CalcPartitionIncrement_3D
CalcPartitionIncrement()

Vx = u
Vy = v
Vz = w

Xtrc = np.arange(0.01, dR * Vz.shape[1] + 0.01, dR)
Ztrc = np.arange(0.01, dZ * Vz.shape[0] + 0.01, dZ)
ismethod = 2
n1, n2 = Xtrc.shape
nr, nz = Xtrc.shape

Ndebut = 1
DeltaN = 0

Nfin = Ndebut + DeltaN

# Call CalcDRDir_2D
CalcDRDir_2D()

# DRdir2dt = spsol * philsmooth
# DRdir = np.reshape(DRdir2dt, (n1, n2, n3, Nls, nt))
# lDRdir = lsingd

# # Write output to file
# with Dataset('DRdir_era5.nc', 'w', format='NETCDF4') as nc:
#     # Define dimensions and variables here
#     # Example:
#     # nc.createDimension('lon', len(lon))
#     # nc.createDimension('lat', len(lat))
#     # nc.createDimension('lev', len(lev))
#     # nc.createDimension('time', n4)
    
#     # var = nc.createVariable('DRdir', 'f4', ('time', 'lev', 'lat', 'lon'))
#     # var[:] = DRdir
#     pass

