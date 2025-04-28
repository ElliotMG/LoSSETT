import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import figure
# import metpy.calc as mpcalc
import matplotlib.ticker as ticker
import xarray as xr
import os
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
# from mpl_toolkits.basemap import Basemap
import pandas as pd
from dask.diagnostics import ProgressBar
from tqdm import tqdm

# ds_CTC5RAL = xr.open_dataset('/gws/nopw/j04/kscale/USERS/emg/data/DYAMOND_Summer/Symm/profile_200/channel_RAL3_n2560/profile_RAL3n2560_200hPa_DS_CTC.nc')
# u_CTC5RAL = ds_CTC5RAL['x_wind']
# v_CTC5RAL = ds_CTC5RAL['y_wind']

# # KE
# ke_CTC5RAL = 0.5 * (u_CTC5RAL**2 + v_CTC5RAL**2)

# ds_CTC5RAL['kinetic_energy'] = ke_CTC5RAL
# ds_CTC5RAL['kinetic_energy'].attrs = {
#     'long_name': 'Kinetic Energy',
#     'units': 'm^2 s^-2',
# }

# output_filename = '/gws/nopw/j04/kscale/USERS/emg/data/DYAMOND_Summer/Symm/profile_200/channel_RAL3_n2560/RAL3n2560_200hPa_KE_DS_CTC.nc'

# # Save to file
# ds_CTC5RAL.to_netcdf(output_filename)

# print(f"Kinetic energy dataset saved to {output_filename}")

# R = 6371000
# Lat = np.deg2rad(ds_CTC5RAL['latitude'])
# Lon = np.deg2rad(ds_CTC5RAL['longitude'])
# lat_grid, lon_grid = np.meshgrid(ds_CTC5RAL['latitude'].values, ds_CTC5RAL['longitude'].values, indexing="ij")
# lat_grid = np.deg2rad(lat_grid)
# lon_grid = np.deg2rad(lon_grid)

# dlon = np.gradient(lon_grid, axis=1)  # Spacing along longitude
# dlat = np.gradient(lat_grid, axis=0)  # Spacing along latitude

# dx = R * np.cos(lat_grid) * dlon  # Zonal spacing in meters
# dy = R * dlat                    # Meridional spacing in meters
# print('calculating du_dx using centred differences....')
# du_dx = u_CTC5RAL.differentiate('longitude') / dx
# print('du_dx calculated')
# print('calculating dv_dy using centred differences....')
# dv_dy = v_CTC5RAL.differentiate('latitude') / dy
# print('dv_dy calculated')
# print('calculating div')
# div_CTC5RAL = du_dx + dv_dy
# print('div calculated. Now writing to file...')
# div_CTC5RAL.name = "divergence"
# div_CTC5RAL.attrs["units"] = "s^-1"  
# div_CTC5RAL.to_netcdf('/home/users/emg97/emgScripts/LoSSETT/out_nc/divergence_CTC5RAL_0p5deg_3h_DS.nc')
# print('divergence written to file')

ds_CTC5GAL = xr.open_dataset('/gws/nopw/j04/kscale/USERS/emg/data/DYAMOND_Summer/Symm/profile_200/channel_GAL9_n2560/profile_GAL9n2560_200hPa_DS_CTC.nc')
u_CTC5GAL = ds_CTC5GAL['x_wind']
v_CTC5GAL = ds_CTC5GAL['y_wind']

ke_CTC5GAL = 0.5 * (u_CTC5GAL**2 + v_CTC5GAL**2)

ds_CTC5GAL['kinetic_energy'] = ke_CTC5GAL
ds_CTC5GAL['kinetic_energy'].attrs = {
    'long_name': 'Kinetic Energy',
    'units': 'm^2 s^-2',
}

output_filenameG = '/gws/nopw/j04/kscale/USERS/emg/data/DYAMOND_Summer/Symm/profile_200/channel_GAL9_n2560/GAL9n2560_200hPa_KE_DS_CTC.nc'

# Save to file
ds_CTC5GAL.to_netcdf(output_filenameG)

print(f"Kinetic energy dataset saved to {output_filenameG}")

# R = 6371000
# Lat = np.deg2rad(ds_CTC5GAL['latitude'])
# Lon = np.deg2rad(ds_CTC5GAL['longitude'])
# lat_grid, lon_grid = np.meshgrid(ds_CTC5GAL['latitude'].values, ds_CTC5GAL['longitude'].values, indexing="ij")
# lat_grid = np.deg2rad(lat_grid)
# lon_grid = np.deg2rad(lon_grid)

# dlon = np.gradient(lon_grid, axis=1)  # Spacing along longitude
# dlat = np.gradient(lat_grid, axis=0)  # Spacing along latitude

# dx = R * np.cos(lat_grid) * dlon  # Zonal spacing in meters
# dy = R * dlat                    # Meridional spacing in meters
# print('calculating du_dx using centred differences....')
# du_dx = u_CTC5GAL.differentiate('longitude') / dx
# print('du_dx calculated')
# print('calculating dv_dy using centred differences....')
# dv_dy = v_CTC5GAL.differentiate('latitude') / dy
# print('dv_dy calculated')
# print('calculating div')
# div_CTC5GAL = du_dx + dv_dy
# print('div calculated. Now writing to file...')
# div_CTC5GAL.name = "divergence"
# div_CTC5GAL.attrs["units"] = "s^-1"  
# div_CTC5GAL.to_netcdf('/home/users/emg97/emgScripts/LoSSETT/out_nc/divergence_CTC5GAL_0p5deg_3h_DS.nc')
# print('divergence written to file')

# # ## Read in winds to make divergence
# ds_u_ERA5 = xr.open_dataset('/gws/nopw/j04/kscale/USERS/dship/ERA5/3hourly/era5_u_component_of_wind_201608_3h_0p5deg.nc')
# u_ERA5 = ds_u_ERA5['u']
# ds_v_ERA5 = xr.open_dataset('/gws/nopw/j04/kscale/USERS/dship/ERA5/3hourly/era5_v_component_of_wind_201608_3h_0p5deg.nc')
# v_ERA5 = ds_v_ERA5['v']

# ke_ERA5 = 0.5 * (u_ERA5**2 + v_ERA5**2)

# ds_u_ERA5['kinetic_energy'] = ke_ERA5
# ds_u_ERA5['kinetic_energy'].attrs = {
#     'long_name': 'Kinetic Energy',
#     'units': 'm^2 s^-2',
# }

# output_filenameE = '/gws/nopw/j04/kscale/USERS/emg/data/DYAMOND_Summer/Symm/profile_200/ERA5_200hPa_KE_DS_CTC.nc'

# # Save to file
# ds_u_ERA5.to_netcdf(output_filenameE)

# print(f"Kinetic energy dataset saved to {output_filenameE}")

# R = 6371000
# Lat = np.deg2rad(ds_u_ERA5['lat'])
# Lon = np.deg2rad(ds_u_ERA5['lon'])
# lat_grid, lon_grid = np.meshgrid(ds_u_ERA5['lat'].values, ds_u_ERA5['lon'].values, indexing="ij")
# lat_grid = np.deg2rad(lat_grid)
# lon_grid = np.deg2rad(lon_grid)

# dlon = np.gradient(lon_grid, axis=1)  # Spacing along longitude
# dlat = np.gradient(lat_grid, axis=0)  # Spacing along latitude

# dx = R * np.cos(lat_grid) * dlon  # Zonal spacing in meters
# dy = R * dlat                    # Meridional spacing in meters
# print('calculating du_dx using centred differences....')
# du_dx = u_ERA5.differentiate('lon') / dx
# print('du_dx calculated')
# print('calculating dv_dy using centred differences....')
# dv_dy = v_ERA5.differentiate('lat') / dy
# print('dv_dy calculated')
# print('calculating div')
# div_ERA5 = du_dx + dv_dy
# print('div calculated. Now writing to file...')
# div_ERA5.name = "divergence"
# div_ERA5.attrs["units"] = "s^-1"  
# div_ERA5.to_netcdf('/home/users/emg97/emgScripts/LoSSETT/out_nc/divergence_ERA5_0p5deg_3h_DS.nc')
# print('divergence written to file')
# ds = xr.open_dataset('/gws/nopw/j04/kscale/USERS/dship/ERA5/hourly/era5_vertical_velocity_201609_hourly.nc')
# w = ds['w']

# resample_obj = w.resample(valid_time='3h')
# n_steps = len(resample_obj.groups)
# resampled_chunks = []
# for i, (time, group) in tqdm(enumerate(resample_obj), total=n_steps, desc="Resampling w Progress"):
#    resampled_chunk = group.mean(dim='valid_time')
#    resampled_chunks.append(resampled_chunk)
# w_3h = xr.concat(resampled_chunks, dim='valid_time')
# w3h_0p5deg = w_3h.coarsen(latitude=2, longitude=2, boundary="trim").mean()

# w3h_0p5deg = xr.DataArray(w3h_0p5deg, dims=['valid_time', 'pressure_level', 'latitude', 'longitude'],
#                           coords={'time':w3h_0p5deg.valid_time, 'level':w3h_0p5deg.pressure_level,\
#                                    'latitude':w3h_0p5deg.latitude, 'longitude':w3h_0p5deg.longitude}, name='w')
# w3h_0p5deg.to_netcdf('/gws/nopw/j04/kscale/USERS/dship/ERA5/era5_vertical_velocity_201609_3h_0p5deg.nc',compute=True)
# print('w3h0p5deg written to NetCDF file')

# dsv = xr.open_dataset('/gws/nopw/j04/kscale/USERS/dship/ERA5/3hourly/era5_temperature_201608_3hourly.nc')
# v = dsv['t']

# resample_obj = v.resample(valid_time='3h')
# n_steps = len(resample_obj.groups)
# resampled_chunks = []
# for i, (time, group) in tqdm(enumerate(resample_obj), total=n_steps, desc="Resampling T Progress"):
#     resampled_chunk = group.mean(dim='valid_time')
#     resampled_chunks.append(resampled_chunk)
# v_3h = xr.concat(resampled_chunks, dim='valid_time')
# v3h_0p5deg = v_3h.coarsen(latitude=2, longitude=2, boundary="trim").mean()

# v3h_0p5deg = xr.DataArray(v3h_0p5deg, dims=['valid_time', 'pressure_level', 'latitude', 'longitude'],
#                            coords={'time':v3h_0p5deg.valid_time, 'level':v3h_0p5deg.pressure_level,\
#                                     'latitude':v3h_0p5deg.latitude, 'longitude':v3h_0p5deg.longitude}, name='T')
# v3h_0p5deg.to_netcdf('/gws/nopw/j04/kscale/USERS/dship/ERA5/3hourly/era5_temperature_201608_3h_0p5deg.nc',compute=True)
# print('T3h0p5deg written to NetCDF file')
