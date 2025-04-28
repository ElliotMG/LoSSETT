######## PACKAGES ###########
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
import matplotlib.cm as cm

######### READ IN DATA - COMBINE DAILY FILES ###############
directory = "/gws/nopw/j04/kscale/USERS/emg/data/LoSSETT_out/ERA5_0p5deg_3h"
dates = pd.date_range(start="2016-08-01", end="2016-09-09").strftime('%Y-%m-%d')

file_paths = [f"{directory}/DRdir2dt_Nlmax5_era5_{date}.nc" for date in dates]
ds = xr.open_mfdataset(file_paths, combine='by_coords',engine='netcdf4')

# Time mean #
# fname = '/gws/nopw/j04/kscale/USERS/emg/data/LoSSETT_out/ERA5_0p5deg_3h/DRdir2dt_Nlmax5_era5_DYAMOND_Summer_tmean.nc'
# if not os.path.exists(fname):
#     DR_tm = ds['LoSSET_DR'].mean(dim='time').compute()
#     DR_tm.to_netcdf(fname)
# DR_tm = xr.open_dataset(fname)['LoSSET_DR']

# levl = 500

# DR_map = DR_tm.sel(level=levl,n_scales=0)

# plt.figure(figsize=(18,6))
# ax = plt.axes(projection=ccrs.PlateCarree())
# lon = ds['longitude']
# lat = ds['latitude']
# # ax.set_extent([180,-180,-15,15],crs=ccrs.PlateCarree())
# plt.contourf(lon,lat,DR_map.T,cmap='bwr',levels=np.arange(-5e-3,5e-3,5e-5),extend='both')
# ax.coastlines()
# plt.title(r'$\mathcal{D}_\ell(\mathbf{u})$' + f' at {levl}hPa ' + r'(m$^2$ s$^{-3}$) for $\ell = 220$ km ERA5 | DYAMOND Summer')
# cbar=plt.colorbar(orientation='vertical', shrink=0.25, pad=0.025)
# # cbar.set_label(r'$\mathcal{D}_\ell(\mathbf{u})$ (m$^2$ s$^{-3}$)')
# ax.axhline(y=0,color='k',linewidth=0.5,linestyle='--')
# cbar.formatter = ticker.ScalarFormatter()
# cbar.formatter.set_scientific(True)
# cbar.formatter.set_powerlimits((-2, 3))
# cbar.update_ticks() 
# cbar.set_ticks([-0.5,0,0.5])

# plt.savefig(f'/home/users/emg97/emgPlots/LO_tm_DS_{levl}hPa_ERA5_l55_FULLGlobe.png',dpi=600)

##### Animation #####
DR_anim = ds['LoSSET_DR'].sel(level=200,n_scales=3)
time = ds['time']
for i in range(len(time)):
    plt.figure(figsize=(18,3))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([180,-180,-15,15],crs=ccrs.PlateCarree())
    data_at_time = DR_anim.isel(time=i)

    lon = data_at_time['longitude']
    lat = data_at_time['latitude']
    ax.set_extent([180,-180,-15,15],crs=ccrs.PlateCarree())
    plt.contourf(lon, lat, -data_at_time.T, cmap='bwr', levels=np.arange(-0.5,0.5,5e-3),extend='both')
    ax.coastlines()
    cbar=plt.colorbar(orientation='vertical', shrink=0.5, pad=0.025)
    cbar.set_label(r'$\mathcal{D}_\ell(\mathbf{u})$ (m$^2$ s$^{-3}$)')
    ax.axhline(y=0,color='k',linewidth=0.5,linestyle='--')
    cbar.formatter = ticker.ScalarFormatter()
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((-2, 3))
    cbar.update_ticks() 
    cbar.set_ticks([-0.5,0,0.5])
    
    file_number = str(i).zfill(3)

    plt.title(f'L = 220 km | DYAMOND Summer | 200hPa | ERA5 | Day: {i//8}')
    plt.savefig(f'/home/users/emg97/emgPlots/frames_LO/frame_{file_number}.png')
    print(f'frame {i} saved')
    plt.close()