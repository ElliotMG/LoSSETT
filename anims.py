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

# Merging datasets
directory = "/gws/nopw/j04/kscale/USERS/emg/data/LoSSETT_out/channel_n2560_RAL3p2"
dates = pd.date_range(start="2016-08-01", end="2016-09-09").strftime('%Y%m%d')

file_paths = [f"{directory}/DRdir2dt_Nlmax5_kscaleRAL3n2560_0p5_DS_{date}.nc" for date in dates]
ds = xr.open_mfdataset(file_paths, combine='by_coords',engine='netcdf4')
ds

with ProgressBar():
    DR_trop_ERA = ds['LoSSET_DR'].sel(latitude=slice(-15,15)).compute()
DR_trop_ERA

DR_anim = DR_trop_ERA.sel(level=200,n_scales=3)
time = ds['time']
for i in range(len(time)):
    plt.figure(figsize=(18,3))
    ax = plt.axes(projection=ccrs.PlateCarree())

    data_at_time = DR_anim.isel(time=i)

    lon = data_at_time['longitude']
    lat = data_at_time['latitude']
    ax.set_extent([180,-180,-15,15],crs=ccrs.PlateCarree())
    plt.contourf(lon, lat, data_at_time.T, cmap='bwr', levels=np.arange(-0.5, 0.5, 5e-3),extend='both')
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

    plt.title(f'L = 220 km | DYAMOND Summer | 200hPa | 5km expl | Day: {i//8}')

    plt.savefig(f'/home/users/emg97/emgPlots/frames_LO/frame_{file_number}.png')
    print(f'Saved frame number {file_number}')
    plt.close()
