import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy as cpy

RADIUS_EARTH = 6371000 # Earth radius in meters

def calculate_great_circle_distance(lat1, lon1, lat2, lon2, R=RADIUS_EARTH):
    """
    Calculates the great-circle distance between two points on a spherical surface
    using the Haversine formula.

    Inputs:
    lat1, lon1, lat2, lon2 (assumed to be in degrees)

    Outputs:
    r : great circle distance between (lat1,lon1) and (lat2,lon2).
    """
    
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Compute required differences
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    # Compute great circle distance
    a = np.sin(delta_lat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    r = R * c
    
    return r;


def calculate_initial_bearing(lat1, lon1, lat2, lon2, return_degrees=False):
    """
    Calculates the clockwise angle between a great circle path and a line of 
    constant longitude (meridian).
    Can return the bearing in either degrees or radians.

    Inputs:
    lat1, lon1, lat2, lon2 (assumed to be in degrees)

    Outputs:
    initial_bearing : clockwise angle at point (lat1,lon1) between North and the great circle path connecting 
                      (lat1,lon1) to (lat2,lon2). Can return angle in either radians (default) or degrees.
    """
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Compute required differences
    delta_lon = lon2 - lon1

    # Compute initial_bearing in radians clockwise from North
    y = np.sin(delta_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    initial_bearing = np.arctan2(y, x)
    
    return initial_bearing;

def calculate_great_circle_distance_and_bearing(lat1, lon1, lat2, lon2, R=RADIUS_EARTH, return_degrees=False):
    """
    Calculates:
      1. the great-circle distance between two points on a spherical surface
    using the Haversine formula;
      2. the clockwise angle between a great circle path and a line of 
    constant longitude (meridian).

    Inputs:
    lat1, lon1, lat2, lon2 (assumed to be in degrees)

    Outputs:
    r               : great circle distance between (lat1,lon1) and (lat2,lon2).
    initial_bearing : clockwise angle at point (lat1,lon1) between North and the great circle path connecting 
                      (lat1,lon1) to (lat2,lon2). Can return angle in either radians (default) or degrees.
    """
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Compute required differences
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    # Compute great circle distance
    a = np.sin(delta_lat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    r = R * c

    # Compute initial_bearing in radians clockwise from North
    y = np.sin(delta_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    initial_bearing = np.arctan2(y, x)
    
    return r, initial_bearing;

if __name__ == "__main__":
    # 0p5deg
    lon_step = 0.5
    lat_step = 0.5
    # n320
    #lon_step = 0.5625
    #lat_step = 0.375
    # n640
    #lon_step = 0.28125
    #lat_step = 0.1875
    # n1280
    #lon_step = 0.140625
    #lat_step = 0.09375
    # n2560
    #lon_step = 0.0703125
    #lat_step = 0.046875
    lons = np.arange(-180.,180.,lon_step)
    lats = np.arange(-90.,90.,lat_step)
    lats = np.append(lats, lats[-1] + lat_step)
    lons, lats = np.meshgrid(lons, lats)

    lon_origin = 25.
    lat_origin = np.array([0.,10.,20.,30.,40.,50.,60.,70.,80.])

    distances = []
    bearings = []
    for _lat_origin in lat_origin:
        _distances, _bearings = np.vectorize(
            calculate_great_circle_distance_and_bearing
        )(
            _lat_origin, lon_origin, lats, lons
        )
        _distances = xr.DataArray(
            _distances,
            dims = ["latitude","longitude"],
            coords = {
                "latitude": np.unique(lats),
                "longitude": np.unique(lons)
            }
        )
        _bearings = xr.DataArray(
            _bearings,
            dims = ["latitude","longitude"],
            coords = {
                "latitude": np.unique(lats),
                "longitude": np.unique(lons)
            }
        )
        distances.append(_distances)
        bearings.append(_bearings)
    #endfor
    distances = xr.concat(
        distances,
        dim="origin_latitude"
    ).assign_coords({"origin_latitude":lat_origin})
    bearings = xr.concat(
        bearings,
        dim="origin_latitude"
    ).assign_coords({"origin_latitude":lat_origin})

    print(distances)
    sys.exit(1)

    delta_r = 220.  # km
    delta_r *= 1000 # convert to m
    r1 = 5000. # km
    r1 *= 1000 # convert to m
    r2 = 10000. # km
    r2 *= 1000 # convert to m

    dist_mask1 = np.abs(distances - r1) < delta_r /2
    dist_mask1 = dist_mask1.astype("float")
    dist_mask1[dist_mask1==0] = np.nan

    dist_mask2 = np.abs(distances - r2) < delta_r /2
    dist_mask2 = dist_mask2.astype("float")
    dist_mask2[dist_mask2==0] = np.nan

    fig, axes = plt.subplots(
        nrows=1, ncols=2,
        subplot_kw={"projection":cpy.crs.PlateCarree()},
        figsize=(20,10)
    )
    pc_r = axes[0].pcolormesh(lons,lats,distances/1000,cmap="magma_r")
    axes[0].plot(lon_origin, lat_origin, marker="x", color="C2")
    axes[0].pcolormesh(lons,lats,dist_mask1,cmap="binary")
    axes[0].pcolormesh(lons,lats,dist_mask2,cmap="binary")
    plt.colorbar(
        pc_r, orientation="horizontal",
        label=f"great circle distance from {lat_origin:.4g}N, {lon_origin:.4g}E [km]"
    )
    pc_b = axes[1].pcolormesh(lons,lats,bearings,cmap="twilight_shifted")
    axes[1].plot(lon_origin, lat_origin, marker="x", color="C2")
    plt.colorbar(
        pc_b, orientation="horizontal",
        label=f"initial bearing along great circle path\nfrom {lat_origin:.4g}N, {lon_origin:.4g}E [rad.]"
    )
    for ax in axes:
        ax.coastlines()
    plt.show()

    print("\n\n\nEND.")

    
