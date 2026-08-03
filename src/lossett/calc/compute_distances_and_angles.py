import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy as cpy

RADIUS_EARTH = 6371000 # Earth radius in metres

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
    Calculates the clockwise angle between a great circle path between two points 
    and the line of constant longitude (meridian) passing through the initial point.
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

def calculate_final_bearing(lat1, lon1, lat2, lon2, return_degrees=False):
    """
    Calculates the clockwise angle between a great circle path between two points 
    and the line of constant longitude (meridian) passing through the final point.
    Can return the bearing in either degrees or radians.

    Inputs:
    lat1, lon1, lat2, lon2 (assumed to be in degrees)

    Outputs:
    final_bearing : clockwise angle at point (lat2,lon2) between North and the great circle path connecting 
                      (lat1,lon1) to (lat2,lon2). Can return angle in either radians (default) or degrees.
    """
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Compute required differences
    delta_lon = lon1 - lon2

    # Compute final_bearing in radians clockwise from North
    y = np.sin(delta_lon) * np.cos(lat1)
    x = np.cos(lat2) * np.sin(lat1) - np.sin(lat2) * np.cos(lat1) * np.cos(delta_lon)
    final_bearing = np.arctan2(-y, -x)
    
    return final_bearing;

def calculate_great_circle_distance_and_bearing(lat1, lon1, lat2, lon2, R=RADIUS_EARTH, return_degrees=False):
    """
    Calculates:
      1. the great-circle distance between two points on a spherical surface
         using the Haversine formula;
      2. the clockwise angle (bearing) between the great circle path and a line of 
         constant longitude (meridian) passing through the initial point;
      3. the clockwise angle (bearing) between the great-circle path and a line of
         constant longitude (meridian) passing through the final point.

    Inputs:
    lat1, lon1, lat2, lon2 (assumed to be in degrees)

    Outputs:
    r               : great circle distance between (lat1,lon1) and (lat2,lon2).
    initial_bearing : clockwise angle at point (lat1,lon1) between North and the 
                      great circle path connecting (lat1,lon1) to (lat2,lon2). Can return
                      angle in either radians (default) or degrees.
    final_bearing   : clockwise angle at point (lat2,lon2) between North and the 
                      great circle path connecting (lat1,lon1) to (lat2,lon2). Can return
                      angle in either radians (default) or degrees.
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

    # Compute final_bearing in radians clockwise from North
    y = np.sin(-delta_lon) * np.cos(lat1)
    x = np.cos(lat2) * np.sin(lat1) - np.sin(lat2) * np.cos(lat1) * np.cos(-delta_lon)
    final_bearing = np.arctan2(-y, -x)
    
    return r, initial_bearing, final_bearing;

def calculate_tangent_and_normal_components(u, v, bearing):
    u_tangent = u*np.sin(bearing) + v*np.cos(bearing)
    u_normal = u*np.cos(bearing) - v*np.sin(bearing)
    return u_tangent, u_normal;

if __name__ == "__main__":
    grid =  "n320"
    force = False
    #force = True
    if grid == "0p5deg":
        lon_step = 0.5
        lat_step = 0.5
    elif grid == "n320":
        lon_step = 0.5625
        lat_step = 0.375
    elif grid == "n640":
        lon_step = 0.28125
        lat_step = 0.1875
    elif grid == "n1280":
        lon_step = 0.140625
        lat_step = 0.09375
    elif grid == "n2560":
        lon_step = 0.0703125
        lat_step = 0.046875
    #endif

    # construct grid
    lons = np.arange(-180.,180.,lon_step)
    lats = np.arange(-90.,90.,lat_step)
    lats = np.append(lats, lats[-1] + lat_step)
    lons, lats = np.meshgrid(lons, lats)

    ds_n1280 = xr.open_dataset(
        "/gws/ssde/j25b/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/"\
        "DYAMOND_SUMMER/glm.n1280_GAL9.uvw_20160815T00.nc"
    )

    save_path = "/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/"
    
    lon_attrs = ds_n1280.longitude.attrs
    ds_n1280.coords["longitude"] = (ds_n1280.coords["longitude"] + 180) % 360 - 180
    ds_n1280 = ds_n1280.sortby(ds_n1280.longitude)
    ds_n1280 = ds_n1280.isel(time=0).sel(pressure=200)
    print(ds_n1280)

    u_n1280 = ds_n1280.u
    v_n1280 = ds_n1280.v
    
    lon_origin = 25.
    lat_origin = np.array([30.])
    #lat_origin = np.array([0.,10.,20.,30.,40.,50.,60.,70.,80.])

    dist_fpath = os.path.join(save_path, f"distances_and_bearings_{grid}.nc")
    
    if not os.path.exists(dist_fpath) or force:

        distances = []
        init_bearings = []
        final_bearings = []
        for _lat_origin in lat_origin:
            _distances, _init_bearings, _fin_bearings = np.vectorize(
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
                },
                name = "great_circle_distance"
            )
            _init_bearings = xr.DataArray(
                _init_bearings,
                dims = ["latitude","longitude"],
                coords = {
                    "latitude": np.unique(lats),
                    "longitude": np.unique(lons)
                },
                name = "initial_bearing"
            )
            _fin_bearings = xr.DataArray(
                _fin_bearings,
                dims = ["latitude","longitude"],
                coords = {
                    "latitude": np.unique(lats),
                    "longitude": np.unique(lons)
                },
                name = "final_bearing"
            )
            distances.append(_distances)
            init_bearings.append(_init_bearings)
            final_bearings.append(_fin_bearings)
        #endfor
        distances = xr.concat(
            distances,
            dim="origin_latitude"
        ).assign_coords({"origin_latitude":lat_origin})
        init_bearings = xr.concat(
            init_bearings,
            dim="origin_latitude"
        ).assign_coords({"origin_latitude":lat_origin})
        final_bearings = xr.concat(
            final_bearings,
            dim="origin_latitude"
        ).assign_coords({"origin_latitude":lat_origin})

        distances = distances.squeeze()
        init_bearings = init_bearings.squeeze()
        final_bearings = final_bearings.squeeze()
        
        ds_geom = xr.merge([distances, init_bearings, final_bearings])
        ds_geom.to_netcdf(dist_fpath)
    #endif
    
    ds_geom = xr.open_dataset(dist_fpath)
    distances = ds_geom.great_circle_distance
    init_bearings = ds_geom.initial_bearing
    final_bearings = ds_geom.final_bearing

    ### PLOT DISTANCES AND ANGLES
    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(20,15),
        subplot_kw={"projection":cpy.crs.PlateCarree()}
    )
    ax = axes[0,0]
    pc1 = ax.pcolormesh(
        distances.longitude, distances.latitude, distances
    )
    plt.colorbar(pc1)

    ax = axes[0,1]
    pc2 = ax.pcolormesh(
        distances.longitude, distances.latitude, init_bearings,
        cmap="twilight_shifted"
    )
    plt.colorbar(pc2)

    ax = axes[1,0]
    pc3 = ax.pcolormesh(
        distances.longitude, distances.latitude, final_bearings,
        cmap="twilight_shifted"
    )
    plt.colorbar(pc3)

    ax = axes[1,1]
    pc4 = ax.pcolormesh(
        distances.longitude, distances.latitude, final_bearings - init_bearings,
        cmap="twilight_shifted"
    )
    ax.plot([25, -100], [30, -30], color="k", transform=cpy.crs.Geodetic())
    ax.plot([25, 170], [30, 30], color="k", transform=cpy.crs.Geodetic())
    plt.colorbar(pc4)

    for ax in axes.flatten():
        ax.coastlines()
        ax.grid()

    #plt.show()

    delta_r = 220.  # km
    delta_r *= 1000 # convert to m
    r1 = 1000. # km
    r1 *= 1000 # convert to m
    r2 = 5000. # km
    r2 *= 1000 # convert to m
    r3 = 10000. # km
    r3 *= 1000 # convert to m

    # interpolate to coarser grid for testing
    u_coarse = u_n1280.interp(latitude=distances.latitude, longitude=distances.longitude)
    v_coarse = u_n1280.interp(latitude=distances.latitude, longitude=distances.longitude)

    u_origin = u_coarse.sel(
        latitude=distances.origin_latitude,
        longitude=lon_origin,
        method="nearest"
    )
    v_origin = v_coarse.sel(
        latitude=distances.origin_latitude,
        longitude=lon_origin,
        method="nearest"
    )

    u_tangent_init, u_normal_init = calculate_tangent_and_normal_components(
        u_origin, v_origin, init_bearings
    )

    u_tangent_final, u_normal_final = calculate_tangent_and_normal_components(
        u_coarse, v_coarse, final_bearings
    )

    # exact increments
    du_tangent = u_tangent_final - u_tangent_init
    du_normal = u_normal_final - u_normal_init
    du_sq = du_tangent**2 + du_normal**2
    du_cubed = du_tangent * du_sq

    # approximate increments
    du = u_coarse-u_origin
    dv = v_coarse-v_origin
    du_sq_approx = du**2 + dv**2
    du_dot_rhat = du*np.sin(init_bearings) + dv*np.cos(init_bearings)
    du_cubed_approx = du_dot_rhat * du_sq_approx

    dist_mask1 = xr.where(np.abs(distances - r1) < delta_r/2, 1., np.nan)
    dist_mask2 = xr.where(np.abs(distances - r2) < delta_r/2, 1., np.nan)
    dist_mask3 = xr.where(np.abs(distances - r3) < delta_r/2, 1., np.nan)
    u_mask1 = xr.where(np.abs(distances - r1) < delta_r/2, u_coarse, np.nan)
    u_mask2 = xr.where(np.abs(distances - r2) < delta_r/2, u_coarse, np.nan)
    u_mask3 = xr.where(np.abs(distances - r3) < delta_r/2, u_coarse, np.nan)
    du_mask1 = xr.where(np.abs(distances - r1) < delta_r/2, du, np.nan)
    du_mask2 = xr.where(np.abs(distances - r2) < delta_r/2, du, np.nan)
    du_mask3 = xr.where(np.abs(distances - r3) < delta_r/2, du, np.nan)
    dv_mask1 = xr.where(np.abs(distances - r1) < delta_r/2, dv, np.nan)
    dv_mask2 = xr.where(np.abs(distances - r2) < delta_r/2, dv, np.nan)
    dv_mask3 = xr.where(np.abs(distances - r3) < delta_r/2, dv, np.nan)
    du_sq_approx_mask1 = xr.where(np.abs(distances - r1) < delta_r/2, du_sq_approx, np.nan)
    du_sq_approx_mask2 = xr.where(np.abs(distances - r2) < delta_r/2, du_sq_approx, np.nan)
    du_sq_approx_mask3 = xr.where(np.abs(distances - r3) < delta_r/2, du_sq_approx, np.nan)
    du_dot_rhat_mask1 = xr.where(np.abs(distances - r1) < delta_r/2, du_dot_rhat, np.nan)
    du_dot_rhat_mask2 = xr.where(np.abs(distances - r2) < delta_r/2, du_dot_rhat, np.nan)
    du_dot_rhat_mask3 = xr.where(np.abs(distances - r3) < delta_r/2, du_dot_rhat, np.nan)
    
    du_tan_mask1 = xr.where(np.abs(distances - r1) < delta_r/2, du_tangent, np.nan)
    du_tan_mask2 = xr.where(np.abs(distances - r2) < delta_r/2, du_tangent, np.nan)
    du_tan_mask3 = xr.where(np.abs(distances - r3) < delta_r/2, du_tangent, np.nan)
    du_norm_mask1 = xr.where(np.abs(distances - r1) < delta_r/2, du_normal, np.nan)
    du_norm_mask2 = xr.where(np.abs(distances - r2) < delta_r/2, du_normal, np.nan)
    du_norm_mask3 = xr.where(np.abs(distances - r3) < delta_r/2, du_normal, np.nan)
    du_sq_mask1 = xr.where(np.abs(distances - r1) < delta_r/2, du_sq, np.nan)
    du_sq_mask2 = xr.where(np.abs(distances - r2) < delta_r/2, du_sq, np.nan)
    du_sq_mask3 = xr.where(np.abs(distances - r3) < delta_r/2, du_sq, np.nan)

    ### PLOT DISTANCES, INITIAL BEARINGS, FINAL BEARINGS ALONG GREAT CIRCLE PATHS
    fig, axes = plt.subplots(
        nrows=1, ncols=3,
        subplot_kw={"projection":cpy.crs.PlateCarree()},
        figsize=(20,10)
    )
    pc_r = axes[0].pcolormesh(lons,lats,distances/1000,cmap="magma_r", transform=cpy.crs.PlateCarree())
    axes[0].plot(lon_origin, lat_origin, marker="x", color="C2", transform=cpy.crs.PlateCarree())
    axes[0].pcolormesh(lons,lats,dist_mask1,cmap="binary", transform=cpy.crs.PlateCarree())
    axes[0].pcolormesh(lons,lats,dist_mask2,cmap="binary", transform=cpy.crs.PlateCarree())
    axes[0].pcolormesh(lons,lats,dist_mask3,cmap="binary", transform=cpy.crs.PlateCarree())
    plt.colorbar(
        pc_r, orientation="horizontal",
        label=f"great circle distance from {distances.origin_latitude:.4g}N, {lon_origin:.4g}E [km]"
    )
    pc_b = axes[1].pcolormesh(lons,lats,init_bearings,cmap="twilight_shifted", transform=cpy.crs.PlateCarree())
    plt.colorbar(
        pc_b, orientation="horizontal",
        label=f"initial bearing along great circle path\nfrom "\
        f"{distances.origin_latitude:.4g}N, {lon_origin:.4g}E [rad.]"
    )
    pc_b = axes[2].pcolormesh(lons,lats,final_bearings,cmap="twilight_shifted", transform=cpy.crs.PlateCarree())
    plt.colorbar(
        pc_b, orientation="horizontal",
        label=f"final bearing along great circle path\nfrom "\
        f"{distances.origin_latitude:.4g}N, {lon_origin:.4g}E [rad.]"
    )
    for ax in axes:
        ax.coastlines()
        ax.plot(lon_origin, distances.origin_latitude, marker="x", color="C2", transform=cpy.crs.PlateCarree())
    
    ### PLOT u, delta u_tangent, delta u_normal, delta_u**2 (exact)
    fig, axes = plt.subplots(
        nrows=2, ncols=2,
        subplot_kw={"projection":cpy.crs.PlateCarree()},
        figsize=(20,10)
    )
    mag_u = 60
    # u
    ax = axes[0,0]
    pc_u = ax.pcolormesh(
        lons,lats,u_mask1,cmap="RdBu_r",
        vmin=-mag_u,
        vmax=mag_u,
        transform=cpy.crs.PlateCarree()
    )
    ax.pcolormesh(
        lons,lats,u_mask2,cmap="RdBu_r",
        vmin=-mag_u,
        vmax=mag_u,
        transform=cpy.crs.PlateCarree()
    )
    ax.pcolormesh(
        lons,lats,u_mask3,cmap="RdBu_r",
        vmin=-mag_u,
        vmax=mag_u,
        transform=cpy.crs.PlateCarree()
    )
    plt.colorbar(
        pc_u, orientation="horizontal",
        label=f"zonal velocity at 200 hPa [m s-1]",
        extend="both"
    )
    # delta u_tangent
    ax = axes[0,1]
    pc_du = ax.pcolormesh(
        lons,lats,du_tan_mask1,cmap="RdBu_r",
        vmin=-mag_u,
        vmax=mag_u,
        transform=cpy.crs.PlateCarree()
    )
    ax.pcolormesh(
        lons,lats,du_tan_mask2,cmap="RdBu_r",
        vmin=-mag_u,
        vmax=mag_u,
        transform=cpy.crs.PlateCarree()
    )
    ax.pcolormesh(
        lons,lats,du_tan_mask3,cmap="RdBu_r",
        vmin=-mag_u,
        vmax=mag_u,
        transform=cpy.crs.PlateCarree()
    )
    plt.colorbar(
        pc_du, orientation="horizontal",
        label=f"difference velocity tangent to geodesic at 200 hPa [m s-1]",
        extend="both"
    )
    # delta u_normal
    ax = axes[1,0]
    pc_du = ax.pcolormesh(
        lons,lats,du_norm_mask1,cmap="RdBu_r",
        vmin=-mag_u,
        vmax=mag_u,
        transform=cpy.crs.PlateCarree()
    )
    ax.pcolormesh(
        lons,lats,du_norm_mask2,cmap="RdBu_r",
        vmin=-mag_u,
        vmax=mag_u,
        transform=cpy.crs.PlateCarree()
    )
    ax.pcolormesh(
        lons,lats,du_norm_mask3,cmap="RdBu_r",
        vmin=-mag_u,
        vmax=mag_u,
        transform=cpy.crs.PlateCarree()
    )
    plt.colorbar(
        pc_du, orientation="horizontal",
        label=f"difference velocity normal to geodesic at 200 hPa [m s-1]",
        extend="both"
    )
    # delta u squared
    ax = axes[1,1]
    pc_du_sq = ax.pcolormesh(
        lons,lats,du_sq_mask1,cmap="RdBu_r",
        vmin=-mag_u**2,
        vmax=mag_u**2,
        transform=cpy.crs.PlateCarree()
    )
    ax.pcolormesh(
        lons,lats,du_sq_mask2,cmap="RdBu_r",
        vmin=-mag_u**2,
        vmax=mag_u**2,
        transform=cpy.crs.PlateCarree()
    )
    ax.pcolormesh(
        lons,lats,du_sq_mask3,cmap="RdBu_r",
        vmin=-mag_u**2,
        vmax=mag_u**2,
        transform=cpy.crs.PlateCarree()
    )
    plt.colorbar(
        pc_du_sq, orientation="horizontal",
        label=f"square of difference velocity at 200 hPa [m2 s-2]",
        extend="both"
    )
    # tidying up
    for ax in axes.flatten():
        ax.coastlines()
        ax.plot(lon_origin, distances.origin_latitude, marker="x", color="C2", transform=cpy.crs.PlateCarree())

    ### COMPARISON OF \delta u dot rhat, \delta u ^2 EXACT VS. APPROX.
    fig, axes = plt.subplots(
        nrows=2, ncols=3,
        subplot_kw={"projection":cpy.crs.PlateCarree()},
        figsize=(20,10)
    )
    mag_u = 60
    # \delta u_tangent (= \delta u  \cdot \hat{r} *exact*)
    ax = axes[0,0]
    pc_du = ax.pcolormesh(
        lons, lats, du_tangent, cmap="RdBu_r",
        vmin=-mag_u,
        vmax=mag_u,
        transform=cpy.crs.PlateCarree()
    )
    plt.colorbar(
        pc_du, orientation="horizontal",
        label=r"exact $\delta u \cdot \hat{\mathbf{r}}$ at 200 hPa [m s-1]",
        extend="both",
        ax=ax
    )
    # du_dot_rhat (= \delta u  \cdot \hat{r} *approx.*)
    ax = axes[0,1]
    pc_du = ax.pcolormesh(
        lons, lats, du_dot_rhat, cmap="RdBu_r",
        vmin=-mag_u,
        vmax=mag_u,
        transform=cpy.crs.PlateCarree()
    )
    plt.colorbar(
        pc_du, orientation="horizontal",
        label=r"approx. $\delta u \cdot \hat{\mathbf{r}}$ at 200 hPa [m s-1]",
        extend="both",
        ax=ax
    )
    # difference (exact minus approx.)
    ax = axes[0,2]
    pc_du = ax.pcolormesh(
        lons, lats, du_tangent - du_dot_rhat, cmap="RdBu_r",
        vmin=-mag_u,
        vmax=mag_u,
        transform=cpy.crs.PlateCarree()
    )
    plt.colorbar(
        pc_du, orientation="horizontal",
        label=r"exact minus approx. $\delta u \cdot \hat{\mathbf{r}}$ at 200 hPa [m s-1]",
        extend="both",
        ax=ax
    )
    # (\delta u)^2 (exact)
    ax = axes[1,0]
    pc_du_sq = ax.pcolormesh(
        lons, lats, du_sq, cmap="RdBu_r",
        vmin=-mag_u**2,
        vmax=mag_u**2,
        transform=cpy.crs.PlateCarree()
    )
    plt.colorbar(
        pc_du_sq, orientation="horizontal",
        label=r"exact $\delta u \cdot \hat{\mathbf{r}}$ at 200 hPa [m2 s-2]",
        extend="both",
        ax=ax
    )
    # (\delta u)^2 (approx.)
    ax = axes[1,1]
    pc_du_sq = ax.pcolormesh(
        lons, lats, du_sq_approx, cmap="RdBu_r",
        vmin=-mag_u**2,
        vmax=mag_u**2,
        transform=cpy.crs.PlateCarree()
    )
    plt.colorbar(
        pc_du_sq, orientation="horizontal",
        label=r"approx. $\delta u \cdot \hat{\mathbf{r}}$ at 200 hPa [m2 s-2]",
        extend="both",
        ax=ax
    )
    # difference (exact minus approx.)
    ax = axes[1,2]
    pc_du_sq = ax.pcolormesh(
        lons, lats, du_sq - du_sq_approx, cmap="RdBu_r",
        vmin=-mag_u**2,
        vmax=mag_u**2,
        transform=cpy.crs.PlateCarree()
    )
    plt.colorbar(
        pc_du_sq, orientation="horizontal",
        label=r"exact minus approx. $| \delta u |^2$ at 200 hPa [m2 s-2]",
        extend="both",
        ax=ax
    )
    # tidying up
    for ax in axes.flatten():
        ax.coastlines()
        ax.plot(lon_origin, distances.origin_latitude, marker="x", color="C2", transform=cpy.crs.PlateCarree())
    #plt.show()

    ### PLOT (\delta u)^3
    fig, axes= plt.subplots(nrows=1, ncols=3, figsize=(20,10), subplot_kw={"projection":cpy.crs.PlateCarree()})
    # exact
    ax = axes[0]
    pc_du_cu = ax.pcolormesh(
        lons, lats, du_cubed, cmap="RdBu_r",
        vmin=-mag_u**3,
        vmax=mag_u**3,
        transform=cpy.crs.PlateCarree()
    )
    plt.colorbar(
        pc_du_cu, orientation="horizontal",
        label=r"exact $( \delta u )^3$ at 200 hPa [m3 s-3]",
        extend="both",
        ax=ax
    )
    # approx.
    ax = axes[1]
    pc_du_cu = ax.pcolormesh(
        lons, lats, du_cubed_approx, cmap="RdBu_r",
        vmin=-mag_u**3,
        vmax=mag_u**3,
        transform=cpy.crs.PlateCarree()
    )
    plt.colorbar(
        pc_du_cu, orientation="horizontal",
        label=r"approx. $( \delta u )^3$ at 200 hPa [m3 s-3]",
        extend="both",
        ax=ax
    )
    # difference (exact minus approx.)
    ax = axes[2]
    pc_du_cu = ax.pcolormesh(
        lons, lats, du_cubed - du_cubed_approx, cmap="RdBu_r",
        vmin=-mag_u**3,
        vmax=mag_u**3,
        transform=cpy.crs.PlateCarree()
    )
    plt.colorbar(
        pc_du_cu, orientation="horizontal",
        label=r"exact minus approx. $( \delta u )^3$ at 200 hPa [m3 s-3]",
        extend="both",
        ax=ax
    )
    # tidying up
    for ax in axes:
        ax.coastlines()
        ax.plot(lon_origin, distances.origin_latitude, marker="x", color="C2", transform=cpy.crs.PlateCarree())

    ### ANGULAR INTEGRALS LINE PLOT
    du_cubed_ang_av = du_cubed.groupby_bins(group=distances, bins=100).mean()
    du_cubed_approx_ang_av = du_cubed_approx.groupby_bins(group=distances, bins=100).mean()

    distance_bin_centres = np.array([val.mid for val in du_cubed_ang_av.great_circle_distance_bins.values])

    print(du_cubed_ang_av)
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(distance_bin_centres/1000., du_cubed_ang_av)
    ax.plot(distance_bin_centres/1000., du_cubed_approx_ang_av)
    ax.grid()
    ax.set_ylabel(r"angular average of $( \delta \mathbf{u} )^3$ at 200 hPa [m3 s-3]")
    ax.set_xlabel("great circle distance [km]")
    plt.show()
    
    

    print("\n\n\nEND.")

    
