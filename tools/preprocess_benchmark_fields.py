#!/usr/bin/env python3
"""Spectrally low-pass global winds before sampling benchmark grids."""

import argparse
import json
import time
from pathlib import Path

import ducc0
import numpy as np
import xarray as xr

from lossett.calc.compute_spherical_geometry import GRID_DEFS, build_regular_latlon_grid


def spectral_limits(nlat, nlon):
    """Return taper start and inclusive cutoff below the grid's safe bandlimit."""
    safe = min(nlat - 2, (nlon - 1) // 2)
    if safe < 4:
        raise ValueError("target grid is too small for spectral filtering")
    return int(0.7 * safe), int(0.9 * safe)


def _tapered_coefficients(coefficients, source_limit, pass_limit, stop_limit):
    result = np.empty((coefficients.shape[0], (stop_limit + 1) * (stop_limit + 2) // 2), dtype=coefficients.dtype)
    for m in range(stop_limit + 1):
        degree = np.arange(m, stop_limit + 1)
        weight = np.ones(degree.size)
        transition = degree > pass_limit
        weight[transition] = (1 + np.cos(np.pi * (degree[transition] - pass_limit) / (stop_limit - pass_limit))) / 2
        source = m * (2 * source_limit + 1 - m) // 2 + degree
        target = m * (2 * stop_limit + 1 - m) // 2 + degree
        result[:, target] = coefficients[:, source] * weight
    return result


def _regularize_poles(u, v, w, longitude):
    """Replace each polar value by its longitude-mean physical vector."""
    u, v, w = (np.array(value, dtype=float, copy=True) for value in (u, v, w))
    cosine, sine = np.cos(longitude), np.sin(longitude)
    adjustments = []
    for index, pole in ((0, 1.), (-1, -1.)):
        x = -u[index] * sine - pole * v[index] * cosine
        y = u[index] * cosine - pole * v[index] * sine
        mean_x, mean_y, mean_w = x.mean(), y.mean(), w[index].mean()
        adjustments.append(float(np.max(np.sqrt((x - mean_x)**2 + (y - mean_y)**2 + (w[index] - mean_w)**2))))
        u[index] = -mean_x * sine + mean_y * cosine
        v[index] = -pole * (mean_x * cosine + mean_y * sine)
        w[index] = mean_w
    return u, v, w, adjustments


def filter_shell(u, v, w, latitude, longitude, grids, *, threads=1):
    """Filter one pole-to-pole shell and synthesize it on requested grids.

    Latitude and longitude are degrees; input arrays are (latitude, longitude).
    Horizontal E/B vector harmonics and scalar radial harmonics use the same
    degree response, without mixing horizontal and radial velocity.
    """
    lat, lon = np.asarray(latitude, dtype=float), np.asarray(longitude, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1 or lat.size < 3 or lon.size < 4:
        raise ValueError("latitude and longitude must be one-dimensional global coordinates")
    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise ValueError("coordinates must be finite")
    if not np.allclose(lat, np.linspace(-90., 90., lat.size), rtol=0, atol=1e-6):
        raise ValueError("latitude must be uniform, south-to-north and include both poles")
    if not np.allclose(np.diff(lon), 360. / lon.size, rtol=1e-7, atol=1e-6):
        raise ValueError("longitude must be uniform and global without a duplicate endpoint")
    arrays = [np.asarray(value) for value in (u, v, w)]
    if any(value.shape != (lat.size, lon.size) or not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("velocity fields must be finite and match the latitude-longitude grid")
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if not grids:
        raise ValueError("at least one target grid is required")
    limits = {name: spectral_limits(len(target_lat), len(target_lon)) for name, (target_lat, target_lon) in grids.items()}
    source_limit = max(stop for _, stop in limits.values())
    if source_limit > min(lat.size - 2, (lon.size - 1) // 2):
        raise ValueError("target spectral bandlimit exceeds the source grid's safe bandlimit")
    u, v, w, adjustments = _regularize_poles(
        arrays[0][::-1], arrays[1][::-1], arrays[2][::-1], np.deg2rad(lon)
    )
    kwargs = dict(lmax=source_limit, geometry="CC", nthreads=threads, phi0=np.deg2rad(lon[0]))
    horizontal = ducc0.sht.analysis_2d(map=np.stack((-v, u)), spin=1, **kwargs)
    radial = ducc0.sht.analysis_2d(map=w[None], spin=0, **kwargs)
    outputs = {}
    for name, (target_lat, target_lon) in grids.items():
        target_lat, target_lon = np.asarray(target_lat), np.asarray(target_lon)
        if not np.allclose(target_lat, np.linspace(-90., 90., target_lat.size), rtol=0, atol=1e-6):
            raise ValueError(f"{name}: target latitude must be uniform and include both poles")
        if not np.allclose(np.diff(target_lon), 360. / target_lon.size, rtol=1e-7, atol=1e-6):
            raise ValueError(f"{name}: target longitude must be uniform and global")
        passed, stopped = limits[name]
        synth = dict(
            lmax=stopped, geometry="CC", ntheta=len(target_lat), nphi=len(target_lon),
            nthreads=threads, phi0=np.deg2rad(target_lon[0]),
        )
        h = ducc0.sht.synthesis_2d(
            alm=_tapered_coefficients(horizontal, source_limit, passed, stopped), spin=1, **synth
        )
        r = ducc0.sht.synthesis_2d(
            alm=_tapered_coefficients(radial, source_limit, passed, stopped), spin=0, **synth
        )
        outputs[name] = (h[1, ::-1], -h[0, ::-1], r[0, ::-1])
    return outputs, limits, adjustments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity-file", required=True)
    parser.add_argument("--output-dir", required=True, help="new directory for filtered NetCDF outputs")
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--pressure", type=float, nargs="+", required=True, help="pressure levels in hPa")
    parser.add_argument("--grid", nargs="+", choices=GRID_DEFS, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if len(set(args.grid)) != len(args.grid) or len(set(args.pressure)) != len(args.pressure):
        parser.error("grid names and pressure levels must be unique")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    grids = {}
    for name in args.grid:
        lon, lat = build_regular_latlon_grid(*GRID_DEFS[name])
        grids[name] = (lat, lon)
    limits = {name: spectral_limits(len(lat), len(lon)) for name, (lat, lon) in grids.items()}
    with xr.open_dataset(args.velocity_file, decode_timedelta=False) as ds:
        if ds.pressure.attrs.get("units") != "hPa":
            raise ValueError("input pressure coordinate must have units hPa")
        for name in ("u", "v", "w"):
            if name not in ds or ds[name].attrs.get("units") not in ("m s-1", "m/s"):
                raise ValueError(f"{name} must be present with velocity units m s-1")
        selection = ds.sel(pressure=args.pressure).isel(time=args.time_index)
        levels = selection.pressure.values
        output = {name: {field: [] for field in ("u", "v", "w")} for name in grids}
        adjustments = []
        start = time.perf_counter()
        for pressure in levels:
            print(f"Filtering {pressure:g} hPa", flush=True)
            level = selection.sel(pressure=pressure)
            values = [level[name].transpose("latitude", "longitude").values for name in ("u", "v", "w")]
            filtered, _, pole_changes = filter_shell(
                values[0], values[1], values[2], selection.latitude.values,
                selection.longitude.values, grids, threads=args.threads
            )
            adjustments.append(pole_changes)
            for grid in grids:
                for field, value in zip(("u", "v", "w"), filtered[grid]):
                    output[grid][field].append(value)
        time_value = np.asarray([selection.time.values])
        for grid, (lat, lon) in grids.items():
            passed, stopped = limits[grid]
            result = xr.Dataset(
                {field: (("time", "pressure", "latitude", "longitude"), np.stack(values)[None])
                 for field, values in output[grid].items()},
                coords={"time": time_value, "pressure": levels, "latitude": lat, "longitude": lon},
                attrs={
                    "source_file": str(Path(args.velocity_file).resolve()),
                    "filter": "spherical harmonic scalar and tangent-vector low-pass",
                    "filter_response": "unity to pass degree; cosine taper to zero at stop degree",
                    "filter_pass_degree": passed, "filter_stop_degree": stopped,
                    "pole_treatment": "longitude mean of physical Cartesian velocity at each pole",
                    "pole_max_adjustment_m_s-1": json.dumps(adjustments),
                    "spherical_transform_library": f"ducc0 {ducc0.__version__}",
                },
            )
            result.pressure.attrs["units"] = "hPa"
            result.latitude.attrs["units"] = "degrees_north"
            result.longitude.attrs["units"] = "degrees_east"
            for field in ("u", "v", "w"):
                result[field].attrs["units"] = "m s-1"
            path = output_dir / f"filtered_{grid}.nc"
            result.to_netcdf(path, mode="w")
            print(f"Wrote {path} (pass={passed}, stop={stopped})", flush=True)
    print(f"Completed in {time.perf_counter() - start:.2f} s", flush=True)


if __name__ == "__main__":
    main()
