"""Single-shell SO(3) increment Zarr workflow and vertical reconstruction."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import warnings

import numpy as np
import xarray as xr

from lossett.calc.group_increments import (
    group_increment_grid,
    hydrostatic_radius,
    latlon_shell_fields,
    reconstruct_group_vertical,
)


def _solid_angles(latitude, longitude):
    lat = np.asarray(latitude, dtype=float)
    lon = np.asarray(longitude, dtype=float)
    if lat.size < 3 or lon.size < 4:
        raise ValueError("the global grid needs at least three latitudes and four longitudes")
    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise ValueError("grid coordinates must be finite")
    descending = lat[1] < lat[0]
    ordered = lat[::-1] if descending else lat
    step = np.diff(ordered)
    if np.any(step <= 0) or not np.allclose(step, step[0], rtol=1e-7, atol=1e-12):
        raise ValueError("latitude must be uniformly spaced and monotonic")
    if ordered[0] < -np.pi / 2 - 1e-12 or ordered[-1] > np.pi / 2 + 1e-12:
        raise ValueError("latitudes must be within the poles")
    if ordered[0] + np.pi / 2 > step[0] or np.pi / 2 - ordered[-1] > step[0]:
        raise ValueError("latitude grid must cover the whole sphere")
    if not np.allclose(np.diff(lon), 2 * np.pi / lon.size, rtol=1e-7, atol=1e-12):
        raise ValueError("longitude must be uniform and global without a duplicate endpoint")
    edges = np.r_[-np.pi / 2, (ordered[:-1] + ordered[1:]) / 2, np.pi / 2]
    weights = np.diff(np.sin(edges)) * 2 * np.pi / lon.size
    if descending:
        weights = weights[::-1]
    return np.broadcast_to(weights[:, None], (lat.size, lon.size))


def _open_velocity(path):
    if str(path).endswith(".zarr"):
        return xr.open_zarr(path)
    return xr.open_dataset(path)


def shell_dataset(
    ds, *, time_index, level_index, level_dim, coordinate_type, coordinate_unit,
    radius, heat_time=None, mollifier_scale=None, atmosphere="deep",
    quadrature_tolerance=0.01, cache_longitude_moments=False, compact_support=False, pair_backend="numpy",
):
    """Calculate and package one shell; velocities are east/north/radial in m/s."""
    valid_units = {"height": {"m": 1.0}, "pressure": {"Pa": 1.0, "hPa": 100.0}}
    if coordinate_type not in valid_units or coordinate_unit not in valid_units[coordinate_type]:
        raise ValueError("coordinate unit must be m for height or Pa/hPa for pressure")
    if heat_time is not None and mollifier_scale is None:
        kernel_parameter = heat_time
    elif mollifier_scale is not None and heat_time is None:
        kernel_parameter = mollifier_scale
    else:
        raise ValueError("specify exactly one of heat_time and mollifier_scale")
    if level_dim not in ds.dims or "time" not in ds.dims:
        raise ValueError("input requires time and the selected level dimension")
    declared_unit = str(ds[level_dim].attrs.get("units", "")).strip().lower()
    aliases = {"pa": "Pa", "hpa": "hPa", "mbar": "hPa", "millibar": "hPa",
               "millibars": "hPa", "m": "m", "meter": "m", "metre": "m",
               "meters": "m", "metres": "m"}
    if declared_unit in aliases and aliases[declared_unit] != coordinate_unit:
        raise ValueError("coordinate unit conflicts with input metadata")
    selected = ds.isel(time=time_index, **{level_dim: level_index})
    for name in ("u", "v", "w"):
        if name not in selected or set(selected[name].dims) != {"latitude", "longitude"}:
            raise ValueError(f"{name} must have exactly latitude and longitude dimensions after selection")
        units = str(selected[name].attrs.get("units", "")).replace(" ", "").lower()
        if units and units not in {"m/s", "ms-1", "ms^-1", "ms**-1", "ms**(-1)", "m.s-1"}:
            raise ValueError(f"{name} units must describe velocity in m/s, not {units!r}")
    lat = np.asarray(selected.latitude.values, dtype=float)
    lon = np.asarray(selected.longitude.values, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("latitude and longitude must be one-dimensional")
    latitude = np.deg2rad(lat)
    longitude = np.deg2rad(lon)
    weights = _solid_angles(latitude, longitude)
    values = [selected[name].transpose("latitude", "longitude").values for name in ("u", "v", "w")]
    directions, velocity, grad, gradw = latlon_shell_fields(latitude, longitude, *values, radius=radius)
    result = group_increment_grid(
        directions, velocity, grad, gradw, weights, radius=radius, heat_time=heat_time,
        mollifier_scale=mollifier_scale, atmosphere=atmosphere,
        quadrature_tolerance=quadrature_tolerance, cache_longitude_moments=cache_longitude_moments,
        compact_support=compact_support, pair_backend=pair_backend,
    )
    coordinate = float(selected[level_dim].values) * valid_units[coordinate_type][coordinate_unit]
    if not np.isfinite(coordinate) or (coordinate_type == "pressure" and coordinate <= 0):
        raise ValueError("vertical coordinate must be finite and pressure positive")
    dims = ("latitude", "longitude")
    out = xr.Dataset(
        {name: (dims + (("component",) if name == "mixed_moment" else ()), data)
         for name, data in result.items()},
        coords={"latitude": lat, "longitude": lon, "component": ["x", "y", "z"]},
        attrs={
            "diagnostic": "coherent SO(3) matched-group increments",
            "sign_convention": "positive downscale",
            "coordinate_type": coordinate_type,
            "coordinate_value": coordinate,
            "coordinate_units": "Pa" if coordinate_type == "pressure" else "m",
            "radius_m": float(radius),
            "atmosphere": atmosphere,
            "kernel": "heat" if heat_time is not None else "mollifier",
            "kernel_parameter": float(kernel_parameter),
            "pair_backend": pair_backend,
            "quadrature_tolerance": float(quadrature_tolerance),
            "time_value": str(selected.time.values),
        },
    )
    out["direction"] = (dims + ("component",), directions)
    out["velocity"] = (dims + ("component",), velocity)
    for name, unit in {
        "horizontal": "m2 s-3", "structure": "m2 s-2", "vertical_flux": "m3 s-3",
        "mixed_moment": "m2 s-2", "kernel_mass": "1", "direction": "1", "velocity": "m s-1",
    }.items():
        out[name].attrs["units"] = unit
    out.latitude.attrs["units"] = "degrees_north"
    out.longitude.attrs["units"] = "degrees_east"
    return out


def vertical_dataset(shells, *, fluid, reference_density=None, gravity=None):
    """Reconstruct vertical transfer from compatible, ordered shell datasets."""
    if len(shells) < 3:
        raise ValueError("at least three shell outputs are required")
    first = shells[0]
    keys = ("diagnostic", "coordinate_type", "coordinate_units", "atmosphere", "kernel",
            "kernel_parameter", "quadrature_tolerance", "time_value")
    for shell in shells[1:]:
        if any(shell.attrs[key] != first.attrs[key] for key in keys):
            raise ValueError("shell metadata do not match")
        if any(not np.array_equal(shell[dim].values, first[dim].values)
               for dim in ("latitude", "longitude", "component")):
            raise ValueError("shell grids do not match")
        if not np.allclose(shell.direction.values, first.direction.values, rtol=0, atol=1e-12):
            raise ValueError("shell radial directions do not match")
    coord = np.array([s.attrs["coordinate_value"] for s in shells])
    radius = np.array([s.attrs["radius_m"] for s in shells])
    rho = None if reference_density is None else np.asarray(reference_density, dtype=float)
    if rho is not None and (rho.shape != coord.shape or not np.all(np.isfinite(rho)) or np.any(rho <= 0)):
        raise ValueError("reference density must be a positive finite value for each shell")
    coordinate_type = first.attrs["coordinate_type"]
    if coordinate_type == "height" and gravity is not None:
        raise ValueError("gravity is only used for pressure reconstruction")
    if coordinate_type == "height" and fluid == "boussinesq" and rho is not None:
        warnings.warn("Reference density is not used in Boussinesq height reconstruction.", UserWarning, stacklevel=2)
    if coordinate_type == "pressure":
        if rho is None or gravity is None:
            raise ValueError("pressure reconstruction requires reference density and gravity")
        profile_keys = ("hydrostatic_pressure_pa", "hydrostatic_density", "hydrostatic_initial_radius",
                        "hydrostatic_gravity")
        has_profile = [all(key in s.attrs for key in profile_keys) for s in shells]
        any_profile = any(any(key in s.attrs for key in profile_keys) for s in shells)
        if any_profile and not all(has_profile):
            raise ValueError("shell hydrostatic profile provenance is inconsistent")
        if all(has_profile):
            if any(any(s.attrs[key] != first.attrs[key] for key in profile_keys) for s in shells[1:]):
                raise ValueError("shell hydrostatic profiles do not match")
            pressure = np.asarray(first.attrs["hydrostatic_pressure_pa"])
            density = np.asarray(first.attrs["hydrostatic_density"])
            if gravity != first.attrs["hydrostatic_gravity"]:
                raise ValueError("gravity does not match the shell hydrostatic profile")
            radii = hydrostatic_radius(pressure, density, gravity, first.attrs["hydrostatic_initial_radius"])
            indices = [np.flatnonzero(pressure == value) for value in coord]
            if any(len(index) != 1 for index in indices):
                raise ValueError("shell pressure is absent from the reference hydrostatic profile")
            indices = np.array([index[0] for index in indices])
            if not np.allclose(radius, radii[indices], rtol=1e-9, atol=1e-4) or not np.allclose(rho, density[indices]):
                raise ValueError("shell radii or densities do not match the reference hydrostatic profile")
        else:
            warnings.warn(
                "Shells have no hydrostatic profile provenance; verify that radii, pressure, "
                "density and gravity represent the same reference hydrostatic profile.",
                UserWarning,
                stacklevel=2,
            )
    fields = {name: np.stack([s[name].values for s in shells]) for name in
              ("velocity", "structure", "vertical_flux", "mixed_moment", "horizontal")}
    nlat, nlon = first.sizes["latitude"], first.sizes["longitude"]
    vertical = np.empty((len(shells), nlat, nlon))
    for i in range(nlat):
        for j in range(nlon):
            vertical[:, i, j] = reconstruct_group_vertical(
                coord, radius, first.direction.values[i, j], fields["velocity"][:, i, j],
                fields["structure"][:, i, j], fields["vertical_flux"][:, i, j],
                fields["mixed_moment"][:, i, j], coordinate_type=coordinate_type,
                fluid=fluid, atmosphere=first.attrs["atmosphere"],
                reference_density=rho, gravity=gravity,
            )
    attrs = dict(first.attrs)
    attrs.pop("coordinate_value")
    attrs.pop("radius_m")
    attrs.pop("velocity_source", None)
    if all("velocity_source" in s.attrs for s in shells):
        attrs["velocity_sources"] = [s.attrs["velocity_source"] for s in shells]
    attrs.update(fluid=fluid, reconstruction="second-order finite differences in the vertical coordinate")
    if gravity is not None:
        attrs["gravity_m_s-2"] = float(gravity)
    out = xr.Dataset(
        {
            "horizontal": (("level", "latitude", "longitude"), fields["horizontal"]),
            "vertical": (("level", "latitude", "longitude"), vertical),
            "total": (("level", "latitude", "longitude"), fields["horizontal"] + vertical),
            "radius": (("level",), radius),
        },
        coords={"level": coord, "latitude": first.latitude.values, "longitude": first.longitude.values},
        attrs=attrs,
    )
    if rho is not None:
        out["reference_density"] = (("level",), rho)
        out.reference_density.attrs["units"] = "kg m-3"
    out.level.attrs["units"] = first.attrs["coordinate_units"]
    out.radius.attrs["units"] = "m"
    for name in ("horizontal", "vertical", "total"):
        out[name].attrs["units"] = "m2 s-3"
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    shell = sub.add_parser("shell", help="calculate one level and save its moments")
    shell.add_argument("--velocity-file", required=True)
    shell.add_argument("--output", required=True)
    shell.add_argument("--time-index", type=int, required=True)
    shell.add_argument("--level-index", type=int, required=True)
    shell.add_argument("--level-dim", default="pressure")
    shell.add_argument("--coordinate-type", choices=("height", "pressure"), required=True)
    shell.add_argument("--coordinate-unit", choices=("m", "Pa", "hPa"), required=True)
    shell.add_argument("--radius", type=float, help="shell radius in metres")
    shell.add_argument("--initial-radius", type=float, help="radius at the first input pressure level in metres")
    shell.add_argument("--density", type=float, nargs="+", help="reference densities in input level order, kg/m3")
    shell.add_argument("--gravity", type=float, help="reference gravity in m/s2")
    kernel = shell.add_mutually_exclusive_group(required=True)
    kernel.add_argument("--heat-time", type=float)
    kernel.add_argument("--mollifier-scale", type=float)
    shell.add_argument("--atmosphere", choices=("deep", "shallow"), default="deep")
    shell.add_argument("--quadrature-tolerance", type=float, default=0.01)
    shell.add_argument("--cache-longitude-moments", action="store_true",
                       help="reuse fiber moments across regular global longitude rotations")
    shell.add_argument("--compact-support", action="store_true",
                       help="restrict matched-mollifier targets to its exact compact support")
    shell.add_argument("--pair-backend", choices=("numpy", "numba"), default="numpy",
                       help="pairwise accumulation backend")
    vertical = sub.add_parser("vertical", help="reconstruct from at least three ordered shell stores")
    vertical.add_argument("--shells", nargs="+", required=True)
    vertical.add_argument("--output", required=True)
    vertical.add_argument("--fluid", choices=("boussinesq", "anelastic"), required=True)
    vertical.add_argument("--density", type=float, nargs="+", help="reference densities in shell order, kg/m3")
    vertical.add_argument("--gravity", type=float, help="reference gravity in m/s2")
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.command == "shell":
        with _open_velocity(args.velocity_file) as ds:
            radius = args.radius
            hydrostatic_metadata = None
            if args.initial_radius is not None:
                if radius is not None or args.coordinate_type != "pressure" or args.density is None or args.gravity is None:
                    parser.error("--initial-radius requires pressure, --density and --gravity, and excludes --radius")
                if args.coordinate_unit not in ("Pa", "hPa"):
                    parser.error("pressure coordinate unit must be Pa or hPa")
                pressure = np.asarray(ds[args.level_dim].values, dtype=float) * (
                    100 if args.coordinate_unit == "hPa" else 1
                )
                radius = hydrostatic_radius(pressure, args.density, args.gravity, args.initial_radius)[args.level_index]
                hydrostatic_metadata = {
                    "hydrostatic_pressure_pa": pressure.tolist(),
                    "hydrostatic_density": list(args.density),
                    "hydrostatic_initial_radius": float(args.initial_radius),
                    "hydrostatic_gravity": float(args.gravity),
                }
            if radius is None:
                parser.error("provide --radius or a hydrostatic --initial-radius profile")
            if args.initial_radius is None and (args.density is not None or args.gravity is not None):
                parser.error("--density and --gravity in the shell command require --initial-radius")
            out = shell_dataset(
                ds, time_index=args.time_index, level_index=args.level_index, level_dim=args.level_dim,
                coordinate_type=args.coordinate_type, coordinate_unit=args.coordinate_unit, radius=radius,
                heat_time=args.heat_time, mollifier_scale=args.mollifier_scale,
                atmosphere=args.atmosphere, quadrature_tolerance=args.quadrature_tolerance,
                cache_longitude_moments=args.cache_longitude_moments,
                compact_support=args.compact_support,
                pair_backend=args.pair_backend,
            )
            out.attrs["velocity_source"] = str(Path(args.velocity_file).resolve())
            if hydrostatic_metadata is not None:
                out.attrs.update(hydrostatic_metadata)
    else:
        with ExitStack() as stack:
            opened = [stack.enter_context(xr.open_zarr(path)) for path in args.shells]
            out = vertical_dataset(opened, fluid=args.fluid, reference_density=args.density, gravity=args.gravity)
    out.to_zarr(args.output, mode="w-", zarr_version=2)


if __name__ == "__main__":
    main()
