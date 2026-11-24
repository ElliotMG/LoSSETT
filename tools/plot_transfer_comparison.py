#!/usr/bin/env python3
"""Plot saved horizontal transfer fields from the benchmark paths."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import numpy as np
import xarray as xr


PATHS = (
    ("cartesian", "Original Cartesian", "Dl_u"),
    ("legacy", "Old spherical", "DL_u_longitudinal"),
    ("numba", "Group Numba", "horizontal"),
    ("fft", "Group FFT", "horizontal"),
)


def load_numpy(path, latitude, longitude, coordinates):
    """Load experimental seven-channel output, preserving sampled row indices."""
    report_path = path.with_name("results.json")
    with report_path.open() as stream:
        report = json.load(stream)
    if report.get("grid_shape") != [len(latitude), len(longitude)]:
        raise ValueError(f"{report_path}: grid_shape does not match the coordinate file")
    source = report.get("velocity_file", report.get("input"))
    if source is not None and Path(source).resolve() != coordinates.resolve():
        raise ValueError(f"{report_path}: input field does not match {coordinates}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.ndim != 3 or values.shape[1:] != (len(longitude), 7):
        raise ValueError(f"{path}: expected shape (latitude rows, {len(longitude)}, 7)")
    if values.shape[0] == len(latitude):
        rows = np.arange(len(latitude))
    else:
        if "origin_rows" not in report:
            raise ValueError(f"{report_path}: sampled output requires origin_rows")
        rows = np.asarray(report["origin_rows"])
        if (rows.ndim != 1 or rows.size != values.shape[0]
                or not np.issubdtype(rows.dtype, np.integer)
                or np.any(rows < 0) or np.any(rows >= len(latitude))
                or np.any(np.diff(rows) <= 0)):
            raise ValueError(f"{report_path}: invalid or unordered origin_rows")
    return rows, np.asarray(values[..., 0])


def load_zarr(path, variable, latitude, longitude):
    """Load one singleton-level benchmark field on the specified grid."""
    with xr.open_zarr(path) as dataset:
        if variable not in dataset:
            raise ValueError(f"{path}: missing variable {variable}")
        field = dataset[variable]
        for dim in field.dims:
            if dim not in ("latitude", "longitude"):
                if field.sizes[dim] != 1:
                    raise ValueError(f"{path}: {dim} must have exactly one value")
                field = field.isel({dim: 0}, drop=True)
        if set(field.dims) != {"latitude", "longitude"}:
            raise ValueError(f"{path}: {variable} must have latitude and longitude dimensions")
        for dim, expected in (("latitude", latitude), ("longitude", longitude)):
            actual = field[dim].values
            if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=1e-8):
                raise ValueError(f"{path}: {dim} does not match the coordinate file")
        units = field.attrs.get("units")
        if units and units != "m2 s-3":
            raise ValueError(f"{path}: unexpected {variable} units {units!r}")
        values = field.transpose("latitude", "longitude").values
    return np.arange(len(latitude)), values


def load_field(path, variable, latitude, longitude, coordinates):
    if path.suffix == ".npy":
        if variable != "horizontal":
            raise ValueError(f"{path}: NumPy benchmark output is only supported for group paths")
        return load_numpy(path, latitude, longitude, coordinates)
    return load_zarr(path, variable, latitude, longitude)


def plot(fields, latitude, longitude, requested_latitude, output, grid_label):
    finite_peaks = [
        np.max(np.abs(values[np.isfinite(values)]))
        for _, values in fields.values() if np.isfinite(values).any()
    ]
    if not finite_peaks:
        raise ValueError("the supplied outputs contain no finite horizontal transfer values")
    limit = max(finite_peaks) or 1.0
    selected_row = int(np.argmin(np.abs(latitude - requested_latitude)))
    selected_latitude = latitude[selected_row]

    fig = plt.figure(figsize=(14, 11), layout="constrained")
    layout = fig.add_gridspec(3, 2, height_ratios=(1, 1, 0.8))
    map_axes = [fig.add_subplot(layout[i // 2, i % 2]) for i in range(4)]
    profile = fig.add_subplot(layout[2, :])
    image = None
    for ax, (key, label, _) in zip(map_axes, PATHS):
        ax.set_title(label)
        ax.set_xlabel("Longitude (degrees)")
        ax.set_ylabel("Latitude (degrees)")
        ax.set_xlim(longitude.min(), longitude.max())
        ax.set_ylim(-90, 90)
        if key not in fields:
            ax.text(0.5, 0.5, "Not supplied", ha="center", va="center", transform=ax.transAxes)
            ax.set_facecolor("0.94")
            continue
        rows, values = fields[key]
        if len(rows) == len(latitude):
            image = ax.pcolormesh(
                longitude, latitude, np.ma.masked_invalid(values),
                shading="nearest", cmap="RdBu_r", vmin=-limit, vmax=limit, rasterized=True,
            )
        else:
            x = np.broadcast_to(longitude, values.shape)
            y = np.broadcast_to(latitude[rows, None], values.shape)
            image = ax.scatter(
                x.ravel(), y.ravel(), c=values.ravel(), s=3, marker=MarkerStyle("s"),
                cmap="RdBu_r", vmin=-limit, vmax=limit, linewidths=0, rasterized=True,
            )
            ax.text(0.02, 0.02, f"{len(rows)} sampled rings only",
                    transform=ax.transAxes, ha="left", va="bottom",
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        index = np.flatnonzero(rows == selected_row)
        if index.size:
            profile.plot(longitude, values[index[0]], label=label, linewidth=1)
    fig.colorbar(image, ax=map_axes, label=r"Horizontal transfer (m$^2$ s$^{-3}$)")
    profile.axhline(0, color="0.5", linewidth=0.6)
    profile.set_title(f"Longitude section at {selected_latitude:g} degrees latitude")
    profile.set_xlabel("Longitude (degrees)")
    profile.set_ylabel(r"Transfer (m$^2$ s$^{-3}$)")
    if profile.lines[1:]:
        profile.legend(ncol=4, fontsize="small")
    else:
        profile.text(0.5, 0.5, "No supplied output has this latitude",
                     transform=profile.transAxes, ha="center", va="center")
    fig.suptitle(
        f"{grid_label}: horizontal outputs; positive downscale\n"
        "Different diagnostics, kernels and boundary treatments; not equivalent transfers",
        fontsize=12,
    )
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}; longitude section at {selected_latitude:g} degrees latitude.")
    if "numba" in fields and "fft" in fields:
        numba_rows, numba = fields["numba"]
        fft_rows, fft = fields["fft"]
        common, ni, fi = np.intersect1d(numba_rows, fft_rows, return_indices=True)
        if common.size:
            difference = numba[ni] - fft[fi]
            finite = np.isfinite(difference)
            if finite.any():
                print(f"Numba minus FFT: max absolute difference {np.max(np.abs(difference[finite])):.6g} "
                      f"over {finite.sum()} finite points on {len(common)} common rings.")
            else:
                print("Numba and FFT have no common finite values.")
        else:
            print("Numba and FFT have no common latitude rings.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", type=Path, required=True,
                        help="Input NetCDF file containing the full latitude and longitude grid")
    for key, label, _ in PATHS:
        parser.add_argument(f"--{key}", type=Path, help=f"{label} Zarr store or group NumPy output")
    parser.add_argument("--latitude", type=float, default=0.0, help="Latitude of longitude section (default: 0)")
    parser.add_argument("--label", default="Benchmark grid", help="Title identifying this grid and run")
    parser.add_argument("--output", type=Path, required=True, help="Output figure path; must not exist")
    args = parser.parse_args(argv)
    if not any(getattr(args, key) is not None for key, _, _ in PATHS):
        parser.error("provide at least one benchmark output")
    if not np.isfinite(args.latitude) or not -90 <= args.latitude <= 90:
        parser.error("--latitude must be finite and between -90 and 90 degrees")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    with xr.open_dataset(args.coordinates, decode_timedelta=False) as dataset:
        latitude = np.asarray(dataset["latitude"].values)
        longitude = np.asarray(dataset["longitude"].values)
    for name, coord in (("latitude", latitude), ("longitude", longitude)):
        if coord.ndim != 1 or len(coord) < 2 or not np.isfinite(coord).all() or np.any(np.diff(coord) <= 0):
            raise ValueError(f"{name} must be a finite, strictly increasing one-dimensional coordinate")
    fields = {
        key: load_field(path, variable, latitude, longitude, args.coordinates)
        for key, _, variable in PATHS if (path := getattr(args, key)) is not None
    }
    plot(fields, latitude, longitude, args.latitude, args.output, args.label)


if __name__ == "__main__":
    main()
