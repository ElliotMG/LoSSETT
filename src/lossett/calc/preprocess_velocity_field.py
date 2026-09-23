#!/usr/bin/env python3
import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path
import xarray as xr

from lossett.calc.compute_spherical_geometry import (
    GRID_DEFS,
    build_regular_latlon_grid,
)

logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Interpolate a velocity field onto a LoSSETT "
            "analysis grid while retaining all times and "
            "pressure levels."
        )
    )

    parser.add_argument(
        "--input-file",
        required=True,
        help="Input velocity file",
    )

    parser.add_argument(
        "--grid",
        required=True,
        choices=GRID_DEFS.keys(),
        help="Target LoSSETT grid",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for preprocessed output file."
    )

    parser.add_argument(
        "--output-fname-root",
        default=None,
        help=(
            "Root of output filename. "
            "Grid information will be appended automatically. "
            "Default is the input filename stem."
        )
    )

    parser.add_argument(
        "--include-w",
        action="store_true",
        help="Include vertical velocity",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ],
    )

    return parser.parse_args()


def setup_logging(level="INFO"):

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(message)s"
        ),
    )


def load_velocity_file(fpath):

    logger.info(f"Opening {fpath}")

    ds = xr.open_dataset(
        fpath,
        decode_timedelta=False,
    )

    drop_vars = [
        v
        for v in [
            "forecast_reference_time",
            "forecast_period",
        ]
        if v in ds
    ]

    if drop_vars:
        ds = ds.drop_vars(drop_vars)

    logger.info(
        "Converting longitude coordinates "
        "to [-180, 180)"
    )

    lon_attrs = ds.longitude.attrs

    ds.coords["longitude"] = (
        (ds.longitude + 180) % 360
    ) - 180

    ds = ds.sortby("longitude")

    ds.longitude.attrs = lon_attrs

    return ds


def interpolate_to_grid(
    ds,
    grid,
    include_vertical=False,
):

    lon_step, lat_step = GRID_DEFS[grid]

    target_lons, target_lats = (
        build_regular_latlon_grid(
            lon_step,
            lat_step,
        )
    )

    logger.info(
        f"Interpolating to grid {grid} "
        f"({len(target_lats)} latitudes, "
        f"{len(target_lons)} longitudes)"
    )

    data_vars = {
        "u": ds.u.interp(
            latitude=target_lats,
            longitude=target_lons,
        ),
        "v": ds.v.interp(
            latitude=target_lats,
            longitude=target_lons,
        ),
    }

    if include_vertical:

        data_vars["w"] = ds.w.interp(
            latitude=target_lats,
            longitude=target_lons,
        )

    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": ds.time,
            "pressure": ds.pressure,
            "latitude": target_lats,
            "longitude": target_lons,
        },
        attrs=ds.attrs,
    )

    ds_out.attrs.update(
        {
            "target_grid": grid,
            "history": (
                ds.attrs.get("history", "")
                + "\n"
                + f"{datetime.now(UTC).isoformat()}: "
                + "Interpolated to LoSSETT grid "
                + grid
            ),
        }
    )

    return ds_out


def main():

    args = parse_args()

    input_file = Path(args.input_file)

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.output_fname_root is None:
        output_fname_root = input_file.stem
    else:
        output_fname_root = args.output_fname_root

    output_fname = (
        f"{output_fname_root}_{args.grid}.nc"
    )

    output_file = (
        output_dir
        / output_fname
    )

    setup_logging(args.log_level)

    logger.info(
        "\n"
        "#########################################\n"
        "### LoSSETT velocity preprocessing ######\n"
        "#########################################\n"
        f"\nInput file:  {args.input_file}\n"
        f"Grid:        {args.grid}\n"
        f"Output file: {output_file}\n"
    )

    ds = load_velocity_file(input_file)

    ds_interp = interpolate_to_grid(
        ds,
        args.grid,
        include_vertical=args.include_w,
    )

    ds_interp.attrs.update(
        {"source_file": repr(input_file)}
    )

    logger.info("Writing output")

    ds_interp.to_netcdf(
        output_file,
    )

    logger.info("\n\n\nEND.")


if __name__ == "__main__":
    main()
