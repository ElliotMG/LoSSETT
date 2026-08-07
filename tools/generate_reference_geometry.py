#!/usr/bin/env python3

import numpy as np
import xarray as xr
from pathlib import Path

from lossett.calc.spherical_geometry import (
    RADIUS_EARTH,
)
from lossett.calc.compute_spherical_geometry import (
    initialize_geometry_store,
    write_geometry_chunk,
    iter_chunks,
)

OUTFILE = (
    Path(__file__).parent.parent
    / "tests"
    / "reference_data"
    / "reference_geometry.zarr"
)

origin_lats = np.array([
    -30.0,
    0.0,
    30.0,
])

target_lats = origin_lats

delta_lons = np.array([
    -90.0,
    0.0,
    90.0,
])

distance_edges = np.linspace(
    0.0,
    np.pi * RADIUS_EARTH,
    8,
)

compute_kwargs = {
    "dtype": np.float32,
    "bin_dtype": np.uint8,
    "trig_fns": True,
    "radius": RADIUS_EARTH,
}

initialize_geometry_store(
    geom_fpath=str(OUTFILE),
    origin_lats=origin_lats,
    target_lats=target_lats,
    delta_lons=delta_lons,
    chunk_origin=1,
    chunk_lat=len(target_lats),
    chunk_lon=len(delta_lons),
    dtype=np.float32,
    bin_dtype=np.uint8,
    trig_fns=True,
    distance_edges=distance_edges,
    radius=RADIUS_EARTH,
)

for i0, i1 in iter_chunks(
    len(origin_lats),
    1,
):
    write_geometry_chunk(
        geom_fpath=str(OUTFILE),
        i0=i0,
        i1=i1,
        origin_lats=origin_lats,
        target_lats=target_lats,
        delta_lons=delta_lons,
        distance_edges=distance_edges,
        compute_kwargs=compute_kwargs,
    )

# Add reference metadata
ds = xr.open_zarr(OUTFILE)

ds.attrs.update(
    {
        "reference_version": "1.0",
        "reference_type": "geometry_archive",
        "dtype": "float32",
        "bin_dtype": "uint8",
        "trig_fns": True
    }
)

ds.to_zarr(
    OUTFILE,
    mode="w",
    zarr_format=2,
)

print(f"Wrote {OUTFILE}")
