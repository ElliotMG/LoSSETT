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

REFERENCE_GEOMETRY = (
    Path(__file__).parent
    / "reference_data"
    / "reference_geometry.zarr"
)

# Helpers

def generate_test_archive(
    geom_fpath,
    origin_lats,
    target_lats,
    delta_lons,
    distance_edges,
):

    initialize_geometry_store(
        geom_fpath=str(geom_fpath),
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
    )

    compute_kwargs = {
        "dtype": np.float32,
        "bin_dtype": np.uint8,
        "trig_fns": True,
        "radius": RADIUS_EARTH,
    }

    for i0, i1 in iter_chunks(
        len(origin_lats),
        1,
    ):
        write_geometry_chunk(
            geom_fpath=str(geom_fpath),
            i0=i0,
            i1=i1,
            origin_lats=origin_lats,
            target_lats=target_lats,
            delta_lons=delta_lons,
            distance_edges=distance_edges,
            compute_kwargs=compute_kwargs,
        )

# Tests

def test_reference_geometry_archive(tmp_path):

    reference = xr.open_zarr(
        REFERENCE_GEOMETRY
    )

    assert reference.attrs["reference_type"] == "geometry_archive"
    assert reference.attrs["reference_version"] == "1.0"

    reference = reference.copy()
    for attr in (
            "reference_version",
            "reference_type",
    ):
        reference.attrs.pop(attr, None)

    distance_edges = np.asarray(
        reference[
            "great_circle_distance_bin"
        ].attrs["distance_bin_edges"]
    )

    candidate_path = (
        tmp_path
        / "candidate_geometry.zarr"
    )

    generate_test_archive(
        candidate_path,
        reference.origin_latitude.values,
        reference.latitude.values,
        reference.longitude.values,
        distance_edges,
    )

    candidate = xr.open_zarr(
        candidate_path
    )

    for var in candidate.data_vars:
        print(var)
        print("candidate attrs:")
        print(candidate[var].attrs)
        print("reference attrs:")
        print(reference[var].attrs)
        print()

        try:
            xr.testing.assert_identical(
                candidate[var],
                reference[var],
            )

        except AssertionError as e:
            print()
            print(f"FAILED: {var}")
            print(e)

            raise

    #xr.testing.assert_identical(
    xr.testing.assert_equal(
        candidate,
        reference,
    )
