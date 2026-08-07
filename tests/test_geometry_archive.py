import numpy as np
import xarray as xr

from lossett.calc.spherical_geometry import (
    compute_geometry,
    RADIUS_EARTH
)
from lossett.calc.compute_spherical_geometry import(
    initialize_geometry_store,
    create_geometry_template,
    write_geometry_chunk,
    iter_chunks
)

# Helpers

def assert_metadata_equal(expected, actual):

    assert actual.attrs == expected.attrs

    for var in expected.data_vars:

        assert var in actual

        assert (
            actual[var].attrs
            == expected[var].attrs
        ), (
            f"Metadata mismatch for variable "
            f"{var}"
        )

# Tests

def test_archive_vs_direct_metadata(tmp_path):
    origin_lats = np.array([
        -30.,
        0.,
        30.,
    ])
    target_lats = origin_lats
    delta_lons = np.array([
        -90.,
        0.,
        90.,
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
        "radius": RADIUS_EARTH
    }

    # Reference geometry
    ds_ref = compute_geometry(
        origin_lats[:1],
        target_lats[:1],
        delta_lons[:1],
        distance_edges,
        **compute_kwargs,
    )

    # Create archive
    geom_fpath = tmp_path / "test.zarr"

    initialize_geometry_store(
        geom_fpath=str(geom_fpath),
        origin_lats=origin_lats,
        target_lats=target_lats,
        delta_lons=delta_lons,
        chunk_origin=1,
        chunk_lat=len(target_lats),
        chunk_lon=len(delta_lons),
        distance_edges=distance_edges,
        **compute_kwargs,
    )

    ds_archive = xr.open_zarr(
        geom_fpath,
    )

    assert_metadata_equal(
        ds_ref,
        ds_archive,
    )

def test_archived_vs_direct_geometry(tmp_path):
    """
    Test whether archived geometry bit-compares with saved
    Zarr archive.
    """

    origin_lats = np.array([
        -30.0,
        0.0,
        30.0,
    ])

    target_lats = np.array([
        -30.0,
        0.0,
        30.0,
    ])

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
    }

    # Direct computation
    ds_direct = compute_geometry(
        origin_lats,
        target_lats,
        delta_lons,
        distance_edges,
        **compute_kwargs,
    )

    # Chunked archive computation
    geom_fpath = tmp_path / "test_geometry.zarr"

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

    ds_chunked = xr.open_zarr(geom_fpath)

    # Compare all data variables
    for var in ds_direct.data_vars:
        if np.issubdtype(
                ds_direct[var].dtype,
                np.integer,
        ):
            xr.testing.assert_equal(
                ds_chunked[var],
                ds_direct[var],
            )
        else:
            xr.testing.assert_equal(
                ds_chunked[var],
                ds_direct[var],
            )

    # Compare all coordinates
    for coord in ds_direct.coords:
        xr.testing.assert_equal(
            ds_chunked.coords[coord],
            ds_direct.coords[coord]
        )

def test_geometry_template_chunking():
    chunk_origin = 2
    chunk_lat = 5
    chunk_lon = 6
    template = create_geometry_template(
        origin_latitudes=np.arange(4),
        target_latitudes=np.arange(5),
        delta_longitudes=np.arange(6),
        chunk_origin=chunk_origin,
        chunk_lat=chunk_lat,
        chunk_lon=chunk_lon,
        distance_edges=np.arange(10),
    )

    assert (
        template["great_circle_distance"]
        .data.chunksize
        == (chunk_origin, chunk_lat, chunk_lon)
    )
