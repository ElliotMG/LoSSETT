import numpy as np

from lossett.calc.compute_spherical_geometry import (
    compute_geometry,
    RADIUS_EARTH
)

# Helpers

def make_test_geometry():

    lats = np.array([
        -60.,
        -30.,
        0.,
        30.,
        60.,
    ])

    lons = np.arange(
        -180.,
        180.,
        30.,
    )

    distance_edges = np.linspace(
        0.,
        np.pi * RADIUS_EARTH,
        16,
    )

    ds = compute_geometry(
        lats,
        lats,
        lons,
        distance_edges,
        trig_fns=True,
    )

    return ds, distance_edges

def make_test_geometry_fine_grid():

    lats = np.arange(-90, 91, 2)
    lons = np.arange(-180, 180, 2)

    distance_edges = np.linspace(
        0,
        np.pi * 6371000,
        64,
    )

    ds = compute_geometry(
        lats,
        lats,
        lons,
        distance_edges,
        trig_fns=True,
    )

    return ds, distance_edges

# Tests

def test_geometry_weight_conservation():

    ds, distance_edges = make_test_geometry()

    nbins = len(distance_edges) - 1

    for ilat0 in range(
        ds.sizes["origin_latitude"]
    ):

        bins = (
            ds.great_circle_distance_bin
            .isel(origin_latitude=ilat0)
            .values
        )

        weights = (
            ds.angular_weight
            .isel(origin_latitude=ilat0)
            .values
        )

        sinchi = (
            ds.sine_initial_bearing
            .isel(origin_latitude=ilat0)
            .values
        )

        coschi = (
            ds.cosine_initial_bearing
            .isel(origin_latitude=ilat0)
            .values
        )

        valid = (
            np.isfinite(sinchi)
            & np.isfinite(coschi)
        )

        for ibin in range(nbins):

            mask = (
                (bins == ibin)
                & valid
            )

            print(f"ibin = {ibin}")
            
            print(
                "weights in mask:",
                weights[mask].sum(),
            )
            print(
                "weights outside mask:",
                weights[(bins == ibin) & ~valid].sum(),
            )

            if np.any(mask):

                assert np.isclose(
                    weights[mask].sum(),
                    2*np.pi,
                    atol=1e-4,
                )

def test_geometry_weights_finite():

    ds, _ = make_test_geometry()

    weights = ds.angular_weight.values

    assert np.all(
        np.isfinite(weights)
    )

def test_geometry_weights_nonnegative():

    ds, _ = make_test_geometry()

    weights = ds.angular_weight.values

    assert np.all(weights >= 0.)

def test_geometry_constant_integral():

    ds, distance_edges = make_test_geometry()

    nbins = len(distance_edges) - 1

    for ilat0 in range(
        ds.sizes["origin_latitude"]
    ):

        bins = (
            ds.great_circle_distance_bin
            .isel(origin_latitude=ilat0)
            .values
        )

        weights = (
            ds.angular_weight
            .isel(origin_latitude=ilat0)
            .values
        )

        valid = np.isfinite(
            ds.sine_initial_bearing
            .isel(origin_latitude=ilat0)
            .values
        )

        for ibin in range(nbins):

            mask = (
                (bins == ibin)
                & valid
            )

            if np.any(mask):

                integral = np.sum(
                    weights[mask]
                )

                assert np.isclose(
                    integral,
                    2*np.pi,
                    atol=1e-4,
                )

def test_geometry_sine_integral():

    ds, distance_edges = make_test_geometry_fine_grid()

    nbins = len(distance_edges) - 1

    for ilat0 in range(
            ds.sizes["origin_latitude"]
    ):
                
        bins = (
            ds.great_circle_distance_bin
            .isel(origin_latitude=ilat0)
            .values
        )

        weights = (
            ds.angular_weight
            .isel(origin_latitude=ilat0)
            .values
        )

        sinchi = (
            ds.sine_initial_bearing
            .isel(origin_latitude=ilat0)
            .values
        )

        coschi = (
            ds.cosine_initial_bearing
            .isel(origin_latitude=ilat0)
            .values
        )

        valid = (
            np.isfinite(sinchi)
            & np.isfinite(coschi)
        )

        for ibin in range(nbins):

            mask = (
                (bins == ibin)
                & valid
            )

            if mask.sum() < 50:
                # reasonable angular sampling is required for
                # this test to be meaningful
                continue

            integral = np.sum(
                sinchi[mask]
                * weights[mask]
            )

            assert abs(integral) < 1e-2

def test_zero_distance_bearings_nan():

    ds, _ = make_test_geometry()

    zero_distance = (
        ds.great_circle_distance.values == 0.0
    )

    assert np.any(
        zero_distance
    )

    assert np.all(
        np.isnan(
            ds.sine_initial_bearing.values[
                zero_distance
            ]
        )
    )

    assert np.all(
        np.isnan(
            ds.cosine_initial_bearing.values[
                zero_distance
            ]
        )
    )

    assert np.all(
        np.isnan(
            ds.sine_final_bearing.values[
                zero_distance
            ]
        )
    )

    assert np.all(
        np.isnan(
            ds.cosine_final_bearing.values[
                zero_distance
            ]
        )
    )

    assert np.all(
        ds.angular_weight.values[
            zero_distance
        ] == 0.0
    )
