import numpy as np

from lossett.calc.spherical_geometry import (
    compute_geometry,
    compute_great_circle_distance,
    compute_distance_bins,
    compute_initial_bearing,
    compute_final_bearing,
    compute_initial_bearing_trig,
    compute_final_bearing_trig,
    compute_angular_weights,
    RADIUS_EARTH
)

RND_SEED = 0

# Helpers

def make_test_lats_lons():
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

    return lats, lons

def make_test_geometry():

    lats, lons = make_test_lats_lons()

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

def bearing_trig(lat0_deg, lat_deg, dlon_deg):

    lat0 = np.deg2rad(lat0_deg)
    lat = np.deg2rad(lat_deg)
    dlon = np.deg2rad(dlon_deg)

    return compute_initial_bearing_trig(
        np.sin(dlon),
        np.cos(dlon),
        np.sin(lat0),
        np.cos(lat0),
        np.sin(lat),
        np.cos(lat),
    )

# Tests

### TOP-LEVEL: SHAPES

def test_compute_geometry_shapes():

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

    expected_shape = (
        len(lats),
        len(lats),
        len(lons),
    )

    for var in ds.data_vars:
        assert ds[var].shape == expected_shape

### ANGULAR WEIGHTS

def test_geometry_weights_valid():

    ds, _ = make_test_geometry()

    weights = ds.angular_weight.values

    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)

def test_geometry_weight_conservation():
    """
    Note that this is the same as ensuring that integrating a constant gives
    2 * pi times that constant.
    """

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

### GREAT CIRCLE DISTANCE

def test_great_circle_distance_zero():

    lat0 = np.deg2rad(0.0)
    lat = np.deg2rad(0.0)
    dlon = np.deg2rad(0.0)

    distance = compute_great_circle_distance(
        lat0,
        lat,
        np.cos(lat0),
        np.cos(lat),
        dlon,
    )

    np.testing.assert_allclose(
        distance,
        0.0,
        atol=1e-12,
    )

def test_great_circle_distance_quarter_circumference():

    lat0 = np.deg2rad(0.0)
    lat = np.deg2rad(0.0)
    dlon = np.deg2rad(90.0)

    distance = compute_great_circle_distance(
        lat0,
        lat,
        np.cos(lat0),
        np.cos(lat),
        dlon,
    )

    np.testing.assert_allclose(
        distance,
        np.pi * RADIUS_EARTH / 2,
        rtol=1e-10,
    )

def test_great_circle_distance_antipodes():

    lat0 = np.deg2rad(0.0)
    lat = np.deg2rad(0.0)
    dlon = np.deg2rad(180.0)

    distance = compute_great_circle_distance(
        lat0,
        lat,
        np.cos(lat0),
        np.cos(lat),
        dlon,
    )

    np.testing.assert_allclose(
        distance,
        np.pi * RADIUS_EARTH,
        rtol=1e-10,
    )

def test_great_circle_distance_symmetric():

    lat_a = np.deg2rad(15.0)
    lon_a = np.deg2rad(40.0)

    lat_b = np.deg2rad(-35.0)
    lon_b = np.deg2rad(120.0)

    dab = compute_great_circle_distance(
        lat_a,
        lat_b,
        np.cos(lat_a),
        np.cos(lat_b),
        lon_b - lon_a,
    )

    dba = compute_great_circle_distance(
        lat_b,
        lat_a,
        np.cos(lat_b),
        np.cos(lat_a),
        lon_a - lon_b,
    )

    np.testing.assert_allclose(
        dab,
        dba,
        rtol=1e-12,
    )

### DISTANCE BINS

def test_distance_bin_assignment():

    distance = np.array([
        5.0,
        15.0,
        25.0,
    ])

    edges = np.array([
        0.0,
        10.0,
        20.0,
        30.0,
    ])

    bins = compute_distance_bins(
        distance,
        edges,
    )

    np.testing.assert_array_equal(
        bins,
        [0, 1, 2],
    )

### BEARINGS

def test_zero_distance_bearings_nan():
    """
    Tests what bearings are assigned to a great-circle distance of 0.
    This should be np.nan; similarly the associated angular weight should be 0.
    """

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

def test_bearing_due_north():

    sin_b, cos_b = bearing_trig(
        0.0,
        10.0,
        0.0,
    )

    np.testing.assert_allclose(
        sin_b,
        0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        cos_b,
        1.0,
        atol=1e-12,
    )

def test_bearing_due_south():

    sin_b, cos_b = bearing_trig(
        10.0,
        0.0,
        0.0,
    )

    np.testing.assert_allclose(
        sin_b,
        0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        cos_b,
        -1.0,
        atol=1e-12,
    )

def test_bearing_due_east():

    sin_b, cos_b = bearing_trig(
        0.0,
        0.0,
        10.0,
    )

    np.testing.assert_allclose(
        sin_b,
        1.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        cos_b,
        0.0,
        atol=1e-12,
    )

def test_bearing_due_west():

    sin_b, cos_b = bearing_trig(
        0.0,
        0.0,
        -10.0,
    )

    np.testing.assert_allclose(
        sin_b,
        -1.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        cos_b,
        0.0,
        atol=1e-12,
    )

def test_initial_bearing_trig_identity():

    lats, lons = make_test_lats_lons()

    lat0 = np.deg2rad(lats)[:, None, None]

    lat = np.deg2rad(lats)[None, :, None]

    dlon = np.deg2rad(lons)[None, None, :]

    sin_b, cos_b = bearing_trig(
        lat0,
        lat,
        dlon
    )

    valid = (
        np.isfinite(sin_b)
        & np.isfinite(cos_b)
    )

    np.testing.assert_allclose(
        sin_b[valid]**2
        + cos_b[valid]**2,
        1.0,
        atol=1e-6,
    )

def test_bearing_reversal_many_points():
    """
    Calculation of initial and final bearings should be symmetric,
    i.e. if chi(A,B) represents the bearing from A to B, then
    
        chi(A,B) = chi(B,A) + pi mod (2 * pi)
    
    We test this using sin(chi), cos(chi) rather than chi directly
    as this makes it easier to handle edge cases.
    """
    rng = np.random.default_rng(RND_SEED)

    for _ in range(100):

        # Avoid poles and pathological cases
        lat_a = np.deg2rad(
            rng.uniform(-80.0, 80.0)
        )
        lat_b = np.deg2rad(
            rng.uniform(-80.0, 80.0)
        )

        lon_a = np.deg2rad(
            rng.uniform(-180.0, 180.0)
        )
        lon_b = np.deg2rad(
            rng.uniform(-180.0, 180.0)
        )

        # don't consider coincident points
        if (
            np.isclose(lat_a, lat_b)
            and np.isclose(lon_a, lon_b)
        ):
            continue

        dlon_ab = lon_b - lon_a
        dlon_ba = lon_a - lon_b

        #
        # A -> B
        #
        sin_i_ab, cos_i_ab = (
            compute_initial_bearing_trig(
                np.sin(dlon_ab),
                np.cos(dlon_ab),
                np.sin(lat_a),
                np.cos(lat_a),
                np.sin(lat_b),
                np.cos(lat_b),
            )
        )

        sin_f_ab, cos_f_ab = (
            compute_final_bearing_trig(
                np.sin(dlon_ab),
                np.cos(dlon_ab),
                np.sin(lat_a),
                np.cos(lat_a),
                np.sin(lat_b),
                np.cos(lat_b),
            )
        )

        #
        # B -> A
        #
        sin_i_ba, cos_i_ba = (
            compute_initial_bearing_trig(
                np.sin(dlon_ba),
                np.cos(dlon_ba),
                np.sin(lat_b),
                np.cos(lat_b),
                np.sin(lat_a),
                np.cos(lat_a),
            )
        )

        sin_f_ba, cos_f_ba = (
            compute_final_bearing_trig(
                np.sin(dlon_ba),
                np.cos(dlon_ba),
                np.sin(lat_b),
                np.cos(lat_b),
                np.sin(lat_a),
                np.cos(lat_a),
            )
        )

        #
        # initial(A->B) = final(B->A) + π
        #
        np.testing.assert_allclose(
            sin_i_ab,
            -sin_f_ba,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            cos_i_ab,
            -cos_f_ba,
            atol=1e-12,
        )

        #
        # final(A->B) = initial(B->A) + π
        #
        np.testing.assert_allclose(
            sin_f_ab,
            -sin_i_ba,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            cos_f_ab,
            -cos_i_ba,
            atol=1e-12,
        )

def test_initial_bearing_matches_trig_functions():

    rng = np.random.default_rng(RND_SEED)

    for _ in range(1000):

        lat0 = np.deg2rad(
            rng.uniform(-80.0, 80.0)
        )

        lat = np.deg2rad(
            rng.uniform(-80.0, 80.0)
        )

        dlon = np.deg2rad(
            rng.uniform(-180.0, 180.0)
        )

        # Skip coincident points
        if (
            np.isclose(lat0, lat)
            and np.isclose(dlon, 0.0)
        ):
            continue

        bearing = compute_initial_bearing(
            np.sin(dlon),
            np.cos(dlon),
            np.sin(lat0),
            np.cos(lat0),
            np.sin(lat),
            np.cos(lat),
        )

        sin_b, cos_b = compute_initial_bearing_trig(
            np.sin(dlon),
            np.cos(dlon),
            np.sin(lat0),
            np.cos(lat0),
            np.sin(lat),
            np.cos(lat),
        )

        valid = (
            np.isfinite(sin_b)
            and np.isfinite(cos_b)
        )

        if not valid:
            continue

        np.testing.assert_allclose(
            np.sin(bearing),
            sin_b,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            np.cos(bearing),
            cos_b,
            atol=1e-12,
        )

def test_final_bearing_matches_trig_functions():

    rng = np.random.default_rng(RND_SEED)

    for _ in range(1000):

        lat0 = np.deg2rad(
            rng.uniform(-80.0, 80.0)
        )

        lat = np.deg2rad(
            rng.uniform(-80.0, 80.0)
        )

        dlon = np.deg2rad(
            rng.uniform(-180.0, 180.0)
        )

        # Skip coincident points
        if (
            np.isclose(lat0, lat)
            and np.isclose(dlon, 0.0)
        ):
            continue

        bearing = compute_final_bearing(
            np.sin(dlon),
            np.cos(dlon),
            np.sin(lat0),
            np.cos(lat0),
            np.sin(lat),
            np.cos(lat),
        )

        sin_b, cos_b = compute_final_bearing_trig(
            np.sin(dlon),
            np.cos(dlon),
            np.sin(lat0),
            np.cos(lat0),
            np.sin(lat),
            np.cos(lat),
        )

        valid = (
            np.isfinite(sin_b)
            and np.isfinite(cos_b)
        )

        if not valid:
            continue

        np.testing.assert_allclose(
            np.sin(bearing),
            sin_b,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            np.cos(bearing),
            cos_b,
            atol=1e-12,
        )

### EDGE CASES

def test_coincident_points():

    lat0 = np.deg2rad(30.0)
    lat = np.deg2rad(30.0)
    dlon = 0.0

    distance = compute_great_circle_distance(
        lat0,
        lat,
        np.cos(lat0),
        np.cos(lat),
        dlon,
    )

    assert distance == 0.0

    sin_b, cos_b = compute_initial_bearing_trig(
        np.sin(dlon),
        np.cos(dlon),
        np.sin(lat0),
        np.cos(lat0),
        np.sin(lat),
        np.cos(lat),
    )

    assert np.isnan(sin_b)
    assert np.isnan(cos_b)
    
    sin_b, cos_b = compute_final_bearing_trig(
        np.sin(dlon),
        np.cos(dlon),
        np.sin(lat0),
        np.cos(lat0),
        np.sin(lat),
        np.cos(lat),
    )

    assert np.isnan(sin_b)
    assert np.isnan(cos_b)

def test_pole_to_pole_distance():

    lat0 = np.deg2rad(90.0)
    lat = np.deg2rad(-90.0)
    dlon = 0.0

    distance = compute_great_circle_distance(
        lat0,
        lat,
        np.cos(lat0),
        np.cos(lat),
        dlon,
    )

    np.testing.assert_allclose(
        distance,
        np.pi * RADIUS_EARTH,
        rtol=1e-12,
    )

def test_longitude_wraparound_distance():

    lat0 = np.deg2rad(20.0)
    lat = np.deg2rad(-10.0)

    dist_pos = compute_great_circle_distance(
        lat0,
        lat,
        np.cos(lat0),
        np.cos(lat),
        np.deg2rad(180.0),
    )

    dist_neg = compute_great_circle_distance(
        lat0,
        lat,
        np.cos(lat0),
        np.cos(lat),
        np.deg2rad(-180.0),
    )

    np.testing.assert_allclose(
        dist_pos,
        dist_neg,
        rtol=1e-12,
    )

def test_initial_bearing_pm180_consistent_trig():

    lat0 = np.deg2rad(20.0)
    lat = np.deg2rad(-10.0)

    sin_pos, cos_pos = compute_initial_bearing_trig(
        np.sin(np.deg2rad(180.0)),
        np.cos(np.deg2rad(180.0)),
        np.sin(lat0),
        np.cos(lat0),
        np.sin(lat),
        np.cos(lat),
    )

    sin_neg, cos_neg = compute_initial_bearing_trig(
        np.sin(np.deg2rad(-180.0)),
        np.cos(np.deg2rad(-180.0)),
        np.sin(lat0),
        np.cos(lat0),
        np.sin(lat),
        np.cos(lat),
    )

    np.testing.assert_allclose(
        cos_pos,
        cos_neg,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        sin_pos,
        sin_neg,
        atol=1e-12,
    )

def test_initial_bearing_pm180_consistent():

    lat0 = np.deg2rad(20.0)
    lat = np.deg2rad(-10.0)

    bearing_pos = compute_initial_bearing(
        np.sin(np.deg2rad(180.0)),
        np.cos(np.deg2rad(180.0)),
        np.sin(lat0),
        np.cos(lat0),
        np.sin(lat),
        np.cos(lat),
    )

    bearing_neg = compute_initial_bearing(
        np.sin(np.deg2rad(-180.0)),
        np.cos(np.deg2rad(-180.0)),
        np.sin(lat0),
        np.cos(lat0),
        np.sin(lat),
        np.cos(lat),
    )

    np.testing.assert_allclose(
        bearing_pos,
        bearing_neg,
        atol=1e-12,
    )

def test_angular_weights_small_bin_uniform():

    distance_bin = np.zeros(
        (2, 2),
        dtype=np.int32,
    )

    chi = np.array([
        [0.0,        np.pi / 2],
        [np.pi,      3 * np.pi / 2],
    ])

    sin_chi = np.sin(chi)
    cos_chi = np.cos(chi)

    weights = compute_angular_weights(
        distance_bin,
        sin_chi,
        cos_chi,
        nbins=1,
    )

    expected = np.full(
        (2, 2),
        2 * np.pi / 4,
    )

    np.testing.assert_allclose(
        weights,
        expected,
    )

    np.testing.assert_allclose(
        weights.sum(),
        2 * np.pi,
    )
