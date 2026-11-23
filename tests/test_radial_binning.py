import numpy as np
import xarray as xr

from lossett.calc.angular_integration import angular_integral_by_distance_bin
from lossett.calc.field_increments import (
    compute_delta_u_cubed,
    compute_delta_u_cubed_tangent_plane,
    compute_du3_angular_integral_global,
)
from lossett.calc.spherical_geometry import (
    RADIUS_EARTH,
    build_distance_bins,
    compute_geometry,
    compute_angular_weights,
)
from lossett.calc.compute_inter_scale_transfers_spherical import (
    compute_inter_scale_kinetic_energy_transfer,
)
from lossett.filtering.kernels import filter_kernel


def spherical_mollifier_derivative(r, length_scale):
    _, derivative = filter_kernel(
        length_scale,
        r,
        return_derivative=True,
        normalization="sphere",
        sphere_radius=RADIUS_EARTH,
    )
    return derivative


def make_affine_shell_samples(nbins, radial_samples_per_bin, n_angles, length_scale):
    edges, distances = build_distance_bins(
        nbins,
        max_r=2 * length_scale,
    )
    fractions = (
        np.arange(radial_samples_per_bin) + 0.5
    ) / radial_samples_per_bin
    radii = (
        edges[:-1, None]
        + fractions[None, :] * np.diff(edges)[:, None]
    )
    radii = np.repeat(radii.ravel(), n_angles)
    angles = np.tile(
        np.linspace(0, 2 * np.pi, n_angles, endpoint=False),
        nbins * radial_samples_per_bin,
    )
    bins = np.repeat(
        np.arange(nbins),
        radial_samples_per_bin * n_angles,
    )

    sin_bearing = np.sin(angles)
    cos_bearing = np.cos(angles)
    angular_weights = compute_angular_weights(
        bins[None, :],
        sin_bearing[None, :],
        cos_bearing[None, :],
        nbins=nbins,
        dtype=np.float64,
    )[0]

    east = radii * sin_bearing
    north = radii * cos_bearing
    return (
        distances,
        bins,
        sin_bearing,
        cos_bearing,
        angular_weights,
        east,
        north,
    )


def affine_du3(field, east, north, sin_bearing, cos_bearing, amplitude=1.0):
    if field == "divergence":
        u = amplitude * east
        v = amplitude * north
    elif field == "strain":
        u = amplitude * east
        v = -amplitude * north
    elif field == "rotation":
        u = -amplitude * north
        v = amplitude * east
    else:
        raise ValueError(f"Unknown affine field: {field}")

    du3, _ = compute_delta_u_cubed(
        u,
        v,
        0.0,
        0.0,
        sin_bearing,
        cos_bearing,
        sin_bearing,
        cos_bearing,
    )
    return du3


def binned_transfer(nbins, radial_samples_per_bin, field, length_scale):
    (
        distances,
        bins,
        sin_bearing,
        cos_bearing,
        angular_weights,
        east,
        north,
    ) = make_affine_shell_samples(
        nbins,
        radial_samples_per_bin,
        n_angles=64,
        length_scale=length_scale,
    )
    du3 = affine_du3(
        field,
        east,
        north,
        sin_bearing,
        cos_bearing,
    )
    shell_integral = angular_integral_by_distance_bin(
        du3,
        bins,
        nbins,
        weights=angular_weights,
    )
    dG_dr = spherical_mollifier_derivative(distances, length_scale)
    radial_area = RADIUS_EARTH * np.sin(distances / RADIUS_EARTH)
    return 0.25 * np.trapezoid(
        shell_integral * dG_dr * radial_area,
        x=distances,
    )


def direct_divergence_transfer(nradial, length_scale):
    """Evaluate the spherical radial integral at high resolution."""
    distances = np.linspace(
        0.0,
        2 * length_scale,
        nradial,
        endpoint=False,
    )
    dG_dr = spherical_mollifier_derivative(distances, length_scale)
    radial_area = RADIUS_EARTH * np.sin(distances / RADIUS_EARTH)
    angular_integral = 2 * np.pi * distances**3
    return 0.25 * np.trapezoid(
        angular_integral * dG_dr * radial_area,
        x=distances,
    )


def tangent_plane_binned_transfer(nbins, radial_samples_per_bin, length_scale):
    (
        distances,
        bins,
        sin_bearing,
        cos_bearing,
        angular_weights,
        east,
        north,
    ) = make_affine_shell_samples(
        nbins,
        radial_samples_per_bin,
        n_angles=64,
        length_scale=length_scale,
    )
    du3 = compute_delta_u_cubed_tangent_plane(
        east,
        north,
        0.0,
        0.0,
        sin_bearing,
        cos_bearing,
    )
    shell_integral = angular_integral_by_distance_bin(
        du3,
        bins,
        nbins,
        weights=angular_weights,
    )
    _, derivative = filter_kernel(
        length_scale,
        distances,
        return_derivative=True,
        normalization="2D",
    )
    return 0.25 * np.trapezoid(
        shell_integral * derivative * distances,
        x=distances,
    )


def direct_tangent_plane_divergence_transfer(nradial, length_scale):
    distances = np.linspace(
        0.0,
        2 * length_scale,
        nradial,
        endpoint=False,
    )
    _, derivative = filter_kernel(
        length_scale,
        distances,
        return_derivative=True,
        normalization="2D",
    )
    exact_angular_integral = 2 * np.pi * distances**3
    return 0.25 * np.trapezoid(
        exact_angular_integral * derivative * distances,
        x=distances,
    )


def test_spherical_transfer_uses_spherical_radial_area():
    length_scale = 0.4 * RADIUS_EARTH
    r = np.linspace(
        0.0,
        2 * length_scale,
        128,
        endpoint=False,
    )
    du3_long = xr.DataArray(
        np.ones((len(r), 1, 1)),
        dims=("r", "origin_latitude", "origin_longitude"),
        coords={
            "r": r,
            "origin_latitude": [0.0],
            "origin_longitude": [0.0],
        },
    )
    du3_vert = xr.DataArray(
        2 * np.ones((len(r), 1, 1)),
        dims=("r", "origin_latitude", "origin_longitude"),
        coords={
            "r": r,
            "origin_latitude": [0.0],
            "origin_longitude": [0.0],
        },
    )
    result = compute_inter_scale_kinetic_energy_transfer(
        du3_long,
        [length_scale],
        du3_vert=du3_vert,
        ratio_rmax_to_ell=2.0,
        ratio_L_to_ell=2.0,
    )
    kernel, derivative = filter_kernel(
        length_scale,
        r,
        return_derivative=True,
        normalization="sphere",
        sphere_radius=RADIUS_EARTH,
    )
    radial_area = RADIUS_EARTH * np.sin(r / RADIUS_EARTH)
    expected_long = 0.25 * np.trapezoid(
        derivative * radial_area,
        x=r,
    )
    expected_vert = -0.5 * np.trapezoid(
        kernel * radial_area,
        x=r,
    )

    np.testing.assert_allclose(
        result.DL_u_longitudinal.values.squeeze(),
        expected_long,
    )
    np.testing.assert_allclose(
        result.DL_u_vert_transverse_times_dz.values.squeeze(),
        expected_vert,
    )


def test_tangent_plane_affine_divergence_transfer_converges():
    length_scale = 1.0
    reference = direct_tangent_plane_divergence_transfer(
        nradial=8193,
        length_scale=length_scale,
    )
    coarse = tangent_plane_binned_transfer(
        nbins=16,
        radial_samples_per_bin=4,
        length_scale=length_scale,
    )
    fine = tangent_plane_binned_transfer(
        nbins=256,
        radial_samples_per_bin=4,
        length_scale=length_scale,
    )

    assert reference != 0.0
    assert abs(fine - reference) < abs(coarse - reference)
    np.testing.assert_allclose(
        fine,
        reference,
        rtol=2e-3,
    )


def test_pure_strain_and_rotation_have_zero_shell_integrals():
    """Keep the angular controls separate from radial-bin convergence."""
    (
        _,
        bins,
        sin_bearing,
        cos_bearing,
        angular_weights,
        east,
        north,
    ) = make_affine_shell_samples(
        nbins=16,
        radial_samples_per_bin=4,
        n_angles=64,
        length_scale=1.0,
    )

    for field in ("strain", "rotation"):
        du3 = affine_du3(
            field,
            east,
            north,
            sin_bearing,
            cos_bearing,
        )
        shell_integral = angular_integral_by_distance_bin(
            du3,
            bins,
            nbins=16,
            weights=angular_weights,
        )

        np.testing.assert_allclose(
            shell_integral,
            0.0,
            atol=1e-12,
        )


def test_pure_divergence_shell_integral_is_exact_angularly():
    (
        _,
        bins,
        sin_bearing,
        cos_bearing,
        angular_weights,
        east,
        north,
    ) = make_affine_shell_samples(
        nbins=16,
        radial_samples_per_bin=4,
        n_angles=64,
        length_scale=1.0,
    )
    du3 = affine_du3(
        "divergence",
        east,
        north,
        sin_bearing,
        cos_bearing,
    )
    shell_integral = angular_integral_by_distance_bin(
        du3,
        bins,
        nbins=16,
        weights=angular_weights,
    )

    edges, _ = build_distance_bins(
        16,
        max_r=2.0,
    )
    fractions = (np.arange(4) + 0.5) / 4
    radii = (
        edges[:-1, None]
        + fractions[None, :] * np.diff(edges)[:, None]
    )
    exact_shell_integral = 2 * np.pi * np.mean(radii**3, axis=1)

    np.testing.assert_allclose(
        shell_integral,
        exact_shell_integral,
        rtol=1e-12,
    )


def test_divergence_transfer_converges_as_radial_bins_are_refined():
    """Check finite-width shells against a direct radial integral."""
    length_scale = 1.0
    reference = direct_divergence_transfer(
        nradial=8193,
        length_scale=length_scale,
    )
    coarse = binned_transfer(
        nbins=16,
        radial_samples_per_bin=4,
        field="divergence",
        length_scale=length_scale,
    )
    fine = binned_transfer(
        nbins=256,
        radial_samples_per_bin=4,
        field="divergence",
        length_scale=length_scale,
    )

    assert reference != 0.0
    assert abs(fine - reference) < abs(coarse - reference)
    np.testing.assert_allclose(
        fine,
        reference,
        rtol=2e-3,
    )


def test_full_transfer_for_curl_free_spherical_gradient_flow():
    origin_latitude = np.deg2rad(35.0)
    origin_longitude = 0.25
    speed_scale = 12.0
    length_scale = 1.0e6
    target_latitudes = np.arange(-89.5, 90.0, 1.0)
    target_longitudes = np.arange(-179.5, 180.0, 1.0)
    delta_longitudes = target_longitudes - origin_longitude
    origin_latitudes = np.rad2deg([origin_latitude])
    edges, distances = build_distance_bins(180)

    geometry = compute_geometry(
        origin_latitudes,
        target_latitudes,
        delta_longitudes,
        edges,
        dtype=np.float64,
        bin_dtype=np.uint8,
        trig_fns=True,
    )
    u = xr.DataArray(
        np.zeros((len(target_latitudes), len(target_longitudes))),
        dims=("latitude", "longitude"),
        coords={
            "latitude": target_latitudes,
            "longitude": target_longitudes,
        },
    )
    v = xr.DataArray(
        speed_scale
        * np.broadcast_to(
            np.cos(np.deg2rad(target_latitudes))[:, None],
            (len(target_latitudes), len(target_longitudes)),
        ),
        dims=("latitude", "longitude"),
        coords=u.coords,
    )
    u0 = xr.DataArray(
        [0.0],
        dims=("origin_latitude",),
        coords={"origin_latitude": origin_latitudes},
    )
    v0 = xr.DataArray(
        [speed_scale * np.cos(origin_latitude)],
        dims=("origin_latitude",),
        coords={"origin_latitude": origin_latitudes},
    )
    shell_integrals, _ = compute_du3_angular_integral_global(
        u,
        v,
        u0,
        v0,
        geometry,
        len(distances),
        use_angular_weights=True,
    )
    du3_long = xr.DataArray(
        shell_integrals.reshape(1, 1, -1),
        dims=("origin_latitude", "origin_longitude", "r"),
        coords={
            "origin_latitude": origin_latitudes,
            "origin_longitude": [origin_longitude],
            "r": distances,
        },
    )

    transfer = compute_inter_scale_kinetic_energy_transfer(
        du3_long,
        [length_scale],
        ratio_rmax_to_ell=2.0,
        ratio_L_to_ell=2.0,
    )

    theta = distances / RADIUS_EARTH
    a = -np.sin(origin_latitude) * np.sin(theta)
    b = np.cos(origin_latitude) * (np.cos(theta) - 1)
    exact_angular_integral = 2 * np.pi * speed_scale**3 * (
        a**3 + 1.5 * a * b**2
    )
    derivative = spherical_mollifier_derivative(distances, length_scale)
    radial_area = RADIUS_EARTH * np.sin(theta)
    within_support = distances <= 2 * length_scale
    exact_transfer = 0.25 * np.trapezoid(
        exact_angular_integral[within_support]
        * derivative[within_support]
        * radial_area[within_support],
        x=distances[within_support],
    )

    assert exact_transfer != 0.0
    np.testing.assert_allclose(
        transfer.DL_u_longitudinal.values.squeeze(),
        exact_transfer,
        rtol=0.03,
    )
