import numpy as np

from lossett.calc.spherical_geometry import (
    RADIUS_EARTH,
    compute_final_bearing_trig,
    compute_great_circle_distance,
    compute_initial_bearing_trig,
)
from lossett.filtering.spherical_vector_filter import (
    filter_vector_field_tangent_normal,
    geodesic_vector_increment,
)


def spherical_gradient_flow(latitude, speed_scale):
    return 0.0, speed_scale * np.cos(latitude)


def destination(latitude, longitude, bearing, angular_distance):
    target_latitude = np.arcsin(
        np.sin(latitude) * np.cos(angular_distance)
        + np.cos(latitude) * np.sin(angular_distance) * np.cos(bearing)
    )
    target_longitude = longitude + np.arctan2(
        np.sin(bearing)
        * np.sin(angular_distance)
        * np.cos(latitude),
        np.cos(angular_distance)
        - np.sin(latitude) * np.sin(target_latitude),
    )
    return target_latitude, target_longitude


def geometry_from_origin(latitude, longitude, target_latitudes, target_longitudes):
    origin_latitude = latitude
    target_latitude = target_latitudes
    delta_longitude = target_longitudes - longitude
    sin_latitude = np.sin(origin_latitude)
    cos_latitude = np.cos(origin_latitude)
    sin_target_latitude = np.sin(target_latitude)
    cos_target_latitude = np.cos(target_latitude)
    sin_delta_longitude = np.sin(delta_longitude)
    cos_delta_longitude = np.cos(delta_longitude)

    distance = compute_great_circle_distance(
        origin_latitude,
        target_latitude,
        cos_latitude,
        cos_target_latitude,
        delta_longitude,
    )
    sin_initial, cos_initial = compute_initial_bearing_trig(
        sin_delta_longitude,
        cos_delta_longitude,
        sin_latitude,
        cos_latitude,
        sin_target_latitude,
        cos_target_latitude,
    )
    sin_final, cos_final = compute_final_bearing_trig(
        sin_delta_longitude,
        cos_delta_longitude,
        sin_latitude,
        cos_latitude,
        sin_target_latitude,
        cos_target_latitude,
    )
    return distance, sin_initial, cos_initial, sin_final, cos_final


def filtered_vector_at(latitude, longitude, speed_scale, flow, mu, mu_weights, longitudes):
    target_latitudes = np.arcsin(mu)[:, None]
    target_longitudes = longitudes[None, :]
    distance, sin_initial, cos_initial, sin_final, cos_final = geometry_from_origin(
        latitude,
        longitude,
        target_latitudes,
        target_longitudes,
    )

    eastward = np.zeros_like(target_latitudes) + np.zeros_like(target_longitudes)
    northward = speed_scale * np.cos(target_latitudes)
    if flow == "rotated_gradient":
        eastward, northward = -northward, np.zeros_like(northward)
    elif flow != "gradient":
        raise ValueError(f"Unknown flow: {flow}")

    mu_grid = np.linspace(-1, 1, 65537)
    kernel = np.exp(8 * (mu_grid - 1))
    area_weights = (
        RADIUS_EARTH**2
        * mu_weights[:, None]
        * (2 * np.pi / len(longitudes))
    )
    eastward = np.broadcast_to(eastward, distance.shape)
    northward = np.broadcast_to(northward, distance.shape)
    return filter_vector_field_tangent_normal(
        eastward.ravel(),
        northward.ravel(),
        sin_initial.ravel(),
        cos_initial.ravel(),
        sin_final.ravel(),
        cos_final.ravel(),
        kernel,
        np.broadcast_to(area_weights, distance.shape).ravel(),
        mu=np.cos(distance / RADIUS_EARTH).ravel(),
        mu_grid=mu_grid,
    )


def estimate_filtered_divergence_and_curl(
    latitude,
    longitude,
    speed_scale,
    flow,
    angular_step,
    mu,
    mu_weights,
    longitudes,
):
    center = filtered_vector_at(
        latitude, longitude, speed_scale, flow, mu, mu_weights, longitudes
    )
    displacements = {
        "east": (np.pi / 2, 0.0),
        "west": (3 * np.pi / 2, 0.0),
        "north": (0.0, 0.0),
        "south": (np.pi, 0.0),
    }
    filtered_targets = {}
    for name, (bearing, _) in displacements.items():
        target_latitude, target_longitude = destination(
            latitude,
            longitude,
            bearing,
            angular_step,
        )
        filtered_targets[name] = (
            target_latitude,
            target_longitude,
            *filtered_vector_at(
                target_latitude,
                target_longitude,
                speed_scale,
                flow,
                mu,
                mu_weights,
                longitudes,
            ),
        )

    def increment(target):
        target_latitude, target_longitude, eastward, northward = target
        _, sin_initial, cos_initial, sin_final, cos_final = geometry_from_origin(
            latitude,
            longitude,
            target_latitude,
            target_longitude,
        )
        return geodesic_vector_increment(
            center[0],
            center[1],
            eastward,
            northward,
            sin_initial,
            cos_initial,
            sin_final,
            cos_final,
        )

    distance = angular_step * RADIUS_EARTH
    east_derivative = (
        np.asarray(increment(filtered_targets["east"]))
        - np.asarray(increment(filtered_targets["west"]))
    ) / (2 * distance)
    north_derivative = (
        np.asarray(increment(filtered_targets["north"]))
        - np.asarray(increment(filtered_targets["south"]))
    ) / (2 * distance)
    divergence = east_derivative[0] + north_derivative[1]
    vertical_curl = east_derivative[1] - north_derivative[0]
    return center, divergence, vertical_curl


def test_spherical_gradient_and_rotated_gradient_filtering():
    latitude = np.deg2rad(35.0)
    longitude = np.deg2rad(17.0)
    speed_scale = 12.0
    mu, mu_weights = np.polynomial.legendre.leggauss(32)
    longitudes = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    angular_step = 2e-3

    gradient_center, gradient_divergence, gradient_curl = (
        estimate_filtered_divergence_and_curl(
            latitude,
            longitude,
            speed_scale,
            "gradient",
            angular_step,
            mu,
            mu_weights,
            longitudes,
        )
    )
    gradient_scale = gradient_center[1] / (
        speed_scale * np.cos(latitude)
    )
    exact_filtered_divergence = (
        -2
        * gradient_scale
        * speed_scale
        * np.sin(latitude)
        / RADIUS_EARTH
    )

    rotated_center, rotated_divergence, rotated_curl = (
        estimate_filtered_divergence_and_curl(
            latitude,
            longitude,
            speed_scale,
            "rotated_gradient",
            angular_step,
            mu,
            mu_weights,
            longitudes,
        )
    )
    rotated_scale = -rotated_center[0] / (
        speed_scale * np.cos(latitude)
    )
    exact_rotated_curl = (
        -2
        * rotated_scale
        * speed_scale
        * np.sin(latitude)
        / RADIUS_EARTH
    )

    np.testing.assert_allclose(
        gradient_divergence,
        exact_filtered_divergence,
        rtol=2e-4,
    )
    np.testing.assert_allclose(
        gradient_curl,
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rotated_divergence,
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rotated_curl,
        exact_rotated_curl,
        rtol=2e-4,
    )
