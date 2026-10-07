using LoSSETT
using Test

@testset "LoSSETT package load" begin
    @test isdefined(LoSSETT, :kinetic_energy_transfer)
end

@testset "LoSSETT kinetic energy transfer" begin
    @testset "independent radial/angular reference" begin
        x = collect(0.0:1.0:4.0)
        y = collect(0.0:1.0:4.0)
        u = zeros(5, 5)
        v = repeat(reshape(y, 5, 1), 1, 5)
        w = zeros(5, 5)
        result = kinetic_energy_transfer(
            u, v, w, x, y, [1.0];
            max_radius=3.0, periodic=(false, false),
        )

        # Direct hand calculation for the eight offsets in the r=1 annulus.
        angles = [-3π / 4, -π / 2, -π / 4, 0.0, π / 4, π / 2, 3π / 4, π]
        angular_values = [inv(sqrt(2.0)), 1.0, inv(sqrt(2.0)), 0.0,
                          inv(sqrt(2.0)), 1.0, inv(sqrt(2.0)), 0.0]
        angular_reference = sum(
            (angles[i + 1] - angles[i]) * (angular_values[i] + angular_values[i + 1]) / 2
            for i in 1:(length(angles) - 1)
        )
        expected = -angular_reference / (9π)
        @test result.transfer[1, 3, 3] ≈ expected rtol=1e-12
        @test result.length_scales == [1.0]
        @test result.radii == [0.0, 1.0]
    end

    @testset "Spherical LoSSETT geometry" begin
        earth_radius = 6_371_000.0

        @testset "great-circle references and Cartesian limit" begin
            one_degree = deg2rad(1.0)
            distance, sin_initial, cos_initial, sin_final, cos_final =
                LoSSETT._great_circle_geometry(0.0, 0.0, one_degree, earth_radius)
            @test distance ≈ earth_radius * one_degree rtol=1e-13
            @test atan(sin_initial, cos_initial) ≈ π / 2 atol=1e-13
            @test atan(sin_final, cos_final) ≈ π / 2 atol=1e-13

            north_distance, north_sin, north_cos, _, _ =
                LoSSETT._great_circle_geometry(0.0, one_degree, 0.0, earth_radius)
            @test north_distance ≈ earth_radius * one_degree rtol=1e-13
            @test atan(north_sin, north_cos) ≈ 0.0 atol=1e-13

            latitude0 = deg2rad(40.0)
            delta_latitude = deg2rad(1e-5)
            delta_longitude = deg2rad(2e-5)
            distance, sin_bearing, cos_bearing, _, _ =
                LoSSETT._great_circle_geometry(
                    latitude0, latitude0 + delta_latitude,
                    delta_longitude, earth_radius,
                )
            east = earth_radius * cos(latitude0) * delta_longitude
            north = earth_radius * delta_latitude
            @test distance ≈ hypot(east, north) rtol=1e-9
            @test atan(sin_bearing, cos_bearing) ≈ atan(east, north) atol=1e-9

            seam_distance, seam_sin, seam_cos, _, _ =
                LoSSETT._great_circle_geometry(
                    0.0, 0.0, deg2rad(358.0), earth_radius,
                )
            @test seam_distance ≈ earth_radius * deg2rad(2.0) rtol=1e-13
            @test atan(seam_sin, seam_cos) ≈ -π / 2 atol=1e-13
            _, anti_sin, _, _, _ =
                LoSSETT._great_circle_geometry(0.0, 0.0, π, earth_radius)
            @test isnan(anti_sin)
            pole_distance, pole_sin, _, _, _ =
                LoSSETT._great_circle_geometry(
                    π / 2, π / 2, π / 2, earth_radius,
                )
            @test pole_distance ≈ 0.0 atol=1e-8
            @test isnan(pole_sin)
        end

        @testset "spherical area measure and mollifier normalization" begin
            @test LoSSETT._spherical_area_jacobian(0.0, earth_radius) == 0.0
            @test LoSSETT._spherical_area_jacobian(π * earth_radius / 2, earth_radius) ≈ earth_radius
            @test LoSSETT._spherical_area_jacobian(1.0, earth_radius) ≈ 1.0 rtol=1e-14

            radii = collect(range(0.0, 2_000.0; length=2_001))
            ell = 500.0
            normalization, _, jacobian =
                LoSSETT._spherical_mollifier(radii, ell, earth_radius)
            raw = [LoSSETT._mollifier(r, ell) for r in radii]
            @test 2π * LoSSETT._trapz(jacobian .* raw .* normalization, radii) ≈ 1.0 rtol=1e-12
            @test LoSSETT._spherical_angular_integral(
                ones(8), collect(range(0.0; step=π / 4, length=8)), true,
            ) ≈ 2π rtol=1e-12
        end

        @testset "spherical entry point, dimensions, and boundaries" begin
            longitude = collect(-0.02:0.01:0.02)
            latitude = collect(-0.02:0.01:0.02)
            u = zeros(5, 5)
            v = zeros(5, 5)
            w = zeros(5, 5)
            grid_step = earth_radius * deg2rad(0.01)
            result = kinetic_energy_transfer(
                u, v, w, longitude, latitude, [grid_step];
                max_radius=2.5 * grid_step, geometry=:spherical,
            )
            @test size(result.transfer) == (1, 5, 5)
            @test result.length_scales == [grid_step]
            @test all(iszero, result.transfer)
            @test first(result.radii) > 0

            local_longitude = [-0.01, 0.0, 0.01]
            local_latitude = [-0.01, 0.0, 0.01]
            eastward = repeat(reshape([-1.0, 0.0, 1.0], 1, 3), 3, 1)
            northward = repeat(reshape([-1.0, 0.0, 1.0], 3, 1), 1, 3)
            direct = kinetic_energy_transfer(
                eastward, northward, zeros(3, 3),
                local_longitude, local_latitude, [grid_step];
                max_radius=2.5 * grid_step, geometry=:spherical,
            )
            expected_radii = direct.radii
            radial_step = expected_radii[2] - expected_radii[1]
            angular_values = [Float64[] for _ in expected_radii]
            for iy in 1:3, ix in 1:3
                (iy == 2 && ix == 2) && continue
                distance, sin_initial, cos_initial, sin_final, cos_final =
                    LoSSETT._great_circle_geometry(
                        0.0, deg2rad(local_latitude[iy]),
                        deg2rad(local_longitude[ix]), earth_radius,
                    )
                bin = floor(Int, distance / radial_step) + 1
                du_t = eastward[iy, ix] * sin_final +
                       northward[iy, ix] * cos_final
                du_n = eastward[iy, ix] * cos_final -
                       northward[iy, ix] * sin_final
                push!(angular_values[bin], du_t * (du_t^2 + du_n^2))
            end
            expected_angular = [
                isempty(values) ? 0.0 : 2π * sum(values) / length(values)
                for values in angular_values
            ]
            _, derivative, jacobian =
                LoSSETT._spherical_mollifier(expected_radii, grid_step, earth_radius)
            expected_transfer =
                LoSSETT._trapz(derivative .* jacobian .* expected_angular, expected_radii) / 4
            @test direct.transfer[1, 2, 2] ≈ expected_transfer rtol=1e-12

            quadratic_longitude = [-0.02, 0.0, 0.02]
            quadratic_latitude = [40.0, 40.01, 40.02]
            quadratic_u = [
                0.2 * ix - 0.3 * iy + 0.1 * ix * iy
                for iy in 1:3, ix in 1:3
            ]
            quadratic_v = [
                -0.1 * ix + 0.5 * iy + 0.07 * ix^2
                for iy in 1:3, ix in 1:3
            ]
            quadratic_w = [
                0.15 * ix - 0.2 * iy + 0.03 * ix * iy
                for iy in 1:3, ix in 1:3
            ]
            quadratic_step = earth_radius * deg2rad(0.02)
            quadratic = kinetic_energy_transfer(
                quadratic_u, quadratic_v, quadratic_w,
                quadratic_longitude, quadratic_latitude, [quadratic_step];
                max_radius=2.5 * quadratic_step,
                geometry=:tangent_quadratic,
            )

            quadratic_radii = quadratic.radii
            quadratic_dr = quadratic_radii[2] - quadratic_radii[1]
            quadratic_bins = [Float64[] for _ in quadratic_radii]
            origin_latitude = deg2rad(quadratic_latitude[2])
            for iy in 1:3, ix in 1:3
                (iy == 2 && ix == 2) && continue
                latitude_rad = deg2rad(quadratic_latitude[iy])
                delta_longitude = deg2rad(
                    quadratic_longitude[ix] - quadratic_longitude[2],
                )
                haversine =
                    sin((latitude_rad - origin_latitude) / 2)^2 +
                    cos(origin_latitude) * cos(latitude_rad) *
                    sin(delta_longitude / 2)^2
                distance = 2 * earth_radius *
                           atan(sqrt(haversine), sqrt(1 - haversine))
                sine_bearing = sin(delta_longitude) * cos(latitude_rad)
                cosine_bearing =
                    cos(origin_latitude) * sin(latitude_rad) -
                    sin(origin_latitude) * cos(latitude_rad) *
                    cos(delta_longitude)
                bearing_norm = hypot(sine_bearing, cosine_bearing)
                sine_bearing /= bearing_norm
                cosine_bearing /= bearing_norm
                delta_alpha = distance * sine_bearing *
                              tan(origin_latitude) / earth_radius
                delta_u = quadratic_u[iy, ix] - quadratic_u[2, 2]
                delta_v = quadratic_v[iy, ix] - quadratic_v[2, 2]
                delta_t = delta_u * sine_bearing + delta_v * cosine_bearing +
                          delta_alpha * (
                              quadratic_u[iy, ix] * cosine_bearing -
                              quadratic_v[iy, ix] * sine_bearing
                          )
                delta_n = delta_u * cosine_bearing - delta_v * sine_bearing -
                          delta_alpha * (
                              quadratic_u[iy, ix] * sine_bearing +
                              quadratic_v[iy, ix] * cosine_bearing
                          )
                delta_w = quadratic_w[iy, ix] - quadratic_w[2, 2]
                bin = floor(Int, distance / quadratic_dr) + 1
                push!(
                    quadratic_bins[bin],
                    delta_t * (delta_t^2 + delta_n^2 + delta_w^2),
                )
            end
            quadratic_angular = [
                isempty(values) ? 0.0 : 2π * sum(values) / length(values)
                for values in quadratic_bins
            ]
            _, quadratic_derivative, quadratic_jacobian =
                LoSSETT._spherical_mollifier(
                    quadratic_radii, quadratic_step, earth_radius,
                )
            quadratic_reference = LoSSETT._trapz(
                quadratic_derivative .* quadratic_jacobian .* quadratic_angular,
                quadratic_radii,
            ) / 4
            @test quadratic.transfer[1, 2, 2] ≈ quadratic_reference rtol=1e-12
            @test quadratic.length_scales == [quadratic_step]

            batched = kinetic_energy_transfer(
                zeros(2, 5, 5), zeros(2, 5, 5), zeros(2, 5, 5),
                longitude, latitude, [grid_step];
                max_radius=2.5 * grid_step, geometry="spherical",
                xdim=3, ydim=2,
            )
            @test size(batched.transfer) == (1, 2, 5, 5)
            @test all(iszero, batched.transfer)

            @test_throws ArgumentError kinetic_energy_transfer(
                u, v, w, longitude, latitude, [grid_step];
                max_radius=π * earth_radius + 1, geometry=:spherical,
            )
            @test_throws ArgumentError kinetic_energy_transfer(
                u, v, w, longitude, [0.0, 0.01, 90.01, 90.02, 90.03], [grid_step];
                max_radius=2.5 * grid_step, geometry=:spherical,
            )
            @test_throws ArgumentError kinetic_energy_transfer(
                u, v, w, longitude, [-90.0, -89.99, -89.98, -89.97, -89.96],
                [grid_step];
                max_radius=2.5 * grid_step, geometry=:tangent_quadratic,
            )
            @test_throws ArgumentError kinetic_energy_transfer(
                u, v, w, [-180.0, -90.0, 0.0, 90.0, 180.0], latitude, [grid_step];
                max_radius=2.5 * grid_step, geometry=:spherical,
            )
        end
    end

    @testset "dimensions, clipping, and constant-flow invariance" begin
        x = collect(0.0:1.0:4.0)
        y = collect(0.0:1.0:4.0)
        u = fill(3.0, 2, 3, 5, 5)
        v = fill(-2.0, 2, 3, 5, 5)
        w = fill(1.0, 2, 3, 5, 5)
        result = kinetic_energy_transfer(
            u, v, w, x, y, [0.5, 1.0, 2.0];
            max_radius=3.0, xdim=4, ydim=3,
        )
        @test size(result.transfer) == (1, 2, 3, 5, 5)
        @test result.length_scales == [1.0]
        @test all(iszero, result.transfer)
        @test result.dimension_order == (:length_scale, :dim1, :dim2, :dim3, :dim4)
    end

    @testset "boundary mask and input validation" begin
        x = collect(0.0:1.0:4.0)
        y = collect(0.0:1.0:4.0)
        u = zeros(5, 5)
        v = zeros(5, 5)
        w = zeros(5, 5)
        u[3, 1] = 1.0
        bounded = kinetic_energy_transfer(
            u, v, w, x, y, [1.0];
            max_radius=3.0, periodic=(false, false),
        )
        wrapped = kinetic_energy_transfer(
            u, v, w, x, y, [1.0];
            max_radius=3.0, periodic=(true, false),
        )
        @test all(isfinite, bounded.transfer)
        @test bounded.transfer[1, 3, 5] != wrapped.transfer[1, 3, 5]
        @test_throws ArgumentError kinetic_energy_transfer(
            u, v, w, x, y, [0.5];
            max_radius=3.0, periodic=(false, false),
        )
        @test_throws ArgumentError kinetic_energy_transfer(
            u, v, w, [0.0, 1.0, 2.1, 3.0, 4.0], y, [1.0];
            max_radius=3.0, periodic=(false, false),
        )
    end
end
