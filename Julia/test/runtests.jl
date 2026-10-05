using LoSSETT
using Test

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
