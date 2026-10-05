module LoSSETT

export kinetic_energy_transfer

function _trapz(values, coordinates)
    length(values) == length(coordinates) || throw(ArgumentError("values and coordinates must have equal lengths"))
    length(values) < 2 && return 0.0
    total = 0.0
    @inbounds for i in 1:(length(values) - 1)
        total += (coordinates[i + 1] - coordinates[i]) * (values[i] + values[i + 1]) / 2
    end
    return total
end

function _mollifier(r, ell)
    r < 2ell ? exp(-1 / (1 - (r / (2ell))^2)) : 0.0
end

function _mollifier_derivative(r, ell, normalization)
    r < 2ell || return 0.0
    q = 1 - (r / (2ell))^2
    return -normalization * (r / (2ell^2)) * exp(-1 / q) / q^2
end

function _uniform_spacing(coordinates, name)
    length(coordinates) >= 2 || throw(ArgumentError("$name must contain at least two coordinates"))
    all(isfinite, coordinates) || throw(ArgumentError("$name coordinates must be finite"))
    differences = diff(coordinates)
    all(d -> d > 0, differences) || throw(ArgumentError("$name coordinates must be strictly increasing"))
    spacing = first(differences)
    all(d -> isapprox(d, spacing; rtol=1e-8, atol=abs(spacing) * 1e-10), differences) ||
        throw(ArgumentError("$name coordinates must be regularly spaced"))
    return Float64(spacing)
end

function _shift_source(index, offset, extent, periodic)
    source = index + offset
    if periodic
        return mod1(source, extent)
    end
    return 1 <= source <= extent ? source : 0
end

"""
    kinetic_energy_transfer(u, v, w, x, y, length_scales;
        max_radius, periodic=(true, false), xdim=ndims(u),
        ydim=ndims(u)-1)

Compute the 2-D kinetic-energy transfer
`D_ell = (1/4) ∫ (dG_ell/dr) r ∫ (delta_u ⋅ r_hat) |delta_u|^2 dphi dr`
on a regular Cartesian grid. The mollifier is normalized with the 2-D
Euclidean area element `2πr dr`.

`u`, `v`, and `w` must be equally shaped real arrays. `xdim` and `ydim`
identify the longitude-like and latitude-like axes; by default, the final
two axes are `(y, x)`. Coordinates `x` and `y`, `max_radius`, and
`length_scales` must use the same units. `periodic` is `(x_periodic,
y_periodic)`. Nonperiodic out-of-domain increments contribute `NaN`, which
is treated as zero in the angular integral, matching LoSSETT's integration
handling of masked boundaries.

Returns a named tuple. `transfer` has shape
`(length_scale, remaining input axes in original order, y, x)`;
`length_scales` contains the sorted requested scales clipped to the
resolvable range, and `radii` contains the sampled radial annuli. Input
dimensions are retained, with spatial axes placed last in `(y, x)` order.

For latitude/longitude data in degrees, convert coordinates and all scales
to metres before calling (the ANCIL tutorial uses 110,000 m per degree).
"""
function kinetic_energy_transfer(
    u::AbstractArray{<:Real},
    v::AbstractArray{<:Real},
    w::AbstractArray{<:Real},
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    length_scales::AbstractVector{<:Real};
    max_radius::Real,
    periodic::Tuple{Bool,Bool}=(true, false),
    xdim::Integer=ndims(u),
    ydim::Integer=ndims(u) - 1,
)
    size(u) == size(v) == size(w) || throw(ArgumentError("u, v, and w must have identical shapes"))
    ndims(u) >= 2 || throw(ArgumentError("velocity arrays must have at least two dimensions"))
    1 <= xdim <= ndims(u) || throw(ArgumentError("xdim is outside the array dimensions"))
    1 <= ydim <= ndims(u) || throw(ArgumentError("ydim is outside the array dimensions"))
    xdim != ydim || throw(ArgumentError("xdim and ydim must be different"))
    length(x) == size(u, xdim) || throw(ArgumentError("length(x) must match size(u, xdim)"))
    length(y) == size(u, ydim) || throw(ArgumentError("length(y) must match size(u, ydim)"))
    isempty(length_scales) && throw(ArgumentError("length_scales must not be empty"))
    isfinite(max_radius) && max_radius > 0 || throw(ArgumentError("max_radius must be finite and positive"))
    all(s -> isfinite(s) && s > 0, length_scales) ||
        throw(ArgumentError("length_scales must contain finite positive values"))

    dx = _uniform_spacing(x, "x")
    dy = _uniform_spacing(y, "y")
    nx, ny = length(x), length(y)
    lx, ly = Float64(x[end] - x[1]), Float64(y[end] - y[1])
    dr = max(dx, dy)

    requested = Float64.(length_scales)
    requested_radius = Float64(max_radius)
    if maximum(requested) < requested_radius / 2
        requested_radius = 2 * maximum(requested)
    end
    radial_limit = min(requested_radius + dr, min(lx, ly) / 2)
    radial_count = max(0, ceil(Int, radial_limit / dr))
    radii = Float64[i * dr for i in 0:(radial_count - 1) if i * dr < radial_limit]
    length(radii) >= 2 || throw(ArgumentError("max_radius and domain size must provide at least two radial samples"))

    minimum_scale = radii[2]
    maximum_scale = radii[fld(length(radii), 2) + 1]
    scales = sort!(unique!(requested))
    filter!(ell -> minimum_scale <= ell <= maximum_scale, scales)
    isempty(scales) && throw(ArgumentError(
        "no length scales are resolvable; allowed range is [$minimum_scale, $maximum_scale]"
    ))

    remaining_dims = [d for d in 1:ndims(u) if d != xdim && d != ydim]
    permutation = (remaining_dims..., ydim, xdim)
    up = permutedims(Float64.(u), permutation)
    vp = permutedims(Float64.(v), permutation)
    wp = permutedims(Float64.(w), permutation)
    remaining_sizes = Tuple(size(u, d) for d in remaining_dims)
    profiles = isempty(remaining_sizes) ? 1 : prod(remaining_sizes)
    up = reshape(up, profiles, ny, nx)
    vp = reshape(vp, profiles, ny, nx)
    wp = reshape(wp, profiles, ny, nx)

    origin_x, origin_y = fld(nx, 2) + 1, fld(ny, 2) + 1
    annuli = [Dict{Float64,Vector{Tuple{Int,Int}}}() for _ in radii]
    for iy in 1:ny, ix in 1:nx
        offset_x, offset_y = ix - origin_x, iy - origin_y
        radius = hypot(offset_x * dx, offset_y * dy)
        for ir in 2:length(radii)
            ring_radius = radii[ir]
            if ring_radius - dr / 2 <= radius < ring_radius + dr / 2
                phi = atan(offset_y * dy, offset_x * dx)
                push!(get!(annuli[ir], phi, Tuple{Int,Int}[]), (offset_x, offset_y))
            end
        end
    end

    radial_integrand = zeros(Float64, length(radii), profiles, ny, nx)
    for ir in 2:length(radii)
        groups = annuli[ir]
        isempty(groups) && continue
        angles = sort!(collect(keys(groups)))
        previous_values = nothing
        previous_angle = 0.0
        for phi in angles
            offsets = groups[phi]
            current_values = fill(NaN, profiles, ny, nx)
            for iy in 1:ny, ix in 1:nx, profile in 1:profiles
                total = 0.0
                valid = true
                for (offset_x, offset_y) in offsets
                    source_x = _shift_source(ix, offset_x, nx, periodic[1])
                    source_y = _shift_source(iy, offset_y, ny, periodic[2])
                    if source_x == 0 || source_y == 0
                        valid = false
                        break
                    end
                    delta_u = up[profile, source_y, source_x] - up[profile, iy, ix]
                    delta_v = vp[profile, source_y, source_x] - vp[profile, iy, ix]
                    delta_w = wp[profile, source_y, source_x] - wp[profile, iy, ix]
                    if isnan(delta_u) || isnan(delta_v) || isnan(delta_w)
                        valid = false
                        break
                    end
                    radius = hypot(offset_x * dx, offset_y * dy)
                    along_r = (delta_u * offset_x * dx + delta_v * offset_y * dy) / radius
                    total += along_r * (delta_u^2 + delta_v^2 + delta_w^2)
                end
                if valid
                    current_values[profile, iy, ix] = total / length(offsets)
                end
            end

            if previous_values !== nothing
                delta_phi = phi - previous_angle
                for profile in 1:profiles, iy in 1:ny, ix in 1:nx
                    first_value = previous_values[profile, iy, ix]
                    next_value = current_values[profile, iy, ix]
                    isnan(first_value) && (first_value = 0.0)
                    isnan(next_value) && (next_value = 0.0)
                    radial_integrand[ir, profile, iy, ix] +=
                        delta_phi * (first_value + next_value) / 2
                end
            end
            previous_values = current_values
            previous_angle = phi
        end
    end

    result = Array{Float64}(undef, length(scales), profiles, ny, nx)
    for (iscale, ell) in enumerate(scales)
        raw_kernel = [_mollifier(radius, ell) for radius in radii]
        normalization_integral = 2π * _trapz(radii .* raw_kernel, radii)
        normalization_integral > 0 && isfinite(normalization_integral) ||
            throw(ArgumentError("unable to normalize the mollifier for length scale $ell"))
        normalization = inv(normalization_integral)
        derivative = [_mollifier_derivative(radius, ell, normalization) for radius in radii]
        support = findall(radius -> radius <= 2ell, radii)
        for profile in 1:profiles, iy in 1:ny, ix in 1:nx
            weighted = [
                derivative[ir] * radii[ir] * radial_integrand[ir, profile, iy, ix]
                for ir in support
            ]
            result[iscale, profile, iy, ix] = _trapz(weighted, radii[support]) / 4
        end
    end

    output_shape = (length(scales), remaining_sizes..., ny, nx)
    output_order = (:length_scale, (Symbol("dim", string(d)) for d in remaining_dims)...,
                    Symbol("dim", string(ydim)), Symbol("dim", string(xdim)))
    return (
        transfer=reshape(result, output_shape),
        length_scales=scales,
        radii=radii,
        dimension_order=output_order,
    )
end

function _kinetic_energy_transfer_python(args...; kwargs...)
    result = kinetic_energy_transfer(args...; kwargs...)
    return (
        result.transfer,
        result.length_scales,
        result.radii,
        string.(result.dimension_order),
    )
end

end
