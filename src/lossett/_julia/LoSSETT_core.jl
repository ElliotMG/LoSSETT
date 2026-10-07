export kinetic_energy_transfer

const _DEFAULT_SPHERE_RADIUS = 6_371_000.0

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

_spherical_area_jacobian(r, sphere_radius) =
    sphere_radius * sin(r / sphere_radius)

function _great_circle_geometry(lat0, lat, delta_lon, sphere_radius)
    sin_lat0, cos_lat0 = sincos(lat0)
    sin_lat, cos_lat = sincos(lat)
    sin_half_lat = sin((lat - lat0) / 2)
    sin_half_lon = sin(delta_lon / 2)
    haversine = clamp(
        sin_half_lat^2 + cos_lat0 * cos_lat * sin_half_lon^2,
        0.0, 1.0,
    )
    distance = 2 * sphere_radius * atan(sqrt(haversine), sqrt(1 - haversine))

    sin_lon, cos_lon = sincos(delta_lon)
    yi = sin_lon * cos_lat
    xi = cos_lat0 * sin_lat - sin_lat0 * cos_lat * cos_lon
    bearing_norm = hypot(xi, yi)
    bearing_norm > 32eps(Float64) ||
        return (distance, NaN, NaN, NaN, NaN)

    sin_initial, cos_initial = yi / bearing_norm, xi / bearing_norm
    sin_final = sin_lon * cos_lat0 / bearing_norm
    cos_final = -(cos_lat * sin_lat0 - sin_lat * cos_lat0 * cos_lon) /
                bearing_norm
    return (distance, sin_initial, cos_initial, sin_final, cos_final)
end

function _spherical_mollifier(radii, ell, sphere_radius)
    area_jacobian = [_spherical_area_jacobian(r, sphere_radius) for r in radii]
    raw_kernel = [_mollifier(r, ell) for r in radii]
    normalization_integral = 2π * _trapz(area_jacobian .* raw_kernel, radii)
    normalization_integral > 0 && isfinite(normalization_integral) ||
        throw(ArgumentError("unable to normalize the spherical mollifier for length scale $ell"))
    normalization = inv(normalization_integral)
    derivative = [_mollifier_derivative(r, ell, normalization) for r in radii]
    return normalization, derivative, area_jacobian
end

function _spherical_angular_integral(samples, angles, use_angular_weights)
    isempty(samples) && return 0.0
    if !use_angular_weights
        return 2π * sum(samples) / length(samples)
    end

    groups = Dict{Tuple{Int,Int},Vector{Int}}()
    for i in eachindex(samples)
        key = (round(Int, cos(angles[i]) / 1e-6),
               round(Int, sin(angles[i]) / 1e-6))
        push!(get!(groups, key, Int[]), i)
    end

    group_angles = Float64[]
    group_values = Float64[]
    for indices in Base.values(groups)
        sin_mean = sum(sin(angles[i]) for i in indices) / length(indices)
        cos_mean = sum(cos(angles[i]) for i in indices) / length(indices)
        push!(group_angles, mod(atan(sin_mean, cos_mean), 2π))
        push!(group_values, sum(samples[i] for i in indices) / length(indices))
    end

    if length(group_angles) <= 4
        return 2π * sum(samples) / length(samples)
    end

    order = sortperm(group_angles)
    sorted_angles = group_angles[order]
    gaps = diff(vcat(sorted_angles, sorted_angles[1] + 2π))
    cell_widths = (gaps .+ circshift(gaps, 1)) ./ 2
    return sum(cell_widths[i] * group_values[order[i]] for i in eachindex(order))
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

function _spherical_kinetic_energy_transfer(
    u, v, w, longitude, latitude, length_scales;
    max_radius, sphere_radius, use_angular_weights, xdim, ydim,
    tangent_quadratic,
)
    size(u) == size(v) == size(w) ||
        throw(ArgumentError("u, v, and w must have identical shapes"))
    ndims(u) >= 2 || throw(ArgumentError("velocity arrays must have at least two dimensions"))
    1 <= xdim <= ndims(u) || throw(ArgumentError("xdim is outside the array dimensions"))
    1 <= ydim <= ndims(u) || throw(ArgumentError("ydim is outside the array dimensions"))
    xdim != ydim || throw(ArgumentError("xdim and ydim must be different"))
    length(longitude) == size(u, xdim) ||
        throw(ArgumentError("length(longitude) must match size(u, xdim)"))
    length(latitude) == size(u, ydim) ||
        throw(ArgumentError("length(latitude) must match size(u, ydim)"))
    isempty(length_scales) && throw(ArgumentError("length_scales must not be empty"))
    isfinite(max_radius) && max_radius > 0 ||
        throw(ArgumentError("max_radius must be finite and positive"))
    isfinite(sphere_radius) && sphere_radius > 0 ||
        throw(ArgumentError("sphere_radius must be finite and positive"))
    max_radius <= π * sphere_radius ||
        throw(ArgumentError("max_radius cannot exceed half the sphere circumference"))
    all(s -> isfinite(s) && s > 0, length_scales) ||
        throw(ArgumentError("length_scales must contain finite positive values"))
    all(isfinite, longitude) && all(isfinite, latitude) ||
        throw(ArgumentError("longitude and latitude coordinates must be finite"))
    all(lat -> -90 <= lat <= 90, latitude) ||
        throw(ArgumentError("latitude coordinates must lie in [-90, 90] degrees"))
    tangent_quadratic && any(lat -> abs(lat) == 90, latitude) &&
        throw(ArgumentError("tangent_quadratic geometry is undefined at the poles"))

    dlon = _uniform_spacing(longitude, "longitude")
    dlat = _uniform_spacing(latitude, "latitude")
    longitude[end] - longitude[1] < 360 - dlon * 1e-8 ||
        throw(ArgumentError("longitude must not contain a duplicated 360-degree endpoint"))

    nlon, nlat = length(longitude), length(latitude)
    requested = Float64.(length_scales)
    radial_extent = Float64(max_radius)
    if maximum(requested) < radial_extent / 2
        radial_extent = 2 * maximum(requested)
    end
    nominal_step = Float64(sphere_radius) * deg2rad(max(dlon, dlat))
    nradial = ceil(Int, radial_extent / nominal_step)
    nradial >= 2 ||
        throw(ArgumentError("max_radius and grid spacing must provide at least two radial samples"))
    radial_step = radial_extent / nradial
    radii = Float64[(i - 0.5) * radial_step for i in 1:nradial]

    scales = sort!(unique!(requested))
    minimum_scale = nominal_step * (1 - 1e-10)
    filter!(ell -> minimum_scale <= ell <= radial_extent / 2, scales)
    isempty(scales) && throw(ArgumentError(
        "no length scales are resolvable; allowed range is [$minimum_scale, $(radial_extent / 2)]"
    ))

    remaining_dims = [d for d in 1:ndims(u) if d != xdim && d != ydim]
    permutation = (remaining_dims..., ydim, xdim)
    up = permutedims(Float64.(u), permutation)
    vp = permutedims(Float64.(v), permutation)
    wp = permutedims(Float64.(w), permutation)
    remaining_sizes = Tuple(size(u, d) for d in remaining_dims)
    profiles = isempty(remaining_sizes) ? 1 : prod(remaining_sizes)
    up = reshape(up, profiles, nlat, nlon)
    vp = reshape(vp, profiles, nlat, nlon)
    wp = reshape(wp, profiles, nlat, nlon)
    lon_rad = deg2rad.(Float64.(longitude))
    lat_rad = deg2rad.(Float64.(latitude))

    angular_integrand = zeros(Float64, nradial, profiles, nlat, nlon)
    for iy0 in 1:nlat, ix0 in 1:nlon
        bin_values = [[Float64[] for _ in 1:nradial] for _ in 1:profiles]
        bin_angles = [[Float64[] for _ in 1:nradial] for _ in 1:profiles]
        lat0 = lat_rad[iy0]
        for iy in 1:nlat, ix in 1:nlon
            lat = lat_rad[iy]
            delta_lon = lon_rad[ix] - lon_rad[ix0]
            distance, sin_initial, cos_initial, sin_final, cos_final =
                _great_circle_geometry(lat0, lat, delta_lon, sphere_radius)
            distance < radial_extent || continue
            ir = floor(Int, distance / radial_step) + 1
            ir <= nradial || continue
            isfinite(sin_initial) || continue
            initial_bearing = atan(sin_initial, cos_initial)

            for profile in 1:profiles
                delta_u = up[profile, iy, ix] - up[profile, iy0, ix0]
                delta_v = vp[profile, iy, ix] - vp[profile, iy0, ix0]
                if tangent_quadratic
                    delta_alpha =
                        distance * sin_initial * tan(lat0) / sphere_radius
                    delta_t = delta_u * sin_initial + delta_v * cos_initial +
                              delta_alpha * (
                                  up[profile, iy, ix] * cos_initial -
                                  vp[profile, iy, ix] * sin_initial
                              )
                    delta_n = delta_u * cos_initial - delta_v * sin_initial -
                              delta_alpha * (
                                  up[profile, iy, ix] * sin_initial +
                                  vp[profile, iy, ix] * cos_initial
                              )
                else
                    delta_t = up[profile, iy, ix] * sin_final +
                              vp[profile, iy, ix] * cos_final -
                              up[profile, iy0, ix0] * sin_initial -
                              vp[profile, iy0, ix0] * cos_initial
                    delta_n = up[profile, iy, ix] * cos_final -
                              vp[profile, iy, ix] * sin_final -
                              up[profile, iy0, ix0] * cos_initial +
                              vp[profile, iy0, ix0] * sin_initial
                end
                delta_w = wp[profile, iy, ix] - wp[profile, iy0, ix0]
                value = delta_t * (delta_t^2 + delta_n^2 + delta_w^2)
                isfinite(value) || continue
                push!(bin_values[profile][ir], value)
                push!(bin_angles[profile][ir], initial_bearing)
            end
        end

        for ir in 1:nradial, profile in 1:profiles
            angular_integrand[ir, profile, iy0, ix0] =
                _spherical_angular_integral(
                    bin_values[profile][ir], bin_angles[profile][ir],
                    use_angular_weights,
                )
        end
    end

    result = Array{Float64}(undef, length(scales), profiles, nlat, nlon)
    for (iscale, ell) in enumerate(scales)
        _, derivative, area_jacobian =
            _spherical_mollifier(radii, ell, sphere_radius)
        support = findall(radius -> radius < 2ell, radii)
        length(support) >= 2 ||
            throw(ArgumentError("length scale $ell has fewer than two radial samples in kernel support"))
        weights = derivative[support] .* area_jacobian[support]
        for profile in 1:profiles, iy in 1:nlat, ix in 1:nlon
            radial_values = [
                weights[j] * angular_integrand[ir, profile, iy, ix]
                for (j, ir) in enumerate(support)
            ]
            result[iscale, profile, iy, ix] =
                _trapz(radial_values, radii[support]) / 4
        end
    end

    output_shape = (length(scales), remaining_sizes..., nlat, nlon)
    output_order = (:length_scale, (Symbol("dim", string(d)) for d in remaining_dims)...,
                    Symbol("dim", string(ydim)), Symbol("dim", string(xdim)))
    return (
        transfer=reshape(result, output_shape),
        length_scales=scales,
        radii=radii,
        dimension_order=output_order,
    )
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
        ydim=ndims(u)-1, geometry=:cartesian, sphere_radius=6_371_000.0)

Compute the 2-D kinetic-energy transfer
`D_ell = (1/4) ∫ (dG_ell/dr) J(r) ∫ (delta_u ⋅ r_hat) |delta_u|^2 dphi dr`
on a regular Cartesian grid by default, where `J(r)=r`. Set
`geometry=:spherical` to use longitude/latitude coordinates in degrees,
great-circle displacements and bearings, and `J(r)=R sin(r/R)` for both
mollifier normalization and radial integration. `sphere_radius` is in the
same distance units as `max_radius` and `length_scales`. Set
`geometry=:tangent_quadratic` to use great-circle radial bins with an
initial-bearing tangent-plane velocity increment corrected for leading-order
spherical curvature; it uses the same spherical radial kernel as `:spherical`.

`u`, `v`, and `w` must be equally shaped real arrays. `xdim` and `ydim`
identify the longitude-like and latitude-like axes; by default, the final
two axes are `(y, x)`. For Cartesian geometry, `x`, `y`, `max_radius`, and
`length_scales` use common linear units. For spherical geometry, `x` and `y`
are longitude and latitude in degrees, while the radii and scales use linear
units (metres by default). `periodic` is `(x_periodic, y_periodic)` for
Cartesian geometry and is ignored for spherical geometry, where distances
are computed directly between the supplied coordinates. Spherical annuli use
only supplied grid points; empty annuli contribute zero, and regional grids
therefore provide incomplete directional coverage.
For `:tangent_quadratic`, the increment correction is
`delta_alpha = (r/R) sin(initial_bearing) tan(latitude_origin)`. Both spherical
modes include the vertical-velocity increment in the increment norm.
`:tangent_quadratic` rejects grids containing either pole because its
curvature correction is undefined there.

Returns a named tuple. `transfer` has shape
`(length_scale, remaining input axes in original order, y, x)`;
`length_scales` contains the sorted requested scales clipped to the
resolvable range, and `radii` contains the sampled radial annuli. Input
dimensions are retained, with spatial axes placed last in `(y, x)` order.

The Python `spherical_geometry` workflow's `tangent_quadratic` increment uses
the initial-bearing tangent-plane projection plus the leading-order curvature
correction above. Julia retains that velocity-increment approximation while
using `R sin(r/R)` consistently for spherical mollifier normalization and
transfer integration; the Python workflow currently integrates with `r dr`.
Julia also retains the Cartesian API's `w` contribution to `|delta u|^2`,
which the Python tangent-quadratic increment omits. The separate legacy
Python `calc_scale_increments` path still computes Euclidean coordinate-offset
distances and angles.
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
    geometry::Union{Symbol,AbstractString}=:cartesian,
    sphere_radius::Real=_DEFAULT_SPHERE_RADIUS,
    use_angular_weights::Bool=false,
)
    geometry_kind = Symbol(geometry)
    if geometry_kind == :spherical || geometry_kind == :tangent_quadratic
        return _spherical_kinetic_energy_transfer(
            u, v, w, x, y, length_scales;
            max_radius, sphere_radius, use_angular_weights, xdim, ydim,
            tangent_quadratic=geometry_kind == :tangent_quadratic,
        )
    elseif geometry_kind != :cartesian
        throw(ArgumentError(
            "geometry must be :cartesian, :spherical, or :tangent_quadratic",
        ))
    end

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
