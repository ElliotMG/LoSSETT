# Julia kinetic-energy transfer

The Julia core is available as a standalone project or through the optional
Python bridge. Its default remains the Cartesian API:

```julia
using LoSSETT

result = kinetic_energy_transfer(
    u, v, w, x_metres, y_metres, [100_000.0, 200_000.0];
    max_radius=400_000.0,
)
```

Pass `geometry=:spherical` to use a regular longitude/latitude grid. Longitude
and latitude coordinates are in degrees; `length_scales`, `max_radius`, and
`sphere_radius` are linear distances (metres by default). `u` and `v` are local
eastward and northward velocity components. For example:

```julia
result = kinetic_energy_transfer(
    u, v, w, longitude_degrees, latitude_degrees, [100_000.0, 200_000.0];
    max_radius=400_000.0,
    geometry=:spherical,
    sphere_radius=6_371_000.0,
)
```

Spherical mode computes great-circle distances and initial/final bearings,
then projects endpoint velocity components into the geodesic tangent frame.
It uses `R*sin(r/R)` as the radial area Jacobian both when normalizing the
mollifier and when integrating the transfer. Longitude coordinates must be
strictly increasing and regularly spaced, without a duplicated 360-degree
endpoint; latitude coordinates must be regularly spaced in `[-90, 90]`.
`periodic` only affects Cartesian mode. `use_angular_weights=true` applies
periodic Voronoi sectors to the sampled bearings; the default uses the
Python spherical workflow's uniform `2π`-scaled sample average.
Only supplied grid points participate in each annulus: empty annuli contribute
zero, and a regional grid has incomplete directional coverage near its edges.

Use `geometry=:tangent_quadratic` for an alternate spherical approximation.
It keeps great-circle distances and the spherical radial/kernel treatment but
uses the initial bearing for both endpoint projections, with the leading-order
curvature correction
`delta_alpha = (r / sphere_radius) * sin(initial_bearing) * tan(latitude_origin)`.
As in Julia's full spherical mode, the increment norm includes `w`; the Python
branch's corresponding tangent-quadratic function omits vertical velocity.
The approximation is intended for spherical-cap/subset calculations and does
not change the default Cartesian or full spherical modes. Because the
correction contains `tan(latitude_origin)`, grids that include either pole are
rejected.

The Python bridge accepts the same geometry options, including
`geometry="tangent_quadratic"`:

```python
from lossett.julia import kinetic_energy_transfer

result = kinetic_energy_transfer(
    u, v, w, longitude_degrees, latitude_degrees, [100_000.0],
    max_radius=200_000.0,
    geometry="spherical",
    sphere_radius=6_371_000.0,
    xdim=4, ydim=3,  # e.g. (time, pressure, latitude, longitude)
)
```

Install the optional bridge dependency with `pip install 'lossett[julia]'`.
The first bridge call starts Julia and compiles the core.

## Relationship to the Python spherical branch

The `spherical_geometry` branch's separate geometry/increment workflow uses
great-circle distance and endpoint bearings, and its mollifier normalization
uses the spherical area element. Its current final transfer integration still
weights by `r dr`, rather than `R*sin(r/R) dr`. Julia's spherical mode uses the
spherical Jacobian consistently for normalization and transfer, so its radial
integral intentionally corrects that discrepancy rather than reproducing it.
The older Python `calc_scale_increments` workflow remains Euclidean in its
coordinate-offset distances and angles; it is not a spherical API. The
Python spherical workflow also currently omits vertical velocity from its
increment norm, whereas Julia includes `w` in the squared increment norm.

This is a direct sampled-grid implementation, not a port of the Python
geometry archive, chunked Zarr workflow, or Dask/Numba execution. Spherical
calculation compares each origin with the supplied grid points and therefore
has quadratic spatial work; it is intended for modest grids or prototyping.
For Python's uniform angular weighting, results can also differ from its
optional precomputed angular Voronoi weights.

Run the Julia tests with:

```sh
julia --project=Julia -e 'using Pkg; Pkg.test()'
```
