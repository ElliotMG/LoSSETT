# LoSSETT (Local Scale-to-Scale Energy Transfer Tool)

LoSSETT (the Local Scale-to-Scale Energy Transfer Tool) is a Python package for calculating local energy transfer across specified length scales. The kinetic energy transfer from scales larger than $\ell$ to scales smaller than $\ell$ is derived in [Duchon & Robert (2000)](https://iopscience.iop.org/article/10.1088/0951-7715/13/1/312) to be:

$$\mathcal{D}_{\ell} := \frac{1}{4} \int \nabla G _\ell(\mathbf{r}) \cdot \delta \mathbf{u} |\delta \mathbf{u}|^2 \mathrm{d}^d \mathbf{r}.$$

Here $\delta \mathbf{u} := \mathbf{u}(\mathbf{x}+\mathbf{r}) - \mathbf{u}(\mathbf{x})$ is a velocity increment, and $G _\ell(\mathbf{r})$ is a filter kernel with characteristic length scale $\ell$. Similar expressions may be derived for the inter-scale transfer of any quadratic scalar (e.g. buoyancy variance, moisture variance).

With thanks to Valerio Lembo (ISAC-CNR) for sharing some of the original MATLAB code developed with colleagues at CNRS ([Kuzzay et al. 2015](https://pubs.aip.org/aip/pof/article-abstract/27/7/075105/103779), [Faranda et al. 2018](https://journals.ametsoc.org/view/journals/atsc/75/7/jas-d-17-0114.1.xml)). Development work funded as part of the UPSCALE project at the University of Reading funded by the Met Office.

Note that this code is currently a work-in-progress.


> **Notice:** the previous functionality that was in `lossett_control` and `lossett_plotting` is now available in the new repo **[LoSSETT_ANCIL](https://github.com/ElliotMG/LoSSETT_ANCIL)**. This repository is now only focused on the core computation.

---


## Repository Structure

Within `src`:

* `lossett/calc` contains the core computational routines for calculating inter-scale energy transfers.
* `lossett/filtering` contains utilities for filtering and integration.

## Prerequisites
Current distribution of python (Python 3) - built with `xarray` and `numpy`. See `pyproject.toml` for full list of requirements.

## Spherical vector filtering

`lossett.filtering.spherical_vector_filter.filter_vector_field_tangent_normal`
defaults to the gradient-commuting vector filter. Pass the scalar-kernel profile
as `kernel`, sampled on a strictly increasing `mu_grid` spanning `[-1, 1]`,
and pass `mu=cos(distance / sphere_radius)` at the vector sample points.
Provide spherical area quadrature weights and the initial and final bearing
components. The scalar and vector kernels share the scalar normalization.
The profile must resolve the kernel support, especially for small filter
scales; finite quadrature and missing samples make commutation approximate.

The previous same-kernel parallel-transport average is still available with
`mode="parallel_transport"`; in that mode `kernel` contains sample values
and neither `mu` nor `mu_grid` is used.

## Reference SO(3) increment calculation

`lossett.calc.group_increments.group_increment_grid` computes the coherent
group diagnostic and single-shell moments on a full sphere. Its inputs are
NumPy arrays: unit radial directions and Cartesian velocities with shape
`(*grid, 3)`, the intrinsic horizontal covariant gradient of horizontal
velocity embedded in Cartesian coordinates with shape `(*grid, 3, 3)`, the
horizontal gradient of radial velocity with shape `(*grid, 3)`, and solid
angle quadrature weights with shape `grid`. The weights must sum to `4*pi`.
Gradients must include spherical connection terms; ordinary Cartesian
derivatives of east/north component arrays are not interchangeable with them.
Missing data and regional grids are not supported.
`latlon_shell_fields` constructs these Cartesian fields and gradients from
eastward, northward and radial velocity on a regular global longitude grid,
using second-order finite differences. It requires latitudes and longitudes
in radians. At a pole it estimates Cartesian derivatives from the first
Fourier mode on the nearest latitude ring; the pole velocities must
represent the same physical Cartesian vector at all longitudes. This
assumes a smooth field and should be checked for convergence. Supply the
grid's own solid-angle weights.

Select exactly one kernel:

* `heat_time=t` uses the matched increment filter with degree response
  `exp(-2*t*n*(n+1))`; the corresponding LES filter has response
  `exp(-t*n*(n+1))`. `t` is dimensionless.
* `mollifier_scale=ell_over_radius` uses the matched group convolution of
  two LES filters whose scalar endpoint kernel is the existing standard
  mollifier. The scale is in radians and must be less than `pi/2`; the
  increment kernel is not the original mollifier.

`lossett.calc.group_filter.group_filter_shell` applies the matching LES
filter H to vectors and general rank-two tensors. To compute the coherent
LES stress, supply the target velocity and its outer product, then subtract
the outer product of the returned filtered velocity from the returned
filtered tensor. The heat time or mollifier scale must match the value
passed to the increment routine. The tensor's horizontal trace,
antisymmetric and trace-free parts and its radial components use their
respective paired fiber moments.

Keep the same angular scale at each radius for the vertical reconstruction.
`atmosphere="deep"` is the default; `"shallow"` drops explicit curvature
terms and is an approximation, not the exact deep-spherical matched identity.
The returned `horizontal`, `structure`, `vertical_flux`, and `mixed_moment`
are per unit mass. The returned `kernel_mass` measures the discrete integral
of the endpoint kernel; the grid routine raises if it differs from one by
more than `quadrature_tolerance` (default 0.01). This is only a necessary
resolution check, not a guarantee of accuracy. Pass the moments from multiple levels, origin velocity,
and radial direction to `reconstruct_group_vertical` for one radial ray.
It accepts `fluid="boussinesq"` or `"anelastic"` and
`coordinate_type="height"` or `"pressure"`. Pressure is in Pa and requires
positive reference density and gravity; anelastic calculations also require
reference density. Supplied radius, pressure, and density must represent the
same reference hydrostatic profile. `hydrostatic_radius` can reconstruct
the radius profile from pressure, reference density, gravity, and an
initial radius. Pressure surfaces differing from
fixed-radius shells require an additional approximation.

### Single-shell Zarr workflow

The separate `lossett.calc.compute_group_increments` CLI preserves the
existing spherical CLI and its diagnostic. It reads a full global, regular
latitude/longitude grid from NetCDF or Zarr, including the native poles.
Input `u`, `v`, and `w` are respectively eastward, northward, and **outward
geometric radial** velocities in m/s; pressure velocity (Pa/s) is not `w`.
Nonempty velocity-unit metadata must be recognized as m/s. If units are absent,
the user must ensure the inputs meet this contract. The input must contain
`time`, a one-dimensional level coordinate, `latitude`, and `longitude`;
latitude and longitude are in degrees and longitude is global, increasing,
uniform, and has no repeated endpoint. Missing data, inconsistent physical
velocity at a pole, and regional grids are rejected. Cell-boundary weights
integrate the full solid angle; they and the finite differences require
independent convergence checks.

For example, with three pressure levels and their reference densities:

```bash
for i in 0 1 2; do
  python -m lossett.calc.compute_group_increments shell \
    --velocity-file velocity.nc --output "shell-${i}.zarr" \
    --time-index 0 --level-index "$i" \
    --coordinate-type pressure --coordinate-unit hPa \
    --initial-radius 6371000 --density 1.2 1.0 0.8 --gravity 9.81 \
    --heat-time 0.4 --atmosphere deep
done
python -m lossett.calc.compute_group_increments vertical \
  --shells shell-0.zarr shell-1.zarr shell-2.zarr \
  --output transfer.zarr --fluid anelastic --density 1.2 1.0 0.8 --gravity 9.81
```

In the shell command `--density` must list one density per input pressure
level in input order; `--initial-radius` is the radius of the first input
level. The hydrostatic integration uses the trapezoidal rule. The vertical
command takes at least three shells in strictly monotonic coordinate order,
and its density values correspond to those shells in that order. The
recorded full profile allows non-adjacent shells to be checked without
reintegrating over a sparse subset. Alternatively, supply `--radius` for
each shell; pressure reconstruction then warns that consistency with the
reference hydrostatic profile is the user's responsibility.

For height input use `--level-dim height --coordinate-type height
--coordinate-unit m --radius ...`. Height and radius differences must agree.
Choose `--fluid boussinesq` or `anelastic` during reconstruction, and
`--atmosphere shallow` on each shell for the shallow approximation. Pressure
coordinates still require density and gravity for Boussinesq hydrostatic
conversion. Use `--mollifier-scale` instead of `--heat-time` to select the
matched mollifier. Keep the same angular kernel parameter and atmosphere
on all shells. The CLI will not overwrite an existing output.
For a regular global longitude grid, `--cache-longitude-moments` reuses
the converged fiber moments across longitude rotations within each origin
latitude. The grid is checked before caching; no interpolation of the
kernel is used. The default remains the uncached reference path.

For the matched mollifier only, `--compact-support` limits targets to the
exact support of the increment kernel. If `--mollifier-scale` is `s`
radians, the scalar endpoint kernel of the LES filter H is supported
within `2*s` and G = H*H within `min(4*s, pi)`. The calculation validates
the entire input and retains the original solid-angle weights; it does not
renormalize the selected targets. It requires a regular global longitude
grid and can be combined with `--cache-longitude-moments`. The mollifier
moments use a converged spectral approximation, set to zero outside their
exact support; they may still require significant computation at fine
resolution. Spectral coefficients are reused across calls at the same
angular scale and degree through a bounded in-process cache; convergence
is still checked for each set of moments. Heat kernels have global support
and cannot use this option.

Each shell store contains `horizontal`, `structure`, `vertical_flux`,
`mixed_moment`, `kernel_mass`, Cartesian `direction` and `velocity`, plus
coordinate, radius, kernel and provenance metadata. The second stage writes
`horizontal`, `vertical`, and `total` transfer on the supplied levels.
Values are per unit mass and positive transfer is downscale. The horizontal
calculation is independent of the reference-density profile, whereas the
anelastic vertical reconstruction includes its derivative. Pressure
surfaces that differ from fixed-radius shells remain an approximation.
The coherent LES tensor filter is still a reference API, not part of this CLI.

The implementation is a correctness-oriented reference, not optimized for
large grids. Even with compact support the pairwise cost can be substantial;
never infer accuracy from the kernel-mass check alone.

### Anti-aliased benchmark inputs

`tools/preprocess_benchmark_fields.py` prepares global benchmark fields using
the optional `benchmark` dependency (`pip install -e '.[benchmark]'`). It
selects one time and specified pressure levels from a regular, pole-inclusive
global grid. Before sampling each target grid, it applies the same
spherical-harmonic low-pass response to the horizontal velocity's vector
harmonics and the radial velocity's scalar harmonics. This avoids mixing the
horizontal velocity into `w`. The response is one through 70% of the target
grid's safe harmonic degree, tapers smoothly to zero at 90%, and is zero
above that cutoff. Output attributes record both degrees and the input file.
The taper is a benchmark-input filter, not the matched diagnostic filter
`H` or `G`; it is not asserted to have a nonnegative spatial kernel.

The source may have multiple inconsistent representations of each pole.
This tool replaces each with the longitude-mean **Cartesian** physical
velocity before filtering, and records the maximum polar adjustment for
each level (north, then south). Its output retains both poles. It rejects
missing data, unsupported units, non-global grids, and an existing output
directory; it does not overwrite the source.

The older preprocessing workflow interpolates directly to the target grid.
Interpolation alone is not an adequate anti-aliasing filter when fine-scale
structure exceeds the target grid's sampling limit: it can appear as false
larger-scale structure. The spectral preprocessing prevents retained output
harmonics from exceeding its stated cutoff, subject to the source sampling
and transform accuracy. It cannot undo aliasing already present in the
input, and it changes the fields by design. Different target grids use
different cutoffs, so a grid-convergence experiment on one *identically
filtered* field needs a common cutoff rather than these default outputs.

For example:

```bash
python tools/preprocess_benchmark_fields.py \
  --velocity-file velocity.nc --output-dir filtered-benchmark \
  --time-index 0 --pressure 150 200 250 \
  --grid 5p0deg n160 n320 n640 2p0deg 1p0deg 0p5deg 0p25deg
```

### Small synthetic benchmark

On a shared AMD EPYC 9654 host with BLAS/OpenMP threads limited to one, one
warm-up and two timed repetitions gave 0.107--0.112 s for the 8x16 group
grid kernel (`heat_time=0.2`) and 0.266--0.283 s for the original checkout's
16x16 Cartesian calculation. These are **different grids, fields, kernels,
and diagnostics**, not an equivalent-work speedup. The spherical checkout's
legacy Cartesian entry point currently fails because its integration-kernel
return contract changed, so the baseline uses the original checkout.
For a separate three-level 16x16 original-checkout run, calculation took
0.282--0.284 s, NetCDF input reading 0.007--0.040 s, Zarr output writing
0.319--0.352 s, and output rereading 0.016--0.020 s. This synthetic wrapper
uses the original calculation and adds I/O; it does not calculate the group
moments or perform a vertical reconstruction.

For a three-level synthetic `heat_time=0.4` Zarr workflow, 9x16 timed shell
calls ranged from 1.18 to 5.99 s and vertical reconstruction from 1.72 to
2.60 s; timings were noisy on the shared machine. Compared at shared nodes
with a 17x32 run, maximum absolute coarse/fine differences divided by the
fine-grid maximum magnitude were 0.28% (horizontal), 0.17% (vertical),
1.08% (total), and 0.12% (structure). Maximum kernel-mass errors were
0.025% and 0.0065%, respectively, illustrating why mass alone is not an
accuracy guarantee. These figures are only a small synthetic baseline, not
production throughput or an error bound.

Reproduce from the spherical checkout with an existing temporary directory:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=src python tools/benchmark_group_reference.py group
PYTHONPATH=/path/to/original/LoSSETT/src python tools/benchmark_group_reference.py legacy
PYTHONPATH=/path/to/original/LoSSETT/src python tools/benchmark_group_reference.py legacy-workflow --workspace /path/to/temporary-directory
PYTHONPATH=src python tools/benchmark_group_workflow.py --workspace /path/to/temporary-directory
```

### Real-field pilot

On LOTUS, one CPU and single-threaded BLAS were used for a 5-degree,
200 hPa field and a nominal 4000 km scale. The original Cartesian
calculation took 1.151--1.161 s after warm-up, excluding input and output.
The prior spherical minimum-transport CLI took 7.883--8.450 s including
startup and increment output, with 4.184--4.583 s logged for computation;
the logged latitude loop also includes Zarr writes. Separate geometry
generation took 10.008 s. It used unweighted angular
bins, a 4000 km distance cutoff, and included `w`; its vertical result is
a flux intermediate, not a reconstructed vertical transfer. Both transfer
outputs were non-finite at all 144 polar grid entries: two in-support
radial bins contain no samples at either pole on this grid.
The uncached group shell took 37.659--38.437 s, including field setup but
excluding input and output. A three-shell reconstruction took 0.358 s.
These are different diagnostics, output sets, boundaries and kernels,
and the timing boundaries differ; they are not equivalent-work speedups.

The group heat kernel matches the spherical mollifier's degree-one
response for the increment filter, not its complete shape. The 150, 200
and 250 hPa shells use a coarse vertical stencil and a US Standard
Atmosphere reference radius, including the geopotential-height
approximation. Sampled group kernel-mass errors reached 0.58% at the poles;
this is not an accuracy bound. The fiber-moment evaluation dominated a
sampled uncached origin profile. On a separate node, alternating uncached
and cached shell runs took 43.30--45.35 s and 8.51--9.51 s respectively,
about 4.8--5.1 times faster for this identical workload. The maximum
absolute differences were 6.8e-18 for horizontal transfer and 2.8e-12
for structure; all other moments also agreed within the recorded
floating-point tolerances. This is a coarse-grid result, not a
production-grid performance estimate.
`tools/benchmark_real_fields.py`, `tools/benchmark_real_spherical.py`,
`tools/benchmark_group_cache.py`, and `tools/benchmark_group_origins.py`
record the detailed workflows and timing boundaries.

Five uncached single-origin samples at the same scale took 0.326--0.335 s
on 1-degree, 1.289--1.320 s on 0.5-degree, and 1.517--1.547 s on n320
grids. These are not full-grid or cached timings. The separately filtered
input grids do not form an identical-field convergence test.

`tools/benchmark_spherical_comparison.py` compares the previous spherical
increment-and-integration path with the cached group shell in the same
process, on the same input level and with warm-up and timed repetitions.
It reads the previously generated spherical geometry in each increment
run, but reports its archive opening separately; it excludes shared input
reading and final output writing from the computation timings. It can
also run the original Cartesian checkout on the same allocation. The
legacy radial increment intermediate is held in memory instead of written
and reread, so this is not a full legacy CLI timing. The legacy vertical
output is a flux intermediate, while the group shell includes additional
moments but not vertical reconstruction. The heat and mollifier kernels,
domain treatment and diagnostics differ; the ratio is workload context,
not an equivalent-work speedup. The benchmark reports non-finite output,
and optional saved references verify the independently executed paths.
It defaults to a 3000-point safety limit; raising it does not make
all-pairs runs inexpensive. `--legacy-coverage both` also times the
previous spherical whole-globe increment calculation, while retaining the
same compact-support scale integration. `--mollifier-scale` selects the
matched mollifier in place of `--heat-time`; `--compact-support` enables
its support-limited group path.

On one LOTUS allocation, the 5-degree, 200 hPa case gave two post-warmup
times of 1.30--1.36 s for the original Cartesian calculation,
2.64--2.69 s for the previous spherical calculation plus scale integration,
and 8.77--9.05 s for the cached group shell. The respective ratios are
about 6.5--7.0 and 3.3--3.4, with the workload and timing caveats above.
The legacy geometry archive used float32; its increment calculations and
the group calculations used float64. The legacy and group outputs matched
their previously saved references, including the legacy polar missing values.
The legacy radius was 6371 km; the group shell used the reference
200 hPa radius of about 6383 km. The Cartesian calculation timing includes
its verbose calculation logging.
A separate cached-group profile took 9.40 s under `cProfile`, with 3.68 s
cumulative in the two transport operations per origin and 0.52 s in
fiber-moment evaluation. This profile suggests, but does not establish,
that fusing pairwise operations is a useful next optimization.

On a 2-degree grid, with the same level, scale and one-CPU setup, two
post-warmup runs took 7.32--7.34 s for Cartesian, 15.65--15.78 s for the
previous spherical path and 282.38--292.84 s for the cached group shell.
These are approximately 38.5--40.0 times the Cartesian time and
17.9--18.7 times the previous spherical time for the different workloads
described above. The previous spherical outputs were non-finite at all
360 polar entries; the group outputs were finite and their maximum
kernel-mass error was 0.094%, not an accuracy bound. The legacy geometry
archive took 19.92 s to generate separately. The prior spherical CLI
reproduced the comparison outputs to floating-point tolerance, and eight
cached group origins, including both poles, agreed with uncached calculations.
The 2-degree input has a different preprocessing cutoff from the 5-degree
input, so this is not an identical-field convergence test.

Before adding the spectral coefficient cache, a separate 2-degree, 200 hPa
equal-support comparison on one LOTUS CPU gave
the old spherical cutoff and whole-globe increment paths
12.50--12.60 s and 21.38--21.71 s respectively; the original Cartesian
path took 10.31--11.31 s, and the support-limited, cached matched-group
shell took 276.66--292.17 s. H has 2000 km support and G has 4000 km
support, so the support matches the previous mollifier, not its shape or
degree-one response. The group loop selected 32.56 million of 268.30
million origin--target pairs. These remain different diagnostics and output
sets, not equivalent-work speedups. The whole-globe spherical run still
integrates the same 4000 km mollifier; its outputs agree exactly with the
cutoff run, including 360 non-finite polar entries in each output.
The group outputs were finite; maximum kernel-mass error was 0.171%,
which is not an accuracy bound. Timings exclude output writes and legacy
geometry creation, include legacy geometry chunk reads, and include group
moment calculation. The Cartesian calculation uses a 4000 km separation
cutoff and periodic longitude with missing latitude boundaries, not
whole-globe spherical geometry.

Before adding the spectral coefficient cache, an equivalent-work check of
the compact group path used alternating full-target and support-limited
runs on the 5-degree field with the same matched mollifier. These took
250.88/94.58 s and 178.86/106.27 s respectively.
The paired speedups were 2.65 and 1.68, showing substantial timing
variability. All output fields agreed to floating-point tolerance.
This coarse grid has a 1.042% polar kernel-mass error, exceeding the
default 1% gate, so these runs used a 2% gate and are performance checks,
not evidence of an adequately resolved calculation. Four 2-degree origins,
including both poles, also agreed with uncached full-target calculations.
In a separate pre-coefficient-cache 5-degree compact profile, matched
fiber moments accounted for 43.65 of 45.64 profiled seconds, including
27.22 s in repeated Gauss--Legendre node generation and 15.56 s in
Chebyshev evaluation. These are profiled, coarse-grid timings, not a
breakdown of the 2-degree run.
On a separate 2-degree allocation, alternating runs of the compact group
path with and without spectral coefficient reuse took 255.27/306.52 s and
254.61/327.03 s respectively, with the coefficient cache cleared at the
start of each cached run. The maximum absolute output differences were zero.
Those group timings are from a different allocation than the three-path
comparison above.

For example, using an existing geometry archive and approved kernel parameters:

```bash
PYTHONPATH=src python tools/benchmark_spherical_comparison.py \
  --velocity-file filtered_5p0deg.nc --grid 5p0deg \
  --geometry-dir spherical-geometry --output-dir comparison \
  --physical-scale-km 4000 --radius 6382758.772639869 \
  --heat-time 0.012932180921700717 \
  --cartesian-src /path/to/original/LoSSETT/src
```

For the equal-support 2-degree comparison, use a separate output directory:

```bash
PYTHONPATH=src python tools/benchmark_spherical_comparison.py \
  --velocity-file filtered_2p0deg.nc --grid 2p0deg \
  --geometry-dir spherical-geometry --output-dir compact-comparison \
  --physical-scale-km 4000 --radius 6382758.772639869 \
  --mollifier-scale 0.1566720654220191 --compact-support \
  --legacy-coverage both --max-points 16380 \
  --cartesian-src /path/to/original/LoSSETT/src
```

Use the optional `--legacy-reference` and `--group-reference` arguments
to verify against existing output stores. The optional Cartesian driver
currently requires a single-time input at 200 hPa.

## Installation

As a user: activate a suitable environment then pip install:

```bash
pip install  git+https://github.com/ElliotMG/LoSSETT.git
```

As a developer: fork then clone the repository (please create a branch before making any changes!), activate a suitable Python environment, navigate to your LoSSETT directory and

```bash
pip install -e .
```

This will install as the user installation but using the editable cloned code. Please commit code improvements and discuss merging with the master branch with Elliot McKinnon-Gray, Dan Shipley, and other users.

Alternatively:
1. Install poetry [https://github.com/python-poetry/poetry], `pip install poetry`.
2. Clone the repository & install using `poetry`:
   ```bash
   git clone git@github.com:ElliotMG/LoSSETT.git
   cd LoSSETT
   poetry install
   ```
