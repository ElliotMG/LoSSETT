"""Compare equivalent group backends on the same allocation."""
import json
import os
import time
from pathlib import Path

import numpy as np
import xarray as xr

from lossett.calc._group_numba import accumulate_group_pairs
from lossett.calc.compute_group_increments import shell_dataset

input_path = Path("/gws/ssde/j25b/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/DYAMOND_SUMMER/benchmark_antialiased_20160801T00/filtered_2p0deg.nc")
directory = Path("/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/real_pilot_20160801T00_2p0deg/compact_matched_comparison_v2")
output = directory / "numba_same_allocation_v1.json"
if output.exists():
    raise FileExistsError(output)
with xr.open_dataset(input_path, decode_timedelta=False) as source:
    data = source.isel(time=[0]).sel(pressure=[200.]).load()
with xr.open_zarr(directory / "group-shell.zarr") as source:
    reference = source.load()

x = np.array([1., 0., 0.])
points = x[None, :]
zeros = np.zeros((1, 3))
start = time.perf_counter()
accumulate_group_pairs(x, x, points, zeros, np.zeros((1, 3, 3)), zeros,
                       np.ones(1), np.ones(1), np.ones(1), np.ones(1), 0.,
                       np.array([0], dtype=np.intp))
compile_seconds = time.perf_counter() - start
report = {
    "note": "One 2-degree shell, same allocation, exact support; first call compiles with an initially empty Numba cache.",
    "numba_compile_and_one_pair_seconds": compile_seconds,
    "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR"),
    "runs": [],
}
options = dict(time_index=0, level_index=0, level_dim="pressure", coordinate_type="pressure",
               coordinate_unit="hPa", radius=6382758.772639869, mollifier_scale=0.1566720654220191,
               cache_longitude_moments=True, compact_support=True)
results = {}
for backend in ("numba", "numpy", "numba", "numpy", "numba"):
    start = time.perf_counter()
    result = shell_dataset(data, pair_backend=backend, **options)
    elapsed = time.perf_counter() - start
    errors = {}
    for name in ("horizontal", "structure", "vertical_flux", "mixed_moment", "kernel_mass", "direction", "velocity"):
        actual = result[name].transpose(*reference[name].dims).values
        expected = reference[name].values
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
        errors[name] = float(np.max(np.abs(actual - expected)))
    if "numpy" in results and backend == "numba":
        for name in errors:
            np.testing.assert_allclose(result[name], results["numpy"][name], rtol=1e-10, atol=1e-10)
    results[backend] = result
    run = {"backend": backend, "seconds": elapsed, "max_absolute_errors_vs_direct_fiber": errors,
           "kernel_mass_error": float(np.max(np.abs(result.kernel_mass.values - 1)))}
    report["runs"].append(run)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(run), flush=True)
