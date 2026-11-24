"""Measure reusable moments and an experimental exact FFT on a real shell."""
import json
import socket
import time
from pathlib import Path

import numpy as np
import xarray as xr

import lossett.calc.group_increments as increments
from lossett.calc._group_numba import accumulate_group_pairs
from lossett.calc.compute_group_increments import _solid_angles, shell_dataset
from longitude_fft_prototype import fft_group


input_path = Path("/gws/ssde/j25b/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/DYAMOND_SUMMER/benchmark_antialiased_20160801T00/filtered_2p0deg.nc")
output = Path("/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/real_pilot_20160801T00_2p0deg/compact_matched_comparison_v2/precompute_fft_v1.json")
if output.exists():
    raise FileExistsError(output)
with xr.open_dataset(input_path, decode_timedelta=False) as source:
    data = source.isel(time=[0]).sel(pressure=[200.]).load()
field = data.isel(time=0, pressure=0)
latitude = np.deg2rad(field.latitude.values)
longitude = np.deg2rad(field.longitude.values)
radius = 6382758.772639869
scale = 0.1566720654220191
weights = _solid_angles(latitude, longitude)
directions, velocity, grad, gradw = increments.latlon_shell_fields(
    latitude, longitude, *(field[name].values for name in ("u", "v", "w")), radius=radius,
)

x = np.array([1., 0., 0.])
points = x[None, :]
zeros = np.zeros((1, 3))
accumulate_group_pairs(x, x, points, zeros, np.zeros((1, 3, 3)), zeros,
                       np.ones(1), np.ones(1), np.ones(1), np.ones(1), 0.,
                       np.array([0], dtype=np.intp))
options = dict(time_index=0, level_index=0, level_dim="pressure", coordinate_type="pressure",
               coordinate_unit="hPa", radius=radius, mollifier_scale=scale,
               cache_longitude_moments=True, compact_support=True, pair_backend="numba")
start = time.perf_counter()
reference = shell_dataset(data, **options)
baseline = time.perf_counter() - start
report = {"host": socket.gethostname(), "baseline_numba_seconds": baseline, "runs": []}
cache = {}
original = increments.mollifier_fiber_moments


def memoized(mu, angular_scale, *, tolerance):
    key = (mu.tobytes(), angular_scale, tolerance)
    if key not in cache:
        cache[key] = original(mu, angular_scale, tolerance=tolerance)
    return cache[key]


try:
    increments.mollifier_fiber_moments = memoized
    for run in ("populate", "reuse"):
        start = time.perf_counter()
        result = shell_dataset(data, **options)
        seconds = time.perf_counter() - start
        errors = {}
        for name in ("horizontal", "structure", "vertical_flux", "mixed_moment", "kernel_mass"):
            np.testing.assert_allclose(result[name], reference[name], rtol=1e-10, atol=1e-10)
            errors[name] = float(np.max(np.abs(result[name].values - reference[name].values)))
        cache_bytes = sum(len(key[0]) + sum(a.nbytes for a in values) for key, values in cache.items())
        report["runs"].append({"method": "moments_" + run, "seconds": seconds,
                               "array_and_key_bytes": cache_bytes, "entries": len(cache), "errors": errors})
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["runs"][-1]), flush=True)
finally:
    increments.mollifier_fiber_moments = original

start = time.perf_counter()
fft = fft_group(latitude, longitude, velocity, grad, gradw, weights, radius=radius, mollifier_scale=scale)
seconds = time.perf_counter() - start
errors = {}
for name, actual in fft.items():
    expected = reference[name].values
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    errors[name] = float(np.max(np.abs(actual - expected)))
report["runs"].append({"method": "full_sphere_longitude_fft_prototype", "seconds": seconds, "errors": errors})
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report["runs"][-1]), flush=True)
