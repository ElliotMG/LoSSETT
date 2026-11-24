"""Transport-only comparison of computed and precomputed rotation geometry."""
import json
import socket
import time
from pathlib import Path

import numpy as np
import xarray as xr
from numba import njit

from lossett.calc._group_numba import _transport
from lossett.calc.group_increments import _transport as numpy_transport, latlon_shell_fields


@njit
def transport_sweep(directions, velocity, active, matrix, origin_row, nlon, precomputed):
    results = np.zeros((nlon, 6))
    for lon in range(nlon):
        x = directions[origin_row * nlon + lon]
        u = velocity[origin_row * nlon + lon]
        wx = np.dot(x, u)
        vx = u - wx * x
        angle = 2 * np.pi * lon / nlon
        c, s = np.cos(angle), np.sin(angle)
        for n in range(len(active)):
            j = (active[n] // nlon) * nlon + (active[n] % nlon + lon) % nlon
            y, u_y = directions[j], velocity[j]
            wy = np.dot(y, u_y)
            vy = u_y - wy * y
            if precomputed:
                vx0, vx1, vx2 = c * vx[0] + s * vx[1], -s * vx[0] + c * vx[1], vx[2]
                vy0, vy1, vy2 = c * vy[0] + s * vy[1], -s * vy[0] + c * vy[1], vy[2]
                m = matrix[n]
                p0 = m[0, 0] * vx0 + m[0, 1] * vx1 + m[0, 2] * vx2
                p1 = m[1, 0] * vx0 + m[1, 1] * vx1 + m[1, 2] * vx2
                p2 = m[2, 0] * vx0 + m[2, 1] * vx1 + m[2, 2] * vx2
                b0 = m[0, 0] * vy0 + m[1, 0] * vy1 + m[2, 0] * vy2
                b1 = m[0, 1] * vy0 + m[1, 1] * vy1 + m[2, 1] * vy2
                b2 = m[0, 2] * vy0 + m[1, 2] * vy1 + m[2, 2] * vy2
                results[lon, 0] += c * p0 - s * p1
                results[lon, 1] += s * p0 + c * p1
                results[lon, 2] += p2
                results[lon, 3] += c * b0 - s * b1
                results[lon, 4] += s * b0 + c * b1
                results[lon, 5] += b2
            else:
                p = _transport(vx[0], vx[1], vx[2], x[0], x[1], x[2], y[0], y[1], y[2])
                back = _transport(vy[0], vy[1], vy[2], y[0], y[1], y[2], x[0], x[1], x[2])
                for a in range(3):
                    results[lon, a] += p[a]
                    results[lon, a + 3] += back[a]
    return results


input_path = Path("/gws/ssde/j25b/kscale/USERS/dship/LoSSETT_in/preprocessed_kscale_data/DYAMOND_SUMMER/benchmark_antialiased_20160801T00/filtered_2p0deg.nc")
output = Path("/work/scratch-pw5/dship/upscale/LoSSETT/spherical_geometry/real_pilot_20160801T00_2p0deg/compact_matched_comparison_v2/geometry_reuse_v1.json")
if output.exists():
    raise FileExistsError(output)
with xr.open_dataset(input_path, decode_timedelta=False) as source:
    field = source.isel(time=0).sel(pressure=200.).load()
lat = np.deg2rad(field.latitude.values)
lon = np.deg2rad(field.longitude.values)
directions, velocity, _, _ = latlon_shell_fields(
    lat, lon, *(field[name].values for name in ("u", "v", "w")), radius=6382758.772639869,
)
nlon = len(lon)
directions, velocity = directions.reshape(-1, 3), velocity.reshape(-1, 3)
report = {"host": socket.gethostname(), "scope": "transport-only sweep, not the complete diagnostic",
          "rows": []}
for row in (0, 30, 45):
    x = directions[row * nlon]
    start = time.perf_counter()
    mu = np.clip(directions @ x, -1, 1)
    active = np.flatnonzero(mu >= np.cos(4 * 0.1566720654220191) - 64 * np.finfo(float).eps)
    y = directions[active]
    matrix = np.stack([numpy_transport(np.eye(3)[a], x, y) for a in range(3)], axis=-1)
    setup = time.perf_counter() - start
    transport_sweep(directions, velocity, active[:1], matrix[:1], row, nlon, False)
    transport_sweep(directions, velocity, active[:1], matrix[:1], row, nlon, True)
    times = {}
    values = {}
    for method, precomputed in (("computed", False), ("precomputed", True)):
        times[method] = []
        for _ in range(3):
            start = time.perf_counter()
            values[method] = transport_sweep(directions, velocity, active, matrix, row, nlon, precomputed)
            times[method].append(time.perf_counter() - start)
    np.testing.assert_allclose(values["computed"], values["precomputed"], rtol=1e-10, atol=1e-8)
    result = {"latitude_degrees": float(np.rad2deg(lat[row])), "targets": len(active),
              "geometry_setup_seconds": setup, "geometry_bytes": matrix.nbytes + active.nbytes,
              "seconds": times, "max_absolute_sum_difference": float(np.max(np.abs(values["computed"] - values["precomputed"])))}
    report["rows"].append(result)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(result), flush=True)
