import time
import numpy as np

from lossett.calc.spherical_geometry import (
    compute_initial_bearing_trig,
    compute_final_bearing_trig,
)

from lossett.calc.field_increments import (
    compute_delta_u_cubed_NEW,
)

##############################################################################
# USER SETTINGS
##############################################################################

DLAT = 0.25
NCHUNK = 16

NREP = 20
NTRIAL = 5

##############################################################################
# GRID
##############################################################################

nlat = int(180 / DLAT) + 1
nlon = int(360 / DLAT)

print(
    f"nlat={nlat:,}, "
    f"nlon={nlon:,}, "
    f"nchunk={NCHUNK:,}, "
    f"nrep={NREP}, "
    f"ntrial={NTRIAL}"
)

##############################################################################
# GEOMETRY INPUTS
##############################################################################

origin_latitudes = np.linspace(
    -90,
    90,
    NCHUNK,
    dtype=np.float64,
)

target_latitudes = np.linspace(
    -90,
    90,
    nlat,
    dtype=np.float64,
)

delta_longitudes = np.linspace(
    -180,
    180,
    nlon,
    endpoint=False,
    dtype=np.float64,
)

lat0 = np.deg2rad(origin_latitudes)[:, None, None]
lat = np.deg2rad(target_latitudes)[None, :, None]
dlon = np.deg2rad(delta_longitudes)[None, None, :]

sin_lat0 = np.sin(lat0)
cos_lat0 = np.cos(lat0)

sin_lat = np.sin(lat)
cos_lat = np.cos(lat)

sin_dlon = np.sin(dlon)
cos_dlon = np.cos(dlon)

##############################################################################
# PRECOMPUTE GEOMETRY
##############################################################################

sin_init, cos_init = (
    compute_initial_bearing_trig(
        sin_dlon,
        cos_dlon,
        sin_lat0,
        cos_lat0,
        sin_lat,
        cos_lat,
    )
)

sin_final, cos_final = (
    compute_final_bearing_trig(
        sin_dlon,
        cos_dlon,
        sin_lat0,
        cos_lat0,
        sin_lat,
        cos_lat,
    )
)

##############################################################################
# DU3 INPUTS
##############################################################################

rng = np.random.default_rng(1234)

u = rng.standard_normal((nlat, nlon))
v = rng.standard_normal((nlat, nlon))
w = rng.standard_normal((nlat, nlon))

u0 = rng.standard_normal((NCHUNK, 1, 1))
v0 = rng.standard_normal((NCHUNK, 1, 1))
w0 = rng.standard_normal((NCHUNK, 1, 1))

##############################################################################
# MEMORY INFORMATION
##############################################################################

geom_bytes = (
    sin_init.nbytes
    + cos_init.nbytes
    + sin_final.nbytes
    + cos_final.nbytes
)

print()
print(
    f"Stored bearings = "
    f"{geom_bytes / 1024**2:.1f} MB"
)

##############################################################################
# WARMUP
##############################################################################

for _ in range(5):

    compute_initial_bearing_trig(
        sin_dlon,
        cos_dlon,
        sin_lat0,
        cos_lat0,
        sin_lat,
        cos_lat,
    )

    compute_final_bearing_trig(
        sin_dlon,
        cos_dlon,
        sin_lat0,
        cos_lat0,
        sin_lat,
        cos_lat,
    )

    compute_delta_u_cubed_NEW(
        u,
        v,
        u0,
        v0,
        sin_init,
        cos_init,
        sin_final,
        cos_final,
        w=w,
        w0=w0,
    )

##############################################################################
# BENCHMARKS
##############################################################################

def benchmark_recompute_geometry():

    checksum = 0.0

    t0 = time.perf_counter()

    for _ in range(NREP):

        si, ci = compute_initial_bearing_trig(
            sin_dlon,
            cos_dlon,
            sin_lat0,
            cos_lat0,
            sin_lat,
            cos_lat,
        )

        sf, cf = compute_final_bearing_trig(
            sin_dlon,
            cos_dlon,
            sin_lat0,
            cos_lat0,
            sin_lat,
            cos_lat,
        )

        checksum += float(si[0, 1, 1])

    return (
        (time.perf_counter() - t0) / NREP,
        checksum,
    )


def benchmark_stream_geometry():

    checksum = 0.0

    t0 = time.perf_counter()

    for _ in range(NREP):

        checksum += (
            float(np.sum(sin_init))
            + float(np.sum(cos_init))
            + float(np.sum(sin_final))
            + float(np.sum(cos_final))
        )

    return (
        (time.perf_counter() - t0) / NREP,
        checksum,
    )


def benchmark_du3_precomputed():

    checksum = 0.0

    t0 = time.perf_counter()

    for _ in range(NREP):

        long_, vert_ = compute_delta_u_cubed_NEW(
            u,
            v,
            u0,
            v0,
            sin_init,
            cos_init,
            sin_final,
            cos_final,
            w=w,
            w0=w0,
        )

        checksum += float(long_[0, 0, 0])

    return (
        (time.perf_counter() - t0) / NREP,
        checksum,
    )


def benchmark_du3_recompute():

    checksum = 0.0

    t0 = time.perf_counter()

    for _ in range(NREP):

        si, ci = compute_initial_bearing_trig(
            sin_dlon,
            cos_dlon,
            sin_lat0,
            cos_lat0,
            sin_lat,
            cos_lat,
        )

        sf, cf = compute_final_bearing_trig(
            sin_dlon,
            cos_dlon,
            sin_lat0,
            cos_lat0,
            sin_lat,
            cos_lat,
        )

        long_, vert_ = compute_delta_u_cubed_NEW(
            u,
            v,
            u0,
            v0,
            si,
            ci,
            sf,
            cf,
            w=w,
            w0=w0,
        )

        checksum += float(long_[0, 0, 0])

    return (
        (time.perf_counter() - t0) / NREP,
        checksum,
    )

##############################################################################
# RUN TRIALS
##############################################################################

recompute_geom = []
stream_geom = []
du3_pre = []
du3_recompute = []

print()
print("TRIAL RESULTS")
print("-------------")

for itrial in range(NTRIAL):

    t1, _ = benchmark_recompute_geometry()
    t2, _ = benchmark_stream_geometry()
    t3, _ = benchmark_du3_precomputed()
    t4, _ = benchmark_du3_recompute()

    recompute_geom.append(t1)
    stream_geom.append(t2)
    du3_pre.append(t3)
    du3_recompute.append(t4)

    print(
        f"Trial {itrial+1}: "
        f"geom={t1:.4f}s "
        f"stream={t2:.4f}s "
        f"du3_pre={t3:.4f}s "
        f"du3_recompute={t4:.4f}s"
    )

##############################################################################
# SUMMARY
##############################################################################

print()
print("SUMMARY")
print("-------")

print(
    f"Geometry recompute : "
    f"{np.mean(recompute_geom):.6f} s"
)

print(
    f"Stream geometry    : "
    f"{np.mean(stream_geom):.6f} s"
)

print(
    f"du3 precomputed    : "
    f"{np.mean(du3_pre):.6f} s"
)

print(
    f"du3 recompute geom : "
    f"{np.mean(du3_recompute):.6f} s"
)

print()
print(
    "Extra cost of recomputing geometry:"
)

print(
    f"{np.mean(du3_recompute) / np.mean(du3_pre):.2f}x"
)

print()
print(
    "Geometry recompute / stream ratio:"
)

print(
    f"{np.mean(recompute_geom) / np.mean(stream_geom):.2f}x"
)
