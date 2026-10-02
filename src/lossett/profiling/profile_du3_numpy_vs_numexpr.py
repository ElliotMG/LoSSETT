import time
import numpy as np
import numexpr as ne

from lossett.calc.field_increments import (
    compute_delta_u_cubed_NEW,
    compute_delta_u_cubed_numexpr_NEW,
    compute_delta_u_cubed_numexpr_FUSED,
)

##############################################################################
# USER SETTINGS
##############################################################################

DLAT = 1.0
NCHUNK = 16

NREP = 20
NTRIAL = 5

NUMEXPR_THREADS = 16

##############################################################################
# NUMEXPR CONFIG
##############################################################################

ne.set_num_threads(NUMEXPR_THREADS)

print(
    f"NumExpr threads = {ne.get_num_threads()}"
)

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
# DATA
##############################################################################

rng = np.random.default_rng(1234)

shape_geom = (
    NCHUNK,
    nlat,
    nlon,
)

u = rng.standard_normal((nlat, nlon))
v = rng.standard_normal((nlat, nlon))
w = rng.standard_normal((nlat, nlon))

u0 = rng.standard_normal((NCHUNK, 1, 1))
v0 = rng.standard_normal((NCHUNK, 1, 1))
w0 = rng.standard_normal((NCHUNK, 1, 1))

sin_init = rng.standard_normal(shape_geom)
cos_init = rng.standard_normal(shape_geom)

sin_final = rng.standard_normal(shape_geom)
cos_final = rng.standard_normal(shape_geom)

##############################################################################
# VALIDATION
##############################################################################

long_np, vert_np = compute_delta_u_cubed_NEW(
    u, v,
    u0, v0,
    sin_init, cos_init,
    sin_final, cos_final,
    w=w, w0=w0,
)

long_ne, vert_ne = compute_delta_u_cubed_numexpr_NEW(
    u, v,
    u0, v0,
    sin_init, cos_init,
    sin_final, cos_final,
    w=w, w0=w0,
)

long_fused, vert_fused = compute_delta_u_cubed_numexpr_FUSED(
    u, v,
    u0, v0,
    sin_init, cos_init,
    sin_final, cos_final,
    w=w, w0=w0,
)

print()
print(
    "max abs diff NumPy vs NumExpr:",
    np.max(np.abs(long_np - long_ne))
)

print(
    "max abs diff NumPy vs Fused:",
    np.max(np.abs(long_np - long_fused))
)

##############################################################################
# WARMUP
##############################################################################

for _ in range(5):

    compute_delta_u_cubed_NEW(
        u, v,
        u0, v0,
        sin_init, cos_init,
        sin_final, cos_final,
        w=w, w0=w0,
    )

    compute_delta_u_cubed_numexpr_NEW(
        u, v,
        u0, v0,
        sin_init, cos_init,
        sin_final, cos_final,
        w=w, w0=w0,
    )

    compute_delta_u_cubed_numexpr_FUSED(
        u, v,
        u0, v0,
        sin_init, cos_init,
        sin_final, cos_final,
        w=w, w0=w0,
    )

##############################################################################
# BENCHMARK
##############################################################################

def benchmark(func):

    t0 = time.perf_counter()

    checksum = 0.0

    for _ in range(NREP):

        long_, vert_ = func(
            u, v,
            u0, v0,
            sin_init, cos_init,
            sin_final, cos_final,
            w=w,
            w0=w0,
        )

        checksum += float(long_[0, 0, 0])

    elapsed = time.perf_counter() - t0

    return elapsed / NREP, checksum

##############################################################################
# TRIALS
##############################################################################

numpy_times = []
numexpr_times = []
fused_times = []

print()
print("TRIAL RESULTS")
print("-------------")

for itrial in range(NTRIAL):

    t_np, chk_np = benchmark(
        compute_delta_u_cubed_NEW
    )

    t_ne, chk_ne = benchmark(
        compute_delta_u_cubed_numexpr_NEW
    )

    t_fused, chk_fused = benchmark(
        compute_delta_u_cubed_numexpr_FUSED
    )

    numpy_times.append(t_np)
    numexpr_times.append(t_ne)
    fused_times.append(t_fused)

    print(
        f"Trial {itrial+1}: "
        f"NumPy={t_np:.6f}s  "
        f"NumExpr={t_ne:.6f}s  "
        f"Fused={t_fused:.6f}s"
    )

##############################################################################
# SUMMARY
##############################################################################

numpy_mean = np.mean(numpy_times)
numpy_std = np.std(numpy_times)

numexpr_mean = np.mean(numexpr_times)
numexpr_std = np.std(numexpr_times)

fused_mean = np.mean(fused_times)
fused_std = np.std(fused_times)

print()
print("SUMMARY")
print("-------")

print(
    f"NumPy        : "
    f"{numpy_mean:.6f} ± {numpy_std:.6f} s/call"
)

print(
    f"NumExpr      : "
    f"{numexpr_mean:.6f} ± {numexpr_std:.6f} s/call"
)

print(
    f"NumExprFused : "
    f"{fused_mean:.6f} ± {fused_std:.6f} s/call"
)

print()
print(
    f"NumExpr speedup: "
    f"{numpy_mean / numexpr_mean:.2f}x"
)

print(
    f"Fused speedup: "
    f"{numpy_mean / fused_mean:.2f}x"
)

print(
    f"Fused vs NumExpr: "
    f"{numexpr_mean / fused_mean:.2f}x"
)
