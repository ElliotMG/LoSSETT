import time
import numpy as np
import numexpr as ne

from lossett.calc.field_increments import (
    EXPR_DUT,
    EXPR_DUN,
    EXPR_DUSQ,
    EXPR_DU3_LONG_FUSED,
    EXPR_DU3_VERT_FUSED,
)

##############################################################################
# SETTINGS
##############################################################################

#DLAT = 1.0
#DLAT = 0.5
#DLAT = 0.25
#DLAT = 0.125
DLAT = 0.0625
NCHUNK = 16
NREP = 20

nlat = int(180 / DLAT) + 1
nlon = int(360 / DLAT)

print(
    f"nlat={nlat:,}, "
    f"nlon={nlon:,}, "
    f"nchunk={NCHUNK:,}, "
    f"nrep={NREP}"
)

print(
    f"NumExpr threads = "
    f"{ne.get_num_threads()}"
)

##############################################################################
# TEST DATA
##############################################################################

rng = np.random.default_rng(1234)

shape = (
    NCHUNK,
    nlat,
    nlon,
)

u = rng.standard_normal(
    (nlat, nlon)
)

v = rng.standard_normal(
    (nlat, nlon)
)

w = rng.standard_normal(
    (nlat, nlon)
)

u0 = rng.standard_normal(
    (NCHUNK, 1, 1)
)

v0 = rng.standard_normal(
    (NCHUNK, 1, 1)
)

w0 = rng.standard_normal(
    (NCHUNK, 1, 1)
)

sin_init = rng.standard_normal(shape)
cos_init = rng.standard_normal(shape)

sin_final = rng.standard_normal(shape)
cos_final = rng.standard_normal(shape)

##############################################################################
# PURE NUMPY
##############################################################################

def kernel_numpy():

    du_t = (
        u * sin_final
        + v * cos_final
        - u0 * sin_init
        - v0 * cos_init
    )

    du_n = (
        u * cos_final
        - v * sin_final
        - u0 * cos_init
        + v0 * sin_init
    )

    dw = w - w0

    du_sq = (
        du_t * du_t
        + du_n * du_n
        + dw * dw
    )

    du_long = (
        du_t * du_sq
    )

    du_vert = (
        dw * du_sq
    )

    del du_t
    del du_n
    del dw
    del du_sq

    return du_long, du_vert


##############################################################################
# HYBRID NUMEXPR
##############################################################################

def kernel_hybrid_numexpr():

    du_t = EXPR_DUT(
        cos_final,
        cos_init,
        sin_final,
        sin_init,
        u,
        u0,
        v,
        v0,
    )

    du_n = EXPR_DUN(
        cos_final,
        cos_init,
        sin_final,
        sin_init,
        u,
        u0,
        v,
        v0,
    )

    dw = w - w0

    du_sq = EXPR_DUSQ(
        du_n,
        du_t,
        dw,
    )

    du_long = (
        du_t * du_sq
    )

    du_vert = (
        dw * du_sq
    )

    del du_t
    del du_n
    del dw
    del du_sq

    return du_long, du_vert

##############################################################################
# FULLY FUSED NUMEXPR
##############################################################################

def kernel_fused_numexpr():

    du_long = EXPR_DU3_LONG_FUSED(
        cos_final,
        cos_init,
        sin_final,
        sin_init,
        u,
        u0,
        v,
        v0,
        w,
        w0,
    )

    du_vert = EXPR_DU3_VERT_FUSED(
        cos_final,
        cos_init,
        sin_final,
        sin_init,
        u,
        u0,
        v,
        v0,
        w,
        w0,
    )

    return du_long, du_vert

##############################################################################
# WARMUP
##############################################################################

for _ in range(3):

    kernel_numpy()

    kernel_hybrid_numexpr()

    kernel_fused_numexpr()

##############################################################################
# VALIDATION
##############################################################################

long_np, vert_np = kernel_numpy()

long_h, vert_h = kernel_hybrid_numexpr()

long_f, vert_f = kernel_fused_numexpr()

print()

print(
    "max abs diff hybrid long:",
    np.max(
        np.abs(
            long_np - long_h
        )
    )
)

print(
    "max abs diff hybrid vert:",
    np.max(
        np.abs(
            vert_np - vert_h
        )
    )
)

print(
    "max abs diff fused long:",
    np.max(
        np.abs(
            long_np - long_f
        )
    )
)

print(
    "max abs diff fused vert:",
    np.max(
        np.abs(
            vert_np - vert_f
        )
    )
)

##############################################################################
# BENCHMARK
##############################################################################

def benchmark(label, func):

    checksum = 0.0

    t0 = time.perf_counter()

    for _ in range(NREP):

        long_, vert_ = func()

        checksum += float(
            long_[0, 0, 0]
        )

    elapsed = (
        time.perf_counter()
        - t0
    ) / NREP

    print(
        f"{label:<12}"
        f"{elapsed:.6f} s"
    )

    return elapsed, checksum

##############################################################################
# RUN
##############################################################################

print()
print("TIMINGS")
print("-------")

t_numpy, chk_numpy = benchmark(
    "NumPy",
    kernel_numpy,
)

t_hybrid, chk_hybrid = benchmark(
    "Hybrid",
    kernel_hybrid_numexpr,
)

t_fused, chk_fused = benchmark(
    "Fused",
    kernel_fused_numexpr,
)

print()
print("CHECKSUMS")
print("---------")

print(
    "NumPy :",
    chk_numpy,
)

print(
    "Hybrid:",
    chk_hybrid,
)

print(
    "Fused :",
    chk_fused,
)

print()
print(
    "Hybrid speedup:",
    f"{t_numpy / t_hybrid:.2f}x"
)

print(
    "Fused speedup :",
    f"{t_numpy / t_fused:.2f}x"
)

print(
    "Fused vs Hybrid:",
    f"{t_hybrid / t_fused:.2f}x"
)
