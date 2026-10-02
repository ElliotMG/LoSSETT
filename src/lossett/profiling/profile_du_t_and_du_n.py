import time
import numpy as np
import numexpr as ne

from lossett.calc.field_increments import (
    EXPR_DUT,
    EXPR_DUN,
)

##############################################################################
# SETTINGS
##############################################################################

DLAT = 0.125
NCHUNK = 16
NREP = 20

##############################################################################
# GRID
##############################################################################

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
# DATA
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

u0 = rng.standard_normal(
    (NCHUNK, 1, 1)
)

v0 = rng.standard_normal(
    (NCHUNK, 1, 1)
)

sin_init = rng.standard_normal(shape)
cos_init = rng.standard_normal(shape)

sin_final = rng.standard_normal(shape)
cos_final = rng.standard_normal(shape)

##############################################################################
# NUMPY IMPLEMENTATION
##############################################################################

def numpy_du_t_du_n():

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

    return du_t, du_n

##############################################################################
# NUMEXPR IMPLEMENTATION
##############################################################################

def numexpr_du_t_du_n():

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

    return du_t, du_n

##############################################################################
# WARMUP
##############################################################################

for _ in range(3):

    numpy_du_t_du_n()
    numexpr_du_t_du_n()

##############################################################################
# VALIDATION
##############################################################################

du_t_np, du_n_np = numpy_du_t_du_n()

du_t_ne, du_n_ne = numexpr_du_t_du_n()

print()

print(
    "max abs diff du_t:",
    np.max(
        np.abs(
            du_t_np - du_t_ne
        )
    )
)

print(
    "max abs diff du_n:",
    np.max(
        np.abs(
            du_n_np - du_n_ne
        )
    )
)

##############################################################################
# TIMING HELPER
##############################################################################

def benchmark(label, func):

    checksum = 0.0

    t0 = time.perf_counter()

    for _ in range(NREP):

        du_t, du_n = func()

        checksum += float(
            du_t[0, 0, 0]
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
    numpy_du_t_du_n,
)

t_numexpr, chk_numexpr = benchmark(
    "NumExpr",
    numexpr_du_t_du_n,
)

print()
print("CHECKSUMS")
print("---------")

print(
    "NumPy  :",
    chk_numpy,
)

print(
    "NumExpr:",
    chk_numexpr,
)

print()
print(
    "NumExpr speedup:",
    f"{t_numpy / t_numexpr:.2f}x"
)
