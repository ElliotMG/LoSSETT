import time
import numpy as np
import numexpr as ne

from lossett.calc.field_increments import (
    EXPR_DUSQ,
)

##############################################################################
# SETTINGS
##############################################################################

DLAT = 0.125
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
# DATA
##############################################################################

rng = np.random.default_rng(1234)

shape = (
    NCHUNK,
    nlat,
    nlon,
)

du_t = rng.standard_normal(shape)

du_n = rng.standard_normal(shape)

dw = rng.standard_normal(shape)

##############################################################################
# NUMPY IMPLEMENTATION
##############################################################################

def numpy_du_sq():

    return (
        du_t * du_t
        + du_n * du_n
        + dw * dw
    )

##############################################################################
# NUMEXPR IMPLEMENTATION
##############################################################################

def numexpr_du_sq():

    return EXPR_DUSQ(
        du_n,
        du_t,
        dw,
    )

##############################################################################
# WARMUP
##############################################################################

for _ in range(3):

    numpy_du_sq()

    numexpr_du_sq()

##############################################################################
# VALIDATION
##############################################################################

du_sq_np = numpy_du_sq()

du_sq_ne = numexpr_du_sq()

print()

print(
    "max abs diff:",
    np.max(
        np.abs(
            du_sq_np - du_sq_ne
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

        du_sq = func()

        checksum += float(
            du_sq[0, 0, 0]
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
    numpy_du_sq,
)

t_numexpr, chk_numexpr = benchmark(
    "NumExpr",
    numexpr_du_sq,
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
