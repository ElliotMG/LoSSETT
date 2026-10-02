import time
import numpy as np

##############################################################################
# USER SETTINGS
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
# TIMING HELPER
##############################################################################

def benchmark(name, func):

    checksum = 0.0

    t0 = time.perf_counter()

    for _ in range(NREP):

        result = func()

        if isinstance(result, tuple):
            checksum += float(
                result[0].ravel()[0]
            )
        else:
            checksum += float(
                result.ravel()[0]
            )

    elapsed = (
        time.perf_counter()
        - t0
    ) / NREP

    print(
        f"{name:<20}"
        f"{elapsed:.6f} s"
    )

    return elapsed, checksum

##############################################################################
# STAGE: du_t + du_n
##############################################################################

def stage_du_t_du_n():

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
# STAGE: dw
##############################################################################

def stage_dw():

    return w - w0

##############################################################################
# PRECOMPUTE REFERENCES
##############################################################################

du_t_ref, du_n_ref = stage_du_t_du_n()

dw_ref = stage_dw()

##############################################################################
# STAGE: du_sq
##############################################################################

def stage_du_sq():

    return (
        du_t_ref * du_t_ref
        + du_n_ref * du_n_ref
        + dw_ref * dw_ref
    )

##############################################################################
# PRECOMPUTE du_sq
##############################################################################

du_sq_ref = stage_du_sq()

##############################################################################
# STAGE: outputs
##############################################################################

def stage_outputs():

    du_long = (
        du_t_ref * du_sq_ref
    )

    du_vert = (
        dw_ref * du_sq_ref
    )

    return du_long, du_vert

##############################################################################
# WHOLE KERNEL
##############################################################################

def whole_kernel():

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

    du_long = du_t * du_sq
    du_vert = dw * du_sq

    return du_long, du_vert

##############################################################################
# WARMUP
##############################################################################

for _ in range(3):
    whole_kernel()

##############################################################################
# RUN BENCHMARKS
##############################################################################

print()
print("TIMINGS")
print("-------")

t_du, _ = benchmark(
    "du_t + du_n",
    stage_du_t_du_n,
)

t_dw, _ = benchmark(
    "dw",
    stage_dw,
)

t_sq, _ = benchmark(
    "du_sq",
    stage_du_sq,
)

t_out, _ = benchmark(
    "du3 outputs",
    stage_outputs,
)

t_total, _ = benchmark(
    "whole kernel",
    whole_kernel,
)

##############################################################################
# SUMMARY
##############################################################################

print()
print("FRACTION OF WHOLE KERNEL")
print("------------------------")

print(
    f"du_t + du_n : "
    f"{100*t_du/t_total:.1f}%"
)

print(
    f"dw          : "
    f"{100*t_dw/t_total:.1f}%"
)

print(
    f"du_sq       : "
    f"{100*t_sq/t_total:.1f}%"
)

print(
    f"du3 outputs : "
    f"{100*t_out/t_total:.1f}%"
)

print()

print(
    f"sum(stages) : "
    f"{(t_du+t_dw+t_sq+t_out):.6f} s"
)

print(
    f"whole kernel: "
    f"{t_total:.6f} s"
)
