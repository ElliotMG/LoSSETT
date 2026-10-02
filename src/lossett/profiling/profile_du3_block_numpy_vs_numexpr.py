import numpy as np
import time

from lossett.calc.field_increments import (
    compute_delta_u_cubed_NEW,
)

##############################################################################
# USER SETTINGS
##############################################################################

DLAT = 1.0

NCHUNK = 16

NBLOCK = 32

NREP = 10

##############################################################################
# GRID SIZE
##############################################################################

nlat = int(180 / DLAT) + 1
nlon = int(360 / DLAT)

print(
    f"nlat={nlat:,}, "
    f"nlon={nlon:,}, "
    f"nchunk={NCHUNK:,}, "
    f"nblock={NBLOCK:,}"
)

##############################################################################
# SYNTHETIC DATA
##############################################################################

rng = np.random.default_rng(1234)

#
# One field per origin longitude
#
u_serial = rng.standard_normal(
    (NBLOCK, nlat, nlon)
)

v_serial = rng.standard_normal(
    (NBLOCK, nlat, nlon)
)

w_serial = rng.standard_normal(
    (NBLOCK, nlat, nlon)
)

#
# One origin value per
# (origin longitude, origin latitude)
#
u0_serial = rng.standard_normal(
    (NBLOCK, NCHUNK, 1, 1)
)

v0_serial = rng.standard_normal(
    (NBLOCK, NCHUNK, 1, 1)
)

w0_serial = rng.standard_normal(
    (NBLOCK, NCHUNK, 1, 1)
)

shape_geom = (
    NBLOCK,
    NCHUNK,
    nlat,
    nlon,
)

sin_init = rng.standard_normal(shape_geom)
cos_init = rng.standard_normal(shape_geom)

sin_final = rng.standard_normal(shape_geom)
cos_final = rng.standard_normal(shape_geom)

##############################################################################
# SERIAL WARMUP
##############################################################################

compute_delta_u_cubed_NEW(
    u_serial[0],
    v_serial[0],
    u0_serial[0],
    v0_serial[0],
    sin_init[0],
    cos_init[0],
    sin_final[0],
    cos_final[0],
    w=w_serial[0],
    w0=w0_serial[0],
)

##############################################################################
# BLOCKED ARRAYS
##############################################################################

u_block = u_serial[:, None, :, :]
v_block = v_serial[:, None, :, :]
w_block = w_serial[:, None, :, :]

u0_block = u0_serial
v0_block = v0_serial
w0_block = w0_serial

##############################################################################
# SHAPE CHECK
##############################################################################

serial_result = compute_delta_u_cubed_NEW(
    u_serial[0],
    v_serial[0],
    u0_serial[0],
    v0_serial[0],
    sin_init[0],
    cos_init[0],
    sin_final[0],
    cos_final[0],
    w=w_serial[0],
    w0=w0_serial[0],
)

blocked_result = compute_delta_u_cubed_NEW(
    u_block,
    v_block,
    u0_block,
    v0_block,
    sin_init,
    cos_init,
    sin_final,
    cos_final,
    w=w_block,
    w0=w0_block,
)

print()
print(
    "serial result shape =",
    serial_result[0].shape,
)

print(
    "blocked result shape =",
    blocked_result[0].shape,
)

##############################################################################
# VALIDATION
##############################################################################

long_serial = np.stack(
    [
        compute_delta_u_cubed_NEW(
            u_serial[j],
            v_serial[j],
            u0_serial[j],
            v0_serial[j],
            sin_init[j],
            cos_init[j],
            sin_final[j],
            cos_final[j],
            w=w_serial[j],
            w0=w0_serial[j],
        )[0]
        for j in range(NBLOCK)
    ],
    axis=0,
)

long_block = blocked_result[0]

print()

print(
    "max abs diff =",
    np.max(
        np.abs(
            long_serial
            - long_block
        )
    )
)

##############################################################################
# SERIAL TIMING
##############################################################################

t0 = time.perf_counter()

for _ in range(NREP):

    for j in range(NBLOCK):

        compute_delta_u_cubed_NEW(
            u_serial[j],
            v_serial[j],
            u0_serial[j],
            v0_serial[j],
            sin_init[j],
            cos_init[j],
            sin_final[j],
            cos_final[j],
            w=w_serial[j],
            w0=w0_serial[j],
        )

serial_time = (
    time.perf_counter() - t0
)

##############################################################################
# BLOCK TIMING
##############################################################################

t0 = time.perf_counter()

for _ in range(NREP):

    compute_delta_u_cubed_NEW(
        u_block,
        v_block,
        u0_block,
        v0_block,
        sin_init,
        cos_init,
        sin_final,
        cos_final,
        w=w_block,
        w0=w0_block,
    )

blocked_time = (
    time.perf_counter() - t0
)

##############################################################################
# RESULTS
##############################################################################

serial_per_call = serial_time / NREP
blocked_per_call = blocked_time / NREP

print()
print(
    f"Serial   : {serial_per_call:.6f} s"
)

print(
    f"Blocked  : {blocked_per_call:.6f} s"
)

print(
    f"Speedup  : "
    f"{serial_per_call / blocked_per_call:.2f}x"
)
