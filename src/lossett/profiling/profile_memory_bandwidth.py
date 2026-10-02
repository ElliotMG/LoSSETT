import time
import numpy as np

##############################################################################
# SETTINGS
##############################################################################

NREP = 20

# Match production-like du^3 arrays
shape = (
    16,
    721,
    1440,
)

dtype = np.float64

##############################################################################
# ARRAYS
##############################################################################

a = np.ones(shape, dtype=dtype)
b = np.ones(shape, dtype=dtype)
c = np.ones(shape, dtype=dtype)

array_size_gb = a.nbytes / 1024**3

print()
print("CONFIGURATION")
print("-------------")
print(f"shape      = {shape}")
print(f"dtype      = {dtype}")
print(f"array size = {array_size_gb:.3f} GiB")

##############################################################################
# WARMUP
##############################################################################

_ = np.sum(a)
b[:] = a
b[:] = 3.0 * a
b[:] = a + c

##############################################################################
# READ BANDWIDTH
##############################################################################

checksum = 0.0

t0 = time.perf_counter()

for _ in range(NREP):
    checksum += float(np.sum(a))

read_time = (
    time.perf_counter() - t0
) / NREP

read_bw = (
    a.nbytes
    / read_time
    / 1024**3
)

##############################################################################
# WRITE BANDWIDTH
##############################################################################

t0 = time.perf_counter()

for _ in range(NREP):
    b.fill(1.0)

write_time = (
    time.perf_counter() - t0
) / NREP

write_bw = (
    b.nbytes
    / write_time
    / 1024**3
)

##############################################################################
# COPY BANDWIDTH
##############################################################################

t0 = time.perf_counter()

for _ in range(NREP):
    b[:] = a

copy_time = (
    time.perf_counter() - t0
) / NREP

copy_bw = (
    (a.nbytes + b.nbytes)
    / copy_time
    / 1024**3
)

##############################################################################
# SCALE
#
# STREAM SCALE:
#
# b = 3.0 * a
#
# bytes moved:
#   read a
#   write b
##############################################################################

t0 = time.perf_counter()

for _ in range(NREP):
    b[:] = 3.0 * a

scale_time = (
    time.perf_counter() - t0
) / NREP

scale_bw = (
    (a.nbytes + b.nbytes)
    / scale_time
    / 1024**3
)

##############################################################################
# TRIAD
#
# STREAM TRIAD:
#
# b = a + c
#
# bytes moved:
#   read a
#   read c
#   write b
##############################################################################

t0 = time.perf_counter()

for _ in range(NREP):
    b[:] = a + c

triad_time = (
    time.perf_counter() - t0
) / NREP

triad_bw = (
    (a.nbytes + c.nbytes + b.nbytes)
    / triad_time
    / 1024**3
)

##############################################################################
# ADDITIONAL USEFUL METRIC
#
# FLOPS for TRIAD
#
# One floating-point addition per element.
##############################################################################

n_elements = a.size

triad_flops = n_elements

triad_gflops = (
    triad_flops
    / triad_time
    / 1e9
)

##############################################################################
# RESULTS
##############################################################################

print()
print("RESULTS")
print("-------")

print(
    f"Read : {read_time:.6f} s   "
    f"{read_bw:8.2f} GB/s"
)

print(
    f"Write: {write_time:.6f} s   "
    f"{write_bw:8.2f} GB/s"
)

print(
    f"Copy : {copy_time:.6f} s   "
    f"{copy_bw:8.2f} GB/s"
)

print(
    f"Scale: {scale_time:.6f} s   "
    f"{scale_bw:8.2f} GB/s"
)

print(
    f"Triad: {triad_time:.6f} s   "
    f"{triad_bw:8.2f} GB/s"
)

print()
print(
    f"Triad throughput: "
    f"{triad_gflops:.2f} GFLOP/s"
)

print()
print(
    f"Checksum: {checksum:.6e}"
)
