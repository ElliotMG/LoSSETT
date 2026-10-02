import numpy as np
import time
from numba import njit

from lossett.calc.angular_integration import (
    angular_integral_unweighted_numba,
)

##############################################################################
# USER SETTINGS
##############################################################################

DLAT = 0.25
NBLOCK = 128
NREP = 10

##############################################################################
# BLOCK IMPLEMENTATION
##############################################################################

@njit(cache=True)
def angular_integral_block_numba(
    integrand,
    bins,
    nbins,
):
    """
    Parameters
    ----------
    integrand : ndarray
        Shape (nblock, npoints)

    bins : ndarray
        Shape (npoints,)

    Returns
    -------
    out : ndarray
        Shape (nblock, nbins)
    """

    nblock = integrand.shape[0]

    sum_bin = np.zeros(
        (nblock, nbins),
        dtype=np.float64,
    )

    n_total = np.zeros(
        nbins,
        dtype=np.int64,
    )

    for i in range(bins.size):
        n_total[bins[i]] += 1

    for j in range(nblock):

        for i in range(integrand.shape[1]):

            val = integrand[j, i]

            if not np.isnan(val):
                sum_bin[j, bins[i]] += val

    out = np.empty(
        (nblock, nbins),
        dtype=np.float64,
    )

    for b in range(nbins):

        if n_total[b] > 0:

            fac = 2.0 * np.pi / n_total[b]

            for j in range(nblock):
                out[j, b] = fac * sum_bin[j, b]

        else:

            for j in range(nblock):
                out[j, b] = np.nan

    return out

##############################################################################
# SYNTHETIC TEST PROBLEM
##############################################################################

nlat = int(180 / DLAT) + 1
nlon = int(360 / DLAT)

npoints = nlat * nlon
nbins = nlon // 2

print(
    f"npoints={npoints:,}, "
    f"nbins={nbins:,}, "
    f"nblock={NBLOCK}"
)

rng = np.random.default_rng(1234)

du3_block = rng.standard_normal(
    (NBLOCK, npoints)
)

# optional: inject some NaNs
# du3_block[:, ::1000] = np.nan

bins = rng.integers(
    0,
    nbins,
    size=npoints,
    dtype=np.int64,
)

##############################################################################
# WARM-UP
##############################################################################

angular_integral_unweighted_numba(
    du3_block[0],
    bins,
    nbins,
)

angular_integral_block_numba(
    du3_block,
    bins,
    nbins,
)

##############################################################################
# VALIDATION
##############################################################################

out_old = np.empty(
    (NBLOCK, nbins)
)

for j in range(NBLOCK):

    out_old[j] = angular_integral_unweighted_numba(
        du3_block[j],
        bins,
        nbins,
    )

out_new = angular_integral_block_numba(
    du3_block,
    bins,
    nbins,
)

print(
    "max abs diff =",
    np.nanmax(
        np.abs(out_old - out_new)
    )
)

##############################################################################
# REPEATED TIMING
##############################################################################

times_old = []
times_new = []

for _ in range(NREP):

    t0 = time.perf_counter()

    for j in range(NBLOCK):

        angular_integral_unweighted_numba(
            du3_block[j],
            bins,
            nbins,
        )

    times_old.append(
        time.perf_counter() - t0
    )

    t0 = time.perf_counter()

    angular_integral_block_numba(
        du3_block,
        bins,
        nbins,
    )

    times_new.append(
        time.perf_counter() - t0
    )

##############################################################################
# RESULTS
##############################################################################

old_mean = np.mean(times_old)
old_std = np.std(times_old)

new_mean = np.mean(times_new)
new_std = np.std(times_new)

print()
print(
    f"Old kernel: "
    f"{old_mean:.6f} ± {old_std:.6f} s"
)

print(
    f"Block kernel: "
    f"{new_mean:.6f} ± {new_std:.6f} s"
)

print(
    f"Speedup: "
    f"{old_mean/new_mean:.2f}x"
)
