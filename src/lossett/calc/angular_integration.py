import numpy as np
from numba import njit

def voronoi_widths_periodic(chi):
    """
    Compute Voronoi cells widths given angular cell centres chi.
    Assumes chi is sorted and unique.
    Correct handling of periodicity.
    """

    dchi = np.diff(
        np.concatenate(
            [chi, [chi[0] + 2*np.pi]]
        )
    )

    return 0.5 * (
        dchi
        + np.roll(dchi, 1)
    )

def bin_integrate(values, bins, nbins, weights=None):
    if weights is None:
        sum_bin = np.bincount(
            bins,
            weights=values,
            minlength=nbins,
        )
        
    else:    
        sum_bin = np.bincount(
            bins,
            weights=values*weights,
            minlength=nbins,
        )
    
    if len(sum_bin) > nbins:
        raise ValueError(
            "Found distance-bin overflow."
        )

    return sum_bin

@njit(cache=True)
def angular_integral_weighted_numba(
    integrand,
    bins,
    nbins,
    weights,
):
    out = np.zeros(nbins, dtype=np.float64)

    for i in range(integrand.size):

        val = integrand[i]

        if not np.isnan(val):
            out[bins[i]] += val * weights[i]

    return out

@njit(cache=True)
def angular_integral_unweighted_numba(
    integrand,
    bins,
    nbins,
):
    sum_bin = np.zeros(nbins, dtype=np.float64)
    n_total = np.zeros(nbins, dtype=np.int64)

    for i in range(integrand.size):

        b = bins[i]

        n_total[b] += 1

        val = integrand[i]

        if not np.isnan(val):
            sum_bin[b] += val

    out = np.empty(nbins, dtype=np.float64)

    for b in range(nbins):
        if n_total[b] > 0:
            out[b] = (
                2*np.pi
                * sum_bin[b]
                / n_total[b]
            )
        else:
            out[b] = np.nan

    return out

def angular_integral_by_distance_bin(
    integrand,
    bins,
    nbins,
    weights=None,
):
    """
    Compute angular integral for a single origin latitude.

    Parameters
    ----------
    integrand : ndarray
        Integrand values.

    bins : ndarray
        Distance-bin indices corresponding to integrand. Includes both
        valid and invalid points.

    nbins : int
        Number of distance bins.

    weights : ndarray or None
        Angular quadrature weights.
        If None, use uniform weighting.

    Returns
    -------
    integral : ndarray
        Angular integral as a function of distance.

    Description
    -----------
    
    Unweighted mode computes

        (2π / N) Σ f

    where N is the total number of samples in the shell.

    Weighted mode computes

        Σ f Δχ

    using supplied angular quadrature weights.

    """
    if weights is not None:
        if weights.shape != integrand.shape:
            raise ValueError(
                "weights and integrand must have the same shape!"
            )

    valid = np.isfinite(integrand)

    integrand = integrand[valid]
    bins_valid = bins[valid]

    if weights is not None:

        weights = weights[valid]

        return angular_integral_weighted_numba(
            integrand,
            bins_valid,
            nbins,
            weights,
        )
        #return bin_integrate(
        #    integrand,
        #    bins_valid,
        #    nbins,
        #    weights=weights,
        #)

    else:
        #n_total = np.bincount(
        #    bins,
        #    minlength=nbins,
        #)
        #return np.divide(
        #    2*np.pi * bin_integrate(
        #        integrand,
        #        bins_valid,
        #        nbins,
        #    ),
        #    n_total,
        #    out=np.full(nbins, np.nan),
        #    where=n_total > 0,
        #)
        return angular_integral_unweighted_numba(
            integrand,
            bins_valid,
            nbins,
        )
