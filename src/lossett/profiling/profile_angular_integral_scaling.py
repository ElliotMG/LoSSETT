import time
import numpy as np

from lossett.calc.angular_integration import (
    angular_integral_by_distance_bin,
)


def benchmark_wrapper(
    M,
    nbins,
    nrepeat=10,
    frac_nan=0.0,
    seed=1234,
):

    rng = np.random.default_rng(seed)

    integrand = rng.standard_normal(M)

    if frac_nan > 0:
        mask = rng.random(M) < frac_nan
        integrand[mask] = np.nan

    weights = rng.random(M)

    bins = rng.integers(
        0,
        nbins,
        size=M,
        dtype=np.int32,
    )

    #
    # warm-up
    #
    angular_integral_by_distance_bin(
        integrand,
        bins,
        nbins,
        weights=weights,
    )

    times = []

    for _ in range(nrepeat):

        t0 = time.perf_counter()

        angular_integral_by_distance_bin(
            integrand,
            bins,
            nbins,
            weights=weights,
        )

        times.append(
            time.perf_counter() - t0
        )

    return (
        np.mean(times),
        np.std(times),
    )


if __name__ == "__main__":

    NREP=50

    cases = [

        # ~1°
        (1042560, 180),

        # ~0.5°
        (4177920, 360),

        # ~0.25°
        (16711680, 720),
    ]

    print()
    print(
        f"{'M':>12s} "
        f"{'nbins':>8s} "
        f"{'time (s)':>12s} "
        f"{'ns/point':>12s}"
    )

    for M, nbins in cases:

        mean_time, std_time = benchmark_wrapper(
            M,
            nbins,
            nrepeat=NREP,
            frac_nan=0.0,
        )

        ns_per_point = (
            mean_time / M * 1e9
        )

        print(
            f"{M:12d} "
            f"{nbins:8d} "
            f"{mean_time:12.6f} "
            f"{ns_per_point:12.3f}"
        )
