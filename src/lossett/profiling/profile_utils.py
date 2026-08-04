#!/usr/bin/env python3
from contextlib import contextmanager
import time
import logging

logger = logging.getLogger(__name__)

@contextmanager
def profile_block(name, logger=None):
    t0 = time.perf_counter()

    yield

    elapsed = time.perf_counter() - t0

    if logger is not None:
        logger.debug(
            f"{name}: {elapsed:.6f} s"
        )

def benchmark(func, *args, repeats=100):

    t00 = time.perf_counter()
    func(*args)  # warm-up (important for JIT-compiled code using e.g. Numba)
    init = time.perf_counter() - t00

    t0 = time.perf_counter()

    for _ in range(repeats):
        func(*args)

    elapsed = time.perf_counter() - t0

    return elapsed / repeats, init
