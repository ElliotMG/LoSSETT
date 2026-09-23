#!/usr/bin/env python3
from contextlib import contextmanager
import time
import logging

logger = logging.getLogger(__name__)

class Profiler:
    """
    Lightweight profiling accumulator.

    Example
    -------
    profiler = Profiler()

    t0 = time.perf_counter()
    slow_function()
    profiler.add(
        "slow_function",
        time.perf_counter() - t0,
    )

    profiler.report(logger)
    """

    def __bool__(self):
        return True

    def __init__(self):
        self.times = {}

    def add(self, name, dt):
        self.times[name] = (
            self.times.get(name, 0.0) + dt
        )

    def reset(self):
        self.times.clear()

    def report(
        self,
        logger=None,
        total_key=None,
        title="PROFILING SUMMARY",
    ):
        if not self.times:
            return

        if total_key is not None:
            total = self.times.get(total_key, 0.0)
        else:
            total = sum(self.times.values())

        lines = [
            "",
            "###############################",
            f"### {title} #########",
            "###############################",
        ]

        for key, value in sorted(
            self.times.items(),
            key=lambda x: x[1],
            reverse=True,
        ):

            if total > 0:
                frac = 100.0 * value / total
            else:
                frac = 0.0

            lines.append(
                f"{key:32s}"
                f"{value:10.3f} s  "
                f"({frac:5.1f}%)"
            )

        text = "\n".join(lines)

        if logger is None:
            print(text)
        else:
            logger.info(text)

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
