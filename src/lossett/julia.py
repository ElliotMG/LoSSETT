"""Optional Python/Jupyter bridge to LoSSETT's Julia kinetic-energy core."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import NamedTuple

import numpy as np


class JuliaTransferResult(NamedTuple):
    """Python-friendly result returned by :func:`kinetic_energy_transfer`."""

    transfer: np.ndarray
    length_scales: np.ndarray
    radii: np.ndarray
    dimension_order: tuple[str, ...]


_JULIA_MODULE = None
_JULIA_LOCK = Lock()
_JULIA_SOURCE = Path(__file__).parent / "_julia" / "LoSSETT.jl"


def _load_julia_module():
    global _JULIA_MODULE

    if _JULIA_MODULE is not None:
        return _JULIA_MODULE

    with _JULIA_LOCK:
        if _JULIA_MODULE is not None:
            return _JULIA_MODULE

        try:
            from juliacall import Main as jl
        except ImportError as error:
            raise ImportError(
                "The LoSSETT Julia bridge requires the optional 'juliacall' "
                "dependency. Install with `pip install 'lossett[julia]'`."
            ) from error

        if not _JULIA_SOURCE.is_file():
            raise FileNotFoundError(
                f"Packaged LoSSETT Julia source is missing: {_JULIA_SOURCE}"
            )

        jl.seval("include")(str(_JULIA_SOURCE))
        _JULIA_MODULE = jl.LoSSETT

    return _JULIA_MODULE


def _float_array(name, value, ndim):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional; got {array.ndim}")
    return array


def kinetic_energy_transfer(
    u,
    v,
    w,
    x,
    y,
    length_scales,
    *,
    max_radius,
    periodic=(True, False),
    xdim=None,
    ydim=None,
    geometry="cartesian",
    sphere_radius=6_371_000.0,
    use_angular_weights=False,
):
    """Compute kinetic-energy transfer using the Julia core from Python.

    Pass same-shaped real-valued arrays for `u`, `v`, and `w`; one-dimensional
    coordinate vectors `x` and `y`; and length scales. The default
    `geometry="cartesian"` requires coordinates, scales, and `max_radius` in
    common linear units. With `geometry="spherical"`, `x` and `y` are
    longitude and latitude in degrees; scales, `max_radius`, and
    `sphere_radius` are in metres (or another common linear unit). For
    `(time, pressure, latitude, longitude)` data, pass `xdim=4, ydim=3`.
    `use_angular_weights=True` uses bearing-sector Voronoi weights instead of
    the default uniform sample average. Returns a :class:`JuliaTransferResult`
    with NumPy arrays for transfer, clipped scales, and sampled radii.

    The first call starts Julia and compiles the core; warm it up before
    timing subsequent calls if measuring steady-state compute performance.

    Spherical geometry integrates with `R*sin(r/R) dr` and uses that same
    spherical area element to normalize the mollifier. This corrects the
    current Python `spherical_geometry` transfer routine, which normalizes
    spherically but still uses `r dr` in its transfer integral.
    """
    if geometry not in ("cartesian", "spherical"):
        raise ValueError("geometry must be 'cartesian' or 'spherical'")
    u_array = _float_array("u", u, ndim=np.ndim(u))
    if u_array.ndim < 2:
        raise ValueError("u, v, and w must have at least two dimensions")
    v_array = _float_array("v", v, ndim=u_array.ndim)
    w_array = _float_array("w", w, ndim=u_array.ndim)
    if u_array.shape != v_array.shape or u_array.shape != w_array.shape:
        raise ValueError("u, v, and w must have identical shapes")

    x_array = _float_array("x", x, ndim=1)
    y_array = _float_array("y", y, ndim=1)
    scale_array = _float_array("length_scales", length_scales, ndim=1)
    if len(periodic) != 2:
        raise ValueError("periodic must be a pair (x_periodic, y_periodic)")

    if xdim is None:
        xdim = u_array.ndim
    if ydim is None:
        ydim = u_array.ndim - 1

    result = _load_julia_module()._kinetic_energy_transfer_python(
        u_array,
        v_array,
        w_array,
        x_array,
        y_array,
        scale_array,
        max_radius=float(max_radius),
        periodic=(bool(periodic[0]), bool(periodic[1])),
        xdim=int(xdim),
        ydim=int(ydim),
        geometry=geometry,
        sphere_radius=float(sphere_radius),
        use_angular_weights=bool(use_angular_weights),
    )
    return JuliaTransferResult(
        transfer=np.array(result[0], dtype=np.float64, copy=True),
        length_scales=np.array(result[1], dtype=np.float64, copy=True),
        radii=np.array(result[2], dtype=np.float64, copy=True),
        dimension_order=tuple(result[3]),
    )
