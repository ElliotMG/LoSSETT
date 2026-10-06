import numpy as np
import pytest

from lossett.filtering.kernels import filter_kernel


@pytest.mark.parametrize("normalization", ["2D", "sphere", "3D"])
def test_filter_kernel_normalizes_supported_geometries(normalization):
    r = np.linspace(0.0, 2.5, 500)
    sphere_radius = 10.0

    kernel = filter_kernel(
        length_scale=1.0,
        r=r,
        return_derivative=False,
        normalization=normalization,
        sphere_radius=sphere_radius,
    )

    if normalization == "2D":
        integrand = 2 * np.pi * r * kernel
    elif normalization == "sphere":
        integrand = (
            2
            * np.pi
            * sphere_radius
            * np.sin(r / sphere_radius)
            * kernel
        )
    else:
        integrand = 4 * np.pi * r**2 * kernel

    np.testing.assert_allclose(np.trapz(integrand, x=r), 1.0, rtol=1e-5)
