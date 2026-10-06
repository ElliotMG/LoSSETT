import numpy as np
import pytest
import xarray as xr

import lossett.calc.calc_inter_scale_transfers as transfers


@pytest.mark.parametrize(
    ("kernel_gradient", "expected"),
    [
        (True, [6.0, 32.0]),
        (False, [2.0, 16.0]),
    ],
)
def test_calc_scale_space_integral_reads_kernels_from_dataset(
        monkeypatch,
        kernel_gradient,
        expected
):
    r = np.arange(5.0)
    length_scales = np.array([1.0, 2.0])
    integrand = xr.DataArray(
        np.ones_like(r),
        coords={"r": r},
        dims="r",
    )
    integrand.r.attrs["units"] = "m"

    def fake_get_integration_kernels(*args, **kwargs):
        return xr.Dataset(
            {
                "G": (
                    ("length_scale", "r"),
                    np.array([np.ones_like(r), 2 * np.ones_like(r)]),
                ),
                "dG_dr": (
                    ("length_scale", "r"),
                    np.array([3 * np.ones_like(r), 4 * np.ones_like(r)]),
                ),
            },
            coords={"length_scale": length_scales, "r": r},
        )

    monkeypatch.setattr(
        transfers,
        "get_integration_kernels",
        fake_get_integration_kernels,
    )

    result = transfers.calc_scale_space_integral(
        integrand,
        name="transfer",
        length_scales=length_scales,
        kernel_gradient=kernel_gradient,
    )

    np.testing.assert_allclose(result.values, expected)
    np.testing.assert_array_equal(result.length_scale, length_scales)
    assert result.length_scale.attrs["units"] == "m"
