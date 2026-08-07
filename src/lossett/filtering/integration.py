import xarray as xr

def integrate_over_scales(
    field,
    weight,
    ratio_rmax_to_ell=None,
    scale_dim="length_scale",
    radial_dim="r",
):
    """
    Compute

        I(ell) = ∫ K(r, ell) f(r) dr

    for every length scale ell.

    Parameters
    ----------
    field : xr.DataArray
        Quantity to integrate.
        Must contain radial_dim.

    weight : xr.DataArray
        Integration weight.
        Must contain dimensions
        (scale_dim, radial_dim).

    ratio_rmax_to_ell : float, optional
        Truncate integration at

            r <= ratio_rmax_to_ell * ell

    Returns
    -------
    xr.DataArray
    """
    
    scales = weight[scale_dim]

    integrals = []

    for iscale, scale in enumerate(scales.values):

        w = weight.isel(
            {scale_dim: iscale}
        )

        if ratio_rmax_to_ell is not None:

            rmax = ratio_rmax_to_ell * scale

            w = w.sel(
                {radial_dim: slice(0, rmax)}
            )

            f = field.sel(
                {radial_dim: slice(0, rmax)}
            )

        else:

            f = field

        integrals.append(
            (w * f).integrate(radial_dim)
        )

    return xr.concat(
        integrals,
        dim=scales,
    )
