import numpy as np
import matplotlib.pyplot as plt

def standard_mollifier(x,length_scale=1,normalization=1):
    return np.where(
        x<2*length_scale,
        normalization * np.exp(  -1 / (1 - (x / (2*length_scale))**2) ),
        0
    );

def derivative_of_standard_mollifier(x, length_scale=1,normalization=1):
    return np.where(
        x<2*length_scale,
        - normalization * (x / (2 * length_scale**2)) * \
        np.exp(  -1 / (1 - (x / (2*length_scale))**2) ) / (1 - (x / (2*length_scale))**2)**2,
        0
    );

def filter_kernel(length_scale,r,npts=101,return_derivative=True):
    """
    Compute standard mollifier with length scale \ell, defined by:
        G_\ell(r) = N_\ell \exp( -1 / (1 - (r / 2\ell)^2) )
    with N_\ell determined by the constraint \int_0^\infty dr G_\ell(r) = 1.

    Optionally compute also its derivative:
                                     r                 1                /        - 1          \
        dG_\ell / dr = - N_\ell . -------- . --------------------- . exp| ------------------- |
                                  2 \ell^2   [1 - (r / 2\ell)^2]^2      \ [1 - (r / 2\ell)^2] /
    Required inputs:
      - length_scale: \ell
      - r: sampling points in r-space

    Optional inputs:
      - npts: number of points to compute normalization factor
      - return_derivative: boolean, switch to determine whether to also compute and return derivative.
    """
    
    return 0;

if __name__ == "__main__":
    x = np.arange(0,100,0.01)
    print(x)
    colours = ["k","C0","C1","C2","C3","C4","C5"]
    fig,axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10,5),
        #sharex=True
    )
    length_scales = [1,2.5,5,10,20,40,50]
    sin_scale = 2*100/np.pi
    norms = []
    sin_norms = []
    for il, length_scale in enumerate(length_scales):
        print(f"\n\nLength scale = {length_scale}")
        _kernel = standard_mollifier(x, length_scale=length_scale)
        _integral = 2*np.pi*np.trapz(x*_kernel,x=x)
        _sin_integral = 2*np.pi*np.trapz(sin_scale*np.sin(x/sin_scale)*_kernel,x=x)
        print(f"Integral = {_integral:.4g}")
        print(f"Sine integral = {_sin_integral:.4g}")
        normalization = 1 / _integral
        sin_normalization = 1 / _sin_integral
        norms.append(normalization)
        sin_norms.append(sin_normalization)
        kernel = standard_mollifier(
            x,
            length_scale=length_scale,
            normalization=normalization
        )
        sin_kernel = standard_mollifier(
            x,
            length_scale=length_scale,
            normalization=sin_normalization
        )
        integral = np.trapz(x*kernel,x=x)
        sin_integral = np.trapz(np.sin(x/sin_scale)*kernel,x=x)
        print(f"Integral = {integral:.4g}")
        print(f"Sine integral = {sin_integral:.4g}")
        ax=axes[0]
        ax.plot(x, _kernel, linestyle="--", color=colours[il], label="raw")
        ax.plot(x, kernel, color=colours[il], label="normalized")
        ax.axvline(2*length_scale, color=colours[il], linestyle=":", label="$2\ell$")
        if il == 0:
            axes[0].legend(loc="upper right")

    _best_fit = np.polynomial.Polynomial.fit(np.log(length_scales),np.log(norms),1,full=True)
    print(_best_fit)
    best_fit = _best_fit[0].convert().coef
    norm_zero = norms[0]
    print(best_fit)
    axes[1].loglog(length_scales, norms, "+--", label="$\mathcal{N}(\ell)$")
    axes[1].loglog(
        length_scales,
        np.exp(best_fit[0])*(np.array(length_scales)**best_fit[1]),
        linestyle="--",
        label=r"$"+f"{np.exp(best_fit[0]):.4g}"+r"\times \ell^{"+f"{best_fit[1]:.4g}"+r"}$ (best fit)",
    )
    axes[1].loglog(
        length_scales,
        norm_zero / (np.array(length_scales)**2.0),
        linestyle=":",
        label=r"$"+f"{norm_zero:.4g}"+r"\times \ell^{-2} $ (anal.)",
    )
    axes[1].loglog(
        length_scales,
        sin_norms,
        "+--",
        color="k",
        label="$\mathcal{N}(\ell)$ (sine)"
    )
    # tidying
    axes[0].set_ylabel("$G_\ell(r)$ [2D]")
    axes[0].set_xlabel("$r$")
    axes[1].legend(loc="upper right")
    axes[1].set_ylabel("Normalization")
    axes[1].set_xlabel("$\ell$")
    plt.savefig("standard_mollifier_normalization.png")
    plt.show()

    fig,axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10,5),
        sharex=True
    )
    ax = axes[0]
    ax.plot(
        x,
        standard_mollifier(x, length_scale=1, normalization=norm_zero),
        label=r"$G_\ell (r) = \ell^2 \times \mathcal{N}_0\ \mathrm{exp}\left(\frac{-1}{1-(r/2\ell)^2}\right)$"
    )
    ax.plot(
        x,
        x*standard_mollifier(x, length_scale=1, normalization=norm_zero),
        label=r"$(r / \ell) \times G_\ell (r)$"
    )
    ax.axhline(
        norm_zero/np.exp(1),
        color="k",
        linestyle=":",
        label=r"$\ell^2 \times \mathcal{N}_\ell / e = \mathcal{N}_0 / e$"
    )
    ax.set_xlim([0,4])
    ax.set_xlabel("$r / \ell$")
    ax.legend(loc="upper right")
    ax.grid()
    ax = axes[1]
    ax.plot(
        x,
        np.gradient(standard_mollifier(x, length_scale=1, normalization=norm_zero),x),
        label=r"$d G_\ell (r) / dr$ (numerical)",
        color="C0"
    )
    ax.plot(
        x,
        derivative_of_standard_mollifier(x, length_scale=1, normalization=norm_zero),
        label=r"$d G_\ell (r) / dr$ (analytical)",
        linestyle=":",
        color="C3"
    )
    ax.plot(
        x,
        x*derivative_of_standard_mollifier(x, length_scale=1, normalization=norm_zero),
        label=r"$(r / \ell) \times d G_\ell (r) / dr$ (analytical)",
        linestyle="-",
        color="C1"
    )
    ax.set_xlim([0,4])
    ax.set_xlabel("$r / \ell$")
    ax.legend(loc="upper right")
    ax.grid()
    plt.savefig("standard_mollifier_and_derivative_2D_integral_weights.png")
    plt.show()
