#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

def plot_scale_increments(scale_incs, Rsel=[10,20,40], show=False):
    distance = ds_mask.distance
    angle = ds_mask.angle
    mask = scale_incs.mask
    r_x = ds_mask.r_x
    r_y = ds_mask.r_y
    
    # plot
    fig,axes = plt.subplots(
        nrows=2, ncols=3, sharex=True, sharey=True,
        figsize=(18,10)
    )
    vmax_r = distance.max()
    ticks_phi = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    ticklabels_phi = ["$-\pi$", "$-\pi / 2$", "0", "$\pi / 2$", "$\pi$"]
    # plot r
    ax=axes[0,0]
    ax.set_title("$r$")
    pc_r = ax.pcolormesh(r_x,r_y,distance.T,cmap="viridis",vmin=0,vmax=vmax_r)
    plt.colorbar(pc_r)
    # plot phi
    ax=axes[0,1]
    ax.set_title("$\phi$")
    pc_phi = ax.pcolormesh(r_x,r_y,angle.T,cmap="twilight_shifted")
    cb_phi = plt.colorbar(pc_phi, ticks=ticks_phi)
    cb_phi.ax.set_yticklabels(ticklabels_phi)
    # plot mask
    ax=axes[0,2]
    ax.set_title("mask summed over $r$")
    pc_mask = ax.pcolormesh(r_x,r_y,mask.sum("r").T,cmap="viridis")
    plt.colorbar(pc_mask)

    # plot r, phi for various R
    for ir, R in enumerate(Rsel):
        R_circ = mask.sel(r=R,method="nearest").r
        # plot r (select r = R)
        ax=axes[1,0]
        pc_r = ax.pcolormesh(
            r_x,r_y,distance.where(mask.sel(r=R, method="nearest") == 1).T,
            cmap="viridis",vmin=0,vmax=vmax_r
        )
        circle=plt.Circle((r_0.r_x,r_0.r_y), R_circ, color='w', fill=False)
        ax.add_patch(circle)
        if ir == 0:
            ax.set_title(f"$r$ for $r \in [{Rsel[0]:.2g},{Rsel[1]:.2g},{Rsel[2]:.2g}]$")
            cb_r = plt.colorbar(pc_r)
        cb_r.ax.axhline(R_circ,color="w")
        # plot phi (select r = R)
        ax=axes[1,1]
        pc_phi = ax.pcolormesh(
            r_x,r_y,angle.where(mask.sel(r=R, method="nearest") == 1).T,
            cmap="twilight_shifted"
        )
        circle=plt.Circle((r_0.r_x,r_0.r_y), R_circ, color='w', fill=False)
        ax.add_patch(circle)
        if ir == 0:
            ax.set_title(f"$\phi$ for $r \in [{Rsel[0]:.2g},{Rsel[1]:.2g},{Rsel[2]:.2g}]$")
            cb_phi = plt.colorbar(pc_phi, ticks=ticks_phi)
            cb_phi.ax.set_yticklabels(ticklabels_phi)
    # aspect ratio
    for row in axes:
        for ax in row:
            ax.set_aspect("equal")
    # save and show
    plt.savefig(
        f"scale_increments_mask_Lx_{L_x:.3g}_Ly_{L_y:.3g}_Rmax_{max_r:.3g}_dx_{delta_x:.3g}_dy_{delta_y:.3g}.png"
    )
    if show:
        plt.show()
    plt.close()
    return 0;
