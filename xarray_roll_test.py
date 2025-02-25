import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import sys

x = np.arange(0,2*np.pi,0.01)
u = xr.DataArray(np.sin(x),coords={"x":x},dims="x")
pi_over_2_shift = int(np.floor(np.pi/2/0.01))

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    sharex=True,
    sharey=True
)
for i, roll_coords in enumerate([False, True]):
    ax=axes[i]
    ax.plot(u.x,u,label="unshifted")
    u_rolled_plus = u.roll(x=+pi_over_2_shift, roll_coords=roll_coords)
    u_rolled_minus = u.roll(x=-pi_over_2_shift, roll_coords=roll_coords)
    ax.plot(
        u_rolled_plus.x,
        u_rolled_plus,
        label="roll +"
    )
    ax.plot(
        u_rolled_minus.x,
        u_rolled_minus,
        label="roll -"
    )
    ax.legend(loc="best")
    ax.grid()
    ax.set_title(f"roll_coords={str(roll_coords)}")
plt.show()
