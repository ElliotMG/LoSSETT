import numpy as np

# Load the field and pad it with symmetric conditions.
n1, n2, n3, nt = u.shape

# Pad Vy and Vz symmetrically everywhere
u_init = np.pad(u, ((Nlmax, Nlmax), (Nlmax, Nlmax), (0, 0), (0, 0)), mode='reflect')
v_init = np.pad(v, ((Nlmax, Nlmax), (Nlmax, Nlmax), (0, 0), (0, 0)), mode='reflect')
w_init = np.pad(w, ((Nlmax, Nlmax), (Nlmax, Nlmax), (0, 0), (0, 0)), mode='reflect')

DeltaUcubemoy = np.full((Nlmax, nt), np.nan)

# 1. Calculate the increments on these base vectors
# Check on an example; works if R=R[:,1] (column) and Z=Z[1,:] (row)

for ic in range(Nlmax):
    
    duDRt = np.full((n1, n2, n3, ntm, nt), np.nan, dtype=np.float32)
    
    for im in range(nphiinc[ic]):
        
        nlx = llx[ic, im]
        nly = lly[ic, im]
        
        # circshift(A,K) circularly shifts elements in array A by K
        # positions. If K is a vector, each element indicates the shift
        # amount in the corresponding dimension of A. np.roll should do the same thing in python
        du_l = np.roll(u_init, shift=[-nlx, -nly, 0, 0]) - u_init
        dv_l = np.roll(v_init, shift=[-nlx, -nly, 0, 0]) - v_init
        dw_l = np.roll(w_init, shift=[-nlx, -nly, 0, 0]) - w_init
        
        dusquare = du_l**2 + dv_l**2 + dw_l**2
        
        # Below is calculating component of du_l_3D along radial vector
        du_l_3D = (du_l * nlx * dR + dv_l * nly * dR + dw_l) / np.sqrt((nlx * dR)**2 + (nly * dR)**2)
        
        duDRt[:n1, :n2, :n3, im, :nt] = du_l_3D[Nlmax:Nlmax+n1, Nlmax:Nlmax+n2, :n3, :nt] * dusquare[Nlmax:Nlmax+n1, Nlmax:Nlmax+n2, :n3, :nt]
    
    # Calculate the angular average
    duDRt = np.nanmean(duDRt, axis=3)
    print('Average done')
    SulocDR[:, :, :, ic, :] = duDRt
    
    # Computation of the average DR
    DeltaUcubemoy[ic, :] = np.mean(duDRt, axis=(0, 1, 2))

# 1) Calcul de Duchon Robert [DYN] (calculation of DR DYN)

n1, n2, n3, n4, nt = SulocDR.shape

spsol = np.reshape(SulocDR / 4, (-1, n4))
spsol[np.isnan(spsol)] = 0

# DRdir2dt = spsol * philsmooth
# DRdir = np.reshape(DRdir2dt, (n1, n2, n3, Nls, nt))
# lDRdir = lsingd