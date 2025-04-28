import sys
import numpy as np
import xarray as xr
import os
import psutil

from test_kernels import get_integration_kernels

radius_earth = 6.371e6 # radius of Earth in m

def CalcDRDir_2D(dR, Nlmax, u, v, w, nphiinc, llx, lly, philsmooth, Nls, fname_str, diro, verbose=False, new_kernel=True):
    # Load the fields and pad them with symmetric conditions.
    # Note 29.7.24: commented out the padding becuase should be handled in np.roll
    # u_init = np.pad(u, ((Nlmax, Nlmax), (Nlmax, Nlmax), (0, 0), (0, 0)), mode='reflect')
    # v_init = np.pad(v, ((Nlmax, Nlmax), (Nlmax, Nlmax), (0, 0), (0, 0)), mode='reflect')
    # w_init = np.pad(w, ((Nlmax, Nlmax), (Nlmax, Nlmax), (0, 0), (0, 0)), mode='reflect')
    # Dimensions
    nt = len(u.time)
    nz = len(u.level)
    ny = len(u.latitude)
    nx = len(u.longitude)
    # DeltaUcubemoy = np.full((Nlmax, nt), np.nan)
    # ^EMG 29.7.24 removed because script was getting stuck later on this even though
    # we never refer to it. See inside loop for comment where error is thrown up.

    SulocDR = xr.full_like(u, np.nan)
    SulocDR = SulocDR.expand_dims(dim={"n_scales":range(Nlmax)}, axis=0).copy()

    pid = os.getpid()
    python_process = psutil.Process(pid)
    
    for ic in range(Nlmax):
        
        ntm = int(nphiinc[ic])
        if verbose:
            print('ntm: ',ntm)
        
        duDRt = xr.full_like(u, np.nan)
        duDRt = duDRt.expand_dims(dim={"angle_incs":range(ntm)}, axis=0).copy()
        
        for im in range(ntm):
            
            nlx = llx[ic, im]
            nly = lly[ic, im]
            nlx=int(nlx)
            nly=int(nly)
            du_l = u.roll(longitude=-nlx, latitude=-nly, roll_coords=False) - u
            dv_l = v.roll(longitude=-nlx, latitude=-nly, roll_coords=False) - v
            dw_l = w.roll(longitude=-nlx, latitude=-nly, roll_coords=False) - w
            dusquare = du_l**2 + dv_l**2 + dw_l**2

            # Below is calculating component of du_l_3D along radial vector
            du_l_3D = (du_l * nlx * dR + dv_l * nly * dR + dw_l) / np.sqrt((nlx * dR)**2 + (nly * dR)**2)
            # I think the above line is wrong in 2D. w cannot have a component along the horizontal surface
            # by definition, so it should instead be:
            # du_l_2D = (du_l * nlx * dR + dv_l * nly * dR) / np.sqrt((nlx * dR)**2 + (nly * dR)**2)
            
            duDRt[im,:, :, :, :] = du_l_3D * dusquare

        # Calculate the angular average
        duDRt = 2*np.pi*duDRt.mean(dim='angle_incs') # note that this isn't weighted properly.
        # (i.e. we should *really* have \delta\theta angle increments associated with each angle)
        # Also the actual calculation is \int_0^2\pi duDRt d\theta \approx \sum_0^2\pi duDRt_i \delta\theta_i
        # \approx \delta\theta \sum_i duDRt_i if \delta_\theta_i can be approximated as constant.
        # We thus need to normalize the sum by 2 \pi (since \delta \theta \approx 2 \pi / number of increments)

        print(f'Average {ic} done')
        SulocDR[ic, :, :, :, :] = duDRt
        memory_use = python_process.memory_info()[0]/(10**9) # RAM usage in GB
        print(f"\nMemory usage at filter integration step: {memory_use:5g} GB")
        # Computation of the average DR
        # DeltaUcubemoy[ic, :] = np.mean(duDRt, axis=(0, 1, 2))
        # ^ EMG 29.7.24 14:42 - got stuck here but I don't think we ever use it
        # Error message: ValueError: could not broadcast input array from shape (480,) into shape (62,)

    # save integrand after angular average
    print("\nSaving integrand after angular average")
    SulocDR = SulocDR.rename("Duchon-Robert_integrand_after_angular_integral")
    print(SulocDR)
    SulocDR = SulocDR.chunk(chunks={"time":1})
    print(SulocDR)
    SulocDR.to_netcdf(os.path.join(diro,f'DR_integrand_Nlmax{Nlmax}_{fname_str}.nc'))

    # 1) Calcul de Duchon Robert [DYN] (calculation of DR DYN)
    if new_kernel:
        # Use new filter kernel code
        # check indexing in CalcPartitionIncrement -- may start at dR rather than 0
        r = np.arange(dR, (Nlmax+1)*dR, dR)
        length_scales = r[:len(r)//2] # should really take these as (required?) input
        print(r)
        print(length_scales)
        G, dG_dr = get_integration_kernels(
            r,
            length_scales,
            normalization="sphere",
            sphere_radius=radius_earth,
            return_deriv=True
        )
        print("\n\n",G)
        print("\n\n",dG_dr)
        integrand = SulocDR.rename({"n_scales":"r"}).assign_coords(r=r)
        print("\n\n",integrand)
        # integrate against kernel over r
        #DRdir2dt = (radius_earth*dG_dr*np.sin(dG_dr.r/radius_earth)*integrand).integrate("r")
        # flat earth integral
        DRdir2dt = (dG_dr*dG_dr.r*integrand).integrate("r").rename("LoSSET_DR")
        print("\n\n\nLoSSET_DR:\n", DRdir2dt)
        
        filo = os.path.join(diro,f'DRdir2dt_Nlmax{Nlmax}_{fname_str}_NEW_KERNEL.nc')
        DRdir2dt.to_netcdf(filo)
        print(f'NC File created: {filo}')
        memory_use = python_process.memory_info()[0]/(10**9) # RAM usage in GB
        print(f"\nMemory usage at end of loop over scales: {memory_use:5g} GB")
        
    else:
        # use old code
        SulocDR_np = SulocDR.values  # Extract the NumPy array [DS 2025-02-24: why?]
        n1, n2, n3, n4, nt = SulocDR_np.shape
        print('SuloDR_np.shape:',SulocDR_np.shape)
        # sys.exit()
        # spsol_np = np.reshape(SulocDR_np / 4, (-1, n4))
    
        # spsol_np[np.isnan(spsol_np)] = 0
        # spsol = xr.DataArray(spsol_np)
        # print("Shape of spsol_np:", spsol.shape)
        # print("Shape of philsmooth:", philsmooth.shape)
    
        # Adjustment to make for broadcasting arrays of different sizes
        # MATLAB does this implicitly hence why it worked in only one line in the Matlab code.
        # philsmooth_reshaped = philsmooth.reshape(Nlmax, Nls)
        print("Applying Phil")
        print("philsmooth.shape:",philsmooth.shape)
        # print("spsol.shape:",spsol.shape)
        # philsmooth_reshaped = np.broadcast_to(philsmooth,spsol.shape)
        # DRdir2dt = np.dot(spsol,philsmooth)
        # print('SulocDR_np:',SulocDR_np)
        DRdir2dt = np.tensordot(SulocDR_np,philsmooth,axes=(0,0))
        print('DRdir2dt_orig:',DRdir2dt.shape)
        DRdir2dt = np.transpose(DRdir2dt,[4,3,2,1,0])
        print('DRdir2dt reshaped by np.transpose:',DRdir2dt.shape)
    
        # DRdir2dt_np = DRdir2dt.values
        # DRdir2dt = xr.DataArray(DRdir2dt, dims=['new_dim', 'n4'])

    
        DRdir2dt = xr.DataArray(DRdir2dt, dims=['n_scales', 'longitude', 'latitude', 'level', 'time'],
                                coords={'latitude': u.latitude, 'longitude': u.longitude, 'level': u.level,\
                                        'n_scales': range(Nlmax), 'time': u.time}, name='LoSSET_DR')
        # lDRdir = lsingd

        filo = os.path.join(diro,f'DRdir2dt_Nlmax{Nlmax}_{fname_str}.nc')
        DRdir2dt.to_netcdf(filo)
        print(f'NC File created: {filo}')
        memory_use = python_process.memory_info()[0]/(10**9) # RAM usage in GB
        print(f"\nMemory usage at end of loop over scales: {memory_use:5g} GB")
        
        return DRdir2dt;
    
    # DRdir.max
# if __name__ == "__main__":
#     dR      = np.double(sys.argv[1])
#     Nlmax   = int(sys.argv[2])
#     u       = np.double(sys.argv[3])
#     v       = np.double(sys.argv[4])
#     w       = np.double(sys.argv[5])
#     # print("dR: ", dR, dR.type())
#     # print("Nlmax: ", Nlmax, Nlmax.type())
#     CalcDRDir_2D(dR,Nlmax,u,v,w)
