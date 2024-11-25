import sys
import numpy as np
import xarray as xr

def CalcDR_Scalar(dR, Nlmax, u, v, w, scalar, nphiinc, llx, lly, philsmooth, Nls, fname_str, verbose=False):

    # Dimensions
    nt = len(u.time)
    nz = len(u.level)
    ny = len(u.latitude)
    nx = len(u.longitude)

    SulocDR_s = xr.full_like(u, np.nan)
    SulocDR_s = SulocDR_s.expand_dims(dim={"n_scales":range(Nlmax)}, axis=0).copy()

    
    for ic in range(Nlmax):
        
        ntm = int(nphiinc[ic])
        if verbose:
            print('ntm: ',ntm)
        
        duDRt_s = xr.full_like(u, np.nan)
        duDRt_s = duDRt_s.expand_dims(dim={"angle_incs":range(ntm)}, axis=0).copy()
        
        for im in range(ntm):
            
            nlx = llx[ic, im]
            nly = lly[ic, im]
            # print("nlx: ", nlx)
            nlx=int(nlx)
            nly=int(nly)
            # print("nlx: ",nlx)
            du_l = u.roll(longitude=-nlx, latitude=-nly, roll_coords=False) - u
            dv_l = v.roll(longitude=-nlx, latitude=-nly, roll_coords=False) - v
            dw_l = w.roll(longitude=-nlx, latitude=-nly, roll_coords=False) - w
            dscalar_l = scalar.roll(longitude=-nlx, latitude=-nly, roll_coords=False) - scalar
            
            # Below is calculating component of du_l_3D along radial vector
            du_l_3D = (du_l * nlx * dR + dv_l * nly * dR + dw_l) / np.sqrt((nlx * dR)**2 + (nly * dR)**2)
            
            duDRt_s[im,:, :, :, :] = du_l_3D * dscalar_l**2

        # Calculate the angular average
        duDRt_s = duDRt_s.mean(dim='angle_incs')

        print(f'Average {ic} done')
        SulocDR_s[ic, :, :, :, :] = duDRt_s
        
    # 1) Calcul de Duchon Robert [THERMO] (calculation of DR THERMO)

    SulocDR_s_np = SulocDR_s.values  # Extract the NumPy array
    n1, n2, n3, n4, nt = SulocDR_s_np.shape
    print('SuloDR_np.shape:',SulocDR_s_np.shape)
    
    print("Applying Phil")
    
    print("philsmooth.shape:",philsmooth.shape)
    
    DRdir2dt_s = np.tensordot(SulocDR_s_np,philsmooth,axes=(0,0))
    print('DRdir2dt_orig:',DRdir2dt_s.shape)
    DRdir2dt_s = np.transpose(DRdir2dt_s,[4,3,2,1,0])
    print('DRdir2dt reshaped by np.transpose:',DRdir2dt_s.shape)
    
   
    DRdir2dt_s = xr.DataArray(DRdir2dt_s, dims=['n_scales', 'longitude', 'latitude', 'level', 'time'],
                         coords={'latitude': u.latitude, 'longitude': u.longitude, 'level': u.level,\
                             'n_scales': range(Nlmax), 'time': u.time}, name=f'LoSSET_DR_{scalar.name}')
    # lDRdir = lsingd
    DRdir2dt_s.to_netcdf(f'/home/users/emg97/emgScripts/LoSSETT/out_nc/DRdir2dt_{scalar.name}_Nlmax{Nlmax}_{fname_str}.nc')
    print(f'DRdir2dt_s written to NetCDF file')
    return DRdir2dt_s
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