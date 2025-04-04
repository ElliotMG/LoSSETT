import sys
import numpy as np

def CalcPartitionIncrement(dR,Nlmax,verbose=False):
    """
    Calculate the partition of increments
    Calculate the smoothing function one time for all
    Normalisation
    """
    # --------- Define Global Variables ----------- #
    dims = 2  # dimension of integration
    face = 1.27
    normphieps = 1

    # Calculate the length of the increments one time for all
    # Define average angle scale
    dlbb = np.ones(Nlmax)
    dR = np.array(dR)
    # dxs = dR
    # print('dlbb: ',dlbb)
    # print('dR: ', dR)
    dlbd = dlbb*dR
    dld = dlbd
    
    # define scales $\ell$

    ld = np.array([dR+(1+ic)*dlbd[ic] for ic in range(Nlmax)])
    if verbose:
        print("ld: ", ld)
    
    # Calculate the increment lengths one time for all
    # for nlx
    ibase = 0
    nmaxl = (2*Nlmax+1)**2
    llxt = np.zeros(nmaxl)
    llyt = np.zeros(nmaxl)

    for nlx in range(-Nlmax,Nlmax):
        for nly in range(-Nlmax,Nlmax):
            
            ibase=ibase+1;
            llxt[ibase] = np.single(nlx)
            llyt[ibase] = np.single(nly)

    llmov = np.zeros_like(ld)
    nphiinc = np.zeros_like(ld)
    lur = np.sqrt((np.array(llxt)*dR)**2 + (np.array(llyt)*dR)**2)

    llx = np.zeros((Nlmax,nmaxl))
    lly = np.zeros((Nlmax,nmaxl))
    if verbose:
        print("dld: ", dld)
        print("dlbd: ", dlbd)

    for ic in range(Nlmax):
        dll = dld[ic]
        llmov[ic] = ld[ic]
        nmov = np.where((lur <= ld[ic]+dll) & (lur > ld[ic]-dll))[0]
        nphiinc[ic] = len(nmov)

        for im in range(len(nmov)):
            llx[ic, im] = llxt[nmov[im]]
            lly[ic, im] = llyt[nmov[im]]

    # Useful for the rest
    ntm = np.max(nphiinc)

    # Calculate the smoothing function
    deps = dR
    epsl = np.arange(dR, Nlmax*dR+deps, deps)
    Nls = len(epsl)

    llmov = ld
    dllmov = dR

    lsingd = epsl

    # Calculate phismooth and psieps
    philsmooth = np.zeros((Nlmax, Nls))
    phils = np.zeros((Nlmax, Nls))

    for iic in range(Nlmax):
        for ic in range(Nls):
            epsr = epsl[ic]
            llt = ld[iic]

            dpsieps = 2 * llt * np.exp(-1/(1-(llt**2/face/epsr**2))) / normphieps / (epsr**dims) / (face*epsr**2) / (1-(llt**2/face/epsr**2))**2

            psieps = np.exp(-1/(1-(llt**2/face/epsr**2))) / normphieps / (epsr**dims)

            # if llt >= np.sqrt(face) * epsr:
            #     dpsieps = 0
            #     psieps = 0

            philsmooth[iic, ic] = dpsieps * llt**(dims-1) * dld[iic]
            phils[iic, ic] = psieps * dld[iic]
        
    return dR, Nlmax, nphiinc, llx, lly, philsmooth, Nls

if __name__ == "__main__":
    dR = np.double(sys.argv[1])
    Nlmax = int(sys.argv[2])
    print("dR: ", dR, dR.type())
    print("Nlmax: ", Nlmax, Nlmax.type())
    CalcPartitionIncrement(dR,Nlmax)
