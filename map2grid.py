import numpy as np

#Function creates an numpy array of values based on lattice coordinates
def map2grid(inab, inVal, *args):
    # Inputs
    #   inab (ndarray): grid coordinates
    #   inVal (ndarray): value per coordinate
    #   args   :
    #       alim :   a limits [amin, amax] (will crop if data extends beyond)
    #       blim :   b limits [bmin, bmax] (will crop if data extends beyond)

    # Outputs
    #   Valgrid: Value mapped to grid
    #   abrng:   Boundaries of grid [amin, amax, bmin, bmax]

    default_val = np.nan
    a = inab[:,0].astype('int')
    b = inab[:,1].astype('int')

    if len(args)==2:
        alim, blim = args
        al = alim[0]
        ah = alim[1]
        bl = blim[0]
        bh = blim[1]
    else:
        al = np.nanmin(a)
        ah = np.nanmax(a)
        bl = np.nanmin(b)
        bh = np.nanmax(b)
    abrng = np.array([al, ah, bl, bh])

    #crop
    ind = np.where((np.isfinite(a)) & (a>=al) & (a<=ah) & (b>=bl) & (b<=bh))[0]
    abind = inab[ind,:]
    abind[:,0] -= al
    abind[:,1] -= bl

    #assign to grid
    if inVal.ndim>1:
        n = inVal.shape[1]
        Valgrid = np.ones((bh-bl+1, ah-al+1,n))*default_val
        for i in range(n):
            Valgrid[abind[:,1].astype(int),abind[:,0].astype(int),i]=inVal[ind,i]
    else:
        Valgrid = np.ones((bh-bl+1, ah-al+1))*default_val
        Valgrid[abind[:,1].astype(int),abind[:,0].astype(int)]=inVal[ind]

    return Valgrid, abrng