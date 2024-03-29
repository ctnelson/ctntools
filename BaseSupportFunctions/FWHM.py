import numpy as np

def FWHM(inarray, ind0 = None, cutoff=0.5, normalize=True):
    #FWHM measures the Full Width at Half Maximimum of a peak in 'inarray' which should be an [n,] vector or [n,m] array (if 2D, along axis 0). Determined as the first crossing of the halfway point of the normalized array (so recommend smoothing noisy data)
    ### Inputs ###
    #inarray            :       [n,] or [n,m] vector or array to find FWHM
    #ind0   (optional)  :       this is the central point from which the function will search for 50% crossings. For 2D, currently coded only for a single center index (so must be shared across axis 1)
    ### Outputs ###
    #fhwm               :       [m,] full width half max(es) 

    dim = np.ndim(inarray)
    if dim==1:
        inarray = inarray[:,np.newaxis]
    
    sz = np.array(inarray.shape,dtype='int')

    #normalize
    if normalize:
        iamin = np.repeat(np.nanmin(inarray,axis=0,keepdims=True),sz[0],axis=0)
        iamax = np.repeat(np.nanmin(inarray,axis=0,keepdims=True),sz[0],axis=0)
        inarray = (inarray-iamin) - (iamax-iamin)

    #center point
    if ind0 is None:
        ind0 = np.nanargmax(np.nanmean(inarray,axis=1))

    #positive branch
    fwhmp = np.nanargmax(inarray[ind0:,:]<cutoff,axis=0)
    dyp = inarray[tuple([ind0+fwhmp-1,np.arange(sz[1])])] - inarray[tuple([ind0+fwhmp,np.arange(sz[1])])]
    dxp = (inarray[tuple([ind0+fwhmp-1,np.arange(sz[1])])]-cutoff)/dyp * 1
    fwhmp_fr = fwhmp+dxp-1

    #negative branch
    fwhmn = np.nanargmax(np.flip(inarray[:(ind0+1),:],axis=0)<cutoff,axis=0)
    dyn = inarray[tuple([ind0-fwhmn,np.arange(sz[1])])] - inarray[tuple([ind0-fwhmn+1,np.arange(sz[1])])]
    dxn = (inarray[tuple([ind0-fwhmn+1,np.arange(sz[1])])]-cutoff)/dyn * 1
    fwhmn_fr = -fwhmn+dxn+1

    #total
    fwhm = fwhmp_fr - fwhmn_fr

    return fwhm
