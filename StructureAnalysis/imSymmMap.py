import numpy as np
from ctntools.SlidingWindow.swMtransf_radial_CC import swMtransf_IJradial_CC
from ctntools.BaseSupportFunctions.matTransf import makeMtransf   
#Maps given symmetries (defined by given [2,2] transform matrix) via a sliding window analysis

#wrapper function to running the radial sliding window symmetry calc swMtransf_IJradial_CC for a given (series of) transformation(s)
def imSymmMap(inim, iM, swRad, symmCalc='ZeroNormCrossCorr', inax=None, verbose=False, **kwargs):
    ### Inputs ###
    #inim           :   input image
    #iM             :   Dictionary of symmetries to apply. Example: {'i':0,'r':[90,120]} creates a [2,2,3] stack of inversion, 90deg rot and 120deg rot. More details in def makeMtransf() description
    #swRad          :   search window transform calculation radius
    #symmCalc       :   calculation type for sliding window symmetry 'ZeroNormCrossCorr', or 'MeanAbsDiff'   
    #inax           :   axis for plotting
    #verbose        :   flag to print execuation details       
    ### Outputs ###
    #swSymm         :   stack of the sliding window symmetry calculations
    #swCounts       :   stack of the count of valid datapoints used for M transform calculation
    #ds             :   [2,] downsizing scalars
    #Mlbls          :   str labels for the symmetries applied

    #get transform matrices
    M, Mlbls = makeMtransf(iM,verbose=verbose)
    n = M.shape[2]

    #Sliding window symmetry
    swRad = np.ceil(swRad).astype('int')
    swSymm = np.ones((inim.shape[0],inim.shape[1],n))*np.nan
    swCounts = np.ones((inim.shape[0],inim.shape[1],n))*np.nan
    for i in range(n):
        if verbose:
            print('Checking '+Mlbls[i]+'...')
        tM = M[:,:,i].copy()
        swSymm[:,:,i],_,_, swCounts[:,:,i] = swMtransf_IJradial_CC(inim, tM, swRad, calc=symmCalc)
        swCounts[:,:,i] = swCounts[:,:,i]/np.nanmax(swCounts[:,:,i].ravel())
    if verbose:
        print('Completed')

    #Plots
    if not (inax is None):
        if n>1:
            for i in range(n):
                inax[i].imshow(swSymm[:,:,i], origin='lower')
                inax[i].set_title(Mlbls[i])
        else:
            inax.imshow(swSymm[:,:,i], origin='lower')
            inax.set_title(Mlbls[i])

    return swSymm, swCounts, Mlbls
