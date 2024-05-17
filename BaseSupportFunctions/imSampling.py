#### Image sampling functions ###
import numpy as np
from skimage.measure import block_reduce

################################ condDownsample ###################################################
#Conditional Downsampling. Performs a block_reduce downsampling based on closest power of 2 ratio of a given dimension to a target dimension (e.g. unit cell spacing vs. a target pixels per unit cell)
def condDownsample(im,dim,dimTarget,verbose=False):
    ### Inputs ###
    #im         :   input image
    #dim        :   given dimension
    #dimTarget  :   target of dimension
    ### Outputs ###
    #im         :   (downsampled?) image
    #ds         :   [2,] downsampling factor used
    ds = np.floor(np.log2(dim/dimTarget))
    ds = np.round(2**np.max([ds,np.zeros_like(ds)],axis=0)).astype('int')       #downsample factor (as an integer power of 2)
    if ds.size==1:
        ds = np.tile(ds,2)
    if np.any(ds>1):
        outim = block_reduce(im, block_size=tuple(ds), func=np.mean)
        if verbose:
            print('Downsampled to {:d} by factor {:d}'.format(im.shape,ds))
    else:
        outim = im.copy()
        
    return outim, ds
