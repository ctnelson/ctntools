######################################  Table of Contents  ##########################################
#  gKernel1D    :    a 1D gaussian kernel   (normalized to have total area=1)
#  gKernel2D    :    a 2D gaussian kernel    (normalized to have total volume=1)

######################################## 1D Gauss Kernel ############################################
def gKernel2D(sig, rdist=None, rscalar=2, normalize=True):
    #creates a 1D guassian kernel of size rdist*2+1
    #inputs
    #sig      :  [1,] gaussian sigma
    #rdist    :  [1,] distance from center to define. Must either be an integer or will be rounded up to one. If 'None', rdist will be based on a scalar of sigma (rscalar)
    #rscalar  :  scalar of sigma to determine rdist if rdist = None, otherwise ignored (default 2)
    #output
    #outval   :  [rdist*2+1,] 1D gaussian centered at rdist

    #set rdist
    if rdist is None:
      rdist = np.ceil(rscalar*sig)
    rdist = np.ceil(rdist).astype('int')
    #create guassian
    xx, yy = np.arange(-rdist,rdist+1,1)
    outval = np.exp(-.5*(xx**2/sig**2))
    if normalize:
      outval = outval/np.sum(outval)
    return outval

######################################## 2D Gauss Kernel ############################################
def gKernel2D(sig, rdist=None, rscalar=2, normalize=True):
    #creates a 2D guassian kernel of size rdist[0]*2+1 x rdist[1]*2+1
    #inputs
    #sig      :  [1,] or [2,] gaussian sigma
    #rdist    :  [1,] or [2,] distance from center to define. Must either be an integer or will be rounded up to one. If 'None', rdist will be based on a scalar of sigma (rscalar)
    #rscalar  :  scalar of sigma to determine rdist if rdist = None, otherwise ignored (default 2)
    #output
    #outval   :  [rdist[0]*2+1, rdist[1]*2+1] 2D gaussian centered at rdist,rdist

    #set rdist
    if sig.size==1:
      sig = np.array([sig,sig])
    if rdist is None:
      rdist = np.ceil(rscalar*sig)
    if rdist.size==1:
      rdist = np.array([rdist,rdist])
    rdist = np.ceil(rdist).astype('int')
    #create guassian
    xx, yy = np.meshgrid(np.arange(-rdist[0],rdist[0]+1,1),np.arange(-rdist[1],rdist[1]+1,1))
    outval = np.exp(-.5*(xx**2/sig[0]**2+yy**2/sig[1]**2))
    if normalize:
      outval = outval/np.sum(outval.ravel())
    return outval
