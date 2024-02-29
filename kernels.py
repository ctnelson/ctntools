######################################  Table of Contents  ##########################################
#  gKernel1D    :    a 1D gaussian kernel       (normalized to have total area=1)
#  gKernel2D    :    a 2D gaussian kernel       (normalized to have total volume=1)
#  bKernel1D    :    a 1D bump function kernel. (normalized to have total area=1)
#  bKernel2D    :    a 2D bump function kernel. Can apply an optional input transform [2,2] transforme matrix M. (normalized to have total volume=1)

######################################## Imports  ###################################################
import numpy as np

######################################## 1D Gaussian Function ############################################
def gKernel1D(sig, rdist=None, rscalar=2, rstp=1, normalize=True):
    #creates a 1D guassian kernel of size rdist*2+1
    #inputs
    #sig      :  [1,] gaussian sigma
    #rdist    :  [1,] distance from center to define. Must either be an integer or will be rounded up to one. If 'None', rdist will be based on a scalar of sigma (rscalar)
    #rscalar  :  scalar of sigma to determine rdist if rdist = None, otherwise ignored (default 2)
    #output
    #z        :  [rdist*2+1,] 1D gaussian centered at rdist

    #set rdist
    if rdist is None:
      rdist = rscalar*sig
    assert(np.mod(rdist/rstp,1)==0)
    #create guassian
    xx = np.arange(-rdist,rdist+rstp,rstp)
    z = np.exp(-.5*(xx**2/sig**2))
    if normalize:
      z = z/np.sum(z)
    return z

######################################## 2D Gaussian Function ############################################
def gKernel2D(sig, rdist=None, rscalar=2, normalize=True):
    #creates a 2D guassian kernel of size rdist[0]*2+1 x rdist[1]*2+1
    #inputs
    #sig      :  [1,] or [2,] gaussian sigma
    #rdist    :  [1,] or [2,] distance from center to define. Must either be an integer or will be rounded up to one. If 'None', rdist will be based on a scalar of sigma (rscalar)
    #rscalar  :  scalar of sigma to determine rdist if rdist = None, otherwise ignored (default 2)
    #output
    #z        :  [rdist[0]*2+1, rdist[1]*2+1] 2D gaussian centered at rdist,rdist

    #set rdist
    sig = np.array(sig,dtype='float')
    if np.size(sig)==1:
      sig = np.array([sig,sig])
    if rdist is None:
      rdist = np.ceil(rscalar*sig)
    if np.size(rdist)==1:
      rdist = np.array([rdist,rdist])
    rdist = np.ceil(rdist).astype('int')
    #create guassian
    xx, yy = np.meshgrid(np.arange(-rdist[0],rdist[0]+1,1,dtype='float'),np.arange(-rdist[1],rdist[1]+1,1,dtype='float'))
    z = np.exp(-.5*(xx**2/sig[0]**2+yy**2/sig[1]**2))
    if normalize:
      z = z/np.sum(z.ravel())
    return z

######################################## 1D Bump Function ############################################
def bKernel1D(rdist, hwhm=None, rstp=1, normalize=True, normalize_withX=True, returngrid=False):
    #broadness rescaling achieved with distance transform x -> x**n.  so n=1 for unadjusted linear
    #Inputs:
    #rdist                          :   distance cutoff
    #hwhm               (optional)  :   broadness variable (the half width half max). Defaults to 'None' which will be set as 0.5 * rdist. Here there is no xscale adjustment.
    #rstp               (optional)  :   interpolation grid stepsize
    #normalize          (optional)  :   flag to normalize output
    #normalize_withX    (optional)  :   normalize accounting for rstp (normalization grid step size)
    #returngrid         (optional)  :   return normalization grid

    #Outputs:
    #z              :       bump kernel
    #x  (optional)  :       interpolation grid

    assert(np.mod(rdist/rstp,1)==0)
    assert(hwhm<rdist)

    #determine width / scaling
    xc=0.5     
    if hwhm is None:
        hwhm = rdist/2
    n = np.log(xc)/np.log(hwhm/rdist)
    
    #x
    ind0 = np.int64(rdist/rstp)
    x = np.arange(0,rdist+rstp,rstp)
    x = np.hstack((-x[1:][::-1],x))
    xx = np.linspace(0,1,ind0+1)
    xx = np.hstack((-xx[1:][::-1],xx))
    xadj = xx[ind0:]**n
    xadj = np.hstack((-xadj[1:][::-1],xadj))
    
    #z
    z = np.zeros_like(xadj,dtype='float')
    ind = np.where((np.abs(xx)>0) & (np.abs(xx)<1))[0]
    z[ind] = 1/(1 + np.exp((1-2*np.abs(xadj[ind])) / (xadj[ind]**2-np.abs(xadj[ind]))))
    z[ind0] = 1
    if normalize:
        if normalize_withX:
            zscalar = 1/np.sum((z[ind0:-1] + z[ind0+1:])*rstp) 
        else:
            zscalar = 1/np.sum(z.ravel())
        z = z * zscalar

    if returngrid:
        return z, x
    else:
        return z

#########################################  2D Bump Function  ###########################################
#2D bump function kernel (advantage as a constrained function)
def bKernel2D_old(rdist, n=1, M=[[1,0],[0,1]], normalize=True):
    #Inputs:
    #rmax       :       distance cutoff
    #n          :       broadness variable, larger values increase flattening of the central plateau (defaults to 1)
    #M          :       Transform matrix (defaults to identity)
    #Outputs:
    #z          :       bump kernel
    M=np.array(M)
    if np.size(rdist)==1:
        rdist = [rdist,rdist]
    rdistn = rdist.copy()
    rdistn[0] = np.ceil((M[0,0]+M[0,1])*rdist[0]).astype('int')
    rdistn[1] = np.ceil((M[1,0]+M[1,1])*rdist[1]).astype('int')
    xv = np.arange(-rdistn[0],rdistn[0]+1)/rdist[0]
    yv = np.arange(-rdistn[1],rdistn[1]+1)/rdist[1]
    xx,yy = np.meshgrid(xv, yv)
    xy = np.vstack((xx.ravel(),yy.ravel()))
    xyn = np.linalg.inv(M)@xy
    r = (xyn[0,:]**2+xyn[1,:]**2)**n
    r = np.reshape(r,xx.shape)
    z = np.zeros_like(xx,dtype='float')
    ind = np.where(r<1)
    z[ind] =  np.exp(-1/(r[ind]-1))
    if normalize:
        z = z/np.sum(z.ravel())
    return z

def bKernel2D(rdist, n=1, rstp=1, M=[[1,0],[0,1]], normalize=True):
    #Inputs:
    #rmax       :       distance cutoff
    #n          :       broadness variable, larger values increase flattening of the central plateau (defaults to 1)
    #M          :       Transform matrix (defaults to identity)
    #Outputs:
    #z          :       bump kernel
    M=np.array(M)
    if np.size(rdist)==1:
        rdist = [rdist,rdist]
    rdistn = rdist.copy()
    rdistn[0] = np.ceil((M[0,0]+M[0,1])*rdist[0]).astype('int')
    rdistn[1] = np.ceil((M[1,0]+M[1,1])*rdist[1]).astype('int')
    xv = np.arange(-rdistn[0],rdistn[0]+1)/rdist[0]
    yv = np.arange(-rdistn[1],rdistn[1]+1)/rdist[1]
    xx,yy = np.meshgrid(xv, yv)
    xy = np.vstack((xx.ravel(),yy.ravel()))
    xyn = np.linalg.inv(M)@xy
    r = (xyn[0,:]**2+xyn[1,:]**2)**n
    r = np.reshape(r,xx.shape)
    z = np.zeros_like(xx,dtype='float')
    ind = np.where(r<1)
    z[ind] =  np.exp(-1/(r[ind]-1))
    if normalize:
        z = z/np.sum(z.ravel())
    return z
