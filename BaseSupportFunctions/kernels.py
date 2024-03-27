######################################  Table of Contents  ##########################################
#  gKernel1D    :    a 1D gaussian kernel       (normalized to have total area=1)
#  gKernel2D    :    a 2D gaussian kernel       (normalized to have total volume=1)
#  bKernel1D    :    a 1D bump function kernel. (normalized to have total area=1)
#  bKernel2D    :    a 2D bump function kernel. Can apply an optional input transform [2,2] transforme matrix M. (normalized to have total volume=1)

######################################## Imports  ###################################################
import numpy as np
import math

#############################################################################################################
################################### The Radial Functions  ###################################################
#return value(s) corresponding to radial input

##### Gaussian Function #####
def GaussFun(r,sig):
    ###  Input ###
    #r  :   radius
    #sig:   standard deviation
    z = np.exp(-.5*(r**2/sig**2))
    return z

#####  Bump Function  ######
#Returns values as a function of radius, nonzero on interval [0->1)
def BumpFun(r):
    #bump function is 1 at r=0, and 1 at |r|>1
    ###  Input ###
    #r  :   radius

    ###  Outputs  ###
    #z  :   kernel values

    ###  Main  ###
    z = np.zeros_like(r,dtype='float')
    ind = np.where((np.abs(r)>0) & (np.abs(r)<1))
    z[ind] = 1/(1 + np.exp((1-2*np.abs(r[ind])) / (r[ind]**2-np.abs(r[ind]))))
    ind = np.where(r==0)
    z[ind] = 1
    return z

##########################################################################################################################
################################### 2D Functions, w/ Matrix Transform  ###################################################
#These functions transform the xy space and call the radial functions.
#Return value(s) corresponding to x,y input
#Transform Matrix, allows for ellipsoidal and rotation transformation of radial functions

######################################
##### 2D Gaussian (w/ Transform) #####
def Gauss2DAFun(dx,dy,M=[[1,0],[0,1]]):
    ###  Inputs  ###
    #dx                 :   x values (relative to kernel center)
    #dy                 :   y values (relative to kernel center)
    #M      (optional)  :   transform matrix (defaults to identity)

    ###  Outputs  ###
    #z                  :   output kernel values

    ###  Main  ###
    M=np.array(M)
    if np.ndim(M)==2:
        M=M[:,:,np.newaxis]
    Mn = M.shape[2]
    #coordinate transform
    xy = np.vstack((dx.ravel(),dy.ravel())).T
    for i in range(Mn):
        iM = np.linalg.inv(M[:,:,Mn-i-1])
        xy = xy@iM
    dx = np.reshape(xy[:,0],dx.shape)
    dy = np.reshape(xy[:,1],dy.shape)
    #get Gaussian
    r = (dx**2+dy**2)**.5
    z = GaussFun(r,1)
    return z

###########################################
##### 2D Bump Function (w/ Transform) #####
def Bump2DAFun(dx,dy,M=np.array([[1,0],[0,1]]),rdist=None, hwhm=None):
    ###  Inputs  ###
    #dx                 :   x values (relative to kernel center)
    #dy                 :   y values (relative to kernel center)
    #M      (optional)  :   transform matrix (defaults to identity)
    #rdist  (optional)  :   outside radius      where the bump function hits zero.  One or both of rdist & hwhm must be provided.
    #hwhm   (optional)  :   half-width-half-max where the bump function hits 50%.   One or both of rdist & hwhm must be provided.

    ###  Outputs  ###
    #z                  :   output kernel values

    #param_default
    rscalar = 2     #default ratio of rdist/hwhm if not provided
    xc=0.5          #parameter to target "half" for hwhm. Don't adjust unless trying to change hwhm as the input parameter.

    #Input handling
    assert((not (rdist is None)) | (not (hwhm is None)))        #check that either hwhm or rdist is provided
    #hwhm
    if hwhm is None:
        hwhm = np.array(rdist/rscalar)
    hwhm = np.array(hwhm)
    if hwhm.size==1:
        hwhm = np.array([hwhm,hwhm],dtype='int')
    M1 = np.array([[hwhm[0], 0],[0,hwhm[1]]])  
    #rdist
    if rdist is None:
        rdist = np.array([hwhm[0]*rscalar,hwhm[1]*rscalar])
    else:
        rdist = np.array(rdist,dtype='float')
        if rdist.size==1:
            rdist = np.array([rdist,rdist],dtype='float') 
    #Transform
    if np.ndim(M)==2:
        M=M[:,:,np.newaxis]
    M = np.dstack((M1,M))
    Mn = M.shape[2]

    #coordinate transform
    xy = np.vstack((dx.ravel(),dy.ravel())).T
    for i in range(Mn):
        iM = np.linalg.inv(M[:,:,Mn-i-1])
        xy = xy@iM
    dx = np.reshape(xy[:,0],dx.shape)
    dy = np.reshape(xy[:,1],dy.shape)
    dx = dx/(rdist[0]/hwhm[0])
    dy = dy/(rdist[1]/hwhm[1])

    #get Bump
    n = np.log(xc)/np.log(hwhm/rdist)
    if not math.isclose(n[0], n[1], rel_tol=1e-6):
        print(hwhm)
        print(rdist)
        raise ValueError('Not codded for differing hwhm/rdist ratios on x & y')
    r = (dx**2+dy**2)**.5
    r = r**n[0]
    z = BumpFun(r)
    return z

###########################################################################################################
################################### 2D Kernel Creation  ###################################################
#creates a kernel of size rdist*2+1 (which may be modified by transform)

################################
##### 2D Gaussian Creation #####
def gKernel2D(sig, rdist=None, rscalar=2, M = np.array([[1,0],[0,1]]), rstp=1, normMethod = None): 
    ###  Inputs  ###
    #sig                      :  [1,] or [2,] gaussian sigma
    #rdist        (optional)  :  [1,] or [2,] distance from center to sample. (rdist/rstp) Must either be an integer or will be rounded up to one. If 'None', rdist will be based on a scalar of sigma (rscalar)
    #rscalar      (optional)  :  scalar of sigma to determine rdist if rdist = None, otherwise ignored (default 2)
    #M            (optional)  :  transformation matrix (or matrices, and applied in order) [2,2], [2,2,1], or [2,2,n]
    #normMethod   (optional)  :  None, 'Sum', 'Integrate', or 'Exact'. 'Sum' ignores the r stepsize, 'Integrate' includes it. 'Exact' is the analytical scalar
    ###  Output  ###
    #z        :  [rdist*2+1,] 1D gaussian centered at rdist

    ####  Main  ####
    #sigmas/ellipticity
    sig = np.array(sig)
    if sig.size==2:
        M1 = np.array([[sig[0], 0],[0,sig[1]]])
    else:
        M1 = np.array([[sig, 0],[0,sig]])
    #Transform M
    if np.ndim(M)==2:
        M=M[:,:,np.newaxis]
    M = np.dstack((M1,M))
    Mn = M.shape[2]
    A = M[:,:,0]
    for i in range(1,Mn):
        A = A@M[:,:,i]

    #rdist
    if rdist is None:
        #get range (from 4 corners)
        xx = np.array([-rscalar, rscalar, rscalar, -rscalar],dtype='float')
        yy = np.array([-rscalar, -rscalar, rscalar, rscalar],dtype='float')
        xy = np.vstack((xx.ravel(),yy.ravel())).T
        for i in range(Mn):
            xy = xy@M[:,:,i]
        rdist = np.array([np.ceil(np.max(np.abs(xy[:,0]))),np.ceil(np.max(np.abs(xy[:,1])))],dtype='int')
    else:
        rdist = np.array(rdist,dtype='int')
        if rdist.size==1:
            rdist = np.array([rdist,rdist],dtype='int')

    #get Gaussian
    xx, yy = np.meshgrid(np.arange(-rdist[0],rdist[0]+rstp,rstp,dtype='float'),np.arange(-rdist[1],rdist[1]+rstp,rstp,dtype='float'))
    z = Gauss2DAFun(xx,yy,M)

    #normalize
    if not (normMethod is None):
        if normMethod == 'Sum':
            zscalar = 1/np.sum(z.ravel())
        elif normMethod == 'Integrate':
            zscalar = 1/(np.trapz(np.trapz(z,axis=1)*rstp,axis=0)*rstp)
        elif normMethod == 'Exact':
            zscalar = 1/np.sqrt(np.linalg.det(2*np.pi*A@A.T))
        else:
            raise ValueError('normMethod not recognized, must be None type, "Sum", "Integrate", or "Exact"')
        z = z*zscalar

    return z

############################
##### 2D Bump Creation #####
def bKernel2D(hwhm, rdist=None, M = np.array([[1,0],[0,1]]), rstp=1, normMethod = None): 
    ###  Inputs  ###
    #hwhm                       :  [1,] or [2,] half width half max
    #rdist        (optional)    :  [1,] or [2,] outer distance
    #M            (optional)    :  transformation matrix (or matrices, and applied in order) [2,2], [2,2,1], or [2,2,n]
    #normMethod   (optional)    :  None, 'Sum' or 'Integrate'. 'Sum' ignores the r stepsize, 'Integrate' includes it.
    ###  Output  ###
    #z        :  [rdist*2+1,] 1D gaussian centered at rdist

    ####  Main  ####
    #param_default
    rscalar = 2
    #hwhm/ellipticity
    hwhm = np.array(hwhm)
    if hwhm.size==2:
        M1 = np.array([[hwhm[0], 0],[0,hwhm[1]]])
    else:
        M1 = np.array([[hwhm, 0],[0, hwhm]])

    #rdist
    if rdist is None:
        rdist = np.array([hwhm[0]*rscalar,hwhm[1]*rscalar])
    else:
        rdist = np.array(rdist,dtype='float')
        if rdist.size==1:
            rdist = np.array([rdist,rdist],dtype='float')   

    #Transform M
    if np.ndim(M)==2:
        M=M[:,:,np.newaxis]
    M2 = np.dstack((M1,M))
    Mn = M2.shape[2]

    #get transformed interpolation range (from 4 corners)
    xx = np.array([-rdist[0], rdist[0], rdist[0], -rdist[0]],dtype='float')/hwhm[0]
    yy = np.array([-rdist[1], -rdist[1], rdist[1], rdist[1]],dtype='float')/hwhm[1]
    xy = np.vstack((xx.ravel(),yy.ravel())).T
    for i in range(Mn):
        xy = xy@M2[:,:,i]
    rdistM = np.array([np.ceil(np.max(np.abs(xy[:,0]))),np.ceil(np.max(np.abs(xy[:,1])))],dtype='int')

    #get Bump Function
    xx, yy = np.meshgrid(np.arange(-rdistM[0],rdistM[0]+rstp,rstp,dtype='float'),np.arange(-rdistM[1],rdistM[1]+rstp,rstp,dtype='float'))
    z = Bump2DAFun(xx,yy,M=M,rdist=rdist, hwhm=hwhm)

    #normalize
    if not (normMethod is None):
        if normMethod == 'Sum':
            zscalar = 1/np.sum(z.ravel())
        elif normMethod == 'Integrate':
            zscalar = 1/(np.trapz(np.trapz(z,axis=1)*rstp,axis=0)*rstp)
        else:
            raise ValueError('normMethod not recognized, must be None type, "Sum" or "Integrate"')
        z = z*zscalar

    return z

##########################################################################################################
######################################## 1D Function Creation ############################################
#Creates 1D Kernel of size rdist*2+1
#######################
##### 1D Gaussian #####
def gKernel1D(sig, rdist=None, rscalar=2, rstp=1, normalize=True, normalize_withX=True):
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
        if normalize_withX:
            ind0 = np.int64(rdist/rstp)
            zscalar = 1/np.sum((z[ind0:-1] + z[ind0+1:])*rstp) 
        else:
            zscalar = 1/np.sum(z.ravel())
        z = z*zscalar
    return z

###################
##### 1D Bump #####
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
######################################## Archive #########################################################
#Old functions (and in the case of Bump, incl. errors)
######################################## 2D Gaussian Function ############################################
def gKernel2D_old(sig, rdist=None, rscalar=2, normalize=True):
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
