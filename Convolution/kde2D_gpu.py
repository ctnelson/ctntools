#A GPU enabled version of Kernel Density Estimation. Useful if cannot be achieved by convolution such as non-uniform sampling grids or unique kernels for different datapoints
#still to do: intensity weights
######################################  Table of Contents  ##########################################
#Support Functions
#  GaussFun_gpu                 :   GPU Funtion, value of gaussian at r,sig
#  dMinPeriodic                 :   GPU Funtion, checks if a closer datapoint is present a periodic boundary (for single axis)
#  M22inv                       :   GPU Funtion, inverse of 2x2 matrix
#GPU kernels
#  kdeGauss2d_Radial_core_gpu   :   GPU kernel, calculate contribution of all radial gaussian at given sampling point
#  kdeGauss2d_MTransf_core_gpu  :   GPU kernel, calculate contribution of all Gaussian at given sampling point. Allows matrix transform. 
#Gaussian Kernel Density Estimator
#  kdeGauss2d_gpu               :   Gaussian kernel is applied at given datapoints, calculated at given sampling points. Allows matrix transforms of kernel.       
#  kdeGauss2d_SRtransf_gpu      :   Invokes kdeGauss2d_gpu, kernel is Scaled (S) and rotated (R). 

######################################## Imports  ###################################################
from numba import cuda
import math
import numpy as np
import time

#############################################################################################################
###################################### Support Functions  ###################################################

#Gaussian radial function
@cuda.jit(device=True)
def GaussFun_gpu(r, sig):
    ###  Input ###
    #r  :   radius
    #sig:   standard deviation
    z = math.exp(-.5*(r**2/sig**2))
    return z

#checks for an improved minimum distance on periodic boundaries
@cuda.jit(device=True)
def dMinPeriodic(d,bound):
    ###  Inputs  ###
    #d      :   distance
    #bound  :   periodic boundary
    ###  Output  ###
    dP = d      #new minimum distance
    ###  Main  ###
    if abs(d+bound) < abs(dP):
        dP = d+bound
    if abs(d-bound) < abs(dP):
        dP = d-bound 
    return dP

#Calculates coefficients for inverse of [2,2] matrix
@cuda.jit(device=True)
def M22inv(M):
    ###  Inputs  ###
    #M                  :   [2,2] matrix to find inverse
    ###  Outputs  ###
    #Minv00,01,10,11    :   inverse matrix coefficients
    A = 1/(M[0,0]*M[1,1]-M[0,1]*M[1,0])
    Minv00 = A*M[1,1]
    Minv01 = -A*M[0,1]
    Minv10 = -A*M[1,0]
    Minv11 = A*M[0,0]
    return Minv00, Minv01, Minv10, Minv11

#############################################################################################################
######################################## KDE GPU Kernels  ###################################################

#Calculates the radial basis function for all k datapoints contributing to a single sampling point. Calculation is run for all sampling points in parallel on the gpu.
#This function allows for radial scaling only via (M).
@cuda.jit
def kdeGauss2d_Radial_core_gpu(X,Y,kx,ky,kwt,M,PerBnds,scutoff,samplingMode,kdeDens,kdeVals):
    ###  Inputs  ###
    #X & Y        :   if samplingMode 'Manual' X & Y sampling points. samplingMode 'Grid' [3,] vectors of [size, offset, stepsize]
    #kx & ky      :   input xy positions
    #kwt          :   value at the datapoint
    #M            :   [1,] radial scaling
    #PerBnds      :   None or [2,]. Periodic boundary x and y values
    #scutoff      :   cutoff radius (in units of s)
    #samplingMode :   0=Grid, 1=Manual. How sample positions are handled. using Grid trades a memory call for a small conversion calcualation

    ###  Outputs  ###
    #kdeVals      :   values
    #kdeDens      :   density

    ####  Main  ####
    #thread position
    ii = cuda.grid(1)
    #get xy position
    if samplingMode==0:
        if (ii>=X[0]*Y[0]):
            return
        #convert thread position to grid position
        x = ii%X[0]
        y = ii//X[0]
        x = X[1]+x*X[2]
        y = Y[1]+y*Y[2]
    else: 
        if (ii>=X.size):
            return
        x=X[ii]
        y=Y[ii]
    
    kdeDens[ii]=0
    kdeVals[ii]=0
    #loop through all datapoints
    for i in range(kx.size):
        dx = kx[i]-x
        dy = ky[i]-y
        if PerBnds[0]!=np.inf:
            dx = dMinPeriodic(dx,PerBnds[0])
            dy = dMinPeriodic(dy,PerBnds[1])
        if (abs(dx)>M*scutoff) or (abs(dy)>M*scutoff):
            continue
            
        #converted radius
        r = (dx**2+dy**2)**.5
        if (r>M*scutoff):
            continue
        #Gaussian Value
        z = GaussFun_gpu(r,M)
        kdeDens[ii] += z
        kdeVals[ii] += z*kwt[i]


#Calculates the 'radial' basis function for all k datapoints contributing to a single sampling point. Calculation is run for all sampling points in parallel on the gpu.
#This function allows for M matrix transforms to produce non-radial kernels. This is accomplished by calling a radial kernel on a transformation of the xy space. Provides an easy method to use or create different kernel functions (e.g. bump functions). Just replace the  GaussFun_gpu() call.
#If M is constant for all k datapoints, M must be a precalculated inverse matrix. This avoids considerable overhead to recalculate it for each thread.
@cuda.jit
def kdeGauss2d_MTransf_core_gpu(X,Y,kx,ky,kwt,M,estRng,PerBnds,scutoff,samplingMode,kScalar,kdeDens,kdeVals):
    ###  Inputs  ###
    #X & Y        :   if samplingMode 'Manual' X & Y sampling points. samplingMode 'Grid' [3,] vectors of [size, offset, stepsize]
    #kx & ky      :   input xy positions
    #kwt          :   value at the datapoint
    #M            :   [1,] radial scaling, [2,2,m] precalculated inverse transform matrix, or [n,2,2,m] transform matrix (a unique one for each n datapoint) where each inverse is calculated here.
    #estRng       :   [2,] precalculated max range estimate (not applicable to uniqe M per each datapoint). Used to quickly skip datapoints out of range. [max|dx|, max|dy|] 
    #PerBnds      :   None or [2,]. Periodic boundary x and y values (currently assumes lower boundary is at 0)
    #scutoff      :   cutoff radius (in units of s)
    #samplingMode :   0=Grid, or 1=Manual. How sample positions are handled. Using a grid trades a memory call for a small conversion calcualation
    #kScalar      :   Kernel scalar. Only use here is a flag (if ==-1) to indicate this needs to calculated here

    ###  Outputs  ###
    #kdeVals      :   values
    #kdeDens      :   density

    ####  Main  ####
    #thread position
    ii = cuda.grid(1)
    #get xy position
    if samplingMode==0:
        if (ii>=X[0]*Y[0]):
            return
        #convert thread position to grid position
        x = ii%X[0]
        y = ii//X[0]
        x = X[1]+x*X[2]
        y = Y[1]+y*Y[2]
    else: 
        if (ii>=X.size):
            return
        x=X[ii]
        y=Y[ii]
    
    Mn=M.shape[-1]
    Md=M.ndim
    kdeDens[ii]=0
    kdeVals[ii]=0
    #loop through all datapoints
    for i in range(kx.size):
        dx = kx[i]-x
        dy = ky[i]-y
        if PerBnds[0]!=np.inf:
            dx = dMinPeriodic(dx,PerBnds[0])
            dy = dMinPeriodic(dy,PerBnds[1])
        #Transform
        if Md==3:       #all datapoints share M transform(s)
            if ((dx<-estRng[0]) or (dx>estRng[0]) or (dy<-estRng[1]) or (dy>estRng[1])):    #a quick square boundary cutoff to save doing additional calculations
                continue
            for j in range(Mn-1,-1,-1):
                xn = dx*M[0,0,j] + dy*M[1,0,j]
                yn = dx*M[0,1,j] + dy*M[1,1,j]
                dx = xn
                dy = yn
        elif Md==4:      #each datapoint has unique M transform(s)
            #normalize?
            #if kScalar==-1:
            #    #A = M[:,:,0].copy()
            #    #for i in range(1,Mn):
            #    #    tM = A.copy()
            #    #    A[0,0]=tM[0,0]*M[0,0,i]+tM[0,1]*M[1,0,i]
            #    #    A[0,1]=tM[0,0]*M[0,1,i]+tM[0,1]*M[1,1,i]
            #    #    A[1,0]=tM[1,0]*M[0,0,i]+tM[1,1]*M[1,0,i]
            #    #    A[1,1]=tM[1,0]*M[0,1,i]+tM[1,1]*M[1,1,i]
            #    #kwtS = 1/np.sqrt(np.linalg.det(2*np.pi*A@A.T))
            #    #kwt[i] = kwt[i]*kwtS
            #calc inv M transform
            for j in range(Mn-1,-1,-1):
                Minv00, Minv01, Minv10, Minv11 = M22inv(M[i,:,:,j])
                xn = dx*Minv00 + dy*Minv10
                yn = dx*Minv01 + dy*Minv11
                dx = xn
                dy = yn
            
        #converted radius
        r = (dx**2+dy**2)**.5
        if (r>scutoff):
            continue
        #Gaussian Value
        z = GaussFun_gpu(r,1.)
        kdeDens[ii] += z
        kdeVals[ii] += z*kwt[i]

#############################################################################################################
############################### Gaussian Kernel Density Estimator  ##########################################

#Calculates Gaussian 'radial' basis functions located at the given k datapoints at sampling points. GPU enabled
#Can use provided matrix transforms for scaling and non-radial kernels. Boundaries can be periodic if given boundary values PerBnds.
#Kernel normalization is available via 'normKernel' flag to approximately preserve mass. It is an analytical scalar so discrete sampling and the 'scutoff' sigma cropping will deviate.
#Except for unique at each point, this normalization is not yet implemented so mass is not conserved.
def kdeGauss2d_gpu(sX, sY, kx, ky, kwt, M=1, samplingMode=0, PerBnds=np.array([np.inf,np.inf],dtype=np.float32), scutoff=3, normKernel=True, tpb=64, verbose=False):
    ###  Inputs  ###
    #sX & sY                    :   [3,] start,stop,step parameters if grid. Otherwise array of sampling positions (shape ignored, will be flattened)
    #kx & ky                    :   [n,] datapoint xy positions
    #kwt                        :   [n,] or scalar, datapoint value. If single value, will be formed into an array.
    #samplingMode   (optional)  :   0=Grid, or 1=Manual
    #PerBnds        (optional)  :   none or [2,] max xy boundaries (dx dy translations occur on these values). If none, periodic boundaries not considered
    #M              (optional)  :   [1,], [2,2,m], or [n,2,2,m] transform matrix for gausian kernel. If m>1, [2,2] transform is iterated over m. If None, M default to identity matrix (a round sigma=1 gaussian)
    #scutoff        (optional)  :   cutoff radius (in units of s)
    #normKernel     (optional)  :   flag to normalize the kernel to an ~total volume of 1. This is the analytically derived scalar so low sampling density and cutting off the tails will deviate from a real value of 1. 
    #tpb            (optional)  :   threads per block for gpu kernel
    #verbose        (optional)  :   flag to print timings

    ###  Outputs  ###
    #kdeVals      :   values
    #kdeDens      :   density

    ###  Main  ###
    tic = time.perf_counter()
    if np.array(kwt).size==1:
        kwt = np.ones((kx.shape),dtype=np.float32)*kwt
    #Determine output size & preallocate
    if samplingMode==0:
        if ((sX.size==3) and (sY.size==3)):     #check correct size input. If not, will assume manual sampling points were provided instead.
            sz = np.array([(sX[1]-sX[0])/sX[2]+1, (sY[1]-sY[0])/sY[2]+1])
            if np.isclose(sz[0],np.round(sz[0]/sX[2])*sX[2]) and np.isclose(sz[1],np.round(sz[1]/sY[2])*sY[2]):     #check if range values are integers of stepsize (using this approach instead of fmod to allow some tolerance)
                sz = sz.astype('int')
            else:
                raise ValueError('sampling range not an integer of the step size. sX and sY for grid sampling are formated [start,stop,step]')
            sX = np.array([sz[0],sX[0],sX[2]],dtype=np.float32)
            sY = np.array([sz[1],sY[0],sY[2]],dtype=np.float32)
            kdeDens = np.ones((sz[0]*sz[1],),np.float32)*np.nan
            kdeVals = np.ones((sz[0]*sz[1],),np.float32)*np.nan
        else:
            samplingMode=1
    if samplingMode==1:
        sX=sX.ravel().astype(np.float32) #flatten and ensure contiguous
        sY=sY.ravel().astype(np.float32)
        kdeDens = np.ones((sX.size,),np.float32)*np.nan
        kdeVals = np.ones((sX.size,),np.float32)*np.nan
    #flatten arrays (also ensures they are contiguous, required for the numba kernel). Use of 32bit floating to save memory.
    kx = kx.ravel().astype(np.float32)
    ky = ky.ravel().astype(np.float32)
    kwt = kwt.ravel().astype(np.float32)
    #other typecasts
    M = np.array(M,dtype=np.float32)
    scutoff = np.float32(scutoff)

    #Normalize? (the case of unique M transforms per datapoint is handled in the gpu kernel)
    Md = np.ndim(M)
    if normKernel:
        if Md==3:                    #if shared transform
            A = M[:,:,0].copy()
            for i in range(1,Mn):
                A = A@M[:,:,i]
            kScalar = 1/np.sqrt(np.linalg.det(2*np.pi*A@A.T))
            kwt = kwt*kScalar
            print('shared transform normalization parameter: {:.4}'.format(kScalar))
        elif M.size==1:              #if single radial value
            kScalar = 1/(2*np.pi*M**2)
            kwt = kwt*kScalar
            print('radial normalization parameter: {:.4}'.format(kScalar))
        elif Md==4:
            #kScalar = -1             #this value is used as a flag for the gpu kernel to recalculate the normalization scalar in each thread
            kScalar = 1
            print('unique transform normalization not implemented')
        else:
            raise ValueError('Failure in Normalization Routine, Transform case not recognized')
    else:
        kScalar = 1
    
    #Precalculations (if all datapoints share M). M is replaced by inverse matrices (as only these are needed later)
    
    estRng = np.array([np.inf, np.inf],dtype=np.float32)
    if Md==3:            
        # M inverse
        Minv = np.ones(M.shape,dtype=np.float32)*np.nan
        Mn = M.shape[2]
        for i in range(Mn):
            Minv[:,:,i] = np.linalg.inv(M[:,:,i])
        #Boundaries: get simple outer range from transform or 4 corners
        xx = np.array([-scutoff, scutoff, scutoff, -scutoff],dtype='float')
        yy = np.array([-scutoff, -scutoff, scutoff, scutoff],dtype='float')
        xy = np.vstack((xx.ravel(),yy.ravel())).T
        for i in range(Mn):
            xy = xy@M[:,:,i]
        M = Minv
        estRng = np.array([np.ceil(np.max(np.abs(xy[:,0]))),np.ceil(np.max(np.abs(xy[:,1])))],dtype='int')

    tock1 = time.perf_counter()

    #transfer to GPU memory
    d_sX = cuda.to_device(sX)
    d_sY = cuda.to_device(sY)
    d_kx = cuda.to_device(kx)
    d_ky = cuda.to_device(ky)
    d_kwt = cuda.to_device(kwt)
    d_M = cuda.to_device(M)
    d_Dens = cuda.to_device(kdeDens)
    d_Vals = cuda.to_device(kdeVals)
    tock2 = time.perf_counter()

    #invoke kernel
    blockspergrid = (kdeVals.size // tpb + 1)
    if M.size==1:
        kdeGauss2d_Radial_core_gpu[blockspergrid, tpb](d_sX, d_sY, d_kx, d_ky, d_kwt, np.float32(M), PerBnds, scutoff, samplingMode, d_Dens, d_Vals)
    else:
        kdeGauss2d_MTransf_core_gpu[blockspergrid, tpb](d_sX, d_sY, d_kx, d_ky, d_kwt, d_M, estRng, PerBnds, scutoff, samplingMode, kScalar, d_Dens, d_Vals)
    tock3 = time.perf_counter()

    #memory transfer back to host
    kdeVals = d_Vals.copy_to_host()
    kdeDens = d_Dens.copy_to_host()
    tock4 = time.perf_counter()
    if samplingMode==0:
        kdeVals = np.reshape(kdeVals,sz[[1,0]])
        kdeDens = np.reshape(kdeDens,sz[[1,0]])
        if sX[2]<0:
            kdeVals = np.flip(kdeVals,axis=1)
            kdeDens = np.flip(kdeDens,axis=1)
        if sY[2]<0:
            kdeVals = np.flip(kdeVals,axis=0)
            kdeDens = np.flip(kdeDens,axis=0)

    #print(timings)
    if verbose:
        timings = np.array([tic, tock1, tock2, tock3, tock4])
        timing_lbl = ['T0', 'Initialize', 'Memory Transfer to GPU', 'Run Kernel', 'Memory Transfer to GPU']
        print('kdeGauss2d_multi_gpu Total execution time: {:.4f}'.format(timings[-1]-timings[0]))
        for i in range(1,timings.size):
            print('{:}: {:.4f}'.format(timing_lbl[i],timings[i]-timings[i-1]))

    return kdeVals, kdeDens

#At each kxy datapoint creates a 2D Gaussian for given dilation S then rotation R.
#This function is really just a wrapper for kdeGauss2d_gpu that formats M from S & R.
def kdeGauss2d_SRtransf_gpu(sX, sY, kx, ky, kwt, S, R, **kwargs):
    ###  Inputs  ###
    #sX & sY    :   sampling point xy positions
    #kx & ky    :   [n,] datapoint xy positions
    #kwt        :   [n,] or scalar, datapoint value. If single value, will be formed into an array.
    #S          :   [1,], [2,], [n,1] or [n,2] xy scalar
    #R          :   [1,] or [n,] rotation (radians)
    ### for optional variables see kdeGauss2d_gpu ###
    ###  Outputs  ###
    #kdeVals      :   values
    #kdeDens      :   density

    #Create transform matrices
    n = R.size
    #Single S & R value shared for all datapoints
    if n==1:
        M = np.zeros((2,2,2))
        #Scaling
        if S.size==1:
            M[0,0,0] = S
            M[1,1,0] = S
        elif S.size==2:
            M[0,0,0] = S[0]
            M[1,1,0] = S[1]
        #Rotation
        M[0,0,1] = np.cos(R)
        M[0,1,1] = -np.sin(R)
        M[1,0,1] = np.sin(R)
        M[1,1,1] = np.cos(R)
    #S & R values unique for all datapoints
    else:
        assert(n == kx.size)
        #Scaling
        M = np.zeros((n,2,2,2))
        if np.ndim(S)==1:
            M[:,0,0,0] = S
            M[:,1,1,0] = S
        elif np.ndim(S)==2:
            M[:,0,0,0] = S[:,0]
            M[:,1,1,0] = S[:,1]
        #Rotation
        M[:,0,0,1] = np.cos(R)
        M[:,0,1,1] = -np.sin(R)
        M[:,1,0,1] = np.sin(R)
        M[:,1,1,1] = np.cos(R)

    #intensity scaling

    kdeVals, kdeDens = kdeGauss2d_gpu(sX, sY, kx, ky, kwt, M=M, **kwargs)
    return kdeVals, kdeDens
