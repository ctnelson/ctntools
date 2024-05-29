import numpy as np
from tqdm import tqdm
import time
from ctntools.Convolution.kde2D_gpu import kdeGauss2d_gpu, kdeGauss2d_SRtransf_gpu

########################################### Table of Contents #########################################################
#Support:
#convKDE2arr              :  converts output sampling to an array (only works if ordered as flattened meshgrid)

#CreateDatapoints:
#CreateUCArray            :  create lattice of datapoints from basis vectors & primitives
#CreateRandArray          :  create random datapoints (can specify number and exclusion distance)

#Sample Guass Kernel @ Datapoints
#GenerateDatasetUCArray   : sample points for gauss kernel at all (Lattice) datapoints
#GenerateDatasetRand      : sample points for gauss kernel at all (Random) datapoints

#Wrapper Function
#CreateDataset            :  Wrapper function to create dataset of gaussian datapoints including some noise terms

########################################### Support ##########################################################
### Convert flattened meshgrid to array ### 
def convKDE2arr(ival,isampling,isampMesh):
    #converts input vector to array. Predicated on vector being in a meshgrid sampled order.
    ### Inputs ###
    #ival       :   [n,]    vector of input values
    #isampling  :   [n,2]   array of xy sampling points
    #isampMesh  :   [6,]    If samplingXY is an nominal meshgrid array [xstart,xstop,xstep,ystart,ystop,ystep]
    ### Outputs ###
    #outval     :   reshaped ival
    #xx         :   x positions
    #yy         :   y positions
    #meshstep   :   step of meshgrid (either the calculated version or imeshstep is passed through)

    ### Main ###
    if not (isampMesh is None):
        #sz = np.array([(sbounds[1]-sbounds[0])/meshstep[0]+1,(sbounds[3]-sbounds[2])/meshstep[1]+1],dtype='int')
        sz = np.array([(isampMesh[1]-isampMesh[0])/isampMesh[2],(isampMesh[4]-isampMesh[3])/isampMesh[5]],dtype='int')
        #outval = np.flip(np.reshape(ival.copy(),sz[[1,0]]),axis=0)
        outval = np.reshape(ival.copy(),sz[[1,0]])
        xx,yy = np.meshgrid(np.arange(isampMesh[0],isampMesh[1],isampMesh[2]),np.arange(isampMesh[3],isampMesh[4],isampMesh[5]))
    else:
        raise ValueError('isampMesh not provided')
        
    return outval, xx, yy, isampMesh
    
### Convert flattened meshgrid to array ### 
def convKDE2arr_old(ival,isampling,imeshstep):
    #converts input vector to array. Predicated on vector being in a meshgrid sampled order.
    ### Inputs ###
    #ival       :   [n,]    vector of input values
    #isampling  :   [n,2]   array of xy sampling points
    #imeshstep  :   [2,] or None, meshgrid stepsize. If none, will calculate it based on xy bounds and assertion that x&y steps are equal
    ### Outputs ###
    #outval     :   reshaped ival
    #xx         :   x positions
    #yy         :   y positions
    #meshstep   :   step of meshgrid (either the calculated version or imeshstep is passed through)

    ### Main ###
    sbounds = np.array([np.min(isampling[:,0]),np.max(isampling[:,0]),np.min(isampling[:,1]),np.max(isampling[:,1])])
    if imeshstep is None:     #solve assuming x & y steps are equal
        sz = np.array([sbounds[1]-sbounds[0],sbounds[3]-sbounds[2]])
        a=1-ival.size
        b=sz[0]+sz[1]
        c=sz[0]*sz[1]
        meshstep = (-b-(b**2-4*a*c)**.5)/(2*a)
        meshstep = np.array([meshstep,meshstep])
    elif np.array(imeshstep).size ==1:
        meshstep = np.array([imeshstep,imeshstep])
    else:
        meshstep=imeshstep
    sz = np.array([(sbounds[1]-sbounds[0])/meshstep[0]+1,(sbounds[3]-sbounds[2])/meshstep[1]+1],dtype='int')
    outval = np.flip(np.reshape(ival.copy(),sz[[1,0]]),axis=0)
    xx,yy = np.meshgrid(np.arange(sbounds[0],sbounds[1]+meshstep[0],meshstep[0]),np.arange(sbounds[2],sbounds[3]+meshstep[1],meshstep[1]))
    
    return outval, xx, yy, meshstep

########################################### Create DataPoints ##########################################################
### Lattice ###
def CreateUCArray(bounds, xy0=[0,0], primitive=[[0,0],[.5,.5]], a=[1,0],b=[0,1], edge=0, display=False):
    ###  Inputs  ###
    #bounds                 :       [4,] boundary to populate [xmin, xmax, ymin, ymax]
    #xy0        (optional)  :       [2,] origin
    #primitive  (optional)  :       [[x0,y0],[x1,y1],...]   points in unit cell (fractional coords)
    #a          (optional)  :       [2,] a vector   
    #b          (optional)  :       [2,] b vector
    #edge       (optional)  :       edge exclusion distance (default is 0)
    #display    (optional)  :       flag to plot outcome
    ###  Outputs  ###
    #pts                    :       [n,5] array of pts [:,[x,y,wt,a,b]] 
    lbls = ['x','y','wt','a','b']   #labels for pt columns

    primitive = np.array(primitive)
    if np.ndim(primitive)==1:                   #ensure 2D array so indexing below is consistent
        primitive = primitive[np.newaxis,:,]
    primitive = np.hstack((primitive,np.arange(primitive.shape[0])[:,np.newaxis]))    #append an index
    M = np.array([a,b],dtype='float')           #transformation matrix ab->xy

    #get bounds in ab space
    xv = np.array([bounds[0],bounds[1],bounds[0],bounds[1]],dtype='float')-xy0[0]
    yv = np.array([bounds[2],bounds[2],bounds[3],bounds[3]],dtype='float')-xy0[1]
    ab = np.vstack((xv,yv)).T
    ab = ab@np.linalg.inv(M)
    abbounds = np.array([np.floor(np.min(ab[:,0]))-1, np.ceil(np.max(ab[:,0])), np.floor(np.min(ab[:,1]))-1, np.ceil(np.max(ab[:,1]))],dtype='int')

    #tile unit cells in xy space
    aa,bb = np.meshgrid(np.arange(abbounds[0],abbounds[1]+1),np.arange(abbounds[2],abbounds[3]+1))
    xy = np.vstack((aa.ravel(),bb.ravel())).T
    xy = xy@M
    xy[:,0]+=xy0[0]
    xy[:,1]+=xy0[1]

    #populate primitive
    sz = np.array([xy.shape[0],primitive.shape[0]],dtype='int')
    primitive[:,:2] = ((primitive[:,:2])@M)
    pts = np.repeat(primitive[np.newaxis,:,:],sz[0],axis=0)
    pts[:,:,0] += np.repeat(xy[:,np.newaxis,0],sz[1],axis=1)
    pts[:,:,1] += np.repeat(xy[:,np.newaxis,1],sz[1],axis=1)
    temp = np.repeat(np.vstack((aa.ravel(),bb.ravel())).T[:,np.newaxis,:],sz[1],axis=1)
    pts = np.dstack((pts,temp))

    #trim at xybounds
    pts = np.reshape(pts,(sz[0]*sz[1],-1))
    ind = np.where((pts[:,0]>=bounds[0]+edge) & (pts[:,0]<=bounds[1]-edge) & (pts[:,1]>=bounds[2]+edge) & (pts[:,1]<=bounds[3]-edge))[0]
    pts = pts[ind,:]

    if display:
        brdr = np.sqrt(np.sum(np.array(a)**2))
        fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi = 100)
        ax.scatter(pts[:,0].ravel(),pts[:,1].ravel(),c=pts[:,2].ravel(),s=5)
        ax.set_xlim([bounds[0]-brdr,bounds[1]+brdr])
        ax.set_ylim([bounds[2]-brdr,bounds[3]+brdr])
        ax.plot([bounds[0],bounds[1],bounds[1],bounds[0],bounds[0]],[bounds[0],bounds[0],bounds[1],bounds[1],bounds[0]],'-k')

    return pts, lbls

### Random Points ###
def CreateRandArray(bounds, num=None, minR=None, edge = 0, verbose=False, **kwargs):
    ###  Inputs  ###
    #bounds                 :       [4,] boundary to populate [xmin, xmax, ymin, ymax]
    #num        (optional)  :       number of points (required if no minR supplied)
    #minR       (optional)  :       minimum neighbor spacing
    #verbose    (optional)  :       flag to print status
    ###  Outputs  ###
    #xy                    :       [n,2] array of pts [:,[x,y]] 

    if minR is None:
        assert(not (num is None))
        if verbose:
            print('Random datapoints, no exclusion distance')
        xy = np.random.rand(num,2)
        xy[:,0] = xy[:,0]*(bounds[1]-bounds[0])+bounds[0]
        xy[:,1] = xy[:,1]*(bounds[3]-bounds[2])+bounds[2]
    else:
        sz = np.array([bounds[3]-bounds[2],bounds[1]-bounds[0]],dtype='int')
        if num is None:         #will use a num (over)estimate corresponding to closest packing
            a = np.ceil(sz[1]/minR)
            b = np.ceil(sz[0]/(minR*np.sqrt(.75)))
            num = a*b
            num = np.int32(num)
            if verbose:
                print('initial num {:d} used as None provided'.format(num))
        gmask = np.zeros(sz,dtype='bool')
        if edge>0:
            gmask[edge:-edge,edge:-edge] = True
        else:
            gmask[:,:] = True
        xx,yy = np.meshgrid(np.arange(sz[1]),np.arange(sz[0]))
        xy = np.zeros((num,2),dtype='float')
        for i in tqdm(range(num)):
            ind = np.where(gmask.ravel())[0]
            if ind.size>0:
                loc = np.round(np.random.rand() * ind.size -.5).astype('int')
                x = xx.ravel()[ind[loc]] + np.random.rand(1)-.5
                y = yy.ravel()[ind[loc]] + np.random.rand(1)-.5
                xy[i,0] = x
                xy[i,1] = y
                r = ((xx-xy[i,0])**2 + (yy-xy[i,1])**2)**.5
                gmask = np.where(r>=minR,gmask,False)
            else:
                if verbose:
                    print('Unable to reach predefined # of gaussians due to exlusion rules, terminated at: '+str(i))
                num=i
                xy = xy[:num,:]
                break
    return xy

########################################### Sample Gaussian Kernel at Datapoints ##########################################################
### Lattice ###
def GenerateDatasetUCArray(bounds=[0,512,0,256], a=[10,1], b=[-1,10], xy0=[0,0], primitive=[[0,0,1,1.5,1.5,0],[.5,.5,.5,1,1,0]], edge=0, samplingXY=None, sampMesh=None, verbose=False, pRandType=[1,1,1,1,1,1], pRandRng=[0,0,0,0,0,0], sRandType=True, sRandRng=[0,0], **kwargs):
    ### Creates a dataset of Gaussians from a repeated unit cell sampled at given 'samplingXY' points
    
    ### Notes ###
    #not currently using grid sampling mode (samplingMode=0), some initial testing did not show any speedup on my system. Maybe useful in some use cases (very large # sampling points?)
    
    ### Inputs ###
    #bounds             [4,] window which will be populated [xmin, xmax, ymin, ymax]
    #a                  [2,] a basis vector
    #b                  [2,] b basis vector
    #xy0                [2,] xy origin
    #primitive          [pN,6] Atom Parameters [a, b, A, s1, s2, theta] for each atom in the unit cell. a and b are fractional position coordinates, A the weight, s1 & s2 the widths, and theta the rotation (radians) of the major axis (if s1!=s2)
    #samplingXY         [n,2] xy sampling points. Will default to a meshgrid of the bounds.
    #sampMesh           [6,] or None. If samplingXY is an nominal meshgrid array [xstart,xstop,xstep,ystart,ystop,ystep].
    #verbose            flag to display execution information
    #noise parameters:
    #pRandType          [6,] Flag for noise type applied to each parameter in primitive. True for normal dist, False for uniform
    #pRandRng           [6,] Scalar to define the range for each respective noise parameter
    #sRandtype          bool Flag for noise type applied to scan xy locations 
    #sRandRng           [2,] Scalar to define the range for xy noise parameters

    ### Outputs ###
    #kdeVal             [n,]    scan location values
    #samplingXY         [n,2]   nominal scan positions
    #sXY                [n,2]   actual scan positions
    #pts                [sN,5]  datapoint grid information [x, y, primitive index, row, col]
    #params             [sN,6] or [pN,6] datapoint gaussian parameters [x, y, A, s1, s2, theta]
    #a                  [2,]    returns a basis vector used
    #b                  [2,]    returns b basis vector used
    #primitive          [pN,6] returns primitive used
    #dN                 [sN,6] or [pN,2] noise applied to gaussian parameters

    ### Initial Parameters
    meshstepdefault = 0.25
    bounds = np.array(bounds,dtype='float')
    primitive = np.array(primitive,dtype='float')
    pRandType = np.array(pRandType,dtype='bool')
    pRandRng = np.array(pRandRng,dtype='float')
    sRandRng = np.array(sRandRng,dtype='float')

    #ensure 2D primitive
    if np.ndim(primitive)==1:
        primitive = primitive[np.newaxis,:]
    pN = primitive.shape[0]
    radialFlag = (primitive[:,3]==primitive[:,4])
    #Sampling Points (if not supplied)
    if samplingXY is None:
        if sampMesh is None:
            #samplingMeshStep = meshstepdefault
            sampMesh = np.array([bounds[0],bounds[1],meshstepdefault,bounds[2],bounds[3],meshstepdefault])
        #sX = np.array([0,bounds[1]-samplingMeshStep,samplingMeshStep])
        #sY = np.array([bounds[3]-samplingMeshStep,0,-samplingMeshStep])
        sX = np.array([sampMesh[0],sampMesh[1],sampMesh[2]])
        sY = np.array([sampMesh[3],sampMesh[4],sampMesh[5]])
        if verbose:
            #print('Generating default sampling grid, X[{:.2f}:{:.2f}:{:.2f}], Y[{:.2f}:{:.2f}:{:.2f}]'.format(bounds[0],samplingMeshStep,bounds[1],bounds[2],samplingMeshStep,bounds[3]))
            print('Generating default sampling grid, X[{:.2f}:{:.2f}:{:.2f}], Y[{:.2f}:{:.2f}:{:.2f}]'.format(sampMesh[0],sampMesh[2],sampMesh[1],sampMesh[3],sampMesh[5],sampMesh[4]))
        #xx,yy = np.meshgrid(np.arange(sX[0],sX[1]+sX[2],sX[2]),np.arange(sY[0],sY[1]+sY[2],sY[2]))
        xx,yy = np.meshgrid(np.arange(sX[0],sX[1],sX[2]),np.arange(sY[0],sY[1],sY[2]))
        samplingXY = np.vstack((xx.ravel(),yy.ravel())).T
    sN = samplingXY.shape[0]

    ### Main ###
    #Sampling Noise
    if np.any(sRandRng>0):
        if sRandType:   #normal
            if verbose:
                print('Sampling Grid added Normal Noise')
            dsX = np.random.randn(sN,)*sRandRng[0]
            dsY = np.random.randn(sN,)*sRandRng[1]
        else:           #uniform
            if verbose:
                print('Sampling Grid added Uniform Noise')
            dsX = sRandRng[0]*(np.random.rand(sN,) - .5)
            dsY = sRandRng[1]*(np.random.rand(sN,) - .5)
        sX = samplingXY[:,0]+dsX
        sY = samplingXY[:,1]+dsY
    else:
        sX = samplingXY[:,0]
        sY = samplingXY[:,1]

    ### Get sampled kernel ###
    if np.all(pRandRng[2:]==0):          #if no atom parameter noise all atoms (of same type) are identical and a faster kernel is run
        #initialize parameters
        params = primitive[:,2:]                         #[A, s1, s2, theta]
        for i in range(pN):                                     #loops through atom type in primitive
            #positions
            tpts = CreateUCArray(bounds, xy0=xy0, primitive=primitive[i,:2], a=a, b=b, edge=edge)[0]
            tpts[:,2] = i
            n = tpts.shape[0]
            #position noise
            dXY = np.zeros((n,2),dtype='float')
            if np.any(pRandRng[:2]>0):
                if verbose:
                    print('Datapoint Position added Noise')
                for j in range(dXY.shape[1]):
                    if pRandType[j]:   #normal
                        dXY[:,j] = np.random.randn(n,)*pRandRng[j]
                    else:           #uniform
                        dXY[:,j] = (np.random.rand(n,) - .5)*pRandRng[j]
                tpts[:,:2]+=dXY
            #kernel
            if radialFlag[i]:       #use simplest radial kernel
                if verbose:
                    print('Primitive atom# {:d} radial kernel executed, no Parameter Noise'.format(i))
                tkdeVal,_ = kdeGauss2d_gpu(sX,sY,tpts[:,0],tpts[:,1],params[i,0], M=params[i,1], samplingMode=1, verbose=verbose, **kwargs)
            else:                   #use a shared transform kernel
                if verbose:
                    print('Primitive atom# {:d} shared M transform kernel executed, no Parameter Noise'.format(i))
                tkdeVal,_ = kdeGauss2d_SRtransf_gpu(sX,sY,tpts[:,0],tpts[:,1],params[i,0], params[i,[1,2]], params[i,3], samplingMode=1, verbose=verbose, **kwargs)
            if i>0:
                kdeVal +=tkdeVal
                pts = np.append(pts,tpts,axis=0)
                dN = np.append(dN,dXY,axis=0)
            else:
                kdeVal = tkdeVal.copy()
                pts = tpts.copy()
                dN = dXY.copy()
    else:                                                       #add atom parameter noise, result in each being unique and a slower kernel
        #positions
        pts = CreateUCArray(bounds, xy0=xy0, primitive=primitive[:,:2], a=a, b=b)[0]
        n = pts.shape[0]
        #initialize parameters
        params = np.ones((n,6))*np.nan                      #[x, y, A, s1, s2, theta]
        params[:,:2] = pts[:,:2]                            #xy position
        params[:,2:] = primitive[pts[:,2].astype('int'),2:] #intensity
        #add parameter noise
        dN = np.zeros_like(params,dtype='float')
        if verbose:
            print('Added Parameter Noise of Scalar X:{:.2f}, Y:{:.2f}, A:{:.2f}, S1:{:.2f}, S2:{:.2f}, Theta:{:.2f}'.format(pRandRng[0],pRandRng[1],pRandRng[2],pRandRng[3],pRandRng[4],pRandRng[5]))
        for i in range(params.shape[1]):
            if pRandType[i]:   #normal
                dN[:,i] = np.random.randn(n,)*pRandRng[i]
            else:           #uniform
                dN[:,i] = (np.random.rand(n,) - .5)*pRandRng[i]
        params += dN
        #sampling kernel
        if np.all(pRandRng[3:]==0):     #if noise only on weights
            if verbose:
                print('shared M transform kernel executed, Parameter Noise on Weights')
            kdeVal,_ = kdeGauss2d_SRtransf_gpu(sX, sY, params[:,0], params[:,1], params[:,2], params[0,[3,4]], params[0,5], samplingMode=1, verbose=verbose, **kwargs)
        else:
            if verbose:
                print('Unique M transform kernel executed due for added Parameter Noise')
            kdeVal,_ = kdeGauss2d_SRtransf_gpu(sX, sY, params[:,0], params[:,1], params[:,2], params[:,[3,4]], params[:,5], samplingMode=1, verbose=verbose, **kwargs)

    sXY = np.vstack((sX,sY)).T
    return kdeVal, samplingXY, sXY, pts, params, dN, a, b, primitive, sampMesh

### Random ###
def GenerateDatasetRand(bounds=[0,512,0,256], num=None, minR=10, edge=1, params=[1.,1.,1.,0.], samplingXY=None, sampMesh=None, verbose=False, pRandType=[1,1,1,1,1,1], pRandRng=[0,0,0,0,0,0], sRandType=True, sRandRng=[0,0], **kwargs):
    ### Creates a dataset of Gaussians from a repeated unit cell sampled at given 'samplingXY' points
    
    ### Notes ###
    #not currently using grid sampling mode (samplingMode=0), some initial testing did not show any speedup on my system. Maybe useful in some use cases (very large # sampling points?)
    
    ### Inputs ###
    #bounds             [4,] window which will be populated [xmin, xmax, ymin, ymax]
    #num                number of datapoints
    #minR               exclusion radius
    #edge               edge exclusion distance
    #params             [4,] gauss kernel parameters [A,s1,s2,theta]
    #samplingXY         [n,2] xy sampling points. Will default to a meshgrid of the bounds.
    #sampMesh           [6,] or None. If samplingXY is an nominal meshgrid array [xstart,xstop,xstep,ystart,ystop,ystep].
    #verbose            flag to display execution information
    #noise parameters:
    #pRandType          [6,] Flag for noise type applied to each parameter in primitive. True for normal dist, False for uniform
    #pRandRng           [6,] Scalar to define the range for each respective noise parameter
    #sRandtype          bool Flag for noise type applied to scan xy locations 
    #sRandRng           [2,] Scalar to define the range for xy noise parameters

    ### Outputs ###
    #kdeVal             [n,]    scan location values
    #samplingXY         [n,2]   nominal scan positions
    #sXY                [n,2]   actual scan positions
    #pts                [sN,5]  datapoint grid information [x, y, primitive index, row, col]
    #params             [sN,6] or [pN,6] datapoint gaussian parameters [x, y, A, s1, s2, theta]
    #minR
    #dN                 [sN,6] or [pN,2] noise applied to gaussian parameters

    ### Initial Parameters
    meshstepdefault = 0.25
    bounds = np.array(bounds,dtype='float')
    params = np.array(params,dtype='float')
    pRandType = np.array(pRandType,dtype='bool')
    pRandRng = np.array(pRandRng,dtype='float')
    sRandRng = np.array(sRandRng,dtype='float')
    radialFlag = (params[1]==params[2])
    
    #Sampling Points (if not supplied)
    if samplingXY is None:
        if sampMesh is None:
            #samplingMeshStep = meshstepdefault
            sampMesh = np.array([bounds[0],bounds[1],meshstepdefault,bounds[2],bounds[3],meshstepdefault])
        #sX = np.array([0,bounds[1]-samplingMeshStep,samplingMeshStep])
        #sY = np.array([bounds[3]-samplingMeshStep,0,-samplingMeshStep])
        sX = np.array([sampMesh[0],sampMesh[1],sampMesh[2]])
        sY = np.array([sampMesh[3],sampMesh[4],sampMesh[5]])
        if verbose:
            #print('Generating default sampling grid, X[{:.2f}:{:.2f}:{:.2f}], Y[{:.2f}:{:.2f}:{:.2f}]'.format(bounds[0],samplingMeshStep,bounds[1],bounds[2],samplingMeshStep,bounds[3]))
            print('Generating default sampling grid, X[{:.2f}:{:.2f}:{:.2f}], Y[{:.2f}:{:.2f}:{:.2f}]'.format(sampMesh[0],sampMesh[2],sampMesh[1],sampMesh[3],sampMesh[5],sampMesh[4]))
        #xx,yy = np.meshgrid(np.arange(sX[0],sX[1]+sX[2],sX[2]),np.arange(sY[0],sY[1]+sY[2],sY[2]))
        xx,yy = np.meshgrid(np.arange(sX[0],sX[1],sX[2]),np.arange(sY[0],sY[1],sY[2]))
        samplingXY = np.vstack((xx.ravel(),yy.ravel())).T
    sN = samplingXY.shape[0]

    ### Main ###
    #Sampling Noise
    if np.any(sRandRng>0):
        if sRandType:   #normal
            if verbose:
                print('Sampling Grid added Normal Noise')
            dsX = np.random.randn(sN,)*sRandRng[0]
            dsY = np.random.randn(sN,)*sRandRng[1]
        else:           #uniform
            if verbose:
                print('Sampling Grid added Uniform Noise')
            dsX = sRandRng[0]*(np.random.rand(sN,) - .5)
            dsY = sRandRng[1]*(np.random.rand(sN,) - .5)
        sX = samplingXY[:,0]+dsX
        sY = samplingXY[:,1]+dsY
    else:
        sX = samplingXY[:,0]
        sY = samplingXY[:,1]

    #positions
    pts = CreateRandArray(bounds, num=num, minR=minR, edge=edge, verbose=verbose)
    n = pts.shape[0]
    #position noise
    dXY = np.zeros((n,2),dtype='float')
    if np.any(pRandRng[:2]>0):
        if verbose:
            print('Added Parameter Noise of Scalar X:{:.2f}, Y:{:.2f}'.format(pRandRng[0],pRandRng[1]))
        for j in range(dXY.shape[1]):
            if pRandType[j]:   #normal
                dXY[:,j] = np.random.randn(n,)*pRandRng[j]
            else:           #uniform
                dXY[:,j] = (np.random.rand(n,) - .5)*pRandRng[j]
        pts[:,:2]+=dXY

    #Get sampled kernel
    if np.all(pRandRng[2:]==0):          #if no atom parameter noise all atoms (of same type) are identical and a faster kernel is run
        if radialFlag:       #use simplest radial kernel
            if verbose:
                print('radial kernel executed, no Parameter Noise')
            kdeVal,_ = kdeGauss2d_gpu(sX,sY,pts[:,0],pts[:,1],params[0], M=params[1], samplingMode=1, verbose=verbose, **kwargs)
        else:                #use a shared transform kernel
            if verbose:
                print('shared M transform kernel executed, no Parameter Noise')
            kdeVal,_ = kdeGauss2d_SRtransf_gpu(sX,sY,pts[:,0],pts[:,1],params[0], params[[1,2]], params[3], samplingMode=1, verbose=verbose, **kwargs)
        dN = dXY
    else:                                                       #add atom parameter noise, result in each being unique and a slower kernel
        #add parameter noise
        dN = np.zeros((n,6),dtype='float')
        dN[:,:2] = dXY
        params = np.repeat(params[np.newaxis,:],n,axis=0)
        if verbose:
            print('Added Parameter Noise of Scalar A:{:.2f}, S1:{:.2f}, S2:{:.2f}, Theta:{:.2f}'.format(pRandRng[2],pRandRng[3],pRandRng[4],pRandRng[5]))
        for i in range(params.shape[1]):
            if pRandType[i+2]:   #normal
                dN[:,i+2] = np.random.randn(n,)*pRandRng[i+2]
            else:           #uniform
                dN[:,i+2] = (np.random.rand(n,) - .5)*pRandRng[i+2]
        params += dN[:,2:]
        #sampling kernel
        if np.all(pRandRng[3:]==0):     #if noise only on weights
            if verbose:
                print('shared M transform kernel executed, Parameter Noise on Weights')
            kdeVal,_ = kdeGauss2d_SRtransf_gpu(sX, sY, pts[:,0], pts[:,1], params[:,0], params[0,[1,2]], params[0,3], samplingMode=1, verbose=verbose, **kwargs)
        else:
            if verbose:
                print('Unique M transform kernel executed due for added Parameter Noise')
            kdeVal,_ = kdeGauss2d_SRtransf_gpu(sX, sY, pts[:,0], pts[:,1], params[:,0], params[:,[1,2]], params[:,3], samplingMode=1, verbose=verbose, **kwargs)

    sXY = np.vstack((sX,sY)).T
    return kdeVal, samplingXY, sXY, pts, params, dN, minR, sampMesh
  
########################################### Generate Dataset ##########################################################
def createDataset(method='Grid', sampMesh=None, countsPerUnit=0, baseNoiseRng=0, discretize=False, normKernel=False, verbose=False, **kwargs):
    ###  Inputs  ###
    #method                     :   'Grid' or 'Random'
    #sampMesh                   :   [6,] or None. If samplingXY is an nominal meshgrid array [xstart,xstop,xstep,ystart,ystop,ystep]. Used to reshaped and dose for shotnoise.
    #countsPerUnit              :   If grid & sampled meshgrid = counts/primitive. If random & sampled meshgrid = counts/minR_circle. If not sampled meshgrid = totalcounts.
    #baseNoiseRng               :   added gaussian background noise
    #discretize                 :   flag to output as an integer (output will be rounded). If false, shot noise will be scaled back to the primitive amplitude.
    #normKernel                 :   flag whether to normalize the gaussian kernel (conserves total mass), or not (preserves the primitive intensity value)
    #verbose            flag to display execution information
    
    ### Common inputs passed through to sub functions ###
    #bounds             [4,] window which will be populated [xmin, xmax, ymin, ymax]
    #samplingXY         [n,2] xy sampling points. Will default to a meshgrid of the bounds.
    #edge               edge exclusion distance
    #noise parameters:
    #pRandType          [6,] Flag for noise type applied to each parameter in primitive. True for normal dist, False for uniform
    #pRandRng           [6,] Scalar to define the range for each respective noise parameter
    #sRandtype          bool Flag for noise type applied to scan xy locations 
    #sRandRng           [2,] Scalar to define the range for xy noise parameters

    ### Outputs ###
    #Vals               [n,]    scan location values
    #samplingXY         [n,2]   nominal scan positions
    #sXY                [n,2]   actual scan positions
    #pts                [sN,5]  datapoint grid information [x, y, primitive index, row, col]
    #params             [sN,6] or [pN,6] datapoint gaussian parameters [x, y, A, s1, s2, theta]
    #dN                 [sN,6] or [pN,2] noise applied to gaussian parameters
    #sampMesh           [6,]    meshgrid sampling parameters (see input)
    #iScalar            intensity scalar for creating shot-noise output. Can use this to convert back to input units.
    
    ### For Random Datapoints ###
    #num                number of datapoints
    #minR               exclusion radius

    ### For Lattice ###
    #a                  [2,] a basis vector
    #b                  [2,] b basis vector
    #xy0                [2,] xy origin
    #primitive          [pN,6] Atom Parameters [a, b, A, s1, s2, theta] for each atom in the unit cell. a and b are fractional position coordinates, A the weight, s1 & s2 the widths, and theta the rotation (radians) of the major axis (if s1!=s2)

    ### Main ###
    # Get sampled Kernel
    if method=='Grid':
        Vals, samplingXY, sXY, pts, params, dN, a, b, primitive, sampMesh = GenerateDatasetUCArray(sampMesh=sampMesh,normKernel=normKernel,verbose=verbose,**kwargs)
    elif method=='Random':
        Vals, samplingXY, sXY, pts, params, dN, minR, sampMesh = GenerateDatasetRand(sampMesh=sampMesh,normKernel=normKernel,verbose=verbose,**kwargs)
    else:
        raise ValueError('method must be "Grid" or "Random"')
    
    #Convert to array?
    if not(sampMesh is None):
        if verbose:
            print('reshaping as meshgrid array')
        Vals,sxx,syy,sampMesh = convKDE2arr(Vals,samplingXY,sampMesh)

    #Shot Noise?
    if countsPerUnit>0:
        if verbose:
            print('Incorporating Shot Noise by drawing from Poisson Distribution')
        Valmx = np.sum(Vals.ravel())
        Valsnorm = Vals/Valmx        #divide by sum to convert to a probability distribution
        if not(sampMesh is None):
            if (method=='Grid'):
                a = np.array(a)
                b = np.array(b)
                pArea = np.linalg.norm(np.cross(a, b))
            elif method=='Random':
                pArea = 2*np.pi*minR**2
            else:
                raise ValueError('invalid method')
            gArea = (Vals.shape[1]*sampMesh[2]) * (Vals.shape[0]*sampMesh[5])
            numP = gArea/pArea
            totCounts = numP*countsPerUnit          #expected counts
            Vals = Valsnorm*totCounts               #expected value
            iScalar = Valmx/totCounts
        else:
            Vals = Valsnorm*countsPerUnit
            iScalar = Valmx/countsPerUnit
        Vals = np.random.poisson(lam=Vals, size=Vals.shape) #sample expected value from Poisson distribution
    else:
        iScalar=1.

    #Background Gaussian Noise?
    if baseNoiseRng>0:
        if verbose:
            print('Adding background noise, normal distribution sigma {:.2f}'.format(baseNoiseRng))
        if np.ndim(Vals)==1:
            bGNoise = np.random.randn(Vals.size)*baseNoiseRng
        elif np.ndim(Vals)==2:
            bGNoise = np.random.randn(Vals.shape[0],Vals.shape[1])*baseNoiseRng
        else:
            raise ValueError('Vals dimension not accounted for')
        #bGNoise = np.where(bGNoise<0,0,bGNoise)
        Vals = Vals.astype(np.float32) + bGNoise
        
    if discretize:
        if verbose:
            print('Discretizing Output')
        Vals = np.round(Vals).astype(np.int32)
    elif countsPerUnit>0:
        Vals=Vals*iScalar        #this resets the scale change created by the shot noise function

    return Vals, samplingXY, sXY, pts, params, dN, sampMesh, iScalar
