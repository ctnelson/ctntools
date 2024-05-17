import numpy as np

#Finds closest neighbor to all xy points in given set. Allows a defined relative offset, for example to find closest to expected neighbor location(s).
#This function uses a large datacubes [n,n,m], probably not ideal if [n,2] array is a significant fraction of memory
def nbrByDist(ixy, RelativeOffsets=[0,0], minDist=0, maxDist=np.inf, missingVal=np.nan):
    ### Inputs ###
    #ixy                            :   input xy positions [n,2]
    #RelativeOffsets    (optional)  :   xy postions of nbr locations [m, 2]
    #minDist            (optional)  :   minimum exclusion distance for assigning neighbor
    #maxDist            (optional)  :   maximum search radius 
    #missingVal         (optional)  :   value in output if no valid point is available

    ### Outputs ###
    #output                         :   [n,m,3] closest valid points 

    #Parameters
    ixy = np.array(ixy)
    assert((np.ndim(ixy)==2) and (ixy.shape[1]==2))
    n = ixy.shape[0]
    RelativeOffsets=np.array(RelativeOffsets)
    if np.ndim(RelativeOffsets)==1:
        RelativeOffsets=RelativeOffsets[np.newaxis,:]
    m = RelativeOffsets.shape[0]

    #find delta xy between atoms
    x = np.repeat(ixy[:,0,np.newaxis],n,axis=1) 
    y = np.repeat(ixy[:,1,np.newaxis],n,axis=1)
    x = x-x.T
    y = y-y.T
    #ignore self distance
    mI = np.diag(np.ones((n,))*np.nan)+1
    x=x*mI
    y=y*mI
    #add offsets
    x=np.repeat(x[:,:,np.newaxis],m,axis=2)
    y=np.repeat(y[:,:,np.newaxis],m,axis=2)
    xOffs = np.repeat(np.repeat(RelativeOffsets[np.newaxis,np.newaxis,:,0],n,axis=0),n,axis=1)
    yOffs = np.repeat(np.repeat(RelativeOffsets[np.newaxis,np.newaxis,:,1],n,axis=0),n,axis=1)
    x=x+xOffs
    y=y+yOffs
    #distance criteria
    r = (x**2+y**2)**.5
    r = np.where(r<minDist,np.nan,r)    #min distance
    r = np.where(r>maxDist,np.nan,r)    #max distance
    #find minima
    rInd = np.nanargmin(np.nan_to_num(r,nan=np.inf),axis=0)
    mm,nn = np.meshgrid(np.arange(m),np.arange(n))
    ind = (rInd,nn,mm)
    rDist = r[ind]
    #check for none found
    ind = np.where(np.isnan(rDist))
    #overwrite index with 'missingVal'
    if ((not np.isfinite(missingVal)) | (np.mod(missingVal,1)!=0)):
        rInd=rInd.astype('float')
    rInd[ind]=missingVal

    #output
    output = np.dstack((nn,rInd,rDist))

    return output
