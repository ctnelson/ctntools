import numpy as np

#Assigns neighbor categories to point array by proximity to 'inbrclusters' relative xy positions
#This function uses a large datacube [n,n,m,2], probably not ideal if [n,2] array is a significant fraction of memory
def nbrassignment(ixy, inbrcluster, mindist=0, maxdist=np.inf, missing_val = -1):
    #Inputs
    #ixy                    :       input xy positions [n,2]
    #inbrcluster            :       xy postions of nbr locations [m, 2]
    #mindist (optional)     :       minimum exclusion distance for assigning neighbor
    #maxdist (optional)     :       maximum search radius 
    #missing_val(optional)  :       value in output if no valid point is available

    #Outputs
    #nc_ind                 :       index of pts assigned to respective 

    #Parameters
    n = ixy.shape[0]
    inum = inbrcluster.shape[0]

    #find delta xy between atoms
    x = np.repeat(ixy[:,0,np.newaxis],n,axis=1) 
    y = np.repeat(ixy[:,1,np.newaxis],n,axis=1)
    dxy = np.stack((x-x.T,y-y.T),axis=2)
    dxy = np.where(dxy==0,np.nan,dxy)      
    r = (np.sum(dxy**2,axis=2))**.5
    dxy = np.where(np.repeat(r[:,:,np.newaxis],2,axis=2)<mindist,np.nan,dxy)      #apply min distance tolerance
    #stack data to compare to m nbr posibilities
    dxy = np.repeat(dxy[:,:,np.newaxis,:],inum,axis=2)
    ncgrid = np.repeat(np.repeat(inbrcluster[np.newaxis,:,:],n,axis=0)[np.newaxis,:,:,:],n,axis=0)
    #compute the distances to nbrcluster positions
    dxy_nc = dxy-ncgrid
    r = (np.sum(dxy_nc**2,axis=3))**.5
    r = np.where(r<=maxdist,r,np.inf)       #apply max distance tolerance
    #find the best match
    nc_ind = np.nanargmin(r,axis=0)
    if ~np.isfinite(missing_val):
        nc_ind = nc_ind.astype('float')
    #if non found, fill with missing_val
    temp = np.where(np.all(np.isinf(r.astype('float')),axis=0)==True)
    nc_ind[temp] = missing_val

    return nc_ind