#Imports
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering

#Function categorizes nearest neighbors of points. Approach is Nearest neighbors -> agglomerative clustering -> sorting
def nbrclustering(ixy, inum, mintol=.2):
    #Inputs
    #ixy      :     input xy points [n, 2]
    #inum     :     number of neighbors to return

    #Outputs
    nbrmn_xyra = np.ones((inum,4))*np.nan   #   neighbor position in x,y,dist,angle [n, 4]
    rbin = []                               #   distance bins (not yet implemented)

    #Parameters
    max_pts = 10000   #for memory considerations, if>max_pts uses a random subsampling of points

    ###get average nearest neighbor distance
    #scipy nearest neighbors
    knnnum = inum*2+1
    nbrs = NearestNeighbors(n_neighbors=knnnum, algorithm='ball_tree').fit(ixy)
    distances, indices = nbrs.kneighbors(ixy)
    nnbrmn = np.median(distances[:,1])

    #get delta xy
    dxy = ixy[indices[:,1:knnnum],:]-np.transpose(np.repeat(ixy[:, :, np.newaxis], knnnum-1, axis=2),(0, 2, 1))
    dxy = np.reshape(dxy,[np.shape(dxy)[0]*np.shape(dxy)[1],2])
    if ixy.shape[0] > max_pts:
        subind = np.array(random.sample(range(np.size(dxy,axis=0)),k=max_pts))
    else:
        subind = np.arange(ixy.shape[0])
    dist_thresh = nnbrmn*.5

    #Clustering
    clustering_sub = AgglomerativeClustering(n_clusters=None, distance_threshold=dist_thresh, compute_full_tree=True, linkage='average').fit(dxy[subind,:])

    #Get mean position of each cluster
    ac_mnxy = np.nan*np.ones([clustering_sub.n_clusters_,2],dtype=float)
    for i in np.arange(clustering_sub.n_clusters_):
        c_ind = np.where(clustering_sub.labels_==i)[0]
        ac_mnxy[i,0] = np.mean(dxy[subind[c_ind],0])
        ac_mnxy[i,1] = np.mean(dxy[subind[c_ind],1]) 

    ###Sort and categorize neighbors:
    #nearest neighbors (avoiding close to dxy=0,0)
    r = np.sum(ac_mnxy**2,axis=1)**.5
    d_ind_at = np.where(r>mintol*nnbrmn)[0]     #min distance search threshold       
    d_ind = np.argsort(r[d_ind_at])
    nnbr_ind = d_ind_at[d_ind[:inum]]     #restricts to nnum nearest

    #get cluster closest to x axis direction
    ang = np.arctan2(ac_mnxy[nnbr_ind,1],ac_mnxy[nnbr_ind,0])
    a0_a = np.argmin(np.abs(ang))                            
    #and sort by angle
    ang = np.where(ang-ang[a0_a]<0,ang+2*np.pi,ang)
    ang_ind = np.argsort(ang)

    #sort
    nnbr_ind = nnbr_ind[ang_ind]
    ang = ang[ang_ind]

    #Output
    nbrmn_xyra[:,:2] = ac_mnxy[nnbr_ind,:]
    nbrmn_xyra[:,2] = r[nnbr_ind]
    nbrmn_xyra[:,3] = ang

    return nbrmn_xyra, rbin