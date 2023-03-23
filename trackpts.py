#Imports
from sklearn.neighbors import NearestNeighbors
import numpy as np

#iterates through frames to identify and track points within a search radius
def trackpts(ixy,iframe,search_dist=.5):
    #Inputs:
        #   ixy     :   input xy positions [n x 2]
        #   iframe  :   "frame" of each xy position [n, 1]
        #   (optional)
        #   search_dist  :   maximum search radius in units of median nearest neighbor distance

        #Outputs:
        #   pt_lst  :   list of points [unique point num x unique frame num]
        #   nnbmdn  :   nearest neighbor median value
    
    def trackpts_core(ixy, iframe, minrad=0, maxrad = 10):
        #Inputs:
        #   ixy     :   input xy positions [n x 2]
        #   iframe  :   "frame" of each xy position [n, 1]
        #   (optional)
        #   minrad  :   minimum search radius
        #   maxrad  :   maximum search radius

        #Outputs:
        #   pt_lst  :   list of points [unique point num x unique frame num]

        frame_N = np.unique(iframe)             #frames
        fun_ind = np.arange(np.shape(ixy)[0])   #indices of all ixy rows

        #initialize using atom# in first frame
        ind = np.where(iframe==iframe[0])[0]   
        pt_N = np.size(ind)
        pt_lst = np.ones((pt_N,frame_N.size),dtype='int')*-1
        pt_lst[:,0] = fun_ind[ind]
    
        for i,val in enumerate(frame_N):
            ind = np.where(iframe==val)[0]              #index for atoms in current frame
            lp_pt_N = np.size(ind) 
            #find last valid position
            xx = np.arange(frame_N.size,0,-1,dtype='int')-1
            p_lastval = frame_N.size-np.argmax(pt_lst[:,xx] >-1,axis=1)-1
            p_x = ixy[pt_lst[np.arange(pt_N),p_lastval],0]
            p_y = ixy[pt_lst[np.arange(pt_N),p_lastval],1]
            #create neighbor grid (hopefully not tons of atoms!)
            p_x = np.repeat(p_x[:,np.newaxis], lp_pt_N, axis=1)
            n_x = np.repeat(ixy[ind,0][:,np.newaxis].T,pt_N,axis=0)    #grid of new frame x values
            dx = n_x - p_x
            p_y = np.repeat(p_y[:,np.newaxis], lp_pt_N, axis=1)
            n_y = np.repeat(ixy[ind,1][:,np.newaxis].T,pt_N,axis=0)
            dy = n_y - p_y
            r = (dx**2+dy**2)**.5

            #find closest distance match
            #closest in new frame to priors
            nrstn = np.nanargmin(r,axis=1)
            nrstn_val = np.nanmin(r,axis=1)
            #closest in priors to new frame
            nrstp = np.nanargmin(r,axis=0)
            nrstp_val = np.nanmin(r,axis=0)

            #find conflicts
            nbritern = nrstn[nrstp]==np.arange(lp_pt_N)       #are found closest neighbors mutual? (new frame index)
            nbriterp = nrstp[nrstn]==np.arange(pt_N)          #are found closest neighbors mutual? (prior index)
            nvalid = np.where((nbritern==True) & (nrstp_val>=minrad) & (nrstp_val<=maxrad))[0]    #valid prior atom found in new frame
            pvalid = nrstp[nvalid]
            pinvalid = np.where(~((nbriterp==True) & (nrstn_val>=minrad) & (nrstn_val<=maxrad)))[0]
            ninvalid = np.where(~((nbritern==True) & (nrstp_val>=minrad) & (nrstp_val<=maxrad)))[0] 

            #assign
            pt_lst[pvalid,i] = fun_ind[ind[nvalid]]

            #append new atoms?
            ninvsz = np.size(ninvalid)
            if ninvsz>0:
                #print('Frame '+str(i)+ ' '+str(ninvsz)+' added')
                temp = np.ones((ninvsz,frame_N.size),dtype='int')*-1
                pt_lst = np.append(pt_lst,temp,axis=0)
                pt_lst[-ninvsz:,i] = fun_ind[ind[ninvalid]]
                pt_N = pt_N+ninvsz

        return pt_lst

    #get average nearest neighbor distance
    knnnum = 2
    frame_N = np.unique(iframe)
    ind = np.where(iframe==frame_N[0])[0]
    nbrs = NearestNeighbors(n_neighbors=knnnum, algorithm='ball_tree').fit(ixy[ind,:])
    distances, _ = nbrs.kneighbors(ixy[ind,:])
    nnbrmdn = np.median(distances[:,1])                                       
    #print('Median nearest neighbor distance: '+str(nnbrmn) + ' pixels')
    
    #do point tracking
    pt_lst = trackpts_core(ixy, iframe, minrad=0, maxrad = search_dist*nnbrmdn) 

    return pt_lst, nnbrmdn