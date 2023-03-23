import numpy as np

#Function assigns lattice coordinates from 2 inputs: an array of local neighbors and defined relative lattice offsets for each neighbor type
def assign_ab_from_neighbors(inbrs, inbr_ab, i0=0):
    #inputs
    #inbrs      :      #neighbor index array [num atoms x num neighbors]. Invalid neighbors should be set to non finite values (e.g. np.inf or np.nan) 
    #inbr_ab    :      #atom local neighbor relative grid addresses [num neighbors x 2]
    #i0         :      #(optional) starting point index

    #outputs
    grd = np.ones((inbrs.shape[0],2))*np.nan       #array of grid address of all atoms [n,2]. if not found will remain nan

    #flags & parameters
    n = inbrs.shape[0]                #number of atoms
    grd_flag = np.zeros((n,))         #flag of new atoms to interrogate each loop
    grd_fitflag = np.zeros((n,))      #flag of atoms w/ grid assignments
    iter_lim = n

    #starting point
    grd[i0,:] = [0,0]
    grd_flag[i0] = 1

    #Grid Assignment Loop
    for i in range(iter_lim):              #loop
        ind = np.where(grd_flag==1)[0]
        #print(str(i)+': '+str(ind.size)+' new atoms')
        if len(ind)==0:                                     #if no new atoms to assign, exit
            #print('loop complete on iteration '+str(i))
            break
        else:
            for x in ind:                         #loop through new atoms
                ii = np.where(np.isfinite(inbrs[x,:]))[0]
                iii = np.where(grd_fitflag[inbrs[x,ii].astype('int')]==0)[0]
                ii = ii[iii]
                if ii.size>0:
                    grd[inbrs[x,ii].astype('int'),:] = grd[x,:]+inbr_ab[ii,:]   #neighbor atom grid assignments
                    grd_flag[inbrs[x,ii].astype('int')] = 1

                #set flags
                grd_flag[x] = 0
                grd_fitflag[x] = 1  
                grd_flag[inbrs[x,ii].astype('int')] = 1    


    #set minimum to 0
    grd[:,0] = grd[:,0]-np.floor(np.nanmin(grd[:,0]))
    grd[:,1] = grd[:,1]-np.floor(np.nanmin(grd[:,1]))

    return grd