import numpy as np

#finds regional maximum values (pixel-level)
def imregionalmax(inim, exclRadius, insubmask=np.empty([0],dtype='bool'), prominence=0, normalmode=0, exclusionmode=0, verbose=False):
    ### Inputs ###
    #inim                     :   input image
    #exclRadius               :   radius of exclusion range
    #insubmask (optional)     :   mask of potential peak regions
    #prominence (optional)    :   normalized (set to 0 to ignore), required prominence of peak
    #normalmode (optional)    :   normalization mode (0=full range, 1=percentile & crop)
    #exclusionmode (optional) :   0=requires value to be peak value within radius, 1=not required to be peak
    ### Outputs ###
    #pks                      :   found maxima [3,n] [x,y,value]
    #pkindex                  :   corresponding maxima indices (flattened)
    #pkprominence             :   maxima prominence (measured against min within exclRadius)
    #zoneAssigned             :   image of the sequence of ROI assignments

    #mask
    if len(insubmask.flatten())==0 or not np.all(np.shape(inim)==np.shape(insubmask)):
        submask = np.zeros_like(inim, dtype='int')
    else:
        submask = -(np.logical_not(np.copy(insubmask).astype('bool'))).astype('int')
    #
    exclRadius=np.round(exclRadius).astype('int')

    #Allocate outputs
    outxyval = []                                 #peak x,y,value array
    pkindex = np.empty([0],dtype='int')           #index position of maxima
    pkprominence = np.empty([0],dtype='int')      #prominence of peak

    #Normalize
    if normalmode==0:   #full range
        inim = inim-np.nanmin(inim.flatten())
        inim = inim/np.nanmax(inim.flatten())
    elif normalmode==1: #percentile
        inim = inim-np.nanpercentile(inim.flatten(),1)
        inim = inim/np.nanpercentile(inim.flatten(),99)

    #Local variables
    imsz=np.array(inim.shape)
    xx, yy = np.meshgrid(np.arange(np.shape(inim)[1]), np.arange(np.shape(inim)[0]))
    xx = xx.flatten()                   #x position
    yy = yy.flatten()                   #y position
    val = inim.flatten()                #image value
    zoneAssigned = submask.flatten()    #tracks the ROI for each peak assignment. Also used as the test flag to continue to iterate
    valmasked = np.where(submask.ravel()==False,val,np.min(val))
    sortind = np.flip(np.argsort(valmasked))  #pre-sort by descending value 

    #relative radial indices
    dx,dy = np.meshgrid(np.arange(-exclRadius,exclRadius+1), np.arange(-exclRadius,exclRadius+1))
    r = (dx.ravel()**2+dy.ravel()**2)**.5
    ind = np.where(r<=exclRadius)[0]
    dx = dx.ravel()[ind]
    dy = dy.ravel()[ind]

    #Loop
    n=0
    for ii in np.arange(len(val)):
        if zoneAssigned[sortind[ii]]==False:
            #current point
            currpos = sortind[ii]
            cx = xx[sortind[ii]]
            cy = yy[sortind[ii]]

            #get valid relative indices
            ind = np.where((cx+dx>=0) & (cx+dx<imsz[1]) & (cy+dy>=0) & (cy+dy<imsz[0]))[0]  
            x=dx[ind]+cx
            y=dy[ind]+cy
            yx = np.vstack([y,x])
            ind = np.ravel_multi_index(yx, np.shape(inim))

            #prominence test
            pk_p = val[currpos]-np.nanmin(val[ind])
            if  pk_p < prominence:
                if verbose:
                    print(str(ii)+': prominence failure ('+str(pk_p))
                continue

            #local max test
            if exclusionmode==0:
                if np.any(val[ind]>=val[currpos]):    #check if max within radius
                    if verbose:
                        print(str(ii)+': not regional max:')
                    continue
            
            tfind = np.where(zoneAssigned[ind]==0)[0]
            zoneAssigned[ind[tfind]]=n+1

            pkindex = np.append(pkindex,currpos)
            pkprominence = np.append(pkprominence, pk_p)

            n+=1
            
    zoneAssigned = np.reshape(zoneAssigned,inim.shape)
    pks = np.array([xx[pkindex],yy[pkindex],val[pkindex]])
    return pks, pkindex, pkprominence, zoneAssigned

#Legacy version (and flawed)
def imregionalmax_legacy(inim, windowsize, insubmask = np.empty([0],dtype='bool'), prominence=0, normalmode=0, exclusionmode=0):
    #Variables
    #inim :                       input image
    #windowsize :                 radius of exclusion range
    #submask (optional) :         mask of potential peak regions
    #prominence (optional) :      normalized (set to 0 to ignore), required prominence of peak
    #normalmode (optional) :      normalization mode (0=full range, 1=percentile & crop)
    #exclusionmode (optional) :   0=requires value to be peak value within radius, 1=not required to be peak

    if len(insubmask.flatten())==0 or not np.all(np.shape(inim)==np.shape(insubmask)):
        submask = np.ones_like(inim, dtype='bool')
    else:
        submask = np.copy(insubmask)

    #outputs
    outxyval = []                                 #peak x,y,value array
    maxposn = np.empty([0],dtype='int')           #index position of maxima
    pkprominence = np.empty([0],dtype='int')      #prominence of peak

    #Normalize
    if normalmode==0:   #full range
        inim = inim-np.nanmin(inim.flatten())
        inim = inim/np.nanmax(inim.flatten())
    elif normalmode==1: #percentile
        inim = inim-np.nanpercentile(inim.flatten(),1)
        inim = inim/np.nanpercentile(inim.flatten(),99)

    #Local variables
    xx, yy = np.meshgrid(np.arange(np.shape(inim)[1]), np.arange(np.shape(inim)[0]))
    xx = xx.flatten()                   #x position
    yy = yy.flatten()                   #y position
    val = inim.flatten()                #image value
    vind = np.arange(len(val))          #indices
    testflag = submask.flatten()        #flag whether to bother testing a point (are masked out when within exclusion radius of larger value)
    sortind = np.flip(np.argsort(val))  #pre-sort by descending value 

    #boundary mask
    bmx,bmy = np.meshgrid(np.arange(-np.floor(windowsize/2),np.floor(windowsize/2)+1), np.arange(-np.floor(windowsize/2),np.floor(windowsize/2)+1))
    b_mask = (bmx**2+bmy**2)**.5>np.floor(windowsize/2)
    b_mask[np.floor(windowsize/2).astype('int'), np.floor(windowsize/2).astype('int')] = True

    #Loop
    n=0
    for ii in np.arange(len(val)): 
        if testflag[sortind[ii]]==True:
            #get next viable maxima in sorted array
            currpos = sortind[ii]
            
            #get window
            xmin = np.maximum(xx[currpos]-np.floor(windowsize/2),0).astype('int')
            xmax = np.minimum(xx[currpos]+np.floor(windowsize/2),np.shape(inim)[1]).astype('int')
            xrng = np.arange(xmin,xmax,1,dtype='int')
            ymin = np.maximum(yy[currpos]-np.floor(windowsize/2),0).astype('int')
            ymax = np.minimum(yy[currpos]+np.floor(windowsize/2),np.shape(inim)[0]).astype('int')
            yrng = np.arange(ymin,ymax,1,dtype='int')
            #print(str(ii)+': x('+str(xmin)+':'+str(xmax)+') y('+str(ymin)+':'+str(ymax)+')')
            frameim = inim[ymin:ymax,xmin:xmax]

            #matching exclusion radius mask
            xmin_bmsk = (np.floor(windowsize/2)-xx[currpos]+xmin).astype('int')
            xmax_bmsk = (np.floor(windowsize/2)+xmax-xx[currpos]).astype('int')
            ymin_bmsk = (np.floor(windowsize/2)-yy[currpos]+ymin).astype('int')
            ymax_bmsk = (np.floor(windowsize/2)+ymax-yy[currpos]).astype('int')
            #print(str(ii)+': x('+str(xmin_bmsk)+':'+str(xmax_bmsk)+') y('+str(ymin_bmsk)+':'+str(ymax_bmsk)+')')

            mskind = np.where(b_mask[ymin_bmsk:ymax_bmsk,xmin_bmsk:xmax_bmsk].flatten()==False)

            #prominence test
            pk_p = val[currpos]-np.nanmin(inim[ymin:ymax,xmin:xmax].flatten()[mskind])
            if  pk_p < prominence:
                #print(str(ii)+': prominence failure ('+str(pk_p))
                continue

            #mask out exlusion radius
            submask[ymin:ymax,xmin:xmax]=np.logical_and(submask[ymin:ymax,xmin:xmax],b_mask[ymin_bmsk:ymax_bmsk,xmin_bmsk:xmax_bmsk])
            lx,ly = np.meshgrid(xrng,yrng)
            lxy = np.vstack([ly.flatten(),lx.flatten()])
            sind = np.ravel_multi_index(lxy, np.shape(inim))
            testflag[sind]=False

            #local peak
            if exclusionmode==0:
                if np.any(frameim.flatten()[mskind]>=val[currpos]):    #check if max within radius
                    #print(str(ii)+': not regional max:')
                    continue
            maxposn = np.append(maxposn,currpos)
            pkprominence = np.append(pkprominence, pk_p)
            n+=1
    
    outxyval = np.array([xx[maxposn],yy[maxposn],val[maxposn]])
    return outxyval, maxposn, pkprominence, b_mask
