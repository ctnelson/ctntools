#swSlices      #Slices window grid on image give winSz and stride. Returns stack of image patches, grid size, grid origins, and an index stack indicating overlaps.

###################################################   Imports   ####################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pch
from tqdm import tqdm

###################################################   swSlices   ####################################################
def swSlices(inim, winSz, stride, iplot=0, verbose=0):
    ### Inputs ###
    #inim               :   [h,w]   input image
    #winSz              :   [2,]    window size
    #stride             :   [2,]    stride
    #iplot (optional)   :   index of patch to plot (if plotting)
    #verbose (optional) :   if >1 display process

    ### Outputs ###
    #imPatchStack   :   [winSz[0],winSz[1],patchNum]    stack of image patches   
    #sliceWinSz     :   array size of the image patch slice grid
    #imPatchOrigin  :   tuple meshgrid of imPatch origins within source inim
    #sliceInd       :   [h,w,stackN] indices of slices the size of inim. 3rd dimension shows overlapping slice indices

    imSz = np.array(inim.shape)[[1,0]]        #image dimensions  
    d = np.array([np.ceil(winSz[0]/stride[0]).astype('int'), np.ceil(winSz[1]/stride[1]).astype('int')])        #window overlap in x and y axis
    #x axis
    xv = np.arange(0,imSz[0]-winSz[0]+1,stride[0])
    temp = np.arange(xv.size)
    xSliceInd = np.ones((imSz[0],d[0]))*np.nan
    for i in range(d[0]):
        xSliceInd[i*stride[0]:stride[0]*xv.size+i*stride[0],i] = np.repeat(temp,stride[0])
    #y axis
    yv = np.arange(0,imSz[1]-winSz[1]+1,stride[1])
    temp = np.arange(yv.size)
    ySliceInd = np.ones((imSz[1],d[1]))*np.nan
    for i in range(d[1]):
        ySliceInd[i*stride[1]:stride[1]*yv.size+i*stride[1],i] = np.repeat(temp,stride[1])
    #Index stack
    sliceWinSz = np.array([xv.size, yv.size],dtype='int')
    xSliceInd = np.repeat(xSliceInd[np.newaxis,:,:],imSz[1],axis=0)
    xSliceInd = xSliceInd[:,:,np.tile(np.arange(d[0]),d[1])]
    ySliceInd = np.repeat(ySliceInd[:,np.newaxis,:],imSz[0],axis=1)
    ySliceInd = np.repeat(ySliceInd,d[0],axis=2)
    sliceInd = np.ones_like(xSliceInd)*np.nan
    ind = np.where(np.isfinite(xSliceInd) & np.isfinite(ySliceInd))
    sliceInd[ind] = xSliceInd[ind] + ySliceInd[ind]*xv.size

    #meshgrid
    imPatchOrigin = np.meshgrid(xv, yv)
    ii,jj = np.meshgrid(np.arange(sliceWinSz[0]), np.arange(sliceWinSz[1]))
    #get image patch array
    n = imPatchOrigin[0].size
    imPatchStack = np.ones((winSz[0],winSz[1],n))*np.nan
    for i in tqdm(range(n), desc='calculating sliding window patches', disable=(verbose==0)):
        xx = ii.ravel()[i]
        yy = jj.ravel()[i]
        ind = np.array([xx*stride[0], xx*stride[0]+winSz[0], yy*stride[1], yy*stride[1]+winSz[1]])
        imPatchStack[:,:,i] = inim[ind[2]:ind[3], ind[0]:ind[1]]

    if verbose>1:
        #plotting
        fig, ax = plt.subplots(1, 2, figsize=(18,9), dpi = 100)
        #Image w/ slice window overlay
        color = plt.cm.jet(np.linspace(0, 1, n))
        winxx = imPatchOrigin[0]
        winyy = imPatchOrigin[1]
        ax[0].imshow(inim,origin='lower',cmap='gray')
        for i in range(n):
            x = np.array([winxx.ravel()[i],winxx.ravel()[i]+winSz[0],winxx.ravel()[i]+winSz[0],winxx.ravel()[i],winxx.ravel()[i]])
            y = np.array([winyy.ravel()[i],winyy.ravel()[i],winyy.ravel()[i]+winSz[1],winyy.ravel()[i]+winSz[1],winyy.ravel()[i]])
            ax[0].plot(x,y,'-',color=color[i])
            if i==iplot:
                temp = np.vstack((x,y)).T
                winPatch = pch.Polygon(np.vstack((x,y)).T, closed=True, edgecolor=color[i], facecolor=[1,1,1], fill=True)
                ax[0].add_patch(winPatch)
        ax[0].set_aspect(1)
        #Select window
        ax[1].imshow(imPatchStack[:,:,iplot],origin='lower')
        ax[1].set_title('Image Patch {:d}'.format(iplot))
        
    return imPatchStack, sliceWinSz, imPatchOrigin, sliceInd
