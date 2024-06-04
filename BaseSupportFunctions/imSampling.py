### NOTE ### KDE not currently linked to getUCStack. currently two distinct functions.

######################################### Table of Contents #####################################
#condDownsample    :    Performs a conditional downsampling (using block_reduce) based on comparison of a ref to given dimension
#getUCStack        :    Creates a 3D stack of subimages from a given image, array of reference points, & basis vectors
#imDatapointsInUC  :    Return indices of datapoints found in unit cells (subpixel), designed by centerpoint,a,&b vectors

############################################# Imports ############################################
import numpy as np
from tqdm import tqdm
from skimage.measure import block_reduce
from scipy.interpolate import RegularGridInterpolator                      
from ctntools.Geometry.pointInPolygon import pointInPoly, imIndInPoly   

########################################### condDownsample ###################################################
#Conditional Downsampling. Performs a block_reduce downsampling based on closest power of 2 ratio of a given dimension to a target dimension (e.g. unit cell spacing vs. a target pixels per unit cell)
def condDownsample(im,dim,dimTarget,verbose=False):
    ### Inputs ###
    #im         :   input image
    #dim        :   given dimension
    #dimTarget  :   target of dimension
    ### Outputs ###
    #im         :   (downsampled?) image
    #ds         :   [2,] downsampling factor used
    ds = np.floor(np.log2(dim/dimTarget))
    ds = np.round(2**np.max([ds,np.zeros_like(ds)],axis=0)).astype('int')       #downsample factor (as an integer power of 2)
    if ds.size==1:
        ds = np.tile(ds,2)
    if np.any(ds>1):
        outim = block_reduce(im, block_size=tuple(ds), func=np.mean)
        if verbose:
            print('Downsampled to {:d} by factor {:d}'.format(im.shape,ds))
    else:
        outim = im.copy()
        
    return outim, ds

############################################################ get UC stack ##################################
#creates a 3D stack of subimages from a given image, array of reference points, & basis vectors.
def getUCStack(inim, xy, a, b, abUCoffset=[.5,.5], method='round',KDEs=.5,KDErscalar=3,verbose=False, **kwargs):
    ### Inputs ###
    #inim       :   input image
    #xy         :   [n,2] xy locations in inim
    #a          :   [2,] a vector
    #b          :   [2,] b vector
    #abUCoffset :   [2,] offset (units of a&b) corresponding to xy locations. Example, [.5,.5] is the unit cell center.
    #method     :   'interp', 'KDE', or 'round'
    #KDEs       :   KDE interpolation sigma
    #KDErscalar :   KDE interpolation cutoff distance (units of sigma)
    #verbose    :   flag to print execution details
    ### Outputs ###
    #UCstack    :   stack of subimages
    #UCmask     :   mask for unit cells

    ### Main ###
    if (method=='round') or (method=='interp'):
        brdr = 1
    elif method=='KDE':
        brdr = np.int32(np.ceil(KDErscalar/KDEs))
    
    dx=(np.abs(a[0])+np.abs(b[0]))
    dy=(np.abs(a[1])+np.abs(b[1]))
    #bounding box of unitcell (with extended border)
    dxrng = np.array([-brdr/2-np.abs(a[0])*abUCoffset[0]-np.abs(b[0])*abUCoffset[1], dx+brdr/2-np.abs(a[0])*abUCoffset[0]-np.abs(b[0])*abUCoffset[1]])
    dyrng = np.array([-brdr/2-np.abs(a[1])*abUCoffset[0]-np.abs(b[1])*abUCoffset[1], dy+brdr/2-np.abs(a[1])*abUCoffset[0]-np.abs(b[1])*abUCoffset[1]])
    #relative meshgrid of bounding box
    x = np.arange(np.floor(dxrng[0]),np.ceil(dxrng[1])+1)
    y = np.arange(np.floor(dyrng[0]),np.ceil(dyrng[1])+1)
    dx,dy = np.meshgrid(x,y)
    #Create UCstack
    n = xy.shape[0]
    UCstack = np.ones((np.ceil(dyrng[1]).astype('int')-np.floor(dyrng[0]).astype('int')+1, np.ceil(dxrng[1]).astype('int')-np.floor(dxrng[0]).astype('int')+1,n))*np.nan
    if method=='interp':
        xx = np.arange(inim.shape[0])
        yy = np.arange(inim.shape[1])
        interp = RegularGridInterpolator((xx, yy), inim)
    for i in tqdm(range(n),'Creating Unit Cell Stack...',disable=(not verbose)):
        #UC datalocations in inim
        tdx = dx+xy[i,0]
        tdy = dy+xy[i,1]
        tind = np.where((np.floor(tdx)>=0) & (np.ceil(tdx)<inim.shape[1]) & (np.floor(tdy)>=0) & (np.ceil(tdy)<inim.shape[0])) #valid indices
        #Round
        if method=='round':
            tdx = np.round(tdx).astype('int')
            tdy = np.round(tdy).astype('int')
            #tdyx = np.array([tdx[tind],tdy[tind]])
            tdyx = (tdy[tind],tdx[tind])
            UCstack[:,:,i][tind]=inim[tdyx]
        elif method=='interp':
            tdyx = np.array([tdy[tind],tdx[tind]]).T
            UCstack[:,:,i][tind]=interp(tdyx)
        elif method=='KDE':
            raise ValueError('not yet coded')

    #Unit Cell Mask    
    xy0 = np.array([-abUCoffset[0]*a[0]-abUCoffset[1]*b[0], -abUCoffset[0]*a[1]-abUCoffset[1]*b[1]])
    #mask = maskFromABOrigin(a,b,xy0,dx,dy)
    verts = np.array([[xy0[0],xy0[0]+a[0],xy0[0]+a[0]+b[0],xy0[0]+b[0]],[xy0[1],xy0[1]+a[1],xy0[1]+a[1]+b[1],xy0[1]+b[1]]]).T
    test = np.array([dx.ravel(),dy.ravel()]).T
    mask = pointInPoly(test,verts)[0]
    mask = np.reshape(mask,dx.shape)
    #strip border
    UCstack = UCstack[brdr:-brdr,brdr:-brdr,:]
    mask = mask[brdr:-brdr,brdr:-brdr]
    
    return UCstack, mask

################################################################################################################
#This function collects the image datapoints found within unit cells for given location, a, and b vectors
def imDatapointsInUC(imSz,ucxy,a,b,ucScore=None):
    ### Inputs ###
    #imSz       :   [2,] image size
    #ucxy       :   [n,2] unit cell center points
    #a          :   [2,] a vector
    #b          :   [2,] b vector
    #ucScore    :   [n,] unit cell score (if conflicts will default to the uc with higher score)

    ### Outputs ###
    #UCab       :   [n,2] Unit cell datapoints in ab coordinates
    #UCxy       :   [n,2] Unit cell datapoints in xy coordinates
    #imUCind    :   [h,w] Map of uc index assignments matching im.shape         (duplicates are resolved)
    #imUCscore  :   [h,w] Map of corresponding score of uc index assignments    (duplicates are resolved)
    #UCimInd    :   [n,] Flat Array of indices of UC assignments                (includes/allows duplicates)
    #UCimI      :   [n,] Flat Array of corresponding unit cell #                (includes/allows duplicates)

    #general setting
    if ucScore is None:
        ucScore = np.ones(ucxy.shape[0],)
    xx,yy = np.meshgrid(np.arange(im[1]),np.arange(im[0]))                                              #x,y position meshgrid
    dAB = np.array([[0,a[0],a[0]+b[0],b[0]]-a[0]/2-b[0]/2,[0,a[1],a[1]+b[1],b[1]]-a[1]/2-b[1]/2]).T     #relative unit cell vertices
    #preallocate
    UCimInd = np.array([],dtype='int')
    UCimI = np.array([],dtype='int')
    imUCind = np.ones(imSz,dtype='int')
    imUCscore = np.ones(imSz,dtype='float')*np.nan
    #loop through unit cells
    for i in np.arange(ucxy.shape[0],dtype='int'):
        #get indices within UC
        idAB = dAB+np.repeat(ucxy[i,np.newaxis,:2],4,axis=0)    #unit cell vertices at index
        ind = imIndInPoly(imSz,idAB,xx,yy)                        #return indices in unit cell
        #update maps (includes conflict resolution. If any points are multiply subscribed priority is to the highest score)
        compareArr = np.array([imUCscore.ravel()[ind],np.ones(ind.size,)*ucScore[i]])   #stack current and new score values
        maxScoreInd = np.nanargmax(compareArr,axis=0)                                   #get index of max
        imUCscore.ravel()[ind] = compareArr[(maxScoreInd,np.arange(ind.size))]          #assign new score
        compareArr = np.array([imUCind.ravel()[ind],np.ones(ind.size,)*i])              #stack current and new uc index values
        imUCind.ravel()[ind] = compareArr[(maxScoreInd,np.arange(ind.size))]            #assign new index
        #update index arrays
        UCimInd = np.append(UCimInd,ind)                    
        UCimI = np.append(UCimI,np.ones(ind.size,dtype='int')*i)

    #transformation Matrix for xy->ab spaces
    M  = np.linalg.inv(np.vstack((a,b)))
    #Coordinates in Unit Cell coordinates (xy)
    UCdx = xx.ravel()[UCimInd]-ucxy[UCimI,0]
    UCdy = yy.ravel()[UCimInd]-ucxy[UCimI,1]
    #Coordinates in Unit Cell coordinates (ab)
    UCxy = np.vstack((UCdx,UCdy)).T
    UCab = UCxy@M

    return UCab, UCxy, imUCind, imUCscore, UCimInd, UCimI
