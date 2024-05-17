### NOTE ### KDE not coded for getUCStack

#### Image sampling functions ###
#condDownsample    :    Performs a conditional downsampling (using block_reduce) based on comparison of a ref to given dimension
#getUCStack        :    Creates a 3D stack of subimages from a given image, array of reference points, & basis vectors

#Imports
import numpy as np
from tqdm import tqdm
from skimage.measure import block_reduce
from scipy.interpolate import RegularGridInterpolator                      
from ctntools.Geometry.pointInPolygon import pointInPoly   

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
            tdyx = np.array([tdy[tind],tdx[tind]])
            UCstack[:,:,i][tind]=im[tdyx]
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
