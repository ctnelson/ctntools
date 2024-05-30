#Creates a 'Mean' unit cell via a gaussian radial basis function
################################################# Table of Contents ############################################
#ucKDE        :  This function interpolates a 'mean' unit cell. Results is in ab lattice space (so always tiles in a grid)
#valFromUCab  :  From a unit cell ab image, interpolate values at given a- and b- coordinates
#ucxyFromKDE  :  Returns the mean unit cell converted from abspace in xyspace (not gauranteed to tile)

################################################### Imports ###################################################
import numpy as np
import matplotlib.pyplot as plt
from ctntools.Convolution.kde2D_gpu import kdeGauss2d_gpu

##################################################### ucKDE ##################################################
#This function interpolates a 'mean' unit cell. Results is in ab lattice space, that is the x- and y- orthogonal axis correspond to a- and b- vectors, even if a- and b- themselves are not orthogonal 
def ucKDE(UCwt, UCab, a, b, UCabImSz=None, UCabStpSz=None, KDEsig=.5, scutoff=6, **kwargs):
    ### Inputst ###
    #UCwt               :   [n,] datapoint values
    #UCab               :   [n,2] datapoint positions
    #UCabImSz           :   [2,] predefined UCab frame size 
    #UCabStpSz          :   UCab target stepsize (units are pixels of input image)
    #KDEsig             :   sigma of gaussian kernel used for KDE 'mean' image. This is effectively a smoothing parameter.
    #scutoff            :   scalar (sigmas) as cutoff radius for gaussian kernel

    ### Outputs ###
    #KDEim              :   'mean' image of 1 unit cell in ab space
    #KDEcnt             :   datapoint density

    ### Main ###
    #Get ucab image Scaling
    if (UCabImSz is None):
        assert(not (UCabStpSz is None))
        UCabStpSz = np.array(UCabStpSz)
        if UCabStpSz.size==1:
            UCabStpSz = np.array([UCabStpSz,UCabStpSz])
        asz = np.round((a[0]**2+a[1]**2)**.5 / UCabStpSz[0]).astype('int')
        bsz = np.round((b[0]**2+b[1]**2)**.5 / UCabStpSz[1]).astype('int')
        UCabImSz = np.array([asz,bsz],dtype='int')
    else:
        UCabImSz = np.array(UCabImSz)
        if (not np.issubdtype(UCabImSz, np.integer)):
            raise ValueError('UC ab image size must be an integer value')
        if UCabImSz.size==1:
            UCabImSz = np.array([UCabImSz,UCabImSz],dtype='int')

    #convert points
    UCabKDE = (UCab+.5)*np.repeat(UCabImSz[np.newaxis,:],UCab.shape[0],axis=0)

    #transform matrix
    Mk1 = np.linalg.inv(np.vstack((a/UCabImSz[0],b/UCabImSz[1])))     #transform matrix from xy to UCab coordinates
    Mk2 = np.array([[KDEsig,0],[0,KDEsig]])
    Mk = np.stack((Mk1,Mk2),axis=2)

    #KDE
    KDEim,KDEcnt = kdeGauss2d_gpu(np.array([0,UCabImSz[0]-1,1]), np.array([0,UCabImSz[1]-1,1]), UCabKDE[:,0], UCabKDE[:,1], UCwt, M=Mk, scutoff=scutoff, samplingMode=0, PerBnds=UCabImSz, normKernel=True, **kwargs)

    return KDEim, KDEcnt

####################################################### valFromUCab #################################################
#From a unit cell image, interpolate values at given a- and b- coordinates
def valFromUCab(UCim,ab):
    ### Input ###
    #UCim   :   UC Image in ab basis-vector space. Assume a is the horizontal and b is the vertical axis such that the imag is indexed UCim[b,a]
    #ab     :   [n,2] ab coordinates. Note, 1 unit cell is the interval 0->1, values above and below are wrapped onto this inveral
    ### Output ###
    #val    :   value of unit at given ab coordinates. Determined by linear interpolation

    ### Main ###
    imsz = np.array(UCim.shape,dtype='int')[[1,0]]
    #wrap and scale ab inputs to UCim coordinates
    ab = np.mod(ab,1.)            #wrap
    ab[:,0] = ab[:,0]*imsz[0]     #scale
    ab[:,1] = ab[:,1]*imsz[1]
    #linear interpolation
    aL = np.floor(ab[:,0])
    aH = np.mod(aL+1,imsz[0])
    aHw = ab[:,0]-aL
    aLw = 1-aHw
    bL = np.floor(ab[:,1])
    bH = np.mod(bL+1,imsz[1])
    bHw = ab[:,1]-bL
    bLw = 1-bHw
    #cast as int
    aL=aL.astype('int')
    aH=aH.astype('int')
    bL=bL.astype('int')
    bH=bH.astype('int')
    #value
    val = UCim[bL,aL]*aLw*bLw + UCim[bL,aH]*aHw*bLw + UCim[bH,aL]*aLw*bHw + UCim[bH,aH]*aHw*bHw 

    return val

####################################################### ucxyFromKDE #################################################
#returns the mean unit cell from abspace in xyspace. Size is a bounding box surrounding the ab range provided by 'abRng'.
def ucxyFromKDE(ucKDEab, a, b, abRng=[0.,1.,0.,1.], inax=None):
    ### Inputs ###
    #ucKDEab    :   mean unit cell in ab coordinates
    #a          :   [2,] a basis vector    
    #b          :   [2,] b basis vector
    #abRng      :   [4,] ab range   

    ### Outputs ###
    #xyKDE      :   mean image in xy space
    #xyKDEmask  :   mask of region within the ab vector range
    
    #Determine xy bounding box
    x = [abRng[0]*a[0]+abRng[2]*b[0], abRng[1]*a[0]+abRng[2]*b[0], abRng[1]*a[0]+abRng[3]*b[0], abRng[0]*a[0]+abRng[3]*b[0]]
    y = [abRng[0]*a[1]+abRng[2]*b[1], abRng[1]*a[1]+abRng[2]*b[1], abRng[1]*a[1]+abRng[3]*b[1], abRng[0]*a[1]+abRng[3]*b[1]]
    xxkde = np.arange(np.floor(np.min(x)).astype('int'),np.ceil(np.max(x)).astype('int')+1)
    yykde = np.arange(np.floor(np.min(y)).astype('int'),np.ceil(np.max(y)).astype('int')+1)
    xxkde, yykde = np.meshgrid(xxkde,yykde)
    #conversion to ab coords
    M  = np.linalg.inv(np.vstack((a,b))) 
    d_ab = np.vstack((xxkde.ravel(),yykde.ravel())).T @ M 
    #interpolate from the ab mean cell
    xyKDE = valFromUCab(ucKDEab,d_ab)
    xyKDE = np.reshape(xyKDE,xxkde.shape)
    #valid mask
    xyKDEmask = np.where((d_ab[:,0]>=abRng[0]) & (d_ab[:,0]<=abRng[1]) & (d_ab[:,1]>=abRng[2]) & (d_ab[:,1]<=abRng[3]),True,False)
    xyKDEmask = np.reshape(xyKDEmask,xxkde.shape)

    #Plotting?
    if not (inax is None):
        xy0 = np.array([.5*(a[0]+b[0]),.5*(a[1]+b[1])])
        xy0 = np.array([(xy0[0]-xxkde[0,0])/(xxkde[0,-1]-xxkde[0,0])*xxkde.shape[1], (xy0[1]-yykde[0,0])/(yykde[-1,0]-yykde[0,0])*yykde.shape[0]])
        dAB = np.array([[0,a[0],a[0]+b[0],b[0]]-a[0]/2-b[0]/2,[0,a[1],a[1]+b[1],b[1]]-a[1]/2-b[1]/2]).T     #relative unit cell vertices
        inax.imshow(xyKDE,origin='lower')                                                                   #image of mean unit cell
        inax.plot(np.append(dAB[:,0],dAB[0,0])+xy0[0],np.append(dAB[:,1],dAB[0,1])+xy0[1],'--y')            #plot unit cell

    return xyKDE, xyKDEmask
