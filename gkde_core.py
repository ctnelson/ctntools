import numpy as np
from scipy.ndimage import convolve

########################################################### Support Functions ##################################################################
#increments the given indices by weighted value (defaults to 1)
#functionally a 2D extension of numpy bincount
def accumarray(rc,ashape,wt=None):
    #Inputs
    #rc         :   [2,n] row-column indices that will increment
    #ashape     :   shape of array
    #wt         :   [n,] or [1] weights to increment

    #Outputs
    #outarray   :   accumlated array


    n = rc.shape[1]
    if wt is None:
        wt = np.ones((n,))
    elif wt.size==1:
        wt = np.ones((n,))*wt
    accmap = np.ravel_multi_index(rc,ashape)
    outarray = np.bincount(accmap, weights = wt, minlength=ashape[0]*ashape[1])
    outarray = np.reshape(outarray,ashape)

    return outarray

#2D gaussian kernel
def gkernel(s,wincutoff=3):
    if np.size(s)==1:
        s = [s,s]
    xv = np.arange(-np.ceil(s[0])*wincutoff,np.ceil(s[0])*wincutoff+1)
    yv = np.arange(-np.ceil(s[1])*wincutoff,np.ceil(s[1])*wincutoff+1)
    xx,yy = np.meshgrid(xv, yv)
    z =  1/(2*np.pi*s[0]*s[1]) * np.exp(-.5*xx**2/s[0]**2 - .5*yy**2/s[1]**2)
    return z

#2D gaussian kernel (allowing non integer values of s but can create sharp edges depending on s vs rmax)
def gkernel_float(s,rmax):
    if np.size(s)==1:
        s = [s,s]
    if np.size(rmax)==1:
        rmax = [rmax,rmax]
    xv = np.arange(-rmax[0],rmax[0]+1)*s[0]**-1
    yv = np.arange(-rmax[1],rmax[1]+1)*s[1]**-1
    xx,yy = np.meshgrid(xv, yv)
    z =  np.exp(-.5*xx**2 - .5*yy**2)
    z = z/np.sum(z.ravel())
    return z

#2D bump function kernel (advantage as a constrained function)
def bkernel(rmax,n=1):
    #Inputs:
    #rmax       :       distance cutoff
    #n          :       broadness variable, larger values increase flattening of the central plateau
    #Outputs:
    #z          :       bump kernel
    if np.size(rmax)==1:
        rmax = [rmax,rmax]
    xv = np.arange(-rmax[0],rmax[0]+1)/rmax[0]
    yv = np.arange(-rmax[1],rmax[1]+1)/rmax[1]
    xx,yy = np.meshgrid(xv, yv)
    r = (xx**2+yy**2)**n
    z = np.zeros_like(xx)
    ind = np.where(r<1)
    z[ind] =  np.exp(1/(r[ind]-1))
    z = z/np.sum(z.ravel())
    return z

def xytointerpgrid(ix, iy, x0=None, y0=None, ixv=[1,0], iyv=[0,1]):
    #creates and applies transformation matrix on input x & y vectors
    #inpute can either be basis vectors or the interpolation meshgrid
    
    #Inputs
    #ix     :       x positions
    #iy     :       y positions
    #x0     :       x origin
    #y0     :       y origin
    #ixv    :       x vector or x meshgrid
    #iyv    :       y vector or y meshgrid
    
    #Outputs
    #oxy    :       transformed xy coords

    ixv = np.array(ixv)
    iyv = np.array(iyv)

    #ixv & iyv are vectors
    if ((np.size(ixv)==2) & (np.size(iyv)==2)):
        xy = np.vstack((ix.ravel(),iy.ravel(),np.ones_like(iy.ravel())))
        M = np.array([[ixv[0],ixv[1],x0],[iyv[0],iyv[1],y0],[0,0,1]],dtype='float')
        oxy = M @ xy
        oxy = oxy[:2,:].T
    
    #ixv & iyv are meshgrids
    #note, will not rigorously check if actually a meshgrid
    if (np.ndim(ixv)==2) & (np.ndim(iyv)==2) & (np.size(ixv)>=4) & (np.size(iyv)>=4):
        if x0 is None:
            x0 = -np.min(ixv)
        if y0 is None:
            y0 = -np.min(iyv)
        ixv = np.array([ixv[0,1] - ixv[0,0],0])
        iyv = np.array([0,iyv[1,0] - iyv[0,0]])
        xy = np.vstack((ix.ravel(),iy.ravel(),np.ones_like(iy.ravel())))
        M = np.array([[ixv[0],ixv[1],x0],[iyv[0],iyv[1],y0],[0,0,1]],dtype='float')
        oxy = M @ xy
        oxy = oxy[:2,:].T

    return oxy

############################################## Radial Basis Function Interpolator: Periodic Unit Cell ##################################################
#This RBFI uses periodic boundaries. Inputs are asserted to be in units of fractional 'unit cell' coordinates.
#a and b vectors (av and bv) are required basis vectors used to map to the a & b coordinates. Their magnitude determines the output image size.
#Their full vector is required in order to determine the 'radial' kernel in xy space (as opposed to the ab space used here)
def gkde_core_periodic(ia, ib, iz, av=None, bv=None, ds=1, isig=np.array([1,1]), r_cutoff=.1, mode='interp'):
    #Inputs:
    #position data:
    #ia & ib        :   input ab positions (should be in a range from 0 to 1)
    #iz             :   value at the datapoints

    #interpolation space info:
    #av             :   a vector [x,y] (in source xy coordinates)
    #bv             :   b vector [x,y] (in source xy coordinates)
    #ds             :   (down)scalar of vectors

    #parameters:
    #isig           :   sigma for gausian kernel (in source xy coordinates)
    #r_cutoff
    #mode           :   'interp' will consider subpixel positions as bilinear interpolations of integer values which are convolved with a gaussian kernel 
    #                   'nearest' subpixel values are rounded to closes integer position                       which are convolved with a gaussian kernel
    #                   'exact' exact solutions calculated against gaussian equation, will be slow

    #Returns:
    #densitymap     :   A map of the density of input positions
    #valsmap        :   A map of the interpolated data

    #Parameters
    am = np.round(((av[0]**2+av[1]**2)**.5)*ds)
    bm = np.round(((av[0]**2+av[1]**2)**.5)*ds)
    kl = np.array([np.floor(-r_cutoff*am), np.ceil(r_cutoff*am), np.floor(-r_cutoff*bm), np.ceil(r_cutoff*bm)],dtype=np.int16)
    #convert fractional to image
    apos = ia*am
    bpos = ib*bm

    if isig.size==1:
        sig = np.array(isig,(2,))
    else:
        sig=isig

    #Kernel
    #creates a gaussian kernel for convolution
    #defined in terms of the source xy space, will be transformed to the ab space here
    kaa,kbb = np.meshgrid(np.arange(kl[0],kl[1]+1),np.arange(kl[2],kl[3]+1))
    M = np.vstack((av,bv))
    ab = np.vstack((kaa.ravel(),kbb.ravel())).T.astype(np.float32)
    ab[:,0] = ab[:,0]/am
    ab[:,1] = ab[:,1]/bm
    kxy = ab@M
    #k =  1/(2*np.pi*isig[0]*isig[1]) * np.exp(-.5*kxy[:,0]**2/isig[0]**2 - .5*kxy[:,1]**2/isig[1]**2)
    k =  np.exp(-.5*kxy[:,0]**2/isig[0]**2 - .5*kxy[:,1]**2/isig[1]**2)
    k = k/np.sum(k)
    k = np.reshape(k,kaa.shape)

    #Main
    vals = np.zeros((bm.astype('int'),am.astype('int')),dtype='float')
    dens = np.zeros((bm.astype('int'),am.astype('int')),dtype='float')

    #split subpixel as a bilinear interpolation of surrounding integer positions
    if mode=='interp':
        #indices & weights
        aH = apos-np.floor(apos)
        aL = 1-aH
        bH = bpos-np.floor(bpos)
        bL = 1-bH
        aposL = np.int16((np.floor(apos)  )    % am)
        aposH = np.int16((np.floor(apos)+1)    % am)
        bposL = np.int16((np.floor(bpos)  )    % bm)
        bposH = np.int16((np.floor(bpos)+1)    % bm)

        #density & value ~histogram
        ind = np.vstack((bposL,aposL))
        vals += accumarray(ind, vals.shape, wt=aL*bL*iz)    #p00 value
        dens += accumarray(ind, dens.shape, wt=aL*bL)       #p00 density
        ind = np.vstack((bposH,aposL))
        vals += accumarray(ind, vals.shape, wt=aL*bH*iz)    #p01
        dens += accumarray(ind, dens.shape, wt=aL*bH)       #p01
        ind = np.vstack((bposL,aposH))
        vals += accumarray(ind, vals.shape, wt=aH*bL*iz)    #p10
        dens += accumarray(ind, dens.shape, wt=aH*bL)       #p10
        ind = np.vstack((bposH,aposH))
        vals += accumarray(ind, vals.shape, wt=aH*bH*iz)    #p11
        dens += accumarray(ind, dens.shape, wt=aH*bH)       #p11
    
    #round to nearest pixel position
    elif mode=='nearest':
        apos = np.round(apos)
        bpos = np.round(bpos)
        #values
        ind = np.vstack((bpos.astype('int'),apos.astype('int')))
        vals += accumarray(ind, vals.shape, wt=iz)   
        dens += accumarray(ind, dens.shape)    

    elif mode=='exact':
        print('not yet coded')
    else:
        raise ValueError('mode not recognized')

    #convolve w/ kernel
    densitymap = convolve(dens,k,mode='wrap')
    valsmap = convolve(vals,k,mode='wrap')
    valsmap = np.divide(valsmap,densitymap)

    return densitymap, valsmap

########################################################### Radial Basis Function Interpolator ##################################################
#Interpolates unstructured data as a convolution of a radial kernel with a ~histogram of data points
def gkde_core(ix,iy,ival,s,ixx,iyy,interplinear=True):
    #Inputs:
    #ix & iy        :   input xy positions
    #ival           :   value at the datapoint
    #s              :   sigma for gausian kernel
    #ixx & iyy      :   meshgrid of x and y positions to interpolate onto
    #interplinear   :   whether to calculate from subpixel positions or linear interpolate from pixel positions (former will be slower)


        if interplinear:
        #get kernel
        k = gkernel_float(s,np.ceil(3*s))
        #k = gkernel(s)
        #k = bkernel(s)

        #convert to interpolation grid
        xy = xytointerpgrid(ix,iy,ixv=ixx,iyv=iyy)
        #linear interpolation points
        #x
        xpos = np.floor(xy[:,0]).astype('int')
        xH = xy[:,0]-xpos
        xL = 1-xH
        #y
        ypos = np.floor(xy[:,1]).astype('int')
        yH = xy[:,1]-ypos
        yL = 1-yH
        
        #values
        vals = np.zeros_like(ixx).astype('float')
        #density
        dens = np.zeros_like(ixx).astype('float')
        
        #density & value ~histogram
        ind = np.vstack((ypos,xpos))
        vals += accumarray(ind, vals.shape, wt=xL*yL*ival)    #p00 value
        dens += accumarray(ind, dens.shape, wt=xL*yL)                 #p00 density
        ind = np.vstack((ypos+1,xpos))
        vals += accumarray(ind, vals.shape, wt=xL*yH*ival)    #p01
        dens += accumarray(ind, dens.shape, wt=xL*yH)                 #p01
        ind = np.vstack((ypos,xpos+1))
        vals += accumarray(ind, vals.shape, wt=xH*yL*ival)    #p10
        dens += accumarray(ind, dens.shape, wt=xH*yL)                 #p10
        ind = np.vstack((ypos+1,xpos+1))
        vals += accumarray(ind, vals.shape, wt=xH*yH*ival)    #p11
        dens += accumarray(ind, dens.shape, wt=xH*yH)                 #p11

        #convolve w/ kernel
        densitymap = convolve(dens,k,mode='nearest')
        valsmap = convolve(vals,k,mode='nearest')
        valsmap = np.divide(valsmap,densitymap)

    else:
        densitymap = np.zeros_like(ixx).astype('float')
        valsmap = np.zeros_like(ixx).astype('float')
        for xx in ixx:
            for yy in iyy:
                for i in np.arange(ix.size):
                    dx = ixx[yy,xx] - ix[i]
                    dy = iyy[yy,xx] - iy[i]
                    z = 1/(2*np.pi*s[0]*s[1]) * np.exp(-.5*dx**2/s[0]**2 - .5*dy**2/s[1]**2)
                    densitymap[yy,xx] = densitymap[yy,xx] + z
                    valsmap[yy,xx] = valsmap[yy,xx]+z*ival[i]
        valsmap = np.divide(valsmap,densitymap)
        
    return densitymap, valsmap
