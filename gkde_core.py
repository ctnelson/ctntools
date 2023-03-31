import numpy as np
from scipy import signal

#Kernel Density Estimation
def gkde_core(ix,iy,ival,s,ixx,iyy,interplinear=True):
    #Inputs:
    #ix & iy        :   input xy positions
    #ival           :   value at the datapoint
    #s              :   sigma for gausian kernel
    #ixx & iyy      :   meshgrid of x and y positions to interpolate onto
    #interplinear   :   whether to calculate from subpixel positions or linear interpolate from pixel positions (former will be slower)


    if interplinear:
        def gkernel(s,wincutoff=3):
            if np.size(s)==1:
                s = [s,s]
            xv = np.arange(-np.ceil(s[0])*wincutoff,np.ceil(s[0])*wincutoff+1)
            yv = np.arange(-np.ceil(s[1])*wincutoff,np.ceil(s[1])*wincutoff+1)
            xx,yy = np.meshgrid(xv, yv)
            z =  1/(2*np.pi*s[0]*s[1]) * np.exp(-.5*xx**2/s[0]**2 - .5*yy**2/s[1]**2)
            return z

        #get guassian kernel
        k = gkernel(s)

        xv = ixx[0,:]
        yv = iyy[:,0]

        #estimate each datapoint as sum of 4 at integer positions
        stpsz = xv[1]-xv[0]
        xpos = np.floor((ix-xv[0])/stpsz+1).astype('int')     #get first position
        xh = (ix-(xv[xpos-1]))/stpsz                          #fraction of xpos
        xl = 1-xh                                             #fraction of xpos+1
        stpsz = yv[1]-yv[0]
        ypos = np.floor((iy-yv[0])/stpsz+1).astype('int')
        yh = (iy-(yv[ypos-1]))/stpsz
        yl = 1-yh

        #values
        vals = np.zeros_like(ixx).astype('float')
        vals[ypos-1,xpos-1] = vals[ypos-1,xpos-1]   + xl*yl*ival    #p00
        vals[ypos,xpos-1]   = vals[ypos,xpos-1]     + xl*yh*ival    #p10
        vals[ypos-1,xpos]   = vals[ypos-1,xpos]     + xh*yl*ival    #p01
        vals[ypos,xpos]     = vals[ypos,xpos]       + xh*yh*ival    #p11

        #density
        dens = np.zeros_like(ixx).astype('float')
        dens[ypos-1,xpos-1] = dens[ypos-1,xpos-1]   + xl*yl    #p00
        dens[ypos,xpos-1]   = dens[ypos,xpos-1]     + xl*yh    #p10
        dens[ypos-1,xpos]   = dens[ypos-1,xpos]     + xh*yl    #p01
        dens[ypos,xpos]     = dens[ypos,xpos]       + xh*yh    #p11

        #convolve w/ kernel
        densitymap = signal.convolve2d(dens,k,mode='same')
        valsmap = signal.convolve2d(vals,k,mode='same')
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
