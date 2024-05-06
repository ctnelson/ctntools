import numpy as np
from ctntools.Convolution.kde1D import kde1D

#Creates a radial profile via kde1D. This function is wrapper feed the GPU function kde1D 
def radKDE(inim, xy0 = None, rmax = None, rstp=1, xyscale=[1,1], **kwargs):
    ### Inputs ###
    #inim                   :       Input image
    #xy0    (optional)      :       center point (will default to center)
    #rmax   (optional)      :       Outer radius (data beyond is ignored)
    #rstp   (optional)      :       stepsize
    #xyscale(optional)      :       xy and y axis scales

    ### kwargs - passed thru to KDE ###   
    #s              (optional)      :       sigma of gaussian (or approximation thereof for bump function)
    #k_rdist        (optional)      :       kernel max radial distance
    #k              (optional)      :       user supplied kernel
    #kernel         (optional)      :       'Gauss' or 'Bump', type of Kernel to use 
    #method         (optional)      :       'round', 'interp', or 'exact', method to use. 'Round' rounds to closest interpolation grid, 'Interp' linearly interpolates kernel as two fractional kernels at boundary grid points, 'exact' (Gauss only) uses exact calculation (but total graph area will not precisely equal total weighted datapoints)
    #returnstat     (optional)      :       statistic calculated 'Mean', 'Max', or 'Min'
    #processor      (optional)      :       'cpu' or 'gpu', for large datasets may benefit to push onto gpu
    #sRescaleByX    (optional)      :       flag whether to consider s in units of x or #x-steps (index)
    #xSizeDefault   (optional)      :       if x is not provided, x will be a vector between min & max of pts_x with xSizeDefault steps
    #tpb            (optional)      :       threads per block (GPU only)

    ### Outputs ###
    #rv                     :       r-axis, the interpolation grid
    #distr                  :       Distribution
    #density                :       Density of datapoints
    #k                      :       smoothing kernel used
    #r                      :       radius

    rscalar=3   #kernel window scalar (does not significantly affect the width of the function, use s for that, this extends the tails)

    inim_sz = np.array(inim.shape)

    #apply defaults to optional parameters
    if xy0==None:
        xy0 = np.floor(inim_sz/2)
    if rmax==None:
        rmax = np.min(np.floor(np.array(inim_sz/2))) 

    #convert to polar coords
    xx,yy = np.meshgrid(np.arange(0,inim_sz[0]),np.arange(0,inim_sz[1]))
    xy0 = xy0
    dx = xx-xy0[0]
    dy = yy-xy0[1]
    dx = dx*xyscale[0]
    dy = dy*xyscale[1]
    r = ((dx)**2+(dy)**2)**.5

    #get r vector to interpolate on
    rv = np.arange(0,rmax,rstp)

    #distr, density, k = kde1D(r, inim, rv, kernel='Gauss', method='exact')
    distr, density, k = kde1D(r, inim, rv, **kwargs)
 
    return rv, distr, density, k, r
