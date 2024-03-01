from ctntools.BaseSupportFunctions.kernels import bKernel1D, gKernel1D
import numpy as np
#from scipy.ndimage import convolve
from numba import cuda
import math

@cuda.jit
#gaussian exact GPU
def gkde1Dexact_gpu(x, wt, xgrd, k, returnstat, result, dens):
    #Inputs
    #x          :       x position of elements
    #wt         :       weight / value of element
    #xgrd       :       interpolation grid
    #k          :       gaussian kernel parameters [sigma, rdist, zoffset, zscalar]
    #returnstat :       0=Mean, 1=Min, 2=Max
    ### Outputs (are also inputs for gpu functions) ###
    #result     :       the output weighted density
    #dens       :       the output density

    #Check boundaries
    ii = cuda.grid(1)
    if ii>result.size-1:
        return
    #initial values
    if returnstat==0:
        result[ii] = 0.0
        dens[ii] = 0
    elif returnstat==1:
        result[ii] = -np.inf
        dens[ii] = -np.inf
    elif returnstat==2:
        result[ii] = -np.inf
        dens[ii] = -np.inf
    #Scan through elements
    for i in range(x.size):
        xdel = x[i]-xgrd[ii]
        if xdel<0:
            xdel = -xdel
        if xdel<=k[1]:
            if returnstat==0:
                ydel = k[3]*(math.exp(-.5*(xdel**2/k[0]**2))+k[2])
                result[ii]+= ydel*wt[i]
                dens[ii]+=ydel
            elif returnstat==1:
                ydel = math.exp(-.5*(xdel**2/k[0]**2))
                if (1-wt[i])*ydel > result[ii]:
                    #result[ii]=ydel*wt[i]
                    #dens[ii]=ydel
                    result[ii]=(1-wt[i])*ydel
                    dens[ii]=1
            elif returnstat==2:
                ydel = math.exp(-.5*(xdel**2/k[0]**2))
                if wt[i]*ydel > result[ii]:
                    #result[ii]=ydel*wt[i]
                    #dens[ii]=ydel
                    result[ii]=wt[i]*ydel
                    dens[ii]=1
    if returnstat==1:
        result[ii]=-(result[ii]-1)        
    return

@cuda.jit
#gaussian convolution GPU
def gkde1Dconv_gpu(x, wt, xgrd, k, returnstat, method, result, dens):
    #Inputs
    #x          :       x position of elements (supply this input in units of stepsize, i.e. xgrid is integers)
    #wt         :       weight / value of element
    #xgrd       :       interpolation grid (supply this input in units of stepsize, i.e. this should be a linear integer array)
    #k          :       gaussian kernel
    #returnstat :       0=Mean, 1=Min, 2=Max
    #method     :       0=interp, 1=round. Datapoint value/density is assigned to nearest gridpoint(s) via this method
    ### Outputs (are also inputs for gpu functions) ###
    #result     :       the output weighted density
    #dens       :       the output density

    #Check boundaries
    ii = cuda.grid(1)
    if ii>result.size-1:
        return
    #initial values
    if returnstat==0:
        result[ii] = 0.0
        dens[ii] = 0
    elif returnstat==1:
        result[ii] = -np.inf
        dens[ii] = -np.inf
    elif returnstat==2:
        result[ii] = -np.inf
        dens[ii] = -np.inf
    #Scan through elements
    krad = np.int64(round((k.size-1)/2))
    for i in range(x.size):
        if method==0:   #interpolate
            xdelL = math.floor(x[i])
            xdelH = math.ceil(x[i])
            if (xdelL==xdelH):        #check for condition x integer
                xdelH=xdelL+1
            xdelL = np.int64(xgrd[ii]-xdelL)       #distance to leftmost x interp position
            xdelH = np.int64(xgrd[ii]-xdelH)       #distance to rightmost x interp position
            wtH = math.modf(x[i])[0]                        #weight of left position
            #Low position
            if (xdelL<=krad) & (xdelL>=-krad):        #check if within distance of kernel
                xdelL += krad
                if returnstat==0:
                    result[ii]+= k[xdelL]*wt[i]*(1-wtH)
                    dens[ii]+=k[xdelL]*(1-wtH)
            #High position
            if (xdelH<=krad) & (xdelH>=-krad):        #check if within distance of kernel
                xdelH += krad
                if returnstat==0:
                    result[ii]+= k[xdelH]*wt[i]*wtH
                    dens[ii]+=k[xdelH]*wtH
        elif method==1:   #round
            xdel = np.int64(xgrd[ii]-round(x[i]))   #distance to round(x position)
            if (xdel<=krad) & (xdel>=-krad):        #check if within distance of kernel
                xdel += krad
                if returnstat==0:
                    result[ii]+= k[xdel]*wt[i]
                    dens[ii]+=k[xdel]           
    return

def kde1D(pts_x, pts_wt=None, x=None, s=1, k_rdist=None, k=None, kernel='Gauss', method='interp', returnstat='Mean', processor='gpu', sRescaleByX=True, xSizeDefault=100, tpb=32):
    #calculates the distribution of pts_x by convolving pts_x with a kernel (e.g. gaussian). Functions like blurring/smoothing of a histogram.
    #inputs        
    #pts_x                          :       data points to calculate distribution
    #pts_wt         (optional)      :       weights / values (default is 1, the point density)
    #x              (optional)      :       linear interpolation vector (defaults to min -> max value in 'xSizeDefault' #steps).
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
    
    #Returns
    #y          :       distribution
    #dens       :       distribution density
    #k          :       kernel used

    #Some Parameters
    k_rdistDefaultScalar    = 3                 #if not provided, kernel rdist will be this scalar times s
    k_rdistMin              = 1                 #minimum k_rdist
    k_BumpHWHMScalar        = 2*np.log(2)       #bump function broadness input is HWHM, this ~converts from the gaussian s=sigma
    rstat_enum = ['Mean', 'Min', 'Max']         #used to convert returnstat string -> index (i.e. 'Mean'=0,...) for gpu functions
    method_enum = ['interp', 'round']
    
    #x values
    pts_x = pts_x.astype(np.float32).ravel()
    s = np.array(s,dtype=np.float32)

    #weights
    if pts_wt is None:
        pts_wt = np.ones_like(pts_x,dtype=np.float32)
    else:
        pts_wt = pts_wt.astype(np.float32).ravel()

    #x: interpolation grid
    #if not provided, create it
    if x is None:
        x = np.linspace(np.nanmin(pts_x.ravel()),np.nanmax(pts_x.ravel()),xSizeDefault,dtype=np.float32)
    else:
        x = np.array(x,dtype=np.float32)
        assert(x.size>0)
    xstp = x[1]-x[0]
    
    #Initialize
    scalar=1.0      #scalar applied to kernel
    k_offset=0.0    #offset applied to kernel
    #outputs
    y = np.ones_like(x,dtype=np.float32)*np.nan         #weighted distribution
    dens = np.ones_like(x,dtype=np.float32)*np.nan      #density (wt=1 distribution)

    #kernel parameters
    if sRescaleByX & (method!='exact'):       #scale s in units of x? If not, s will be in units of index
        s = s/xstp
    if k_rdist is None:
        k_rdist = s*k_rdistDefaultScalar
    else:
        k_rdist = k_rdist/xstp
    k_rdist = np.int32(np.ceil(k_rdist))    #round up to integer
    k_rdist = np.max([k_rdist,k_rdistMin])  #ensure larger than min value

    #create kernel array
    if (k is None) & (method!='exact'):
        #print('create kernel array '+kernel)
        #get kernel
        if kernel=='Bump':
            #print('Bump')
            k = bKernel1D(k_rdist,hwhm=s*k_BumpHWHMScalar,normalize=False)
            scalar = 1/np.sum(k)
            k_offset=0.0
        elif kernel=='Gauss':
            #print('Gaussian')
            k = gKernel1D(s,rdist=k_rdist,normalize=False)
            k_offset = -np.min(k)
            scalar = 1/np.sum(k+k_offset)
        else:
            raise ValueError('Unknown Kernel')
        k = (k+k_offset)*scalar
    elif not (k is None):
        #print('Kernel provided as input')
        kernel='User Kernel'

    #if exact, loop needs to perform distance calculations, so pass s & k_rdist
    if method == 'exact':
        scalar = 1/(2*np.pi)**.5/s  #normalization scalar for gaussian distribution
        if processor=='gpu':
            #print('exact gpu')
            #transfer to gpu
            d_x = cuda.to_device(pts_x)
            d_wt = cuda.to_device(pts_wt)
            d_xgrid = cuda.to_device(x)
            d_y = cuda.to_device(y)
            d_dens = cuda.to_device(dens)
            #Kernel memory 
            blockspergrid = np.ceil(x.size / tpb).astype('int')
            #Execuate based on kernel type
            #print(kernel)
            if kernel=='Bump':
                #print('exact Bump GPU')
                raise ValueError('method "exact" not implemented for Bump Function, switch to Gaussian kernel')
            elif kernel=='Gauss':
                #print('exact Gaussian GPU')
                kparms = np.array([s,k_rdist,k_offset,scalar],dtype=np.float32)
                d_k = cuda.to_device(kparms)
                gkde1Dexact_gpu[blockspergrid, tpb](d_x, d_wt, d_xgrid, d_k, rstat_enum.index(returnstat), d_y, d_dens)
            #return to cpu
            y = d_y.copy_to_host()
            dens = d_dens.copy_to_host()
            #divide by density
            #y = y/dens
        else:
            raise ValueError('exact cpu not yet coded')

    #else convolve
    else:
        x = np.round(x/xstp).astype('int64')
        pts_x = pts_x/xstp
        if processor=='gpu':
            #print(kernel+'_convolution_'+method+'_'+processor)
            #transfer to gpu
            d_x = cuda.to_device(pts_x)
            d_wt = cuda.to_device(pts_wt)
            d_xgrid = cuda.to_device(x)
            d_k = cuda.to_device(k)
            d_y = cuda.to_device(y)
            d_dens = cuda.to_device(dens)
            #Kernel memory 
            blockspergrid = np.ceil(x.size / tpb).astype('int')
            #Execute
            gkde1Dconv_gpu[blockspergrid, tpb](d_x, d_wt, d_xgrid, d_k, rstat_enum.index(returnstat), method_enum.index(method), d_y, d_dens)
            #return to cpu
            y = d_y.copy_to_host()
            dens = d_dens.copy_to_host()
            #divide by density
            #y = y/dens
        else:
            #print(kernel+'_convolution_'+method+'_'+processor)
            raise ValueError('CPU not yet coded')
        
    return y, dens, k
