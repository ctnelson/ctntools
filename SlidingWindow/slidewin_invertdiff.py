from numba import cuda
import math
import numpy as np

@cuda.jit
def swinvrtd_GPU(inim, winrng, searchwin, wincnt, winmean, outval): 
    ii,jj = cuda.grid(2)

    #if (ii >= inim.shape[0]-1) or (ii < 1) or (jj >= inim.shape[1]-1) or (jj < 1):
    if (ii >= searchwin[3]) or (ii < searchwin[2]) or (jj >= searchwin[1]) or (jj < searchwin[0]):
      return

    outval[ii,jj] = 0
    step = 0
    winmean[ii,jj] = 0

    #default index ranges
    x0 = np.int64(jj-winrng[0])
    x1 = np.int64(jj+winrng[0])
    y0 = np.int64(ii-winrng[1])
    y1 = np.int64(ii+winrng[1])

    #check for edges
    #x
    xlow = winrng[0] - jj
    xhigh = -inim.shape[1]+jj+winrng[0]+1
    if (xlow>0) | (xhigh>0):
      if xlow>xhigh:
        x0 = x0+xlow
        x1 = x1-xlow
      else:
        x0 = x0+xhigh
        x1 = x1-xhigh

    #y
    ylow = winrng[1] - ii
    yhigh = -inim.shape[0]+ii+winrng[1]+1
    if (ylow>0) | (yhigh>0):
      if ylow>yhigh:
        y0 = y0+ylow
        y1 = y1-ylow
      else:
        y0 = y0+yhigh
        y1 = y1-yhigh

    for xx in range(x1-x0):
      for yy in range(y1-y0):
        isnanflag = False
        if inim.ndim==3:
          isnanflag = (~math.isnan(inim[yy+y0,xx+x0,0]) and ~math.isnan(inim[y1-yy-1,x1-xx-1,0]))
        elif inim.ndim==2:
          isnanflag = (~math.isnan(inim[yy+y0,xx+x0]) and ~math.isnan(inim[y1-yy-1,x1-xx-1]))
        if isnanflag:
          if inim.ndim==3:
            lp_3 = inim.shape[2]
          elif inim.ndim==2:
            lp_3 = 1
          for l in range(lp_3):
            temp = 0
            if inim.ndim==3:
              temp = inim[y1-yy-1,x1-xx-1,l]-inim[yy+y0,xx+x0,l]
              winmean[ii,jj] += inim[yy+y0,xx+x0,l]
            elif inim.ndim==2:
              temp = inim[y1-yy-1,x1-xx-1]-inim[yy+y0,xx+x0]
              winmean[ii,jj] += inim[yy+y0,xx+x0]
            if temp>0:
              outval[ii,jj] = outval[ii,jj]+temp
            else:
              outval[ii,jj] = outval[ii,jj]-temp
            step += 1
    winmean[ii,jj] = winmean[ii,jj]/step
    outval[ii,jj] = outval[ii,jj]/step
    wincnt[ii,jj] = step
    

def swinvrtd_CPU(inim, winrng, searchwin):
    from tqdm import tqdm
    inim_sz = np.array(inim.shape)
    result = np.ones_like(inim)*np.nan
    #for xlp in tqdm(np.arange(inim_sz[1])):
    for xlp in tqdm(np.arange(searchwin[0],searchwin[1]+1)):
        #for ylp in np.arange(inim_sz[0]):
        for ylp in np.arange(searchwin[2],searchwin[3]+1):
            xtrim = np.max(np.array([winrng[0] - xlp, -inim_sz[0]+xlp+winrng[0]+1, 0]))      #a check if indices extend over image edge
            ytrim = np.max(np.array([winrng[1] - ylp, -inim_sz[1]+ylp+winrng[1]+1, 0]))      #a check if indices extend over image edge
            xx = np.arange(xlp-winrng[0]+xtrim,xlp+winrng[0]+1-xtrim)
            yy = np.arange(ylp-winrng[1]+ytrim,ylp+winrng[1]+1-ytrim)
            x0 = xlp-winrng[0]+xtrim
            if x0==0:
                x0f=None
            else:
                x0f=x0-1
            x1 = xlp+winrng[0]+1-xtrim
            y0 = ylp-winrng[1]+ytrim
            if y0==0:
                y0f=None
            else:
                y0f = y0-1
            y1 = ylp+winrng[1]+1-ytrim
            im1 = inim[y0:y1:1,x0:x1:1]
            im2 = inim[y1-1:y0f:-1,x1-1:x0f:-1]
            temp = np.abs(im1-im2).ravel()
            result[ylp,xlp] = np.nanmean(temp)
            wincnt = np.where(np.isfinite(temp))[0].size
            winmean = np.nanmean(im1.ravel())
    return result, wincnt, winmean

def slidewin_invertdiff(inimage, winrng, trygpu=True, searchwin = None):

    if searchwin is None:
        searchwin = np.array([1, inimage.shape[1]-1, 1, inimage.shape[0]-1], dtype=np.int16)
    
    result = np.ones_like(inimage,np.float32)*np.nan
    winmean = np.ones_like(inimage,np.float32)*np.nan
    wincnt = np.zeros(inimage.shape,np.int64)

    if trygpu:
        #try:
        blockdim = (32, 32)
        print('Blocks dimensions:', blockdim)
        griddim = (result.shape[0] // blockdim[0] + 1, result.shape[1] // blockdim[1] + 1)
        print('Grid dimensions:', griddim)
        swinvrtd_GPU[griddim, blockdim](inimage,winrng,searchwin,wincnt,winmean,result)
        #except:
        #print('GPU Execution failed, fall back to cpu')
        #result = swinvrtd_CPU(inimage,winrng)
    else:
        result, wincnt, winmean = swinvrtd_CPU(inimage,winrng,searchwin)

    return result, wincnt, winmean

##################################################################################################################################
#########################################    Zero Normalized Cross Correlation     ###############################################
##################################################################################################################################
@cuda.jit
def swinvrt_ccorr_GPU(inim, winrng, searchwin, result_counts, result_mean, result_var, result_ccorr): 
    ii,jj = cuda.grid(2)

    #if (ii >= inim.shape[0]-1) or (ii < 1) or (jj >= inim.shape[1]-1) or (jj < 1):
    if (ii >= searchwin[3]) or (ii < searchwin[2]) or (jj >= searchwin[1]) or (jj < searchwin[0]):
      return

    result_counts[ii,jj] = 0
    result_mean[ii,jj] = 0
    result_var[ii,jj] = 0
    result_ccorr[ii,jj] = 0
    step = 0

    #default index ranges
    x0 = np.int32(jj-winrng[0])
    x1 = np.int32(jj+winrng[0])
    y0 = np.int32(ii-winrng[1])
    y1 = np.int32(ii+winrng[1])

    #check for edges
    #x
    xlow = winrng[0] - jj
    xhigh = -inim.shape[1]+jj+winrng[0]+1
    if (xlow>0) | (xhigh>0):
      if xlow>xhigh:
        x0 = x0+xlow
        x1 = x1-xlow
      else:
        x0 = x0+xhigh
        x1 = x1-xhigh

    #y
    ylow = winrng[1] - ii
    yhigh = -inim.shape[0]+ii+winrng[1]+1
    if (ylow>0) | (yhigh>0):
      if ylow>yhigh:
        y0 = y0+ylow
        y1 = y1-ylow
      else:
        y0 = y0+yhigh
        y1 = y1-yhigh

    if inim.ndim==3:
      lp_3 = inim.shape[2]
    elif inim.ndim==2:
      lp_3 = 1
        
    #Get window Mean
    for xx in range(x1-x0):
      for yy in range(y1-y0):
        isnanflag = False
        if inim.ndim==3:
          isnanflag = (~math.isnan(inim[yy+y0,xx+x0,0]) and ~math.isnan(inim[y1-yy-1,x1-xx-1,0]))
        elif inim.ndim==2:
          isnanflag = (~math.isnan(inim[yy+y0,xx+x0]) and ~math.isnan(inim[y1-yy-1,x1-xx-1]))
        if isnanflag:
          for l in range(lp_3):
            if inim.ndim==3:
              result_mean[ii,jj,0] += inim[yy+y0,xx+x0,l]
              result_mean[ii,jj,1] += inim[y1-yy-1,x1-xx-1,l]
            elif inim.ndim==2:
              result_mean[ii,jj,0] += inim[yy+y0,xx+x0]
              result_mean[ii,jj,1] += inim[y1-yy-1,x1-xx-1]
          step += 1
    result_mean[ii,jj,0] = result_mean[ii,jj,0]/step/lp_3
    result_mean[ii,jj,1] = result_mean[ii,jj,1]/step/lp_3
    result_counts[ii,jj] = step
    
    #Get window standard deviation
    for xx in range(x1-x0):
      for yy in range(y1-y0):
        isnanflag = False
        if inim.ndim==3:
          isnanflag = (~math.isnan(inim[yy+y0,xx+x0,0]) and ~math.isnan(inim[y1-yy-1,x1-xx-1,0]))
        elif inim.ndim==2:
          isnanflag = (~math.isnan(inim[yy+y0,xx+x0]) and ~math.isnan(inim[y1-yy-1,x1-xx-1]))
        if isnanflag:
          for l in range(lp_3):
            if inim.ndim==3:
              result_var[ii,jj,0] += (inim[yy+y0,xx+x0,l]-result_mean[ii,jj,0])**2
              result_varn[ii,jj,1] += (inim[y1-yy-1,x1-xx-1,l]-result_mean[ii,jj,1])**2
            elif inim.ndim==2:
              result_var[ii,jj,0] += (inim[yy+y0,xx+x0]-result_mean[ii,jj,0])**2
              result_var[ii,jj,1] += (inim[y1-yy-1,x1-xx-1]-result_mean[ii,jj,1])**2
    result_var[ii,jj,0] = result_var[ii,jj,0]/step/lp_3
    result_var[ii,jj,1] = result_var[ii,jj,1]/step/lp_3
    
    #The normalized cross correlation    
    for xx in range(x1-x0):
      for yy in range(y1-y0):
        isnanflag = False
        if inim.ndim==3:
          isnanflag = (~math.isnan(inim[yy+y0,xx+x0,0]) and ~math.isnan(inim[y1-yy-1,x1-xx-1,0]))
        elif inim.ndim==2:
          isnanflag = (~math.isnan(inim[yy+y0,xx+x0]) and ~math.isnan(inim[y1-yy-1,x1-xx-1]))
        if isnanflag:
          for l in range(lp_3):
            if inim.ndim==3:
              result_ccorr[ii,jj] += (inim[y1-yy-1,x1-xx-1,l]-result_mean[ii,jj,1])*(inim[yy+y0,xx+x0,l]-result_mean[ii,jj,0])/(result_var[ii,jj,0]**.5*result_var[ii,jj,1]**.5)
            elif inim.ndim==2:
              result_ccorr[ii,jj] += (inim[y1-yy-1,x1-xx-1]-result_mean[ii,jj,1])*(inim[yy+y0,xx+x0]-result_mean[ii,jj,0])/(result_var[ii,jj,0]**.5*result_var[ii,jj,1]**.5)
    result_ccorr[ii,jj] = result_ccorr[ii,jj]/step/lp_3
    

def swinvrt_ccorr_CPU(inim, winrng, searchwin):
    from tqdm import tqdm
    inim_sz = np.array(inim.shape)
    result = np.ones_like(inim)*np.nan
    #for xlp in tqdm(np.arange(inim_sz[1])):
    for xlp in tqdm(np.arange(searchwin[0],searchwin[1]+1)):
        #for ylp in np.arange(inim_sz[0]):
        for ylp in np.arange(searchwin[2],searchwin[3]+1):
            xtrim = np.max(np.array([winrng[0] - xlp, -inim_sz[0]+xlp+winrng[0]+1, 0]))      #a check if indices extend over image edge
            ytrim = np.max(np.array([winrng[1] - ylp, -inim_sz[1]+ylp+winrng[1]+1, 0]))      #a check if indices extend over image edge
            xx = np.arange(xlp-winrng[0]+xtrim,xlp+winrng[0]+1-xtrim)
            yy = np.arange(ylp-winrng[1]+ytrim,ylp+winrng[1]+1-ytrim)
            x0 = xlp-winrng[0]+xtrim
            if x0==0:
                x0f=None
            else:
                x0f=x0-1
            x1 = xlp+winrng[0]+1-xtrim
            y0 = ylp-winrng[1]+ytrim
            if y0==0:
                y0f=None
            else:
                y0f = y0-1
            y1 = ylp+winrng[1]+1-ytrim
            im1 = inim[y0:y1:1,x0:x1:1]
            im2 = inim[y1-1:y0f:-1,x1-1:x0f:-1]
            temp = np.abs(im1-im2).ravel()
            result[ylp,xlp] = np.nanmean(temp)
            wincnt = np.where(np.isfinite(temp))[0].size
            winmean = np.nanmean(im1.ravel())
    return result, wincnt, winmean

def slidewin_invertccorr(inimage, winrng, trygpu=True, searchwin=None):
    #Inputs:
    #inimage                :     Input image
    #winrng                 :     Sliding window half-width
    #trygpu(optional)       :     attemp to execute first on gpu
    #searchwin              :     manually dictate the search window [xmin, xmax, ymin, ymax] (make sure < image bounds)

    #Outputs:
    #invccorr               :     sliding window cross correlation
    #invcounts              :     pixel-count of window size per window location
    #invmean                :     mean value of sliding windows
    #invvar                 :     variance of sliding windows

    if searchwin is None:
        searchwin = np.array([1, inimage.shape[1]-1, 1, inimage.shape[0]-1], dtype=np.int16)
    
    invcounts = np.ones_like(inimage,np.int32)*-1
    invccorr = np.ones_like(inimage,np.float32)*np.nan
    invmean = np.ones((inimage.shape[0],inimage.shape[1],2),dtype=np.float32)*np.nan
    invvar = np.ones((inimage.shape[0],inimage.shape[1],2),dtype=np.float32)*np.nan
    
    if trygpu:
        #try:
        blockdim = (32, 32)
        print('Blocks dimensions:', blockdim)
        griddim = (invccorr.shape[0] // blockdim[0] + 1, invccorr.shape[1] // blockdim[1] + 1)
        print('Grid dimensions:', griddim)
        swinvrt_ccorr_GPU[griddim, blockdim](inimage, winrng, searchwin, invcounts, invmean, invvar, invccorr) 
        #except:
        #print('GPU Execution failed, fall back to cpu')
        #result = swinvrtd_CPU(inimage,winrng)
    else:
        invccorr, invcounts, invmean = swinvrt_ccorr_CPU(inimage,winrng,searchwin)

    return invccorr, invcounts, invmean, invvar
