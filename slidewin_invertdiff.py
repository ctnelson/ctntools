from numba import cuda
import numpy as np

@cuda.jit
def swinvrtd_GPU(inim, winrng, ptnum, outval): 
    ii,jj = cuda.grid(2)

    if (ii >= inim.shape[0]) or (jj >= inim.shape[1]): 
      return

    #default index ranges
    x0 = np.int64(jj-winrng[1])
    x1 = np.int64(jj+winrng[0]+1)
    y0 = np.int64(ii-winrng[0])
    y1 = np.int64(ii+winrng[1]+1)

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
          isnanflag = (~math.isnan(inim[yy+y0,xx+x0,0]) and ~math.isnan(inim[y1-yy,x1-xx,0]))
        elif inim.ndim==2:
          isnanflag = (~math.isnan(inim[yy+y0,xx+x0]) and ~math.isnan(inim[yy+y0,xx+x0]))
        if isnanflag:
          if inim.ndim==3:
            lp_3 = inim.shape[2]
          elif inim.ndim==2:
            lp_3 = 1
          for l in range(lp_3):
            if inim.ndim==3:
              temp = inim[yy+y0,xx+x0,l]-inim[yy+y0,xx+x0,l]
            elif inim.ndim==2:
              temp = inim[yy+y0,xx+x0]-inim[yy+y0,xx+x0]
            if temp>0:
              outval[jj,ii] = outval[jj,ii]+temp
            else:
              outval[jj,ii] = outval[jj,ii]-temp
            step += 1
    outval[jj,ii] = outval[jj,ii]/step
    ptnum[jj,ii] = step

def swinvrtd_CPU(inim, winrng):
    from tqdm import tqdm
    inim_sz = np.array(inim.shape)
    result = np.ones_like(inim)*np.nan
    for xlp in tqdm(np.arange(inim_sz[1])):
        for ylp in np.arange(inim_sz[0]):
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
            ptnum = np.where(np.isfinite(temp))[0].size
    return result, ptnum

def slidewin_invertdiff(inimage, winrng, trygpu=True):

    result = np.ones(inimage.shape,np.float32)*np.nan
    ptnum = np.zeros(inimage.shape,np.int64)

    if trygpu:
        #try:
        blockdim = (32, 32)
        #print('Blocks dimensions:', blockdim)
        griddim = (result.shape[0] // blockdim[0] + 1, result.shape[1] // blockdim[1] + 1)
        #print('Grid dimensions:', griddim)
        swinvrtd_GPU[griddim, blockdim](inimage,winrng,ptnum,result)
        #except:
        #print('GPU Execution failed, fall back to cpu')
        #result = swinvrtd_CPU(inimage,winrng)
    else:
        result, ptnum = swinvrtd_CPU(inimage,winrng)

    return result, ptnum
