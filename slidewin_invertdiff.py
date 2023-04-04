from numba import cuda
import numpy as np

@cuda.jit
def swinvrtd_GPU(inim, winrng, outval): 
    ii,jj = cuda.grid(2)

    if (ii >= inim.shape[0]) or (jj >= inim.shape[0]): 
      return

    #window indices
    xtrim = np.max(np.array([winrng[0] - jj, -inim.shape[0]+jj+winrng[0]+1, 0]))      #check if indices extend over image edge
    ytrim = np.max(np.array([winrng[1] - ii, -inim.shape[1]+ii+winrng[1]+1, 0]))      
    x0 = jj-winrng[0]+xtrim
    if x0==0:
      x0f=None
    else:
      x0f=x0-1
    x1 = jj+winrng[0]+1-xtrim
    y0 = ii-winrng[1]+ytrim
    if y0==0:
      y0f=None
    else:
      y0f = y0-1
    y1 = ii+winrng[1]+1-ytrim

    #Comparison
    im1 = inim[y0:y1:1,x0:x1:1]
    im2 = inim[y1-1:y0f:-1,x1-1:x0f:-1]
    outval[ii,jj] = np.nanmean(np.abs(im1-im2).ravel())

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
            result[ylp,xlp] = np.nanmean(np.abs(im1-im2).ravel())
    return result

def slidewin_invertdiff(inimage, winrng, trygpu=True):

    result = np.ones(inimage.shape,np.float32)*np.nan

    if trygpu:
        #try:
        blockdim = (32, 32)
        #print('Blocks dimensions:', blockdim)
        griddim = (result.shape[0] // blockdim[0] + 1, result.shape[1] // blockdim[1] + 1)
        #print('Grid dimensions:', griddim)
        swinvrtd_GPU[griddim, blockdim](inimage,winrng,result)
        #except:
        #print('GPU Execution failed, fall back to cpu')
        #result = swinvrtd_CPU(inimage,winrng)
    else:
        result = swinvrtd_CPU(inimage,winrng)

    return result
