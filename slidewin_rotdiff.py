from numba import cuda
import math
import numpy as np

@cuda.jit
def angarray_rotdiff_core_gpu(inim, itheta, ixy0, ixx, iyy, irad, mode, result):
    #Which thread
    ii = cuda.grid(1)

    #is thread valid?
    if (ii >= itheta.size): 
      return

    #polar
    rx = ixx.ravel()-ixy0[0]
    ry = iyy.ravel()-ixy0[1]
    r = ((rx)**2+(ry)**2)**.5
    rang = np.arctan2(ry,rx)

    #limit to radius
    ind = np.where(r<=irad)[0]

    #coordinate transform
    x1 = (r[ind]*np.cos(rang[ind]+itheta[i]) + ixy0[0]).ravel()
    y1 = (r[ind]*np.sin(rang[ind]+itheta[i]) + ixy0[1]).ravel()

    #limit to in bounds
    ind2 = np.where((x1>=0) & (x1<=inim.shape[1]) & (y1>=0) & (y1<=inim.shape[0]))[0].astype('int')

    if ind2.size==0:
       result[ii] = np.nan
       return
    else:
      ind = ind[ind2]

    #values
    x1 = x1[ind2]
    y1 = y1[ind2]
    z0 = inim.ravel()[ind]

    if mode=='bilinear':
      xL = (x1-np.floor(x1))
      xH = 1-xL
      yL = (y1-np.floor(y1))
      yH = 1-yL
      f00 = inim[np.floor(y1).astype('int'),np.floor(x1).astype('int')]
      f10 = inim[np.floor(y1).astype('int'),np.ceil(x1).astype('int')]
      f01 = inim[np.ceil(y1).astype('int'),np.floor(x1).astype('int')]
      f11 = inim[np.ceil(y1).astype('int'),np.ceil(x1).astype('int')]
      z1 = f00*xH*yH + f10*xL*yH + f01*xH*yL + f11*xL*yL

    elif mode=='nearest':
      z1 = inim[np.round(y1).astype('int'),np.round(x1).astype('int')]

    else:
      raise RaiseValueError('mode parameter not recognized')

    result[ii] = np.mean(np.abs(z1.ravel()-z0.ravel()))

def angarray_rotdiff_core_cpu(inim, itheta, ixy0, ixx, iyy, irad, mode, result):
    print('CPU run')
    result = np.nan
    return result


def angarray_rotdiff(inim, stride=1, ixy0=None, irad=None, iang=None, mode = 'bilinear', trygpu=True):
  #Inputs:
  #inim                   :     Input image
  #stride(optional)       :     downsampling of inim (will default to 1, all points)
  #ixy0(optional)         :     Center point (will default to image center)
  #irad(optional)         :     Maximum radius to consider (will default to half image)
  #iang(optional)         :     Array of rotation angles (radians) to test (will default to 0->180deg, 2 deg steps)
  #mode(optional)         :     interpolation method, 'nearest', or 'bilinear'
  #plotoutput(optional)   :     Plot results

  #Outputs:
  #rotdif                :     rotation mean abs difference
  
  inim_sz = np.array(inim.shape)
  if iang is None:
      print('no input angles, fallback to default')
      iang = np.arange(0,np.pi,np.pi/180*2)      #angles to check
  if ixy0 is None:
        ixy0 = np.floor(inim_sz/2)
  if irad is None:
        irad = np.min(np.floor(np.array(inim_sz/2)))

  brdr = 1
  inim = np.pad(inim,brdr,mode='edge')
  xx,yy = np.meshgrid(np.arange(0,inim_sz[0]+brdr*2,stride,dtype=np.int64),np.arange(0,inim_sz[1]+brdr*2,stride,dtype=np.int64))
  xy0 = ixy0+brdr

  rotdif = np.ones((iang.size,),np.float32)*np.nan

  if trygpu:
    #try:
    threadsperblock = 32
    blockspergrid = (iang.size + (threadsperblock - 1)) // threadsperblock
    angarray_rotdiff_core_gpu[blockspergrid, threadsperblock](inim, iang, xy0, xx, yy, irad, mode, rotdif)
    #except:
    #print('GPU Execution failed, fall back to cpu')
    #rotdif = angarray_rotdiff_core_cpu(inim, iang, xy0, xx, yy, irad, mode, rotdif)
  else:
    rotdif = angarray_rotdiff_core_cpu(inim, iang, xy0, xx, yy, irad, mode, rotdif)

  return rotdif
