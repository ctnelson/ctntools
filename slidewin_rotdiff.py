from numba import cuda
import math
import numpy as np

@cuda.jit
def angarray_rotdiff_core_gpu(inim, itheta, ixy0, ixx, iyy, irad, mode, result):
    #Which thread
  hh = cuda.grid(1)

  #is thread valid?
  if (hh >= itheta.size): 
    return

  ii=ixy0[0]
  jj=ixy0[1]

  result[hh] = 0
  step = 0

  #default index ranges
  x0 = np.int64(math.floor(ii-irad))
  x1 = np.int64(math.ceil(ii+irad))
  y0 = np.int64(math.floor(jj-irad))
  y1 = np.int64(math.ceil(jj+irad))

  #check for edges
  #x
  xlow = np.int64(math.ceil(irad - jj))
  xhigh = np.int64(math.floor(-inim.shape[1]+jj+irad+1))
  if (xlow>0) | (xhigh>0):
    if xlow>xhigh:
      x0 = x0+xlow
      x1 = x1-xlow
    else:
      x0 = x0+xhigh
      x1 = x1-xhigh

  #y
  ylow = np.int64(math.ceil(irad - ii))
  yhigh = np.int64(math.floor(-inim.shape[0]+ii+irad+1))
  if (ylow>0) | (yhigh>0):
    if ylow>yhigh:
      y0 = y0+ylow
      y1 = y1-ylow
    else:
      y0 = y0+yhigh
      y1 = y1-yhigh

  for xx in range(x1-x0):
    for yy in range(y1-y0):

      #polar
      rx = np.float32(xx)-ii
      ry = np.float32(yy)-jj
      r = ((rx)**2+(ry)**2)**.5
      rang = math.atan2(ry,rx)

      #limit to radius
      if r>irad:
          continue

      #coordinate transform
      xt = (r*math.cos(rang+itheta[hh]) + ii)
      yt = (r*math.sin(rang+itheta[hh]) + jj)

      #limit to in bounds
      if (xt<0) | (xt > inim.shape[1]) | (yt<0) | (yt > inim.shape[0]):
        continue

      z0 = inim[yy+y0,xx+x0]

      if mode=='bilinear':
        xL = (xt-math.floor(xt))
        xH = 1-xL
        yL = (yt-math.floor(yt))
        yH = 1-yL
        f00 = inim[np.int64(math.floor(yt)),np.int64(math.floor(xt))]
        f10 = inim[np.int64(math.floor(yt)),np.int64(math.ceil(xt))]
        f01 = inim[np.int64(math.ceil(yt)),np.int64(math.floor(xt))]
        f11 = inim[np.int64(math.ceil(yt)),np.int64(math.ceil(xt))]
        if math.isnan(f00) | math.isnan(f10) | math.isnan(f01) | math.isnan(f11):
          continue
        z1 = f00*xH*yH + f10*xL*yH + f01*xH*yL + f11*xL*yL

      elif mode=='nearest':
        z1 = inim[np.int64(np.round(yt)),np.int64(np.round(xt))]
        if math.isnan(z1):
          continue

      zdiff = z1-z0
      if zdiff>=0:
        result[hh] += zdiff
      else:
        result[hh] -= zdiff
      step += 1

  result[hh] = result[hh]/step

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
      iang = np.arange(0,np.pi,np.pi/180*2,dtype=np.float32)      #angles to check
  if ixy0 is None:
        ixy0 = np.floor(inim_sz/2).astype(np.float32)
  if irad is None:
        irad = np.min(np.floor(np.array(inim_sz/2))).astype(np.float32)

  brdr = 1
  inim = np.pad(inim,brdr,mode='edge')
  xx,yy = np.meshgrid(np.arange(0,inim_sz[0]+brdr*2,stride,dtype=np.float32),np.arange(0,inim_sz[1]+brdr*2,stride,dtype=np.float32))
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
