from numba import cuda
import math
import numpy as np
from scipy.signal import find_peaks

########################################## Array of Angles (fixed center point) ###############################################
@cuda.jit
def angarray_rotdiff_core_gpu(inim, itheta, ixy0, irad, mode, result):
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

      if mode==0:
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

      else:
        z1 = inim[np.int64(round(yt)),np.int64(round(xt))]
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
    from tqdm import tqdm
    for ii in tqdm(range(itheta.size)):
        #polar
        rx = ixx-ixy0[0]
        ry = iyy-ixy0[1]
        rx = rx.ravel()
        ry = ry.ravel()
        r = ((rx)**2+(ry)**2)**.5
        rang = np.arctan2(ry,rx)

        #limit to radius
        ind = np.where(r<=irad)[0]

        #coordinate transform
        x1 = (r[ind]*np.cos(rang[ind]+itheta[ii]) + ixy0[0]).ravel()
        y1 = (r[ind]*np.sin(rang[ind]+itheta[ii]) + ixy0[1]).ravel()

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
          print('mode parameter not recognized')

        result[ii] = np.mean(np.abs(z1.ravel()-z0.ravel()))

#Calculates the mean abs image difference of an image with a rotation transformed version of itself. 
#This function handles the case of an array of angles applied to a single rotation center 
def angarray_rotdiff(inim, stride=1, ixy0=None, irad=None, iang=None, mode = 0, inax=None, trygpu=True):
  #Inputs:
  #inim                   :     Input image
  #stride(optional)       :     downsampling of inim (will default to 1, all points)
  #ixy0(optional)         :     Center point (will default to image center)
  #irad(optional)         :     Maximum radius to consider (will default to half image)
  #iang(optional)         :     Array of rotation angles (radians) to test (will default to 0->180deg, 2 deg steps)
  #mode(optional)         :     interpolation method, 0 ='bilinear', anything else uses 'nearest'
  #inax(optional)         :     plots results if given an axis
  #trygpu(optional)       :     attemp to execute first on gpu

  #Outputs:
  #rotdif                :     rotation mean abs difference
  #pks                   :     peak similarity angle
  
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
    try:
        threadsperblock = 32
        blockspergrid = (iang.size + (threadsperblock - 1)) // threadsperblock
        angarray_rotdiff_core_gpu[blockspergrid, threadsperblock](inim, iang, xy0, irad, mode, rotdif)
    except:
        print('GPU Execution failed, fall back to cpu')
        angarray_rotdiff_core_cpu(inim, iang, xy0, xx, yy, irad, mode, rotdif)
  else:
    angarray_rotdiff_core_cpu(inim, iang, xy0, xx, yy, irad, mode, rotdif)

  #normalize & find peaks
  rotdif = rotdif/np.nanmax(rotdif)
  rotdif = 1-rotdif
  pk_ind, ht = find_peaks(rotdif,height=.2,width=5)
  ht = ht['peak_heights']

  #outputs
  pks = np.empty((pk_ind.size,2))
  pks[:,0] = iang[pk_ind]
  pks[:,1] = ht

  if ~(inax is None):
    inax.plot(np.rad2deg(iang),rotdif,'-k')
    inax.scatter(np.rad2deg(pks[:,0]),pks[:,1],s=100,c='r',marker='o')
    for i in np.arange(pks.shape[0]):
      inax.text(np.rad2deg(pks[i,0]),pks[i,1],str(i)+':'+str(pks[i,1]),va='bottom')
    inax.set_xlim([0,180])
    inax.set_xlabel('Angle (Degrees)')
    inax.set_ylabel('Self Similarity')
  
  return np.vstack((iang,rotdif)), pks   

########################################## Sliding Window (moving center point) ###############################################

@cuda.jit
def slidewin_rotdiff_core_gpu(inim, itheta, irad, mode, result):
  #Which thread
  ii,jj = cuda.grid(2)

  #is thread valid?
  if (ii >= inim.shape[0]-1) or (ii < 1) or (jj >= inim.shape[1]-1) or (jj < 1): 
      return

  result[ii,jj] = 0
  step = 0

  #default index ranges
  x0 = np.int32(ii-irad)
  x1 = np.int32(ii+irad)
  y0 = np.int32(jj-irad)
  y1 = np.int32(jj+irad)

  #check for edges
  #x
  xlow = irad - jj
  xhigh = -inim.shape[1]+jj+irad+1
  if (xlow>0) | (xhigh>0):
    if xlow>xhigh:
      x0 = x0+xlow
      x1 = x1-xlow
    else:
      x0 = x0+xhigh
      x1 = x1-xhigh

  #y
  ylow = irad - ii
  yhigh = -inim.shape[0]+ii+irad+1
  if (ylow>0) | (yhigh>0):
    if ylow>yhigh:
      y0 = y0+ylow
      y1 = y1-ylow
    else:
      y0 = y0+yhigh
      y1 = y1-yhigh

  for xx in range(x0,x1):
    for yy in range(y0,y1):
      
      #polar
      rx = np.float32(xx-ii)
      ry = np.float32(yy-jj)
      r = ((rx)**2+(ry)**2)**.5
      rang = math.atan2(ry,rx)

      #limit to radius
      if r>irad:
          continue

      #coordinate transform
      xt = (r*math.cos(rang+itheta) + np.float32(ii))
      yt = (r*math.sin(rang+itheta) + np.float32(jj))

      #limit to in bounds
      if (xt<0) | (xt > inim.shape[1]) | (yt<0) | (yt > inim.shape[0]):
        continue

      z0 = inim[yy,xx]

      if mode==0:
        xL = (xt-math.floor(xt))
        xH = 1-xL
        yL = (yt-math.floor(yt))
        yH = 1-yL
        f00 = inim[np.int32(math.floor(yt)),np.int32(math.floor(xt))]
        f10 = inim[np.int32(math.floor(yt)),np.int32(math.ceil(xt))]
        f01 = inim[np.int32(math.ceil(yt)),np.int32(math.floor(xt))]
        f11 = inim[np.int32(math.ceil(yt)),np.int32(math.ceil(xt))]
        if math.isnan(f00) | math.isnan(f10) | math.isnan(f01) | math.isnan(f11):
          continue
        z1 = f00*xH*yH + f10*xL*yH + f01*xH*yL + f11*xL*yL

      else:
        z1 = inim[np.int32(round(yt)),np.int32(round(xt))]
        if math.isnan(z1):
          continue

      zdiff = z1-z0
      if zdiff>=0:
        result[ii,jj] = result[ii,jj]+zdiff
      else:
        result[ii,jj] = result[ii,jj]-zdiff
      step += 1

  result[ii,jj] = result[ii,jj]/step
  #result[ii,jj] = step

    ############################################################ debug #############################
@cuda.jit
def slidewin_rotdiff_core_test(inim, itheta, irad, mode, result):
  #Which thread
  ii,jj = cuda.grid(2)

  #is thread valid?
  if (ii >= inim.shape[0]-1) or (ii < 1) or (jj >= inim.shape[1]-1) or (jj < 1): 
      return

  result[ii,jj] = 0
  step = 0

  #default index ranges
  x0 = np.int32(ii-irad)
  x1 = np.int32(ii+irad)
  y0 = np.int32(jj-irad)
  y1 = np.int32(jj+irad)

  #check for edges
  #x
  xlow = irad - jj
  xhigh = -inim.shape[1]+jj+irad+1
  if (xlow>0) | (xhigh>0):
    if xlow>xhigh:
      x0 = x0+xlow
      x1 = x1-xlow
    else:
      x0 = x0+xhigh
      x1 = x1-xhigh

  #y
  ylow = irad - ii
  yhigh = -inim.shape[0]+ii+irad+1
  if (ylow>0) | (yhigh>0):
    if ylow>yhigh:
      y0 = y0+ylow
      y1 = y1-ylow
    else:
      y0 = y0+yhigh
      y1 = y1-yhigh

  for xx in range(x0,x1):
    for yy in range(y0,y1):
      
      #polar
      rx = np.float32(xx-ii)
      ry = np.float32(yy-jj)
      r = ((rx)**2+(ry)**2)**.5
      rang = math.atan2(ry,rx)

      #limit to radius
      if r>irad:
          continue

      #coordinate transform
      xt = (r*math.cos(rang+itheta) + np.float32(ii))
      yt = (r*math.sin(rang+itheta) + np.float32(jj))

      #limit to in bounds
      if (xt<0) | (xt > inim.shape[1]) | (yt<0) | (yt > inim.shape[0]):
        continue

      z0 = inim[yy,xx]
    
      if mode==0:
        xL = (xt-math.floor(xt))
        xH = 1-xL
        yL = (yt-math.floor(yt))
        yH = 1-yL
        f00 = inim[np.int32(math.floor(yt)),np.int32(math.floor(xt))]
        f10 = inim[np.int32(math.floor(yt)),np.int32(math.ceil(xt))]
        f01 = inim[np.int32(math.ceil(yt)),np.int32(math.floor(xt))]
        f11 = inim[np.int32(math.ceil(yt)),np.int32(math.ceil(xt))]
        if math.isnan(f00) | math.isnan(f10) | math.isnan(f01) | math.isnan(f11):
          continue
        z1 = f00*xH*yH + f10*xL*yH + f01*xH*yL + f11*xL*yL

      else:
        z1 = inim[np.int32(round(yt)),np.int32(round(xt))]
        if math.isnan(z1):
          continue

      step += 1

  #result[ii,jj] = result[ii,jj]/step
  result[ii,jj] = step
    
def slidewin_rotdiff_core_cpu(inim, itheta, ixx, iyy, irad, mode, result):
    from tqdm import tqdm

    hh = 0
    for ii in tqdm(range(inim.shape[0])):
      for jj in range(inim.shape[1]):
        #polar
        rx = ixx-ii
        ry = iyy-jj
        rx = rx.ravel()
        ry = ry.ravel()
        r = ((rx)**2+(ry)**2)**.5
        rang = np.arctan2(ry,rx)

        #limit to radius
        ind = np.where(r<=irad)[0]

        #coordinate transform
        x1 = (r[ind]*np.cos(rang[ind]+itheta[hh]) + ii).ravel()
        y1 = (r[ind]*np.sin(rang[ind]+itheta[hh]) + jj).ravel()

        #limit to in bounds
        ind2 = np.where((x1>=0) & (x1<=inim.shape[1]) & (y1>=0) & (y1<=inim.shape[0]))[0].astype('int')

        if ind2.size==0:
           result[ii,jj] = np.nan
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
          print('mode parameter not recognized')

        result[ii,jj] = np.mean(np.abs(z1.ravel()-z0.ravel()))

#Calculates the mean abs image difference of an image with a rotation transformed version of itself. 
#This function handles the case of a sliding window with a fixed radius 
def slidewin_rotdiff(inim, iang, irad, mode = 0, trygpu=True):
  #Inputs:
  #inim                   :     Input image
  #iang                   :     Rotation angle (radians)
  #irad                   :     Radius to calculate
  #mode(optional)         :     interpolation method, 0 ='bilinear', anything else uses 'nearest'
  #trygpu(optional)       :     attemp to execute first on gpu

  #Outputs:
  #rotdif                :     rotation mean abs difference
  
  inim_sz = np.array(inim.shape)

  #edge padding (to aid interpolation at edges where neighbors are required)
  brdr = 1
  inim = np.pad(inim,brdr,mode='edge')

  rotdif = np.ones_like(inim,np.float32)*-1
    
  print('angle: '+str(iang))
  print('radius: '+str(irad))
  if trygpu:
    #try:
    blockdim = (32, 32)
    print('Blocks dimensions:', blockdim)
    griddim = (rotdif.shape[0] // blockdim[0] + 1, rotdif.shape[1] // blockdim[1] + 1)
    print('Grid dimensions:', griddim)
    slidewin_rotdiff_core_test[griddim, blockdim](inim, iang, irad, mode, rotdif)
    #except:
    #print('GPU Execution failed, fall back to cpu')
    #xx,yy = np.meshgrid(np.arange(0,inim_sz[0]+brdr*2,dtype=np.float32),np.arange(0,inim_sz[1]+brdr*2,dtype=np.float32))
    #slidewin_rotdiff_core_cpu(inim, iang, xx, yy, irad, mode, rotdif)
  else:
    xx,yy = np.meshgrid(np.arange(0,inim_sz[0]+brdr*2,dtype=np.float32),np.arange(0,inim_sz[1]+brdr*2,dtype=np.float32))
    slidewin_rotdiff_core_cpu(inim, iang, xx, yy, irad, mode, rotdif)

  #outputs
  rotdif = rotdif[brdr:-brdr,brdr:-brdr]
  
  return rotdif   
