from numba import cuda
import math
import numpy as np

@cuda.jit
def swid_rng_GPU(inimage, lpx, lpy, xx, yy, outval): 
    ii,jj = cuda.grid(2)

    if (ii >= lpx.shape[0]) or (jj >= lpy.shape[0]): 
      return

    #template indices
    if lpx[ii]<0:
      xxmint = 0
      xxmaxt = xx.shape[1]-1+lpx[ii]
    else:
      xxmint = lpx[ii]
      xxmaxt = xx.shape[1]-1
    if lpy[jj]<0:
      yymint = 0
      yymaxt = xx.shape[0]-1+lpy[jj]
    else:
      yymint = lpy[jj]
      yymaxt = xx.shape[0]-1

    #corresponding main indices 
    if lpx[ii]>0:
      xxmin = 0
      xxmax = xx.shape[1]-1-lpx[ii]
    else:
      xxmin = -lpx[ii]
      xxmax = xx.shape[1]-1
    if lpy[jj]>0:
      yymin = 0
      yymax = xx.shape[0]-1-lpy[jj]
    else:
      yymin = -lpy[jj]
      yymax = xx.shape[0]-1
    
    #comparison
    outval[jj,ii] = 0
    step = 0
    for j in range(0,xxmaxt-xxmint):
      for k in range(0,yymaxt-yymint):
        isnanflag = False
        if inimage.ndim==3:
          isnanflag = (~math.isnan(inimage[k+yymint,j+xxmint,0]) and ~math.isnan(inimage[k+yymin,j+xxmin,0]))
        elif inimage.ndim==2:
          isnanflag = (~math.isnan(inimage[k+yymint,j+xxmint]) and ~math.isnan(inimage[k+yymin,j+xxmin]))
        if isnanflag:
        #if ~math.isnan(inimage[k+yymint,j+xxmint,0]): 
          if inimage.ndim==3:
            lp_3 = inimage.shape[2]
          elif inimage.ndim==2:
            lp_3 = 1
          for l in range(lp_3):
            if inimage.ndim==3:
              temp = inimage[k+yymint,j+xxmint,l]-inimage[k+yymin,j+xxmin,l]
            elif inimage.ndim==2:
              temp = inimage[k+yymint,j+xxmint]-inimage[k+yymin,j+xxmin]
            if temp>0:
              outval[jj,ii] = outval[jj,ii]+temp
            else:
              outval[jj,ii] = outval[jj,ii]-temp
            step += 1
    #outval[jj,ii] = outval[jj,ii]/((j+1)*(k+1)*(l+1))
    outval[jj,ii] = outval[jj,ii]/step

def swid_rng_CPU2(inimage, lpx, lpy, xx, yy, outval):
  from tqdm import tqdm
  #outval = np.ones_like(xx)*np.nan
  #print(lpx)
  #print(lpx.size)
  for ii in tqdm(np.arange(lpx.size)):
    for jj in np.arange(lpy.size):
      #template indices
      if lpx[ii]<0:
        xxmint = 0
        xxmaxt = xx.shape[1]-1+lpx[ii]
      else:
        xxmint = lpx[ii]
        xxmaxt = xx.shape[1]-1
      if lpy[jj]<0:
        yymint = 0
        yymaxt = xx.shape[0]-1+lpy[jj]
      else:
        yymint = lpy[jj]
        yymaxt = xx.shape[0]-1

      #corresponding main indices 
      if lpx[ii]>0:
        xxmin = 0
        xxmax = xx.shape[1]-1-lpx[ii]
      else:
        xxmin = -lpx[ii]
        xxmax = xx.shape[1]-1
      if lpy[jj]>0:
        yymin = 0
        yymax = xx.shape[0]-1-lpy[jj]
      else:
        yymin = -lpy[jj]
        yymax = xx.shape[0]-1

      #comparison
      outval[jj,ii] = 0
      step = 0
      for j in range(0,xxmaxt-xxmint):
        for k in range(0,yymaxt-yymint):
          isnanflag = False
          if inimage.ndim==3:
            isnanflag = (~math.isnan(inimage[k+yymint,j+xxmint,0]) and ~math.isnan(inimage[k+yymin,j+xxmin,0]))
          elif inimage.ndim==2:
            isnanflag = (~math.isnan(inimage[k+yymint,j+xxmint]) and ~math.isnan(inimage[k+yymin,j+xxmin]))
          if isnanflag:
          #if ~math.isnan(inimage[k+yymint,j+xxmint,0]): 
            if inimage.ndim==3:
              lp_3 = inimage.shape[2]
            elif inimage.ndim==2:
              lp_3 = 1
            for l in range(lp_3):
              if inimage.ndim==3:
                temp = inimage[k+yymint,j+xxmint,l]-inimage[k+yymin,j+xxmin,l]
              elif inimage.ndim==2:
                temp = inimage[k+yymint,j+xxmint]-inimage[k+yymin,j+xxmin]
              if temp>0:
                outval[jj,ii] = outval[jj,ii]+temp
              else:
                outval[jj,ii] = outval[jj,ii]-temp
              step += 1
      #outval[jj,ii] = outval[jj,ii]/((j+1)*(k+1)*(l+1))
      #outval[jj,ii] = outval[jj,ii]/step

def swid_rng_CPU(inimage, lpx, lpy, xx, yy):
  from tqdm import tqdm  
  pdsz = ((-np.min(lpx),np.max(lpx)),(-np.min(lpy),np.max(lpy)))
  inimage = np.pad(inimage,pdsz,mode='constant',constant_values = np.nan)
  result = np.ones((lpy.size,lpx.size))*np.nan
  for i,x in tqdm(enumerate(lpx)):
      for j,y in enumerate(lpy):
          result[j,i] = np.nanmean(np.abs(inimage[yy+pdsz[1][0]+y,xx+pdsz[0][0]+x]-inimage[yy+pdsz[1][0]-y,xx+pdsz[0][0]-x]).ravel())
  return result

def slidewin_imdiff(inimage, inrng, stride=1, trygpu=True):
  lpx = np.arange(-inrng[0], inrng[0]+1, stride, dtype='int64')
  lpy = np.arange(-inrng[1], inrng[1]+1, stride,dtype='int64')
  xx,yy = np.meshgrid(np.arange(0,np.shape(inimage)[1],dtype='int64'), np.arange(0,np.shape(inimage)[0],dtype='int64'))
  result = np.ones((inrng[1]*2+1,inrng[0]*2+1),np.float32)

  if trygpu:
    try:
      blockdim = (32, 32)
      #print('Blocks dimensions:', blockdim)
      griddim = (result.shape[0] // blockdim[0] + 1, result.shape[1] // blockdim[1] + 1)
      #print('Grid dimensions:', griddim)
      swid_rng_GPU[griddim, blockdim](inimage,lpx,lpy,xx,yy,result)
    except:
      print('GPU Execution failed, fall back to cpu')
      result = swid_rng_CPU(inimage,lpx,lpy,xx,yy)
  else:
    result = swid_rng_CPU(inimage,lpx,lpy,xx,yy)

  return result