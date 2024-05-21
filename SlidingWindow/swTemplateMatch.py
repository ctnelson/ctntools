#############################################################################################################################################
####################################             Template Matching             ##############################################
#############################################################################################################################################
#A GPU enabled sliding window similarity analysis of a Template to an Image.
#Calculates either the mean absolute difference (slightly faster) or zero normalized cross correlation
#Allows masks/weights to be provided (same size as Template)
#This incarnation is pixel-level only

########################################### Table of Contents ############################
#gpu kernels:
#swMtransf_matchT_IJ_Mean_gpu    :  gpu kernel to calculate sw mean
#swMtransf_matchT_IJ_Var_gpu     :  gpu kernel to calculate sw var
#swMtransf_matchT_IJ_CC_gpu      :  gpu kernel to calculate zero normalized cross correlation
#calling functions:
#swMtransf_matchT_IJ_Mean        :  get sliding window Mean
#swMtransf_matchT_IJ_Var         :  get sliding window Var
#swMtransf_matchTemplate         :  perform the sliding window template match

###########################################################################################
from numba import cuda
import math
import numpy as np
############################################   Mean   #####################################
@cuda.jit
#GPU core routine of sliding window transform comparison. This function asserts the search points are i,j image coordinates. 
def swMtransf_matchT_IJ_Mean_gpu(im, dij, wt, ijBounds, result_counts, result_mean, stride):
  ### Inputs ###
  #im             :   source image / array of data
  #dij            :   [n,2] relative positions to current (iterated over in this function)
  #wt             :   [n,] weight
  #ijBounds       :   [4,1] array bounds to interrogate [jmin, jmax, imin, imax]
  #stride         :   ij stepsize
  ### Outputs (also inputs) ###
  #result_counts  :   #datapoints considered at each point
  #result_mean    :   #mean of local windows

  #Which thread
  ii,jj = cuda.grid(2)
  ii = ii*stride
  jj = jj*stride

  #check thread validity
  if (ii >= ijBounds[3]) or (ii < ijBounds[2]) or (jj >= ijBounds[1]) or (jj < ijBounds[0]):      #check if within prescribed search bounds
      return
  if (ii >= im.shape[0]) or (ii < 0) or (jj >= im.shape[1]) or (jj < 0): 
      return

  #initialize
  iic = np.int32(round(ii/stride))
  jjc = np.int32(round(jj/stride))
  result_counts[iic,jjc] = 0
  result_mean[iic,jjc] = 0
  step = 0

  for i in range(dij.shape[0]):
    it = ii+dij[i,0]
    jt = jj+dij[i,1]

    #check in range
    if (it >= im.shape[0]) or (it < 0) or (jt >= im.shape[1]) or (jt < 0):  #if not within im
      continue

    #check valid value
    z0 = np.float32(im[it,jt])
    w0 = np.float32(wt[i])
    if (not (math.isfinite(z0))) or (not (math.isfinite(w0))):              #if not finite (inf or nan)
      continue

    #increment with result
    result_mean[iic,jjc] += z0*w0
    step +=1*w0

  #divide by number counts
  result_mean[iic,jjc] = result_mean[iic,jjc]/step
  result_counts[iic,jjc] = step

############################################   Variance   #####################################
@cuda.jit
#GPU core routine of sliding window transform comparison. This function asserts the search points are i,j image coordinates. 
def swMtransf_matchT_IJ_Var_gpu(im, dij, wt, ijBounds, iMean, result_counts, result_var, stride):
  ### Inputs ###
  #im             :   source image / array of data
  #dij            :   [n,2] relative positions to current (iterated over in this function)
  #wt             :   [n,] weight
  #ijBounds       :   [4,1] array bounds to interrogate [jmin, jmax, imin, imax]
  #iMean          :   sliding window mean value map
  #stride         :   ij stepsize
  ### Outputs (also inputs) ###
  #result_counts  :   #datapoints considered at each point
  #result_var    :   #sliding window Var

  #Which thread
  ii,jj = cuda.grid(2)
  ii = ii*stride
  jj = jj*stride

  #check thread validity
  if (ii >= ijBounds[3]) or (ii < ijBounds[2]) or (jj >= ijBounds[1]) or (jj < ijBounds[0]):      #check if within prescribed search bounds
      return
  if (ii >= im.shape[0]) or (ii < 0) or (jj >= im.shape[1]) or (jj < 0): 
      return
  
  #initialize
  iic = np.int32(round(ii/stride))
  jjc = np.int32(round(jj/stride))
  result_counts[iic,jjc] = 0
  result_var[iic,jjc] = 0
  step = 0

  for i in range(dij.shape[0]):
    it = ii+dij[i,0]
    jt = jj+dij[i,1]

    #check in range
    if (it >= im.shape[0]) or (it < 0) or (jt >= im.shape[1]) or (jt < 0):  #if not within im
      continue

    #check valid value
    z0 = np.float32(im[it,jt])
    w0 = np.float32(wt[i])
    if (not (math.isfinite(z0))) or (not (math.isfinite(w0))):              #if not finite (inf or nan)
      continue

    #increment with result
    result_var[iic,jjc] += ((z0-iMean[iic,jjc])**2)*w0
    step +=1*w0

  #divide by number counts
  result_var[iic,jjc] = result_var[iic,jjc]/step
  result_counts[iic,jjc] = step

############################################   Cross Correlation   #####################################
@cuda.jit
#GPU core routine of sliding window transform comparison. This function asserts the search points are i,j image coordinates. 
def swMtransf_matchT_IJ_CC_gpu(im, tim, dij, wt, ijBounds, method, iMean, iVar, result_counts, result_CC, stride):
  ### Inputs ###
  #im             :   source image / array of data
  #tim            :   [n,] template image values
  #tMean          :   Mean of template
  #tVar           :   Var of template
  #dij            :   [n,2] relative positions to current (iterated over in this function)
  #wt             :   [n,] weight
  #ijBounds       :   [4,1] array bounds to interrogate [jmin, jmax, imin, imax]
  #method         :   0=norm cross corr, 1=mean abs diff
  #iMean          :   Mean sliding window values at each image position
  #iVar           :   Var sliding window values at each position
  #stride         :   ij stepsize
  ### Outputs (also inputs) ###
  #result_counts  :   #datapoints considered at each point
  #result_CC      :   #cross correlation calculation
  
  #Which thread
  ii,jj = cuda.grid(2)
  ii = ii*stride
  jj = jj*stride

  #check thread validity
  if (ii >= ijBounds[3]) or (ii < ijBounds[2]) or (jj >= ijBounds[1]) or (jj < ijBounds[0]):      #check if within prescribed search bounds
      return
  if (ii >= im.shape[0]) or (ii < 0) or (jj >= im.shape[1]) or (jj < 0): 
      return
  
  #initialize
  iic = np.int32(round(ii/stride))
  jjc = np.int32(round(jj/stride))
  result_counts[iic,jjc] = 0
  result_CC[iic,jjc] = 0
  step = 0

  for i in range(dij.shape[0]):
    it = ii+dij[i,0]
    jt = jj+dij[i,1]

    #check in range
    if (it >= im.shape[0]) or (it < 0) or (jt >= im.shape[1]) or (jt < 0):  #if not within im
      continue

    #check valid value
    z0 = np.float32(im[it,jt])
    w0 = np.float32(wt[i])
    if (not (math.isfinite(z0))) or (not (math.isfinite(w0))):              #if not finite (inf or nan)
      continue

    #increment with result
    if method==0:
      #ccij = (tim[i]-tMean)*(z0-iMean[iic,jjc])/(tVar**.5*iVar[iic,jjc]**.5)
      ccij = tim[i]*(z0-iMean[iic,jjc])/(iVar[iic,jjc]**.5)
      result_CC[iic,jjc] += ccij
    else:
      ccij = tim[i]-z0
      if ccij>=0:
        result_CC[iic,jjc] += ccij
      else:
        result_CC[iic,jjc] -= ccij
    step +=1*w0
  
  #divide by number counts
  result_CC[iic,jjc] = result_CC[iic,jjc]/step
  result_counts[iic,jjc] = step

##################################### Mean Function to Call ##########################################  
def swMtransf_matchT_IJ_Mean(im, templArr, templArriijj=None, wt=None, ijBounds=None, stride=1, tpb=(16,16)):
  ### Inputs ###
  #im                         :   source image
  #templArr                   :   [w,h] or [n,] template image or array
  #templArriijj   (optional)  :   [n,2] x&y positions of template Array values. If 'None' will assume it is a meshgrid.
  #wt             (optional)  :   weights (either [w,h] or [n,])
  #ijBounds       (optional)  :   [4,1] array bounds to interrogate [jmin, jmax, imin, imax]
  #stride         (optional)  :   ij stepsize
  #tpb            (optional)  :   threads per block
  ### Outputs ###
  #swMean                     :   Mean
  #swCounts                   :   #datapoints considered at each point

  if ijBounds is None:
    ijBounds = np.array([0,im.shape[1],0,im.shape[0]])

  #variables
  templArr=np.array(templArr,dtype=np.float32)    
  n = templArr.size
  nd = np.ndim(templArr)
  tAsz = np.array(templArr.shape)

  #Create meshgrid if needed
  if templArriijj is None:
    if nd==1:
      raise ValueError('If templArr is [n,] xx,yy positions must be provided')
    elif tAsz[1]==1:
      raise ValueError('If templArr is [n,] xx,yy positions must be provided')
    ij0 = (np.floor(tAsz/2)).astype(np.int16)
    jj,ii = np.meshgrid(np.arange(tAsz[1],dtype=np.int16)-ij0[1],np.arange(tAsz[0],dtype=np.int16)-ij0[0])
    templArriijj = np.vstack((ii.ravel(),jj.ravel())).T

  templArr=templArr.ravel()

  #weights & crop out invalid or zero
  if wt is None:
    wt = np.ones((n,),dtype=np.float32)
  else:
    assert(wt.size==n)
    wt = wt.ravel().astype(np.float32)
    ind = np.where((np.isfinite(wt)) & (wt!=0))[0]     #cropt out zero and NaNs
    assert(ind.size>0)
    templArr = templArr[ind]
    templArriijj = templArriijj[ind,:]
    wt = wt[ind]

  #variables
  imsz = np.floor(np.array(im.shape)/stride).astype(np.int32)
  swCounts  = np.ones(imsz,np.float32)*np.nan
  swMean = np.ones(imsz,np.float32)*np.nan

  #transfer to gpu
  d_im      = cuda.to_device(im)
  d_tiijj   = cuda.to_device(templArriijj)
  d_wt      = cuda.to_device(wt)
  d_ijbnd   = cuda.to_device(ijBounds)
  d_Counts  = cuda.to_device(swCounts)
  d_Mean = cuda.to_device(swMean)
  
  #kernel settings
  blockspergrid_i = math.ceil(imsz[0] / tpb[0])
  blockspergrid_j = math.ceil(imsz[1] / tpb[1])
  blockspergrid = (blockspergrid_i, blockspergrid_j)

  #Get Mean
  swMtransf_matchT_IJ_Mean_gpu[blockspergrid, tpb](d_im, d_tiijj, d_wt, d_ijbnd, d_Counts, d_Mean, stride)            #calculate sliding window mean
  
  #return to host
  swMean = d_Mean.copy_to_host()
  swCounts = d_Counts.copy_to_host()

  return swMean, swCounts

##################################### Variance Function to Call ##########################################
def swMtransf_matchT_IJ_Var(im, templArr, templArriijj=None, wt=None, ijBounds=None, swMean=None, stride=1, tpb=(16,16)):
  ### Inputs ###
  #im                         :   source image
  #templArr                   :   [w,h] or [n,] template image or array
  #templArriijj   (optional)  :   [n,2] x&y positions of template Array values. If 'None' will assume it is a meshgrid.
  #wt             (optional)  :   weights (either [w,h] or [n,])
  #ijBounds       (optional)  :   [4,1] array bounds to interrogate [jmin, jmax, imin, imax]
  #swMean         (optional)  :   Mean values at each sliding window point
  #stride         (optional)  :   ij stepsize
  #tpb            (optional)  :   threads per block
  ### Outputs ###
  #swVar                      :   #Variance
  #swMean                     :   Mean
  #swCounts                   :   #datapoints considered at each point

  if ijBounds is None:
    ijBounds = np.array([0,im.shape[1],0,im.shape[0]])

  #variables
  templArr=np.array(templArr,dtype=np.float32)    
  n = templArr.size
  nd = np.ndim(templArr)
  tAsz = np.array(templArr.shape)

  #Create meshgrid if needed
  if templArriijj is None:
    if nd==1:
      raise ValueError('If templArr is [n,] xx,yy positions must be provided')
    elif tAsz[1]==1:
      raise ValueError('If templArr is [n,] xx,yy positions must be provided')
    ij0 = (np.floor(tAsz/2)).astype(np.int16)
    jj,ii = np.meshgrid(np.arange(tAsz[1],dtype=np.int16)-ij0[1],np.arange(tAsz[0],dtype=np.int16)-ij0[0])
    #jj,ii = np.meshgrid(np.arange(tAsz[1],dtype=np.int16),np.arange(tAsz[0],dtype=np.int16))
    templArriijj = np.vstack((ii.ravel(),jj.ravel())).T

  templArr=templArr.ravel()

  #weights & crop out invalid or zero
  if wt is None:
    wt = np.ones((n,),dtype=np.float32)
  else:
    assert(wt.size==n)
    wt = wt.ravel().astype(np.float32)
    ind = np.where((np.isfinite(wt)) & (wt!=0))[0]     #cropt out zero and NaNs
    assert(ind.size>0)
    templArr = templArr[ind]
    templArriijj = templArriijj[ind,:]
    wt = wt[ind]

  #variables
  imsz = np.floor(np.array(im.shape)/stride).astype(np.int32)
  swCounts  = np.ones(imsz,np.float32)*np.nan
  swVar = np.ones(imsz,np.float32)*np.nan

  #transfer to gpu
  d_im      = cuda.to_device(im)
  d_tiijj   = cuda.to_device(templArriijj)
  d_wt      = cuda.to_device(wt)
  d_ijbnd   = cuda.to_device(ijBounds)
  d_Counts  = cuda.to_device(swCounts)
  d_Var     = cuda.to_device(swVar)
  
  #kernel settings
  blockspergrid_i = math.ceil(imsz[0] / tpb[0])
  blockspergrid_j = math.ceil(imsz[1] / tpb[1])
  blockspergrid = (blockspergrid_i, blockspergrid_j)

  if swMean is None:
    swMean = np.ones(imsz,np.float32)*np.nan
    d_Mean = cuda.to_device(swMean)
    swMtransf_matchT_IJ_Mean_gpu[blockspergrid, tpb](d_im, d_tiijj, d_wt, d_ijbnd, d_Counts, d_Mean, stride)            #calculate sliding window mean
    #swMtransf_IJradial_Var_gpu[blockspergrid, tpb](d_im, d_M, d_ijbnd, irad, d_Mean, d_Counts, d_Var, stride)
  else:
    d_Mean = cuda.to_device(swMean)
    #swMtransf_IJradial_Var_gpu[blockspergrid, tpb](d_im, d_M, d_ijbnd, irad, d_Mean, d_Counts, d_Var, stride)
  swMtransf_matchT_IJ_Var_gpu[blockspergrid, tpb](d_im, d_tiijj, d_wt, d_ijbnd, d_Mean, d_Counts, d_Var, stride)
  
  #return to host
  swVar = d_Var.copy_to_host()
  swMean = d_Mean.copy_to_host()
  swCounts = d_Counts.copy_to_host()

  return swVar, swMean, swCounts

##################################### Transform Cross Correlation Function to Call ##########################################
def swMtransf_matchTemplate(im, templArr, templArriijj=None, wt=None, calc='ZeroNormCrossCorr', ijBounds=None, stride=1, tpb=(16,16)):
    ### Inputs ###
    #im                         :   source image
    #templArr                   :   [w,h] or [n,] template image or array
    #templArriijj   (optional)  :   [n,2] x&y positions of template Array values. If 'None' will assume it is a meshgrid.
    #wt             (optional)  :   weights (either [w,h] or [n,])
    #calc           (optional)  :   'ZeroNormCrossCorr', 'MeanAbsDiff'
    #ijBounds       (optional)  :   [4,1] array bounds to interrogate [jmin, jmax, imin, imax]
    #stride         (optional)  :   ij stepsize
    #tpb            (optional)  :   threads per block
    ### Outputs ###
    #swCC                       :   template match score
    #swVar                      :   Variance
    #swMean                     :   Mean
    #swCounts                   :   #datapoints considered at each point
    
    #### Main ####
    #default ij bounds (full array)
    if ijBounds is None:
        ijBounds = np.array([0,im.shape[1],0,im.shape[0]])

    #variables
    templArr=np.array(templArr,dtype=np.float32)    
    n = templArr.size
    nd = np.ndim(templArr)
    tAsz = np.array(templArr.shape)

    #Create meshgrid if needed
    if templArriijj is None:
        if nd==1:
            raise ValueError('If templArr is [n,] xx,yy positions must be provided')
        elif tAsz[1]==1:
            raise ValueError('If templArr is [n,] xx,yy positions must be provided')
        ij0 = (np.floor(tAsz/2)).astype(np.int16)
        jj,ii = np.meshgrid(np.arange(tAsz[1],dtype=np.int16)-ij0[1],np.arange(tAsz[0],dtype=np.int16)-ij0[0])
        #jj,ii = np.meshgrid(np.arange(tAsz[1],dtype=np.int16),np.arange(tAsz[0],dtype=np.int16))
        templArriijj = np.vstack((ii.ravel(),jj.ravel())).T

    templArr=templArr.ravel()

    #weights & crop out invalid or zero
    if wt is None:
        wt = np.ones((n,),dtype=np.float32)
    else:
        assert(wt.size==n)
        wt = wt.ravel().astype(np.float32)
        ind = np.where((np.isfinite(wt)) & (wt!=0))[0]     #cropt out zero and NaNs
        assert(ind.size>0)
        templArr = templArr[ind]
        templArriijj = templArriijj[ind,:]
        wt = wt[ind]

    #variables
    imsz = np.floor(np.array(im.shape)/stride).astype(np.int32)
    swCC    = np.ones((imsz[0],imsz[1]),np.float32)*np.nan
    swCounts  = np.ones(imsz,np.int32)*-1
    lpCC      = np.ones((imsz[0],imsz[1]),np.float32)*np.nan

    #transfer to gpu
    d_im      = cuda.to_device(im)
    d_tArr    = cuda.to_device(templArr)
    d_tiijj   = cuda.to_device(templArriijj)
    d_wt      = cuda.to_device(wt)
    d_ijbnd   = cuda.to_device(ijBounds)
    d_Counts  = cuda.to_device(swCounts)
    d_CC      = cuda.to_device(lpCC)

    if calc=='ZeroNormCrossCorr':
        templMean = np.mean(templArr)
        templVar = np.var(templArr)
        #d_tMean = cuda.to_device(templMean)
        #d_tVar = cuda.to_device(templVar)
        templArr = (templArr-templMean)/templVar**.5    #precalculate the template half of the equation
    
    #kernel settings
    blockspergrid_i = math.ceil(imsz[0] / tpb[0])
    blockspergrid_j = math.ceil(imsz[1] / tpb[1])
    blockspergrid = (blockspergrid_i, blockspergrid_j)

    #Get Mean & Var values
    if calc=='ZeroNormCrossCorr':
        #Calculate Mean & Variance
        swMean = np.ones((imsz[0],imsz[1]),np.float32)*np.nan
        swVar = np.ones((imsz[0],imsz[1]),np.float32)*np.nan
        d_Mean = cuda.to_device(swMean)
        d_Var = cuda.to_device(swVar)
        swMtransf_matchT_IJ_Mean_gpu[blockspergrid, tpb](d_im, d_tiijj, d_wt, d_ijbnd, d_Counts, d_Mean, stride)            #calculate sliding window mean
        swMtransf_matchT_IJ_Var_gpu[blockspergrid, tpb](d_im, d_tiijj, d_wt, d_ijbnd, d_Mean, d_Counts, d_Var, stride)       #calculate sliding window variance
        method = 0
    elif calc=='MeanAbsDiff':
        #sets Mean to zero and Variance to 1
        swMean = np.zeros((imsz[0],imsz[0]),np.float32)
        swVar = np.ones((imsz[0],imsz[1]),np.float32)
        d_Mean = cuda.to_device(swMean)
        d_Var = cuda.to_device(swVar)
        method = 1
    else:
        raise ValueError('calc must be either "ZeroNormCrossCorr" or "MeanAbsDiff"')
    #Calculation
    #swMtransf_IJradial_CC_gpu[blockspergrid, tpb](d_im, M, d_ijbnd, irad, method, d_Mean, d_Var, d_Counts, d_CC, stride)       #calculate sliding window cross-correlation
    swMtransf_matchT_IJ_CC_gpu[blockspergrid, tpb](d_im, d_tArr, d_tiijj, d_wt, d_ijbnd, method, d_Mean, d_Var, d_Counts, d_CC, stride) #calculate sliding window cross-correlation
    swCC = d_CC.copy_to_host()
    
    #return to host
    swVar = d_Var.copy_to_host()
    swMean = d_Mean.copy_to_host()
    swCounts = d_Counts.copy_to_host()

    return swCC, swVar, swMean, swCounts
