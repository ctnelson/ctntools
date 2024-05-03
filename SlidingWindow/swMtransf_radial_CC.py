#############################################################################################################################################
####################################             Zero Normalized Cross-Correlation             ##############################################
#############################################################################################################################################
#A GPU enabled sliding window where a radius around each point is compared to a M transformed version centered at the point

########################################### Table of Contents ############################
#gpu kernels:
#swAffine_IJradial_Mean_gpu    :  gpu kernel to calculate sw mean
#swAffine_IJradial_Var_gpu     :  gpu kernel to calculate sw var
#swAffine_IJradial_CC_gpu      :  gpu kernel to calculate zero normalized cross correlation
#calling functions:
#swAffine_IJradial_Mean        :  get sliding window Mean
#swAffine_IJradial_Var         :  get sliding window Var
#swMtransf_radial_CC           :  perform the sliding window CC

###########################################################################################
from numba import cuda
import math
import numpy as np
############################################   Mean   #####################################
@cuda.jit
#GPU core routine of sliding window transform comparison. This function asserts the search points are i,j image coordinates. 
def swAffine_IJradial_Mean_gpu(im, M, ijBounds, irad, result_counts, result_mean):
  ### Inputs ###
  #im             :   source image / array of data
  #M              :   [2,2] Transform matrix for symmetry operator (e.g. for inversion M=[[-1,0],[0,-1]])
  #ijBounds       :   [4,1] array bounds to interrogate [jmin, jmax, imin, imax]
  #irad           :   radius to consider
  ### Outputs (also inputs) ###
  #result_counts  :   #datapoints considered at each point
  #result_mean      :   #cross correlation
  
  #Which thread
  ii,jj = cuda.grid(2)

  #check thread validity
  if (ii >= ijBounds[3]) or (ii < ijBounds[2]) or (jj >= ijBounds[1]) or (jj < ijBounds[0]):      #check if within prescribed search bounds
      return
  if (ii >= im.shape[0]) or (ii < 0) or (jj >= im.shape[1]) or (jj < 0): 
      return
  
  #initialize
  result_counts[ii,jj] = 0
  result_mean[ii,jj] = 0
  step = 0

  #default index ranges
  #i0 = ii-np.int32(irad)
  i0 = ii-np.int32(math.ceil(irad))
  i1 = ii+np.int32(math.ceil(irad))
  j0 = jj-np.int32(math.ceil(irad))
  j1 = jj+np.int32(math.ceil(irad))

  #Get window Mean
  for i in range(i0,i1+1):
    for j in range(j0,j1+1):
      #validity checks
      if (i >= im.shape[0]) or (i < 0) or (j >= im.shape[1]) or (j < 0):    #if out of range of array
        continue
      
      #polar
      di = np.float32(i-ii)
      dj = np.float32(j-jj)
      r = ((di)**2+(dj)**2)**.5
      if r>np.float64(irad):                                                            #if out of radius
        continue
      
      #value
      z0 = np.float32(im[i,j])
      if not (math.isfinite(z0)):                                              #if not finite (inf or nan)
        continue

      #coordinate transform
      it = di*M[0,0]+dj*M[1,0]
      jt = di*M[0,1]+dj*M[1,1]
      it = it+np.float32(ii)
      jt = jt+np.float32(jj)

      #transform validity checks
      if (it >= im.shape[0]) or (it < 0) or (jt >= im.shape[1]) or (jt < 0):    #if out of range of array
        continue
      
      #interpolation of transformed coord
      itL = np.int32(math.floor(it))  #floor i
      itH = np.int32(math.ceil(it))   #ceil i
      jtL = np.int32(math.floor(jt))  #floor j
      jtH = np.int32(math.ceil(jt))   #ceil j
      iL = it-np.float32(itL)         #low-side fraction i
      jL = jt-np.float32(jtL)         #low-side fraction j
      f00 = im[itL,jtL]
      f10 = im[itH,jtL]
      f01 = im[itL,jtH]
      f11 = im[itH,jtH]
      if math.isnan(f00) | math.isnan(f10) | math.isnan(f01) | math.isnan(f11):
        continue
      zt = f00*iL*jL + f10*(1-iL)*jL + f01*iL*(1-jL) + f11*(1-iL)*(1-jL)

      #increment with result
      result_mean[ii,jj,0] += z0
      result_mean[ii,jj,1] += zt
      step +=1
  
  #divide by number counts
  result_mean[ii,jj,0] = result_mean[ii,jj,0]/step
  result_mean[ii,jj,1] = result_mean[ii,jj,1]/step
  result_counts[ii,jj] = step

############################################   Variance   #####################################
@cuda.jit
#GPU core routine of sliding window transform comparison. This function asserts the search points are i,j image coordinates. 
def swAffine_IJradial_Var_gpu(im, M, ijBounds, irad, iMean, result_counts, result_var):
  ### Inputs ###
  #im             :   source image / array of data
  #M              :   [2,2] Transform matrix for symmetry operator (e.g. for inversion M=[[-1,0],[0,-1]])
  #ijBounds       :   [4,1] array bounds to interrogate [jmin, jmax, imin, imax]
  #irad           :   radius to consider
  #iMean          :   Mean values at each position
  ### Outputs (also inputs) ###
  #result_counts  :   #datapoints considered at each point
  #result_var     :   #cross correlation
  
  #Which thread
  ii,jj = cuda.grid(2)

  #check thread validity
  if (ii >= ijBounds[3]) or (ii < ijBounds[2]) or (jj >= ijBounds[1]) or (jj < ijBounds[0]):      #check if within prescribed search bounds
      return
  if (ii >= im.shape[0]) or (ii < 0) or (jj >= im.shape[1]) or (jj < 0): 
      return
  
  #initialize
  result_counts[ii,jj] = 0
  result_var[ii,jj] = 0
  step = 0

  #default index ranges
  r = np.int32(irad)
  i0 = np.int32(ii-irad)
  i1 = np.int32(ii+irad)
  j0 = np.int32(jj-irad)
  j1 = np.int32(jj+irad)

  #Get window Mean
  for i in range(i0,i1+1):
    for j in range(j0,j1+1):
      #validity checks
      if (i >= im.shape[0]) or (i < 0) or (j >= im.shape[1]) or (j < 0):    #if out of range of array
        continue
      
      #polar
      di = np.float32(i-ii)
      dj = np.float32(j-jj)
      r = ((di)**2+(dj)**2)**.5
      if r>irad:                                                            #if out of radius
        continue
      
      #value
      z0 = np.float32(im[i,j])
      if not (math.isfinite(z0)):                                              #if not finite (inf or nan)
        continue

      #coordinate transform
      it = di*M[0,0]+dj*M[1,0]
      jt = di*M[0,1]+dj*M[1,1]
      it = it+np.float32(ii)
      jt = jt+np.float32(jj)

      #transform validity checks
      if (it >= im.shape[0]) or (it < 0) or (jt >= im.shape[1]) or (jt < 0):    #if out of range of array
        continue
      
      #interpolation of transformed coord
      itL = np.int32(math.floor(it))  #floor i
      itH = np.int32(math.ceil(it))   #ceil i
      jtL = np.int32(math.floor(jt))  #floor j
      jtH = np.int32(math.ceil(jt))   #ceil j
      iL = it-np.float32(itL)         #low-side fraction i
      jL = jt-np.float32(jtL)         #low-side fraction j
      f00 = im[itL,jtL]
      f10 = im[itH,jtL]
      f01 = im[itL,jtH]
      f11 = im[itH,jtH]
      if math.isnan(f00) | math.isnan(f10) | math.isnan(f01) | math.isnan(f11):
        continue
      zt = f00*iL*jL + f10*(1-iL)*jL + f01*iL*(1-jL) + f11*(1-iL)*(1-jL)

      #increment with result
      result_var[ii,jj,0] += (z0-iMean[ii,jj,0])**2
      result_var[ii,jj,1] += (zt-iMean[ii,jj,1])**2
      step +=1
  
  #divide by number counts
  result_var[ii,jj,0] = result_var[ii,jj,0]/step
  result_var[ii,jj,1] = result_var[ii,jj,1]/step
  result_counts[ii,jj] = step

############################################   Cross Correlation   #####################################
@cuda.jit
#GPU core routine of sliding window transform comparison. This function asserts the search points are i,j image coordinates. 
def swAffine_IJradial_CC_gpu(im, M, ijBounds, irad, method, iMean, iVar, result_counts, result_CC):
  ### Inputs ###
  #im             :   source image / array of data
  #M              :   [2,2] Transform matrix for symmetry operator (e.g. for inversion M=[[-1,0],[0,-1]])
  #ijBounds       :   [4,1] array bounds to interrogate [jmin, jmax, imin, imax]
  #irad           :   radius to consider
  #method         :   0=norm cross corr, 1=mean abs diff
  #iMean          :   Mean values at each position
  #iVar           :   Var values at each position
  ### Outputs (also inputs) ###
  #result_counts  :   #datapoints considered at each point
  #result_CC      :   #cross correlation calculation
  
  #Which thread
  ii,jj = cuda.grid(2)

  #check thread validity
  if (ii >= ijBounds[3]) or (ii < ijBounds[2]) or (jj >= ijBounds[1]) or (jj < ijBounds[0]):      #check if within prescribed search bounds
      return
  if (ii >= im.shape[0]) or (ii < 0) or (jj >= im.shape[1]) or (jj < 0): 
      return
  
  #initialize
  result_counts[ii,jj] = 0
  result_CC[ii,jj] = 0
  step = 0

  #default index ranges
  r = np.int32(irad)
  i0 = np.int32(ii-irad)
  i1 = np.int32(ii+irad)
  j0 = np.int32(jj-irad)
  j1 = np.int32(jj+irad)

  #Get window Mean
  for i in range(i0,i1+1):
    for j in range(j0,j1+1):
      #validity checks
      if (i >= im.shape[0]) or (i < 0) or (j >= im.shape[1]) or (j < 0):    #if out of range of array
        continue
      
      #polar
      di = np.float32(i-ii)
      dj = np.float32(j-jj)
      r = ((di)**2+(dj)**2)**.5
      if r>irad:                                                            #if out of radius
        continue
      
      #value
      z0 = np.float32(im[i,j])
      if not (math.isfinite(z0)):                                              #if not finite (inf or nan)
        continue

      #coordinate transform
      it = di*M[0,0]+dj*M[1,0]
      jt = di*M[0,1]+dj*M[1,1]
      it = it+np.float32(ii)
      jt = jt+np.float32(jj)

      #transform validity checks
      if (it >= im.shape[0]) or (it < 0) or (jt >= im.shape[1]) or (jt < 0):    #if out of range of array
        continue
      
      #interpolation of transformed coord
      itL = np.int32(math.floor(it))  #floor i
      itH = np.int32(math.ceil(it))   #ceil i
      jtL = np.int32(math.floor(jt))  #floor j
      jtH = np.int32(math.ceil(jt))   #ceil j
      iL = it-np.float32(itL)         #low-side fraction i
      jL = jt-np.float32(jtL)         #low-side fraction j
      f00 = im[itL,jtL]
      f10 = im[itH,jtL]
      f01 = im[itL,jtH]
      f11 = im[itH,jtH]
      if math.isnan(f00) | math.isnan(f10) | math.isnan(f01) | math.isnan(f11):
        continue
      zt = f00*iL*jL + f10*(1-iL)*jL + f01*iL*(1-jL) + f11*(1-iL)*(1-jL)

      #increment with result
      if method==0:
        ccij = (zt-iMean[ii,jj,1])*(z0-iMean[ii,jj,0])/(iVar[ii,jj,0]**.5*iVar[ii,jj,1]**.5)
        result_CC[ii,jj] += ccij
      else:
        ccij = zt-z0
        if ccij>=0:
          result_CC[ii,jj] += ccij
        else:
          result_CC[ii,jj] -= ccij
      step +=1
  
  #divide by number counts
  result_CC[ii,jj] = result_CC[ii,jj]/step
  result_counts[ii,jj] = step

##################################### Mean Function to Call ##########################################  
def swAffine_IJradial_Mean(im, M, irad, ijBounds=None, tpb=(16,16)):
  ### Inputs ###
  #im                   :   source image / array of data
  #M                    :   [2,2] Transform matrix for symmetry operator (e.g. for inversion M=[[-1,0],[0,-1]])
  #irad                 :   radius to consider
  #ijBounds (optional)  :   [4,1] array bounds to interrogate [jmin, jmax, imin, imax]
  ### Outputs ###
  #swCounts             :   #datapoints considered at each point
  #swMean               :   #Mean

  if ijBounds is None:
    ijBounds = np.array([0,im.shape[1],0,im.shape[0]])

  #initialize
  swCounts  = np.ones(im.shape,np.int32)*100#-1
  swMean    = np.ones((im.shape[0],im.shape[1],2),np.float32)*np.nan
  
  #invoke Kernel
  blockspergrid_i = math.ceil(im.shape[0] / tpb[0])
  blockspergrid_j = math.ceil(im.shape[1] / tpb[1])
  blockspergrid = (blockspergrid_i, blockspergrid_j)
  swAffine_IJradial_Mean_gpu[blockspergrid, tpb](im, M, ijBounds, irad, swCounts, swMean)

  return swMean, swCounts

##################################### Variance Function to Call ##########################################
def swAffine_IJradial_Var(im, M, irad, ijBounds=None, swMean=None, tpb=(16,16)):
  ### Inputs ###
  #im                   :   source image / array of data
  #M                    :   [2,2] Transform matrix for symmetry operator (e.g. for inversion M=[[-1,0],[0,-1]])
  #irad                 :   radius to consider
  #ijBounds (optional)  :   [4,1] array bounds to interrogate [jmin, jmax, imin, imax]
  #swMean   (optional)  :   Mean values at each sliding window point
  #tpb      (optional)  :   threads per block
  ### Outputs ###
  #swVar                :   #Variance
  #swMean               :   #Mean
  #swCounts             :   #datapoints considered at each point
  
  #### Function ####
  if ijBounds is None:
    ijBounds = np.array([0,im.shape[1],0,im.shape[0]])

  #initialize
  swCounts  = np.ones(im.shape,np.int32)*-1
  swVar    = np.ones((im.shape[0],im.shape[1],2),np.float32)*np.nan
  #transfer to gpu
  d_im      = cuda.to_device(im)
  d_M       = cuda.to_device(M)
  #d_r       = cuda.to_device(irad)         #if I do this it seems to convert from scalar to a 0D array and doesn't play nice in the kernel 
  d_ijbnd   = cuda.to_device(ijBounds)
  d_Var     = cuda.to_device(swVar)
  d_Counts  = cuda.to_device(swCounts)
  
  #invoke Kernel
  blockspergrid_i = math.ceil(im.shape[0] / tpb[0])
  blockspergrid_j = math.ceil(im.shape[1] / tpb[1])
  blockspergrid = (blockspergrid_i, blockspergrid_j)
  if swMean is None:
    swMean = np.ones((im.shape[0],im.shape[1],2),np.float32)*np.nan
    d_Mean = cuda.to_device(swMean)
    swAffine_IJradial_Mean_gpu[blockspergrid, tpb](d_im, d_M, d_ijbnd, irad, d_Counts, d_Mean)
    swAffine_IJradial_Var_gpu[blockspergrid, tpb](d_im, d_M, d_ijbnd, irad, d_Mean, d_Counts, d_Var)
  else:
    d_Mean = cuda.to_device(swMean)
    swAffine_IJradial_Var_gpu[blockspergrid, tpb](d_im, d_M, d_ijbnd, irad, d_Mean, d_Counts, d_Var)
  
  #return to host
  swVar = d_Var.copy_to_host()
  swMean = d_Mean.copy_to_host()
  swCounts = d_Counts.copy_to_host()

  return swVar, swMean, swCounts

##################################### Transform Cross Correlation Function to Call ##########################################
def swMtransf_radial_CC(im, M, irad, calc='ZeroNormCrossCorr', ijBounds=None, tpb=(16,16)):
  ### Inputs ###
  #im                   :   source image / array of data
  #M                    :   [2,2,n] Transform matrix for symmetry operator(s) (e.g. for inversion M=[[-1,0],[0,-1]])
  #irad                 :   radius to consider
  #calc     (optional)  :   'ZeroNormCrossCorr', 'MeanAbsDiff'
  #ijBounds (optional)  :   [4,1] array bounds to interrogate [jmin, jmax, imin, imax]
  #tpb      (optional)  :   threads per block
  ### Outputs ###
  #swVar                :   #Variance
  #swMean               :   #Mean
  #swCounts             :   #datapoints considered at each point
  
  #### Function ####
  #default ij bounds (full array)
  if ijBounds is None:
    ijBounds = np.array([0,im.shape[1],0,im.shape[0]])

  #Transform Matrix
  msz = np.array(M.shape,dtype='int')
  nd = np.ndim(M)
  if nd==2:
    n = 1
    M = M[:,:,np.newaxis]
  elif nd==3:
    n = M.shape[2]
  else:
    raise ValueError('M transform matrix must be [2,2,n] in size')
  assert((msz[0]==2) & (msz[1]==2))

  #initialize
  swCounts  = np.ones(im.shape,np.int32)*-1
  swCC    = np.ones((im.shape[0],im.shape[1],n),np.float32)*np.nan

  #transfer to gpu
  d_im      = cuda.to_device(im)
  d_ijbnd   = cuda.to_device(ijBounds)
  d_Counts  = cuda.to_device(swCounts)
  lpCC      = np.ones((im.shape[0],im.shape[1]),np.float32)*np.nan
  d_CC      = cuda.to_device(lpCC)
  
  #kernel settings
  blockspergrid_i = math.ceil(im.shape[0] / tpb[0])
  blockspergrid_j = math.ceil(im.shape[1] / tpb[1])
  blockspergrid = (blockspergrid_i, blockspergrid_j)

  #Get Mean & Var values (assumes this is consistent across all M so only run once)
  if calc=='ZeroNormCrossCorr':
    #Calculate Mean & Variance
      swMean = np.ones((im.shape[0],im.shape[1],2),np.float32)*np.nan
      swVar = np.ones((im.shape[0],im.shape[1],2),np.float32)*np.nan
      d_Mean = cuda.to_device(swMean)
      d_Var = cuda.to_device(swVar)
      swAffine_IJradial_Mean_gpu[blockspergrid, tpb](d_im, M[:,:,0], d_ijbnd, irad, d_Counts, d_Mean)        #calculate sliding window mean
      swAffine_IJradial_Var_gpu[blockspergrid, tpb](d_im, M[:,:,0], d_ijbnd, irad, d_Mean, d_Counts, d_Var)  #calculate sliding window variance
      method = 0
  elif calc=='MeanAbsDiff':
    #sets Mean to zero and Variance to 1
    swMean = np.zeros((im.shape[0],im.shape[1],2),np.float32)
    swVar = np.ones((im.shape[0],im.shape[1],2),np.float32)
    d_Mean = cuda.to_device(swMean)
    d_Var = cuda.to_device(swVar)
    method = 1
  else:
    raise ValueError('calc must be either "ZeroNormCrossCorr" or "MeanAbsDiff"')
  #Calculation
  for i in range(n):
    swAffine_IJradial_CC_gpu[blockspergrid, tpb](d_im, M[:,:,i], d_ijbnd, irad, method, d_Mean, d_Var, d_Counts, d_CC)       #calculate sliding window cross-correlation
    swCC[:,:,i] = d_CC.copy_to_host()
  
  #return to host
  swVar = d_Var.copy_to_host()
  swMean = d_Mean.copy_to_host()
  swCounts = d_Counts.copy_to_host()

  return swCC, swVar, swMean, swCounts
