import numpy as np
from sklearn import decomposition

from ctntools.BaseSupportFunctions.LineAnalysis import findScreeElbow                      #finds elbow in scree plot
from ctntools.Classification.LatentSpaceSupportFuns import ClassLatPositions, plotPCA      #get latent class positions, plot PCA results

def imStackCluster_PCA(imStack, Ndims=None, compN=None, componentMax=10, pcaNormalize='Global', pcaScreeThresh=.1, verbose=0, axPCAscree=None, **kwargs):
  ### Inputs ###
  #imStack            :       [h,w,N] stack of images to cluster
  #Ndims              :       [h,w], [N,2], or None type. Dimensions of N datapoints (only used to plot a loading map)
  #compN              :       manually predefine number of PCA components. If None type will be auto determined from variance scree plot.
  #componentMax       :       max # of components to consider
  #pcaNormalize       :       'Global', 'Frame', or 'none'. Method to normalize data for PCA. 
  #pcaScreeThresh     :       threshold for finding # compononents (kink in PCA scree plot)

  #Clustering
  #nnThreshScalar     :        
  
  ### Outputs ###

  ############
  ############## PCA ###############
  #Prepare Data
  imSz = np.array(imStack.shape,dtype='int')
  Xvec = np.abs(imStack.reshape(-1,np.shape(imStack)[-1])).T        #reshape to 2D Datavector
  
  #Normalize?
  if pcaNormalize=='Global':  #by global values
      normscalars = np.array([np.nanmean(Xvec.ravel()), np.nanstd(Xvec.ravel())])
      normXvec = (Xvec-normscalars[0])/normscalars[1]
  elif pcaNormalize=='Frame': #framewise
      normscalars = np.vstack((np.nanmean(Xvec,axis=0), np.nanstd(Xvec,axis=0)))
      normXvec = (Xvec-normscalars[0,:])/normscalars[1,:]
  elif pcaNormalize=='none':
      normsclars = np.array([0.,1.])
      normXvec = Xvec
  else:
      raise ValueError('pca normalization method pcaNormalize not recognized, should be Global, Frame, or none') 
    
  #Remove invalid
  pcaValidInd=np.where(np.all(np.isfinite(normXvec),axis=0))[0]
  normXvec = normXvec[:,pcaValidInd]
  #Perform PCA
  pca = decomposition.PCA(componentMax)
  pca.fit(normXvec)
  
  #PCA Outputs
  explained_var = pca.explained_variance_ratio_ 
  XvecTransf = pca.fit_transform(normXvec)
  components = np.ones((fftSz[0]*fftSz[1],componentMax))*np.nan
  components[pcaValidInd,:] = pca.components_.T
  components = components.reshape(fftSz[0], fftSz[1],-1)
  
  #Autoselect # of Components
  if compN is None:
      compN = findScreeElbow(explained_var, elbowMethod='LineOutlier', kinkThresh=pcaScreeThresh, minLinearLen=3, fSEnormalize=True, inax=None)+1 

  #reduced dataset to threshold of pca variables
  XvecPCA = XvecTransf[:,:compN]

  ### Clustering ###
  #get threshold
  nnThresh = getNNbrThresh(XvecPCA, nbrKT=nbrKT, nbrNum=nbrNum, verbose=verbose)
  
  #find Clusters
  cLabels, cClassN = Cluster_NumUnknown(XvecPCA, hyperparams=nnThresh*nnThreshScalar, clustMethod=clustMethod, verbose=verbose, **kwargs)

  if verbose>0:
    print('Autoselected {:d} PCA Components'.format(compN))
    print('Autoselected cluster number: {:d}'.format(dbClassN))
    #plots
    if verbose>1:
      if not(Ndims is None):
        if Ndims.size==2:
          compMap = np.reshape(XvecPCA,(Ndim[0],Ndim[1],compN))
        elif ((Ndims.shape[0]==imSz[-1]) and (Ndims.shape[1]==2)):
          raise ValueError('Nx2 loading map xy positions not yet coded')
        else
          raise ValueError('Ndims not recognized. Must be either Nonetype, [h,w], or [N,2]')
      else:
        compMap = None
      axPCAscree = plotPCA(explained_var,components[:,:,:compN],compMap)
      axPCAscree.clear()
      findScreeElbow(explained_var, elbowMethod='LineOutlier', kinkThresh=pcaScreeThresh, minLinearLen=3, fSEnormalize=True, inax=axPCAscree)
      axPCAscree.set_title('Scree Plot')
      axPCAscree.set_xlabel('# Components')
      axPCAscree.set_ylabel('Explained Variance')

  return cLabels, cClassN, XvecPCA, components
