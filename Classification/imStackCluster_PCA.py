#Searches for classses in given image stack. Based on clustering PCA decomposition loadings

############################################## Imports #########################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition

from ctntools.BaseSupportFunctions.LineAnalysis import findScreeElbow                      #finds elbow in scree plot
from ctntools.Classification.LatentSpaceSupportFuns import ClassLatPositions, plotPCA      #get latent class positions, plot PCA results

############################################## imStackCluster_PCA ###############################################
def imStackCluster_PCA(imStack, Ndims=None, compN=None, componentMax=10, pcaNormalize='Global', pcaScreeThresh=.1, clustMethod='DBSCAN', nnThreshScalar=1.0, nbrKT=.03, nbrNum=1, verbose=0, **kwargs):
    ### Inputs ###
    #imStack            :       [h,w,N] stack of images to cluster
    #Ndims              :       [h,w], [N,2], or None type. Dimensions of N datapoints (only used to plot a loading map)

    #PCA
    #compN              :       manually predefine number of PCA components. If None type will be auto determined from variance scree plot.
    #componentMax       :       max # of components to consider
    #pcaNormalize       :       'Global', 'Frame', or 'none'. Method to normalize data for PCA. 
    #pcaScreeThresh     :       threshold for finding # compononents (kink in PCA scree plot)

    #Clustering
    #clustMethod        :       clustering method 'DBSCAN', 'HDBSCAN', 'OPTICS', 'Agglomerative', 'Birch'
    #nnThreshScalar     :       scalar to apply to distance threshold criteria
    #nbrKT              :       Threshold parameter to find elbow in NN distance distribution
    #nbrNum             :       number of closest neighbors to consider

    #other
    #verbose            :       flag to display execution details

    ### Outputs ###
    #cLabels            :      [N,] cluster label
    #cClassN            :      number of classes
    #PCAloading         :      [N, componentNum] PCA scores of downselected # of components
    #PCAcomponents      :      PCA components (downselected)


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

    #Autoselect # of Components from scree plot
    explained_var = pca.explained_variance_ratio_ 
    if compN is None:
        compN = findScreeElbow(explained_var, elbowMethod='LineOutlier', kinkThresh=pcaScreeThresh, minLinearLen=3, fSEnormalize=True, inax=None)+1 

    #PCA variables
    PCAloading = pca.fit_transform(normXvec)
    PCAloading = PCAloading[:,:compN]
    PCAcomponents = np.ones((imSz[0]*imSz[1],componentMax))*np.nan
    PCAcomponents[pcaValidInd,:] = pca.components_.T
    PCAcomponents = PCAcomponents.reshape(imSz[0], imSz[1],-1)
    PCAcomponents = PCAcomponents[:,:,:compN]

    ############## Clustering ###############
    #get threshold
    nnThresh = getNNbrThresh(PCAloading, nbrKT=nbrKT, nbrNum=nbrNum, verbose=verbose)

    #find Clusters
    cLabels, cClassN = Cluster_NumUnknown(PCAloading, hyperparams=nnThresh*nnThreshScalar, clustMethod=clustMethod, verbose=verbose, **kwargs)

    ############## Display ###############
    if verbose>0:
        print('Autoselected {:d} PCA Components'.format(compN))
        print('Autoselected cluster number: {:d}'.format(cClassN))
    #plots
    if verbose>1:
        if not(Ndims is None):
            if Ndims.size==2:
                compMap = np.reshape(PCAloading,(Ndims[0],Ndims[1],compN))
            elif ((Ndims.shape[0]==imSz[-1]) and (Ndims.shape[1]==2)):
                raise ValueError('Nx2 loading map xy positions not yet coded')
            else:
                raise ValueError('Ndims not recognized. Must be either Nonetype, [h,w], or [N,2]')
        else:
            compMap = None
        axPCAscree = plotPCA(explained_var,PCAcomponents[:,:,:compN],compMap)
        axPCAscree.clear()
        findScreeElbow(explained_var, elbowMethod='LineOutlier', kinkThresh=pcaScreeThresh, minLinearLen=3, fSEnormalize=True, inax=axPCAscree)
        axPCAscree.set_title('Scree Plot')
        axPCAscree.set_xlabel('# Components')
        axPCAscree.set_ylabel('Explained Variance')

    return cLabels, cClassN, PCAloading, PCAcomponents
