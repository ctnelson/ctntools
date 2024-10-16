#Classify an image by a sliding window FFT

############################################## Contents ###################################################
### Sub Functions ###
#getWinSz           :   Determine slicing window size (ensures multiple of stride)
#getClassAvg        :   Determine the average by class
#getClassMed        :   Determine the median by class
#plotScore          :   Plots Score & Label Maps
#plotLabelIm        :   plots final segmentation labels & colorized image

### Main ###
#swSegmentationFFT  :   classify an image by a sliding window FFT

########################################### Imports #################################################
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator

from papuc.example_maps import colormaps

#Custom imports from repo
from ctntools.BaseSupportFunctions.plottingFuns import plotImStackGridColorbars                 #quick helper function to plot image stacks & scalebars
from ctntools.StructureAnalysis.fftPeaks import fftPeaks                                        #performs FFT and finds peaks
from ctntools.SlidingWindow.swSlicing import swSlices, swScoreMin                               #returns image patch slices given size & stride
from ctntools.Classification.LatentSpaceSupportFuns import ClassLatPositions, plotClassDistr    #get latent class positions, plot Classes in latent joint distributions  
from ctntools.Classification.clusterPrune import clusterPrune                                   #Prune clusters in a coordinate space given selection criteria (e.g. max number of clusters, min population size)
from ctntools.Classification.imStackCluster_PCA import imStackCluster_PCA                       #Cluster image stack via PCA variables

########################################### Sub Functions #################################################
############################################# get WinSz ###################################################
#determine slicing window size (ensures multiple of stride)
def getWinSz(targetSz, istride):
    if istride<1:
        winSz = (np.ceil(targetSz*istride)/istride).astype('int')    #ensure divisible by stride
    else:
        #winSz = (np.ceil(targetSz/2)*2).astype('int')               #ensure even
        winSz = np.ceil(targetSz).astype('int')
    return winSz
########################################### getClassAvg ###################################################
#Determine the average by class
def getClassAvg(inDataStack, inClassLabels, cN=None):
    ### Inputs ###
    #inDataStack    :   [,,,n]     n datapoints of whatever dims
    #inClassLabels  :   [n,]       n class labels
    ### Outputs ###
    #classAvg       :   [,,,classN]
    ###
    if cN is None:
        cN = np.nanmax(inClassLabels)+1
    dataDims = np.array(inDataStack.shape)
    avgDims = dataDims.copy()
    avgDims[-1] = cN
    #Average of classes
    dVec = np.reshape(inDataStack,(-1,dataDims[-1]))    #reshape to 2 dims
    classAvg = np.ones((dVec.shape[0],cN))*np.nan     #preallocate output
    #loop through classes & average
    for i in range(cN):
        ind = np.where(inClassLabels==i)[0]
        if ind.size>0:
            classAvg[:,i] = np.nanmean(dVec[:,ind],axis=1)
    #reshape like input    
    classAvg = np.reshape(classAvg,avgDims)
    return classAvg
########################################### getClassMed ###################################################
#Determine the median by class
def getClassMed(inDataStack, inClassLabels, cN=None):
    ### Inputs ###
    #inDataStack    :   [,,,n]     n datapoints of whatever dims
    #inClassLabels  :   [n,]       n class labels
    ### Outputs ###
    #classMed       :   [,,,classN]
    ###
    if cN is None:
        cN = np.nanmax(inClassLabels)+1
    dataDims = np.array(inDataStack.shape)
    avgDims = dataDims.copy()
    avgDims[-1] = cN
    #Median of classes
    dVec = np.reshape(inDataStack,(-1,dataDims[-1]))    #reshape to 2 dims
    classMed = np.ones((dVec.shape[0],cN))*np.nan     #preallocate output
    #loop through classes & average
    for i in range(cN):
        ind = np.where(inClassLabels==i)[0]
        if ind.size>0:
            classMed[:,i] = np.nanpercentile(dVec[:,ind],50,axis=1)
    #reshape like input    
    classMed = np.reshape(classMed,avgDims)
    return classMed

########################################### plotScore ###################################################
#Plots Score & Label Maps
def plotScore(inLabels, inScore, cN=None, classColors=None, nonClustColor=[.9,.9,.9,1], figW=8):
    ### Inputs ###
    #inLabels   :   [h,w]   Labels
    #inScore    :   [h,w]   Score
    ### Outputs ###
    #ax         :   plot axes
    imSz = np.array([inLabels.shape[0],inLabels.shape[1]])                    #image dimensions
    if cN is None:
        cN = np.nanmax(inLabels.ravel()).astype('int')
    if classColors is None:
        classColors = plt.cm.jet(np.linspace(0, 1, cN))
        classColors = np.append(classColors,np.array([nonClustColor]),axis=0)
    c_ = classColors[np.reshape(inLabels,(-1,inLabels.shape[-1])),:]
    c_ = np.reshape(c_,(imSz[0],imSz[1],4))
    if inScore is None:
        fig, ax = plt.subplots(1, 1, figsize=(figW/2,figW/2*imSz[0]/imSz[1]))
        ax.imshow(c_,origin='lower')
        ax.set_title('Class Labels')
    else:
        fig, ax = plt.subplots(1, 2, figsize=(figW,figW/2*imSz[0]/imSz[1]))
        ax[0].imshow(c_,origin='lower')
        ax[0].set_title('Class Labels')
        ax[1].imshow(inScore,origin='lower')
        ax[1].set_title('Score')
    return ax
########################################### plotLabelIm ###################################################
#Plots Class Map and (optionally) a colorization of the input image
def plotLabelIm(inim, inLabels, cN=None, normPrcntile=.1, classColors=None, nonClustColor=[.9,.9,.9,1], figW=12):
    ### Inputs ###
    #inim           :   [h,w] input image
    #inLabels       :   [h,w] labels
    #cN             :   class number
    #normPrcntile   :   percentile value for normalization (used for colorization)
    #classColors    :   [cN,4] colors for Class Map (if provided, cannot be used for a colorized image)
    #nonClustColor  :   [4,] color for nonclassified (=-1) labels
    #figW           :   figure width
    ### Outputs ###
    #ax     :   plotting axis
    ###
    imSz = np.array([inim.shape[0],inim.shape[1]])                    #image dimensions
    if cN is None:
        cN = np.nanmax(inLabels.ravel())+1

    #if colors provided (only plots class map)
    if not(classColors is None):
        fig, ax = plt.subplots(1, 1, figsize=(figW,figW*imSz[0]/imSz[1]))
        #generate image
        classColors = np.append(classColors,np.array([nonClustColor]),axis=0)
        c_ = classColors[np.nan_to_num(inLabels,nan=-1).ravel().astype('int'),:]
        c_ = np.reshape(c_,(imSz[0], imSz[1], 4))
        #plot
        ax.imshow(c_,origin='lower')
        ax.set_title('Class Label Mask')
    #if colors to be determined (plots class map & colorized image)
    else:
        fig, ax = plt.subplots(1, 2, figsize=(figW,figW/2*imSz[0]/imSz[1]))
        #generate image
        my_map = colormaps['default']
        #Label Map
        angle = np.arange(cN)/cN*np.pi*2
        mag = np.ones((cN,))
        classColors = np.ones((cN,4))
        classColors[:,:3] = my_map(angle, mag)
        classColors = np.append(classColors,np.array([nonClustColor]),axis=0)
        lblMap = classColors[inLabels.ravel(),:]
        lblMap = np.reshape(lblMap,(imSz[0],imSz[1],4))
        #colorized inim
        ind = np.where(inLabels.ravel()>=0)[0]
        angle = inLabels.ravel()[ind]/cN*np.pi*2
        mag = (inim.ravel()[ind]-np.nanpercentile(inim.ravel(),normPrcntile)) / (np.nanpercentile(inim.ravel(),100-normPrcntile)-np.nanpercentile(inim.ravel(),normPrcntile))
        mag = np.clip(mag,0,1)
        clrIm = np.repeat(np.array(nonClustColor)[np.newaxis,:],np.prod(imSz),axis=0)
        clrIm[ind,:3] = my_map(angle, mag)
        clrIm = np.reshape(clrIm,(imSz[0],imSz[1],4))
        #plot
        ax[1].imshow(lblMap,origin='lower')
        ax[1].set_title('Class Map')
        ax[0].imshow(clrIm,origin='lower')
        ax[0].set_title('Class Colorized Image')
    return ax

################################################ Main ########################################################
########################################## swSegmentationFFT #################################################
def swSegmentationFFT(im, imNormalize='none', winSz=None, stride=.5, fft_s=None, fft_zeroflag=False, fftMaxScalar=1.25, clustHyperparameter=1.0, compN=None, componentMax=10, pcaNormalize='Global', pcaScreeThresh=.1, RefMethod='minfract', RefThresh=None, RefThresh_maxN=3, RefThresh_minPop=10, RefThresh_minFract=.04, ReClassMethod='latDist', ReClass_latDist=5, latNorm='classStd', reClassOnlyPruned=True, scoreMethod='diff2Avg', returnClass=None, verbose=0):
    ########################################### Inputs ###################################################
    #im                 :       image for segmentation
    #imNormalize        :       'Global', 'Frame', or 'none'. Method to normalize image patches prior to PCA.
    ###### FFT & Slicing Windows #####
    #winSz              :       value or None.   Sliding window size.     If none autodetermined by an FFT
    #stride             :       Stride (as a fraction) of sliding window size
    #fft_s              :       value or None.   FFT min period.          If none autodeteremined by an FFT
    #fft_zeroflag       :       flag to include zero frequency (Mean of patch)?
    #fftMaxScalar       :       If autoselecting the winSz, this scalar is applied to the max FFT period.
    ############## PCA Clustering ###############
    #clustHyperparameter:       adjustment scalar for clustering algorithms
    #compN              :       manually predefine number of PCA components. If None type will be auto determined from variance scree plot.
    #componentMax       :       max # of components to consider
    #pcaNormalize       :       'Global', 'Frame', or 'none'. Method to normalize data for PCA. 
    #pcaScreeThresh     :       threshold for finding # compononents (kink in PCA scree plot)
    ####### Cluster Pruning ##########
    #RefMethod          :       Prune clusters if fail criteria of 'maxN' (limit maximum numbe of clusters), 'minpop' (minimum population size of cluster), or 'minfrac' (minimum fractional population size of cluster)
    #RefThresh          :       Threshold used for RefMethod. Set to None type to use default values.
    #RefThresh_maxN     :       default max number of clusters
    #RefThresh_minPop   :       default cluster min population size
    #RefThresh_minFract :       default cluster min population fraction
    ####### Cluster Reclassify #######
    #ReClassMethod      :       'latDist', 'KMeans', or 'none'. Method of reclassifying clusters deemed invalid in pruning step. If latent distance, can avoid redundant computation by providing classLatCoords.
    #ReClass_latDist    :       if reclassify by latent distance, use this cutoff sigma (beyond this datapoints will be classified 'noise')
    #latNorm            :       'none', 'globalStd', or 'classStd'
    #reClassOnlyPruned  :       flag whether to reclassify only removed clusters? Only applies to LatDist, KMeans will reclassify all points for the determined pruned number of clusters.
    ##### Some Final Variables  ######
    #scoreMethod        :       'latDist', 'diff2Avg', or 'all'. How to handle selection if slices overlap. Can use either minimum latent distance, image difference to class average, or use all points.
    #returnClass        :       'Avg', 'Med', or 'PCAinv' to return the class representations as an average, median, or the inverted PCA representation. Defaults to None type.
    #verbose            :       Display debug / progress information. 0/False for silent. 1 for minimal, 2 for lots of info/plots

    ########################################### Outputs #################################################
    #imLabel            :       class labels of image. Will be integers and non-classified / noise will be assigned -1
    #classAvgIm         :       class average FFTs (if ReturnClass != None)


    ########################################### Main #####################################################
    ### Parameters
    imSz = np.array(im.shape)[[1,0]]        #image dimensions
    
    ### Spacing from FFT ###
    stride = np.array(stride) 
    if verbose>1:
        fig, ax = plt.subplots(1, 3, figsize=(20,20/3), dpi = 100)
        inax=ax
    else:
        inax = [None]*3
    #Get Spacing Settings
    if (winSz is None) or (fft_s is None): 
        xy_fft, xy_v, radDistr, radPks, radPks0, im_fft = fftPeaks(im, gaussSigma=2, minRExclusionMethod='afterminima', principlePeakMethod='first', inax=inax)     #Global FFT
        if fft_s is None:
            r_v = (xy_v[0,:]**2 + xy_v[1,:]**2)**.5
            fft_s = np.nanmin(r_v)
        if winSz is None:
            #window size
            winSz = getWinSz(radPks[0,radPks0]*fftMaxScalar, stride)
        else:
            winSz = getWinSz(np.array(winSz), stride)

        if verbose>0:
            print('Scale estimate by FFT [{:.2f}, {:.2f}]'.format(fft_s, radPks[0,radPks0]))
    else:
        winSz = getWinSz(np.array(winSz), stride)
        xy_v=None
    #WinSz
    if winSz.size == 1:
        winSz = np.tile(winSz,2)
    else:
        assert winSz.size==2
        assert winSz[0]==winSz[1]
    #Stride
    stride = np.round(stride*winSz).astype('int')
    if stride.size == 1:
        stride = np.tile(stride,2)
    else:
        assert stride.size==2
        assert stride[0]==stride[1]
    #print?
    if verbose > 0:
        print('Window Size: {:d}x{:d}'.format(winSz[0],winSz[1]))
        print('Stride: {:d}x{:d}'.format(stride[0],stride[1]))

    ### Slicing ###
    imPatches, sWinSz, imPatchOrigin, swSliceInd = swSlices(im, winSz, stride, verbose=verbose)
    imPchN = imPatches.shape[-1]

    ### Image Stack Normalization ###
    if imNormalize=='Global':  #by global values
        imNormScalars = np.array([np.nanmean(im.ravel()), np.nanstd(im.ravel())])
        imPtchNorm = (imPatches-imNormScalars[0])/imNormScalars[1]
    elif imNormalize=='Frame': #framewise
        temp = np.reshape(imPatches,(-1,imPatches.shape[-1]))
        imNormScalars = np.vstack((np.nanmean(temp,axis=0), np.nanstd(temp,axis=0)))
        imPtchNorm = (temp-imNormScalars[0,:])/imNormScalars[1,:]
        imPtchNorm = np.reshape(imPtchNorm,imPatches.shape)
    elif imNormalize=='none':
        imNormScalars = np.array([0.,1.])
        imPtchNorm = imPatches
    else:
        raise ValueError('Image normalization method imNormalize not recognized, should be Global, Frame, or none')
    
    ### FFT ###
    if fft_zeroflag:
        rmin=0
    else:
        rmin=1
    imPchFFT = np.ones((winSz[0],winSz[1],imPchN),dtype='complex')*np.nan       #full size (will crop)
    for i in tqdm(range(imPchN), desc='calculating FFTs', disable=(verbose==0)):
        fFFT = np.fft.fftshift(np.fft.fft2(imPtchNorm[:,:,i]))
        imPchFFT[:,:,i]=fFFT
    #crop
    fftSz = np.ceil(winSz/fft_s).astype('int')
    cntr = np.array([np.floor(winSz[0]/2),np.floor(winSz[1]/2)],dtype='int')
    imPchFFT = imPchFFT[cntr[1]-fftSz[1]:cntr[1]+fftSz[1]+1,cntr[0]-fftSz[0]:cntr[0]+fftSz[0]+1,:]
    #mask
    xx,yy = np.meshgrid(np.arange(-fftSz[1],fftSz[1]+1), np.arange(-fftSz[1],fftSz[1]+1))
    r = (xx**2+yy**2)**.5
    fftmask = np.where(((r>=rmin) & (r<=fftSz[0])),1,np.nan)
    #apply mask
    imPchFFT = imPchFFT*np.repeat(fftmask[:,:,np.newaxis],imPchN,axis=2)
    #
    fftSz = fftSz*2+1

    ### PCA Clustering ###
    cLabels, cClassN, PCAValidInd, PCAloading, PCAcomponents, pca = imStackCluster_PCA(imPchFFT, Ndims=sWinSz, componentMax=componentMax, pcaNormalize=pcaNormalize, pcaScreeThresh=pcaScreeThresh, clustMethod='DBSCAN', nnThreshScalar=clustHyperparameter, nbrKT=.03, nbrNum=1, verbose=verbose)
    compN = PCAcomponents.shape[-1]

    ### Prune Clusters ###
    dReClassLabels, dReClassN = clusterPrune(PCAloading, cLabels, dClassN=cClassN, RefMethod=RefMethod, RefThresh=RefThresh, RefThresh_maxN=RefThresh_maxN, RefThresh_minPop=RefThresh_minPop, RefThresh_minFract=RefThresh_minFract, ReClassMethod=ReClassMethod, ReClass_latDist=ReClass_latDist, latNorm=latNorm, reClassOnlyPruned=reClassOnlyPruned, verbose=verbose)
    #plot?
    if verbose>1:
        reclassLatCoords = ClassLatPositions(PCAloading, dReClassLabels, lblArray=np.arange(dReClassN))
        temp = np.reshape(PCAloading,(sWinSz[1],sWinSz[0],compN))
        plotClassDistr(temp, dReClassLabels, reclassLatCoords, colorMethod='ClassLabel', nonClustColor=[.9,.9,.9,1], classColors=None, magSig=2, figW=18, s_=10)
    else:
        reclassLatCoords = None

    Xvec = np.abs(imPchFFT.reshape(-1,np.shape(imPchFFT)[-1])).T

    ### Determine Output Image Labels & Class Averages ###
    #Class Averages
    if not (returnClass is None):
        if returnClass=='Avg':
            classAvg = getClassAvg(Xvec[:,PCAValidInd].T, dReClassLabels)
        elif returnClass=='Med':
            classAvg = getClassMed(Xvec[:,PCAValidInd].T, dReClassLabels)
        elif returnClass=='PCAinv':
            if reclassLatCoords is None:
                reclassLatCoords = ClassLatPositions(PCAloading, dReClassLabels, lblArray=np.arange(dReClassN))
            temp = np.zeros((reclassLatCoords.shape[0],componentMax))
            temp[:,:compN] = reclassLatCoords
            classAvg = pca.inverse_transform(temp).T
        else:
            raise ValueError('returnClass not recognized. Must be Avg, PCAinv, or None Type')
    else: 
        classAvg = None
    
    #Score
    #image difference to class averages
    if scoreMethod=='diff2Avg':
        if classAvg is None:
            classAvg = getClassAvg(Xvec[:,PCAValidInd].T, dReClassLabels)
        score = np.ones((dReClassLabels.size,))*np.nan
        ind = np.where(dReClassLabels>=0)[0]
        temp = np.reshape(classAvg,(-1,classAvg.shape[-1]))
        classAvgAvg = np.mean(temp,axis=0)
        temp = temp[:,dReClassLabels[ind]].T
        temp = Xvec[ind,:][:,PCAValidInd]-temp
        temp = np.sqrt(np.sum(temp**2,axis=1))
        #normalize
        temp = temp/classAvgAvg[dReClassLabels[ind]]
        score[ind] = temp
        score = score.reshape((sWinSz[1],sWinSz[0]))
        #Get best score map
        imLabel, minScore = swScoreMin(swSliceInd, score, dReClassLabels)
        imLabel = np.nan_to_num(imLabel,nan=-1).astype('int')
    #distance in latent space
    elif scoreMethod=='latDist':
        if reclassLatCoords is None:
            reclassLatCoords = ClassLatPositions(PCAloading, dReClassLabels, lblArray=np.arange(dReClassN))
        score = np.ones((dReClassLabels.size,))*np.nan
        ind = np.where(dReClassLabels>=0)[0]
        score[ind] = np.sqrt(np.sum((PCAloading[ind,:]-reclassLatCoords[dReClassLabels[ind],:])**2,axis=1))
        score = score.reshape((sWinSz[1],sWinSz[0]))
        #Get best score map
        imLabel, minScore = swScoreMin(swSliceInd, score, dReClassLabels)
        imLabel = np.nan_to_num(imLabel,nan=-1).astype('int')
    #all labels are interpolated to the image dimensions
    elif scoreMethod=='all':
        score = None
        minScore = None
        pts = np.append(imPatchOrigin[0][:,:,np.newaxis],imPatchOrigin[1][:,:,np.newaxis],axis=2)
        pts[:,:,0] = pts[:,:,0]+winSz[0]/2
        pts[:,:,1] = pts[:,:,1]+winSz[1]/2
        interp = RegularGridInterpolator((pts[0,:,0],pts[:,0,1]), np.reshape(dReClassLabels,sWinSz[[1,0]]).T, bounds_error=False, method='nearest', fill_value=-1)
        xx,yy = np.meshgrid(np.arange(imSz[0]), np.arange(imSz[1]))
        imLabel = interp((xx,yy)).astype('int')
    else:
        raise ValueError('scoreMethod not recognized. Must be Diff2Avg, LatDist, or All')

    #Class representatives
    if (not (returnClass is None)) or (not (classAvg is None)):
        classAvgIm = np.ones((fftSz[0],fftSz[1],classAvg.shape[-1]))*np.nan
        for i in range(classAvg.shape[-1]):
            temp = np.ones((fftSz))*np.nan
            temp.ravel()[PCAValidInd]=classAvg[:,i]
            classAvgIm[:,:,i] = temp.copy()
        #display
        if verbose>1:
            _,_,fig = plotImStackGridColorbars(classAvgIm, axLbls=['Class {:d}'.format(i) for i in np.arange(classAvg.shape[-1])])
            fig.set_facecolor([.75,.25,.25])
    else:
        classAvgIm = None
    
    if verbose>1:
        plotScore(np.reshape(dReClassLabels,(sWinSz[1],sWinSz[0])), score, cN=dReClassN)
        plotScore(imLabel, minScore, cN=dReClassN)
        
    if verbose>0:
        plotLabelIm(im, imLabel, cN=dReClassN)

    ### Return ###
    return imLabel, classAvgIm
