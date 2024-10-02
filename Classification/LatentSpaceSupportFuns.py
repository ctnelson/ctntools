#Simple Functions Designed for Analysis or Plotting of Latent Space Data
############################################################ Contents ################################################
### Calculations ###
#ClassLatPositions    :    Retuns class median values                                    (design intent was for latent space)
#getDists2Refs        :    Returns distances of data to reference class positions        (design intent was for latent space)
#getRefNormDist       :    std normalized distances of data to reference class positions (design intent was for latent space)

### Plotting ###
#plotPCA              :    plots *PCA Results of Image Patches (a scree plot, components, and score maps). *or similar dim reduction
#plotPCAlatent        :    plots latent joint distributions in real & latent spaces using a perceptually uniform colorscheme
#plotClassDistr       :    class colorized plots of latent joint distributions in real & latent spaces

############################################################ Imports #################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from papuc.example_maps import colormaps

##########################################################################################################################
################################################   Calculation Funs   ####################################################
##########################################################################################################################

###############################################   ClassLatPositions   ####################################################
# get class latent positions
def ClassLatPositions(inData, inDataLbls, lblArray=None, method='Median'):
    ### Inputs ###
    #inData     :   [m,n]   input data coordinates (m points, n dimensions)
    #inDataLbls :   [m,]    data labels
    #lblArray   :   [v,]    array of labels (will default to arange(max))
    #method     :   'Median' (default) or 'Mean'

    ### Outputs ###
    #classPos   :   [v,n]   class median

    #some value checks
    assert np.ndim(inData)==2
    dims_ = np.array(inData.shape,dtype='int')
    if lblArray is None:
        lblMax = np.max(inDataLbls).astype('int')
        lblArray = np.arange(lblMax)
        classPos = np.zeros((lblMax,dims_[1]))
    else:
        classPos = np.zeros((lblArray.size,dims_[1]))

    for i in lblArray:
        ind = np.where(inDataLbls==i)[0]
        if method=='Median':
            classPos[i,:] = np.nanpercentile(inData[ind,:],50,axis=0)      #median
        elif method=='Mean':
            classPos[i,:] = np.nanmean(inData[ind,:],axis=0)               #mean
        else:
            raise ValueError('method not recognized. Must be Median or Mean')

    return classPos

###############################################   getDists2Refs   ################################################
#get Distance of Datapoints
def getDists2Refs(inCoords, refCoords):
    ### Inputs ###
    #inCoords   :   [m,n] datapoint coordinates (m points, n dims)
    #refCoords  :   [v,n] reference point coordinates (v reference points, n dims)
    ### Outputs ###
    #dist       :   [m,v] distance to each reference point
    #delta      :   [m,v,n] delta to each reference point 

    #Some value checks
    assert np.ndim(inCoords)==2
    assert np.ndim(refCoords)==2 
    iDims = np.array(inCoords.shape,dtype='int')
    rDims = np.array(refCoords.shape,dtype='int')
    assert iDims[1] == rDims[1]

    #get data distance to Ref positions
    temp1 = np.repeat(inCoords[:,np.newaxis,:], rDims[0], axis=1)
    temp2 = np.repeat(refCoords[np.newaxis,:,:], iDims[0], axis=0)
    delta = temp1 - temp2
    dist = np.sqrt(np.sum((delta)**2,axis=2))

    return dist, delta

###############################################   getRefNormDist   ################################################
#get std normalized distances to reference points
def getRefNormDist(inData, inDataLbls, inRefCoords, dDeltas2Ref=None, lblArray=None):
    ### Inputs ###
    #inData                 :   [m,n]   data coordinates (m points, n dimensions)
    #inDataLbls             :   [m,]    data labels
    #inRefCoords            :   [v,n]   reference point coordinates (v reference points, n dimensions)
    #dDeltas2Ref (optional) :   [m,v]   if already pre-calculated, distance of data to reference points (m points, v ref points)
    #lblArray (optional)    :   [v,]    array of labels
    
    ### Outputs ###
    #normDist               :   [m,v] distance to each reference point
    #normDelta              :   [m,v,n] delta to each reference point   
    #deltaStd               :   [v,n] standard deviation of each reference point 

    #some value checks
    assert np.ndim(inData)==2
    assert np.ndim(inRefCoords)==2
    dDims = np.array(inData.shape,dtype='int')
    rDims = np.array(inRefCoords.shape,dtype='int')
    assert dDims[1] == rDims[1] 
    if lblArray is None:
        lblMax = np.max(inDataLbls).astype('int')
        lblArray = np.arange(lblMax)
    if dDeltas2Ref is None:
        dDeltas2Ref = getDists2Refs(inData, inRefCoords)[1]

    normDist = np.ones((dDims[0],rDims[0]),dtype='float')*np.nan
    deltaStd = np.ones((rDims[0],rDims[1]))
    for i in lblArray:
        ind = np.where(inDataLbls==i)[0]
        deltaStd[i,:] = np.nanstd(dDeltas2Ref[ind,i,:],axis=0)  #get stdev of each axis
        stdInd = np.where(deltaStd[i,:]>0)[0]
        normDelta = dDeltas2Ref[:,i,:][:,stdInd] / np.repeat(deltaStd[i,stdInd][np.newaxis,:],dDims[0],axis=0)
        normDist[:,i] = np.sqrt(np.sum((normDelta)**2,axis=1))

    return normDist, normDelta, deltaStd
##########################################################################################################################
#####################################################   Plotting   #######################################################
##########################################################################################################################

#####################################################   plotPCA   ########################################################
#Plots PCA Results of Image Patches (a scree plot, components, and score maps)
def plotPCA(inScree, inComponents, inComponentMap, figW=18, figH=None, scalebarWratio=.1, screeWratio=2, figMargins=.1, fig_wspace=.05, fig_hspace=.05):
    #inScree        :   [n,]    array of scree plot values
    #inComponenets  :   [w,h,n] array of components  
    #inComponentMap :   [a,b,n] array of componnent score maps 

    ### Outputs ###
    #axScree        :   scree plot axis
    
    #dimensions
    assert inComponents.ndim==3
    dimsComp = np.array(inComponents.shape)
    dimsCompMap = np.array(inComponentMap.shape)
    n=dimsComp[2]
    whRatComp = dimsComp[1]/dimsComp[0]
    whRatCompMap = dimsCompMap[1]/dimsCompMap[0]
    wRat = np.hstack((screeWratio,np.tile((1,scalebarWratio),n)))

    if figH is None:
        figH = (np.sum(wRat)+n*2*fig_wspace) 
        figH = figW/figH
        figH = figH * (1/whRatComp + 1/whRatCompMap)
    fig = plt.figure(figsize=(figW,figH))
    gs = fig.add_gridspec(2, n*2+1,  width_ratios=wRat, height_ratios=(1, 1), left=figMargins, right=1-figMargins, bottom=figMargins, top=1-figMargins, wspace=fig_wspace, hspace=fig_hspace)

    #Scalebar images
    compScalebarIm = np.arange(0,dimsComp[0])/dimsComp[0]*2-1
    r = np.ceil(dimsComp[1]*scalebarWratio*whRatComp).astype('int')
    compScalebarIm = np.repeat(compScalebarIm[:,np.newaxis],r,axis=1)
    compMapScalebarIm = np.arange(0,dimsCompMap[0])/dimsCompMap[0]*2-1
    r = np.ceil(dimsCompMap[1]*scalebarWratio*whRatCompMap).astype('int')
    compMapScalebarIm = np.repeat(compMapScalebarIm[:,np.newaxis],r,axis=1)

    fig.suptitle('PCA Plots')

    #Scree plot
    axScree = fig.add_subplot(gs[:, 0])
    axScree.plot(inScree,'-k')
    axScree.set_title('Scree Plot')
    axScree.set_xlabel('# Components')
    axScree.set_ylabel('Explained Variance')

    for i in range(n):
        #components
        mag = np.nanmax(np.abs(inComponents[:,:,i].ravel()))
        ax = fig.add_subplot(gs[0, i*2+1])
        ax.imshow(inComponents[:,:,i], origin='lower', cmap='RdBu_r', vmin=-mag, vmax=mag)
        ax.set_axis_off()

        #labels
        #ax.set_title('#{:d}'.format(i))
        ax.text(dimsComp[1]/2,dimsComp[0], '#{:d}'.format(i), ha='center', va='bottom')
        if i==0:
            ax.text(0, dimsComp[0]/2, 'Components', rotation='vertical', ha='right', va='center')

        #components scalebars
        ax = fig.add_subplot(gs[0, i*2+2])
        ax.imshow(compScalebarIm*mag,origin='lower', cmap='RdBu_r', vmin=-mag, vmax=mag, aspect='auto')
        ax.plot([0,compScalebarIm.shape[1]], [compScalebarIm.shape[0]/2,compScalebarIm.shape[0]/2],'-k')
        ax.set_axis_off()

        #component maps
        mag = np.nanmax(np.abs(inComponentMap[:,:,i].ravel()))
        ax = fig.add_subplot(gs[1, i*2+1])
        ax.imshow(inComponentMap[:,:,i], origin='lower', cmap='RdBu_r', vmin=-mag, vmax=mag)
        ax.set_axis_off()

        #labels
        if i==0:
            ax.text(0, dimsCompMap[0]/2, 'Latent Maps', rotation='vertical', ha='right', va='center')

        #component map scalebars
        ax = fig.add_subplot(gs[1, i*2+2])
        ax.imshow(compMapScalebarIm*mag,origin='lower', cmap='RdBu_r', vmin=-mag, vmax=mag, aspect='auto')
        ax.plot([0,compMapScalebarIm.shape[1]], [compMapScalebarIm.shape[0]/2,compMapScalebarIm.shape[0]/2],'-k')
        ax.set_axis_off()

    return axScree
  
###################################################   plotPCAlatent   ########################################################
#Plots latent joint distributions in real & latent spaces
def plotPCAlatent(inLat, magSig=2, figW=18, s_=10):
    #inLat          :   [w,h,n]  array of Latent Coordinates
    #magSig         :   distance to maximum color saturation (in units of std) 

    my_map = colormaps['default']
    n = inLat.shape[-1]
    pairN = np.int32(n*(n-1)/2)
    normVals = np.reshape(inLat,(-1,inLat.shape[-1]))
    normVals = np.nanstd(normVals,axis=0)
    if pairN==1:
        fig, ax = plt.subplots(1, 2, figsize=(figW,figW/2))
    else:
        figH = figW/pairN*2
        fig, ax = plt.subplots(2, pairN, figsize=(figW,figH))

    ii=0
    for i in range(n):
        for j in range(i+1,n):
            x=inLat[:,:,i]/normVals[i]
            y=inLat[:,:,j]/normVals[j]
            r = (x**2 + y**2)**.5
            magnitude_norm = r/magSig
            angle = np.arctan2(y,x)
            c_ = np.ones((x.shape[0],x.shape[1],4))
            c_[:,:,:3] = my_map(angle, np.clip(magnitude_norm,0,1))

            #Plots
            if pairN==1:
                #Real Space Image
                ax[0].imshow(c_,origin='lower')
                ax[0].set_title('[{:d},{:d}] Joint Loading Map'.format(i,j))
                #Plot Latent Space
                ax[1].scatter(inLat[:,:,i].ravel(),inLat[:,:,j].ravel(),color=np.reshape(c_,(-1,c_.shape[-1])),s=s_)
                ax[1].set_title('[{:d},{:d}] Joint Latent Distribution'.format(i,j))
                ax[1].set_xlabel('PCA component {:d}'.format(i))
                ax[1].set_ylabel('PCA component {:d}'.format(j))
            else:
                #Real Space Image
                ax[0,ii].imshow(c_,origin='lower')
                ax[0,ii].set_title('[{:d},{:d}] Joint Loading Map'.format(i,j))
                #Plot Latent Space
                ax[1,ii].scatter(inLat[:,:,i].ravel(),inLat[:,:,j].ravel(),color=np.reshape(c_,(-1,c_.shape[-1])),s=s_)
                ax[1,ii].set_title('[{:d},{:d}] Joint Latent Distribution'.format(i,j))
                ax[1,ii].set_xlabel('PCA component {:d}'.format(i))
                ax[1,ii].set_ylabel('PCA component {:d}'.format(j))
            ii+=1

###################################################   plotClassDistr   ########################################################
#Plots class colorized latent joint distributions in real & latent spaces
def plotClassDistr(inDataLat, inClassLabels, inClassLat, colorMethod='ClassLabel', nonClustColor=[.9,.9,.9,1], classColors=None, magSig=2, figW=18, s_=10):
    ### Inputs ###
    #inDataLat      :   [w,h,n] array of Latent Coordinates
    #inClassLabels  :   [w,h]   array of Class index / labels. Non-labeled / noise points are expected to be -1
    #inClassLat     :   [m,n]   array of Class Latent Positions
    ### Inputs (optional)
    #colorMethod    :   'ClassLabel', or 'LatentPos'
    #nonClustColor  :   [4,]    color of non-indexed points
    #classColors    :   [n,4]   color of classes (inClassLabels should index this colormap). If None type, a jet colormap is used
    #magSig         :   distance to maximum color saturation (in units of std) 
    #figW           :   figure Width
    #s_             :   latent scatter size

    #General
    cN = inClassLat.shape[0]
    n = inDataLat.shape[-1]
    pairN = np.int32(n*(n-1)/2)

    #Colormap
    if colorMethod=='ClassLabel':
        if classColors is None:
            classColors = plt.cm.jet(np.linspace(0, 1, cN))
            classColors = np.append(classColors,np.array([nonClustColor]),axis=0)
    elif colorMethod=='LatentPos':
        my_map = colormaps['default']
        normVals = np.reshape(inDataLat,(-1,inDataLat.shape[-1]))
        normVals = np.nanstd(normVals,axis=0)
    else:
        raise ValueError('ClassLabel variable not recognized, should be either ClassLabel or LatentPos')
    
    #Create Figure    
    if pairN==1:
        fig, ax = plt.subplots(1, 2, figsize=(figW,figW/2))
    else:
        if colorMethod=='ClassLabel':
            figH = figW/(pairN+1)
            fig, ax = plt.subplots(1, pairN+1, figsize=(figW,figH))
        elif colorMethod=='LatentPos':
            figH = figW/pairN*2
            fig, ax = plt.subplots(2, pairN, figsize=(figW,figH))

    #Loop through pairs
    ii=0
    for i in range(n):
        for j in range(i+1,n):
            if colorMethod=='LatentPos':
                x=inClassLat[:,i]/normVals[i]
                y=inClassLat[:,j]/normVals[j]
                r = (x**2 + y**2)**.5
                magnitude_norm = r/magSig
                angle = np.arctan2(y,x)
                classColors = np.ones((cN,4))
                classColors[:,:3] = my_map(angle, np.clip(magnitude_norm,0,1))
                classColors = np.append(classColors,np.array([nonClustColor]),axis=0)
            c_ = classColors[inClassLabels.ravel(),:]
            c_ = np.reshape(c_,(inDataLat.shape[0],inDataLat.shape[1],4))

            #axis
            if pairN==1:
                ind0=0
                ind1=1
            else:
                if colorMethod=='LatentPos':
                    ind0 = [0,ii]
                    ind1 = [1,ii]
                elif colorMethod=='ClassLabel':
                    ind0 = 0
                    ind1 = ii+1
            #Plots
            #Real Space Image
            if (colorMethod=='LatentPos') | (ii==0):
                ax[ind0].imshow(c_,origin='lower')
                ax[ind0].set_title('[{:d},{:d}] Joint Loading Map'.format(i,j))
            #Plot Latent Space
            #All Datapoints
            ax[ind1].scatter(inDataLat[:,:,i].ravel(), inDataLat[:,:,j].ravel(), color=np.reshape(c_,(-1,c_.shape[-1])), s=s_, zorder=0)
            #Class Centers
            ax[ind1].scatter(inClassLat[:,i], inClassLat[:,j], color=classColors[:inClassLat.shape[0],:], s=300, zorder=1, marker='d', edgecolor='k')
            for k in range(cN):
                txt = ax[ind1].text(inClassLat[k,i], inClassLat[k,j], k, color='k', ha='center', va='center')
                txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
            #Graph Labels
            ax[ind1].set_title('[{:d},{:d}] Class'.format(i,j))
            ax[ind1].set_xlabel('PCA component {:d}'.format(i))
            ax[ind1].set_ylabel('PCA component {:d}'.format(j))

            ii+=1
