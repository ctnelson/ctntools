#Plotting Helper Functions
############################################# Contents #################################################
### Support ###
#createGridAxesWithColorbars    :    creates axes array to plot on
#plotGridAxesWithColorbars      :    plots to axes array

### Create & Plot ###
#plotImStackGridColorbars       :    overall function to create and plot images

############################################# Imports ##################################################
import numpy as np
import matplotlib.pyplot as plt

############################################# Supporting Functions ###########################################
########################################## createGridAxesWithScalebars #######################################
#Creates a grid plotting axis for array of images & separated axes for colorbars. Part of a shortcut function to display grids of images w/ scalebars
def createGridAxesWithColorbars(imStack, gridDims, figW=12, figH=None, scalebarWratio=.1, fig_wspace=.05, fig_hspace=.05, **kwargs):
    ### Inputs ###
    #imStack            :   [h,w,n]    array of scree plot values
    #gridDims           :   [r,c]      Axis grid. Must satisfy r*c > n
    ### Optional ###
    #figW               :   figure width
    #figH               :   figure height. Can be None type and will autocalculate based on aspect ratio 
    #scalebarWratio     :   width of colorbar relative to image
    #fig_wspace         :   horizontal spacing between plots
    #fig_hspace         :   vertical spacing between plots
    ### Outputs ###
    #imAx           :   [r,c] array of image axes
    #colorBarAx     :   [r,c] array of colorbar axes
    #oFig           :   figure object
    #scalebarWratio :   scalebar width relative to image

    ### Some Initial Parameters & Checks ###
    dims = np.array(imStack.shape,dtype='int')
    r = np.int32(gridDims[0])
    c = np.int32(gridDims[1])
    if dims.size==3:
        n = dims[2]
    elif dims.size==2:
        n = 1
        imStack = imStack[:,:,np.newaxis]
    else:
        raise ValueError('input image(s) must be 2 or 3 dimension')
    assert r>0
    assert c>0
    assert r*c >= n

    ### Create figure
    #dims of Figure
    wRat = np.tile((1,scalebarWratio),c)
    hRat = np.ones((r,))
    if figH is None:
        whRat = dims[0]/dims[1]
        figH = (np.sum(wRat)+(c*2+1)*fig_wspace) 
        figH = figW/figH
        figH = figH * ((np.sum(hRat)+(r+1)*fig_wspace)) * whRat
    #create figure

    gs_kw = dict(width_ratios=wRat, height_ratios=hRat, left=0, right=1, bottom=0, top=1, wspace=fig_wspace, hspace=fig_hspace)
    fig, ax = plt.subplots(nrows=r, ncols=c*2, gridspec_kw=gs_kw, figsize=(figW,figH))

    #split axes arrays
    if r==2:
        imAx = ax[:,::2]
        colorBarAx = ax[:,1::2]
    if r==1:
        imAx = ax[::2]
        colorBarAx = ax[1::2]

    #erase axes for excess
    if r*c > n:
        for i in range(n,r*c):
            imAx.ravel()[i].set_axis_off()
            colorBarAx.ravel()[i].set_axis_off()

    return imAx, colorBarAx, fig, scalebarWratio
  
############################################# plotGridAxesWithScalebars ##############################
#Plots image stack to axes grid for ims & colorbars. Part of a shortcut function to display grids of images w/ scalebars
def plotGridAxesWithColorbars(imStack, imAx, colorBarAx, scalebarIm=None, inCmap='gray', iVLims=.00, ColorBarZeroCentered=False, labelValues=[0,.5,1], scalebarWratio=.1, labelColor=[.25,.25,0], **kwargs):
    #imStack                :   array of images [h,w,n], or [h,w,4,n] (where colorbar must be explicitly provided in scalebarIm)
    #imAx                   :   [r,c]      array of image axes
    #colorBarAx             :   [r,c]      array of colorbar axes
    ### Optional ###
    #scalebarIm             :   None Type or [h,w2,4,n] array of color bar images
    #inCmap                 :   cmap for plotting. Either cmap string or None
    #iVLims                 :   plotting intensity limits. Either single value used for percentile, or [n,2] array of min-max values per plot. Set to 0 to use min/max
    #ColorBarZeroCentered   :   Flag for colorbar symmetric around zero
    #labelValues            :   flag to label values. Either None type or array of fractional values (of colorbar range 0->1)
    #scaleBarWratio         :   width of scale bar (relative to image width)

    ### Outputs ###
    #None  

    colorbarSteps = 128

    ### Some Initial Parameters & Checks ###
    dims = np.array(imStack.shape,dtype='int')
    if dims.size==3:
        #Determine vLims?
        iVLims = np.array(iVLims,dtype='float')
        if iVLims.size==1:
            Vec = np.reshape(imStack,(-1,dims[-1]))
            #use min/max
            if iVLims==0:
                vLims = np.vstack((np.nanmin(Vec,axis=0),np.nanmax(Vec,axis=0))).T
            #use percentile
            elif ((iVLims>0) and (iVLims<=100)):
                vLims = np.vstack((np.nanpercentile(Vec,iVLims,axis=0),np.nanpercentile(Vec,100-iVLims,axis=0))).T
            else:
                raise ValueError('iVLims value not recognized. Must be [n,2] array or single value between 0-100 for percentile function')
        elif ((iVLims.shape[0]==dims[-1]) and (iVLims.shape[1]==2)):
            vLims=iVLims
        else:
            raise ValueError('iVLims value not recognized. Must be [n,2] array or single values for percentile function')
        #symmetric about zero?
        if ColorBarZeroCentered:
            mx = np.max(np.abs(vLims),axis=1)
            vLims = np.vstack((-mx,mx)).T
            
    elif dims.size==4:
        assert not (scalebarIm is None)
        assert scalebarIm.shape[-1]==dims[-1]
        assert imStack.shape[2]==4
        vLims = None
    else:
        raise ValueError('image stack dimensions not recognized. Should be 3 or 4 dimensional')

    ### Plots ###
    if dims.size==3:
        scalebarIm = np.linspace(0,1,colorbarSteps)
        scalebarW = np.ceil(colorbarSteps*scalebarWratio).astype('int')
        scalebarIm = np.repeat(scalebarIm[:,np.newaxis],scalebarW,axis=1)
        for i in range(dims[-1]):
            imAx.ravel()[i].imshow(imStack[:,:,i], cmap=inCmap, origin='lower', vmin=vLims[i,0], vmax=vLims[i,1])
            colorBarAx.ravel()[i].imshow(scalebarIm*(vLims[i,1]-vLims[i,0])+vLims[i,0], cmap=inCmap, origin='lower', vmin=vLims[i,0], vmax=vLims[i,1])
            imAx.ravel()[i].set_axis_off()
            colorBarAx.ravel()[i].set_axis_off()
    elif dims.size==4:
        for i in range(dims[-1]):
            imAx.ravel()[i].imshow(imStack[:,:,:,i], origin='lower')
            colorBarAx.ravel()[i].imshow(scalebarIm[:,:,:,i], origin='lower')
            imAx.ravel()[i].set_axis_off()
            colorBarAx.ravel()[i].set_axis_off()

    #Label colorbar?
    if not (labelValues is None):
        labelValues = np.array(labelValues,dtype='float')
        assert labelValues.size>0
        for i in range(dims[-1]):
            for j in range(labelValues.size):
                if labelValues[j]==0:
                   va_='top'
                else:
                   va_='bottom'
                lbl_ = labelValues[j]*(vLims[i,1]-vLims[i,0])+vLims[i,0]
                colorBarAx.ravel()[i].text(scalebarIm.shape[1]/2-.5,scalebarIm.shape[0]*labelValues[j]-.5, '{:.2f}'.format(lbl_), ha='center', va=va_, color=labelColor)
                colorBarAx.ravel()[i].plot([-.5,scalebarIm.shape[1]-.5],[scalebarIm.shape[0]*labelValues[j]-.5, scalebarIm.shape[0]*labelValues[j]-.5], '-', color=labelColor)

    return vLims

################################################ Plotting Functions ################################################
############################################# plotImStackGridColorbars #############################################
#overall function to create and plot images
def plotImStackGridColorbars(imStack, gridDims=None, axLbls=None, **kwargs):
    ### Inputs ###
    #imStack        :   [h,w,n]    array of scree plot values
    #gridDims       :   [r,c]      Axis grid. Must satisfy r*c > n

    ### Outputs ###
    #imAx           :   [r,c] array of image axes
    #colorBarAx     :   [r,c] array of colorbar axes
    #oFig           :   figure object

    ### kwargs for subfunctions ###
    #figW                   :   figure width
    #figH                   :   figure height. Can be None type and will autocalculate based on aspect ratio 
    #scalebarWratio         :   width of colorbar relative to image
    #fig_wspace             :   horizontal spacing between plots
    #fig_hspace             :   vertical spacing between plots
    #scalebarIm             :   None Type or [h,w2,4,n] array of color bar images
    #inCmap                 :   cmap for plotting. Either cmap string or None
    #iVLims                 :   plotting intensity limits. Either single value used for percentile, or [n,2] array of min-max values per plot. Set to 0 to use min/max
    #ColorBarZeroCentered   :   Flag for colorbar symmetric around zero
    #labelValues            :   flag to label values. Either None type or array of fractional values (of colorbar range 0->1)

    #######
    if gridDims is None:
        gridDims = [1,imStack.shape[-1]]

    #Create Fig/Axes
    imAx, colorBarAx, fig, scaleBarWratio = createGridAxesWithColorbars(imStack, gridDims, **kwargs)
    #Plot Images & Colorbars
    _=plotGridAxesWithColorbars(imStack, imAx, colorBarAx, scalebarWratio=scaleBarWratio, **kwargs)

    #Labels
    if not (axLbls is None):
        for i in range(imStack.shape[-1]):
            imAx.ravel()[i].set_title(axLbls[i])

    return imAx, colorBarAx, fig
