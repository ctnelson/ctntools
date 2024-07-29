#Protocol to try and automatically find unit cells based on given symmetry requirements.

#Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
#Custom Imports
from ctntools.BaseSupportFunctions.imSampling import condDownsample, getUCStack             #downsampling & creating stack of UC subimages
from ctntools.StructureAnalysis.imSymmMap import imSymmMap                                  #symmetry mapping
from ctntools.StructureAnalysis.ucFindAB import ucFindAB                                    #find basis vectors
from ctntools.StructureAnalysis.Classify import subImStackKMeans                            #classify unit cells by kmeans
from ctntools.PeakFinding.findPeaks import findPeaks                                        #peak finding protocol
from ctntools.Networks.findDefinedNbrs import nbrByDist                                     #finding closest neighbor

############################################# Support Functions ###############################
#counts the number of items in dictionary
def DictItemNum(iDict):
    n=0
    for key, value in iDict.items():
        value=np.array(value)
        n+=value.size
    return n

#gets plotted axis size
def get_ax_size(ifig,iax):
    bbox = iax.get_window_extent().transformed(ifig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= ifig.dpi
    height *= ifig.dpi
    return width, height

############################################# Main ########################################
def ucFromSymm(im, M, abUCoffset=[.5,.5], swRadiusScalar=1.1, Mwt=None, symmCalc='ZeroNormCrossCorr', alphaGuess=120, rExclScalar=.85, edgeExclScalar=1., pkThresh=0.25, pxlprUC=20, downsampleFlag=True, principlePeakMethod='max', UCstackMethod='interp', ClassNaNMethod='random', verbose=1, **kwargs):
    ### Required Input ###
    #im                                 :   2D image
    #M                  (Symmetry)      :   {Dict} Symmetries to apply. 'i' inversion, 'r' rotation, 'm' mirror. example format:{'i':0,'r':[60,120]}
    ### Optional Inputs (defaults provided) ###
    #abUCoffset         (CreateUC)      :   [2,] position in a,b space the symmetry peaks will be assigned (fractional coordinates). e.g. [.5,.5] means each point will be considered a unit cell center
    #swRadiusScalar     (Symmetry)      :   radial distance for symmetry transform (scalar applied to ab vector magnitude)
    #Mwt                (Symmetry)      :   weights for multiple symmetries to be summed (each are normalized)
    #symmCalc           (Symmetry)      :   calculation type for sliding window symmetry 'ZeroNormCrossCorr', or 'MeanAbsDiff' 
    #alphaGuess         (ab Finding)    :   guess for lattice parameter internal angle
    #rExclScalar        (Peak Finding)  :   peak finding exclusion radius (scalar applied to ab vecor magnitude)
    #edgeExclScalar     (Peak Finding)  :   exclusion zone at borders
    #pkThresh           (Peak Finding)  :   Intensity threshold
    #pxlprUC            (Downsampling)  :   target pixels per unit cell (used for downsampling, ignored if flag is False)
    #downsampleFlag     (Downsampling)  :   flag to allow automatic downsampling
    #principlePeakMethod(ab Finding)    :   'max' or 'first'. method to select fft peak for initial scale guess. (passed to ctntools function fftpeaks)
    #UCstackMethod      (CreateUC)      :   'interp', 'KDE', or 'round'. Method to map to sampling grid.
    #ClassNaNMethod     (Classify)      :   'random' or 'remove'. Method to handle nonfinite datapoints for classification. 'Random' replaces with scaled random noise
    #verbose                            :   flag to display execution details 0=None, 1=minimal status & display, 2=detailed
    
    ### Outputs ###
    #ucAvg                              :   [w,h,class#] stack of averaged unit cells
    #ucMask                             :   [w,h] mask of the inner unit cell region
    #pks                                :   [:,4] positions of unit cells [x,y,score,class#]
    #UCSymmAvg                          :   [w2,h2,symm#,class#] stack of averaged Symmetry maps. If downsampled for symmetry calcs w2<w & h2<h.
    #UCSymmMask                         :   [w2,h2] mask of the inner unit cell region
    #outDict                            :   Dictionary of parameters used. 'a','b','abOffset','M','Mwt','Mlbls','Symm','SymmMethod', 'SymmCounts', 'SymmDownscalar','SymmRadius'

    ### Initial Settings
    fsz_ = 20   #figure size (w)
    lw_ = 5     #line width

    if verbose==0:
        warnings.filterwarnings("ignore")   #common to get warnings from underutilized gpu functions (and at present a memory leak from kmeans). Bad practice but I'm suppressing them here.
    else:
        warnings.resetwarnings()
    n=DictItemNum(M)
    if Mwt is None:
        Mwt = np.ones((n,),dtype='float')  # matrix transform weights
    else:
        Mwt=np.array(Mwt)
    n=Mwt.size
    ### Initialize plots?
    if verbose==2:
        figAB, axAB = plt.subplots(1, 3, figsize=(fsz_, fsz_/3), dpi = 100)
        figAB.suptitle('Finding the Lattice Vectors')
        figSymm, axSymm = plt.subplots(1, n, figsize=(fsz_, fsz_/(n)*im.shape[0]/im.shape[1]), dpi = 100)
        figSymm.suptitle('Local Symmetry Maps')
        figSymmfit, axSymmfit = plt.subplots(1, 2, figsize=(fsz_, fsz_/2*im.shape[0]/im.shape[1]), dpi = 100)
        figSymmfit.suptitle('UC Locations from Symmetry')
        figClassify, axClassify = plt.subplots(1, 1, figsize=(fsz_/2, fsz_/2*im.shape[0]/im.shape[1]), dpi = 100)
        figClassify.suptitle('UC Classification KMeans')
    else:
        axAB = [None]*3
        axSymm = None
        axSymmfit = [None]*2
        axClassify = None

    ### Get ab lattice vectors ###
    a,b,ucFabDict = ucFindAB(im, alphaGuess=alphaGuess, pxlprUC=pxlprUC, downsampleFlag=downsampleFlag, verbose=verbose, inax=axAB, principlePeakMethod=principlePeakMethod, **kwargs)
    atmMinR = ucFabDict['dists'][0]
    
    ### find UC candidates by symmetry ###
    #Downsample?
    abmag = np.array([(a[0]**2+a[1]**2)**.5,(b[0]**2+b[1]**2)**.5])
    ucScaleGuess = np.max(abmag)            #lattice vector size (used for downsampling)
    if downsampleFlag:
        imds,ds = condDownsample(im,ucScaleGuess,pxlprUC)
        if ds.size==1:
            ds = np.tile(ds,2)
        if (verbose!=0) & (np.any(ds>1)):
            print('Downsampled by [{:d}x{:d}]'.format(ds[0],ds[1]))
    else:
        imds=im
        ds = np.array([1,1],dtype='int')
    #get sliding window symmetry
    swRad = np.max(abmag)*swRadiusScalar/ds[0]    #sliding window transform calculatoin radius
    swSymm, swCounts, Mlbls = imSymmMap(imds, M, swRad, symmCalc=symmCalc, verbose=verbose, inax=axSymm, **kwargs)
    #Weighted symmetry image sum
    temp = np.repeat(np.repeat(Mwt[np.newaxis,np.newaxis,:],swSymm.shape[0],axis=0),swSymm.shape[1],axis=1)
    swSymm_combined = np.sum(swSymm*swCounts*temp, axis=2)
    #Find peaks
    pkExclRadius=np.min(abmag/ds)*rExclScalar
    edgeExcl=np.max(abmag/ds)*edgeExclScalar
    print(atmMinR)
    print(ds)
    pkRefineWinsz = np.array([atmMinR, atmMinR])/ds/2
    pks_sp,_ = findPeaks(swSymm_combined, pFsig=1, pkExclRadius=pkExclRadius, edgeExcl=edgeExcl, iThresh=pkThresh, pkRefineWinsz=pkRefineWinsz, progressDescr='Fitting Candidate Peaks...',  inax=axSymmfit[0], verbose=verbose, **kwargs)
    
    ### Create UC stack ###
    pks_sp[:,0]=pks_sp[:,0]*ds[0]
    pks_sp[:,1]=pks_sp[:,1]*ds[1]
    if verbose==2:
        axSymmfit[0].set_title('Fit positions: Weighted Sum Symmetries')
        axSymmfit[1].imshow(im,origin='lower',cmap='gray')
        axSymmfit[1].scatter(pks_sp[:,0],pks_sp[:,1],s=10,c='k',marker='x')
        axSymmfit[1].set_title('Fit positions: Source Image')
    UCstack, UCmask = getUCStack(im, pks_sp[:,:2], a, b, abUCoffset=abUCoffset, method=UCstackMethod, verbose=verbose, **kwargs)

    ### Classify (KMeans)
    ucAvg, ucClass = subImStackKMeans(UCstack, inax=axClassify, verbose=verbose, ClassNaNMethod=ClassNaNMethod, **kwargs)
    ucn = ucAvg.shape[2]

    ### Create UC stacks & class averaged of symmetry images
    UCSymmStack, UCSymmMask = getUCStack(swSymm[:,:,0], pks_sp[:,:2]/ds, a/ds, b/ds, abUCoffset=abUCoffset, method=UCstackMethod, verbose=verbose)
    UCSymmStack = np.repeat(UCSymmStack[:,:,:,np.newaxis],n,axis=3)
    if n>1:
        for i in range(1,n):
            UCSymmStack[:,:,:,i] = getUCStack(swSymm[:,:,i], pks_sp[:,:2]/ds, a/ds, b/ds, abUCoffset=abUCoffset, method=UCstackMethod, verbose=verbose)[0]
    UCSymmAvg = np.ones((UCSymmStack.shape[0],UCSymmStack.shape[1],ucn,n))*np.NaN
    for ucni in range(ucn):
        ind = np.where(ucClass==ucni)[0]
        for ni in range(n):
            UCSymmAvg[:,:,ucni,ni] = np.nanmean(UCSymmStack[:,:,ind,ni],axis=2)

    #Display Results
    if verbose>0:
        color = plt.cm.brg(np.linspace(0, 1, ucn))
        figUCs = plt.figure(tight_layout=True,figsize=(fsz_, fsz_/(ucn+1)), dpi = 100)
        gs = gridspec.GridSpec(2, ucn+1)
        ax0 = figUCs.add_subplot(gs[:, 0])
        axUCs = [None]*ucn
        axUCsMasked = [None]*ucn
        #scatter datapoint scaling
        pkspacing = np.percentile(nbrByDist(pks_sp[:,:2])[:,0,2],50)
        axsz = get_ax_size(figUCs,ax0)                 #axis plot size
        s_ax = pkspacing/np.array(im.shape)     #spacing in percent of axis
        s_ = np.min(s_ax*axsz)
        s_ = np.max([s_**2*.9,1])               #set minimum size to 1

        ax0.set_title('UC Positions by Class')
        ax0.imshow(im,origin='lower',cmap='gray')
        for i in range(ucn):
            axUCs[i]= figUCs.add_subplot(gs[0,i+1])
            axUCsMasked[i]= figUCs.add_subplot(gs[1,i+1])
            axUCs[i].imshow(ucAvg[:,:,i],origin='lower',cmap='gray')
            axUCsMasked[i].imshow(ucAvg[:,:,i]*np.where(UCmask,UCmask,np.nan),origin='lower',cmap='gray')
            for axis in ['top','bottom','left','right']:
                axUCs[i].spines[axis].set_linewidth(lw_)
                axUCs[i].spines[axis].set_color(color[i])
            axUCs[i].set_title('Class {:d} Average UC'.format(i),color=color[i])
            ind = np.where(ucClass==i)[0]
            ax0.scatter(pks_sp[ind,0], pks_sp[ind,1], s=s_, color=color[i], marker='o',label='Class '+str(i),edgecolor=None)
            axUCsMasked[i].set_axis_off()
            #if verbose=1:
            axUCs[i].tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False)
            ax0.set_axis_off()

    warnings.resetwarnings()

    #Outputs
    pks = np.hstack((pks_sp,ucClass[:,np.newaxis]))
    outDict = { 'a':a, 'b':b, 'abOffset':abUCoffset, 'M':M, 'Mwt':Mwt, 'Mlbls':Mlbls, 'SymmMethod':symmCalc, 'Symm':swSymm, 'SymmCounts':swCounts, 'SymmDownscalar':ds, 'SymmRadius':swRad}
    
    return ucAvg, UCmask, pks, UCSymmAvg, UCSymmMask, outDict
