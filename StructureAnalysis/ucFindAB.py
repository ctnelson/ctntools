### Searches for periodic a- & b-vectors in an image ###
#abGuessFromScoredPeaks  :  function used by 'ucFindAB' to select basis vectors from candidates
#ucFindAB                :  Protocol to find basis vectors. 1) find fft peaks, 2) bounded 'autocorrelation', 3) peak finding & scoring

#Imports
import numpy as np
import matplotlib.pyplot as plt
#Custom Imoports
from ctntools.StructureAnalysis.fftPeaks import fftPeaks                                    #performs FFT and finds peaks
from ctntools.StructureAnalysis.RadialProfile import radKDE                                 #Reduces 2D to a radial profile
from ctntools.SlidingWindow.slidewin_imdiff import slidewin_imdiff                          #a sliding window self-similarity analysis
from ctntools.SlidingWindow.slidewin_rotdiff import angarray_rotdiff                        #looks for rotation symmetry
from ctntools.BaseSupportFunctions.imSampling import condDownsample

### Notes ### replace with findPeaks?
from ctntools.PeakFinding.imregionalmax import imregionalmax
from ctntools.PeakFinding.peakfitting import refinePeaks


################################## select basis vectors for an xy list of scored candidates #############################
#attempts to guess ab vectors as selections from an array of candidate peaks with associated scores
def abGuessFromScoredPeaks(pks, xy0=np.zeros((2,)), alphaGuess=90, rexcl=0, rGuess=None, aOrientTarget=0, awt = [.75,0.,.25], bwt=[1/3,1/3,1/3], normalize=1, **kwargs):
    ### Inputs ###:
    #pks                        :   [m,3] or [m,5] array of peaks [x,y,score] or [x,y,score,rmag,rang]
    #xy0            (optional)  :   center point (defaults to [0,0])
    #alphaGuess     (optional)  :   target for internal angle (deg)
    #rexcl          (optional)  :   minimum exclusion radius
    #rGuess         (optional)  :   guess for a and b magnitude
    #aOrientTarget  (optional)  :   target for a-vector orientation (deg)
    #awt            (optional)  :   a-vector weights for [score, rmag, orientation]
    #bwt            (optional)  :   b-vector weights for [score, rmag, internalangle (alpha)]
    #normalize      (optional)  :   normalize? 0=no, 1=by max only, 2=max & min (scales to 0-1 range)
    ### Outputs ###:
    #a                          :   a vector index
    #b                          :   b vector index
    #ascore                     :   the scores as a-vector of all candidates
    #bscore                     :   the scores as b-vector of all candidates

    #Set setup
    alphaGuess = np.deg2rad(alphaGuess)         #convert to radians
    aOrientTarget = np.deg2rad(aOrientTarget)

    #add radial information
    if pks.shape[1]==3:
        pks = np.hstack((pks,(((pks[:,0]-xy0[0])**2 + (pks[:,1]-xy0[1])**2)**.5)[:,np.newaxis]))                #add radius
        pks = np.hstack((pks,np.arctan2(pks[:,1]-xy0[0],pks[:,0]-xy0[1])[:,np.newaxis]))                        #add angle

    #normalize score
    if normalize==1:
        pksA = 1-(pks[:,2]/np.max(pks[:,2]))
    elif normalize==2:
        pksA = 1-(pks[:,2]-np.min(pks[:,2]))/np.ptp(pks[:,2])
    
    #min radius exclusion
    ind = np.where(pks[:,3]<rexcl)[0]

    #a vector score
    angdelta = np.min(np.abs(np.vstack((pks[:,4]-aOrientTarget,pks[:,4]-(aOrientTarget+2*np.pi)))),axis=0)
    angdelta = angdelta/np.max(np.abs(angdelta))
    if rGuess is None:
        ascore = awt[2]*np.abs(angdelta) + awt[0]*pksA
    else:
        pksR = np.max(np.vstack((np.abs((pks[:,3]-rGuess)/rGuess),np.zeros_like(pks[:,3]))),axis=0)
        ascore = awt[1]*pksR + awt[2]*np.abs(angdelta) + awt[0]*pksA
    ascore[ind] = np.nan
    aind = np.nanargmin(ascore) #a vector selection
    #a=pks[aind,:2]-xy0

    #b vector score
    if rGuess is None:
        rGuess = pks[aind,3]
    pksR = np.max(np.vstack((np.abs((pks[:,3]-rGuess)/rGuess),np.zeros_like(pks[:,3]))),axis=0)
    angdelta = np.min(np.abs(np.vstack((pks[:,4]-pks[aind,4]-alphaGuess,pks[:,4]-(pks[aind,4]-alphaGuess+2*np.pi)))),axis=0)
    angdelta = angdelta/np.max(np.abs(angdelta))
    bscore = bwt[1]*pksR + bwt[2]*(angdelta) + bwt[0]*pksA
    bscore[ind] = np.nan
    bscore[aind] = np.nan
    bind = np.nanargmin(bscore) #b vector selection
    #b=pks[bind,:2]-xy0

    return aind, bind, ascore, bscore

################################## Protocol to find ab basis vectors #############################
#Searches for periodic a- & b-vectors in an image
def ucFindAB(im, imMask=None, ucScaleGuess=None, swUCScalar=4.1, pxlprUC=20, downsampleFlag=True, rExclScalar = .75, alphaGuess=None, inax=[None]*3, verbose=0, **kwargs):
    ### Inputs ###
    #im                         :   input image
    #imMask         (optional)  :   mask im (only true region will be analyzed)
    #ucScaleGuess   (optional)  :   #estimate of the unit cell size (if not provided will initially estimate via FFT)
    #swUCScalar     (optional)  :   #scalar # of unit cells bounds to perform the sliding window "autocorrelation"
    #pxlprUC        (optional)  :   target pixels / UC for downsampling.
    #downsampleFlag (optional)  :   downsample? flag (will only happen if oversampled relative to 'pxlprUC' by at least a factor of 2)
    #rExclScalar
    #alphaGuess
    #verbose        (optional)  :   print execution details, 0=silent (no plots), 1=basic plots, 2=detailed
    #inax           (optional)  :   axes to plot on [3x]

    #initial settings
    radKDE_stp = .1

    #set outside mask to NaN
    if not (imMask is None):    
        im = np.where(imMask,im,np.nan)

    #Initial Unit Cell size estimate (if not provided estimate from FFT)
    if ucScaleGuess is None: 
        ucScaleGuess = fftPeaks(im, inax=inax, **kwargs)[2]     #FFT
        if verbose==2:
            print('Initial scale estimate by FFT: {:.2f}'.format(ucScaleGuess))
    ucScaleGuess = np.ceil(np.array(ucScaleGuess)).astype('int')
    if ucScaleGuess.size==1:
        ucScaleGuess = np.tile(ucScaleGuess,2)
    assert(ucScaleGuess.size==2)

    #Downsample?
    if downsampleFlag:
        im,ds = condDownsample(im,ucScaleGuess,pxlprUC)
        if ds.size==1:
            ds = np.tile(ds,2)
        if (verbose!=0) & (np.any(ds>1)):
            print('Downsampled by [{:d}x{:d}]'.format(ds[0],ds[1]))
    else:
        ds = np.array([1,1],dtype='int')
    xyscale=ds/np.min(ds)

    #Get lattice vector & range estimates
    rangeGuess = np.ceil(ucScaleGuess*swUCScalar/np.min(ds)).astype('int')

    #Real space self-similarity (a sliding window mean abs difference)
    swImDiff = slidewin_imdiff(im,rangeGuess)                       #Sliding Window Image difference
    imsz = rangeGuess*2+1

    #Normalize (replaces 0 center point with a nearest neighbor average, normalize in 0-1 range, inverts)
    xy0 = [np.floor(swImDiff.shape[1]/2).astype('int'),np.floor(swImDiff.shape[0]/2).astype('int')]
    swImDiff[xy0[0],xy0[1]] = (swImDiff[xy0[0]-1,xy0[1]] + swImDiff[xy0[0],xy0[1]-1] + swImDiff[xy0[0]+1,xy0[1]] + swImDiff[xy0[0],xy0[1]+1])/4
    swImDiff = (swImDiff - np.nanmin(swImDiff.ravel())) / (np.nanmax(swImDiff.ravel())-np.nanmin(swImDiff.ravel()))
    swImDiff = 1-swImDiff

    #Display progress
    if not (inax[0] is None):
        inax[0].clear() 
        inax[0].imshow(swImDiff,cmap='gray',origin='lower',vmin=np.nanmin(swImDiff.ravel()),vmax=np.nanmax(swImDiff.ravel()))
        inax[0].set_title('Image sliding window MeanAbsDiff')

    #Radial distribution
    x, distr, density, _, _ = radKDE(swImDiff, rstp=radKDE_stp, method='interp', xyscale=xyscale)
    distr = distr/density
    distr = np.vstack((x,distr,density))
    #Find first minima
    distrdx = np.gradient(distr[1,:])
    ind = np.argmax(distrdx<0)              
    minRind = np.argmax(distrdx[ind:]>0)+ind
    rdistmin=x[minRind]
    #Find Max peak after minima
    ppkind = np.nanargmax(distr[1,minRind:])+minRind
    ppkDist = x[ppkind]
    ppkDist = np.tile(ppkDist,2)

    exclDist = ppkDist[0]*rExclScalar   #peak finding exclusion radius

    #Display progress
    if not (inax[1] is None):
        inax[1].clear()
        inax[1].set_aspect('auto')
        inax[1].plot(distr[0,:], distr[1,:],'-k')
        inax[1].plot([x[minRind],x[minRind]],[0,np.nanmax(distr[1,:])],'-c')
        inax[1].text(x[minRind],np.nanmax(distr[1,:]),'min radius',c='c',ha='left',va='top',rotation='vertical')
        inax[1].plot([exclDist,exclDist],[0,np.nanmax(distr[1,:])],'-y')
        inax[1].text(exclDist,np.nanmax(distr[1,:]),'peakfind exclusion',c='y',ha='left',va='top',rotation='vertical')
        inax[1].scatter(x[ppkind],distr[1,ppkind],s=50,c='r')
        inax[1].set_title('swMAD Radial distribution')

    #Find Peaks
    #Peaks (pixel)
    xx,yy = np.meshgrid(np.arange(swImDiff.shape[0]),np.arange(swImDiff.shape[1]))
    xx = xx-xy0[0]
    yy = yy-xy0[1]
    r = np.sqrt(xx**2+yy**2)
    rmsk = r>exclDist
    #rmsk = (swImDiff>ithresh) & (r>exclDist)
    pks,_ = findPeaks(swImDiff, imask=rmsk, pkExclRadius=exclDist, edgeExcl=rdistmin, pkRefineWinsz=np.array([rdistmin, rdistmin])/2, progressDescr='Fitting swImDIff Peaks...', verbose=verbose, **kwargs)

    #select a- and b-vector candidates
    if alphaGuess is None:
        #Angular self similarity
        _, rotpks = angarray_rotdiff(swImDiff, inax=inax[2])
        alphaGuess = np.rad2deg(rotpks[0,0])
        if not (inax[2] is None):
            inax[2].set_title('swMAD Angular Self-Similarity')

    a, b, ascore, bscore = abGuessFromScoredPeaks(pks, xy0=xy0, alphaGuess=alphaGuess, rexcl=x[minRind], rGuess=ppkDist[0], **kwargs)
    a = (pks[a,:2]-xy0)
    b = (pks[b,:2]-xy0)

    #plot
    if not (inax[0] is None):
        inax[0].plot([xy0[0],xy0[0]+a[0]],[xy0[1],xy0[1]+a[1]],'-b')
        inax[0].text(xy0[0]+a[0],xy0[1]+a[1],'a-vector',ha='left',c='b')
        inax[0].plot([xy0[0],xy0[0]+b[0]],[xy0[1],xy0[1]+b[1]],'-r')
        inax[0].text(xy0[0]+b[0],xy0[1]+b[1],'b-vector',ha='right',va='bottom',c='r')
        tt = np.linspace(0,2*np.pi,100)
        inax[0].plot(xy0[0]+np.sin(tt)*rdistmin, xy0[1]+np.cos(tt)*rdistmin, '-c',alpha=.25)
        inax[0].plot(xy0[0]+np.sin(tt)*exclDist,xy0[1]+np.cos(tt)*exclDist,'-y',alpha=.25)
        inax[0].plot([rdistmin,imsz[1]-rdistmin,imsz[1]-rdistmin,rdistmin,rdistmin],[rdistmin,rdistmin,imsz[0]-rdistmin,imsz[0]-rdistmin,rdistmin],'-y',alpha=.25)
        inax[0].text(rdistmin,rdistmin,'edge exclusion',c='y',ha='left',va='top',alpha=.25)

    #adjust for downsampling
    a = a*ds
    b = b*ds
    rdistmin=rdistmin*np.min(ds)

    return a, b, rdistmin