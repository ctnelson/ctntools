### A basic protocol to find & refine peak positions in an image ###
#1) gaussian smooth
#2) Laplace (find regions of <0 2nd derivative)
#3) Find local maxima pixels
#4) Refine peak with parabaloid fit

#Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
#Custom imports
from ctntools.BaseSupportFunctions.kernels import gKernel2D                                 #gaussian kernel used for smoothing
from ctntools.PeakFinding.imregionalmax import imregionalmax                                #coarse finds peaks by local maxima
from ctntools.PeakFinding.peakfitting import refinePeaks                                    #refines peaks by parabaloid fit

def findPeaks(inim, imask=None, pFsig=1, iThresh=-np.inf, pkExclRadius=1, edgeExcl=0, pkRefineWinsz=[3,3],  inax=None, verbose=False, **kwargs):
    ### Inputs ###
    #inim           :   input image
    #imask          :   mask out regions for peak finding
    #pFsig          :   pre-smoothing sigma
    #iThresh        :   intensity threshold
    #pkExclRadius   :   exclusion radius of intial peak search
    #edgeExcl       :   exclusion at edges
    #pkRefineWinsz  :   windowsize around initial peak for refining paraboloid fit
    #inax           :   plotting axis
    #verbose        :   flag to print execution details
    ### Outputs ###
    #pks_sp         :   refined peaks [n,3] of [x,y,value]
    #im_sm          :   smoothed image used for fitting

    #Initial
    pkExclRadius = np.ceil(np.array(pkExclRadius)).astype('int')
    edgeExcl = np.ceil(np.array(edgeExcl)).astype('int')
    pkRefineWinsz = np.ceil(np.array(pkRefineWinsz)).astype('int')
    if imask is None:
        imask = np.ones_like(inim,dtype='bool')

    #Smooth?
    if pFsig>0:
        k = gKernel2D(pFsig, rdist=np.max([pFsig*3,2]), normMethod='Sum')
        im_sm = ndimage.convolve(inim,k,mode='nearest')
    else:
        im_sm = inim

    #Masks
    #2nd derivative <0
    im_sm_L = ndimage.laplace(im_sm)
    Lsubmask = im_sm_L<0
    #Intensity
    Isubmask = im_sm>=iThresh
    #Edge
    if edgeExcl>0:
        Esubmask = np.zeros_like(inim,dtype='bool')
        Esubmask[edgeExcl:-edgeExcl,edgeExcl:-edgeExcl] = True
    else:
        Esubmask = np.ones_like(inim,dtype='bool')
    #mask = np.logical_and(Lsubmask,Esubmask)
    mask = imask & Lsubmask & Esubmask & Isubmask

    if np.any(mask.ravel()):
        #Coarse find peaks (pixel level)
        pks = imregionalmax(im_sm, pkExclRadius, localMaxRequired=False, imask=mask)[0].T
        #Peak Refine (parabaloid fit)
        pks_sp = refinePeaks(im_sm, pks[:,:2], pkRefineWinsz, verbose=verbose, **kwargs)[:,:3]
        #crop for valid & in-bounds
        ind = np.where((np.isfinite(pks_sp[:,0])) & (pks_sp[:,0]>0) & (pks_sp[:,0]<inim.shape[1]) & (pks_sp[:,1]>0) & (pks_sp[:,1]<inim.shape[0]))[0]
        pks_sp = pks_sp[ind,:]
    else:
        pks_sp = np.empty((0,3))

    if not (inax is None):
        inax.imshow(im_sm, origin='lower')
        inax.scatter([pks_sp[:,0]],pks_sp[:,1],s=10,c='k',marker='x')

    return pks_sp, im_sm
