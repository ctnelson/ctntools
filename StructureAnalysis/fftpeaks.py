#takes fft of image & finds peaks
from ctntools.PeakFinding.imregionalmax import imregionalmax
from ctntools.BaseSupportFunctions.kernels import gKernel2D
from ctntools.PeakFinding.peakfitting import refinePeaks
from ctntools.Convolution.kde1D import kde1D
from ctntools.StructureAnalysis.RadialProfile import radKDE
from scipy.ndimage import convolve
import numpy as np
import matplotlib.pyplot as plt

def fftpeaks(inim, gaussSigma = 1, subpixelfit=True, thresh=0.15, normalize=True, plotoutput = False, plotax=None):
  #Inputs
  #inim                   :       input image
  #gaussSigma(optional)   :       sigma for gaussian blur
  #sub
  #thresh                 :       Intensity threshold for peak finding
  #normalize(optional)    :       flag to normalize input image
  #plotoutput(optional)   :       flag to plot output
  #plotax                 :       if plot=True, can define an [2,] axis to plot to. If None a new one is created. 

  #Outputs
  #rpk                    :       radius of peak value
  #xyv                    :       xy vectors of peaks
  #rhist                  :       fft mean radial histogram

  #Manual Settings
  rminexcl = 2          #set a minimum exclusion zone from zero freqeuncy to avoid strong signals from 0(non zero mean) and 1 (the hanning window)
  spfit_winsz = 3       #subpixel fitting window size (winsz*2+1,winsz*2+1)
  spfit_ithresh = .7    #subpixel fitting intensity threshold
  #Parameters
  inim_sz = np.array(inim.shape)
  xy0 = np.floor(inim_sz/2)
  
  #Normalize
  if normalize:
      inim = (inim - np.min(inim.ravel())) / np.ptp(inim.ravel())
      inim = inim-np.nanmean(inim.ravel())

  #FFT
  hann = np.outer(np.hanning(inim_sz[0]),np.hanning(inim_sz[1]))
  #im_fft = np.abs(np.fft.fftshift(np.fft.fft2(hann*inim-np.nanmean((inim*hann).ravel()))))
  im_fft = np.abs(np.fft.fftshift(np.fft.fft2(hann*inim)))

  #smooth
  k = gKernel2D(gaussSigma,rscalar=3)
  im_fft_sm = convolve(im_fft,k,mode='nearest')

  #histogram
  #rhist, rbins = radhist(im_fft, percentile=99, binwidth=1, trygpu=False)
  #rpk = rbins[np.nanargmax(rhist[rminexcl:])+rminexcl]
  x, distr, density, _, r = radKDE(im_fft_sm, rstp=.1, s=1, method='interp')
  distr = distr/density
  distrdx = np.gradient(distr)
  ind = np.argmax(distrdx<0)              
  ind = np.argmax(distrdx[ind:]>0)+ind
  rpk = np.argmax(distrdx[ind:]<0)+ind

  #normalize
  im_fft_sm = (im_fft_sm - np.min(im_fft_sm))/np.ptp(im_fft_sm)

  #peaks
  rmsk = im_fft_sm>thresh
  im_fft_xy, mxpos, pkprom, msk = imregionalmax(im_fft_sm,rpk*.75,exclusionmode=False, insubmask=rmsk)

  #refine peaks
  if subpixelfit:
     spfit_winsz = np.array([spfit_winsz,spfit_winsz],dtype='int')
     im_fft_xy_sp = refinePeaks(im_fft_sm,im_fft_xy[[0,1],:].T,winsz=spfit_winsz,ithresh=spfit_ithresh)[:,:3].T
  
  xyv = im_fft_xy.copy()
  dx = xyv[0,:]-xy0[0]
  dy = xyv[1,:]-xy0[1]
  r = np.sqrt(dx**2+dy**2)
  xyv[0,:] = inim_sz[0]/dx
  xyv[1,:] = inim_sz[1]/dy

  if plotoutput:
    rng = 1.1 * np.max(r)
    xlim_ = [np.floor(im_fft.shape[0]/2-rng), np.ceil(im_fft.shape[0]/2+rng)]
    ylim_ = [np.floor(im_fft.shape[1]/2-rng), np.ceil(im_fft.shape[1]/2+rng)]
    
    if plotax is None:
      fig, ax = plt.subplots(1, 2, figsize=(14, 7), dpi = 100)
    ax[0].imshow(im_fft,cmap='gray')
    ax[0].scatter(xy0[0],xy0[1],s=30,c='c',marker='+')
    ax[0].set_xlim(xlim_)
    ax[0].set_ylim(ylim_)
    ax[0].set_title('Image FFT')

    ax[1].imshow(im_fft_sm,cmap='gray')
    ax[1].scatter(xy0[0],xy0[1],s=30,c='c',marker='+')
    ax[1].scatter(im_fft_xy[0,:],im_fft_xy[1,:],s=5,c='r')
    if subpixelfit:
      ax[1].scatter(im_fft_xy_sp[0,:],im_fft_xy_sp[1,:],s=5,c='c')
    ax[1].set_xlim(xlim_)
    ax[1].set_ylim(ylim_)
    ax[1].set_title('SmoothedFFT & Peaks')

  return rpk, xyv, distr
