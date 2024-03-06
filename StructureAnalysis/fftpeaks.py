#takes fft of image & finds peaks
from ctntools.PeakFinding.imregionalmax import imregionalmax
from ctntools.BaseSupportFunctions.kernels import gKernel2D
from ctntools.PeakFinding.peakfitting import refinePeaks
from ctntools.Convolution.kde1D import kde1D
from ctntools.StructureAnalysis.RadialProfile import radKDE
from scipy.ndimage import convolve
import numpy as np
import matplotlib.pyplot as plt

def fftpeaks(inim, gaussSigma = 1, subpixelfit=True, thresh=0.15, normalize=True, minRExclusionMethod = 'afterminima', principlepeakmethod = 'first', plotoutput = False, inax=[None]*3):
  #Inputs
  #inim                     :       input image
  #gaussSigma           (optional)  :   sigma for gaussian blur
  #subpixelfit          (optional)  :   flag to subpixel fit peaks as parabaloid (otherwise returns pixel maxima)
  #thresh               (optional)  :   Intensity threshold for peak finding
  #normalize            (optional)  :   flag to normalize input image
  #minRExclusionMethod  (optional)  :   'afterminima' or #  :set a minimum radius exclusion. 'afterminima' (default) considers only peaks beyond an initial minimum. Provide a numeric input to set manually.
  #principlepeakmethod  (optional)  :   'first' or 'max'    :How to select principle peak. 'first' is the longest frequency, 'max' is the largest value
  #inax                 (optional)  :   list of 3 axis to plot to. If None, no plotting. 

  #Outputs
  #rpk                      :       radius of peak value
  #xyv                      :       xy vectors of peaks
  #rdistr                   :       fft radial distribution

  #Manual Settings
  spfit_winsz = 3       #subpixel fitting window size (winsz*2+1,winsz*2+1)
  spfit_ithresh = .7    #subpixel fitting intensity threshold
  scattersz = 20        #marker size for fit positions
  kde1Dsigma = .5
  #Parameters
  inim_sz = np.array(inim.shape)
  xy0 = np.floor(inim_sz/2)
  
  #Normalize
  if normalize:
      inim = (inim - np.min(inim.ravel())) / np.ptp(inim.ravel())
      inim = inim-np.nanmean(inim.ravel())

  #FFT
  hann = np.outer(np.hanning(inim_sz[0]),np.hanning(inim_sz[1]))
  im_fft = np.abs(np.fft.fftshift(np.fft.fft2(hann*inim)))

  #smooth
  k = gKernel2D(gaussSigma,rscalar=3)
  im_fft_sm = convolve(im_fft,k,mode='nearest')

  #Radial distribution
  x, distr, density, _, _ = radKDE(im_fft_sm, rstp=.1, s=kde1Dsigma, method='interp')
  distr = distr/density
  distr = np.vstack((x,distr,density))

  #find principle peak
  distrdx = None
  if minRExclusionMethod=='afterminima':
    distrdx = np.gradient(distr[1,:])
    ind = np.argmax(distrdx<0)              
    minRind = np.argmax(distrdx[ind:]>0)+ind
  elif np.isfinite(minRExclusionMethod):
    minRind = np.round(minRExclusionMethod).astype('int')
  else:
    raise ValueError('unknown minRExclusionMethod, must be "afterminima" or a finite integer value')
  if principlepeakmethod == 'max':
    rpk = x[np.nanargmax(distr[1,minRind:])+minRind]
  elif principlepeakmethod == 'first':
    if distrdx is None:
        distrdx = np.gradient(distr[1,:])
    ind = np.argmax(distrdx[minRind:]<0)+minRind
    rpk = x[ind]  
  else:
    raise ValueError('unknown principlepeakmethod, must be "first" or "max"')

  #normalize
  im_fft_sm = (im_fft_sm - np.min(im_fft_sm))/np.ptp(im_fft_sm)

  #peaks
  xx,yy = np.meshgrid(np.arange(inim_sz[0]),np.arange(inim_sz[1]))
  dx = xx-xy0[0]
  dy = yy-xy0[1]
  r = np.sqrt(dx**2+dy**2)
  rmsk = (im_fft_sm>thresh) & (r>x[minRind])
  xy_fft = imregionalmax(im_fft_sm,rpk*.75,exclusionmode=False, insubmask=rmsk)[0]

  #refine peaks
  if subpixelfit:
     spfit_winsz = np.array([spfit_winsz,spfit_winsz],dtype='int')
     xy_fft_sp = refinePeaks(im_fft_sm, xy_fft[[0,1],:].T, winsz=spfit_winsz, ithresh=spfit_ithresh)[:,:3].T
     xy_v = xy_fft_sp.copy()
  else:
     xy_v = xy_fft.copy()
 
  #convert to real space
  dx = xy_v[0,:]-xy0[0]
  dy = xy_v[1,:]-xy0[1]
  r = np.sqrt(dx**2+dy**2)
  xy_v[0,:] = inim_sz[0]/dx
  xy_v[1,:] = inim_sz[1]/dy

  #Plot?
  rng = 1.1 * np.max(r)
  xlim_ = [np.floor(im_fft.shape[0]/2-rng), np.ceil(im_fft.shape[0]/2+rng)]
  ylim_ = [np.floor(im_fft.shape[1]/2-rng), np.ceil(im_fft.shape[1]/2+rng)]
    
  if ~(inax[0] is None):
    inax[0].imshow(im_fft,cmap='gray')
    inax[0].scatter(xy0[0],xy0[1],s=30,c='c',marker='+')
    inax[0].set_xlim(xlim_)
    inax[0].set_ylim(ylim_)
    inax[0].set_title('Image FFT')

  if ~(inax[1] is None):
    inax[1].imshow(im_fft_sm,cmap='gray')
    inax[1].scatter(xy0[0],xy0[1],s=30,c='c',marker='+')
    inax[1].scatter(xy_fft[0,:],xy_fft[1,:],s=scattersz,c='b',marker='+')
    if subpixelfit:
      inax[1].scatter(xy_fft_sp[0,:],xy_fft_sp[1,:],s=scattersz,c='r',marker='x')
    inax[1].set_xlim(xlim_)
    inax[1].set_ylim(ylim_)
    inax[1].set_title('SmoothedFFT & Peaks')

  if ~(inax[2] is None):
    ax[2].plot(x,distr[1,:],'-k')
    #inax[2].scatter(x[minRind],distr[minRind],s=50,c='b')
    #inax[2].text(x[minRind],distr[minRind],'exclusion radius',c='b',ha='left',va='top')
    inax[2].text(x[minRind],0,'exclusion radius',c='b',ha='left',va='top')
    inax[2].plot([x[minRind],x[minRind]],[0,np.nanmax(distr[1,:])],'-b')
    inax[2].scatter(x[ind],distr[1,ind],s=50,c='r')
    inax[2].text(x[ind],distr[1,ind],'principle peak',c='r',ha='left',va='bottom')
    inax[2].set_title('FFT Radial Profile')

  #Return
  if subpixelfit:
    xy_fft = xy_fft_sp

  return xy_fft, xy_v, rpk, distr, im_fft
