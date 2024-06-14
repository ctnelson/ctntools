#takes fft of image & finds peaks
from ctntools.PeakFinding.imregionalmax import imregionalmax
from ctntools.BaseSupportFunctions.kernels import gKernel2D
from ctntools.PeakFinding.peakfitting import refinePeaks
from ctntools.Convolution.kde1D import kde1D
from ctntools.StructureAnalysis.RadialProfile import radKDE
from ctntools.Geometry.LineFuncs import LineFromFractIntercepts, NearestPointonLine

from scipy.ndimage import convolve
import numpy as np
import matplotlib.pyplot as plt

def fftPeaks(inim, gaussSigma = 1, subpixelfit=True, thresh=0.15, normalize=True, minRExclusionMethod = 'afterminima', principlePeakMethod = 'first', plotoutput = False, inax=[None]*3, verbose=False, **kwargs):
  #Inputs
  #inim                     :       input image
  #gaussSigma           (optional)  :   sigma for gaussian blur
  #subpixelfit          (optional)  :   flag to subpixel fit peaks as parabaloid (otherwise returns pixel maxima)
  #thresh               (optional)  :   Intensity threshold for peak finding
  #normalize            (optional)  :   flag to normalize input image
  #minRExclusionMethod  (optional)  :   'afterminima' or #  :set a minimum radius exclusion. 'afterminima' (default) considers only peaks beyond an initial minimum. Provide a numeric input to set manually.
  #principlePeakMethod  (optional)  :   'first' or 'max'    :How to select principle peak. 'first' is the longest frequency, 'max' is the largest value
  #inax                 (optional)  :   list of 3 axis to plot to. If None, no plotting. 
  #verbose              (optional)  :   flag to display details of execution

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
  inim_sz = np.array(inim.shape)[[1,0]]
  XYscale = 1/inim_sz
  normXYscale = XYscale/np.min(XYscale)
  xy0 = np.floor(inim_sz/2)
  
  #Normalize
  if normalize:
      inim = (inim - np.min(inim.ravel())) / np.ptp(inim.ravel())
      inim = inim-np.nanmean(inim.ravel())
  
  #FFT
  hann = np.outer(np.hanning(inim_sz[1]),np.hanning(inim_sz[0]))
  im_fft = np.abs(np.fft.fftshift(np.fft.fft2(hann*inim)))
  
  #smooth
  k = gKernel2D(gaussSigma,rscalar=3)
  im_fft_sm = convolve(im_fft,k,mode='nearest')
  
  #Radial distribution
  x, distr, density, _, _ = radKDE(im_fft_sm, rstp=.1, xyscale=normXYscale, s=kde1Dsigma, method='interp')
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
  if principlePeakMethod == 'max':
      ind = np.nanargmax(distr[1,minRind:])+minRind
      rpk = x[ind]
  elif principlePeakMethod == 'first':
      if distrdx is None:
          distrdx = np.gradient(distr[1,:])
      ind = np.argmax(distrdx[minRind:]<0)+minRind
      rpk = x[ind]  
  else:
      raise ValueError('unknown principlePeakMethod, must be "first" or "max"')
  
  #normalize
  im_fft_sm = (im_fft_sm - np.min(im_fft_sm))/np.ptp(im_fft_sm)
  
  #peaks
  xx,yy = np.meshgrid(np.arange(inim_sz[0]),np.arange(inim_sz[1]))
  dx = xx-xy0[0]
  dy = yy-xy0[1]
  dx = dx*normXYscale[0]
  dy = dy*normXYscale[1]
  r = np.sqrt(dx**2+dy**2)
  rmsk = (im_fft_sm>thresh) & (r>x[minRind])
  xy_fft,_,_,temp = imregionalmax(im_fft_sm, rpk*.75, imask=rmsk, xyscale=normXYscale)
  
  #refine peaks
  if subpixelfit:
      spfit_winsz = np.array([spfit_winsz,spfit_winsz],dtype='int')
      xy_fft_sp = refinePeaks(im_fft_sm, xy_fft[[0,1],:].T, winsz=spfit_winsz, ithresh=spfit_ithresh, progressDescr='Fitting peaks in FFT', verbose=verbose)[:,:3].T
      xy_v = xy_fft_sp.copy()
  else:
      xy_v = xy_fft.copy()
  
  #convert to real space
  dx = xy_v[0,:]-xy0[0]
  dy = xy_v[1,:]-xy0[1]
  abc = LineFromFractIntercepts(inim_sz,dx,dy)
  xy = NearestPointonLine(abc[:,0],abc[:,1],abc[:,2],[0,0])
  #r = np.sqrt((dx*normXYscale[0])**2+(dy*normXYscale[1])**2)
  xy_v[0,:] = xy[:,0]
  xy_v[1,:] = xy[:,1]
  
  #Plot?
  dx = xy_fft[0,:]-xy0[0]
  dy = xy_fft[1,:]-xy0[1]
  dx = dx*normXYscale[0]
  dy = dy*normXYscale[1]
  r = np.sqrt(dx**2+dy**2)
  rng = 1.1 * np.nanmax(r)
  print('Range: {:.2f}'.format(rng))
  xlim_ = [np.floor(im_fft.shape[1]/2-rng), np.ceil(im_fft.shape[1]/2+rng)]
  ylim_ = [np.floor(im_fft.shape[0]/2-rng), np.ceil(im_fft.shape[0]/2+rng)]
      
  if not (inax[0] is None):
      inax[0].imshow(im_fft,cmap='gray')
      inax[0].scatter(xy0[0],xy0[1],s=30,c='c',marker='+')
      inax[0].set_xlim(xlim_)
      inax[0].set_ylim(ylim_)
      inax[0].set_aspect(normXYscale[1]/normXYscale[0])
      inax[0].set_title('Image FFT')
  
  if not (inax[1] is None):
      tt = np.linspace(0,2*np.pi,100)
      inax[1].imshow(im_fft_sm,cmap='gray')
      inax[1].scatter(xy0[0],xy0[1],s=30,c='c',marker='+')
      inax[1].scatter(xy_fft[0,:],xy_fft[1,:],s=scattersz,c='b',marker='+')
      if subpixelfit:
          inax[1].scatter(xy_fft_sp[0,:],xy_fft_sp[1,:],s=scattersz,c='r',marker='x')
      inax[1].set_xlim(xlim_)
      inax[1].set_ylim(ylim_)
      inax[1].plot(xy0[0]+np.sin(tt)*x[minRind]/normXYscale[0],xy0[1]+np.cos(tt)*x[minRind]/normXYscale[1],'-b')
      inax[1].plot(xy0[0]+np.sin(tt)*x[ind]/normXYscale[0]    ,xy0[1]+np.cos(tt)*x[ind]/normXYscale[1],'-r')
      inax[1].set_aspect(normXYscale[1]/normXYscale[0])
      inax[1].set_title('SmoothedFFT & Peaks')
  
  if not (inax[2] is None):
      inax[2].plot(x,distr[1,:],'-k')
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
