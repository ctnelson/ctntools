#Fits a local minima/maxima of an image with a paraboloid. Creates a mask by thresholding to a boundary box further reduced to a convex hull (optional). 
import numpy as np
from ctntools.bboxthresh import bboxthresh

def fit1peak(inim, ipkxy=[], ithresh=.8, calcCVHullMask=True, calcprediction=True, calcresidual=True):
    #Inputs:
    #inim               :           input image
    #ipkxy(optional)    :           [2,1] guess of xy location, will default to peak value
    #ithresh(optional)  :           threshold value to create fitting mask
    #calcCVHullMask     :           flag to calculate a convex hull of above-threshold region
    #calcprediction     :           flag to calculate the fitting result (required to return valid zeval)
    #calcresidual       :           flag to calculate the residual of indata-fit_result (required to return resid)
    
    #Outputs:
    #outparams      :           [5,1] array of parameters [x, y, maximum/minimum, rotation, elipticity]
    #bbx            :           rectangular boundary box around peak containing fit region
    #msk            :           the mask withing the bbx selecting the fit region
    zeval = []#     :           evaluated fit result (within bbx)
    resid = []#     :           residual (within bbx)

    def polyval(inx,iny,inA):
        output = np.ones_like(inx)*inA[0] + inA[1]*inx + inA[2]*iny + inA[3]*inx**2 + inA[4]*inx*iny + inA[5]*iny**2
        return output

    if ipkxy.size==0:
        ipkxy = np.array([np.argmax(np.max(inim,axis=0)),np.argmax(np.max(inim,axis=1))]) 
    bbx, msk = bboxthresh(inim,ipkxy,ithresh,convexhullmask=calcCVHullMask,normalize=True)
    ind=np.where(msk==1)
    xx,yy = np.meshgrid(np.arange(0,bbx[1]-bbx[0]+1),np.arange(0,bbx[3]-bbx[2]+1))
    X = xx[ind].ravel()
    Y = yy[ind].ravel()
    B = inim[bbx[2]:bbx[3]+1,bbx[0]:bbx[1]+1][ind]
    A = np.array([np.ones_like(X), X, Y, X**2, X*Y, Y**2]).T
    coeff= np.linalg.lstsq(A, B)[0]

    xf = (2*coeff[5]*coeff[1]-coeff[2]*coeff[4])/(-4*coeff[3]*coeff[5]+coeff[4]**2)
    yf = (-2*coeff[3]*xf-coeff[1])/coeff[4]
    theta = np.arctan(coeff[4]/(coeff[3]-coeff[5]))/2
    elipt = ((coeff[3]*(np.sin(theta)**2)-coeff[4]*np.sin(theta)*np.cos(theta) + coeff[5]*(np.cos(theta)**2))/(coeff[3]*(np.cos(theta)**2)+coeff[4]*(np.cos(theta)*np.sin(theta))+coeff[5]*(np.sin(theta)**2)))**.5
    Af = polyval(xf,yf,coeff)

    if calcprediction:
        zeval=polyval(xx.ravel(),yy.ravel(),coeff)
        zeval = np.reshape(zeval,xx.shape)
        if calcresidual:
            resid = zeval - inim[bbx[2]:bbx[3]+1,bbx[0]:bbx[1]+1]

    outparams = np.array([xf+bbx[0],yf+bbx[2],Af,theta,elipt])

    return outparams, bbx, msk, zeval, resid


#Performs subpixel fitting of local maxima/minima with least square fit to parabaloids. Meant as a refinement, requires initial guesses as an input.
def refinePeaks(inim,ipkxy,winsz=[]):
    #Inputs:
    #inim           :           input image
    #ipkxy          :           [n x 2] array of local maxima guesses
    #winsz          :           [1,], [2,1], or [n,2] array of cropping window to use around each guess point

    #Outpus:
    #outparams      :           returns [n x 5] array of parameters [x, y, maximum/minimum, rotation, elipticity]

    if ipkxy.ndim==1:
        ipkxy = ipkxy[np.newaxis,:]
    ipkxy = np.round(ipkxy).astype('int')

    outparams = np.ones((ipkxy.shape[0],5))*np.nan

    if winsz.size==0:
        winsz = np.array(inim.shape)
    if winsz.size==1:
        winsz[1] = winsz[0]
    winsz = np.ceil(winsz).astype('int')
    

    for i in range(ipkxy.shape[0]):
        try:
            if winsz.size==2:
                winsz_lp = winsz
            else:
                winsz_lp = winsz[i,:]
            xmin_ = np.max(np.array([ipkxy[i,0]-winsz_lp[0],0],dtype='int'))
            xmax_ = np.min(np.array([ipkxy[i,0]+winsz_lp[0]+1,inim.shape[1]],dtype='int'))
            ymin_ = np.max(np.array([ipkxy[i,1]-winsz_lp[1],0],dtype='int'))
            ymax_ = np.min(np.array([ipkxy[i,1]+winsz_lp[1]+1,inim.shape[0]],dtype='int'))

            pkxy_lp = ipkxy[i,:]-np.array([xmin_,ymin_])
            outparams[i,:] = fit1peak(inim[ymin_:ymax_,xmin_:xmax_], pkxy_lp, calcCVHullMask=True, calcprediction=False, calcresidual=False)[0]
            outparams[i,:2] += np.array([xmin_,ymin_])
        except:
            print('Peak#'+str(i)+'failure')
            continue
        
    return outparams