#determines a threshold boundary box around input point (desiged to crop around peaks)
import numpy as np
from scipy import spatial

def bboxthresh(inim, ptxy=[], thr=.5, normalize=True, convexhullmask=True, minHW=np.array([0,0])):
    #Inputs:
    #inim                       :           input image
    #ptxy (optional)            :           rough center guess
    #thr (optional)             :           threshold value
    #normalize (optional)       :           flag to normalize data
    #convexhullmask (optional)  :           flag to calculate a mask in addition to the boundary box (uses the scipy ConvexHull function)
    #minHW (optional)           :           minimum boundary distance from ptxy point (assures a minimum height/width)

    #Outputs:
    #ROI                :           Boundary Box [xmin, xmax, ymin, ymax]
    #thrmsk             :           Threshold mask within boundary box. Default is simple threshold

    if ptxy.size==0:
        ptxy = np.array([np.argmax(np.max(inim,axis=0)),np.argmax(np.max(inim,axis=1))])

    if normalize:
        inim = (inim-np.nanmin(inim.ravel()))/(np.nanmax(inim.ravel())-np.nanmin(inim.ravel()))

    def decaythresh1D(inarray,inthresh):
        if np.any(inarray<=inthresh):
            t = np.argmax(inarray<=inthresh)
        else:
            t = np.size(inarray)
        return t
    #x
    tx = np.max(inim,axis=0)
    txc = tx[np.ceil(ptxy[0]).astype('int'):]
    txp = decaythresh1D(txc,thr)-1
    txp = np.max(np.array([txp,ptxy[0]+minHW[0]]))
    txc = np.flip(tx[:np.ceil(ptxy[0]).astype('int')])
    txn = decaythresh1D(txc,thr)
    txn = np.min(np.array([txn,ptxy[0]-minHW[0]]))

    #y
    ty = np.max(inim/np.max(inim.ravel()),axis=1)
    tyc = ty[np.ceil(ptxy[1]).astype('int'):]
    typ = decaythresh1D(tyc,thr)-1
    typ = np.max(np.array([typ,ptxy[1]+minHW[1]]))
    tyc = np.flip(ty[:np.ceil(ptxy[1]).astype('int')])
    tyn = decaythresh1D(tyc,thr)
    tyn = np.max(np.array([tyn,ptxy[1]-minHW[1]]))

    xmin = ptxy[0]-txn
    xmax = ptxy[0]+txp
    ymin = ptxy[1]-tyn
    ymax = ptxy[1]+typ

    #outputs
    ROI = np.array([xmin,xmax,ymin,ymax])
    thrmsk = (inim[ymin:ymax+1,xmin:xmax+1]>thr).astype('int')
    if np.all(thrmsk.ravel()==0):
        print('Error with calculated bbx bounds: xmin: '+str(xmin)+', xmax: '+str(xmax)+', ymin: '+str(ymin)+', ymax: '+str(ymax))
        #plt.imshow(inim)
        #plt.scatter(ptxy[0],ptxy[1],s=10,c='k',marker='o')

    assert np.any(thrmsk.ravel()==1), 'No points above threshold found'

    if convexhullmask:
        pts = np.array(np.where(thrmsk==1)).T[:,::-1]
        hull = spatial.ConvexHull(pts)
        delny = spatial.Delaunay(pts[hull.vertices])
        #plt.scatter(pts[hull.vertices,0],pts[hull.vertices,1],s=10,c='r',marker='o')
        xx,yy = np.meshgrid(np.arange(xmax-xmin+1),np.arange(ymax-ymin+1))
        pts = np.array([xx.ravel(),yy.ravel()]).T
        #plt.scatter(pts[:,0],pts[:,1],s=1,c='k',marker='o')
        #plt.xlim([0,inim.shape[1]])
        #plt.ylim([0,inim.shape[0]])
        inhull = delny.find_simplex(pts)
        inhull = np.reshape(inhull,xx.shape)
        thrmsk = inhull>-1

    return ROI, thrmsk
