#determines a threshold boundary box around input point (desiged to crop around peaks)
import numpy as np
from scipy import spatial

def bboxthresh(inim, ptxy=[], thr=.5, normalize=True, convexhullmask=True, minHW=np.array([0,0],dtype=np.int8)):
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

    ptxy=np.array(ptxy)
    if ptxy.size==0:
        ptxy = np.array([np.argmax(np.max(inim,axis=0)),np.argmax(np.max(inim,axis=1))],dtype=np.int32)
    else:
        ptxy = np.int32(np.ceil(ptxy))

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
    #txc = tx[np.ceil(ptxy[0]).astype('int'):]
    txc = tx[ptxy[0]:]
    txp = decaythresh1D(txc,thr)-1
    txp = np.max(np.array([txp,minHW[0]]))
    #txc = np.flip(tx[:np.ceil(ptxy[0]).astype('int')])
    txc = np.flip(tx[:ptxy[0]])
    txn = decaythresh1D(txc,thr)
    txn = np.max(np.array([txn,minHW[0]]))

    #y
    ty = np.max(inim/np.max(inim.ravel()),axis=1)
    #tyc = ty[np.ceil(ptxy[1]).astype('int'):]
    tyc = ty[ptxy[1]:]
    typ = decaythresh1D(tyc,thr)-1
    typ = np.max(np.array([typ,minHW[1]]))
    #tyc = np.flip(ty[:np.ceil(ptxy[1]).astype('int')])
    tyc = np.flip(ty[:ptxy[1]])
    tyn = decaythresh1D(tyc,thr)
    tyn = np.max(np.array([tyn,minHW[1]]))

    xmin = np.int32(ptxy[0]-txn)
    xmax = np.int32(ptxy[0]+txp)
    ymin = np.int32(ptxy[1]-tyn)
    ymax = np.int32( ptxy[1]+typ)
    
    #outputs
    ROI = np.array([xmin,xmax,ymin,ymax])
    thrmsk = (inim[ymin:ymax+1,xmin:xmax+1]>thr).astype('int')

    if ~np.any(thrmsk.ravel()==1) & np.all(minHW==0):
        raise ValueError('No points above threshold found')

    if convexhullmask:
        if np.any(thrmsk.ravel()==1):
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
        else:
            thrmsk=np.ones_like(thrmsk)
            Warning('No points above threshold, returning minimum bbx as mask, xmin: '+str(xmin)+', xmax: '+str(xmax)+', ymin: '+str(ymin)+', ymax: '+str(ymax))
    
    #guarantee min HW is in mask (at minimum this will be the zero point)
    thrmsk[ptxy[1]-minHW[1]:ptxy[1]+minHW[1]+1,ptxy[0]-minHW[0]:ptxy[0]+minHW[0]+1] = True
            
    return ROI, thrmsk
