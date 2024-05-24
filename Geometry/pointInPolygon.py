#Checks if test points lie in polygon
import numpy as np
from scipy.spatial import Delaunay

#pointInTriangle  :  check if test points are within triangle
#pointInPoly      :  check if test points are within polygon
#imIndInPoly      :  return indices in image that are within polygon

####################################################### Test if Points in Triangle #######################################
#Checks if test points are within triangle
#code snippet from Cédric Dufour https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle#:~:text=The%20point%20p%20is%20inside,coordinates%20of%20the%20point%20p%20.
def pointInTriangle(tp, tri):
    ### Inputs
    #tp     :   [n,2] test points
    #tri    :   [3,2] triangle vertices
    ### Outputs
    #output :   [n,]  whether test point lies within triangle
  
    if np.ndim(tp)==1:
        tp=tp[np.newaxis,:]
    assert(tp.shape[1]==2)

    dX = tp[:,0]-tri[2,0]
    dY = tp[:,1]-tri[2,1]
    dX21 = tri[2,0]-tri[1,0]
    dY12 = tri[1,1]-tri[2,1]
    D = dY12*(tri[0,0]-tri[2,0]) + dX21*(tri[0,1]-tri[2,1])
    s = dY12*dX + dX21*dY
    t = (tri[2,1]-tri[0,1])*dX + (tri[0,0]-tri[2,0])*dY
    if D<0:
        output = ((s<=0) & (t<=0) & (s+t>=D))
    else:
        output = ((s>=0) & (t>=0) & (s+t<=D))

    return output

####################################################### Test if Points in Polygon #######################################
#Checks if test points are within polygon
#Polygon is defined by Delaunay triangulation
def pointInPoly(tp,poly):
    ### Inputs
    #tp     :   [n,2] test points
    #poly   :   [v,2] polygon vertices
    ### Outputs
    #output :   [n,]  whether test point lies within polygon
    #sets   :   triangle sets from Delaunay
  
    output = np.zeros((tp.shape[0],))
    tri = Delaunay(poly)
    sets = tri.simplices
    for i in range(sets.shape[0]):
        output += pointInTriangle(tp,poly[sets[i,:],:])
    output = np.clip(output,0,1)
    return output, sets

################################################### Return Image indices in Polygon #######################################
#This function is a wrapper for in-polygon testing function 'pointInPoly' made to return indices of a 2D image within a polygon. 
#It's principle function is a speedupt by initially cropping the test points with a bounding box before testing with pointInPoly.
def imIndInPoly(im,vrts,ixx=None,iyy=None):
    ### Inputs ###
    #im     :   input image
    #vrts   :   [n,2] polygon vertices
    ### Output ###
    #ind    :   indices of points within polygon

    if (ixx is None) or (iyy is None):
        ixx,iyy = np.meshgrid(np.arange(im.shape[1]),np.arange(im.shape[0]))
    #boundary box
    bnds = np.array([np.floor(np.min(vrts[:,0])), np.ceil(np.max(vrts[:,0])), np.floor(np.min(vrts[:,1])), np.ceil(np.max(vrts[:,1]))],dtype='int')
    #valid bounds
    xxl = np.max([bnds[0],0])
    xxh = np.min([bnds[1],im.shape[1]-1])+1
    yyl = np.max([bnds[2],0])
    yyh = np.min([bnds[3],im.shape[0]-1])+1
    #crop index to just test the bounding box
    ind = np.ravel_multi_index((iyy[yyl:yyh,xxl:xxh],ixx[yyl:yyh,xxl:xxh]),im.shape)
    #in polygon test
    xy = np.array([ixx.ravel()[ind].ravel(),iyy.ravel()[ind].ravel()]).T
    test = np.reshape(pointInPoly(xy,vrts)[0],ind.shape)
    subInd = np.where(test)
    ind = ind[subInd]

    return ind
