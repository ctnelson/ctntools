import numpy as np

########################################### Contents ####################################
#NearestPointonLine          :   returns the point on a line closest to a given point 
#LineFromFractIntercepts     :   returns general line coeffecients from fractional intercept in frame size 'Sz'

######################################  NearestPointonLine  ##############################
#returns the point on a line closest to a given point
def NearestPointonLine(a,b,c,xy0):
    #ax + by + c = 0
    #a,b,c      :   [n,]            line equation coefficients
    #xy0        :   [2,] or [n,2]   point(s)
    ### Output ###
    #xyL        :   [n,2]           closest position on line

    #Setup
    a   = np.array(a)
    b   = np.array(b)
    c   = np.array(c)
    xy0 = np.array(xy0)

    #Dimension Handling
    n   = a.size
    if xy0.size==2:
        if np.ndim(xy0)==1:
            xy0=xy0[np.newaxis,:]
        elif np.ndim(xy0)==2 and np.shape[0]==2:
            xy0=xy0.T
    if n>1 and xy0.size==2:
        xy0 = np.repeat(xy0,n,axis=0)
    if n==1 and xy0.size>2:
        xyN = xy0.shape[0]
        a = np.repeat(a,xyN)
        b = np.repeat(b,xyN)
        c = np.repeat(c,xyN)
    
    #Calculation
    xL = (b*(b*xy0[:,0] - a*xy0[:,1]) - a*c) / (a**2 + b**2)
    yL = (a*(-b*xy0[:,0] + a*xy0[:,1]) - b*c) / (a**2 + b**2)
    xyL = np.vstack((xL.ravel(),yL.ravel())).T

    return xyL

######################################  LineFromFractIntercepts  ##############################
#returns general line coeffecients from fractional intercept in frame size 'Sz'. Designed to return 2D plane normal vectors from reciprocal space points.
def LineFromFractIntercepts(Sz,xi,yi):
    ### Inputs ###
    #Sz     :   [2,] Frame size
    #xi     :   [n,] x divisor(s)
    #yi     :   [n,] y divisor(s)
    ### Outputs ###
    #abc    :   a,b,c line equation coefficients (ax + by + c = 0)

    #Setup
    Sz = np.array(Sz)
    xi = np.array(xi)
    yi = np.array(yi)

    #Dimensions & allocation
    n=xi.size
    abc = np.ones((n,3))*np.nan
    
    #Conditions
    ind = np.where((xi!=0) | (yi!=0))[0]
    if ind.size==0:
        raise ValueError('both fractional intercepts cannot be zero')
    indx0 = np.where(xi[ind]==0)[0]
    indy0 = np.where(yi[ind]==0)[0]
    indxy = np.where((xi[ind]!=0) & (yi[ind]!=0))[0]

    #no x intercept (fracx=0)
    if indx0.size>0:
        abc[ind[indx0],0] = 0
        abc[ind[indx0],1] = -1
        abc[ind[indx0],2] = Sz[1]/yi[ind[indx0]]

    #no x intercept (fracy=0)
    if indy0.size>0:
        abc[ind[indy0],0] = -1
        abc[ind[indy0],1] = 0
        abc[ind[indy0],2] = Sz[0]/xi[ind[indy0]]

    #normal
    if indxy.size>0:
        abc[ind[indxy],0] = -(Sz[1]/yi[ind[indxy]])/(Sz[0]/xi[ind[indxy]])
        abc[ind[indxy],1] = -1
        abc[ind[indxy],2] = Sz[1]/yi[ind[indxy]]

    return abc
