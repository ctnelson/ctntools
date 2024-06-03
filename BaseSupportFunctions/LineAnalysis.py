import numpy as np
import matplotlib.pyplot as plt

#Function finds the 'elbow' in a monotonic function (decaying gradient)
#Includes the consideration that it is not present
#Function is designed to find the 'elbow' for optimum latent parameter number for classfications such as PCA, NMF, KMeans, etc. 
#Choice of method: 'GradientThresh' will return for the first point the gradient falls below a threshold. 'LineOutlier' looks for the start of a linear region within a threshold value.
def findScreeElbow(y, elbowMethod='LineOutlier', gradThresh=.03, kinkThresh=.01, minLinearLen=3, fSEnormalize=True, inax=None, **kwargs):
    ### Inputs ###
    #y                  :   [n,] input array. 
    #fSEMethod          :   'GradientThresh', 'LineOutlier'. Method to find eblow. 'GradientThresh' uses a slope threshold, 'LineOutlier' looks for a point of deviation of an array that ends in a linear region
    #gradThresh         :   threshold of improvement for incrementing additional class # (this is the criteria used to autoselect class #)
    #asymThresh         :   threshold for finding a kink. Looks for outlier from a linear regression of high # classes.
    #minLinearLen       :   for 'LineOutlier' method, minimum length at end of plot for linear regression
    #fSEnormalize       :   flag to normalize y
    #inax               :   axis for optional plotting of results

    ### Outputs ###
    #elbowInd           :   index of elbow

    n = y.size
    x = np.arange(n)

    #normalize?
    if fSEnormalize:
        y=y/np.max(y)

    #Find by gradient threshold
    if elbowMethod=='GradientThresh':            
        dyn = y[0:-1]-y[1:]
        elbowInd = np.argmax(np.abs(dyn)<gradThresh)

    #Find by onset of linear region at end of function
    elif elbowMethod=='LineOutlier':
        nn = n-minLinearLen
        #Linear regression of scree plot
        m = np.ones((nn,))*np.nan
        c = np.ones((nn,))*np.nan
        predictError = np.ones((nn,))*np.nan
        for i in range(nn):
            #Linear regresion
            A = np.vstack([x[i+1:], np.ones((x.size-i-1,))]).T
            f = np.linalg.lstsq(A, y[i+1:], rcond=None)
            m[i],c[i]=f[0]
            #check for first outlier from prediction
            ypredict = x[i]*m[i]+c[i]
            predictError[i] = y[i]-ypredict
        elbowInd = np.argmax(predictError<kinkThresh)

    else:
        raise ValueError('Unknown method must be GradientThresh or LineOutlier')
    
    #Plot?
    if not (inax is None):
        inax.plot(x,y,'-k',zorder=1)
        inax.scatter(elbowInd,y[elbowInd],s=100,color='r',zorder=2)
        inax.scatter(x,y,s=10,color='k',zorder=1)
        if elbowMethod=='GradientThresh':
            b = y[elbowInd]-elbowInd*(-gradThresh)
            x0 = (y[0]-b)/(-gradThresh)
            x1 = (y[-1]-b)/(-gradThresh)
            inax.plot([x0,x1],[y[0],y[-1]],'-b',alpha=.25,zorder=0)
            inax.text(elbowInd,y[elbowInd],' Gradient Threshold',va='top',ha='right',color='b',rotation=-90,alpha=.25)
        elif elbowMethod=='LineOutlier':
            inax.plot(x,m[elbowInd]*x+c[elbowInd],'--g',zorder=0,alpha=.25)
            inax.plot(x[:elbowInd+2],m[elbowInd]*x[:elbowInd+2]+c[elbowInd]+kinkThresh,'-r',alpha=.25,zorder=0)
            inax.text(elbowInd,m[elbowInd]*elbowInd+c[elbowInd]+kinkThresh,'Gradient Kink Threshold',va='bottom',ha='left',color='r',alpha=.25)

    return elbowInd
