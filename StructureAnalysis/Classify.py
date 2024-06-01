import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from ctntools.BaseSupportFunctions.LineAnalysis import findScreeElbow

def subImStackKMeans(iStack, maxClasses=10, findClassNum=True, classNaNMethod='random', inax=None, verbose=False, **kwargs):
    ### Inputs ###
    #iStack             :   [h,w,n]
    #maxClasses         :   maximum number of classes
    #findClassNum       :   flag to automatically find optimum # of classes. If false will use 'maxClasses' value
    #classNaNMethod     :   'random', or 'remove'. Method to handle invalid datapoints (e.g. NaN or Inf)
    #inax               :   axis for optional plotting of outputs
    #verbose            :   flag to print execution details

    ### Outputs ###
    #km_uc_avg          :   [h,w,class#] Stack of Class averages
    #oClass             :   [n,] Class labels
    
    #Setup
    X_vec = iStack.reshape(np.shape(iStack)[0]*np.shape(iStack)[1],np.shape(iStack)[2]).T
    oClass = np.ones((iStack.shape[2],))*np.nan
    km_labels = np.empty([maxClasses,X_vec.shape[0]])
    km_inertia = np.empty([maxClasses])
    oind = np.arange(X_vec.shape[0])                #subindex of subimages that are classified (defaults to all)

    ### handling invalid values ###
    ind=np.where(np.logical_not(np.isfinite(X_vec)))
    if ind[0].size>0:
        if classNaNMethod=='remove':
            oind=np.where(np.all(np.isfinite(X_vec),axis=1))[0]     #skip indices of any subImages with invalid points
            X_vec = X_vec[oind]
            if verbose:
                print('{:d} subimages with invalid datapoints were removed'.format(iStack.shape[2]-oind.size))
        elif classNaNMethod=='random':
            temp = np.random.randn(ind[0].size,)*np.nanvar(X_vec.ravel())+np.nanmean(X_vec.ravel()) #replace with random noise (scaled to dataset statistics)
            X_vec[ind]=temp

    ### KMeans & Cluster number selection ###
    if not findClassNum:
        kmClustN = maxClasses
        kmeans = KMeans(n_clusters=kmClustN, random_state=0).fit(X_vec)
        if verbose:
                print('Manually selected {:d} classes'.format(kmClustN))
    else:
        #Loop performing Kmeans for different #classes
        for i in np.arange(1,maxClasses+1):
            kmeans = KMeans(n_clusters=i, random_state=0).fit(X_vec)
            km_labels[i-1,:] = kmeans.labels_ #kmeans.predict(kmdata)
            km_inertia[i-1] = kmeans.inertia_

        #Find Elbow in Scree Plot
        x = np.arange(1,maxClasses+1)
        y = km_inertia/np.max(km_inertia)
        kmClustN = findScreeElbow(y, elbowMethod='GradientThresh', gradThresh=.03, kinkThresh=.01, minLinearLen=3, fSEnormalize=True, inax=inax, **kwargs) + 1    
        if verbose:
            print('Autoselected {:d} classes'.format(kmClustN))

    #Unit Cell Class Averages
    km_uc_avg = np.empty([iStack.shape[0],iStack.shape[1],kmClustN])
    for i in np.arange(0,kmClustN):
        ind = np.where(km_labels[kmClustN-1,:]==i)[0]
        km_uc_avg[:,:,i]=np.nanmean(iStack[:,:,ind],axis=2)

    #Plot?
    if not (inax is None):
        inax.set_title('KMeans Scree Plot')
        inax.set_xlabel('# Classes')
        inax.set_ylabel('Intertia (normalized)')

    #Outputs
    oClass[oind] = km_labels[kmClustN-1,:]

    return km_uc_avg, oClass
