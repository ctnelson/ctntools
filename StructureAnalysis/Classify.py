import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#KMeans clustering of a subimage stack [h,w,n]. Increments through number of classes and selects optimum as a failure to reach an threshold improvement ('ClassIncThresh'), i.e. looking for the 'elbow' of the scree plot
def subImStackKMeans(iStack, maxClasses=7, ClassSelMethod = 'GradientThresh', ClassIncThresh=.04, ClassNaNMethod='random', inax=None, verbose=False, **kwargs):
    ### Inputs ###
    #iStack             :   [h,w,n]
    #maxClasses         :   maximum number of classes
    #ClassSelMethod     :   'GradientThresh', 'GradientKink', 'Manual'. Method to autoselect number of classes. 'Manual' uses 'maxClasses', 'Gradient...' uses the slope of the normalized scree plot. '...Thresh' uses a slope threshold, '...Kink' looks for a slope change
    #ClassIncThresh     :   threshold of improvement for incrementing additional class # (this is the criteria used to autoselect class #)
    #ClassNaNMethod     :   'random', or 'remove'. Method to handle invalid datapoints (e.g. NaN or Inf)
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
        if ClassNaNMethod=='remove':
            oind=np.where(np.all(np.isfinite(X_vec),axis=1))[0]     #skip indices of any subImages with invalid points
            X_vec = X_vec[oind]
            if verbose:
                print('{:d} subimages with invalid datapoints were removed'.format(iStack.shape[2]-oind.size))
        elif ClassNaNMethod=='random':
            temp = np.random.randn(ind[0].size,)*np.nanvar(X_vec.ravel())+np.nanmean(X_vec.ravel()) #replace with random noise (scaled to dataset statistics)
            X_vec[ind]=temp

    ### KMeans & Cluster number selection ###
    if ClassSelMethod=='Manual':
        km_clusternum = maxClasses
        kmeans = KMeans(n_clusters=km_clusternum, random_state=0).fit(X_vec)
    else:
        #Loop performing Kmeans for different #classes
        for i in np.arange(1,maxClasses+1):
            kmeans = KMeans(n_clusters=i, random_state=0).fit(X_vec)
            km_labels[i-1,:] = kmeans.labels_ #kmeans.predict(kmdata)
            km_inertia[i-1] = kmeans.inertia_

        #Autoselect class number by gradient threshold
        if ClassSelMethod=='GradientThresh':            
            km_gain = (km_inertia[0:-1]-km_inertia[1:])/km_inertia[0]
            km_clusternum = np.argmax(km_gain<ClassIncThresh)+1
            if verbose:
                print('Autoselected {:d} classes'.format(km_clusternum))

        #Autoselect class number by gradient change ('kink')
        elif ClassSelMethod=='GradientKink':
            raise ValueError('not yet coded')

        elif ClassSelMethod!='Manual':
            raise ValueError('KMeans class number selection variable ClassSelMethod must be "Manual", "GradientThresh", or "GradientKink"')
            
    #Unit Cell Class Averages
    km_uc_avg = np.empty([iStack.shape[0],iStack.shape[1],km_clusternum])
    for i in np.arange(0,km_clusternum):
        ind = np.where(km_labels[km_clusternum-1,:]==i)[0]
        km_uc_avg[:,:,i]=np.nanmean(iStack[:,:,ind],axis=2)

    #Outputs
    oClass[oind] = km_labels[km_clusternum-1,:]

    #Scree Plot?
    if not (inax is None):
        inax.plot(np.arange(1,maxClasses+1),km_inertia)
        inax.scatter(km_clusternum,km_inertia[km_clusternum-1])
        if ClassSelMethod=='GradientThresh':
            inax.plot([km_clusternum, km_clusternum+1],[km_inertia[km_clusternum-1],km_inertia[i]-ClassIncThresh*km_inertia[0]],'--b')
        inax.set_title('KMeans Scree Plot')
        inax.set_xlabel('# Classes')
        inax.set_ylabel('Intertia')

    return km_uc_avg, oClass
