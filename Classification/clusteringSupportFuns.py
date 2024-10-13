################################################ Contents ###################################################
#getNNbrThresh         :    Returns a nearest-neighbor threshold via an elbow in a cumulative distribution function of NN-distances
#Cluster_NumUnknown    :    A subset of sklearn clustering protocols that cluster quickly without an a priori number of clusters

################################################# Imports ###################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.neighbors import NearestNeighbors
from ctntools.BaseSupportFunctions.LineAnalysis import findScreeElbow


#Returns a nearest-neighbor threshold via an elbow in a cumulative distribution function of NN-distances.
#Note, uniformly distributed data will probably cause this to fail (and return 0)
############################################### getNNbrThresh ###################################################
def getNNbrThresh(inData, nbrKT=.03, nbrNum=1, verbose=0, inax=None):
    ### Input ###
    #inData         :   [Ndatapoints, nDims] input data
    #nbrKT          :   Threshold parameter to find elbow in NN distance distribution
    #nbrNum         :   number of closest neighbors to consider
    #verbose        :   flag to display execution details

    ### Output ###
    #nnThresh       :   nearest neighbor threshold

    ### Parameters
    targetBinNum = 100

    ###
    #nearest neighbor distances
    nbrs = NearestNeighbors(n_neighbors=nbrNum+1, algorithm='ball_tree').fit(inData)
    distances, indices = nbrs.kneighbors(inData)
    #(inverted) cumulative distribution of distances
    bins_ = np.linspace(0,np.nanmax(distances[:,1:].ravel()), targetBinNum)
    distHist, bin_edges = np.histogram(distances[:,1:].ravel(), bins=bins_, density=True)
    distHist = np.cumsum(distHist/np.sum(distHist))
    #find elbow

    if verbose>1:
        fig, inax = plt.subplots(1, 1, tight_layout=True, figsize=(12, 6), dpi = 100)
        inax.set_title('Nearest Neighbor Distance Threshold')
    else:
        inax = None

    ind = findScreeElbow(1-distHist, kinkThresh=nbrKT, inax=inax)
    nnThresh = bin_edges[ind]

    if verbose>0:
        print('threshold nearest-neighbor distance: {:.2f}'.format(nnThresh))

    return nnThresh


#A subset of sklearn clustering protocols that cluster quickly without an a priori number of clusters
############################################# Cluster_NumUnknown ################################################
def cluster_NumUnknown(inData, hyperparams=1, clustMethod='DBSCAN', verbose=0, **kwargs):
    ### Inputs ###
    #inData         :   [Ndatapoints, nDims] input data
    #hyperparams    :   threshold parameters for the various methods
    #clustMethod    :   'DBSCAN', 'HDBSCAN', 'OPTICS', 'Agglomerative', 'Birch'
    #verbose        :   flag to print execution details
    #kwargs         :   to pass thru name-value arguements to clustering algorithms 

    ### Outputs ###
    #cLabels        :   [Ndatapoints,] cluster labels
    #cNum           :   Number of clusters

    ###
    if clustMethod=='DBSCAN':
        c = cluster.DBSCAN(eps=hyperparams, **kwargs)
        c.fit(inData)
        cLabels = c.labels_
    elif clustMethod=='HDBSCAN':
        c = cluster.HDBSCAN(cluster_selection_epsilon=hyperparams, **kwargs)
        c.fit(inData)
        cLabels = c.labels_
    elif clustMethod=='OPTICS':
        c = cluster.OPTICS(xi=hyperparams, cluster_method='xi', algorithm='auto', **kwargs)
        c.fit(inData)
        cLabels = c.labels_
    elif clustMethod=='Agglomerative':
        c = cluster.AgglomerativeClustering(distance_threshold=hyperparams, n_clusters=None, linkage='single', **kwargs)
        c.fit(inData)
        cLabels = c.labels_
    elif clustMethod=='Birch':
        c = cluster.Birch(n_clusters=None, threshold=hyperparams, **kwargs)
        c.fit(inData)
        cLabels = c.labels_

    cNum = np.max(cLabels)+1
    if verbose>0:
        print('{:s} cluster number: {:d}'.format(clustMethod, cNum))

    return cLabels, cNum
