# Prune clusters in a coordinate space given selection criteria (e.g. max number of clusters, min population size)
################################################################# Imports ###########################################################
import numpy as np
from sklearn.cluster import KMeans

from ctntools.Classification.LatentSpaceSupportFuns import ClassLatPositions, getDists2Refs, getRefNormDist

#########################################################   clusterPrune   ###########################################################
#Prune Clusters in a (Latent?) Coordinate Space
def clusterPrune(dataLatCoords, dClassLabels, classLatCoords=None, dClassN=None, RefMethod='minfrac', RefThresh=None, RefThresh_maxN=3, RefThresh_minPop=10, RefThresh_minFract=.04, ReClassMethod='latDist', ReClass_latDist=3, latNorm='classStd', reClassOnlyPruned=False, verbose=verbose):
    ### Inputs (Required) ###
    #dataLatCoords      :   [m,n]   data coordinates
    #dClassLabels       :   [m,]    data labels (numeric), noise/unclassified = -1

    ### Inputs (Optional). Defaults provided for all below ###
    #classLatCoords     :   [dClassN, latDimN]  representative coordinates of classes 
    #dClassN            :   number of classes 
    
    ### Prune ###
    #RefMethod          :   Prune clusters if fail criteria of 'maxN' (limit maximum numbe of clusters), 'minpop' (minimum population size of cluster), or 'minfrac' (minimum fractional population size of cluster)
    #RefThresh          :   Threshold used for RefMethod. Set to None type to use default values.
    #RefThresh_maxN     :   default max number of clusters
    #RefThresh_minPop   :   default cluster min population size
    #RefThresh_minFract :   default cluster min population fraction

    ### Reclassify ###
    #ReClassMethod      :   'latDist', 'KMeans', or 'none'. Method of reclassifying clusters deemed invalid in pruning step. If latent distance, can avoid redundant computation by providing classLatCoords.
    #ReClass_latDist    :   if reclassify by latent distance, use this cutoff sigma (beyond this datapoints will be classified 'noise')
    #latNorm            :   'none', 'globalStd', or 'classStd'. If reclassifying by latent distance, method to normalize
    #reClassOnlyPruned  :   flag whether to reclassify only removed clusters? Only applies to LatDist, KMeans will reclassify all points for the determined pruned number of clusters.

    ### Other ###
    #verbose            :   flag to plot execution progress
     
    ### Outputs ###
    #dReClassLabels     :   class labels
    #dReClassN          :   number of classes

    ### Setup ###
    #A few value checks
    assert np.ndim(dataLatCoords)==2
    assert dataLatCoords.shape[0]==dClassLabels.size

    #latDimN = dataLatCoords.shape[1]
    if dClassN is None:
        dClassN = np.max(dClassLabels)+1
    dN = dataLatCoords.shape[0]

    ############# Check for clusters that fail criteria and need to be reclassified ###########
    #Max class limit
    if RefMethod=='maxN':  
        if RefThresh is None:                     #check if input value, else use default
            RefThresh = RefThresh_maxN
        RefThresh = np.int32(RefThresh) #ensure integer
        if dClassN > RefThresh:
            dLblCnt = np.bincount(np.where(dClassLabels>=0,dClassLabels,dClassN),minlength=dClassN)    #counts number in each class (unclassified 'noise' where =-1 is set to highest bin to ensure array length)
            indClassOrderByPop = np.argsort(dLblCnt[:dClassN])[::-1]   #sort classes by population
            indReClass = np.append(indClassOrderByPop[RefThresh:],-1)
            indValClass = indClassOrderByPop[:RefThresh]
            if verbose>0:
                print('{:d} classes exceeds prescribed maximimum of {:d}, {:d} classes removed'.format(dClassN, RefThresh, len(indReClass)))
        else:
            indReclass = np.empty(0,dtype='int')
            indValClass = np.arange(dClassN)

    #Minimum Class Population Size
    elif RefMethod=='minpop':
        if RefThresh is None:                     #check if input value, else use default
            RefThresh = RefThresh_minPop
        #RefThresh = np.int32(RefThresh) #ensure integer
        dLblCnt = np.bincount(np.where(dClassLabels>=0,dClassLabels,dClassN),minlength=dClassN)    #counts number in each class (unclassified 'noise' where =-1 is set to highest bin to ensure array length)
        #indReClass = np.where(dLblCnt[:dClassN]<RefThresh)[0]
        indReClass = np.append(np.where(dLblCnt[:dClassN]<RefThresh)[0],-1)
        indValClass = np.where(dLblCnt[:dClassN]>RefThresh)[0]
        if verbose>0:
            if indReClass.size>0:
                print('{:d} classes are below the population limit ({:d}) and will be removed'.format(indReClass.size, RefThresh))

    #Minimum Class Fraction Size
    elif RefMethod=='minfract':
        if RefThresh is None:                     #check if input value, else use default
            RefThresh = RefThresh_minFract
        RefThresh = np.ceil(RefThresh*dN).astype('int')
        dLblCnt = np.bincount(np.where(dClassLabels>=0,dClassLabels,dClassN),minlength=dClassN)    #counts number in each class (unclassified 'noise' where =-1 is set to highest bin to ensure array length)
        #indReClass = np.where(dLblCnt[:dClassN]<RefThresh)[0]
        indReClass = np.append(np.where(dLblCnt[:dClassN]<RefThresh)[0],-1)
        indValClass = np.where(dLblCnt[:dClassN]>=RefThresh)[0]
        if verbose>0:
            if indReClass.size>0:
                print('{:d} classes are below the population limit ({:d}) and will be removed'.format(indReClass.size-1, RefThresh))
    else:
        raise ValueError('RefMethod value not recognized. Must be maxN, minpop, or minfract')
    
    ######################### Reclassify ############################
    #Reclassify
    reindex = np.ones((dClassN+1),dtype='int')*-1
    reindex[indValClass] = np.arange(indValClass.size)
    dReClassLabels = np.ones_like(dClassLabels,dtype='int')*-1
    dValClassLabels = reindex[dClassLabels]

    #Latent Distance
    if indReClass.size>0:
        if ReClassMethod =='latDist':
            #print('latDist')
            if reClassOnlyPruned:
                rind = np.where(np.isin(dClassLabels,indReClass))[0]
            else:
                rind = np.arange(dN)

            if classLatCoords is None:
                cCoords = ClassLatPositions(dataLatCoords, dValClassLabels, lblArray=np.arange(indValClass.size))
            else:
                cCoords = classLatCoords.copy()
                cCoords = cCoords[indValClass,:]
            
            if latNorm=='none':
                dist, delta = getDists2Refs(dataLatCoords[rind,:], cCoords)          #get delta to reference points
                nDist = dist
                nDelta = delta
            elif latNorm=='globalStd':
                dist, delta = getDists2Refs(dataLatCoords[rind,:], cCoords)          #get delta to reference points
                dataStd = np.sqrt(np.sum(np.std(dataLatCoords,axis=0)**2))
                nDist = dist/dataStd
                nDelta = delta/dataStd
                nDist = nDist[rind,:]
            elif latNorm=='classStd':
                dist, delta = getDists2Refs(dataLatCoords, cCoords)          #get delta to reference points
                nDist, nDelta, deltaStd = getRefNormDist(dataLatCoords, dValClassLabels, cCoords, dDeltas2Ref=delta, lblArray=np.arange(indValClass.size))
                #nDist, nDelta, deltaStd = getRefNormDist(dataLatCoords, dValClassLabels, cCoords, lblArray=np.arange(indValClass.size))
                nDist = nDist[rind,:]
            else:
                raise ValueError('ReClassMethod not recognized. Should be none, globalStd, or classStd')
            #find closest valid class average and reclassify
            distMin = np.nanmin(nDist,axis=1)
            ind = np.where(np.isfinite(distMin) & (distMin<=ReClass_latDist))[0]
            distMinInd = np.nanargmin(nDist[ind,:],axis=1)
            if reClassOnlyPruned:
                dReClassLabels = dValClassLabels
            dReClassLabels[rind[ind]] = distMinInd
        #KMeans
        elif ReClassMethod =='KMeans':
            #print('kMeans')
            kmeans = KMeans(n_clusters=indValClass.size, random_state=0).fit(dataLatCoords)
            dReClassLabels = kmeans.labels_ 
        #Don't Reclass
        elif ReClassMethod == 'none':
            #print('dont reclassify')
            dReClassLabels = dValClassLabels
        else:
            raise ValueError('ReClassMethod not recognized. Must be latDist or KMeans.')
    #No reclassification
    else:
        #print('no clusters to reclassify')
        dReClassLabels = dValClassLabels

    dReClassN = indValClass.size

    return dReClassLabels, dReClassN
