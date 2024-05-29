import numpy as np
import matplotlib.pyplot as plt
from ctntools.Networks.findDefinedNbrs import nbrByDist
from ctntools.Networks.nbrClustering import nbrClustering
from ctntools.StructureAnalysis.ucFindAB import abGuessFromScoredPeaks

def latticeNbrs(xy, a=None, b=None, searchDist=np.inf, nbrNum=9, awt = [.25,1,1], bwt = [.25,1,1], inax=None, verbose=0, **kwargs):
    ### Inputs ###
    #xy                         :   [n,2] locations
    #a              (optional)  :   provided a vector
    #b              (optional)  :   provided b vector
    #searchDist     (optional)  :   exclusion radius for finding neighboring peaks
    #nbrNum         (optional)  :   intial downselect of nearest neighbors to analyze
    #awt            (optional)  :   a-vector weights for [score, rmag, orientation]
    #bwt            (optional)  :   b-vector weights for [score, rmag, internalangle (alpha)]
    #inax           (optional)  :   input axis to plot results
    #verbose        (optional)  :   flag to print execution information 

    ### (Select) Inputs for finding ab from scratch (see ctntools function 'abGuessFromScoredPeaks' for all options) ### 
    #alphaGuess     (optional)  :   target for internal angle (deg)
    #rexcl          (optional)  :   minimum exclusion radius
    #rGuess         (optional)  :   guess for a and b magnitude
    #aOrientTarget  (optional)  :   target for a-vector orientation (deg)

    ### Outputs ###
    #nbrInd                     :   neighbor indices (note the output is not integer to allow NaN for none found)
    #a                          :   a basis vector
    #b                          :   b basis vector

    if (not (a is None)) and (not (b is None)):     #if a and b vector guesses are provided    
        #find lattice neighbors
        abNbrInd = nbrByDist(xy[:,:2], RelativeOffsets=np.array([a,b]), minDist=0, maxDist=searchDist, missingVal=np.nan)
        ar=a
        br=a
    else:
        #neighbors by clusering
        nbrmn_xyra, rbin = nbrClustering(xy[:,:2],nbrNum)
        #find lattice neighbors
        nbrInd = nbrByDist(xy[:,:2], RelativeOffsets=nbrmn_xyra[:,:2], minDist=0, maxDist=searchDist, missingVal=-1)
        #median values
        temp = np.append(xy.copy(),np.ones((1,3))*np.nan,axis=0)
        dx = temp[nbrInd[:,:,1].astype('int'),0]-np.repeat(xy[:,np.newaxis,0],nbrNum,axis=1)
        dy = temp[nbrInd[:,:,1].astype('int'),1]-np.repeat(xy[:,np.newaxis,1],nbrNum,axis=1)
        score =  (temp[nbrInd[:,:,1].astype('int'),2]+np.repeat(xy[:,np.newaxis,2],nbrNum,axis=1))/2
        nbrmed = np.stack((np.nanpercentile(dx,50,axis=0),np.nanpercentile(dy,50,axis=0),np.nanpercentile(score,50,axis=0)),axis=1)
        nbrR = (nbrmed[:,0]**2+nbrmed[:,1]**2)**.5
        rGuess = np.nanpercentile(nbrR,50,axis=0)
        #find lattice vectors
        aind, bind,_,_ = abGuessFromScoredPeaks(nbrmed, xy0=np.zeros((2,)), rGuess=rGuess, awt=awt, bwt=bwt, **kwargs)
        #aind, bind, ascore, bscore = abGuessFromScoredPeaks(nbrmed, xy0=np.zeros((2,)), alphaGuess=alphaGuess, rGuess=rGuess, awt=awt, bwt=bwt)
        ar = nbrmed[aind,:2]    #refined a vector
        br = nbrmed[bind,:2]    #refined b vector
        abNbrInd=nbrInd[:,[aind,bind],:]

    #plot?
    if not (inax is None):
        color = plt.cm.brg(np.linspace(0, 1, nbrNum))
        for i in range(nbrNum):
            inax.scatter(dx[:,i],dy[:,i],s=1,color=color[i])
        inax.scatter(nbrmed[:,0],nbrmed[:,1],s=50,marker='+',c='k',zorder=1)
        inax.scatter(0,0,s=100,marker='o',c='r')
        inax.plot([0,ar[0]],[0,ar[1]],'-b')     #a vector
        inax.plot([0,br[0]],[0,br[1]],'-r')     #b vector
        inax.set_aspect(1)
    
    return abNbrInd, ar, br
