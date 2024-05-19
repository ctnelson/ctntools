import numpy as np

#Generates [2,2] matrix transforms for some given symmetry shorthands (a [2,2,n] stack if multiple are provided)
def makeMtransf(iDict, verbose=False):
    ### Inputs ###
    #iDict  :   input dictionary of symmetry shorthands. Example: {'i':0,'r':[90,120]} creates a [2,2,3] stack of inversion, 90deg rot and 120deg rot.
    #keys and values are:
    #i (inversion)  :   value ignored
    #r (rotation)   :   scalar, list, or array [ang1, ang2...] (in deg)
    #m (mirror)     :   scalar, list, or array [ang1, ang2...] (in deg)

    ### Outputs ###
    #oM             :   [2,2,n] matrix transform stack
    #oMlbls         :   list of str labels

    ### Main ###
    oMlbls = []
    n=0
    for key, value in iDict.items():
        dictValue=np.array(value)
        if (dictValue.size==1) & (np.ndim(dictValue)==0):
            dictValue = dictValue[np.newaxis]
        if key=='i':
            if verbose:
                print('{:d} Adding Inversion Symmetry'.format(n))
            tM = np.array([[-1.,0.],[0.,-1.]])
            if n==0:
                oM=tM.copy()[:,:,np.newaxis]
            else:
                oM=np.append(oM,tM[:,:,np.newaxis],axis=2)
            oMlbls.append('Inversion')
            n+=1
        #Rotation
        elif key=='r':
            rn = dictValue.size
            for i in range(rn):
                if verbose:
                    print('{:d} Adding Rotation Symmetry {:.2f} degrees'.format(n,dictValue[i]))
                rotang=np.deg2rad(dictValue[i])                   #rotation angle
                tM = np.array([[np.cos(rotang),-np.sin(rotang)],[np.sin(rotang),np.cos(rotang)]])
                if n==0:
                    oM=tM.copy()[:,:,np.newaxis]
                else:
                    oM=np.append(oM,tM[:,:,np.newaxis],axis=2)
                oMlbls.append('Rotation ({:.2f} deg)'.format(dictValue[i]))
                n+=1
        elif key=='m':  #mirror
            mn = dictValue.size
            for i in range(mn):
                if verbose:
                    print('{:d} Adding Mirror Symmetry {:.2f} degrees'.format(n,dictValue[i]))
                mirang=np.deg2rad(dictValue[i])                   #mirror angle
                tM = np.array([[np.cos(mirang*2),np.sin(mirang*2)],[np.sin(mirang*2),-np.cos(mirang*2)]])
                if n==0:
                    oM=tM.copy()[:,:,np.newaxis]
                else:
                    oM=np.append(oM,tM[:,:,np.newaxis],axis=2)
                oMlbls.append('Mirror ({:.2f}deg)'.format(dictValue[i]))
                n=n+1
        else:
            raise ValueError('dictionary item not recognized')

    return oM, oMlbls
