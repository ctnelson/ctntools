import numpy as np

#############################  Table of Contents  ##############################
# distWrapped    :    Shortest distance on 1-D periodic axis

################################## distWrapped ################################# 
#######################  Shortest distance on 1-D periodic axis ################
#Output has sign, take absolute value if only need distance 
def distWrapped(d1,d2,per=2*np.pi):
    ### Inputs ###
    #d1    :    first position(s)
    #d2    :    second position(s)
    #per   :    wrap period (defaults to radian circle 2*pi)
    ### Outputs ###
    #deld  :    distance(s)

    #Setup
    d1 = np.array(d1)
    d2 = np.array(d2)
    if d1.size != d2.size:
        if d2.size==1:
            d2 = np.ones_like(d1)*d2
        elif d1.size==1:
            d1 = np.ones_like(d2)*d1
        else:
            raise ValueError('input sizes {:d},{:d} not compatible'.format(d1.size,d2.size))
    #Function
    deld = (( (d2-d1) + per/2) % per - per/2)
    #Return
    return deld
