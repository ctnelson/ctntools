from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ctntools.PeakFinding.peakfitting import refinePeaks2D
from ctntools.BaseSupportFunctions.FWHM import FWHM

#Aligns the EELS zero loss peak of an n-dim EELS dataset.  
#Requires the ZLP to be the maximum within a predefined window of the unaligned data.
#Energy axis assumed to be be last dimension.
def EELS_ZLP_Align(indata, zlpmethod='gaussian', Estep=1, i0=None, croptovalid=False, zlpinterp = True, plotoutput=False,  outlierrng=np.array([-np.inf,np.inf])):
    #Inputs:
    #indata             #       EELS n-dimensional array (last dimension must be the EELS energy axis)
    #zlpmethod          #       'max' = max value, 'quad' = parabolic fit around max value, 'gausian' = gaussian fit around max value
    #zlpinterp          #       flag whether to interpolate subpixel fits to new integer axis
    #Estep              #       Energy scale (e.g. eV/pixel)
    #i0                 #       subindex of energy axis to limit the search for ZLP
    #croptovalid        #       flag whether crop EELS cube to energy range valid for all spectra after shifts
    #plotoutput         #       flag to plot output
    

    #Outputs:
    #ind_f              #       position of peak (in initial non-shifted indata)
    #zlp_params         #       ZLP parameters [maxvalue, FWHM]
    #EELS_aligned       #       Aligned Energy Values (n-dims of indata are flattened)
    #xEs                #       Aligned shared x-axis (Energy Axis)
    #xEs0               #       Index of zero in xEs
    #outlierptflag      #       flag of suspected outliers. Triggered where fits fail & when zlp max exceeds min & max scalars of outlierrng * the zlp mean

    #Hardcoded Parameters (potentially will make into inputs if end up adjusting these)
    bkgnd = 10**-5      #       display background

    #input parameters
    idsz = np.array(indata.shape,dtype='int')
    nd = indata.ndim
    #reshape
    indata = np.reshape(indata,(np.prod(idsz[:nd-1]),idsz[-1]))
    #preallocate outputs
    outlierptflag = np.zeros((indata.shape[0],),dtype=bool)
    zlp_params = np.ones((indata.shape[0],2))
    x = np.arange(indata.shape[-1])
    xE = x*Estep

    #if restrict to a prefined range:
    if i0 is None:
        i0 = x     

    #initial coarse fit: max position
    print('Finding maximum (pixel-level)')
    ind_0 = np.argmax(indata[:,i0],axis=1) + i0[0]

    #fit ZLP
    if zlpmethod!='max':
        print('Fitting maximum (subpixel via '+zlpmethod+' fit)')
        #refine peaks
        winsz_ = np.int32(5)
        p = refinePeaks2D(indata.T,ind_0,winsz=winsz_,method=zlpmethod)
        ind_f = p[:,0]
        Rsq = p[:,5]

        #where fit failed
        outlierptflag[np.where(np.any(np.isnan(p),axis=1))] = True                #fit didn't complete
        outlierptflag[np.where(p[:,0]<ind_0-winsz_)] = True                         #returned fit max is outside of window
        outlierptflag[np.where(p[:,0]>ind_0+winsz_)] = True                         #returned fit max is outside of window
    else:
        ind_f = ind_0

    #Interpolate
    if (zlpinterp==True) & (zlpmethod!='max'):
        yi = np.ones_like(indata)
        for i in tqdm(range(indata.shape[0]),desc='Subpixel Interpolation'):
        #for i in range(1):
            cs = interpolate.CubicSpline(x,indata[i,:], bc_type='natural')
            r = ind_f[i]%1
            yi[i,:] = cs(x + r)
            #yi = np.interp(x + r,x,indata[i,:])
        ind_i = np.floor(ind_f)       #new integer index
    else:
        yi = indata
        ind_i = ind_0

    #find outliers (ZLP intensity) 
    zlpmaxval = np.ones((indata.shape[0],))
    ind = np.where(np.isfinite(ind_i))[0]
    zlpmaxval[ind]  = yi[tuple([np.arange(yi.shape[0])[ind],ind_i[ind].astype('int')])]
    zlpmaxmean = np.nanmean(zlpmaxval)
    outlierptflag[ind][np.where(yi[tuple([np.arange(indata.shape[0])[ind],ind_i[ind].astype('int')])]>outlierrng[1]*zlpmaxmean)] = True      #above max
    outlierptflag[ind][np.where(yi[tuple([np.arange(indata.shape[0])[ind],ind_i[ind].astype('int')])]>outlierrng[1]*zlpmaxmean)] = True      #below min

    ibnds = np.array([np.nanmin(ind_i), np.nanmax(ind_i), np.nanargmin(ind_i), np.nanargmax(ind_i)],dtype='int')

    #apply shifts
    print('Shift to Align ZLP')
    pdsz = ibnds[1] - ibnds[0]
    EELS_aligned = np.ones((indata.shape[0],indata.shape[-1]+pdsz))*np.nan
    sind = np.repeat(x[np.newaxis,:],indata.shape[0],axis=0) - np.repeat(ind_i[:,np.newaxis],indata.shape[-1],axis=1) + ibnds[1]
    xx = np.repeat(np.arange(indata.shape[0])[:,np.newaxis],indata.shape[-1],axis=1)
    EELS_aligned[xx[ind,:],sind[ind,:].astype('int')] = yi[ind,:]

    if croptovalid:
        EELS_aligned = EELS_aligned[:,pdsz:-pdsz]
        x = np.arange(EELS_aligned.shape[-1])-ibnds[0]
    else:
        x = np.arange(EELS_aligned.shape[-1])-pdsz-ibnds[0]

    xEs = x*Estep
    xEs0 = np.argmax(xEs>0)-1

    #FWHM
    print('Measure FWHM')
    temp = EELS_aligned/np.repeat(EELS_aligned[:,xEs0][:,np.newaxis],EELS_aligned.shape[1],axis=1)
    fwhm = FWHM(temp.T,ind0=xEs0)
    
    #positive branch
    #fwhmp = np.argmax(temp[:,xEs0:]<0.5,axis=1)
    #dyp = temp[tuple([np.arange(EELS_aligned.shape[0]),xEs0+fwhmp-1])] - temp[tuple([np.arange(EELS_aligned.shape[0]),xEs0+fwhmp])]
    #dxp = (temp[tuple([np.arange(EELS_aligned.shape[0]),xEs0+fwhmp-1])]-.5)/dyp * 1
    #fwhmp_fr = fwhmp+dxp-1
    #negative branch
    #fwhmn = np.argmax(np.flip(temp[:,:xEs0],axis=1)<0.5,axis=1) + 1
    #dyn = temp[tuple([np.arange(EELS_aligned.shape[0]),xEs0-fwhmn])] - temp[tuple([np.arange(EELS_aligned.shape[0]),xEs0-fwhmn+1])]
    #dxn = (temp[tuple([np.arange(EELS_aligned.shape[0]),xEs0-fwhmn+1])]-.5)/dyn * 1
    #fwhmn_fr = -fwhmn+dxn+1
    #total
    #fwhm = fwhmp_fr - fwhmn_fr

    zlp_params[:,0] = EELS_aligned[:,xEs0]
    zlp_params[:,1] = fwhm * Estep

    if plotoutput:
        print('Plotting Results')
        ind = np.where(outlierptflag==False)[0]
        #plot (normalizes data)
        stp = int(np.ceil(indata.shape[0]/indata.shape[-1]))
        #zbnds = np.vstack((np.nanpercentile(indata,0.01,axis=1),np.nanpercentile(indata,99.99,axis=1))).T       #using percentile
        zbnds = np.vstack((np.nanmin(indata,axis=1),np.nanmax(indata,axis=1))).T       #using min/max

        #normalize
        zmin_arr = np.repeat(zbnds[:,0][:,np.newaxis],indata.shape[1],axis=1)
        zmax_arr = np.repeat(zbnds[:,1][:,np.newaxis],indata.shape[1],axis=1)
        dsv_norm = (indata      -zmin_arr)/(zmax_arr-zmin_arr)
        
        zmin_arr = np.repeat(zbnds[:,0][:,np.newaxis],EELS_aligned.shape[1],axis=1)
        zmax_arr = np.repeat(zbnds[:,1][:,np.newaxis],EELS_aligned.shape[1],axis=1)
        dss_norm = (EELS_aligned-zmin_arr)/(zmax_arr-zmin_arr)
        #dsv_norm = (indata-zbnds[0])/(zbnds[1]-zbnds[0])
        #dss_norm = (dstack_shift-zbnds[0])/(zbnds[1]-zbnds[0])

        fig = plt.figure(layout='constrained', figsize=(20, 10))
        gs=plt.GridSpec(2,4,figure=fig)

        #Original
        ax0 = fig.add_subplot(gs[0,:2])
        #ax0.plot(xE,np.log10(np.nanmax(dsv_norm[ind,:]+bkgnd,axis=0)),'-k')  #plot envelope
        ax0.fill_between(xE,np.log10(np.nanmax(dsv_norm[ind,:]+bkgnd,axis=0)),np.log10(bkgnd),alpha=.1,color='k', label='max value')  #plot envelope
        ax0.fill_between(xE,np.log10(np.nanpercentile(dsv_norm[ind,:]+bkgnd,75,axis=0)),np.log10(np.nanpercentile(dsv_norm[ind,:]+bkgnd,25,axis=0)),alpha=1,color='c', label='1st-3rd quartile')  #plot envelope
        ax0.plot(xE,np.log10(dsv_norm[ibnds[3],:]+bkgnd),'-r', label='max ZLP position')
        ax0.plot(xE[np.nanmax(ind_i.ravel()).astype('int')],np.log10(dsv_norm[ibnds[3],np.nanmax(ind_i.ravel()).astype('int')]+bkgnd),'or')
        ax0.plot(xE,np.log10(dsv_norm[ibnds[2],:]+bkgnd),'-b', label='min ZLP position')
        ax0.plot(xE[np.nanmin(ind_i.ravel()).astype('int')],np.log10(dsv_norm[ibnds[2],np.nanmin(ind_i.ravel()).astype('int')]+bkgnd),'ob')
        ax0.set_title('Initial')
        ax0.legend()

        #Image
        ax1 = fig.add_subplot(gs[0,2])
        ax1.imshow(np.log10(dsv_norm[::stp,:]+bkgnd))
        if stp>1:
            ax1.set_title('Initial Dataset (subsampled)')
        else:
            ax1.set_title('Initial Dataset')

        #Corrected
        ax2 = fig.add_subplot(gs[1,:2])
        #ax2.plot(xEs,np.log10(np.nanmax(dss_norm[ind,:]+bkgnd,axis=0)),'-k')  #plot envelope
        ax2.fill_between(xEs,np.log10(np.nanmax(dss_norm[ind,:]+bkgnd,axis=0)),np.log10(bkgnd),alpha=.1,color='k')  #plot envelope
        ax2.fill_between(xEs,np.log10(np.nanpercentile(dss_norm[ind,:]+bkgnd,75,axis=0)),np.log10(np.nanpercentile(dss_norm[ind,:]+bkgnd,25,axis=0)),alpha=1,color='c')  #plot envelope
        ax2.set_title('Aligned')

        #Image
        ax3 = fig.add_subplot(gs[1,2])
        ax3.imshow(np.log10(dss_norm[::stp,:]+bkgnd))
        if stp>1:
            ax3.set_title('Aligned Dataset (subsampled)')
        else:
            ax3.set_title('Aligned Dataset')

        fwhmmn = np.ceil(np.nanmean(fwhm/2)).astype('int')
        ax4 = fig.add_subplot(gs[:,3])
        ax4.imshow((dss_norm[::stp*10,xEs0-fwhmmn:xEs0+fwhmmn+1]+bkgnd))
        if stp>1:
            ax4.set_title('Aligned ZLP (subsampled)')
        else:
            ax4.set_title('Aligned ZLP')

    #outputs
    return ind_f, EELS_aligned, xEs, xEs0, zlp_params, outlierptflag
