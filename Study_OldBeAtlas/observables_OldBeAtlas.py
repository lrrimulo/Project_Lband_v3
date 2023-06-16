"""
This is a program used to calculate a lot of observables, derived from 
the SEDs, from the BeAtlas grid. 

It reads the grid. Then, it performs a lot of automatic calculations 
on the grids's files in order to calculate the observables. 

Since this program usually takes several seconds or minutes to run, 
it was made separated from the project's main program ('analysis_BeAtlas.py'). 
"""

import glob as glob
import numpy as np
import pyhdust as hdt
import pyhdust.phc as phc
import pyhdust.lrr as lrr
import pyhdust.lrr.roche_singlestar as rss
import pyhdust.spectools as spt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


###########################

### Name of the output file, where the observables will be printed:
outputfile = "observables_BeAtlas.inp"


### Reading the fullsed, source and temperature files from the grid of 
### HDUST models (This may take a few seconds or even minutes.)

import read_everything as read_everything

files_fullsed_new, files_source_new, files_temps_new, fullsed_contents, \
        fullsed_path, source_path, temps_path, dist_std = \
        read_everything.read_everything()
   

### Turn this on to check if the grid was completely computed:
if 1==2:
    fraction_computed, missing = \
        read_everything.check_completeness(files_fullsed_new)
        
    print("Fraction of the grid that was computed = "+\
        str(fraction_computed))
    import sys; sys.exit()


#############################
### 

### Comment the next command line if you want to skip this part.
if 1==2:
    for ifile in range(0,len(fullsed_contents)):
        print(fullsed_contents[ifile][1][4])
    
    import sys; sys.exit()




#############################
### Obtaining S/N ratios of the HDUST models for some intervals
### (These are numbers that we should use to evaluate if the number of 
### photons used in the HDUST simulations are sufficient.)

def powerlaw(x,C,alpha):
    """
    Assuming a power-law for the fitting of a portion of the SED.
    """
    return C*x**alpha

def SNratios_procedure(SNratios, files_fullsed_new, fullsed_contents):
    """
    In this procedure, it is tried to fit a power-law to certain regions 
    of the SED calculated by HDUST. Then, a signal and a noise are 
    calculated and their ratio is saved, for future evaluation of the 
    number of photons used in the HDUST simulations.
    
    I believe that the S/N ratios of the HDUST models should be always 
    greater than 100 in all of the desired domains.
    """


    ### Left and right boundaries of the regions where the 
    ### S/N ratio will be estimated:
    lbda = [3.41,3.93,0.50] ### left boundaries [microns]
    lbdb = [3.47,4.00,0.55] ### right boundaries [microns]

    for ifile in range(0,len(files_fullsed_new)):
        print("Obtaining S/N ratios for file")
        print(files_fullsed_new[ifile][0])

        contents=fullsed_contents[ifile][1]

        auxiSNratios=[]
        ### Loop over inclinations
        for incs in range(0,len(contents[1])):
            xlp=contents[2][incs,:,0]   ### lambda [microns]
            ylp=contents[2][incs,:,1]   ### HDUST's flux [microns^-1]

            ### Notice here that I am assuming that 
            ### len(lbda) == len(lbdb) and len(lbda) > 0
            auxi2SNratios=[]
            ### Loop over the defined regions for extracting S/N ratio
            for ilbda in range(0,len(lbda)):
                ### Collecting indexes in the region inside boundaries
                idx=[]
                for ii in range(0,len(xlp)):
                    if lbda[ilbda] <= xlp[ii] <= lbdb[ilbda]:
                        idx.append(ii)
                idx=np.array(idx)
                
                
                if len(idx) > 1:
                    try:
                        ### Arrays in the domain for extracting S/N ratio:
                        xlp_ajust = np.array(xlp[idx])
                        ylp_ajust = np.array(ylp[idx])
                        
                        ### Finding coefficients of the power-law fitting
                        popt, pcov = curve_fit(powerlaw, xlp_ajust, ylp_ajust)
                        ylp_ajusted = powerlaw(xlp_ajust,popt[0],popt[1])
                        ### The noise is the mean of the quadratic variations
                        ### between the HDUST points and the fitted power-law
                        diff2 = (ylp_ajust-ylp_ajusted)**2.
                        noise = np.sqrt(diff2.mean())
                        ### The signal is the value of the fitted power-law 
                        ### in the middle of the domain:
                        signal = powerlaw(0.5*(lbda[ilbda]+lbdb[ilbda]),\
                                    popt[0],popt[1])
                        auxi2SNratios.append([lbda[ilbda],lbdb[ilbda],signal/noise])
                    except:
                        auxi2SNratios.append([lbda[ilbda],lbdb[ilbda],np.nan])
                else:
                    auxi2SNratios.append([lbda[ilbda],lbdb[ilbda],np.nan])
                    
            auxiSNratios.append(auxi2SNratios)
        ### 'SNratios' is of the form:
        ### 'SNratios[ifile][inc][bondary]' = [lamb1,lamb2,SNR]
        SNratios.append(auxiSNratios)

    return SNratios


SNratios = []
### Comment the next command line if you want to skip this part.
### (But remember that this part is necessary in the output writing part.)
SNratios = SNratios_procedure(SNratios, files_fullsed_new, fullsed_contents)


#############################
### Obtaining line observables

### Maximum level for the hydrogen atom (25 was the number assumed in 
### the Old BeAtlas)
N=25
### Left and right boundaries of the spectral region where the transitions
### are to be found [microns]
lamb_left = 3.3
lamb_right = 4.1

### 
selectedlines=[]
lambs=[]
ji=[]
### transitions, from j to i (emission)
for i in range(1,N):
    for j in range(i+1,N+1):
        lamb = spt.hydrogenlinewl(j, i)*1e6 ### lambda [microns]
        ### If this transition is in the desired lambda domain, save it:
        if lamb_left <= lamb <= lamb_right:
            selectedlines.append([j,i,lamb])
            lambs.append(lamb)
            ji.append([j,i])

def polynom_3(x,x0,a0,a1,a2,a3):
    """
    A third degree polynomial, used for continuum fitting.
    """
    return a0+a1*(x-x0)+a2*(x-x0)*(x-x0)+a3*(x-x0)*(x-x0)*(x-x0)

def HDUST_Lband_spectra(contents,lamb_left,lamb_right,file_pars):

    ### assumed distance [parsecs] for the calculations
    dist_std = 10.

    ### Exclude lambdas within |lambda - lamb0| = 'v_exclude' around the 
    ### lamb0's:
    v_exclude = 1200e5
    intervals_exclude = []
    for ilamb in range(0,len(lambs)):
        intervals_exclude.append([  
                lambs[ilamb]*(1.-v_exclude/phc.c.cgs),\
                lambs[ilamb]*(1.+v_exclude/phc.c.cgs)\
                                    ])
    
    ### 
    lambss = []
    fluxes = []
    flux_fc = []
    ### Loop over inclinations
    for incs in range(0,len(contents[1])):
        
        ### 'xlp' and 'ylp' will contain the spectrum in the desired region
        ### in [microns] and [erg/s cm2 A].
        ### 
        ### Obtaining HDUST spectrum
        xlp=contents[2][incs,:,0]   ### lambda [microns]
        ylp=contents[2][incs,:,1]   ### HDUST's flux [microns^-1]\
        ### Flux [erg/s cm2 A] (assuming distance of 'dist_std')
        ylp = np.array([el*contents[4][3]*phc.Lsun.cgs*1e-4/\
                4./np.pi/(dist_std*dist_std*phc.pc.cgs*phc.pc.cgs) \
                for el in ylp])
        if 1==2:
            plt.plot(np.log10(xlp),np.log10(ylp))
            plt.show()
        ### Selecting spectrum in the desired region
        xaux = []
        yaux = []
        for ix in range(0,len(xlp)):
            if lamb_left <= xlp[ix] <= lamb_right:
                xaux.append(xlp[ix])
                yaux.append(ylp[ix])
        xlp = np.array([el for el in xaux])
        ylp = np.array([el for el in yaux])
        
        ### Now, fitting the continuum for its normalization with the spectrum
        ###
        ### First, excluding the regions around the hydrogen lines
        x_exc = np.array([el for el in xlp])
        y_exc = np.array([el for el in ylp])
        for iint in range(0,len(intervals_exclude)):
            for iel in range(0,len(x_exc)):
                if ~np.isnan(x_exc[iel]) and \
                        intervals_exclude[iint][0] <= x_exc[iel] <= \
                        intervals_exclude[iint][1]:
                    x_exc[iel] = np.nan
                    y_exc[iel] = np.nan
        ### Excluding the NaNs too, for the fitting process (below)
        log10_x_exc_nonan = []
        log10_y_exc_nonan = []
        for ix in range(0,len(x_exc)):
            if ~np.isnan(x_exc[ix]):
                log10_x_exc_nonan.append(np.log10(x_exc[ix]))
                log10_y_exc_nonan.append(np.log10(y_exc[ix]))
        log10_x_exc_nonan = np.array([el for el in log10_x_exc_nonan])
        log10_y_exc_nonan = np.array([el for el in log10_y_exc_nonan])
        ### If there is, at least, 4 points in the lineless spectrum, 
        ### make the fitting:
        if len(log10_x_exc_nonan) > 3:
            converged = False
            trials = 1
            first_guess = [
                    1.30388011,
                    -14.66027978,
                    -14.12250988,
                    10.19202748,
                    -1.73518028
                    ]
            current_guess = [el for el in first_guess]
            ### 
            while converged == False:
                try:
                    popt, pcov = curve_fit(polynom_3, log10_x_exc_nonan, \
                            log10_y_exc_nonan, p0 = current_guess)
                    converged = True
                    #print("Converged after "+str(trials)+" trials.")
                
                except:
                    popt = [np.nan,np.nan,np.nan,np.nan,np.nan]
                    current_guess = [el*np.random.normal(0.,2.) for el in current_guess]
                    trials += 1
                
    
            y_r_lin = np.array([ylp[i]/10.**polynom_3(np.log10(xlp[i]),popt[0],popt[1],\
                    popt[2],popt[3],popt[4]) for i in range(0,len(xlp))])
                    
            x_curve = np.linspace(np.nanmin(xlp),np.nanmax(xlp),401)
            y_curve = np.array([10.**polynom_3(np.log10(x_curve[i]),popt[0],popt[1],\
                    popt[2],popt[3],popt[4]) for i in range(0,len(x_curve))])
        else:
            popt = [np.nan,np.nan,np.nan,np.nan,np.nan]
            y_r_lin = np.array([np.nan for i in range(0,len(xlp))])
            print("SOMETHING IS WRONG WITH THIS HDUST OUTPUT!")
    
        ### Turn this on to check the fitting that was done.
        if 1==2:
            if 1==1:
                ymax = np.nanmax(ylp)
                ymin = np.nanmin(ylp)
                for i in range(0,len(lambs)):
                    plt.plot([lambs[i],lambs[i]],[ymin,ymax],color="black",linestyle=":")
            plt.plot(xlp,ylp,color="blue")
            plt.plot(x_exc,y_exc,color="green")
            if True in [~np.isnan(y_r_lin[i]) for i in range(0,len(y_r_lin))]:
                plt.plot(x_curve,y_curve,color="green")
            plt.show()
        
            if True in [~np.isnan(y_r_lin[i]) for i in range(0,len(y_r_lin))]:
                plt.plot([np.nanmin(xlp),np.nanmax(xlp)],[1.,1.],color="black",linestyle=":")
                plt.plot(xlp,y_r_lin,color="green")
                plt.ylim([0.,np.nanmax(y_r_lin)*1.1])
                plt.show()
        
        ### Turn this on to make figures of the spectra and normalized spectra.        
        if 1==2:
            ### Destination folder
            dest_folder = "./../../"
            ### Parameters for the names of the output files.
            par0 = str(file_pars[0])
            par1 = str(file_pars[1])
            par2 = str(file_pars[2])
            par3 = str(file_pars[3])
            parcosi = str(abs(round(contents[1][incs],2)))
            ### Plot of spectrum 
            plt.plot(xlp,ylp,color="black")
            plt.xlabel("$\lambda\,[\mathrm{\mu m}]$")
            plt.ylabel("$F_\lambda\,[\mathrm{erg\,s^{-1}\,cm^{-2}\,A}]$")
            if ~np.isnan(np.nanmax(ylp)):
                plt.ylim([0.,1.1*np.nanmax(ylp)])
            plt.tight_layout()
            plt.savefig(dest_folder+"Flux__"+par0+"_"+par1+"_"+par2+"_"+par3+"_"+parcosi+".png")
            plt.close()
            ### Plot of normalized spectrum 
            plt.plot(xlp,y_r_lin,color="black")
            plt.xlabel("$\lambda\,[\mathrm{\mu m}]$")
            plt.ylabel("$F_\lambda/F_\mathrm{c}$")
            if ~np.isnan(np.nanmax(y_r_lin)):
                plt.ylim([0.,1.1*np.nanmax(y_r_lin)])
            plt.tight_layout()
            plt.savefig(dest_folder+"F_Fc__"+par0+"_"+par1+"_"+par2+"_"+par3+"_"+parcosi+".png")
            plt.close()
        
        ### 
        lambss.append(xlp)
        fluxes.append(ylp)
        flux_fc.append(y_r_lin)
        
            
    return lambss, fluxes, flux_fc

def polynom_1(x,x0,a0,a1):
    """
    A first degree polynomial, used for continuum fitting of a 
    small portion of spectrum.
    """
    return a0+a1*(x-x0)

def HDUST_Halpha_spectra(contents,lamb_left,lamb_right,file_pars):

    ### assumed distance [parsecs] for the calculations
    dist_std = 10.

    ### Exclude lambdas within |lambda - lamb0| = 'v_exclude' around the 
    ### lamb0's:
    v_exclude = 1200e5
    intervals_exclude = []
    intervals_exclude.append([  
            0.5*(lamb_left+lamb_right)*(1.-v_exclude/phc.c.cgs),\
            0.5*(lamb_left+lamb_right)*(1.+v_exclude/phc.c.cgs)\
                                ])
    
    ### 
    lambss = []
    fluxes = []
    flux_fc = []
    ### Loop over inclinations
    for incs in range(0,len(contents[1])):
        
        ### 'xlp' and 'ylp' will contain the spectrum in the desired region
        ### in [microns] and [erg/s cm2 A].
        ### 
        ### Obtaining HDUST spectrum
        xlp=contents[2][incs,:,0]   ### lambda [microns]
        ylp=contents[2][incs,:,1]   ### HDUST's flux [microns^-1]\
        ### Flux [erg/s cm2 A] (assuming distance of 'dist_std')
        ylp = np.array([el*contents[4][3]*phc.Lsun.cgs*1e-4/\
                4./np.pi/(dist_std*dist_std*phc.pc.cgs*phc.pc.cgs) \
                for el in ylp])
        ### Selecting spectrum in the desired region
        xaux = []
        yaux = []
        for ix in range(0,len(xlp)):
            if lamb_left <= xlp[ix] <= lamb_right:
                xaux.append(xlp[ix])
                yaux.append(ylp[ix])
        xlp = np.array([el for el in xaux])
        ylp = np.array([el for el in yaux])
        
        ### Now, fitting the continuum for its normalization with the spectrum
        ###
        ### First, excluding the regions around the hydrogen lines
        x_exc = np.array([el for el in xlp])
        y_exc = np.array([el for el in ylp])
        for iint in range(0,len(intervals_exclude)):
            for iel in range(0,len(x_exc)):
                if ~np.isnan(x_exc[iel]) and \
                        intervals_exclude[iint][0] <= x_exc[iel] <= \
                        intervals_exclude[iint][1]:
                    x_exc[iel] = np.nan
                    y_exc[iel] = np.nan
        ### Excluding the NaNs too, for the fitting process (below)
        log10_x_exc_nonan = []
        log10_y_exc_nonan = []
        for ix in range(0,len(x_exc)):
            if ~np.isnan(x_exc[ix]):
                log10_x_exc_nonan.append(np.log10(x_exc[ix]))
                log10_y_exc_nonan.append(np.log10(y_exc[ix]))
        log10_x_exc_nonan = np.array([el for el in log10_x_exc_nonan])
        log10_y_exc_nonan = np.array([el for el in log10_y_exc_nonan])
        ### If there is, at least, 4 points in the lineless spectrum, 
        ### make the fitting:
        if len(log10_x_exc_nonan) > 3:
            converged = False
            trials = 1
            first_guess = [
                    1.30388011,
                    -14.66027978,
                    -14.12250988
                    ]
            current_guess = [el for el in first_guess]
            ### 
            while converged == False:
                try:
                    popt, pcov = curve_fit(polynom_1, log10_x_exc_nonan, \
                            log10_y_exc_nonan, p0 = current_guess)
                    converged = True
                    #print("Converged after "+str(trials)+" trials.")
                
                except:
                    popt = [np.nan,np.nan,np.nan]
                    current_guess = [el*np.random.normal(0.,2.) for el in current_guess]
                    trials += 1
                
    
            y_r_lin = np.array([ylp[i]/10.**polynom_1(np.log10(xlp[i]),popt[0],popt[1],\
                    popt[2]) for i in range(0,len(xlp))])
                    
            x_curve = np.linspace(np.nanmin(xlp),np.nanmax(xlp),401)
            y_curve = np.array([10.**polynom_1(np.log10(x_curve[i]),popt[0],popt[1],\
                    popt[2]) for i in range(0,len(x_curve))])
        else:
            popt = [np.nan,np.nan,np.nan]
            y_r_lin = np.array([np.nan for i in range(0,len(xlp))])
            print("SOMETHING IS WRONG WITH THIS HDUST OUTPUT!")
    
        ### Turn this on to check the fitting that was done.
        if 1==2:
            plt.plot(xlp,ylp,color="blue")
            plt.plot(x_exc,y_exc,color="green")
            if True in [~np.isnan(y_r_lin[i]) for i in range(0,len(y_r_lin))]:
                plt.plot(x_curve,y_curve,color="green")
            plt.show()
        
            if True in [~np.isnan(y_r_lin[i]) for i in range(0,len(y_r_lin))]:
                plt.plot([np.nanmin(xlp),np.nanmax(xlp)],[1.,1.],color="black",linestyle=":")
                plt.plot(xlp,y_r_lin,color="green")
                plt.ylim([0.,np.nanmax(y_r_lin)*1.1])
                plt.show()
        
        ### Turn this on to make figures of the spectra and normalized spectra.        
        if 1==2:
            ### Destination folder
            dest_folder = "./../../"
            ### Parameters for the names of the output files.
            par0 = str(file_pars[0])
            par1 = str(file_pars[1])
            par2 = str(file_pars[2])
            par3 = str(file_pars[3])
            parcosi = str(abs(round(contents[1][incs],2)))
            ### Plot of spectrum 
            plt.plot(xlp,ylp,color="black")
            plt.xlabel("$\lambda\,[\mathrm{\mu m}]$")
            plt.ylabel("$F_\lambda\,[\mathrm{erg\,s^{-1}\,cm^{-2}\,A}]$")
            plt.tight_layout()
            plt.savefig(dest_folder+"HaFlux__"+par0+"_"+par1+"_"+par2+"_"+par3+"_"+parcosi+".png")
            plt.close()
            ### Plot of normalized spectrum 
            plt.plot(xlp,y_r_lin,color="black")
            plt.xlabel("$\lambda\,[\mathrm{\mu m}]$")
            plt.ylabel("$F_\lambda/F_\mathrm{c}$")
            plt.tight_layout()
            plt.savefig(dest_folder+"HaF_Fc__"+par0+"_"+par1+"_"+par2+"_"+par3+"_"+parcosi+".png")
            plt.close()
        
        ### 
        lambss.append(xlp)
        fluxes.append(ylp)
        flux_fc.append(y_r_lin)
        
            
    return lambss, fluxes, flux_fc


### WARNING!: If you add or remove lines from the lists below, 
### you will have to change the writing to an external file 
### at the end of this program.

### Central lambda [microns], width [km/s]
lbc_Bralpha = [spt.hydrogenlinewl(5, 4)*1e6,1000.]
lbc_Pfgamma = [spt.hydrogenlinewl(8, 5)*1e6,1000.]
lbc_Brgamma = [spt.hydrogenlinewl(7, 4)*1e6,1000.]
lbc_Humphreyoth =   [
                    [spt.hydrogenlinewl(14, 6)*1e6,1000.,"Humphrey14"],
                    [spt.hydrogenlinewl(15, 6)*1e6,1000.,"Humphrey15"],
                    [spt.hydrogenlinewl(16, 6)*1e6,1000.,"Humphrey16"],
                    ###[spt.hydrogenlinewl(17, 6)*1e6,1000.,"Humphrey17"],  
                            ### This line merges with Pfgamma
                    [spt.hydrogenlinewl(18, 6)*1e6,1000.,"Humphrey18"],
                    [spt.hydrogenlinewl(19, 6)*1e6,1000.,"Humphrey19"],
                    [spt.hydrogenlinewl(20, 6)*1e6,1000.,"Humphrey20"],
                    [spt.hydrogenlinewl(21, 6)*1e6,1000.,"Humphrey21"],
                    [spt.hydrogenlinewl(22, 6)*1e6,1000.,"Humphrey22"],
                    [spt.hydrogenlinewl(23, 6)*1e6,1000.,"Humphrey23"],
                    [spt.hydrogenlinewl(24, 6)*1e6,1000.,"Humphrey24"],
                    [spt.hydrogenlinewl(25, 6)*1e6,1000.,"Humphrey25"]   
                                                    ### The last Humphrey 
                                                    ### transition calculated 
                                                    ### by HDUST
                    ]
lbc_Bracketoth =    [
                    [spt.hydrogenlinewl(3, 2)*1e6,1000.,"Halpha"],
                    [spt.hydrogenlinewl(4, 2)*1e6,1000.,"Hbeta"],
                    [spt.hydrogenlinewl(5, 2)*1e6,1000.,"Hgamma"]
                    ]


### Limits of the BL and RL bands [microns], as defined by 
### Mennickent et al. 2009PASP..121..125M
lamb1_BL = 3.41 ; lamb2_BL = 3.47
lamb1_RL = 3.93 ; lamb2_RL = 4.00

def gaussian_fit(x,sigma2,A):
    """
    This is a gaussian function centered on x = 0 and with area 'A' and 
    variance 'sigma2'.
    """
    return 0.+A/np.sqrt(2.*np.pi*sigma2)*np.exp(-0.5*(x-0.)**2./sigma2)

def obtaining_flux_ew_PS(contents,lbc,hwidth):
    """
    A procedure for calculating the flux, EW, peak separation and FWHM 
    parameters of a line whose center is 'lbc' [microns] and whose 
    width is 'hwidth' [km/s].
    
    Returns arrays of (for every inclination):
    * line flux [erg/s cm^2]
    * EW [Angstroms]
    * Peak separation [km/s]
    * [FWHM, area] [km/s, km/s]
    """
    
    print("obtaining_flux_ew_PS, for lambda = "+str(lbc)+" [microns]")
    
    auxilinflux=[]
    auxiew=[]
    auxiPS=[]
    auxiFWHM=[]
    ### Loop over inclinations
    for incs in range(0,len(contents[1])):
        xlp=contents[2][incs,:,0]   ### lambda [microns]
        ylp=contents[2][incs,:,1]   ### HDUST's flux [microns^-1]
  
        ### Doppler velocities [km/s]
        vels = (xlp-lbc)/lbc*phc.c.cgs*1e-5
        ### line flux [erg/s cm^2]
        linflux=spt.absLineCalc(vels, ylp, vw=hwidth)
        linflux=linflux * contents[4][3]*phc.Lsun.cgs/4./np.pi/\
                    (dist_std*phc.pc.cgs)**2.*1e5*lbc/phc.c.cgs
        auxilinflux.append(linflux)

        ### Returning an array of velocities [km/s]: 'xplot' 
        ### and an array with the normalized flux: 'yplot'
        xplot,yplot=spt.lineProf(xlp, ylp, lbc, hwidth=hwidth)
        ### Turn this on to see the line profile:
        if 1==2:
            plt.plot(xplot,abs(yplot-1.),color="black")
            plt.show()
        
        ### Equivalent width [Angstroms]
        ew=spt.EWcalc(xplot, yplot, vw=hwidth)
        ew = ew*lbc/phc.c.cgs*1e9
        auxiew.append(ew)
            
        ### Try to calculate the peak separation in [km/s]
        try:
            v1,v2=spt.PScalc(xplot, yplot, vc=0.0, ssize=0.05, \
                            gaussfit=True)
        except:
            v1=np.nan; v2=np.nan
        auxiPS.append(v2-v1)
        
        ### Trial of calculating the FWHM: A gaussian is ajusted to 
        ### the absolute value of the line profile. The FWHM of this 
        ### gaussian is extracted [km/s]. Also, the area [km/s] is extracted.
        trials = 1
        trials_max = 20
        converged = False
        while converged == False and trials <= trials_max:
            try:
                popt, pcov = curve_fit(gaussian_fit, xplot, abs(yplot-1.),\
                        p0=[1000.*abs(np.random.normal(0.,2.)),\
                            10.*abs(np.random.normal(0.,2.))])
                fwhm = np.sqrt(8.*popt[0]*np.log(2))
                area = popt[1]
                ### Turn this on to see the line profile and the fitted gaussian
                if 1==2:
                    plt.plot(xplot,abs(yplot-1.),color="black")
                    plt.plot(xplot,gaussian_fit(xplot,popt[0],popt[1]),color="red")
                    plt.show()
                converged = True
                #print("Converged after "+str(trials)+" trials.")
            except:
                fwhm = np.nan
                area = np.nan
                trials += 1
        if trials > trials_max:
            print("Didn't find a gaussian fit for the line profile!!")
        auxiFWHM.append([fwhm,area])
            
    auxilinflux=np.array(auxilinflux)
    auxiew=np.array(auxiew)
    auxiPS=np.array(auxiPS)
    auxiFWHM=np.array(auxiFWHM)
    #print(auxiFWHM)

    return auxilinflux,auxiew,auxiPS,auxiFWHM


def line_observables(Bralpha,Pfgamma,Brgamma,Humphreys,Brackets,BL,RL,
        Lbandspectrum,Halphaspectrum,files_fullsed_new, fullsed_contents):
    """
    In this procedure, the interesting line observables are calculated for 
    each of the desired lines.
    """

    
    for ifile in range(0,len(files_fullsed_new)):
        print("Obtaining line observables for file")
        print(files_fullsed_new[ifile][0])

        ### First, the L-band spectra and normalized spectra are obtained
        file_pars = files_fullsed_new[ifile][0]
        contents=fullsed_contents[ifile][1]
        
        lambss, fluxes, flux_fc = \
                HDUST_Lband_spectra(contents,lamb_left,lamb_right,file_pars)
        
        Lbandspectrum.append([lambss, fluxes, flux_fc])
        
        ### Then, the Halpha spectra and normalized spectra are obtained
        file_pars = files_fullsed_new[ifile][0]
        contents=fullsed_contents[ifile][1]
        
        lambHa = spt.hydrogenlinewl(3, 2)*1e6
        lambHa_left = lambHa*(1.-10000e5/phc.c.cgs)
        lambHa_right = lambHa*(1.+10000e5/phc.c.cgs)
        lambss, fluxes, flux_fc = \
                HDUST_Halpha_spectra(contents,lambHa_left,lambHa_right,file_pars)
        
        Halphaspectrum.append([lambss, fluxes, flux_fc])
        


        ### Then, the line observables of the desired lines are calculated

        ### Line observables for the Br alpha    
        hwidth=lbc_Bralpha[1]
        lbc=lbc_Bralpha[0]
        contents=fullsed_contents[ifile][1]
            
        auxilinflux,auxiew,auxiPS,auxiFWHM = \
            obtaining_flux_ew_PS(contents,lbc,hwidth)
            
        Bralpha.append([auxilinflux,auxiew,auxiPS,auxiFWHM])

        ### Line observables for the Br gamma
        hwidth=lbc_Brgamma[1]
        lbc=lbc_Brgamma[0]
        contents=fullsed_contents[ifile][1]
            
        auxilinflux,auxiew,auxiPS,auxiFWHM = \
            obtaining_flux_ew_PS(contents,lbc,hwidth)
            
        Brgamma.append([auxilinflux,auxiew,auxiPS,auxiFWHM])
        
        ### Line observables for the Pf gamma
        hwidth=lbc_Pfgamma[1]
        lbc=lbc_Pfgamma[0]
        contents=fullsed_contents[ifile][1]
            
        auxilinflux,auxiew,auxiPS,auxiFWHM = \
            obtaining_flux_ew_PS(contents,lbc,hwidth)
            
        Pfgamma.append([auxilinflux,auxiew,auxiPS,auxiFWHM])

        ### Line observables for the Bracket's
        Bracketsauxi=[]
        for iHump in range(0,len(lbc_Bracketoth)):
        
            hwidth=lbc_Bracketoth[iHump][1]
            lbc=lbc_Bracketoth[iHump][0]
            contents=fullsed_contents[ifile][1]
            
            auxilinflux,auxiew,auxiPS,auxiFWHM = \
                obtaining_flux_ew_PS(contents,lbc,hwidth)
            
            Bracketsauxi.append([auxilinflux,auxiew,auxiPS,auxiFWHM])
        Brackets.append(Bracketsauxi)

        ### Line observables for the Humphrey's
        Humphreysauxi=[]
        for iHump in range(0,len(lbc_Humphreyoth)):
        
            hwidth=lbc_Humphreyoth[iHump][1]
            lbc=lbc_Humphreyoth[iHump][0]
            contents=fullsed_contents[ifile][1]
            
            auxilinflux,auxiew,auxiPS,auxiFWHM = \
                obtaining_flux_ew_PS(contents,lbc,hwidth)
            
            Humphreysauxi.append([auxilinflux,auxiew,auxiPS,auxiFWHM])
        Humphreys.append(Humphreysauxi)

        


        ### Now, obtaining the B and R fluxes defined by 
        ### Mennickent et al. 2009PASP..121..125M
        print("Obtaining the B and R fluxes defined by Mennickentet al. 2009.")
        contents=fullsed_contents[ifile][1]
        BLfluxauxi=[]
        RLfluxauxi=[]
        ### Loop over inclinations
        for incs in range(0,len(contents[1])):
            xlp=contents[2][incs,:,0]   ### lambda [microns]
            ylp=contents[2][incs,:,1]   ### HDUST's flux [microns^-1]
            Nnpts=int(50.00)    ### this number must be >= 3. A good choice is 50.
            
            ### Obtaining F(BL) [erg/s cm^2]
            llamb = np.array([lamb1_BL+(lamb2_BL-lamb1_BL)/\
                float(Nnpts-1)*float(i) for i in range(0,Nnpts)])
            dllamb = np.array([llamb[i+1]-llamb[i] for i in range(0,Nnpts-1)])
            ylpf = np.array([lrr.interpLinND([llamb[i]],[xlp],ylp) \
                for i in range(0,Nnpts)])
            BLflux = lrr.integrate_trapezia(ylpf,dllamb)
            BLflux = BLflux * contents[4][3]*phc.Lsun.cgs/4./np.pi/\
                            (dist_std*phc.pc.cgs)**2.
            BLfluxauxi.append(BLflux)
            
            ### Obtaining F(RL) [erg/s cm^2]
            llamb = np.array([lamb1_RL+(lamb2_RL-lamb1_RL)/\
                float(Nnpts-1)*float(i) for i in range(0,Nnpts)])
            dllamb = np.array([llamb[i+1]-llamb[i] for i in range(0,Nnpts-1)])
            ylpf = np.array([lrr.interpLinND([llamb[i]],[xlp],ylp) \
                for i in range(0,Nnpts)])
            RLflux = lrr.integrate_trapezia(ylpf,dllamb)
            RLflux = RLflux * contents[4][3]*phc.Lsun.cgs/4./np.pi/\
                            (dist_std*phc.pc.cgs)**2.
            RLfluxauxi.append(RLflux)
        
        BLfluxauxi = np.array(BLfluxauxi)    
        BL.append(BLfluxauxi)
        RLfluxauxi = np.array(RLfluxauxi)    
        RL.append(RLfluxauxi)        
        

    return Bralpha,Pfgamma,Brgamma,Humphreys,Brackets,BL,RL,Lbandspectrum


### 
Bralpha=[]
Pfgamma=[]
Brgamma=[]
Humphreys=[]
Brackets=[]
BL=[]
RL=[]
Lbandspectrum=[]
Halphaspectrum=[]
### Comment the next command lines if you want to skip this part.
### (But remember that this part is necessary in the output writing part.)
Bralpha,Pfgamma,Brgamma,Humphreys,Brackets,BL,RL,Lbandspectrum = \
        line_observables(\
        Bralpha,Pfgamma,Brgamma,Humphreys,Brackets,BL,RL,Lbandspectrum,\
        Halphaspectrum,files_fullsed_new, fullsed_contents)



#############################
### Obtaining magnitudes and alpha_WISEs


filters=[   'bess-u','bess-b','bess-v','bess-r','bess-i',\
            'bess-j','bess-h','bess-k',\
            'filt_Ha',\
            'wise1','wise2','wise3','wise4',\
            'irac_ch1','irac_ch2','irac_ch3','irac_ch4'
        ]

npts_interp = int(50.00) ### this number must be >= 3. A good choice is 50.

### defining the vector of alphas
alphamin = -6.
alphamax = 2.
N_pts = 80
alphavec = np.array([alphamin+(alphamax-alphamin)*float(i)/float(N_pts)\
    for i in range(0,N_pts+1)])
   
### obtaining the 'zp's for the evaluation of colors as a function 
### of alphas
zp1 = lrr.obtain_pogson_zp('spct1',"wise1")
zp2 = lrr.obtain_pogson_zp('spct1',"wise2")
zp3 = lrr.obtain_pogson_zp('spct1',"wise3")
zp4 = lrr.obtain_pogson_zp('spct1',"wise4")
    
### obtaining colors as functions of alphas
colorW1W2, err_facW1W2 = lrr.color_from_alpha(alphavec,"wise1","wise2",zp1,zp2)
colorW2W3, err_facW2W3 = lrr.color_from_alpha(alphavec,"wise2","wise3",zp2,zp3)
colorW3W4, err_facW3W4 = lrr.color_from_alpha(alphavec,"wise3","wise4",zp3,zp4)

iwise1 = filters.index("wise1")
iwise2 = filters.index("wise2")
iwise3 = filters.index("wise3")
iwise4 = filters.index("wise4")
    
print("Obtaining zero point constants...")
zp=[]
for j in range(0,len(filters)):
    zp.append(lrr.obtain_pogson_zp('spct1',filters[j],\
                npts=npts_interp))
print("")



def obtaining_mags_and_alphas(all_photflux,all_Mag,all_alphaWISE,\
        files_fullsed_new, files_source_new):

    for ifile in range(0,len(files_fullsed_new)):
        
        print("Obtaining magnitudes for file")
        print(files_fullsed_new[ifile][0])
        
        fullsedtest=files_fullsed_new[ifile][1]
        sourcetest=files_source_new[ifile]


        photflux_vec=[]
        Qphotflux_vec=[]; poldeg_vec=[]
        for j in range(0,len(filters)):
            print("Obtaining photon fluxes for filter "+str(filters[j]))
            mu,lamb,flambda,photflux=lrr.fullsed2photonflux(fullsedtest,\
                sourcetest,filters[j],npts=npts_interp,dist=10.,forcedred=5)
            photflux_vec.append(photflux)
            
            print("Obtaining Q photon fluxes for filter "+str(filters[j]))
            mu,lamb,flambda,qphotflux=lrr.fullsed2photonflux(fullsedtest,\
                sourcetest,filters[j],npts=npts_interp,dist=10.,forcedred=5,\
                Q=True)
            Qphotflux_vec.append(qphotflux)
            
            poldeg_auxi = []
            for i in range(0,len(photflux)):
                if photflux[i] != 0. and qphotflux[i] is not None \
                        and photflux[i] is not None:
                    poldeg_auxi.append(abs(qphotflux[i])/photflux[i])
                else:
                    poldeg_auxi.append(np.nan)
            poldeg_auxi = np.array(poldeg_auxi)
            poldeg_vec.append(poldeg_auxi)
        
        all_photflux.append(photflux_vec)
        all_Qphotflux.append(Qphotflux_vec)
        all_poldeg.append(poldeg_vec)
        

        Mag_vec=[]
        for j in range(0,len(filters)):
            if not None in photflux_vec[j]:
                Mag_vec.append(lrr.pogson(photflux_vec[j],zp[j]))
            else:
                Mag_vec.append(np.array([np.nan for el in photflux_vec[j]]))
        all_Mag.append(Mag_vec)
        
        alphaWISE_vec=[]
        
        alphaW1W2now = []
        for ii in range(0,len(mu)):
            if ~np.isnan(all_Mag[-1][iwise1][ii]-all_Mag[-1][iwise2][ii]):
                alphaW1W2now.append(\
                        lrr.interpLinND([all_Mag[-1][iwise1][ii]-\
                        all_Mag[-1][iwise2][ii]],\
                        [colorW1W2],alphavec,allow_extrapolation="no")
                        )
            else:
                alphaW1W2now.append(np.nan)
        
        
        alphaW2W3now = []
        for ii in range(0,len(mu)):
            if ~np.isnan(all_Mag[-1][iwise2][ii]-all_Mag[-1][iwise3][ii]):
                alphaW2W3now.append(\
                        lrr.interpLinND([all_Mag[-1][iwise2][ii]-\
                        all_Mag[-1][iwise3][ii]],\
                        [colorW2W3],alphavec,allow_extrapolation="no") \
                        )
            else:
                alphaW2W3now.append(np.nan)
                
                
        alphaW3W4now = []
        for ii in range(0,len(mu)):
            if ~np.isnan(all_Mag[-1][iwise3][ii]-all_Mag[-1][iwise4][ii]):
                alphaW3W4now.append(\
                        lrr.interpLinND([all_Mag[-1][iwise3][ii]-\
                        all_Mag[-1][iwise4][ii]],\
                        [colorW3W4],alphavec,allow_extrapolation="no") \
                        )
            else:
                alphaW3W4now.append(np.nan)
                
        
        alphaWISE_vec.append(alphaW1W2now)
        alphaWISE_vec.append(alphaW2W3now)
        alphaWISE_vec.append(alphaW3W4now)
        ###
        all_alphaWISE.append(alphaWISE_vec)

        print("")

    return all_photflux, all_Qphotflux, all_Mag, all_poldeg, all_alphaWISE


### 
all_photflux=[]
all_Qphotflux=[]
all_Mag=[]
all_poldeg=[]
all_alphaWISE=[]
### Comment the next command lines if you want to skip this part.
### (But remember that this part is necessary in the output writing part.)
all_photflux,all_Qphotflux, all_Mag, all_poldeg, all_alphaWISE = \
        obtaining_mags_and_alphas(\
        all_photflux,all_Mag,all_alphaWISE,\
        files_fullsed_new, files_source_new)


#############################
### Obtaining vsini

def obtaining_vsini(all_vsini, files_fullsed_new, fullsed_contents):
    """
    In this procedure, the vsini is calculated.
    """
    
    for ifile in range(0,len(files_fullsed_new)):
        contents=fullsed_contents[ifile][1]
        stelpars=[elem for elem in contents[4]]
        rpolenow=stelpars[1]
        massnow=stelpars[0]
        Wnow=stelpars[2]
        lixo,omeganow,lixo,Wnow=rss.rocheparams(Wnow,"W")
        veqfile=rss.cte_veq(rpolenow,massnow,omeganow,1.0)
        
        auxiifile=[]
        for iobs in range(0,len(mu)):
            auxiifile.append(veqfile*(1.-mu[iobs]**2.)**0.5)
        all_vsini.append(np.array(auxiifile))

    return all_vsini

### Obtaining the array of cosi's (a necessary step for obtaining vsini):
contents=fullsed_contents[0][1]
mu = contents[1]
### 
all_vsini = []
### Comment the next command lines if you want to skip this part.
### (But remember that this part is necessary in the output writing part.)
all_vsini = obtaining_vsini(all_vsini, files_fullsed_new, fullsed_contents)



#############################
### Writing in the external file
print("Writing in the external file")


f0=open(outputfile,"w")

### Writing a few explanations in the output file:
f0.write("# This is the output file of observables_OldBeatlas.py. It contains lots of "+"\n")
f0.write("# observables calculated from the SEDs computed by HDUST. This output is "+"\n")
f0.write("# used to feed the main program \"analysis_BeAtlas.py\"." +"\n")
f0.write("# \n")
f0.write("# Explanations of the lines:"+"\n")
f0.write("# \n")
f0.write("# * MODEL: contains the values of n, Sigma [g/cm2], M [Msun] and oblateness, "+"\n")
f0.write("# and it marks the beginning of the subfolder of parameters for that model. "+"\n")
f0.write("# (The model is completely specified by these parameters.)"+"\n")
f0.write("# \n")
f0.write("# * SOURCE: contains the values of M [Msun], Rpole [Rsun], W, L [Lsun], beta, "+"\n")
f0.write("# associated with that disk model."+"\n")
f0.write("# * TEMP_R and TEMP_T: the first contains list R/Req for the plane of the disk; "+"\n")
f0.write("# the second, contains the temperature in the plane of the disk [K]."+"\n")
f0.write("# * Halpha_lbd: Contains the list of lambdas [microns] of HDUST models of Halpha."+"\n")
f0.write("# * LBAND_lbd: Contains the list of lambdas [microns] of HDUST models of the L-band."+"\n")
f0.write("# * COSI: contains the cosine of the inclination angle i, "+"\n")
f0.write("# and it marks the beginning of the subfolder of parameters for that model and "+"\n")
f0.write("# inclination."+"\n")
f0.write("# \n")
f0.write("# * Halpha_flx: Contains the flux densities [erg/s cm2 A] associated to the lambdas"+"\n")
f0.write("# given by Halpha_lbd."+"\n")
f0.write("# * Halpha_f_fc: Contains the flux/continuum associated to Halpha_lbd."+"\n")
f0.write("# * LBAND_flx: Contains the flux densities [erg/s cm2 A] associated to the lambdas"+"\n")
f0.write("# given by LBAND_lbd."+"\n")
f0.write("# * LBAND_f_fc: Contains the flux/continuum associated to LBAND_lbd."+"\n")
f0.write("# * VSINI: contains vsini [km/s] "+"\n")
f0.write("# * SNRATIOS: contains tryads of: left boundary [microns], right boundary [microns], "+"\n")
f0.write("# S/N ratio obtained (or tried) from the SEDs between the previous two boundaries."+"\n")
f0.write("# * UBVRI: Contains the absolute magnitudes U, B, V, R, I [mag]."+"\n")
f0.write("# * poldegUBVRI: Contains the polarization degrees in U, B, V, R, I."+"\n")
f0.write("# * JHK: Contains the absolute magnitudes J, H, K [mag]."+"\n")
f0.write("# * poldegJHK: Contains the polarization degrees in J, H, K."+"\n")
f0.write("# * HALPHA_SOAR: Contains the absolute magnitude from SOAR's Halpha filter [mag], "+"\n")
f0.write("# under the assumption that the magnitude of Vega is defined to be 9999 mag."+"\n")
f0.write("# * poldegHALPHA_SOAR: Contains the polarization degrees from SOAR's Halpha filter."+"\n")
f0.write("# * WISEfilters: Contains the absolute magnitudes W1, W2, W3, W4 [mag]."+"\n")
f0.write("# * poldegWISEfilters: Contains the polarization degrees W1, W2, W3, W4."+"\n")
f0.write("# * ALPHA_WISE: contains the spectral indexes associated with W1-W2, W2-W3, W3-W4."+"\n")
f0.write("# * IRAC_filters: Contains the absolute magnitudes in the four IRAC filters."+"\n")
f0.write("# * poldegIRAC_filters: Contains the polarization degrees in the four IRAC filters."+"\n")
f0.write("# * LINE_XXX: For the specific line XXX, contains: flux (distance = 10pc) [erg/s cm2], "+"\n")
f0.write("# EW [angstroms], peak separation [km/s], a type of gaussian FWHM [km/s], "+"\n")
f0.write("# area of the gaussian fit [km/s]."+"\n")
f0.write("# * BL_FLUX: Contains the (Mennickent's \"blue\") flux [erg/s cm2] between "+"\n")
f0.write("# lbd1 and lbd2, lbd1 [microns], lbd2 [microns]."+"\n")
f0.write("# * RL_FLUX: Contains the (Mennickent's \"red\") flux [erg/s cm2] between "+"\n")
f0.write("# lbd1 and lbd2, lbd1 [microns], lbd2 [microns]."+"\n")
f0.write("# \n")
f0.write("# \n")
f0.write("# \n")



for ifile in range(0,len(files_fullsed_new)):
    ### Printing model parameters:
    ### n, Sigma0 [g cm^-2], Stellar mass [Msun], oblateness
    f0.write("MODEL "+\
            str(files_fullsed_new[ifile][0][0])+" "+\
            str(files_fullsed_new[ifile][0][1])+" "+\
            str(files_fullsed_new[ifile][0][2])+" "+\
            str(files_fullsed_new[ifile][0][3])+"\n")
            
    contents=fullsed_contents[ifile][1]
    
    ### Printing source parameters:
    ### Stellar mass [Msun], Rpole [Rsun], W, Lum [Lsun], beta
    f0.write("    SOURCE "+\
            str(contents[4][0])+" "+\
            str(contents[4][1])+" "+\
            str(contents[4][2])+" "+\
            str(contents[4][3])+" "+\
            str(contents[4][4])+"\n")
    
    ### Printing the radial temperature in the plane of the disk:
    ### Radial distance [Rstar]
    f0.write("    TEMP_R ")
    for ii in range(0,len(contents[6][0,:])):
        f0.write(str(contents[6][0,ii])+" ")
    f0.write("\n")
    ### Temperature [K]
    f0.write("    TEMP_T ")
    for ii in range(0,len(contents[6][1,:])):
        f0.write(str(contents[6][1,ii])+" ")
    f0.write("\n")

    ### Halpha lambda [microns]
    f0.write("    Halpha_lbd ")
    for ii in range(0,len(Halphaspectrum[ifile][0][0])):
        f0.write(str(Halphaspectrum[ifile][0][0][ii])+" ")
    f0.write("\n")
    ### L-band lambda [microns]
    f0.write("    LBAND_lbd ")
    for ii in range(0,len(Lbandspectrum[ifile][0][0])):
        f0.write(str(Lbandspectrum[ifile][0][0][ii])+" ")
    f0.write("\n")
    
    ### Now, we enter into the loop over cosi:
    for incs in range(0,len(contents[1])):
        f0.write("    COSI "+\
            str(contents[1][incs])+"\n")

        ### Halpha flux density [cgs]
        f0.write("        Halpha_flx ")
        for ii in range(0,len(Halphaspectrum[ifile][1][incs])):
            f0.write(str(Halphaspectrum[ifile][1][incs][ii])+" ")
        f0.write("\n")
        ### Halpha flux/continuum
        f0.write("        Halpha_f_fc ")
        for ii in range(0,len(Halphaspectrum[ifile][2][incs])):
            f0.write(str(Halphaspectrum[ifile][2][incs][ii])+" ")
        f0.write("\n")

        ### L-band flux density [cgs]
        f0.write("        LBAND_flx ")
        for ii in range(0,len(Lbandspectrum[ifile][1][incs])):
            f0.write(str(Lbandspectrum[ifile][1][incs][ii])+" ")
        f0.write("\n")
        ### L-band flux/continuum
        f0.write("        LBAND_f_fc ")
        for ii in range(0,len(Lbandspectrum[ifile][2][incs])):
            f0.write(str(Lbandspectrum[ifile][2][incs][ii])+" ")
        f0.write("\n")

        ### Printing the vsini [km/s]
        f0.write("        VSINI "+\
            str(all_vsini[ifile][incs])+"\n")
        
        ### Printing the evaluated S/N ratio of the models
        ### It contains series of tryads: 
        ### left boundary [microns], right boundary [microns], S/N ratio
        elements = []
        for ii in range(0,len(SNratios[ifile][incs])):
            elements.append(SNratios[ifile][incs][ii][0])
            elements.append(SNratios[ifile][incs][ii][1])
            elements.append(SNratios[ifile][incs][ii][2])
        f0.write("        SNRATIOS ")
        for elem in elements:
            f0.write(str(elem)+" ")
        f0.write("\n")
        
        ### Printing the absolute UBVRI magnitudes [mag] (dist = 10 pc)
        f0.write("        UBVRI "+\
            str(all_Mag[ifile][0][incs])+" "+\
            str(all_Mag[ifile][1][incs])+" "+\
            str(all_Mag[ifile][2][incs])+" "+\
            str(all_Mag[ifile][3][incs])+" "+\
            str(all_Mag[ifile][4][incs])+"\n")
        ### Printing the polarization degrees in UBVRI
        f0.write("        poldegUBVRI "+\
            str(all_poldeg[ifile][0][incs])+" "+\
            str(all_poldeg[ifile][1][incs])+" "+\
            str(all_poldeg[ifile][2][incs])+" "+\
            str(all_poldeg[ifile][3][incs])+" "+\
            str(all_poldeg[ifile][4][incs])+"\n")
        ### Printing the absolute JHK magnitudes [mag] (dist = 10 pc)
        f0.write("        JHK "+\
            str(all_Mag[ifile][5][incs])+" "+\
            str(all_Mag[ifile][6][incs])+" "+\
            str(all_Mag[ifile][7][incs])+"\n")
        ### Printing the polarization degrees in JHK
        f0.write("        poldegJHK "+\
            str(all_poldeg[ifile][5][incs])+" "+\
            str(all_poldeg[ifile][6][incs])+" "+\
            str(all_poldeg[ifile][7][incs])+"\n")
        ### Printing the absolute Ha (SOAR) magnitude [mag] (dist = 10 pc)
        ### (See my notes on this non-standard filter.)
        f0.write("        HALPHA_SOAR "+\
            str(all_Mag[ifile][8][incs])+"\n")
        ### Printing the polarization degree in Ha (SOAR)
        f0.write("        poldegHALPHA_SOAR "+\
            str(all_Mag[ifile][8][incs])+"\n")
        ### Printing the absolute WISE magnitudes [mag] (dist = 10 pc)
        f0.write("        WISE_filters "+\
            str(all_Mag[ifile][9][incs])+" "+\
            str(all_Mag[ifile][10][incs])+" "+\
            str(all_Mag[ifile][11][incs])+" "+\
            str(all_Mag[ifile][12][incs])+"\n")
        ### Printing the polarization degrees in WISE
        f0.write("        poldegWISE_filters "+\
            str(all_poldeg[ifile][9][incs])+" "+\
            str(all_poldeg[ifile][10][incs])+" "+\
            str(all_poldeg[ifile][11][incs])+" "+\
            str(all_poldeg[ifile][12][incs])+"\n")
        ### Printing the alphaW1W2, alphaW2W3 and alphaW3W4
        f0.write("        ALPHA_WISE "+\
            str(all_alphaWISE[ifile][0][incs])+" "+\
            str(all_alphaWISE[ifile][1][incs])+" "+\
            str(all_alphaWISE[ifile][2][incs])+"\n")
        ### Printing the absolute IRAC magnitudes [mag] (dist = 10 pc)
        f0.write("        IRAC_filters "+\
            str(all_Mag[ifile][13][incs])+" "+\
            str(all_Mag[ifile][14][incs])+" "+\
            str(all_Mag[ifile][15][incs])+" "+\
            str(all_Mag[ifile][16][incs])+"\n")
        ### Printing the polarization degrees in IRAC
        f0.write("        poldegIRAC_filters "+\
            str(all_poldeg[ifile][13][incs])+" "+\
            str(all_poldeg[ifile][14][incs])+" "+\
            str(all_poldeg[ifile][15][incs])+" "+\
            str(all_poldeg[ifile][16][incs])+"\n")
        
        ### Printing the derived quantities for each line:
        ### line flux (dist = 10 pc) [erg/s cm^2]
        ### Equivalent Width [Angstroms]
        ### Peak separation [km/s]
        ### gaussian FWHM (over abs(F/Fc-1)) [km/s]
        ### area of the gaussian fit [km/s]
        f0.write("        LINE_HALPHA "+\
            str(Brackets[ifile][0][0][incs])+" "+\
            str(Brackets[ifile][0][1][incs])+" "+\
            str(Brackets[ifile][0][2][incs])+" "+\
            str(Brackets[ifile][0][3][incs][0])+" "+\
            str(Brackets[ifile][0][3][incs][1])+"\n")
        f0.write("        LINE_HBETA "+\
            str(Brackets[ifile][1][0][incs])+" "+\
            str(Brackets[ifile][1][1][incs])+" "+\
            str(Brackets[ifile][1][2][incs])+" "+\
            str(Brackets[ifile][1][3][incs][0])+" "+\
            str(Brackets[ifile][1][3][incs][1])+"\n")
        f0.write("        LINE_HGAMMA "+\
            str(Brackets[ifile][2][0][incs])+" "+\
            str(Brackets[ifile][2][1][incs])+" "+\
            str(Brackets[ifile][2][2][incs])+" "+\
            str(Brackets[ifile][2][3][incs][0])+" "+\
            str(Brackets[ifile][2][3][incs][1])+"\n")

        f0.write("        LINE_BRGAMMA "+\
            str(Brgamma[ifile][0][incs])+" "+\
            str(Brgamma[ifile][1][incs])+" "+\
            str(Brgamma[ifile][2][incs])+" "+\
            str(Brgamma[ifile][3][incs][0])+" "+\
            str(Brgamma[ifile][3][incs][1])+"\n")

        f0.write("        LINE_BRALPHA "+\
            str(Bralpha[ifile][0][incs])+" "+\
            str(Bralpha[ifile][1][incs])+" "+\
            str(Bralpha[ifile][2][incs])+" "+\
            str(Bralpha[ifile][3][incs][0])+" "+\
            str(Bralpha[ifile][3][incs][1])+"\n")
        f0.write("        LINE_PFGAMMA "+\
            str(Pfgamma[ifile][0][incs])+" "+\
            str(Pfgamma[ifile][1][incs])+" "+\
            str(Pfgamma[ifile][2][incs])+" "+\
            str(Pfgamma[ifile][3][incs][0])+" "+\
            str(Pfgamma[ifile][3][incs][1])+"\n")
        f0.write("        LINE_HUMPHREY14 "+\
            str(Humphreys[ifile][0][0][incs])+" "+\
            str(Humphreys[ifile][0][1][incs])+" "+\
            str(Humphreys[ifile][0][2][incs])+" "+\
            str(Humphreys[ifile][0][3][incs][0])+" "+\
            str(Humphreys[ifile][0][3][incs][1])+"\n")
        f0.write("        LINE_HUMPHREY15 "+\
            str(Humphreys[ifile][1][0][incs])+" "+\
            str(Humphreys[ifile][1][1][incs])+" "+\
            str(Humphreys[ifile][1][2][incs])+" "+\
            str(Humphreys[ifile][1][3][incs][0])+" "+\
            str(Humphreys[ifile][1][3][incs][1])+"\n")
        f0.write("        LINE_HUMPHREY16 "+\
            str(Humphreys[ifile][2][0][incs])+" "+\
            str(Humphreys[ifile][2][1][incs])+" "+\
            str(Humphreys[ifile][2][2][incs])+" "+\
            str(Humphreys[ifile][2][3][incs][0])+" "+\
            str(Humphreys[ifile][2][3][incs][1])+"\n")
        f0.write("        LINE_HUMPHREY18 "+\
            str(Humphreys[ifile][3][0][incs])+" "+\
            str(Humphreys[ifile][3][1][incs])+" "+\
            str(Humphreys[ifile][3][2][incs])+" "+\
            str(Humphreys[ifile][3][3][incs][0])+" "+\
            str(Humphreys[ifile][3][3][incs][1])+"\n")
        f0.write("        LINE_HUMPHREY19 "+\
            str(Humphreys[ifile][4][0][incs])+" "+\
            str(Humphreys[ifile][4][1][incs])+" "+\
            str(Humphreys[ifile][4][2][incs])+" "+\
            str(Humphreys[ifile][4][3][incs][0])+" "+\
            str(Humphreys[ifile][4][3][incs][1])+"\n")
        f0.write("        LINE_HUMPHREY20 "+\
            str(Humphreys[ifile][5][0][incs])+" "+\
            str(Humphreys[ifile][5][1][incs])+" "+\
            str(Humphreys[ifile][5][2][incs])+" "+\
            str(Humphreys[ifile][5][3][incs][0])+" "+\
            str(Humphreys[ifile][5][3][incs][1])+"\n")
        f0.write("        LINE_HUMPHREY21 "+\
            str(Humphreys[ifile][6][0][incs])+" "+\
            str(Humphreys[ifile][6][1][incs])+" "+\
            str(Humphreys[ifile][6][2][incs])+" "+\
            str(Humphreys[ifile][6][3][incs][0])+" "+\
            str(Humphreys[ifile][6][3][incs][1])+"\n")
        f0.write("        LINE_HUMPHREY22 "+\
            str(Humphreys[ifile][7][0][incs])+" "+\
            str(Humphreys[ifile][7][1][incs])+" "+\
            str(Humphreys[ifile][7][2][incs])+" "+\
            str(Humphreys[ifile][7][3][incs][0])+" "+\
            str(Humphreys[ifile][7][3][incs][1])+"\n")
        f0.write("        LINE_HUMPHREY23 "+\
            str(Humphreys[ifile][8][0][incs])+" "+\
            str(Humphreys[ifile][8][1][incs])+" "+\
            str(Humphreys[ifile][8][2][incs])+" "+\
            str(Humphreys[ifile][8][3][incs][0])+" "+\
            str(Humphreys[ifile][8][3][incs][1])+"\n")
        f0.write("        LINE_HUMPHREY24 "+\
            str(Humphreys[ifile][9][0][incs])+" "+\
            str(Humphreys[ifile][9][1][incs])+" "+\
            str(Humphreys[ifile][9][2][incs])+" "+\
            str(Humphreys[ifile][9][3][incs][0])+" "+\
            str(Humphreys[ifile][9][3][incs][1])+"\n")
        f0.write("        LINE_HUMPHREY25 "+\
            str(Humphreys[ifile][10][0][incs])+" "+\
            str(Humphreys[ifile][10][1][incs])+" "+\
            str(Humphreys[ifile][10][2][incs])+" "+\
            str(Humphreys[ifile][10][3][incs][0])+" "+\
            str(Humphreys[ifile][10][3][incs][1])+"\n")
        ### Printing the Mennickent's fluxes [erg s^-1 cm^-2]
        ### and boundary limits used [microns]
        f0.write("        BL_FLUX "+\
            str(BL[ifile][incs])+" "+\
            str(lamb1_BL)+" "+\
            str(lamb2_BL)+"\n")
        f0.write("        RL_FLUX "+\
            str(RL[ifile][incs])+" "+\
            str(lamb1_RL)+" "+\
            str(lamb2_RL)+"\n")
        
        f0.write("    END_COSI \n")
    f0.write("END_MODEL \n")

f0.close()





        
        
        
        
