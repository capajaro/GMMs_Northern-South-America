# -*- coding: utf-8 -*-
"""
GMM Arteta et al., 2021
Python Code V1.0. 22JUL2021.
coding by: César Pájaro, please report bugs or issues to this email:
    capajaro@uninorte.edu.co
Implements GMM developed by Carlos Arteta et al (2021) and published as 
Arteta, C. A., Pajaro, C. A., Mercado, V., Montejo, J., 
Arcila, M., & Abrahamson, N. A. (2021). Ground-motion model for subduction 
earthquakes in northern South America. 
Earthquake Spectra, https://doi.org/10.1177/87552930211027585.

    Inputs:
        T = Period of interest [float]
        Tec_Environment: 'Interface'(0) of 'Intra-slab'(1) [string/integer/boolean]
        Mag: Magnitude (Mw) [float]
        R: Rupture distance for Interface, Hypocentral distance for Intra-slab [float]
        Cat: The site soil category according to Table 1, Page 8. [integer]
        Amp_HVRSR: amplitude of the peak of the mean HVRSR if unknown use 'Average' 
                see Table 1, Page 8 [float/string]
        FBA: 0:forearc / 1:backarc classification see Page 13 [Integer]
    
    Output: 
        List containing the following values:
        mean: The mean value of the GMM
        tau: Inter-event standard deviation
        phi: Intra-event standard deviation
        sigma: total standard deviation
        SigmaSS: Single-station standard deviation
        
User Guidance
The NSAm SUB GMM models the horizontal-component RotD50, 5% damped, spectral
acceleration of interface and intra-slab subduction earthquakes for spectral periods up to
10 s. The input parameters required to use the NSAm SUB GMM are (1) the moment
magnitude, Mw; (2) the rupture (for interface) or hypocentral (for intra-slab) distance to
the site, Rrup or Rhypo (km); (3) a site category based on natural period according to Table
1; and (4) a fore/back arc flag for intra-slab events, for example, FFABA = 0 for forearc
sites and FFABA = 1 for backarc sites. ‘‘Category s2 can be used to represent sites which
have been typically characterized as ‘‘generic rock’’ in previous models for Colombia; this
category includes sites with a Vs of around 760 m/s. Average values of P* found in Table
1 may be used to characterize such ‘‘generic rock’’ stations in case no HVRSR information
is available’’. The range of magnitudes for the application of the NSAm SUB GMM is
4.5 < Mw < 9.5, for interface earthquakes, and 4.5 < Mw 8.0 for intra-slab events. The
large-magnitude extrapolation is feasible thanks to the constraints imposed by the global
model. The distance range is 10 < Rrup < 450 km for interface earthquakes and
70 < Rhypo < 450 km for intra-slab. This model is only intended for applications in
Colombia and Ecuador. For other regions, without region-specific models, the global
GMMs should be considered.
    
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from NoSAm_Sub2021 import NoSAm_Sub2021


# Name for the .csv with the results and figures
outfile_name = 'out'

# Selected site condition, Tec_Environment, FBA and Periods
Tec_Environment = 'Intra-slab'
Cat = 3
Amp_HVRSR = 'Average'
FBA =0
Periods=[0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2,0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7.5, 10]; 

# For plot 1: Selected distance for differente magnitudes
R = 100
Magnitudes = [5,6,7,8]   # Magnitudes to be tested
Col_names = ['M%0.2f'%(el) for el in Magnitudes]
Col_names = ['Periods'] + Col_names
Mag_GMM_Results = pd.DataFrame(data = np.zeros([len(Periods), len(Magnitudes)+1]), 
                               columns = Col_names)
Mag_GMM_Results['Periods'] = Periods

# For plot 2: Selected magnitude for differente distances
Mag = 6.5
Rs = [25,50,100,300]   # distances to be tested
Col_names = ['R%0.0f'%(el) for el in Rs]
Col_names = ['Periods'] + Col_names
Dist_GMM_Results = pd.DataFrame(data = np.zeros([len(Periods), len(Rs)+1]), 
                               columns = Col_names)
Dist_GMM_Results['Periods'] = Periods

#%% Gráfico Magnitudes y Distancias
Figure_title_size = 11
colors = ['orangered', 'darkslategrey', 'lawngreen', 'deepskyblue',
          'darkred', 'darkgoldenrod', 'orchid']
AVG_P_star = {'s2': 3.29, 's3': 4.48, 's4': 4.24, 's5': 3.47}
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(15/2.54,9/2.54))
for i_Mag in range(len(Magnitudes)):
    PSA = np.zeros_like(Periods)
    for i_Per in range(len(Periods)):
        T = Periods[i_Per]
        GMM_Results = NoSAm_Sub2021(T, Tec_Environment,Magnitudes[i_Mag], R, Cat, Amp_HVRSR, FBA)
        PSA[i_Per] = GMM_Results[0]    
    
    Mag_GMM_Results.iloc[:,i_Mag+1] = PSA
    axs[0].loglog(Periods, PSA, linewidth=1.5, color=colors[i_Mag],
                     label='Mw: %0.1f' % (Magnitudes[i_Mag]))
    axs[0].set_title('R: %0.0f [km]'%(R), fontdict={
        'fontsize': 9, 'fontname': 'serif'})
    axs[0].set_xlabel('Period [s]', fontdict={
        'fontsize': 9, 'fontname': 'serif'})
    axs[0].set_ylabel('Sa[g]',
               fontdict={'fontsize': 9, 'fontname': 'serif'})
    axs[0].legend(loc='lower left', ncol=1, prop={
        'size': 9, 'family': 'serif'})
    plt.setp(axs[0].get_xticklabels(), size=8)
    plt.setp(axs[0].get_yticklabels(), size=8)
    if Cat == 1:
           fig.suptitle('%s - Cat: %0.0f; FBA:%0.0f' % (Tec_Environment,  Cat, FBA),
                        size = Figure_title_size, weight = 'bold', fontdict={'fontname': 'serif'})
    else:
        if isinstance('Average', str):
            AVG_P_star = {'s2': 3.29, 's3': 4.48, 's4': 4.24, 's5': 3.47}
            
            fig.suptitle('%s - Cat: %0.0f; P*:%0.2f; FBA:%0.0f' % (Tec_Environment,  Cat, AVG_P_star["s%0.0f"%(Cat)], FBA),
                         size = Figure_title_size, weight = 'bold', fontdict={'fontname': 'serif'})
        else:
            fig.suptitle('%s - Cat: %0.0f; P*:%0.2f; FBA:%0.0f' % (Tec_Environment,  Cat, Amp_HVRSR, FBA), 
                         size = Figure_title_size, weight = 'bold', fontdict={'fontname': 'serif'})
    
    axs[0].grid(True, which='minor')
    axs[0].set_xlim(0.01, 10)
    axs[0].set_ylim(1e-5, 2)


for i_Rs in range(len(Rs)):
    PSA = np.zeros_like(Periods)
    for i_Per in range(len(Periods)):
        T = Periods[i_Per]
        GMM_Results = NoSAm_Sub2021(T, Tec_Environment,Mag, Rs[i_Rs], Cat, Amp_HVRSR, FBA)
        PSA[i_Per] = GMM_Results[0]    
    
    Dist_GMM_Results.iloc[:,i_Rs+1] = PSA
    axs[1].loglog(Periods, PSA, linewidth=1.5, color=colors[i_Rs],
                     label='R: %0.0f' % (Rs[i_Rs]))
    
    axs[1].set_title('M: %0.2f [Mw]'%(Mag), fontdict={
        'fontsize': 9, 'fontname': 'serif'})
    axs[1].set_xlabel('Period [s]', fontdict={
        'fontsize': 9, 'fontname': 'serif'})
    axs[1].legend(loc='lower left', ncol=1, prop={
        'size': 9, 'family': 'serif'})
    plt.setp(axs[1].get_xticklabels(), size=8)
    plt.setp(axs[1].get_yticklabels(), size=8)
    axs[1].grid(True, which='minor')
    axs[1].set_xlim(0.01, 10)
    axs[1].set_ylim(1e-5, 2)
fig.subplots_adjust(wspace=0.13)
fig.savefig(outfile_name+'.tiff', dpi=600, facecolor='w', edgecolor='k',
        orientation='portrait',  format='tiff',
        transparent=False, bbox_inches=None, pad_inches=0.1,)
fig.savefig(outfile_name+'.png', dpi=600, facecolor='w', edgecolor='k',
        orientation='portrait',  format='png',
        transparent=False, bbox_inches=None, pad_inches=0.1,)
Dist_GMM_Results.to_csv(outfile_name+'_Distance.csv', index = False)
Mag_GMM_Results.to_csv(outfile_name+'_Magnitudes.csv', index = False)

