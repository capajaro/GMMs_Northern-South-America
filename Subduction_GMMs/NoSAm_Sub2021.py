# -*- coding: utf-8 -*-
"""
GMM Arteta et al., 2021
Python Code V1.0. 22JUL2021.
coding by: César Pájaro, please report bugs or issues to this email:
    capajaro@uninorte.edu.co, cesar.pajaromiranda@canterbury@ac.nz
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
from scipy import interpolate
import pandas as pd

def NoSAm_Sub2021(T, Tec_Environment, Mag, R, Cat, Amp_HVRSR, FBA):

    def _get_Coef(Tec_Environment, T):
        if Tec_Environment == 'Interface' or Tec_Environment == 0:
            Coef = """\
            Period	    q1	  q2	  q3	    q4	     q5	    q6	   s1	    s2	  s3	    s4	    s5	    C1	  tau	  phi1	phi2	  Sss
            0.010	   4.329	0.730	-0.021	-1.450	-0.006	0.000	  0.000	0.740	  0.966	  0.959	  0.986	  8.200	0.452	0.726	0.817	0.690
            0.020	   4.347	0.730	-0.021	-1.450	-0.006	0.000	  0.000	0.828	  0.990	  0.994	  0.971	  8.200	0.448	0.739	0.832	0.688
            0.030	   4.360	0.730	-0.021	-1.450	-0.006	0.000	  0.000	0.896	  0.964	  0.989	  0.959	  8.200	0.439	0.777	0.841	0.723
            0.050	   4.473	0.730	-0.021	-1.450	-0.006	0.000	  0.000	1.011	  0.941	  0.955	  0.885	  8.200	0.439	0.798	0.854	0.792
            0.075	   4.679	0.730	-0.021	-1.450	-0.007	0.000	  0.000	1.196	  0.948	  0.899	  0.783	  8.200	0.446	0.836	0.892	0.808
            0.100	   4.893	0.730	-0.021	-1.450	-0.008	0.000	  0.000	1.237	  1.022	  0.850	  0.692	  8.200	0.444	0.753	0.965	0.771
            0.150	   5.070	0.730	-0.023	-1.425	-0.008	0.000	  0.000	1.180	  1.182	  0.772	  0.582	  8.200	0.459	0.708	0.982	0.796
            0.200	   4.950	0.730	-0.025	-1.335	-0.008	0.000	  0.000	1.016	  1.210	  0.706	  0.514	  8.200	0.519	0.730	0.962	0.830
            0.250	   4.900	0.730	-0.029	-1.275	-0.008	0.000	  0.000	0.850	  1.221	  0.693	  0.509	  8.200	0.559	0.830	0.916	0.871
            0.300	   4.850	0.730	-0.038	-1.231	-0.008	0.000	  0.000	0.650	  1.119	  0.732	  0.531	  8.200	0.535	0.806	0.944	0.826
            0.400	   4.650	0.730	-0.057	-1.165	-0.008	0.000	  0.000	0.227	  0.628	  0.934	  0.601	  8.200	0.508	0.704	0.850	0.780
            0.500	   4.334	0.730	-0.072	-1.115	-0.007	0.000	  0.000	0.094	  0.434	  0.949	  0.687	  8.200	0.509	0.715	0.867	0.775
            0.750	   3.564	0.730	-0.099	-1.020	-0.007	0.000	  0.000	-0.047	0.233	  0.784	  0.865	  8.150	0.536	0.627	0.821	0.732
            1.000	   2.957	0.730	-0.118	-0.950	-0.006	0.000	  0.000	-0.146	0.143	  0.632	  0.881	  8.100	0.598	0.658	0.822	0.745
            1.500	   1.986	0.730	-0.145	-0.860	-0.006	0.000	  0.000	-0.287	0.026	  0.384	  0.692	  8.050	0.631	0.675	0.804	0.749
            2.000	   1.323	0.730	-0.164	-0.820	-0.005	0.000	  0.000	-0.386	-0.011	0.225	  0.386	  8.000	0.592	0.716	0.807	0.761
            3.000	   0.518	0.730	-0.191	-0.793	-0.005	0.000	  0.000	-0.438	-0.041	0.090	  0.130	  7.900	0.570	0.696	0.759	0.787
            4.000	  -0.022	0.730	-0.210	-0.793	-0.004	0.000	  0.000	-0.438	-0.041	0.042	  0.052	  7.850	0.525	0.748	0.731	0.789
            5.000	  -0.437	0.730	-0.220	-0.793	-0.003	0.000	  0.000	-0.438	-0.041	0.022	  0.012	  7.800	0.494	0.747	0.775	0.710
            6.000	  -0.784	0.730	-0.224	-0.793	-0.003	0.000	  0.000	-0.438	-0.041	0.022	  0.012	  7.800	0.455	0.791	0.807	0.662
            7.500	  -1.281	0.730	-0.224	-0.793	-0.002	0.000	  0.000	-0.438	-0.041	0.022	  0.012	  7.800	0.440	0.746	0.831	0.625
            10.000	-1.883	0.730	-0.224	-0.793	-0.001	0.000	  0.000	-0.438	-0.041	0.022	  0.012	  7.800	0.468	0.724	0.791	0.609
            """
        else:
            Coef = """\
            Period	q1	    q2	    q3	    q4	    q5	    q6	    s1	    s2	    s3	    s4	    s5	    C1	    tau	  phi1	phi2	Sss
            0.010	  4.639	  1.070	  -0.027	-1.450	-0.005	-0.653	0.000	  0.745	  0.892	  0.933	  0.886	  6.500	  0.364	0.707	0.834	0.676
            0.020	  4.714	  1.070	  -0.027	-1.450	-0.005	-0.653	0.000	  0.723	  0.879	  0.932	  0.862	  6.500	  0.356	0.699	0.848	0.678
            0.030	  4.752	  1.070	  -0.027	-1.450	-0.005	-0.653	0.000	  0.725	  0.863	  0.936	  0.810	  6.500	  0.359	0.704	0.856	0.684
            0.050	  4.951	  1.070	  -0.027	-1.450	-0.005	-0.653	0.000	  0.752	  0.806	  0.889	  0.747	  6.500	  0.340	0.725	0.873	0.678
            0.075	  5.126	  1.070	  -0.027	-1.420	-0.006	-0.717	0.000	  0.823	  0.795	  0.825	  0.653	  6.500	  0.318	0.791	0.902	0.703
            0.100	  5.153	  1.070	  -0.027	-1.364	-0.006	-0.807	0.000	  0.929	  0.820	  0.785	  0.561	  6.500	  0.308	0.855	0.977	0.770
            0.150	  4.975	  1.070	  -0.027	-1.298	-0.006	-0.862	0.000	  0.953	  0.917	  0.785	  0.505	  6.500	  0.351	0.853	0.995	0.793
            0.200	  4.650	  1.070	  -0.027	-1.258	-0.006	-0.857	0.000	  0.849	  1.018	  0.950	  0.553	  6.500	  0.358	0.793	0.983	0.757
            0.250	  4.300	  1.070	  -0.027	-1.227	-0.005	-0.824	0.000	  0.691	  1.101	  1.066	  0.659	  6.500	  0.345	0.782	0.942	0.715
            0.300	  4.000	  1.070	  -0.027	-1.201	-0.004	-0.766	0.000	  0.556	  1.113	  1.153	  0.744	  6.500	  0.358	0.731	0.972	0.729
            0.400	  3.500	  1.070	  -0.030	-1.161	-0.003	-0.628	0.000	  0.499	  1.003	  1.281	  0.904	  6.500	  0.382	0.600	0.889	0.652
            0.500	  3.118	  1.070	  -0.037	-1.130	-0.003	-0.521	0.000	  0.442	  0.829	  1.283	  1.082	  6.500	  0.416	0.656	0.913	0.698
            0.750	  2.400	  1.070	  -0.056	-1.074	-0.003	-0.329	0.000	  0.391	  0.535	  1.146	  1.403	  6.500	  0.452	0.748	0.880	0.695
            1.000	  1.821	  1.070	  -0.072	-1.000	-0.003	-0.192	0.000	  0.355	  0.409	  0.980	  1.350	  6.500	  0.466	0.694	0.850	0.679
            1.500	  0.953	  1.070	  -0.085	-0.958	-0.002	-0.089	0.000	  0.304	  0.317	  0.756	  1.092	  6.500	  0.453	0.581	0.839	0.680
            2.000	  0.340	  1.070	  -0.095	-0.938	-0.002	-0.036	0.000	  0.278	  0.282	  0.666	  0.909	  6.500	  0.422	0.552	0.841	0.664
            3.000	  -0.458	1.070	  -0.104	-0.933	-0.002	-0.018	0.000	  0.267	  0.257	  0.597	  0.751	  6.500	  0.398	0.573	0.809	0.616
            4.000	  -1.033	1.070	  -0.107	-0.933	-0.002	-0.018	0.000	  0.267	  0.247	  0.577	  0.724	  6.500	  0.405	0.549	0.757	0.578
            5.000	  -1.468	1.070	  -0.109	-0.933	-0.002	-0.018	0.000	  0.267	  0.247	  0.567	  0.724	  6.500	  0.415	0.578	0.781	0.621
            6.000	  -1.825	1.070	  -0.109	-0.933	-0.002	-0.018	0.000	  0.267	  0.247	  0.567	  0.724	  6.500	  0.427	0.627	0.813	0.658
            7.500	  -2.265	1.070	  -0.109	-0.933	-0.002	-0.018	0.000	  0.267	  0.247	  0.567	  0.724	  6.500	  0.446	0.680	0.825	0.684
            10.000	-2.755	1.070	  -0.109	-0.933	-0.002	-0.018	0.000	  0.267	  0.247	  0.567	  0.724	  6.500	  0.486	0.691	0.761	0.730
            """
            

        Coef = str.split(Coef)
        sw = 0
        i = 0
        while sw == 0:
            try:
                float(Coef[i])
                sw = 1
                Begining = i
                COEFFS = np.zeros([int(len(Coef)/(i))-1, i])

            except ValueError:
                i = i+1
                n_col_header = i

        k = 0
        for i in range(int(len(Coef)/(Begining))-1):
            for j in range(Begining):
                COEFFS[i, j] = float(Coef[Begining+k])
                k = k+1

        COEFFS = pd.DataFrame(data = COEFFS, columns = Coef[0:n_col_header])
        
        # Base Rock Coefficients
        f_q1 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['q1'])
        f_q2 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['q2'])
        f_q3 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['q3'])
        f_q4 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['q4'])
        f_q5 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['q5'])
        f_q6 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['q6'])
        f_C1 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['C1'])

        # Site Coefficients
        f_s1 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['s1'])
        f_s2 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['s2'])
        f_s3 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['s3'])
        f_s4 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['s4'])
        f_s5 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['s5'])

        # Standard Deviations
        f_t = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['tau'])
        f_f1 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['phi1'])
        f_f2 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['phi2'])
        f_Sss = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['Sss'])

        C = {"q1": f_q1(np.log10(T)).tolist(), "q2": f_q2(np.log10(T)).tolist(), "q3": f_q3(np.log10(T)).tolist(),
            "q4": f_q4(np.log10(T)).tolist(), "q5": f_q5(np.log10(T)).tolist(), "q6": f_q6(np.log10(T)).tolist(),
            "C1": f_C1(np.log10(T)).tolist(), 
            "s1": f_s1(np.log10(T)).tolist(), "s2": f_s2(np.log10(T)).tolist(), "s3": f_s3(np.log10(T)).tolist(),
            "s4": f_s4(np.log10(T)).tolist(), "s5": f_s5(np.log10(T)).tolist(),
            "tau": f_t(np.log10(T)).tolist(), "phi1": f_f1(np.log10(T)).tolist(), "phi2": f_f2(np.log10(T)).tolist(),
            "Sss": f_Sss(np.log10(T)).tolist()}

        return C

    def _get_global_magnitude_scaling_term(C, Mag):
            """
            Returns the global magnitude term Eq. 5
            """
            if Mag<= C['C1']:
              return C["q2"] * (Mag - C['C1'])
            else:
              return 0

    def _get_regional_magnitude_scaling_term(C, Mag):
            """
            Returns the global magnitude term Eq. 6
            """
            
            return C["q3"] * (10 - Mag)**2
            
    def _get_global_distance_scaling_term(C,Mag, R):
        """
        Returns the global distance scaling term Eq.7
        """
        return (C["q4"] + 0.1*(Mag-7)) * np.log(R + 10*np.exp(0.4*(Mag - 6)))

    def _get_regional_distance_scaling_term(C, R):
        """
        Returns the regional distance scaling term Eq.8
        """
        return (C["q5"] * R)

    def _get_site_scaling_term(C, Cat, Amp_HVRSR):
        """
        Returns the site scaling term Eq.9
        """
        if Cat == 1:
            return 0
        else:
            if isinstance('Average', str):
                AVG_P_star = {'s2': 3.29, 's3': 4.48, 's4': 4.24, 's5': 3.47}
                
                return (C["s%0.0f"%(Cat)] * np.log(AVG_P_star["s%0.0f"%(Cat)]))
            else:
                return (C["s%0.0f"%(Cat)] * np.log(Amp_HVRSR))

    def _get_FBA_term(C, FBA):
        """
        Returns the regional distance scaling term 
        """
        return (C["q6"] * FBA)

    def _get_mean(C, Mag, R, Cat, Amp_HVRSR, FBA):
        """
        Returns the mean ground motion
        """
        return (C['q1'] + _get_global_magnitude_scaling_term(C, Mag) +
                _get_regional_magnitude_scaling_term(C, Mag) +
                _get_global_distance_scaling_term(C, Mag, R) +
                _get_regional_distance_scaling_term(C, R) +
                _get_site_scaling_term(C, Cat, Amp_HVRSR)+
                _get_FBA_term(C, FBA))

    def _get_stddevs(C,R):
        """
        Return standard deviations.
        """

        tau = C['tau']
        if R <= 150:
          phi = C['phi1']
        elif R> 200:
          phi = C['phi2']
        else:
          phi = C['phi1'] + (C['phi2'] - C['phi1'])*(R - 150)*(1/50)
          
        Sigma = np.sqrt(phi**2+tau**2)
        SigmaSS = C['Sss']

        return tau, phi, Sigma, SigmaSS
    
    C = _get_Coef(Tec_Environment, T)
    
    def get_mean_and_stddevs(C, Mag, R, Cat, Amp_HVRSR, FBA, T, Tec_Environment):
            
            
            mean = np.exp(_get_mean(C, Mag, R, Cat, Amp_HVRSR, FBA))
            
            [tau, phi, Sigma, SigmaSS] = _get_stddevs(C,R)
            
            Ndec = 3
            mean = np.round(mean, Ndec+5)
            tau = np.round(tau, Ndec)
            phi = np.round(phi, Ndec)
            Sigma = np.round(Sigma, Ndec)
            SigmaSS = np.round(SigmaSS, Ndec)
            
            return mean, tau, phi, Sigma, SigmaSS

    return get_mean_and_stddevs(C, Mag, R, Cat, Amp_HVRSR, FBA, T, Tec_Environment)
