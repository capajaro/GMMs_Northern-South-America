# -*- coding: utf-8 -*-
"""
GMM Arteta et al., 2023
Python Code V1.0. 22FEB2023.
coding by: César Pájaro, please report bugs or issues to this email:
    capajaro@uninorte.edu.co, cesar.pajaromiranda@canterbury@ac.nz
Implements GMM developed by Carlos Arteta et al (2023) and published as 
Arteta, C. A., C. A. Pajaro, V. Mercado, J. Montejo, M. Arcila,
and N. A. Abrahamson (2023). Ground-Motion Model (GMM) for Crustal
Earthquakes in Northern South America (NoSAm Crustal GMM), Bull. Seismol. Soc.
Am. 113, 186–203, doi: 10.1785/0120220168

    Inputs:
        T = Period of interest [float]
        Mag: Magnitude (Mw) [float]
        R: Rupture distance [float]
        Cat: The site soil category according to Table 1, Page 190. [integer]
        Amp_HVRSR: amplitude of the peak of the mean HVRSR if unknown use 'Average' 
                see Table 1, Page 8 [float/string]
        HypoD: Hypocentral Distance [float]
        Rvolc: The horizontal portion of the ray path crossing the volcanic regions [float]
    
    Output: 
        List containing the following values:
        mean: The mean value of the GMM
        tau: Inter-event standard deviation
        phi: Intra-event standard deviation
        sigma: total standard deviation
        SigmaSS: Single-station standard deviation
        
User Guidance
The NoSAm crustal model estimates the horizontal-component
RotD50, 5%-damped, spectral acceleration of crustal earthquakes
in NoSAm for spectral periods ≤10 s. The range of magnitudes
for applying the NoSAm crustal GMM is 4:5 ≤ Mw ≤ 8:0. The
rupture distance range is 5 ≤ Rrup ≤ 350 km. This model is
only intended for applications in NoSAm. The global GMMs
should be considered for other regions without region-specific
models.
The input parameters required are:
1. the moment magnitude Mw;
2. the rupture distance to the site Rrup (km);
3. site category based on the predominant period according to
Table 1;
4. the hypocentral depth of the earthquake Zhypo (km);
and
5. the horizontal portion of the ray path crossing the volcanic
regions, Rvolc (km). The median Rvolc = 31 km may be
used when estimates for ray paths crossing the arc are
unavailable.

"""
import numpy as np
from scipy import interpolate
import pandas as pd

def NoSAm_Crustal_2023(T, Mag, R, Cat, Amp_HVRSR, HypoD, Rvolc):

    def _get_Coef(T):
        
        Coef = """\
        Period	  q1	 q2	      q3	  q4	   q5	       q6	  q7      s1	  s2	  s3	  s4	  s5	 M1	    tau	    phi	    Sss
        0.01	-0.090	-0.1	 0.000	-0.790	-0.00352	-0.0055	 0.0083	  0  	0.337	0.692	0.679	0.609	6.75	0.43	0.76	0.67
        0.02	-0.032	-0.1	 0.000	-0.790	-0.00352	-0.0053	 0.0083	  0	    0.337	0.683	0.672	0.609	6.75	0.40	0.78	0.66
        0.03	 0.038	-0.1	 0.000	-0.790	-0.00363	-0.0052	 0.0083	  0	    0.337	0.672	0.658	0.578	6.75	0.41	0.79	0.67
        0.05	 0.273	-0.1	 0.000	-0.790	-0.00401	-0.0051	 0.0083	  0	    0.337	0.643	0.580	0.505	6.75	0.44	0.80	0.71
        0.075	 0.604	-0.1	 0.000	-0.790	-0.00452	-0.0050	 0.0083	  0	    0.337	0.617	0.500	0.418	6.75	0.46	0.86	0.77
        0.1	     0.773	-0.1	 0.000	-0.790	-0.00468	-0.0050	 0.0083	  0	    0.363	0.649	0.477	0.366	6.75	0.49	0.86	0.80
        0.15	 0.830	-0.1	 0.000	-0.790	-0.00458	-0.0049	 0.0083	  0	    0.551	0.750	0.546	0.379	6.75	0.55	0.84	0.83
        0.2	     0.772	-0.1	 0.000	-0.790	-0.00429	-0.0048	 0.0083	  0	    0.527	0.832	0.620	0.457	6.75	0.55	0.81	0.78
        0.25	 0.744	-0.1	-0.002	-0.790	-0.00392	-0.0048	 0.0083	  0	    0.345	0.857	0.680	0.518	6.75	0.53	0.78	0.74
        0.3	     0.698	-0.1	-0.005	-0.790	-0.00365	-0.0047	 0.0083	  0	    0.186	0.830	0.769	0.582	6.75	0.53	0.76	0.71
        0.4	     0.626	-0.1	-0.020	-0.790	-0.00302	-0.0047	 0.0083	  0	    0.021	0.728	0.913	0.741	6.75	0.50	0.76	0.68
        0.5	     0.570	-0.1	-0.045	-0.790	-0.00248	-0.0046	 0.0083	  0	   -0.040	0.529	1.000	0.849	6.75	0.49	0.75	0.67
        0.75	 0.468	-0.1	-0.078	-0.790	-0.00172	-0.0046	 0.0062	  0	   -0.178	0.281	0.953	1.087	6.75	0.45	0.69	0.62
        1	     0.395	-0.1	-0.106	-0.790	-0.00141	-0.0044	 0.0048	  0	   -0.261	0.156	0.690	1.279	6.75	0.43	0.68	0.61
        1.5	     0.177	-0.1	-0.147	-0.790	-0.00117	-0.0039	 0.0027	  0	   -0.320	0.113	0.488	1.065	6.75	0.39	0.70	0.58
        2	    -0.082	-0.1	-0.168	-0.790	-0.00106	-0.0031	 0.0012	  0	   -0.318	0.071	0.350	0.849	6.75	0.37	0.69	0.57
        3	    -0.577	-0.1	-0.185	-0.790	-0.00096	-0.0012	-0.0008	  0	   -0.248	0.029	0.264	0.705	6.82	0.37	0.62	0.53
        4	    -0.878	-0.1	-0.197	-0.790	-0.00096	-0.0004	-0.0023	  0	   -0.212	0.028	0.225	0.642	6.92	0.36	0.57	0.48
        5	    -1.214	-0.1	-0.207	-0.765	-0.00096	-0.0001	-0.0034	  0	   -0.210	0.028	0.203	0.597	7.00	0.36	0.57	0.48
        6	    -1.647	-0.1	-0.215	-0.711	-0.00096	 0.0000	-0.0043	  0	   -0.210	0.028	0.203	0.597	7.06	0.35	0.56	0.47
        7.5	    -2.255	-0.1	-0.224	-0.634	-0.00096	 0.0000	-0.0055	  0	   -0.210	0.028	0.203	0.597	7.15	0.35	0.54	0.45
        10	    -3.042	-0.1	-0.236	-0.529	-0.00096	 0.0000	-0.0069	  0	   -0.210	0.028	0.203	0.597	7.25	0.35	0.59	0.43

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
        f_q7 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['q7'])
        f_M1 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['M1'])

        # Site Coefficients
        f_s1 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['s1'])
        f_s2 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['s2'])
        f_s3 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['s3'])
        f_s4 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['s4'])
        f_s5 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['s5'])

        # Standard Deviations
        f_t = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['tau'])
        f_f = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['phi'])
        f_Sss = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['Sss'])

        C = {"q1": f_q1(np.log10(T)).tolist(), "q2": f_q2(np.log10(T)).tolist(), "q3": f_q3(np.log10(T)).tolist(),
            "q4": f_q4(np.log10(T)).tolist(), "q5": f_q5(np.log10(T)).tolist(), "q6": f_q6(np.log10(T)).tolist(),
            "q7": f_q7(np.log10(T)).tolist(),"M1": f_M1(np.log10(T)).tolist(), 
            "s1": f_s1(np.log10(T)).tolist(), "s2": f_s2(np.log10(T)).tolist(), "s3": f_s3(np.log10(T)).tolist(),
            "s4": f_s4(np.log10(T)).tolist(), "s5": f_s5(np.log10(T)).tolist(),
            "tau": f_t(np.log10(T)).tolist(), "phi": f_f(np.log10(T)).tolist(), 
            "Sss": f_Sss(np.log10(T)).tolist()}

        return C

    def _get_global_magnitude_scaling_term(C, Mag):
            """
            Returns the global magnitude term Eq. 5
            """
            if Mag <= C['M1']:
              return C["q2"] * (Mag - C['M1'])
            else:
              return 0

    def _get_regional_magnitude_scaling_term(C, Mag):
            """
            Returns the global magnitude term Eq. 6
            """
            
            return C["q3"] * (8.5 - Mag)**2
            
    def _get_global_distance_scaling_term(C,Mag, R):
        """
        Returns the global distance scaling term Eq.7
        """
        return (C["q4"] + 0.275*(Mag-C['M1'])) * np.log(np.sqrt(R**2 + 4.5**2))

    def _get_regional_distance_scaling_term(C, R):
        """
        Returns the regional distance scaling term Eq.8
        """
        return (C["q5"] * R)
    
    def _get_regional_volcanic_term(C, Rvolc):
        """
        Returns the regional distance scaling term Eq.8
        """
        return (C["q6"] * Rvolc)

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

    def _get_FZhypo_term(C, HypoD):
        """
        Returns the hypocentral depth scaling term 
        """
        return (C["q7"] * HypoD)

    def _get_mean(C, Mag, R, Cat, Amp_HVRSR, HypoD):
        """
        Returns the mean ground motion
        """
        
        return (C['q1'] + _get_global_magnitude_scaling_term(C, Mag) +
                _get_regional_magnitude_scaling_term(C, Mag) +
                _get_global_distance_scaling_term(C, Mag, R) +
                _get_regional_distance_scaling_term(C, R) +
                _get_site_scaling_term(C, Cat, Amp_HVRSR)+
                _get_regional_volcanic_term(C, Rvolc) +
                _get_FZhypo_term(C, HypoD))

    def _get_stddevs(C):
        """
        Return standard deviations.
        """

        tau = C['tau']
        phi = C['phi'] 
          
        Sigma = np.sqrt(phi**2+tau**2)
        SigmaSS = C['Sss']

        return tau, phi, Sigma, SigmaSS
    
    C = _get_Coef(T)
    
    def get_mean_and_stddevs(C, Mag, R, Cat, Amp_HVRSR, HypoD):
            
            
            mean = np.exp(_get_mean(C, Mag, R, Cat, Amp_HVRSR, HypoD))
            
            [tau, phi, Sigma, SigmaSS] = _get_stddevs(C)
            
            Ndec = 3
            mean = np.round(mean, Ndec+5)
            tau = np.round(tau, Ndec)
            phi = np.round(phi, Ndec)
            Sigma = np.round(Sigma, Ndec)
            SigmaSS = np.round(SigmaSS, Ndec)
            
            return mean, tau, phi, Sigma, SigmaSS

    return get_mean_and_stddevs(C, Mag, R, Cat, Amp_HVRSR, HypoD)
