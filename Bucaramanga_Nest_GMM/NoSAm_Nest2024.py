# -*- coding: utf-8 -*-
"""
GMM Pajaro et al., 2024
Python Code V1.0. 05MAR2024.
coding by: César Pájaro, please report bugs or issues to this email:
    capajaro@uninorte.edu.co, cesar.pajaromiranda@canterbury@ac.nz
Implements GMM developed by Cesar Pajaro et al (2024) and published as 
Pajaro, C.A., Arteta, C.A., Mercado, V. et al. Partially non-ergodic ground motion model 
for the Bucaramanga seismic nest in Northern South America (NoSAm Nest GMM). 
Bull Earthquake Eng (2024). https://doi.org/10.1007/s10518-024-01898-w

    Inputs:
        T = Period of interest [float]
        Mag: Magnitude (Mw) [float]
        R: Hypocentral distance [float]
        Cat: The site soil category according to Table 1, Page 190. [integer]
        Amp_HVRSR: amplitude of the peak of the mean HVRSR if unknown use 'Average' 
                see Table 1, Page 8 [float/string]
        
    
    Output: 
        List containing the following values:
        mean: The mean value of the GMM
        tau: Inter-event standard deviation
        phi: Intra-event standard deviation
        sigma: total standard deviation
        SigmaSS: Single-station standard deviation
        
User Guidance
The NoSAm Nest GMM model estimates the horizontal-component RotD50, 5% damped,
spectral acceleration of in-slab earthquakes associated with Bucaramanga nest in northern
South America for spectral periods up to 10 s. The range of magnitudes for applying the
NoSAm Nest GMM is 4.5 ≤ Mw ≤ 7.0. The hypocentral distance range is 100 ≤ Rhypo ≤ 450
km. This model is only intended for applications in northern South America. Hence should
be used with caution in other regions. 
The input parameters required are:
1. The moment magnitude, Mw.
2. The hypocentral distance to the site, Rhypo (km).
3. Site category based on the predominant period according to Table 3.

"""
import numpy as np
from scipy import interpolate
import pandas as pd
import io

def NoSAm_Nest(T, Mag, R, Cat, Amp_HVRSR):

    def _get_Coef(T):
        Coef = """
        Period,  q1,      q2,     q3,        q4,       q5,         s1,   s2,       s3,      s4,       s5,      C1,  tau,   phi,    phiS2S,  phiSS,  Sss
        0.010,   5.951,   1.07,   -0.0392,   -1.400,   -0.00271,   0,    0.603,    0.500,   0.440,    0.201,   7,   0.4,   0.63,   0.37,    0.52,   0.65
        0.020,   5.983,   1.07,   -0.0392,   -1.400,   -0.00271,   0,    0.603,    0.500,   0.435,    0.240,   7,   0.4,   0.62,   0.36,    0.51,   0.65
        0.030,   6.097,   1.07,   -0.0392,   -1.400,   -0.00292,   0,    0.615,    0.500,   0.425,    0.240,   7,   0.4,   0.65,   0.36,    0.53,   0.67
        0.050,   6.370,   1.07,   -0.0392,   -1.400,   -0.00332,   0,    0.692,    0.500,   0.399,    0.201,   7,   0.4,   0.72,   0.36,    0.62,   0.74
        0.075,   6.490,   1.07,   -0.0392,   -1.370,   -0.00419,   0,    0.800,    0.505,   0.378,    0.120,   7,   0.4,   0.76,   0.38,    0.66,   0.77
        0.100,   6.494,   1.07,   -0.0392,   -1.314,   -0.00432,   0,    0.830,    0.530,   0.334,    0.076,   7,   0.4,   0.71,   0.36,    0.61,   0.73
        0.150,   6.494,   1.07,   -0.0450,   -1.248,   -0.00415,   0,    0.800,    0.600,   0.203,   -0.016,   7,   0.4,   0.68,   0.37,    0.57,   0.70
        0.200,   6.494,   1.07,   -0.0541,   -1.208,   -0.00373,   0,    0.600,    0.620,   0.140,   -0.058,   7,   0.4,   0.70,   0.41,    0.57,   0.69
        0.250,   6.494,   1.07,   -0.0656,   -1.177,   -0.00310,   0,    0.257,    0.580,   0.120,   -0.052,   7,   0.4,   0.70,   0.45,    0.54,   0.67
        0.300,   6.494,   1.07,   -0.0762,   -1.151,   -0.00272,   0,    0.004,    0.499,   0.133,    0.000,   7,   0.4,   0.66,   0.45,    0.48,   0.62
        0.400,   6.494,   1.07,   -0.0943,   -1.111,   -0.00240,   0,   -0.237,    0.283,   0.336,    0.240,   7,   0.4,   0.63,   0.44,    0.45,   0.61
        0.500,   6.464,   1.07,   -0.1136,   -1.070,   -0.00220,   0,   -0.293,    0.200,   0.539,    0.347,   7,   0.4,   0.64,   0.45,    0.45,   0.60
        0.750,   6.324,   1.07,   -0.1501,   -1.004,   -0.00205,   0,   -0.316,    0.097,   0.559,    0.678,   7,   0.4,   0.63,   0.46,    0.43,   0.59
        1.000,   6.038,   1.07,   -0.1700,   -0.950,   -0.00200,   0,   -0.326,    0.047,   0.418,    0.750,   7,   0.4,   0.68,   0.50,    0.46,   0.61
        1.500,   5.318,   1.07,   -0.175,    -0.908,   -0.00200,   0,   -0.336,   -0.011,   0.360,    0.650,   7,   0.4,   0.62,   0.43,    0.45,   0.60
        2.000,   4.643,   1.07,   -0.175,    -0.888,   -0.00200,   0,   -0.336,   -0.021,   0.350,    0.591,   7,   0.4,   0.58,   0.42,    0.40,   0.56
        3.000,   3.833,   1.07,   -0.175,    -0.883,   -0.00200,   0,   -0.336,   -0.031,   0.340,    0.570,   7,   0.4,   0.58,   0.42,    0.40,   0.55
        4.000,   3.258,   1.07,   -0.175,    -0.883,   -0.00200,   0,   -0.336,   -0.031,   0.340,    0.570,   7,   0.4,   0.59,   0.43,    0.40,   0.54
        5.000,   2.812,   1.07,   -0.175,    -0.883,   -0.00200,   0,   -0.336,   -0.031,   0.340,    0.570,   7,   0.4,   0.58,   0.42,    0.41,   0.55
        6.000,   2.448,   1.07,   -0.175,    -0.883,   -0.00200,   0,   -0.336,   -0.031,   0.340,    0.570,   7,   0.4,   0.58,   0.42,    0.40,   0.55
        7.500,   2.002,   1.07,   -0.175,    -0.883,   -0.00200,   0,   -0.336,   -0.031,   0.340,    0.570,   7,   0.4,   0.59,   0.43,    0.41,   0.57
       10.000,   1.427,   1.07,   -0.175,    -0.883,   -0.00200,   0,   -0.336,   -0.031,   0.340,    0.570,   7,   0.4,   0.63,   0.48,    0.41,   0.58
       """
       
        data = io.StringIO(Coef.replace(' ',''))
        COEFFS = pd.read_csv(data,  sep=',')
          
        # Base Rock Coefficients
        f_q1 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['q1'])
        f_q2 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['q2'])
        f_q3 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['q3'])
        f_q4 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['q4'])
        f_q5 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['q5'])
        f_C1 = interpolate.interp1d(np.log10(COEFFS['Period']), COEFFS['C1'])

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
            "q4": f_q4(np.log10(T)).tolist(), "q5": f_q5(np.log10(T)).tolist(),
            "C1": f_C1(np.log10(T)).tolist(), 
            "s1": f_s1(np.log10(T)).tolist(), "s2": f_s2(np.log10(T)).tolist(), "s3": f_s3(np.log10(T)).tolist(),
            "s4": f_s4(np.log10(T)).tolist(), "s5": f_s5(np.log10(T)).tolist(),
            "tau": f_t(np.log10(T)).tolist(), "phi": f_f(np.log10(T)).tolist(), "Sss": f_Sss(np.log10(T)).tolist()}

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

    def _get_mean(C, Mag, R, Cat, Amp_HVRSR):
        """
        Returns the mean ground motion
        """
        return (C['q1'] + _get_global_magnitude_scaling_term(C, Mag) +
                _get_regional_magnitude_scaling_term(C, Mag) +
                _get_global_distance_scaling_term(C, Mag, R) +
                _get_regional_distance_scaling_term(C, R) +
                _get_site_scaling_term(C, Cat, Amp_HVRSR))

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
    
    def get_mean_and_stddevs(C, Mag, R, Cat, Amp_HVRSR, T):
            
            
            mean = np.exp(_get_mean(C, Mag, R, Cat, Amp_HVRSR))
            
            [tau, phi, Sigma, SigmaSS] = _get_stddevs(C)
            
            Ndec = 3
            mean = np.round(mean, Ndec+5)
            tau = np.round(tau, Ndec)
            phi = np.round(phi, Ndec)
            Sigma = np.round(Sigma, Ndec)
            SigmaSS = np.round(SigmaSS, Ndec)
            
            return mean, tau, phi, Sigma, SigmaSS

    return get_mean_and_stddevs(C, Mag, R, Cat, Amp_HVRSR, T)

