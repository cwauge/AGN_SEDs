'''
Series of Functions from the literature to be plotted and compared to data.
Designed to be read in with SED_plots_v2.py plotting class
'''

import numpy as np
import matplotlib.pyplot as plt


def Durras_Lbol(L,typ):
    '''X-ray to bolometric correction from Durras et al. 2020'''
    if typ == 'Lx':
        a, b, c = 15.33, 11.48, 15.20
    elif typ == 'Lbol':    
        a, b, c = 10.96, 11.93, 17.79
    else:
        print('Specify typ. Options are:    Lx    Lbol')
        return

    kx = a*(1+((L - np.log10(3.8E33))/b)**c)
    return kx


def Stern_MIR(L6):
    '''X-ray to MIR correlation from Stern 2015. Returns 2-10keV X-ray luminosity for a give 6μm luminosity.'''

    a, b, c = 40.981, 1.024, 0.047
    # L6 /= 1E41
    L6 -= 41
    Lx = a + b*L6 - c*L6**2

    Lx += np.log10(1.64)

    return Lx


