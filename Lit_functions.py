'''
Series of Functions from the literature to be plotted and compared to data.
Designed to be read in with SED_plots_v2.py plotting class
'''

import numpy as np
import matplotlib.pyplot as plt


def Durras_Lbol(L,typ,err=False):
    '''X-ray to bolometric correction from Durras et al. 2020'''
    if typ == 'Lx':
        a, b, c = 15.33, 11.48, 16.20
        alo, blo, clo = 15.33-0.06, 11.48-0.01, 16.20-0.16
        aup, bup, cup = 15.33+0.06, 11.48+0.01, 16.20+0.16
        std = 0.37
    elif typ == 'Lbol':    
        a, b, c = 10.96, 11.93, 17.79
        alo, blo, clo = 10.96-0.06, 11.93-0.01, 17.79-0.10
        aup, bup, cup = 10.96+0.06, 11.93+0.01, 17.79+0.10
        std = 0.27
    else:
        print('Specify typ. Options are:    Lx    Lbol')
        return

    # L += np.log10(0.611)

    kx = a*(1+((L - np.log10(3.8E33))/b)**c)
    kx *= 1/1.64
    kx_lo = alo*(1+((L - np.log10(3.8E33))/blo)**clo)
    kx_up = aup*(1+((L - np.log10(3.8E33))/bup)**cup)
    kx_lo = kx - std
    kx_up = kx + std
    if err:
        return kx, kx_up, kx_lo
    else:
        return kx


def Stern_MIR(L6):
    '''X-ray to MIR correlation from Stern 2015. Returns 2-10keV X-ray luminosity for a give 6μm luminosity.'''

    a, b, c = 40.981, 1.024, 0.047
    # L6 /= 1E41
    L6 -= 41
    Lx = a + b*L6 - c*L6**2

    Lx += np.log10(1.64)

    return Lx

def Stern_Lx(Lx):
    a, b, c = 40.981, 1.024, 0.047

    

def Hopkins_Lbol(L,band='Lx'):
    '''X-ray to bolometric correction from Hopkins et al. 2007'''
    if band == 'Lx':
        c1 = 10.83 
        k1 = 0.28
        c2 = 6.08
        k2 = -0.020

    kx = c1*((10**L)/(10E10*3.8E33))**k1+c2*((10**L)/(10E10*3.8E33))**k2
    kx *= 1/1.64
    return kx

def Just_alpha_ox(Luv):
    a = -0.140
    b = 2.705
    c = -0.093
    d = 0.899
    w = 2500*1E-8  # observed wavelength from Angstroms to cm
    nu_uv = 3E10/w  # convert obs wavelength to a frequ

    wx = 6.2*1E-8
    nu_x = 3E10/wx

    # Luv =- np.log10(nu)
    Lnu = Luv - np.log10(nu_uv)

    # Lx = (1/c)*(a*Lnu+b-d)
    Lx = 0.709*Lnu+4.822
    Lx_out = Lx

    Lx_out = Lx + np.log10(nu_x)
    Lx_out += 0.67

    return Lx_out


def Ranalli(L_FIR):
    Lhx = L_FIR - 3.62

    return Lhx


def Torres(L_FIR):
    Lhx = -0.17*L_FIR**2 + 15.2*L_FIR - 169

    return Lhx


def Yang_FIR_Lx(L_FIR):
    Lsun = 3.8E33

    SFR = 1.09E-10*10**(L_FIR - np.log10(Lsun))
    log_SFR = np.log10(SFR)

    BHAR = 0.92*log_SFR-2.47

    Lbol = (1E46*10**BHAR)/1.58

    Lx = Durras_Lbol(np.log10(Lbol), typ='Lbol')

    print('Lx',Lx+np.log10(Lsun))

    return Lx+np.log10(Lsun)


# def Hao_gal_frac(fg):
    
#     nuLnu_gal = 
#     nuLnu_agn = 
#     nuLnu_mix = fg*nuLnu_gal+(1-fg)*nuLnu_agn
    


