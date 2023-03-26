import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def solar(x):
    return x - np.log10(3.8E33)

def ergs(x):
    return x + np.log10(3.8E33)

def Durras(l,typ):
    if typ == 'Lx':
        a = 15.33
        b = 11.48
        c = 15.20
        # kx = a*(1+(np.log10(lx)/b)**c)
    elif typ == 'Lbol':
        a = 10.96
        b = 11.93
        c = 17.79
    kx = a*(1+((l - np.log10(3.8E33))/b)**c)
    return kx



with fits.open('/Users/connor_auge/Desktop/GOALS_CIGALE_out/goals_out/resutls.fits') as hdul:
    cols = hdul[1].columns
    data = hdul[1].data 

# chi = data['']