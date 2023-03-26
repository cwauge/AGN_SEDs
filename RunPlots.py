import numpy as np
import matplotlib.pyplot as plt 
from astropy.io import fits 
from SED_plots_v2 import Plotter
from SED_shape_plots import SED_shape_Plotter
from plots_Letter import Plotter_Letter
from plots_Letter2 import Plotter_Letter2


path = '/Users/connor_auge/Research/Disertation/catalogs/'
with fits.open(path+'AGN_SEDs_out.fits') as hdul:
    cols = hdul[1].columns
    data = hdul[1].data 

id = data['ID']
z = data['z']
x = data['x']
y = data['y']
Lx = data['Lx']
norm = data['norm']
FIR_upper_lims = data['FIR_upper_lims']
frac_err = data['frac_err']

F025 = data['F025']
F1 = data['F1']
F6 = data['F6']
F100 = data['F100']

shape = data['shape']
Lbol = data['Lbol_sub']

int_x = data['int_x']
int_y = data['int_y']
wfir = data['wfir']
ffir = data['ffir']




