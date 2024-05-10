import time
import numpy as np
import matplotlib.pyplot as plt
import os
import Lit_functions
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io import ascii
from astropy.cosmology import FlatLambdaCDM
from scipy.constants import c
from SED_v8 import AGN
from SED_plots_v2 import Plotter
from filters import Filters
from plots_Letter import Plotter_Letter
from plots_Letter2 import Plotter_Letter2
from SED_shape_plots import SED_shape_Plotter
from match import match
from mag_flux import mag_to_flux
from mag_flux import magerr_to_fluxerr

# def Lum_to_Flux(L,z):
#     '''Function to convert flux to luminosity'''
#     cosmo = FlatLambdaCDM(H0=70, Om0=0.29, Tcmb0=2.725)

#     dl = cosmo.luminosity_distance(z).value # Distance in Mpc
#     dl_cgs = dl*(3.0856E24) # Distance from Mpc to cm
#     surf = 4*np.pi*dl_cgs**2
#     k_corr_SED = 1e-26 * surf * c / (2.0664e-4 * 1e-6)

#     # convert flux to luminosity 
#     F = L / k_corr_SED

#     return F


def Lum_to_Flux(L,z):
    '''Function to convert flux to luminosity'''
    cosmo = FlatLambdaCDM(H0=70, Om0=0.29, Tcmb0=2.725)

    dl = cosmo.luminosity_distance(z).value # Distance in Mpc
    dl_cgs = dl*(3.0856E22) # Distance from Mpc to m
    surf = 4*np.pi*dl_cgs**2
    k_corr_SED = 1e-29 * surf * c / (2.0664e-4 * 1e-6)

    # convert flux to luminosity 
    L = L/1E7
    F = L / k_corr_SED

    return F

def Flux_to_Lum(F,z):
    '''Function to convert flux to luminosity'''
    cosmo = FlatLambdaCDM(H0=70, Om0=0.29, Tcmb0=2.725)

    dl = cosmo.luminosity_distance(z).value # Distance in Mpc
    dl_cgs = dl*(3.0856E22) # Distance from Mpc to m
    surf = 4*np.pi*dl_cgs**2
    k_corr_SED = 1e-29 * surf * c / (2.0664e-4 * 1e-6)

    # convert flux to luminosity 
    L = F*k_corr_SED
    L = L*1E7

    return L

ti = time.perf_counter() # Start timer
path = '/Users/connor_auge/Research/Disertation/catalogs/' # Path for photometry catalogs

Lx_min = 42.5

'''
Read in data files for GOALS sources
Data comes from several different files for both the X-ray and photometry data:
    Photometry
        The catalog from Vivian U (U2012_GOALS2.fits)
        Gather individual from NED by myself (NED_GOALS.fits)
    X-ray 
        The C-GOALS II catalog - combination of Table 1 and A.1. (CGOALS_Xray.txt) - contains hard and soft X-ray flux - potentially dont need
        Selection of sources from the Ricci 2021 paper (ULIRG_Xray.csv) - contains observed and intrinsic Lx values and flux values
'''

# Read in the Photometry data data
with fits.open('../catalogs/U2012_GOALS2.fits') as hdul:
    U_phot_data = hdul[1].data 

with fits.open('../catalogs/GOALS_total.fits') as hdul:
    Ned_phot_data = hdul[1].data 

U_phot_ID = U_phot_data['ID']
Ned_phot_ID = Ned_phot_data['ID']

# Read in the X-ray data
cgoals = ascii.read('../catalogs/CGOALS_Xray.txt',guess=False,delimiter=',',encoding='utf-8')
ricci = ascii.read('../catalogs/ULIRG_Xray3.csv',guess=False,delimiter=',')

cgoals_ID = cgoals['ID']
cgoals_z = cgoals['z']
cgoals_Lx = cgoals['Lhx']*1E40
cgoals_Lir = cgoals['LIR']
cgoals_Fhx = cgoals['Fhx']*1E-14
cgoals_Fsx = cgoals['Fsx']*1E-14

ricci_ID = np.asarray(ricci['ID'])
ricci_z = np.asarray(ricci['z'])
ricci_Lx_hard = np.asarray(ricci['Lx2_10'])
ricci_Lx_full = np.asarray(ricci['Lx2_10']) + np.log10(1.64)
ricci_Lx_hard_obs = np.asarray(ricci['Lx2_10_obs'])
ricci_Fhx = np.asarray(ricci['Fx2_10'])
ricci_Fsx = np.asarray(ricci['Fx05_2'])
ricci_Nh = np.asarray(ricci['Nh'])
ricci_LIR = np.asarray(ricci['LIR'])

ricci_Lx_har_flt = np.asarray([10**i for i in ricci_Lx_hard])
ricci_Lx_har_obs_flt = np.asarray([10**i for i in ricci_Lx_hard_obs])
correction = ricci_Lx_har_flt/ricci_Lx_har_obs_flt 

Fxh_int_mjy = Lum_to_Flux(ricci_Lx_har_flt, ricci_z)


goals_cond = ricci_Lx_full > Lx_min
ricci_ID = ricci_ID[goals_cond]
ricci_z = ricci_z[goals_cond]
ricci_Lx_hard = ricci_Lx_hard[goals_cond]
ricci_Lx_full = ricci_Lx_full[goals_cond]
ricci_Lx_hard_obs = ricci_Lx_hard_obs[goals_cond]
ricci_Fhx = ricci_Fhx[goals_cond]
ricci_Fsx = ricci_Fsx[goals_cond]
Fxh_int_mjy = Fxh_int_mjy[goals_cond]
ricci_Nh = ricci_Nh[goals_cond]
ricci_LIR = ricci_LIR[goals_cond]
correction = correction[goals_cond]

ix, iy = match(ricci_ID,U_phot_ID)
ricci_ID_match_U = ricci_ID[ix]
ricci_z_match_U = ricci_z[ix]
ricci_Lx_hard_match_U = ricci_Lx_hard[ix]
ricci_Lx_full_match_U = ricci_Lx_full[ix]
ricci_Lx_hard_obs_match_U = ricci_Lx_hard_obs[ix]
ricci_Fhx_match_U = ricci_Fhx[ix]
ricci_Fsx_match_U = ricci_Fsx[ix]
ricci_Nh_match_U = ricci_Nh[ix]
ricci_LIR_match_U = ricci_LIR[ix]
Fxh_int_mjy_match_U = Fxh_int_mjy[ix]

U_phot_ID_match = U_phot_ID[iy]

ricci_Fx_hard_match_mjy_U = ricci_Fhx_match_U*4.136E8/(10-2)
ricci_Fx_soft_match_mjy_U = ricci_Fsx_match_U*4.136E8/(2-0.5)

goals_nan_array = np.zeros(np.shape(U_phot_data['HX'][iy]))
goals_nan_array[goals_nan_array == 0] = np.nan

U_goals_flux = np.asarray([
    ricci_Fx_hard_match_mjy_U*1000,
    ricci_Fx_soft_match_mjy_U*1000,
    # U_phot_data['HX'][iy]*1E6,
    # U_phot_data['SX'][iy]*1E6,
    goals_nan_array,
    U_phot_data['FUV'][iy]*1E6,
    U_phot_data['NUV'][iy]*1E6,
    U_phot_data['U'][iy]*1E6,
    U_phot_data['B'][iy]*1E6,
    U_phot_data['V'][iy]*1E6,
    U_phot_data['R'][iy]*1E6,
    U_phot_data['I'][iy]*1E6,
    U_phot_data['J'][iy]*1E6,
    U_phot_data['H'][iy]*1E6,
    U_phot_data['Ks'][iy]*1E6,
    U_phot_data['IRAC1'][iy]*1E6,
    U_phot_data['IRAC2'][iy]*1E6,
    U_phot_data['IRAC3'][iy]*1E6,
    U_phot_data['IRAC4'][iy]*1E6,
    U_phot_data['IRAS1'][iy]*1E6,
    U_phot_data['MIPS1'][iy]*1E6,
    U_phot_data['IRAS2'][iy]*1E6,
    U_phot_data['IRAS3'][iy]*1E6,
    U_phot_data['MIPS2'][iy]*1E6,
    U_phot_data['PACS2'][iy]*1E6,
    U_phot_data['MIPS3'][iy]*1E6,
    U_phot_data['F250'][iy]*1E6,
    U_phot_data['F350'][iy]*1E6,
    U_phot_data['F500'][iy]*1E6
    # U_phot_data['SCUBA2'][iy]*1E6,
    # U_phot_data['VLA1'][iy]*1E6,
    # U_phot_data['VLA2'][iy]*1E6
])

U_goals_flux_err = np.asarray([
    ricci_Fx_hard_match_mjy_U*1000*0.2,
    ricci_Fx_soft_match_mjy_U*1000*0.2,
    # U_phot_data['HXerr'][iy]*1E6,
    # U_phot_data['SXerr'][iy]*1E6,
    goals_nan_array,
    U_phot_data['FUVerr'][iy]*1E6,
    U_phot_data['NUVerr'][iy]*1E6,
    U_phot_data['Uerr'][iy]*1E6,
    U_phot_data['Berr'][iy]*1E6,
    U_phot_data['Verr'][iy]*1E6,
    U_phot_data['Rerr'][iy]*1E6,
    U_phot_data['Ierr'][iy]*1E6,
    U_phot_data['Jerr'][iy]*1E6,
    U_phot_data['Herr'][iy]*1E6,
    U_phot_data['Kserr'][iy]*1E6,
    U_phot_data['IRAC1err'][iy]*1E6,
    U_phot_data['IRAC2err'][iy]*1E6,
    U_phot_data['IRAC3err'][iy]*1E6,
    U_phot_data['IRAC4err'][iy]*1E6,
    U_phot_data['IRAS1err'][iy]*1E6,
    U_phot_data['MIPS1err'][iy]*1E6,
    U_phot_data['IRAS2err'][iy]*1E6,
    U_phot_data['IRAS3err'][iy]*1E6,
    U_phot_data['MIPS2err'][iy]*1E6,
    U_phot_data['PACS2err'][iy]*1E6,
    U_phot_data['MIPS3err'][iy]*1E6,
    U_phot_data['F250err'][iy]*1E6,
    U_phot_data['F350err'][iy]*1E6,
    U_phot_data['F500err'][iy]*1E6
    # U_phot_data['SCUBA2err'][iy]*1E6,
    # U_phot_data['VLA1err'][iy]*1E6,
    # U_phot_data['VLA2err'][iy]*1E6
])

U_goals_flux = U_goals_flux.T
U_goals_flux_err = U_goals_flux_err.T



ix, iy = match(ricci_ID, Ned_phot_ID)
ricci_ID_match_Ned = ricci_ID[ix]
ricci_z_match_Ned = ricci_z[ix]
ricci_Lx_hard_match_Ned = ricci_Lx_hard[ix]
ricci_Lx_full_match_Ned = ricci_Lx_full[ix]
ricci_Lx_hard_obs_match_Ned = ricci_Lx_hard_obs[ix]
ricci_Fhx_match_Ned = ricci_Fhx[ix]
ricci_Fsx_match_Ned = ricci_Fsx[ix]
ricci_Nh_match_Ned = ricci_Nh[ix]
ricci_LIR_match_Ned = ricci_LIR[ix]
correction_match_Ned = correction[ix]
Fxh_int_mjy_match_Ned = Fxh_int_mjy[ix]

Ned_phot_ID_match = Ned_phot_ID[iy]

ricci_Fx_hard_match_mjy_Ned = ricci_Fhx_match_Ned*4.136E8/(10-2)
ricci_Fx_soft_match_mjy_Ned = ricci_Fsx_match_Ned*4.136E8/(2-0.5)

L_test = Flux_to_Lum(ricci_Fx_hard_match_mjy_Ned,ricci_z_match_Ned)
L_test2 = Flux_to_Lum(Fxh_int_mjy_match_Ned,ricci_z_match_Ned)

for i in range(len(L_test)):
     print(ricci_ID_match_Ned[i],ricci_Lx_hard_match_Ned[i],np.log10(L_test[i]),np.log10(L_test2[i]))


plt.plot(ricci_Lx_hard_match_Ned,np.log10(L_test),'.',label='xspec')
plt.plot(ricci_Lx_hard_match_Ned,np.log10(L_test2),'.',label='lum to flux')
plt.plot(np.arange(42.5,45.5),np.arange(42.5,45.5),color='k')
plt.legend()
plt.grid()
plt.show()


# Fxh_int_mjy_match_Ned = ricci_Fx_hard_match_mjy_Ned#*correction_match_Ned

# Fxh_int_match_Ned_mjy = Fxh_int_match_Ned*4.136E8/(10-2) 

# goals_Fx_int_array = np.array([ricci_Fx_hard_match_mjy_Ned])
# goals_Fx_int_err_array = np.array([ricci_Fx_hard_match_mjy_Ned*0.3])

goals_Fx_int_array = np.array([Fxh_int_mjy_match_Ned])
goals_Fx_int_err_array = np.array([Fxh_int_mjy_match_Ned*0.3])


goals_nan_array = np.zeros(np.shape(Ned_phot_data['Fx_h'][iy]))
goals_nan_array[goals_nan_array == 0] = np.nan

# print(Ned_phot_data['Fx_h'][iy])

Ned_goals_flux = np.asarray([
    Fxh_int_mjy_match_Ned*1000,
    ricci_Fx_soft_match_mjy_Ned*1000,
    # np.asarray(Ned_phot_data['Fx_h'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['Fx_s'][iy],dtype=float)*1E6,
    goals_nan_array,
    np.asarray(Ned_phot_data['FUV'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['NUV'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['u'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['U'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['B'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['g'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['V'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['r'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['R'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['i'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['I'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['z'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['J'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['H'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['Ks'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAC1'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAC2'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAC3'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAC4'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAS1'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['MIPS1'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['IRAS2'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAS3'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['MIPS2'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['PACS2'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['MIPS3'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['F250'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['F350'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['F500'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['SCUBA2'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['VLA1'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['VLA2'][iy],dtype=float)*1E6
])

Ned_goals_flux_err = np.asarray([
    Fxh_int_mjy_match_Ned*1000*0.2,
    ricci_Fx_soft_match_mjy_Ned*1000*0.2,
    # np.asarray(Ned_phot_data['Fx_h'][iy],dtype=float)*1E6*0.2,
    # np.asarray(Ned_phot_data['Fx_s'][iy],dtype=float)*1E6*0.2,
    goals_nan_array,
    np.asarray(Ned_phot_data['FUVerr'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['NUVerr'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['uerr'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['Uerr'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['Berr'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['gerr'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['Verr'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['rerr'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['Rerr'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['ierr'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['Ierr'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['zerr'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['Jerr'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['Herr'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['Kserr'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAC1err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAC2err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAC3err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAC4err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAS1err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['MIPS1err'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['IRAS2err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['IRAS3err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['MIPS2err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['PACS2err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['MIPS3err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['F250err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['F350err'][iy],dtype=float)*1E6,
    np.asarray(Ned_phot_data['F500err'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['SCUBA2err'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['VLA1err'][iy],dtype=float)*1E6,
    # np.asarray(Ned_phot_data['VLA2err'][iy],dtype=float)*1E6
])


Ned_goals_flux = Ned_goals_flux.T
Ned_goals_flux_err = Ned_goals_flux_err.T


#### Read in Galaxy Templates ####
temp = ascii.read('/Users/connor_auge/Research/templets/A10_templates.txt')
temp_wave = np.asarray(temp['Wave'])
temp_flux = np.asarray(temp['E'])*1E-16  # erg/s/cm^-2/Hz
temp_wave_cgs = temp_wave*1E-8
temp_freq = 3E10/temp_wave_cgs
temp_nuFnu = temp_flux*temp_freq

dl = 10
dl_cgs = dl*3.086E18
temp_lum = temp_nuFnu*4*np.pi*dl_cgs**2

scale_array = [1.87E44, 2.33E44, 3.93E44]
###################################


# goals_filter_name = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'B_FLUX_APER2', 'V_FLUX_APER2', 'R', 'I', 'J_FLUX_APER2', 'H_FLUX_APER2',
                                # 'Ks_FLUX_APER2', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'IRAS1', 'FLUX_24', 'IRAS2', 'IRAS3', 'MIPS2', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500', 'SCUBA2', 'VLA1', 'VLA2'])


goals_filter_name = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U_JC', 'B_FLUX_APER2', 'V_FLUX_APER2', 'R_JC', 'I_Cousins', 'J_FLUX_APER2', 'H_FLUX_APER2',
                                'Ks_FLUX_APER2', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'IRAS1', 'FLUX_24', 'IRAS3', 'MIPS2', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])
GOALS_CIGALE_Filters = np.asarray(['Fx_hard', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U_JC', 'B_FLUX_APER2', 'V_FLUX_APER2', 'R_JC', 'I_Cousins', 'J_FLUX_APER2', 'H_FLUX_APER2',
                                'Ks_FLUX_APER2', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'IRAS1', 'FLUX_24', 'IRAS3', 'MIPS2', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])

# goals_filter_name = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'B_FLUX_APER2', 'V_FLUX_APER2', 'R', 'I', 'J_FLUX_APER2', 'H_FLUX_APER2',
#                                 'Ks_FLUX_APER2', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'IRAS1','IRAS2', 'FLUX_24', 'IRAS3', 'MIPS2', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])

# GOALS_CIGALE_Filters = np.asarray(['Fx_hard', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'B_FLUX_APER2', 'V_FLUX_APER2', 'R', 'I', 'J_FLUX_APER2', 'H_FLUX_APER2',
#                                 'Ks_FLUX_APER2', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'IRAS1','IRAS2', 'FLUX_24', 'IRAS3', 'MIPS2', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])


# goals_filter_name = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U','u_FLUX_APER2', 'B_FLUX_APER2', 'G', 'V_FLUX_APER2', 'R', 'r_FLUX_APER2', 'I', 'ip_FLUX_APER2','Z' , 'J_FLUX_APER2', 'H_FLUX_APER2',
#                                 'Ks_FLUX_APER2', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'IRAS1', 'FLUX_24', 'IRAS3', 'MIPS2', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])
# GOALS_CIGALE_Filters = np.asarray(['Fx_hard', 'Fx_soft', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U','u_FLUX_APER2', 'B_FLUX_APER2', 'G', 'V_FLUX_APER2', 'R', 'r_FLUX_APER2', 'I', 'ip_FLUX_APER2','Z' ,'J_FLUX_APER2', 'H_FLUX_APER2',
#                                 'Ks_FLUX_APER2', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'IRAS1', 'FLUX_24', 'IRAS3', 'MIPS2', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])


# goals_filter_name = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'B_FLUX_APER2', 'V_FLUX_APER2', 'R', 'I', 'J_FLUX_APER2', 'H_FLUX_APER2',
#                                 'Ks_FLUX_APER2', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX','SCUBA2', 'VLA1', 'VLA2'])
# GOALS_CIGALE_Filters = np.asarray(['Fx_hard', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'B_FLUX_APER2', 'V_FLUX_APER2', 'R', 'I', 'J_FLUX_APER2', 'H_FLUX_APER2',
#                                 'Ks_FLUX_APER2', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX','SCUBA2', 'VLA1', 'VLA2'])

###############################################################################
############################## Start CIGALE File ##############################
cigale_name = 'GOALS_new9.mag'
inf = open(f'../xcigale/data_input/{cigale_name}', 'w')
header = np.asarray(['# id', 'redshift'])
cigale_filters = Filters('filter_list.dat').pull_filter(GOALS_CIGALE_Filters, 'xcigale name')
for i in range(len(cigale_filters)):
    header = np.append(header, cigale_filters[i])
    header = np.append(header, cigale_filters[i]+'_err')
np.savetxt(inf, header, fmt='%s', delimiter='    ', newline=' ')
print('header: ',np.shape(header))
inf.close()
###############################################################################


'''CGOALS ULIRGS'''
norm_ulirg, F025_ulirg, F6_ulirg, F10_ulirg, F100_ulirg = [], [], [], [], []
FFIR_ulirg, WFIR_ulirg = [], []
UVslope_ulirg, MIRslope1_ulirg, MIRslope2_ulirg = [], [], []
uv_lum_ulirg, opt_lum_ulirg, mir_lum_ulirg, fir_lum_ulirg = [], [], [], []
ulirg_id, ulirg_z, ulirg_x, ulirg_y, ulirg_frac_err = [], [], [], [], []
median_x_ulirg, median_y_ulirg = [], []
median_fir_x_ulirg, median_fir_y_ulirg = [], []
ulirg_Lx_out, ulirg_Lx_corr_out = [], []
ulirg_Lx_hard_corr_out = []
Lbol_ulirg, Lbol_ulirg_sub = [], []
ulirg_field = []
ulirg_up_check = []
ulirg_Nh_out, ulirg_LIR_out = [], []
goals_irac_ch1, goals_irac_ch2, goals_irac_ch3, goals_irac_ch4 = [], [], [], []
int_x, int_y = [], []

ulirg_frac_error = []
ulirg_FIR_upper_lims = []

med_flux = []

IRx, IRy, IRagn = [], [], []

ulirg_shape = []

xout_ulirg = []

# '''
# fill_nan = np.zeros(len(ned_goals_filter_name)-len(cgoals_filter_name))
# fill_nan[fill_nan == 0] = np.nan
# for i in range(len(ricci_ID_match_U)):
#     if ricci_ID_match_U[i] == 'UGC 08058':
#         continue
#     elif len(U_goals_flux[i][U_goals_flux[i] > 0]) > 1:
#         print('U: ',ricci_ID_match_U[i])
#         source = AGN(ricci_ID_match_U[i],ricci_z_match_U[i],goals_filter_name,U_goals_flux[i],U_goals_flux_err[i])
#         source.MakeSED()
#         med_flux.append(U_goals_flux[i])
#         source.FIR_extrap(['FLUX_24','IRAS2', 'IRAS3', 'MIPS2', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])

#         ix, iy = source.Int_SED(xmin=1E-1, xmax=1E1)
#         median_x_ulirg.append(ix)
#         median_y_ulirg.append(iy)

#         wfir, ffir, f100 = source.Int_SED_FIR(Find_value=100.0, discreet=True)
#         WFIR_ulirg.append(wfir)
#         FFIR_ulirg.append(ffir)
#         F100_ulirg.append(f100/source.Find_value(1.0))

#         lbol = source.Find_Lbol()
#         lbol_sub = source.Find_Lbol_temp_sub(scale_array, temp_wave, temp_lum)
#         shape = source.SED_shape()

#         f1 = source.Find_value(1.0)
#         xval = source.Find_value(3E-4)
#         f6 = source.Find_value(6.0)
#         f025 = source.Find_value(0.25)
#         f10 = source.Find_value(10)
#         f2kev = source.Find_value(6.1992e-4)

#         Id, redshift, w, f, frac_err, up_check = source.pull_plot_info(norm_w=1)
#         ulirg_id.append(Id)
#         ulirg_x.append(w)
#         ulirg_y.append(f)
#         ulirg_frac_error.append(frac_err)
#         ulirg_FIR_upper_lims.append(up_check)
#         ulirg_Lx_out.append(np.log10(ricci_Lx_hard_obs_match_U[i]))
#         ulirg_Lx_corr_out.append(ricci_Lx_full_match_U[i])
#         ulirg_Nh_out.append(ricci_Nh_match_U[i])
#         ulirg_LIR_out.append(ricci_LIR_match_U[i])
#         ulirg_z.append(ricci_z_match_U[i])

#         # plot = Plotter(Id, redshift, w, f, ricci_Lx_full_match_U[i],f1,up_check)
#         # plot.PlotSED(point_x=100,point_y=f100/f1)

#         norm_ulirg.append(f1)
#         F025_ulirg.append(f025)
#         F6_ulirg.append(f6)
#         F10_ulirg.append(f10)
#         F100_ulirg.append(f100)
#         Lbol_ulirg.append(lbol)
#         Lbol_ulirg_sub.append(lbol_sub)

#         UVslope_ulirg.append(source.Find_slope(0.15, 1.0))
#         MIRslope1_ulirg.append(source.Find_slope(1.0, 6.5))
#         MIRslope2_ulirg.append(source.Find_slope(6.5, 10))

#         uv_lum_ulirg.append(source.find_Lum_range(0.1,0.35))
#         opt_lum_ulirg.append(source.find_Lum_range(0.35,3))
#         mir_lum_ulirg.append(source.find_Lum_range(3,30))
#         fir_lum_ulirg.append(source.find_Lum_range(30,500/(1+ricci_z_match_U[i])))

#         goals_irac_ch1.append(U_goals_flux[i][goals_filter_name == 'SPLASH_1_FLUX'][0])
#         goals_irac_ch2.append(U_goals_flux[i][goals_filter_name == 'SPLASH_2_FLUX'][0])
#         goals_irac_ch3.append(U_goals_flux[i][goals_filter_name == 'SPLASH_3_FLUX'][0])
#         goals_irac_ch4.append(U_goals_flux[i][goals_filter_name == 'SPLASH_4_FLUX'][0])
        
#         # source.write_cigale_file(cigale_name,goals_filter_name)
#         ulirg_field.append(5)

# print(np.shape(ricci_ID_match_Ned))

# run_ids = ['IRAS F05189-2524','NGC 7674','NGC6240N','UGC 08058','UGC 09913']

print(np.shape(Ned_goals_flux),len(goals_filter_name),np.shape(Ned_goals_flux_err))
for i in range(len(ricci_ID_match_Ned)):
    # if ricci_ID_match_Ned[i] == 'UGC 08058':
        # continue
    # if len(Ned_goals_flux[i][Ned_goals_flux[i] > 0]) > 1:
    # if ricci_ID_match_Ned[i] in run_ids:
        # print(i,ricci_ID_match_Ned[i], ricci_z_match_Ned[i], ricci_LIR_match_Ned[i])
        source = AGN(ricci_ID_match_Ned[i],ricci_z_match_Ned[i],goals_filter_name,Ned_goals_flux[i],Ned_goals_flux_err[i])
        source.MakeSED()
        med_flux.append(Ned_goals_flux[i])
        source.FIR_extrap(['FLUX_24', 'IRAS3', 'MIPS2', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])

        goals_flux_dict = source.MakeDict(goals_filter_name,Ned_goals_flux[i])
        goals_flux_err_dict = source.MakeDict(goals_filter_name,Ned_goals_flux_err[i])

        ix, iy = source.Int_SED(xmin=1E-1, xmax=1E1)
        median_x_ulirg.append(ix)
        median_y_ulirg.append(iy)

        wfir, ffir, f100 = source.Int_SED_FIR(Find_value=100.0, discreet=True)
        WFIR_ulirg.append(wfir)
        FFIR_ulirg.append(ffir)
        F100_ulirg.append(f100)

        ir_x, ir_y, ir_agn = source.IR_colors('SPLASH_1_FLUX','SPLASH_2_FLUX','SPLASH_3_FLUX','SPLASH_4_FLUX')
        IRx.append(ir_x)
        IRy.append(ir_y)
        IRagn.append(ir_agn)

        f1 = source.Find_value(1.0)
        xval = source.Find_value(3E-4)
        f6 = source.Find_value(6.0)
        f025 = source.Find_value(0.25)
        f10 = source.Find_value(10)
        f2kev = source.Find_value(6.1992e-4)

        lbol = source.Find_Lbol()
        lbol_sub = source.Find_Lbol(sub=True, Lscale=scale_array, Lnorm=f1, temp_x=temp_wave, temp_y=temp_lum)

        shape = source.SED_shape()

        Id, redshift, w, f, frac_err, up_check = source.pull_plot_info(norm_w=1)
        ulirg_id.append(Id)
        ulirg_x.append(w)
        ulirg_y.append(f)
        ulirg_frac_error.append(frac_err)
        ulirg_FIR_upper_lims.append(up_check)
        ulirg_Lx_out.append(ricci_Lx_hard_obs_match_Ned[i])
        ulirg_Lx_corr_out.append(ricci_Lx_full_match_Ned[i])
        ulirg_Lx_hard_corr_out.append(ricci_Lx_hard_match_Ned[i])
        ulirg_Nh_out.append(ricci_Nh_match_Ned[i])
        ulirg_LIR_out.append(ricci_LIR_match_Ned[i])
        ulirg_z.append(ricci_z_match_Ned[i])
        
        print(np.log10(xval), ricci_Lx_full_match_Ned[i])
        plot = Plotter(Id, redshift, w, f, 10**ricci_Lx_full_match_Ned[i],f1,up_check)
        plot.PlotSED(point_x=[3E-4,0.25,6,100],point_y=[xval/f1,f025/f1,f6/f1,f100/f1])

        norm_ulirg.append(f1)
        xout_ulirg.append(xval)
        F025_ulirg.append(f025)
        F6_ulirg.append(f6)
        F10_ulirg.append(f10)
        # F100_ulirg.append(f100)
        Lbol_ulirg.append(lbol)
        Lbol_ulirg_sub.append(lbol_sub)
        ulirg_shape.append(shape)

        UVslope_ulirg.append(source.Find_slope(0.15, 1.0))
        MIRslope1_ulirg.append(source.Find_slope(1.0, 6.5))
        MIRslope2_ulirg.append(source.Find_slope(6.5, 10))

        uv_lum_ulirg.append(source.find_Lum_range(0.1,0.35))
        opt_lum_ulirg.append(source.find_Lum_range(0.35,3))
        mir_lum_ulirg.append(source.find_Lum_range(3,30))
        fir_lum_ulirg.append(source.find_Lum_range(30,500/(1+ricci_z_match_Ned[i])))

        goals_irac_ch1.append(Ned_goals_flux[i][goals_filter_name == 'SPLASH_1_FLUX'][0])
        goals_irac_ch2.append(Ned_goals_flux[i][goals_filter_name == 'SPLASH_2_FLUX'][0])
        goals_irac_ch3.append(Ned_goals_flux[i][goals_filter_name == 'SPLASH_3_FLUX'][0])
        goals_irac_ch4.append(Ned_goals_flux[i][goals_filter_name == 'SPLASH_4_FLUX'][0])
        source.write_cigale_file2(cigale_name, GOALS_CIGALE_Filters, goals_flux_dict, goals_flux_err_dict, int_fx=goals_Fx_int_array[0][i],int_fx_err=goals_Fx_int_err_array[0][i])
        # source.write_cigale_file(cigale_name, goals_filter_name, use_int_fx=False)
        ulirg_field.append(5)

# '''
print('Done with ULIRGS')

ulirg_id, ulirg_z, ulirg_x, ulirg_y, ulirg_Lx_out, ulirg_Lx_corr_out, median_x_ulirg, median_y_ulirg = np.asarray(ulirg_id), np.asarray(ulirg_z), np.asarray(ulirg_x), np.asarray(ulirg_y), np.asarray(ulirg_Lx_out), np.asarray(ulirg_Lx_corr_out), np.asarray(median_x_ulirg), np.asarray(median_y_ulirg)
norm_ulirg, xout_ulirg, F025_ulirg, F6_ulirg, F10_ulirg, F100_ulirg = np.asarray(norm_ulirg), np.asarray(xout_ulirg), np.asarray(F025_ulirg), np.asarray(F6_ulirg), np.asarray(F10_ulirg), np.asarray(F100_ulirg)
UVslope_ulirg, MIRslope1_ulirg, MIRslope2_ulirg = np.asarray(UVslope_ulirg), np.asarray(MIRslope1_ulirg), np.asarray(MIRslope2_ulirg)
FFIR_ulirg, WFIR_ulirg = np.asarray(FFIR_ulirg), np.asarray(WFIR_ulirg)
Lbol_ulirg, Lbol_ulirg_sub = np.asarray(Lbol_ulirg), np.asarray(Lbol_ulirg_sub)
median_fir_x_ulirg, median_fir_y_ulirg = np.asarray(median_fir_x_ulirg), np.asarray(median_fir_y_ulirg)
uv_lum_ulirg, opt_lum_ulirg, mir_lum_ulirg, fir_lum_ulirg = np.asarray(uv_lum_ulirg), np.asarray(opt_lum_ulirg), np.asarray(mir_lum_ulirg), np.asarray(fir_lum_ulirg)
ulirg_field = np.asarray(ulirg_field)
ulirg_up_check = np.asarray(ulirg_up_check)
ulirg_Nh_out = np.asarray(ulirg_Nh_out)
ulirg_LIR_out = np.asarray(ulirg_LIR_out)
goals_irac_ch1, goals_irac_ch2, goals_irach_ch3, goals_irach_ch4 = np.asarray(goals_irac_ch1), np.asarray(goals_irac_ch2), np.asarray(goals_irac_ch3), np.asarray(goals_irac_ch4)
IRx, IRy, IRagn = np.asarray(IRx), np.asarray(IRy), np.asarray(IRagn)
ulirg_shape = np.asarray(ulirg_shape)
ulirg_Lx_hard_corr_out = np.asarray(ulirg_Lx_hard_corr_out)

with fits.open('/Users/connor_auge/Research/Disertation/catalogs/AHA_SEDs_out.fits') as hdul:
    aha_cols = hdul[1].columns
    aha_data = hdul[1].data 

aha_id = aha_data['ID']
aha_z = aha_data['z'] 
aha_x = aha_data['x']
aha_y = aha_data['y']
aha_Lx = aha_data['Lx']
aha_Lx_hard = aha_data['Lx_hard']
aha_norm = aha_data['norm']
aha_FIR_upper_lims = aha_data['FIR_upper_lims']
aha_frac_err = aha_data['frac_err']

aha_F025 = aha_data['F025']
aha_F1 = aha_data['F1']
aha_F6 = aha_data['F6']
aha_F10 = aha_data['F10']
aha_F100 = aha_data['F100']

aha_shape = aha_data['shape']
aha_Lbol = aha_data['Lbol_sub']

aha_int_x = aha_data['int_x']
aha_int_y = aha_data['int_y']
aha_wfir = aha_data['wfir']
aha_ffir = aha_data['ffir']

aha_Nh = aha_data['Nh']
aha_UV_lum = aha_data['UV_lum']
aha_MIR_lum = aha_data['MIR_lum']
aha_FIR_lum = aha_data['FIR_lum']

aha_field = aha_data['field']
aha_uv_slope = aha_data['uv_slope']
aha_mir_slope1 = aha_data['mir_slope1']
aha_mir_slope2 = aha_data['mir_slope2']

aha_Nh_check = aha_data['Nh_check']
# ind = 60
# s = 3
ulirg_plot = Plotter_Letter(ulirg_id,ulirg_z,ulirg_x,ulirg_y,ulirg_frac_err)
plot = Plotter(ulirg_id, ulirg_z, ulirg_x, ulirg_y, ulirg_Lx_corr_out, norm_ulirg, ulirg_FIR_upper_lims)
AHA_plot_Letter = Plotter_Letter(aha_id, aha_z, aha_x, aha_y, aha_frac_err)
# AHA_plot = Plotter(aha_id[aha_shape == s][ind], aha_z[aha_shape == s][ind], aha_x[aha_shape == s][ind], aha_y[aha_shape == s][ind], aha_Lx[aha_shape == s][ind], aha_norm[aha_shape == s][ind], aha_FIR_upper_lims[aha_shape == s][ind])
AHA_plot = Plotter(aha_id, aha_z, aha_x, aha_y, aha_Lx, aha_norm, aha_FIR_upper_lims)
AHA_plot2 = Plotter_Letter2(aha_id, aha_z, aha_x, aha_y, aha_frac_err)
# for i in range(len(ulirg_id)):
#     print(i,ulirg_id[i], ulirg_LIR_out[i],IRagn[i])

med_flux = np.asarray(med_flux)
med_flux_combine = np.nanmedian(med_flux,axis=0)

cigale_Lagn_goals = np.asarray([6.52542985e+44, 3.41348582e+45, 8.00824173e+44, 1.59974679e+44,
                                 2.66964585e+44, 3.99133031e+44, 3.42646075e+44, 4.74850437e+44,
                                 6.45289326e+44, 8.58714120e+44, 3.11057516e+44, 8.16531397e+44,
                                 1.15867334e+43, 7.42606770e+44, 3.44895503e+45, 1.10673834e+45,
                                 5.20256742e+44])

cigale_Lx_goals = np.asarray([2.80337802e+43, 1.84189379e+44, 2.29516011e+43, 2.28154231e+43,
                              7.84260205e+42, 1.71470814e+43, 1.47203556e+43, 1.16297103e+43,
                              1.49451528e+43, 9.08612633e+43, 1.56374298e+43, 6.39142650e+43,
                              5.48583874e+42, 9.40886970e+43, 4.46329229e+43, 2.60954700e+43,
                              1.22669864e+43])*1.64

# AHA_plot.PlotSED()

# source_combine = AGN('Med_GOALS_obs', np.nanmedian(ricci_z_match_Ned), goals_filter_name,med_flux_combine,med_flux_combine*0.3)
# source_combine.write_cigale_file(cigale_name, goals_filter_name)

# plot.L_hist('GOALS_figs/shape_hist',ulirg_shape,'SED Shape',[0,7],[0,7,1])
# plot.L_hist('GOALS_figs/L6_hist',np.log10(F6_ulirg),r'log L (6$\mu{\rm m})$ [erg/s]',[43,46],[43,46,0.25])
# plot.L_hist('GOALS_figs/Nh_hist',np.log10(ulirg_Nh_out),r'log $N_{\rm H}$ [cm$^{-2}$]',[19,25],[19,25,0.25])
# plot.L_hist('GOALS_figs/L100_hist',np.log10(F100_ulirg), r'log L (100$\mu{\rm m})$ [erg/s]', [43, 46], [43, 46, 0.25])
# plot.L_hist('GOALS_figs/Lone_hist',np.log10(norm_ulirg),r'log L (1$\mu{\rm m})$ [erg/s]',[43,46],[43,46,0.25])
# plot.L_hist('GOALS_figs/Lx_hist', ulirg_Lx_corr_out,r'log $L_{\rm X}$ [erg/s]', [42, 46], [42, 46, 0.25])
# plt.scatter(np.log10(xout_ulirg), ulirg_Lx_corr_out)
# plt.plot(np.arange(41,46),np.arange(41,46),color='k')
# plt.xlabel('value found')
# plt.ylabel('Lx')
# plt.grid()
# plt.show()

# print(Lbol_ulirg_sub)
# print(aha_Lbol)
# print(aha_MIR_lum)
# print(mir_lum_ulirg)

# print(ulirg_id[Lbol_ulirg_sub < 10**44])

# AHA_plot2.Upanels_ratio_plots('GOALS_figs/Lum_Lbol','Lbol','UV-MIR-FIR/Lbol','Bins',aha_Nh,aha_Lx,np.log10(aha_Lbol),np.log10(aha_UV_lum),np.log10(aha_MIR_lum),np.log10(aha_FIR_lum),np.log10(aha_F025),np.log10(aha_F6),np.log10(aha_F100),np.log10(aha_F10),aha_F1,aha_field,aha_z,aha_uv_slope,aha_mir_slope1,aha_mir_slope2,aha_FIR_upper_lims,shape=aha_shape,Nh_upper=aha_Nh_check,compare=True,comp_L=np.log10(Lbol_ulirg_sub),comp_UV=np.log10(uv_lum_ulirg),comp_MIR=np.log10(mir_lum_ulirg),comp_FIR=np.log10(fir_lum_ulirg))

# fig = plt.figure(figsize=(10, 8))
# ax1 = plt.subplot(111, aspect='equal', adjustable='box')
# pts = plt.scatter(np.log10(ulirg_Nh_out[np.log10(ulirg_Nh_out) > 23]),ulirg_Lx_corr_out[np.log10(ulirg_Nh_out) > 23]-ulirg_Lx_out[np.log10(ulirg_Nh_out) > 23], c=ulirg_LIR_out[np.log10(ulirg_Nh_out) > 23], edgecolor='k', s=100)
# axcb = fig.colorbar(pts)  # make colorbar
# axcb.mappable.set_clim(10.75, 12.25)  # initialize colorbar limits Lir
# axcb.set_label(label=r'log $L_{\rm IR}/L_{\odot}$')
# plt.xlabel(r'log $N_{\rm H}$ [cm$^{-2}$]')
# plt.ylabel(r'log L$_{\rm X,int}$/L$_{\rm X,obs}$')
# # plt.plot(np.arange(40, 47), np.arange(40, 47), color='k')
# plt.xlim(22, 25.5)
# plt.ylim(-0.25, 3.25)
# ax1.set_xticks([22, 23, 24, 25])
# ax1.set_yticks([0.0, 1.0, 2.0, 3.0])
# plt.grid()
# plt.savefig(f'/Users/connor_auge/Desktop/Final_plots/GOALS_figs/Lx_correction_Nh.pdf')
# plt.show()


# print(ulirg_Lx_corr_out)
# print(ulirg_Lx_hard_corr_out)
# print(len(ulirg_Lx_corr_out))
# print(len(ulirg_Lx_hard_corr_out))


# plt.figure(figsize=(10,10),facecolor='w')
# # plt.hist(ulirg_LIR_out/(ulirg_Lx_corr_out-np.log10(3.8E33)),bins=np.arange(-0.5,3,0.25),alpha=0.75,label='GOALS (U)LIRGs AGN')
# plt.hist(ulirg_LIR_out/(ulirg_Lx_hard_corr_out-np.log10(3.8E33)),bins=np.arange(-0.5,3,0.25),alpha=0.75,label='GOALS (U)LIRGs AGN')
# # plt.hist(np.log10(aha_FIR_lum[aha_FIR_lum > 1E45])-aha_Lx[aha_FIR_lum > 1E45],color='gray',alpha=0.5,bins=np.arange(-0.5,3,0.25),label=r'High $L_{\rm IR}$ AHA AGN')
# plt.hist(np.log10(aha_FIR_lum[aha_FIR_lum > 1E45])-aha_Lx_hard[aha_FIR_lum > 1E45],color='gray',alpha=0.5,bins=np.arange(-0.5,3,0.25),label=r'High $L_{\rm IR}$ AHA AGN')
# plt.axvline(46.7-45.2, c='r', lw=3,label='IRAS 09104+4109')
# plt.xlabel(r'log $L_{\rm IR}$/log $L_{\rm X}$',fontsize=18)
# plt.legend()
# plt.show()

# plt.scatter((ulirg_Lx_corr_out-np.log10(3.8E33)), ulirg_LIR_out)
# plt.plot(np.arange(9,13),np.arange(9,13),color='k')
# plt.show()

# plt.scatter(aha_Lx,aha_Lx_hard)
# plt.plot(np.arange(41,48),np.arange(41,48),color='k')
# plt.show()

# ulirg_plot.multi_SED('ULIRG_new',ulirg_x,ulirg_y,ulirg_Lx_out,median_x_ulirg,median_y_ulirg,suptitle='SEDs of ULIRGs',norm=norm_ulirg,mark=ulirg_field,spec_z=ulirg_z,wfir=None,ffir=None,up_check=ulirg_up_check,med_x_fir=median_fir_x_ulirg,med_y_fir=median_fir_y_ulirg)
plot.multi_SED('GOALS_figs/SEDs_intFx_new',median_x=median_x_ulirg,median_y=median_y_ulirg,wfir=WFIR_ulirg,ffir=FFIR_ulirg,Median_line=True,FIR_upper='data only',GOALS=True)
#
#  plot.L_scatter_comp('GOALS_figs/Lx_comp_Nh',ulirg_Lx_out,ulirg_Lx_corr_out,color_array=np.log10(ulirg_Nh_out),xlabel=r'Observed log L$_{\rm X}$ [erg/s]', ylabel=r'Intrinsic log L$_{\rm X}$ [erg/s]',colorbar_label=r'log N$_{\rm H}$ [cm$^{-2}$]')
plot.IR_colors('GOALS_figs/IR_colors',IRx,IRy,ulirg_LIR_out,colorbar=True,colorbar_label=r'log L$_{\rm IR}$ [L$_\odot$]',Lacy=True,agn=IRagn,select_sources=True)

# AHA_plot.L_scatter_3panels('GOALS_figs/AGN_Lx_scatter_fit_comp','UV-MIR-FIR', 'Lx', 'X-axis', aha_F1, aha_F025, aha_F6, aha_F100, shape=aha_shape, L=aha_Lbol, compare=True, comp_L=ulirg_Lx_corr_out, comp_uv=F025_ulirg, comp_mir=F6_ulirg, comp_fir=F100_ulirg)
# AHA_plot.median_SED_1panel('GOALS_figs/median_SEDs_shape_comp', median_x=aha_int_x, median_y=aha_int_y, wfir=aha_wfir, ffir=aha_ffir, shape=aha_shape, FIR_upper='upper lims', bins='shape',compare=True,comp_med_x=median_x_ulirg,comp_med_y=median_y_ulirg,comp_wfir=WFIR_ulirg,comp_ffir=FFIR_ulirg)
# AHA_plot.L_ratio_1panel('GOALS_figs/Lx_Lbol_comp','Lbol','Lbol/Lx','X-axis',aha_F1,aha_F025,aha_F6,aha_F100,shape=aha_shape,L=np.log10(aha_Lbol),compare=True,comp_x=np.log10(Lbol_ulirg_sub),comp_y=np.log10(Lbol_ulirg_sub)-ulirg_Lx_corr_out)
# AHA_plot.L_ratio_1panel('GOALS_figs/Lx_Lbol_comp','Lbol','Lbol/Lx','X-axis',aha_F1,aha_F025,aha_F6,aha_F100,shape=aha_shape,L=np.log10(aha_Lbol),compare=True,comp_x=np.log10(cigale_Lagn_goals),comp_y=np.log10(cigale_Lagn_goals)-np.log10(cigale_Lx_goals))

# plt.figure(figsize=(10,10))
# plt.plot(ulirg_LIR_out+np.log10(3.8E33),np.log10(Lbol_ulirg_sub),'.')
# plt.plot(np.arange(44,48),np.arange(44,48),color='k')
# plt.xlabel('LIR')
# plt.ylabel('Lbol')
# plt.grid()
# plt.show()


ranalli = Lit_functions.Ranalli(np.arange(40,55))
ulirg_LIR_out += np.log10(3.8E33)

# plt.figure(figsize=(8,8))
# plt.scatter(ulirg_LIR_out,ulirg_Lx_corr_out,edgecolor='k',s=45,color='gray')
# plt.plot(np.arange(40,55),ranalli,color='k',label='Ranalli')
# plt.xlabel(r'L$_{\rm IR}$ [erg/s]')
# plt.ylabel(r'L$_{\rm X}$ [erg/s]')
# plt.xlim(41,49)
# plt.ylim(38,48)
# plt.grid()
# plt.legend(fontsize=16)
# plt.show()

# print(ulirg_LIR_out)
# plot.L_scatter_1panel('GOALS_figs/FIR_Lx_comp.pdf','Lx','X-axis',np.log10(norm_ulirg),np.log10(F025_ulirg),np.log10(F6_ulirg),np.log10(F100_ulirg),ulirg_Nh_out,ulirg_shape,Lbol_ulirg_sub,line='Ranalli')

# print(Lbol_ulirg_sub)
# print(ulirg_Lx_corr_out)
# print(ulirg_LIR_out) 

# for i in range(len(ulirg_Lx_out)):
#     print(ulirg_Lx_out[i],ulirg_Lx_corr_out[i],ulirg_Lx_out[i]/ulirg_Lx_corr_out[i])

# print(ulirg_Lx_out/ulirg_Lx_corr_out)
# print(np.nanmedian(ulirg_Lx_out/ulirg_Lx_corr_out))