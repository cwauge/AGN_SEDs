'''
Script to run AGN class from SED.py and plot results with SED_plots.py Plotter class. 
Reads in Photometry and X-ray data from the COSMOS field, S82X field, GOODS-N/S fields, and for individual GOALS galaxies. 
Updated - August 22, 2022
'''

from statistics import median_grouped
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import Lit_functions
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io import ascii
from SED_v8 import AGN
from SED_plots_v2 import Plotter
from filters import Filters
from plots_Letter import Plotter_Letter
from plots_Letter2 import Plotter_Letter2
from SED_shape_plots import SED_shape_Plotter
from match import match
from mag_flux import mag_to_flux
from mag_flux import magerr_to_fluxerr
from PlotSED_Check import IntPlot
from astropy.table import Table


ti = time.perf_counter() # Start timer

path = '/Users/connor_auge/Research/Disertation/catalogs/' # Path for photometry catalogs

# set redshift and X-ray luminosity limits
z_min = 0.0
z_max = 1.2

Lx_min = 43
Lx_max = 60

# set the X-ray luminosity limits for the GOALS sources
goals_Lx_min = 35

###################################################################################
###################################################################################
############################## Read in COSMOS files ###############################

# Read in the data
# COSMOS2020 catalog
# cosmos = fits.open(path+'cosmos2020/classic/COSMOS2020_CLASSIC_R1_v2.0_master.fits')
# cosmos_data = cosmos[1].data
# cosmos.close()

# COSMOS2020 catalog
cosmos = fits.open(path+'COSMOS2020_CLASSIC_R1_v2.2_p3.fits')
cosmos_data = cosmos[1].data
cosmos.close()

# COSMOS2015 catalog
cosmos2015 = fits.open(path+'COSMOS2015_Laigle+_v1.1.fits')
cosmos2015_data = cosmos2015[1].data
cosmos2015.close()

# COSMOS FIR catalog 
cosmos_fir = fits.open(path+'cosmos_superdeblended.fit')
cosmos_fir_data = cosmos_fir[1].data
cosmos.close()

# Chandra 2016 catalog
chandra_cosmos = fits.open(path+'chandra_COSMOS_legacy_opt_NIR_counterparts_20160113_4d.fits')
chandra_cosmos_data = chandra_cosmos[1].data
chandra_cosmos.close()

# Chandra updated fits catalog
chandra_cosmos2 = fits.open(path+'chandra_cosmos_legacy_spectra_bestfit_20210225.fits')
chandra_cosmos2_data = chandra_cosmos2[1].data
chandra_cosmos2.close()

# Chandra compton thick catalog
chandra_cosmos_ct = fits.open(path+'chandra_cosmos_legacy_spectra_bestfit_ComptonThick_Lanzuisi18.fits')
chandra_cosmos_ct_data = chandra_cosmos_ct[1].data
chandra_cosmos_ct.close()

# DEIMOS 10k Spec z cat
deimos = ascii.read('/Users/connor_auge/Downloads/deimos_10k_March2018_new/deimos_redshifts.tbl')
deimos_id = np.asarray(deimos['ID'])
deimos_z = np.asarray(deimos['zspec'])
deimos_remarks = np.asarray(deimos['Remarks'])
deimos_ID = np.asarray([int(i[1:]) for i in deimos_id if 'L' in i])
deimos_z_spec = np.asarray([deimos_z[i] for i in range(len(deimos_z)) if 'L' in deimos_id[i]])

# Gather all IDs
chandra_cosmos_phot_id = chandra_cosmos_data['id_k_uv']
cosmos_laigle_id = cosmos_data['ID_COSMOS2015']
cosmos_fir_id = cosmos_fir_data['ID']
cosmos2015_ID = cosmos2015_data['NUMBER']
cosmos_xid = cosmos_data['id_chandra']
chandra_cosmos_xid = chandra_cosmos_data['id_x']
chandra_cosmos2_xid = chandra_cosmos2_data['id_x']
chandra_cosmos_ct_xid = chandra_cosmos_ct_data['id_x']

# X-ray coords
chandra_cosmos_RA = chandra_cosmos_data['RA_x']
chandra_cosmos_DEC = chandra_cosmos_data['DEC_x']

# Redshfits
# cosmos_sz = cosmos_data['sz_zspec']
# cosmos_ez = cosmos_data['ez_z_spec']
chandra_cosmos_z = chandra_cosmos_data['z_spec']
chandra_cosmos_z_phot = chandra_cosmos_data['z_best']

chandra_cosmos2_z = chandra_cosmos2_data['z_best']

# X-ray Flux
chandra_cosmos_Fx_hard = chandra_cosmos_data['flux_h']
chandra_cosmos_Fx_soft = chandra_cosmos_data['flux_s']
chandra_cosmos_Fx_full = chandra_cosmos_data['flux_f']

chandra_cosmos2_Fx_hard = chandra_cosmos2_data['flux_210']
chandra_cosmos2_Fx_soft = chandra_cosmos2_data['flux_052']
chandra_cosmos2_Fx_full = chandra_cosmos2_data['flux_0510']

# X-ray Luminosity (non-log)
chandra_cosmos_Lx_hard = np.asarray([10**i for i in chandra_cosmos_data['Lx_210']])
chandra_cosmos_Lx_soft = np.asarray([10**i for i in chandra_cosmos_data['Lx_052']])
chandra_cosmos_Lx_full = np.asarray([10**i for i in chandra_cosmos_data['Lx_0510']])

chandra_cosmos2_Lx_hard = np.asarray([10**i for i in chandra_cosmos2_data['Lx_210']])
chandra_cosmos2_Lx_full = np.asarray([(10**i)*1.64 for i in chandra_cosmos2_data['Lx_210']]) # Correction from hard to full band

# Other Chandra Data
# Spec-type from hardness ratio
chandra_cosmos_spec_type = chandra_cosmos_data['spec_type'] # spec type

# Column Density
chandra_cosmos_nh = chandra_cosmos_data['Nh']
chandra_cosmos_nh_lo = chandra_cosmos_data['Nh_lo']
chandra_cosmos_nh_hi = chandra_cosmos_data['Nh_up']

chandra_cosmos2_nh = chandra_cosmos2_data['nh']
chandra_cosmos2_nh_lo_err = chandra_cosmos2_data['nh_lo_err']
chandra_cosmos2_nh_up_err = chandra_cosmos2_data['nh_up_err']


# Absorption Correction - full, hard, and soft bands with uppr and lower limts
chandra_cosmos_abs_corr_h = chandra_cosmos_data['abs_corr_210'] 
chandra_cosmos_abs_corr_up_h = chandra_cosmos_data['abs_corr_210_up']
chandra_cosmos_abs_corr_lo_h = chandra_cosmos_data['abs_corr_210_lo']
chandra_cosmos_abs_corr_s = chandra_cosmos_data['abs_corr_052']
chandra_cosmos_abs_corr_up_s = chandra_cosmos_data['abs_corr_052_up']
chandra_cosmos_abs_corr_lo_s = chandra_cosmos_data['abs_corr_052_lo']
chandra_cosmos_abs_corr_f = chandra_cosmos_data['abs_corr_0510']
chandra_cosmos_abs_corr_up_f = chandra_cosmos_data['abs_corr_0510_up']
chandra_cosmos_abs_corr_lo_f = chandra_cosmos_data['abs_corr_0510_lo']


# X-ray luminosities and Column densites from the Chandra Compoton thick catalog
chandra_cosmos_ct_Lx_hard_obs = np.asarray([10**i for i in chandra_cosmos_ct_data['loglx']])
chandra_cosmos_ct_Lx_full_obs = np.asarray([(10**i)*1.64 for i in chandra_cosmos_ct_data['loglx']])
chandra_cosmos_ct_Lx_hard = np.asarray([10**i for i in chandra_cosmos_ct_data['loglxcor']])
chandra_cosmos_ct_Lx_full = np.asarray([(10**i)*1.64 for i in chandra_cosmos_ct_data['loglxcor']])
chandra_cosmos_ct_nh = chandra_cosmos_ct_data['nh2_22']

# Gather DEIMOS spec redshifts to redshift array
for i in range(len(chandra_cosmos_z)):
    ind = np.where(deimos_ID == chandra_cosmos_phot_id[i])[0]
    if len(ind) > 0:
        if deimos_z_spec[ind][0] > 0:
            chandra_cosmos_z[i] = deimos_z_spec[ind][0]
        else:
            continue
    else:
        continue

# Gather Luminosity absorption corrections from the Chandra COSMOS cat1 and correct the luminositeis
abs_corr_use_h = []
abs_corr_use_s = []
abs_corr_use_f = []
check_abs = []
for i in range(len(chandra_cosmos_abs_corr_f)):
    if chandra_cosmos_abs_corr_f[i] != -99.0:
        abs_corr_use_f.append(chandra_cosmos_abs_corr_f[i])
        abs_corr_use_h.append(chandra_cosmos_abs_corr_h[i])
        abs_corr_use_s.append(chandra_cosmos_abs_corr_s[i])
        check_abs.append(0)

    elif chandra_cosmos_abs_corr_up_f[i] != -99.0:
        abs_corr_use_f.append(chandra_cosmos_abs_corr_up_f[i])
        abs_corr_use_h.append(chandra_cosmos_abs_corr_up_h[i])
        abs_corr_use_s.append(chandra_cosmos_abs_corr_up_s[i])
        check_abs.append(1)

    elif chandra_cosmos_abs_corr_lo_f[i] != -99.0:
        abs_corr_use_f.append(chandra_cosmos_abs_corr_lo_f[i])
        abs_corr_use_h.append(chandra_cosmos_abs_corr_lo_h[i])
        abs_corr_use_s.append(chandra_cosmos_abs_corr_lo_s[i])
        check_abs.append(2)

    else:
        print('NO GOOD ABSORPTION CORRECTION DATA')

check_abs = np.asarray(check_abs)
# Turn final absorption lists into arrays
abs_corr_use_h = np.asarray(abs_corr_use_h)
abs_corr_use_s = np.asarray(abs_corr_use_s)
abs_corr_use_f = np.asarray(abs_corr_use_f)

# Correct the X-ray luminosity from the 2016 Chandra Catalog for absorption
chandra_cosmos_Lx_hard /= abs_corr_use_h
chandra_cosmos_Lx_soft /= abs_corr_use_s
chandra_cosmos_Lx_full /= abs_corr_use_f

# Gather the column density from the three chandra catalogs for the full and hard band Lx
chandra_cosmos_Nh = []
check = []
cosmos_Nh_check = []
for i in range(len(chandra_cosmos_Lx_full)):    
    ind = np.where(chandra_cosmos2_xid == chandra_cosmos_xid[i])[0] # Check if there is a match to updated Chandra catalog 
    ind_ct = np.where(chandra_cosmos_ct_xid == chandra_cosmos_xid[i])[0] # Check if there is a match to compton thick Chandra catalog 

    if len(ind_ct) > 0:
        chandra_cosmos_Nh.append(chandra_cosmos_ct_nh[ind_ct][0]) # if there is a match append Nh from compton thick catalog 
        chandra_cosmos_Lx_hard[i] = chandra_cosmos_ct_Lx_hard[ind_ct] # replace Lx from original Chandra catalog with that from the CT cat
        chandra_cosmos_Lx_full[i] = chandra_cosmos_ct_Lx_full[ind_ct]
        check.append(3) # count which catalog data is from
        cosmos_Nh_check.append(0)
        check_abs[i] = 0

    elif len(ind) > 0:
        chandra_cosmos_Lx_hard[i] = chandra_cosmos2_Lx_hard[ind] # replace Lx from orginal Chandra catalog with that from updated cat
        chandra_cosmos_Lx_full[i] = chandra_cosmos2_Lx_full[ind]
        check_abs[i] = 0
        if chandra_cosmos2_nh_lo_err[ind][0] == -99.:
            chandra_cosmos_Nh.append(chandra_cosmos2_nh[ind][0]+chandra_cosmos2_nh_up_err[ind][0]) # if there is Nh upper limit in updated cat append to Nh list
            # chandra_cosmos_Nh.append(0.0)
            check.append(2.5)
            cosmos_Nh_check.append(1)
        else:
            chandra_cosmos_Nh.append(chandra_cosmos2_nh[ind][0]) # if there is a match append Nh from updated catalog
            check.append(2)
            cosmos_Nh_check.append(0)
    else: # if no matches to updated or CT catalogs take Nh value from original catalog
        if chandra_cosmos_nh[i] == -99.: # If no good value take upper or lower limits 
            if chandra_cosmos_nh_lo[i] != -99.:
                # chandra_cosmos_Nh.append(0.0)
                chandra_cosmos_Nh.append(chandra_cosmos_nh_lo[i])
                cosmos_Nh_check.append(2)
            else:
                # chandra_cosmos_Nh.append(0.0)
                chandra_cosmos_Nh.append(chandra_cosmos_nh_hi[i])
                cosmos_Nh_check.append(1)
        else:    
            chandra_cosmos_Nh.append(chandra_cosmos_nh[i])
            cosmos_Nh_check.append(0)
        check.append(1)
chandra_cosmos_Nh = np.asarray(chandra_cosmos_Nh)*1E22
check = np.asarray(check)
cosmos_Nh_check = np.asarray(cosmos_Nh_check)


print('COSMOS All Lx cat: ', len(chandra_cosmos_Lx_full))

# Limit chandra sample to sources in z and Lx range
cosmos_condition = (chandra_cosmos_z > z_min) & (chandra_cosmos_z <= z_max) & (np.log10(chandra_cosmos_Lx_full) >= Lx_min) & (np.log10(chandra_cosmos_Lx_full) <= Lx_max) & (chandra_cosmos_phot_id != -99.)
# cosmos_condition = (chandra_cosmos_z > z_min) & (chandra_cosmos_z <= z_max) & (np.log10(chandra_cosmos_Lx_hard) >= Lx_min) & (np.log10(chandra_cosmos_Lx_hard) <= Lx_max) & (chandra_cosmos_phot_id != -99.)

chandra_cosmos_phot_id = chandra_cosmos_phot_id[cosmos_condition]
chandra_cosmos_xid = chandra_cosmos_xid[cosmos_condition]
chandra_cosmos_RA = chandra_cosmos_RA[cosmos_condition]
chandra_cosmos_DEC = chandra_cosmos_DEC[cosmos_condition]
chandra_cosmos_z = chandra_cosmos_z[cosmos_condition]
chandra_cosmos_Fx_full = chandra_cosmos_Fx_full[cosmos_condition]
chandra_cosmos_Fx_hard = chandra_cosmos_Fx_hard[cosmos_condition]
chandra_cosmos_Fx_soft = chandra_cosmos_Fx_soft[cosmos_condition]
chandra_cosmos_Lx_full = chandra_cosmos_Lx_full[cosmos_condition]
chandra_cosmos_Lx_hard = chandra_cosmos_Lx_hard[cosmos_condition]
chandra_cosmos_Lx_soft = chandra_cosmos_Lx_soft[cosmos_condition]
chandra_cosmos_spec_type = chandra_cosmos_spec_type[cosmos_condition]
chandra_cosmos_Nh = chandra_cosmos_Nh[cosmos_condition]
cosmos_Nh_check = cosmos_Nh_check[cosmos_condition]
abs_corr_use_h = abs_corr_use_h[cosmos_condition]
abs_corr_use_s = abs_corr_use_s[cosmos_condition]
abs_corr_use_f = abs_corr_use_f[cosmos_condition]
check_abs = check_abs[cosmos_condition]
print('COSMOS Lx z: ', len(chandra_cosmos_phot_id))

# Match chandra subsample to photometry catalog
cosmos_ix, cosmos_iy = match(chandra_cosmos_phot_id,cosmos_laigle_id) # match cats based on laigle ID
# cosmos_ix, cosmos_iy = match(chandra_cosmos_xid,cosmos_xid) # match cats based on chandra ID in COSMOS2020

cosmos_laigle_id_match = cosmos_laigle_id[cosmos_iy]
chandra_cosmos_phot_id_match = chandra_cosmos_phot_id[cosmos_ix]
chandra_cosmos_xid_match = chandra_cosmos_xid[cosmos_ix]
chandra_cosmos_RA_match = chandra_cosmos_RA[cosmos_ix]
chandra_cosmos_DEC_match = chandra_cosmos_DEC[cosmos_ix]
chandra_cosmos_z_match = chandra_cosmos_z[cosmos_ix]
chandra_cosmos_Fx_full_match = chandra_cosmos_Fx_full[cosmos_ix]
chandra_cosmos_Fx_hard_match = chandra_cosmos_Fx_hard[cosmos_ix]
chandra_cosmos_Fx_soft_match = chandra_cosmos_Fx_soft[cosmos_ix]
chandra_cosmos_Lx_full_match = chandra_cosmos_Lx_full[cosmos_ix]
chandra_cosmos_Lx_hard_match = chandra_cosmos_Lx_hard[cosmos_ix]
chandra_cosmos_Lx_soft_match = chandra_cosmos_Lx_soft[cosmos_ix]
chandra_cosmos_spec_type_match = chandra_cosmos_spec_type[cosmos_ix]
chandra_cosmos_Nh_match = chandra_cosmos_Nh[cosmos_ix]
cosmos_Nh_check_match = cosmos_Nh_check[cosmos_ix]
abs_corr_use_h_match = abs_corr_use_h[cosmos_ix]
abs_corr_use_s_match = abs_corr_use_s[cosmos_ix]
abs_corr_use_f_match = abs_corr_use_f[cosmos_ix]
check_abs_match = check_abs[cosmos_ix]

cosmos2015_ix, cosmos2015_iy = match(cosmos_laigle_id_match,cosmos2015_ID)


cosmos_F24, cosmos_F100, cosmos_F160, cosmos_F250, cosmos_F350, cosmos_F500 = [], [], [], [], [], []
cosmos_F24err, cosmos_F100err, cosmos_F160err, cosmos_F250err, cosmos_F350err, cosmos_F500err = [], [], [], [], [], []

ix2,iy2 = match(cosmos_laigle_id_match,cosmos_fir_id)
print('HERE: ')
print(len(cosmos_fir_id[iy2]))
# for i in range(len(cosmos_laigle_id_match)):
#     ind = np.where(cosmos_fir_id == cosmos_laigle_id_match[i])[0]
#     print(ind)
#     if len(ind) > 0:
#         cosmos_F24.append(cosmos_fir_data['F24'][ind])
#         cosmos_F24err.append(cosmos_fir_data['e_F24'][ind])
#         cosmos_F24.append(cosmos_fir_data['F100'][ind])
#         cosmos_F24err.append(cosmos_fir_data['e_F100'][ind])
#         cosmos_F24.append(cosmos_fir_data['F160'][ind])
#         cosmos_F24err.append(cosmos_fir_data['e_F160'][ind])
#         cosmos_F24.append(cosmos_fir_data['F250'][ind])
#         cosmos_F24err.append(cosmos_fir_data['e_F250'][ind])
#         cosmos_F24.append(cosmos_fir_data['F350'][ind])
#         cosmos_F24err.append(cosmos_fir_data['e_F350'][ind])
#         cosmos_F24.append(cosmos_fir_data['F500'][ind])
#         cosmos_F24err.append(cosmos_fir_data['e_F500'][ind])
#     else:
#         cosmos_F24.append(-9999.)
#         cosmos_F24err.append(-9999.)
#         cosmos_F24.append(-9999.)
#         cosmos_F24err.append(-9999.)
#         cosmos_F24.append(-9999.)
#         cosmos_F24err.append(-9999.)
#         cosmos_F24.append(-9999.)
#         cosmos_F24err.append(-9999.)
#         cosmos_F24.append(-9999.)
#         cosmos_F24err.append(-9999.)
#         cosmos_F24.append(-9999.)
#         cosmos_F24err.append(-9999.)
# cosmos_F24, cosmos_F100, cosmos_F160, cosmos_F250, cosmos_F350, cosmos_F500 = np.asarray(cosmos_F24), np.asarray(cosmos_F100), np.asarray(cosmos_F160), np.asarray(cosmos_F250), np.asarray(cosmos_F350), np.asarray(cosmos_F500) 
# cosmos_F24err, cosmos_F100err, cosmos_F160err, cosmos_F250err, cosmos_F350err, cosmos_F500err = np.asarray(cosmos_F24err), np.asarray(cosmos_F100err), np.asarray(cosmos_F160err), np.asarray(cosmos_F250err), np.asarray(cosmos_F350err), np.asarray(cosmos_F500err)

# print(len(cosmos_laigle_id_match),len(cosmos_F24))
print('COSMOS phot match: ', len(chandra_cosmos_phot_id_match))

# print(len(check_abs_match),len(check_abs_match[check_abs_match == 0]))

# Convert the X-ray flux values to from cgs to mJy
chandra_cosmos_Fx_full_match_mjy = chandra_cosmos_Fx_full_match*4.136E8/(10-0.5)
chandra_cosmos_Fx_hard_match_mjy = chandra_cosmos_Fx_hard_match*4.136E8/(10-2)
chandra_cosmos_Fx_soft_match_mjy = chandra_cosmos_Fx_soft_match*4.136E8/(2-0.5)

chandra_cosmos_fx_full_match_mjy_int = chandra_cosmos_Fx_full_match_mjy/abs_corr_use_f_match
chandra_cosmos_Fx_hard_match_mjy_int = chandra_cosmos_Fx_hard_match_mjy/abs_corr_use_h_match
chandra_cosmos_Fx_soft_match_mjy_int = chandra_cosmos_Fx_soft_match_mjy/abs_corr_use_s_match

# cosmos_Fx_int_array = np.array([chandra_cosmos_Fx_hard_match_mjy.int, chandra_cosmos_Fx_soft_match_mjy.int]).T
cosmos_Fx_int_array = np.array([chandra_cosmos_Fx_hard_match_mjy_int])

# Create a 1D array of NaN that is the length of the number of COSMOS sources - used to create "blank" values in flux array
cosmos_nan_array = np.zeros(np.shape(cosmos_laigle_id_match))
cosmos_nan_array[cosmos_nan_array == 0] = np.nan

# Create flux and flux error arrays for the COSMOS data. Matched to chandra data. NaN array separating the X-ray from the FUV data.
cosmos_flux_array = np.array([
    chandra_cosmos_Fx_hard_match_mjy*1000, chandra_cosmos_Fx_soft_match_mjy*1000,
    cosmos_nan_array,
    cosmos_data['GALEX_FUV_FLUX'][cosmos_iy],
    cosmos_data['GALEX_NUV_FLUX'][cosmos_iy],
    cosmos_data['CFHT_u_FLUX_APER2'][cosmos_iy],
    cosmos_data['HSC_g_FLUX_APER2'][cosmos_iy],
    cosmos_data['HSC_r_FLUX_APER2'][cosmos_iy],
    cosmos_data['HSC_i_FLUX_APER2'][cosmos_iy],
    cosmos_data['HSC_z_FLUX_APER2'][cosmos_iy],
    cosmos_data['HSC_y_FLUX_APER2'][cosmos_iy],
    cosmos_data['UVISTA_J_FLUX_APER2'][cosmos_iy],
    cosmos_data['UVISTA_H_FLUX_APER2'][cosmos_iy],
    cosmos_data['UVISTA_Ks_FLUX_APER2'][cosmos_iy],
    cosmos_data['SPLASH_CH1_FLUX'][cosmos_iy],
    cosmos_data['SPLASH_CH2_FLUX'][cosmos_iy],
    cosmos_data['SPLASH_CH3_FLUX'][cosmos_iy],
    cosmos_data['SPLASH_CH4_FLUX'][cosmos_iy],
    # cosmos_data['FIR_24_FLUX'][cosmos_iy],
    # cosmos_data['FIR_100_FLUX'][cosmos_iy],
    # cosmos_data['FIR_160_FLUX'][cosmos_iy],
    # cosmos_data['FIR_250_FLUX'][cosmos_iy],
    # cosmos_data['FIR_350_FLUX'][cosmos_iy],
    # cosmos_data['FIR_500_FLUX'][cosmos_iy],
    cosmos2015_data['Flux_24'][cosmos2015_iy],
    cosmos2015_data['Flux_100'][cosmos2015_iy]*1000,
    cosmos2015_data['Flux_160'][cosmos2015_iy]*1000,
    cosmos2015_data['Flux_250'][cosmos2015_iy]*1000,
    cosmos2015_data['Flux_350'][cosmos2015_iy]*1000,
    cosmos2015_data['Flux_500'][cosmos2015_iy]*1000
    # cosmos_F24*1000,
    # cosmos_F100*1000,
    # cosmos_F160*1000,
    # cosmos_F250*1000,
    # cosmos_F350*1000,
    # cosmos_F500*1000
    # cosmos_data['FIR_850_FLUX'][cosmos_iy],
    # cosmos_data['FIR_1100_FLUX'][cosmos_iy],
    # cosmos_data['FIR_20CM_FLUX'][cosmos_iy]
])

cosmos_flux_err_array = np.array([
    chandra_cosmos_Fx_hard_match_mjy*1000*0.2, chandra_cosmos_Fx_soft_match_mjy*1000*0.2,
    cosmos_nan_array,
    cosmos_data['GALEX_FUV_FLUXERR'][cosmos_iy],
    cosmos_data['GALEX_NUV_FLUXERR'][cosmos_iy],
    cosmos_data['CFHT_u_FLUXERR_APER2'][cosmos_iy],
    cosmos_data['HSC_g_FLUXERR_APER2'][cosmos_iy],
    cosmos_data['HSC_r_FLUXERR_APER2'][cosmos_iy],
    cosmos_data['HSC_i_FLUXERR_APER2'][cosmos_iy],
    cosmos_data['HSC_z_FLUXERR_APER2'][cosmos_iy],
    cosmos_data['HSC_y_FLUXERR_APER2'][cosmos_iy],
    cosmos_data['UVISTA_J_FLUXERR_APER2'][cosmos_iy],
    cosmos_data['UVISTA_H_FLUXERR_APER2'][cosmos_iy],
    cosmos_data['UVISTA_Ks_FLUXERR_APER2'][cosmos_iy],
    cosmos_data['SPLASH_CH1_FLUXERR'][cosmos_iy],
    cosmos_data['SPLASH_CH2_FLUXERR'][cosmos_iy],
    cosmos_data['SPLASH_CH3_FLUXERR'][cosmos_iy],
    cosmos_data['SPLASH_CH4_FLUXERR'][cosmos_iy],
    # cosmos_data['FIR_24_FLUXERR'][cosmos_iy],
    # cosmos_data['FIR_100_FLUXERR'][cosmos_iy],
    # cosmos_data['FIR_160_FLUXERR'][cosmos_iy],
    # cosmos_data['FIR_250_FLUXERR'][cosmos_iy],
    # cosmos_data['FIR_350_FLUXERR'][cosmos_iy],
    # cosmos_data['FIR_500_FLUXERR'][cosmos_iy],
    cosmos2015_data['Fluxerr_24'][cosmos2015_iy],
    cosmos2015_data['Fluxerr_100'][cosmos2015_iy]*1000,
    cosmos2015_data['Fluxerr_160'][cosmos2015_iy]*1000,
    cosmos2015_data['Fluxerr_250'][cosmos2015_iy]*1000,
    cosmos2015_data['Fluxerr_350'][cosmos2015_iy]*1000,
    cosmos2015_data['Fluxerr_500'][cosmos2015_iy]*1000
    # cosmos_F24err*1000,
    # cosmos_F100err*1000,
    # cosmos_F160err*1000,
    # cosmos_F250err*1000,
    # cosmos_F350err*1000,
    # cosmos_F500err*1000
    # cosmos_data['FIR_850_FLUXERR'][cosmos_iy],
    # cosmos_data['FIR_1100_FLUXERR'][cosmos_iy],
    # cosmos_data['FIR_20CM_FLUXERR'][cosmos_iy]
])

# Transpose arrays so each row is a new source and each column is a obs filter
cosmos_flux_array = cosmos_flux_array.T
cosmos_flux_err_array = cosmos_flux_err_array.T

print(np.shape(cosmos_flux_array))


###################################################################################

###################################################################################
###################################################################################
############################ Read in Stripe82X files ##############################

# Most recent LaMassa catalog - Includes Photometry and X-ray data
lamassa = fits.open(path+'S82X_catalog_with_photozs_unique_Xraysrcs_likely_cps_w_mbh.fits')
lamassa_data = lamassa[1].data
lamassa.close()

# Updated X-ray catalog from Peca et al.
peca = fits.open(path+'Peca_S82X.fits')
peca_data = peca[1].data
peca.close()

with fits.open(path+'Peca_S82X_full.fit') as hdul:
    peca2_data = hdul[1].data

# WISE catalog with forced photometry on SLOAN sources
unwise = ascii.read('/Users/connor_auge/Research/desktop_catalogs/unwise_matches.csv')
unwise_ID = np.asarray(unwise['ID'])
unwise_W1 = np.asarray(unwise['unW1'])
unwise_W2 = np.asarray(unwise['unW2'])
unwise_W3 = np.asarray(unwise['unW3'])
unwise_W4 = np.asarray(unwise['unW4'])
unwise_W1_err = np.asarray(unwise['unW1_err'])
unwise_W2_err = np.asarray(unwise['unW2_err'])
unwise_W3_err = np.asarray(unwise['unW3_err'])
unwise_W4_err = np.asarray(unwise['unW4_err'])

# Gather IDs and sort into usable ID array for LaMassa
lamassa_id = lamassa_data['msid']
lamassa_id_recno = lamassa_data['rec_no']
lamassa_obsID = lamassa_data['OBSID']

s82x_id = []
for i,j in enumerate(lamassa_id):
    if j == 0:
        s82x_id.append(lamassa_id_recno[i])
    else:
        s82x_id.append(j)
s82x_id = np.asarray(s82x_id)

# Additional data from LaMassa
s82x_ra = lamassa_data['XRAY_RA']
s82x_dec = lamassa_data['XRAY_DEC']
s82x_cat = lamassa_data['XRAY_SRC']
s82x_z = lamassa_data['SPEC_Z']
s82x_phot_z = lamassa_data['PHOTO_Z']
s82x_Lx_full = np.asarray([10**i for i in lamassa_data['FULL_LUM']])
s82x_Lx_hard = np.array([10**i for i in lamassa_data['HARD_LUM']])
s82x_Lx_soft = np.array([10**i for i in lamassa_data['SOFT_LUM']])
s82x_Fx_full = lamassa_data['FULL_FLUX']
s82x_Fx_hard = lamassa_data['HARD_FLUX']
s82x_Fx_soft = lamassa_data['SOFT_FLUX']
# s82x_spec_class = lamassa_data['SPEC_CLASS']
s82x_Fx_full_int = s82x_Fx_full.copy()
s82x_Fx_hard_int = s82x_Fx_hard.copy()
s82x_Fx_soft_int = s82x_Fx_soft.copy()

# S82X Spec type data
s82x_spec_type_in = lamassa_data['spec_class']
s82x_spec_class = np.ones(np.shape(s82x_spec_type_in))
for i in range(len(s82x_spec_type_in)):
    if i != 'QSO':
        s82x_spec_class[i] = 2
    else:
        continue

# Read in WISE data 
s82x_W1 = lamassa_data['W1']
s82x_W2 = lamassa_data['W2']
s82x_W3 = lamassa_data['W3']
s82x_W4 = lamassa_data['W4']
s82x_W1_err = lamassa_data['W1_err']
s82x_W2_err = lamassa_data['W2_err']
s82x_W3_err = lamassa_data['W3_err']
s82x_W4_err = lamassa_data['W4_err']

# Replace W3 and W4 data with good data from the unWISE catalog
for i, j in enumerate(s82x_id):
    ind = np.where(unwise_ID == j)[0]
    if len(ind) == 1:
        if np.isnan(unwise_W3[ind]):
            continue
        elif unwise_W3[ind] <= 0.0:
            continue
        elif magerr_to_fluxerr(unwise_W3[ind], unwise_W3_err[ind],'W3',AB=True)/mag_to_flux(unwise_W3[ind],'W3',AB=True) > 0.4:
            continue
        else:
            s82x_W3[i] = unwise_W3[ind][0]
            s82x_W3_err[i] = unwise_W3_err[ind][0]

        if np.isnan(unwise_W4[ind]):
            continue
        elif unwise_W4[ind] <= 0.0:
            continue
        elif magerr_to_fluxerr(unwise_W4[ind], unwise_W4_err[ind], 'W4', AB=True)/mag_to_flux(unwise_W4[ind], 'W4', AB=True) > 0.4:
            continue
        else:
            s82x_W4[i] = unwise_W4[ind][0]
            s82x_W4_err[i] = unwise_W4_err[ind][0]


# Read in Peca data
# peca_ID = peca_data['srcid']
# peca_Lx_full = peca_data['lumin_f']
# peca_Lx_hard = peca_data['lumin_h']
# peca_Lx_soft = peca_data['lumin_s']
# peca_Lx_full_obs = peca_data['lumin_of']
# peca_Lx_hard_obs = peca_data['lumin_oh']
# peca_Lx_soft_obs = peca_data['lumin_os']
# peca_Fx_full = peca_data['flux_f']
# peca_Fx_hard = peca_data['flux_h']
# peca_Fx_soft = peca_data['flux_s']
# peca_Nh = peca_data['nh']
# peca_abs_corr_full = peca_Lx_full_obs/peca_Lx_full
# peca_abs_corr_hard = peca_Lx_hard_obs/peca_Lx_hard
# peca_abs_corr_soft = peca_Lx_soft_obs/peca_Lx_soft

# peca_Fx_full_int = peca_Fx_full/peca_abs_corr_full
# peca_Fx_hard_int = peca_Fx_hard/peca_abs_corr_hard
# peca_Fx_soft_int = peca_Fx_soft/peca_abs_corr_soft

# New Peca
peca_ID = peca2_data['source']
peca_Lx_full = peca2_data['LintF']
peca_Lx_hard = peca2_data['LintH']
peca_Lx_soft = peca2_data['LintS']
peca_Lx_full_obs = peca2_data['LobsF']
peca_Lx_hard_obs = peca2_data['LobsH']
peca_Lx_soft_obs = peca2_data['LobsS']
peca_Fx_full = peca2_data['FluxF']
peca_Fx_hard = peca2_data['FluxH']
peca_Fx_soft = peca2_data['FluxS']
peca_Nh = peca2_data['NH']
peca_abs_corr_full = peca_Lx_full_obs/peca_Lx_full
peca_abs_corr_hard = peca_Lx_hard_obs/peca_Lx_hard
peca_abs_corr_soft = peca_Lx_soft_obs/peca_Lx_soft

peca_Fx_full_int = peca_Fx_full/peca_abs_corr_full
peca_Fx_hard_int = peca_Fx_hard/peca_abs_corr_hard
peca_Fx_soft_int = peca_Fx_soft/peca_abs_corr_soft


# Fill in Nh data from Peca and replace LaMassa X-ray data with Peca 
s82x_Nh = []
s82x_Nh_check = []
for i, j in enumerate(s82x_id):
    ind = np.where(peca_ID == j)[0]
    if len(ind) == 1:
        s82x_Lx_full[i] = peca_Lx_full[ind][0]
        s82x_Lx_hard[i] = peca_Lx_hard[ind][0]
        s82x_Lx_soft[i] = peca_Lx_soft[ind][0]
        s82x_Fx_full[i] = peca_Fx_full[ind][0]
        s82x_Fx_hard[i] = peca_Fx_hard[ind][0]
        s82x_Fx_soft[i] = peca_Fx_soft[ind][0]
        s82x_Fx_full_int[i] = peca_Fx_full_int[ind][0]
        s82x_Fx_hard_int[i] = peca_Fx_hard_int[ind][0]
        s82x_Fx_soft_int[i] = peca_Fx_soft_int[ind][0]
        
        s82x_Nh.append(peca_Nh[ind][0])
        s82x_Nh_check.append(0)
    else:
        s82x_Nh.append(0.0)
        s82x_Nh_check.append(3)
s82x_Nh = np.asarray(s82x_Nh)
s82x_Nh_check = np.asarray(s82x_Nh_check)

print('S82X All: ', len(s82x_id))

# Limit Stripe82X sample to sources in z and Lx range
s82x_condition = (s82x_z > z_min) & (s82x_z <= z_max) & (np.log10(s82x_Lx_full) >= Lx_min) & (np.log10(s82x_Lx_full) <= Lx_max) & (np.logical_and(s82x_ra >= 13, s82x_ra <=37))
# s82x_condition = (s82x_phot_z > z_min) & (s82x_phot_z <= z_max) & (np.log10(s82x_Lx_full) >= Lx_min) & (np.log10(s82x_Lx_full) <= Lx_max) 

s82x_id = s82x_id[s82x_condition]
s82x_cat = s82x_cat[s82x_condition]
s82x_z = s82x_z[s82x_condition]
s82x_phot_z = s82x_phot_z[s82x_condition]
s82x_ra = s82x_ra[s82x_condition]
s82x_dec = s82x_dec[s82x_condition]
s82x_Lx_full = s82x_Lx_full[s82x_condition]
s82x_Lx_hard = s82x_Lx_hard[s82x_condition]
s82x_Lx_soft = s82x_Lx_soft[s82x_condition]
s82x_Fx_full = s82x_Fx_full[s82x_condition]
s82x_Fx_hard = s82x_Fx_hard[s82x_condition]
s82x_Fx_soft = s82x_Fx_soft[s82x_condition]
s82x_Fx_full_int = s82x_Fx_full_int[s82x_condition]
s82x_Fx_hard_int = s82x_Fx_hard_int[s82x_condition]
s82x_Fx_soft_int = s82x_Fx_soft_int[s82x_condition]
s82x_Nh = s82x_Nh[s82x_condition]
s82x_Nh_check = s82x_Nh_check[s82x_condition]
s82x_spec_class = s82x_spec_class[s82x_condition]

print('S82X Lx z coords: ', len(s82x_id))
print('S82X match: ', len(s82x_id))

# Convert the X-ray flux values to from cgs to mJy
s82x_Fx_full_mjy = s82x_Fx_full*4.136E8/(10-0.5)
s82x_Fx_hard_mjy = s82x_Fx_hard*4.136E8/(10-2)
s82x_Fx_soft_mjy = s82x_Fx_soft*4.136E8/(2-0.5)
s82x_Fxerr_full_mjy = s82x_Fx_full_mjy*0.2 # Place holder errors for X-ray flux values
s82x_Fxerr_hard_mjy = s82x_Fx_hard_mjy*0.2 # Place holder errors for X-ray flux values
s82x_Fxerr_soft_mjy = s82x_Fx_soft_mjy*0.2 # Place holder errors for X-ray flux values

s82x_Fx_full_int_mjy = s82x_Fx_full_int*4.136E8/(10-0.5)
s82x_Fx_hard_int_mjy = s82x_Fx_hard_int*4.136E8/(10-2)
s82x_Fx_soft_int_mjy = s82x_Fx_soft_int*4.136E8/(2-0.5)
s82x_Fxerr_full_int_mjy = s82x_Fx_full_int_mjy*0.2 # Place holder errors for X-ray flux values
s82x_Fxerr_hard_int_mjy = s82x_Fx_hard_int_mjy*0.2 # Place holder errors for X-ray flux values
s82x_Fxerr_soft_int_mjy = s82x_Fx_soft_int_mjy*0.2 # Place holder errors for X-ray flux values

s82x_Fx_int_array = np.array([s82x_Fx_hard_int_mjy,s82x_Fx_soft_int_mjy]).T

# Create a 1D array of NaN that is the length of the number of COSMOS sources - used to create "blank" values in flux array
s82x_nan_array = np.zeros(np.shape(s82x_id))
s82x_nan_array[s82x_nan_array == 0] = np.nan

# Create flux and flux error arrays for the S82X data. NaN array separating the X-ray from the FUV data and MIR data from FIR data.
s82x_flux_array = np.array([
    s82x_Fx_hard_mjy*1000, s82x_Fx_soft_mjy*1000,
    s82x_nan_array,
    mag_to_flux(lamassa_data['mag_FUV'][s82x_condition],'FUV')*1E6,
    mag_to_flux(lamassa_data['mag_NUV'][s82x_condition],'NUV')*1E6,
    mag_to_flux(lamassa_data['u'][s82x_condition], 'sloan_u')*1E6,
    mag_to_flux(lamassa_data['g'][s82x_condition], 'sloan_g')*1E6,
    mag_to_flux(lamassa_data['r'][s82x_condition], 'sloan_r')*1E6,
    mag_to_flux(lamassa_data['i'][s82x_condition], 'sloan_i')*1E6,
    mag_to_flux(lamassa_data['z'][s82x_condition], 'sloan_z')*1E6,
    mag_to_flux(lamassa_data['JVHS'][s82x_condition], 'JVHS')*1E6,
    mag_to_flux(lamassa_data['HVHS'][s82x_condition], 'HVHS')*1E6,
    mag_to_flux(lamassa_data['KVHS'][s82x_condition], 'KVHS')*1E6,
    mag_to_flux(s82x_W1[s82x_condition], 'W1')*1E6,
    # mag_to_flux(lamassa_data['CH1_SPIES'][s82x_condition],'Ch1Spies')*1E6,
    # mag_to_flux(lamassa_data['CH2_SPIES'][s82x_condition],'Ch2Spies')*1E6,
    mag_to_flux(s82x_W2[s82x_condition], 'W2')*1E6,
    mag_to_flux(s82x_W3[s82x_condition], 'W3')*1E6,
    mag_to_flux(s82x_W4[s82x_condition], 'W4')*1E6,
    s82x_nan_array,
    lamassa_data['F250'][s82x_condition]*1000,
    lamassa_data['F350'][s82x_condition]*1000,
    lamassa_data['F500'][s82x_condition]*1000
])

s82x_flux_err_array = np.array([
    s82x_Fxerr_hard_mjy*1000, s82x_Fxerr_soft_mjy*1000,
    s82x_nan_array,
    magerr_to_fluxerr(lamassa_data['mag_FUV'][s82x_condition],
                      lamassa_data['magerr_FUV'][s82x_condition], 'FUV')*1E6,
    magerr_to_fluxerr(lamassa_data['mag_NUV'][s82x_condition],
                      lamassa_data['magerr_NUV'][s82x_condition], 'NUV')*1E6,
    magerr_to_fluxerr(lamassa_data['u'][s82x_condition],
                      lamassa_data['u_err'][s82x_condition], 'sloan_u')*1E6,
    magerr_to_fluxerr(lamassa_data['g'][s82x_condition],
                      lamassa_data['g_err'][s82x_condition], 'sloan_g')*1E6,
    magerr_to_fluxerr(lamassa_data['r'][s82x_condition],
                      lamassa_data['r_err'][s82x_condition], 'sloan_r')*1E6,
    magerr_to_fluxerr(lamassa_data['i'][s82x_condition],
                      lamassa_data['i_err'][s82x_condition], 'sloan_i')*1E6,
    magerr_to_fluxerr(lamassa_data['z'][s82x_condition],
                      lamassa_data['z_err'][s82x_condition], 'sloan_z')*1E6,
    magerr_to_fluxerr(lamassa_data['JVHS'][s82x_condition],
                      lamassa_data['JVHS_err'][s82x_condition], 'JVHS')*1E6,
    magerr_to_fluxerr(lamassa_data['HVHS'][s82x_condition],
                      lamassa_data['HVHS_err'][s82x_condition], 'HVHS')*1E6,
    magerr_to_fluxerr(lamassa_data['KVHS'][s82x_condition],
                      lamassa_data['KVHS_err'][s82x_condition], 'KVHS')*1E6,
    magerr_to_fluxerr(s82x_W1[s82x_condition], s82x_W1_err[s82x_condition], 'W1')*1E6,
    # magerr_to_fluxerr(lamassa_data['CH1_SPIES'][s82x_condition],
    #                   lamassa_data['CH1_SPIES_err'][s82x_condition], 'Ch1Spies')*1E6,
    # magerr_to_fluxerr(lamassa_data['CH2_SPIES'][s82x_condition],
    #                   lamassa_data['CH2_SPIES'][s82x_condition], 'Ch2Spies')*1E6,                  
    magerr_to_fluxerr(s82x_W2[s82x_condition], s82x_W2_err[s82x_condition], 'W2')*1E6,
    magerr_to_fluxerr(s82x_W3[s82x_condition], s82x_W3_err[s82x_condition], 'W3')*1E6,
    magerr_to_fluxerr(s82x_W4[s82x_condition], s82x_W4_err[s82x_condition], 'W4')*1E6,
    s82x_nan_array,
    lamassa_data['F250_err'][s82x_condition]*1000,
    lamassa_data['F350_err'][s82x_condition]*1000,
    lamassa_data['F500_err'][s82x_condition]*1000
])

# Transpose arrays so each row is a new source and each column is a obs filter
s82x_flux_array = s82x_flux_array.T
s82x_flux_err_array = s82x_flux_err_array.T
###################################################################################


###################################################################################
###################################################################################
############################## Read in GOODS-N files ##############################
goodsN_auge = fits.open(path+'GOODsN_full_cat_update.fits')
goodsN_auge_data = goodsN_auge[1].data
goodsN_auge.close()

goodsN_auge_ID = goodsN_auge_data['id_xray']
goodsN_auge_phot_ID = goodsN_auge_data['id_rainbow']
goodsN_RA = goodsN_auge_data['xRA']
goodsN_DEC = goodsN_auge_data['xDec']
goodsN_auge_Lx = goodsN_auge_data['Lx']
goodsN_auge_Lx_hard = goodsN_auge_data['Lx']*0.611
goodsN_auge_z = goodsN_auge_data['z_spec']
goodsN_auge_Nh = goodsN_auge_data['Nh']
goodsN_auge_Nh_lo = goodsN_auge_data['Nh_lo']
goodsN_auge_Nh_hi = goodsN_auge_data['Nh_hi']

goodsN_auge_condition = (np.log10(goodsN_auge_Lx) >= Lx_min) & (np.log10(goodsN_auge_Lx) <= Lx_max) &(goodsN_auge_z > z_min) & (goodsN_auge_z <= z_max) & (goodsN_auge_z != 0.0)

goodsN_auge_ID_match = goodsN_auge_ID[goodsN_auge_condition]
goodsN_auge_phot_ID_match = goodsN_auge_phot_ID[goodsN_auge_condition]
goodsN_auge_RA_match = goodsN_RA[goodsN_auge_condition]
goodsN_auge_DEC_match = goodsN_DEC[goodsN_auge_condition]
goodsN_auge_Lx_match = goodsN_auge_Lx[goodsN_auge_condition]
goodsN_auge_Lx_hard_match = goodsN_auge_Lx_hard[goodsN_auge_condition]
goodsN_auge_z_match = goodsN_auge_z[goodsN_auge_condition]
goodsN_auge_Nh_match = goodsN_auge_Nh[goodsN_auge_condition]
goodsN_auge_Nh_lo_match = goodsN_auge_Nh_lo[goodsN_auge_condition]
goodsN_auge_Nh_hi_match = goodsN_auge_Nh_hi[goodsN_auge_condition]

goodsN_Nh_check = []
for i in range(len(goodsN_auge_Nh_match)):
    if goodsN_auge_Nh_match[i] <= 0.0:
        if goodsN_auge_Nh_lo_match[i] <= 0.0:
            goodsN_auge_Nh_match[i] = goodsN_auge_Nh_hi_match[i]
            goodsN_Nh_check.append(1)
        else:
            goodsN_auge_Nh_match[i] = goodsN_auge_Nh_lo_match[i]
            goodsN_Nh_check.append(2)
    else:
        goodsN_Nh_check.append(0)


print('GOODS-N 2 match: ',len(goodsN_auge_ID_match))

goodsN_auge_Fx_hard_match_mjy = goodsN_auge_data['Fx_hard'][goodsN_auge_condition]*4.136E8/(10-2)
goodsN_auge_Fx_soft_match_mjy = goodsN_auge_data['Fx_soft'][goodsN_auge_condition]*4.136E8/(2-0.5)

goodsN_nan_array = np.zeros(np.shape(goodsN_auge_ID_match)) # Create nan array with length == to the number of sources to be input to the photometry array
goodsN_nan_array[goodsN_nan_array == 0] = np.nan
goodsN_flux_array_auge = np.asarray([goodsN_auge_Fx_hard_match_mjy*1000, goodsN_auge_Fx_soft_match_mjy*1000,
	goodsN_nan_array,
    goodsN_auge_data['FUV'][goodsN_auge_condition],
	goodsN_auge_data['NUV'][goodsN_auge_condition],
    goodsN_auge_data['U'][goodsN_auge_condition],
    goodsN_auge_data['F435W'][goodsN_auge_condition],
	goodsN_auge_data['B'][goodsN_auge_condition], 
	goodsN_auge_data['V'][goodsN_auge_condition],
    goodsN_auge_data['F606W'][goodsN_auge_condition],
	goodsN_auge_data['R'][goodsN_auge_condition], 
	goodsN_auge_data['I'][goodsN_auge_condition],
    goodsN_auge_data['F775W'][goodsN_auge_condition],
    goodsN_auge_data['F814W'][goodsN_auge_condition],
	goodsN_auge_data['z'][goodsN_auge_condition],
    goodsN_auge_data['F105W'][goodsN_auge_condition],
    goodsN_auge_data['F125W'][goodsN_auge_condition],
	goodsN_auge_data['J'][goodsN_auge_condition],
    goodsN_auge_data['F140W'][goodsN_auge_condition],
    goodsN_auge_data['F160W'][goodsN_auge_condition],
	goodsN_auge_data['H'][goodsN_auge_condition],
	goodsN_auge_data['K'][goodsN_auge_condition],
	goodsN_auge_data['irac_ch1'][goodsN_auge_condition],
	goodsN_auge_data['irac_ch2'][goodsN_auge_condition],
	goodsN_auge_data['irac_ch3'][goodsN_auge_condition],
	goodsN_auge_data['irac_ch4'][goodsN_auge_condition],
	goodsN_auge_data['f24'][goodsN_auge_condition],
    goodsN_auge_data['f70'][goodsN_auge_condition],
	goodsN_auge_data['f100'][goodsN_auge_condition],
	goodsN_auge_data['f160'][goodsN_auge_condition],
	goodsN_auge_data['f250'][goodsN_auge_condition],
	goodsN_auge_data['f350'][goodsN_auge_condition],
	goodsN_auge_data['f500'][goodsN_auge_condition]
]) 

goodsN_flux_err_array_auge = np.asarray([goodsN_auge_Fx_hard_match_mjy*1000*0.2, goodsN_auge_Fx_soft_match_mjy*1000*0.2,
	goodsN_nan_array,
    goodsN_auge_data['FUVerr'][goodsN_auge_condition],
	goodsN_auge_data['NUVerr'][goodsN_auge_condition],
    goodsN_auge_data['Uerr'][goodsN_auge_condition],
    goodsN_auge_data['F435Werr'][goodsN_auge_condition], 
	goodsN_auge_data['Berr'][goodsN_auge_condition], 
	goodsN_auge_data['Verr'][goodsN_auge_condition],
    goodsN_auge_data['F606Werr'][goodsN_auge_condition],
	goodsN_auge_data['Rerr'][goodsN_auge_condition], 
	goodsN_auge_data['Ierr'][goodsN_auge_condition],
    goodsN_auge_data['F775Werr'][goodsN_auge_condition],
    goodsN_auge_data['F814Werr'][goodsN_auge_condition],
	goodsN_auge_data['zerr'][goodsN_auge_condition],
    goodsN_auge_data['F105Werr'][goodsN_auge_condition],
    goodsN_auge_data['F125Werr'][goodsN_auge_condition],
	goodsN_auge_data['Jerr'][goodsN_auge_condition],
    goodsN_auge_data['F140Werr'][goodsN_auge_condition],
    goodsN_auge_data['F160Werr'][goodsN_auge_condition],
	goodsN_auge_data['Herr'][goodsN_auge_condition],
	goodsN_auge_data['Kerr'][goodsN_auge_condition],
	goodsN_auge_data['irac_ch1err'][goodsN_auge_condition],
	goodsN_auge_data['irac_ch2err'][goodsN_auge_condition],
	goodsN_auge_data['irac_ch3err'][goodsN_auge_condition],
	goodsN_auge_data['irac_ch4err'][goodsN_auge_condition],
	goodsN_auge_data['f24err'][goodsN_auge_condition],
    goodsN_auge_data['f70err'][goodsN_auge_condition],
	goodsN_auge_data['f100err'][goodsN_auge_condition],
	goodsN_auge_data['f160err'][goodsN_auge_condition],
	goodsN_auge_data['f250err'][goodsN_auge_condition],
	goodsN_auge_data['f350err'][goodsN_auge_condition],
	goodsN_auge_data['f500err'][goodsN_auge_condition]
]) 


goodsN_flux_array_auge = goodsN_flux_array_auge.T
goodsN_flux_err_array_auge = goodsN_flux_err_array_auge.T
###################################################################################

###################################################################################
###################################################################################
############################## Read in GOODS-S files ##############################
goodsS_auge = fits.open(path+'GOODsS_full_cat_update.fits')
goodsS_auge_data = goodsS_auge[1].data
goodsS_auge.close()

goodsS_auge_ID = goodsS_auge_data['id_xray']
goodsS_auge_phot_ID = goodsS_auge_data['id_rainbow']
goodsS_auge_RA = goodsS_auge_data['xRA']
goodsS_auge_DEC = goodsS_auge_data['xDEC']
goodsS_auge_Lx = goodsS_auge_data['Lxc']
goodsS_auge_Lx_hard = goodsS_auge_data['Lxc']*0.611
goodsS_auge_z = goodsS_auge_data['z_spec']
goodsS_auge_Nh = goodsS_auge_data['Nh']
goodsS_auge_Nh_lo = goodsS_auge_data['Nh_lo']
goodsS_auge_Nh_hi = goodsS_auge_data['Nh_hi']

goodsS_auge_condition = (np.log10(goodsS_auge_Lx) >= Lx_min) & (np.log10(goodsS_auge_Lx) <= Lx_max) &(goodsS_auge_z > z_min) & (goodsS_auge_z <= z_max) & (goodsS_auge_z != 0.0)

goodsS_auge_ID_match = goodsS_auge_ID[goodsS_auge_condition]
goodsS_auge_phot_ID_match = goodsS_auge_phot_ID[goodsS_auge_condition]
goodsS_auge_RA_match = goodsS_auge_RA[goodsS_auge_condition]
goodsS_auge_DEC_match = goodsS_auge_DEC[goodsS_auge_condition]
goodsS_auge_Lx_match = goodsS_auge_Lx[goodsS_auge_condition]
goodsS_auge_Lx_hard_match = goodsS_auge_Lx_hard[goodsS_auge_condition]
goodsS_auge_z_match = goodsS_auge_z[goodsS_auge_condition]
goodsS_auge_Nh_match = goodsS_auge_Nh[goodsS_auge_condition]
goodsS_auge_Nh_lo_match = goodsS_auge_Nh_lo[goodsS_auge_condition]
goodsS_auge_Nh_hi_match = goodsS_auge_Nh_hi[goodsS_auge_condition]

goodsS_Nh_check = []
for i in range(len(goodsS_auge_Nh_match)):
    if goodsS_auge_Nh_match[i] <= 0.0:
        if goodsS_auge_Nh_lo_match[i] <= 0.0:
            goodsS_auge_Nh_match[i] = goodsS_auge_Nh_hi_match[i]
            goodsS_Nh_check.append(1)
        else:
            goodsS_auge_Nh_match[i] = goodsS_auge_Nh_lo_match[i]
            goodsS_Nh_check.append(2)
    else:
        goodsS_Nh_check.append(0)

print('GOODS-S 2 match: ',len(goodsS_auge_ID_match))

goodsS_auge_Fx_hard_match_mjy = goodsS_auge_data['Fx_hard'][goodsS_auge_condition]*4.136E8/(10-2)
goodsS_auge_Fx_soft_match_mjy = goodsS_auge_data['Fx_soft'][goodsS_auge_condition]*4.136E8/(2-0.5)

goodsS_nan_array = np.zeros(np.shape(goodsS_auge_ID_match)) # Create nan array with length == to the number of sources to be input to the photometry array
goodsS_nan_array[goodsS_nan_array == 0] = np.nan


goodsS_flux_array_auge = np.asarray([goodsS_auge_Fx_hard_match_mjy*1000, goodsS_auge_Fx_soft_match_mjy*1000,
    goodsS_nan_array,
    goodsS_auge_data['FUV'][goodsS_auge_condition],
    goodsS_auge_data['NUV'][goodsS_auge_condition],
    goodsS_auge_data['U'][goodsS_auge_condition],
    # goodsS_auge_data['IA427'][goodsS_auge_condition],
    goodsS_auge_data['F435W'][goodsS_auge_condition],
    # goodsS_auge_data['IA445'][goodsS_auge_condition],
    goodsS_auge_data['B'][goodsS_auge_condition],
    # goodsS_auge_data['IA464'][goodsS_auge_condition],
    # goodsS_auge_data['IA484'][goodsS_auge_condition],
    # goodsS_auge_data['IA505'][goodsS_auge_condition],
    # goodsS_auge_data['IA527'][goodsS_auge_condition],
    goodsS_auge_data['V'][goodsS_auge_condition],
    # goodsS_auge_data['IA550'][goodsS_auge_condition],
    # goodsS_auge_data['IA574'][goodsS_auge_condition],
    goodsS_auge_data['F606W'][goodsS_auge_condition],
    # goodsS_auge_data['IA598'][goodsS_auge_condition],
    # goodsS_auge_data['IA624'][goodsS_auge_condition],
    goodsS_auge_data['R'][goodsS_auge_condition],
    # goodsS_auge_data['IA651'][goodsS_auge_condition],
    # goodsS_auge_data['IA679'][goodsS_auge_condition],
    # goodsS_auge_data['IA709'][goodsS_auge_condition],
    # goodsS_auge_data['IA738'][goodsS_auge_condition],
    goodsS_auge_data['I'][goodsS_auge_condition],
    goodsS_auge_data['F775W'][goodsS_auge_condition],
    # goodsS_auge_data['IA767'][goodsS_auge_condition],
    # goodsS_auge_data['IA797'][goodsS_auge_condition],
    goodsS_auge_data['F814W'][goodsS_auge_condition],
    # goodsS_auge_data['IA827'][goodsS_auge_condition],
    # goodsS_auge_data['IA856'][goodsS_auge_condition],
    goodsS_auge_data['z'][goodsS_auge_condition],
    goodsS_auge_data['F850LP'][goodsS_auge_condition],
    goodsS_auge_data['F098M'][goodsS_auge_condition],
    goodsS_auge_data['F105W'][goodsS_auge_condition],
    goodsS_auge_data['F125W'][goodsS_auge_condition],
    goodsS_auge_data['J'][goodsS_auge_condition],
    goodsS_auge_data['F140W'][goodsS_auge_condition],
    goodsS_auge_data['F160W'][goodsS_auge_condition],
    goodsS_auge_data['H'][goodsS_auge_condition],
    goodsS_auge_data['K'][goodsS_auge_condition],
    goodsS_auge_data['irac_ch1'][goodsS_auge_condition],
    goodsS_auge_data['irac_ch2'][goodsS_auge_condition],
    goodsS_auge_data['irac_ch3'][goodsS_auge_condition],
    goodsS_auge_data['irac_ch4'][goodsS_auge_condition],
    goodsS_auge_data['F24'][goodsS_auge_condition],
    goodsS_auge_data['F70'][goodsS_auge_condition],
    goodsS_auge_data['F100'][goodsS_auge_condition],
    goodsS_auge_data['F160'][goodsS_auge_condition],
    goodsS_auge_data['F250'][goodsS_auge_condition],
    goodsS_auge_data['F350'][goodsS_auge_condition],
    goodsS_auge_data['F500'][goodsS_auge_condition],
    ])

goodsS_flux_err_array_auge = np.asarray([goodsS_auge_Fx_hard_match_mjy*1000*0.2, goodsS_auge_Fx_soft_match_mjy*1000*0.2,
    goodsS_nan_array,
    goodsS_auge_data['FUVerr'][goodsS_auge_condition],
    goodsS_auge_data['NUVerr'][goodsS_auge_condition],
    goodsS_auge_data['Uerr'][goodsS_auge_condition],
    # goodsS_auge_data['IA427err'][goodsS_auge_condition],
    goodsS_auge_data['F435Werr'][goodsS_auge_condition],
    # goodsS_auge_data['IA445err'][goodsS_auge_condition],
    goodsS_auge_data['Berr'][goodsS_auge_condition],
    # goodsS_auge_data['IA464err'][goodsS_auge_condition],
    # goodsS_auge_data['IA484err'][goodsS_auge_condition],
    # goodsS_auge_data['IA505err'][goodsS_auge_condition],
    # goodsS_auge_data['IA527err'][goodsS_auge_condition],
    goodsS_auge_data['Verr'][goodsS_auge_condition],
    # goodsS_auge_data['IA550err'][goodsS_auge_condition],
    # goodsS_auge_data['IA574err'][goodsS_auge_condition],
    goodsS_auge_data['F606Werr'][goodsS_auge_condition],
    # goodsS_auge_data['IA598err'][goodsS_auge_condition],
    # goodsS_auge_data['IA624err'][goodsS_auge_condition],
    goodsS_auge_data['Rerr'][goodsS_auge_condition],
    # goodsS_auge_data['IA651err'][goodsS_auge_condition],
    # goodsS_auge_data['IA679err'][goodsS_auge_condition],
    # goodsS_auge_data['IA709err'][goodsS_auge_condition],
    # goodsS_auge_data['IA738err'][goodsS_auge_condition],
    goodsS_auge_data['Ierr'][goodsS_auge_condition],
    goodsS_auge_data['F775Werr'][goodsS_auge_condition],
    # goodsS_auge_data['IA767err'][goodsS_auge_condition],
    # goodsS_auge_data['IA797err'][goodsS_auge_condition],
    goodsS_auge_data['F814Werr'][goodsS_auge_condition],
    # goodsS_auge_data['IA827err'][goodsS_auge_condition],
    # goodsS_auge_data['IA856err'][goodsS_auge_condition],
    goodsS_auge_data['zerr'][goodsS_auge_condition],
    goodsS_auge_data['F850LPerr'][goodsS_auge_condition],
    goodsS_auge_data['F098Merr'][goodsS_auge_condition],
    goodsS_auge_data['F105Werr'][goodsS_auge_condition],
    goodsS_auge_data['F125Werr'][goodsS_auge_condition],
    goodsS_auge_data['Jerr'][goodsS_auge_condition],
    goodsS_auge_data['F140Werr'][goodsS_auge_condition],
    goodsS_auge_data['F160Werr'][goodsS_auge_condition],
    goodsS_auge_data['Herr'][goodsS_auge_condition],
    goodsS_auge_data['Kerr'][goodsS_auge_condition],
    goodsS_auge_data['irac_ch1err'][goodsS_auge_condition],
    goodsS_auge_data['irac_ch2err'][goodsS_auge_condition],
    goodsS_auge_data['irac_ch3err'][goodsS_auge_condition], 
    goodsS_auge_data['irac_ch4err'][goodsS_auge_condition],
    goodsS_auge_data['F24err'][goodsS_auge_condition],
    goodsS_auge_data['F70err'][goodsS_auge_condition],
    goodsS_auge_data['F100err'][goodsS_auge_condition],
    goodsS_auge_data['F160err'][goodsS_auge_condition],
    goodsS_auge_data['F250err'][goodsS_auge_condition],
    goodsS_auge_data['F350err'][goodsS_auge_condition],
    goodsS_auge_data['F500err'][goodsS_auge_condition]
    ])


goodsS_flux_array_auge = goodsS_flux_array_auge.T
goodsS_flux_err_array_auge = goodsS_flux_err_array_auge.T

######## Figure 1 ########
'''
plt.rcParams['font.size'] = 24
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.figure(figsize=(10,10))
# ax1 = plt.subplot(111,)
plt.plot(goodsS_auge_z_match,np.log10(goodsS_auge_Lx_match),'.',ms=10,color='gray',rasterized=True)
plt.plot(goodsN_auge_z_match,np.log10(goodsN_auge_Lx_match),'.',ms=10,color='gray',rasterized=True,label='GOODS-N/S')
# plt.plot(xue_z_match,np.log10(xue_Lx_match),'.',color='gray',rasterized=True)
# plt.plot(luo_z_match,np.log10(luo_Lx_match),'.',color='gray',rasterized=True,label='GOODS-N/S')
plt.plot(chandra_cosmos_z_match,np.log10(chandra_cosmos_Lx_full_match),'+',ms=10,color='b',rasterized=True,alpha=0.8,label='COSMOS')
# plt.plot(s82x_z_sp_match,np.log10(s82x_≥Lx_sp_full_match),'x',ms=10,color='r',rasterized=True,alpha=0.8,label='Stripe82X')
plt.plot(s82x_z,np.log10(s82x_Lx_full),'x',ms=10,color='r',rasterized=True,alpha=0.8,label='Stripe82X')
# plt.plot(ulirg_z,ulirg_Lx,'*',color='g',ms=12,rasterized=True,label='GOALS')
plt.xlabel('Spectroscopic Redshift')
plt.ylabel(r'log$_{10}$ L$_{0.5 - 10\mathrm{keV}}$ [erg s$^{-1}$]')
# plt.text(2.15, 40.55, f'n = {len(goodsS_auge_z_match)+len(goodsN_auge_z_match)+len(xue_z_match)+len(luo_z_match)+len(chandra_cosmos_z_match)+len(s82x_z_sp)+len(ulirg_z)}')
plt.text(3.25, 40.55, f'n = {len(goodsS_auge_z_match)+len(goodsN_auge_z_match)+len(chandra_cosmos_z_match)+len(s82x_z)}')
# plt.text(3.25, 40.55, f'n = {len(xue_z_match)+len(luo_z_match)+len(chandra_cosmos_z_match)+len(s82x_z_sp)}')
plt.legend()
# plt.axvline(1.2,color='k',ls='--',lw=3)
plt.xlim(-0.05,1.25)
plt.grid()
plt.tight_layout()
plt.savefig('/Users/connor_auge/Desktop/Final_plots/a_new/Lx_z_spec3.pdf')
plt.show()
'''
####################


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

scale_array = [1.12E44, 1.72E44, 2.27E44]
###################################

#### Read in M82 Templates ####
m82 = ascii.read('/Users/connor_auge/Research/templets/M82.csv')
m82_wave = np.asarray(m82['wave'])
m82_lum = np.asarray(m82['lum'])
###################################



# Print time taken to read in all files
tfl = time.perf_counter()
print(f'Done with file reading ({tfl - ti:0.4f} second)')


# Filters used in COSMOS field 
COSMOS_filters = np.array(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'G', 'R', 'I', 'Z', 'yHSC_FLUX_APER2', 'JVHS', 'H_FLUX_APER2',
                          'Ks_FLUX_APER2', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])

COSMOS_CIGALE_filters = np.array(['Fx_hard', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'G', 'R', 'I', 'Z', 'yHSC_FLUX_APER2', 'JVHS', 'H_FLUX_APER2',
                                  'Ks_FLUX_APER2', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])

# COSMOS_filters = np.array(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'G', 'R', 'I', 'Z', 'yHSC_FLUX_APER2', 'J_FLUX_APER2', 'H_FLUX_APER2',
#                           'Ks_FLUX_APER2', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500',
#                            'SCUBA2', 'VLA2'])
# Filters used in the S82X field
S82X_filters = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'MAG_FUV', 'MAG_NUV', 'U', 'G', 'R', 'I', 'Z',
                          'JVHS', 'HVHS', 'KVHS', 'W1', 'W2', 'W3', 'W4', 'nan', 'FLUX_250_s82x', 'FLUX_350_s82x', 'FLUX_500_s82x'])

# Filters used in the GOODS-N/S field
GOODSS_auge_filters = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'F435W', 'B_FLUX_APER2', 'V_FLUX_APER2', 'F606W', 'R', 'I',
                                  'F775W', 'F814W', 'Z', 'F850LP', 'F098M', 'F105W', 'F125W', 'JVHS', 'F140W', 'F160W', 'HVHS', 'KVHS', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'MIPS2', 'FLUX_100_goodsS', 'FLUX_160_goodsS', 'FLUX_250_goodsS', 'FLUX_350_goodsS', 'FLUX_500_goodsS'])

GOODSN_auge_filters = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'F435W', 'B_FLUX_APER2', 'V_FLUX_APER2', 'F606W', 'R', 'I', 'F775W', 'F814W', 'Z', 'F105W', 'F125W', 'JVHS', 'F140W', 'F160W', 'HVHS', 'KVHS',
                                  'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'MIPS2', 'FLUX_100_goodsN', 'FLUX_160_goodsN', 'FLUX_250_goodsN', 'FLUX_350_goodsN', 'FLUX_500_goodsN'])

# Filters for writting the output file
filter_total = np.array(['Fxh','Fxs','FUV','NUV','u','g','r','i','z','y','J','H','Ks','W1','IRAC1','IRAC2','W2','IRAC3','IRAC4','W3','W4','F24','F250','F350','F500'])
filter_COSMOS_match = np.array(['Fxh','Fxs','nan','FUV','NUV','u','g','r','i','z','y','J','H','Ks','IRAC1','IRAC2','IRAC3','IRAC4','F24','F250','F350','F500'])
filter_S82X_match = np.array(['Fxh','Fxs','nan','FUV','NUV','u','g','r','i','z','J','H','Ks','W1','W2','W3','W4','nan','F250','F350','F500'])
filter_GOODSN_match = np.asarray(['Fxh','Fxs','nan','FUV','NUV','I','F435W','B','V','F606W','R','I','F775W','F814W', 'z', 'F105W', 'F125W', 'J', 'F140W', 'F160W', 'H', 'Ks','IRAC1','IRAC2','IRAC3','IRAC4','F24','F70','F100','F160','F250','F350','F500'])
filter_GOODSS_match = np.asarray(['Fxh','Fxs','nan','FUV','NUV','I','F435W','B','V','F606W','R','I','F775W','F814W', 'z', 'F850LP', 'F098M','F105W', 'F125W', 'J', 'F140W', 'F160W', 'H', 'Ks','IRAC1','IRAC2','IRAC3','IRAC4','F24','F70','F100','F160','F250','F350','F500'])

filter_COSMOS_total = np.array(['Fxh','Fxs','FUV','NUV','u','g','r','i','z','y','J','H','Ks','IRAC1','IRAC2','IRAC3','IRAC4','F24','F250','F350','F500'])
filter_S82X_total = np.array(['Fxh','Fxs','FUV','NUV','u','g','r','i','z','J','H','Ks','W1','W2','W3','W4','F250','F350','F500'])
filter_GOODSN_total = np.asarray(['Fxh','Fxs','FUV','NUV','U','F435W','B','V','F606W','R','I','F775W','F814W', 'z', 'F105W', 'F125W', 'J', 'F140W', 'F160W', 'H', 'Ks','IRAC1','IRAC2','IRAC3','IRAC4','F24','F70','F100','F160','F250','F350','F500'])
filter_GOODSS_total = np.asarray(['Fxh','Fxs','FUV','NUV','U','F435W','B','V','F606W','R','I','F775W','F814W', 'z', 'F850LP', 'F098M','F105W', 'F125W', 'J', 'F140W', 'F160W', 'H', 'Ks','IRAC1','IRAC2','IRAC3','IRAC4','F24','F70','F100','F160','F250','F350','F500'])
filter_GOODS_total  = np.asarray(['Fxh','Fxs','FUV','NUV','U','F435W','B','V','F606W','R','I','F775W','F814W', 'z', 'F850LP', 'F098M','F105W', 'F125W', 'J', 'F140W', 'F160W', 'H', 'Ks','IRAC1','IRAC2','IRAC3','IRAC4','F24','F70','F100','F160','F250','F350','F500'])
###############################################################################
############################## Start CIGALE File ##############################
cigale_name = 'AHA_sample_FINAL2/AHA_COSMOS_shape5_3.mag'
# cigale_name = 'test_FIR_cosmos3.mag'
inf = open(f'../xcigale/data_input/{cigale_name}', 'w')
header = np.asarray(['# id', 'redshift'])
cigale_filters = Filters('filter_list.dat').pull_filter(COSMOS_CIGALE_filters, 'xcigale name')
for i in range(len(cigale_filters)):
    header = np.append(header, cigale_filters[i])
    header = np.append(header, cigale_filters[i]+'_err')
np.savetxt(inf, header, fmt='%s', delimiter='    ', newline=' ')
inf.close()
###############################################################################


###############################################################################
# ############################# Start Check SED File #############################
# check_sed_fname = 'COSMOS_SED_Check'
# with open(f'/Users/connor_auge/Desktop/sed_check_output/{check_sed_fname}.txt', 'w') as outf:
#     outf.writelines('ID,Bad_SED,UV_extrap,F1_extrap,MIR_extrap,FIR_extrap,Bad_FIR,Manual_Check\n')
###############################################################################


# Slope lims
uv1, uv2 = 0.2, 1.0
mir11, mir12 = 1.0, 6.0
mir21, mir22 = 6.0, 10


###################################################################################
###################################################################################
###################################################################################
########################## Run AGN Class over each source #########################

# Make empty lists to be filled with SED outputs 
out_ID, out_z, out_x, out_y, out_frac_error = [], [], [], [], []
out_Lx, out_Lx_hard, out_Lx_soft = [], [], []
out_SED_shape = []
check_sed = []
norm = []
wfir_out, ffir_out = [], []
int_x, int_y = [],[]
FIR_upper_lims = []
F025, F1, F6, F10, F100 = [], [], [], [], []
F2 = []
xval_out, F2kev = [], []
field = []
uv_slope, mir_slope1, mir_slope2 = [], [], []
Lbol_out, Lbol_sub_out = [], []
Nh = []
UV_lum_out, OPT_lum_out, MIR_lum_out, FIR_lum_out = [], [], [], []
X_UV_lum_out = []
FIR_R_lum = []
Nh_check = []
abs_check = []
mix_x, mix_y = [], []
F025_boot, F1_boot, F6_boot, F10_boot, F100_boot = [], [], [], [], []
F2_boot = []
xval_out_boot, F2kev_boot = [], []
spec_class = []
check_sed6 = []

F24_out = []
F100_ratio =[]
stack_bin = []

Lbol_sf_sub_out = []
###############################################################################
###############################################################################
############################### Run COSMOS SEDs ###############################
# '''
nustar_id = np.asarray(
    [215929, 317570, 338071, 348646, 397900, 420912, 450055, 452180, 486971, 487826, 494577, 508074, 529855, 567452, 572088, 589540, 609017, 635561, 662467, 859808, 919147])

fill_nan = np.zeros(len(GOODSS_auge_filters)-len(COSMOS_filters))
fill_nan[fill_nan == 0] = np.nan
cigale_count = 0
for i in range(len(chandra_cosmos_phot_id_match)):
# for i in range(50):
    # if chandra_cosmos_phot_id_match[i] in nustar_id:
        source = AGN(chandra_cosmos_phot_id_match[i], chandra_cosmos_z_match[i], COSMOS_filters, cosmos_flux_array[i], cosmos_flux_err_array[i])
        source.MakeSED(data_replace_filt=['FLUX_24'])
        source.FIR_extrap(['FLUX_24', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])

        cosmos_flux_dict = source.MakeDict(COSMOS_filters,cosmos_flux_array[i])
        cosmos_flux_err_dict = source.MakeDict(COSMOS_filters,cosmos_flux_err_array[i])
        
        Lcheck = source.Lum_filter('FLUX_24')
        F24_out.append(Lcheck)
        stack_bin.append(7)

        ix, iy = source.Int_SED(xmin=1E-1, xmax=1E1)
        int_x.append(ix)
        int_y.append(iy)

        # wfir, ffir, f100, f100_boot = source.Int_SED_FIR(Find_value=100.0,discreet=True,boot=True)
        wfir, ffir, f100 = source.Int_SED_FIR(Find_value=100.0,discreet=True)
        wfir_out.append(wfir)
        ffir_out.append(ffir)

        L100_ratio = source.flux_ratio_lower()
        F100_ratio.append(L100_ratio)

        # f1, f1_boot = source.Find_value(1.0, boot=True)
        # f2, f2_boot = source.Find_value(2.0, boot=True)
        # xval, xval_boot = source.Find_value(3E-4, boot=True)
        # f6, f6_boot = source.Find_value(6.0, boot=True)
        # f025, f025_boot = source.Find_value(0.25, boot=True)
        # f10, f10_boot = source.Find_value(10, boot=True)
        # f2kev, f2kev_boot = source.Find_value(6.1992e-4, boot=True)

        # f015, f015_boot = source.Find_value(0.15, boot=True)
        # f65, f65_boot = source.Find_value(6.5, boot=True)

        f1 = source.Find_value(1.0)
        f2 = source.Find_value(2.0)
        xval = source.Find_value(3E-4)
        f6 = source.Find_value(6.0)
        f025 = source.Find_value(0.25)
        f10 = source.Find_value(10)
        f2kev = source.Find_value(6.1992e-4)

        f015 = source.Find_value(0.2)
        f65 = source.Find_value(6.5)

        lbol = source.Find_Lbol()
        lbol_sub = source.Find_Lbol(sub=True,Lscale=scale_array, Lnorm=f1, temp_x=temp_wave, temp_y=temp_lum)
        lbol_sf_sub = source.Find_Lbol(sub=True,Lscale=scale_array, Lnorm=f1, temp_x=m82_wave, temp_y=m82_lum)
        # lbol_sub = source.Find_Lbol_temp_sub(scale_array, f1, temp_wave, temp_lum)
        # lbol_sf_sub = source.Find_Lbol_temp_sub(scale_array, f1, m82_wave, m82_lum)
        # print(lbol,lbol_sub,lsbol_sf_sub)
        # lbol_sf_sub = source.Find_Lbol(xmax=30)
        # lbol_sf_sub, tx, ty, xsub, ysub = source.Find_Lbol_temp_sub(scale_array, f1, m82_wave, m82_lum,sed=True)
        # print(np.log10(lbol),np.log10(lbol_sub),np.log10(lbol_sf_sub))
        # lbol_sub = source.Find_Lbol_temp_sub(scale_array, f1, temp_wave, temp_lum, xmax=50)
        shape = source.SED_shape()

        Id, redshift, w, f, frac_err, up_check = source.pull_plot_info(norm_w=1)
        w = np.append(w, fill_nan)
        f = np.append(f, fill_nan)
        frac_err = np.append(frac_err, fill_nan)
        out_ID.append(Id)
        out_x.append(w)
        out_y.append(f)
        out_frac_error.append(frac_err)
        out_Lx.append(chandra_cosmos_Lx_full_match[i])
        out_Lx_hard.append(chandra_cosmos_Lx_hard_match[i])
        out_Lx_soft.append(chandra_cosmos_Lx_soft_match[i])
        out_z.append(chandra_cosmos_z_match[i])
        FIR_upper_lims.append(up_check)

        hao_x, hao_y = source.mix_loc([1,0.3],[3,1])
        # hao_x, hao_y = source.mix_loc([0.3, 1.0], [1.0, 3.0])
        mix_x.append(hao_x)
        mix_y.append(hao_y)

        norm.append(f1)
        F025.append(f025)
        F1.append(f1)
        F2.append(f2)
        F6.append(f6)
        F10.append(f10)
        F100.append(f100)
        F2kev.append(f2kev)
        xval_out.append(xval)
        # F1_boot.append(f1_boot)
        # F025_boot.append(f025_boot)
        # F2_boot.append(f2_boot)
        # F6_boot.append(f6_boot)
        # F10_boot.append(f10_boot)
        # xval_out_boot.append(xval_boot)
        # F100_boot.append(f100_boot)
        # F2kev_boot.append(f2kev_boot)
        Lbol_out.append(lbol)
        Lbol_sub_out.append(lbol_sub)
        Lbol_sf_sub_out.append(lbol_sf_sub)
        Nh.append(chandra_cosmos_Nh_match[i])
        Nh_check.append(cosmos_Nh_check[i])
        abs_check.append(check_abs_match[i])

        shape = source.SED_shape()
        # shape = source.SED_shape(uv1, uv2, mir11, mir12, mir21, mir22)

        uv_slope.append(source.Find_slope(uv1, uv2))
        mir_slope1.append(source.Find_slope(mir11, mir12))
        mir_slope2.append(source.Find_slope(mir21, mir22))
        out_SED_shape.append(shape)
        UV_lum_out.append(source.find_Lum_range(0.1,0.35))
        OPT_lum_out.append(source.find_Lum_range(0.35,3))
        MIR_lum_out.append(source.find_Lum_range(3,30))
        FIR_lum_out.append(source.find_Lum_range(30,500/(1+chandra_cosmos_z_match[i])))
        X_UV_lum_out.append(source.find_Lum_range(1E-3,0.1))
        FIR_R_lum.append(source.find_Lum_range(40,400))

        plot = Plotter(Id, redshift, w, f, chandra_cosmos_Lx_full_match[i],f1,up_check)

        check = source.check_SED(10, check_span=2.75)
        check6 = source.check_SED(6, check_span=2.75)
        check_sed.append(check)
        check_sed6.append(check6)
        field.append('COSMOS')
        spec_class.append(chandra_cosmos_spec_type_match[i])

        # if check6 == 'GOOD':
        # # # # # #     print(shape, source.Find_slope(uv1,uv2), source.Find_slope(mir11, mir12), source.Find_slope(mir21, mir22))
        # # #     if shape == 3:
        # # print(Id,check6,shape,up_check,source.Find_slope(uv1, uv2),source.Find_slope(mir11, mir12),mir_slope2[i])
        #     plot.PlotSED(point_x=[0.2,1.0,6.,10,100],point_y=[f015/f1,f1/f1,f6/f1,f10/f1,f100/f1],fir_x = wfir, fir_y = ffir,temp_x=tx,temp_y=ty)


        # if check6 == 'GOOD':
        #     c = SkyCoord(ra = chandra_cosmos_RA_match[i]*u.degree, dec = chandra_cosmos_DEC_match[i]*u.degree)
        #     coords = c.to_string('hmsdms').split()

        #     cols, data, dtyp = source.output_properties('COSMOS',chandra_cosmos_xid_match[i],coords[0],coords[1],chandra_cosmos_Lx_full_match[i],chandra_cosmos_Nh_match[i])
        #     source.write_output_file('AGN_Properties7.fits',data,cols,dtyp,'w')

        #     cols, data, dtyp = source.output_phot('COSMOS',filter_COSMOS_total,filter_COSMOS_match)
        #     source.write_output_file('AGN_photometry_cosmos7.fits',data,cols,dtyp,'w')

        # if Id == 847966:


        # if check == 'GOOD':
        #     int_plot = IntPlot(check_sed_fname,Id, redshift, w, f, chandra_cosmos_Lx_full_match[i],f1,up_check)
        #     int_plot.Plot([0.25,1.0,6.0,100],[f025/f1,f1/f1,f6/f1,f100/f1])
        
        
        # # if check == 'GOOD':
        #     # plot.Plot_FIR_SED(wfir, ffir/f1)
            # print(chandra_cosmos_phot_id_match[i])
            # print('shape: ', shape)
            # print('uv: ', source.Find_slope(0.15, 1.0))
            # print('mir1: ', source.Find_slope(1.0, 6.5))
            # print('mir2: ', source.Find_slope(6.5, 10))
            # plot.PlotSED(point_x=100,point_y=f100/f1)
        
        # elif Id == 662467:
        # if check == 'GOOD':
        # if Id == 687415:
            # plot.Plot_FIR_SED(wfir, ffir/f1)
        #     source.write_cigale_file(cigale_name,int_fx=cosmos_Fx_int_array[i],use_int_fx=True)
            # plot.PlotSED(point_x=100,point_y=f100/f1)
            
        # if Id == 536989 or Id == 576750:
        #     # if check == 'GOOD':
        #     # plot.Plot_FIR_SED(wfir, ffir/f1)
        #     plot.PlotSED(point_x=100, point_y=f100/f1)
            # source.Find_Lbol()

        if check6 == 'GOOD':
            if shape == 5:
                cigale_count += 1
                if (cigale_count > 60) & (cigale_count <= 90):
                # if cigale_count <= 30:
                    source.write_cigale_file2(cigale_name, COSMOS_CIGALE_filters, cosmos_flux_dict, cosmos_flux_err_dict, int_fx=cosmos_Fx_int_array[0][i])
                    # source.write_cigale_file(cigale_name,int_fx=cosmos_Fx_int_array[i],use_int_fx=True)
                else:
                    continue
            else:
                continue
        else:
            continue

        # if check6 == 'GOOD':
        #     if shape == 4:
        #         cigale_count += 1
        #         if (cigale_count > 150) & (cigale_count <= 180):
                # if cigale_count <= 30:
                    # source.write_cigale_file(cigale_name,COSMOS_filters)
                    # source.write_cigale_file(cigale_name,int_fx=cosmos_Fx_int_array[i],use_int_fx=True)
        #         else:
        #             continue
        #     else:
        #         continue
        # else:
            # continue

    # else:
        # continue    
    

# '''
tc = time.perf_counter()
print(f'Done with COSMOS sources ({tc - tfl:0.4f} second)')


###############################################################################
###############################################################################
############################## Run Stripe82X SEDs #############################
# '''
# Make array of NaNs to fill in the output wave and Lum arrays so output is consistent shape
fill_nan = np.zeros(len(GOODSS_auge_filters)-len(S82X_filters)) 
fill_nan[fill_nan == 0] = np.nan
# bad_source = np.asarray([2585,2685,2748,2774,2795,2931,2964,3129,3201,3296,3451,3522,3531,3851,4158,4349,4435,4453,4490,4728,4857,4872,15292,105703])
bad_source = np.asarray([2685,2748,2774,2795,2931,2964,3129,3201,3451,3522,3531,4158,4349,4435,4453,4490,4857,4872,15292,105703])

for i in range(len(s82x_id)):
    # if s82x_id[i] == 15292:
    if s82x_id[i] in bad_source:
        continue
    else:
        try:
            source = AGN(s82x_id[i],s82x_z[i],S82X_filters,s82x_flux_array[i],s82x_flux_err_array[i])
            source.MakeSED(data_replace_filt=['W4'])
            source.FIR_extrap(['W4','FLUX_250_s82x', 'FLUX_350_s82x', 'FLUX_500_s82x'],stack=True)
            # stack_out = source.stack_check()
            # print(stack_out)
            stack_bin.append(7)

            Lcheck = source.Lum_filter('W4')
            F24_out.append(Lcheck)

            ix, iy = source.Int_SED(xmin=1E-1, xmax=1E1)
            int_x.append(ix)
            int_y.append(iy)

            # wfir, ffir, f100, f100_boot = source.Int_SED_FIR(Find_value=100.0,discreet=True,boot=True)
            wfir, ffir, f100 = source.Int_SED_FIR(Find_value=100.0,discreet=True)
            wfir_out.append(wfir)
            ffir_out.append(ffir)

            L100_ratio = source.flux_ratio_lower()
            F100_ratio.append(L100_ratio)

            # f1, f1_boot = source.Find_value(1.0, boot=True)
            # f2, f2_boot = source.Find_value(2.0, boot=True)
            # xval, xval_boot = source.Find_value(3E-4, boot=True)
            # f6, f6_boot = source.Find_value(6.0, boot=True)
            # f025, f025_boot = source.Find_value(0.25, boot=True)
            # f10, f10_boot = source.Find_value(10, boot=True)
            # f2kev, f2kev_boot = source.Find_value(6.1992e-4, boot=True)

            # f015, f015_boot = source.Find_value(0.15, boot=True)
            # f65, f65_boot = source.Find_value(6.5, boot=True)

            f1 = source.Find_value(1.0)
            f2 = source.Find_value(2.0)
            xval = source.Find_value(3E-4)
            f6 = source.Find_value(6.0)
            f025 = source.Find_value(0.25)
            f10 = source.Find_value(10)
            f2kev = source.Find_value(6.1992e-4)

            f015 = source.Find_value(0.2)
            f65 = source.Find_value(6.5)

            hao_x, hao_y = source.mix_loc([1,0.3],[3,1])
            mix_x.append(hao_x)
            mix_y.append(hao_y)

            lbol = source.Find_Lbol()
            lbol_sub = source.Find_Lbol(sub=True,Lscale=scale_array, Lnorm=f1, temp_x=temp_wave, temp_y=temp_lum)
            lbol_sf_sub = source.Find_Lbol(sub=True,Lscale=scale_array, Lnorm=f1, temp_x=m82_wave, temp_y=m82_lum)
            # lbol_sub = source.Find_Lbol_temp_sub(scale_array, f1, temp_wave, temp_lum)
            # lbol_sf_sub = source.Find_Lbol_temp_sub(scale_array, f1, m82_wave, m82_lum)
            # lbol_sf_sub = source.Find_Lbol(xmax=30)
            # lbol_sub = source.Find_Lbol_temp_sub(scale_array, f1, temp_wave, temp_lum, xmax=50)
            shape = source.SED_shape()

            Id, redshift, w, f, frac_err, up_check = source.pull_plot_info(norm_w=1)
            w = np.append(w,fill_nan)
            f = np.append(f,fill_nan)
            frac_err = np.append(frac_err,fill_nan)
            out_ID.append(Id)
            out_x.append(w)
            out_y.append(f)
            out_frac_error.append(frac_err)
            out_Lx.append(s82x_Lx_full[i])
            out_Lx_hard.append(s82x_Lx_hard[i])
            out_z.append(s82x_z[i])
            FIR_upper_lims.append(up_check)

            norm.append(f1)
            F025.append(f025)
            F1.append(f1)
            F2.append(f2)
            F6.append(f6)
            F10.append(f10)
            F100.append(f100)
            xval_out.append(xval)
            F2kev.append(f2kev)
            # F1_boot.append(f1_boot)
            # F025_boot.append(f025_boot)
            # F2_boot.append(f2_boot)
            # F6_boot.append(f6_boot)
            # F10_boot.append(f10_boot)
            # xval_out_boot.append(xval_boot)
            # F100_boot.append(f100_boot)
            # F2kev_boot.append(f2kev_boot)
            Lbol_out.append(lbol)
            Lbol_sub_out.append(lbol_sub)
            Lbol_sf_sub_out.append(lbol_sf_sub)
            Nh.append(s82x_Nh[i])
            Nh_check.append(s82x_Nh_check[i])
            abs_check.append(0)

            # shape = source.SED_shape(uv1, uv2, mir11, mir12, mir21, mir22)
            shape = source.SED_shape()

            uv_slope.append(source.Find_slope(uv1, uv2))
            mir_slope1.append(source.Find_slope(mir11, mir12))
            mir_slope2.append(source.Find_slope(mir21, mir22))
            out_SED_shape.append(shape)
            UV_lum_out.append(source.find_Lum_range(0.1,0.35))
            OPT_lum_out.append(source.find_Lum_range(0.35,3))
            MIR_lum_out.append(source.find_Lum_range(3,30))
            FIR_lum_out.append(source.find_Lum_range(30,500/(1+s82x_z[i])))
            X_UV_lum_out.append(source.find_Lum_range(1E-3, 0.1))
            FIR_R_lum.append(source.find_Lum_range(40,400))


            plot = Plotter(Id, redshift, w, f, s82x_Lx_full[i],f1,up_check)
            check = source.check_SED(10, check_span=2.75)
            check6 = source.check_SED(6, check_span=2.75)
            check_sed.append(check)
            check_sed6.append(check6)
            field.append('S82X')
            spec_class.append(s82x_spec_class[i])
           
            # if check6 == 'GOOD':
            # # # #     print(shape, source.Find_slope(uv1,uv2), source.Find_slope(mir11, mir12), source.Find_slope(mir21, mir22))
            #     # if shape == 3:
            #         print(Id,check6,shape,up_check)#source.Find_slope(uv1, uv2),source.Find_slope(mir11, mir12),mir_slope2[i])
            #         plot.PlotSED(point_x=[0.25,1.0,6.5,10,100],point_y=[f025/f1,f1/f1,f6/f1,f10/f1,f100/f1],fir_x = wfir, fir_y = ffir)

            # if Id == 187:
            #     print(source.Find_slope(0.15,1.0))
            #     plot.PlotSED(point_x=100,point_y=f100/f1)
            #     source.write_cigale_file(cigale_name,S82X_filters,int_fx=s82x_Fx_int_array[i],use_int_fx=True)

            # if check6 == 'GOOD':
            #     if shape == 2:
            #         cigale_count += 1
            #         if (cigale_count > 30) & (cigale_count <= 60):
            #         # if cigale_count <= 30:
            #             source.write_cigale_file(cigale_name, int_fx=s82x_Fx_int_array[i], use_int_fx=True)
            #         else:
            #             continue
            #     else:
            #         continue
            # else:
            #     continue

            # if check6 == 'GOOD':
            #     c = SkyCoord(ra = s82x_ra[i]*u.degree, dec = s82x_dec[i]*u.degree)
            #     coords = c.to_string('hmsdms').split()

            #     cols, data, dtyp = source.output_properties('Stripe82X',int(s82x_id[i]),coords[0],coords[1],s82x_Lx_full[i],s82x_Nh[i])
            #     source.write_output_file('AGN_Properties7.fits',data,cols,dtyp,'w')

            #     cols, data, dtyp = source.output_phot('Stripe82X',filter_S82X_total,filter_S82X_match)
            #     source.write_output_file('AGN_photometry_s82x7.fits',data,cols,dtyp,'w')

            
        except ValueError:
            continue
    # else:
    #     continue
# '''
ts = time.perf_counter()
print(f'Done with Stripe82X sources ({ts - tc:0.4f} second)')

###############################################################################
###############################################################################
############################## Run GOODS-N SEDs ###############################
# '''
fill_nan = np.zeros(len(GOODSS_auge_filters)-len(GOODSN_auge_filters))
fill_nan[fill_nan == 0] = np.nan
for i in range(len(goodsN_auge_ID_match)):
    # for i in range(50):
    source = AGN(goodsN_auge_phot_ID_match[i], goodsN_auge_z_match[i], GOODSN_auge_filters, goodsN_flux_array_auge[i], goodsN_flux_err_array_auge[i])
    source.MakeSED(data_replace_filt=['FLUX_24'])
    source.FIR_extrap(['FLUX_24', 'MIPS2', 'FLUX_100_goodsN', 'FLUX_160_goodsN', 'FLUX_250_goodsN', 'FLUX_350_goodsN', 'FLUX_500_goodsN'])

    Lcheck = source.Lum_filter('FLUX_24')
    F24_out.append(Lcheck)
    stack_bin.append(7)

    ix, iy = source.Int_SED(xmin=1E-1, xmax=1E1)
    int_x.append(ix)
    int_y.append(iy)

    # wfir, ffir, f100, f100_boot = source.Int_SED_FIR(Find_value=100.0, discreet=True, boot=True)
    wfir, ffir, f100 = source.Int_SED_FIR(Find_value=100.0, discreet=True)
    wfir_out.append(wfir)
    ffir_out.append(ffir)

    L100_ratio = source.flux_ratio_lower()
    F100_ratio.append(L100_ratio)

    # f1, f1_boot = source.Find_value(1.0, boot=True)
    # f2, f2_boot = source.Find_value(2.0, boot=True)
    # xval, xval_boot = source.Find_value(3E-4, boot=True)
    # f6, f6_boot = source.Find_value(6.0, boot=True)
    # f025, f025_boot = source.Find_value(0.25, boot=True)
    # f10, f10_boot = source.Find_value(10, boot=True)
    # f2kev, f2kev_boot = source.Find_value(6.1992e-4, boot=True)
    # f015, f015_boot = source.Find_value(0.15,boot=True)
    # f65, f65_boot = source.Find_value(6.5,boot=True)

    f1 = source.Find_value(1.0)
    f2 = source.Find_value(2.0)
    xval = source.Find_value(3E-4)
    f6 = source.Find_value(6.0)
    f025 = source.Find_value(0.25)
    f10 = source.Find_value(10)
    f2kev = source.Find_value(6.1992e-4)
    f015 = source.Find_value(0.2)
    f65 = source.Find_value(6.5)


    lbol = source.Find_Lbol()
    lbol_sub = source.Find_Lbol(sub=True,Lscale=scale_array, Lnorm=f1, temp_x=temp_wave, temp_y=temp_lum)
    lbol_sf_sub = source.Find_Lbol(sub=True,Lscale=scale_array, Lnorm=f1, temp_x=m82_wave, temp_y=m82_lum)
    # lbol_sub = source.Find_Lbol_temp_sub(scale_array, f1, temp_wave, temp_lum)
    # lbol_sf_sub = source.Find_Lbol_temp_sub(scale_array, f1, m82_wave, m82_lum)
    # lbol_sf_sub = source.Find_Lbol(xmax=30)
    # lbol_sub = source.Find_Lbol_temp_sub(scale_array, f1, temp_wave, temp_lum, xmax=50)
    shape = source.SED_shape()

    hao_x, hao_y = source.mix_loc([0.3,1],[1,3])
    mix_x.append(hao_x)
    mix_y.append(hao_y)

    Id, redshift, w, f, frac_err, up_check = source.pull_plot_info(norm_w=1)
    w = np.append(w, fill_nan)
    f = np.append(f, fill_nan)
    frac_err = np.append(frac_err, fill_nan)
    out_ID.append(Id)
    out_x.append(w)
    out_y.append(f)
    out_frac_error.append(frac_err)
    out_Lx.append(goodsN_auge_Lx_match[i])
    out_Lx_hard.append(goodsN_auge_Lx_hard_match[i])
    out_z.append(goodsN_auge_z_match[i])
    FIR_upper_lims.append(up_check)

    norm.append(f1)
    F025.append(f025)
    F1.append(f1)
    F2.append(f2)
    F6.append(f6)
    F10.append(f10)
    F100.append(f100)
    xval_out.append(xval)
    F2kev.append(f2kev)
    # F1_boot.append(f1_boot)
    # F025_boot.append(f025_boot)
    # F2_boot.append(f2_boot)
    # F6_boot.append(f6_boot)
    # F10_boot.append(f10_boot)
    # xval_out_boot.append(xval_boot)
    # F100_boot.append(f100_boot)
    # F2kev_boot.append(f2kev_boot)
    Lbol_out.append(lbol)
    Lbol_sub_out.append(lbol_sub)
    Lbol_sf_sub_out.append(lbol_sf_sub)
    Nh.append(goodsN_auge_Nh_match[i])
    Nh_check.append(goodsN_Nh_check[i])
    abs_check.append(0)

    # shape = source.SED_shape(uv1, uv2, mir11, mir12, mir21, mir22)
    shape = source.SED_shape()

    uv_slope.append(source.Find_slope(uv1, uv2))
    mir_slope1.append(source.Find_slope(mir11, mir12))
    mir_slope2.append(source.Find_slope(mir21, mir22))
    out_SED_shape.append(shape)
    UV_lum_out.append(source.find_Lum_range(0.1,0.35))
    OPT_lum_out.append(source.find_Lum_range(0.35,3))
    MIR_lum_out.append(source.find_Lum_range(3,30))
    FIR_lum_out.append(source.find_Lum_range(30,500/(1+goodsN_auge_z_match[i])))
    X_UV_lum_out.append(source.find_Lum_range(1E-3,0.1))
    FIR_R_lum.append(source.find_Lum_range(40,400))

    plot = Plotter(Id, redshift, w, f, goodsN_auge_Lx_match[i], f1, up_check)

    # check = source.check_SED(10, check_span=5.5)

    
    check = source.check_SED(10, check_span=2.75)
    check6 = source.check_SED(6, check_span=2.75)
    check_sed.append(check)
    check_sed6.append(check6)
    field.append('GOODS-N')
    spec_class.append(0)

    # if check6 == 'GOOD':
    # # #     print(shape, source.Find_slope(uv1,uv2), source.Find_slope(mir11, mir12), source.Find_slope(mir21, mir22))
    #     # if shape == 6:
    #         print(Id,check6,shape,up_check,source.Find_slope(uv1, uv2),source.Find_slope(mir11, mir12),mir_slope2[i])
    #         plot.PlotSED(point_x=[0.25,1.0,6.5,10],point_y=[f025/f1,f1/f1,f6/f1,f10/f1],fir_x = wfir, fir_y = ffir)


    # if check6 == 'GOOD':

    #     c = SkyCoord(ra = goodsN_auge_RA_match[i], dec = goodsN_auge_DEC_match[i], unit=(u.hourangle, u.deg))
    #     coords = c.to_string('hmsdms').split()

    #     cols, data, dtyp = source.output_properties(
    #         'GOODS-N', int(goodsN_auge_ID_match[i]), coords[0], coords[1], goodsN_auge_Lx_match[i], goodsN_auge_Nh_match[i])
    #     source.write_output_file('AGN_Properties7.fits',data,cols,dtyp,'w')

    #     cols, data, dtyp = source.output_phot('GOODS-N',filter_GOODS_total,filter_GOODSN_match)
    #     source.write_output_file('AGN_photometry_GOODS7.fits',data,cols,dtyp,'w')

    # if Id == 167601:
    # plot.Plot_FIR_SED(wfir, ffir/f1)
        # plot.PlotSED(point_x=6.0,point_y=f6/f1)
    # source.Find_Lbol()


# '''
tgn = time.perf_counter()
print(f'Done with GOODS-N sources ({tgn - ts:0.4f} second)')

###############################################################################
###############################################################################
############################## Run GOODS-S SEDs ###############################
# '''
bad_source = np.asarray([409,520,579])
for i in range(len(goodsS_auge_ID_match)):
    if goodsS_auge_ID_match[i] in bad_source:
        continue
    else:
    # for i in range(50):
        try:
            source = AGN(goodsS_auge_phot_ID_match[i], goodsS_auge_z_match[i], GOODSS_auge_filters, goodsS_flux_array_auge[i], goodsS_flux_err_array_auge[i])
            source.MakeSED(data_replace_filt=['FLUX_24'])
            source.FIR_extrap(['FLUX_24', 'MIPS2', 'FLUX_100_goodsS', 'FLUX_160_goodsS', 'FLUX_250_goodsS', 'FLUX_350_goodsS', 'FLUX_500_goodsS'])

            Lcheck = source.Lum_filter('FLUX_24')
            F24_out.append(Lcheck)
            stack_bin.append(7)

            ix, iy = source.Int_SED(xmin=1E-1, xmax=1E1)
            int_x.append(ix)
            int_y.append(iy)

            # wfir, ffir, f100, f100_boot = source.Int_SED_FIR(Find_value=100.0, discreet=True, boot=True)
            wfir, ffir, f100 = source.Int_SED_FIR(Find_value=100.0, discreet=True)
            wfir_out.append(wfir)
            ffir_out.append(ffir)

            L100_ratio = source.flux_ratio_lower()
            F100_ratio.append(L100_ratio)

            # f1, f1_boot = source.Find_value(1.0, boot=True)
            # f2, f2_boot = source.Find_value(2.0, boot=True)
            # xval, xval_boot = source.Find_value(3E-4, boot=True)
            # f6, f6_boot = source.Find_value(6.0, boot=True)
            # f025, f025_boot = source.Find_value(0.25, boot=True)
            # f10, f10_boot = source.Find_value(10, boot=True)
            # f2kev, f2kev_boot = source.Find_value(6.1992e-4, boot=True)
            # f015, f015_boot = source.Find_value(0.15,boot=True)
            # f65, f65_boot = source.Find_value(6.5,boot=True)

            f1 = source.Find_value(1.0)
            f2 = source.Find_value(2.0)
            xval = source.Find_value(3E-4)
            f6 = source.Find_value(6.0)
            f025 = source.Find_value(0.25)
            f10 = source.Find_value(10)
            f2kev = source.Find_value(6.1992e-4)
            f015 = source.Find_value(0.2)
            f65 = source.Find_value(6.5)

            hao_x, hao_y = source.mix_loc([0.3, 1], [1, 3])
            mix_x.append(hao_x)
            mix_y.append(hao_y)

            lbol = source.Find_Lbol()
            lbol_sub = source.Find_Lbol(sub=True,Lscale=scale_array, Lnorm=f1, temp_x=temp_wave, temp_y=temp_lum)
            lbol_sf_sub = source.Find_Lbol(sub=True,Lscale=scale_array, Lnorm=f1, temp_x=m82_wave, temp_y=m82_lum)
            # lbol_sub = source.Find_Lbol_temp_sub(scale_array, f1, temp_wave, temp_lum)
            # lbol_sf_sub = source.Find_Lbol_temp_sub(scale_array, f1, m82_wave, m82_lum)
            # lbol_sf_sub = source.Find_Lbol(xmax=30)
            # lbol_sub = source.Find_Lbol_temp_sub(scale_array, f1, temp_wave, temp_lum, xmax=50)
            shape = source.SED_shape()

            Id, redshift, w, f, frac_err, up_check = source.pull_plot_info(norm_w=1)
            out_ID.append(Id)
            out_x.append(w)
            out_y.append(f)
            out_frac_error.append(frac_err)
            out_Lx.append(goodsS_auge_Lx_match[i])
            out_Lx_hard.append(goodsS_auge_Lx_hard_match[i])
            out_z.append(goodsS_auge_z_match[i])
            FIR_upper_lims.append(up_check)

            norm.append(f1)
            F025.append(f025)
            F1.append(f1)
            F2.append(f2)
            F6.append(f6)
            F10.append(f10)
            F100.append(f100)
            xval_out.append(xval)
            F2kev.append(f2kev)
            # F1_boot.append(f1_boot)
            # F025_boot.append(f025_boot)
            # F2_boot.append(f2_boot)
            # F6_boot.append(f6_boot)
            # F10_boot.append(f10_boot)
            # xval_out_boot.append(xval_boot)
            # F100_boot.append(f100_boot)
            # F2kev_boot.append(f2kev_boot)
            Lbol_out.append(lbol)
            Lbol_sub_out.append(lbol_sub)
            Lbol_sf_sub_out.append(lbol_sf_sub)
            Nh.append(goodsS_auge_Nh_match[i])
            Nh_check.append(goodsS_Nh_check[i])
            abs_check.append(0)

            # shape = source.SED_shape(uv1, uv2, mir11, mir12, mir21, mir22)
            shape = source.SED_shape()

            uv_slope.append(source.Find_slope(uv1, uv2))
            mir_slope1.append(source.Find_slope(mir11, mir12))
            mir_slope2.append(source.Find_slope(mir21, mir22))
            out_SED_shape.append(shape)
            UV_lum_out.append(source.find_Lum_range(0.1,0.35))
            OPT_lum_out.append(source.find_Lum_range(0.35,3))
            MIR_lum_out.append(source.find_Lum_range(3,30))
            FIR_lum_out.append(source.find_Lum_range(30,500/(1+goodsS_auge_z_match[i])))
            X_UV_lum_out.append(source.find_Lum_range(1E-3,0.1))
            FIR_R_lum.append(source.find_Lum_range(40,400))

            plot = Plotter(Id, redshift, w, f, goodsS_auge_Lx_match[i], f1, up_check)
            check = source.check_SED(10, check_span=2.75)
            check6 = source.check_SED(6, check_span=2.75)
            check_sed.append(check)
            check_sed6.append(check6)
            spec_class.append(0)
            field.append('GOODS-S')

            # if check == 'GOOD':
                # print(Id,check,shape)
                # plot.PlotSED(point_x=[0.25,6,100],point_y=[f025/f1,f6/f1,f100/f1],fir_x = wfir, fir_y = ffir)

            # if check6 == 'GOOD':
            # # #     print(shape, source.Find_slope(uv1,uv2), source.Find_slope(mir11, mir12), source.Find_slope(mir21, mir22))
            #     # if shape == 6:
            #         print(Id,check6,shape,up_check,source.Find_slope(uv1, uv2),source.Find_slope(mir11, mir12),mir_slope2[i])
            #         plot.PlotSED(point_x=[0.25,1.0,6.5,10],point_y=[f025/f1,f1/f1,f6/f1,f10/f1],fir_x = wfir, fir_y = ffir)

            
            # if check6 == 'GOOD':

            #     c = SkyCoord(ra = goodsS_auge_RA_match[i]*u.degree, dec = goodsS_auge_DEC_match[i]*u.degree)
            #     coords = c.to_string('hmsdms').split()

            #     cols, data, dtyp = source.output_properties('GOODS-S', int(goodsS_auge_ID_match[i]), coords[0], coords[1], goodsS_auge_Lx_match[i], goodsS_auge_Nh_match[i])
            #     source.write_output_file('AGN_Properties7.fits',data,cols,dtyp,'w')

            #     cols, data, dtyp = source.output_phot('GOODS-S',filter_GOODS_total,filter_GOODSS_match)
            #     source.write_output_file('AGN_photometry_GOODS7.fits',data,cols,dtyp,'w')
                
            # if Id == 567:
                # plot.Plot_FIR_SED(wfir, ffir/f1)
            # if Id == 911:
                # print(source.Find_slope(0.1, 1.0))
                # plot.PlotSED(point_x=100,point_y=f100/f1)
            # source.Find_Lbol()
        except ValueError:
            continue


# '''
tgs = time.perf_counter()
print(f'Done with GOODS-S sources ({tgs - tgn:0.4f} second)')

# Make all output lists into arrays with only good sources
check_sed = np.asarray(check_sed)
check_sed6 = np.asarray(check_sed6)
GOOD_SED = check_sed == 'GOOD'
GOOD_SED = check_sed6 == 'GOOD'



# Make all output lists into arrays and remove the bad SEDs
# out_ID, out_z, out_x, out_y, out_frac_error = np.asarray(out_ID)[GOOD_SED], np.asarray(out_z)[GOOD_SED], np.asarray(out_x)[GOOD_SED], np.asarray(out_y)[GOOD_SED], np.asarray(out_frac_error)[[GOOD_SED]]
# out_Lx, out_Lx_hard = np.log10(np.asarray(out_Lx)[GOOD_SED]), np.log10(np.asarray(out_Lx_hard)[GOOD_SED])
# wfir_out, ffir_out = np.asarray(wfir_out)[GOOD_SED], np.asarray(ffir_out)[GOOD_SED]
# int_x, int_y = np.asarray(int_x)[GOOD_SED], np.asarray(int_y)[GOOD_SED]
# norm = np.asarray(norm)[GOOD_SED]
# FIR_upper_lims = np.asarray(FIR_upper_lims)[GOOD_SED]
# F025, F1, F6, F10, F100 = np.asarray(F025)[GOOD_SED], np.asarray(F1)[GOOD_SED], np.asarray(F6)[GOOD_SED], np.asarray(F10)[GOOD_SED], np.asarray(F100)[GOOD_SED]
# F2 = np.asarray(F2)[GOOD_SED]
# xval_out = np.asarray(xval_out)[GOOD_SED]

# # F025_boot, F1_boot, F6_boot, F10_boot, F100_boot = np.asarray(F025_boot)[GOOD_SED], np.asarray(F1_boot)[GOOD_SED], np.asarray(F6_boot)[GOOD_SED], np.asarray(F10_boot)[GOOD_SED], np.asarray(F100_boot)[GOOD_SED]
# # F2_boot = np.asarray(F2_boot)[GOOD_SED]
# # xval_out_boot = np.asarray(xval_out_boot)[GOOD_SED]

# F025_boot, F1_boot, F6_boot, F10_boot, F100_boot = np.asarray(F025), np.asarray(F1), np.asarray(F6), np.asarray(F10), np.asarray(F100)
# F2_boot = np.asarray(F2)
# xval_out_boot = np.asarray(xval_out)

# field = np.asarray(field)[GOOD_SED]
# out_SED_shape = np.asarray(out_SED_shape)[GOOD_SED]
# uv_slope, mir_slope1, mir_slope2 = np.asarray(uv_slope)[GOOD_SED], np.asarray(mir_slope1)[GOOD_SED], np.asarray(mir_slope2)[GOOD_SED]
# Lbol_out, Lbol_sub_out = np.asarray(Lbol_out)[GOOD_SED], np.asarray(Lbol_sub_out)[GOOD_SED]
# Nh = np.asarray(Nh)[GOOD_SED]
# UV_lum_out, OPT_lum_out, MIR_lum_out, FIR_lum_out = np.asarray(UV_lum_out)[GOOD_SED], np.asarray(OPT_lum_out)[GOOD_SED], np.asarray(MIR_lum_out)[GOOD_SED], np.asarray(FIR_lum_out)[GOOD_SED]
# X_UV_lum_out = np.asarray(X_UV_lum_out)[GOOD_SED]
# FIR_R_lum = np.asarray(FIR_R_lum)[GOOD_SED]
# Nh_check = np.asarray(Nh_check)[GOOD_SED]
# abs_check = np.asarray(abs_check)[GOOD_SED]
# spec_class = np.asarray(spec_class)[GOOD_SED]

# mix_x, mix_y = np.asarray(mix_x)[GOOD_SED], np.asarray(mix_y)[GOOD_SED]

# check_sed = check_sed[GOOD_SED]


out_ID, out_z, out_x, out_y, out_frac_error = np.asarray(out_ID), np.asarray(out_z), np.asarray(out_x), np.asarray(out_y), np.asarray(out_frac_error)
out_Lx, out_Lx_hard = np.log10(np.asarray(out_Lx)), np.log10(np.asarray(out_Lx_hard))
wfir_out, ffir_out = np.asarray(wfir_out), np.asarray(ffir_out)
int_x, int_y = np.asarray(int_x), np.asarray(int_y)
norm = np.asarray(norm)
FIR_upper_lims = np.asarray(FIR_upper_lims)
F025, F1, F6, F10, F100 = np.asarray(F025), np.asarray(F1), np.asarray(F6), np.asarray(F10), np.asarray(F100)
F2 = np.asarray(F2)
xval_out = np.asarray(xval_out)

# F025_boot, F1_boot, F6_boot, F10_boot, F100_boot = np.asarray(F025_boot)[GOOD_SED], np.asarray(F1_boot)[GOOD_SED], np.asarray(F6_boot)[GOOD_SED], np.asarray(F10_boot)[GOOD_SED], np.asarray(F100_boot)[GOOD_SED]
# F2_boot = np.asarray(F2_boot)[GOOD_SED]
# xval_out_boot = np.asarray(xval_out_boot)[GOOD_SED]

F025_boot, F1_boot, F6_boot, F10_boot, F100_boot = np.asarray(F025), np.asarray(F1), np.asarray(F6), np.asarray(F10), np.asarray(F100)
F2_boot = np.asarray(F2)
xval_out_boot = np.asarray(xval_out)

field = np.asarray(field)
out_SED_shape = np.asarray(out_SED_shape)
uv_slope, mir_slope1, mir_slope2 = np.asarray(uv_slope), np.asarray(mir_slope1), np.asarray(mir_slope2)
Lbol_out, Lbol_sub_out = np.asarray(Lbol_out), np.asarray(Lbol_sub_out)
Nh = np.asarray(Nh)
UV_lum_out, OPT_lum_out, MIR_lum_out, FIR_lum_out = np.asarray(UV_lum_out), np.asarray(OPT_lum_out), np.asarray(MIR_lum_out), np.asarray(FIR_lum_out)
X_UV_lum_out = np.asarray(X_UV_lum_out)
FIR_R_lum = np.asarray(FIR_R_lum)
Nh_check = np.asarray(Nh_check)
abs_check = np.asarray(abs_check)
spec_class = np.asarray(spec_class)

mix_x, mix_y = np.asarray(mix_x), np.asarray(mix_y)

check_sed = check_sed
check_sed6 = np.asarray(check_sed6)

F24_out = np.asarray(F24_out)
stack_bin = np.asarray(stack_bin)
F100_ratio = np.asarray(F100_ratio)

Lbol_sf_sub_out = np.asarray(Lbol_sf_sub_out)

print(out_y)
print(out_y.shape)
print('check dtype: ',out_y.dtype)



# values, base = np.histogram(FIR_lum_out[FIR_upper_lims == 1]/Lbol_out[FIR_upper_lims == 1], bins=40)
# #evaluate the cumulative
# cumulative = np.cumsum(values)
# # plot the cumulative function
# plt.plot(base[:-1], cumulative, c='blue')
# plt.show()


# # Sort all output data by the intrinsic X-ray luminosity
# sort = out_Lx.argsort()
# out_ID, out_z, out_x, out_y, out_frac_error = out_ID[sort], out_z[sort], out_x[sort], out_y[sort], out_frac_error[sort]
# out_Lx, out_Lx_hard = out_Lx[sort], out_Lx_hard[sort]
# wfir_out, ffir_out = wfir_out[sort], ffir_out[sort]
# int_x, int_y = int_x[sort], int_y[sort]
# norm = norm[sort]
# FIR_upper_lims = FIR_upper_lims[sort]
# F025, F1, F6, F10, F100 = F025[sort], F1[sort], F6[sort], F10[sort], F100[sort]
# F2 = F2[sort]
# xval_out = xval_out[sort]

# # F025_boot, F1_boot, F6_boot, F10_boot, F100_boot = F025_boot[sort], F1_boot[sort], F6_boot[sort], F10_boot[sort], F100_boot[sort]
# # F2_boot = F2_boot[sort]
# # xval_out_boot = xval_out_boot[sort]

# field = field[sort]
# out_SED_shape = out_SED_shape[sort]
# uv_slope, mir_slope1, mir_slope2 = uv_slope[sort], mir_slope1[sort], mir_slope2[sort]
# Lbol_out, Lbol_sub_out = Lbol_out[sort], Lbol_sub_out[sort]
# Nh = Nh[sort]
# UV_lum_out, OPT_lum_out, MIR_lum_out, FIR_lum_out = UV_lum_out[sort], OPT_lum_out[sort], MIR_lum_out[sort], FIR_lum_out[sort]
# FIR_R_lum = FIR_R_lum[sort]
# Nh_check = Nh_check[sort]
# abs_check = abs_check[sort]
# spec_class = spec_class[sort]
# mix_x, mix_y = mix_x[sort], mix_y[sort]

# check_sed = check_sed[sort]

# F24_out = F24_out[sort]
# stack_bin = stack_bin[sort]
# F100_ratio = F100_ratio[sort]

stack_bin1_ID = np.asarray([2387,   2667,   2700,   2731,   2803,   2832,   2846,   2850,   2868,   2873,
                            2960,   3015,   3029,   3092,   3131,   3171,   3318,   3335,   3388,   3398,
                            3485,   3504,   3739,   3783,   3840,   3851,   3854,   3936,   3939,   3976,
                            4007,   4034,   4053,   4214,   4222,   4276,   4290,   4295,   4334,   4387,
                            4409,   4414,   4592,   4596,   4602,   4739,   4747,   4898,   4964,   5028,
                            5062,   5089,   5135,   5143,   5151,   5172,    417,    434,    492,    514,
                            520,    521,  57494,  89316, 129876, 129885])
stack_bin2_ID = np.asarray([2360,   2463,   2471,   2525,   2563,   2598,   2635,   2693,   2728,   2782,
                            2811,   2831,   2840,   2906,   3053,   3241,   3246,   3259,   3264,   3327,
                            3427,   3488,   3540,   3547,   3626,   3628,   3647,   3708,   3763,   3810,
                            3831,   3846,   3861,   3872,   3884,   3909,   3966,   3979,   3982,   4010,
                            4028,   4031,   4051,   4073,   4087,   4139,   4159,   4264,   4272,   4407,
                            4418,   4422,   4424,   4437,   4456,   4512,   4696,   4758,   4791,   4838,
                            4867,   5031,   5087,    405,    425,  57498, 129884, 129887])
stack_bin3_ID = np.asarray([2363,   2388,   2420,   2442,   2446,   2482,   2522,   2536,   2673,   2675,
                            2702,   2711,   2753,   2845,   2878,   2886,   2925,   2935,   2940,   2948,
                            2949,   3037,   3106,   3116,   3179,   3232,   3291,   3304,   3305,   3312,
                            3339,   3354,   3408,   3868,   3912,   3921,   3929,   3934,   3949,   3975,
                            3983,   4019,   4021,   4029,   4060,   4158,   4174,   4273,   4278,   4287,
                            4306,   4321,   4442,   4467,   4510,   4557,   4558,   4591,   4624,   4630,
                            4766,   4781,   4799,   4816,   4833,   4836,   4845,   4853,   4877,   4881,
                            4902,   4913,   4939,   4957,   4991,   5005,   5025,   5064,   5068,   5078,
                            5213,    411,    418,    488,    491,  15292,  15296,  15306,  42255,  50025,
                            107987, 129802, 180997])
stack_bin4_ID = np.asarray([2379,   2407,   2413,   2469,   2524,   2587,   2741,   2871,   2971,   3005,
                            3085,   3147,   3194,   3209,   3211,   3219,   3316,   3555,   3557,   3603,
                            3636,   3652,   3654,   3719,   3808,   4001,   4054,   4096,   4134,   4136,
                            4194,   4220,   4235,   4292,   4329,   4398,   4495,   4544,   4625,   4626,
                            4645,   4666,   4771,   4920,   5043,   5076,   5079,   5093,   5094,   5163,
                            399,    505,  15297,  50021, 129821])
stack_bin5_ID = np.asarray([2458,   2474,   2521,   2542,   2555,   2600,   2622,   2627,   2686,   2691,
                            2788,   2793,   2847,   2893,   2895,   2901,   2974,   2988,   3050,   3059,
                            3185,   3258,   3274,   3281,   3306,   3322,   3361,   3381,   3382,   3410,
                            3482,   3492,   3511,   3608,   3627,   3705,   3711,   3744,   3761,   3794,
                            3816,   3823,   3835,   3837,   3853,   3859,   3873,   3880,   3908,   3914,
                            3937,   4020,   4074,   4112,   4118,   4122,   4127,   4196,   4211,   4212,
                            4217,   4231,   4258,   4260,   4267,   4284,   4303,   4342,   4365,   4395,
                            4420,   4441,   4449,   4488,   4585,   4692,   4707,   4716,   4728,   4778,
                            4807,   4810,   4811,   4824,   4825,   4831,   4832,   4870,   4884,   4918,
                            4934,   4940,   4982,   5007,   5012,   5041,   5100,   5139,   5148,   5155,
                            489,  89309, 105728, 129814, 180999])
stack_bin6_ID = np.asarray([2368,  2476,  2570,  2606,  2660,  2707,  2708,  2808,  2909,  3027,  3212,  3228,
                            3268,  3376,  3487,  3544,  3610,  3633,  3660,  3709,  3726,  3862,  3865,  3915,
                            4111,  4225,  4421,  4446,  4531,  4598,  4615,  4676,  4678,  4784,  4793,  4850,
                            4851,  4915,  5106,   503, 50029])
stack_bin7_ID = np.asarray([2445,   2516,   2533,   2637,   2654,   2706,   2729,   2829,   2936,   3070,
                            3076,   3099,   3168,   3247,   3249,   3517,   3552,   3674,   3692,   3766,
                            3768,   3800,   3803,   3822,   3836,   3946,   3953,   3965,   3967,   4059,
                            4147,   4241,   4304,   4448,   4476,   4504,   4704,   4705,   4735,   4865,
                            4998,   5101,   5158,   5162,   5211,    493,  89310, 107991, 129819])
stack_bin8_ID = np.asarray([2685,   2748,   2774,   2795,   2931,   2964,   3129,   3201,   3296,   3451,
                            3531,   4349,   4435,   4453,   4490,   4857,   4872,   5136,   5142,   5178,
                            5185, 105703])
bin_stack_out = []
for i in range(len(out_ID)):
    if out_ID[i] in stack_bin1_ID:
        bin_stack_out.append(1)
    elif out_ID[i] in stack_bin2_ID:
        bin_stack_out.append(2)
    elif out_ID[i] in stack_bin3_ID:
        bin_stack_out.append(3)
    elif out_ID[i] in stack_bin4_ID:
        bin_stack_out.append(4)
    elif out_ID[i] in stack_bin5_ID:
        bin_stack_out.append(5)
    elif out_ID[i] in stack_bin6_ID:
        bin_stack_out.append(6)
    elif out_ID[i] in stack_bin7_ID:
        bin_stack_out.append(7)
    elif out_ID[i] in stack_bin8_ID:
        bin_stack_out.append(8)
    else:
        bin_stack_out.append(9)

# print(bin_stack_out)
bin_stack_out = np.asarray(bin_stack_out)
# print(len(bin_stack_out))
print('Here: ')
print(out_x)
print('Check 1: ',type(out_ID),type(field),type(out_Lx),type(out_Lx[0]))
print('Check 2: ',type(out_x),print(type(out_x[0])),print(out_x.dtype))

h = np.asarray(['ID','field','shape','z','x','y','frac_err','Lx','Lx_hard','wfir','ffir','int_x','int_y','norm','FIR_upper_lims',
                'F025','F025_boot','F1','F1_boot','F6','F6_boot','F10','F10_boot','F100','F100_boot','F2','F2_boot','uv_slope','mir_slope1','mir_slope2',
                'Lbol','Lbol_sub','Lbol_sf_sub','Nh','UV_lum','OPT_lum','MIR_lum','FIR_lum','FIR_R_lum','Nh_check','abs_check','mix_x','mix_y','spec_class','sed_check','F24_lum','stack_bin','F100_ratio','check6'])
t = Table([ out_ID,field,out_SED_shape,out_z,out_x,out_y,out_frac_error,
            out_Lx,out_Lx_hard,wfir_out,ffir_out,int_x,int_y,norm,FIR_upper_lims,
            F025,F025_boot,F1,F1_boot,F6,F6_boot,F10,F10_boot,F100,F100_boot,F2,F2_boot,uv_slope,mir_slope1,mir_slope2,
            Lbol_out,Lbol_sub_out,Lbol_sf_sub_out,Nh,UV_lum_out,OPT_lum_out,MIR_lum_out,FIR_lum_out,FIR_R_lum,Nh_check,abs_check,mix_x,mix_y,spec_class,check_sed,F24_out,bin_stack_out,F100_ratio,check_sed6],names=(h))

t.write('/Users/connor_auge/Research/Disertation/catalogs/AHA_SEDs_out_ALL_F6_FINAL3.fits',format='fits',overwrite=True)



# print(len(abs_check),len(abs_check[abs_check == 0]),len(abs_check[abs_check == 1]),len(abs_check[abs_check == 2]))

Lbol_out = np.log10(Lbol_out)
Lbol_sub_out = np.log10(Lbol_sub_out)
# Lbol_duras = np.log10(Lit_functions.Durras_Lbol((out_Lx+np.log10(0.611)),typ='Lx'))+out_Lx
Lbol_duras = np.log10(Lit_functions.Durras_Lbol(out_Lx,typ='Lx'))+out_Lx
check_durras = np.log10(Lit_functions.Durras_Lbol(np.arange(43,46,0.25),typ='Lx'))+np.arange(43,46,0.25)
check = np.log10(Lit_functions.Durras_Lbol(np.arange(42,48,0.05),typ='Lbol'))

print('Total SEDs')
print('Stripe82X: ', len(out_ID[field == 'S82X']))
print('COSMOS: ', len(out_ID[field == 'COSMOS']))
print('GOODS-N: ', len(out_ID[field == 'GOODS-N']))
print('GOODS-S: ', len(out_ID[field == 'GOODS-S']))
print('All: ', len(out_ID))
print('~~~~~~~~~~')
print('Total GOOD SEDs')
print('Stripe82X: ', len(out_ID[GOOD_SED][field[GOOD_SED] == 'S82X']))
print('COSMOS: ', len(out_ID[GOOD_SED][field[GOOD_SED] == 'COSMOS']))
print('GOODS-N: ', len(out_ID[GOOD_SED][field[GOOD_SED] == 'GOODS-N']))
print('GOODS-S: ', len(out_ID[GOOD_SED][field[GOOD_SED] == 'GOODS-S']))
print('All: ', len(out_ID[GOOD_SED]))

# Begin Plotting
# plot = Plotter(out_ID, out_z, out_x, out_y, out_Lx, norm, FIR_upper_lims)
# plot2 = Plotter_Letter2(out_ID, out_z, out_x, out_y, out_frac_error)
# plot_shape = SED_shape_Plotter(out_ID, out_z, out_x, out_y, out_Lx, norm, FIR_upper_lims, out_SED_shape)

# plot = Plotter(out_ID, out_z, out_x, out_y, out_Lx, norm, FIR_upper_lims)
# plot2 = Plotter_Letter2(out_ID, out_z, out_x, out_y, out_frac_error)
# plot_shape = SED_shape_Plotter(out_ID, out_z, out_x, out_y, out_Lx, norm, FIR_upper_lims, out_SED_shape)

# plot = Plotter(out_ID[FIR_upper_lims == 0], out_z[FIR_upper_lims == 0], out_x[FIR_upper_lims == 0], out_y[FIR_upper_lims == 0], out_Lx[FIR_upper_lims == 0], norm[FIR_upper_lims == 0], FIR_upper_lims[FIR_upper_lims == 0])
# plot2 = Plotter_Letter2(out_ID[FIR_upper_lims == 0], out_z[FIR_upper_lims == 0], out_x[FIR_upper_lims == 0], out_y[FIR_upper_lims == 0], out_frac_error[FIR_upper_lims == 0])
# plot_shape = SED_shape_Plotter(out_ID[FIR_upper_lims == 0], out_z[FIR_upper_lims == 0], out_x[FIR_upper_lims == 0], out_y[FIR_upper_lims == 0], out_Lx[FIR_upper_lims == 0], norm[FIR_upper_lims == 0], FIR_upper_lims[FIR_upper_lims == 0], out_SED_shape[FIR_upper_lims == 0])

# plot = Plotter(out_ID[out_SED_shape == 1], out_z[out_SED_shape == 1], out_x[out_SED_shape == 1], out_y[out_SED_shape == 1], out_Lx[out_SED_shape == 1], norm[out_SED_shape == 1], FIR_upper_lims[out_SED_shape == 1])


plot = Plotter(out_ID, out_z, out_x, out_y, out_Lx, norm, FIR_upper_lims)
plot2 = Plotter_Letter2(out_ID, out_z, out_x, out_y, out_frac_error)
plot_shape = SED_shape_Plotter(out_ID, out_z, out_x, out_y, out_Lx, norm, FIR_upper_lims, out_SED_shape)


# plt.figure(figsize=(8,8))
# plt.scatter(mix_x, mix_y)
# plt.xlim(-4,2.5)
# plt.ylim(-3,3)
# plt.grid()
# plt.show()

# plt.figure(figsize=(9,9))
# plt.hist(out_Lx,bins=np.arange(42.5,46,0.25),color='gray',alpha=0.3)
# plt.hist(out_Lx[field == 'c'],bins=np.arange(42.5,46,0.25),histtype='step',color='b',alpha=0.8,lw=5,label='COSMOS')
# plt.hist(out_Lx[field == 's'],bins=np.arange(42.5,46,0.25),histtype='step',color='r',alpha=0.8,lw=5,label='Stripe82X')
# plt.hist(out_Lx[(field == 'gn') | (field == 'gs')],bins=np.arange(42.5,46,0.25),histtype='step',color='k',alpha=0.7,lw=5,label='GOODS-N/S')
# plt.axvline(np.nanmean(out_Lx[field == 'c']), color = 'b', lw=3, ls='--', alpha=0.7)
# plt.axvline(np.nanmean(out_Lx[field == 's']), color = 'r', lw=3, ls='--', alpha=0.7)
# plt.axvline(np.nanmean(out_Lx[(field == 'gs') | (field == 'gn')]), color = 'k', lw=3, ls='--', alpha=0.8)
# plt.xlabel(r'log L$_{\mathrm{X}}$ [erg/s]')
# plt.legend()
# plt.grid()
# plt.xlim(42.75,46)
# plt.savefig('/Users/connor_auge/Desktop/Final_Plots/a_new/Lx_sample_update.pdf')
# plt.show()


# opt_x = np.ones(len(xval_out))
# opt_x[opt_x == 1.] = 3E-4
# plot.multi_SED('a_new/All_SEDs_test',median_x=int_x,median_y=int_y,wfir=wfir_out,ffir=ffir_out,Median_line=True,FIR_upper='upper lims')
# plot.multi_SED('Shape1',median_x=int_x[out_SED_shape == 1],median_y=int_y[out_SED_shape == 1],wfir=wfir_out[out_SED_shape == 1],ffir=ffir_out[out_SED_shape == 1],Median_line=True,FIR_upper='upper lims')
# plot.multi_SED_bins('a_new/All_z_bins_norm',bin='redshift',field=field,median_x=int_x,median_y=int_y,wfir=wfir_out,ffir=ffir_out,Median_line=True,FIR_upper='upper lims',scale=True)
# plot.multi_SED_bins('a_new/All_Lx_bins_norm',bin='Lx',field=field,median_x=int_x,median_y=int_y,wfir=wfir_out,ffir=ffir_out,Median_line=True,FIR_upper='upper lims',scale=True)
# plot.multi_SED_bins('All_z_field',bin='field',field=field,median_x=int_x,median_y=int_y,wfir=wfir_out,ffir=ffir_out,Median_line=True,FIR_upper='upper lims')
# plot.median_SED_plot('All_median_SEDs2', median_x=int_x, median_y=int_y, wfir=wfir_out, ffir=ffir_out, shape=out_SED_shape, FIR_upper='upper lims')
# plot.median_SED_1panel('a_new/median_SEDs_shape', median_x=int_x, median_y=int_y, wfir=wfir_out, ffir=ffir_out, shape=out_SED_shape, FIR_upper='upper lims', bins='shape')
# plot.median_SED_1panel('median_SEDs_Lx_bins', median_x=int_x, median_y=int_y, wfir=wfir_out, ffir=ffir_out, shape=out_SED_shape, FIR_upper='upper lims', bins='Lx_5')
# plot.median_SED_1panel('median_SEDs_Lx_bins', median_x=int_x, median_y=int_y, wfir=wfir_out, ffir=ffir_out, shape=out_SED_shape, FIR_upper='upper lims', bins='Lx_3')
# print('mix')
# print(mix_x)
# print(mix_y)

# plot.mix_plot('a_new/mix_plot4',mix_x,mix_y,out_SED_shape)

# plot.L_hist('Lx_hist',out_Lx-np.log10(F1),r'log L$_{\rm X}$) [erg/s]',[-3.5,3.5],[-3.5,3.5,0.25],median=True,std=True)
# plot.L_hist('L_UV_hist',np.log10(F025)-np.log10(F1),r'log L (0.25 $\mu$m) [erg/s]',[-3.5,3.5],[-3.5,3.5,0.25],median=True,std=True)
# plot.L_hist('L_MIR_hist', np.log10(F6)-np.log10(F1), r'log L (6 $\mu$m) [erg/s]', [-3.5, 3.5], [-3.5, 3.5, 0.25],median=True, std=True)
# plot.L_hist('L_FIR_hist',np.log10(F100)-np.log10(F1),r'log L (100 $\mu$m) [erg/s]',[-3.5,3.5],[-3.5,3.5,0.25],median=True,std=True)

# print(len(Nh[Nh > 0]))
# print(len(Nh[Nh_check == 0]))
# plot.violin_plot('Lx_Violin_plot','Lx',out_Lx,out_SED_shape,bins='shape')
# plot.violin_plot('Lone_Violin_plot','Lone', np.log10(F1), out_SED_shape, bins='shape')
# plot.violin_plot('Nh_Violin_plot_detec','Nh', np.log10(Nh[Nh_check == 0]), out_SED_shape[Nh_check == 0], bins='shape')
# plot.violin_plot('a_new/Lbol_Violin_plot', 'Lbol', Lbol_sub_out, out_SED_shape, bins='shape')
# plot.violin_plot('a_new/Lbol_Lx_Violin_plot', 'Lbol/Lx', Lbol_sub_out - out_Lx, out_SED_shape, bins='shape')


# plot_shape.L_hist_panels('a_new/z_hist_panels',out_z,r'z',[0,2],[0,1.5,0.2],median=True,bins='shape')
# plot_shape.L_hist_panels('a_new/Lx_hist_panels',out_Lx,r'log L$_{\rm X}$',[43,46],[43,46,0.25],median=True,bins='shape')
# plot_shape.L_hist_panels('a_new/Lbol_Lx_hist_panels',Lbol_sub_out - out_Lx,r'log L$_{\rm bol}$/L$_{\rm X}$',[0,3],[0,3,0.25],median=True,bins='shape')
# plot_shape.L_hist_panels('a_new/Lone_hist_panels',np.log10(F1),r'log L (1$\mu \rm m)$',[43,46],[43,46,0.25],median=True,bins='shape')
# plot_shape.L_hist_panels('a_new/Lbol_hist_panels',Lbol_sub_out,r'log L$_{\rm bol}$',[44,46.5],[44,46.5,0.25],median=True,bins='shape')
# plot_shape.L_hist_panels('a_new/Nh_hist_panels',np.log10(Nh),r'log N$_{\rm H}$',[20,24.5],[20,24.5,0.1],median=True,bins='shape')

# for i in range(len(out_SED_shape)):
    # print(i,out_ID[i],out_SED_shape[i],uv_slope[i],mir_slope1[i],mir_slope2[i])


# plot_shape.shape_1bin_h('horizantal_3_panel_NSF3',median_x=int_x,median_y=int_y,wfir=wfir_out,ffir=ffir_out,uv_slope=uv_slope,mir_slope1=mir_slope1,mir_slope2=mir_slope2,Median_line=True,FIR_med=False,FIR_upper='upper lims')
# plot_shape.shape_1bin_h('horizantal_5_panel',median_x=int_x,median_y=int_y,wfir=wfir_out,ffir=ffir_out,uv_slope=uv_slope,mir_slope1=mir_slope1,mir_slope2=mir_slope2,Median_line=True,FIR_med=False,FIR_upper='upper lims')

# plot.plot_medians('new_med_plot',F1,F025,F6,F100)

# plot_shape.shape_1bin_v('a_new/vertical_5_panel_check',median_x=int_x,median_y=int_y,wfir=wfir_out,ffir=ffir_out,uv_slope=uv_slope,mir_slope1=mir_slope1,mir_slope2=mir_slope2,Median_line=True,FIR_upper='upper lims',bins='shape')
# plot_shape.shape_1bin_v('vertical_5_panel_Lx',median_x=int_x,median_y=int_y,wfir=wfir_out,ffir=ffir_out,uv_slope=uv_slope,mir_slope1=mir_slope1,mir_slope2=mir_slope2,Median_line=True,FIR_upper='upper lims',bins='Lx')

# plot.L_ratio_3panels('AGN_Lx_ratio5','Lx','UV-MIR-FIR','X-axis',F1,UV_lum_out,MIR_lum_out,FIR_lum_out,shape=out_SED_shape,L=Lbol_sub_out)
# plot.L_ratio_3panels('AGN_Lx_ratio_new','Lx','UV-MIR-FIR','X-axis',F1,F025,F6,F100,shape=out_SED_shape,L=Lbol_sub_out)

# plot.L_scatter_3panels('a_new/AGN_Lx_scatter_no_upper_other', 'UV-MIR-FIR', 'Lx', 'X-axis', F1[FIR_upper_lims == 0], F025[FIR_upper_lims == 0],F6[FIR_upper_lims == 0], F100[FIR_upper_lims == 0], shape=out_SED_shape[FIR_upper_lims == 0], L=Lbol_sub_out[FIR_upper_lims == 0])
# plot.L_scatter_3panels('a_new/AGN_Lx_scatter_fit1', 'UV-MIR-FIR', 'Lx', 'X-axis', F1, F025,F6, F100, shape=out_SED_shape, L=Lbol_sub_out)


# print(out_Lx_hard)
# plot.L_scatter_1panel('Ranalli','FIR_lum','Lx_h','None',np.log10(F1),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(Nh),shape=out_SED_shape,L=Lbol_sub_out,line='Ranalli',Lx_h=out_Lx_hard,Lum_range=FIR_R_lum)

# plot.L_scatter_3panels('a_new/AGN_Lx_scatter', 'Lx','UV-MIR-FIR','X-axis',F1,F025,F6,F100,shape=out_SED_shape,L=Lbol_sub_out)
# plot.L_ratio_1panel('a_new/Lx_Lbol','Lbol','Lbol/Lx','X-axis',F1,F025,F6,F100,shape=out_SED_shape,L=Lbol_sub_out)
# plot.L_ratio_1panel('Lx_Lbol_check_durras','Lbol','Lbol/Lx','X-axis',F1,F025,F6,F100,shape=out_SED_shape,L=Lbol_durras)

# goals_Lx = np.array([43.31484385, 43.78484385, 43.62484385, 43.72484385, 43.39484385, 43.26484385,
#                      43.56484385, 43.34484385, 43.51484385, 43.81484385, 43.32484385, 43.43484385,
#                      43.93484385, 43.18484385, 43.26484385, 43.91484385, 43.64484385, 43.14484385,
#                      43.13484385])
# goals_Lbol = np.log10(np.array([5.02707167e+45, 5.51831157e+45, 1.23614371e+45, 1.02375172e+45,
#                        2.23677376e+45, 9.15001901e+44, 7.91779926e+44, 1.37116974e+45,
#                        2.22296043e+45, 1.96278907e+45, 1.00068966e+45, 1.06311908e+45,
#                        2.86611956e+45, 1.51105456e+44, 9.54617436e+44, 3.97036310e+43,
#                        2.45095918e+45, 5.18452425e+45, 3.32507746e+45]))
# goals_LIR = np.array([12.32, 12.1, 11.17, 11.41, 11.93, 11.38, 11.3, 11.35, 11.65, 11.56, 11.4, 11.26,
#                       11.93, 10.73, 11.42, 11.3, 12.01, 12.21, 12.28]) + np.log10(3.8E33)

# plot.L_ratio_1panel_GOALS('Lx_Lbol_GOALS','Lbol','Lbol/Lx','X-axis',F1,F025,F6,F100,shape=out_SED_shape,L=Lbol_sub_out,goals_Lx=goals_Lx,goals_Lbol=goals_Lbol)
# plot.L_ratio_1panel_GOALS('Lx_Lbol_GOALS_LIR','Lbol','Lbol/Lx','X-axis',F1,F025,F6,F100,shape=out_SED_shape,L=Lbol_sub_out,goals_Lx=goals_Lx,goals_Lbol=goals_LIR)

# print('x: ', Lbol_sub_out)
# print('y: ', Lbol_sub_out - out_Lx)

# plot.L_scatter_1panel('Nh_Lx','Lx','Nh','X-axis',F1,F025,F6,F100,Nh,shape=out_SED_shape,L=Lbol_sub_out)
# plot.L_scatter_1panel('MIR_Lx_test_S82X_COSMOS4','MIR','Lx','X-axis',F1,F025,F6,F100,Nh,shape=out_SED_shape,L=Lbol_sub_out)
# plot.L_hist('a_new/Lone_hist',np.log10(F1),r'log L (1 $\mu$m) [erg/s]',[41.5,46],[41.5,46,0.25],median=True,std=True)
# plot.L_hist('all_Lx_hist',out_Lx,r'log L$_{\mathrm{X}}$ [erg/s]',[42.5,47],[42.5,47,0.25])


# plt.hist(Nh_check,bins=np.arange(0,5,1))
# plt.xlim(-1,4)
# plt.xlabel('NH Check')
# plt.show()

# print('Nh: ',len(Nh),len(Nh[Nh > 0]),len(Nh[Nh_check == 0]),len(Nh[Nh_check == 1]),len(Nh[Nh_check == 2]))
# plot.L_hist('a_new/Nh_hist',np.log10(Nh),r'log N$_{\mathrm{H}}$ [cm$^{-2}$]',[19.5,25],[19.5,24.5,0.25],median=True,std=False)
# plot.L_hist('Nh_hist_good',np.log10(Nh[Nh_check == 0]),r'log N$_{\mathrm{H}}$ [cm$^{-2}$]',[19.5,25],[19.5,24.5,0.25],median=True,std=False)
# plot.L_hist('Nh_hist_upper',np.log10(Nh[Nh_check == 1]),r'log N$_{\mathrm{H}}$ [cm$^{-2}$]',[19.5,25],[19.5,24.5,0.25],median=True,std=False)
# plot.L_hist('Nh_hist_lower',np.log10(Nh[Nh_check == 2]),r'log N$_{\mathrm{H}}$ [cm$^{-2}$]',[19.5,25],[19.5,24.5,0.25],median=True,std=False)

# plot.L_hist('a_new/Lbol_hist',Lbol_out,r'Total log L$_{\mathrm{bol}}$ [erg/s]',[43,48],[43,48,0.25],std=True)
# plot.L_hist('a_new/Lbol_sub_hist',Lbol_out,r'log L$_{\mathrm{bol}}$ [erg/s]',[43,48],[43,48,0.25],std=True)

# plot.L_hist('Lone_hist',np.log10(F1),r'log L (1 $\mu$m) [erg/s]',[41.5,46],[41.5,46,0.25],median=True,std=False)
# plot.L_hist_zbins('a_new/Lone_hist_zbins',np.log10(F1),r'log L (1 $\mu$m) [erg/s]',[41.5,46],[41.5,46,0.25],median=True,std=False)
# plot.L_hist_zbins('L2_hist_zbins',np.log10(F2),r'log L (2 $\mu$m) [erg/s]',[41.5,46],[41.5,46,0.25],median=True,std=False)
# plot.L_hist_zbins('Lx_hist_bins',out_Lx,r'log L$_{\mathrm{X}}$ [erg/s]',[42.5,46],[42.5,46,0.25],median=True,std=False,bin_type='Lx')
# plot.L_hist_zbins('Lone_Lx_hist_bins',np.log10(F1),r'log L (1$\mu$m) [erg/s]',[42.5,46],[42.5,46,0.25],median=True,std=False,bin_type='Lx')

# plot.L_hist_zbins('L2_1_hist_zbins',np.log10(F2) - np.log10(F1),r'log (L (2 $\mu$m)/ L (1$\mu$m))',[-1,1],[-1,1,0.25],median=True,std=False)
# plot.L_hist('Lx_sample',out_Lx,r'log L$_{\mathrm{X}}$ [erg/s]',[42.5,46],[42.5,46,0.25],median=True,std=False)
# plot.L_hist('Lbol_tot_hist_check',np.log10(Lbol_out),r'Total log L$_{\mathrm{bol}}$ [erg/s]',[43,48],[43,48,0.25],median=True,std=True)
# plot.L_hist('Lbol_sub_hist',Lbol_sub_out,r'log L$_{\mathrm{bol}}$ [erg/s]',[43,48],[43,48,0.25],median=True,std=True)


# print(len(F1[FIR_upper_lims == 1]))
# print(len(F1[FIR_upper_lims == 1][FIR_lum_out[FIR_upper_lims == 1]/Lbol_out[FIR_upper_lims == 1]>=0.5]))
# print(len(F1))
# print(len(F1[FIR_lum_out/Lbol_out>=0.5]))

# plt.scatter(np.log10(F1[FIR_upper_lims == 1]), FIR_lum_out[FIR_upper_lims == 1]/Lbol_out[FIR_upper_lims == 1])
# plt.scatter(np.log10(F1[FIR_upper_lims == 0]), FIR_lum_out[FIR_upper_lims == 0]/Lbol_out[FIR_upper_lims == 0],c='gray',alpha=0.5)
# plt.xlabel(r'L$_{1\mu m}$')
# plt.ylabel(r'L$_{\rm FIR}$/L$_{\rm bol}$')
# plt.show()

# plot_shape.L_hist_bins('Lone_hist_shape_bins',np.log10(F1),r'log L (1 $\mu$m) [erg/s]',[41.5,46],[41.5,46,0.25],median=True,std=False,bins='shape')
# plot_shape.L_hist_bins('Lone_hist_Lx_bins',np.log10(F1),r'log L (1 $\mu$m) [erg/s]',[41.5,46],[41.5,46,0.25],median=True,std=False,bins='Lx')

# plot_shape.L_hist_bins('Lbol_Duras_hist_shape_bins',Lbol_duras,r'log L (1 $\mu$m) [erg/s]',[42.5,47],[42.5,47,0.25],median=True,std=False)
# plot_shape.L_hist_bins('Lbol_hist_shape_bins',Lbol_sub_out,r'log L (1 $\mu$m) [erg/s]',[42.5,47],[42.5,47,0.25],median=True,std=False)
# plot_shape.L_hist_bins('ratio_hist_shape_bins',Lbol_sub_out-out_Lx,r'log L$_{\mathrm{bol}}$/L$_{\mathrm{X}}$',[0,5],[0,5,0.25],median=True,std=False)
# plot_shape.L_hist_bins('ratio_Duras_hist_shape_bins',Lbol_duras-out_Lx,r'log L$_{\mathrm{bol}}$/L$_{\mathrm{X}}$',[0,5],[0,5,0.25],median=True,std=False)

# plot_shape.L_hist_panels('a_new/Lx_hist_panels.pdf',out_Lx,r'log L$_{\rm X}$',[42.75,45.75],[42.5,46,0.25])


# print(np.nanmedian(np.log10(Lbol_sub_out[out_SED_shape == 1])))
# print(np.nanmedian(np.log10(Lbol_sub_out[out_SED_shape == 2])))
# print(np.nanmedian(np.log10(Lbol_sub_out[out_SED_shape == 3])))
# print(np.nanmedian(np.log10(Lbol_sub_out[out_SED_shape == 4])))
# print(np.nanmedian(np.log10(Lbol_sub_out[out_SED_shape == 5])))

# plot2.Box_1panel('Lone_box_1panel_v', 'Lone', np.log10(F1), uv_slope, mir_slope1, mir_slope2,shape=out_SED_shape)


# plot2.Upanels_ratio_plots('a_new/Nh_Upanels_check','Nh','UV/MIR-UV/Lx-MIR/Lx','Bins',Nh,out_Lx,Lbol_duras,np.log10(UV_lum_out),np.log10(MIR_lum_out),np.log10(FIR_lum_out),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,out_z,uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,shape=out_SED_shape,Nh_upper=Nh_check)
# plot2.Upanels_ratio_plots('a_new/Lum_Lbol','Lbol','UV-MIR-FIR/Lbol','Bins',Nh,out_Lx,Lbol_sub_out,np.log10(UV_lum_out),np.log10(MIR_lum_out),np.log10(FIR_lum_out),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,out_z,uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,shape=out_SED_shape,Nh_upper=Nh_check)

# plot2.Box_1panel('Nh_box_1panel_v', 'Nh', np.log10(Nh), uv_slope, mir_slope1, mir_slope2,shape=out_SED_shape)
# plot2.Upanels_ratio_plots('a_new/Lum_Lbol','Lbol','UV-MIR-FIR/Lbol','Bins',Nh,out_Lx,Lbol_duras,np.log10(UV_lum_out),np.log10(MIR_lum_out),np.log10(FIR_lum_out),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,out_z,uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,shape=out_SED_shape)
# plot2.Box_1panel('Lx_box_1panel_v', 'Lx', out_Lx, uv_slope, mir_slope1, mir_slope2,shape=out_SED_shape)
# plot2.Box_1panel('Lbol_Lx_box_fix2', 'Lbol/Lx', Lbol_sub_out-out_Lx, uv_slope, mir_slope1, mir_slope2, shape=out_SED_shape, L2=Lbol_duras-out_Lx)
# plot2.Box_1panel('Lbol_box_fix2', 'Lbol', Lbol_sub_out, uv_slope, mir_slope1, mir_slope2, shape=out_SED_shape, L2= Lbol_duras-np.log10(3.8E33))
# plot2.Box_1panel('Lbol_box_1panel_sub3', 'Lbol', np.log10(Lbol_sub_out), uv_slope, mir_slope1, mir_slope2, shape=out_SED_shape)
# plot2.scatter_1panel('UV_Lx_Lx_norm_new','Lx','UV/Lx','None','Both',Nh,out_Lx,np.log10(Lbol_sub_out),F1,np.log10(F025/norm),np.log10(F6/norm),np.log10(F100/norm),np.log10(F10/norm),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims)
# plot2.scatter_1panel('new/UV_FIR','FIR','UV/FIR','None','Bins',Nh,out_Lx,np.log10(Lbol_sub_out),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,out_SED_shape)
# plot2.scatter_1panel('new/UV_FIR2','UV','FIR/UV','None','Bins',Nh,out_Lx,np.log10(Lbol_sub_out),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,out_SED_shape)
# plot2.scatter_1panel('new/MIR_FIR','FIR','MIR6/FIR','None','Bins',Nh,out_Lx,np.log10(Lbol_sub_out),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,out_SED_shape)
# plot2.scatter_1panel('new/MIR_FIR2','MIR6','FIR/MIR6','None','Bins',Nh,out_Lx,np.log10(Lbol_sub_out),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,out_SED_shape)

# plot2.scatter_1panel('new/FIR_Lx_1panel', 'Lx', 'FIR', 'Y-axis', 'Bins', Nh, out_Lx, np.log10(Lbol_sub_out), F1, np.log10(F025/F1), np.log10(F6/F1), np.log10(F100/F1), np.log10(F10/F1), uv_slope, mir_slope1, mir_slope2, FIR_upper_lims)
# plot2.scatter_1panel('Lx_Lbol_1panel_sub_xmed_lim', 'Lbol', 'Lbol/Lx', 'None', 'Both', Nh, out_Lx, np.log10(Lbol_sub_out), F1, np.log10(F025), np.log10(F6), np.log10(F100), np.log10(F10), uv_slope, mir_slope1, mir_slope2, FIR_upper_lims, shape = out_SED_shape, durras=True)
# plot2.scatter_1panel('a_new/UV_MIR','MIR6','UV/MIR6','None','Bins',Nh,out_Lx,np.log10(Lbol_sub_out),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,out_SED_shape)
# plot2.scatter_1panel('a_new/UV_Lx','Lx','UV/Lx','None','Bins',Nh,out_Lx,np.log10(Lbol_sub_out),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,out_SED_shape)
# plot2.scatter_1panel('a_new/MIR_Lx','Lx','MIR6/Lx','None','Bins',Nh,out_Lx,np.log10(Lbol_sub_out),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,out_SED_shape)

# plot2.scatter_1panel('FIR_Lx_no_upper_shape2','Lx','FIR/Lx','None','Bins',Nh[FIR_upper_lims == 0],out_Lx[FIR_upper_lims == 0],np.log10(Lbol_sub_out[FIR_upper_lims == 0]),F1[FIR_upper_lims == 0],np.log10(F025[FIR_upper_lims == 0]),np.log10(F6[FIR_upper_lims == 0]),np.log10(F100[FIR_upper_lims == 0]),np.log10(F10[FIR_upper_lims == 0]),uv_slope[FIR_upper_lims == 0],mir_slope1[FIR_upper_lims == 0],mir_slope2[FIR_upper_lims == 0],FIR_upper_lims[FIR_upper_lims == 0],out_SED_shape[FIR_upper_lims == 0])
# plot2.scatter_1panel('a_new/FIR_Lx_no_upper_other','Lx','FIR/Lx','None','X-axis',Nh[FIR_upper_lims == 0],out_Lx[FIR_upper_lims == 0],np.log10(Lbol_sub_out[FIR_upper_lims == 0]),F1[FIR_upper_lims == 0],np.log10(F025[FIR_upper_lims == 0]),np.log10(F6[FIR_upper_lims == 0]),np.log10(F100[FIR_upper_lims == 0]),np.log10(F10[FIR_upper_lims == 0]),uv_slope[FIR_upper_lims == 0],mir_slope1[FIR_upper_lims == 0],mir_slope2[FIR_upper_lims == 0],FIR_upper_lims[FIR_upper_lims == 0],out_SED_shape[FIR_upper_lims == 0])
# plot2.scatter_1panel('a_new/FIR_Lx_check','Lx','FIR/Lx','None','X-axis',Nh,out_Lx,np.log10(Lbol_sub_out),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,out_SED_shape)


# plot2.scatter_1panel('a_new/FIR_Lx_test','Lx','FIR/Lx','None','Bins',Nh,out_Lx,np.log10(Lbol_sub_out),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,out_SED_shape)
# plot2.scatter_1panel('FIR_Lx_no_upper2','Lx','FIR/Lx','None','X-axis',Nh,out_Lx,np.log10(Lbol_sub_out),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,out_SED_shape)
# plot2.scatter_1panel('FIR_Lx_no_upper2_both','Lx','FIR/Lx','None','Both',Nh,out_Lx,np.log10(Lbol_sub_out),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,out_SED_shape)



# plot2.scatter_1panel('FIR_Lx_no_upper_meds','Lx','FIR/Lx','None','X-axis',Nh,out_Lx,np.log10(Lbol_sub_out),F1,np.log10(F025),np.log10(F6),np.log10(F10),np.log10(F10),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,out_SED_shape)

# plot2.scatter_1panel('UV_FIR','FIR','UV/FIR','None','Bins',Nh,out_Lx,np.log10(Lbol_sub_out),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,out_SED_shape)
# plot2.scatter_1panel('Lx_Lbol_shape_lim','Lbol','Lbol/Lx','None','Bins',Nh,out_Lx,Lbol_sub_out,F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,out_SED_shape)
