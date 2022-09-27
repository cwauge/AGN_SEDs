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
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io import ascii
from SED_v8 import AGN
from SED_plots_v2 import Plotter
from plots_Letter import Plotter_Letter
from plots_Letter2 import Plotter_Letter2
from SED_shape_plots import SED_shape_Plotter
from match import match
from mag_flux import mag_to_flux
from mag_flux import magerr_to_fluxerr


ti = time.perf_counter() # Start timer

path = '/Users/connor_auge/Research/Disertation/catalogs/' # Path for photometry catalogs

# set redshift and X-ray luminosity limits
z_min = 0.0
z_max = 3

Lx_min = 42
Lx_max = 60

# set the X-ray luminosity limits for the GOALS sources
goals_Lx_min = 35

###################################################################################
###################################################################################
############################## Read in COSMOS files ###############################

# Read in the data
# COSMOS2020 catalog
cosmos = fits.open(path+'cosmos2020/classic/COSMOS2020_CLASSIC_R1_v2.0_master.fits')
cosmos_data = cosmos[1].data
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
cosmos_xid = cosmos_data['id_chandra']
chandra_cosmos_xid = chandra_cosmos_data['id_x']
chandra_cosmos2_xid = chandra_cosmos2_data['id_x']
chandra_cosmos_ct_xid = chandra_cosmos_ct_data['id_x']

# X-ray coords
chandra_cosmos_RA = chandra_cosmos_data['RA_x']
chandra_cosmos_DEC = chandra_cosmos_data['DEC_x']

# Redshfits
cosmos_sz = cosmos_data['sz_zspec']
cosmos_ez = cosmos_data['ez_z_spec']
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
for i in range(len(chandra_cosmos_abs_corr_f)):
    if chandra_cosmos_abs_corr_f[i] != -99.0:
        abs_corr_use_f.append(chandra_cosmos_abs_corr_f[i])
        abs_corr_use_h.append(chandra_cosmos_abs_corr_h[i])
        abs_corr_use_s.append(chandra_cosmos_abs_corr_s[i])

    elif chandra_cosmos_abs_corr_up_f[i] != -99.0:
        abs_corr_use_f.append(chandra_cosmos_abs_corr_up_f[i])
        abs_corr_use_h.append(chandra_cosmos_abs_corr_up_h[i])
        abs_corr_use_s.append(chandra_cosmos_abs_corr_up_s[i])

    elif chandra_cosmos_abs_corr_lo_f[i] != -99.0:
        abs_corr_use_f.append(chandra_cosmos_abs_corr_lo_f[i])
        abs_corr_use_h.append(chandra_cosmos_abs_corr_lo_h[i])
        abs_corr_use_s.append(chandra_cosmos_abs_corr_lo_s[i])

    else:
        print('NO GOOD ABSORPTION CORRECTION DATA')

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
for i in range(len(chandra_cosmos_Lx_full)):    
    ind = np.where(chandra_cosmos2_xid == chandra_cosmos_xid[i])[0] # Check if there is a match to updated Chandra catalog 
    ind_ct = np.where(chandra_cosmos_ct_xid == chandra_cosmos_xid[i])[0] # Check if there is a match to compton thick Chandra catalog 

    if len(ind_ct) > 0:
        chandra_cosmos_Nh.append(chandra_cosmos_ct_nh[ind_ct][0]) # if there is a match append Nh from compton thick catalog 
        chandra_cosmos_Lx_hard[i] = chandra_cosmos_ct_Lx_hard[ind_ct] # replace Lx from original Chandra catalog with that from the CT cat
        chandra_cosmos_Lx_full[i] = chandra_cosmos_ct_Lx_full[ind_ct]
        check.append(3) # count which catalog data is from

    elif len(ind) > 0:
        chandra_cosmos_Lx_hard[i] = chandra_cosmos2_Lx_hard[ind] # replace Lx from orginal Chandra catalog with that from updated cat
        chandra_cosmos_Lx_full[i] = chandra_cosmos2_Lx_full[ind]
        if chandra_cosmos2_nh_lo_err[ind][0] == -99.:
            chandra_cosmos_Nh.append(chandra_cosmos2_nh[ind][0]+chandra_cosmos2_nh_up_err[ind][0]) # if there is Nh upper limit in updated cat append to Nh list
            check.append(2.5)
        else:
            chandra_cosmos_Nh.append(chandra_cosmos2_nh[ind][0]) # if there is a match append Nh from updated catalog
            check.append(2)
    else: # if no matches to updated or CT catalogs take Nh value from original catalog
        if chandra_cosmos_nh[i] == -99.: # If no good value take upper or lower limits 
            if chandra_cosmos_nh_lo[i] != -99.:
                chandra_cosmos_Nh.append(chandra_cosmos_nh_lo[i])
            else:
                chandra_cosmos_Nh.append(chandra_cosmos_nh_hi[i])
        else:    
            chandra_cosmos_Nh.append(chandra_cosmos_nh[i])
        check.append(1)
chandra_cosmos_Nh = np.asarray(chandra_cosmos_Nh)*1E22
check = np.asarray(check)

print('COSMOS All Lx cat: ', len(chandra_cosmos_Lx_full))

# Limit chandra sample to sources in z and Lx range
cosmos_condition = (chandra_cosmos_z > z_min) & (chandra_cosmos_z <= z_max) & (np.log10(chandra_cosmos_Lx_full) >= Lx_min) & (np.log10(chandra_cosmos_Lx_full) <= Lx_max) & (chandra_cosmos_phot_id != -99.)

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
print('COSMOS phot match: ', len(chandra_cosmos_phot_id_match))

# Convert the X-ray flux values to from cgs to mJy
chandra_cosmos_Fx_full_match_mjy = chandra_cosmos_Fx_full_match*4.136E8/(10-0.5)
chandra_cosmos_Fx_hard_match_mjy = chandra_cosmos_Fx_hard_match*4.136E8/(10-2)
chandra_cosmos_Fx_soft_match_mjy = chandra_cosmos_Fx_soft_match*4.136E8/(2-0.5)

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
    cosmos_data['FIR_24_FLUX'][cosmos_iy],
    cosmos_data['FIR_100_FLUX'][cosmos_iy],
    cosmos_data['FIR_160_FLUX'][cosmos_iy],
    cosmos_data['FIR_250_FLUX'][cosmos_iy],
    cosmos_data['FIR_350_FLUX'][cosmos_iy],
    cosmos_data['FIR_500_FLUX'][cosmos_iy],
    cosmos_data['FIR_850_FLUX'][cosmos_iy],
    # cosmos_data['FIR_1100_FLUX'][cosmos_iy],
    cosmos_data['FIR_20CM_FLUX'][cosmos_iy]
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
    cosmos_data['FIR_24_FLUXERR'][cosmos_iy],
    cosmos_data['FIR_100_FLUXERR'][cosmos_iy],
    cosmos_data['FIR_160_FLUXERR'][cosmos_iy],
    cosmos_data['FIR_250_FLUXERR'][cosmos_iy],
    cosmos_data['FIR_350_FLUXERR'][cosmos_iy],
    cosmos_data['FIR_500_FLUXERR'][cosmos_iy],
    cosmos_data['FIR_850_FLUXERR'][cosmos_iy],
    # cosmos_data['FIR_1100_FLUXERR'][cosmos_iy],
    cosmos_data['FIR_20CM_FLUXERR'][cosmos_iy]
])

# Transpose arrays so each row is a new source and each column is a obs filter
cosmos_flux_array = cosmos_flux_array.T
cosmos_flux_err_array = cosmos_flux_err_array.T
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

# WISE catalog with forced photometry on SLOAN sources
unwise = ascii.read('/Users/connor_auge/Desktop/desktop_catalogs/unwise_matches.csv')
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
s82x_Lx_full = np.asarray([10**i for i in lamassa_data['FULL_LUM']])
s82x_Lx_hard = np.array([10**i for i in lamassa_data['HARD_LUM']])
s82x_Lx_soft = np.array([10**i for i in lamassa_data['SOFT_LUM']])
s82x_Fx_full = lamassa_data['FULL_FLUX']
s82x_Fx_hard = lamassa_data['HARD_FLUX']
s82x_Fx_soft = lamassa_data['SOFT_FLUX']
s82x_spec_class = lamassa_data['SPEC_CLASS']

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
peca_ID = peca_data['srcid']
peca_Lx_full = peca_data['lumin_f']
peca_Lx_hard = peca_data['lumin_h']
peca_Lx_soft = peca_data['lumin_s']
peca_Lx_full_obs = peca_data['lumin_of']
peca_Fx_full = peca_data['flux_f']
peca_Fx_hard = peca_data['flux_h']
peca_Fx_soft = peca_data['flux_s']
peca_Nh = peca_data['nh']

# Fill in Nh data from Peca and replace LaMassa X-ray data with Peca 
s82x_Nh = []
for i, j in enumerate(s82x_id):
    ind = np.where(peca_ID == j)[0]
    if len(ind) == 1:
        s82x_Lx_full[i] = peca_Lx_full[ind][0]
        s82x_Lx_hard[i] = peca_Lx_hard[ind][0]
        s82x_Lx_soft[i] = peca_Lx_soft[ind][0]
        s82x_Fx_full[i] = peca_Fx_full[ind][0]
        s82x_Fx_hard[i] = peca_Fx_hard[ind][0]
        s82x_Fx_soft[i] = peca_Fx_soft[ind][0]
        s82x_Nh.append(peca_Nh[ind][0])
    else:
        s82x_Nh.append(0.0)
s82x_Nh = np.asarray(s82x_Nh)

print('S82X All: ', len(s82x_id))

# Limit Stripe82X sample to sources in z and Lx range
s82x_condition = (s82x_z > z_min) & (s82x_z <= z_max) & (np.log10(s82x_Lx_full) >= Lx_min) & (np.log10(s82x_Lx_full) <= Lx_max) & (np.logical_and(s82x_ra >= 13, s82x_ra <=37))

s82x_id = s82x_id[s82x_condition]
s82x_cat = s82x_cat[s82x_condition]
s82x_z = s82x_z[s82x_condition]
s82x_ra = s82x_ra[s82x_condition]
s82x_dec = s82x_dec[s82x_condition]
s82x_Lx_full = s82x_Lx_full[s82x_condition]
s82x_Lx_hard = s82x_Lx_hard[s82x_condition]
s82x_Lx_soft = s82x_Lx_soft[s82x_condition]
s82x_Fx_full = s82x_Fx_full[s82x_condition]
s82x_Fx_hard = s82x_Fx_hard[s82x_condition]
s82x_Fx_soft = s82x_Fx_soft[s82x_condition]
s82x_Nh = s82x_Nh[s82x_condition]
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
goodsN_auge = fits.open(path+'GOODsN_full_cat.fits')
goodsN_auge_data = goodsN_auge[1].data
goodsN_auge.close()

goodsN_auge_ID = goodsN_auge_data['id_xray']
goodsN_auge_Lx = goodsN_auge_data['Lx']
goodsN_auge_Lx_hard = goodsN_auge_data['Lx']*0.611
goodsN_auge_z = goodsN_auge_data['z_spec']

goodsN_auge_condition = (np.log10(goodsN_auge_Lx) >= Lx_min) & (np.log10(goodsN_auge_Lx) <= Lx_max) &(goodsN_auge_z > z_min) & (goodsN_auge_z <= z_max) & (goodsN_auge_z != 0.0)

goodsN_auge_ID_match = goodsN_auge_ID[goodsN_auge_condition]
goodsN_auge_Lx_match = goodsN_auge_Lx[goodsN_auge_condition]
goodsN_auge_Lx_hard_match = goodsN_auge_Lx_hard[goodsN_auge_condition]
goodsN_auge_z_match = goodsN_auge_z[goodsN_auge_condition]

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
goodsS_auge = fits.open(path+'GOODsS_full_cat.fits')
goodsS_auge_data = goodsS_auge[1].data
goodsS_auge.close()

goodsS_auge_ID = goodsS_auge_data['id_xray']
goodsS_auge_Lx = goodsS_auge_data['Lxc']
goodsS_auge_Lx_hard = goodsS_auge_data['Lxc']*0.611
goodsS_auge_z = goodsS_auge_data['z_spec']

goodsS_auge_condition = (np.log10(goodsS_auge_Lx) >= Lx_min) & (np.log10(goodsS_auge_Lx) <= Lx_max) &(goodsS_auge_z > z_min) & (goodsS_auge_z <= z_max) & (goodsS_auge_z != 0.0)

goodsS_auge_ID_match = goodsS_auge_ID[goodsS_auge_condition]
goodsS_auge_Lx_match = goodsS_auge_Lx[goodsS_auge_condition]
goodsS_auge_Lx_hard_match = goodsS_auge_Lx_hard[goodsS_auge_condition]
goodsS_auge_z_match = goodsS_auge_z[goodsS_auge_condition]

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
plt.savefig('/Users/connor_auge/Desktop/Final_plots/Lx_z_spec3.pdf')
plt.show()
'''
####################


#### Read in Galaxy Templates ####
temp = ascii.read('/Users/connor_auge/Desktop/templets/A10_templates.txt')
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


# Print time taken to read in all files
tfl = time.perf_counter()
print(f'Done with file reading ({tfl - ti:0.4f} second)')

# Filters used in COSMOS field 
# COSMOS_filters = np.array(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'G', 'R', 'I', 'Z', 'yHSC_FLUX_APER2', 'J_FLUX_APER2', 'H_FLUX_APER2',
                        #   'Ks_FLUX_APER2', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])

COSMOS_filters = np.array(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'G', 'R', 'I', 'Z', 'yHSC_FLUX_APER2', 'J_FLUX_APER2', 'H_FLUX_APER2',
                          'Ks_FLUX_APER2', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500',
                           'SCUBA2', 'VLA2'])
# Filters used in the S82X field
S82X_filters = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'MAG_FUV', 'MAG_NUV', 'U', 'G', 'R', 'I', 'Z',
                          'JVHS', 'HVHS', 'KVHS', 'W1', 'W2', 'W3', 'W4', 'nan', 'FLUX_250_s82x', 'FLUX_350_s82x', 'FLUX_500_s82x'])

# Filters used in the GOODS-N/S field
GOODSS_auge_filters = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'F435W', 'B_FLUX_APER2', 'V_FLUX_APER2', 'F606W', 'R', 'I',
                                  'F775W', 'F814W', 'Z', 'F850LP', 'F098M', 'F105W', 'F125W', 'JVHS', 'F140W', 'F160W', 'HVHS', 'KVHS', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'MIPS2', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])

GOODSN_auge_filters = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'F435W', 'B_FLUX_APER2', 'V_FLUX_APER2', 'F606W', 'R', 'I', 'F775W', 'F814W', 'Z', 'F105W', 'F125W', 'JVHS', 'F140W', 'F160W', 'HVHS', 'KVHS',
                                  'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'MIPS2', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])


# Filters for writting the output file
filter_total = np.array(['Fxh','Fxs','FUV','NUV','u','g','r','i','z','y','J','H','Ks','W1','IRAC1','IRAC2','W2','IRAC3','IRAC4','W3','W4','F24','F250','F350','F500'])
filter_COSMOS_match = np.array(['Fxh','Fxs','nan','FUV','NUV','u','g','r','i','z','y','J','H','Ks','IRAC1','IRAC2','IRAC3','IRAC4','F24','F250','F350','F500'])
filter_S82X_match = np.array(['Fxh','Fxs','nan','FUV','NUV','u','g','r','i','z','J','H','Ks','W1','W2','W3','W4','nan','F250','F350','F500'])

###################################################################################
###################################################################################
###################################################################################
########################## Run AGN Class over each source #########################

# Make empty lists to be filled with SED outputs 
out_ID, out_z, out_x, out_y, out_frac_error = [], [], [], [], []
out_Lx, out_Lx_hard = [], []
out_SED_shape = []
check_sed = []
norm = []
wfir_out, ffir_out = [], []
int_x, int_y = [],[]
FIR_upper_lims = []
F025, F1, F6, F10, F100 = [], [], [], [], []
xval_out = []
field = []
uv_slope, mir_slope1, mir_slope2 = [], [], []
Lbol_out, Lbol_sub_out = [], []
Nh = []
UV_lum_out, OPT_lum_out, MIR_lum_out, FIR_lum_out = [], [], [], []
###############################################################################
###############################################################################
############################### Run COSMOS SEDs ###############################
'''
fill_nan = np.zeros(len(GOODSS_auge_filters)-len(COSMOS_filters))
fill_nan[fill_nan == 0] = np.nan
for i in range(len(chandra_cosmos_phot_id_match)):
# for i in range(50):
    # if chandra_cosmos_phot_id_match[i] == 222544:
        source = AGN(chandra_cosmos_phot_id_match[i], chandra_cosmos_z_match[i], COSMOS_filters, cosmos_flux_array[i], cosmos_flux_err_array[i])
        source.MakeSED()
        source.FIR_extrap(['FLUX_24', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])

        ix, iy = source.Int_SED(xmin=1E-1, xmax=1E1)
        int_x.append(ix)
        int_y.append(iy)

        wfir, ffir, f100 = source.Int_SED_FIR(Find_value=100.0,discreet=True)
        wfir_out.append(wfir)
        ffir_out.append(ffir)

        lbol = source.Find_Lbol()
        lbol_sub = source.Find_Lbol_temp_sub(scale_array, temp_wave, temp_lum)
        shape = source.SED_shape()

        f1 = source.Find_value(1.0)
        xval = source.Find_value(3E-4)
        f6 = source.Find_value(6.0)
        f025 = source.Find_value(0.25)
        f10 = source.Find_value(10)

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
        out_z.append(chandra_cosmos_z_match[i])
        FIR_upper_lims.append(up_check)

        norm.append(f1)
        F025.append(f025)
        F1.append(f1)
        F6.append(f6)
        F10.append(f10)
        F100.append(f100)
        xval_out.append(xval)
        Lbol_out.append(lbol)
        Lbol_sub_out.append(lbol_sub)
        Nh.append(chandra_cosmos_Nh_match[i])

        uv_slope.append(source.Find_slope(0.15, 1.0))
        mir_slope1.append(source.Find_slope(1.0, 6.5))
        mir_slope2.append(source.Find_slope(6.5, 10))
        out_SED_shape.append(source.SED_shape())
        UV_lum_out.append(source.find_Lum_range(0.1,0.35))
        OPT_lum_out.append(source.find_Lum_range(0.35,3))
        MIR_lum_out.append(source.find_Lum_range(3,30))
        FIR_lum_out.append(source.find_Lum_range(30,500/(1+chandra_cosmos_z_match[i])))

        plot = Plotter(Id, redshift, w, f, chandra_cosmos_Lx_full_match[i],f1,up_check)

        check_sed.append(source.check_SED(10, check_span=2.75))
        field.append('c')
        # if check_sed[i] == 'GOOD':
        #     cols, data = source.output_properties('COSMOS',chandra_cosmos_xid_match[i],chandra_cosmos_RA_match[i],chandra_cosmos_DEC_match[i],chandra_cosmos_Lx_full_match[i],chandra_cosmos_Nh_match[i])
        #     source.write_output_file('AGN_Properties',data,cols,'w')

        #     cols, data = source.output_phot('COSMOS',filter_total,filter_COSMOS_match)
        #     source.write_output_file('AGN_photometry',data,cols,'w')
        # if Id == 167601:
            # plot.Plot_FIR_SED(wfir, ffir/f1)
            # plot.PlotSED(point_x=3E-4,point_y=xval/f1)
            # source.Find_Lbol()
    

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
for i in range(len(s82x_id)):
# for i in range(50):
    try:
        source = AGN(s82x_id[i],s82x_z[i],S82X_filters,s82x_flux_array[i],s82x_flux_err_array[i])
        source.MakeSED()
        source.FIR_extrap(['W4','FLUX_250_s82x', 'FLUX_350_s82x', 'FLUX_500_s82x'])

        ix, iy = source.Int_SED(xmin=1E-1, xmax=1E1)
        int_x.append(ix)
        int_y.append(iy)

        wfir, ffir, f100 = source.Int_SED_FIR(Find_value=100.0,discreet=True)
        wfir_out.append(wfir)
        ffir_out.append(ffir)

        lbol = source.Find_Lbol()
        lbol_sub = source.Find_Lbol_temp_sub(scale_array, temp_wave, temp_lum)
        shape = source.SED_shape()

        f1 = source.Find_value(1.0)
        xval = source.Find_value(3E-4)
        f6 = source.Find_value(6.0)
        f025 = source.Find_value(0.25)
        f10 = source.Find_value(10)

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
        F6.append(f6)
        F10.append(f10)
        F100.append(f100)
        xval_out.append(xval)
        Lbol_out.append(lbol)
        Lbol_sub_out.append(lbol_sub)
        Nh.append(s82x_Nh[i])

        uv_slope.append(source.Find_slope(0.15, 1.0))
        mir_slope1.append(source.Find_slope(1.0, 6.5))
        mir_slope2.append(source.Find_slope(6.5, 10))
        out_SED_shape.append(source.SED_shape())
        UV_lum_out.append(source.find_Lum_range(0.1,0.35))
        OPT_lum_out.append(source.find_Lum_range(0.35,3))
        MIR_lum_out.append(source.find_Lum_range(3,30))
        FIR_lum_out.append(source.find_Lum_range(30,500/(1+s82x_z[i])))


        plot = Plotter(Id, redshift, w, f, s82x_Lx_full[i],f1,up_check)

        # if Id == 2363:
        #     print(source.Find_slope(0.15,1.0))
        #     plot.PlotSED(point_x=3E-4,point_y=xval/f1)


        check_sed.append(source.check_SED(10, check_span=2.75))
        field.append('s')

    except ValueError:
        continue
# '''
ts = time.perf_counter()
print(f'Done with Stripe82X sources ({ts - tc:0.4f} second)')

###############################################################################
###############################################################################
############################## Run GOODS-N SEDs ###############################
'''
fill_nan = np.zeros(len(GOODSS_auge_filters)-len(GOODSN_auge_filters))
fill_nan[fill_nan == 0] = np.nan
for i in range(len(goodsN_auge_ID_match)):
    # for i in range(50):
    source = AGN(goodsN_auge_ID_match[i], goodsN_auge_z_match[i], GOODSN_auge_filters, goodsN_flux_array_auge[i], goodsN_flux_err_array_auge[i])
    source.MakeSED()
    source.FIR_extrap(['FLUX_24', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])

    ix, iy = source.Int_SED(xmin=1E-1, xmax=1E1)
    int_x.append(ix)
    int_y.append(iy)

    wfir, ffir, f100 = source.Int_SED_FIR(Find_value=100.0, discreet=True)
    wfir_out.append(wfir)
    ffir_out.append(ffir)

    lbol = source.Find_Lbol()
    lbol_sub = source.Find_Lbol_temp_sub(scale_array, temp_wave, temp_lum)
    shape = source.SED_shape()

    f1 = source.Find_value(1.0)
    xval = source.Find_value(3E-4)
    f6 = source.Find_value(6.0)
    f025 = source.Find_value(0.25)
    f10 = source.Find_value(10)

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
    F6.append(f6)
    F10.append(f10)
    F100.append(f100)
    xval_out.append(xval)
    Lbol_out.append(lbol)
    Lbol_sub_out.append(lbol_sub)
    Nh.append(0.0)

    uv_slope.append(source.Find_slope(0.15, 1.0))
    mir_slope1.append(source.Find_slope(1.0, 6.5))
    mir_slope2.append(source.Find_slope(6.5, 10))
    out_SED_shape.append(source.SED_shape())
    UV_lum_out.append(source.find_Lum_range(0.1,0.35))
    OPT_lum_out.append(source.find_Lum_range(0.35,3))
    MIR_lum_out.append(source.find_Lum_range(3,30))
    FIR_lum_out.append(source.find_Lum_range(30,500/(1+goodsN_auge_z_match[i])))

    plot = Plotter(Id, redshift, w, f, goodsN_auge_Lx_match[i], f1, up_check)

    check_sed.append(source.check_SED(10, check_span=2.75))
    field.append('gn')
    # if check_sed[i] == 'GOOD':
    #     cols, data = source.output_properties('COSMOS',chandra_cosmos_xid_match[i],chandra_cosmos_RA_match[i],chandra_cosmos_DEC_match[i],chandra_cosmos_Lx_full_match[i],chandra_cosmos_Nh_match[i])
    #     source.write_output_file('AGN_Properties',data,cols,'w')

    #     cols, data = source.output_phot('COSMOS',filter_total,filter_COSMOS_match)
    #     source.write_output_file('AGN_photometry',data,cols,'w')
    # if Id == 167601:
    # plot.Plot_FIR_SED(wfir, ffir/f1)
    # plot.PlotSED(point_x=3E-4,point_y=xval/f1)
    # source.Find_Lbol()


'''
tgn = time.perf_counter()
print(f'Done with GOODS-N sources ({tgn - ts:0.4f} second)')

###############################################################################
###############################################################################
############################## Run GOODS-S SEDs ###############################
'''
for i in range(len(goodsS_auge_ID_match)):
    # for i in range(50):
    try:
        source = AGN(goodsS_auge_ID_match[i], goodsS_auge_z_match[i], GOODSS_auge_filters, goodsS_flux_array_auge[i], goodsS_flux_err_array_auge[i])
        source.MakeSED()
        source.FIR_extrap(['FLUX_24', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])

        ix, iy = source.Int_SED(xmin=1E-1, xmax=1E1)
        int_x.append(ix)
        int_y.append(iy)

        wfir, ffir, f100 = source.Int_SED_FIR(Find_value=100.0, discreet=True)
        wfir_out.append(wfir)
        ffir_out.append(ffir)

        lbol = source.Find_Lbol()
        lbol_sub = source.Find_Lbol_temp_sub(scale_array, temp_wave, temp_lum)
        shape = source.SED_shape()

        f1 = source.Find_value(1.0)
        xval = source.Find_value(3E-4)
        f6 = source.Find_value(6.0)
        f025 = source.Find_value(0.25)
        f10 = source.Find_value(10)

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
        F6.append(f6)
        F10.append(f10)
        F100.append(f100)
        xval_out.append(xval)
        Lbol_out.append(lbol)
        Lbol_sub_out.append(lbol_sub)
        Nh.append(0.0)

        uv_slope.append(source.Find_slope(0.15, 1.0))
        mir_slope1.append(source.Find_slope(1.0, 6.5))
        mir_slope2.append(source.Find_slope(6.5, 10))
        out_SED_shape.append(source.SED_shape())
        UV_lum_out.append(source.find_Lum_range(0.1,0.35))
        OPT_lum_out.append(source.find_Lum_range(0.35,3))
        MIR_lum_out.append(source.find_Lum_range(3,30))
        FIR_lum_out.append(source.find_Lum_range(30,500/(1+goodsS_auge_z_match[i])))

        plot = Plotter(Id, redshift, w, f, goodsS_auge_Lx_match[i], f1, up_check)

        check_sed.append(source.check_SED(10, check_span=2.75))
        field.append('gs')
        # if check_sed[i] == 'GOOD':
        #     cols, data = source.output_properties('COSMOS',chandra_cosmos_xid_match[i],chandra_cosmos_RA_match[i],chandra_cosmos_DEC_match[i],chandra_cosmos_Lx_full_match[i],chandra_cosmos_Nh_match[i])
        #     source.write_output_file('AGN_Properties',data,cols,'w')

        #     cols, data = source.output_phot('COSMOS',filter_total,filter_COSMOS_match)
        #     source.write_output_file('AGN_photometry',data,cols,'w')
        # if Id == 167601:
        # plot.Plot_FIR_SED(wfir, ffir/f1)
        # if Id == 911:
            # print(source.Find_slope(0.1, 1.0))
            # plot.PlotSED(point_x=3E-4,point_y=xval/f1)
        # source.Find_Lbol()
    except ValueError:
        continue


'''
tgs = time.perf_counter()
print(f'Done with GOODS-S sources ({tgs - tgn:0.4f} second)')

# Make all output lists into arrays with only good sources
check_sed = np.asarray(check_sed)
GOOD_SED = check_sed == 'GOOD'

# Make all output lists into arrays and remove the bad SEDs
out_ID, out_z, out_x, out_y, out_frac_error = np.asarray(out_ID)[GOOD_SED], np.asarray(out_z)[GOOD_SED], np.asarray(out_x)[GOOD_SED], np.asarray(out_y)[GOOD_SED], np.asarray(out_frac_error)[[GOOD_SED]]
out_Lx, out_Lx_hard = np.log10(np.asarray(out_Lx)[GOOD_SED]), np.log10(np.asarray(out_Lx_hard)[GOOD_SED])
wfir_out, ffir_out = np.asarray(wfir_out)[GOOD_SED], np.asarray(ffir_out)[GOOD_SED]
int_x, int_y = np.asarray(int_x)[GOOD_SED], np.asarray(int_y)[GOOD_SED]
norm = np.asarray(norm)[GOOD_SED]
FIR_upper_lims = np.asarray(FIR_upper_lims)[GOOD_SED]
F025, F1, F6, F10, F100 = np.asarray(F025)[GOOD_SED], np.asarray(F1)[GOOD_SED], np.asarray(F6)[GOOD_SED], np.asarray(F10)[GOOD_SED], np.asarray(F100)[GOOD_SED]
xval_out = np.asarray(xval_out)[GOOD_SED]
field = np.asarray(field)[GOOD_SED]
out_SED_shape = np.asarray(out_SED_shape)[GOOD_SED]
uv_slope, mir_slope1, mir_slope2 = np.asarray(uv_slope)[GOOD_SED], np.asarray(mir_slope1)[GOOD_SED], np.asarray(mir_slope2)[GOOD_SED]
Lbol_out, Lbol_sub_out = np.asarray(Lbol_out)[GOOD_SED], np.asarray(Lbol_sub_out)[GOOD_SED]
Nh = np.asarray(Nh)[GOOD_SED]
UV_lum_out, OPT_lum_out, MIR_lum_out, FIR_lum_out = np.asarray(UV_lum_out)[GOOD_SED], np.asarray(OPT_lum_out)[GOOD_SED], np.asarray(MIR_lum_out)[GOOD_SED], np.asarray(FIR_lum_out)[GOOD_SED]

# for i in range(len(out_ID)):
#     print(out_ID[i],out_Lx[i],Lbol_sub_out[i])

# Sort all output data by the intrinsic X-ray luminosity
sort = out_Lx.argsort()
out_ID, out_z, out_x, out_y, out_frac_error = out_ID[sort], out_z[sort], out_x[sort], out_y[sort], out_frac_error[sort]
out_Lx, out_Lx_hard = out_Lx[sort], out_Lx_hard[sort]
wfir_out, ffir_out = wfir_out[sort], ffir_out[sort]
int_x, int_y = int_x[sort], int_y[sort]
norm = norm[sort]
FIR_upper_lims = FIR_upper_lims[sort]
F025, F1, F6, F10, F100 = F025[sort], F1[sort], F6[sort], F10[sort], F100[sort]
xval_out = xval_out[sort]
field = field[sort]
out_SED_shape = out_SED_shape[sort]
uv_slope, mir_slope1, mir_slope1 = uv_slope[sort], mir_slope1[sort], mir_slope2[sort]
Lbol_out, Lbol_sub_out = Lbol_out[sort], Lbol_sub_out[sort]
Nh = Nh[sort]
UV_lum_out, OPT_lum_out, MIR_lum_out, FIR_lum_out = UV_lum_out[sort], OPT_lum_out[sort], MIR_lum_out[sort], FIR_lum_out[sort]

print('Total GOOD SEDs')
print('Stripe82X: ', len(out_ID[field == 's']))
print('COSMOS: ', len(out_ID[field == 'c']))
print('GOODS-N: ', len(out_ID[field == 'gn']))
print('GOODS-S: ', len(out_ID[field == 'gs']))

# Begin Plotting
plot = Plotter(out_ID, out_z, out_x, out_y, out_Lx_hard, norm, FIR_upper_lims)
plot2 = Plotter_Letter2(out_ID, out_z, out_x, out_y, out_frac_error)
plot_shape = SED_shape_Plotter(out_ID, out_z, out_x, out_y, out_Lx, norm, FIR_upper_lims, out_SED_shape)

# plot = Plotter(out_ID[out_SED_shape == 1], out_z[out_SED_shape == 1],
            #    out_x[out_SED_shape == 1], out_y[out_SED_shape == 1], out_Lx[out_SED_shape == 1], norm[out_SED_shape == 1], FIR_upper_lims[out_SED_shape == 1])

# plt.figure(figsize=(8,8))
# plt.hist(np.log10(norm),bins=np.arange(39,46,0.25))
# plt.axvline(np.nanmedian(np.log10(norm)),c='k')
# plt.xlabel(r'L (1$\mu$m)')
# plt.ylim(0,300)
# plt.show()

# plt.figure(figsize=(8,8))
# plt.hist(np.log10(Lbol_sub_out),bins=np.arange(42,49,0.25))
# plt.axvline(np.nanmedian(np.log10(Lbol_sub_out)),c='k')
# plt.xlabel(r'L$_{\mathrm{bol}}$')
# plt.ylim(0,300)
# plt.show()

# plt.figure(figsize=(10,9))
# plt.hist(out_SED_shape, bins=np.arange(0, 7, 1))
# plt.xlabel('SED Shape')
# plt.ylim(0,400)
# plt.show()

# print(print(len(out_Lx[field == 's']), len(out_Lx[field == 'c']), len(out_Lx[field == 'gn']), len(out_Lx[field == 'gs'])))
# print(len(out_Lx))
# print(min(out_Lx),max(out_Lx))
# print(max(out_Lx[field == 's']), max(out_Lx[field == 'c']), max(out_Lx[field == 'gn']), max(out_Lx[field == 'gs']))

# plt.figure(figsize=(9,9))
# plt.hist(out_Lx,bins=np.arange(42.5,46,0.25),color='gray',alpha=0.3)
# plt.hist(out_Lx[field == 'c'],bins=np.arange(42.5,46,0.25),histtype='step',color='b',alpha=0.8,lw=5,label='COSMOS')
# plt.hist(out_Lx[field == 's'],bins=np.arange(42.5,46,0.25),histtype='step',color='r',alpha=0.8,lw=5,label='Stripe82X')
# plt.hist(out_Lx[(field == 'gn') | (field == 'gs')],bins=np.arange(42.5,46,0.25),histtype='step',color='k',alpha=0.7,lw=5,label='GOODS-N/S')
# plt.axvline(np.nanmean(out_Lx[field == 'c']), color = 'b', lw=3, ls='--', alpha=0.7)
# plt.axvline(np.nanmean(out_Lx[field == 's']), color = 'r', lw=3, ls='--', alpha=0.7)
# plt.axvline(np.nanmean(out_Lx[(field == 'gs') | (field == 'gn')]), color = 'k', lw=3, ls='--', alpha=0.8)
# plt.xlabel(r'log L$_{\mathrm{X}}$ [erg/s]')
# # plt.legend()
# plt.grid()
# plt.xlim(42.75,46)
# plt.savefig('/Users/connor_auge/Desktop/Final_Plots/Lx_sample2.pdf')
# plt.show()


# opt_x = np.ones(len(xval_out))
# opt_x[opt_x == 1.] = 3E-4
# plot.multi_SED('All',median_x=int_x,median_y=int_y,wfir=wfir_out,ffir=ffir_out,Median_line=True,FIR_upper='upper lims')
# plot.multi_SED('Shape1',median_x=int_x[out_SED_shape == 1],median_y=int_y[out_SED_shape == 1],wfir=wfir_out[out_SED_shape == 1],ffir=ffir_out[out_SED_shape == 1],Median_line=True,FIR_upper='upper lims')
# plot.multi_SED_bins('All_z_bins',bin='redshift',field=field,median_x=int_x,median_y=int_y,wfir=wfir_out,ffir=ffir_out,Median_line=True,FIR_upper='upper lims')
# plot.multi_SED_bins('All_z_field',bin='field',field=field,median_x=int_x,median_y=int_y,wfir=wfir_out,ffir=ffir_out,Median_line=True,FIR_upper='upper lims')
# plot.median_SED_plot('All_median_SEDs2', median_x=int_x, median_y=int_y, wfir=wfir_out, ffir=ffir_out, shape=out_SED_shape, FIR_upper='upper lims')
# plot.median_SED_1panel('median_shape1', median_x=int_x, median_y=int_y, wfir=wfir_out, ffir=ffir_out, shape=out_SED_shape, FIR_upper='upper lims')
# plot_shape.shape_1bin_h('horizantal_5_panel3',median_x=int_x,median_y=int_y,wfir=wfir_out,ffir=ffir_out,uv_slope=uv_slope,mir_slope1=mir_slope1,mir_slope2=mir_slope2,Median_line=True,FIR_upper='upper lims')
# plot.plot_medians('new_med_plot',F1,F025,F6,F100)

# plot_shape.shape_1bin_v('goodss_vertical_5_panel',median_x=int_x,median_y=int_y,wfir=wfir_out,ffir=ffir_out,uv_slope=uv_slope,mir_slope1=mir_slope1,mir_slope2=mir_slope2,Median_line=True,FIR_upper='upper lims')
# plot.L_ratio_3panels('AGN_Lx_ratio5','Lx','UV-MIR-FIR','X-axis',F1,UV_lum_out,MIR_lum_out,FIR_lum_out,shape=out_SED_shape,L=Lbol_sub_out)
# plot.L_ratio_3panels('AGN_Lx_ratio3','Lx','UV-MIR-FIR','X-axis',F1,F025,F6,F100,shape=out_SED_shape,L=Lbol_sub_out)
# plot.L_ratio_1panel('Lx_Lbol_ratio2','Lx','Lbol/Lx','X-axis',F1,F025,F6,F100,shape=out_SED_shape,L=Lbol_sub_out)


# plot.L_scatter_1panel('Nh_Lx','Lx','Nh','X-axis',F1,F025,F6,F100,Nh,shape=out_SED_shape,L=Lbol_sub_out)
plot.L_scatter_1panel('MIR_Lx_test','MIR','Lx','X-axis',F1,F025,F6,F100,Nh,shape=out_SED_shape,L=Lbol_sub_out)

# plot.L_hist('Lone_hist2',np.log10(F1),r'log L (1 $\mu$m) [erg/s]',[41.5,46],[41.5,46,0.25],median=True,std=False)
# plot.L_hist('all_Lx_hist',out_Lx,r'log L$_{\mathrm{X}}$ [erg/s]',[42.5,47],[42.5,47,0.25])
# plot.L_hist('Nh_hist',np.log10(Nh),r'log N$_{\mathrm{H}}$ [cm$^{-2}$]',[19.5,25],[19.5,24.5,0.25],median=True,std=False)
# plot.L_hist('all_Lbol',np.log10(Lbol_sub_out),r'log L$_{\mathrm{bol}}$',[43,48],[43,48,0.25])
# plot.L_hist('Lone_hist2',np.log10(F1),r'log L (1 $\mu$m) [erg/s]',[41.5,46],[41.5,46,0.25],median=True,std=False)
# plot.L_hist_zbins('Lone_hist_zbins',np.log10(F1),r'log L (1 $\mu$m) [erg/s]',[41.5,46],[41.5,46,0.25],median=True,std=False)
# plot.L_hist('Lx_sample',out_Lx,r'log L$_{\mathrm{X}}$ [erg/s]',[42.5,46],[42.5,46,0.25],median=True,std=False)


# print(out_Lx)
# print(F025)
# print(norm)
# print(np.log10(F025))
# print(np.log10(F025/norm))

# print(np.nanmedian(np.log10(Lbol_sub_out[out_SED_shape == 1])))
# print(np.nanmedian(np.log10(Lbol_sub_out[out_SED_shape == 2])))
# print(np.nanmedian(np.log10(Lbol_sub_out[out_SED_shape == 3])))
# print(np.nanmedian(np.log10(Lbol_sub_out[out_SED_shape == 4])))
# print(np.nanmedian(np.log10(Lbol_sub_out[out_SED_shape == 5])))


# plot2.Upanels_ratio_plots('Lum_Lbol','Lbol','UV-MIR-FIR/Lbol','Bins',Nh,out_Lx,np.log10(Lbol_sub_out),np.log10(UV_lum_out),np.log10(MIR_lum_out),np.log10(FIR_lum_out),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,out_z,uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,shape=out_SED_shape)
# plot2.Box_1panel('Lx_box_1panel', 'Lx', out_Lx, uv_slope, mir_slope1, mir_slope2,shape=out_SED_shape)
# plot2.Box_1panel('Lbol_Lx_box_1panel_sub2', 'Lbol/Lx', np.log10(Lbol_sub_out)-out_Lx, uv_slope, mir_slope1, mir_slope2, shape=out_SED_shape)
# plot2.Box_1panel('Lbol_box_1panel_sub3', 'Lbol', np.log10(Lbol_sub_out), uv_slope, mir_slope1, mir_slope2, shape=out_SED_shape)
# plot2.scatter_1panel('UV_Lx_Lx_norm_new','Lx','UV/Lx','None','Both',Nh,out_Lx,np.log10(Lbol_sub_out),F1,np.log10(F025/norm),np.log10(F6/norm),np.log10(F100/norm),np.log10(F10/norm),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims)
# plot2.scatter_1panel('new/UV_FIR_1panel','FIR','UV','Both','Bins',Nh,out_Lx,np.log10(Lbol_sub_out),F1,np.log10(F025/F1),np.log10(F6/F1),np.log10(F100/F1),np.log10(F10/F1),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims)
# plot2.scatter_1panel('new/FIR_Lx_1panel', 'Lx', 'FIR', 'Y-axis', 'Bins', Nh, out_Lx, np.log10(Lbol_sub_out), F1, np.log10(F025/F1), np.log10(F6/F1), np.log10(F100/F1), np.log10(F10/F1), uv_slope, mir_slope1, mir_slope2, FIR_upper_lims)
# plot2.scatter_1panel('new/Lx_Lbol_1panel_sub_xmed_lbol', 'Lbol', 'Lbol/Lx', 'None', 'Both', Nh, out_Lx, np.log10(Lbol_sub_out), F1, np.log10(F025), np.log10(F6), np.log10(F100), np.log10(F10), uv_slope, mir_slope1, mir_slope2, FIR_upper_lims, shape = out_SED_shape, durras=True)
# plot2.scatter_1panel('UV_MIR','MIR6','UV/MIR6','None','Bins',Nh,out_Lx,np.log10(Lbol_sub_out),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,out_SED_shape)
# plot2.scatter_1panel('UV_Lx','Lx','UV/Lx','None','Bins',Nh,out_Lx,np.log10(Lbol_sub_out),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,out_SED_shape)
# plot2.scatter_1panel('MIR_Lx','Lx','MIR6/Lx','None','Bins',Nh,out_Lx,np.log10(Lbol_sub_out),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,out_SED_shape)
# plot2.scatter_1panel('FIR_Lx','Lx','FIR/Lx','None','Bins',Nh,out_Lx,np.log10(Lbol_sub_out),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,out_SED_shape)
# plot2.scatter_1panel('UV_FIR','FIR','UV/FIR','None','Bins',Nh,out_Lx,np.log10(Lbol_sub_out),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,out_SED_shape)
# plot2.scatter_1panel('Lx_Lbol_shape','Lbol','Lx/Lbol','None','Bins',Nh,out_Lx,np.log10(Lbol_sub_out),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),uv_slope,mir_slope1,mir_slope2,FIR_upper_lims,out_SED_shape)
