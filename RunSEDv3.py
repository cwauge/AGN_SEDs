import time
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io import ascii
from filters import Filters
from SED_v7 import AGN
from SED_plots import Plotter
from plots_Letter import Plotter_Letter
from plots_Letter2 import Plotter_Letter2
from SED_plots_v2 import Plotter as Plotter_v2
from SED_shape_plots import SED_shape_Plotter
from match import match 
from mag_flux import mag_to_flux
from mag_flux import magerr_to_fluxerr

ti = time.perf_counter()

path = '/Users/connor_auge/Research/Disertation/catalogs/' # Path for photometry catalogs

# set redshift and X-ray luminosity limits
z_min = 0.0
z_max = 1.2

Lx_min = 43
Lx_max = 50

goals_Lx_min = 35

###################################################################################
###################################################################################
############################ Read in COSMOS2020 files #############################

# Read in data
cosmos = fits.open(path+'cosmos2020/classic/COSMOS2020_CLASSIC_R1_v2.0_master.fits')
cosmos_data = cosmos[1].data
cosmos.close()

chandra_cosmos = fits.open(path+'chandra_COSMOS_legacy_opt_NIR_counterparts_20160113_4d.fits')
chandra_cosmos_data = chandra_cosmos[1].data
chandra_cosmos.close()

chandra_cosmos2 = fits.open(path+'chandra_cosmos_legacy_spectra_bestfit_20210225.fits')
chandra_cosmos2_data = chandra_cosmos2[1].data
chandra_cosmos2.close()

chandra_cosmos_ct = fits.open(path+'chandra_cosmos_legacy_spectra_bestfit_ComptonThick_Lanzuisi18.fits')
chandra_cosmos_ct_data = chandra_cosmos_ct[1].data
chandra_cosmos_ct.close()

deimos = ascii.read('/Users/connor_auge/Downloads/deimos_10k_March2018_new/deimos_redshifts.tbl')
deimos_id = np.asarray(deimos['ID'])
deimos_z = np.asarray(deimos['zspec'])
deimos_remarks = np.asarray(deimos['Remarks'])
deimos_ID = np.asarray([int(i[1:]) for i in deimos_id if 'L' in i])
deimos_z_spec = np.asarray([deimos_z[i] for i in range(len(deimos_z)) if 'L' in deimos_id[i]])

# cosmos_cutouts = os.listdir('./cosmos_cutouts/')

chandra_cosmos_phot_id = chandra_cosmos_data['id_k_uv']
cosmos_laigle_id = cosmos_data['ID_COSMOS2015']
cosmos_xid = cosmos_data['id_chandra']
chandra_cosmos_xid = chandra_cosmos_data['id_x']
chandra_cosmos2_xid = chandra_cosmos2_data['id_x']
chandra_cosmos_ct_xid = chandra_cosmos_ct_data['id_x']

chandra_cosmos_RA = chandra_cosmos_data['RA_x']
chandra_cosmos_DEC = chandra_cosmos_data['DEC_x']

cosmos_sz = cosmos_data['sz_zspec']
cosmos_ez = cosmos_data['ez_z_spec']
chandra_cosmos_z = chandra_cosmos_data['z_spec']
chandra_cosmos_z_phot = chandra_cosmos_data['z_best']
chandra_cosmos_Fx_hard = chandra_cosmos_data['flux_h']
chandra_cosmos_Fx_soft = chandra_cosmos_data['flux_s']
chandra_cosmos_Fx_full = chandra_cosmos_data['flux_f']
chandra_cosmos_Lx_hard = np.asarray([10**i for i in chandra_cosmos_data['Lx_210']])
chandra_cosmos_Lx_soft = np.asarray([10**i for i in chandra_cosmos_data['Lx_052']])
chandra_cosmos_Lx_full = np.asarray([10**i for i in chandra_cosmos_data['Lx_0510']])
chandra_cosmos_spec_type = chandra_cosmos_data['spec_type']
chandra_cosmos_nh = chandra_cosmos_data['Nh']
chandra_cosmos_nh_lo = chandra_cosmos_data['Nh_lo']
chandra_cosmos_nh_hi = chandra_cosmos_data['Nh_up']
chandra_cosmos_abs_corr_h = chandra_cosmos_data['abs_corr_210']
chandra_cosmos_abs_corr_up_h = chandra_cosmos_data['abs_corr_210_up']
chandra_cosmos_abs_corr_lo_h = chandra_cosmos_data['abs_corr_210_lo']
chandra_cosmos_abs_corr_s = chandra_cosmos_data['abs_corr_052']
chandra_cosmos_abs_corr_up_s = chandra_cosmos_data['abs_corr_052_up']
chandra_cosmos_abs_corr_lo_s = chandra_cosmos_data['abs_corr_052_lo']
chandra_cosmos_abs_corr_f = chandra_cosmos_data['abs_corr_0510']
chandra_cosmos_abs_corr_up_f = chandra_cosmos_data['abs_corr_0510_up']
chandra_cosmos_abs_corr_lo_f = chandra_cosmos_data['abs_corr_0510_lo']
chandra_cosmos2_Lx_hard = np.asarray([10**i for i in chandra_cosmos2_data['Lx_210']])
chandra_cosmos2_Lx_full = np.asarray([(10**i)*1.64 for i in chandra_cosmos2_data['Lx_210']])
chandra_cosmos2_z = chandra_cosmos2_data['z_best']
chandra_cosmos2_nh = chandra_cosmos2_data['nh']
chandra_cosmos2_nh_lo_err = chandra_cosmos2_data['nh_lo_err']
chandra_cosmos2_nh_up_err = chandra_cosmos2_data['nh_up_err']
chandra_cosmos2_Fx_hard = chandra_cosmos2_data['flux_210']
chandra_cosmos2_Fx_soft = chandra_cosmos2_data['flux_052']
chandra_cosmos2_Fx_full = chandra_cosmos2_data['flux_0510']

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
    # if chandra_cosmos_z[i] < 0:
    #     chandra_cosmos_z[i] = chandra_cosmos_z_phot[i]
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
     
abs_corr_use_h = np.asarray(abs_corr_use_h)
abs_corr_use_s = np.asarray(abs_corr_use_s)
abs_corr_use_f = np.asarray(abs_corr_use_f)

chandra_cosmos_Lx_hard /= abs_corr_use_h
chandra_cosmos_Lx_soft /= abs_corr_use_s
chandra_cosmos_Lx_full /= abs_corr_use_f

chandra_cosmos_Nh = []
check = []
for i in range(len(chandra_cosmos_Lx_full)):
    ind = np.where(chandra_cosmos2_xid == chandra_cosmos_xid[i])[0]
    ind_ct = np.where(chandra_cosmos_ct_xid == chandra_cosmos_xid[i])[0]

    if len(ind_ct) > 0:
        chandra_cosmos_Nh.append(chandra_cosmos_ct_nh[ind_ct][0])
        chandra_cosmos_Lx_hard[i] = chandra_cosmos_ct_Lx_hard[ind_ct]
        chandra_cosmos_Lx_full[i] = chandra_cosmos_ct_Lx_full[ind_ct]
        check.append(3)

    elif len(ind) > 0:
        chandra_cosmos_Lx_hard[i] = chandra_cosmos2_Lx_hard[ind]
        chandra_cosmos_Lx_full[i] = chandra_cosmos2_Lx_full[ind]
        if chandra_cosmos2_nh_lo_err[ind][0] == -99.:
            chandra_cosmos_Nh.append(chandra_cosmos2_nh[ind][0]+chandra_cosmos2_nh_up_err[ind][0])
            check.append(2.5)
        else:
            chandra_cosmos_Nh.append(chandra_cosmos2_nh[ind][0])
            check.append(2)
    else:
        if chandra_cosmos_nh[i] == -99.:
            if chandra_cosmos_nh_lo[i] != -99.:
                chandra_cosmos_Nh.append(chandra_cosmos_nh_lo[i])
            else:
                chandra_cosmos_Nh.append(chandra_cosmos_nh_hi[i])
        else:    
            chandra_cosmos_Nh.append(chandra_cosmos_nh[i])
        check.append(1)
chandra_cosmos_Nh = np.asarray(chandra_cosmos_Nh)*1E22
check = np.asarray(check)

# print(chandra_cosmos_Nh)
# for i in range(len(chandra_cosmos_Lx_full)):
#     if chandra_cosmos_nh[i] == -99.:
#         if chandra_cosmos_nh_lo[i] != -99.:
#             chandra_cosmos_Nh.append(chandra_cosmos_nh_lo[i])
#         else:
#             chandra_cosmos_Nh.append(chandra_cosmos_nh_hi[i])
#     else:    
#         chandra_cosmos_Nh.append(chandra_cosmos_nh[i])
# chandra_cosmos_Nh = np.asarray(chandra_cosmos_Nh)*1E2


print('COSMOS All Lx cat: ',len(chandra_cosmos_Lx_full))


# Select COSMOS sources from the Chandra catalog based on conditions listed at the top of the file 
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


# Select COSMOS sources from the Chandra catalog that have spectral types
cosmos_spec_type_cond = np.logical_or(chandra_cosmos_spec_type == 1, chandra_cosmos_spec_type == 2)

# chandra_cosmos_phot_id = chandra_cosmos_phot_id[cosmos_spec_type_cond]
# chandra_cosmos_xid = chandra_cosmos_xid[cosmos_spec_type_cond]
# chandra_cosmos_RA = chandra_cosmos_RA[cosmos_spec_type_cond]
# chandra_cosmos_DEC = chandra_cosmos_DEC[cosmos_spec_type_cond]
# chandra_cosmos_z = chandra_cosmos_z[cosmos_spec_type_cond]
# chandra_cosmos_Fx_full = chandra_cosmos_Fx_full[cosmos_spec_type_cond]
# chandra_cosmos_Fx_hard = chandra_cosmos_Fx_hard[cosmos_spec_type_cond]
# chandra_cosmos_Fx_soft = chandra_cosmos_Fx_soft[cosmos_spec_type_cond]
# chandra_cosmos_Lx_full = chandra_cosmos_Lx_full[cosmos_spec_type_cond]
# chandra_cosmos_Lx_hard = chandra_cosmos_Lx_hard[cosmos_spec_type_cond]
# chandra_cosmos_Lx_soft = chandra_cosmos_Lx_soft[cosmos_spec_type_cond]
# chandra_cosmos_spec_type = chandra_cosmos_spec_type[cosmos_spec_type_cond]
# chandra_cosmos_Nh = chandra_cosmos_Nh[cosmos_spec_type_cond]
# print('COSMOS Lx z spec type: ',len(chandra_cosmos_phot_id))

############### Subset match ###############
'''
subset_id = np.asarray([262286,673312,818825,859825,950643,215847,385450,385450,935830])

cosmos_subset_ix, cosmos_supbset_iy = match(chandra_cosmos_phot_id,subset_id)

chandra_cosmos_phot_id = chandra_cosmos_phot_id[cosmos_subset_ix]
chandra_cosmos_xid = chandra_cosmos_xid[cosmos_subset_ix]
chandra_cosmos_RA = chandra_cosmos_RA[cosmos_subset_ix]
chandra_cosmos_DEC = chandra_cosmos_DEC[cosmos_subset_ix]
chandra_cosmos_z = chandra_cosmos_z[cosmos_subset_ix]
chandra_cosmos_Fx_full = chandra_cosmos_Fx_full[cosmos_subset_ix]
chandra_cosmos_Fx_hard = chandra_cosmos_Fx_hard[cosmos_subset_ix]
chandra_cosmos_Fx_soft = chandra_cosmos_Fx_soft[cosmos_subset_ix]
chandra_cosmos_Lx_full = chandra_cosmos_Lx_full[cosmos_subset_ix]
chandra_cosmos_Lx_hard = chandra_cosmos_Lx_hard[cosmos_subset_ix]
chandra_cosmos_Lx_soft = chandra_cosmos_Lx_soft[cosmos_subset_ix]
chandra_cosmos_spec_type = chandra_cosmos_spec_type[cosmos_subset_ix]
chandra_cosmos_Nh = chandra_cosmos_Nh[cosmos_subset_ix]
'''
############## ############## ##############

# Match data from COSMOS Chandra catalog to data from the COSMOS2020 photometry catalog 
# cosmos_ix, cosmos_iy = match(chandra_cosmos_xid, cosmos_xid) # Match Chandra and COSMOS data with X-ray id
cosmos_ix, cosmos_iy = match(chandra_cosmos_phot_id, cosmos_laigle_id)

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

chandra_cosmos_Fx_hard_match_mjy = chandra_cosmos_Fx_hard_match*4.136E8/(10-2)
chandra_cosmos_Fx_soft_match_mjy = chandra_cosmos_Fx_soft_match*4.136E8/(2-0.5)
chandra_cosmos_Fx_full_match_mjy = chandra_cosmos_Fx_full_match*4.136E8/(10-0.5)

cosmos_nan_array = np.zeros(np.shape(cosmos_laigle_id_match))

cosmos_flux_array = np.array([
    chandra_cosmos_Fx_hard_match_mjy*1000,chandra_cosmos_Fx_soft_match_mjy*1000,
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
    cosmos_data['FIR_500_FLUX'][cosmos_iy]
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
    cosmos_data['FIR_500_FLUXERR'][cosmos_iy]
])

cosmos_flux_array = cosmos_flux_array.T
cosmos_flux_err_array = cosmos_flux_err_array.T
###############################################################################


###############################################################################
###############################################################################
########################### Read in Stripe82X files ###########################

eboss = fits.open('/Users/connor_auge/Desktop/desktop_catalogs/lamassa2019.fit')
eboss_data = eboss[1].data
eboss.close()

lamassa = fits.open(path+'S82X_catalog_with_photozs_unique_Xraysrcs_likely_cps_w_mbh.fits')
lamassa_cols = lamassa[1].columns
lamassa_data = lamassa[1].data
lamassa.close()

# peca = ascii.read('/Users/connor_auge/Desktop/desktop_catalogs/s82_spec_results/Auge_spec_results_safe.txt')
peca = fits.open(path+'Peca_S82X.fits')
peca_cols = peca[1].columns
peca_data = peca[1].data
peca.close()

unwise = ascii.read('/Users/connor_auge/Desktop/desktop_catalogs/unwise_matches.csv')

# peca_ID = np.asarray(peca['ID'])
# peca_Lx_full = np.asarray(peca['lumin_f'])
# peca_Lx_hard = np.asarray(peca['lumin_h'])
# peca_Lx_full_obs = np.asarray(peca['lumin_of'])
# peca_Nh = np.asarray(peca['nh'])

peca_ID = peca_data['srcid']
peca_Lx_full = peca_data['lumin_f']
peca_Lx_hard = peca_data['lumin_h']
peca_Lx_full_obs = peca_data['lumin_of']
peca_Nh = peca_data['nh']

unwise_ID = np.asarray(unwise['ID'])
# unwise_W1 = mag_to_flux(np.asarray(unwise['unW1']),'W1')*1E6
# unwise_W1_err = magerr_to_fluxerr(np.asarray(unwise['unW1']),np.asarray(unwise['unW1_err']),'W1')*1E6
# unwise_W2 = mag_to_flux(np.asarray(unwise['unW2']),'W2')*1E6
# unwise_W2_err = magerr_to_fluxerr(np.asarray(unwise['unW2']),np.asarray(unwise['unW2_err']),'W2')*1E6
# unwise_W3 = mag_to_flux(np.asarray(unwise['unW3']),'W3')*1E6
# unwise_W3_err = magerr_to_fluxerr(np.asarray(unwise['unW3']),np.asarray(unwise['unW3_err']),'W3')*1E6
# unwise_W4 = mag_to_flux(np.asarray(unwise['unW4']),'W4')*1E6
# unwise_W4_err = magerr_to_fluxerr(np.asarray(unwise['unW4']),np.asarray(unwise['unW4_err']),'W4')*1E6
unwise_W1 = np.asarray(unwise['unW1'])
unwise_W1_err = np.asarray(unwise['unW1_err'])
unwise_W2 = np.asarray(unwise['unW2'])
unwise_W2_err = np.asarray(unwise['unW2_err'])
unwise_W3 = np.asarray(unwise['unW3'])
unwise_W3_err = np.asarray(unwise['unW3_err'])
unwise_W4 = np.asarray(unwise['unW4'])
unwise_W4_err = np.asarray(unwise['unW4_err'])


lamassa_id = lamassa_data['msid']
lamassa_id2 = lamassa_data['rec_no']
lamassa_id3 = lamassa_data['OBSID']
lamassa_z = lamassa_data['SPEC_Z']
lamassa_z_phot = lamassa_data['PHOTO_Z']

lamassa_id_use = []
lamassa_z_use = []
for i in range(len(lamassa_id)):
    if lamassa_id[i] == 0:
        lamassa_id_use.append(lamassa_id2[i])
    else:
       lamassa_id_use.append(lamassa_id[i])
    # if lamassa_z[i] <=0:
    #     lamassa_z_use.append(lamassa_z_phot[i])
    # else:
    #     lamassa_z_use.append(lamassa_z[i])
lamassa_id_use = np.asarray(lamassa_id_use)
# lamassa_z = np.asarray(lamassa_z_use)

# ID1 = lamassa_data['MSID']
# ID2 = lamassa_data['REC_NO']
# ID3 = lamassa_data['OBSID']
# lamassa_z = lamassa_data['spec_z']
# lamassa_ra = lamassa_data['XRAY_RA']
# lamassa_id = []
# for i in range(len(ID1)):
#     if ID1[i] == 0:
#         lamassa_id.append(ID2[i])
#     else:
#         lamassa_id.append(ID1[i])
# lamassa_id_use = np.asarray(lamassa_id)


lamassa_ra = lamassa_data['XRAY_RA']
lamassa_dec = lamassa_data['XRAY_DEC']
lamassa_cat = lamassa_data['XRAY_SRC']

s82x_Lx_sp_full = np.asarray([10**i for i in lamassa_data['FULL_LUM']])
s82x_Lx_sp_hard = np.asarray([10**i for i in lamassa_data['HARD_LUM']])
s82x_Fx_hard = lamassa_data['HARD_FLUX']
s82x_Fx_soft = lamassa_data['SOFT_FLUX']
s82x_Fx_full = lamassa_data['FULL_FLUX']
# s82x_W1 = mag_to_flux(lamassa_data['W1'],'W1')*1E6
# s82x_W2 = mag_to_flux(lamassa_data['W2'],'W2')*1E6
# s82x_W3 = mag_to_flux(lamassa_data['W3'],'W3')*1E6
# s82x_W4 = mag_to_flux(lamassa_data['W4'],'W4')*1E6
# s82x_W1_err = magerr_to_fluxerr(lamassa_data['W1'],lamassa_data['W1_err'],'W1')*1E6
# s82x_W2_err = magerr_to_fluxerr(lamassa_data['W2'],lamassa_data['W2_err'],'W2')*1E6
# s82x_W3_err = magerr_to_fluxerr(lamassa_data['W3'],lamassa_data['W3_err'],'W3')*1E6
# s82x_W4_err = magerr_to_fluxerr(lamassa_data['W4'],lamassa_data['W4_err'],'W4')*1E6
# s82x_ch1_spies = mag_to_flux(lamassa_data['CH1_SPIES'],'Ch1Spies')*1E6
# s82x_ch2_spies = mag_to_flux(lamassa_data['CH2_SPIES'],'Ch2Spies')*1E6
# s82x_ch1_shela = mag_to_flux(lamassa_data['CH1_SHELA'],'Ch1Spies')*1E6
# s82x_ch2_shela = mag_to_flux(lamassa_data['CH2_SHELA'],'Ch2Spies')*1E6
# s82x_ch1_spies_err = magerr_to_fluxerr(lamassa_data['CH1_SPIES'],lamassa_data['CH1_SPIES_ERR'], 'Ch1Spies')*1E6
# s82x_ch2_spies_err = magerr_to_fluxerr(lamassa_data['CH2_SPIES'],lamassa_data['CH2_SPIES_ERR'], 'Ch2Spies')*1E6
# s82x_ch1_shela_err = magerr_to_fluxerr(lamassa_data['CH1_SHELA'],lamassa_data['CH1_SHELA_ERR'],'Ch1Spies')*1E6
# s82x_ch2_shela_err = magerr_to_fluxerr(lamassa_data['CH2_SHELA'],lamassa_data['CH2_SHELA_ERR'],'Ch2Spies')*1E6
s82x_spec_class = lamassa_data['SPEC_CLASS']

s82x_W1 = lamassa_data['W1']
s82x_W2 = lamassa_data['W2']
s82x_W3 = lamassa_data['W3']
s82x_W4 = lamassa_data['W4']
s82x_W1_err = lamassa_data['W1_err']
s82x_W2_err = lamassa_data['W2_err']
s82x_W3_err = lamassa_data['W3_err']
s82x_W4_err = lamassa_data['W4_err']

s82x_spec_class[s82x_spec_class == 'QSO'] = '1'
s82x_spec_class[s82x_spec_class == 'QSO_BAL'] = '1'
s82x_spec_class[s82x_spec_class == 'QSO(BA'] = '1'
s82x_spec_class[s82x_spec_class == 'GALAXY'] = '2'
s82x_spec_class[s82x_spec_class == 'AGN'] = '1'
s82x_spec_class[s82x_spec_class == 'N/A'] = '3'
s82x_spec_class[s82x_spec_class == 'STAR'] = '3'
s82x_spec_class[s82x_spec_class == 'NELG'] = '3'
s82x_spec_class[s82x_spec_class == '    '] = '3'
s82x_spec_class[s82x_spec_class == ''] = '3'
s82x_spec_class = np.asarray(s82x_spec_class)


for i in range(len(lamassa_id_use)):
    ind = np.where(unwise_ID == lamassa_id_use[i])[0]
    if len(ind) > 0:
        if np.isnan(unwise_W3[ind]):
            continue
        elif unwise_W3[ind] <= 0.0:
            continue
        elif np.isnan(unwise_W3[ind]):
            continue
        elif (magerr_to_fluxerr(unwise_W3[ind], unwise_W3_err[ind], 'W3', AB=True)/mag_to_flux(unwise_W3[ind], 'W3')) > 0.4:
            continue
        else:
            s82x_W3[i] = unwise_W3[ind][0]
            s82x_W3_err[i] = unwise_W3_err[ind][0]

        if np.isnan(unwise_W4[ind]):
            continue
        elif unwise_W4[ind] <= 0.0:
            continue
        elif np.isnan(unwise_W4[ind]):
            continue
        elif (magerr_to_fluxerr(unwise_W4[ind], unwise_W4_err[ind], 'W4', AB=True)/mag_to_flux(unwise_W4[ind], 'W4')) > 0.4:
            continue
        else:
            s82x_W4[i] = unwise_W4[ind][0]
            s82x_W4_err[i] = unwise_W4_err[ind][0]

# s82x_W1 = mag_to_flux(s82x_W1,'W1')*1E6
# s82x_W2 = mag_to_flux(s82x_W2,'W2')*1E6
# s82x_W3 = mag_to_flux(s82x_W3,'W3')*1E6
# s82x_W4 = mag_to_flux(s82x_W4,'W4')*1E6

# s82x_W1_err = magerr_to_fluxerr(s82x_W1,s82x_W1_err,'W1')*1E6
# s82x_W2_err = magerr_to_fluxerr(s82x_W2,s82x_W2_err,'W2')*1E6
# s82x_W3_err = magerr_to_fluxerr(s82x_W3,s82x_W3_err,'W3')*1E6
# s82x_W4_err = magerr_to_fluxerr(s82x_W4,s82x_W4_err,'W4')*1E6

s82x_Nh = []
for i in range(len(lamassa_id_use)):
    ind = np.where(peca_ID == lamassa_id_use[i])[0]
    if len(ind) > 0:
        s82x_Lx_sp_full[i] = peca_Lx_full[ind]
        # s82x_Lx_use[i] = peca_Lx_full[ind]
        s82x_Nh.append(peca_Nh[ind][0])

    else:
        s82x_Nh.append(0.0)

s82x_Nh = np.asarray(s82x_Nh)

# s82x_irac1, s82x_irac2 = [], []
# s82x_irac1_err, s82x_irac2_err = [], []
# for i in range(len(lamassa_id_use)):
#     if s82x_ch1_spies[i] > 0:
#         s82x_irac1.append(s82x_ch1_spies[i])
#         s82x_irac1_err.append(s82x_ch1_spies_err[i])
#     else:
#         s82x_irac1.append(s82x_ch1_shela[i])
#         s82x_irac1_err.append(s82x_ch1_shela_err[i])
#     if s82x_ch2_spies[i] > 0:
#         s82x_irac2.append(s82x_ch2_spies[i])
#         s82x_irac2_err.append(s82x_ch2_spies_err[i])
#     else:
#         s82x_irac2.append(s82x_ch2_shela[i])
#         s82x_irac2_err.append(s82x_ch2_shela_err[i])

# s82x_irac1, s82x_irac2 = np.asarray(s82x_irac1), np.asarray(s82x_irac2)
# s82x_irac1_err, s82x_irac2_err = np.asarray(s82x_irac1_err), np.asarray(s82x_irac2_err)




# mir1, mir2 = [], []
# mir1_err, mir2_err = [], []
# for i in range(len(lamassa_id_use)):
#     # if s82x_irac1[i] > 0:
#     #     mir1.append(s82x_irac1[i])
#     #     mir1_err.append(s82x_irac1_err[i])
#     # else: 
#         mir1.append(s82x_W1[i])
#         mir1_err.append(s82x_W1_err[i])

#     # if s82x_irac2[i] > 0:
#     #     mir2.append(s82x_irac2[i])
#     #     mir2_err.append(s82x_irac2_err[i])
#     # else:
#         mir2.append(s82x_W2[i])
#         mir2_err.append(s82x_W2_err[i])

# mir1, mir2 = np.asarray(mir1), np.asarray(mir2)
# mir1_err, mir2_err = np.asarray(mir1_err), np.asarray(mir2_err)

# for i in range(len(s82x_irac2)):
#     print(s82x_irac2[i], s82x_W2[i], mir2[i])


print('S82X All: ', len(lamassa_id_use))


# Select S82X sources from the Ananna2017 catalog based on conditions listed at the top of the file
s82x_condition = (lamassa_z > z_min) & (lamassa_z < z_max) & (np.logical_and(lamassa_ra >= 13, lamassa_ra <= 37)) & (
    np.logical_and(np.log10(s82x_Lx_sp_full) <= Lx_max, np.log10(s82x_Lx_sp_full) >= Lx_min))


lamassa_id_use = lamassa_id_use[s82x_condition]
lamassa_cat = lamassa_cat[s82x_condition]
s82x_z_sp = lamassa_z[s82x_condition]
lamassa_ra = lamassa_ra[s82x_condition]
lamassa_dec = lamassa_dec[s82x_condition]
s82x_Lx_sp_full = s82x_Lx_sp_full[s82x_condition]
s82x_Lx_sp_hard = s82x_Lx_sp_hard[s82x_condition]
s82x_Fx_hard = s82x_Fx_hard[s82x_condition]
s82x_Fx_soft = s82x_Fx_soft[s82x_condition]
s82x_Fx_full = s82x_Fx_full[s82x_condition]
s82x_Nh = s82x_Nh[s82x_condition]
s82x_spec_class = s82x_spec_class[s82x_condition]

print('S82X Lx z coords: ', len(lamassa_id_use))


print('S82X match: ', len(lamassa_id_use))


s82x_Fx_hard_match_mjy = s82x_Fx_hard*4.136E8/(10-2)
s82x_Fx_soft_match_mjy = s82x_Fx_soft*4.136E8/(2-0.5)
s82x_Fx_full_match_mjy = s82x_Fx_full*4.136E8/(10-0.5)
s82x_Fx_hard_err_match_mjy = s82x_Fx_hard_match_mjy*0.2
s82x_Fx_soft_err_match_mjy = s82x_Fx_soft_match_mjy*0.2
s82x_Fx_full_err_match_mjy = s82x_Fx_full_match_mjy*0.2


# Create nan array with length == to the number of sources to be input to the photometry array
s82x_nan_array = np.zeros(np.shape(lamassa_id_use))
s82x_nan_array[s82x_nan_array == 0] = np.nan

# Flux array for Stripe82X sources in Jy
s82x_flux_array = np.array([
	s82x_Fx_hard_match_mjy*1000, s82x_Fx_soft_match_mjy*1000,
	s82x_nan_array,
	mag_to_flux(lamassa_data['mag_FUV'][s82x_condition], 'FUV')*1E6,
	mag_to_flux(lamassa_data['mag_NUV'][s82x_condition], 'FUV')*1E6,
	mag_to_flux(lamassa_data['u'][s82x_condition], 'sloan_u')*1E6,
	mag_to_flux(lamassa_data['g'][s82x_condition], 'sloan_g')*1E6,
	mag_to_flux(lamassa_data['r'][s82x_condition], 'sloan_r')*1E6,
	mag_to_flux(lamassa_data['i'][s82x_condition], 'sloan_i')*1E6,
	mag_to_flux(lamassa_data['z'][s82x_condition], 'sloan_z')*1E6,
	mag_to_flux(lamassa_data['JVHS'][s82x_condition], 'JVHS')*1E6,
	mag_to_flux(lamassa_data['HVHS'][s82x_condition], 'HVHS')*1E6,
	mag_to_flux(lamassa_data['KVHS'][s82x_condition], 'HVHS')*1E6,
    # s82x_irac1[s82x_condition],
	# s82x_W1[s82x_condition],
	# s82x_W2[s82x_condition],
    # s82x_irac2[s82x_condition],
    # mir1[s82x_condition],
    # mir2[s82x_condition],
	# s82x_W3[s82x_condition],
	# s82x_W4[s82x_condition],
    mag_to_flux(s82x_W1[s82x_condition],'W1')*1E6,
    mag_to_flux(s82x_W2[s82x_condition],'W2')*1E6,
    mag_to_flux(s82x_W3[s82x_condition],'W3')*1E6,
    mag_to_flux(s82x_W4[s82x_condition],'W4')*1E6,    
	s82x_nan_array,
	lamassa_data['F250'][s82x_condition]*1000,
	lamassa_data['F350'][s82x_condition]*1000,
	lamassa_data['F500'][s82x_condition]*1000
])


s82x_flux_err_array = np.array([
	s82x_Fx_hard_err_match_mjy*1000, s82x_Fx_soft_err_match_mjy*1000,
	s82x_nan_array,
	magerr_to_fluxerr(lamassa_data['mag_FUV'][s82x_condition],lamassa_data['magerr_FUV'][s82x_condition], 'FUV')*1E6,
	magerr_to_fluxerr(lamassa_data['mag_NUV'][s82x_condition],lamassa_data['magerr_NUV'][s82x_condition], 'FUV')*1E6,
	magerr_to_fluxerr(lamassa_data['u'][s82x_condition],lamassa_data['u_err'][s82x_condition], 'sloan_u')*1E6,
	magerr_to_fluxerr(lamassa_data['g'][s82x_condition],lamassa_data['g_err'][s82x_condition], 'sloan_g')*1E6,
	magerr_to_fluxerr(lamassa_data['r'][s82x_condition],lamassa_data['r_err'][s82x_condition], 'sloan_r')*1E6,
	magerr_to_fluxerr(lamassa_data['i'][s82x_condition],lamassa_data['i_err'][s82x_condition], 'sloan_i')*1E6,
	magerr_to_fluxerr(lamassa_data['z'][s82x_condition],lamassa_data['z_err'][s82x_condition], 'sloan_z')*1E6,
	magerr_to_fluxerr(lamassa_data['JVHS'][s82x_condition],lamassa_data['JVHS_err'][s82x_condition], 'JVHS', AB=True)*1E6,
	magerr_to_fluxerr(lamassa_data['HVHS'][s82x_condition],lamassa_data['HVHS_err'][s82x_condition], 'HVHS', AB=True)*1E6,
	magerr_to_fluxerr(lamassa_data['KVHS'][s82x_condition],lamassa_data['KVHS_err'][s82x_condition], 'HVHS', AB=True)*1E6,
    # s82x_irac1_err[s82x_condition],
	# s82x_W1_err[s82x_condition],
	# s82x_W2_err[s82x_condition],
    # s82x_irac1_err[s82x_condition],
    # mir1_err[s82x_condition],
    # mir2_err[s82x_condition],
	# s82x_W3_err[s82x_condition],
	# s82x_W4_err[s82x_condition],
    magerr_to_fluxerr(s82x_W1[s82x_condition],s82x_W1_err[s82x_condition],'W1')*1E6,
    magerr_to_fluxerr(s82x_W2[s82x_condition],s82x_W2_err[s82x_condition],'W2')*1E6,
    magerr_to_fluxerr(s82x_W3[s82x_condition],s82x_W3_err[s82x_condition],'W3')*1E6,
    magerr_to_fluxerr(s82x_W4[s82x_condition],s82x_W4_err[s82x_condition],'W4')*1E6,
    s82x_nan_array,
	lamassa_data['F250_err'][s82x_condition]*1000,
	lamassa_data['F350_err'][s82x_condition]*1000,
	lamassa_data['F500_err'][s82x_condition]*1000
])

s82x_flux_array = s82x_flux_array.T
s82x_flux_err_array = s82x_flux_err_array.T



###############################################################################


###############################################################################
###############################################################################
############################ Read in GOODS-N files ############################

xue = fits.open(path+'XueCDFN.fit')
xue_data = xue[1].data
xue.close()

goodsN = ascii.read('/Users/connor_auge/Research/REU/2021/GOODS-N_phot2.txt')

xue_id = xue_data['seq']
xue_xRA = xue_data['RAJ2000']
xue_xDEC = xue_data['DEJ2000']
xue_Lx = xue_data['Lx']*1.180  # convert the 0.5-7keV Lx to 0.5-10kev
xue_Lx_hard = xue_data['Lx']*0.721 # convert the 0.5-7keV Lx to 2-10keV
xue_Fx_full = xue_data['FFlux']
xue_Fx_hard = xue_data['HFlux']
xue_Fx_soft = xue_data['SFlux']
xue_z = xue_data['zspec']
xue_offset = xue_data['C-XOff']
# xue_z = xue_data['zadopt']

# Select GOODS-N sources from the Xue catalog based on conditions listed at the top of the file
goodsN_condition = (np.log10(xue_Lx) >= Lx_min) & (np.log10(xue_Lx) <= Lx_max) & (xue_z > z_min) & (xue_z <= z_max) & (xue_z != 0.0)
print('GOODS-N All:', len(xue_id))

xue_xRA = xue_xRA[goodsN_condition]
xue_xDEC = xue_xDEC[goodsN_condition]
xue_Lx = xue_Lx[goodsN_condition]
xue_Lx_hard = xue_Lx_hard[goodsN_condition]
xue_Fx_full = xue_Fx_full[goodsN_condition]
xue_Fx_hard = xue_Fx_hard[goodsN_condition]
xue_Fx_soft = xue_Fx_soft[goodsN_condition]
xue_z = xue_z[goodsN_condition]
xue_id = xue_id[goodsN_condition]
xue_offset = xue_offset[goodsN_condition]

goodsN_phot_RA = np.asarray(goodsN['ALPHA_GR_DEC_ORDER'])
goodsN_phot_DEC = np.asarray(goodsN['DELTA_GR_DEC_ORDER'])
goodsN_phot_id = np.asarray(goodsN['object'])

print('GOODS-N Lx z:', len(xue_id))

# goodsN_coords_file = open('/Users/connor_auge/Research/Disertation/catalogs/goodsN_coords.txt','w')
# goodsN_coords_file.writelines('#ID,RA,Dec\n')
# for i in range(len(xue_id)):
#     goodsN_coords_file.writelines('%s,%s,%s\n' % (xue_id[i], xue_xRA[i], xue_xDEC[i]))
# goodsN_coords_file.close()

# Match the X-ray catalog and photometry catalog based on their corrdinates (5 arcsecond sep)
cat_match_dist = 1.5  # arcseconds
catalog = SkyCoord(ra=goodsN_phot_RA*u.degree, dec=goodsN_phot_DEC*u.degree)
c1 = SkyCoord(ra=xue_xRA, dec=xue_xDEC, unit=(u.hourangle, u.deg))
idx, d2d, d3d = c1.match_to_catalog_sky(catalog)
goodsN_idx_match = idx[d2d.arcsecond <= cat_match_dist]

goodsN_phot_id_match = goodsN_phot_id[goodsN_idx_match]
xue_xRA_match = xue_xRA[d2d.arcsecond < cat_match_dist]
xue_xDEC_match = xue_xDEC[d2d.arcsecond < cat_match_dist]
xue_z_match = xue_z[d2d.arcsecond < cat_match_dist]
xue_Lx_match = xue_Lx[d2d.arcsecond < cat_match_dist]
xue_Lx_hard_match = xue_Lx_hard[d2d.arcsecond < cat_match_dist]
xue_Fx_full_match = xue_Fx_full[d2d.arcsecond < cat_match_dist]
xue_Fx_hard_match = xue_Fx_hard[d2d.arcsecond < cat_match_dist]
xue_Fx_soft_match = xue_Fx_soft[d2d.arcsecond < cat_match_dist]
xue_id_match = xue_id[d2d.arcsecond < cat_match_dist]

print('GOODS-N match:', len(xue_id_match))

xue_Fx_hard_match_mjy = xue_Fx_hard_match*4.136E8/(10-2)
xue_Fx_soft_match_mjy = xue_Fx_soft_match*4.136E8/(2-0.5)
xue_Fx_full_match_mjy = xue_Fx_full_match*4.136E8/(10-0.5)
xue_Fx_hard_match_err_mjy = xue_Fx_hard_match_mjy*0.2
xue_Fx_soft_match_err_mjy = xue_Fx_soft_match_mjy*0.2
xue_Fx_full_match_err_mjy = xue_Fx_full_match_mjy*0.2

goodsN_Nh_match = np.zeros(np.shape(xue_Lx_match))

# Create nan array with length == to the number of sources to be input to the photometry array
goodsN_nan_array = np.zeros(np.shape(goodsN_phot_id_match))
goodsN_nan_array[goodsN_nan_array == 0] = np.nan

goodsN_flux_array = np.asarray([xue_Fx_hard_match_mjy*1000, xue_Fx_soft_match_mjy*1000,
                                goodsN_nan_array,
                                goodsN['GALEX_fuv'][goodsN_idx_match],
                                goodsN['GALEX_nuv'][goodsN_idx_match],
                                goodsN['TFIT_KPNO_U'][goodsN_idx_match],
                                goodsN['TFIT_GOODS_ACS_F435W'][goodsN_idx_match],
                                goodsN['TFIT_GOODS_ACS_F606W'][goodsN_idx_match],
                                goodsN['TFIT_GOODS_ACS_F775W'][goodsN_idx_match],
                                goodsN['TFIT_GOODS_ACS_F814W'][goodsN_idx_match],
                                goodsN['TFIT_CANDELS_WFC3_F105W'][goodsN_idx_match],
                                goodsN['TFIT_CANDELS_WFC3_F125W'][goodsN_idx_match],
                                goodsN['TFIT_CANDELS_WFC3_F140W'][goodsN_idx_match],
                                goodsN['TFIT_CANDELS_WFC3_F160W'][goodsN_idx_match],
                                goodsN['TFIT_MOIRCS_K'][goodsN_idx_match],
                                goodsN['TFIT_SEDS_irac36'][goodsN_idx_match],
                                goodsN['TFIT_SEDS_irac45'][goodsN_idx_match],
                                goodsN['TFIT_SEDS_irac58'][goodsN_idx_match],
                                goodsN['TFIT_SEDS_irac80'][goodsN_idx_match],
                                goodsN['mips24_cryo'][goodsN_idx_match],
                                goodsN['mips70_cryo'][goodsN_idx_match],
                                goodsN['pacs100_merged'][goodsN_idx_match],
                                goodsN['pacs160_merged'][goodsN_idx_match],
                                goodsN['spire250_merged'][goodsN_idx_match],
                                goodsN['spire350_merged'][goodsN_idx_match],
                                goodsN['spire500_merged'][goodsN_idx_match]
                                ])

goodsN_flux_err_array = np.asarray([xue_Fx_hard_match_err_mjy*1000, xue_Fx_soft_match_err_mjy*1000,
                                    goodsN_nan_array,
                                    goodsN['err_GALEX_fuv'][goodsN_idx_match],
                                    goodsN['err_GALEX_nuv'][goodsN_idx_match],
                                    goodsN['err_TFIT_KPNO_U'][goodsN_idx_match],
                                    goodsN['err_TFIT_GOODS_ACS_F435W'][goodsN_idx_match],
                                    goodsN['err_TFIT_GOODS_ACS_F606W'][goodsN_idx_match],
                                    goodsN['err_TFIT_GOODS_ACS_F775W'][goodsN_idx_match],
                                    goodsN['err_TFIT_GOODS_ACS_F814W'][goodsN_idx_match],
                                    goodsN['err_TFIT_CANDELS_WFC3_F105W'][goodsN_idx_match],
                                    goodsN['err_TFIT_CANDELS_WFC3_F125W'][goodsN_idx_match],
                                    goodsN['err_TFIT_CANDELS_WFC3_F140W'][goodsN_idx_match],
                                    goodsN['err_TFIT_CANDELS_WFC3_F160W'][goodsN_idx_match],
                                    goodsN['err_TFIT_MOIRCS_K'][goodsN_idx_match],
                                    goodsN['err_TFIT_SEDS_irac36'][goodsN_idx_match],
                                    goodsN['err_TFIT_SEDS_irac45'][goodsN_idx_match],
                                    goodsN['err_TFIT_SEDS_irac58'][goodsN_idx_match],
                                    goodsN['err_TFIT_SEDS_irac80'][goodsN_idx_match],
                                    goodsN['err_mips24_cryo'][goodsN_idx_match],
                                    goodsN['err_mips70_cryo'][goodsN_idx_match],
                                    goodsN['err_pacs100_merged'][goodsN_idx_match],
                                    goodsN['err_pacs160_merged'][goodsN_idx_match],
                                    goodsN['err_spire250_merged'][goodsN_idx_match],
                                    goodsN['err_spire350_merged'][goodsN_idx_match],
                                    goodsN['err_spire500_merged'][goodsN_idx_match]
                                    ])

goodsN_flux_array = goodsN_flux_array.T
goodsN_flux_err_array = goodsN_flux_err_array.T


###############################################################################

###############################################################################
###############################################################################
############################ Read in GOODS-S files ############################
luo = fits.open(path+'LuoCDFS.fit')
luo_data = luo[1].data
luo.close()

goodsS = ascii.read('/Users/connor_auge/Research/REU/2021/GOODS-S_phot2.txt')

luo_id = luo_data['seq']
# luo_xRA = luo_data['RACdeg']
# luo_xDEC = luo_data['DECdeg']
luo_xRA = luo_data['RAJ2000']
luo_xDEC = luo_data['DEJ2000']
luo_Lx = luo_data['Lxc']*1.180 # convert the 0.5-7kev Lx to 0.5-10kev
luo_Lx_hard = luo_data['Lxc']*0.721  # convert the 0.5-7kev Lx to 2-10kev
luo_Fx_full = luo_data['FFB']
luo_Fx_hard = luo_data['FHB']
luo_Fx_soft = luo_data['FSB']
luo_z = luo_data['zspec']
goodsS_Nh = luo_data['Nh']
# luo_z = luo_data['zf']

print('GOODS-S All: ', len(luo_id))

# Select GOODS-S sources from the Xue catalog based on conditions listed at the top of the file 
goodsS_condition = (np.log10(luo_Lx) >= Lx_min) & (np.log10(luo_Lx) <= Lx_max) &(luo_z > z_min) & (luo_z <= z_max) & (luo_z != 0.0)

luo_id = luo_id[goodsS_condition]
luo_xRA = luo_xRA[goodsS_condition]
luo_xDEC = luo_xDEC[goodsS_condition]
luo_Lx = luo_Lx[goodsS_condition]
luo_Lx_hard = luo_Lx_hard[goodsS_condition]
luo_Fx_full = luo_Fx_full[goodsS_condition]
luo_Fx_hard = luo_Fx_hard[goodsS_condition]
luo_Fx_soft = luo_Fx_soft[goodsS_condition]
luo_z = luo_z[goodsS_condition]
goodsS_Nh = goodsS_Nh[goodsS_condition]

print('GOODS-S Lx z: ', len(luo_id))

goodsS_phot_RA = np.asarray(goodsS['ALPHA_GR_DEC_ORDER'])
goodsS_phot_DEC = np.asarray(goodsS['DELTA_GR_DEC_ORDER'])
goodsS_phot_id = np.asarray(goodsS['object'])

# goodsS_coords_file = open('/Users/connor_auge/Research/Disertation/catalogs/goodsS_coords.txt','w')
# goodsS_coords_file.writelines('#ID,RA,Dec\n')
# for i in range(len(luo_id)):
#     goodsS_coords_file.writelines('%s,%s,%s\n' % (luo_id[i], luo_xRA[i], luo_xDEC[i]))
# goodsS_coords_file.close()

# Match the X-ray catalog and photometry catalog based on their corrdinates (5 arcsecond sep)
cat_match_dist = 5 #arcseconds
catalog = SkyCoord(ra=goodsS_phot_RA*u.degree,dec=goodsS_phot_DEC*u.degree)
c1 = SkyCoord(ra=luo_xRA*u.degree, dec=luo_xDEC*u.degree)
idx, d2d, d3d = c1.match_to_catalog_sky(catalog)
goodsS_idx_match = idx[d2d.arcsecond < cat_match_dist]

goodsS_phot_id_match = goodsS_phot_id[goodsS_idx_match]
luo_id_match = luo_id[d2d.arcsecond < cat_match_dist]
luo_xRA_match = luo_xRA[d2d.arcsecond < cat_match_dist]
luo_xDEC_match = luo_xDEC[d2d.arcsecond < cat_match_dist]
luo_z_match = luo_z[d2d.arcsecond < cat_match_dist]
luo_Lx_match = luo_Lx[d2d.arcsecond < cat_match_dist]
luo_Lx_hard_match = luo_Lx_hard[d2d.arcsecond < cat_match_dist]
luo_Fx_full_match = luo_Fx_full[d2d.arcsecond < cat_match_dist]
luo_Fx_hard_match = luo_Fx_hard[d2d.arcsecond < cat_match_dist]
luo_Fx_soft_match = luo_Fx_soft[d2d.arcsecond < cat_match_dist]
goodsS_NH_match = goodsS_Nh[d2d.arcsecond < cat_match_dist]

print('GOODS-S match: ', len(luo_id_match))

luo_Fx_hard_match_mjy = luo_Fx_hard_match*4.136E8/(10-2)
luo_Fx_soft_match_mjy = luo_Fx_soft_match*4.136E8/(2-0.5)
luo_Fx_full_match_mjy = luo_Fx_full_match*4.136E8/(10-0.5)
luo_Fx_hard_match_err_mjy = luo_Fx_hard_match_mjy*0.2
luo_Fx_soft_match_err_mjy = luo_Fx_soft_match_mjy*0.2
luo_Fx_full_match_err_mjy = luo_Fx_full_match_mjy*0.2

goodsS_nan_array = np.zeros(np.shape(goodsS_phot_id_match)) # Create nan array with length == to the number of sources to be input to the photometry array
goodsS_nan_array[goodsS_nan_array == 0] = np.nan

goodsS_flux_array = np.asarray([luo_Fx_hard_match_mjy*1000,luo_Fx_soft_match_mjy*1000,
	goodsS_nan_array,
	goodsS['TFIT_130701_VIMOS_U'][goodsS_idx_match],
	goodsS['TFIT_130701_ACS_b'][goodsS_idx_match], #435
	goodsS['TFIT_130701_ACS_v'][goodsS_idx_match], #606
	goodsS['TFIT_130701_ACS_i'][goodsS_idx_match], #775
	goodsS['TFIT_130701_ACS_f814w'][goodsS_idx_match],
	goodsS['TFIT_130701_WFC3_F098M'][goodsS_idx_match],
	goodsS['TFIT_130701_WFC3_F105W'][goodsS_idx_match],
	goodsS['TFIT_130701_WFC3_F125W'][goodsS_idx_match],
	goodsS['TFIT_130701_WFC3_F160W'][goodsS_idx_match],
	goodsS['TFIT_130701_ISAAC_Ks'][goodsS_idx_match],
	goodsS['TFIT_130701_IRAC36'][goodsS_idx_match],
	goodsS['TFIT_130701_IRAC45'][goodsS_idx_match],
	goodsS['TFIT_130701_IRAC58'][goodsS_idx_match],
	goodsS['TFIT_130701_IRAC80'][goodsS_idx_match],
	goodsS['mips24_cryo'][goodsS_idx_match],
	goodsS['mips70_cryo'][goodsS_idx_match],
	goodsS['pacs100_merged'][goodsS_idx_match],
	goodsS['pacs160_merged'][goodsS_idx_match],
	goodsS['spire250_merged'][goodsS_idx_match],
	goodsS['spire350_merged'][goodsS_idx_match],
	goodsS['spire500_merged'][goodsS_idx_match]
])

goodsS_flux_err_array = np.asarray([luo_Fx_hard_match_err_mjy*1000,luo_Fx_soft_match_err_mjy*1000,
	goodsS_nan_array,
	goodsS['err_TFIT_130701_VIMOS_U'][goodsS_idx_match],
	goodsS['err_TFIT_130701_ACS_b'][goodsS_idx_match],
	goodsS['err_TFIT_130701_ACS_v'][goodsS_idx_match],
	goodsS['err_TFIT_130701_ACS_i'][goodsS_idx_match],
	goodsS['err_TFIT_130701_ACS_f814w'][goodsS_idx_match],
	goodsS['err_TFIT_130701_WFC3_F098M'][goodsS_idx_match],
	goodsS['err_TFIT_130701_WFC3_F105W'][goodsS_idx_match],
	goodsS['err_TFIT_130701_WFC3_F125W'][goodsS_idx_match],
	goodsS['err_TFIT_130701_WFC3_F160W'][goodsS_idx_match],
	goodsS['err_TFIT_130701_ISAAC_Ks'][goodsS_idx_match],
	goodsS['err_TFIT_130701_IRAC36'][goodsS_idx_match],
	goodsS['err_TFIT_130701_IRAC45'][goodsS_idx_match],
	goodsS['err_TFIT_130701_IRAC58'][goodsS_idx_match],
	goodsS['err_TFIT_130701_IRAC80'][goodsS_idx_match],
	goodsS['err_mips24_cryo'][goodsS_idx_match],
	goodsS['err_mips70_cryo'][goodsS_idx_match],
	goodsS['err_pacs100_merged'][goodsS_idx_match],
	goodsS['err_pacs160_merged'][goodsS_idx_match],
	goodsS['err_spire250_merged'][goodsS_idx_match],
	goodsS['err_spire350_merged'][goodsS_idx_match],
	goodsS['err_spire500_merged'][goodsS_idx_match]
])

goodsS_flux_array = goodsS_flux_array.T
goodsS_flux_err_array = goodsS_flux_err_array.T


###############################################################################
###############################################################################
############################ Read in GOODS-N files 2 ##########################

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

###############################################################################
###############################################################################
############################ Read in GOODS-S files 2 ##########################

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




# goodsS_flux_array_auge = np.asarray([goodsS_auge_Fx_hard_match_mjy*1000, goodsS_auge_Fx_soft_match_mjy*1000,
# 	goodsS_nan_array,
#     mag_to_flux(goodsS_auge_data['FUV'][goodsS_auge_condition],'FUV')*1E6,
# 	mag_to_flux(goodsS_auge_data['NUV'][goodsS_auge_condition],'NUV')*1E6,
# 	mag_to_flux(goodsS_auge_data['Umag'][goodsS_auge_condition],'U')*1E6,
#     mag_to_flux(goodsS_auge_data['F435W'][goodsS_auge_condition],'AB')*1E6,
# 	mag_to_flux(goodsS_auge_data['Bmag'][goodsS_auge_condition],'AB')*1E6,
# 	mag_to_flux(goodsS_auge_data['Vmag'][goodsS_auge_condition],'V')*1E6,
#     mag_to_flux(goodsS_auge_data['F606W'][goodsS_auge_condition], 'AB')*1E6,
# 	mag_to_flux(goodsS_auge_data['Rmag'][goodsS_auge_condition],'R')*1E6,
# 	mag_to_flux(goodsS_auge_data['Imag'][goodsS_auge_condition], 'I')*1E6,
#     mag_to_flux(goodsS_auge_data['F775W'][goodsS_auge_condition], 'AB')*1E6,
#    	mag_to_flux(goodsS_auge_data['F814W'][goodsS_auge_condition], 'AB')*1E6,
# 	mag_to_flux(goodsS_auge_data['zmag'][goodsS_auge_condition],'sloan_z')*1E6,
#     mag_to_flux(goodsS_auge_data['F098M'][goodsS_auge_condition], 'AB')*1E6,
#     mag_to_flux(goodsS_auge_data['F125W'][goodsS_auge_condition],'AB')*1E6,
# 	mag_to_flux(goodsS_auge_data['Jmag'][goodsS_auge_condition],'JUK')*1E6,
# 	mag_to_flux(goodsS_auge_data['Hmag'][goodsS_auge_condition],'HUK')*1E6,
# 	mag_to_flux(goodsS_auge_data['Kmag'][goodsS_auge_condition], 'KUK')*1E6,
# 	goodsS_auge_data['IRAC1'][goodsS_auge_condition],
# 	goodsS_auge_data['IRAC2'][goodsS_auge_condition],
# 	goodsS_auge_data['IRAC3'][goodsS_auge_condition],
# 	goodsS_auge_data['IRAC4'][goodsS_auge_condition],
# 	goodsS_auge_data['F24'][goodsS_auge_condition],
#     goodsS_auge_data['F70'][goodsS_auge_condition],
# 	goodsS_auge_data['F100'][goodsS_auge_condition],
# 	goodsS_auge_data['F160'][goodsS_auge_condition],
# 	goodsS_auge_data['F250'][goodsS_auge_condition]*1000,
# 	goodsS_auge_data['F350'][goodsS_auge_condition]*1000,
# 	goodsS_auge_data['F500'][goodsS_auge_condition]*1000
# ]) 

# goodsS_flux_err_array_auge = np.asarray([goodsS_auge_Fx_hard_match_mjy*1000*0.2, goodsS_auge_Fx_soft_match_mjy*1000*0.2,
# 	goodsS_nan_array,
#     magerr_to_fluxerr(goodsS_auge_data['FUV'][goodsS_auge_condition],goodsS_auge_data['FUVerr'][goodsS_auge_condition],'U')*1E6,
#     magerr_to_fluxerr(goodsS_auge_data['F435W'][goodsS_auge_condition],goodsS_auge_data['F435W'][goodsS_auge_condition],'AB')*1E6,
# 	magerr_to_fluxerr(goodsS_auge_data['NUV'][goodsS_auge_condition],goodsS_auge_data['NUVerr'][goodsS_auge_condition], 'B')*1E6,
#     magerr_to_fluxerr(goodsS_auge_data['Umag'][goodsS_auge_condition],goodsS_auge_data['Uerrmag'][goodsS_auge_condition],'U')*1E6,
# 	magerr_to_fluxerr(goodsS_auge_data['Bmag'][goodsS_auge_condition],goodsS_auge_data['Berrmag'][goodsS_auge_condition], 'B')*1E6,
# 	magerr_to_fluxerr(goodsS_auge_data['Vmag'][goodsS_auge_condition],goodsS_auge_data['Verrmag'][goodsS_auge_condition], 'V')*1E6,
#     magerr_to_fluxerr(goodsS_auge_data['F606W'][goodsS_auge_condition],goodsS_auge_data['F606W'][goodsS_auge_condition],'AB')*1E6,
# 	magerr_to_fluxerr(goodsS_auge_data['Rmag'][goodsS_auge_condition],goodsS_auge_data['Rerrmag'][goodsS_auge_condition], 'R')*1E6,
# 	magerr_to_fluxerr(goodsS_auge_data['Imag'][goodsS_auge_condition],goodsS_auge_data['Imagerr'][goodsS_auge_condition],'I')*1E6,
#     magerr_to_fluxerr(goodsS_auge_data['F775W'][goodsS_auge_condition],goodsS_auge_data['F775W'][goodsS_auge_condition],'AB')*1E6,
#     magerr_to_fluxerr(goodsS_auge_data['F814W'][goodsS_auge_condition],goodsS_auge_data['F814W'][goodsS_auge_condition],'AB')*1E6,
# 	magerr_to_fluxerr(goodsS_auge_data['zmag'][goodsS_auge_condition],goodsS_auge_data['zmagerr'][goodsS_auge_condition],'sloan_z')*1E6,
# 	magerr_to_fluxerr(goodsS_auge_data['Jmag'][goodsS_auge_condition],goodsS_auge_data['Jmagerr'][goodsS_auge_condition],'JVHS')*1E6,
#     magerr_to_fluxerr(goodsS_auge_data['F098M'][goodsS_auge_condition],goodsS_auge_data['F098M'][goodsS_auge_condition],'AB')*1E6,
#     magerr_to_fluxerr(goodsS_auge_data['F125W'][goodsS_auge_condition],goodsS_auge_data['F125W'][goodsS_auge_condition],'AB')*1E6,
# 	magerr_to_fluxerr(goodsS_auge_data['Hmag'][goodsS_auge_condition],goodsS_auge_data['Hmagerr'][goodsS_auge_condition],'HVHS')*1E6,
# 	magerr_to_fluxerr(goodsS_auge_data['Kmag'][goodsS_auge_condition],goodsS_auge_data['Kmagerr'][goodsS_auge_condition],'KVHS')*1E6,
# 	goodsS_auge_data['IRAC1err'][goodsS_auge_condition],
# 	goodsS_auge_data['IRAC2err'][goodsS_auge_condition],
# 	goodsS_auge_data['IRAC3err'][goodsS_auge_condition],
# 	goodsS_auge_data['IRAC4err'][goodsS_auge_condition],
# 	goodsS_auge_data['F24err'][goodsS_auge_condition],
#     goodsS_auge_data['F70err'][goodsS_auge_condition],
# 	goodsS_auge_data['F100err'][goodsS_auge_condition],
# 	goodsS_auge_data['F160err'][goodsS_auge_condition],
# 	goodsS_auge_data['F250err'][goodsS_auge_condition]*1000,
# 	goodsS_auge_data['F350err'][goodsS_auge_condition]*1000,
# 	goodsS_auge_data['F500err'][goodsS_auge_condition]*1000
# ])

print(np.shape(goodsS_flux_array_auge))

goodsS_flux_array_auge = goodsS_flux_array_auge.T
goodsS_flux_err_array_auge = goodsS_flux_err_array_auge.T


##############################################################################
##############################################################################
'''
C-GOALS
'''

goals = fits.open('../catalogs/U2012_GOALS2.fits')
goals_data = goals[1].data

cgoals = ascii.read('../catalogs/CGOALS_Xray.txt',guess=False,delimiter=',',encoding="utf-8")

ned = fits.open('../catalogs/NED_GOALS.fits')
ned_data = ned[1].data

goals_photID = goals_data['ID']

print('GOALS_ID: ', goals_photID)

cgoals_ID = cgoals['ID']
cgoals_z = cgoals['z']
cgoals_Lx = cgoals['Lhx']*1E40
cgoals_Lir = cgoals['LIR']
cgoals_Fhx = cgoals['Fhx']*1E-14
cgoals_Fsx = cgoals['Fsx']*1E-14

ulirg_xray = ascii.read('../catalogs/ULIRG_Xray.csv',guess=False,delimiter=',')

ulirg_ID = ulirg_xray['ID']
ulirg_z = ulirg_xray['z']
ulirg_Lx = ulirg_xray['Lx2_10']
ulirg_Lx_obs = ulirg_xray['Lx2_10_obs']
ulirg_Fhx = ulirg_xray['Fx2_10']
# ulirg_Lx2 = np.log10(Flux_to_Lum(ulirg_Fhx,ulirg_z))
ulirg_Fsx = ulirg_xray['Fx05_2']
ulirg_Nh = ulirg_xray['Nh']
ulirg_LIR = ulirg_xray['LIR']

ulirg_Lx_cond = ulirg_Lx > goals_Lx_min
ulirg_ID = ulirg_ID[ulirg_Lx_cond]
ulirg_z = ulirg_z[ulirg_Lx_cond]
ulirg_Lx = ulirg_Lx[ulirg_Lx_cond]
ulirg_Lx_obs = ulirg_Lx_obs[ulirg_Lx_cond]
# ulirg_Lx2 = ulirg_Lx2[ulirg_Lx_cond]
ulirg_Fhx = ulirg_Fhx[ulirg_Lx_cond]
ulirg_Fsx = ulirg_Fsx[ulirg_Lx_cond]
ulirg_Nh = ulirg_Nh[ulirg_Lx_cond]
ulirg_LIR = ulirg_LIR[ulirg_Lx_cond]


# ulirg_short = np.asarray(['NGC 6240', 'IRAS F05189-2524', 'UGC 05101', 'UGC 08058', 'UGC 08696', 'UGC 09913'])

# ix,iy = match(ulirg_short,ulirg_ID)
# ulirg_ID = ulirg_ID[iy]
# ulirg_z = ulirg_z[iy]
# ulirg_Lx = ulirg_Lx[iy]
# ulirg_Lx_obs = ulirg_Lx_obs[iy]
# # ulirg_Lx2 = ulirg_Lx2[iy]
# ulirg_Fhx = ulirg_Fhx[iy]
# ulirg_Fsx = ulirg_Fsx[iy]




Lx_cond = (cgoals_Lx > 0) & (np.log10(cgoals_Lx) < 42.5)
cgoals_ID = cgoals_ID[Lx_cond]
cgoals_z = cgoals_z[Lx_cond]
cgoals_Fhx = cgoals_Fhx[Lx_cond]
cgoals_Fsx = cgoals_Fsx[Lx_cond]
cgoals_Lir = cgoals_Lir[Lx_cond]
cgoals_Lx = cgoals_Lx[Lx_cond]

Lir_cond = (cgoals_Lir < 11.5)
cgoals_ID = cgoals_ID[Lir_cond]
cgoals_z = cgoals_z[Lir_cond]
cgoals_Fhx = cgoals_Fhx[Lir_cond]
cgoals_Fsx = cgoals_Fsx[Lir_cond]
cgoals_Lx = cgoals_Lx[Lir_cond]
cgoals_Lir = cgoals_Lir[Lir_cond]


############### Subset match ###############

# # ulirgs = np.asarray(['NGC 6286','NGC 4922','NGC 34','IC 4518A','IC 4518B','NGC 3690(W)','Arp 220(W)','NGC 6240(N)','NGC 6240(S)','NGC 7130','NGC 7674','NGC 7469','NGC 833','NGC 835','CGCG 468-002W','MCG +04-48-002','NGC 6921','UGC 05101','IRAS 13120-5453','Mrk 273','Mrk 231','IRAS 05189-2524','NGC 7679','NGC 7682'])
# ulirgs = np.asarray(['NGC 6286','NGC 4922','NGC0034','NGC 3690/IC 694','UGC 09913','NGC 7674','NGC 7469','UGC 05101','UGC 08696','UGC 08058','IRAS F05189-2524'])

single = np.asarray(['UGC 09913'])
ulirgs = np.asarray(['IRAS F05189-2524','UGC 05101','UGC 08058','UGC 08696'])
lirgs = np.asarray(['NGC 3690/IC 694','NGC 4922','NGC 7469','NGC 7674','MCG +04-48-002','NGC 6240','NGC 6921','NGC 7130','NGC 7682','UGC 09913'])

ix2, iy2 = match(goals_photID,ulirgs)

# print('HERE:',len(goals_photID[ix2]))
# print(goals_photID[ix2])
# goals_photID = goals_photID[ix2]

# goals_photID_match = goals_photID[ix2]
goals_photID_match = goals_photID

# cgoals_ID = cgoals_ID[ix2]
# cgoals_z = cgoals_z[ix2]
# cgoals_Fhx = cgoals_Fhx[ix2]
# cgoals_Fsx = cgoals_Fsx[ix2]
# cgoals_Lx = cgoals_Lx[ix2]
# cgoals_Lir = cgoals_Lir[ix2]

############### ############### ###############


# ix,iy = match(goals_photID_match,cgoals_ID)

ulirg_ix,ulirg_iy = match(goals_photID_match,ulirg_ID)


ulirg_cgoals_z = ulirg_z[ulirg_iy]
ulirg_cgoals_Lx_match = 10**ulirg_Lx[ulirg_iy]
# ulirg_cgoals_Lx_match = (10**ulirg_Lx[ulirg_iy])*0.638 # convert Lx(2-10keV) to Lx(0.5-2keV)
ulirg_cgoals_Lx_obs_match = 10**ulirg_Lx_obs[ulirg_iy]
ulirg_cgoals_Nh_match = np.log10(ulirg_Nh[ulirg_iy])
ulirg_cgoals_Lir_match = 10**ulirg_LIR[ulirg_iy]

# cgoals_z = cgoals_z[iy]
# cgoals_Lx_match = cgoals_Lx[iy]
# cgoals_Fhx = cgoals_Fhx[iy]
# cgoals_Fsx = cgoals_Fsx[iy]

# cgoals_ID_match = goals_photID_match[ix]
ulirg_cgoals_ID_match = goals_photID_match[ulirg_ix]

ulirg_cgoals_Fx_hard_match_mjy = ulirg_Fhx[ulirg_iy]*4.136E8/(10-2)
ulirg_cgoals_Fx_soft_match_mjy = ulirg_Fsx[ulirg_iy]*4.136E8/(2-0.5)
# cgoals_Fx_hard_match_mjy = cgoals_Fhx*4.136E8/(7-2)
# cgoals_Fx_soft_match_mjy = cgoals_Fsx*4.136E8/(2-0.5)

# cgoals_nan_array = np.zeros(np.shape(goals_data['HX'][ix]))
# cgoals_nan_array[cgoals_nan_array == 0] = np.nan
# cgoals_flux = np.asarray([
# 	# cgoals_Fx_hard_match_mjy*1000,
# 	# cgoals_Fx_soft_match_mjy*1000,
# 	goals_data['HX'][ix]*1E6,
# 	goals_data['SX'][ix]*1E6,
# 	cgoals_nan_array,
# 	goals_data['FUV'][ix]*1E6,
# 	goals_data['NUV'][ix]*1E6,
# 	goals_data['U'][ix]*1E6,
# 	goals_data['B'][ix]*1E6,
# 	goals_data['V'][ix]*1E6,
# 	goals_data['R'][ix]*1E6,
# 	goals_data['I'][ix]*1E6,
# 	goals_data['J'][ix]*1E6,
# 	goals_data['H'][ix]*1E6,
# 	goals_data['Ks'][ix]*1E6,
# 	goals_data['IRAC1'][ix]*1E6,
# 	goals_data['IRAC2'][ix]*1E6,
# 	goals_data['IRAC3'][ix]*1E6,
# 	goals_data['IRAC4'][ix]*1E6,
# 	goals_data['IRAS1'][ix]*1E6,
# 	goals_data['MIPS1'][ix]*1E6,
# 	goals_data['IRAS2'][ix]*1E6,
# 	goals_data['IRAS3'][ix]*1E6,
# 	goals_data['MIPS2'][ix]*1E6,
# 	goals_data['IRAS4'][ix]*1E6,
# 	goals_data['SCUBA2'][ix]*1E6,
# 	goals_data['VLA1'][ix]*1E6,
# 	goals_data['VLA2'][ix]*1E6
# 	])

# cgoals_flux_err = np.asarray([
# 	# cgoals_Fx_hard_match_mjy*1000*0.2,
# 	# cgoals_Fx_soft_match_mjy*1000*0.2,
# 	goals_data['HXerr'][ix]*1E6,
# 	goals_data['SXerr'][ix]*1E6,
# 	cgoals_nan_array,
# 	goals_data['FUVerr'][ix]*1E6,
# 	goals_data['NUVerr'][ix]*1E6,
# 	goals_data['Uerr'][ix]*1E6,
# 	goals_data['Berr'][ix]*1E6,
# 	goals_data['Verr'][ix]*1E6,
# 	goals_data['Rerr'][ix]*1E6,
# 	goals_data['Ierr'][ix]*1E6,
# 	goals_data['Jerr'][ix]*1E6,
# 	goals_data['Herr'][ix]*1E6,
# 	goals_data['Kserr'][ix]*1E6,
# 	goals_data['IRAC1err'][ix]*1E6,
# 	goals_data['IRAC2err'][ix]*1E6,
# 	goals_data['IRAC3err'][ix]*1E6,
# 	goals_data['IRAC4err'][ix]*1E6,
# 	goals_data['IRAS1err'][ix]*1E6,
# 	goals_data['MIPS1err'][ix]*1E6,
# 	goals_data['IRAS2err'][ix]*1E6,
# 	goals_data['IRAS3err'][ix]*1E6,
# 	goals_data['MIPS2err'][ix]*1E6,
# 	goals_data['IRAS4err'][ix]*1E6,
# 	goals_data['SCUBA2err'][ix]*1E6,
# 	goals_data['VLA1err'][ix]*1E6,
# 	goals_data['VLA2err'][ix]*1E6
# 	])

# cgoals_flux = cgoals_flux.T
# cgoals_flux_err = cgoals_flux_err.T

ulirgs_observed_hard = []
ulirgs_observed_soft = []
ulirgs_corrected_hard = []
ulirgs_corrected_soft = []

for i in range(len(goals_data['HX'][ulirg_ix])):
	ulirgs_observed_hard.append(goals_data['HX'][ulirg_ix][i]*1E6)
	ulirgs_observed_soft.append(goals_data['SX'][ulirg_ix][i]*1E6)
	ulirgs_corrected_hard.append(ulirg_cgoals_Fx_hard_match_mjy[i]*1000)
	ulirgs_corrected_soft.append(ulirg_cgoals_Fx_soft_match_mjy[i]*1000)

ulirg_cgoals_nan_array = np.zeros(np.shape(goals_data['HX'][ulirg_ix]))
ulirg_cgoals_nan_array[ulirg_cgoals_nan_array == 0] = np.nan
ulirg_cgoals_flux = np.asarray([
	ulirg_cgoals_Fx_hard_match_mjy*1000,
	ulirg_cgoals_Fx_soft_match_mjy*1000,
	# goals_data['HX'][ulirg_ix]*1E6,
	# goals_data['SX'][ulirg_ix]*1E6,
	ulirg_cgoals_nan_array,
	goals_data['FUV'][ulirg_ix]*1E6,
	goals_data['NUV'][ulirg_ix]*1E6,
	goals_data['U'][ulirg_ix]*1E6,
	goals_data['B'][ulirg_ix]*1E6,
	goals_data['V'][ulirg_ix]*1E6,
	goals_data['R'][ulirg_ix]*1E6,
	goals_data['I'][ulirg_ix]*1E6,
	goals_data['J'][ulirg_ix]*1E6,
	goals_data['H'][ulirg_ix]*1E6,
	goals_data['Ks'][ulirg_ix]*1E6,
	goals_data['IRAC1'][ulirg_ix]*1E6,
	goals_data['IRAC2'][ulirg_ix]*1E6,
	goals_data['IRAC3'][ulirg_ix]*1E6,
	goals_data['IRAC4'][ulirg_ix]*1E6,
	goals_data['IRAS1'][ulirg_ix]*1E6,
	goals_data['MIPS1'][ulirg_ix]*1E6,
	goals_data['IRAS2'][ulirg_ix]*1E6,
	goals_data['IRAS3'][ulirg_ix]*1E6,
	goals_data['MIPS2'][ulirg_ix]*1E6,
	goals_data['PACS2'][ulirg_ix]*1E6,
	goals_data['MIPS3'][ulirg_ix]*1E6,
	goals_data['F250'][ulirg_ix]*1E6,
	goals_data['F350'][ulirg_ix]*1E6,
	goals_data['F500'][ulirg_ix]*1E6,
	goals_data['SCUBA2'][ulirg_ix]*1E6,
	goals_data['VLA1'][ulirg_ix]*1E6,
	goals_data['VLA2'][ulirg_ix]*1E6
	])

ulirg_cgoals_flux_err = np.asarray([
	ulirg_cgoals_Fx_hard_match_mjy*1000*0.2,
	ulirg_cgoals_Fx_soft_match_mjy*1000*0.2,
	# goals_data['HXerr'][ulirg_ix]*1E6*0.2,
	# goals_data['SXerr'][ulirg_ix]*1E6*0.2,
	ulirg_cgoals_nan_array,
	goals_data['FUVerr'][ulirg_ix]*1E6,
	goals_data['NUVerr'][ulirg_ix]*1E6,
	goals_data['Uerr'][ulirg_ix]*1E6,
	goals_data['Berr'][ulirg_ix]*1E6,
	goals_data['Verr'][ulirg_ix]*1E6,
	goals_data['Rerr'][ulirg_ix]*1E6,
	goals_data['Ierr'][ulirg_ix]*1E6,
	goals_data['Jerr'][ulirg_ix]*1E6,
	goals_data['Herr'][ulirg_ix]*1E6,
	goals_data['Kserr'][ulirg_ix]*1E6,
	goals_data['IRAC1err'][ulirg_ix]*1E6,
	goals_data['IRAC2err'][ulirg_ix]*1E6,
	goals_data['IRAC3err'][ulirg_ix]*1E6,
	goals_data['IRAC4err'][ulirg_ix]*1E6,
	goals_data['IRAS1err'][ulirg_ix]*1E6,
	goals_data['MIPS1err'][ulirg_ix]*1E6,
	goals_data['IRAS2err'][ulirg_ix]*1E6,
	goals_data['IRAS3err'][ulirg_ix]*1E6,
	goals_data['MIPS2err'][ulirg_ix]*1E6,
	goals_data['PACS2err'][ulirg_ix]*1E6,
	goals_data['MIPS3err'][ulirg_ix]*1E6,
	goals_data['F250err'][ulirg_ix]*1E6,
	goals_data['F350err'][ulirg_ix]*1E6,
	goals_data['F500err'][ulirg_ix]*1E6,
	goals_data['SCUBA2err'][ulirg_ix]*1E6,
	goals_data['VLA1err'][ulirg_ix]*1E6,
	goals_data['VLA2err'][ulirg_ix]*1E6
	])

ulirg_cgoals_flux = ulirg_cgoals_flux.T
ulirg_cgoals_flux_err = ulirg_cgoals_flux_err.T


single_cgoals_nan_array = np.zeros(np.shape(goals_data['HX'][goals_photID == single]))
single_cgoals_nan_array[single_cgoals_nan_array == 0] = np.nan
single_cgoals_flux = np.asarray([
	# single_cgoals_Fx_hard_match_mjy*1000,
	# single_cgoals_Fx_soft_match_mjy*1000,
	goals_data['HX'][goals_photID == single]*1E6,
	goals_data['SX'][goals_photID == single]*1E6,
	single_cgoals_nan_array,
	goals_data['FUV'][goals_photID == single]*1E6,
	goals_data['NUV'][goals_photID == single]*1E6,
	goals_data['U'][goals_photID == single]*1E6,
	goals_data['B'][goals_photID == single]*1E6,
	goals_data['V'][goals_photID == single]*1E6,
	goals_data['R'][goals_photID == single]*1E6,
	goals_data['I'][goals_photID == single]*1E6,
	goals_data['J'][goals_photID == single]*1E6,
	goals_data['H'][goals_photID == single]*1E6,
	goals_data['Ks'][goals_photID == single]*1E6,
	goals_data['IRAC1'][goals_photID == single]*1E6,
	goals_data['IRAC2'][goals_photID == single]*1E6,
	goals_data['IRAC3'][goals_photID == single]*1E6,
	goals_data['IRAC4'][goals_photID == single]*1E6,
	goals_data['IRAS1'][goals_photID == single]*1E6,
	goals_data['MIPS1'][goals_photID == single]*1E6,
	goals_data['IRAS2'][goals_photID == single]*1E6,
	goals_data['IRAS3'][goals_photID == single]*1E6,
	goals_data['MIPS2'][goals_photID == single]*1E6,
	goals_data['PACS2'][goals_photID == single]*1E6,
	goals_data['MIPS3'][goals_photID == single]*1E6,
	goals_data['F250'][goals_photID == single]*1E6,
	goals_data['F350'][goals_photID == single]*1E6,
	goals_data['F500'][goals_photID == single]*1E6,
	goals_data['SCUBA2'][goals_photID == single]*1E6,
	goals_data['VLA1'][goals_photID == single]*1E6,
	goals_data['VLA2'][goals_photID == single]*1E6
])

single_cgoals_flux_err = np.asarray([
	# single_cgoals_Fx_hard_match_mjy*1000*0.2,
	# single_cgoals_Fx_soft_match_mjy*1000*0.2,
	goals_data['HXerr'][goals_photID == single]*1E6*0.2,
	goals_data['SXerr'][goals_photID == single]*1E6*0.2,
	single_cgoals_nan_array,
	goals_data['FUVerr'][goals_photID == single]*1E6,
	goals_data['NUVerr'][goals_photID == single]*1E6,
	goals_data['Uerr'][goals_photID == single]*1E6,
	goals_data['Berr'][goals_photID == single]*1E6,
	goals_data['Verr'][goals_photID == single]*1E6,
	goals_data['Rerr'][goals_photID == single]*1E6,
	goals_data['Ierr'][goals_photID == single]*1E6,
	goals_data['Jerr'][goals_photID == single]*1E6,
	goals_data['Herr'][goals_photID == single]*1E6,
	goals_data['Kserr'][goals_photID == single]*1E6,
	goals_data['IRAC1err'][goals_photID == single]*1E6,
	goals_data['IRAC2err'][goals_photID == single]*1E6,
	goals_data['IRAC3err'][goals_photID == single]*1E6,
	goals_data['IRAC4err'][goals_photID == single]*1E6,
	goals_data['IRAS1err'][goals_photID == single]*1E6,
	goals_data['MIPS1err'][goals_photID == single]*1E6,
	goals_data['IRAS2err'][goals_photID == single]*1E6,
	goals_data['IRAS3err'][goals_photID == single]*1E6,
	goals_data['MIPS2err'][goals_photID == single]*1E6,
	goals_data['PACS2err'][goals_photID == single]*1E6,
	goals_data['MIPS3err'][goals_photID == single]*1E6,
	goals_data['F250err'][goals_photID == single]*1E6,
	goals_data['F350err'][goals_photID == single]*1E6,
	goals_data['F500err'][goals_photID == single]*1E6,
	goals_data['SCUBA2err'][goals_photID == single]*1E6,
	goals_data['VLA1err'][goals_photID == single]*1E6,
	goals_data['VLA2err'][goals_photID == single]*1E6
])

single_cgoals_flux = single_cgoals_flux.T
single_cgoals_flux_err = single_cgoals_flux_err.T


ned_goals_ID = np.asarray(ned_data['ID'])

print('NED ID: ', ned_goals_ID)
# print(ned_goals_ID)
# print(ulirgs)

# ix2, iy2 = match(ned_goals_ID, ulirgs)
# ned_goals_ID = ned_goals_ID[ix2]

ix,iy = match(ned_goals_ID,ulirg_ID)

ned_goals_ID_match = ned_goals_ID[ix]
ned_goals_LIR_match = 10**ulirg_LIR[iy]

ned_goals_Lx = 10**ulirg_Lx[iy]
# ned_goals_Lx = np.log10((10**ulirg_Lx[iy])*0.638) # convert Lx(2-10keV) to Lx(0.5-2keV)
ned_goals_Lx_obs = ulirg_Lx_obs[iy]
ned_goals_z = ulirg_z[iy]
ned_goals_Nh = np.log10(ulirg_Nh[iy])

for i in range(len(np.asarray(ned_data['HX'][ix],dtype=float))):
	ulirgs_observed_hard.append(np.asarray(ned_data['HX'][ix],dtype=float)[i]*4.136E8/(10-2)*1000)
	ulirgs_observed_soft.append(np.asarray(ned_data['SX'][ix],dtype=float)[i]*4.136E8/(2-0.5)*1000)
	ulirgs_corrected_hard.append(np.asarray(ulirg_Fhx[iy],dtype=float)[i]*4.136E8/(10-2)*1000)
	ulirgs_corrected_soft.append(np.asarray(ulirg_Fsx[iy],dtype=float)[i]*4.136E8/(2-0.5)*1000)


ned_goals_Fhx = np.asarray(ulirg_Fhx[iy],dtype=float)
ned_goals_Fsx = np.asarray(ulirg_Fsx[iy],dtype=float)
# ned_goals_Fhx = np.asarray(ned_data['HX'][ix],dtype=float)
# ned_goals_Fsx = np.asarray(ned_data['SX'][ix],dtype=float)

ned_goals_Fx_hard_match_mjy = ned_goals_Fhx*4.136E8/(10-2)
ned_goals_Fx_soft_match_mjy = ned_goals_Fsx*4.136E8/(2-0.5)
print('GOALS:',ned_goals_Fx_hard_match_mjy)


ned_goals_nan_array = np.zeros(np.shape(ned_data['HX'][ix]))
ned_goals_nan_array[ned_goals_nan_array == 0] = np.nan
ned_goals_flux = np.asarray([
	ned_goals_Fx_hard_match_mjy*1000,
	ned_goals_Fx_soft_match_mjy*1000,
	ned_goals_nan_array,
	np.asarray(ned_data['FUV'][ix],dtype=float)*1E6,
	np.asarray(ned_data['NUV'][ix],dtype=float)*1E6,
	np.asarray(ned_data['u'][ix],dtype=float)*1E6,
	np.asarray(ned_data['U'][ix],dtype=float)*1E6,
	np.asarray(ned_data['B'][ix],dtype=float)*1E6,
	np.asarray(ned_data['g'][ix],dtype=float)*1E6,
	np.asarray(ned_data['V'][ix],dtype=float)*1E6,
	np.asarray(ned_data['r'][ix],dtype=float)*1E6,
	np.asarray(ned_data['i'][ix],dtype=float)*1E6,
	np.asarray(ned_data['z'][ix],dtype=float)*1E6,
	np.asarray(ned_data['J'][ix],dtype=float)*1E6,
	np.asarray(ned_data['H'][ix],dtype=float)*1E6,
	np.asarray(ned_data['Ks'][ix],dtype=float)*1E6,
	# np.asarray(ned_data['W1'][ix],dtype=float)*1E6,
	np.asarray(ned_data['IRAC1'][ix],dtype=float)*1E6,
	np.asarray(ned_data['IRAC2'][ix],dtype=float)*1E6,
	# np.asarray(ned_data['W2'][ix],dtype=float)*1E6,
	np.asarray(ned_data['IRAC3'][ix],dtype=float)*1E6,
	np.asarray(ned_data['IRAC4'][ix],dtype=float)*1E6,
	# np.asarray(ned_data['W3'][ix],dtype=float)*1E6,
	np.asarray(ned_data['IRAS1'][ix],dtype=float)*1E6,
	# np.asarray(ned_data['W4'][ix],dtype=float)*1E6,
	# np.asarray(ned_data['MIPS1'][ix],dtype=float)*1E6,
	# np.asarray(ned_data['IRAS2'][ix],dtype=float)*1E6,
	np.asarray(ned_data['IRAS3'][ix],dtype=float)*1E6,
	np.asarray(ned_data['MIPS2'][ix],dtype=float)*1E6,
	np.asarray(ned_data['IRAS4'][ix],dtype=float)*1E6,
	np.asarray(ned_data['MIPS3'][ix],dtype=float)*1E6,
	np.asarray(ned_data['250'][ix],dtype=float)*1E6,
	np.asarray(ned_data['350'][ix],dtype=float)*1E6,
	np.asarray(ned_data['500'][ix],dtype=float)*1E6
	])



ned_goals_flux = ned_goals_flux.T
# ned_goals_flux_err = ned_goals_flux_err.T
ned_goals_flux_err = ned_goals_flux*0.2

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

scale_array = [1.87E44,2.33E44,3.93E44]
###################################



##### Figure 1 & 2 plots #####

plt.rcParams['font.size']=24
plt.rcParams['axes.linewidth']=3
plt.rcParams['xtick.major.size']=3
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.size']=3
plt.rcParams['ytick.major.width'] = 3

for i in range(len(ulirg_ID)):
    print(ulirg_ID[i],ulirg_z[i],ulirg_Lx[i])

# plt.figure(figsize=(10,10))
# plt.plot(goodsS_auge_z_match,np.log10(goodsS_auge_Lx_match),'.',color='gray',rasterized=True)
# plt.plot(goodsN_auge_z_match,np.log10(goodsN_auge_Lx_match),'.',color='gray',rasterized=True,label='GOODS-N/S')
# # plt.plot(xue_z_match,np.log10(xue_Lx_match),'.',color='gray',rasterized=True)
# # plt.plot(luo_z_match,np.log10(luo_Lx_match),'.',color='gray',rasterized=True,label='GOODS-N/S')
# plt.plot(chandra_cosmos_z_match,np.log10(chandra_cosmos_Lx_full_match),'+',ms=10,color='b',rasterized=True,alpha=0.8,label='COSMOS')
# # plt.plot(s82x_z_sp_match,np.log10(s82x_≥Lx_sp_full_match),'x',ms=10,color='r',rasterized=True,alpha=0.8,label='Stripe82X')
# plt.plot(s82x_z_sp,np.log10(s82x_Lx_sp_full),'x',ms=10,color='r',rasterized=True,alpha=0.8,label='Stripe82X')
# # plt.plot(ulirg_z,ulirg_Lx,'*',color='g',ms=12,rasterized=True,label='GOALS')
# plt.xlabel('Spectroscopic Redshift')
# plt.ylabel(r'log$_{10}$ L$_{0.5 - 10\mathrm{keV}}$ [erg s$^{-1}$]')
# # plt.text(2.15, 40.55, f'n = {len(goodsS_auge_z_match)+len(goodsN_auge_z_match)+len(xue_z_match)+len(luo_z_match)+len(chandra_cosmos_z_match)+len(s82x_z_sp)+len(ulirg_z)}')
# plt.text(2.15, 40.55, f'n = {len(goodsS_auge_z_match)+len(goodsN_auge_z_match)+len(chandra_cosmos_z_match)+len(s82x_z_sp)+len(ulirg_z)}')
# # plt.text(3.25, 40.55, f'n = {len(xue_z_match)+len(luo_z_match)+len(chandra_cosmos_z_match)+len(s82x_z_sp)}')
# plt.legend()
# plt.xlim(-0.075,5.5)
# plt.grid()
# plt.tight_layout()
# plt.savefig('/Users/connor_auge/Desktop/Final_plots/Lx_z_spec.pdf')
# plt.show()

# cdf_z = np.append(xue_z_match,luo_z_match)
# cdf_lx = np.append(xue_Lx_match,luo_Lx_match)

# cdf_z = np.append(cdf_z, goodsS_auge_z_match)
# cdf_lx = np.append(cdf_lx, goodsS_auge_Lx_match)

cdf_z = np.append(goodsS_auge_z_match, goodsN_auge_z_match)
cdf_lx = np.append(goodsS_auge_Lx_match, goodsN_auge_Lx_match)

print('Stripe82X: ',len(s82x_Lx_sp_full))
print('COSMOS: ',len(chandra_cosmos_z_match))
print('GOODS: ',len(cdf_z))

# fig = plt.figure(figsize=(20,10))
# ax1 = plt.subplot(121)
# ax1.hist(cdf_z,bins=np.arange(0,6,0.25),histtype='step',color='gray',lw=3,label='GOODS-N/S')
# ax1.hist(chandra_cosmos_z_match,bins=np.arange(0,6,0.25),histtype='step',color='b',lw=3,label='COSMOS')
# ax1.hist(s82x_z_sp,bins=np.arange(0,6,0.25),histtype='step',color='r',lw=3,label='Stripe82X')
# ax1.set_xlabel('Spectroscopic Redshift')
# ax1.set_xlim(-0.075, 5.5)
# ax1.set_ylim(0,475)
# ax1.set_ylabel('N')
# ax1.grid()

# ax2 = plt.subplot(122)
# ax2.hist(np.log10(cdf_lx),bins=np.arange(39,46.5,0.25),histtype='step',color='gray',lw=3)
# ax2.hist(np.log10(chandra_cosmos_Lx_full_match),bins=np.arange(39,46.5,0.25),histtype='step',color='b',lw=3)
# ax2.hist(np.log10(s82x_Lx_sp_full),bins=np.arange(39,46.5,0.25),histtype='step',color='r',lw=3)
# ax2.set_xlabel(r'log$_{10}$ L$_{0.5 - 10\mathrm{keV}}$ [erg s$^{-1}$]')
# # ax2.set_ylabel('N')
# ax2.set_ylim(0,475)
# ax2.set_yticklabels([])
# ax2.grid()
# plt.tight_layout()
# plt.savefig('/Users/connor_auge/Desktop/Final_Plots/Lx_z_hist_spec.pdf')
# plt.show()


##############################

###############################################################################
COSMOS_filters = np.array(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'G', 'R', 'I', 'Z', 'yHSC_FLUX_APER2', 'J_FLUX_APER2', 'H_FLUX_APER2',
                          'Ks_FLUX_APER2', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])
S82X_filters = np.asarray(['Fx_hard','Fx_soft','nan','MAG_FUV','MAG_NUV','U','G','R','I','Z','JVHS','HVHS','KVHS','W1','W2','W3','W4','nan','FLUX_250_s82x','FLUX_350_s82x','FLUX_500_s82x'])
# S82X_filters = np.asarray(['Fx_hard','Fx_soft','nan','MAG_FUV','MAG_NUV','U','G','R','I','Z','JVHS','HVHS','KVHS','W1','W2','W3','W4','nan'])
GOODSN_filters  = np.asarray(['Fx_hard','Fx_soft','nan','FLUX_GALEX_FUV','FLUX_GALEX_NUV','u_FLUX_APER2','F435W','F606W','F775W','F814W','F105W','F125W','F140W','F160W','Ks_FLUX_APER2','SPLASH_1_FLUX','SPLASH_2_FLUX','SPLASH_3_FLUX','SPLASH_4_FLUX','FLUX_24','MIPS2','FLUX_100','FLUX_160','FLUX_250','FLUX_350','FLUX_500'])
GOODSS_filters = np.asarray(['Fx_hard','Fx_soft','nan','u_FLUX_APER2','F435W','F606W','F775W','F814W','F098M','F105W','F125W','F160W','Ks_FLUX_APER2','SPLASH_1_FLUX','SPLASH_2_FLUX','SPLASH_3_FLUX','SPLASH_4_FLUX','FLUX_24','MIPS2','FLUX_100','FLUX_160','FLUX_250','FLUX_350','FLUX_500'])

# GOODSN_auge_filters = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'U', 'B_FLUX_APER2', 'V_FLUX_APER2', 'R', 'I', 'Z', 'JVHS', 'HVHS', 'KVHS', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])
# GOODSS_auge_filters = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'MAG_FUV', 'MAG_NUV','U', 'F435W', 'B_FLUX_APER2', 'V_FLUX_APER2', 'F606W', 'R', 'I', 'F775W', 'F814W', 'Z', 'F098M', 'F125W', 'JVHS', 'HVHS', 'KVHS', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'MIPS2', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])
# GOODSS_auge_filters = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'IB427_FLUX', 'F435W', 'IB445_FLUX', 'B_FLUX_APER2', 'IB464_FLUX_APER2', 'IA484_FLUX_APER2', 'IB505_FLUX_APER2', 'IA527_FLUX_APER2', 'V_FLUX_APER2', 'IB550_FLUX', 'IB574_FLUX_APER2', 'F606W', 'IB598_FLUX', 'IA624_FLUX_APER2', 'R', 'IB651_FLUX', 'IA679_FLUX_APER2', 'IB709_FLUX_APER2', 'IA738_FLUX_APER2', 'I', 
                                #   'F775W', 'IA767_FLUX_APER2', 'IA797_FLUX', 'F814W', 'IB827_FLUX_APER2', 'IA856_FLUX', 'Z','F850LP', 'F098M', 'F105W', 'F125W', 'JVHS', 'F140W', 'F160W', 'HVHS', 'KVHS', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'MIPS2', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])
# GOODSN_auge_filters = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'B_FLUX_APER2', 'V_FLUX_APER2', 'F606W', 'R', 'I', 'F814W', 'Z', 'F125W', 'JVHS', 'F140W', 'F160W', 'HVHS', 'KVHS',
                                #  'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])
GOODSS_auge_filters = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'F435W', 'B_FLUX_APER2', 'V_FLUX_APER2', 'F606W', 'R', 'I',
                                  'F775W', 'F814W', 'Z', 'F850LP', 'F098M', 'F105W', 'F125W', 'JVHS', 'F140W', 'F160W', 'HVHS', 'KVHS', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'MIPS2', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])

GOODSN_auge_filters = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'F435W', 'B_FLUX_APER2', 'V_FLUX_APER2', 'F606W', 'R', 'I', 'F775W', 'F814W', 'Z', 'F105W', 'F125W', 'JVHS', 'F140W', 'F160W', 'HVHS', 'KVHS',
                                   'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'FLUX_24', 'MIPS2','FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])

cgoals_filter_name = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'B_FLUX_APER2', 'V_FLUX_APER2', 'R', 'I', 'J_FLUX_APER2', 'H_FLUX_APER2',
                                'Ks_FLUX_APER2', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'IRAS1', 'FLUX_24', 'IRAS2', 'IRAS3', 'MIPS2', 'FLUX_100', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500', 'SCUBA2', 'VLA1', 'VLA2'])
# ned_goals_filter_name = np.asarray(['Fx_hard','Fx_soft','nan','FLUX_GALEX_FUV','FLUX_GALEX_NUV','U','u_FLUX_APER2','B_FLUX_APER2','G','V_FLUX_APER2','R','I','Z','J_2mass','H_2mass','Ks_2mass','W1','SPLASH_1_FLUX','SPLASH_2_FLUX','W2','SPLASH_3_FLUX','SPLASH_4_FLUX','W3','IRAS1','W4','FLUX_24','IRAS2','IRAS3','MIPS2','IRAS4','FLUX_160','FLUX_250','FLUX_350','FLUX_500'])
ned_goals_filter_name = np.asarray(['Fx_hard', 'Fx_soft', 'nan', 'FLUX_GALEX_FUV', 'FLUX_GALEX_NUV', 'U', 'u_FLUX_APER2', 'B_FLUX_APER2', 'G', 'V_FLUX_APER2', 'R', 'I', 'Z', 'J_2mass',
                                   'H_2mass', 'Ks_2mass', 'SPLASH_1_FLUX', 'SPLASH_2_FLUX', 'SPLASH_3_FLUX', 'SPLASH_4_FLUX', 'IRAS1', 'IRAS3', 'MIPS2', 'IRAS4', 'FLUX_160', 'FLUX_250', 'FLUX_350', 'FLUX_500'])
# ULIRS_filter_name = np.asarray(['Fx_hard','Fx_soft','FLUX_GALEX_FUV','FLUX_GALEX_NUV','U','B_FLUX_APER2','V_FLUX_APER2','R','I','J_FLUX_APER2','H_FLUX_APER2','Ks_FLUX_APER2','SPLASH_1_FLUX','SPLASH_2_FLUX','SPLASH_3_FLUX','SPLASH_4_FLUX','IRAS1','FLUX_24','IRAS2','IRAS3','MIPS2','FLUX_100','SCUBA2','VLA1','VLA2'])

###############################################################################
tfl = time.perf_counter()
print(f'Done with file reading ({tfl - ti:0.4f} second)')

# Make empty lists to be filled with SED outputs
all_id, all_z, all_x, all_y, all_frac_err = [], [], [], [], []
F1, F025, F6, F10, F100 = [], [], [], [], []
Lx, Lbol = [], []
Lx_hard = []
field = []
UVslope, MIRslope1, MIRslope2 = [], [], []
spec_type = []
Nh = []
FFIR, WFIR = [], []
median_x, median_y = [], []
median_fir_x, median_fir_y = [], []
upper_check, fir_frac = [], []
FFIR_2, WFIR_2, F100_2 = [], [], []
s82x_Lx = []
check_sed = []
uv_lum, opt_lum, mir_lum, fir_lum = [], [], [], []
Lbol_sub = []
sed_shape = []
irac_ch1, irac_ch2, irac_ch3, irac_ch4 = [], [], [], []


###############################################################################
xcigale_name = 'shape2_cosmos.mag'
inf = open('../xcigale/data_input/'+xcigale_name,'w')
header = np.asarray(['# id','redshift'])
x_cigale_filters = Filters('filter_list.dat').pull_filter(COSMOS_filters,'xcigale name')
for i in range(len(x_cigale_filters)):
	header = np.append(header,x_cigale_filters[i])
	header = np.append(header,x_cigale_filters[i]+'_err')
np.savetxt(inf,header,fmt='%s',delimiter='    ',newline=' ')
inf.close()
###############################################################################

###############################################################################
###############################################################################
############################### Run COSMOS SEDs ###############################
fill_nan = np.zeros(len(GOODSS_auge_filters)-len(COSMOS_filters))
fill_nan[fill_nan == 0] = np.nan

cosmos_target_id = []
cosmos_target_ra = []
cosmos_target_dec = []
# '''
a = 0
for i in range(len(chandra_cosmos_phot_id_match)):
    # if chandra_cosmos_phot_id_match[i] == 222544:
        source = AGN(chandra_cosmos_phot_id_match[i],chandra_cosmos_z_match[i],COSMOS_filters,cosmos_flux_array[i],cosmos_flux_err_array[i])
        source.MakeSED()
        check = source.CheckSED(10, check_span=2.75)
        check_sed.append(check)
        f1 = source.Find_nuFnu(1.0)

        F1.append(f1)
        F025.append(source.Find_nuFnu(0.25)/source.Find_nuFnu(1.0))
        F6.append(source.Find_nuFnu(6.0)/source.Find_nuFnu(1.0))
        F10.append(source.Find_nuFnu(10.0)/source.Find_nuFnu(1.0))
        # F100_2.append(source.Find_nuFnu(100)/source.Find_nuFnu(1.0))

        # ffir, wfir,f100 = source.median_FIR_filter(['FLUX_24','FLUX_250','FLUX_350','FLUX_500'],Find_value=100.0)
        ffir, wfir,f100 = source.median_FIR_filter(['FLUX_24','FLUX_100', 'FLUX_160','FLUX_250','FLUX_350','FLUX_500'],Find_value=100.0)
        # print('w: ', wfir)
        # print('f: ', ffir)
        FFIR.append(ffir)
        WFIR.append(wfir)
        F100.append(f100/source.Find_nuFnu(1.0))

        # FFIR_2.append(ffir2)
        # WFIR_2.append(wfir2)
        # F100_2.append(f1002/source.Find_nuFnu(1.0))

        uvs = source.Find_slope(0.15, 1.0)
        mirs1 = source.Find_slope(1.0,6.5)
        mirs2 = source.Find_slope(6.5,10)

        UVslope.append(uvs)
        MIRslope1.append(mirs1)
        MIRslope2.append(mirs2)

        uv_lum.append(source.find_Lum_range(0.1,0.35))
        opt_lum.append(source.find_Lum_range(0.35,3))
        mir_lum.append(source.find_Lum_range(3,30))
        fir_lum.append(source.find_Lum_range(30,500/(1+chandra_cosmos_z_match[i])))

        Id, redshift, w, f, frac_err, up_check = source.pull_plot_info()
        w = np.append(w,fill_nan)
        f = np.append(f,fill_nan)
        frac_err = np.append(frac_err,fill_nan)
        all_id.append(Id)
        all_z.append(redshift)
        all_x.append(w)
        all_y.append(f)
        all_frac_err.append(frac_err)
        upper_check.append(up_check)

        # if chandra_cosmos_phot_id_match[i] == 222544:
        # #     print(chandra_cosmos_xid_match[i])
        # plot = Plotter(Id, redshift, w, f, frac_err, np.log10(f1))
        # plot.PlotSingleSED(flux_point=f100/f1, wfir=wfir, ffir=ffir)
        # elif chandra_cosmos_phot_id_match[i] == 235265:
        #     # print(chandra_cosmos_xid_match[i])
        # plot = Plotter(Id,redshift,w,f,frac_err,np.log10(chandra_cosmos_Lx_full_match[i]))
        # plot.PlotSingleSED(flux_point=f100/source.Find_nuFnu(1.0),wfir=wfir,ffir=ffir/source.Find_nuFnu(1.0))

        med_x, med_y = source.median_SED(['U'], ['FLUX_24'])
        median_x.append(med_x)
        median_y.append(med_y)

        med_fir_x, med_fir_y = source.median_SED(['FLUX_24'], ['FLUX_500'])
        median_fir_x.append(med_fir_x)
        median_fir_y.append(med_fir_y)

        Lx.append(chandra_cosmos_Lx_full_match[i])
        Lx_hard.append(chandra_cosmos_Lx_hard_match[i])
        # Lx.append(chandra_cosmos_Lx_hard_match[i])
        Nh.append(chandra_cosmos_Nh_match[i])
        Lbol.append(source.Find_Lbol())
        Lbol_sub.append(source.Find_Lbol_temp_sub(scale_array,temp_wave,temp_lum))
        fir_frac.append(source.FIR_frac())

        field.append(0)

        cosmos_target_id.append(chandra_cosmos_phot_id_match[i])
        cosmos_target_ra.append(chandra_cosmos_RA_match[i])
        cosmos_target_dec.append(chandra_cosmos_DEC_match[i])

        shape = source.SED_shape(UVslope[i],MIRslope1[i],MIRslope2[i])
        sed_shape.append(shape)

        # print(shape,a)
        if check == 'GOOD':
            if shape == 2:
                a += 1
                if a <= 30:
                    source.write_xcigale_input(xcigale_name,COSMOS_filters)
                else:
                    continue
            else:
                continue

        # irac_ch1.append(cosmos_flux_array[i][COSMOS_filters == 'SPLASH_1_FLUX'][0])
        # irac_ch2.append(cosmos_flux_array[i][COSMOS_filters == 'SPLASH_2_FLUX'][0])
        # irac_ch3.append(cosmos_flux_array[i][COSMOS_filters == 'SPLASH_3_FLUX'][0])
        # irac_ch4.append(cosmos_flux_array[i][COSMOS_filters == 'SPLASH_4_FLUX'][0]) 

        # if check_sed[i] == 'GOOD':
        #     source.SED_output('Test_out_cosmos2.fits','w')
        #     source.AGN_output('Test_out_cosmos_prop2.fits',chandra_cosmos_Lx_full_match[i],chandra_cosmos_Nh_match[i],source.Find_nuFnu(1.0),'w')
        # # print(chandra_cosmos_Lx_full_match[i])
# '''
tc = time.perf_counter()   
print(f'Done with COSMOS sources ({tc - tfl:0.4f} second)')

# cosmos_target_ra = np.asarray(cosmos_target_ra)
# cosmos_target_dec = np.asarray(cosmos_target_dec)
# cosmos_ra_out2 = []
# cosmos_dec_out2 = []
# c2 = SkyCoord(ra=cosmos_target_ra*u.degree, dec=cosmos_target_dec*u.degree)
# str_list = c2.to_string('hmsdms')
# ra_out2, dec_out2 = [], []
# for i in range(len(str_list)):
#     str_list[i] = str_list[i].replace("h","")
#     str_list[i] = str_list[i].replace("m", "")
#     str_list[i] = str_list[i].replace("d", "")
#     str_list[i] = str_list[i].replace("s", "")
#     out_str = str_list[i].split()

#     cosmos_ra_out2.append(out_str[0])
#     cosmos_dec_out2.append(out_str[1])

# outf = open('/Users/connor_auge/Desktop/all_cosmos_targets.csv','w')
# for i in range(len(cosmos_target_id)):
#     outf.writelines('%s,%s,%s,%s\n' % (cosmos_target_id[i],cosmos_ra_out2[i],cosmos_dec_out2[i],2000))

ra_out = []
dec_out = []
###############################################################################
###############################################################################
############################# Run Stripe82X SEDs ##############################
fill_nan = np.zeros(len(GOODSS_auge_filters)-len(S82X_filters))
fill_nan[fill_nan == 0] = np.nan
# '''
for i in range(len(lamassa_id_use)):
        try:
            source = AGN(lamassa_id_use[i], s82x_z_sp[i], S82X_filters, s82x_flux_array[i], s82x_flux_err_array[i])
            source.MakeSED()
            check = source.CheckSED(10, check_span=2.75)
            check_sed.append(check)
           

            F1.append(source.Find_nuFnu(1.0))
            F025.append(source.Find_nuFnu(0.25)/source.Find_nuFnu(1.0))
            F6.append(source.Find_nuFnu(6.0)/source.Find_nuFnu(1.0))
            F10.append(source.Find_nuFnu(10.0)/source.Find_nuFnu(1.0))
            # F100_2.append(source.Find_nuFnu(100)/source.Find_nuFnu(1.0))

            ffir, wfir, f100 = source.median_FIR_filter(['W4', 'FLUX_250', 'FLUX_350_s82x', 'FLUX_500_s82x'],Find_value=100.0)
            ffir = np.append(np.array([np.nan, np.nan]), ffir)
            wfir = np.append(np.array([np.nan, np.nan]), wfir)
            FFIR.append(ffir)
            WFIR.append(wfir)
            F100.append(f100/source.Find_nuFnu(1.0))

            uvs = source.Find_slope(0.15, 1.0)
            mirs1 = source.Find_slope(1.0, 6.5)
            mirs2 = source.Find_slope(6.5, 10)

            UVslope.append(uvs)
            MIRslope1.append(mirs1)
            MIRslope2.append(mirs2)

            uv_lum.append(source.find_Lum_range(0.1,0.35))
            opt_lum.append(source.find_Lum_range(0.35,3))
            mir_lum.append(source.find_Lum_range(3,30))
            fir_lum.append(source.find_Lum_range(30,500/(1+s82x_z_sp[i])))

            # if check_sed[i] == 'GOOD':
            #     source.find_Lum_range(3,30)
            # else:
            #     continue

            Id, redshift, w, f, frac_err, up_check = source.pull_plot_info()
            w = np.append(w,fill_nan)
            f = np.append(f,fill_nan)
            frac_err = np.append(frac_err,fill_nan)
            all_id.append(Id)
            all_z.append(redshift)
            all_x.append(w)
            all_y.append(f)
            all_frac_err.append(frac_err)
            upper_check.append(up_check)

            # if check_sed[i] == 'GOOD':
            plot = Plotter(Id,redshift,w,f,frac_err,np.log10(s82x_Lx_sp_full[i]))
            # if Id == 2363:
            #     print(uvs)
            #     plot.PlotSingleSED(flux_point=f100/source.Find_nuFnu(1.0),wfir=wfir,ffir=ffir/source.Find_nuFnu(1.0))

            med_x, med_y = source.median_SED(['U'], ['W4'])
            median_x.append(med_x)
            median_y.append(med_y)

            med_fir_x, med_fir_y = source.median_SED(['FLUX_24'], ['FLUX_500'])
            median_fir_x.append(med_fir_x)
            median_fir_y.append(med_fir_y)

            Lx.append(s82x_Lx_sp_full[i])
            Lx_hard.append(s82x_Lx_sp_hard[i])
            # Lx.append(s82x_Lx_sp_hard[i])
            Nh.append(s82x_Nh[i])
            Lbol.append(source.Find_Lbol())
            Lbol_sub.append(source.Find_Lbol_temp_sub(scale_array,temp_wave,temp_lum))
            fir_frac.append(source.FIR_frac())

            field.append(1)
            ra_out.append(lamassa_ra[i])
            dec_out.append(lamassa_dec[i])
            shape = source.SED_shape(uvs,mirs1,mirs2)
            sed_shape.append(shape)


            # if check == 'GOOD':
            #     if shape == 2:
            #         a += 1
            #         if a <= 20:
            #             source.write_xcigale_input(xcigale_name, S82X_filters)
            #         else:
            #             continue
            #     else:
            #         continue

            # irac_ch1.append(np.nan)
            # irac_ch2.append(np.nan)
            # irac_ch3.append(np.nan)
            # irac_ch4.append(np.nan) 

        except ValueError:
            continue

# '''
# outf = open('/Users/connor_auge/Desktop/s82x_check_coords.csv','w')
# outf.writelines('ID,RA,DEC\n')
# for i in range(len(ra_out)):
#     outf.writelines('%s,%s,%s\n' % (all_id[i],ra_out[i],dec_out[i]))
# outf.close()

ts = time.perf_counter()
print(f'Done with S82X sources ({ts - tc:0.4f} second)')
###############################################################################
###############################################################################
############################### Run GOODS-N SEDs ##############################

fill_nan = np.zeros(len(GOODSS_auge_filters)-len(GOODSN_filters))
fill_nan[fill_nan == 0] = np.nan

goodsN_id_candels = []
goodsN_id_auge = []

# '''
# for i in range(len(goodsN_phot_id_match)):
#     # if goodsN_flux_array[i][GOODSN_filters == 'FLUX_24'] <= 0:
#     #     continue
#     # elif goodsN_flux_err_array[i][GOODSN_filters == 'FLUX_24']/goodsN_flux_array[i][GOODSN_filters == 'FLUX_24'] > 0.5:
#     #     continue
#     # else:
#         source = AGN(goodsN_phot_id_match[i],xue_z_match[i],GOODSN_filters,goodsN_flux_array[i],goodsN_flux_err_array[i])
#         source.MakeSED()
#         check_sed.append(source.CheckSED(10, check_span=2.75))


#         F1.append(source.Find_nuFnu(1.0))
#         F025.append(source.Find_nuFnu(0.25)/source.Find_nuFnu(1.0))
#         F6.append(source.Find_nuFnu(6.0)/source.Find_nuFnu(1.0))
#         F10.append(source.Find_nuFnu(10.0)/source.Find_nuFnu(1.0))
#         # F100_2.append(source.Find_nuFnu(100)/source.Find_nuFnu(1.0))

#         ffir, wfir, f100 = source.median_FIR_filter(['FLUX_24','FLUX_100', 'FLUX_160','FLUX_250','FLUX_350','FLUX_500'],Find_value=100.0)
#         FFIR.append(ffir)
#         WFIR.append(wfir)
#         F100.append(f100/source.Find_nuFnu(1.0))

#         uvs = source.Find_slope(0.15, 1.0)
#         mirs1 = source.Find_slope(1.0, 6.5)
#         mirs2 = source.Find_slope(6.5, 10)

#         UVslope.append(uvs)
#         MIRslope1.append(mirs1)
#         MIRslope2.append(mirs2)

#         uv_lum.append(source.find_Lum_range(0.1,0.35))
#         opt_lum.append(source.find_Lum_range(0.35,3))
#         mir_lum.append(source.find_Lum_range(3,30))
#         fir_lum.append(source.find_Lum_range(30,500/(1+xue_z_match[i])))

#         Id, redshift, w, f, frac_err, up_check = source.pull_plot_info()
#         w = np.append(w,fill_nan)
#         f = np.append(f,fill_nan)
#         frac_err = np.append(frac_err, fill_nan)
#         all_id.append(Id)
#         all_z.append(redshift)
#         all_x.append(w)
#         all_y.append(f)
#         all_frac_err.append(frac_err)
#         upper_check.append(up_check)

#         med_x, med_y = source.median_SED(['U'], ['MIPS2'])
#         median_x.append(med_x)
#         median_y.append(med_y)
        
#         med_fir_x, med_fir_y = source.median_SED(['FLUX_24'], ['FLUX_500'])
#         median_fir_x.append(med_fir_x)
#         median_fir_y.append(med_fir_y)

#         Lx.append(xue_Lx_match[i])
#         Lx_hard.append(xue_Lx_hard_match[i])
#         # Lx.append(xue_Lx_hard_match[i])
#         Nh.append(goodsN_Nh_match[i])
#         Lbol.append(source.Find_Lbol())
#         Lbol_sub.append(source.Find_Lbol_temp_sub(scale_array,temp_wave,temp_lum))
#         fir_frac.append(source.FIR_frac())
#         shape = source.SED_shape(uvs, mirs1, mirs2)
#         sed_shape.append(shape)

#         # irac_ch1.append(goodsN_flux_array[i][GOODSN_filters == 'SPLASH_1_FLUX'][0])
#         # irac_ch2.append(goodsN_flux_array[i][GOODSN_filters == 'SPLASH_2_FLUX'][0])
#         # irac_ch3.append(goodsN_flux_array[i][GOODSN_filters == 'SPLASH_3_FLUX'][0])
#         # irac_ch4.append(goodsN_flux_array[i][GOODSN_filters == 'SPLASH_4_FLUX'][0]) 
        
#         if source.CheckSED(10, check_span=2.5) == 'GOOD':
#             goodsN_id_candels.append(goodsN_phot_id_match[i])

#         field.append(2)

fill_nan = np.zeros(len(GOODSS_auge_filters)-len(GOODSN_auge_filters))
fill_nan[fill_nan == 0] = np.nan

for i in range(len(goodsN_auge_ID_match)):
    # if goodsN_auge_ID_match[i] in xue_id_match:
    #     print('repeat')
    # if goodsN_auge_ID_match[i] == 348:
        # continue
    # else:
        try:
            source = AGN(goodsN_auge_ID_match[i],goodsN_auge_z_match[i],GOODSN_auge_filters,goodsN_flux_array_auge[i],goodsN_flux_err_array_auge[i])
            source.MakeSED()
            check_sed.append(source.CheckSED(10, check_span=2.75))
            # print(check_sed[i])


            F1.append(source.Find_nuFnu(1.0))
            F025.append(source.Find_nuFnu(0.25)/source.Find_nuFnu(1.0))
            F6.append(source.Find_nuFnu(6.0)/source.Find_nuFnu(1.0))
            F10.append(source.Find_nuFnu(10.0)/source.Find_nuFnu(1.0))
            F100_2.append(source.Find_nuFnu(100)/source.Find_nuFnu(1.0))

            ffir, wfir, f100 = source.median_FIR_filter(['FLUX_24','FLUX_100', 'FLUX_160','FLUX_250','FLUX_350','FLUX_500'],Find_value=100.0)
            FFIR.append(ffir)
            WFIR.append(wfir)
            F100.append(f100/source.Find_nuFnu(1.0))

            uvs = source.Find_slope(0.15, 1.0)
            mirs1 = source.Find_slope(1.0, 6.5)
            mirs2 = source.Find_slope(6.5, 10)

            UVslope.append(uvs)
            MIRslope1.append(mirs1)
            MIRslope2.append(mirs2)

            uv_lum.append(source.find_Lum_range(0.1,0.35))
            opt_lum.append(source.find_Lum_range(0.35,3))
            mir_lum.append(source.find_Lum_range(3,30))
            fir_lum.append(source.find_Lum_range(30,500/(1+goodsN_auge_z_match[i])))

            Id, redshift, w, f, frac_err, up_check = source.pull_plot_info()
            w = np.append(w,fill_nan)
            f = np.append(f,fill_nan)
            frac_err = np.append(frac_err,fill_nan)
            all_id.append(Id)
            all_z.append(redshift)
            all_x.append(w)
            all_y.append(f)
            all_frac_err.append(frac_err)
            upper_check.append(up_check)
            
            # if source.CheckSED(10, check_span=2.5) == 'GOOD':
            #     plot = Plotter(Id,redshift,w,f,frac_err,np.log10(goodsN_auge_Lx_match[i]))
            #     plot.PlotSingleSED(flux_point=f100/source.Find_nuFnu(1.0),wfir=wfir,ffir=ffir/source.Find_nuFnu(1.0))

            med_x, med_y = source.median_SED(['U'], ['FLUX_24'])
            median_x.append(med_x)
            median_y.append(med_y)

            med_fir_x, med_fir_y = source.median_SED(['FLUX_24'], ['FLUX_500'])
            median_fir_x.append(med_fir_x)
            median_fir_y.append(med_fir_y)

            Lx.append(goodsN_auge_Lx_match[i])
            Lx_hard.append(goodsN_auge_Lx_hard_match[i])
            # Lx.append(goodsN_auge_Lx_hard_match[i])    
            Nh.append(0.0)
            Lbol.append(source.Find_Lbol())
            Lbol_sub.append(source.Find_Lbol_temp_sub(scale_array,temp_wave,temp_lum))
            fir_frac.append(source.FIR_frac())
            shape = source.SED_shape(uvs, mirs1, mirs2)
            sed_shape.append(shape)

            # irac_ch1.append(goodsN_flux_array_auge[i][GOODSN_auge_filters == 'SPLASH_1_FLUX'][0])
            # irac_ch2.append(goodsN_flux_array_auge[i][GOODSN_auge_filters == 'SPLASH_2_FLUX'][0])
            # irac_ch3.append(goodsN_flux_array_auge[i][GOODSN_auge_filters == 'SPLASH_3_FLUX'][0])
            # irac_ch4.append(goodsN_flux_array_auge[i][GOODSN_auge_filters == 'SPLASH_4_FLUX'][0]) 

            if source.CheckSED(10, check_span=2.5) == 'GOOD':
                goodsN_id_auge.append(goodsN_auge_ID_match[i])

            field.append(2)
        except ValueError:
            continue



# '''
tgn = time.perf_counter()   
print(f'Done with GOODS-N sources ({tgn - ts:0.4f} second)')

###############################################################################
###############################################################################
############################### Run GOODS-S SEDs ###############################

# fill_nan = np.zeros(len(GOODSN_filters)-len(GOODSS_filters))
# fill_nan[fill_nan == 0] = np.nan

goodsS_id_candels = []
goodsS_id_auge = []

fill_nan = np.zeros(len(GOODSS_auge_filters)-len(GOODSS_filters))
fill_nan[fill_nan == 0] = np.nan

# '''
# for i in range(len(goodsS_phot_id_match)):
#     # if goodsS_flux_array[i][GOODSS_filters == 'FLUX_24'] <= 0:
#     #     continue
#     # elif goodsS_flux_err_array[i][COSMOS_filters == 'FLUX_24']/goodsS_flux_array[i][COSMOS_filters == 'FLUX_24'] > 0.5:
#     #     continue
#     # else:
#         source = AGN(goodsS_phot_id_match[i],luo_z_match[i],GOODSS_filters,goodsS_flux_array[i],goodsS_flux_err_array[i])
#         source.MakeSED()
#         check_sed.append(source.CheckSED(10, check_span=2.75))

#         F1.append(source.Find_nuFnu(1.0))
#         F025.append(source.Find_nuFnu(0.25)/source.Find_nuFnu(1.0))
#         F6.append(source.Find_nuFnu(6.0)/source.Find_nuFnu(1.0))
#         F10.append(source.Find_nuFnu(10.0)/source.Find_nuFnu(1.0))
#         # F100_2.append(source.Find_nuFnu(100)/source.Find_nuFnu(1.0))

#         ffir, wfir, f100 = source.median_FIR_filter(['FLUX_24','FLUX_100', 'FLUX_160','FLUX_250','FLUX_350','FLUX_500'],Find_value=100.0)
#         FFIR.append(ffir)
#         WFIR.append(wfir)
#         F100.append(f100/source.Find_nuFnu(1.0))

#         uvs = source.Find_slope(0.15, 1.0)
#         mirs1 = source.Find_slope(1.0, 6.5)
#         mirs2 = source.Find_slope(6.5, 10)

#         UVslope.append(uvs)
#         MIRslope1.append(mirs1)
#         MIRslope2.append(mirs2)

#         uv_lum.append(source.find_Lum_range(0.1,0.35))
#         opt_lum.append(source.find_Lum_range(0.35,3))
#         mir_lum.append(source.find_Lum_range(3,30))
#         fir_lum.append(source.find_Lum_range(30,500/(1+luo_z_match[i])))

#         Id, redshift, w, f, frac_err, up_check = source.pull_plot_info()
#         w = np.append(w, fill_nan)
#         f = np.append(f, fill_nan)
#         all_id.append(Id)
#         all_z.append(redshift)
#         all_x.append(w)
#         all_y.append(f)
#         all_frac_err.append(frac_err)
#         upper_check.append(up_check)

#         med_x, med_y = source.median_SED(['U'], ['MIPS2'])
#         median_x.append(med_x)
#         median_y.append(med_y)
        
#         med_fir_x, med_fir_y = source.median_SED(['FLUX_24'], ['FLUX_500'])
#         median_fir_x.append(med_fir_x)
#         median_fir_y.append(med_fir_y)

#         Lx.append(luo_Lx_match[i])
#         Lx_hard.append(luo_Lx_match[i])
#         # Lx.append(luo_Lx_hard_match[i])
#         Nh.append(goodsS_NH_match[i])
#         Lbol.append(source.Find_Lbol())
#         Lbol_sub.append(source.Find_Lbol_temp_sub(scale_array,temp_wave,temp_lum))
#         fir_frac.append(source.FIR_frac())
#         shape = source.SED_shape(uvs, mirs1, mirs2)
#         sed_shape.append(shape)

#         # irac_ch1.append(goodsS_flux_array[i][GOODSS_filters == 'SPLASH_1_FLUX'][0])
#         # irac_ch2.append(goodsS_flux_array[i][GOODSS_filters == 'SPLASH_2_FLUX'][0])
#         # irac_ch3.append(goodsS_flux_array[i][GOODSS_filters == 'SPLASH_3_FLUX'][0])
#         # irac_ch4.append(goodsS_flux_array[i][GOODSS_filters == 'SPLASH_4_FLUX'][0]) 

#         if source.CheckSED(10, check_span=2.5) == 'GOOD':
#             goodsS_id_candels.append(goodsS_phot_id_match[i])

#         field.append(3)


for i in range(len(goodsS_auge_ID_match)):
    # if goodsS_auge_ID_match[i] in luo_id_match:
    #     # continue
        # print('repeat')
    # else:
        try:
            source = AGN(goodsS_auge_ID_match[i],goodsS_auge_z_match[i],GOODSS_auge_filters,goodsS_flux_array_auge[i],goodsS_flux_err_array_auge[i])
            source.MakeSED()
            check_sed.append(source.CheckSED(10, check_span=2.75))


            F1.append(source.Find_nuFnu(1.0))
            F025.append(source.Find_nuFnu(0.25)/source.Find_nuFnu(1.0))
            F6.append(source.Find_nuFnu(6.0)/source.Find_nuFnu(1.0))
            F10.append(source.Find_nuFnu(10.0)/source.Find_nuFnu(1.0))
            # F100_2.append(source.Find_nuFnu(100)/source.Find_nuFnu(1.0))

            ffir, wfir, f100 = source.median_FIR_filter(['FLUX_24','FLUX_100', 'FLUX_160','FLUX_250','FLUX_350','FLUX_500'],Find_value=100.0)
            FFIR.append(ffir)
            WFIR.append(wfir)
            F100.append(f100/source.Find_nuFnu(1.0))

            uvs = source.Find_slope(0.15, 1.0)
            mirs1 = source.Find_slope(1.0, 6.5)
            mirs2 = source.Find_slope(6.5, 10)

            UVslope.append(uvs)
            MIRslope1.append(mirs1)
            MIRslope2.append(mirs2)

            uv_lum.append(source.find_Lum_range(0.1,0.35))
            opt_lum.append(source.find_Lum_range(0.35,3))
            mir_lum.append(source.find_Lum_range(3,30))
            fir_lum.append(source.find_Lum_range(30,500/(1+goodsS_auge_z_match[i])))

            Id, redshift, w, f, frac_err, up_check = source.pull_plot_info()
            # w = np.append(w,fill_nan)
            # f = np.append(f,fill_nan)
            # frac_err = np.append(frac_err,fill_nan)
            all_id.append(Id)
            all_z.append(redshift)
            all_x.append(w)
            all_y.append(f)
            all_frac_err.append(frac_err)
            upper_check.append(up_check)
            shape = source.SED_shape(uvs, mirs1, mirs2)
            sed_shape.append(shape)

            # if source.CheckSED(10, check_span=2.5) == 'GOOD': 
            plot = Plotter(Id,redshift,w,f,frac_err,np.log10(goodsS_auge_Lx_match[i]))
            # if Id == 911:
                # print(uvs)
                # plot.PlotSingleSED(flux_point=f100/source.Find_nuFnu(1.0),wfir=wfir,ffir=ffir/source.Find_nuFnu(1.0))


            med_x, med_y = source.median_SED(['U'], ['MIPS2'])
            median_x.append(med_x)
            median_y.append(med_y)
            
            med_fir_x, med_fir_y = source.median_SED(['FLUX_24'], ['FLUX_500'])
            median_fir_x.append(med_fir_x)
            median_fir_y.append(med_fir_y)

            Lx.append(goodsS_auge_Lx_match[i])
            Lx_hard.append(goodsS_auge_Lx_hard_match[i])
            # Lx.append(goodsS_auge_Lx_hard_match[i])
            Nh.append(0.0)
            Lbol.append(source.Find_Lbol())
            Lbol_sub.append(source.Find_Lbol_temp_sub(scale_array,temp_wave,temp_lum))
            fir_frac.append(source.FIR_frac())

            # irac_ch1.append(goodsS_flux_array_auge[i][GOODSS_auge_filters == 'SPLASH_1_FLUX'][0])
            # irac_ch2.append(goodsS_flux_array_auge[i][GOODSS_auge_filters == 'SPLASH_2_FLUX'][0])
            # irac_ch3.append(goodsS_flux_array_auge[i][GOODSS_auge_filters == 'SPLASH_3_FLUX'][0])
            # irac_ch4.append(goodsS_flux_array_auge[i][GOODSS_auge_filters == 'SPLASH_4_FLUX'][0])

            if source.CheckSED(10, check_span=2.5) == 'GOOD':
                goodsS_id_auge.append(goodsS_auge_ID_match[i])

            field.append(3)
        except ValueError:
            continue
# '''

tgs = time.perf_counter()   
print(f'Done with GOODS-S sources ({tgs - tgn:0.4f} second)')


###############################################################################


'''CGOALS ULIRGS'''
F1_ulirg, F025_ulirg, F6_ulirg, F10_ulirg, F100_ulirg = [], [], [], [], []
FFIR_ulirg, WFIR_ulirg = [], []
UVslope_ulirg, MIRslope1_ulirg, MIRslope2_ulirg = [], [], []
uv_lum_ulirg, opt_lum_ulirg, mir_lum_ulirg, fir_lum_ulirg = [], [], [], []
ulirg_id, ulirg_z, ulirg_x, ulirg_y, ulirg_frac_err = [], [], [], [], []
median_x_ulirg, median_y_ulirg = [], []
median_fir_x_ulirg, median_fir_y_ulirg = [], []
ulirg_Lx_out, ulirg_Lx_corr_out = [], []
Lbol_ulirg, Lbol_ulirg_sub = [], []
ulirg_field = []
ulirg_up_check = []
ulirg_Nh_out, ulirg_LIR_out = [], []
goals_irac_ch1, goals_irac_ch2, goals_irac_ch3, goals_irac_ch4 = [], [], [], []

'''
# fill_nan = np.zeros(len(ned_goals_filter_name)-len(cgoals_filter_name))
# fill_nan[fill_nan == 0] = np.nan
for i in range(len(ulirg_cgoals_ID_match)):
    if ulirg_cgoals_ID_match[i] == 'UGC 08058':
        continue
    elif len(ulirg_cgoals_flux[i][ulirg_cgoals_flux[i] > 0]) > 1:
        source = AGN(ulirg_cgoals_ID_match[i],ulirg_cgoals_z[i],cgoals_filter_name,ulirg_cgoals_flux[i],ulirg_cgoals_flux_err[i])
        source.MakeSED()

        F1_ulirg.append(source.Find_nuFnu(1.0))
        F025_ulirg.append(source.Find_nuFnu(0.25)/source.Find_nuFnu(1.0))
        F6_ulirg.append(source.Find_nuFnu(6.0)/source.Find_nuFnu(1.0))
        F10_ulirg.append(source.Find_nuFnu(10.0)/source.Find_nuFnu(1.0))

        ffir, wfir, f100 = source.median_FIR_filter(['FLUX_24','IRAS2','MIPS2','IRAS4'], Find_value=100.0)

        FFIR_ulirg.append(ffir)
        WFIR_ulirg.append(wfir)
        F100_ulirg.append(f100/source.Find_nuFnu(1.0))

        UVslope_ulirg.append(source.Find_slope(0.15, 1.0))
        MIRslope1_ulirg.append(source.Find_slope(1.0, 6.5))
        MIRslope2_ulirg.append(source.Find_slope(6.5, 10))

        uv_lum_ulirg.append(source.find_Lum_range(0.1,0.35))
        opt_lum_ulirg.append(source.find_Lum_range(0.35,3))
        mir_lum_ulirg.append(source.find_Lum_range(3,30))
        fir_lum_ulirg.append(source.find_Lum_range(30,500/(1+ulirg_cgoals_z[i])))

        Id, redshift, w, f, frac_err, up_check = source.pull_plot_info()
        # w = np.append(w,fill_nan)
        # f = np.append(f,fill_nan)
        # frac_err = np.append(frac_err,fill_nan)
        ulirg_id.append(Id)
        ulirg_z.append(redshift)
        ulirg_x.append(w)
        ulirg_y.append(f)
        ulirg_frac_err.append(frac_err)
        ulirg_up_check.append(up_check)

        med_x, med_y = source.median_SED(['U'], ['VLA2'])
        median_x_ulirg.append(med_x)
        median_y_ulirg.append(med_y)

        med_fir_x, med_fir_y = source.median_SED(['FLUX_24'], ['VLA2'])
        median_fir_x_ulirg.append(med_fir_x)
        median_fir_y_ulirg.append(med_fir_y) 

        ulirg_Lx_out.append(np.log10(ulirg_cgoals_Lx_match[i]))
        ulirg_Lx_corr_out.append(np.log10(ulirg_cgoals_Lx_match[i]))
        ulirg_Nh_out.append(ulirg_cgoals_Nh_match[i])
        ulirg_LIR_out.append(ulirg_cgoals_Lir_match[i])

        Lbol_ulirg.append(source.Find_Lbol())
        Lbol_ulirg_sub.append(source.Find_Lbol_temp_sub(scale_array,temp_wave,temp_lum))
        goals_irac_ch1.append(ulirg_cgoals_flux[i][cgoals_filter_name == 'SPLASH_1_FLUX'][0])
        goals_irac_ch2.append(ulirg_cgoals_flux[i][cgoals_filter_name == 'SPLASH_2_FLUX'][0])
        goals_irac_ch3.append(ulirg_cgoals_flux[i][cgoals_filter_name == 'SPLASH_3_FLUX'][0])
        goals_irac_ch4.append(ulirg_cgoals_flux[i][cgoals_filter_name == 'SPLASH_4_FLUX'][0])

        ulirg_field.append(5)



fill_nan = np.zeros(len(cgoals_filter_name) - len(ned_goals_filter_name))
fill_nan[fill_nan == 0] = np.nan
for i in range(len(ned_goals_ID_match)):
    if ned_goals_ID_match[i] == 'UGC 08058':
        continue
    elif len(ned_goals_flux[i][ned_goals_flux[i] > 0]) > 1:
        source = AGN(ned_goals_ID_match[i],ned_goals_z[i],ned_goals_filter_name,ned_goals_flux[i],ned_goals_flux_err[i])
        source.MakeSED()

        F1_ulirg.append(source.Find_nuFnu(1.0))
        F025_ulirg.append(source.Find_nuFnu(0.25)/source.Find_nuFnu(1.0))
        F6_ulirg.append(source.Find_nuFnu(6.0)/source.Find_nuFnu(1.0))
        F10_ulirg.append(source.Find_nuFnu(10.0)/source.Find_nuFnu(1.0))

        try:
            ffir, wfir, f100 = source.median_FIR_filter(['IRAS1', 'IRAS3', 'MIPS2', 'IRAS4'], Find_value=100.0)
        except ValueError:
            f100 = source.Find_nuFnu(100.0)


        FFIR_ulirg.append(ffir)
        WFIR_ulirg.append(wfir)
        F100_ulirg.append(f100/source.Find_nuFnu(1.0))

        UVslope_ulirg.append(source.Find_slope(0.15, 1.0))
        MIRslope1_ulirg.append(source.Find_slope(1.0, 6.5))
        MIRslope2_ulirg.append(source.Find_slope(6.5, 10))

        uv_lum_ulirg.append(source.find_Lum_range(0.1,0.35))
        opt_lum_ulirg.append(source.find_Lum_range(0.35,3))
        mir_lum_ulirg.append(source.find_Lum_range(3,30))
        fir_lum_ulirg.append(source.find_Lum_range(30, 500/(1+ned_goals_z[i])))

        Id, redshift, w, f, frac_err, up_check = source.pull_plot_info()
        w = np.append(w,fill_nan)
        f = np.append(f,fill_nan)
        frac_err = np.append(frac_err,fill_nan)
        ulirg_id.append(Id)
        ulirg_z.append(redshift)
        ulirg_x.append(w)
        ulirg_y.append(f)
        ulirg_frac_err.append(frac_err)
        ulirg_up_check.append(up_check)

        med_x, med_y = source.median_SED(['U'], ['FLUX_500'])
        median_x_ulirg.append(med_x)
        median_y_ulirg.append(med_y)

        med_fir_x, med_fir_y = source.median_SED(['FLUX_24'], ['FLUX_500'])
        median_fir_x_ulirg.append(med_fir_x)
        median_fir_y_ulirg.append(med_fir_y) 

        ulirg_Lx_out.append(np.log10(ned_goals_Lx[i]))
        # ulirg_Lx_corr.append(np.log10(ned_goals_Lx[i]))
        ulirg_Nh_out.append(ned_goals_Nh[i])

        Lbol_ulirg.append(source.Find_Lbol())
        Lbol_ulirg_sub.append(source.Find_Lbol_temp_sub(scale_array,temp_wave,temp_lum))
        ulirg_LIR_out.append(ned_goals_LIR_match[i])
        goals_irac_ch1.append(ned_goals_flux[i][ned_goals_filter_name == 'SPLASH_1_FLUX'][0])
        goals_irac_ch2.append(ned_goals_flux[i][ned_goals_filter_name == 'SPLASH_2_FLUX'][0])
        goals_irac_ch3.append(ned_goals_flux[i][ned_goals_filter_name == 'SPLASH_3_FLUX'][0])
        goals_irac_ch4.append(ned_goals_flux[i][ned_goals_filter_name == 'SPLASH_4_FLUX'][0])

        ulirg_field.append(6)


'''
print('Done with ULIRGS')




###############################################################################
###############################################################################
############################### Begin Plotting ################################
# plt.rcParams['font.size'] = 14
# plt.rcParams['axes.linewidth'] = 2
# plt.rcParams['xtick.major.size'] = 2
# plt.rcParams['xtick.major.width'] = 2
# plt.rcParams['ytick.major.size'] = 2
# plt.rcParams['ytick.major.width'] = 2



check_sed = np.asarray(check_sed)
GOOD_SEDs = np.where(check_sed == 'GOOD')
# GOOD_SEDs = np.where(check_sed == 'BAD')
print('ALL: ', len(check_sed))
print('BAD SEDs: ', len(check_sed[check_sed == 'BAD']))
print('GOOD SEDs: ', len(check_sed[check_sed == 'GOOD']))
# print('Except: ',len(bad_id))
# all_id, all_z, all_x, all_y, Lx, median_x, median_y = np.asarray(all_id), np.asarray(all_z), np.asarray(all_x), np.asarray(all_y), np.log10(np.asarray(Lx)), np.asarray(median_x), np.asarray(median_y)
# F1, F025, F6, F10, F100 = np.asarray(F1), np.asarray(F025), np.asarray(F6), np.asarray(F10), np.asarray(F100)
# field = np.asarray(field)
# UVslope, MIRslope1, MIRslope2 = np.asarray(UVslope), np.asarray(MIRslope1), np.asarray(MIRslope2)
# FFIR, WFIR = np.asarray(FFIR), np.asarray(WFIR)
# spec_type = np.asarray(spec_type)
# upper_check, fir_frac = np.asarray(upper_check), np.asarray(fir_frac)
# Lbol = np.asarray(Lbol)
# Nh = np.asarray(Nh)
# F100_2 = np.asarray(F100_2)
# FFIR_2, WFIR_2 = np.asarray(FFIR_2), np.asarray(WFIR_2)

all_id, all_z, all_x, all_y, Lx, median_x, median_y = np.asarray(all_id)[GOOD_SEDs], np.asarray(all_z)[GOOD_SEDs], np.asarray(all_x)[GOOD_SEDs], np.asarray(all_y)[GOOD_SEDs], np.log10(np.asarray(Lx))[GOOD_SEDs], np.asarray(median_x)[GOOD_SEDs], np.asarray(median_y)[GOOD_SEDs]
F1, F025, F6, F10, F100 = np.asarray(F1)[GOOD_SEDs], np.asarray(F025)[GOOD_SEDs], np.asarray(F6)[GOOD_SEDs], np.asarray(F10)[GOOD_SEDs], np.asarray(F100)[GOOD_SEDs]
field = np.asarray(field)[GOOD_SEDs]
UVslope, MIRslope1, MIRslope2 = np.asarray(UVslope)[GOOD_SEDs], np.asarray(MIRslope1)[GOOD_SEDs], np.asarray(MIRslope2)[GOOD_SEDs]
FFIR, WFIR = np.asarray(FFIR)[GOOD_SEDs], np.asarray(WFIR)[GOOD_SEDs]
# spec_type = np.asarray(spec_type)[GOOD_SEDs]
upper_check, fir_frac = np.asarray(upper_check)[GOOD_SEDs], np.asarray(fir_frac)[GOOD_SEDs]
Lbol, Lbol_sub = np.asarray(Lbol)[GOOD_SEDs], np.asarray(Lbol_sub)[GOOD_SEDs]
Nh = np.asarray(Nh)[GOOD_SEDs]
# F100_2 = np.asarray(F100_2)[GOOD_SEDs]
# FFIR_2, WFIR_2 = np.asarray(FFIR_2)[GOOD_SEDs], np.asarray(WFIR_2)[GOOD_SEDs]
median_fir_x, median_fir_y = np.asarray(median_fir_x)[GOOD_SEDs], np.asarray(median_fir_y)[GOOD_SEDs]
uv_lum, opt_lum, mir_lum, fir_lum = np.asarray(uv_lum)[GOOD_SEDs], np.asarray(opt_lum)[GOOD_SEDs], np.asarray(mir_lum)[GOOD_SEDs], np.asarray(fir_lum)[GOOD_SEDs]
sed_shape = np.asarray(sed_shape)[GOOD_SEDs]
# irac_ch1, irac_ch2, irac_ch3, irac_ch4 = np.asarray(irac_ch1)[GOOD_SEDs], np.asarray(irac_ch2)[GOOD_SEDs], np.asarray(irac_ch3)[GOOD_SEDs], np.asarray(irac_ch4)[GOOD_SEDs]
Lx_hard = np.log10(np.asarray(Lx_hard)[GOOD_SEDs])

ulirg_id, ulirg_z, ulirg_x, ulirg_y, ulirg_Lx_out, ulirg_Lx_corr_out, median_x_ulirg, median_y_ulirg = np.asarray(ulirg_id), np.asarray(ulirg_z), np.asarray(ulirg_x), np.asarray(ulirg_y), np.asarray(ulirg_Lx_out), np.asarray(ulirg_Lx_corr_out), np.asarray(median_x_ulirg), np.asarray(median_y_ulirg)
F1_ulirg, F025_ulirg, F6_ulirg, F10_ulirg, F100_ulirg = np.asarray(F1_ulirg), np.asarray(F025_ulirg), np.asarray(F6_ulirg), np.asarray(F10_ulirg), np.asarray(F100_ulirg)
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


plt.rcParams['font.size'] = 22
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.size'] = 2
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.size'] = 2
plt.rcParams['ytick.major.width'] = 2

plt.rcParams['font.size'] = 25
plt.rcParams['axes.linewidth'] = 3.5
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 4
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 4
plt.rcParams['xtick.minor.size'] = 3.
plt.rcParams['xtick.minor.width'] = 2.
plt.rcParams['ytick.minor.size'] = 3.
plt.rcParams['ytick.minor.width'] = 2.

# plt.figure(figsize=(8,8))
# plt.hist(np.log10(F1),bins=np.arange(39,46,0.25))
# plt.axvline(np.nanmedian(np.log10(F1)),c='k')
# plt.xlabel(r'L (1$\mu$m)')
# plt.ylim(0,300)
# plt.show()

# plt.figure(figsize=(8,8))
# plt.hist(np.log10(Lbol_sub),bins=np.arange(42,49,0.25))
# plt.axvline(np.nanmedian(np.log10(Lbol_sub)),c='k')
# plt.xlabel(r'L$_{\mathrm{bol}}$')
# plt.ylim(0,300)
# plt.show()

plt.figure(figsize=(10,9))
plt.hist(sed_shape,bins=np.arange(0,7,1))
plt.xlabel('SED Shape')
plt.ylim(0,400)
plt.show()

# x1 = np.linspace(0.08,1.5)
# x2 = np.linspace(0.35,1.5)
# # print(np.log10(ulirg_LIR_out))
# # print(ulirg_Lx_out)
# plt.plot(np.log10(irac_ch3/irac_ch1), np.log10(irac_ch4/irac_ch2), 'x',label='COSMOS X-ray',zorder=0)
# # # plt.plot(np.log10(goals_irac_ch3/goals_irac_ch1), np.log10(goals_irac_ch4/goals_irac_ch2), '.',label='GOALS',zorder=0)
# # plt.scatter(np.log10(goals_irac_ch3/goals_irac_ch1), np.log10(goals_irac_ch4/goals_irac_ch2), c=ulirg_Lx_out, s=50, label='GOALS', zorder=1)
# plt.xlim(-0.7,1.2)
# plt.ylim(-0.7,1.3)
# plt.vlines(0.08,ymin=0.15,ymax=0.38,color='k')
# plt.hlines(0.15,xmin=0.08,xmax=0.35,color='k')
# plt.plot(x1, 1.21*x1 + 0.27,color='k')
# plt.plot(x2, 1.21*x2 - 0.27,color='k')
# plt.legend(fontsize=12)
# # plt.colorbar(label=r'L$_{\mathrm{IR}}$')
# plt.show()



# print('HERE: ', len(ulirg_LIR_out), len(ulirg_Nh_out))
# print(np.log10(ulirg_LIR_out))
# print(np.log10(Lbol_ulirg/3.8E33))
# print(ulirg_id)

# plt.plot(np.log10(ulirg_LIR_out),np.log10(Lbol_ulirg/3.8E33),'.')
# plt.plot(np.arange(0,15),np.arange(0,15))
# plt.xlim(10,13)
# plt.ylim(10,13)
# plt.show()



# plt.figure(figsize=(10,10))
# # plt.plot(all_z[field == 2],Lx[field == 2],'.',color='gray',rasterized=True)
# # plt.plot(all_z[field == 3],Lx[field == 3],'.',color='gray',rasterized=True,label='GOODS-N/S')
# # plt.plot(all_z[field == 0],Lx[field == 0],'+',ms=10,color='b',rasterized=True,alpha=0.8,label='COSMOS')
# plt.plot(all_z[field == 1],Lx[field == 1],'x',ms=10,color='r',rasterized=True,alpha=0.8,label='Stripe82X')
# plt.xlabel('Spectroscopic Redshift')
# plt.ylabel(r'log$_{10}$ L$_{0.5 - 10\mathrm{keV}}$ [erg s$^{-1}$]')
# plt.text(0.025,45.5,f'n = {len(all_z)}')
# plt.legend()
# # plt.xlim(-0.075,1.4)
# # plt.ylim(42.9,46.5)
# plt.grid()
# plt.tight_layout()
# # plt.savefig('/Users/connor_auge/Desktop/New_runSED/Lx_z_COSMOS_S82X_good.pdf')
# plt.show()

# fig = plt.figure(figsize=(20,10))
# ax1 = plt.subplot(121)
# ax1.hist(all_z[np.logical_or(field == 2, field == 3)],bins=np.arange(0,1.5,0.25),histtype='step',color='gray',lw=3,label='GOODS-N/S')
# ax1.hist(all_z[field == 0],bins=np.arange(0,1.5,0.25),histtype='step',color='b',lw=3,label='COSMOS')
# ax1.hist(all_z[field == 1],bins=np.arange(0,1.5,0.25),histtype='step',color='r',lw=3,label='Stripe82X')
# ax1.set_xlabel('Spectroscopic Redshift')
# ax1.set_xlim(-0.075, 1.5)
# ax1.set_ylim(0,350)
# ax1.set_ylabel('N')
# ax1.grid()

# ax2 = plt.subplot(122)
# ax2.hist(Lx[np.logical_or(field == 2, field == 3)],bins=np.arange(39,46.5,0.25),histtype='step',color='gray',lw=3)
# ax2.hist(Lx[field == 0],bins=np.arange(39,47.5,0.25),histtype='step',color='b',lw=3,label='COSMOS')
# ax2.hist(Lx[field == 1],bins=np.arange(39,47.5,0.25),histtype='step',color='r',lw=3,label='Stripe82X')
# ax2.set_xlabel(r'log$_{10}$ L$_{0.5 - 10\mathrm{keV}}$ [erg s$^{-1}$]')
# ax2.set_ylabel('N')
# ax2.set_ylim(0,150)
# ax2.set_xlim(42.8,46)
# ax2.grid()
# ax2.legend()
# plt.tight_layout()
# plt.savefig('/Users/connor_auge/Desktop/New_runSED/Lx_z_hist_COSMOS_S82X_good.pdf')
# plt.show()


# plt.plot(np.log10(Lbol),np.log10(Lbol_sub),'.')
# plt.plot(np.arange(40,48),np.arange(40,48),color='k')
# plt.xlabel('Lbol')
# plt.ylabel('Lbol sub')
# plt.show()

for i in range(len(all_id)):
    print(all_id[i],Lx[i],Lbol_sub[i])


print('COSMOS Sample: ', len(Lx[field == 0]))
print('S82X Sample: ', len(Lx[field == 1]))
print('GOODS-N Sample: ', len(Lx[field == 2]))
print('GOODS-S Sample: ', len(Lx[field == 3]))
print('ULIRGS: ', len(ulirg_id))

plot = Plotter_Letter(all_id,all_z,all_x,all_y,all_frac_err)
plot2 = Plotter_Letter2(all_id,all_z,all_x,all_y,all_frac_err)

ulirg_plot = Plotter_Letter(ulirg_id,ulirg_z,ulirg_x,ulirg_y,ulirg_frac_err)
sort = Lx.argsort()

plot_v2 = Plotter_v2(all_id, all_z, all_x, all_y, Lx, F1, upper_check)
plot_shape_v2 = SED_shape_Plotter(all_id, all_z, all_x, all_y, Lx, F1, upper_check, sed_shape)


# plt.plot(np.log10(F100[field == 0]*F1[field == 0]), np.log10(F100_2[field == 0]*F1[field == 0]), '.',color='b',label='COSMOS')
# plt.plot(np.log10(F100[field == 1]*F1[field == 1]), np.log10(F100_2[field == 1]*F1[field == 1]), '.',color='orange',label='Stripe82X')
# plt.plot(np.log10(F100[field == 2]*F1[field == 2]), np.log10(F100_2[field == 2]*F1[field == 2]), '.',color='gray',label='GOODS-N/S')
# plt.plot(np.log10(F100[field == 3]*F1[field == 3]), np.log10(F100_2[field == 3]*F1[field == 3]), '.',color='gray')
# plt.plot(np.arange(40,47,1),np.arange(40,47,1),color='k')
# plt.xlabel('New')
# plt.ylabel('Old')
# plt.legend()
# plt.show()

zlim_1 = 0.0
zlim_2 = 0.6
zlim_3 = 0.9
zlim_4 = 1.2

B1 = np.logical_and(all_z >= zlim_1,all_z <= zlim_2)
B2 = np.logical_and(all_z > zlim_2,all_z <= zlim_3)
B3 = np.logical_and(all_z > zlim_3,all_z <= zlim_4)

# plt.figure(figsize=(8,6))
# plt.hist(np.log10(F1[field==1]),bins=np.arange(40,50,0.25),histtype='step',color='r',label='Stripe82X',lw=4.0)
# plt.axvline(np.log10(np.nanmedian(F1[field==1])),ls='--',color='r',lw=3.5)
# plt.hist(np.log10(F1[field==0]),bins=np.arange(40,50,0.25),histtype='step',color='b',label='COSMOS',lw=3.5)
# plt.axvline(np.log10(np.nanmedian(F1[field==0])),ls='--',color='b',lw=3.5)
# plt.hist(np.log10(F1[np.logical_or(field==2,field==3)]),bins=np.arange(40,50,0.25),histtype='step',color='gray',label='GOODS-N/S',lw=3.0)
# plt.axvline(np.log10(np.nanmedian(F1[np.logical_or(field==2,field==3)])),ls='--',color='gray',lw=3.5)
# plt.xlabel(r'Log L$_{1 \mu \mathrm{m}}$ [erg/s]')
# plt.xlim(42,47)
# plt.legend()
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.hist(Lx[field==1],bins=np.arange(40,50,0.25),histtype='step',color='r',label='Stripe82X',lw=4.0)
# plt.axvline(np.nanmedian(Lx[field==1]),ls='--',color='r',lw=3.5)
# plt.hist(Lx[field==0],bins=np.arange(40,50,0.25),histtype='step',color='b',label='COSMOS',lw=3.5)
# plt.axvline(np.nanmedian(Lx[field==0]),ls='--',color='b',lw=3.5)
# plt.hist(Lx[np.logical_or(field==2,field==3)],bins=np.arange(40,50,0.25),histtype='step',color='gray',label='GOODS-N/S',lw=3.0)
# plt.axvline(np.nanmedian(Lx[np.logical_or(field==2,field==3)]),ls='--',color='gray',lw=3.5)
# plt.xlabel(r'Log L$_{\mathrm{X}}$ [erg/s]')
# plt.xlim(42, 47)
# plt.legend()
# plt.show()

# plt.figure(figsize=(8,6))
# plt.hist(np.log10(F1[B3]),bins=np.arange(40,50,0.25),histtype='step',color='r',label='0.9 < z < 1.2',lw=4.0)
# plt.axvline(np.log10(np.nanmedian(F1[B3])),ls='--',color='r',lw=3.5)
# plt.hist(np.log10(F1[B2]),bins=np.arange(40,50,0.25),histtype='step',color='b',label='0.6 < z < 0.9',lw=3.5)
# plt.axvline(np.log10(np.nanmedian(F1[B2])),ls='--',color='b',lw=3.5)
# plt.hist(np.log10(F1[B1]),bins=np.arange(40,50,0.25),histtype='step',color='gray',label='0.0 < z < 0.6',lw=3.0)
# plt.axvline(np.log10(np.nanmedian(F1[B1])),ls='--',color='gray',lw=3.5)
# plt.xlabel(r'Log L$_{1 \mu \mathrm{m}}$ [erg/s]')
# plt.xlim(42,47)
# plt.legend()
# plt.show()

# print(uv_lum)
# print(Lbol)
# print(np.log10((uv_lum+mir_lum+fir_lum)/Lbol))

# plt.figure(figsize=(10,9))
# plt.hist(np.log10((uv_lum+mir_lum+fir_lum+opt_lum)/Lbol),bins=np.arange(-1,1,0.05))
# plt.xlim(-0.7,0.25)
# plt.xlabel('L(UV+MIR+FIR)/Lbol')
# plt.grid()
# plt.show()

# plot_v2.L_hist('goodss_Lone_hist_old2',np.log10(F1),r'log L (1 $\mu$m) [erg/s]',[41.5,46],[41.5,46,0.25])
# plot_v2.L_hist('all_Lx_hist_old2',Lx,r'log L$_{\mathrm{X}}$ [erg/s]',[42.5,47],[42.5,47,0.25])
# plot_v2.L_hist('all_Lbol_old',np.log10(Lbol_sub),r'log L$_{\mathrm{bol}}$',[43,48],[43,48,0.25])
# plot_v2.L_hist('goodss_Nh_hist_old2',np.log10(Nh),r'log N$_{\mathrm{H}}$ [cm$^{-2}$]',[19.5,25],[19.5,24.5,0.25])

# plot.plot_1panel('43',all_w,all_f,Lx,spec_type,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),median_w,median_f,F1=norm,F2=marker,suptitle=str(z_min)+' < z < '+str(z_max),spec_z=all_z,uv_slope=UVslope,mir_slope1=MIRslope1,mir_slope2=MIRslope2,wfir=np.asarray(WFIR),ffir=np.asarray(FFIR))
# plot.NSF_seds_3panel(all_x[sort],all_y[sort],Lx[sort],UVslope[sort],MIRslope1[sort],MIRslope2[sort],median_x[sort],median_y[sort],F1[sort])
# plot_shape_v2.shape_1bin_h('horizantal_5_panel2',median_x=10**median_x,median_y=median_y,wfir=WFIR,ffir=FFIR,uv_slope=UVslope,mir_slope1=MIRslope1,mir_slope2=MIRslope2,Median_line=True,FIR_upper='upper lims')
# plot_shape_v2.shape_1bin_v('goodss_vertical_5_panel_old2',median_x=10**median_x,median_y=median_y,wfir=WFIR,ffir=FFIR,uv_slope=UVslope,mir_slope1=MIRslope1,mir_slope2=MIRslope2,Median_line=True,FIR_upper='upper lims')

################ Paper Plots ###################
# fig 3

# plt.hist(np.log10(F1),bins=np.arange(35,50,0.5))
# plt.ylim(0,300)
# plt.show()

# plot.multi_SED('all_check',all_x[sort],all_y[sort],Lx[sort],median_x[sort],median_y[sort],suptitle='SEDs of X-ray',norm=F1[sort],mark=field[sort],spec_z=all_z[sort],wfir=WFIR[sort],ffir=FFIR[sort],up_check=upper_check[sort],med_x_fir=median_fir_x[sort],med_y_fir=median_fir_y[sort])

# fig 4 & 5
# plot.multi_SED_zbins('S82X',all_x[sort], all_y[sort], Lx[sort], all_z[sort], median_x[sort], median_y[sort], F1[sort], field[sort], spec_z=all_z[sort],wfir=WFIR[sort],ffir=FFIR[sort],up_check=upper_check[sort],med_x_fir=median_fir_x[sort],med_y_fir=median_fir_y[sort])
# plot.multi_SED_field('AAS', all_x[sort], all_y[sort], Lx[sort], all_z[sort], median_x[sort], median_y[sort], F1[sort], field[sort], spec_z=all_z[sort], wfir=WFIR[sort], ffir=FFIR[sort])

# fig 6
# plot.plot_5panel_zbins('AAS_check',all_x,all_y,Lx,spec_type,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),median_x,median_y,F1=F1,F2=field,suptitle=str(z_min)+' < z < '+str(z_max),spec_z=all_z,uv_slope=UVslope,mir_slope1=MIRslope1,mir_slope2=MIRslope2,wfir=np.asarray(WFIR),ffir=np.asarray(FFIR))

# fig 7 
# plot.plot_median_zbins(np.asarray(FFIR),all_x,all_y,Lx,spec_type,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),median_x,median_y,F1=F1,F2=field,median_FIR_w=np.asarray(WFIR),spec_z=all_z,uv_slope=UVslope,mir_slope1=MIRslope1,mir_slope2=MIRslope2)

# fig 8 & 16
# plot.L_box_3zbins('Lx_box', Lx, F1, all_z, UVslope, MIRslope1, MIRslope2,label='X') # Need to change ylim between these plots
# plot.L_box_3zbins('Lbol_box3_43', np.log10(Lbol), F1, all_z, UVslope, MIRslope1, MIRslope2,label='bol')
# plot.L_box_fields('Lone_box_field_43', np.log10(F1), F1, all_z, UVslope, MIRslope1, MIRslope2,field,label='one')

# fig 9
# plot.Nh_box_3zbins('Nh_box', np.log10(Nh[Nh > 0]), F1[Nh > 0], all_z[Nh > 0], UVslope[Nh > 0], MIRslope1[Nh > 0], MIRslope2[Nh > 0])


# fig 10, 11 & 12
# plot.Emission_Scatter_Comp('UV_MIR_ALL_NEW',np.asarray([np.nan]),np.asarray([np.nan]),np.asarray([np.nan]),np.asarray([np.nan]),np.asarray([np.nan]),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),Nh,F1=F1,F2=field,spec_z=all_z,uv_slope=UVslope,mir_slope1=MIRslope1,mir_slope2=MIRslope2,up_check=upper_check)

# fig 13 & 14
# plot.Lx_Scatter_Comp('Lx_MIR_ALL_NEW',Lx,np.log10(Lbol),np.asarray([np.nan]),np.asarray([np.nan]),np.asarray([np.nan]),np.asarray([np.nan]),np.asarray([np.nan]),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),Nh,F1=F1,F2=field,spec_z=all_z,uv_slope=UVslope,mir_slope1=MIRslope1,mir_slope2=MIRslope2,up_check=upper_check)

# fig 15
# plot.L_Lx_scatter_3zbins('Lx_Lbol_photz', Lx, np.log10(Lbol), np.asarray([np.nan]), np.asarray([np.nan]), np.asarray([np.nan]), np.asarray([np.nan]), np.asarray([np.nan]), np.log10(F025), np.log10(F6), np.log10(F100), np.log10(F10), F1=F1, F2=field, spec_z=all_z, uv_slope=UVslope, mir_slope1=MIRslope1, mir_slope2=MIRslope2, up_check=upper_check, fir_frac=fir_frac)


### New Plots ### 

# fig 13 & 14
# plot2.Lx_Scatter_Comp('Lx_MIR10_test','Lx','MIR10','None','Lx' ,Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Lx_Scatter_Comp('Lx_MIR6','Lx','MIR6','None','X-axis' ,Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)

# plot2.Lx_Scatter_Comp('Lx_UV','Lx','UV','None','X-axis',Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Lx_Scatter_Comp('Lx_FIR_norm','Lx','FIR','Y','Bins',Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Lx_Scatter_Comp('Lx_FIR','Lx','FIR','None','Bins',Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Lx_Scatter_Comp('Lx_FIR_both','Lx','FIR','Both','Bins',Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)

# plot2.Lx_Scatter_Comp('Lx_Lbol','Lx','Lbol','None','X-axis' ,Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Lx_Scatter_Comp('Lx_Lbol_sub','Lx','Lbol','None','X-axis' ,Lx,np.log10(Lbol_sub),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)


# plot2.Box_3panel('Lx_box_3panel_AAS', 'Lx', Lx, all_z, UVslope, MIRslope1, MIRslope2)
# plot2.Box_3panel('Nh_box_3panel_AAS', 'Nh', np.log10(Nh[Nh > 0]), all_z[Nh > 0], UVslope[Nh > 0], MIRslope1[Nh > 0], MIRslope2[Nh > 0])
# plot2.Box_3panel('Lbol_box_3panel_AAS', 'Lbol', np.log10(Lbol_sub), all_z, UVslope, MIRslope1, MIRslope2)

# # fig 10, 11 & 12
# plot2.Emission_Scatter_Comp('UV_MIR_plot_43_med_bins','MIR6','UV','None','Bins' ,Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Emission_Scatter_Comp('UV_FIR','FIR','UV','Both','Bins',Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Emission_Scatter_Comp('MIR10_FIR_NEWplot_43_normY_med_bins','FIR','MIR6','Both','Bins',Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)

# plot2.Nh_frac_plots('Nh_MIR','MIR','Bins',Nh,Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Nh_frac_plots('Nh_UV','UV','Bins',Nh,Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Nh_frac_plots('Nh_UV_MIR','UV/MIR','Bins',Nh,Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Nh_frac_plots('Nh_UV_FIR','UV/FIR','Bins',Nh,Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Nh_frac_plots('Nh_MIR_FIR','MIR/FIR','Bins',Nh,Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Nh_frac_plots('Nh_Lbol','Lbol','Bins',Nh,Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)

# plot2.ratio_plots('Nh_MIR','Nh','MIR6','Bins',Nh,Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.ratio_plots('Lbol_MIR','Lbol','MIR6/Lbol','Bins',Nh,Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.ratio_plots('Lbol_Lx_ratio_medBins','Lx','Lbol','Bins',Nh,Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.ratio_plots('Lbol_Lx_ratio_medLx','Lx','Lbol','X-axis',Nh,Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)


# plot2.Upanels_ratio_plots('new2/Lum_Lbol','Lbol','UV-MIR-FIR/Lbol','Bins',Nh,Lx,np.log10(Lbol_sub),np.log10(uv_lum),np.log10(mir_lum),np.log10(fir_lum),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Upanels_ratio_plots('new2/Nh_ratios','Nh','UV/MIR-UV/Lx-MIR/Lx','Bins',Nh,Lx,np.log10(Lbol_sub),np.log10(uv_lum),np.log10(mir_lum),np.log10(fir_lum),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.Box_1panel('Lbol_box_1panel', 'Lbol', np.log10(Lbol), UVslope, MIRslope1, MIRslope2)
# plot2.ratios_1panel('Lx_Lbol_1panel','Lx','Lbol/Lx','X-axis',Nh,Lx,np.log10(Lbol),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check)

# plot2.ratios_1panel('MIR_Nh_1panel','Nh','MIR6','Bins',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.ratios_1panel('UV_Nh_1panel','Nh','UV','Bins',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.scatter_1panel('UV_MIR_1panel_norm','MIR6','UV','Both','Bins',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.scatter_1panel('UV_Lx_1panel_norm_x','Lx','UV','None','X-axis',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.scatter_1panel('MIR_Lx_norm_x','Lx','MIR6','None','X-axis',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check)



# plot2.scatter_1panel('MIR6_Lx_1panel_norm','Lx','MIR6','Both','Bins',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check)

# plot2.scatter_1panel('UV_MIR_1panel_med_norm','MIR6','UV','Both','X-axis',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.scatter_1panel('UV_Lx_1panel_med_norm','Lx','UV','Both','X-axis',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.scatter_1panel('MIR6_Lx_1panel_med_norm','Lx','MIR6','Both','X-axis',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check)

# plot2.Box_1panel('new2/Lx_box_1panel3', 'Lx', Lx, UVslope, MIRslope1, MIRslope2, shape=sed_shape)
# plot2.Box_1panel('new2/Nh_box_1panel', 'Nh', np.log10(Nh[Nh > 0]), UVslope[Nh>0], MIRslope1[Nh>0], MIRslope2[Nh>0])
# plot2.Box_1panel('new2/Lbol_box_1panel_sub', 'Lbol', np.log10(Lbol_sub), UVslope, MIRslope1, MIRslope2)
# plot2.Box_1panel('new2/Lbol_Lx_box_1panel_sub', 'Lbol/Lx', np.log10(Lbol_sub)-Lx, UVslope, MIRslope1, MIRslope2)
# plot2.Upanels_ratio_plots('new2/Lum_Lbol_sub','Lbol','UV-MIR-FIR/Lbol','Bins',Nh,Lx,np.log10(Lbol_sub),np.log10(uv_lum),np.log10(mir_lum),np.log10(fir_lum),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check)

# plot2.ratios_1panel('new2/Lx_Lbol_1panel_sub_xmed_lbol','Lbol','Lbol/Lx','X-axis',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check)


# plot2.scatter_1panel('new2/Lx_Lbol2','Lbol','Lx/Lbol','None','Both',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check,durras=False)
plot2.scatter_1panel('new2/Lx_Lbol_1panel_sub_xmed_lbol2_allpts','Lbol','Lbol/Lx','None','Both',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check,shape=sed_shape,durras=True)
# plot2.scatter_1panel('new2/Lx_Lbol_1panel_sub_xmed_lbol_hard_allpts','Lbol','Lbol/Lx','None','Both',Nh,Lx_hard,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check,durras=True)
# plot2.scatter_1panel('new2/UV_Lx_Lx_norm_new3','Lx','UV/Lx','None','Bins',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.scatter_1panel('new2/MIR_Lx_Lx_norm_new3','Lx','MIR6/Lx','None','Bins',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.scatter_1panel('new2/UV_MIR_MIR_norm3','MIR6','UV/MIR6','None','Bins',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.scatter_1panel('new2/UV_FIR_1panel_norm_new3','FIR','UV/FIR','None','Bins',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.scatter_1panel('new2/FIR_Lx_1panel_norm_new3','Lx','FIR/Lx','None','Bins',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.scatter_1panel('new2/UV_FIR_1panel_norm_new3','FIR','UV','Both','Bins',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.scatter_1panel('new2/FIR_Lx_1panel_norm_new3','Lx','FIR','Y-axis','Bins',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check)



# plot2.scatter_1panel('UV_Lx_FIR_Lx_1panel','FIR/Lx','UV/Lx','None','Bins',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check)
# plot2.scatter_1panel('FIR_Lx_Lx_1panel','Lx','FIR/Lx','None','Bins',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check)


# print(ulirg_Lx_out)
### ULIRG Plots ###
# ulirg_plot.multi_SED('ULIRG_n',ulirg_x,ulirg_y,ulirg_Lx_out,median_x_ulirg,median_y_ulirg,suptitle='SEDs of ULIRGs',norm=F1_ulirg,mark=ulirg_field,spec_z=ulirg_z,wfir=None,ffir=None,up_check=ulirg_up_check,med_x_fir=median_fir_x_ulirg,med_y_fir=median_fir_y_ulirg)
# plot2.Lx_Scatter_Comp('ULIRG_Lx_MIR_bins','Lx','MIR6','None','Bins' ,Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check,ulirg_Lx=ulirg_Lx_out,ulirg_Flux = np.log10(F6_ulirg),ulirg_F1 = F1_ulirg)
# plot2.Lx_Scatter_Comp('ULIRG_Lx_UV_bins','Lx','UV','None','Bins',Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check,ulirg_Lx=ulirg_Lx_out,ulirg_Flux = np.log10(F025_ulirg),ulirg_F1 = F1_ulirg)
# plot2.ratio_plots('ULIRG_Nh_MIR','Nh','MIR6','Bins',Nh,Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check,ulirg_Nh=ulirg_Nh_out,ulirg_Lx=ulirg_Lx_out,ulirg_Flux=np.log10(F6_ulirg),ulirg_F1=F1_ulirg)
# plot2.ratio_plots('ULIRG_Nh_UV','Nh','UV','Bins',Nh,Lx,np.log10(Lbol),np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),F1,field,all_z,UVslope,MIRslope1,MIRslope2,upper_check,ulirg_Nh=ulirg_Nh_out,ulirg_Lx=ulirg_Lx_out,ulirg_Flux=np.log10(F025_ulirg),ulirg_F1=F1_ulirg)
# plot2.Box_1panel('ULIRG_Lx_box_1panel_n', 'Lx', Lx, UVslope, MIRslope1, MIRslope2, ulirg_x=ulirg_Lx_out-np.log10(3.8E33))
# plot2.Box_1panel('ULIRG_Nh_box_1panel_n', 'Nh', np.log10(Nh[Nh > 0]), UVslope[Nh > 0], MIRslope1[Nh > 0], MIRslope2[Nh > 0], ulirg_x=ulirg_Nh_out[ulirg_Nh_out > 21])
# plot2.Box_1panel('ULIRG_Lbol_box_1panel_n', 'Lbol', np.log10(Lbol_sub), UVslope, MIRslope1, MIRslope2, ulirg_x=np.log10(ulirg_LIR_out*3.8E33))
# plot2.Box_1panel('ULIRG_Lbol_Lx_box_1panel', 'Lbol/Lx', np.log10(Lbol)-Lx, UVslope, MIRslope1, MIRslope2, ulirg_x=np.log10(Lbol_ulirg[Lbol_ulirg > 0])-ulirg_Lx_out[Lbol_ulirg > 0])
# plot2.Box_1panel('ULIRG_Lbol_Lx_box_1panel_sub', 'Lbol/Lx', np.log10(Lbol_sub)-Lx, UVslope, MIRslope1, MIRslope2, ulirg_x=np.log10(ulirg_LIR_out*3.8E33)-ulirg_Lx_out)
# plot2.ratios_1panel('ULIRG_MIR_Nh_1panel','Nh','MIR6','Bins',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check,ulirg_Nh=ulirg_Nh_out,ulirg_Lx=ulirg_Lx_out,ulirg_Flux=np.log10(F6_ulirg),ulirg_F1=F1_ulirg)
# plot2.ratios_1panel('ULIRG_UV_Nh_1panel','Nh','UV','Bins',Nh,Lx,np.log10(Lbol_sub),F1,np.log10(F025),np.log10(F6),np.log10(F100),np.log10(F10),UVslope,MIRslope1,MIRslope2,upper_check,ulirg_Nh=ulirg_Nh_out,ulirg_Lx=ulirg_Lx_out,ulirg_Flux=np.log10(F025_ulirg),ulirg_F1=F1_ulirg)

###############################################################################
###############################################################################
###############################################################################

tf = time.perf_counter()
print(f'Total time: {tf - ti:0.4f} seconds')


